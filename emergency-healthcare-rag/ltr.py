import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import lightgbm as lgb
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import rank_bm25

from langchain.schema import Document
from bm25_retriever import (
    preprocess_func,
    load_cleaned_documents,
    load_topics,
    preprocess_statements,
    normalize_medical_text,
    get_retrieval_predictions,
)
from pathlib import Path
from sklearn.metrics import accuracy_score
import pandas as pd


def build_ltr_training_data(
    docs: list[Document],
    processed_statements: list[list[str]],
    true_labels: list[int],
    bm25: rank_bm25.BM25Okapi,
    negative_per_query: int = 5,
):
    """
    Build feature matrix and labels for learning-to-rank.

    Returns:
        X: np.ndarray of shape (num_pairs, num_features)
        y: list of relevance labels (0 or 1)
        group: list of group sizes (per query)
        meta: list of metadata dicts (optional, for debugging)
    """
    # Precompute TF-IDF on documents + statements (raw text might be needed)
    # For simplicity, reconstruct strings from tokens:
    doc_texts = [" ".join(preprocess_func(doc.page_content)) for doc in docs]
    statement_texts = [" ".join(tokens) for tokens in processed_statements]

    tfidf = TfidfVectorizer()
    tfidf.fit(doc_texts + statement_texts)
    doc_tfidf = tfidf.transform(doc_texts)
    stmt_tfidf = tfidf.transform(statement_texts)

    X_rows = []
    y = []
    group = []
    meta = []

    for qi, (stmt_vec, stmt_tokens, true_topic) in enumerate(
        zip(stmt_tfidf, processed_statements, true_labels)
    ):
        # Identify positives
        positives = [
            (di, doc)
            for di, doc in enumerate(docs)
            if doc.metadata.get("topic_id", -1) == true_topic
        ]
        negatives = [
            (di, doc)
            for di, doc in enumerate(docs)
            if doc.metadata.get("topic_id", -1) != true_topic
        ]

        if len(positives) == 0:
            continue  # skip if no positive to learn from

        # Sample negatives
        sampled_neg = np.random.choice(
            len(negatives), min(negative_per_query, len(negatives)), replace=False
        )
        candidate_pairs = []

        # Add all positives
        for di, doc in positives:
            candidate_pairs.append((di, doc, 1))
        # Add sampled negatives
        for idx in sampled_neg:
            di, doc = negatives[idx]
            candidate_pairs.append((di, doc, 0))

        # For group size bookkeeping
        group.append(len(candidate_pairs))

        for di, doc, label in candidate_pairs:
            features = []

            # 1. BM25 score: compute score of this statement against this doc
            bm25_score = bm25.get_scores(processed_statements[qi])[di]
            features.append(bm25_score)

            # 2. TF-IDF cosine similarity
            doc_vec = doc_tfidf[di]
            cos_sim = cosine_similarity(stmt_vec, doc_vec)[0, 0]
            features.append(cos_sim)

            # 3. Jaccard over token sets (raw tokens before stemming/stop removal if desired)
            stmt_set = set(processed_statements[qi])
            doc_tokens = preprocess_func(doc.page_content)
            doc_set = set(doc_tokens)
            jaccard = (
                len(stmt_set & doc_set) / len(stmt_set | doc_set)
                if len(stmt_set | doc_set) > 0
                else 0.0
            )
            features.append(jaccard)

            # 4. Length features
            features.append(len(processed_statements[qi]))  # query length
            features.append(len(doc_tokens))  # doc length

            # (Optional) add more features here...

            X_rows.append(features)
            y.append(label)
            meta.append(
                {
                    "query_idx": qi,
                    "doc_idx": di,
                    "true_topic": true_topic,
                    "pred_topic": doc.metadata.get("topic_id", -1),
                }
            )

    X = np.array(X_rows, dtype=float)
    y = np.array(y, dtype=int)
    return X, y, group, meta


def train_ltr_model(X, y, group):
    """
    Train LightGBM LambdaMART ranker.
    """
    # Scale features (LightGBM is fairly robust, but scaling helps convergence)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        importance_type="gain",
        learning_rate=0.1,
        n_estimators=500,
        num_leaves=31,
        min_child_samples=20,
    )

    # For simplicity, no separate validation here; in real use, split queries so groups don't leak.
    ranker.fit(
        X_scaled,
        y,
        group=group,
        eval_at=[1, 3, 5],
        verbose=False,
        callbacks=[lgb.early_stopping(stopping_rounds=50)],
    )
    return ranker, scaler


def get_ltr_predictions(
    ranker,
    scaler,
    docs: list[Document],
    statement_tokens: list[str],
    bm25: rank_bm25.BM25Okapi,
    tfidf: TfidfVectorizer,
    doc_tfidf_matrix,
):
    """
    Given one statement, compute ranking scores over all docs.
    Returns list of (doc_idx, score) sorted descending.
    """
    stmt_text = " ".join(statement_tokens)
    stmt_vec = tfidf.transform([stmt_text])

    features = []
    for di, doc in enumerate(docs):
        feat = []

        # BM25 score
        bm25_score = bm25.get_scores(statement_tokens)[di]
        feat.append(bm25_score)

        # TF-IDF cosine
        doc_vec = doc_tfidf_matrix[di]
        cos_sim = cosine_similarity(stmt_vec, doc_vec)[0, 0]
        feat.append(cos_sim)

        # Jaccard
        stmt_set = set(statement_tokens)
        doc_tokens = preprocess_func(doc.page_content)
        doc_set = set(doc_tokens)
        jaccard = (
            len(stmt_set & doc_set) / len(stmt_set | doc_set)
            if len(stmt_set | doc_set) > 0
            else 0.0
        )
        feat.append(jaccard)

        feat.append(len(statement_tokens))
        feat.append(len(doc_tokens))

        features.append(feat)

    Xq = np.array(features, dtype=float)
    Xq_scaled = scaler.transform(Xq)
    scores = ranker.predict(Xq_scaled)  # higher is better

    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return ranked  # list of (doc_idx, score)


def ndcg_at_k(relevances: list[int], k: int) -> float:
    """
    Compute NDCG@k for a single query given binary relevances in ranked order.
    """

    def dcg(rs):
        return sum((2**r - 1) / np.log2(idx + 2) for idx, r in enumerate(rs[:k]))

    ideal = sorted(relevances, reverse=True)
    ideal_dcg = dcg(ideal)
    if ideal_dcg == 0:
        return 0.0
    return dcg(relevances) / ideal_dcg


def test_ltr(
    k1: float = 1.5,
    b: float = 0.75,
    normalize: bool = False,
    remove_stopwords: bool = False,
    use_stemming: bool = False,
    negative_per_query: int = 5,
    val_size: float = 0.2,
    random_state: int = 42,
):
    """Train and evaluate a LightGBM LambdaMART learning-to-rank model and compare to BM25 baseline."""
    # Setup paths
    base = Path("data")
    topics_json = base / "topics.json"
    cleaned_root = base / "cleaned_topics"
    statements_dir = base / "synthetic" / "statements"
    answers_dir = base / "synthetic" / "answers"

    # Load data
    print("Loading data...")
    topic2id, _ = load_topics(topics_json)
    docs = load_cleaned_documents(cleaned_root, topic2id, normalize=False)
    print(f"Loaded {len(docs)} documents")

    # Preprocess document texts (for BM25 and TF-IDF)
    if normalize:
        texts_processed = [
            preprocess_func(
                normalize_medical_text(doc.page_content),
                remove_stopwords=remove_stopwords,
                use_stemming=use_stemming,
            )
            for doc in docs
        ]
    else:
        texts_processed = [
            preprocess_func(
                doc.page_content,
                remove_stopwords=remove_stopwords,
                use_stemming=use_stemming,
            )
            for doc in docs
        ]

    processed_statements, true_labels, _ = preprocess_statements(
        statements_dir, answers_dir, normalize, remove_stopwords, use_stemming
    )

    # Baseline: BM25 with given k1, b
    bm25 = rank_bm25.BM25Okapi(texts_processed, k1=k1, b=b)
    y_true_bm25, y_pred_bm25 = get_retrieval_predictions(
        bm25, docs, processed_statements, true_labels
    )
    bm25_accuracy = accuracy_score(y_true_bm25, y_pred_bm25)
    print(f"[BM25 baseline] Top-1 accuracy: {bm25_accuracy:.4f}")

    # ---- Build LTR dataset per statement ----
    # Reconstruct "string" versions for TF-IDF using same preprocessing
    doc_texts = [
        " ".join(
            preprocess_func(
                doc.page_content,
                remove_stopwords=remove_stopwords,
                use_stemming=use_stemming,
            )
        )
        for doc in docs
    ]
    statement_texts = [" ".join(tokens) for tokens in processed_statements]

    tfidf = TfidfVectorizer()
    tfidf.fit(doc_texts + statement_texts)
    doc_tfidf = tfidf.transform(doc_texts)
    stmt_tfidf = tfidf.transform(statement_texts)

    X_per_stmt = []  # list of arrays of shape (num_candidates_for_stmt, num_features)
    y_per_stmt = []  # list of arrays of relevance labels
    meta_per_stmt = []  # list of list of metadata per candidate
    candidate_doc_indices_per_stmt = []  # to recover doc indices later

    rng = np.random.default_rng(random_state)

    for qi, (stmt_vec, stmt_tokens, true_topic) in enumerate(
        zip(stmt_tfidf, processed_statements, true_labels)
    ):
        # Positives: docs with same topic
        positives = [
            (di, doc)
            for di, doc in enumerate(docs)
            if doc.metadata.get("topic_id", -1) == true_topic
        ]
        negatives = [
            (di, doc)
            for di, doc in enumerate(docs)
            if doc.metadata.get("topic_id", -1) != true_topic
        ]

        if len(positives) == 0:
            continue  # skip queries with no positive example

        # Sample negatives
        num_neg = min(negative_per_query, len(negatives))
        neg_indices = rng.choice(len(negatives), num_neg, replace=False)
        candidates = []

        # Add all positives
        for di, doc in positives:
            candidates.append((di, doc, 1))
        # Add sampled negatives
        for idx in neg_indices:
            di, doc = negatives[idx]
            candidates.append((di, doc, 0))

        features_list = []
        labels = []
        metas = []
        doc_indices = []

        for di, doc, label in candidates:
            feats = []

            # 1. BM25 score
            bm25_score = bm25.get_scores(processed_statements[qi])[di]
            feats.append(bm25_score)

            # 2. TF-IDF cosine similarity
            doc_vec = doc_tfidf[di]
            cos_sim = cosine_similarity(stmt_vec, doc_vec)[0, 0]
            feats.append(cos_sim)

            # 3. Jaccard overlap
            stmt_set = set(processed_statements[qi])
            doc_tokens = preprocess_func(
                doc.page_content,
                remove_stopwords=remove_stopwords,
                use_stemming=use_stemming,
            )
            doc_set = set(doc_tokens)
            jaccard = (
                len(stmt_set & doc_set) / len(stmt_set | doc_set)
                if len(stmt_set | doc_set) > 0
                else 0.0
            )
            feats.append(jaccard)

            # 4. Lengths
            feats.append(len(processed_statements[qi]))  # query length
            feats.append(len(doc_tokens))  # doc length

            features_list.append(feats)
            labels.append(label)
            metas.append(
                {
                    "query_idx": qi,
                    "doc_idx": di,
                    "true_topic": true_topic,
                    "doc_topic": doc.metadata.get("topic_id", -1),
                }
            )
            doc_indices.append(di)

        X_per_stmt.append(np.array(features_list, dtype=float))
        y_per_stmt.append(np.array(labels, dtype=int))
        meta_per_stmt.append(metas)
        candidate_doc_indices_per_stmt.append(doc_indices)

    # Split statement indices (queries) into train/validation
    stmt_indices = list(range(len(X_per_stmt)))
    train_idxs, val_idxs = train_test_split(
        stmt_indices, test_size=val_size, random_state=random_state
    )

    def flatten_group(idxs):
        X_list, y_list, group, metas_flat, docidx_flat = [], [], [], [], []
        for i in idxs:
            X_list.append(X_per_stmt[i])
            y_list.append(y_per_stmt[i])
            group.append(len(X_per_stmt[i]))
            metas_flat.extend(meta_per_stmt[i])
            docidx_flat.extend(candidate_doc_indices_per_stmt[i])
        if len(X_list) == 0:
            return None, None, None, None, None
        X = np.vstack(X_list)
        y = np.concatenate(y_list)
        return X, y, group, metas_flat, docidx_flat

    X_train, y_train, group_train, meta_train, docidx_train = flatten_group(train_idxs)
    X_val, y_val, group_val, meta_val, docidx_val = flatten_group(val_idxs)

    if X_train is None or X_val is None:
        raise RuntimeError("Not enough data to build train/validation splits for LTR.")

    FEATURE_NAMES = ["bm25", "tfidf_cosine", "jaccard", "query_len", "doc_len"]

    # ---- training ----
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    X_train_df = pd.DataFrame(X_train_scaled, columns=FEATURE_NAMES)
    X_val_df = pd.DataFrame(X_val_scaled, columns=FEATURE_NAMES)

    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        importance_type="gain",
        learning_rate=0.1,
        n_estimators=500,
        num_leaves=31,
        min_child_samples=20,
        random_state=random_state,
    )

    ranker.fit(
        X_train_df,
        y_train,
        group=group_train,
        eval_set=[(X_val_df, y_val)],
        eval_group=[group_val],
        eval_at=[1, 3, 5],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(period=0)],
    )

    # ---- Evaluation on validation queries ----
    # For each validation statement, compute its own candidate features and predict
    correct = 0
    total = 0
    ndcg1_list, ndcg3_list, ndcg5_list = [], [], []
    y_true_ltr = []
    y_pred_ltr = []

    for idx in val_idxs:
        X_stmt = X_per_stmt[idx]
        y_stmt = y_per_stmt[idx]
        metas_stmt = meta_per_stmt[idx]
        doc_indices_stmt = candidate_doc_indices_per_stmt[idx]

        if X_stmt.shape[0] == 0:
            continue

        X_stmt_scaled = scaler.transform(X_stmt)
        X_stmt_df = pd.DataFrame(X_stmt_scaled, columns=FEATURE_NAMES)
        scores = ranker.predict(X_stmt_df)

        # rank candidates by score descending
        order = np.argsort(scores)[::-1]
        ranked_labels = y_stmt[order].tolist()

        # For accuracy: take top-1 candidate's doc topic and compare to true_topic
        top_idx = order[0]
        top_meta = metas_stmt[top_idx]
        pred_topic = top_meta["doc_topic"]
        true_topic = top_meta["true_topic"]

        y_true_ltr.append(true_topic)
        y_pred_ltr.append(pred_topic)
        if pred_topic == true_topic:
            correct += 1
        total += 1

        # NDCG@k
        ndcg1_list.append(ndcg_at_k(ranked_labels, 1))
        ndcg3_list.append(ndcg_at_k(ranked_labels, 3))
        ndcg5_list.append(ndcg_at_k(ranked_labels, 5))

    ltr_accuracy = correct / total if total > 0 else 0.0
    avg_ndcg1 = np.mean(ndcg1_list) if ndcg1_list else 0.0
    avg_ndcg3 = np.mean(ndcg3_list) if ndcg3_list else 0.0
    avg_ndcg5 = np.mean(ndcg5_list) if ndcg5_list else 0.0

    print(f"[LTR model] Top-1 accuracy on val: {ltr_accuracy:.4f}")
    print(
        f"[LTR model] Avg NDCG@1: {avg_ndcg1:.4f}, @3: {avg_ndcg3:.4f}, @5: {avg_ndcg5:.4f}"
    )

    return {
        "bm25_baseline": {
            "accuracy": bm25_accuracy,
            "y_true": y_true_bm25,
            "y_pred": y_pred_bm25,
        },
        "ltr": {
            "accuracy": ltr_accuracy,
            "ndcg@1": avg_ndcg1,
            "ndcg@3": avg_ndcg3,
            "ndcg@5": avg_ndcg5,
            "y_true": y_true_ltr,
            "y_pred": y_pred_ltr,
            "model": ranker,
            "scaler": scaler,
            "tfidf": tfidf,
        },
    }


def main():
    test_ltr(k1=2.5, b=1.0, normalize=False, remove_stopwords=True)


if __name__ == "__main__":
    main()
