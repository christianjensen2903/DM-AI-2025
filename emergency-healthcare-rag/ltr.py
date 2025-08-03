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
    train_source: str = "synthetic",
    val_source: str = "train",
    k1: float = 1.5,
    b: float = 0.75,
    normalize: bool = False,
    remove_stopwords: bool = False,
    use_stemming: bool = False,
    negative_per_query: int = 5,
    random_state: int = 42,
):
    """Train on data/{train_source} and validate on data/{val_source}."""
    # Setup paths
    base = Path("data")
    topics_json = base / "topics.json"
    cleaned_root = base / "cleaned_topics"

    train_statements_dir = base / train_source / "statements"
    train_answers_dir = base / train_source / "answers"
    val_statements_dir = base / val_source / "statements"
    val_answers_dir = base / val_source / "answers"

    # Load topics & documents (shared)
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

    # Load and preprocess statements/answers for train and val
    processed_statements_train, true_labels_train, _ = preprocess_statements(
        train_statements_dir,
        train_answers_dir,
        normalize,
        remove_stopwords,
        use_stemming,
    )
    processed_statements_val, true_labels_val, _ = preprocess_statements(
        val_statements_dir, val_answers_dir, normalize, remove_stopwords, use_stemming
    )

    # ----- BM25 baseline evaluated on validation set -----
    bm25 = rank_bm25.BM25Okapi(texts_processed, k1=k1, b=b)
    y_true_bm25_val, y_pred_bm25_val = get_retrieval_predictions(
        bm25, docs, processed_statements_val, true_labels_val
    )
    bm25_accuracy_val = accuracy_score(y_true_bm25_val, y_pred_bm25_val)
    print(
        f"[BM25 baseline on validation ({val_source})] Top-1 accuracy: {bm25_accuracy_val:.4f}"
    )

    # ---- Build LTR training data from train_source ----
    # Reconstruct strings for TF-IDF (with same preprocessing)
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
    statement_texts_train = [" ".join(tokens) for tokens in processed_statements_train]

    tfidf = TfidfVectorizer()
    tfidf.fit(doc_texts + statement_texts_train)
    doc_tfidf = tfidf.transform(doc_texts)
    stmt_tfidf_train = tfidf.transform(statement_texts_train)

    # Build training pairs (synthetic / train_source)
    X_rows = []
    y_rows = []
    group = []
    rng = np.random.default_rng(random_state)

    for qi, (stmt_vec, stmt_tokens, true_topic) in enumerate(
        zip(stmt_tfidf_train, processed_statements_train, true_labels_train)
    ):
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
            continue  # skip if no positive example

        num_neg = min(negative_per_query, len(negatives))
        neg_indices = rng.choice(len(negatives), num_neg, replace=False)
        candidate_pairs = []

        for di, doc in positives:
            candidate_pairs.append((di, doc, 1))
        for idx in neg_indices:
            di, doc = negatives[idx]
            candidate_pairs.append((di, doc, 0))

        group.append(len(candidate_pairs))

        for di, doc, label in candidate_pairs:
            feats = []

            # BM25 score
            bm25_score = bm25.get_scores(processed_statements_train[qi])[di]
            feats.append(bm25_score)

            # TF-IDF cosine similarity
            doc_vec = doc_tfidf[di]
            cos_sim = cosine_similarity(stmt_vec, doc_vec)[0, 0]
            feats.append(cos_sim)

            # Jaccard overlap
            stmt_set = set(processed_statements_train[qi])
            doc_tokens_for_jaccard = preprocess_func(
                doc.page_content,
                remove_stopwords=remove_stopwords,
                use_stemming=use_stemming,
            )
            doc_set = set(doc_tokens_for_jaccard)
            jaccard = (
                len(stmt_set & doc_set) / len(stmt_set | doc_set)
                if len(stmt_set | doc_set) > 0
                else 0.0
            )
            feats.append(jaccard)

            feats.append(len(processed_statements_train[qi]))  # query length
            feats.append(len(doc_tokens_for_jaccard))  # doc length

            X_rows.append(feats)
            y_rows.append(label)

    X_train = np.array(X_rows, dtype=float)
    y_train = np.array(y_rows, dtype=int)

    if X_train.size == 0:
        raise RuntimeError(
            "No training data constructed for LTR (check positive examples)."
        )

    FEATURE_NAMES = ["bm25", "tfidf_cosine", "jaccard", "query_len", "doc_len"]

    # Split the per-query groups: 90% for primary training, 10% for early-stopping validation
    stmt_indices = list(range(len(group)))
    train_grp_idxs, es_grp_idxs = train_test_split(
        stmt_indices, test_size=0.1, random_state=random_state
    )

    def flatten_groups(selected_idxs, X_flat, y_flat, group_list):
        X_parts, y_parts, grp = [], [], []
        cursor = 0
        for qi, g in enumerate(group_list):
            segment = slice(cursor, cursor + g)
            if qi in selected_idxs:
                X_parts.append(X_flat[segment])
                y_parts.append(y_flat[segment])
                grp.append(g)
            cursor += g
        if not X_parts:
            raise RuntimeError("No data after internal train/early-stop split.")
        X_res = np.vstack(X_parts)
        y_res = np.concatenate(y_parts)
        return X_res, y_res, grp

    # flatten into main train and early-stop validation
    X_train_main, y_train_main, group_main = flatten_groups(
        train_grp_idxs, X_train, y_train, group
    )
    X_es, y_es, group_es = flatten_groups(es_grp_idxs, X_train, y_train, group)

    # scale based on main training portion
    scaler = StandardScaler()
    X_train_main_scaled = scaler.fit_transform(X_train_main)
    X_es_scaled = scaler.transform(X_es)

    X_train_df = pd.DataFrame(X_train_main_scaled, columns=FEATURE_NAMES)
    X_es_df = pd.DataFrame(X_es_scaled, columns=FEATURE_NAMES)

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
        y_train_main,
        group=group_main,
        eval_set=[(X_es_df, y_es)],
        eval_group=[group_es],
        eval_at=[1, 3, 5],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(period=0)],
    )

    # ---- Evaluate LTR model on validation set (val_source) ----
    stmt_tfidf_val = tfidf.transform(
        [" ".join(tokens) for tokens in processed_statements_val]
    )

    correct = 0
    total = 0
    ndcg1_list, ndcg3_list, ndcg5_list = [], [], []
    y_true_ltr = []
    y_pred_ltr = []

    for qi, (stmt_vec, stmt_tokens, true_topic) in enumerate(
        zip(stmt_tfidf_val, processed_statements_val, true_labels_val)
    ):
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
            continue  # skip if no positive to evaluate

        num_neg = min(negative_per_query, len(negatives))
        neg_indices = rng.choice(len(negatives), num_neg, replace=False)
        candidate_pairs = []

        for di, doc in positives:
            candidate_pairs.append((di, doc, 1))
        for idx in neg_indices:
            di, doc = negatives[idx]
            candidate_pairs.append((di, doc, 0))

        features_list = []
        labels = []
        metas = []

        for di, doc, label in candidate_pairs:
            feats = []

            # BM25
            bm25_score = bm25.get_scores(processed_statements_val[qi])[di]
            feats.append(bm25_score)

            # TF-IDF cosine
            doc_vec = doc_tfidf[di]
            cos_sim = cosine_similarity(stmt_vec, doc_vec)[0, 0]
            feats.append(cos_sim)

            # Jaccard
            stmt_set = set(processed_statements_val[qi])
            doc_tokens_for_jaccard = preprocess_func(
                doc.page_content,
                remove_stopwords=remove_stopwords,
                use_stemming=use_stemming,
            )
            doc_set = set(doc_tokens_for_jaccard)
            jaccard = (
                len(stmt_set & doc_set) / len(stmt_set | doc_set)
                if len(stmt_set | doc_set) > 0
                else 0.0
            )
            feats.append(jaccard)

            feats.append(len(processed_statements_val[qi]))
            feats.append(len(doc_tokens_for_jaccard))

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

        if not features_list:
            continue

        X_stmt = np.array(features_list, dtype=float)
        X_stmt_scaled = scaler.transform(X_stmt)
        X_stmt_df = pd.DataFrame(X_stmt_scaled, columns=FEATURE_NAMES)
        scores = ranker.predict(X_stmt_df)

        order = np.argsort(scores)[::-1]
        ranked_labels = np.array(labels)[order].tolist()

        top_idx = order[0]
        top_meta = metas[top_idx]
        pred_topic = top_meta["doc_topic"]
        true_topic = top_meta["true_topic"]

        y_true_ltr.append(true_topic)
        y_pred_ltr.append(pred_topic)
        if pred_topic == true_topic:
            correct += 1
        total += 1

        ndcg1_list.append(ndcg_at_k(ranked_labels, 1))
        ndcg3_list.append(ndcg_at_k(ranked_labels, 3))
        ndcg5_list.append(ndcg_at_k(ranked_labels, 5))

    ltr_accuracy_val = correct / total if total > 0 else 0.0
    avg_ndcg1 = np.mean(ndcg1_list) if ndcg1_list else 0.0
    avg_ndcg3 = np.mean(ndcg3_list) if ndcg3_list else 0.0
    avg_ndcg5 = np.mean(ndcg5_list) if ndcg5_list else 0.0

    print(
        f"[LTR model on validation ({val_source})] Top-1 accuracy: {ltr_accuracy_val:.4f}"
    )
    print(
        f"[LTR model on validation] Avg NDCG@1: {avg_ndcg1:.4f}, @3: {avg_ndcg3:.4f}, @5: {avg_ndcg5:.4f}"
    )

    return {
        "bm25_baseline_val": {
            "accuracy": bm25_accuracy_val,
            "y_true": y_true_bm25_val,
            "y_pred": y_pred_bm25_val,
        },
        "ltr": {
            "accuracy": ltr_accuracy_val,
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
    # train on synthetic, validate on train
    test_ltr(
        train_source="synthetic",
        val_source="train",
        k1=2.5,
        b=1.0,
        normalize=False,
        remove_stopwords=True,
        use_stemming=False,
        negative_per_query=5,
        random_state=42,
    )


if __name__ == "__main__":
    main()
