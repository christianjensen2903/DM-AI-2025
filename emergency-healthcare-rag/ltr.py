import logging
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import lightgbm as lgb
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

# ---------- CONFIG / CONSTANTS ----------
FEATURE_NAMES = ["bm25", "tfidf_cosine", "jaccard", "query_len", "doc_len"]


@dataclass
class LTRConfig:
    train_source: str = "synthetic"
    val_source: str = "train"
    k1: float = 1.5
    b: float = 0.75
    normalize: bool = False
    remove_stopwords: bool = False
    use_stemming: bool = False
    negative_per_query: int = 5
    random_state: int = 42
    early_stop_frac: float = 0.1  # fraction of queries set aside for early stopping


# ---------- SETUP LOGGING ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------- UTILITIES ----------
def build_candidate_pairs(
    docs: List[Document],
    statement_tokens: List[str],
    true_topic: int,
    bm25: rank_bm25.BM25Okapi,
    tfidf_vectorizer: TfidfVectorizer,
    doc_tfidf_matrix,
    rng: np.random.Generator,
    negative_per_query: int,
    normalize: bool,
    remove_stopwords: bool,
    use_stemming: bool,
) -> Tuple[List[List[float]], List[int], List[Dict[str, Any]]]:
    """
    Build feature vectors, labels, and meta for one statement.
    """
    # Identify positives and negatives
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

    if not positives:
        return [], [], []

    num_neg = min(negative_per_query, len(negatives))
    neg_indices = rng.choice(len(negatives), num_neg, replace=False)

    pairs = []
    for di, doc in positives:
        pairs.append((di, doc, 1))
    for idx in neg_indices:
        di, doc = negatives[idx]
        pairs.append((di, doc, 0))

    stmt_text = " ".join(statement_tokens)
    stmt_vec = tfidf_vectorizer.transform([stmt_text])[0]  # sparse row

    features_list = []
    labels = []
    metas = []

    for di, doc, label in pairs:
        feats = []

        # BM25 score
        bm25_score = bm25.get_scores(statement_tokens)[di]
        feats.append(bm25_score)

        # TF-IDF cosine similarity
        doc_vec = doc_tfidf_matrix[di]
        cos_sim = cosine_similarity(stmt_vec, doc_vec)[0, 0]
        feats.append(cos_sim)

        # Jaccard
        if normalize:
            doc_tokens = preprocess_func(
                normalize_medical_text(doc.page_content),
                remove_stopwords=remove_stopwords,
                use_stemming=use_stemming,
            )
        else:
            doc_tokens = preprocess_func(
                doc.page_content,
                remove_stopwords=remove_stopwords,
                use_stemming=use_stemming,
            )
        stmt_set = set(statement_tokens)
        doc_set = set(doc_tokens)
        jaccard = (
            (len(stmt_set & doc_set) / len(stmt_set | doc_set))
            if (stmt_set | doc_set)
            else 0.0
        )
        feats.append(jaccard)

        # Length features
        feats.append(len(statement_tokens))
        feats.append(len(doc_tokens))

        features_list.append(feats)
        labels.append(label)
        metas.append(
            {
                "doc_idx": di,
                "true_topic": true_topic,
                "doc_topic": doc.metadata.get("topic_id", -1),
            }
        )

    return features_list, labels, metas


def flatten_grouped_data(
    group_idxs: List[int],
    X_flat: np.ndarray,
    y_flat: np.ndarray,
    group: List[int],
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Given indices of which groups to include, reconstruct the flattened arrays with corresponding group sizes.
    """
    X_parts = []
    y_parts = []
    grp = []
    cursor = 0
    for qi, g in enumerate(group):
        segment = slice(cursor, cursor + g)
        if qi in group_idxs:
            X_parts.append(X_flat[segment])
            y_parts.append(y_flat[segment])
            grp.append(g)
        cursor += g

    if not X_parts:
        raise RuntimeError("No data after internal train/early-stop split.")

    return np.vstack(X_parts), np.concatenate(y_parts), grp


def prepare_tfidf(
    docs: List[Document],
    statements: List[List[str]],
    normalize: bool,
    remove_stopwords: bool,
    use_stemming: bool,
) -> Tuple[TfidfVectorizer, List[str], np.ndarray, List[str]]:
    """
    Prepares TF-IDF vectorizer and matrices for docs and statements. Returns vectorizer, doc_texts (strings),
    doc_tfidf_matrix, statement_texts (strings).
    """

    def prep_content(text: str) -> List[str]:
        if normalize:
            return preprocess_func(
                normalize_medical_text(text),
                remove_stopwords=remove_stopwords,
                use_stemming=use_stemming,
            )
        else:
            return preprocess_func(
                text, remove_stopwords=remove_stopwords, use_stemming=use_stemming
            )

    doc_texts = [" ".join(prep_content(doc.page_content)) for doc in docs]
    statement_texts = [" ".join(tokens) for tokens in statements]

    tfidf = TfidfVectorizer()
    tfidf.fit(doc_texts + statement_texts)
    doc_tfidf = tfidf.transform(doc_texts)

    return tfidf, doc_texts, doc_tfidf, statement_texts


def train_ranker(
    X: np.ndarray,
    y: np.ndarray,
    group: List[int],
    early_stop_frac: float,
    random_state: int,
) -> Tuple[lgb.LGBMRanker, StandardScaler]:
    """
    Train LightGBM LambdaMART ranker with internal train / early-stop split.
    """
    if X.size == 0:
        raise RuntimeError("Empty training matrix passed to ranker.")

    # Split groups into main train and early-stop
    num_queries = len(group)
    stmt_indices = list(range(num_queries))
    train_grp_idxs, es_grp_idxs = train_test_split(
        stmt_indices, test_size=early_stop_frac, random_state=random_state
    )

    X_train_main, y_train_main, group_main = flatten_grouped_data(
        train_grp_idxs, X, y, group
    )
    X_es, y_es, group_es = flatten_grouped_data(es_grp_idxs, X, y, group)

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

    return ranker, scaler


def evaluate_ltr(
    ranker: lgb.LGBMRanker,
    scaler: StandardScaler,
    docs: List[Document],
    processed_statements_val: List[List[str]],
    true_labels_val: List[int],
    bm25: rank_bm25.BM25Okapi,
    tfidf: TfidfVectorizer,
    doc_tfidf: np.ndarray,
    config: LTRConfig,
) -> Dict[str, Any]:
    """
    Evaluate the trained LTR model on validation set. Returns metrics and predictions.
    """
    rng = np.random.default_rng(config.random_state)
    correct = 0
    total = 0
    ndcg1_list, ndcg3_list, ndcg5_list = [], [], []
    y_true_ltr, y_pred_ltr = [], []

    for qi, (stmt_tokens, true_topic) in enumerate(
        zip(processed_statements_val, true_labels_val)
    ):
        feature_vecs, labels, metas = build_candidate_pairs(
            docs=docs,
            statement_tokens=stmt_tokens,
            true_topic=true_topic,
            bm25=bm25,
            tfidf_vectorizer=tfidf,
            doc_tfidf_matrix=doc_tfidf,
            rng=rng,
            negative_per_query=config.negative_per_query,
            normalize=config.normalize,
            remove_stopwords=config.remove_stopwords,
            use_stemming=config.use_stemming,
        )
        if not feature_vecs:
            continue  # no positives for this query

        X_stmt = np.array(feature_vecs, dtype=float)
        X_stmt_scaled = scaler.transform(X_stmt)
        X_stmt_df = pd.DataFrame(X_stmt_scaled, columns=FEATURE_NAMES)
        scores = ranker.predict(X_stmt_df)

        order = np.argsort(scores)[::-1]
        ranked_labels = np.array(labels)[order].tolist()
        top_meta = metas[order[0]]
        pred_topic = top_meta["doc_topic"]

        y_true_ltr.append(true_topic)
        y_pred_ltr.append(pred_topic)

        if pred_topic == true_topic:
            correct += 1
        total += 1

        ndcg1_list.append(ndcg_at_k(ranked_labels, 1))
        ndcg3_list.append(ndcg_at_k(ranked_labels, 3))
        ndcg5_list.append(ndcg_at_k(ranked_labels, 5))

    ltr_accuracy_val = correct / total if total > 0 else 0.0
    avg_ndcg1 = float(np.mean(ndcg1_list)) if ndcg1_list else 0.0
    avg_ndcg3 = float(np.mean(ndcg3_list)) if ndcg3_list else 0.0
    avg_ndcg5 = float(np.mean(ndcg5_list)) if ndcg5_list else 0.0

    logger.info(
        f"[LTR model on validation ({config.val_source})] Top-1 accuracy: {ltr_accuracy_val:.4f}"
    )
    logger.info(
        f"[LTR model on validation] Avg NDCG@1: {avg_ndcg1:.4f}, @3: {avg_ndcg3:.4f}, @5: {avg_ndcg5:.4f}"
    )

    return {
        "accuracy": ltr_accuracy_val,
        "ndcg@1": avg_ndcg1,
        "ndcg@3": avg_ndcg3,
        "ndcg@5": avg_ndcg5,
        "y_true": y_true_ltr,
        "y_pred": y_pred_ltr,
    }


def ndcg_at_k(relevances: List[int], k: int) -> float:
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


# ---------- MAIN WORKFLOW ----------
def test_ltr(config: LTRConfig) -> Dict[str, Any]:
    """Train on data/{train_source} and validate on data/{val_source}."""
    base = Path("data")
    topics_json = base / "topics.json"
    cleaned_root = base / "cleaned_topics"

    train_statements_dir = base / config.train_source / "statements"
    train_answers_dir = base / config.train_source / "answers"
    val_statements_dir = base / config.val_source / "statements"
    val_answers_dir = base / config.val_source / "answers"

    logger.info("Loading topics and documents...")
    topic2id, _ = load_topics(topics_json)
    docs = load_cleaned_documents(cleaned_root, topic2id, normalize=False)
    logger.info(f"Loaded {len(docs)} documents")

    # Preprocess documents for BM25 (uses same tokenization as for retrieval)
    if config.normalize:
        texts_processed = [
            preprocess_func(
                normalize_medical_text(doc.page_content),
                remove_stopwords=config.remove_stopwords,
                use_stemming=config.use_stemming,
            )
            for doc in docs
        ]
    else:
        texts_processed = [
            preprocess_func(
                doc.page_content,
                remove_stopwords=config.remove_stopwords,
                use_stemming=config.use_stemming,
            )
            for doc in docs
        ]

    # Load & preprocess statements/answers
    processed_statements_train, true_labels_train, _ = preprocess_statements(
        train_statements_dir,
        train_answers_dir,
        config.normalize,
        config.remove_stopwords,
        config.use_stemming,
    )
    processed_statements_val, true_labels_val, _ = preprocess_statements(
        val_statements_dir,
        val_answers_dir,
        config.normalize,
        config.remove_stopwords,
        config.use_stemming,
    )

    # BM25 baseline on validation
    bm25 = rank_bm25.BM25Okapi(texts_processed, k1=config.k1, b=config.b)
    y_true_bm25_val, y_pred_bm25_val = get_retrieval_predictions(
        bm25, docs, processed_statements_val, true_labels_val
    )
    bm25_accuracy_val = accuracy_score(y_true_bm25_val, y_pred_bm25_val)
    logger.info(
        f"[BM25 baseline on validation ({config.val_source})] Top-1 accuracy: {bm25_accuracy_val:.4f}"
    )

    # Prepare TF-IDF with training statements
    tfidf, _, doc_tfidf, statement_texts_train = prepare_tfidf(
        docs,
        processed_statements_train,
        normalize=config.normalize,
        remove_stopwords=config.remove_stopwords,
        use_stemming=config.use_stemming,
    )

    rng = np.random.default_rng(config.random_state)
    X_rows = []
    y_rows = []
    group = []

    for stmt_tokens, true_topic in zip(processed_statements_train, true_labels_train):
        feature_vecs, labels, _ = build_candidate_pairs(
            docs=docs,
            statement_tokens=stmt_tokens,
            true_topic=true_topic,
            bm25=bm25,
            tfidf_vectorizer=tfidf,
            doc_tfidf_matrix=doc_tfidf,
            rng=rng,
            negative_per_query=config.negative_per_query,
            normalize=config.normalize,
            remove_stopwords=config.remove_stopwords,
            use_stemming=config.use_stemming,
        )
        if not feature_vecs:
            continue  # skip if no positive

        group.append(len(feature_vecs))
        X_rows.extend(feature_vecs)
        y_rows.extend(labels)

    X_train = np.array(X_rows, dtype=float)
    y_train = np.array(y_rows, dtype=int)

    if X_train.size == 0:
        raise RuntimeError(
            "No training data constructed for LTR (check positive examples)."
        )

    ranker, scaler = train_ranker(
        X=X_train,
        y=y_train,
        group=group,
        early_stop_frac=config.early_stop_frac,
        random_state=config.random_state,
    )

    # Evaluate LTR
    ltr_eval = evaluate_ltr(
        ranker=ranker,
        scaler=scaler,
        docs=docs,
        processed_statements_val=processed_statements_val,
        true_labels_val=true_labels_val,
        bm25=bm25,
        tfidf=tfidf,
        doc_tfidf=doc_tfidf,
        config=config,
    )

    return {
        "bm25_baseline_val": {
            "accuracy": bm25_accuracy_val,
            "y_true": y_true_bm25_val,
            "y_pred": y_pred_bm25_val,
        },
        "ltr": {
            **ltr_eval,
            "model": ranker,
            "scaler": scaler,
            "tfidf": tfidf,
        },
    }


def main():
    config = LTRConfig(
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
    results = test_ltr(config)
    # Optionally: persist or further inspect `results`


if __name__ == "__main__":
    main()
