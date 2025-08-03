import logging
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import rank_bm25
from tqdm import tqdm

from langchain.schema import Document

from bm25_retriever import (
    preprocess_func,
    preprocess_statements,
    normalize_medical_text,
    get_retrieval_predictions,
)
from utils import load_cleaned_documents, load_topics

# ---------- CONFIG / CONSTANTS ----------
FEATURE_NAMES = ["bm25", "tfidf_cosine", "jaccard", "query_len", "doc_len"]


@dataclass
class PassageLTRConfig:
    train_source: str = "synthetic"
    val_source: str = "train"
    k1: float = 1.5
    b: float = 0.75
    normalize: bool = False
    remove_stopwords: bool = False
    use_stemming: bool = False
    negative_per_query: int = 5
    random_state: int = 42
    early_stop_frac: float = 0.1
    # New passage-specific parameters
    passage_size: int = 150  # words per passage
    passage_overlap: int = 50  # overlapping words between passages
    use_section_splitting: bool = True  # also split by markdown sections


# ---------- SETUP LOGGING ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------- PASSAGE SPLITTING UTILITIES ----------
def precompute_passage_tokens(
    passages: List[Document],
    normalize: bool,
    remove_stopwords: bool,
    use_stemming: bool,
) -> List[List[str]]:
    """
    Precompute tokenized versions of all passages to avoid repeated preprocessing.
    """
    passage_tokens = []
    for passage in tqdm(passages, desc="Precomputing passage tokens"):
        if normalize:
            tokens = preprocess_func(
                normalize_medical_text(passage.page_content),
                remove_stopwords=remove_stopwords,
                use_stemming=use_stemming,
            )
        else:
            tokens = preprocess_func(
                passage.page_content,
                remove_stopwords=remove_stopwords,
                use_stemming=use_stemming,
            )
        passage_tokens.append(tokens)
    return passage_tokens


def split_by_sections(text: str) -> List[str]:
    """Split text by markdown headers (## Section)."""
    # Split by ## headers, but keep the header with the content
    sections = re.split(r"\n(?=##\s)", text)
    # Remove empty sections
    sections = [section.strip() for section in sections if section.strip()]
    return sections


def split_by_words(text: str, passage_size: int, overlap: int) -> List[str]:
    """Split text into fixed-size passages with overlap."""
    words = text.split()
    if len(words) <= passage_size:
        return [text]

    passages = []
    start = 0

    while start < len(words):
        end = min(start + passage_size, len(words))
        passage = " ".join(words[start:end])
        passages.append(passage)

        if end >= len(words):
            break
        start = end - overlap

    return passages


def split_document_into_passages(
    doc: Document, config: PassageLTRConfig
) -> List[Document]:
    """Split a document into smaller passages."""
    text = doc.page_content
    passages = []

    if config.use_section_splitting:
        # First split by sections
        sections = split_by_sections(text)
        for i, section in enumerate(sections):
            # Then split each section by word count if it's too long
            if len(section.split()) > config.passage_size:
                word_passages = split_by_words(
                    section, config.passage_size, config.passage_overlap
                )
                for j, passage in enumerate(word_passages):
                    passage_doc = Document(
                        page_content=passage,
                        metadata={
                            **doc.metadata,
                            "passage_type": "section_word",
                            "section_id": i,
                            "passage_id": f"{i}_{j}",
                            "parent_doc": doc.metadata.get("source", "unknown"),
                        },
                    )
                    passages.append(passage_doc)
            else:
                # Keep section as single passage
                passage_doc = Document(
                    page_content=section,
                    metadata={
                        **doc.metadata,
                        "passage_type": "section",
                        "section_id": i,
                        "passage_id": str(i),
                        "parent_doc": doc.metadata.get("source", "unknown"),
                    },
                )
                passages.append(passage_doc)
    else:
        # Just split by word count
        word_passages = split_by_words(
            text, config.passage_size, config.passage_overlap
        )
        for i, passage in enumerate(word_passages):
            passage_doc = Document(
                page_content=passage,
                metadata={
                    **doc.metadata,
                    "passage_type": "word",
                    "passage_id": str(i),
                    "parent_doc": doc.metadata.get("source", "unknown"),
                },
            )
            passages.append(passage_doc)

    return passages


def create_passage_corpus(
    docs: List[Document], config: PassageLTRConfig
) -> List[Document]:
    """Convert document corpus to passage corpus."""
    all_passages = []

    logger.info(f"Splitting {len(docs)} documents into passages...")
    for doc in tqdm(docs, desc="Creating passages"):
        passages = split_document_into_passages(doc, config)
        all_passages.extend(passages)

    logger.info(f"Created {len(all_passages)} passages from {len(docs)} documents")
    return all_passages


# ---------- EVALUATION UTILITIES ----------
def analyze_passage_rankings(
    ranked_passages: List[Document], true_topic: int, top_k: int = 10
) -> Dict[str, Any]:
    """Analyze the topic distribution in top-k ranked passages."""
    if not ranked_passages:
        return {"error": "No passages ranked"}

    # Count passages by topic in top-k
    topic_counts = {}
    correct_topic_positions = []

    for i, passage in enumerate(ranked_passages[:top_k]):
        passage_topic = passage.metadata.get("topic_id", -1)
        if passage_topic not in topic_counts:
            topic_counts[passage_topic] = 0
        topic_counts[passage_topic] += 1

        if passage_topic == true_topic:
            correct_topic_positions.append(i + 1)  # 1-indexed

    # Calculate metrics
    correct_in_top_k = len(correct_topic_positions)
    total_in_top_k = min(top_k, len(ranked_passages))
    precision_at_k = correct_in_top_k / total_in_top_k if total_in_top_k > 0 else 0.0

    # First correct position (MRR component)
    first_correct_pos = correct_topic_positions[0] if correct_topic_positions else None

    return {
        "topic_counts": topic_counts,
        "correct_in_top_k": correct_in_top_k,
        "total_in_top_k": total_in_top_k,
        "precision_at_k": precision_at_k,
        "first_correct_position": first_correct_pos,
        "correct_positions": correct_topic_positions,
    }


# ---------- MODIFIED LTR FUNCTIONS ----------
def build_candidate_pairs_passages(
    passages: List[Document],
    statement_tokens: List[str],
    true_topic: int,
    bm25: rank_bm25.BM25Okapi,
    tfidf_vectorizer: TfidfVectorizer,
    passage_tfidf_matrix,
    rng: np.random.Generator,
    negative_per_query: int,
    normalize: bool,
    remove_stopwords: bool,
    use_stemming: bool,
    passage_token_cache: Optional[List[List[str]]] = None,
) -> Tuple[List[List[float]], List[int], List[Dict[str, Any]]]:
    """
    Build feature vectors for passages (optimized version).
    """
    # Identify positives and negatives
    positives = [
        (pi, passage)
        for pi, passage in enumerate(passages)
        if passage.metadata.get("topic_id", -1) == true_topic
    ]
    negatives = [
        (pi, passage)
        for pi, passage in enumerate(passages)
        if passage.metadata.get("topic_id", -1) != true_topic
    ]

    if not positives:
        return [], [], []

    num_neg = min(negative_per_query, len(negatives))
    neg_indices = rng.choice(len(negatives), num_neg, replace=False)

    pairs = []
    for pi, passage in positives:
        pairs.append((pi, passage, 1))
    for idx in neg_indices:
        pi, passage = negatives[idx]
        pairs.append((pi, passage, 0))

    # OPTIMIZATION 1: Compute BM25 scores for ALL passages once, then index
    all_bm25_scores = bm25.get_scores(statement_tokens)

    # OPTIMIZATION 2: Transform statement to TF-IDF once
    stmt_text = " ".join(statement_tokens)
    stmt_vec = tfidf_vectorizer.transform([stmt_text])[0]

    # OPTIMIZATION 3: Precompute sets for all passages we need
    stmt_set = set(statement_tokens)
    passage_indices = [pi for pi, _, _ in pairs]

    # Extract relevant passage vectors for batch cosine similarity
    relevant_passage_vecs = passage_tfidf_matrix[passage_indices]

    # OPTIMIZATION 4: Batch compute cosine similarities
    cos_sims = cosine_similarity(stmt_vec, relevant_passage_vecs)[0]

    features_list = []
    labels = []
    metas = []

    for idx, (pi, passage, label) in enumerate(pairs):
        feats = []

        # BM25 score (from precomputed array)
        bm25_score = all_bm25_scores[pi]
        feats.append(bm25_score)

        # TF-IDF cosine similarity (from batch computation)
        cos_sim = cos_sims[idx]
        feats.append(cos_sim)

        # Jaccard - use cached tokens if available
        if passage_token_cache is not None:
            passage_tokens = passage_token_cache[pi]
        else:
            # Fallback to original computation
            if normalize:
                passage_tokens = preprocess_func(
                    normalize_medical_text(passage.page_content),
                    remove_stopwords=remove_stopwords,
                    use_stemming=use_stemming,
                )
            else:
                passage_tokens = preprocess_func(
                    passage.page_content,
                    remove_stopwords=remove_stopwords,
                    use_stemming=use_stemming,
                )

        passage_set = set(passage_tokens)
        jaccard = (
            (len(stmt_set & passage_set) / len(stmt_set | passage_set))
            if (stmt_set | passage_set)
            else 0.0
        )
        feats.append(jaccard)

        # Length features
        feats.append(len(statement_tokens))
        feats.append(len(passage_tokens))

        features_list.append(feats)
        labels.append(label)
        metas.append(
            {
                "passage_idx": pi,
                "true_topic": true_topic,
                "passage_topic": passage.metadata.get("topic_id", -1),
                "passage_type": passage.metadata.get("passage_type", "unknown"),
                "parent_doc": passage.metadata.get("parent_doc", "unknown"),
            }
        )

    return features_list, labels, metas


def evaluate_ltr_passages(
    ranker: lgb.LGBMRanker,
    scaler: StandardScaler,
    passages: List[Document],
    processed_statements_val: List[List[str]],
    true_labels_val: List[int],
    bm25: rank_bm25.BM25Okapi,
    tfidf: TfidfVectorizer,
    passage_tfidf: np.ndarray,
    config: PassageLTRConfig,
) -> Dict[str, Any]:
    """
    Evaluate the trained LTR model on validation set using passages.
    """
    rng = np.random.default_rng(config.random_state)

    # Traditional metrics (based on top passage)
    correct = 0
    total = 0

    # Passage-specific metrics
    mrr_scores = []
    precision_at_k_scores: Dict[int, List[float]] = {k: [] for k in [1, 3, 5, 10]}
    topic_analysis = []

    ndcg1_list, ndcg3_list, ndcg5_list = [], [], []
    y_true_ltr, y_pred_ltr = [], []

    for qi, (stmt_tokens, true_topic) in enumerate(
        tqdm(
            zip(processed_statements_val, true_labels_val),
            desc="Evaluating LTR",
            total=len(processed_statements_val),
        )
    ):
        feature_vecs, labels, metas = build_candidate_pairs_passages(
            passages=passages,
            statement_tokens=stmt_tokens,
            true_topic=true_topic,
            bm25=bm25,
            tfidf_vectorizer=tfidf,
            passage_tfidf_matrix=passage_tfidf,
            rng=rng,
            negative_per_query=config.negative_per_query,
            normalize=config.normalize,
            remove_stopwords=config.remove_stopwords,
            use_stemming=config.use_stemming,
            passage_token_cache=None,  # For eval, we'll compute on-the-fly for now
        )
        if not feature_vecs:
            continue

        X_stmt = np.array(feature_vecs, dtype=float)
        X_stmt_scaled = scaler.transform(X_stmt)
        X_stmt_df = pd.DataFrame(X_stmt_scaled, columns=FEATURE_NAMES)
        scores = ranker.predict(X_stmt_df)

        # Get ranking order
        order = np.argsort(scores)[::-1]
        ranked_labels = np.array(labels)[order].tolist()
        ranked_metas = [metas[i] for i in order]

        # Traditional accuracy (top passage topic)
        top_meta = ranked_metas[0]
        pred_topic = top_meta["passage_topic"]
        y_true_ltr.append(true_topic)
        y_pred_ltr.append(pred_topic)

        if pred_topic == true_topic:
            correct += 1
        total += 1

        # NDCG scores
        ndcg1_list.append(ndcg_at_k(ranked_labels, 1))
        ndcg3_list.append(ndcg_at_k(ranked_labels, 3))
        ndcg5_list.append(ndcg_at_k(ranked_labels, 5))

        # Create ranked passages for analysis
        ranked_passages = []
        for meta in ranked_metas:
            passage_idx = meta["passage_idx"]
            ranked_passages.append(passages[passage_idx])

        # Passage-specific analysis
        analysis = analyze_passage_rankings(ranked_passages, true_topic, top_k=10)
        topic_analysis.append(analysis)

        # MRR calculation
        first_correct = analysis["first_correct_position"]
        if first_correct:
            mrr_scores.append(1.0 / first_correct)
        else:
            mrr_scores.append(0.0)

        # Precision@k calculation
        for k in precision_at_k_scores.keys():
            analysis_k = analyze_passage_rankings(ranked_passages, true_topic, top_k=k)
            precision_at_k_scores[k].append(analysis_k["precision_at_k"])

    # Calculate final metrics
    ltr_accuracy_val = correct / total if total > 0 else 0.0
    avg_ndcg1 = float(np.mean(ndcg1_list)) if ndcg1_list else 0.0
    avg_ndcg3 = float(np.mean(ndcg3_list)) if ndcg3_list else 0.0
    avg_ndcg5 = float(np.mean(ndcg5_list)) if ndcg5_list else 0.0

    avg_mrr = float(np.mean(mrr_scores)) if mrr_scores else 0.0
    avg_precision_at_k = {
        k: float(np.mean(scores)) for k, scores in precision_at_k_scores.items()
    }

    logger.info(f"[LTR Passages] Top-1 accuracy: {ltr_accuracy_val:.4f}")
    logger.info(f"[LTR Passages] MRR: {avg_mrr:.4f}")
    logger.info(
        f"[LTR Passages] Precision@1: {avg_precision_at_k[1]:.4f}, @3: {avg_precision_at_k[3]:.4f}, @5: {avg_precision_at_k[5]:.4f}, @10: {avg_precision_at_k[10]:.4f}"
    )
    logger.info(
        f"[LTR Passages] NDCG@1: {avg_ndcg1:.4f}, @3: {avg_ndcg3:.4f}, @5: {avg_ndcg5:.4f}"
    )

    return {
        "accuracy": ltr_accuracy_val,
        "mrr": avg_mrr,
        "precision_at_k": avg_precision_at_k,
        "ndcg@1": avg_ndcg1,
        "ndcg@3": avg_ndcg3,
        "ndcg@5": avg_ndcg5,
        "y_true": y_true_ltr,
        "y_pred": y_pred_ltr,
        "topic_analysis": topic_analysis,
    }


def prepare_tfidf_passages(
    passages: List[Document],
    statements: List[List[str]],
    normalize: bool,
    remove_stopwords: bool,
    use_stemming: bool,
) -> Tuple[TfidfVectorizer, List[str], np.ndarray, List[str]]:
    """
    Prepares TF-IDF vectorizer and matrices for passages and statements.
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

    passage_texts = [
        " ".join(prep_content(passage.page_content))
        for passage in tqdm(passages, desc="Preparing passage texts for TF-IDF")
    ]
    statement_texts = [" ".join(tokens) for tokens in statements]

    tfidf = TfidfVectorizer()
    tfidf.fit(passage_texts + statement_texts)
    passage_tfidf = tfidf.transform(passage_texts)

    return tfidf, passage_texts, passage_tfidf, statement_texts


def train_ranker(
    X: np.ndarray,
    y: np.ndarray,
    group: List[int],
    early_stop_frac: float,
    random_state: int,
) -> Tuple[lgb.LGBMRanker, StandardScaler]:
    """
    Train LightGBM LambdaMART ranker (same as original).
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
        learning_rate=0.01,
        n_estimators=1000,
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
def test_ltr_passages(config: PassageLTRConfig) -> Dict[str, Any]:
    """Train on data/{train_source} and validate on data/{val_source} using passages."""
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

    # Create passage corpus
    passages = create_passage_corpus(docs, config)

    # Log passage statistics
    passage_types: Dict[str, int] = {}
    topics_in_passages: Dict[int, int] = {}
    for passage in passages:
        ptype = passage.metadata.get("passage_type", "unknown")
        passage_types[ptype] = passage_types.get(ptype, 0) + 1

        topic = passage.metadata.get("topic_id", -1)
        topics_in_passages[topic] = topics_in_passages.get(topic, 0) + 1

    logger.info(f"Passage types: {passage_types}")
    logger.info(f"Number of unique topics in passages: {len(topics_in_passages)}")

    # Preprocess passages for BM25
    logger.info("Preprocessing passages for BM25...")
    if config.normalize:
        texts_processed = [
            preprocess_func(
                normalize_medical_text(passage.page_content),
                remove_stopwords=config.remove_stopwords,
                use_stemming=config.use_stemming,
            )
            for passage in tqdm(passages, desc="Processing passages for BM25")
        ]
    else:
        texts_processed = [
            preprocess_func(
                passage.page_content,
                remove_stopwords=config.remove_stopwords,
                use_stemming=config.use_stemming,
            )
            for passage in tqdm(passages, desc="Processing passages for BM25")
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

    # BM25 baseline on validation using passages
    bm25 = rank_bm25.BM25Okapi(texts_processed, k1=config.k1, b=config.b)
    y_true_bm25_val, y_pred_bm25_val = get_retrieval_predictions(
        bm25, passages, processed_statements_val, true_labels_val
    )
    bm25_accuracy_val = accuracy_score(y_true_bm25_val, y_pred_bm25_val)
    logger.info(
        f"[BM25 baseline on passages ({config.val_source})] Top-1 accuracy: {bm25_accuracy_val:.4f}"
    )

    # Prepare TF-IDF with training statements and passages
    tfidf, _, passage_tfidf, statement_texts_train = prepare_tfidf_passages(
        passages,
        processed_statements_train,
        normalize=config.normalize,
        remove_stopwords=config.remove_stopwords,
        use_stemming=config.use_stemming,
    )

    # Precompute passage tokens for Jaccard similarity
    logger.info("Precomputing passage tokens for efficiency...")
    passage_token_cache = precompute_passage_tokens(
        passages,
        normalize=config.normalize,
        remove_stopwords=config.remove_stopwords,
        use_stemming=config.use_stemming,
    )

    rng = np.random.default_rng(config.random_state)
    X_rows = []
    y_rows = []
    group = []

    logger.info("Building training candidate pairs...")
    for stmt_tokens, true_topic in tqdm(
        zip(processed_statements_train, true_labels_train),
        desc="Building training pairs",
        total=len(processed_statements_train),
    ):
        feature_vecs, labels, _ = build_candidate_pairs_passages(
            passages=passages,
            statement_tokens=stmt_tokens,
            true_topic=true_topic,
            bm25=bm25,
            tfidf_vectorizer=tfidf,
            passage_tfidf_matrix=passage_tfidf,
            rng=rng,
            negative_per_query=config.negative_per_query,
            normalize=config.normalize,
            remove_stopwords=config.remove_stopwords,
            use_stemming=config.use_stemming,
            passage_token_cache=passage_token_cache,
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

    # Evaluate LTR on passages
    ltr_eval = evaluate_ltr_passages(
        ranker=ranker,
        scaler=scaler,
        passages=passages,
        processed_statements_val=processed_statements_val,
        true_labels_val=true_labels_val,
        bm25=bm25,
        tfidf=tfidf,
        passage_tfidf=passage_tfidf,
        config=config,
    )

    return {
        "config": config,
        "passage_stats": {
            "total_passages": len(passages),
            "passage_types": passage_types,
            "topics_in_passages": len(topics_in_passages),
        },
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
    # Test different passage configurations
    configs = [
        PassageLTRConfig(
            train_source="synthetic",
            val_source="train",
            k1=2.5,
            b=1.0,
            normalize=False,
            remove_stopwords=True,
            use_stemming=False,
            negative_per_query=5,
            random_state=42,
            passage_size=150,
            passage_overlap=50,
            use_section_splitting=True,
        ),
        PassageLTRConfig(
            train_source="synthetic",
            val_source="train",
            k1=2.5,
            b=1.0,
            normalize=False,
            remove_stopwords=True,
            use_stemming=False,
            negative_per_query=5,
            random_state=42,
            passage_size=100,
            passage_overlap=25,
            use_section_splitting=False,
        ),
    ]

    for i, config in enumerate(configs):
        logger.info(f"\n{'='*60}")
        logger.info(
            f"Testing configuration {i+1}: passage_size={config.passage_size}, overlap={config.passage_overlap}, section_splitting={config.use_section_splitting}"
        )
        logger.info(f"{'='*60}")

        try:
            results = test_ltr_passages(config)

            # Print summary
            print(f"\n--- Configuration {i+1} Results ---")
            print(f"Total passages: {results['passage_stats']['total_passages']}")
            print(f"Passage types: {results['passage_stats']['passage_types']}")
            print(
                f"BM25 baseline accuracy: {results['bm25_baseline_val']['accuracy']:.4f}"
            )
            print(f"LTR accuracy: {results['ltr']['accuracy']:.4f}")
            print(f"LTR MRR: {results['ltr']['mrr']:.4f}")
            print(f"LTR Precision@1: {results['ltr']['precision_at_k'][1]:.4f}")
            print(f"LTR Precision@5: {results['ltr']['precision_at_k'][5]:.4f}")
            print(f"LTR NDCG@5: {results['ltr']['ndcg@5']:.4f}")

        except Exception as e:
            logger.error(f"Error with configuration {i+1}: {e}")
            continue


if __name__ == "__main__":
    main()
