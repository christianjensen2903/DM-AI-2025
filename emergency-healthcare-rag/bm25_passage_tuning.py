from pathlib import Path
import json
import csv
import random
from itertools import product
from typing import Tuple, List, Dict, Any
from collections import Counter

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.metrics import accuracy_score
import rank_bm25

from langchain_core.documents import Document
from utils import load_topics, load_cleaned_documents
from text_normalizer import normalize_medical_text

# Ensure NLTK data exists
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

# --- Hyperparameter grid (only these are tuned) ---
K1_VALUES = [1.2, 2.5, 3.0]
B_VALUES = [0.25, 0.5, 0.75, 1.0]
TOP_NS = [1]

# Fixed (non-tuned) options
NORMALIZE = True
REMOVE_STOPWORDS = False
USE_STEMMING = False
PASSAGE_SIZE = 100
OVERLAP = 25

# Random sampling over (k1, b, top_n) triplets if desired
RANDOM_SEARCH = None  # e.g., 10 to sample 10 random configs
SEED = 42

# Paths / other globals
BASE_DIR = Path("data")
OUTPUT_PATH = Path("bm25_tuning_results.csv")


# --- Preprocessing helpers ---
def preprocess_func(
    text: str, remove_stopwords: bool = False, use_stemming: bool = False
) -> List[str]:
    tokens = word_tokenize(text.lower())
    if remove_stopwords:
        stop_words = set(stopwords.words("english"))
        tokens = [token for token in tokens if token not in stop_words]
    if use_stemming:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(token) for token in tokens]
    return tokens


def split_text_to_passages(
    text: str, passage_size: int = 200, overlap: int = 50
) -> List[str]:
    tokens = word_tokenize(text.lower())
    if passage_size <= 0:
        raise ValueError("passage_size must be positive")
    if overlap < 0 or overlap >= passage_size:
        raise ValueError("overlap must be >=0 and < passage_size")
    step = passage_size - overlap
    passages = []
    for start in range(0, max(1, len(tokens)), step):
        end = start + passage_size
        chunk_tokens = tokens[start:end]
        if not chunk_tokens:
            break
        passage = " ".join(chunk_tokens)
        passages.append(passage)
        if end >= len(tokens):
            break
    return passages


def build_passage_corpus(
    docs: List[Document],
    passage_size: int,
    overlap: int,
    normalize: bool,
    remove_stopwords: bool,
    use_stemming: bool,
) -> Tuple[List[List[str]], List[Document]]:
    corpus_tokenized: List[List[str]] = []
    passage_docs: List[Document] = []

    for doc_idx, doc in enumerate(docs):
        content = doc.page_content
        if normalize:
            content = normalize_medical_text(content)
        passages = split_text_to_passages(
            content, passage_size=passage_size, overlap=overlap
        )
        original_doc_id = doc.metadata.get("doc_id", f"doc_{doc_idx}")
        topic_id = doc.metadata.get("topic_id", -1)
        for p_idx, passage_text in enumerate(passages):
            tokens = preprocess_func(
                passage_text,
                remove_stopwords=remove_stopwords,
                use_stemming=use_stemming,
            )
            corpus_tokenized.append(tokens)
            passage_metadata: Dict[str, Any] = {
                "topic_id": topic_id,
                "original_doc_id": original_doc_id,
                "passage_id": f"{original_doc_id}_passage_{p_idx}",
            }
            passage_doc = Document(page_content=passage_text, metadata=passage_metadata)
            passage_docs.append(passage_doc)
    return corpus_tokenized, passage_docs


def preprocess_statements(
    statements_dir: Path,
    answers_dir: Path,
    normalize: bool = False,
    remove_stopwords: bool = True,
    use_stemming: bool = True,
) -> Tuple[List[List[str]], List[int], List[Path]]:
    processed_statements = []
    true_labels: List[int] = []
    statement_paths = []

    for stmt_path in sorted(statements_dir.glob("statement_*.txt")):
        base = stmt_path.stem
        answer_path = answers_dir / f"{base}.json"
        if not answer_path.exists():
            continue
        statement_text = stmt_path.read_text(encoding="utf-8")
        with answer_path.open("r", encoding="utf-8") as f:
            answer = json.load(f)
        true_topic_id = answer.get("statement_topic", -1)
        if normalize:
            statement_text = normalize_medical_text(statement_text, is_query=True)
        statement_text_processed = preprocess_func(
            statement_text,
            remove_stopwords=remove_stopwords,
            use_stemming=use_stemming,
        )
        processed_statements.append(statement_text_processed)
        true_labels.append(true_topic_id)
        statement_paths.append(stmt_path)
    return processed_statements, true_labels, statement_paths


def get_retrieval_predictions(
    retriever: rank_bm25.BM25Okapi,
    passage_docs: List[Document],
    processed_statements: List[List[str]],
    true_labels: List[int],
    n: int = 1,
) -> Tuple[List[int], List[int]]:
    y_true = []
    y_pred = []
    for statement_tokens, true_topic_id in zip(processed_statements, true_labels):
        retrieved = retriever.get_top_n(statement_tokens, passage_docs, n=n)
        if not retrieved:
            y_true.append(true_topic_id)
            y_pred.append(-1)
            continue
        top_topic_ids = [
            doc.metadata.get("topic_id", -1)
            for doc in retrieved
            if doc.metadata.get("topic_id", -1) != -1
        ]
        if not top_topic_ids:
            pred_topic_id = -1
        elif len(top_topic_ids) == 1 or n == 1:
            pred_topic_id = top_topic_ids[0]
        else:
            counts = Counter(top_topic_ids)
            most_common = counts.most_common()
            if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
                pred_topic_id = retrieved[0].metadata.get("topic_id", -1)
            else:
                pred_topic_id = most_common[0][0]
        y_true.append(true_topic_id)
        y_pred.append(pred_topic_id)
    return y_true, y_pred


def evaluate_config_simple(
    k1: float,
    b: float,
    top_n: int,
    passage_corpus_tokens: List[List[str]],
    passage_docs: List[Document],
    processed_statements: List[List[str]],
    true_labels: List[int],
) -> Dict[str, Any]:
    retriever = rank_bm25.BM25Okapi(corpus=passage_corpus_tokens, k1=k1, b=b)
    y_true, y_pred = get_retrieval_predictions(
        retriever,
        passage_docs,
        processed_statements,
        true_labels,
        n=top_n,
    )
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    return {
        "k1": k1,
        "b": b,
        "top_n": top_n,
        "accuracy": accuracy,
        "total_evaluated": len(y_true),
    }


def main():
    # Load topics & documents once
    topic2id, _ = load_topics(BASE_DIR / "topics.json")
    full_docs = load_cleaned_documents(
        BASE_DIR / "cleaned_topics", topic2id, normalize=False
    )

    # Build passage corpus once with fixed preprocessing settings
    passage_corpus_tokens, passage_docs = build_passage_corpus(
        full_docs,
        passage_size=PASSAGE_SIZE,
        overlap=OVERLAP,
        normalize=NORMALIZE,
        remove_stopwords=REMOVE_STOPWORDS,
        use_stemming=USE_STEMMING,
    )

    # Preprocess statements once with fixed settings
    statements_dir = BASE_DIR / "synthetic" / "statements"
    answers_dir = BASE_DIR / "synthetic" / "answers"
    processed_statements, true_labels, _ = preprocess_statements(
        statements_dir,
        answers_dir,
        normalize=NORMALIZE,
        remove_stopwords=REMOVE_STOPWORDS,
        use_stemming=USE_STEMMING,
    )

    # Build tuning grid over just k1, b, top_n
    full_grid = list(product(K1_VALUES, B_VALUES, TOP_NS))
    if RANDOM_SEARCH is not None:
        random.seed(SEED)
        if RANDOM_SEARCH > len(full_grid):
            print(
                f"Requested RANDOM_SEARCH={RANDOM_SEARCH} exceeds total combinations {len(full_grid)}; using full grid instead."
            )
            search_space = full_grid
        else:
            search_space = random.sample(full_grid, RANDOM_SEARCH)
    else:
        search_space = full_grid

    print(f"Total configurations to evaluate: {len(search_space)}")
    best: Dict[str, Any] = None

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "k1",
            "b",
            "top_n",
            "accuracy",
            "total_evaluated",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for idx, (k1, b, top_n) in enumerate(search_space, start=1):
            print(
                f"[{idx}/{len(search_space)}] Evaluating k1={k1}, b={b}, top_n={top_n}"
            )
            result = evaluate_config_simple(
                k1,
                b,
                top_n,
                passage_corpus_tokens,
                passage_docs,
                processed_statements,
                true_labels,
            )
            writer.writerow({k: result[k] for k in fieldnames})
            csvfile.flush()
            if best is None or result["accuracy"] > best["accuracy"]:
                best = result.copy()

    print("=== Best configuration ===")
    print(
        f"k1={best['k1']}, b={best['b']}, top_n={best['top_n']} -> accuracy={best['accuracy']:.4f} (n={best['total_evaluated']})"
    )
    print(f"Results saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
