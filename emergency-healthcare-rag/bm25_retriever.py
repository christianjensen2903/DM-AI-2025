import json
from pathlib import Path
from typing import Any
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from sklearn.metrics import accuracy_score

from utils import load_topics, load_cleaned_documents
from normalized_retrievers import create_normalized_bm25_retriever
from text_normalizer import normalize_medical_text
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import rank_bm25

nltk.download("punkt_tab")
nltk.download("stopwords")


def evaluate_topic_retrieval(
    retriever: rank_bm25.BM25,
    docs: list[Document],
    statements_dir: Path,
    answers_dir: Path,
):
    y_true = []
    y_pred = []
    missing = []
    for stmt_path in sorted(statements_dir.glob("statement_*.txt")):
        base = stmt_path.stem  # e.g., "statement_0001"
        answer_path = answers_dir / f"{base}.json"
        if not answer_path.exists():
            missing.append(base)
            continue
        statement_text = stmt_path.read_text(encoding="utf-8")
        with answer_path.open("r", encoding="utf-8") as f:
            answer = json.load(f)
        true_topic_id = answer.get("statement_topic")
        if true_topic_id is None:
            continue

        # Retrieval with normalized query
        # Note: The normalized retriever will handle query normalization internally
        statement_text_processed = preprocess_func(statement_text)
        retrieved = retriever.get_top_n(statement_text_processed, docs, n=1)
        if len(retrieved) == 0:
            # fallback: predict nothing
            y_true.append(true_topic_id)
            y_pred.append(-1)
            continue

        # Take the top 1 result as the predicted topic
        pred_topic_id = retrieved[0].metadata.get("topic_id", -1)

        y_true.append(true_topic_id)
        y_pred.append(pred_topic_id)

    # Filter out any predictions that are -1 if desired (here we include them)
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Total evaluated statements: {len(y_true)}")
    print(f"\nOverall topic prediction accuracy: {accuracy:.4f}")

    return {
        "accuracy": accuracy,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def preprocess_func(
    text: str, remove_stopwords: bool = False, use_stemming: bool = False
) -> list[str]:
    """
    Preprocess text by tokenizing and optionally removing stopwords and stemming.

    Args:
        text: Input text to preprocess
        remove_stopwords: Whether to remove English stopwords
        use_stemming: Whether to apply Porter stemming

    Returns:
        List of processed tokens
    """
    # Tokenize the text
    tokens = word_tokenize(text.lower())

    # Remove stopwords if requested
    if remove_stopwords:
        stop_words = set(stopwords.words("english"))
        tokens = [token for token in tokens if token not in stop_words]

    # Apply stemming if requested
    if use_stemming:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(token) for token in tokens]

    return tokens


def test_bm25_params(
    k1: float = 1.5,
    b: float = 0.75,
    normalize: bool = False,
    remove_stopwords: bool = False,
    use_stemming: bool = False,
):
    """Test BM25 with specific k1 and b parameters, and text preprocessing options."""
    base = Path("data")
    topics_json = base / "topics.json"
    cleaned_root = base / "cleaned_topics"
    statements_dir = base / "synthetic" / "statements"
    answers_dir = base / "synthetic" / "answers"

    topic2id, _ = load_topics(topics_json)

    # Load documents
    docs = load_cleaned_documents(cleaned_root, topic2id, normalize=normalize)
    print(
        f"Loaded {len(docs)} documents (normalize={normalize}, remove_stopwords={remove_stopwords}, use_stemming={use_stemming})"
    )

    # Create BM25Plus retriever with specified parameters
    if normalize:
        from text_normalizer import normalize_medical_text

        texts_processed = [
            preprocess_func(
                normalize_medical_text(t.page_content), remove_stopwords, use_stemming
            )
            for t in docs
        ]
    else:
        texts_processed = [
            preprocess_func(t.page_content, remove_stopwords, use_stemming)
            for t in docs
        ]

    retriever = rank_bm25.BM25Okapi(corpus=texts_processed, k1=k1, b=b)

    # Temporarily modify the global preprocess_func to handle the current configuration
    global preprocess_func
    original_preprocess = preprocess_func

    def configured_preprocess_func(text: str) -> list[str]:
        if normalize:
            from text_normalizer import normalize_medical_text

            text = normalize_medical_text(text, is_query=True)
        return original_preprocess(text, remove_stopwords, use_stemming)

    preprocess_func = configured_preprocess_func  # type: ignore[assignment]

    try:
        result = evaluate_topic_retrieval(retriever, docs, statements_dir, answers_dir)
    finally:
        # Restore original preprocess function
        preprocess_func = original_preprocess

    return result


def main():
    test_bm25_params(k1=3, b=1, normalize=True)


if __name__ == "__main__":
    main()
