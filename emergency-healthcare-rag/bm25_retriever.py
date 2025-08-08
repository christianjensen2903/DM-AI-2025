import json
from pathlib import Path
from langchain_core.documents import Document
from sklearn.metrics import accuracy_score

from utils import load_topics, load_cleaned_documents
from text_normalizer import normalize_medical_text
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import rank_bm25
from retrieval import build_retrievers

nltk.download("punkt_tab")
nltk.download("stopwords")


def preprocess_statements(
    statements_dir: Path,
    answers_dir: Path,
    normalize: bool = False,
    remove_stopwords: bool = True,
    use_stemming: bool = True,
) -> tuple[list[list[str]], list[int], list[Path]]:
    """
    Pre-process all statements to avoid redundant processing during parameter tuning.

    Returns:
        Tuple of (processed_statements, true_labels, statement_paths)
    """
    processed_statements = []
    true_labels: list[int] = []
    statement_paths = []

    for stmt_path in sorted(statements_dir.glob("statement_*.txt")):
        base = stmt_path.stem  # e.g., "statement_0001"
        answer_path = answers_dir / f"{base}.json"
        if not answer_path.exists():
            continue

        statement_text = stmt_path.read_text(encoding="utf-8")
        with answer_path.open("r", encoding="utf-8") as f:
            answer = json.load(f)
        true_topic_id = answer.get("statement_topic", -1)

        processed_statements.append(statement_text)
        true_labels.append(true_topic_id)
        statement_paths.append(stmt_path)

    return processed_statements, true_labels, statement_paths


def get_retrieval_predictions(
    retriever,
    docs: list[Document],
    processed_statements: list[list[str]],
    true_labels: list[int],
    n: int = 1,
) -> tuple[list[int], list[int]]:
    """
    Get retrieval predictions using the specified retriever.

    Args:
        retriever: The retriever to use (e.g., BM25Okapi instance)
        docs: List of documents (for metadata access)
        processed_statements: Pre-processed statement texts
        true_labels: True topic labels for statements
        n: Number of top results to retrieve (default: 1)

    Returns:
        Tuple of (y_true, y_pred) lists
    """
    y_true = []
    y_pred = []

    for statement_text_processed, true_topic_id in zip(
        processed_statements, true_labels
    ):
        # Retrieve top document(s)
        retrieved = retriever.invoke(statement_text_processed)
        if len(retrieved) == 0:
            # fallback: predict nothing
            y_true.append(true_topic_id)
            y_pred.append(-1)
            continue

        # Take the top 1 result as the predicted topic
        pred_topic_id = retrieved[0].metadata.get("topic_id", -1)
        y_true.append(true_topic_id)
        y_pred.append(pred_topic_id)

    return y_true, y_pred


def evaluate_bm25_config(
    k1: float,
    b: float,
    docs: list[Document],
    processed_statements: list[list[str]],
    true_labels: list[int],
) -> dict:
    """
    Evaluate a specific k1, b configuration for BM25 using pre-processed texts.

    Args:
        k1: BM25 k1 parameter
        b: BM25 b parameter
        docs: List of documents (for metadata access)
        texts_processed: Pre-processed document texts
        processed_statements: Pre-processed statement texts
        true_labels: True topic labels for statements

    Returns:
        Dictionary with evaluation results
    """
    retriever = build_retrievers(docs, k1, b)

    # Get predictions using the extracted function
    y_true, y_pred = get_retrieval_predictions(
        retriever, docs, processed_statements, true_labels
    )

    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)

    return {
        "k1": k1,
        "b": b,
        "accuracy": accuracy,
        "total_evaluated": len(y_true),
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

    result = evaluate_bm25_config(
        k1,
        b,
        docs,
        texts_processed,
        processed_statements,
        true_labels,
    )
    print(f"Accuracy: {result['accuracy']:.4f}")


def main():
    test_bm25_params(k1=2.5, b=1.0, normalize=False, remove_stopwords=True)


if __name__ == "__main__":
    main()
