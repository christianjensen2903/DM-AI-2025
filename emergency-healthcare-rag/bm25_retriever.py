import json
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from sklearn.metrics import accuracy_score

from utils import load_topics, load_cleaned_documents
from normalized_retrievers import create_normalized_bm25_retriever
from text_normalizer import normalize_medical_text
import nltk
from nltk.tokenize import word_tokenize
import rank_bm25

nltk.download("punkt_tab")


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


def preprocess_func(text: str) -> list[str]:
    return text.split()


def test_bm25_params(k1: float = 1.5, b: float = 0.75, normalize: bool = False):
    """Test BM25 with specific k1 and b parameters."""
    base = Path("data")
    topics_json = base / "topics.json"
    cleaned_root = base / "cleaned_topics"
    statements_dir = base / "synthetic" / "statements"
    answers_dir = base / "synthetic" / "answers"

    topic2id, _ = load_topics(topics_json)

    # Load documents
    docs = load_cleaned_documents(cleaned_root, topic2id, normalize=normalize)
    print(f"Loaded {len(docs)} documents (normalize={normalize})")

    # Create BM25Plus retriever with specified parameters
    if normalize:
        from text_normalizer import normalize_medical_text

        texts_processed = [
            preprocess_func(normalize_medical_text(t.page_content)) for t in docs
        ]
    else:
        texts_processed = [preprocess_func(t.page_content) for t in docs]

    retriever = rank_bm25.BM25Plus(corpus=texts_processed, k1=k1, b=b)
    print(f"Created BM25Plus retriever with k1={k1}, b={b}")

    # Update the global preprocess_func to handle normalization if enabled
    global preprocess_func
    if normalize:
        from text_normalizer import normalize_medical_text

        original_preprocess = preprocess_func

        def normalized_preprocess_func(text: str) -> list[str]:
            return original_preprocess(normalize_medical_text(text, is_query=True))

        preprocess_func = normalized_preprocess_func

    result = evaluate_topic_retrieval(retriever, docs, statements_dir, answers_dir)

    # Restore original preprocess function
    if normalize:
        preprocess_func = original_preprocess

    return result


def main():
    """Main function with optimized parameters."""
    print("Testing BM25 with default parameters...")
    test_bm25_params()

    print("\n" + "=" * 50)
    print("To run hyperparameter tuning, use:")
    print("python hyperparameter_tuning.py")
    print("Or for quick tuning:")
    print("python quick_tune.py")
    print("=" * 50)


if __name__ == "__main__":
    main()
