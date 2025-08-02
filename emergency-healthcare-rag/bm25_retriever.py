import json
from pathlib import Path
from langchain_community.retrievers import BM25Retriever
from sklearn.metrics import accuracy_score

from utils import load_topics, load_cleaned_documents
from normalized_retrievers import create_normalized_bm25_retriever
from text_normalizer import normalize_medical_text


def evaluate_topic_retrieval(retriever, statements_dir: Path, answers_dir: Path):
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
        retrieved = retriever.invoke(statement_text)
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


def main():
    base = Path("data")
    topics_json = base / "topics.json"
    cleaned_root = base / "cleaned_topics"
    statements_dir = base / "synthetic" / "statements"
    answers_dir = base / "synthetic" / "answers"

    topic2id, _ = load_topics(topics_json)

    # Load documents with normalization enabled
    docs = load_cleaned_documents(cleaned_root, topic2id, normalize=False)
    print(f"Loaded {len(docs)} normalized documents")

    # Use normalized BM25 retriever
    # retriever = create_normalized_bm25_retriever(docs, k=1)
    retriever = BM25Retriever.from_documents(docs, k=1)
    print("Created normalized BM25 retriever")

    evaluate_topic_retrieval(retriever, statements_dir, answers_dir)


if __name__ == "__main__":
    main()
