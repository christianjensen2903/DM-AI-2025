import json
import os
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from sklearn.metrics import accuracy_score


def load_topics(topics_json_path: Path):
    with topics_json_path.open("r", encoding="utf-8") as f:
        topic2id = json.load(f)
    id2topic = {v: k for k, v in topic2id.items()}
    return topic2id, id2topic


def load_cleaned_documents(root: Path, topic2id: dict):
    """
    Walks data/cleaned_topics/<topic_name>/*.md and returns list of Documents with metadata.
    """
    docs = []
    for topic_dir in root.iterdir():
        if not topic_dir.is_dir():
            continue
        topic_name = topic_dir.name
        topic_id = topic2id.get(topic_name)
        if topic_id is None:
            print(f"Warning: topic '{topic_name}' not found in topics.json; skipping.")
            continue
        for md_path in topic_dir.glob("*.md"):
            try:
                text = md_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                text = md_path.read_text(encoding="latin-1")
            metadata = {
                "source": str(md_path),
                "topic_name": topic_name,
                "topic_id": topic_id,
            }
            docs.append(Document(page_content=text, metadata=metadata))
    return docs


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

        # Retrieval
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
    statements_dir = base / "train" / "statements"
    answers_dir = base / "train" / "answers"

    topic2id, _ = load_topics(topics_json)
    docs = load_cleaned_documents(cleaned_root, topic2id)

    retriever = BM25Retriever.from_documents(docs, k=1)
    evaluate_topic_retrieval(retriever, statements_dir, answers_dir)


if __name__ == "__main__":
    main()
