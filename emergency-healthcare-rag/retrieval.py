import json
import os
from pathlib import Path
from collections import defaultdict

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import EnsembleRetriever
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings

from sklearn.metrics import accuracy_score, classification_report


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


def build_retrievers(docs):
    # Optional splitting if docs are large; here we assume they are manageable
    # embeddings - using sentence-transformers for quick start
    embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")

    # Dense vector store
    vectorstore = Chroma.from_documents(docs, embeddings)  # in-memory
    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # Sparse BM25 retriever
    bm25_retriever = BM25Retriever.from_documents(docs, k=5)

    # Ensemble: adjust weights as needed
    ensemble = EnsembleRetriever(
        retrievers=[bm25_retriever, dense_retriever],
        weights=[0.5, 0.5],  # 50/50 blend; tune this
    )
    return ensemble


def evaluate_topic_retrieval(
    ensemble, statements_dir: Path, answers_dir: Path, id2topic: dict
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

        # Retrieval
        retrieved = ensemble.invoke(statement_text)
        if len(retrieved) == 0:
            # fallback: predict nothing
            y_true.append(true_topic_id)
            y_pred.append(-1)
            continue
        top = retrieved[0]
        pred_topic_id = top.metadata.get("topic_id", -1)
        y_true.append(true_topic_id)
        y_pred.append(pred_topic_id)

    # Filter out any predictions that are -1 if desired (here we include them)
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Total evaluated statements: {len(y_true)}")
    if missing:
        print(
            f"Warning: missing answer JSONs for {len(missing)} statements (examples: {missing[:5]})"
        )
    print(f"\nOverall topic prediction accuracy: {accuracy:.4f}")

    return {
        "accuracy": accuracy,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def main():
    base = Path("data")
    topics_json = base / "topics.json"
    cleaned_root = base / "cleaned_topics"  # adjust if your folder is named differently
    statements_dir = base / "train" / "statements"
    answers_dir = base / "train" / "answers"

    topic2id, id2topic = load_topics(topics_json)
    print(f"Loaded {len(topic2id)} topics from {topics_json}")

    documents = load_cleaned_documents(cleaned_root, topic2id)
    if not documents:
        raise RuntimeError(
            "No documents loaded; check paths and topics.json consistency."
        )
    print(f"Loaded {len(documents)} documents from {cleaned_root}")

    # Verify statement and answer directories exist
    if not statements_dir.exists():
        raise RuntimeError(f"Statements directory not found: {statements_dir}")
    if not answers_dir.exists():
        raise RuntimeError(f"Answers directory not found: {answers_dir}")

    stmt_files = list(statements_dir.glob("statement_*.txt"))
    answer_files = list(answers_dir.glob("statement_*.json"))
    print(
        f"Found {len(stmt_files)} statement files and {len(answer_files)} answer files"
    )

    ensemble = build_retrievers(documents)
    evaluate_topic_retrieval(ensemble, statements_dir, answers_dir, id2topic)


if __name__ == "__main__":
    main()
