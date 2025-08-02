import json
import os
from pathlib import Path
from collections import defaultdict

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers import EnsembleRetriever
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer
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


tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")


def length_fn(text: str) -> int:
    return len(tokenizer(text, truncation=False)["input_ids"])


def split_documents_by_sections(docs):
    """
    Split documents by markdown section headers (##) to create semantically coherent chunks.
    If a section is too long, it will be further split using RecursiveCharacterTextSplitter.
    """
    split_docs = []

    # Fallback splitter for very long sections
    fallback_splitter = RecursiveCharacterTextSplitter(
        chunk_size=256,
        chunk_overlap=64,
        length_function=length_fn,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    for doc in docs:
        content = doc.page_content
        sections = []
        current_section = ""
        current_header = ""

        lines = content.split("\n")

        for line in lines:
            # Check if line is a section header (starts with ##)
            if line.strip().startswith("## "):
                # Save previous section if it exists
                if current_section.strip():
                    section_content = current_section.strip()

                    # If section is too long, split it further
                    section_doc = Document(
                        page_content=section_content,
                        metadata={
                            **doc.metadata,
                            "section_header": current_header,
                            "section_type": "content",
                        },
                    )
                    subsections = fallback_splitter.split_documents([section_doc])
                    # Add subsection info to metadata
                    for i, subsection in enumerate(subsections):
                        subsection.metadata["subsection_index"] = i
                        subsection.metadata["total_subsections"] = len(subsections)
                    sections.extend(subsections)

                # Start new section
                current_header = line.strip()
                current_section = line + "\n"
            else:
                current_section += line + "\n"

        # Don't forget the last section
        if current_section.strip():
            section_content = current_section.strip()

            # If section is too long, split it further
            section_doc = Document(
                page_content=section_content,
                metadata={
                    **doc.metadata,
                    "section_header": current_header,
                    "section_type": "content",
                },
            )
            subsections = fallback_splitter.split_documents([section_doc])
            # Add subsection info to metadata
            for i, subsection in enumerate(subsections):
                subsection.metadata["subsection_index"] = i
                subsection.metadata["total_subsections"] = len(subsections)
            sections.extend(subsections)

        # If no sections found, keep original document
        if not sections:
            split_docs.append(doc)
        else:
            split_docs.extend(sections)

    return split_docs


def build_retrievers(docs):
    # Split documents by sections for better semantic coherence
    split_docs = split_documents_by_sections(docs)
    print(f"Split {len(docs)} documents into {len(split_docs)} section-based chunks")

    # embeddings - using sentence-transformers for quick start
    embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")

    # Dense vector store using chunked documents
    vectorstore = Chroma.from_documents(split_docs, embeddings)  # in-memory
    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # Sparse BM25 retriever using chunked documents
    bm25_retriever = BM25Retriever.from_documents(split_docs, k=5)

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

        # Vote among top 3 results instead of just taking the best
        top_k = min(3, len(retrieved))
        topic_votes = defaultdict(int)

        for i in range(top_k):
            result_topic_id = retrieved[i].metadata.get("topic_id", -1)
            if result_topic_id != -1:  # Only count valid topic IDs
                topic_votes[result_topic_id] += 1

        if not topic_votes:
            # No valid topics found in top results
            pred_topic_id = -1
        else:
            # Select topic with most votes (ties go to first occurrence)
            pred_topic_id = max(topic_votes.items(), key=lambda x: x[1])[0]

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
