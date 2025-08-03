import json
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

from utils import load_topics, load_cleaned_documents


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
    bm25_retriever = BM25Retriever.from_documents(
        split_docs, k=5, bm25_params={"k1": 3.0, "b": 1.0}
    )

    # Ensemble: adjust weights as needed
    ensemble = EnsembleRetriever(
        retrievers=[bm25_retriever, dense_retriever],
        weights=[0.5, 0.5],  # 50/50 blend; tune this
    )
    return ensemble


def evaluate_topic_retrieval(ensemble, statements_dir: Path, answers_dir: Path):
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

    documents = load_cleaned_documents(cleaned_root, topic2id)

    ensemble = build_retrievers(documents)
    evaluate_topic_retrieval(ensemble, statements_dir, answers_dir)


if __name__ == "__main__":
    main()
