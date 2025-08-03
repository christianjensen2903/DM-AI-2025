import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers import EnsembleRetriever
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics import accuracy_score
from utils import load_topics, load_cleaned_documents
from text_normalizer import normalize_medical_text


def length_fn(text: str) -> int:
    return len(text.split())


def split_documents_by_sections(docs):
    """
    Split documents by markdown section headers (##). Short sections are kept as-is;
    long ones get further split with RecursiveCharacterTextSplitter.
    """
    split_docs = []

    fallback_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=25,
        length_function=length_fn,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    header_pattern = re.compile(r"(## .*)")

    def _process_section(text, base_metadata, header):
        text = text.strip()
        metadata = {
            **base_metadata,
            "section_header": header,
            "section_type": "content",
        }

        # Only split if it's longer than chunk_size, otherwise keep as one piece
        if length_fn(text) <= fallback_splitter.chunk_size:
            doc = Document(page_content=text, metadata=metadata)
            doc.metadata["subsection_index"] = 0
            doc.metadata["total_subsections"] = 1
            return [doc]

        subsections = fallback_splitter.split_documents(
            [Document(page_content=text, metadata=metadata)]
        )
        for i, sub in enumerate(subsections):
            sub.metadata["subsection_index"] = i
            sub.metadata["total_subsections"] = len(subsections)
        return subsections

    for doc in docs:
        parts = header_pattern.split(doc.page_content)
        if len(parts) == 1:  # no headers found
            split_docs.append(doc)
            continue

        sections = []

        # preamble before the first header (if any)
        if parts[0].strip():
            sections.extend(_process_section(parts[0], doc.metadata, ""))

        # pair up headers with their following content
        for i in range(1, len(parts), 2):
            header = parts[i].strip()
            content = parts[i + 1] if i + 1 < len(parts) else ""
            section_text = f"{header}\n{content}"
            sections.extend(_process_section(section_text, doc.metadata, header))

        split_docs.extend(sections if sections else [doc])

    return split_docs


def build_retrievers(docs):
    # Split documents by sections for better semantic coherence
    split_docs = split_documents_by_sections(docs)
    print(f"Split {len(docs)} documents into {len(split_docs)} section-based chunks")

    # embeddings - using sentence-transformers for quick start
    # embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")

    # # Dense vector store using chunked documents
    # vectorstore = Chroma.from_documents(split_docs, embeddings)
    # dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # Sparse BM25 retriever using chunked documents
    bm25_retriever = BM25Retriever.from_documents(
        split_docs,
        k=5,
        bm25_params={"k1": 2.5, "b": 0.25},
    )
    return bm25_retriever

    # Ensemble: adjust weights as needed
    # ensemble = EnsembleRetriever(
    #     retrievers=[bm25_retriever, dense_retriever],
    #     weights=[0.75, 0.25],
    # )
    # return ensemble


def evaluate_topic_retrieval(
    ensemble, statements_dir: Path, answers_dir: Path, normalize: bool = False
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

        if normalize:
            statement_text = normalize_medical_text(statement_text)

        # Retrieval
        retrieved = ensemble.invoke(statement_text)
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

    normalize = True

    documents = load_cleaned_documents(cleaned_root, topic2id, normalize=normalize)

    ensemble = build_retrievers(documents)
    evaluate_topic_retrieval(ensemble, statements_dir, answers_dir, normalize=normalize)


if __name__ == "__main__":
    main()
