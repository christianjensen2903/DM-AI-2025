import json
from pathlib import Path
import re
from typing import Iterable, List

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


SECTION_HEADER_RE = re.compile(r"^\s*(##\s+.*)$")


def split_documents_by_sections(
    docs: Iterable["Document"],
    chunk_size: int = 100,
    chunk_overlap: int = 25,
    header_pattern: re.Pattern = SECTION_HEADER_RE,
) -> List["Document"]:
    """
    Split documents by markdown section headers (lines starting with '## ').
    Sections longer than chunk_size are further split using RecursiveCharacterTextSplitter.
    Metadata is preserved and annotated with:
      - section_header: the most recent header line (including the '## ')
      - section_type: "content"
      - subsection_index / total_subsections: if the section was further split
    """

    fallback_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=length_fn,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    def _finalize_section(
        header: str, lines: List[str], base_meta: dict
    ) -> List["Document"]:
        content = "\n".join(lines).strip()
        if not content:
            return []
        section_doc = Document(
            page_content=content,
            metadata={
                **base_meta,
                "section_header": header,
                "section_type": "content",
            },
        )
        subsections = fallback_splitter.split_documents([section_doc])
        for i, subsection in enumerate(subsections):
            subsection.metadata["subsection_index"] = i
            subsection.metadata["total_subsections"] = len(subsections)
        return subsections

    split_docs: List["Document"] = []

    for doc in docs:
        sections: List["Document"] = []
        current_header = ""
        current_lines: List[str] = []

        for line in doc.page_content.splitlines():
            match = header_pattern.match(line)
            if match:
                # flush previous section
                sections.extend(
                    _finalize_section(current_header, current_lines, doc.metadata)
                )
                current_header = match.group(1)  # full header line, e.g., "## Section"
                current_lines = [line]  # include header in the section content
            else:
                current_lines.append(line)

        # flush the last section
        sections.extend(_finalize_section(current_header, current_lines, doc.metadata))

        if sections:
            split_docs.extend(sections)
        else:
            split_docs.append(doc)

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
