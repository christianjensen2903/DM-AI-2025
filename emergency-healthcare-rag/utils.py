import json
from pathlib import Path

from langchain_core.documents import Document
from text_normalizer import normalize_medical_text


def validate_prediction(statement_is_true: int, statement_topic: int):
    """Validate that prediction values are in correct format"""
    assert isinstance(
        statement_is_true, int
    ), f"statement_is_true must be int, got {type(statement_is_true)}"
    assert isinstance(
        statement_topic, int
    ), f"statement_topic must be int, got {type(statement_topic)}"
    assert statement_is_true in [
        0,
        1,
    ], f"statement_is_true must be 0 or 1, got {statement_is_true}"
    assert (
        0 <= statement_topic <= 114
    ), f"statement_topic must be between 0-114, got {statement_topic}"


def load_topics(topics_json_path: Path):
    with topics_json_path.open("r", encoding="utf-8") as f:
        topic2id = json.load(f)
    id2topic = {v: k for k, v in topic2id.items()}
    return topic2id, id2topic


def load_cleaned_documents(
    root: Path, topic2id: dict, normalize: bool = True
) -> list[Document]:
    """
    Walks data/cleaned_topics/<topic_name>/*.md and returns list of Documents with metadata.

    Args:
        root: Path to the cleaned topics directory
        topic2id: Mapping from topic names to topic IDs
        normalize: Whether to apply medical text normalization
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
                original_text = md_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                original_text = md_path.read_text(encoding="latin-1")

            # Apply normalization if requested
            if normalize:
                normalized_text = normalize_medical_text(original_text, is_query=False)
            else:
                normalized_text = original_text

            metadata = {
                "source": str(md_path),
                "topic_name": topic_name,
                "topic_id": topic_id,
                "original_content": original_text,  # Preserve original for reference
                "normalized": normalize,
            }
            docs.append(Document(page_content=normalized_text, metadata=metadata))
    return docs
