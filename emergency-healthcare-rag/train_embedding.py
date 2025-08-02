from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from torch.utils.data import DataLoader
import json
from pathlib import Path
import random
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


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


def split_documents_by_sections(docs, max_section_length=3000):
    """
    Split documents by markdown section headers (##) to create semantically coherent chunks.
    If a section is too long, it will be further split using RecursiveCharacterTextSplitter.
    """
    split_docs = []

    # Fallback splitter for very long sections
    fallback_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        length_function=len,
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
                    if len(section_content) > max_section_length:
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
                    else:
                        section_doc = Document(
                            page_content=section_content,
                            metadata={
                                **doc.metadata,
                                "section_header": current_header,
                                "section_type": "content",
                            },
                        )
                        sections.append(section_doc)

                # Start new section
                current_header = line.strip()
                current_section = line + "\n"
            else:
                current_section += line + "\n"

        # Don't forget the last section
        if current_section.strip():
            section_content = current_section.strip()

            # If section is too long, split it further
            if len(section_content) > max_section_length:
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
            else:
                section_doc = Document(
                    page_content=section_content,
                    metadata={
                        **doc.metadata,
                        "section_header": current_header,
                        "section_type": "content",
                    },
                )
                sections.append(section_doc)

        # If no sections found, keep original document
        if not sections:
            split_docs.append(doc)
        else:
            split_docs.extend(sections)

    return split_docs


model = SentenceTransformer("all-mpnet-base-v2")

base = Path("data")
topic2id, id2topic = load_topics(base / "topics.json")

# 3. Load and split documents (same splitting you use in retrieval) into memory
cleaned_root = base / "cleaned_topics"
documents = load_cleaned_documents(cleaned_root, topic2id)
print(f"Loaded {len(documents)} documents from {cleaned_root}")

# Split documents by sections for better semantic coherence
split_docs = split_documents_by_sections(documents)
print(f"Split {len(documents)} documents into {len(split_docs)} section-based chunks")

# Create chunk index from split documents
chunk_index = []
for doc in split_docs:
    chunk_data = {
        "text": doc.page_content,
        "topic_id": doc.metadata["topic_id"],
        "source": doc.metadata["source"],
        "topic_name": doc.metadata["topic_name"],
    }
    # Add section-specific metadata if available
    if "section_header" in doc.metadata:
        chunk_data["section_header"] = doc.metadata["section_header"]
    chunk_index.append(chunk_data)

# Group chunks by topic for sampling positives
chunks_by_topic: dict[int, list[str]] = {}
for c in chunk_index:
    chunks_by_topic.setdefault(c["topic_id"], []).append(c["text"])

# 4. Build training examples (statement, positive) pairs
train_statements_dir = base / "train" / "statements"
train_answers_dir = base / "train" / "answers"

train_examples = []
for stmt_path in sorted(train_statements_dir.glob("statement_*.txt")):
    base_name = stmt_path.stem
    answer_path = train_answers_dir / f"{base_name}.json"
    if not answer_path.exists():
        continue
    statement_text = stmt_path.read_text(encoding="utf-8").strip()
    with answer_path.open() as f:
        ans = json.load(f)
    topic_id = ans.get("statement_topic")
    if topic_id is None:
        continue
    positives = chunks_by_topic.get(topic_id, [])
    if not positives:
        continue
    # Simple strategy: randomly sample one positive chunk per statement
    positive_chunk = random.choice(positives)
    train_examples.append(InputExample(texts=[statement_text, positive_chunk]))

# 5. Create DataLoader from examples
# Note: sentence-transformers handles InputExample objects correctly despite type checker warnings
train_dataloader: Any = DataLoader(train_examples, shuffle=True, batch_size=16)  # type: ignore

# 6. Loss
train_loss = losses.MultipleNegativesRankingLoss(model=model)

# 7. (Optional) Validation evaluator: build a small dev set of (statement, correct chunk) pairs
# You can use EmbeddingSimilarityEvaluator or compute retrieval metrics yourself.

print(f"Created {len(train_examples)} training examples from statements and chunks")
print(
    f"Topics with chunks: {len([t for t in chunks_by_topic.keys() if chunks_by_topic[t]])}"
)

# 8. Train
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=2,
    warmup_steps=100,
    output_path="models/finetuned-med-embedder",
    show_progress_bar=True,
)

# After initial training, you can implement hard negative mining loop (see next section)
