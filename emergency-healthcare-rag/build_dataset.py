import json
from pathlib import Path
import argparse
from typing import List, Dict, Any

from retrieval import build_retrievers
from utils import load_topics, load_cleaned_documents
from text_normalizer import normalize_medical_text


def get_topic_name_map(topic2id: dict) -> dict:
    return {v: k for k, v in topic2id.items()}


def build_statement_level_examples(
    retriever,
    statements_dir: Path,
    answers_dir: Path,
    topic_name_map: dict,
    normalize: bool,
    top_k: int,
) -> List[Dict[str, Any]]:
    examples = []
    for stmt_path in sorted(statements_dir.glob("statement_*.txt")):
        base = stmt_path.stem
        answer_path = answers_dir / f"{base}.json"
        if not answer_path.exists():
            continue
        statement_text = stmt_path.read_text(encoding="utf-8")
        with answer_path.open("r", encoding="utf-8") as f:
            answer = json.load(f)
        true_topic = answer.get("statement_topic")
        if true_topic is None:
            continue
        if normalize:
            statement_text = normalize_medical_text(statement_text)

        retrieved = retriever.get_relevant_documents(statement_text)[:top_k]
        # Build snippet list
        retrieved_snippets = []
        snippet_labels = []
        for i, doc in enumerate(retrieved):
            topic_id = doc.metadata.get("topic_id", -1)
            snippet = {
                "rank": i + 1,
                "topic_id": topic_id,
                "topic_name": topic_name_map.get(topic_id, ""),
                "source": doc.metadata.get("source", ""),
                "section_header": doc.metadata.get("section_header", ""),
                "subsection_index": doc.metadata.get("subsection_index", None),
                "total_subsections": doc.metadata.get("total_subsections", None),
                "chunk_text": doc.page_content,
            }
            retrieved_snippets.append(snippet)
            snippet_labels.append(topic_id == true_topic)

        example = {
            "statement_id": base,
            "statement_text": statement_text,
            "is_true": bool(answer.get("statement_is_true")),
            "true_topic_id": true_topic,
            "true_topic_name": topic_name_map.get(true_topic, ""),
            "retrieved_snippets": retrieved_snippets,
            "snippet_labels": snippet_labels,
        }
        examples.append(example)
    return examples


def build_flattened_snippet_examples(
    statement_level_examples: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    flat = []
    for ex in statement_level_examples:
        for i, snip in enumerate(ex["retrieved_snippets"]):
            flat_example = {
                "statement_id": ex["statement_id"],
                "statement_text": ex["statement_text"],
                "is_true": ex["is_true"],
                "true_topic_id": ex["true_topic_id"],
                "true_topic_name": ex["true_topic_name"],
                "snippet_text": snip["chunk_text"],
                "snippet_topic_id": snip["topic_id"],
                "snippet_topic_name": snip["topic_name"],
                "snippet_relevant": ex["snippet_labels"][i],
                "is_top1": snip["rank"] == 1,
            }
            flat.append(flat_example)
    return flat


def save_jsonl(data: List[Dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Build BERT-style training dataset")
    parser.add_argument(
        "--data-root", type=str, default="data", help="Base data directory"
    )
    parser.add_argument(
        "--top-k", type=int, default=5, help="Number of top retrieved chunks to include"
    )
    parser.add_argument(
        "--output-root",
        type=str,
        required=True,
        help="Directory to write dataset files",
    )
    parser.add_argument(
        "--normalize", action="store_true", help="Disable medical text normalization"
    )
    parser.add_argument(
        "--flatten-snippets",
        action="store_true",
        help="Also emit flattened (statement, snippet) examples",
    )
    args = parser.parse_args()

    base = Path(args.data_root)
    topics_json = base / "topics.json"
    cleaned_root = base / "cleaned_topics"
    statements_dir = base / "synthetic" / "statements"
    answers_dir = base / "synthetic" / "answers"

    topic2id, _ = load_topics(topics_json)
    topic_name_map = get_topic_name_map(topic2id)

    normalize = args.normalize
    documents = load_cleaned_documents(cleaned_root, topic2id, normalize=normalize)
    retriever = build_retrievers(documents)

    statement_level = build_statement_level_examples(
        retriever,
        statements_dir,
        answers_dir,
        topic_name_map,
        normalize=normalize,
        top_k=args.top_k,
    )

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    statement_path = out_root / "statement_level_dataset.jsonl"
    save_jsonl(statement_level, statement_path)
    print(
        f"Saved statement-level dataset with {len(statement_level)} examples to {statement_path}"
    )

    if args.flatten_snippets:
        flat = build_flattened_snippet_examples(statement_level)
        flat_path = out_root / "flattened_snippet_dataset.jsonl"
        save_jsonl(flat, flat_path)
        print(
            f"Saved flattened snippet dataset with {len(flat)} examples to {flat_path}"
        )


if __name__ == "__main__":
    main()
