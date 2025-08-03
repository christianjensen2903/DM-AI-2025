import json
from pathlib import Path
import argparse
from typing import List, Dict, Any

from retrieval import build_retrievers
from utils import load_topics, load_cleaned_documents
from text_normalizer import normalize_medical_text
from sklearn.metrics import accuracy_score


def get_topic_name_map(topic2id: dict) -> dict:
    return {v: k for k, v in topic2id.items()}


def check_top2_different(retrieved_docs) -> bool:
    """
    Check if the top 2 retrieved snippets are different.

    Args:
        retrieved_docs: List of retrieved documents

    Returns:
        bool: True if top 2 snippets have different text content
    """
    if len(retrieved_docs) < 2:
        return False

    doc1 = retrieved_docs[0]
    doc2 = retrieved_docs[1]

    # Compare chunk text (strip whitespace for comparison)
    text1 = doc1.page_content.strip()
    text2 = doc2.page_content.strip()

    # Check if text is different
    return text1 != text2


def analyze_retrieval_errors(
    retriever,
    statements_dir: Path,
    answers_dir: Path,
    topic_name_map: dict,
    normalize: bool = False,
    top_k: int = 5,
    save_report: bool = True,
):
    """
    Analyze and report retrieval results with full context:
    - Record every statement with its true and predicted topic.
    - Label each case as correct or incorrect.
    - For ALL incorrect cases where top 2 snippets are different, include full retrieved top-k chunks.
    """
    cases: List[Dict[str, Any]] = []  # all evaluated statements
    all_errors: List[Dict[str, Any]] = (
        []
    )  # incorrect cases with different top-2 for reporting
    y_true = []
    y_pred = []

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

        # Retrieve top-k
        retrieved = retriever.get_relevant_documents(statement_text)[:top_k]
        pred_topic = retrieved[0].metadata.get("topic_id", -1) if retrieved else -1
        is_correct = pred_topic == true_topic

        # Check if top 2 snippets are different
        has_different_top2 = check_top2_different(retrieved)

        y_true.append(true_topic)
        y_pred.append(pred_topic)

        is_true = bool(answer.get("statement_is_true"))

        entry = {
            "statement_id": base,
            "statement_text": statement_text,
            "is_true": is_true,
            "true_topic_id": true_topic,
            "true_topic_name": topic_name_map.get(true_topic, ""),
            "predicted_topic_id": pred_topic,
            "predicted_topic_name": topic_name_map.get(pred_topic, ""),
            "is_correct": is_correct,
            "has_different_top2": has_different_top2,
            "retrieved": [],
        }
        for i, doc in enumerate(retrieved):
            topic_id = doc.metadata.get("topic_id", -1)
            entry["retrieved"].append(
                {
                    "rank": i + 1,
                    "topic_id": topic_id,
                    "topic_name": topic_name_map.get(topic_id, ""),
                    "source": doc.metadata.get("source", ""),
                    "section_header": doc.metadata.get("section_header", ""),
                    "subsection_index": doc.metadata.get("subsection_index", None),
                    "total_subsections": doc.metadata.get("total_subsections", None),
                    "chunk_text": doc.page_content,
                }
            )
        cases.append(entry)

        # Collect errors with different top-2 for reporting
        if not is_correct and (has_different_top2 or is_true):
            all_errors.append(entry)

    # Compute overall accuracy
    acc = accuracy_score(y_true, y_pred)
    total = len(y_true)
    print(f"Evaluated {total} statements, accuracy: {acc:.4f}")
    print(f"Incorrect cases with different top-2 snippets: {len(all_errors)}")

    # Save full case list
    detailed_path = Path("error_analysis_detailed.json")
    with detailed_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "accuracy": acc,
                "total": total,
                "incorrect_with_different_top2_count": len(all_errors),
                "cases": cases,
                "y_true": y_true,
                "y_pred": y_pred,
            },
            f,
            indent=2,
        )
    print(f"Saved detailed cases to {detailed_path}")

    if save_report:
        report_path = Path("error_analysis_report.md")
        with report_path.open("w", encoding="utf-8") as f:
            f.write("# Retrieval Error Analysis Report\n")
            f.write(f"**Overall accuracy**: {acc:.4f} on {total} statements.\n\n")
            f.write(
                f"**Incorrect cases with different top-2 snippets**: {len(all_errors)}\n\n"
            )

            f.write(
                f"## Incorrect retrievals with different top-2 snippets ({len(all_errors)} total)\n\n"
            )
            for err in all_errors:
                f.write(
                    f"### Statement {err['statement_id']} (Correct: {err['is_true']})\n"
                )
                f.write(
                    f"**True topic**: {err['true_topic_name']} (ID {err['true_topic_id']})\n\n"
                )
                f.write(
                    f"**Predicted topic**: {err['predicted_topic_name']} (ID {err['predicted_topic_id']})\n\n"
                )
                f.write(
                    f"**Statement text**: {err['statement_text'].replace(' ', ' ')}\n\n"
                )
                f.write(f"#### Top {top_k} retrieved chunks\n")
                for r in err["retrieved"]:
                    f.write(
                        f"- **Rank {r['rank']}**: topic {r['topic_name']} (ID {r['topic_id']}), source: {r['source']}, section: {r.get('section_header','')}\n"
                    )
                    snippet = r["chunk_text"].strip().replace(" ", " ")
                    f.write(f"  - Passage: > {snippet}\n")
                f.write("---")

        print(f"Saved markdown summary report to {report_path}")

    return {
        "accuracy": acc,
        "total": total,
        "incorrect_with_different_top2_count": len(all_errors),
        "cases": cases,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detailed retrieval error analysis")
    parser.add_argument(
        "--data-root", type=str, default="data", help="Base data directory"
    )
    parser.add_argument(
        "--top-k", type=int, default=5, help="Number of top retrieved chunks to inspect"
    )

    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Disable writing the markdown summary report",
    )
    args = parser.parse_args()

    base = Path(args.data_root)
    topics_json = base / "topics.json"
    cleaned_root = base / "cleaned_topics"
    statements_dir = base / "train" / "statements"
    answers_dir = base / "train" / "answers"

    topic2id, _ = load_topics(topics_json)
    topic_name_map = get_topic_name_map(topic2id)

    normalize = True
    documents = load_cleaned_documents(cleaned_root, topic2id, normalize=normalize)

    retriever = build_retrievers(documents)

    analyze_retrieval_errors(
        retriever,
        statements_dir,
        answers_dir,
        topic_name_map,
        normalize=normalize,
        top_k=args.top_k,
        save_report=not args.no_report,
    )
