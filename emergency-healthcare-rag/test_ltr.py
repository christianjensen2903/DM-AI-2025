from ltr_retriever import LTRRetriever
from pathlib import Path
import json
from sklearn.metrics import accuracy_score


retriever = LTRRetriever.load(Path("models/ltr"))
# results = retriever.retrieve(
#     "Coronary heart disease affects approximately 15.5 million people in the United States, with the American Heart Association estimating that a person experiences a heart attack every 41 seconds.",
#     top_k=10,
# )
# for doc, bm25_score, ltr_score in results:
#     print("BM25:", bm25_score, "LTR:", ltr_score)
#     print(doc.metadata["topic_name"])
#     print(doc.page_content)
#     print()


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

        retrieved = retriever.retrieve(statement_text, top_k=1)

        pred_topic_id = retrieved[0].metadata.get("topic_id", -1)

        # print(true_topic_id, pred_topic_id)

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


if __name__ == "__main__":
    evaluate_topic_retrieval(
        retriever, Path("data/train/statements"), Path("data/train/answers")
    )
