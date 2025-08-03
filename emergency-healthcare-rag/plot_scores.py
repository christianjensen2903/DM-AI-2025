import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Import functions from the existing module (adjust module name if needed)
from retrieval import build_retrievers
from utils import load_topics, load_cleaned_documents
from text_normalizer import normalize_medical_text
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
)


def main():
    base = Path("data")
    topics_json = base / "topics.json"
    cleaned_root = base / "cleaned_topics"
    statements_dir = base / "train" / "statements"
    answers_dir = base / "train" / "answers"

    # Load topics and documents
    topic2id, _ = load_topics(topics_json)
    normalize_flag = True
    documents = load_cleaned_documents(cleaned_root, topic2id, normalize=normalize_flag)

    # Build retriever
    retriever = build_retrievers(documents)

    # Prepare lists for scores and correctness
    scores = []
    correctness = []

    for stmt_path in sorted(statements_dir.glob("statement_*.txt")):
        answer_path = answers_dir / f"{stmt_path.stem}.json"
        if not answer_path.exists():
            continue
        statement_text = stmt_path.read_text(encoding="utf-8")
        with answer_path.open("r", encoding="utf-8") as f:
            answer = json.load(f)
        true_topic = answer.get("statement_topic")
        if true_topic is None:
            continue

        # Normalize if needed
        if normalize_flag:
            statement_text = normalize_medical_text(statement_text)

        processed_query = retriever.preprocess_func(statement_text)
        score = max(retriever.vectorizer.get_scores(processed_query))

        scores.append(score)
        correctness.append(answer.get("statement_is_true"))

    # Convert to arrays
    y = np.array([1 if c else 0 for c in correctness])
    s = np.array(scores)

    # ROC / Youden's J
    fpr, tpr, thresh_roc = roc_curve(y, s)
    youden_j = tpr - fpr
    best_j_idx = np.argmax(youden_j)
    best_threshold_j = thresh_roc[best_j_idx]

    # Precision-Recall / F1
    precision, recall, thresh_pr = precision_recall_curve(y, s)
    # precision/recal arrays are one longer than thresh_pr; compute F1 for matching thresholds
    f1_scores = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-8)
    best_f1_idx = np.argmax(f1_scores)
    best_threshold_f1 = thresh_pr[best_f1_idx]

    # Accuracy over unique score thresholds
    unique_thresh = np.unique(s)
    accs = []
    for t in unique_thresh:
        preds = (s >= t).astype(int)
        accs.append((accuracy_score(y, preds), t))
    best_acc, best_threshold_acc = max(accs, key=lambda x: x[0])

    # Summary metrics
    roc_auc = roc_auc_score(y, s)
    pr_auc = average_precision_score(y, s)

    def eval_at_threshold(threshold, name):
        preds = (s >= threshold).astype(int)
        print(f"\n=== Evaluation using threshold optimized for {name} ===")
        print(f"Threshold: {threshold:.5f}")
        print(f"Accuracy: {accuracy_score(y, preds):.4f}")
        print(f"F1:       {f1_score(y, preds):.4f}")
        print("Confusion matrix (rows=true, cols=pred):")
        print(confusion_matrix(y, preds))
        print("Classification report:")
        print(classification_report(y, preds, digits=4))

    print(f"ROC AUC: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}")
    eval_at_threshold(best_threshold_j, "Youden's J (balanced TPR/FPR)")
    eval_at_threshold(best_threshold_f1, "F1")
    eval_at_threshold(best_threshold_acc, "Accuracy")

    # Plot histogram with threshold lines
    plt.figure(figsize=(10, 6))
    plt.hist(
        [s[y == 1], s[y == 0]],
        bins=50,
        label=["True", "False"],
        alpha=0.7,
        edgecolor="black",
        stacked=False,
    )
    plt.axvline(
        best_threshold_j,
        color="C2",
        linestyle="--",
        label=f"Youden's J thresh ({best_threshold_j:.3f})",
    )
    plt.axvline(
        best_threshold_f1,
        color="C3",
        linestyle=":",
        label=f"F1 thresh ({best_threshold_f1:.3f})",
    )
    plt.axvline(
        best_threshold_acc,
        color="C4",
        linestyle="-.",
        label=f"Accuracy thresh ({best_threshold_acc:.3f})",
    )
    plt.xlabel("Retrieval Score")
    plt.ylabel("Frequency")
    plt.title(
        "Histogram of Retrieval Scores for True vs. False with Selected Thresholds"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig("score_histogram_with_thresholds.png")
    plt.show()


if __name__ == "__main__":
    main()
