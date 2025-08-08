import itertools
from pathlib import Path
from typing import List, Optional
import pandas as pd
from langchain_core.documents import Document


from utils import load_topics, load_cleaned_documents
from text_normalizer import normalize_medical_text
from bm25_retriever import preprocess_func, preprocess_statements, evaluate_bm25_config
import traceback


def grid_search_bm25_params(
    docs: List[Document],
    statements_dir: Path,
    answers_dir: Path,
    k1_values: Optional[List[float]] = None,
    b_values: Optional[List[float]] = None,
    normalize: bool = False,
    remove_stopwords: bool = True,
    use_stemming: bool = True,
) -> pd.DataFrame:
    """
    Perform grid search over k1 and b parameters for BM25.

    Args:
        docs: List of documents
        statements_dir: Directory containing statement files
        answers_dir: Directory containing answer files
        k1_values: List of k1 values to test
        b_values: List of b values to test
        normalize: Whether to apply text normalization

    Returns:
        DataFrame with results for all parameter combinations
    """
    if k1_values is None:
        # Default k1 range: test around the default value of 1.5
        k1_values = [0.5, 0.75, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0]

    if b_values is None:
        # Default b range: test around the default value of 0.75
        b_values = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]

    print(f"Pre-processing documents and statements...")

    # Pre-process statements once
    processed_statements, true_labels, _ = preprocess_statements(
        statements_dir, answers_dir, normalize, remove_stopwords, use_stemming
    )

    results = []
    total_combinations = len(k1_values) * len(b_values)

    print(f"Testing {total_combinations} parameter combinations...")
    print(f"k1 values: {k1_values}")
    print(f"b values: {b_values}")
    print(f"Normalization: {'enabled' if normalize else 'disabled'}")
    print()

    for i, (k1, b) in enumerate(itertools.product(k1_values, b_values), 1):
        print(
            f"[{i:2d}/{total_combinations}] Testing k1={k1:.2f}, b={b:.2f}...", end=" "
        )

        try:
            result = evaluate_bm25_config(
                k1,
                b,
                docs,
                processed_statements,
                true_labels,
            )
            results.append(result)
            print(f"Accuracy: {result['accuracy']:.4f}")
        except Exception as e:
            traceback.print_exc()
            print(f"ERROR: {e}")
            results.append(
                {
                    "k1": k1,
                    "b": b,
                    "accuracy": 0.0,
                    "total_evaluated": 0,
                    "error": str(e),
                }
            )

    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(results)
    return df


def analyze_results(df: pd.DataFrame, top_n: int = 10) -> None:
    """
    Analyze and display the hyperparameter tuning results.

    Args:
        df: DataFrame with tuning results
        top_n: Number of top results to display
    """
    # Filter out error results
    valid_df = df[df["accuracy"] > 0].copy()

    if len(valid_df) == 0:
        print("No valid results found!")
        return

    # Sort by accuracy
    valid_df = valid_df.sort_values("accuracy", ascending=False)

    print("=== HYPERPARAMETER TUNING RESULTS ===")
    print(f"Total valid configurations tested: {len(valid_df)}")
    print(f"Best accuracy: {valid_df.iloc[0]['accuracy']:.4f}")
    print()

    print(f"Top {min(top_n, len(valid_df))} configurations:")
    print("Rank | k1    | b     | Accuracy | Evaluated")
    print("-" * 45)
    for i, row in valid_df.head(top_n).iterrows():
        rank = valid_df.index.get_loc(i) + 1
        print(
            f"{rank:4d} | {row['k1']:5.2f} | {row['b']:5.2f} | {row['accuracy']:8.4f} | {row['total_evaluated']:9d}"
        )

    print()

    # Best parameters
    best = valid_df.iloc[0]
    print(f"BEST PARAMETERS:")
    print(f"k1 = {best['k1']}")
    print(f"b = {best['b']}")
    print(f"Accuracy = {best['accuracy']:.4f}")
    print()

    # Parameter distribution analysis
    print("=== PARAMETER ANALYSIS ===")
    print(f"k1 range tested: {valid_df['k1'].min():.2f} - {valid_df['k1'].max():.2f}")
    print(f"b range tested: {valid_df['b'].min():.2f} - {valid_df['b'].max():.2f}")
    print(
        f"Accuracy range: {valid_df['accuracy'].min():.4f} - {valid_df['accuracy'].max():.4f}"
    )
    print()

    # Top k1 values
    k1_performance = (
        valid_df.groupby("k1")["accuracy"]
        .agg(["mean", "max", "count"])
        .sort_values("mean", ascending=False)
    )
    print("Top k1 values by average accuracy:")
    print(k1_performance.head().round(4))
    print()

    # Top b values
    b_performance = (
        valid_df.groupby("b")["accuracy"]
        .agg(["mean", "max", "count"])
        .sort_values("mean", ascending=False)
    )
    print("Top b values by average accuracy:")
    print(b_performance.head().round(4))


def main():
    """Main function to run hyperparameter tuning."""
    print("Starting BM25 hyperparameter tuning...")

    # Setup paths
    base = Path("data")
    topics_json = base / "topics.json"
    cleaned_root = base / "cleaned_topics"
    statements_dir = base / "synthetic" / "statements"
    answers_dir = base / "synthetic" / "answers"

    # Load data
    print("Loading data...")
    topic2id, _ = load_topics(topics_json)
    docs = load_cleaned_documents(cleaned_root, topic2id, normalize=False)
    print(f"Loaded {len(docs)} documents")

    # Define parameter ranges for tuning
    # You can customize these ranges based on your needs
    k1_values = [2.0, 2.5, 3.0, 4.0]
    b_values = [0.0, 0.25, 0.5, 0.75, 1.0]

    remove_stopwords = False
    use_stemming = False

    # Option 1: Test without normalization
    print("\n" + "=" * 50)
    print("TESTING WITHOUT NORMALIZATION")
    print("=" * 50)
    results_no_norm = grid_search_bm25_params(
        docs,
        statements_dir,
        answers_dir,
        k1_values=k1_values,
        b_values=b_values,
        normalize=False,
        remove_stopwords=remove_stopwords,
        use_stemming=use_stemming,
    )

    print("\nResults without normalization:")
    analyze_results(results_no_norm)

    # Compare best results
    print("\n" + "=" * 50)
    print("COMPARISON")
    print("=" * 50)
    best_no_norm = results_no_norm.loc[results_no_norm["accuracy"].idxmax()]

    print(
        f"Best without normalization: k1={best_no_norm['k1']}, b={best_no_norm['b']}, accuracy={best_no_norm['accuracy']:.4f}"
    )

    print(
        f"✓ No normalization performs better by {best_no_norm['accuracy'] - best_with_norm['accuracy']:.4f}"
    )
    print(
        f"RECOMMENDED: No normalization with k1={best_no_norm['k1']}, b={best_no_norm['b']}"
    )


if __name__ == "__main__":
    main()
