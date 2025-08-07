#!/usr/bin/env python3
"""
Evaluation script for Emergency Healthcare RAG system on training data.

This script processes the training data through the LLM and logs:
- Model predictions vs true labels
- Top retrieved snippets and their topics
- Detailed error analysis for misclassified cases
"""

import json
import os
import time
import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, asdict
import csv

# Import necessary modules from the API
from utils import load_topics, load_cleaned_documents
from retrieval import build_retrievers
from text_normalizer import normalize_medical_text
from api import format_prompt, query_llm, parse_llm_response
import dotenv

dotenv.load_dotenv()


@dataclass
class EvaluationResult:
    """Data class to store evaluation results for a single statement"""

    statement_id: str
    statement_text: str
    true_label: bool
    true_topic: int
    predicted_label: bool
    predicted_topic: int
    is_correct_label: bool
    is_correct_topic: bool
    is_correct_both: bool
    processing_time: float
    llm_response: str
    top_snippets: List[Dict[str, Any]]
    error_type: str = ""  # For error analysis


class TrainingDataEvaluator:
    """Evaluator for training data using the RAG system"""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.train_dir = self.data_dir / "train"
        self.statements_dir = self.train_dir / "statements"
        self.answers_dir = self.train_dir / "answers"

        # Load topics and build retriever (same as API)
        topics_json = self.data_dir / "topics.json"
        cleaned_root = self.data_dir / "cleaned_topics"

        self.topic2id, self.id2topic = load_topics(topics_json)
        self.normalize = False

        print("Loading documents...")
        documents = load_cleaned_documents(
            cleaned_root, self.topic2id, normalize=self.normalize
        )

        print("Building retriever...")
        self.retriever = build_retrievers(documents)

        print("Evaluator initialized successfully!")

    def load_training_data(self) -> List[Tuple[str, str, Dict]]:
        """Load all training statements and their corresponding answers"""
        training_data = []

        # Get all statement files
        statement_files = sorted(self.statements_dir.glob("statement_*.txt"))

        for statement_file in statement_files:
            statement_id = statement_file.stem  # e.g., "statement_0001"

            # Read statement text
            with open(statement_file, "r", encoding="utf-8") as f:
                statement_text = f.read().strip()

            # Read corresponding answer
            answer_file = self.answers_dir / f"{statement_id}.json"
            if answer_file.exists():
                with open(answer_file, "r", encoding="utf-8") as f:
                    answer_data = json.load(f)

                training_data.append((statement_id, statement_text, answer_data))
            else:
                print(f"Warning: No answer file found for {statement_id}")

        return training_data

    def predict_single_statement(
        self, statement: str
    ) -> Tuple[bool, int, str, List[Dict], float]:
        """
        Process a single statement through the RAG system
        Returns: (predicted_label, predicted_topic, llm_response, top_snippets, processing_time)
        """
        start_time = time.time()

        # Normalize statement if needed
        if self.normalize:
            statement = normalize_medical_text(statement, is_query=True)

        # Retrieval
        retrieved = self.retriever.invoke(statement)

        # Get top 10 snippets with topic information
        top_snippets = []
        for doc in retrieved[:10]:
            snippet_info = {
                "content": doc.metadata.get("original_content", doc.page_content),
                "topic_name": doc.metadata.get("topic_name", "Unknown"),
                "topic_id": doc.metadata.get("topic_id", -1),
                "score": (
                    getattr(doc.metadata, "score", None)
                    if hasattr(doc, "metadata")
                    else None
                ),
            }
            top_snippets.append(snippet_info)

        # Format prompt and query LLM
        prompt = format_prompt(statement, top_snippets)
        llm_response = query_llm(prompt)

        # Parse LLM response
        predicted_label, predicted_topic = parse_llm_response(llm_response)

        processing_time = time.time() - start_time

        return (
            predicted_label,
            predicted_topic,
            llm_response,
            top_snippets,
            processing_time,
        )

    def evaluate_all(self, max_samples: int = None) -> List[EvaluationResult]:
        """Evaluate all training data"""
        print("Loading training data...")
        training_data = self.load_training_data()

        if max_samples:
            training_data = training_data[:max_samples]
            print(f"Evaluating first {max_samples} samples")
        else:
            print(f"Evaluating all {len(training_data)} samples")

        results = []

        for i, (statement_id, statement_text, answer_data) in enumerate(training_data):
            print(f"Processing {i+1}/{len(training_data)}: {statement_id}")

            # Get true labels
            true_label = bool(answer_data["statement_is_true"])
            true_topic = answer_data["statement_topic"]

            try:
                # Make prediction
                (
                    predicted_label,
                    predicted_topic,
                    llm_response,
                    top_snippets,
                    proc_time,
                ) = self.predict_single_statement(statement_text)

                # Calculate correctness
                is_correct_label = predicted_label == true_label
                is_correct_topic = predicted_topic == true_topic
                is_correct_both = is_correct_label and is_correct_topic

                # Determine error type
                error_type = ""
                if not is_correct_both:
                    if not is_correct_label and not is_correct_topic:
                        error_type = "both_wrong"
                    elif not is_correct_label:
                        error_type = "label_wrong"
                    elif not is_correct_topic:
                        error_type = "topic_wrong"

                result = EvaluationResult(
                    statement_id=statement_id,
                    statement_text=statement_text,
                    true_label=true_label,
                    true_topic=true_topic,
                    predicted_label=predicted_label,
                    predicted_topic=predicted_topic,
                    is_correct_label=is_correct_label,
                    is_correct_topic=is_correct_topic,
                    is_correct_both=is_correct_both,
                    processing_time=proc_time,
                    llm_response=llm_response,
                    top_snippets=top_snippets,
                    error_type=error_type,
                )

                results.append(result)

                # Print progress for errors
                if not is_correct_both:
                    print(
                        f"  ERROR - {error_type}: True({true_label}, {true_topic}) vs Pred({predicted_label}, {predicted_topic})"
                    )

            except Exception as e:
                print(f"  ERROR processing {statement_id}: {e}")
                # Create error result
                error_result = EvaluationResult(
                    statement_id=statement_id,
                    statement_text=statement_text,
                    true_label=true_label,
                    true_topic=true_topic,
                    predicted_label=False,
                    predicted_topic=-1,
                    is_correct_label=False,
                    is_correct_topic=False,
                    is_correct_both=False,
                    processing_time=0.0,
                    llm_response=str(e),
                    top_snippets=[],
                    error_type="processing_error",
                )
                results.append(error_result)

        return results

    def save_results(
        self, results: List[EvaluationResult], output_dir: str = "evaluation_results"
    ):
        """Save evaluation results to various output formats"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1. Save detailed results as JSON
        detailed_file = output_path / f"detailed_results_{timestamp}.json"
        with open(detailed_file, "w") as f:
            json.dump([asdict(result) for result in results], f, indent=2)
        print(f"Detailed results saved to: {detailed_file}")

        # 2. Save summary statistics
        self.save_summary_stats(results, output_path / f"summary_{timestamp}.json")

        # 3. Save error analysis
        self.save_error_analysis(
            results, output_path / f"error_analysis_{timestamp}.json"
        )

        # 4. Save CSV for easy analysis
        self.save_csv_results(results, output_path / f"results_{timestamp}.csv")

    def save_summary_stats(self, results: List[EvaluationResult], filepath: Path):
        """Calculate and save summary statistics"""
        total = len(results)
        if total == 0:
            return

        # Basic accuracy metrics
        correct_labels = sum(r.is_correct_label for r in results)
        correct_topics = sum(r.is_correct_topic for r in results)
        correct_both = sum(r.is_correct_both for r in results)

        # Error breakdown
        error_counts = {}
        for result in results:
            if result.error_type:
                error_counts[result.error_type] = (
                    error_counts.get(result.error_type, 0) + 1
                )

        # Processing time stats
        times = [r.processing_time for r in results if r.processing_time > 0]
        avg_time = sum(times) / len(times) if times else 0

        stats = {
            "total_samples": total,
            "label_accuracy": correct_labels / total,
            "topic_accuracy": correct_topics / total,
            "both_accuracy": correct_both / total,
            "error_breakdown": error_counts,
            "avg_processing_time": avg_time,
            "total_processing_time": sum(times),
        }

        with open(filepath, "w") as f:
            json.dump(stats, f, indent=2)

        print(f"Summary statistics:")
        print(
            f"  Label Accuracy: {stats['label_accuracy']:.3f} ({correct_labels}/{total})"
        )
        print(
            f"  Topic Accuracy: {stats['topic_accuracy']:.3f} ({correct_topics}/{total})"
        )
        print(f"  Both Accuracy: {stats['both_accuracy']:.3f} ({correct_both}/{total})")
        print(f"  Average Time: {avg_time:.2f}s")
        print(f"Summary saved to: {filepath}")

    def save_error_analysis(self, results: List[EvaluationResult], filepath: Path):
        """Save detailed error analysis"""
        errors = [r for r in results if not r.is_correct_both]

        error_analysis = {
            "error_count": len(errors),
            "total_count": len(results),
            "error_rate": len(errors) / len(results) if results else 0,
            "errors": [],
        }

        for error in errors:
            error_info = {
                "statement_id": error.statement_id,
                "statement_text": (
                    error.statement_text[:200] + "..."
                    if len(error.statement_text) > 200
                    else error.statement_text
                ),
                "true_label": error.true_label,
                "true_topic": error.true_topic,
                "true_topic_name": self.id2topic.get(error.true_topic, "Unknown"),
                "predicted_label": error.predicted_label,
                "predicted_topic": error.predicted_topic,
                "predicted_topic_name": self.id2topic.get(
                    error.predicted_topic, "Unknown"
                ),
                "error_type": error.error_type,
                "llm_response": error.llm_response,
                "top_snippets": [
                    {
                        "topic_name": snippet["topic_name"],
                        "topic_id": snippet["topic_id"],
                        "content": snippet["content"],
                    }
                    for snippet in error.top_snippets
                ],
            }
            error_analysis["errors"].append(error_info)

        with open(filepath, "w") as f:
            json.dump(error_analysis, f, indent=2)

        print(f"Error analysis saved to: {filepath}")
        print(
            f"Found {len(errors)} errors out of {len(results)} samples ({len(errors)/len(results)*100:.1f}%)"
        )

    def save_csv_results(self, results: List[EvaluationResult], filepath: Path):
        """Save results in CSV format for easy analysis"""
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # Header
            writer.writerow(
                [
                    "statement_id",
                    "true_label",
                    "true_topic",
                    "predicted_label",
                    "predicted_topic",
                    "is_correct_label",
                    "is_correct_topic",
                    "is_correct_both",
                    "error_type",
                    "processing_time",
                    "true_topic_name",
                    "predicted_topic_name",
                    "top_snippet_topics",
                    "statement_text",
                ]
            )

            # Data rows
            for result in results:
                top_topics = "|".join(
                    [snippet["topic_name"] for snippet in result.top_snippets[:5]]
                )

                writer.writerow(
                    [
                        result.statement_id,
                        result.true_label,
                        result.true_topic,
                        result.predicted_label,
                        result.predicted_topic,
                        result.is_correct_label,
                        result.is_correct_topic,
                        result.is_correct_both,
                        result.error_type,
                        result.processing_time,
                        self.id2topic.get(result.true_topic, "Unknown"),
                        self.id2topic.get(result.predicted_topic, "Unknown"),
                        top_topics,
                        result.statement_text,
                    ]
                )

        print(f"CSV results saved to: {filepath}")


def main():
    """Main function to run the evaluation"""
    print("=" * 80)
    print("Emergency Healthcare RAG - Training Data Evaluation")
    print("=" * 80)

    # Initialize evaluator
    evaluator = TrainingDataEvaluator()

    # Run evaluation (you can limit samples for testing)
    # For full evaluation, set max_samples=None
    # For testing, use max_samples=10
    results = evaluator.evaluate_all()  # Start with 50 samples

    # Save results
    evaluator.save_results(results)

    print("\nEvaluation completed!")


if __name__ == "__main__":
    main()
