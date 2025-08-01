#!/usr/bin/env python3
"""
Script to format medical statements with their related articles.

This script:
1. Processes the first n statement files
2. Displays each statement with clear formatting
3. Looks up their corresponding topics and truth values from answer files
4. Finds all markdown articles for those topics
5. Presents each statement with its answer (TRUE/FALSE) and related articles in a clear, formatted output
"""

import json
import os
import glob
from pathlib import Path
from typing import Set, Dict, Tuple


def load_topics_mapping(topics_file: str = "data/topics.json") -> Dict[int, str]:
    """Load the mapping from topic ID to topic name."""
    with open(topics_file, "r") as f:
        topics_data = json.load(f)

    # Reverse the mapping: topic_name -> topic_id becomes topic_id -> topic_name
    return {topic_id: topic_name for topic_name, topic_id in topics_data.items()}


def get_statement_content(
    statement_num: int, statements_dir: str = "data/train/statements"
) -> str:
    """Get the content of a given statement."""
    statement_file = os.path.join(statements_dir, f"statement_{statement_num:04d}.txt")

    try:
        with open(statement_file, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return f"Statement {statement_num:04d} not found"


def get_statement_topic(
    statement_num: int, answers_dir: str = "data/train/answers"
) -> int:
    """Get the topic ID for a given statement number."""
    answer_file = os.path.join(answers_dir, f"statement_{statement_num:04d}.json")

    with open(answer_file, "r") as f:
        answer_data = json.load(f)

    return answer_data["statement_topic"]


def get_statement_answer(
    statement_num: int, answers_dir: str = "data/train/answers"
) -> Tuple[int, bool]:
    """Get the topic ID and truth value for a given statement number."""
    answer_file = os.path.join(answers_dir, f"statement_{statement_num:04d}.json")

    with open(answer_file, "r") as f:
        answer_data = json.load(f)

    return answer_data["statement_topic"], bool(answer_data["statement_is_true"])


def get_topic_articles(
    topic_name: str, topics_dir: str = "data/cleaned_topics"
) -> list:
    """Get all markdown content for a given topic."""
    topic_path = os.path.join(topics_dir, topic_name)

    if not os.path.exists(topic_path):
        print(f"Warning: Topic directory '{topic_path}' not found")
        return []

    # Find all markdown files in the topic directory
    md_files = glob.glob(os.path.join(topic_path, "*.md"))

    articles = []
    for md_file in md_files:
        try:
            with open(md_file, "r", encoding="utf-8") as f:
                file_content = f.read()
                articles.append(
                    {
                        "title": os.path.basename(md_file).replace(".md", ""),
                        "content": file_content.strip(),
                    }
                )
        except Exception as e:
            print(f"Warning: Could not read file {md_file}: {e}")

    return articles


def get_articles_for_statements(n: int, base_dir: str = ".") -> str:
    """
    Get formatted output with statements and their related articles.

    Args:
        n: Number of statements to process
        base_dir: Base directory containing the data folders

    Returns:
        Formatted text with statements and their related articles
    """
    # Change to the emergency-healthcare-rag directory
    old_cwd = os.getcwd()
    os.chdir(base_dir)

    try:
        # Load topic mapping
        topics_mapping = load_topics_mapping()

        formatted_output = []

        # Process each statement individually
        for i in range(n):
            try:
                # Get statement content
                statement_content = get_statement_content(i)

                # Get topic and truth value for this statement
                topic_id, is_true = get_statement_answer(i)

                if topic_id not in topics_mapping:
                    print(f"Warning: Topic ID {topic_id} not found in topics mapping")
                    continue

                topic_name = topics_mapping[topic_id]
                print(
                    f"Processing statement {i}: topic {topic_id} - {topic_name} - {'TRUE' if is_true else 'FALSE'}"
                )

                # Get articles for this topic
                articles = get_topic_articles(topic_name)

                # Format the statement and its articles
                statement_section = []
                statement_section.append("=" * 100)
                statement_section.append(f"STATEMENT {i:04d}")
                statement_section.append("=" * 100)
                statement_section.append("")
                statement_section.append(statement_content)
                statement_section.append("")
                statement_section.append(
                    f"Truth Value: {'TRUE' if is_true else 'FALSE'}"
                )
                statement_section.append("")
                statement_section.append("-" * 60)
                statement_section.append("Related Articles:")
                statement_section.append("-" * 60)

                if articles:
                    for article_num, article in enumerate(articles, 1):
                        statement_section.append("")
                        statement_section.append(
                            f"Article {article_num}: {article['title']}"
                        )
                        statement_section.append("─" * 40)
                        statement_section.append(article["content"])
                        if article_num < len(articles):
                            statement_section.append("")
                            statement_section.append("─" * 40)
                else:
                    statement_section.append("")
                    statement_section.append("No articles found for this topic.")

                formatted_output.append("\n".join(statement_section))

            except FileNotFoundError:
                print(f"Warning: Statement {i:04d} not found")
            except Exception as e:
                print(f"Warning: Error processing statement {i:04d}: {e}")

        return "\n\n\n".join(formatted_output)

    finally:
        os.chdir(old_cwd)


def main():
    """Main function to run the script."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract articles for the first n statements"
    )
    parser.add_argument("n", type=int, help="Number of statements to process")
    parser.add_argument(
        "--output",
        "-o",
        help="Output file (optional, prints to stdout if not provided)",
    )
    parser.add_argument(
        "--base-dir", default=".", help="Base directory containing data folders"
    )

    args = parser.parse_args()

    print(f"Processing first {args.n} statements...")

    try:
        # Get the concatenated articles
        result = get_articles_for_statements(args.n, args.base_dir)

        # Output the result
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(result)
            print(f"Articles written to {args.output}")
        else:
            print("\n" + "=" * 50 + " RESULT " + "=" * 50)
            print(result)

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
