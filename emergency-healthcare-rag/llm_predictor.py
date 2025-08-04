import json
from pathlib import Path
from typing import List

import openai
import dotenv
from sklearn.metrics import accuracy_score
from tqdm import tqdm

dotenv.load_dotenv()


def format_prompt(statement: str, snippets: List[str]) -> str:
    prompt = """
You are a helpful medical assistant. Your task is to determine whether the following medical statement is supported by the evidence provided.

Statement:
{statement}

Retrieved Snippets:
{snippets}

Based only on the above snippets, is the statement true or false? Reply with a single word: True or False.
"""
    snippets_text = "\n\n".join(f"Snippet {i+1}:\n{s}" for i, s in enumerate(snippets))
    return prompt.format(statement=statement.strip(), snippets=snippets_text.strip())


client = openai.OpenAI(base_url="http://localhost:1234/v1")


def query_llm(prompt: str, model: str = "gpt-4.1-mini") -> str:
    response = client.chat.completions.create(
        model="local", messages=[{"role": "user", "content": prompt}], temperature=0.0
    )
    return response.choices[0].message.content.strip()


def predict_truth_from_dataset(
    dataset_path: Path,
    top_k: int = 5,
):
    with dataset_path.open("r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f]

    pred_labels = []
    true_labels = []
    for example in tqdm(dataset):
        statement = example["statement_text"]
        top_snippets = [s["chunk_text"] for s in example["retrieved_snippets"][:top_k]]

        prompt = format_prompt(statement, top_snippets)
        llm_response = query_llm(prompt)
        prediction = llm_response.lower().startswith("true")

        pred_labels.append(prediction)
        true_labels.append(example["is_true"])

    print(f"Accuracy: {accuracy_score(true_labels, pred_labels)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Predict statement truth using OpenAI LLM"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to statement_level_dataset.jsonl",
    )
    parser.add_argument(
        "--top-k", type=int, default=5, help="Top-K snippets to include"
    )
    args = parser.parse_args()

    predict_truth_from_dataset(Path(args.dataset), top_k=args.top_k)
