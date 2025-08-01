import json
from pathlib import Path
from typing import Callable, Any

import torch
from torch.utils.data import Dataset


class StatementDataset(Dataset):

    def __init__(
        self,
        root_dir: str,
        statements_subdir: str = "statements",
        answers_subdir: str = "answers",
        text_transform: Callable[[str], Any] | None = None,
        preload: bool = False,
    ):
        self.statements_dir = Path(root_dir) / statements_subdir
        self.answers_dir = Path(root_dir) / answers_subdir
        if not self.statements_dir.is_dir():
            raise ValueError(f"{self.statements_dir} is not a directory")
        if not self.answers_dir.is_dir():
            raise ValueError(f"{self.answers_dir} is not a directory")

        # gather all statement_*.txt and pair with corresponding json, skip mismatches
        txt_files = sorted(self.statements_dir.glob("statement_*.txt"))
        self.pairs: list[dict[str, Path]] = []
        for txt_path in txt_files:
            json_path = self.answers_dir / (txt_path.stem + ".json")
            if not json_path.exists():
                # skip or raise depending on desired behavior; here we skip with warning
                print(f"Warning: missing JSON for {txt_path.name}, skipping.")
                continue
            self.pairs.append({"txt": txt_path, "json": json_path})

        if not self.pairs:
            raise RuntimeError("No valid statement/json pairs found.")

        self.text_transform = text_transform
        self.preload = preload

        if self.preload:
            self._cache: list[dict[str, Any]] = []
            for pair in self.pairs:
                text, label, topic = self._load_item(pair["txt"], pair["json"])
                transformed = self.text_transform(text) if self.text_transform else text
                self._cache.append(
                    {
                        "input": transformed,
                        "label": label,
                        "topic": topic,
                        "id": pair["txt"].stem,
                        "raw_text": text,
                    }
                )

    def __len__(self) -> int:
        return len(self.pairs)

    def _load_item(self, txt_path: Path, json_path: Path):
        text = txt_path.read_text(encoding="utf-8")
        with json_path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        if "statement_is_true" not in obj or "statement_topic" not in obj:
            raise KeyError(f"JSON {json_path} missing required keys.")
        label = int(obj["statement_is_true"])
        topic = int(obj["statement_topic"])
        return text, label, topic

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if self.preload:
            entry = self._cache[idx]
            return {
                "input": entry["input"],
                "label": entry["label"],
                "topic": entry["topic"],
                "id": entry["id"],
                "raw_text": entry["raw_text"],
            }
        pair = self.pairs[idx]
        text, label, topic = self._load_item(pair["txt"], pair["json"])
        transformed = self.text_transform(text) if self.text_transform else text
        return {
            "input": transformed,
            "label": label,
            "topic": topic,
            "id": pair["txt"].stem,
            "raw_text": text,
        }

    @staticmethod
    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Default collate: assumes that 'input' is either raw text (kept as list)
        or a dict of torch tensors (e.g., tokenizer output) which will be stacked.
        Returns dict with keys: input, label, topic, id, raw_text
        """
        if not batch:
            return {}

        # Collate labels and topics
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        topics = torch.tensor([b["topic"] for b in batch], dtype=torch.long)
        ids = [b["id"] for b in batch]
        raw_texts = [b["raw_text"] for b in batch]

        first_input = batch[0]["input"]
        collated_input: Any  # Allow both dict and list types
        if isinstance(first_input, dict) or hasattr(first_input, "keys"):
            # assume HuggingFace-style dict of tensors; need to pad/stack manually or rely on tokenizer's batching
            collated_input = {}
            for k in first_input:
                values = [
                    (
                        b["input"][k].squeeze(0)
                        if (
                            isinstance(b["input"][k], torch.Tensor)
                            and b["input"][k].dim() > 0
                            and b["input"][k].shape[0] == 1
                        )
                        else b["input"][k]
                    )
                    for b in batch
                ]
                collated_input[k] = (
                    torch.nn.utils.rnn.pad_sequence(
                        values, batch_first=True, padding_value=0
                    )
                    if values and values[0].dim() >= 1
                    else torch.stack(values)
                )
        else:
            # raw text list
            collated_input = [b["input"] for b in batch]

        return {
            "input": collated_input,
            "label": labels,
            "topic": topics,
            "id": ids,
            "raw_text": raw_texts,
        }


if __name__ == "__main__":
    dataset = StatementDataset(
        root_dir="data/train",
        statements_subdir="statements",
        answers_subdir="answers",
    )
    print(dataset[0])
