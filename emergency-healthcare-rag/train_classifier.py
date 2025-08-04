import json
import logging
from pathlib import Path
from typing import Dict, Any

import torch
from torch.utils.data import Dataset
import numpy as np
import wandb
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    set_seed,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class StatementSnippetDataset(Dataset):
    def __init__(
        self,
        path: Path,
        tokenizer,
        max_length: int = 512,
        sep_token: str = "[SEP]",
        k: int = 3,
    ):
        self.examples = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                if not obj.get("retrieved_snippets"):
                    continue
                statement = obj.get("statement_text", "")
                snippets = [s.get("chunk_text", "") for s in obj["retrieved_snippets"]]
                context = f" {sep_token} ".join(snippets[:k])
                label = 1 if obj.get("is_true") else 0
                self.examples.append(
                    {
                        "statement": statement,
                        "context": context,
                        "label": label,
                    }
                )
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        enc = self.tokenizer(
            ex["statement"],
            ex["context"],
            padding=False,
            truncation="only_second",
            max_length=self.max_length,
            return_tensors=None,
        )
        enc["labels"] = ex["label"]
        return enc


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    preds = np.argmax(probs, axis=1)
    labels = labels.astype(int)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    try:
        auc = roc_auc_score(labels, probs[:, 1])
    except Exception:
        auc = float("nan")
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": auc,
    }


def train_and_evaluate(config: Dict[str, Any]) -> Dict[str, Any]:
    """Train and evaluate with a config dictionary. Returns a dict with results and paths."""
    # Defaults
    defaults = {
        "model_name": "bert-base-uncased",
        "batch_size": 8,
        "epochs": 3,
        "learning_rate": 2e-5,
        "max_length": 512,
        "seed": 42,
        "eval_steps": 500,
        "gradient_accumulation_steps": 1,
        "weight_decay": 0.01,
        "use_wandb": True,
        "wandb_project": "emergency-healthcare-rag",
        "wandb_run_name": None,
        "k": 3,
    }
    cfg = {**defaults, **config}

    # Initialize wandb if enabled
    if cfg.get("use_wandb", True):
        wandb.init(
            project=cfg["wandb_project"],
            name=cfg["wandb_run_name"],
            config=cfg,
            tags=["text-classification", "healthcare", "emergency-medicine"],
        )
        # Log dataset info
        wandb.log(
            {
                "dataset/train_file": cfg.get("train_file", ""),
                "dataset/val_file": cfg.get("val_file", ""),
            }
        )

    required = ["train_file", "output_dir"]
    for r in required:
        if r not in cfg or not cfg[r]:
            raise ValueError(f"Missing required config key: {r}")

    set_seed(cfg["seed"])

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])

    train_path = Path(cfg["train_file"])
    val_path = Path(cfg["val_file"])

    train_dataset = StatementSnippetDataset(
        train_path, tokenizer, max_length=cfg["max_length"], k=cfg["k"]
    )
    val_dataset = StatementSnippetDataset(
        val_path, tokenizer, max_length=cfg["max_length"], k=cfg["k"]
    )

    # Log dataset sizes to wandb
    if cfg.get("use_wandb", True):
        wandb.log(
            {
                "dataset/train_size": len(train_dataset),
                "dataset/val_size": len(val_dataset),
                "dataset/total_size": len(train_dataset) + len(val_dataset),
            }
        )

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model_name"], num_labels=2
    )

    data_collator = DataCollatorWithPadding(tokenizer, padding=True)

    training_args = TrainingArguments(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"],
        eval_strategy="steps",
        eval_steps=cfg["eval_steps"],
        save_strategy="no",
        learning_rate=cfg["learning_rate"],
        num_train_epochs=cfg["epochs"],
        weight_decay=cfg["weight_decay"],
        metric_for_best_model="accuracy",
        greater_is_better=True,
        load_best_model_at_end=False,
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        logging_steps=cfg["eval_steps"],
        report_to=["wandb"] if cfg.get("use_wandb", True) else [],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    logger.info("Starting training...")
    trainer.train()

    metrics = trainer.evaluate()
    logger.info(f"Evaluation metrics: {metrics}")

    trainer.save_model(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])

    preds_output = trainer.predict(val_dataset)
    pred_probs = torch.softmax(torch.tensor(preds_output.predictions), dim=1).numpy()
    pred_labels = np.argmax(pred_probs, axis=1)

    out_pred_file = Path(cfg["output_dir"]) / "val_predictions.jsonl"
    with out_pred_file.open("w", encoding="utf-8") as f:
        for i, ex in enumerate(val_dataset.examples):
            rec = {
                "statement": ex["statement"],
                "label": ex["label"],
                "predicted_label": int(pred_labels[i]),
                "score_true": float(pred_probs[i, 1]),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info(f"Wrote validation predictions to {out_pred_file}")

    # Log final metrics and artifacts to wandb
    if cfg.get("use_wandb", True):
        # Log final evaluation metrics
        final_metrics = {
            f"final/{k}": v for k, v in metrics.items() if not k.startswith("eval_")
        }
        wandb.log(final_metrics)

        # Save model artifact
        model_artifact = wandb.Artifact(
            name=f"model-{wandb.run.id}",
            type="model",
            description=f"Fine-tuned {cfg['model_name']} for healthcare statement classification",
        )
        model_artifact.add_dir(cfg["output_dir"])
        wandb.log_artifact(model_artifact)

        # Save predictions artifact
        pred_artifact = wandb.Artifact(
            name=f"predictions-{wandb.run.id}",
            type="predictions",
            description="Validation set predictions and scores",
        )
        pred_artifact.add_file(str(out_pred_file))
        wandb.log_artifact(pred_artifact)

        # Log summary metrics
        wandb.summary.update(
            {
                "best_f1": metrics.get("eval_f1", 0),
                "best_accuracy": metrics.get("eval_accuracy", 0),
                "best_roc_auc": metrics.get("eval_roc_auc", 0),
                "train_samples": len(train_dataset),
                "val_samples": len(val_dataset),
            }
        )

        wandb.finish()

    result = {
        "metrics": metrics,
        "prediction_file": str(out_pred_file),
        "model_dir": cfg["output_dir"],
    }
    return result


if __name__ == "__main__":
    config = {
        "train_file": "dataset/train.jsonl",
        "val_file": "dataset/val.jsonl",
        "model_name": "bert-base-uncased",
        "output_dir": "out/bert_truth_model",
        "batch_size": 8,
        "epochs": 3,
        "learning_rate": 2e-5,
        "max_length": 512,
        "seed": 42,
        "eval_steps": 500,
        "gradient_accumulation_steps": 1,
        "weight_decay": 0.01,
        "use_wandb": True,
        "wandb_project": "emergency-healthcare-rag",
        "wandb_run_name": "bert-truth-classifier-baseline",
        "k": 3,
    }
    result = train_and_evaluate(config)
    print("Final evaluation metrics:", result["metrics"])
    print(f"Predictions written to: {result['prediction_file']}")
    if config.get("use_wandb", True):
        print(f"Training logged to wandb project: {config['wandb_project']}")
