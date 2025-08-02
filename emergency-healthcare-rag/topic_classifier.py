# file: train_topic_classifier_bert.py
import json, os, random
from pathlib import Path
import torch, numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

# ---------- CONFIG ----------
MODEL_NAME = "bert-base-uncased"  # swap for Bio-/Clinical-BERT if you wish
MAX_LENGTH = 128  # most statements are short
BATCH_SIZE = 8
EPOCHS = 10
LR = 2e-5
FREEZE_UP_TO = None  # e.g. 8 → freeze embeddings + first 8 encoder layers
SEED = 42
DATA_ROOT = Path("data")
OUT_DIR = Path("trained_topic_model_bert")
# -----------------------------

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cuda.matmul.allow_tf32 = True


def load_statements(folder_txt: Path, folder_ans: Path):
    rows = []
    for txt_file in folder_txt.glob("statement_*.txt"):
        ans_file = folder_ans / f"{txt_file.stem}.json"
        if not ans_file.exists():
            continue
        topic_id = json.load(ans_file.open())["statement_topic"]
        rows.append({"text": txt_file.read_text().strip(), "label": int(topic_id)})
    return rows


train_rows = load_statements(
    DATA_ROOT / "synthetic/statements", DATA_ROOT / "synthetic/answers"
)
val_rows = load_statements(DATA_ROOT / "train/statements", DATA_ROOT / "train/answers")
if not train_rows or not val_rows:
    raise RuntimeError("Train/validation folders missing or empty.")

num_labels = max(r["label"] for r in train_rows + val_rows) + 1
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=num_labels
)

# optional layer freezing
if FREEZE_UP_TO is not None:
    for name, param in model.named_parameters():
        layer_num = (
            int(name.split(".")[2])  # works for BERT/RoBERTa
            if name.startswith("encoder.layer")
            else -1
        )
        if layer_num < FREEZE_UP_TO:
            param.requires_grad_(False)


def tokenize(batch):
    return tokenizer(
        batch["text"], padding="max_length", truncation=True, max_length=MAX_LENGTH
    )


train_ds = Dataset.from_list(train_rows).shuffle(SEED).map(tokenize, batched=True)
val_ds = Dataset.from_list(val_rows).map(tokenize, batched=True)

train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
val_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

args = TrainingArguments(
    output_dir=OUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LR,
    weight_decay=0.01,
    fp16=torch.cuda.is_available(),
    logging_steps=20,
    report_to="none",
    seed=SEED,
)


def compute_metrics(pred):
    preds = pred.predictions.argmax(-1)
    acc = (preds == pred.label_ids).mean()
    return {"accuracy": acc}


trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.evaluate()

# save artefacts
# trainer.save_model(OUT_DIR)
# tokenizer.save_pretrained(OUT_DIR)
# print(f"\n✓ Model + tokenizer saved to {OUT_DIR.resolve()}")
