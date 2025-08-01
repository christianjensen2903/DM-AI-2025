from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from statement_dataset import StatementDataset


class BertMultiTask(nn.Module):
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_topics: int = 115,
        dropout_prob: float = 0.1,
    ):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier_label = nn.Linear(hidden_size, 1)
        self.classifier_topic = nn.Linear(hidden_size, num_topics)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        pooled = outputs.pooler_output  # (B, H)
        dropped = self.dropout(pooled)
        logit_label = self.classifier_label(dropped).squeeze(-1)
        logit_topic = self.classifier_topic(dropped)
        return logit_label, logit_topic


def make_train_val_dataloaders(
    dataset,
    val_frac: float = 0.1,
    batch_size: int = 8,
    seed: int = 42,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader]:
    """
    Split dataset into train/val with random sampling.
    Note: Stratification removed due to many topics having only 1 sample.
    """
    indices = list(range(len(dataset)))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_frac,
        random_state=seed,
    )
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=StatementDataset.collate_fn,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues with tokenize_fn
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=StatementDataset.collate_fn,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues with tokenize_fn
        pin_memory=True,
    )
    return train_loader, val_loader


def compute_metrics(
    logit_label, logit_topic, true_label, true_topic
) -> Dict[str, float]:
    """
    label: binary, topic: multi-class
    Returns dict with label_acc, topic_acc, joint_acc
    """
    with torch.no_grad():
        pred_label = (torch.sigmoid(logit_label) > 0.5).long()
        pred_topic = torch.argmax(logit_topic, dim=-1)

        label_correct = (pred_label == true_label).float()
        topic_correct = (pred_topic == true_topic).float()
        joint_correct = (label_correct * topic_correct).float()

        label_acc = label_correct.mean().item()
        topic_acc = topic_correct.mean().item()
        joint_acc = joint_correct.mean().item()

    return {
        "label_acc": label_acc,
        "topic_acc": topic_acc,
        "joint_acc": joint_acc,
    }


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    loss_fns: Tuple[nn.Module, nn.Module],
    loss_weights: Tuple[float, float] = (1.0, 1.0),
) -> Dict[str, float]:
    model.eval()
    total_label_loss = 0.0
    total_topic_loss = 0.0
    total_batch = 0
    accs = {"label_acc": 0.0, "topic_acc": 0.0, "joint_acc": 0.0}

    bce_loss_fn, ce_loss_fn = loss_fns

    with torch.no_grad():
        for batch in dataloader:
            inp = batch["input"]
            label = batch["label"].to(device)  # (B,)
            topic = batch["topic"].to(device)  # (B,)

            # assume tokenizer output dict
            input_ids = (
                inp["input_ids"].squeeze(1)
                if inp["input_ids"].dim() == 3
                else inp["input_ids"]
            )
            attention_mask = inp.get("attention_mask", None)
            if attention_mask is not None and attention_mask.dim() == 3:
                attention_mask = attention_mask.squeeze(1)
            token_type_ids = inp.get("token_type_ids", None)
            if token_type_ids is not None and token_type_ids.dim() == 3:
                token_type_ids = token_type_ids.squeeze(1)

            logit_label, logit_topic = model(
                input_ids=input_ids.to(device),
                attention_mask=(
                    attention_mask.to(device) if attention_mask is not None else None
                ),
                token_type_ids=(
                    token_type_ids.to(device) if token_type_ids is not None else None
                ),
            )

            label_loss = bce_loss_fn(logit_label, label.float().to(device))
            topic_loss = ce_loss_fn(logit_topic, topic.to(device))
            loss = loss_weights[0] * label_loss + loss_weights[1] * topic_loss

            total_label_loss += label_loss.item()
            total_topic_loss += topic_loss.item()

            batch_metrics = compute_metrics(
                logit_label, logit_topic, label.to(device), topic.to(device)
            )
            accs["label_acc"] += batch_metrics["label_acc"]
            accs["topic_acc"] += batch_metrics["topic_acc"]
            accs["joint_acc"] += batch_metrics["joint_acc"]

            total_batch += 1

    # average
    avg_metrics = {
        "label_loss": total_label_loss / total_batch,
        "topic_loss": total_topic_loss / total_batch,
        "label_acc": accs["label_acc"] / total_batch,
        "topic_acc": accs["topic_acc"] / total_batch,
        "joint_acc": accs["joint_acc"] / total_batch,
    }
    return avg_metrics


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_epochs: int = 5,
    lr: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_steps: int = 0,
    gradient_clip: float = 1.0,
    loss_weights: Tuple[float, float] = (1.0, 1.0),
    save_path: Optional[str] = "best_model.pt",
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    bce_loss_fn = nn.BCEWithLogitsLoss()
    ce_loss_fn = nn.CrossEntropyLoss()
    loss_fns = (bce_loss_fn, ce_loss_fn)

    best_joint_acc = -1.0

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_label_loss = 0.0
        running_topic_loss = 0.0
        running_total_loss = 0.0
        running_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch} train", leave=False)
        for batch in pbar:
            inp = batch["input"]
            label = batch["label"].to(device)
            topic = batch["topic"].to(device)

            input_ids = (
                inp["input_ids"].squeeze(1)
                if inp["input_ids"].dim() == 3
                else inp["input_ids"]
            )
            attention_mask = inp.get("attention_mask", None)
            if attention_mask is not None and attention_mask.dim() == 3:
                attention_mask = attention_mask.squeeze(1)
            token_type_ids = inp.get("token_type_ids", None)
            if token_type_ids is not None and token_type_ids.dim() == 3:
                token_type_ids = token_type_ids.squeeze(1)

            optimizer.zero_grad()

            logit_label, logit_topic = model(
                input_ids=input_ids.to(device),
                attention_mask=(
                    attention_mask.to(device) if attention_mask is not None else None
                ),
                token_type_ids=(
                    token_type_ids.to(device) if token_type_ids is not None else None
                ),
            )

            label_loss = bce_loss_fn(logit_label, label.float().to(device))
            topic_loss = ce_loss_fn(logit_topic, topic.to(device))
            loss = loss_weights[0] * label_loss + loss_weights[1] * topic_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            scheduler.step()

            running_label_loss += label_loss.item()
            running_topic_loss += topic_loss.item()
            running_total_loss += loss.item()
            running_batches += 1

            pbar.set_postfix(
                {
                    "lbl_loss": f"{running_label_loss/running_batches:.4f}",
                    "top_loss": f"{running_topic_loss/running_batches:.4f}",
                    "total_loss": f"{running_total_loss/running_batches:.4f}",
                }
            )

        train_metrics = evaluate(model, train_loader, device, loss_fns, loss_weights)
        val_metrics = evaluate(model, val_loader, device, loss_fns, loss_weights)

        # Calculate epoch-averaged training losses
        avg_train_label_loss = running_label_loss / running_batches
        avg_train_topic_loss = running_topic_loss / running_batches
        avg_train_total_loss = running_total_loss / running_batches

        print(
            f"[Epoch {epoch}] train joint_acc: {train_metrics['joint_acc']:.4f} | val joint_acc: {val_metrics['joint_acc']:.4f}"
        )
        print(
            f"            label_acc: {val_metrics['label_acc']:.4f}, topic_acc: {val_metrics['topic_acc']:.4f}"
        )
        print(
            f"            train losses -> label: {avg_train_label_loss:.4f}, topic: {avg_train_topic_loss:.4f}, total: {avg_train_total_loss:.4f}"
        )
        print(
            f"            val losses -> label: {val_metrics['label_loss']:.4f}, topic: {val_metrics['topic_loss']:.4f}"
        )

        # save model when joint accuracy improves
        if val_metrics["joint_acc"] > best_joint_acc + 1e-5:
            best_joint_acc = val_metrics["joint_acc"]
            if save_path:
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "epoch": epoch,
                        "val_metrics": val_metrics,
                    },
                    save_path,
                )
                print(
                    f"  Saved new best model (joint_acc={best_joint_acc:.4f}) to {save_path}"
                )

    return best_joint_acc


if __name__ == "__main__":
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader

    # configuration
    ROOT = "data/train"  # path to your train directory
    MODEL_NAME = "bert-base-uncased"
    NUM_TOPICS = 115
    BATCH_SIZE = 16
    NUM_EPOCHS = 10
    LEARNING_RATE = 2e-5
    VAL_FRAC = 0.1
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # tokenizer and dataset
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_fn(text: str):
        # return with batch dim=1 so collate handles squeezing
        return tokenizer(text, truncation=True, padding=False, return_tensors="pt")

    # you need to have StatementDataset class in scope (from prior answer)
    dataset = StatementDataset(ROOT, text_transform=tokenize_fn, preload=False)

    train_loader, val_loader = make_train_val_dataloaders(
        dataset, val_frac=VAL_FRAC, batch_size=BATCH_SIZE
    )

    model = BertMultiTask(model_name=MODEL_NAME, num_topics=NUM_TOPICS)
    model.to(DEVICE)

    # train: joint accuracy is used for best-model selection
    _ = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=DEVICE,
        num_epochs=NUM_EPOCHS,
        lr=LEARNING_RATE,
        save_path="best_multihead_bert.pt",
        loss_weights=(1.0, 1.0),  # you can tune to emphasize topic vs label
    )
