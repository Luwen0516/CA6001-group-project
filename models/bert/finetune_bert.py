"""Fine-tune bert-base-uncased for binary Sentiment140 classification.

Expected CSV files (columns: text, label):
  data/processed/{train,val,test,manual_test}.csv

Labels are 0=negative and 1=positive. The manual split may also contain
label 2 (neutral); those rows are excluded from its binary evaluation.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)


class TweetDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, tokenizer, max_length: int):
        self.texts = frame["text"].fillna("").astype(str).tolist()
        self.labels = frame["label"].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        encoded = self.tokenizer(
            self.texts[index], max_length=self.max_length,
            truncation=True, padding="max_length", return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[index], dtype=torch.long),
        }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    labels, probabilities = [], []
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        logits = model(input_ids=batch["input_ids"],
                       attention_mask=batch["attention_mask"]).logits
        probabilities.extend(torch.softmax(logits, dim=1)[:, 1].cpu().tolist())
        labels.extend(batch["labels"].cpu().tolist())
    return np.asarray(labels), np.asarray(probabilities)


def metrics(y_true, probabilities):
    pred = (probabilities >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision_positive": float(precision_score(y_true, pred, zero_division=0)),
        "recall_positive": float(recall_score(y_true, pred, zero_division=0)),
        "f1_positive": float(f1_score(y_true, pred, zero_division=0)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "auc": float(roc_auc_score(y_true, probabilities)),
        "confusion_matrix": confusion_matrix(y_true, pred).tolist(),
        "classification_report": classification_report(
            y_true, pred, target_names=["negative", "positive"],
            output_dict=True, zero_division=0,
        ),
    }


def read_split(path: Path, *, binary_only=True) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Required data file not found: {path}. Run split_data.py first."
        )
    frame = pd.read_csv(path)
    missing = {"text", "label"} - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    frame = frame.dropna(subset=["label"]).copy()
    frame["label"] = frame["label"].astype(int)
    if binary_only:
        frame = frame[frame["label"].isin([0, 1])].copy()
    if frame.empty:
        raise ValueError(f"No usable binary examples found in {path}")
    return frame


def deterministic_subset(frame: pd.DataFrame, limit, seed: int) -> pd.DataFrame:
    if limit is None or limit >= len(frame):
        return frame.reset_index(drop=True)
    selected = set(frame.sample(frac=1, random_state=seed).index[:limit])
    return frame.loc[frame.index.isin(selected)].reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("../../data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/bert-base-uncased"))
    parser.add_argument("--model-name", default="bert-base-uncased")
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=None,
                        help="Optional reproducible subset size for CPU/smoke runs.")
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--eval-manual", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}; pretrained model: {args.model_name}")

    train = read_split(args.data_dir / "train.csv")
    val = read_split(args.data_dir / "val.csv")
    test = read_split(args.data_dir / "test.csv")
    train = deterministic_subset(train, args.max_train_samples, args.seed)
    val = deterministic_subset(val, args.max_val_samples, args.seed)
    test = deterministic_subset(test, args.max_test_samples, args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=2,
    ).to(device)
    train_loader = DataLoader(TweetDataset(train, tokenizer, args.max_length),
                              batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TweetDataset(val, tokenizer, args.max_length),
                            batch_size=args.batch_size)
    test_loader = DataLoader(TweetDataset(test, tokenizer, args.max_length),
                             batch_size=args.batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * args.warmup_ratio), total_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    best_val_f1, stale_epochs, history = -1.0, 0, []
    best_path = args.output_dir / "best_model"

    for epoch in range(args.epochs):
        start = time.time()
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=(device.type == "cuda")):
                loss = model(**batch).loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += loss.item()

        y_val, p_val = predict(model, val_loader, device)
        val_scores = metrics(y_val, p_val)
        record = {"epoch": epoch + 1, "train_loss": total_loss / len(train_loader),
                  "val": {k: val_scores[k] for k in
                          ("accuracy", "precision_positive", "recall_positive",
                           "f1_positive", "macro_f1", "auc")}}
        history.append(record)
        print(f"Epoch {epoch + 1}/{args.epochs}: loss={record['train_loss']:.4f}, "
              f"val_acc={val_scores['accuracy']:.4f}, "
              f"val_macro_f1={val_scores['macro_f1']:.4f}, "
              f"time={time.time() - start:.1f}s")

        if val_scores["macro_f1"] > best_val_f1:
            best_val_f1 = val_scores["macro_f1"]
            stale_epochs = 0
            model.save_pretrained(best_path)
            tokenizer.save_pretrained(best_path)
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print("Early stopping: validation macro-F1 did not improve.")
                break

    # Test split is evaluated only after selecting the checkpoint on validation.
    model = AutoModelForSequenceClassification.from_pretrained(best_path).to(device)
    y_test, p_test = predict(model, test_loader, device)
    result = {
        "model": args.model_name,
        "seed": args.seed,
        "hyperparameters": {
            "max_length": args.max_length, "batch_size": args.batch_size,
            "epochs_requested": args.epochs, "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay, "warmup_ratio": args.warmup_ratio,
            "max_train_samples": args.max_train_samples,
            "max_val_samples": args.max_val_samples,
            "max_test_samples": args.max_test_samples,
            "selection_metric": "validation macro-F1",
        },
        "sizes": {"train": len(train), "validation": len(val), "test": len(test)},
        "test": metrics(y_test, p_test),
        "history": history,
    }
    if args.eval_manual:
        manual_path = args.data_dir / "manual_test.csv"
        if manual_path.exists():
            manual = read_split(manual_path)
            manual_loader = DataLoader(TweetDataset(manual, tokenizer, args.max_length),
                                       batch_size=args.batch_size)
            y_manual, p_manual = predict(model, manual_loader, device)
            result["manual_test_size"] = len(manual)
            result["manual_test"] = metrics(y_manual, p_manual)
        else:
            print(f"Skipping manual evaluation; file is missing: {manual_path}")

    with (args.output_dir / "results.json").open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({"test": result["test"],
                      "manual_test": result.get("manual_test")}, indent=2))
    print(f"Saved model and results to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
