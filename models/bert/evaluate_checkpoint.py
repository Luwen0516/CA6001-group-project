"""Evaluate an already fine-tuned BERT checkpoint without retraining."""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from finetune_bert import TweetDataset, metrics, predict, read_split


def subset(frame: pd.DataFrame, limit, seed: int):
    if limit is None or limit >= len(frame):
        return frame.reset_index(drop=True)
    indexes = frame.sample(frac=1, random_state=seed).index[:limit]
    return frame.loc[indexes].reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(device)
    frames = {
        "train": subset(read_split(args.data_dir / "train.csv"), args.max_train_samples, args.seed),
        "validation": subset(read_split(args.data_dir / "val.csv"), args.max_val_samples, args.seed),
        "test": subset(read_split(args.data_dir / "test.csv"), args.max_test_samples, args.seed),
    }
    evaluated = {}
    for split_name in ("validation", "test"):
        frame = frames[split_name]
        loader = DataLoader(TweetDataset(frame, tokenizer, args.max_length),
                            batch_size=args.batch_size)
        y_true, probabilities = predict(model, loader, device)
        evaluated[split_name] = metrics(y_true, probabilities)

    result = {
        "model": "bert-base-uncased",
        "checkpoint": str(args.model_dir.resolve()),
        "device": str(device),
        "seed": args.seed,
        "sizes": {key: len(frame) for key, frame in frames.items()},
        "hyperparameters": {"max_length": args.max_length,
                            "batch_size": args.batch_size,
                            "epochs": 1,
                            "learning_rate": 2e-5,
                            "selection_metric": "validation macro-F1"},
        "validation": evaluated["validation"],
        "test": evaluated["test"],
    }
    output = args.output or args.model_dir.parent / "results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"sizes": result["sizes"],
                      "validation": {k: result["validation"][k] for k in
                                     ("accuracy", "precision_positive", "recall_positive", "f1_positive", "macro_f1", "auc")},
                      "test": {k: result["test"][k] for k in
                               ("accuracy", "precision_positive", "recall_positive", "f1_positive", "macro_f1", "auc")},
                      "output": str(output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
