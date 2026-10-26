# BERT Fine-tuning on Sentiment140

This experiment fine-tunes the pretrained `bert-base-uncased` encoder for binary tweet sentiment classification. Labels are `0` for negative and `1` for positive. It uses the same stratified 80/10/10 split (seed 42) as the other project models, selects a checkpoint by validation macro-F1, and evaluates the held-out test set once after selection.

## Required input files

The repository currently does not contain the processed dataset. From the project root, create these files first:

```text
data/processed/train.csv
data/processed/val.csv
data/processed/test.csv
data/processed/manual_test.csv   # optional unless --eval-manual is used
```

Each CSV must contain `text` and `label` columns. The manual file may include the neutral label `2`; the script excludes it.

## Run

Install the dependencies in an environment with PyTorch and access to Hugging Face model weights:

```bash
pip install torch transformers pandas scikit-learn numpy
cd models/bert
python finetune_bert.py --data-dir ../../data/processed --output-dir runs/bert-base-uncased --eval-manual
```

For a constrained CPU quick run, add `--max-train-samples 5000 --max-val-samples 2000 --max-test-samples 2000 --epochs 1`. Such subset results must be labelled as preliminary and must not be presented as directly comparable with full-data model results.

The checked-in experiment summary is in `results/RESULTS.md`. To re-evaluate its saved checkpoint without retraining, use:

```bash
python evaluate_checkpoint.py --data-dir ../../data/processed --model-dir runs/bert-base-uncased-quick/best_model --max-train-samples 5000 --max-val-samples 2000 --max-test-samples 2000
```

Default settings: maximum sequence length 64, batch size 16, 3 epochs, AdamW learning rate `2e-5`, weight decay `0.01`, and 10% linear warmup. Training uses mixed precision when CUDA is available and early stopping on validation macro-F1. The output directory contains the best checkpoint and `results.json` with test metrics, confusion matrix, and epoch history. A GPU is strongly recommended for the full 1.28-million-example training split.

## Comparison caveat

The current project files include a full-data Logistic Regression result and full-data TextCNN results, but the saved BiLSTM report is based on a 2,000-example test subset and its README reports a separate quick run. Compare the final models only after confirming that the BiLSTM result uses the same test split and sample count. Record the actual BERT run settings and measured values; do not treat expected performance as an experimental result.
