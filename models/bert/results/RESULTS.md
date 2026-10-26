# BERT Fine-tuning Results

## Run configuration

This is a preliminary, resource-constrained run using `bert-base-uncased` and seed 42. The best checkpoint was selected on validation macro-F1.

| Setting | Value |
|---|---:|
| Training examples | 5,000 sampled from the 1,280,000-row training split |
| Validation examples | 2,000 |
| Test examples | 2,000 |
| Epochs | 1 |
| Maximum sequence length | 64 tokens |
| Batch size | 8 |
| Optimizer | AdamW, learning rate 2e-5, weight decay 0.01 |
| Warmup | 10% linear warmup |
| Device | CPU (CPU-only PyTorch build) |
| Training loss | 0.4893 |

## Metrics

| Split | Accuracy | Positive precision | Positive recall | Positive F1 | Macro-F1 | AUC |
|---|---:|---:|---:|---:|---:|---:|
| Validation (n=2,000) | 0.8115 | 0.8038 | 0.8315 | 0.8174 | 0.8113 | 0.8878 |
| Test (n=2,000) | 0.8175 | 0.8084 | 0.8318 | 0.8199 | 0.8175 | 0.8908 |

Test confusion matrix (rows = actual negative/positive; columns = predicted negative/positive):

|  | Predicted negative | Predicted positive |
|---|---:|---:|
| Actual negative | 804 | 197 |
| Actual positive | 168 | 831 |

## Interpretation and limitations

The pipeline completed end-to-end and achieved 0.8175 accuracy on the selected 2,000-example test sample. This is not a full-data BERT result: only 5,000 training examples and one epoch were used because the available environment has CPU-only PyTorch and limited GPU memory. The value should not be ranked directly against the full-data Logistic Regression and TextCNN results. It is also a single seed, with no manual-test evaluation because `manual_test.csv` was not available.

The complete metrics and configuration are stored in `runs/bert-base-uncased-quick/results.json`; the fine-tuned checkpoint is in the same run directory.
