# CA6001 Group Project - Twitter Sentiment Analysis

CA6001 AI Algorithms Fundamentals & Application Assignment

## Project Overview

Comparative study of traditional ML vs deep learning models on Twitter sentiment classification using the Sentiment140 dataset.

## Dataset

**Sentiment140** - 1.6 million automatically labelled tweets from Stanford (Go et al., 2009).

- Labels: `0` = negative, `4` = positive
- Distant supervision: labels derived from emoticons in tweets

### Data Split

| Split | Size | Purpose |
|---|---|---|
| Train | 1,280,000 (80%) | Model training |
| Validation | 160,000 (10%) | Hyperparameter tuning |
| Test | 160,000 (10%) | Final evaluation |
| Manual Test | 498 | Human-annotated supplementary test |

- Random seed: `42`
- Stratified sampling (balanced class distribution)

### How to get the data

```bash
# 1. Download the original dataset from Stanford
cd data/raw
curl -L -o trainingandtestdata.zip http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip
unzip trainingandtestdata.zip

# 2. Run the split script
cd ../..
python split_data.py
```

The processed CSV files will be generated in `data/processed/`.

## Models to Compare

### Traditional ML
- Naive Bayes
- Logistic Regression
- Linear SVM

### Deep Learning
- TextCNN
- BiLSTM
- BERT / TweetBERT

## Project Structure

```
├── data/
│   ├── raw/                    # Original downloaded data
│   └── processed/              # Split train/val/test (generated)
├── split_data.py               # Data download & split script
├── models/                     # Model implementations
├── notebooks/                  # EDA and experiment notebooks
├── report/                     # Final report
└── slides/                     # Presentation slides
```

## Team Members

- 4-person group
