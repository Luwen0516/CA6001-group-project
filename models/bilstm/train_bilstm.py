"""
BiLSTM 文本情感分类模型 - 完整训练框架
=========================================
模型结构: Embedding -> BiLSTM -> Dropout -> FC -> Sigmoid

使用方法:
    python train_bilstm.py

输出:
    - 训练好的模型权重 (best_model.pt)
    - 训练曲线 (training_curves.png)
    - 混淆矩阵 (confusion_matrix.png)
    - 分类报告 (classification_report.txt)
"""

import os
import re
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

# ============================================================
# 配置参数
# ============================================================
CONFIG = {
    # 数据路径
    "train_path": "../../data/processed/train.csv",
    "val_path": "../../data/processed/val.csv",
    "test_path": "../../data/processed/test.csv",

    # 为了快速验证，先用小子集跑通；正式实验时设为 None 用全量数据
    "max_train_samples": 20000,   # 先用2万条快速跑通
    "max_val_samples": 2000,
    "max_test_samples": 2000,

    # 文本处理参数
    "max_vocab_size": 20000,      # 词表大小
    "max_seq_len": 50,            # 每条推文最大截断长度

    # 模型参数
    "embed_dim": 128,             # 词向量维度
    "hidden_dim": 64,             # LSTM隐藏层维度
    "num_layers": 2,              # LSTM层数
    "dropout": 0.5,               # Dropout比率

    # 训练参数
    "batch_size": 64,
    "epochs": 5,
    "lr": 0.001,

    # 设备
    "device": "mps" if torch.backends.mps.is_available() else "cpu",
}

# ============================================================
# 1. 文本预处理
# ============================================================
def clean_text(text):
    """简单的文本清洗"""
    text = str(text).lower()
    text = re.sub(r"@\w+", "", text)        # 去掉@用户名
    text = re.sub(r"http\S+|www\S+", "", text)  # 去掉URL
    text = re.sub(r"[^a-z\s]", "", text)   # 只保留字母和空格
    text = re.sub(r"\s+", " ", text).strip()  # 合并多余空格
    return text


# ============================================================
# 2. 构建词表 (Vocabulary)
# ============================================================
class Vocab:
    def __init__(self, texts, max_size=20000):
        # 特殊符号
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.pad_idx = 0
        self.unk_idx = 1

        # 统计词频
        counter = Counter()
        for text in texts:
            words = text.split()
            counter.update(words)

        # 取最常见的 top-K 词
        most_common = counter.most_common(max_size - 2)

        # 构建词 -> 索引 映射
        self.word2idx = {self.pad_token: self.pad_idx, self.unk_token: self.unk_idx}
        for word, _ in most_common:
            self.word2idx[word] = len(self.word2idx)

        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)
        print(f"词表大小: {self.vocab_size}")

    def encode(self, text, max_len=50):
        """把文本转成索引序列"""
        words = text.split()
        indices = [self.word2idx.get(w, self.unk_idx) for w in words]

        # 截断或填充到固定长度
        if len(indices) < max_len:
            indices += [self.pad_idx] * (max_len - len(indices))
        else:
            indices = indices[:max_len]

        return indices


# ============================================================
# 3. Dataset 类
# ============================================================
class SentimentDataset(Dataset):
    def __init__(self, df, vocab, max_seq_len=50):
        self.texts = df["text"].tolist()
        self.labels = df["label"].tolist()
        self.vocab = vocab
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        input_ids = self.vocab.encode(text, self.max_seq_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.float),
        }


# ============================================================
# 4. BiLSTM 模型定义
# ============================================================
class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=64, num_layers=2, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)  # *2 因为双向

    def forward(self, x):
        # x: (batch, seq_len)
        embedded = self.embedding(x)          # (batch, seq_len, embed_dim)
        lstm_out, (hidden, cell) = self.lstm(embedded)  # hidden: (2*num_layers, batch, hidden_dim)

        # 取最后一层的正向和反向 hidden state 拼接
        # hidden shape: (4, batch, hidden_dim)  (2层*2方向)
        # 取最后一层: hidden[-2] 是正向最后一层, hidden[-1] 是反向最后一层
        forward_last = hidden[-2]             # (batch, hidden_dim)
        backward_last = hidden[-1]            # (batch, hidden_dim)
        combined = torch.cat((forward_last, backward_last), dim=1)  # (batch, hidden_dim*2)

        out = self.dropout(combined)
        out = self.fc(out)                    # (batch, 1)
        return out.squeeze(1)                 # (batch,)


# ============================================================
# 5. 训练一个 epoch
# ============================================================
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    return avg_loss, acc, f1


# ============================================================
# 6. 验证/测试
# ============================================================
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)

    return avg_loss, acc, f1, precision, recall, all_labels, all_preds


# ============================================================
# 7. 可视化
# ============================================================
def plot_training_curves(history, save_path="training_curves.png"):
    """画训练/验证的 loss 和 accuracy 曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss 曲线
    axes[0].plot(history["train_loss"], label="Train Loss", marker="o")
    axes[0].plot(history["val_loss"], label="Val Loss", marker="s")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy 曲线
    axes[1].plot(history["train_acc"], label="Train Acc", marker="o")
    axes[1].plot(history["val_acc"], label="Val Acc", marker="s")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training & Validation Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"训练曲线已保存: {save_path}")


def plot_confusion_matrix(y_true, y_pred, save_path="confusion_matrix.png"):
    """画混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Negative", "Positive"],
                yticklabels=["Negative", "Positive"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"混淆矩阵已保存: {save_path}")


# ============================================================
# 8. 主函数
# ============================================================
def main():
    print("=" * 60)
    print("BiLSTM 情感分类训练")
    print("=" * 60)
    print(f"设备: {CONFIG['device']}")

    # --- 加载数据 ---
    print("\n[1/5] 加载数据...")
    train_df = pd.read_csv(CONFIG["train_path"])
    val_df = pd.read_csv(CONFIG["val_path"])
    test_df = pd.read_csv(CONFIG["test_path"])

    # 抽样（快速验证用）
    if CONFIG["max_train_samples"]:
        train_df = train_df.sample(n=CONFIG["max_train_samples"], random_state=42)
        val_df = val_df.sample(n=CONFIG["max_val_samples"], random_state=42)
        test_df = test_df.sample(n=CONFIG["max_test_samples"], random_state=42)

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # 文本清洗
    print("\n[2/5] 文本清洗 & 构建词表...")
    train_df["clean_text"] = train_df["text"].apply(clean_text)
    val_df["clean_text"] = val_df["text"].apply(clean_text)
    test_df["clean_text"] = test_df["text"].apply(clean_text)

    vocab = Vocab(train_df["clean_text"].tolist(), max_size=CONFIG["max_vocab_size"])

    # 构建 Dataset & DataLoader
    train_dataset = SentimentDataset(train_df, vocab, CONFIG["max_seq_len"])
    val_dataset = SentimentDataset(val_df, vocab, CONFIG["max_seq_len"])
    test_dataset = SentimentDataset(test_df, vocab, CONFIG["max_seq_len"])

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"])
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"])

    # --- 初始化模型 ---
    print("\n[3/5] 初始化模型...")
    model = BiLSTM(
        vocab_size=vocab.vocab_size,
        embed_dim=CONFIG["embed_dim"],
        hidden_dim=CONFIG["hidden_dim"],
        num_layers=CONFIG["num_layers"],
        dropout=CONFIG["dropout"],
    ).to(CONFIG["device"])

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量: {total_params:,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"])

    # --- 训练 ---
    print("\n[4/5] 开始训练...")
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0

    for epoch in range(CONFIG["epochs"]):
        start = time.time()

        train_loss, train_acc, train_f1 = train_epoch(model, train_loader, criterion, optimizer, CONFIG["device"])
        val_loss, val_acc, val_f1, val_prec, val_recall, _, _ = evaluate(model, val_loader, criterion, CONFIG["device"])

        elapsed = time.time() - start

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch+1}/{CONFIG['epochs']} ({elapsed:.1f}s) | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f}")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pt")
            print(f"  -> 新最佳模型! Val Acc: {val_acc:.4f}")

    # --- 测试集评估 ---
    print("\n[5/5] 测试集评估...")
    model.load_state_dict(torch.load("best_model.pt", weights_only=True))
    test_loss, test_acc, test_f1, test_prec, test_recall, y_true, y_pred = evaluate(
        model, test_loader, criterion, CONFIG["device"]
    )

    print(f"\n{'='*50}")
    print(f"测试集结果:")
    print(f"  Accuracy:  {test_acc:.4f}")
    print(f"  Precision: {test_prec:.4f}")
    print(f"  Recall:    {test_recall:.4f}")
    print(f"  F1 Score:  {test_f1:.4f}")
    print(f"{'='*50}")

    # 保存分类报告
    with open("classification_report.txt", "w") as f:
        f.write(f"Test Accuracy:  {test_acc:.4f}\n")
        f.write(f"Test Precision: {test_prec:.4f}\n")
        f.write(f"Test Recall:    {test_recall:.4f}\n")
        f.write(f"Test F1 Score:  {test_f1:.4f}\n")

    # --- 可视化 ---
    plot_training_curves(history)
    plot_confusion_matrix(y_true, y_pred)

    print("\n完成! 所有输出文件:")
    print("  - best_model.pt          (最佳模型权重)")
    print("  - training_curves.png    (训练曲线)")
    print("  - confusion_matrix.png   (混淆矩阵)")
    print("  - classification_report.txt (测试集指标)")


if __name__ == "__main__":
    main()
