# BiLSTM 情感分类模型

## 模型结构

```
Embedding (vocab_size × 128)
    ↓
BiLSTM (2层, hidden_dim=64, bidirectional)
    ↓
Concat forward + backward hidden state (128维)
    ↓
Dropout (0.5)
    ↓
Linear → Sigmoid
```

- 参数量: ~2.76M
- 优化器: Adam (lr=0.001)
- Loss: BCEWithLogitsLoss

## 快速开始

```bash
cd models/bilstm
python train_bilstm.py
```

## 配置参数

在 `train_bilstm.py` 顶部的 `CONFIG` 字典中修改：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `max_train_samples` | 20000 | 训练样本数（设为None用全量128万条） |
| `max_val_samples` | 2000 | 验证样本数 |
| `max_test_samples` | 2000 | 测试样本数 |
| `max_vocab_size` | 20000 | 词表大小 |
| `max_seq_len` | 50 | 每条推文最大长度 |
| `embed_dim` | 128 | 词向量维度 |
| `hidden_dim` | 64 | LSTM隐藏层维度 |
| `num_layers` | 2 | LSTM层数 |
| `dropout` | 0.5 | Dropout比率 |
| `batch_size` | 64 | Batch大小 |
| `epochs` | 5 | 训练轮数 |
| `lr` | 0.001 | 学习率 |

## 输出文件

运行后生成：
- `best_model.pt` — 验证集上表现最好的模型权重
- `training_curves.png` — 训练/验证 Loss 和 Accuracy 曲线
- `confusion_matrix.png` — 测试集混淆矩阵
- `classification_report.txt` — 测试集各项指标

## 当前结果（快速验证版）

基于 2 万条训练样本的快速测试：

| 指标 | 值 |
|---|---|
| Accuracy | 0.7075 |
| Precision | 0.7021 |
| Recall | 0.7197 |
| F1 Score | 0.7108 |

> 注：全量 128 万条数据训练后预期 Accuracy 可达 80-83%。
