# TextCNN 情感分类模型

Kim (2014) 的 TextCNN，用于 Sentiment140 推特情感二分类。本目录只含**模型、训练、消融和误差分析**；预处理沿用组内统一做法（见下方「数据接口」），不在这里重复实现。

## 模型结构

```
Embedding (20,000 × 128)
    ↓
并行 Conv1d：核宽 2 / 3 / 4 / 5，各 100 个滤波器 → ReLU
    ↓
max-over-time pooling（覆盖到补零位置的窗口置 0，避免补零影响）
    ↓
拼接 (4 × 100 = 400 维)
    ↓
Dropout (0.5) → Linear (400 → 2)
```

- 参数量：2,740,402（Embedding 2,560,000 + 卷积 179,600 + 全连接 802）
- 优化器 Adam（lr 1e-3，权重衰减 1e-4），批大小 256，交叉熵
- 最多 20 轮，验证集准确率连续 2 轮不提升则早停，并恢复验证集上最好一轮的权重
- 同一批内句长相近（按长度分桶），减少补零的计算浪费

## 数据接口

训练/评估读取一个目录（默认 `../../data/tokenized`），里面需要：

| 文件 | 内容 |
|---|---|
| `data.npz` | `x_{train,val,test,manual}`：词 id，形状 `(N, 50)`，补 0；`len_*`：真实词数；`y_*`：0 负 / 1 正 |
| `vocab.json` | `{"pad_id": 0, "unk_id": 1, "stoi": {词: id}}` |

这些数组按**组内统一预处理**生成（与 `models/bilstm/train_bilstm.py` 的 `clean_text` / `Vocab` 一致）：小写；去 `@用户名` 和网址；只保留 a–z 与空白；词表取**训练集**最常见的 19,998 词 + `<pad>`(0) + `<unk>`(1)；句长 50。划分用 `split_data.py` 生成的 `data/processed/{train,val,test,manual_test}.csv`。人工测试集去掉中性（label 2）后剩 359 条。

`analyze.py` 另外需要 `data/processed/*.csv`（原文），用来列出错误样本。

## 快速开始

```bash
cd models/textcnn
pip install -r requirements.txt

# 冒烟测试（1 分钟内，只检查能否跑通）
python textcnn.py --max_len 50 --subset 5000 --epochs 2 --out runs/smoke

# 最终结果：全量训练、5 个种子，并在测试集 + 人工测试集上评估一次
python textcnn.py --max_len 50 --seeds 5 --weight_decay 1e-4 --kernels 2,3,4,5 \
    --epochs 20 --eval_test --save_model --out runs/final

# 消融实验（20 万训练 / 2 万验证，3 个种子，只看验证集）
python run_ablations.py --max_len 50 --seeds 3 --out runs/ablation

# 误差分析（需要上面的 --eval_test --save_model）
python analyze.py --run_dir runs/final/seed0 --split test --ngrams
python analyze.py --run_dir runs/final/seed0 --split manual --ngrams
```

> 调参只用验证集。测试集和人工测试集只有加 `--eval_test` 才会被评估，只在配置固定后用一次。

## 文件

| 文件 | 作用 |
|---|---|
| `textcnn.py` | 模型 + 训练 + 评估（多种子、早停、`--glove` 预训练词向量可选、`--merge_val`） |
| `tc_utils.py` | 指标（accuracy / P / R / F1 / macro-F1 / AUC）、McNemar、数据读取、随机种子 |
| `run_ablations.py` | 消融：核宽、句长、dropout、词向量维度、滤波器数、权重衰减 |
| `analyze.py` | 训练曲线、混淆矩阵、按句长准确率、最自信的错误样本、各滤波器最强激活 n-gram |
| `TextCNN_Sentiment140.ipynb` | 带讲解的 notebook：调用上面的脚本，内含结构图、消融图、结果图和误差分析；默认不重训，可直接 Run All |
| `results/` | 最终结果 `RESULTS.md` 和图 `figures/` |

`runs/`（训练输出、模型权重）已在 `.gitignore` 中，不提交。

## 最终结果（128 万条训练，5 个种子，均值 ± 标准差）

| 数据集 | 准确率 | 精确率(正) | 召回率(正) | F1(正) | macro-F1 | AUC |
|---|---|---|---|---|---|---|
| 验证集 16 万 | 0.8163 ± 0.0015 | 0.8179 ± 0.0051 | 0.8138 ± 0.0086 | 0.8158 ± 0.0024 | 0.8163 ± 0.0015 | 0.8965 ± 0.0012 |
| **测试集 16 万** | **0.8166 ± 0.0015** | 0.8170 ± 0.0053 | 0.8160 ± 0.0088 | 0.8165 ± 0.0024 | 0.8166 ± 0.0015 | 0.8973 ± 0.0014 |
| 人工测试集 359 | 0.8201 ± 0.0133 | 0.7975 ± 0.0146 | 0.8648 ± 0.0138 | 0.8297 ± 0.0122 | 0.8195 ± 0.0134 | 0.9106 ± 0.0074 |

验证集与测试集几乎相同，没有对验证集过拟合。人工测试集只有 359 条，准确率的误差范围约 ±4 个百分点，不能说它比测试集更高。完整结果、消融表和误差分析见 [`results/RESULTS.md`](results/RESULTS.md)。

## 主要发现

- **多种卷积核有用**：核宽 2/3/4/5 比 3/4/5 高约 0.7 个百分点（验证集，p = 0.005）；只用核宽 3 显著变差。
- 词向量维度（64）、滤波器数（200）、句长减半（25）对结果影响不大；dropout 0.7 偏大。
- 权重衰减 1e-4 在 20 万条规模下约 +0.8 个百分点，在全量数据上缩小到约 +0.1 个百分点（单种子）。
- 测试集的错误大量来自 Sentiment140 的远程监督标签噪声（最自信的 50 条错误里至多约 2 条像模型自身缺陷）；人工测试集的错误更多来自针对对象的批评/讽刺、表情符号被丢弃和俚语。
