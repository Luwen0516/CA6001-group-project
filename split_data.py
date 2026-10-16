"""
Sentiment140 数据划分脚本
- 读取本地原始 CSV (Stanford 原始格式, 6列无表头)
- 从 160 万条自动标注数据中按 8:1:1 分层抽样划分 train/val/test
- 498 条人工标注测试集单独保存为补充测试集
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import os

# 路径设置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

RANDOM_STATE = 42  # 固定随机种子，保证所有人结果一致

# 原始 CSV 列名 (Stanford 原始格式: 6列, 无表头)
COLUMNS = ["sentiment", "id", "date", "flag", "user", "text"]

print("=" * 50)
print("Step 1: 读取原始数据...")
print("=" * 50)

# 读取 160 万条自动标注训练集
train_full = pd.read_csv(
    os.path.join(RAW_DIR, "training.1600000.processed.noemoticon.csv"),
    encoding="latin-1",
    names=COLUMNS
)
print(f"自动标注训练集总量: {len(train_full):,} 条")

# 读取 498 条人工标注测试集
manual_test = pd.read_csv(
    os.path.join(RAW_DIR, "testdata.manual.2009.06.14.csv"),
    encoding="latin-1",
    names=COLUMNS
)
print(f"人工标注测试集: {len(manual_test)} 条")

# 标签分布
print("\n自动标注集标签分布:")
print(train_full["sentiment"].value_counts().sort_index())

print("\n人工标注测试集标签分布:")
print(manual_test["sentiment"].value_counts().sort_index())

# 只保留 0 (negative) 和 4 (positive)，去掉极少量的 2 (neutral)
train_full = train_full[train_full["sentiment"].isin([0, 4])].copy()
print(f"\n去掉 neutral (sentiment=2) 后剩余: {len(train_full):,} 条")

# 标签简化: 0 -> 0 (negative), 4 -> 1 (positive)
train_full["label"] = train_full["sentiment"].apply(lambda x: 1 if x == 4 else 0)

# 只保留需要的列: text, label
df = train_full[["text", "label"]].copy()
print(f"\n最终用于划分的数据: {len(df):,} 条")
print(f"正面(1): {(df['label']==1).sum():,} 条")
print(f"负面(0): {(df['label']==0).sum():,} 条")

print("\n" + "=" * 50)
print("Step 2: 按 8:1:1 分层抽样划分...")
print("=" * 50)

# 第一步: 分出 test (10%)
train_val_df, test_df = train_test_split(
    df, test_size=0.1, random_state=RANDOM_STATE, stratify=df["label"]
)

# 第二步: 从剩余 90% 中分出 val (占总量 10%，即剩余的 1/9)
train_df, val_df = train_test_split(
    train_val_df, test_size=1/9, random_state=RANDOM_STATE, stratify=train_val_df["label"]
)

print(f"\nTrain: {len(train_df):,} 条 ({len(train_df)/len(df)*100:.1f}%)")
print(f"Val:   {len(val_df):,} 条 ({len(val_df)/len(df)*100:.1f}%)")
print(f"Test:  {len(test_df):,} 条 ({len(test_df)/len(df)*100:.1f}%)")

# 验证各集合标签比例
print("\n各集合标签分布:")
for name, d in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
    pos = (d["label"] == 1).sum()
    neg = (d["label"] == 0).sum()
    print(f"  {name}: 正面={pos:,} ({pos/len(d)*100:.1f}%), 负面={neg:,} ({neg/len(d)*100:.1f}%)")

print("\n" + "=" * 50)
print("Step 3: 保存数据...")
print("=" * 50)

# 保存划分后的数据
train_df.to_csv(os.path.join(PROCESSED_DIR, "train.csv"), index=False)
val_df.to_csv(os.path.join(PROCESSED_DIR, "val.csv"), index=False)
test_df.to_csv(os.path.join(PROCESSED_DIR, "test.csv"), index=False)

# 人工标注测试集也做标签映射
manual_test["label"] = manual_test["sentiment"].apply(lambda x: 1 if x == 4 else (0 if x == 0 else 2))
manual_test[["text", "label"]].to_csv(os.path.join(PROCESSED_DIR, "manual_test.csv"), index=False)

print(f"划分后数据已保存到 data/processed/")
print(f"  - train.csv ({len(train_df):,} 条)")
print(f"  - val.csv ({len(val_df):,} 条)")
print(f"  - test.csv ({len(test_df):,} 条)")
print(f"  - manual_test.csv ({len(manual_test)} 条, 人工标注)")

print("\n" + "=" * 50)
print("完成! 随机种子 =", RANDOM_STATE)
print("=" * 50)
