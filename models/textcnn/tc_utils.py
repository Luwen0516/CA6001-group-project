"""TextCNN 项目的通用函数（只依赖 numpy / pandas，不依赖 PyTorch，便于单独测试）。"""
import json
import math
import os
import random

import numpy as np

PAD_ID, UNK_ID = 0, 1


# --------------------------------------------------------------------------- #
# 数据
# --------------------------------------------------------------------------- #
def load_data(data_dir, max_len=None, subset=None, merge_val=False, max_train=None, max_val=None):
    """读取预处理好的 data.npz / vocab.json（格式见 README「数据接口」）。

    max_len   : 比存储长度更短时，取每条文本的前 max_len 个词（与预处理的 head 截断一致）
    subset    : 冒烟测试：训练取前 subset 条，验证/测试取前 subset//5 条
    merge_val : 把验证集并入训练集（最终模型用完整训练池训练时使用）
    max_train : 训练集最多取前 N 条（消融实验提速；组内划分本身是随机的，所以取前 N 条等价于随机子集）
    max_val   : 验证集最多取前 N 条（测试集和人工测试集永远不截取）
    返回 dict: x_train, len_train, y_train, x_val, ..., x_test, ...（若有：x_manual, ...），stoi, vocab_size
    """
    d = np.load(os.path.join(data_dir, "data.npz"))
    out = {k: d[k] for k in d.files}
    with open(os.path.join(data_dir, "vocab.json"), encoding="utf-8") as f:
        stoi = json.load(f)["stoi"]
    out["stoi"], out["vocab_size"] = stoi, len(stoi)
    splits = [s for s in ("train", "val", "test", "manual") if f"x_{s}" in out]

    width = out["x_train"].shape[1]
    if max_len is not None:
        if max_len > width:
            print(f"[提示] max_len={max_len} 大于预处理时的长度 {width}，改用 {width}")
            max_len = width
        for s in splits:
            out[f"x_{s}"] = out[f"x_{s}"][:, :max_len]
            out[f"len_{s}"] = np.minimum(out[f"len_{s}"], max_len)
    if merge_val:
        for k in ("x", "len", "y"):
            out[f"{k}_train"] = np.concatenate([out[f"{k}_train"], out[f"{k}_val"]])
    if max_train:
        for k in ("x", "len", "y"):
            out[f"{k}_train"] = out[f"{k}_train"][:max_train]
    if max_val:
        for k in ("x", "len", "y"):
            out[f"{k}_val"] = out[f"{k}_val"][:max_val]
    if subset:
        for k in ("x", "len", "y"):
            out[f"{k}_train"] = out[f"{k}_train"][:subset]
            out[f"{k}_val"] = out[f"{k}_val"][: max(subset // 5, 50)]
            out[f"{k}_test"] = out[f"{k}_test"][: max(subset // 5, 50)]
    return out


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# --------------------------------------------------------------------------- #
# GloVe 词向量（可选）
# --------------------------------------------------------------------------- #
def load_glove(path, stoi, cache_dir=None, seed=0):
    """按词表 stoi 构建嵌入矩阵。词表里没有出现在 GloVe 的词用 U(-0.25, 0.25) 随机初始化，<pad> 为 0。
    返回 (matrix[V, dim], coverage)，coverage 为词表中（不含 pad/unk）找到向量的词所占比例。"""
    V = len(stoi)
    cache = None
    if cache_dir:
        tag = os.path.basename(path).replace(".txt", "")
        cache = os.path.join(cache_dir, f"emb_{tag}_V{V}.npy")
        if os.path.exists(cache):
            m = np.load(cache)
            cov = float(np.mean(np.abs(m[2:]).sum(1) > 0))
            return m, cov
    rng = np.random.RandomState(seed)
    found, dim, vecs = 0, None, {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():  # word2vec 文本格式的首行
                continue
            w = parts[0]
            if w not in stoi or w in vecs:
                continue
            v = np.asarray(parts[1:], dtype=np.float32)
            if dim is None:
                dim = v.shape[0]
            if v.shape[0] != dim:
                continue
            vecs[w] = v
    if dim is None:
        raise ValueError(f"在 {path} 中没有找到任何词表中的词，请检查文件格式")
    m = rng.uniform(-0.25, 0.25, size=(V, dim)).astype(np.float32)
    m[PAD_ID] = 0.0
    for w, v in vecs.items():
        m[stoi[w]] = v
        found += 1
    cov = found / max(1, V - 2)
    if cache:
        np.save(cache, m)
    return m, cov


# --------------------------------------------------------------------------- #
# 指标
# --------------------------------------------------------------------------- #
def roc_auc(y, p):
    """ROC-AUC（Mann–Whitney 形式，对相同分数取平均秩）。"""
    y, p = np.asarray(y), np.asarray(p, dtype=np.float64)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    uniq, inv, cnt = np.unique(p, return_inverse=True, return_counts=True)
    ends = np.cumsum(cnt)
    avg_rank = ends - (cnt - 1) / 2.0          # 每个唯一值的平均秩（从 1 开始）
    ranks = avg_rank[inv]
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def binary_metrics(y, prob):
    """y: 0/1 标签；prob: 预测为正面的概率。"""
    y, prob = np.asarray(y).astype(int), np.asarray(prob, dtype=np.float64)
    pred = (prob >= 0.5).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())

    def prf(tp_, fp_, fn_):
        p = tp_ / (tp_ + fp_) if tp_ + fp_ else 0.0
        r = tp_ / (tp_ + fn_) if tp_ + fn_ else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        return p, r, f

    p1, r1, f1 = prf(tp, fp, fn)          # 正面类
    p0, r0, f0 = prf(tn, fn, fp)          # 负面类
    eps = 1e-12
    ll = -np.mean(y * np.log(prob + eps) + (1 - y) * np.log(1 - prob + eps))
    return {
        "accuracy": (tp + tn) / max(1, len(y)),
        "precision_pos": p1, "recall_pos": r1, "f1_pos": f1,
        "precision_neg": p0, "recall_neg": r0, "f1_neg": f0,
        "macro_f1": (f1 + f0) / 2,
        "auc": roc_auc(y, prob),
        "log_loss": float(ll),
        "confusion": [[tn, fp], [fn, tp]],   # 行=真实(负,正)，列=预测(负,正)
        "n": int(len(y)),
    }


def mcnemar(y, pred_a, pred_b):
    """McNemar 检验（带连续性修正）。返回 (chi2, p, b, c)：b=只有 A 对，c=只有 B 对。"""
    y, a, b_ = np.asarray(y), np.asarray(pred_a), np.asarray(pred_b)
    b = int(((a == y) & (b_ != y)).sum())
    c = int(((a != y) & (b_ == y)).sum())
    if b + c == 0:
        return 0.0, 1.0, b, c
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p = math.erfc(math.sqrt(chi2 / 2.0))       # 自由度 1 的卡方分布上尾概率
    return float(chi2), float(p), b, c


def group_accuracy(y, pred, groups, labels):
    """按分组计算准确率。groups 为整数组号，labels 为每组名称。返回 [(label, n, acc)]。"""
    y, pred, groups = np.asarray(y), np.asarray(pred), np.asarray(groups)
    rows = []
    for g, name in enumerate(labels):
        m = groups == g
        rows.append((name, int(m.sum()), float((y[m] == pred[m]).mean()) if m.any() else float("nan")))
    return rows


def mean_std(values):
    v = np.asarray(values, dtype=np.float64)
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0


def to_jsonable(o):
    if isinstance(o, dict):
        return {k: to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o
