#!/usr/bin/env python3
"""
对一次 TextCNN 运行做结果分析和作图（图中文字用英文，方便直接放进报告）。

输入：run_dir（含 metrics.json、{split}_probs.npy；--ngrams 还需要 model.pt）
输出（默认保存到 run_dir/analysis_{split}/）：
    metrics_{split}.json      完整指标
    curves.png                训练/验证曲线
    confusion.png             混淆矩阵
    by_length.png/.csv        按评论长度分组的准确率
    by_rating.png/.csv        按原始评分分组的准确率（仅当 csv 里有 rating 列，如 IMDB）
    errors.csv                最自信的错误预测（用于人工归类）
    ngrams.csv/.md            每个卷积核最强激活的 n-gram（需 --ngrams）

用法：
    python analyze.py --run_dir runs/final/seed0 --split test
    python analyze.py --run_dir runs/final/seed0 --split manual      # 人工标注测试集（已去掉中性）
    python analyze.py --run_dir runs/final/seed0 --split test --ngrams
    （只在最终结果上对 test/manual 做分析；调参阶段请用 --split val）
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from tc_utils import binary_metrics, group_accuracy, load_data, to_jsonable

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)
    ax.title.set_color(INK)


def new_fig(*a, **k):
    fig, ax = plt.subplots(*a, **k)
    fig.patch.set_facecolor(SURF)
    return fig, ax


def plot_curves(hist, path):
    ep = [h["epoch"] for h in hist]
    has_val = "val_acc" in hist[0]
    fig, axes = new_fig(1, 2, figsize=(9, 3.4))
    for ax, key, title in ((axes[0], "loss", "Loss"), (axes[1], "acc", "Accuracy")):
        style(ax)
        ax.plot(ep, [h[f"train_{key}"] for h in hist], color=BLUE, lw=2, marker="o", ms=4, label="train")
        if has_val:
            ax.plot(ep, [h[f"val_{key}"] for h in hist], color=ORANGE, lw=2, marker="o", ms=4, label="validation")
            best = int(np.argmax([h["val_acc"] for h in hist])) + 1
            ax.axvline(best, color=GRID, lw=1.2, ls="--")
        ax.set_xlabel("epoch"); ax.set_title(title, fontsize=11, loc="left")
        ax.set_xticks(ep)
        ax.legend(frameon=False, fontsize=9, labelcolor=INK2)
    fig.tight_layout(); fig.savefig(path, dpi=200, facecolor=SURF); plt.close(fig)


def plot_confusion(cm, path):
    cm = np.asarray(cm)
    frac = cm / cm.sum(1, keepdims=True)
    cmap = LinearSegmentedColormap.from_list("b", [SURF, BLUE])
    fig, ax = new_fig(figsize=(3.8, 3.4))
    ax.imshow(frac, cmap=cmap, vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}\n({frac[i, j]:.1%})", ha="center", va="center", fontsize=10,
                    color="white" if frac[i, j] > 0.55 else INK)
    ax.set_xticks([0, 1], ["negative", "positive"], color=INK2)
    ax.set_yticks([0, 1], ["negative", "positive"], color=INK2)
    ax.set_xlabel("predicted", color=INK2); ax.set_ylabel("true", color=INK2)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=200, facecolor=SURF); plt.close(fig)


def plot_bars(rows, xlabel, path, overall):
    labels = [r[0] for r in rows]; acc = [r[2] for r in rows]; n = [r[1] for r in rows]
    fig, ax = new_fig(figsize=(max(5, 0.9 * len(rows) + 2), 3.4))
    style(ax)
    x = np.arange(len(rows))
    ax.bar(x, acc, color=BLUE, width=0.6, zorder=2)
    ax.axhline(overall, color=ORANGE, lw=1.5, ls="--", zorder=3, label=f"overall accuracy {overall:.1%}")
    for xi, a, k in zip(x, acc, n):       # 标签带底色，避免被虚线穿过
        ax.text(xi, a + 0.012, f"{a:.1%}\nn={k:,}", ha="center", va="bottom", fontsize=8, color=INK2, zorder=4,
                bbox=dict(facecolor=SURF, edgecolor="none", pad=1.0))
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.22); ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_ylabel("accuracy"); ax.set_xlabel(xlabel)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper left", ncol=1)
    fig.tight_layout(); fig.savefig(path, dpi=200, facecolor=SURF); plt.close(fig)


def top_ngrams(run_dir, data, itos, top_filters, split="val"):
    """每个卷积核（按其对"正面-负面"logit 的权重选出最强的若干个）在评论集上激活最强的 n-gram。"""
    import torch
    from textcnn import TextCNN
    ck = torch.load(os.path.join(run_dir, "model.pt"), map_location="cpu", weights_only=False)
    model = TextCNN(ck["vocab_size"], ck["emb_dim"], ck["kernels"], ck["num_filters"], dropout=0.0)
    model.load_state_dict(ck["state_dict"]); model.eval()
    F_ = ck["num_filters"]
    w = (model.fc.weight[1] - model.fc.weight[0]).detach().numpy()          # 每个 filter 对 (正-负) 的权重
    order = np.argsort(w)
    picks = list(order[-top_filters:][::-1]) + list(order[:top_filters])     # 最偏正面 + 最偏负面
    X, L = data[f"x_{split}"], data[f"len_{split}"]
    best = {int(f): (-1.0, None, None) for f in picks}
    with torch.no_grad():
        for s in range(0, len(X), 256):
            xb = torch.from_numpy(X[s:s + 256].astype(np.int64)); lb = torch.from_numpy(L[s:s + 256].astype(np.int64))
            feats = model.features(xb, lb)
            for f in best:
                g, j = divmod(f, F_)
                val, pos = feats[g][:, j, :].max(dim=1)
                i = int(val.argmax())
                if float(val[i]) > best[f][0]:
                    k = ck["kernels"][g]
                    ids = X[s + i, int(pos[i]):int(pos[i]) + k]
                    best[f] = (float(val[i]), " ".join(itos.get(int(t), "?") for t in ids), k)
    rows = [dict(filter=f, kernel_width=best[f][2], weight_pos_minus_neg=float(w[f]),
                 max_activation=best[f][0], top_ngram=best[f][1],
                 leans="positive" if w[f] > 0 else "negative") for f in best]
    return pd.DataFrame(rows).sort_values("weight_pos_minus_neg", ascending=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--data_dir", default="../../data/tokenized", help="含 data.npz / vocab.json 的目录")
    ap.add_argument("--csv_dir", default="../../data/processed", help="含 train/val/test/manual_test.csv（原文）的目录")
    ap.add_argument("--length_edges", default=None, help="长度分组边界，逗号分隔；默认按 max_len 自动选择")
    ap.add_argument("--split", choices=["val", "test", "manual"], default="test")
    ap.add_argument("--out", default=None)
    ap.add_argument("--ngrams", action="store_true", help="提取最强激活 n-gram（需要 model.pt，即训练时加 --save_model）")
    ap.add_argument("--top_filters", type=int, default=8)
    ap.add_argument("--n_errors", type=int, default=50)
    args = ap.parse_args()

    out = args.out or os.path.join(args.run_dir, f"analysis_{args.split}")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(args.run_dir, "metrics.json"), encoding="utf-8") as f:
        run = json.load(f)
    cfg = run["config"]
    pfile = os.path.join(args.run_dir, f"{args.split}_probs.npy")
    if not os.path.exists(pfile):
        raise SystemExit(f"找不到 {pfile}。测试集分析需要训练时加 --eval_test。")
    probs = np.load(pfile)

    data = load_data(args.data_dir, cfg["max_len"])
    y, L = data[f"y_{args.split}"], data[f"len_{args.split}"]
    csv_name = "manual_test.csv" if args.split == "manual" else f"{args.split}.csv"
    df = pd.read_csv(os.path.join(args.csv_dir, csv_name), keep_default_na=False)
    if args.split == "manual":                       # 与预处理一致：去掉中性(label=2)
        df = df[df["label"].isin([0, 1])].reset_index(drop=True)
    if args.split == "val" and len(df) != len(y):    # 训练时可能用 --max_val 截取了验证集
        df = df.iloc[:len(y)].reset_index(drop=True)
    if not (len(df) == len(y) == len(probs) and (df["label"].values == y).all()):
        raise SystemExit("预测结果与数据文件对不上：请确认 data_dir / csv_dir 与训练时一致。")
    pred = (probs >= 0.5).astype(int)
    m = binary_metrics(y, probs)
    with open(os.path.join(out, f"metrics_{args.split}.json"), "w", encoding="utf-8") as f:
        json.dump(to_jsonable(m), f, indent=2)
    print(f"[{args.split}] acc={m['accuracy']:.4f} macro-F1={m['macro_f1']:.4f} AUC={m['auc']:.4f} "
          f"P(pos)={m['precision_pos']:.4f} R(pos)={m['recall_pos']:.4f}")

    if run.get("history"):
        plot_curves(run["history"], os.path.join(out, "curves.png"))
    plot_confusion(m["confusion"], os.path.join(out, "confusion.png"))

    ml = cfg["max_len"]
    if args.length_edges:
        edges = [int(e) for e in args.length_edges.split(",")]
    else:
        edges = [8, 15, 25] if ml <= 100 else [100, 250, ml]
    groups = np.digitize(L, edges)
    names = [f"<{edges[0]}"] + [f"{a}-{b - 1}" for a, b in zip(edges[:-1], edges[1:])] + [f">={edges[-1]}"]
    rows = group_accuracy(y, pred, groups, names)
    pd.DataFrame(rows, columns=["length_group", "n", "accuracy"]).to_csv(os.path.join(out, "by_length.csv"), index=False)
    plot_bars([r for r in rows if r[1] > 0], "length (tokens)", os.path.join(out, "by_length.png"), m["accuracy"])

    if "rating" in df.columns:                       # IMDB 才有原始评分
        ratings = sorted(df["rating"].unique())
        g = np.array([ratings.index(r) for r in df["rating"]])
        rows = group_accuracy(y, pred, g, [str(r) for r in ratings])
        pd.DataFrame(rows, columns=["rating", "n", "accuracy"]).to_csv(os.path.join(out, "by_rating.csv"), index=False)
        plot_bars(rows, "original IMDB rating (1-4 negative, 7-10 positive)", os.path.join(out, "by_rating.png"), m["accuracy"])

    wrong = df[pred != y].copy()
    wrong["prob_pos"] = probs[pred != y]
    wrong["confidence"] = np.abs(wrong["prob_pos"] - 0.5) * 2
    wrong = wrong.sort_values("confidence", ascending=False).head(args.n_errors)
    wrong["text"] = wrong["text"].astype(str).str.slice(0, 600)
    cols = [c for c in ("src_id", "movie", "rating", "label", "prob_pos", "confidence", "text") if c in wrong.columns]
    wrong[cols].to_csv(os.path.join(out, "errors.csv"), index=False, encoding="utf-8-sig")
    print(f"[错误样本] 共 {int((pred != y).sum())} 条错误，已保存最自信的 {len(wrong)} 条到 errors.csv，请人工归类（讽刺/否定/先抑后扬/评分与文字不符等）")

    if args.ngrams:
        if not os.path.exists(os.path.join(args.run_dir, "model.pt")):
            print("[n-gram] 没有 model.pt，训练时请加 --save_model")
        else:
            itos = {i: w for w, i in data["stoi"].items()}
            ng = top_ngrams(args.run_dir, data, itos, args.top_filters, split="val")   # 在验证集上找最强激活 n-gram
            ng.to_csv(os.path.join(out, "ngrams.csv"), index=False)
            with open(os.path.join(out, "ngrams.md"), "w", encoding="utf-8") as f:
                f.write("| leans | kernel | weight(pos-neg) | activation | strongest n-gram (validation set) |\n|---|---|---|---|---|\n")
                for r in ng.itertuples():
                    f.write(f"| {r.leans} | {r.kernel_width} | {r.weight_pos_minus_neg:+.2f} | {r.max_activation:.2f} | {r.top_ngram} |\n")
            print(f"[n-gram] 已保存到 {os.path.join(out, 'ngrams.md')}")
    print(f"[完成] 输出目录：{os.path.abspath(out)}")


if __name__ == "__main__":
    main()
