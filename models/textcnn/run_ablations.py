#!/usr/bin/env python3
"""
TextCNN 消融实验：多个配置 × 多个随机种子，只使用验证集（不碰测试集）。

每次只改动一个因素，和 base 配置对比：
  A 词向量   : glove_frozen / glove_finetune          （需要 --glove，推荐 glove.twitter.27B.100d.txt）
  B 卷积核   : k3 / k2345
  C 序列长度 : maxlen_half（base 的一半）
  D 正则化   : drop0.3 / drop0.7 / wd1e-4
  F 模型容量 : emb64 / filters200

160 万条推特全量训练较慢，消融实验默认只用 20 万条训练样本和 2 万条验证样本（--max_train / --max_val）。
最终结果再用全量训练集重跑选定的配置。

用法：
    python run_ablations.py --seeds 3 --out runs/ablation
    python run_ablations.py --seeds 3 --glove glove.twitter.27B.100d.txt
    python run_ablations.py --groups B,C --seeds 3           # 只跑部分组
    python run_ablations.py --subset 5000 --epochs 2 --seeds 1 --out runs/ablation_smoke   # 冒烟测试
已完成的运行（有 metrics.json）会被跳过，所以中断后重跑同一条命令即可继续；加 --force 强制重跑。
"""
import argparse
import json
import os

import numpy as np

from tc_utils import load_data, mcnemar, mean_std, to_jsonable
import textcnn


def build_experiments(args):
    base = dict(data_dir=args.data_dir, max_len=args.max_len, emb_dim=128, kernels="3,4,5", num_filters=100,
                dropout=0.5, weight_decay=0.0, lr=1e-3, batch_size=256, epochs=args.epochs, patience=2,
                device=args.device, num_threads=args.num_threads, subset=args.subset,
                max_train=args.max_train, max_val=args.max_val)
    E = {"base": ("0", {})}
    if args.glove:
        E["glove_frozen"] = ("A", dict(glove=args.glove, freeze_emb=True))
        E["glove_finetune"] = ("A", dict(glove=args.glove))
    E["k3"] = ("B", dict(kernels="3"))
    E["k2345"] = ("B", dict(kernels="2,3,4,5"))
    E["maxlen_half"] = ("C", dict(max_len=max(8, args.max_len // 2)))
    E["drop0.3"] = ("D", dict(dropout=0.3))
    E["drop0.7"] = ("D", dict(dropout=0.7))
    E["wd1e-4"] = ("D", dict(weight_decay=1e-4))
    E["emb64"] = ("F", dict(emb_dim=64))
    E["filters200"] = ("F", dict(num_filters=200))
    return base, E


def main():
    ap = argparse.ArgumentParser(description="TextCNN 消融实验（仅验证集）")
    ap.add_argument("--data_dir", default="../../data/tokenized")
    ap.add_argument("--out", default="runs/ablation")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--max_train", type=int, default=200000, help="训练集取前 N 条（0 表示全量）")
    ap.add_argument("--max_val", type=int, default=20000, help="验证集取前 N 条（0 表示全量）")
    ap.add_argument("--groups", default="all", help="如 B,C,D；all 表示全部")
    ap.add_argument("--glove", default=None)
    ap.add_argument("--subset", type=int, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--num_threads", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    args.max_train, args.max_val = args.max_train or None, args.max_val or None

    base_cfg, exps = build_experiments(args)
    want = None if args.groups == "all" else set(args.groups.upper().split(","))
    names = [n for n, (g, _) in exps.items() if n == "base" or want is None or g in want]
    print(f"[计划] {len(names)} 个配置 × {args.seeds} 个种子 = {len(names) * args.seeds} 次训练：{names}")

    data_cache = {}

    def get_data(cfg):
        key = (cfg["data_dir"], cfg["max_len"], cfg["subset"], cfg["max_train"], cfg["max_val"])
        if key not in data_cache:
            data_cache[key] = load_data(cfg["data_dir"], cfg["max_len"], cfg["subset"], False,
                                        cfg["max_train"], cfg["max_val"])
        return data_cache[key]

    results = {}
    for name in names:
        group, over = exps[name]
        cfg0 = {**base_cfg, **over}
        results[name] = []
        for s in range(args.seeds):
            out_dir = os.path.join(args.out, name, f"seed{s}")
            mpath = os.path.join(out_dir, "metrics.json")
            if os.path.exists(mpath) and not args.force:
                with open(mpath, encoding="utf-8") as f:
                    r = json.load(f)
                print(f"[跳过] {name} seed{s}（已有结果）验证集 acc={r['val']['accuracy']:.4f}")
            else:
                print(f"\n[运行] {name}（组 {group}） seed{s}")
                r = textcnn.run_one({**cfg0, "seed": s}, out_dir, data=get_data(cfg0))
            results[name].append(r)

    # ---- 汇总 ----
    base_val = get_data(base_cfg)
    y_val = base_val["y_val"]
    base_p0 = np.load(os.path.join(args.out, "base", "seed0", "val_probs.npy"))
    rows = []
    for name in names:
        rs = results[name]
        acc = mean_std([r["val"]["accuracy"] for r in rs])
        f1 = mean_std([r["val"]["macro_f1"] for r in rs])
        auc = mean_std([r["val"]["auc"] for r in rs])
        row = dict(config=name, group=exps[name][0], seeds=len(rs),
                   val_acc_mean=acc[0], val_acc_std=acc[1], val_f1_mean=f1[0], val_f1_std=f1[1],
                   val_auc_mean=auc[0], val_auc_std=auc[1],
                   best_epoch_mean=float(np.mean([r["best_epoch"] for r in rs])),
                   params=rs[0]["n_params"], sec_per_epoch=float(np.mean([r["sec_per_epoch"] for r in rs])))
        if name != "base":
            p = np.load(os.path.join(args.out, name, "seed0", "val_probs.npy"))
            _, pval, _, _ = mcnemar(y_val, (base_p0 >= .5).astype(int), (p >= .5).astype(int))
            row["mcnemar_p_vs_base(seed0)"] = pval
        rows.append(row)
    base_acc = rows[0]["val_acc_mean"]
    for r in rows:
        r["delta_acc_vs_base"] = r["val_acc_mean"] - base_acc

    import pandas as pd
    df = pd.DataFrame(rows)
    os.makedirs(args.out, exist_ok=True)
    df.to_csv(os.path.join(args.out, "ablation_summary.csv"), index=False)
    lines = ["| 配置 | 组 | 验证集准确率 | Macro-F1 | AUC | Δ准确率 | McNemar p | 最佳轮次 | 参数量 | 秒/轮 |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        p = r.get("mcnemar_p_vs_base(seed0)")
        lines.append(f"| {r['config']} | {r['group']} | {r['val_acc_mean']:.4f} ± {r['val_acc_std']:.4f} | "
                     f"{r['val_f1_mean']:.4f} ± {r['val_f1_std']:.4f} | {r['val_auc_mean']:.4f} ± {r['val_auc_std']:.4f} | "
                     f"{r['delta_acc_vs_base']:+.4f} | {'-' if p is None else f'{p:.3f}'} | "
                     f"{r['best_epoch_mean']:.1f} | {r['params']:,} | {r['sec_per_epoch']:.0f} |")
    md = "\n".join(lines)
    with open(os.path.join(args.out, "ablation_summary.md"), "w", encoding="utf-8") as f:
        f.write(md + "\n")
    print("\n" + md)
    print(f"\n汇总已保存：{os.path.join(args.out, 'ablation_summary.csv')} / .md")
    print("提示：差异小于一个标准差、或 McNemar p > 0.05 的配置，不要在报告里声称\"更好\"。")


if __name__ == "__main__":
    main()
