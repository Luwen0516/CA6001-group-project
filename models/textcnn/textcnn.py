#!/usr/bin/env python3
"""
TextCNN（Kim, 2014）用于 Sentiment140 推特情感二分类。读取组内统一预处理得到的 data.npz / vocab.json（格式见 README「数据接口」）。

结构：Embedding -> 多个并行 1D 卷积(不同核宽，每个核宽 num_filters 个) -> ReLU
      -> max-over-time pooling -> 拼接 -> Dropout -> 全连接 -> 2 类
训练：Adam + 交叉熵，按验证集准确率早停并恢复最佳权重。测试集默认不碰，只有加 --eval_test 才评估
      （同时评估人工标注测试集 manual，若数据里有）。

常用命令（在项目根目录执行）：
    # 冒烟测试（1 分钟内跑完，只检查代码能否跑通）
    python textcnn.py --subset 5000 --epochs 2 --out runs/smoke
    # 主配置：为节省时间，训练集取 20 万条、验证集取 2 万条，3 个种子，只看验证集
    python textcnn.py --max_train 200000 --max_val 20000 --seeds 3 --out runs/base
    # 最终结果：全量训练集、5 个种子，并在测试集 + 人工测试集上评估一次
    python textcnn.py --seeds 5 --eval_test --save_model --out runs/final
    # 用完整训练池（训练+验证）重新训练（固定轮数，需和上一步的最佳轮数一致）
    python textcnn.py --merge_val --epochs 5 --eval_test --out runs/final_full
"""
import argparse
import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tc_utils import (PAD_ID, binary_metrics, load_data, load_glove, mean_std, set_seed,
                      to_jsonable)

DEFAULTS = dict(
    data_dir="../../data/tokenized", max_len=64, emb_dim=128, kernels="3,4,5", num_filters=100,
    dropout=0.5, lr=1e-3, weight_decay=0.0, batch_size=256, epochs=10, patience=2,
    glove=None, freeze_emb=False, eval_test=False, save_model=False, merge_val=False,
    subset=None, max_train=None, max_val=None, device="auto", num_threads=None, seed=0,
)


# --------------------------------------------------------------------------- #
# 模型
# --------------------------------------------------------------------------- #
class TextCNN(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, kernel_sizes=(3, 4, 5), num_filters=100,
                 dropout=0.5, pretrained=None, freeze_emb=False, num_classes=2):
        super().__init__()
        self.kernel_sizes = [int(k) for k in kernel_sizes]
        self.num_filters = num_filters
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_ID)
        if pretrained is not None:
            self.emb.weight.data.copy_(torch.from_numpy(pretrained))
        self.emb.weight.data[PAD_ID].zero_()
        self.emb.weight.requires_grad = not freeze_emb
        self.convs = nn.ModuleList([nn.Conv1d(emb_dim, num_filters, k) for k in self.kernel_sizes])
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(self.kernel_sizes), num_classes)

    def features(self, x, lengths):
        """返回每个核宽的卷积激活列表，每项形状 (B, num_filters, L-k+1)。
        窗口只要包含补零位置（i + k > 真实长度）就置 0，避免补零影响 max pooling。"""
        e = self.emb(x).transpose(1, 2)                      # (B, emb_dim, L)
        outs = []
        for k, conv in zip(self.kernel_sizes, self.convs):
            h = F.relu(conv(e))                              # (B, F, T)
            pos = torch.arange(h.size(2), device=x.device).unsqueeze(0)    # (1, T)
            valid = ((pos + k) <= lengths.unsqueeze(1)).to(h.dtype)        # (B, T)
            outs.append(h * valid.unsqueeze(1))
        return outs

    def forward(self, x, lengths):
        pooled = [h.max(dim=2).values for h in self.features(x, lengths)]  # 每项 (B, F)
        return self.fc(self.drop(torch.cat(pooled, dim=1)))


# --------------------------------------------------------------------------- #
# 批处理（同一批内长度相近，减少补零带来的计算浪费）
# --------------------------------------------------------------------------- #
def make_batches(lengths, bs, rng, shuffle, chunk=50):
    n = len(lengths)
    if shuffle:
        idx, batches = rng.permutation(n), []
        for s in range(0, n, bs * chunk):
            c = idx[s:s + bs * chunk]
            c = c[np.argsort(lengths[c], kind="stable")]
            batches += [c[i:i + bs] for i in range(0, len(c), bs)]
        return [batches[i] for i in rng.permutation(len(batches))]
    idx = np.argsort(lengths, kind="stable")
    return [idx[i:i + bs] for i in range(0, n, bs)]


def to_batch(X, L, y, b, min_len, device):
    width = int(min(max(L[b].max(), min_len), X.shape[1]))
    xb = torch.from_numpy(X[b][:, :width].astype(np.int64)).to(device)
    lb = torch.from_numpy(L[b].astype(np.int64)).to(device)
    yb = torch.from_numpy(y[b].astype(np.int64)).to(device)
    return xb, lb, yb


@torch.no_grad()
def predict(model, X, L, y, bs, device, min_len):
    """返回 (正面概率数组, 平均交叉熵)。顺序与输入样本一致。"""
    model.eval()
    probs, loss_sum = np.zeros(len(y), dtype=np.float64), 0.0
    for b in make_batches(L, bs * 2, None, False):
        xb, lb, yb = to_batch(X, L, y, b, min_len, device)
        logits = model(xb, lb)
        loss_sum += F.cross_entropy(logits, yb, reduction="sum").item()
        probs[b] = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    return probs, loss_sum / len(y)


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --------------------------------------------------------------------------- #
# 单次运行
# --------------------------------------------------------------------------- #
def run_one(cfg, out_dir=None, data=None, verbose=True):
    cfg = {**DEFAULTS, **cfg}
    set_seed(cfg["seed"])
    rng = np.random.RandomState(cfg["seed"])
    if cfg["num_threads"]:
        torch.set_num_threads(int(cfg["num_threads"]))
    device = pick_device(cfg["device"])
    kernels = [int(k) for k in str(cfg["kernels"]).split(",")]
    min_len = max(kernels)
    no_val = bool(cfg["merge_val"])          # 完整训练池模式：没有验证集，固定训练 epochs 轮

    if data is None:
        data = load_data(cfg["data_dir"], cfg["max_len"], cfg["subset"], cfg["merge_val"], cfg["max_train"], cfg["max_val"])
    Xtr, Ltr, ytr = data["x_train"], data["len_train"], data["y_train"]
    Xva, Lva, yva = data["x_val"], data["len_val"], data["y_val"]
    if Xtr.shape[1] < min_len:
        raise ValueError(f"序列长度 {Xtr.shape[1]} 小于最大卷积核 {min_len}")

    pretrained, glove_cov, emb_dim = None, None, int(cfg["emb_dim"])
    if cfg["glove"]:
        pretrained, glove_cov = load_glove(cfg["glove"], data["stoi"], cache_dir=cfg["data_dir"], seed=cfg["seed"])
        emb_dim = pretrained.shape[1]
        if verbose:
            print(f"[GloVe] 维度 {emb_dim}，词表覆盖率 {glove_cov:.1%}")

    model = TextCNN(data["vocab_size"], emb_dim, kernels, int(cfg["num_filters"]), cfg["dropout"],
                    pretrained, cfg["freeze_emb"]).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    n_params = sum(p.numel() for p in model.parameters())
    n_train_params = sum(p.numel() for p in params)
    if verbose:
        print(f"[模型] seed={cfg['seed']} 设备={device} 参数量={n_params:,}（可训练 {n_train_params:,}） "
              f"训练样本={len(ytr)} 核宽={kernels} dropout={cfg['dropout']}")

    best = dict(acc=-1.0, loss=float("inf"), epoch=0, state=None)
    history, bad, t_start = [], 0, time.time()
    for ep in range(1, int(cfg["epochs"]) + 1):
        model.train()
        t0, tl, tc, n = time.time(), 0.0, 0, 0
        for b in make_batches(Ltr, int(cfg["batch_size"]), rng, True):
            xb, lb, yb = to_batch(Xtr, Ltr, ytr, b, min_len, device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb, lb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            tl += loss.item() * len(b)
            tc += (logits.argmax(1) == yb).sum().item()
            n += len(b)
        rec = dict(epoch=ep, train_loss=tl / n, train_acc=tc / n, epoch_sec=time.time() - t0)
        if no_val:
            best.update(epoch=ep, state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            if verbose:
                print(f"  epoch {ep:2d} | train loss {rec['train_loss']:.4f} acc {rec['train_acc']:.4f} | {rec['epoch_sec']:.0f}s")
        else:
            pv, vloss = predict(model, Xva, Lva, yva, int(cfg["batch_size"]), device, min_len)
            m = binary_metrics(yva, pv)
            rec.update(val_loss=vloss, val_acc=m["accuracy"], val_macro_f1=m["macro_f1"], val_auc=m["auc"])
            better = (m["accuracy"] > best["acc"] + 1e-9) or \
                     (abs(m["accuracy"] - best["acc"]) <= 1e-9 and vloss < best["loss"])
            if better:
                best.update(acc=m["accuracy"], loss=vloss, epoch=ep,
                            state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
                bad = 0
            else:
                bad += 1
            if verbose:
                print(f"  epoch {ep:2d} | train loss {rec['train_loss']:.4f} acc {rec['train_acc']:.4f} | "
                      f"val loss {vloss:.4f} acc {m['accuracy']:.4f} f1 {m['macro_f1']:.4f} | "
                      f"{rec['epoch_sec']:.0f}s{' *' if better else ''}")
        history.append(rec)
        if not no_val and bad >= int(cfg["patience"]):
            if verbose:
                print(f"  早停：验证集准确率连续 {bad} 轮没有提升，最佳轮次 = {best['epoch']}")
            break
    train_time = time.time() - t_start

    model.load_state_dict(best["state"])
    result = dict(config=cfg, device=str(device), n_params=n_params, n_trainable_params=n_train_params,
                  best_epoch=best["epoch"], epochs_run=len(history), train_time_sec=train_time,
                  sec_per_epoch=float(np.mean([h["epoch_sec"] for h in history])),
                  glove_coverage=glove_cov, history=history, val=None, test=None, manual=None)
    probs = {}
    if not no_val:
        probs["val"], _ = predict(model, Xva, Lva, yva, int(cfg["batch_size"]), device, min_len)
        result["val"] = binary_metrics(yva, probs["val"])
    if cfg["eval_test"]:
        t0 = time.time()
        probs["test"], _ = predict(model, data["x_test"], data["len_test"], data["y_test"],
                                   int(cfg["batch_size"]), device, min_len)
        result["infer_sec_per_1k"] = (time.time() - t0) / len(data["y_test"]) * 1000
        result["test"] = binary_metrics(data["y_test"], probs["test"])
        if verbose:
            print(f"  [测试集] acc {result['test']['accuracy']:.4f} macro-F1 {result['test']['macro_f1']:.4f} "
                  f"AUC {result['test']['auc']:.4f}  （测试集只用于最终报告，请勿据此再调参）")
        if "x_manual" in data and len(data["y_manual"]):          # 人工标注测试集（已去掉中性）
            probs["manual"], _ = predict(model, data["x_manual"], data["len_manual"], data["y_manual"],
                                         int(cfg["batch_size"]), device, min_len)
            result["manual"] = binary_metrics(data["y_manual"], probs["manual"])
            if verbose:
                print(f"  [人工测试集 n={len(data['y_manual'])}] acc {result['manual']['accuracy']:.4f} "
                      f"macro-F1 {result['manual']['macro_f1']:.4f}")
    if verbose and result["val"]:
        v = result["val"]
        print(f"  [验证集·最佳轮次 {best['epoch']}] acc {v['accuracy']:.4f} macro-F1 {v['macro_f1']:.4f} AUC {v['auc']:.4f}")

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(to_jsonable(result), f, ensure_ascii=False, indent=2)
        for k, p in probs.items():
            np.save(os.path.join(out_dir, f"{k}_probs.npy"), p)
        if cfg["save_model"]:
            torch.save(dict(state_dict=best["state"], config=cfg, vocab_size=data["vocab_size"],
                            emb_dim=emb_dim, kernels=kernels, num_filters=int(cfg["num_filters"])),
                       os.path.join(out_dir, "model.pt"))
    return result


# --------------------------------------------------------------------------- #
def summarize(results):
    """多个种子的结果取 均值 ± 标准差。"""
    out = {"n_runs": len(results)}
    for split in ("val", "test", "manual"):
        if all(r.get(split) for r in results):
            out[split] = {}
            for k in ("accuracy", "macro_f1", "auc", "precision_pos", "recall_pos", "log_loss"):
                m, s = mean_std([r[split][k] for r in results])
                out[split][k] = dict(mean=m, std=s)
    out["best_epoch"] = [r["best_epoch"] for r in results]
    out["n_params"] = results[0]["n_params"]
    out["sec_per_epoch"] = float(np.mean([r["sec_per_epoch"] for r in results]))
    return out


def main():
    ap = argparse.ArgumentParser(description="TextCNN 训练", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    D = DEFAULTS
    ap.add_argument("--data_dir", default=D["data_dir"])
    ap.add_argument("--out", default="runs/textcnn")
    ap.add_argument("--max_len", type=int, default=D["max_len"])
    ap.add_argument("--emb_dim", type=int, default=D["emb_dim"], help="随机初始化时的词向量维度；使用 --glove 时由文件决定")
    ap.add_argument("--kernels", default=D["kernels"], help="卷积核宽度，逗号分隔")
    ap.add_argument("--num_filters", type=int, default=D["num_filters"])
    ap.add_argument("--dropout", type=float, default=D["dropout"])
    ap.add_argument("--lr", type=float, default=D["lr"])
    ap.add_argument("--weight_decay", type=float, default=D["weight_decay"])
    ap.add_argument("--batch_size", type=int, default=D["batch_size"])
    ap.add_argument("--epochs", type=int, default=D["epochs"])
    ap.add_argument("--patience", type=int, default=D["patience"])
    ap.add_argument("--glove", default=None, help="GloVe 文本文件路径，如 glove.6B.100d.txt")
    ap.add_argument("--freeze_emb", action="store_true", help="冻结词向量")
    ap.add_argument("--eval_test", action="store_true", help="训练结束后在测试集评估（只在最终结果时使用）")
    ap.add_argument("--save_model", action="store_true")
    ap.add_argument("--merge_val", action="store_true", help="把验证集并入训练集，固定训练 --epochs 轮（需配合 --eval_test）")
    ap.add_argument("--subset", type=int, default=None, help="冒烟测试：训练取前 N 条，验证/测试取前 N/5 条")
    ap.add_argument("--max_train", type=int, default=None, help="训练集最多取前 N 条（消融实验提速）")
    ap.add_argument("--max_val", type=int, default=None, help="验证集最多取前 N 条（测试集不截取）")
    ap.add_argument("--device", default="auto", help="auto / cpu / cuda / mps")
    ap.add_argument("--num_threads", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0, help="起始随机种子")
    ap.add_argument("--seeds", type=int, default=1, help="运行几个种子（seed, seed+1, ...）")
    args = ap.parse_args()
    cfg = {k: v for k, v in vars(args).items() if k not in ("out", "seeds")}
    if cfg["merge_val"] and not cfg["eval_test"]:
        ap.error("--merge_val 没有验证集，必须同时使用 --eval_test")

    data = load_data(cfg["data_dir"], cfg["max_len"], cfg["subset"], cfg["merge_val"], cfg["max_train"], cfg["max_val"])
    results = []
    for i in range(args.seeds):
        c = dict(cfg, seed=cfg["seed"] + i)
        results.append(run_one(c, os.path.join(args.out, f"seed{c['seed']}"), data=data))
    summ = summarize(results)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(to_jsonable(summ), f, ensure_ascii=False, indent=2)
    print(f"\n===== {args.seeds} 个种子汇总（均值 ± 标准差）=====")
    for split in ("val", "test", "manual"):
        if split in summ:
            print(" ", split, " ".join(f"{k}={v['mean']:.4f}±{v['std']:.4f}" for k, v in summ[split].items()
                                       if k in ("accuracy", "macro_f1", "auc")))
    print(f"  参数量 {summ['n_params']:,} | 每轮 {summ['sec_per_epoch']:.0f}s | 最佳轮次 {summ['best_epoch']}")
    print(f"  结果已保存到 {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
