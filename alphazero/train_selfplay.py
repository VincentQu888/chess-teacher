"""Train the attention net on AlphaZero self-play targets.

Loss = policy cross-entropy against the MCTS visit distribution
     + value_weight * MSE(value, target)
where target = value_lambda * game_outcome_z + (1 - value_lambda) * root_q.

Warm-starts from an existing checkpoint (the distilled net or the previous
self-play iteration) so we do expert-iteration / AlphaZero fine-tuning rather
than an infeasible from-scratch laptop self-play run. A fraction of the original
Stockfish-distillation data can be retained (`--retain-dir/--retain-frac`) to
avoid catastrophic forgetting of basic tactics during early self-play.

  python train_selfplay.py --data-dir selfplay_data --warm-start checkpoints_v5/best.pt \
      --epochs 4 --batch-size 512 --out checkpoints_sp1 \
      --retain-dir data_v5 --retain-frac 0.5
"""

from __future__ import annotations

import argparse
import glob
import os
import time
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from model import AttentionChessNet, ModelConfig, count_params
from train import pick_device, policy_ce, topk_best_accuracy
from selfplay import MAX_PT


def _load_dir_padded(data_dir: str, limit_shards: int = 0):
    """Load .npz shards from a dir, padding policy targets to MAX_PT and
    synthesising q_value when absent (distillation shards)."""
    files = sorted(glob.glob(os.path.join(data_dir, "shard_*.npz")))
    if limit_shards:
        files = files[:limit_shards]
    if not files:
        raise FileNotFoundError(f"no shards in {data_dir}")
    pids, gs, pidx, pprob, val, qval = [], [], [], [], [], []
    for f in files:
        d = np.load(f)
        n = d["piece_ids"].shape[0]
        pids.append(d["piece_ids"].astype(np.int8))
        gs.append(d["globals"].astype(np.float32))
        k = d["pol_idx"].shape[1]
        pi = np.full((n, MAX_PT), -1, dtype=np.int64)
        pp = np.zeros((n, MAX_PT), dtype=np.float32)
        w = min(k, MAX_PT)
        pi[:, :w] = d["pol_idx"][:, :w]
        pp[:, :w] = d["pol_prob"][:, :w]
        pidx.append(pi)
        pprob.append(pp)
        v = d["value"].astype(np.float32)
        val.append(v)
        qval.append(d["q_value"].astype(np.float32) if "q_value" in d.files else v.copy())
    return (np.concatenate(pids), np.concatenate(gs), np.concatenate(pidx),
            np.concatenate(pprob), np.concatenate(val), np.concatenate(qval))


class SelfPlayDataset:
    def __init__(self, data_dir: str, value_lambda: float = 1.0,
                 retain_dir: str = "", retain_frac: float = 0.0,
                 limit_shards: int = 0, seed: int = 0):
        pids, gs, pidx, pprob, val, qval = _load_dir_padded(data_dir, limit_shards)
        n_sp = pids.shape[0]

        if retain_dir and retain_frac > 0:
            r = _load_dir_padded(retain_dir)
            n_take = min(r[0].shape[0], int(round(n_sp * retain_frac)))
            rng = np.random.default_rng(seed)
            sel = rng.choice(r[0].shape[0], size=n_take, replace=False)
            pids = np.concatenate([pids, r[0][sel]])
            gs = np.concatenate([gs, r[1][sel]])
            pidx = np.concatenate([pidx, r[2][sel]])
            pprob = np.concatenate([pprob, r[3][sel]])
            val = np.concatenate([val, r[4][sel]])
            qval = np.concatenate([qval, r[5][sel]])
            print(f"[data] self-play {n_sp} + retained distillation {n_take} "
                  f"= {pids.shape[0]} positions")
        else:
            print(f"[data] self-play {n_sp} positions")

        self.piece_ids = torch.from_numpy(pids).long()
        self.globals = torch.from_numpy(gs).float()
        self.pol_idx = torch.from_numpy(pidx).long()
        self.pol_prob = torch.from_numpy(pprob).float()
        z = torch.from_numpy(val).float()
        q = torch.from_numpy(qval).float()
        self.value = value_lambda * z + (1.0 - value_lambda) * q

        # renormalise probs over valid (idx>=0) entries
        mask = (self.pol_idx >= 0).float()
        self.pol_prob = self.pol_prob * mask
        self.pol_prob = self.pol_prob / self.pol_prob.sum(dim=1, keepdim=True).clamp_min(1e-8)
        self.pol_idx = self.pol_idx.clamp_min(0)

    def __len__(self):
        return self.piece_ids.shape[0]


def run_epoch(net, ds, indices, device, batch_size, opt=None, value_weight=1.0):
    train = opt is not None
    net.train(train)
    tot_p = tot_v = tot_acc = 0.0
    n = 0
    if train:
        indices = indices[torch.randperm(indices.shape[0])]
    for start in range(0, indices.shape[0], batch_size):
        idx = indices[start: start + batch_size]
        piece_ids = ds.piece_ids[idx].to(device)
        g = ds.globals[idx].to(device)
        pidx = ds.pol_idx[idx].to(device)
        pprob = ds.pol_prob[idx].to(device)
        val = ds.value[idx].to(device)
        with torch.set_grad_enabled(train):
            logits, value, _ = net(piece_ids, g)
            p_loss = policy_ce(logits, pidx, pprob)
            v_loss = F.mse_loss(value, val)
            loss = p_loss + value_weight * v_loss
        if train:
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)
            opt.step()
        bs = idx.shape[0]
        tot_p += p_loss.item() * bs
        tot_v += v_loss.item() * bs
        tot_acc += topk_best_accuracy(logits.detach(), pidx) * bs
        n += bs
    return tot_p / n, tot_v / n, tot_acc / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="selfplay_data")
    ap.add_argument("--warm-start", default="checkpoints_v5/best.pt")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--value-weight", type=float, default=1.0)
    ap.add_argument("--value-lambda", type=float, default=1.0,
                    help="value target = lambda*outcome + (1-lambda)*root_q")
    ap.add_argument("--val-frac", type=float, default=0.03)
    ap.add_argument("--retain-dir", default="")
    ap.add_argument("--retain-frac", type=float, default=0.0)
    ap.add_argument("--limit-shards", type=int, default=0)
    ap.add_argument("--out", default="checkpoints_sp")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = pick_device()
    print(f"[train-sp] device={device}")

    ds = SelfPlayDataset(args.data_dir, value_lambda=args.value_lambda,
                         retain_dir=args.retain_dir, retain_frac=args.retain_frac,
                         limit_shards=args.limit_shards, seed=args.seed)
    n_val = max(1, int(len(ds) * args.val_frac))
    perm = torch.randperm(len(ds), generator=torch.Generator().manual_seed(args.seed))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    # architecture comes from the warm-start checkpoint (keeps attention net)
    ck = torch.load(args.warm_start, map_location=device)
    cfg = ModelConfig(**ck["cfg"]) if "cfg" in ck else ModelConfig()
    net = AttentionChessNet(cfg).to(device)
    net.load_state_dict(ck["model"])
    print(f"[train-sp] warm-started from {args.warm_start} "
          f"(params {count_params(net):,}, cfg {vars(cfg)})")

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))

    os.makedirs(args.out, exist_ok=True)
    best_val = float("inf")
    for epoch in range(args.epochs):
        t0 = time.time()
        tp, tv, ta = run_epoch(net, ds, train_idx, device, args.batch_size, opt, args.value_weight)
        vp, vv, va = run_epoch(net, ds, val_idx, device, args.batch_size, None, args.value_weight)
        sched.step()
        dt = time.time() - t0
        print(f"[train-sp] epoch {epoch+1}/{args.epochs} ({dt:.0f}s) "
              f"train p={tp:.3f} v={tv:.3f} acc={ta:.3f} | "
              f"val p={vp:.3f} v={vv:.3f} acc={va:.3f} | lr={sched.get_last_lr()[0]:.2e}",
              flush=True)
        out_ck = {"model": net.state_dict(), "cfg": vars(cfg), "epoch": epoch + 1,
                  "val_policy": vp, "val_value": vv, "val_acc": va}
        torch.save(out_ck, os.path.join(args.out, "latest.pt"))
        if vp + vv < best_val:
            best_val = vp + vv
            torch.save(out_ck, os.path.join(args.out, "best.pt"))
            print(f"[train-sp]   new best (val {best_val:.3f}) -> best.pt")
    print("[train-sp] DONE")


if __name__ == "__main__":
    main()
