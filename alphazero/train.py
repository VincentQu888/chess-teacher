"""Supervised training: distil Stockfish into the attention net.

Loss = soft-target policy cross-entropy (over the MultiPV top moves) + value MSE.
Runs on MPS (Apple GPU) if available. Checkpoints go to ``--out``.

  python train.py --data-dir data --epochs 20 --batch-size 512 --out checkpoints
"""

from __future__ import annotations

import argparse
import glob
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from model import AttentionChessNet, ModelConfig, count_params


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class ShardDataset:
    """Loads all .npz shards fully into RAM (dataset is small)."""

    def __init__(self, data_dir: str, limit_shards: int = 0, value_scale: float = 0.0):
        # data_dir may be a comma-separated list of directories (e.g. base corpus +
        # on-policy self-play shards). All must share the pol_idx width (multipv).
        files = []
        for d in str(data_dir).split(","):
            d = d.strip()
            if d:
                files.extend(sorted(glob.glob(os.path.join(d, "shard_*.npz"))))
        if limit_shards:
            files = files[:limit_shards]
        if not files:
            raise FileNotFoundError(f"no shards in {data_dir}")
        pids, gs, pidx, pprob, val, cps = [], [], [], [], [], []
        have_cp = True
        for f in files:
            d = np.load(f)
            pids.append(d["piece_ids"])
            gs.append(d["globals"])
            pidx.append(d["pol_idx"])
            pprob.append(d["pol_prob"])
            val.append(d["value"])
            if "best_cp" in d.files:
                cps.append(d["best_cp"])
            else:
                have_cp = False
        self.piece_ids = torch.from_numpy(np.concatenate(pids)).long()
        self.globals = torch.from_numpy(np.concatenate(gs)).float()
        self.pol_idx = torch.from_numpy(np.concatenate(pidx)).long()
        self.pol_prob = torch.from_numpy(np.concatenate(pprob)).float()
        self.value = torch.from_numpy(np.concatenate(val)).float()
        self.cps = torch.from_numpy(np.concatenate(cps)).float() if have_cp else None
        self.wdl_target = None  # (N,3) filled by build_wdl_targets() when --wdl
        # Optional: recompute a less-saturating value target from raw centipawns.
        if value_scale > 0 and have_cp:
            self.value = torch.tanh(self.cps / value_scale)
            print(f"[data] value target recomputed as tanh(cp/{value_scale:.0f})")
        elif value_scale > 0 and not have_cp:
            print("[data] --value-scale set but shards lack best_cp; using stored value")
        # renormalise probs over valid (idx>=0) entries
        mask = (self.pol_idx >= 0).float()
        self.pol_prob = self.pol_prob * mask
        self.pol_prob = self.pol_prob / self.pol_prob.sum(dim=1, keepdim=True).clamp_min(1e-8)
        self.pol_idx = self.pol_idx.clamp_min(0)  # -1 -> 0, prob is 0 there
        self.n_files = len(files)

    def build_wdl_targets(self, scale: float = 350.0, draw0: float = 0.35):
        """Soft win/draw/loss targets from centipawns (side-to-move POV)."""
        if self.cps is None:
            # fall back to stored scalar value in (-1,1)
            v = self.value
        else:
            v = torch.tanh(self.cps / scale)
        e = (v + 1.0) / 2.0                       # expected score in (0,1)
        pd = draw0 * (1.0 - v.abs())              # draw prob peaks at v=0
        pw = (e - pd / 2).clamp_min(1e-6)
        pl = (1.0 - e - pd / 2).clamp_min(1e-6)
        w = torch.stack([pw, pd.clamp_min(1e-6), pl], dim=1)
        self.wdl_target = w / w.sum(dim=1, keepdim=True)
        print(f"[data] built WDL targets from cp (scale={scale}, draw0={draw0})")

    def __len__(self):
        return self.piece_ids.shape[0]


def wdl_ce(wdl_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Soft cross-entropy for the win/draw/loss value head."""
    return -(target * F.log_softmax(wdl_logits, dim=1)).sum(dim=1).mean()


def policy_ce(logits: torch.Tensor, pol_idx: torch.Tensor, pol_prob: torch.Tensor) -> torch.Tensor:
    """Soft-target cross-entropy over the stored top-K move indices."""
    logp = F.log_softmax(logits, dim=1)                 # (B, 4672)
    gathered = torch.gather(logp, 1, pol_idx)           # (B, K)
    return -(pol_prob * gathered).sum(dim=1).mean()


@torch.no_grad()
def topk_best_accuracy(logits: torch.Tensor, pol_idx: torch.Tensor) -> float:
    """Fraction where the model ranks Stockfish's best move (pol_idx[:,0])
    highest among the stored candidate moves."""
    cand_logits = torch.gather(logits, 1, pol_idx)      # (B, K)
    pred = cand_logits.argmax(dim=1)                    # index within candidates
    return (pred == 0).float().mean().item()


def run_epoch(net, ds, indices, device, batch_size, opt=None, value_weight=1.0):
    """Vectorised batching directly over in-RAM tensors (no DataLoader)."""
    train = opt is not None
    net.train(train)
    tot_p = tot_v = tot_acc = 0.0
    n = 0
    if train:
        indices = indices[torch.randperm(indices.shape[0])]
    for start in range(0, indices.shape[0], batch_size):
        idx = indices[start : start + batch_size]
        piece_ids = ds.piece_ids[idx].to(device, non_blocking=True)
        g = ds.globals[idx].to(device, non_blocking=True)
        pidx = ds.pol_idx[idx].to(device, non_blocking=True)
        pprob = ds.pol_prob[idx].to(device, non_blocking=True)
        use_wdl = getattr(net, "wdl", False) and ds.wdl_target is not None
        with torch.set_grad_enabled(train):
            if use_wdl:
                logits, value, _, wdl_logits = net(piece_ids, g, return_wdl=True)
                wt = ds.wdl_target[idx].to(device, non_blocking=True)
                v_loss = wdl_ce(wdl_logits, wt)
            else:
                val = ds.value[idx].to(device, non_blocking=True)
                logits, value, _ = net(piece_ids, g)
                v_loss = F.mse_loss(value, val)
            p_loss = policy_ce(logits, pidx, pprob)
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
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--value-weight", type=float, default=1.0)
    ap.add_argument("--val-frac", type=float, default=0.03)
    ap.add_argument("--limit-shards", type=int, default=0)
    ap.add_argument("--value-scale", type=float, default=0.0,
                    help="if >0 and shards have best_cp, value target = tanh(cp/scale)")
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--resume", default="")
    ap.add_argument("--warm-start", default="",
                    help="load model weights from this checkpoint (fresh epochs); "
                         "architecture is taken from the checkpoint's cfg")
    ap.add_argument("--wdl", action="store_true",
                    help="use a win/draw/loss value head (CE loss from cp targets); "
                         "warm-start loads the trunk/policy and reinitialises the value head")
    ap.add_argument("--wdl-scale", type=float, default=350.0)
    ap.add_argument("--wdl-draw0", type=float, default=0.35)
    args = ap.parse_args()

    device = pick_device()
    print(f"[train] device={device}")

    ds = ShardDataset(args.data_dir, args.limit_shards, value_scale=args.value_scale)
    print(f"[train] loaded {len(ds)} positions from {ds.n_files} shards")
    if args.wdl:
        ds.build_wdl_targets(scale=args.wdl_scale, draw0=args.wdl_draw0)
    n_val = max(1, int(len(ds) * args.val_frac))
    perm = torch.randperm(len(ds), generator=torch.Generator().manual_seed(0))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    if args.warm_start and os.path.exists(args.warm_start):
        wck = torch.load(args.warm_start, map_location=device)
        cfg = ModelConfig(**wck["cfg"]) if "cfg" in wck else ModelConfig(
            d_model=args.d_model, n_layers=args.layers, n_heads=args.heads)
    else:
        cfg = ModelConfig(d_model=args.d_model, n_layers=args.layers, n_heads=args.heads)
    if args.wdl:
        cfg.wdl = True  # request a WDL value head regardless of the warm-start's head
    net = AttentionChessNet(cfg).to(device)
    print(f"[train] model params: {count_params(net):,} (wdl={cfg.wdl})")
    start_epoch = 0
    if args.warm_start and os.path.exists(args.warm_start):
        wck = torch.load(args.warm_start, map_location=device)
        sd = wck["model"]
        # drop value-head weights when the head shape changed (e.g. scalar -> WDL)
        model_sd = net.state_dict()
        filtered = {k: v for k, v in sd.items()
                    if k in model_sd and model_sd[k].shape == v.shape}
        dropped = [k for k in sd if k not in filtered]
        net.load_state_dict(filtered, strict=False)
        print(f"[train] warm-started {len(filtered)}/{len(sd)} tensors from {args.warm_start} "
              f"(reinit: {sorted(set(k.split('.')[0] for k in dropped)) or 'none'}; cfg {vars(cfg)})")
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location=device)
        net.load_state_dict(ck["model"])
        start_epoch = ck.get("epoch", 0)
        print(f"[train] resumed from {args.resume} @ epoch {start_epoch}")

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    os.makedirs(args.out, exist_ok=True)
    best_val = float("inf")
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        tp, tv, ta = run_epoch(net, ds, train_idx, device, args.batch_size, opt, args.value_weight)
        vp, vv, va = run_epoch(net, ds, val_idx, device, args.batch_size, None, args.value_weight)
        sched.step()
        dt = time.time() - t0
        print(f"[train] epoch {epoch+1}/{args.epochs} ({dt:.0f}s) "
              f"train p={tp:.3f} v={tv:.3f} acc={ta:.3f} | "
              f"val p={vp:.3f} v={vv:.3f} acc={va:.3f} | lr={sched.get_last_lr()[0]:.2e}",
              flush=True)
        ck = {"model": net.state_dict(), "cfg": vars(cfg), "epoch": epoch + 1,
              "val_policy": vp, "val_value": vv, "val_acc": va}
        torch.save(ck, os.path.join(args.out, "latest.pt"))
        if vp + vv < best_val:
            best_val = vp + vv
            torch.save(ck, os.path.join(args.out, "best.pt"))
            print(f"[train]   new best (val {best_val:.3f}) -> best.pt")
    print("[train] DONE")


if __name__ == "__main__":
    main()
