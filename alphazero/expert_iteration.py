"""Expert-iteration loop (self-play generation + Stockfish-anchored gating).

WHY THIS DESIGN (see PLAN.md "Iteration 7"): a first naive AlphaZero self-play
loop *regressed* the net badly (v5 ~2104 -> ~1626 Elo vs Stockfish) because
(a) bot-vs-bot gating at low sims is blind to real strength, and (b) pure
game-outcome value targets destroyed v5's tuned value head. This loop fixes both:

  1. self-play GENERATES on-policy positions (the net plays itself with MCTS),
  2. STOCKFISH LABELS every position (soft policy + value) -> high-quality
     teacher targets on the distribution the bot actually reaches (DAgger /
     expert iteration), trained by the existing distillation trainer warm-started
     from the current best (so the value head is preserved, not overwritten),
  3. every candidate is GATED vs Stockfish@Elo at realistic sims and only
     promoted if it scores >= the incumbent (v5 is the permanent floor) -> the
     shipped net can never be weaker than v5, and only genuine gains are kept.

The attention architecture is unchanged, so chess-teacher's saliency is intact.
Resumable via <workdir>/state.json.

  python expert_iteration.py --workdir ei --warm-start checkpoints_v5/best.pt \
      --iterations 6 --games 120 --sims 100 --workers 10 --sf-depth 12 \
      --epochs 4 --window 3 --base-data data_v5 \
      --gate-games 40 --gate-sims 800 --sf-elo 2000
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time

PY = sys.executable


def sh(cmd, log):
    line = "[ei] $ " + " ".join(str(c) for c in cmd)
    print(line, flush=True)
    log.write(line + "\n"); log.flush()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out = []
    for ln in proc.stdout:
        sys.stdout.write(ln); sys.stdout.flush()
        log.write(ln); log.flush()
        out.append(ln)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(map(str, cmd))}")
    return "".join(out)


def prune_buffer(buffer_dir, keep_from_iter, log):
    removed = 0
    for f in glob.glob(os.path.join(buffer_dir, "shard_iter*.npz")):
        m = re.search(r"shard_iter(\d+)_", os.path.basename(f))
        if m and int(m.group(1)) < keep_from_iter:
            os.remove(f); removed += 1
    if removed:
        msg = f"[ei] pruned {removed} old on-policy shards (< iter {keep_from_iter})"
        print(msg, flush=True); log.write(msg + "\n")


def load_state(workdir):
    p = os.path.join(workdir, "state.json")
    if os.path.exists(p):
        with open(p) as fh:
            return json.load(fh)
    return {"iteration": 0, "best": os.path.join(workdir, "best.pt"),
            "incumbent_score": None, "incumbent_elo": None, "history": []}


def save_state(workdir, state):
    with open(os.path.join(workdir, "state.json"), "w") as fh:
        json.dump(state, fh, indent=2)


def parse_score(output):
    m = re.findall(r"score=([0-9.]+)", output)
    return float(m[-1]) if m else None


def parse_elo(output):
    m = re.findall(r"estimated bot Elo ~=\s*(-?[0-9]+)", output)
    return int(m[-1]) if m else None


def eval_vs_sf(ckpt, args, log, seed):
    out = sh([PY, "evaluate.py", "--ckpt", ckpt, "--games", str(args.gate_games),
              "--sims", str(args.gate_sims), "--batch-size", str(args.gate_batch),
              "--c-puct", str(args.c_puct), "--sf-elo", str(args.sf_elo),
              "--seed", str(seed)], log)
    return parse_score(out), parse_elo(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default="ei")
    ap.add_argument("--warm-start", default="checkpoints_v5/best.pt")
    ap.add_argument("--iterations", type=int, default=6)
    ap.add_argument("--base-data", default="data_v5", help="base distillation corpus (anchors quality)")
    # on-policy self-play generation (Stockfish-labelled)
    ap.add_argument("--games", type=int, default=120)
    ap.add_argument("--sims", type=int, default=100)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--sp-device", default="cpu")
    ap.add_argument("--sf-depth", type=int, default=12)
    ap.add_argument("--sf-multipv", type=int, default=5, help="must match base-data pol_idx width")
    ap.add_argument("--temp-moves", type=int, default=20)
    ap.add_argument("--dirichlet-eps", type=float, default=0.25)
    ap.add_argument("--book-prob", type=float, default=0.6)
    ap.add_argument("--window", type=int, default=3)
    ap.add_argument("--onpolicy-oversample", type=int, default=2,
                    help="repeat the on-policy buffer N times in the training mix so it "
                         "carries real weight against the (larger) base corpus")
    # training
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=768)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--value-weight", type=float, default=1.5)
    ap.add_argument("--value-scale", type=float, default=500.0)
    # gating vs Stockfish
    ap.add_argument("--gate-games", type=int, default=40)
    ap.add_argument("--gate-sims", type=int, default=800)
    ap.add_argument("--gate-batch", type=int, default=4)
    ap.add_argument("--c-puct", type=float, default=1.5)
    ap.add_argument("--sf-elo", type=int, default=2000)
    ap.add_argument("--gate-seed", type=int, default=777)
    ap.add_argument("--promote-margin", type=float, default=0.03,
                    help="candidate must beat incumbent SF score by this margin to promote")
    args = ap.parse_args()

    os.makedirs(args.workdir, exist_ok=True)
    buffer_dir = os.path.join(args.workdir, "buffer")
    os.makedirs(buffer_dir, exist_ok=True)
    best_path = os.path.join(args.workdir, "best.pt")
    log = open(os.path.join(args.workdir, "ei.log"), "a")

    state = load_state(args.workdir)
    if not os.path.exists(best_path):
        shutil.copyfile(args.warm_start, best_path)
        print(f"[ei] initialised best.pt from {args.warm_start}", flush=True)
        log.write(f"[ei] init best from {args.warm_start}\n")

    # Establish the incumbent's Stockfish score once (the floor to beat).
    if state.get("incumbent_score") is None:
        print("[ei] measuring incumbent (v5) vs Stockfish to set the promotion floor...", flush=True)
        sc, elo = eval_vs_sf(best_path, args, log, args.gate_seed)
        state["incumbent_score"] = sc
        state["incumbent_elo"] = elo
        save_state(args.workdir, state)
        msg = f"[ei] incumbent floor: score={sc} (~{elo} Elo) vs SF@{args.sf_elo}"
        print(msg, flush=True); log.write(msg + "\n")

    start_iter = state["iteration"] + 1
    end_iter = state["iteration"] + args.iterations
    print(f"[ei] running iterations {start_iter}..{end_iter} (workdir={args.workdir})", flush=True)

    for it in range(start_iter, end_iter + 1):
        t0 = time.time()
        print(f"\n[ei] ===== ITERATION {it} =====", flush=True)
        log.write(f"\n[ei] ===== ITERATION {it} ===== {time.ctime()}\n")

        # 1. on-policy self-play with Stockfish labels
        sh([PY, "selfplay.py", "--ckpt", best_path, "--games", str(args.games),
            "--sims", str(args.sims), "--workers", str(args.workers), "--device", args.sp_device,
            "--label-stockfish", "--sf-depth", str(args.sf_depth), "--sf-multipv", str(args.sf_multipv),
            "--temp-moves", str(args.temp_moves), "--dirichlet-eps", str(args.dirichlet_eps),
            "--book-prob", str(args.book_prob), "--out", buffer_dir,
            "--tag", f"iter{it:03d}_", "--seed", str(it)], log)
        prune_buffer(buffer_dir, keep_from_iter=it - args.window + 1, log=log)

        # 2. warm-started retrain on base corpus + on-policy shards
        cand_dir = os.path.join(args.workdir, f"cand_iter{it:03d}")
        data_dirs = ",".join([args.base_data] + [buffer_dir] * max(1, args.onpolicy_oversample))
        sh([PY, "train.py", "--data-dir", data_dirs,
            "--warm-start", best_path, "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size), "--lr", str(args.lr),
            "--value-weight", str(args.value_weight), "--value-scale", str(args.value_scale),
            "--out", cand_dir], log)
        cand_best = os.path.join(cand_dir, "best.pt")

        # 3. gate vs Stockfish (paired seed) -> promote only if it beats the floor
        cand_score, cand_elo = eval_vs_sf(cand_best, args, log, args.gate_seed)
        incumbent = state["incumbent_score"] or 0.0
        promoted = cand_score is not None and cand_score >= incumbent + args.promote_margin
        if promoted:
            shutil.copyfile(cand_best, best_path)
            state["incumbent_score"] = cand_score
            state["incumbent_elo"] = cand_elo
            msg = (f"[ei] iter {it}: PROMOTED (cand score={cand_score} ~{cand_elo} Elo "
                   f">= floor {incumbent}+{args.promote_margin})")
        else:
            msg = (f"[ei] iter {it}: kept incumbent (cand score={cand_score} ~{cand_elo} Elo "
                   f"< floor {incumbent}+{args.promote_margin})")
        print(msg, flush=True); log.write(msg + "\n")

        state["iteration"] = it
        state["history"].append({"iter": it, "promoted": promoted, "cand_score": cand_score,
                                 "cand_elo": cand_elo, "incumbent_score": state["incumbent_score"],
                                 "secs": round(time.time() - t0)})
        save_state(args.workdir, state)
        # candidate dirs can be large; keep only the promoted one implicitly (best.pt copied)
        if not promoted:
            shutil.rmtree(cand_dir, ignore_errors=True)
        print(f"[ei] iter {it} done in {time.time()-t0:.0f}s "
              f"(best ~{state['incumbent_elo']} Elo)", flush=True)
        log.flush()

    print(f"[ei] finished. best={best_path} (~{state['incumbent_elo']} Elo vs SF@{args.sf_elo})", flush=True)
    log.close()


if __name__ == "__main__":
    main()
