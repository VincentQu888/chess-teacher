"""Final verification for the self-play net.

Proves (or disproves) that self-play produced a stronger net than the distilled
baseline v5, at the strong play config:

  1. Head-to-head: ei/best.pt (self-play) vs checkpoints_v5/best.pt (distilled).
  2. Elo vs Stockfish@2000 for the self-play net at --sims/--batch (strong config).
  3. (optional) Same Stockfish match for v5 as an apples-to-apples baseline.

  python verify_selfplay.py --sp ei/best.pt --base checkpoints_v5/best.pt \
      --h2h-games 60 --h2h-sims 400 --sf-games 100 --sf-sims 3600 --sf-batch 4 \
      --sf-elo 2000 --also-base-sf
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

PY = sys.executable


def run(cmd):
    print("\n$ " + " ".join(str(c) for c in cmd), flush=True)
    out_lines = []
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        out_lines.append(line)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"failed: {' '.join(map(str, cmd))}")
    return "".join(out_lines)


def grab(pattern, text, cast=float, last=True):
    import re
    m = re.findall(pattern, text)
    if not m:
        return None
    return cast(m[-1] if last else m[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sp", default="ei/best.pt")
    ap.add_argument("--base", default="checkpoints_v5/best.pt")
    ap.add_argument("--h2h-games", type=int, default=60)
    ap.add_argument("--h2h-sims", type=int, default=400)
    ap.add_argument("--sf-games", type=int, default=100)
    ap.add_argument("--sf-sims", type=int, default=3600)
    ap.add_argument("--sf-batch", type=int, default=4)
    ap.add_argument("--sf-elo", type=int, default=2000)
    ap.add_argument("--also-base-sf", action="store_true",
                    help="also run the Stockfish match for the baseline v5")
    args = ap.parse_args()

    for p in (args.sp, args.base):
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    summary = {}

    # 1. head-to-head self-play vs baseline
    out = run([PY, "head_to_head.py", "--a", args.sp, "--b", args.base,
               "--games", str(args.h2h_games), "--sims", str(args.h2h_sims)])
    summary["h2h_sp_vs_base"] = grab(r"A_SCORE=([0-9.]+)", out)

    # 2. self-play net vs Stockfish
    out = run([PY, "evaluate.py", "--ckpt", args.sp, "--games", str(args.sf_games),
               "--sims", str(args.sf_sims), "--batch-size", str(args.sf_batch),
               "--sf-elo", str(args.sf_elo)])
    summary["sp_sf_score"] = grab(r"score=([0-9.]+)", out)
    summary["sp_sf_elo"] = grab(r"estimated bot Elo ~=\s*(-?[0-9]+)", out, int)

    # 3. optional baseline vs Stockfish
    if args.also_base_sf:
        out = run([PY, "evaluate.py", "--ckpt", args.base, "--games", str(args.sf_games),
                   "--sims", str(args.sf_sims), "--batch-size", str(args.sf_batch),
                   "--sf-elo", str(args.sf_elo)])
        summary["base_sf_score"] = grab(r"score=([0-9.]+)", out)
        summary["base_sf_elo"] = grab(r"estimated bot Elo ~=\s*(-?[0-9]+)", out, int)

    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    print(f"  head-to-head  self-play vs v5 baseline : {summary.get('h2h_sp_vs_base')}"
          f"  (>0.5 => self-play stronger)")
    print(f"  self-play net vs SF@{args.sf_elo}         : score={summary.get('sp_sf_score')}"
          f"  est Elo ~= {summary.get('sp_sf_elo')}")
    if args.also_base_sf:
        print(f"  baseline v5   vs SF@{args.sf_elo}         : score={summary.get('base_sf_score')}"
              f"  est Elo ~= {summary.get('base_sf_elo')}")
    gain = None
    if summary.get("sp_sf_elo") is not None and summary.get("base_sf_elo") is not None:
        gain = summary["sp_sf_elo"] - summary["base_sf_elo"]
        print(f"  Elo gain over v5 (same config)        : {gain:+d}")
    print("=" * 60)
    ok = (summary.get("h2h_sp_vs_base") or 0) > 0.5
    print("RESULT:", "self-play net is stronger than the distilled baseline"
          if ok else "no clear improvement yet")


if __name__ == "__main__":
    main()
