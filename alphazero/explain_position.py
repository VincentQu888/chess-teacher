"""CLI bridge: given a FEN, emit the attention-weighted board state as JSON.

This is the integration point for the chess-teacher explainer. The explainer
backend (a different Python env, no torch) shells out to this script using the
alphazero venv and folds the JSON into its ground-truth block.

Output JSON:
{
  "value": float,                 # side-to-move value in (-1,1)
  "chosen_move": "e2e4",          # bot's top policy move (or --move)
  "top_moves": [{"uci","prob"}],  # policy distribution over legal moves
  "value_saliency": [{"square","piece","weight"}],  # squares driving the eval
  "move_saliency":  [{"square","piece","weight"}],  # squares justifying the move
  "prompt_block": "..."           # ready-to-drop text for the LLM
}

Usage:
  python explain_position.py --fen "<FEN>" [--move e2e4] [--ckpt path] [--topk 6]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import chess
import torch

from attention_explain import attention_report, format_report_for_prompt
from evaluate import load_net


def default_ckpt() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    # Prefer the self-play (expert-iteration) net if present; it keeps the same
    # attention architecture, so the attention-weighted saliency is unchanged but
    # comes from the strongest available net.
    for cand in ("ei/best.pt",
                 "checkpoints_v5/best.pt", "checkpoints_v4/best.pt",
                 "checkpoints_v3/best.pt", "checkpoints_v2/best.pt",
                 "checkpoints/best.pt", "checkpoints_v1/best.pt"):
        p = os.path.join(here, cand)
        if os.path.exists(p):
            return p
    raise FileNotFoundError("no trained checkpoint found")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fen", required=True)
    ap.add_argument("--move", default="", help="UCI move to explain (default: bot's top move)")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--topk", type=int, default=6)
    ap.add_argument("--device", default="cpu", help="cpu (default, fast for 1 position) or mps")
    args = ap.parse_args()

    try:
        board = chess.Board(args.fen)
    except ValueError as exc:
        print(json.dumps({"error": f"bad FEN: {exc}"}))
        return 1

    ckpt = args.ckpt or default_ckpt()
    device = torch.device(args.device)
    net = load_net(ckpt, device)

    move = None
    if args.move:
        try:
            mv = chess.Move.from_uci(args.move)
            if mv in board.legal_moves:
                move = mv
        except ValueError:
            pass

    report = attention_report(net, board, device, move=move, topk=args.topk)
    report["prompt_block"] = format_report_for_prompt(report)
    report["checkpoint"] = os.path.basename(os.path.dirname(ckpt)) + "/" + os.path.basename(ckpt)
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
