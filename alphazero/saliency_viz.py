"""Render the attention-weighted board state as a board heatmap.

Produces a PNG with two panels:
  * Value saliency  - how much the value (CLS) token attends to each square
    (which squares drive the model's assessment of the position).
  * Move saliency   - how much the chosen move's from/to square tokens attend to
    each square (which squares justify that move).

Usage:
  python saliency_viz.py --fen "<FEN>" [--move e2e4] [--ckpt checkpoints_v5/best.pt] \
      --out saliency.png
"""

from __future__ import annotations

import argparse

import chess
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from encoding import encode_position, legal_moves_with_indices
from model import CLS_TOKEN, NUM_SQUARES
from attention_explain import _agg_attention
from evaluate import load_net

GLYPH = {
    "P": "\u2659", "N": "\u2658", "B": "\u2657", "R": "\u2656", "Q": "\u2655", "K": "\u2654",
    "p": "\u265F", "n": "\u265E", "b": "\u265D", "r": "\u265C", "q": "\u265B", "k": "\u265A",
}
LIGHT = "#EAE9D2"
DARK = "#7FA160"


@torch.no_grad()
def compute_saliency(net, board, device, move=None):
    net.eval()
    piece_ids, g, mirrored = encode_position(board)
    pid = torch.from_numpy(piece_ids).unsqueeze(0).to(device)
    gt = torch.from_numpy(g).unsqueeze(0).to(device)
    logits, value, attns = net(pid, gt, return_attn=True)
    attn = _agg_attention(attns)  # (66, 66) mean over layers

    moves, idxs = legal_moves_with_indices(board, mirrored)
    logits = logits[0].detach().float().cpu().numpy()
    sel = logits[idxs] - logits[idxs].max()
    pr = np.exp(sel)
    pr /= pr.sum()
    order = np.argsort(-pr)
    if move is None:
        move = moves[int(order[0])]
    top = [(moves[i].uci(), float(pr[i])) for i in order[:5]]

    from_c = chess.square_mirror(move.from_square) if mirrored else move.from_square
    to_c = chess.square_mirror(move.to_square) if mirrored else move.to_square
    value_sal_c = attn[CLS_TOKEN, :NUM_SQUARES]
    move_sal_c = (attn[from_c, :NUM_SQUARES] + attn[to_c, :NUM_SQUARES]) / 2.0

    def to_real(arr):
        real = np.zeros(64, dtype=float)
        for cs in range(64):
            rs = chess.square_mirror(cs) if mirrored else cs
            real[rs] = float(arr[cs])
        return real

    return to_real(value_sal_c), to_real(move_sal_c), float(value.item()), move, top


def _hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


def draw_panel(ax, board, sal, title, highlight=()):
    base_light = np.array(_hex_to_rgb(LIGHT))
    base_dark = np.array(_hex_to_rgb(DARK))
    hot = np.array(_hex_to_rgb("#E23B2E"))  # saliency color
    mx = sal.max() if sal.max() > 0 else 1.0

    for sq in range(64):
        f = chess.square_file(sq)
        r = chess.square_rank(sq)
        x = f
        y = r  # rank 1 at bottom
        base = base_light if (f + r) % 2 == 1 else base_dark
        w = sal[sq] / mx
        color = tuple(base * (1 - 0.85 * w) + hot * (0.85 * w))
        ax.add_patch(Rectangle((x, y), 1, 1, facecolor=color, edgecolor="none"))
        if sq in highlight:
            ax.add_patch(Rectangle((x + 0.03, y + 0.03), 0.94, 0.94, fill=False,
                                   edgecolor="#1b4fd8", linewidth=2.5))
        piece = board.piece_at(sq)
        if piece:
            ax.text(x + 0.5, y + 0.5, GLYPH[piece.symbol()], ha="center", va="center",
                    fontsize=26,
                    color="#111" if piece.color == chess.BLACK else "#fafafa",
                    zorder=3)
    # coordinates
    for f in range(8):
        ax.text(f + 0.5, -0.18, chess.FILE_NAMES[f], ha="center", va="center", fontsize=9, color="#333")
    for r in range(8):
        ax.text(-0.18, r + 0.5, str(r + 1), ha="center", va="center", fontsize=9, color="#333")
    ax.set_xlim(-0.4, 8)
    ax.set_ylim(-0.4, 8)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fen", required=True)
    ap.add_argument("--move", default="")
    ap.add_argument("--ckpt", default="checkpoints_v5/best.pt")
    ap.add_argument("--out", default="saliency.png")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    net = load_net(args.ckpt, device)
    board = chess.Board(args.fen)
    move = None
    if args.move:
        mv = chess.Move.from_uci(args.move)
        if mv in board.legal_moves:
            move = mv

    value_sal, move_sal, value, chosen, top = compute_saliency(net, board, device, move)

    fig, axes = plt.subplots(1, 2, figsize=(13, 7))
    side = "White" if board.turn == chess.WHITE else "Black"
    draw_panel(axes[0], board, value_sal,
               f"Value saliency  (net value {value:+.2f} for {side})")
    hl = {chosen.from_square, chosen.to_square}
    draw_panel(axes[1], board, move_sal,
               f"Move saliency for {chosen.uci()}  (policy top move)", highlight=hl)

    top_str = ", ".join(f"{u} {p:.0%}" for u, p in top)
    fig.suptitle(
        f"Attention-weighted board state (v5 net)\nFEN: {args.fen}\n"
        f"policy top moves: {top_str}",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
