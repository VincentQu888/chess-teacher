"""Attention-weighted board state for the explainer.

Given a position (and optionally the move the bot chose), this runs the network
with attention capture and reports, in *real board* coordinates:

* value_saliency  -- how much the value (CLS) token attends to each square;
  i.e. which squares most influenced the bot's assessment of the position.
* move_saliency   -- how much the chosen move's from/to square tokens attend to
  the rest of the board; i.e. which squares justify that move.

This is the concrete realisation of the HEX-RL idea (arXiv:2112.08907): the
attention weights point at the state elements that drove the decision, giving
the LLM coach verified, model-grounded saliency instead of guesses. The output
is JSON-friendly and can be dropped into the existing ground-truth block.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import chess
import numpy as np
import torch

from encoding import encode_position, legal_moves_with_indices
from model import CLS_TOKEN, NUM_SQUARES


def _piece_name(board: chess.Board, sq: int) -> str:
    p = board.piece_at(sq)
    if p is None:
        return "empty"
    name = chess.piece_name(p.piece_type)
    color = "white" if p.color == chess.WHITE else "black"
    return f"{color} {name}"


def _canon_to_real(sq: int, mirrored: bool) -> int:
    return chess.square_mirror(sq) if mirrored else sq


def _agg_attention(attns: List[torch.Tensor]) -> np.ndarray:
    """Mean attention over layers -> (66, 66) numpy."""
    stack = torch.stack([a[0] for a in attns], dim=0)  # (L, 66, 66)
    return stack.mean(dim=0).detach().float().cpu().numpy()


def _top_squares(weights: np.ndarray, board: chess.Board, mirrored: bool, topk: int):
    """weights: (64,) over canonical squares -> list of dicts in real coords."""
    order = np.argsort(-weights)[:topk]
    out = []
    for canon_sq in order:
        real = _canon_to_real(int(canon_sq), mirrored)
        out.append({
            "square": chess.square_name(real),
            "piece": _piece_name(board, real),
            "weight": round(float(weights[canon_sq]), 4),
        })
    return out


@torch.no_grad()
def attention_report(
    net,
    board: chess.Board,
    device,
    move: Optional[chess.Move] = None,
    topk: int = 6,
) -> Dict:
    """Compute policy/value + attention saliency for ``board``.

    If ``move`` is None, the bot's top policy move is used for move_saliency.
    """
    net.eval()
    piece_ids, g, mirrored = encode_position(board)
    pid_t = torch.from_numpy(piece_ids).unsqueeze(0).to(device)
    g_t = torch.from_numpy(g).unsqueeze(0).to(device)
    logits, value, attns = net(pid_t, g_t, return_attn=True)
    logits = logits[0].detach().float().cpu().numpy()

    moves, idxs = legal_moves_with_indices(board, mirrored)
    if not moves:
        return {"value": float(value.item()), "moves": [], "note": "no legal moves"}
    sel = logits[idxs]
    sel = sel - sel.max()
    priors = np.exp(sel)
    priors = priors / priors.sum()
    order = np.argsort(-priors)
    top_moves = [
        {"uci": moves[i].uci(), "prob": round(float(priors[i]), 4)}
        for i in order[:topk]
    ]

    if move is None:
        move = moves[int(order[0])]
    # canonical from/to for the chosen move
    from_c = chess.square_mirror(move.from_square) if mirrored else move.from_square
    to_c = chess.square_mirror(move.to_square) if mirrored else move.to_square

    attn = _agg_attention(attns)  # (66, 66)
    value_sal = attn[CLS_TOKEN, :NUM_SQUARES].copy()
    move_sal = (attn[from_c, :NUM_SQUARES] + attn[to_c, :NUM_SQUARES]) / 2.0

    return {
        "value": round(float(value.item()), 4),
        "chosen_move": move.uci(),
        "top_moves": top_moves,
        "value_saliency": _top_squares(value_sal, board, mirrored, topk),
        "move_saliency": _top_squares(move_sal, board, mirrored, topk),
    }


def format_report_for_prompt(report: Dict) -> str:
    """Render an attention report as a compact ground-truth block for the LLM."""
    if not report.get("top_moves"):
        return "Attention model: no legal moves."
    lines = [
        "Attention model (neural bot) — attention-weighted board state:",
        f"- value (side to move): {report['value']:+.2f} (>0 favours side to move)",
        "- policy top moves: "
        + ", ".join(f"{m['uci']} ({m['prob']:.0%})" for m in report["top_moves"]),
    ]
    vs = ", ".join(f"{s['square']}({s['piece']})" for s in report["value_saliency"])
    ms = ", ".join(f"{s['square']}({s['piece']})" for s in report["move_saliency"])
    lines.append(f"- squares the value head attends to most: {vs}")
    lines.append(
        f"- squares the attention-weighted policy links to its move "
        f"{report['chosen_move']}: {ms}"
    )
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    from model import AttentionChessNet
    from train import pick_device

    dev = pick_device()
    ckpt = sys.argv[1] if len(sys.argv) > 1 else None
    net = AttentionChessNet().to(dev)
    if ckpt:
        ck = torch.load(ckpt, map_location=dev)
        net.load_state_dict(ck["model"])
        print(f"loaded {ckpt}")
    board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 2 3")
    rep = attention_report(net, board, dev)
    print(format_report_for_prompt(rep))
