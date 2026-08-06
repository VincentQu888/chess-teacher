"""Play a head-to-head match between two checkpoints (candidate vs incumbent).

Used by the expert-iteration loop to gate promotion: the new net must clearly
outscore the current best before it becomes the new best. Both sides use PUCT
MCTS (no Dirichlet noise), alternating colours over varied openings.

  python head_to_head.py --a cand/best.pt --b best.pt --games 20 --sims 100
prints:  A_SCORE=<score in [0,1]>
"""

from __future__ import annotations

import argparse
import random

import chess

from mcts import MCTS
from selfplay import load_net, OPENING_BOOK
from train import pick_device


def play(mcts_a, mcts_b, a_is_white, opening, max_plies):
    board = chess.Board()
    for uci in opening:
        if board.is_game_over():
            break
        board.push_uci(uci)
    mcts_a.clear_cache()
    mcts_b.clear_cache()
    plies = 0
    while not board.is_game_over(claim_draw=True) and plies < max_plies:
        a_to_move = (board.turn == chess.WHITE) == a_is_white
        mcts = mcts_a if a_to_move else mcts_b
        move, _ = mcts.select_move(board, temperature=0.0)
        if move is None:
            break
        board.push(move)
        plies += 1
    res = board.result(claim_draw=True)
    if res == "1-0":
        return 1.0 if a_is_white else 0.0
    if res == "0-1":
        return 0.0 if a_is_white else 1.0
    return 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="candidate checkpoint")
    ap.add_argument("--b", required=True, help="incumbent checkpoint")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--sims", type=int, default=100)
    ap.add_argument("--c-puct", type=float, default=1.5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-plies", type=int, default=200)
    ap.add_argument("--device", default="")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = pick_device() if not args.device else __import__("torch").device(args.device)
    net_a, _ = load_net(args.a, device)
    net_b, _ = load_net(args.b, device)
    mcts_a = MCTS(net_a, device, c_puct=args.c_puct, n_sims=args.sims,
                  dirichlet_eps=0.0, batch_size=args.batch_size)
    mcts_b = MCTS(net_b, device, c_puct=args.c_puct, n_sims=args.sims,
                  dirichlet_eps=0.0, batch_size=args.batch_size)

    rng = random.Random(args.seed)
    score = 0.0
    w = d = l = 0
    for i in range(args.games):
        opening = rng.choice(OPENING_BOOK)
        a_white = (i % 2 == 0)
        g = play(mcts_a, mcts_b, a_white, opening, args.max_plies)
        score += g
        if g == 1.0:
            w += 1
        elif g == 0.5:
            d += 1
        else:
            l += 1
        print(f"[h2h] game {i+1}/{args.games} A={'W' if a_white else 'B'} res={g} "
              f"| A: {w}W {d}D {l}L score={score/(i+1):.3f}", flush=True)
    frac = score / args.games
    print(f"[h2h] FINAL A {w}W {d}D {l}L / {args.games}  A_SCORE={frac:.4f}")


if __name__ == "__main__":
    main()
