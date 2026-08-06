"""Diagnose HOW the bot loses vs Stockfish@2000.

For each bot move we take a strong reference eval (full-strength Stockfish, fixed
depth) of the position before and after the move, both from the bot's
perspective. A large drop => the bot's move was a blunder. This tells us whether
losses come from tactical blunders (policy/value/search problem) or slow
positional decline (eval/strategy problem).

  python diag.py --ckpt checkpoints_v3/best.pt --games 4 --sims 800
"""

from __future__ import annotations

import argparse
import random

import chess
import chess.engine

from mcts import MCTS
from evaluate import load_net, OPENINGS
from stockfish_data import find_stockfish
from train import pick_device


def ref_cp(engine, board, depth, pov):
    """Reference eval in centipawns from ``pov``'s perspective (clamped)."""
    info = engine.analyse(board, chess.engine.Limit(depth=depth))
    score = info["score"].pov(pov)
    if score.is_mate():
        return 3000 if score.mate() > 0 else -3000
    return max(-3000, min(3000, score.score()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints_v3/best.pt")
    ap.add_argument("--games", type=int, default=4)
    ap.add_argument("--sims", type=int, default=800)
    ap.add_argument("--ref-depth", type=int, default=13)
    ap.add_argument("--blunder-cp", type=int, default=150)
    ap.add_argument("--sf-elo", type=int, default=2000)
    ap.add_argument("--max-plies", type=int, default=200)
    args = ap.parse_args()

    device = pick_device()
    net = load_net(args.ckpt, device)
    mcts = MCTS(net, device, n_sims=args.sims, dirichlet_eps=0.0, batch_size=32)

    opp = chess.engine.SimpleEngine.popen_uci(find_stockfish())
    opp.configure({"UCI_LimitStrength": True, "UCI_Elo": args.sf_elo, "Threads": 1})
    ref = chess.engine.SimpleEngine.popen_uci(find_stockfish())
    ref.configure({"Threads": 1})

    rng = random.Random(0)
    totals = {"blunders": 0, "bot_moves": 0, "games": 0}
    try:
        for gi in range(args.games):
            board = chess.Board()
            for uci in rng.choice(OPENINGS):
                board.push_uci(uci)
            bot_white = gi % 2 == 0
            bot_color = chess.WHITE if bot_white else chess.BLACK
            mcts.clear_cache()
            blunders = []
            plies = 0
            while not board.is_game_over(claim_draw=True) and plies < args.max_plies:
                if board.turn == bot_color:
                    cp_before = ref_cp(ref, board, args.ref_depth, bot_color)
                    root_val = None
                    mv, root = mcts.select_move(board)
                    if root is not None:
                        root_val = float(root.Q[int(root.N.argmax())])
                    board.push(mv)
                    cp_after = ref_cp(ref, board, args.ref_depth, bot_color)
                    totals["bot_moves"] += 1
                    drop = cp_after - cp_before
                    if drop <= -args.blunder_cp:
                        blunders.append((board.fullmove_number, mv.uci(), cp_before,
                                         cp_after, root_val))
                else:
                    mv = opp.play(board, chess.engine.Limit(time=0.1)).move
                    if mv is None:
                        break
                    board.push(mv)
                plies += 1
            result = board.result(claim_draw=True)
            totals["blunders"] += len(blunders)
            totals["games"] += 1
            print(f"\n=== Game {gi+1}: bot={'White' if bot_white else 'Black'} "
                  f"result={result} plies={plies} blunders={len(blunders)} ===")
            for fm, uci, cb, ca, rv in blunders[:6]:
                rvs = f"{rv:+.2f}" if rv is not None else "n/a"
                print(f"  move {fm}: {uci}  ref cp {cb:+d} -> {ca:+d} (drop {ca-cb:+d}); "
                      f"bot's own value={rvs}")
    finally:
        opp.quit()
        ref.quit()

    bm = max(totals["bot_moves"], 1)
    print(f"\n[diag] {totals['games']} games, {totals['bot_moves']} bot moves, "
          f"{totals['blunders']} blunders (>= {args.blunder_cp}cp drop) "
          f"= {100*totals['blunders']/bm:.1f}% of moves")


if __name__ == "__main__":
    main()
