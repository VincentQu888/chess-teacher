"""Evaluate the trained bot against Stockfish limited to a target Elo.

Plays a match (alternating colours, varied openings) vs Stockfish with
UCI_LimitStrength / UCI_Elo, reports the score and an Elo estimate:

    elo(bot) ~= target_elo - 400 * log10(1/score - 1)

  python evaluate.py --ckpt checkpoints/best.pt --games 40 --sims 160 \
      --sf-elo 2000
"""

from __future__ import annotations

import argparse
import math
import os
import random
from typing import List

import chess
import chess.engine
import torch

from mcts import MCTS
from model import AttentionChessNet, ModelConfig
from stockfish_data import find_stockfish
from train import pick_device

# A handful of common, sound openings (a few book plies) for game variety.
OPENINGS: List[List[str]] = [
    [],  # start position
    ["e2e4", "e7e5"],
    ["e2e4", "c7c5"],
    ["e2e4", "e7e6"],
    ["e2e4", "c7c6"],
    ["d2d4", "d7d5"],
    ["d2d4", "g8f6"],
    ["c2c4", "e7e5"],
    ["g1f3", "d7d5"],
    ["e2e4", "e7e5", "g1f3", "b8c6"],
    ["d2d4", "d7d5", "c2c4", "e7e6"],
    ["e2e4", "c7c5", "g1f3", "d7d6"],
]


def load_net(ckpt_path: str, device) -> AttentionChessNet:
    ck = torch.load(ckpt_path, map_location=device)
    cfg = ModelConfig(**ck["cfg"]) if "cfg" in ck else ModelConfig()
    net = AttentionChessNet(cfg).to(device)
    net.load_state_dict(ck["model"])
    net.eval()
    return net


def elo_estimate(score: float, games: int, base_elo: float):
    score = min(max(score, 1e-9), 1 - 1e-9)
    diff = -400.0 * math.log10(1.0 / score - 1.0)
    # rough 95% CI from score stderr
    se = math.sqrt(max(score * (1 - score) / games, 1e-12))
    lo = min(max(score - 1.96 * se, 1e-9), 1 - 1e-9)
    hi = min(max(score + 1.96 * se, 1e-9), 1 - 1e-9)
    d_lo = -400.0 * math.log10(1.0 / lo - 1.0)
    d_hi = -400.0 * math.log10(1.0 / hi - 1.0)
    return base_elo + diff, base_elo + d_lo, base_elo + d_hi


def play_game(mcts: MCTS, engine, bot_is_white: bool, opening: List[str],
              sf_limit: chess.engine.Limit, temperature: float, max_plies: int) -> float:
    """Return bot score for one game: 1.0 win, 0.5 draw, 0.0 loss."""
    board = chess.Board()
    for uci in opening:
        board.push_uci(uci)
    plies = 0
    while not board.is_game_over(claim_draw=True) and plies < max_plies:
        bot_to_move = (board.turn == chess.WHITE) == bot_is_white
        if bot_to_move:
            move, _ = mcts.select_move(board, temperature=temperature)
        else:
            move = engine.play(board, sf_limit).move
        if move is None:
            break
        board.push(move)
        plies += 1
    result = board.result(claim_draw=True)
    if result == "1-0":
        return 1.0 if bot_is_white else 0.0
    if result == "0-1":
        return 0.0 if bot_is_white else 1.0
    return 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/best.pt")
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--sims", type=int, default=160)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--c-puct", type=float, default=1.5)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--sf-elo", type=int, default=2000)
    ap.add_argument("--sf-movetime", type=float, default=0.1)
    ap.add_argument("--max-plies", type=int, default=220)
    ap.add_argument("--device", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pgn-out", default="")
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else pick_device()
    net = load_net(args.ckpt, device)
    mcts = MCTS(net, device, c_puct=args.c_puct, n_sims=args.sims,
                dirichlet_eps=0.0, batch_size=args.batch_size)
    print(f"[eval] ckpt={args.ckpt} device={device} sims={args.sims} vs SF Elo {args.sf_elo}")

    engine = chess.engine.SimpleEngine.popen_uci(find_stockfish())
    engine.configure({"UCI_LimitStrength": True, "UCI_Elo": args.sf_elo, "Threads": 1})
    sf_limit = chess.engine.Limit(time=args.sf_movetime)

    rng = random.Random(args.seed)
    score = 0.0
    w = d = l = 0
    try:
        for i in range(args.games):
            opening = rng.choice(OPENINGS)
            bot_white = (i % 2 == 0)
            mcts.clear_cache()
            g = play_game(mcts, engine, bot_white, opening, sf_limit,
                          args.temperature, args.max_plies)
            score += g
            if g == 1.0:
                w += 1
            elif g == 0.5:
                d += 1
            else:
                l += 1
            frac = score / (i + 1)
            est, lo, hi = elo_estimate(frac, i + 1, args.sf_elo)
            print(f"[eval] game {i+1}/{args.games} bot={'W' if bot_white else 'B'} "
                  f"res={g} | W{w} D{d} L{l} score={frac:.3f} "
                  f"est_elo~{est:.0f} [{lo:.0f},{hi:.0f}]", flush=True)
    finally:
        engine.quit()

    frac = score / args.games
    est, lo, hi = elo_estimate(frac, args.games, args.sf_elo)
    print(f"\n[eval] FINAL: {w}W {d}D {l}L / {args.games}  score={frac:.3f}")
    print(f"[eval] estimated bot Elo ~= {est:.0f}  (95% CI [{lo:.0f}, {hi:.0f}])  "
          f"vs Stockfish@{args.sf_elo}")
    if est >= 2000:
        print("[eval] >>> TARGET MET: estimated Elo >= 2000")
    else:
        print("[eval] target not yet met (>=2000)")


if __name__ == "__main__":
    main()
