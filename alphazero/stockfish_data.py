"""Generate training data by distilling Stockfish.

For each visited position we ask Stockfish (fixed depth, MultiPV=K) for its top
moves and evaluation. We store:
  * piece_ids (64) / globals (7)          -- network input
  * up to K (policy_index, prob) pairs     -- soft policy target (softmax over
    the top moves' evals, from the mover's perspective)
  * value in (-1, 1)                       -- tanh of the best move's eval

Positions come from lightly-randomised Stockfish self-play so the distribution
looks like real games while still covering diverse openings/middlegames. Runs
one Stockfish process per worker across all cores.

Usage:
  python stockfish_data.py --games 2000 --depth 12 --multipv 8 \
      --out data --shard-size 20000 --workers 10
"""

from __future__ import annotations

import argparse
import os
import random
import time
from typing import List, Tuple

import chess
import chess.engine
import numpy as np

from encoding import (
    board_to_tokens,
    canonical_board,
    cp_to_value,
    mirror_move,
    move_to_index,
)

STOCKFISH_CANDIDATES = [
    os.environ.get("STOCKFISH_PATH", ""),
    "/opt/homebrew/bin/stockfish",
    "/usr/local/bin/stockfish",
    "/usr/bin/stockfish",
]


def find_stockfish() -> str:
    for c in STOCKFISH_CANDIDATES:
        if c and os.path.exists(c):
            return c
    from shutil import which

    found = which("stockfish")
    if found:
        return found
    raise FileNotFoundError("Stockfish not found; set STOCKFISH_PATH")


def _softmax(x: np.ndarray, temp: float) -> np.ndarray:
    x = x / max(temp, 1e-6)
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def label_from_infos(board: chess.Board, infos):
    """Build (piece_ids, globals, indices, probs, value, top_moves) from a single
    Stockfish MultiPV ``analyse`` result. ``top_moves`` are real moves on
    ``board`` (best first) for game continuation."""
    if isinstance(infos, dict):  # single-pv fallback
        infos = [infos]

    _, mirrored = canonical_board(board)
    canon, _ = canonical_board(board)
    piece_ids, globals_ = board_to_tokens(canon)

    indices: List[int] = []
    cps: List[float] = []
    top_moves: List[chess.Move] = []
    for info in infos:
        pv = info.get("pv")
        if not pv:
            continue
        mv = pv[0]
        score = info["score"].relative  # from side-to-move perspective
        if score.is_mate():
            cp = 100000.0 if score.mate() > 0 else -100000.0
        else:
            cp = float(score.score())
        canon_mv = mirror_move(mv) if mirrored else mv
        try:
            idx = move_to_index(canon_mv)
        except ValueError:
            continue
        indices.append(idx)
        cps.append(cp)
        top_moves.append(mv)

    if not indices:
        raise ValueError("no analysable moves")

    cps_arr = np.asarray(cps, dtype=np.float32)
    # soft policy target: softmax over top moves' evals (mapped through tanh so
    # mate dominates but small cp gaps stay smooth), then sharpened.
    vals = np.tanh(cps_arr / 300.0)
    probs = _softmax(vals * 4.0, temp=1.0).astype(np.float32)
    best_value = float(np.clip(cp_to_value(float(cps_arr[0])), -1.0, 1.0))
    # raw best-move cp (clipped) so the value target can be re-scaled at train time
    best_cp = int(np.clip(cps_arr[0], -3200, 3200))
    return piece_ids.astype(np.int8), globals_, indices, probs, best_value, best_cp, top_moves


def play_labeled_game(
    engine: chess.engine.SimpleEngine,
    depth: int,
    multipv: int,
    max_plies: int,
    rng: random.Random,
    explore_prob: float = 0.0,
):
    """Yield labeled positions from one randomised Stockfish self-play game.

    ``explore_prob`` reintroduces occasional uniformly-random mid-game moves. This
    deliberately creates *imbalanced/losing* positions so the value net learns to
    recognise bad positions (fixing the over-optimism seen in diag.py), at the
    cost of some position realism.
    """
    board = chess.Board()
    # opening diversity: 0..12 random plies
    for _ in range(rng.randint(0, 12)):
        if board.is_game_over():
            return
        board.push(rng.choice(list(board.legal_moves)))

    plies = 0
    while not board.is_game_over() and plies < max_plies:
        infos = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=multipv)
        try:
            piece_ids, globals_, idxs, probs, value, best_cp, top_moves = label_from_infos(board, infos)
        except ValueError:
            break
        yield piece_ids, globals_, idxs, probs, value, best_cp

        # continue the game: mostly temperature-sample Stockfish's top moves
        # (realistic), but with probability explore_prob play a uniformly-random
        # legal move to reach diverse/imbalanced positions for value coverage.
        if explore_prob > 0 and rng.random() < explore_prob:
            move = rng.choice(list(board.legal_moves))
        else:
            w = _softmax(np.arange(len(top_moves), dtype=np.float32) * -0.7, temp=1.0)
            move = top_moves[int(rng.choices(range(len(top_moves)), weights=w)[0])]
        board.push(move)
        plies += 1


def _worker(args):
    """Run games and write shards to disk incrementally (durable, low memory).
    Returns the number of positions written."""
    wid, seed, n_games, depth, multipv, max_plies, out_dir, shard_size, tag, explore_prob = args
    rng = random.Random(seed)
    engine = chess.engine.SimpleEngine.popen_uci(find_stockfish())
    engine.configure({"Threads": 1, "Hash": 64})
    buffer: list = []
    shard_idx = 0
    total = 0
    try:
        for _ in range(n_games):
            for rec in play_labeled_game(engine, depth, multipv, max_plies, rng, explore_prob):
                buffer.append(rec)
            while len(buffer) >= shard_size:
                flush_shard(buffer[:shard_size], out_dir, f"{tag}w{wid:02d}_{shard_idx:05d}", multipv)
                buffer = buffer[shard_size:]
                shard_idx += 1
                total += shard_size
    finally:
        engine.quit()
    if buffer:
        flush_shard(buffer, out_dir, f"{tag}w{wid:02d}_{shard_idx:05d}", multipv)
        total += len(buffer)
    return total


def flush_shard(records, out_dir: str, shard_name, multipv: int):
    n = len(records)
    piece_ids = np.zeros((n, 64), dtype=np.int8)
    globals_ = np.zeros((n, 7), dtype=np.float32)
    pol_idx = np.full((n, multipv), -1, dtype=np.int64)
    pol_prob = np.zeros((n, multipv), dtype=np.float32)
    value = np.zeros((n,), dtype=np.float32)
    best_cp = np.zeros((n,), dtype=np.int16)
    for i, (pid, g, idxs, probs, v, cp) in enumerate(records):
        piece_ids[i] = pid
        globals_[i] = g
        k = min(len(idxs), multipv)
        pol_idx[i, :k] = idxs[:k]
        pol_prob[i, :k] = probs[:k]
        value[i] = v
        best_cp[i] = cp
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"shard_{shard_name}.npz")
    np.savez_compressed(
        path,
        piece_ids=piece_ids,
        globals=globals_,
        pol_idx=pol_idx,
        pol_prob=pol_prob,
        value=value,
        best_cp=best_cp,
    )
    return path, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=1000)
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--multipv", type=int, default=8)
    ap.add_argument("--max-plies", type=int, default=160)
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 1))
    ap.add_argument("--out", default="data")
    ap.add_argument("--shard-size", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="", help="filename prefix so multiple runs don't overwrite shards")
    ap.add_argument("--explore-prob", type=float, default=0.0,
                    help="prob of a uniformly-random mid-game move (diverse/losing positions)")
    args = ap.parse_args()

    import multiprocessing as mp

    games_per_worker = [args.games // args.workers] * args.workers
    for i in range(args.games % args.workers):
        games_per_worker[i] += 1
    tasks = [
        (w, args.seed + w, games_per_worker[w], args.depth, args.multipv,
         args.max_plies, args.out, args.shard_size, args.tag, args.explore_prob)
        for w in range(args.workers)
        if games_per_worker[w] > 0
    ]

    print(f"[gen] {args.games} games across {len(tasks)} workers, depth={args.depth}, "
          f"multipv={args.multipv}, shard-size={args.shard_size}")
    t0 = time.time()
    total = 0
    with mp.Pool(processes=len(tasks)) as pool:
        for wi, n in enumerate(pool.imap_unordered(_worker, tasks)):
            total += n
            elapsed = time.time() - t0
            print(f"[gen] worker done ({wi+1}/{len(tasks)}), +{n} positions, "
                  f"total={total}, {total/max(elapsed,1e-6):.0f} pos/s", flush=True)
    print(f"[gen] DONE {total} positions in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
