"""AlphaZero-style self-play game generation.

For every move the current net + PUCT MCTS produces an *improved* policy (the
root visit-count distribution). We record, per position:

  * piece_ids (64) / globals (7)   -- canonical (side-to-move) network input
  * pol_idx / pol_prob             -- MCTS visit distribution over legal moves
                                      (canonical-frame policy indices)
  * value z                        -- the eventual GAME OUTCOME from the mover's
                                      perspective (+1 win / 0 draw / -1 loss)
  * q_value                        -- MCTS root value (mover POV), for optional
                                      lower-variance value blending at train time

Training on (visit-policy, game-outcome) is the core AlphaZero self-play signal:
the visit target lets the net exceed its (distilled) teacher, and the outcome
target is inherently non-saturating and tactically grounded -- directly fixing
the "value blindness / over-optimism" that `diag.py` identified as the ceiling.

The attention architecture is unchanged (so its attention-weighted board state
still drives the chess-teacher explainer); only the training *targets* change.

Usage:
  python selfplay.py --ckpt checkpoints_v5/best.pt --games 400 --sims 200 \
      --workers 8 --out selfplay_data --tag iter1_
"""

from __future__ import annotations

import argparse
import os
import random
import time
from typing import List, Optional, Tuple

import chess
import chess.engine
import numpy as np
import torch

from encoding import encode_position, legal_moves_with_indices
from mcts import MCTS
from model import AttentionChessNet, ModelConfig
from stockfish_data import find_stockfish, label_from_infos, flush_shard as sd_flush_shard

MAX_PT = 96  # fixed policy-target width per position (top-N legal moves by visits)

# A small opening book (book plies) for extra self-play diversity on laptop compute.
OPENING_BOOK: List[List[str]] = [
    [], [], [],  # weight toward the start position
    ["e2e4", "e7e5"], ["e2e4", "c7c5"], ["e2e4", "e7e6"], ["e2e4", "c7c6"],
    ["d2d4", "d7d5"], ["d2d4", "g8f6"], ["c2c4", "e7e5"], ["g1f3", "d7d5"],
    ["e2e4", "e7e5", "g1f3", "b8c6"], ["d2d4", "d7d5", "c2c4", "c7c6"],
    ["e2e4", "c7c5", "g1f3", "d7d6"], ["d2d4", "g8f6", "c2c4", "g7g6"],
]


def load_net(ckpt_path: str, device) -> Tuple[AttentionChessNet, dict]:
    ck = torch.load(ckpt_path, map_location=device)
    cfg = ModelConfig(**ck["cfg"]) if "cfg" in ck else ModelConfig()
    net = AttentionChessNet(cfg).to(device)
    net.load_state_dict(ck["model"])
    net.eval()
    return net, ck.get("cfg", {})


def _visit_targets(board: chess.Board, root) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (canonical policy indices, visit probs, aligned visit counts) for
    the root's legal moves. Order matches root.moves == legal_moves order."""
    _, _, mirrored = encode_position(board)
    _, idxs = legal_moves_with_indices(board, mirrored)  # same order as root.moves
    visits = np.asarray(root.N, dtype=np.float64)
    if visits.sum() <= 0:
        visits = np.ones_like(visits)
    probs = (visits / visits.sum()).astype(np.float32)
    return idxs.astype(np.int64), probs, visits


def _root_value(root) -> float:
    """Visit-weighted mean action value at the root, from the mover's perspective."""
    n = np.asarray(root.N, dtype=np.float64)
    q = np.asarray(root.Q, dtype=np.float64)
    if n.sum() <= 0:
        return 0.0
    return float(np.average(q, weights=n))


def play_selfplay_game(
    mcts: MCTS,
    rng: random.Random,
    temp_moves: int,
    max_plies: int,
    resign_threshold: float,
    resign_moves: int,
    allow_resign: bool,
    book_prob: float,
):
    """Play one self-play game; return (records, result_str).

    Each record is (piece_ids int8[64], globals f32[7], idxs int64[k],
    probs f32[k], side_to_move bool, root_q f32).
    """
    board = chess.Board()
    if rng.random() < book_prob:
        for uci in rng.choice(OPENING_BOOK):
            if board.is_game_over():
                break
            board.push_uci(uci)

    mcts.clear_cache()
    records = []
    plies = 0
    resign_streak = {chess.WHITE: 0, chess.BLACK: 0}
    resigned_by: Optional[chess.Color] = None

    while not board.is_game_over(claim_draw=True) and plies < max_plies:
        root = mcts.search(board)
        if len(root.moves) == 0:
            break
        idxs, probs, visits = _visit_targets(board, root)
        rq = _root_value(root)
        piece_ids, g, _ = encode_position(board)
        records.append((piece_ids.astype(np.int8), g, idxs, probs, board.turn, rq))

        # resign handling (kept off for a fraction of games to avoid resign bias)
        if allow_resign:
            if rq <= resign_threshold:
                resign_streak[board.turn] += 1
                if resign_streak[board.turn] >= resign_moves:
                    resigned_by = board.turn
                    break
            else:
                resign_streak[board.turn] = 0

        # move selection: temperature 1.0 early (explore), greedy later
        if plies < temp_moves:
            dist = visits / visits.sum()
            a = int(np.random.default_rng().choice(len(visits), p=dist))
        else:
            a = int(np.argmax(visits))
        board.push(root.moves[a])
        plies += 1

    # game result -> per-position outcome z (mover POV)
    if resigned_by is not None:
        winner = not resigned_by
        result_str = "1-0" if winner == chess.WHITE else "0-1"
    else:
        result_str = board.result(claim_draw=True)
    if result_str == "1-0":
        winner = chess.WHITE
    elif result_str == "0-1":
        winner = chess.BLACK
    else:
        winner = None

    out = []
    for piece_ids, g, idxs, probs, stm, rq in records:
        if winner is None:
            z = 0.0
        else:
            z = 1.0 if winner == stm else -1.0
        out.append((piece_ids, g, idxs, probs, np.float32(z), np.float32(rq)))
    return out, result_str


def play_and_label_sf(mcts: MCTS, engine, sf_depth: int, sf_multipv: int, rng: random.Random,
                      temp_moves: int, max_plies: int, book_prob: float):
    """Expert-iteration / DAgger game: the NET (self-play, on-policy) chooses moves,
    but every visited position is labelled by STOCKFISH (soft policy over its MultiPV
    top moves + centipawn value). Returns stockfish_data-format records
    (piece_ids, globals, idxs, probs, value, best_cp) so the existing distillation
    trainer can consume them directly. This trains the net on the distribution it
    actually reaches while keeping high-quality (teacher) targets -> raises strength
    without destroying the value head."""
    board = chess.Board()
    if rng.random() < book_prob:
        for uci in rng.choice(OPENING_BOOK):
            if board.is_game_over():
                break
            board.push_uci(uci)
    mcts.clear_cache()
    records = []
    plies = 0
    while not board.is_game_over(claim_draw=True) and plies < max_plies:
        # expert label for the current (on-policy) position
        try:
            infos = engine.analyse(board, chess.engine.Limit(depth=sf_depth), multipv=sf_multipv)
            pid, g, idxs, probs, value, best_cp, _top = label_from_infos(board, infos)
            records.append((pid.astype(np.int8), g, idxs, probs, value, best_cp))
        except (ValueError, chess.engine.EngineError):
            pass
        # move selection: net + MCTS (on-policy); temperature early for diversity
        root = mcts.search(board)
        if len(root.moves) == 0:
            break
        visits = np.asarray(root.N, dtype=np.float64)
        if visits.sum() <= 0:
            visits = np.ones_like(visits)
        if plies < temp_moves:
            a = int(np.random.default_rng().choice(len(visits), p=visits / visits.sum()))
        else:
            a = int(np.argmax(visits))
        board.push(root.moves[a])
        plies += 1
    return records


def _worker_sf(args):
    (wid, seed, n_games, ckpt, device_str, sims, c_puct, dir_eps, dir_alpha, fpu,
     batch_size, temp_moves, max_plies, book_prob, sf_depth, sf_multipv,
     out_dir, shard_size, tag) = args
    torch.set_num_threads(1)
    device = torch.device(device_str)
    net, _ = load_net(ckpt, device)
    mcts = MCTS(net, device, c_puct=c_puct, n_sims=sims, dirichlet_alpha=dir_alpha,
                dirichlet_eps=dir_eps, fpu=fpu, batch_size=batch_size)
    engine = chess.engine.SimpleEngine.popen_uci(find_stockfish())
    engine.configure({"Threads": 1, "Hash": 64})
    rng = random.Random(seed)
    buffer: list = []
    shard_idx = 0
    total = 0
    try:
        for _gi in range(n_games):
            recs = play_and_label_sf(mcts, engine, sf_depth, sf_multipv, rng,
                                     temp_moves, max_plies, book_prob)
            buffer.extend(recs)
            while len(buffer) >= shard_size:
                sd_flush_shard(buffer[:shard_size], out_dir, f"{tag}w{wid:02d}_{shard_idx:05d}", sf_multipv)
                buffer = buffer[shard_size:]
                shard_idx += 1
                total += shard_size
    finally:
        engine.quit()
    if buffer:
        sd_flush_shard(buffer, out_dir, f"{tag}w{wid:02d}_{shard_idx:05d}", sf_multipv)
        total += len(buffer)
    return total, {"sf_labelled": total}


def _pad_targets(idxs: np.ndarray, probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Pad/truncate a variable-length policy target to MAX_PT (keep top by prob)."""
    if len(idxs) > MAX_PT:
        keep = np.argsort(-probs)[:MAX_PT]
        idxs, probs = idxs[keep], probs[keep]
        probs = probs / probs.sum()
    pi = np.full(MAX_PT, -1, dtype=np.int64)
    pp = np.zeros(MAX_PT, dtype=np.float32)
    pi[: len(idxs)] = idxs
    pp[: len(probs)] = probs
    return pi, pp


def flush_shard(records, out_dir: str, shard_name: str):
    n = len(records)
    piece_ids = np.zeros((n, 64), dtype=np.int8)
    globals_ = np.zeros((n, 7), dtype=np.float32)
    pol_idx = np.full((n, MAX_PT), -1, dtype=np.int64)
    pol_prob = np.zeros((n, MAX_PT), dtype=np.float32)
    value = np.zeros((n,), dtype=np.float32)
    q_value = np.zeros((n,), dtype=np.float32)
    for i, (pid, g, idxs, probs, z, rq) in enumerate(records):
        piece_ids[i] = pid
        globals_[i] = g
        pi, pp = _pad_targets(idxs, probs)
        pol_idx[i] = pi
        pol_prob[i] = pp
        value[i] = z
        q_value[i] = rq
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"shard_{shard_name}.npz")
    np.savez_compressed(path, piece_ids=piece_ids, globals=globals_,
                        pol_idx=pol_idx, pol_prob=pol_prob, value=value, q_value=q_value)
    return path, n


def _worker(args):
    (wid, seed, n_games, ckpt, device_str, sims, c_puct, dir_eps, dir_alpha, fpu,
     batch_size, temp_moves, max_plies, resign_threshold, resign_moves,
     no_resign_frac, book_prob, out_dir, shard_size, tag) = args
    torch.set_num_threads(1)
    device = torch.device(device_str)
    net, _ = load_net(ckpt, device)
    mcts = MCTS(net, device, c_puct=c_puct, n_sims=sims, dirichlet_alpha=dir_alpha,
                dirichlet_eps=dir_eps, fpu=fpu, batch_size=batch_size)
    rng = random.Random(seed)
    buffer: list = []
    shard_idx = 0
    total = 0
    results = {"1-0": 0, "0-1": 0, "1/2-1/2": 0}
    for gi in range(n_games):
        allow_resign = rng.random() >= no_resign_frac
        recs, res = play_selfplay_game(
            mcts, rng, temp_moves, max_plies, resign_threshold, resign_moves,
            allow_resign, book_prob)
        results[res] = results.get(res, 0) + 1
        buffer.extend(recs)
        while len(buffer) >= shard_size:
            flush_shard(buffer[:shard_size], out_dir, f"{tag}w{wid:02d}_{shard_idx:05d}")
            buffer = buffer[shard_size:]
            shard_idx += 1
            total += shard_size
    if buffer:
        flush_shard(buffer, out_dir, f"{tag}w{wid:02d}_{shard_idx:05d}")
        total += len(buffer)
    return total, results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints_v5/best.pt")
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--c-puct", type=float, default=1.5)
    ap.add_argument("--dirichlet-eps", type=float, default=0.25)
    ap.add_argument("--dirichlet-alpha", type=float, default=0.3)
    ap.add_argument("--fpu", type=float, default=0.2)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--temp-moves", type=int, default=24, help="plies of temperature-1 sampling")
    ap.add_argument("--max-plies", type=int, default=200)
    ap.add_argument("--resign-threshold", type=float, default=-0.90)
    ap.add_argument("--resign-moves", type=int, default=4)
    ap.add_argument("--no-resign-frac", type=float, default=0.15)
    ap.add_argument("--book-prob", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--device", default="cpu", help="cpu (multiproc-safe) or mps (single worker)")
    ap.add_argument("--out", default="selfplay_data")
    ap.add_argument("--shard-size", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--label-stockfish", action="store_true",
                    help="expert-iteration mode: net plays (on-policy) but Stockfish labels "
                         "each position (writes stockfish_data-format shards for train.py)")
    ap.add_argument("--sf-depth", type=int, default=12)
    ap.add_argument("--sf-multipv", type=int, default=8)
    args = ap.parse_args()

    if args.device != "cpu" and args.workers > 1:
        print(f"[selfplay] device={args.device} forces workers=1 (GPU not multiprocess-safe)")
        args.workers = 1

    import multiprocessing as mp

    per = [args.games // args.workers] * args.workers
    for i in range(args.games % args.workers):
        per[i] += 1
    worker_fn = _worker_sf if args.label_stockfish else _worker
    if args.label_stockfish:
        tasks = [
            (w, args.seed + 1000 * w, per[w], args.ckpt, args.device, args.sims, args.c_puct,
             args.dirichlet_eps, args.dirichlet_alpha, args.fpu, args.batch_size,
             args.temp_moves, args.max_plies, args.book_prob, args.sf_depth, args.sf_multipv,
             args.out, args.shard_size, args.tag)
            for w in range(args.workers) if per[w] > 0
        ]
    else:
        tasks = [
            (w, args.seed + 1000 * w, per[w], args.ckpt, args.device, args.sims, args.c_puct,
             args.dirichlet_eps, args.dirichlet_alpha, args.fpu, args.batch_size,
             args.temp_moves, args.max_plies, args.resign_threshold, args.resign_moves,
             args.no_resign_frac, args.book_prob, args.out, args.shard_size, args.tag)
            for w in range(args.workers) if per[w] > 0
        ]
    print(f"[selfplay] {args.games} games / {len(tasks)} workers, sims={args.sims}, "
          f"device={args.device}, label_sf={args.label_stockfish}, ckpt={args.ckpt} "
          f"-> {args.out} (tag='{args.tag}')", flush=True)
    t0 = time.time()
    total = 0
    agg = {"1-0": 0, "0-1": 0, "1/2-1/2": 0}
    if len(tasks) == 1:
        n, res = worker_fn(tasks[0])
        total += n
        for k, v in res.items():
            agg[k] = agg.get(k, 0) + v
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=len(tasks)) as pool:
            for wi, (n, res) in enumerate(pool.imap_unordered(worker_fn, tasks)):
                total += n
                for k, v in res.items():
                    agg[k] = agg.get(k, 0) + v
                el = time.time() - t0
                print(f"[selfplay] worker {wi+1}/{len(tasks)} done +{n} pos, total={total}, "
                      f"{total/max(el,1e-6):.0f} pos/s, results={agg}", flush=True)
    print(f"[selfplay] DONE {total} positions in {time.time()-t0:.1f}s; results W/D/L(white) = "
          f"{agg.get('1-0',0)}/{agg.get('1/2-1/2',0)}/{agg.get('0-1',0)}")


if __name__ == "__main__":
    main()
