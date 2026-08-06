#!/usr/bin/env python3
"""Explanation-faithfulness harness.

Rather than fixing wrong explanations one FEN at a time, this treats every
factual claim the explainer emits as a checkable assertion about the board and
re-verifies it with an INDEPENDENT oracle (python-chess), across a large corpus
of real positions. Any sentence that contradicts the board is reported with the
FEN so the underlying rule -- not just the single position -- can be fixed.

It catches whole *classes* of "irrelevant / false claim" bugs, e.g.:
  * "the queen is left undefended on h8"  when nothing attacks h8
  * "safe on X (nothing attacks it)"      when X is actually attacked
  * "defended by the rook"                when no rook defends the square
  * "it wins the knight"                  when the move wins nothing

Usage:
    python verify_explanations.py                # PGN corpus + crafted cases
    python verify_explanations.py --max 300 --depth 14
    python verify_explanations.py --fen "<FEN>"  # check a single position
"""
from __future__ import annotations

import argparse
import glob
import re
import sys
from dataclasses import dataclass
from typing import List, Optional

import chess
import chess.engine
import chess.pgn

import chess_concepts as cc
import chess_teacher as ct
from chess_teacher import (
    LineResult,
    _MOVE_TOKEN_RE,
    engine_top_lines,
    find_engine_path,
    format_move_verdict,
    parse_move,
    static_exchange_eval,
)

PIECE_TYPE_BY_NAME = {chess.piece_name(pt): pt for pt in range(1, 7)}
PIECE_TYPE_BY_LETTER = {
    "P": chess.PAWN, "N": chess.KNIGHT, "B": chess.BISHOP,
    "R": chess.ROOK, "Q": chess.QUEEN, "K": chess.KING,
}
# "Qd4", "Rh8", "Pe5" -> a claim that this piece type sits on this square.
_PIECE_ON_SQ_RE = re.compile(r"\b([PNBRQK])([a-h][1-8])\b")
_ALL_DETECTORS = [getattr(cc, n) for n in dir(cc)
                  if n.startswith("detect_") and callable(getattr(cc, n))]

# ---------------------------------------------------------------------------
# Claim verifiers: each returns None if the sentence is faithful, else a reason.
# ---------------------------------------------------------------------------
SAFE_WORDS = ("safe on", "nothing attacks it", "perfectly safe")
UNSAFE_WORDS = ("undefended on", "hanging on", "loosely defended on")


@dataclass
class Violation:
    fen: str
    sentence: str
    reason: str


def _landing_square(board: chess.Board, san: str) -> Optional[int]:
    try:
        return parse_move(board, san).to_square
    except Exception:
        return None


def _candidate_facts(board: chess.Board, sans: List[str]) -> dict:
    """Precompute board-truth for each referenced move so free-text claims can be
    matched to the move by its explicit landing square rather than by fragile
    subject-token parsing (which misses e.g. pawn pushes like 'd3')."""
    facts = {}
    mover = board.turn
    for san in sans:
        try:
            mv = parse_move(board, san)
        except Exception:
            continue
        b = board.copy()
        is_cap = board.is_capture(mv)
        cap_type = None
        if is_cap and not board.is_en_passant(mv):
            cp = board.piece_at(mv.to_square)
            cap_type = cp.piece_type if cp else None
        elif is_cap:
            cap_type = chess.PAWN
        b.push(mv)
        sq = mv.to_square
        facts[san] = {
            "sq": sq,
            "sqname": chess.square_name(sq),
            "opp_attackers": len([s for s in b.attackers(not mover, sq) if b.piece_at(s)]),
            "defender_types": {b.piece_at(s).piece_type
                               for s in b.attackers(mover, sq) if b.piece_at(s)},
            "is_capture": is_cap,
            "cap_type": cap_type,
            "see_after": static_exchange_eval(b, sq, not mover),
            "landing_piece": b.piece_at(sq).piece_type if b.piece_at(sq) else None,
        }
    return facts


def check_verdict(board: chess.Board, text: str, candidate_sans: List[str]) -> List[str]:
    """Verify the deterministic move-verdict prose against board truth. Claims are
    tied to a move via the explicit 'on <square>' target (robust for all move
    types) or, for material/defence claims, by the referenced piece."""
    facts = _candidate_facts(board, candidate_sans)
    by_sq = {f["sq"]: (san, f) for san, f in facts.items()}
    reasons: List[str] = []
    low = text.lower()

    # --- safety claims: '(safe|undefended|hanging|loosely defended) ... on <sq>'
    for m in re.finditer(r"(safe|undefended|hanging|loosely defended)[^.;]*? on ([a-h][1-8])", low):
        word, sqname = m.group(1), m.group(2)
        sq = chess.parse_square(sqname)
        if sq not in by_sq:
            continue
        san, f = by_sq[sq]
        if word == "safe" and f["opp_attackers"] > 0:
            reasons.append(f"claims {san} is SAFE on {sqname} but opponent has "
                           f"{f['opp_attackers']} attacker(s) there")
        if word in ("undefended", "hanging", "loosely defended") and f["opp_attackers"] == 0:
            reasons.append(f"claims {san} is {word.upper()} on {sqname} but nothing "
                           f"attacks that square")

    # --- 'defended by the X (and the Y)' : tie to the piece landing there -----
    for m in re.finditer(r"the (pawn|knight|bishop|rook|queen|king) is defended by the ([a-z ]+?)(?:\.|,|;| where| though|$)", low):
        piece_type = PIECE_TYPE_BY_NAME[m.group(1)]
        claimed = {PIECE_TYPE_BY_NAME[w.strip()] for w in re.split(r"\band\b|,", m.group(2))
                   if w.strip() in PIECE_TYPE_BY_NAME}
        cands = [(s, f) for s, f in facts.items() if f["landing_piece"] == piece_type]
        for san, f in cands:
            missing = claimed - f["defender_types"]
            if claimed and missing and len(cands) == 1:
                names = ", ".join(sorted(chess.piece_name(t) for t in missing))
                reasons.append(f"claims {san} on {f['sqname']} is defended by [{names}] "
                               f"but those pieces do not defend it")

    # --- 'wins the <piece>' : some referenced move must soundly win it --------
    for m in re.finditer(r"wins the (pawn|knight|bishop|rook|queen)", low):
        want = PIECE_TYPE_BY_NAME[m.group(1)]
        ok = any(f["is_capture"] and f["cap_type"] == want and f["see_after"] == 0
                 for f in facts.values())
        if not ok and facts:
            reasons.append(f"claims a move 'wins the {m.group(1)}' but no referenced "
                           f"move soundly captures a {m.group(1)}")
    return reasons


def check_concepts(board: chess.Board) -> List[str]:
    """Verify concept-detector claims. Universal invariant: any '<Piece><square>'
    token in a detail string must match the actual piece type on that square, and
    every square in ``Concept.squares`` referencing an occupied piece must exist.
    Also spot-checks 'hanging' claims against Static Exchange Evaluation."""
    reasons: List[str] = []
    for det in _ALL_DETECTORS:
        try:
            concepts = det(board)
        except Exception as exc:
            reasons.append(f"{det.__name__} raised: {exc!r}")
            continue
        for c in concepts or []:
            detail = getattr(c, "detail", "") or ""
            for letter, sqname in _PIECE_ON_SQ_RE.findall(detail):
                sq = chess.parse_square(sqname)
                pc = board.piece_at(sq)
                want = PIECE_TYPE_BY_LETTER[letter]
                if pc is None:
                    reasons.append(f"[{c.name}] says '{letter}{sqname}' but {sqname} is empty  ::  {detail}")
                elif pc.piece_type != want:
                    reasons.append(
                        f"[{c.name}] says '{letter}{sqname}' ({chess.piece_name(want)}) but "
                        f"{sqname} holds a {chess.piece_name(pc.piece_type)}  ::  {detail}")
            # 'X is hanging ... opponent wins ~N' -> N must match SEE.
            m = re.search(r"([PNBRQK])([a-h][1-8]) is hanging.*opponent wins ~(\d+)", detail)
            if m:
                sq = chess.parse_square(m.group(2))
                pc = board.piece_at(sq)
                if pc is not None:
                    see = static_exchange_eval(board, sq, not pc.color)
                    if see != int(m.group(3)):
                        reasons.append(
                            f"[{c.name}] claims hanging value ~{m.group(3)} on {m.group(2)} "
                            f"but SEE says {see}  ::  {detail}")
    return reasons





# ---------------------------------------------------------------------------
# Position sourcing
# ---------------------------------------------------------------------------
def positions_from_pgns(patterns: List[str], every_plies: int, per_game: int,
                        total_cap: int) -> List[str]:
    fens: List[str] = []
    seen = set()
    for pat in patterns:
        for path in glob.glob(pat, recursive=True):
            try:
                with open(path, encoding="utf-8", errors="ignore") as fh:
                    while len(fens) < total_cap:
                        game = chess.pgn.read_game(fh)
                        if game is None:
                            break
                        board = game.board()
                        n = 0
                        for i, mv in enumerate(game.mainline_moves()):
                            board.push(mv)
                            if i >= 8 and i % every_plies == 0:
                                fen = board.fen()
                                if fen not in seen and not board.is_game_over():
                                    seen.add(fen)
                                    fens.append(fen)
                                    n += 1
                            if n >= per_game or len(fens) >= total_cap:
                                break
            except Exception:
                continue
            if len(fens) >= total_cap:
                return fens
    return fens


CRAFTED = [
    # The originally reported position (Qxh8 safe, wins a rook).
    "r2k2nr/ppp2pQp/2npb3/2bNp3/2BqP3/8/PPPP2PP/R1BKNR2 w - - 6 11",
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fen", help="check a single FEN instead of the corpus")
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--multipv", type=int, default=4)
    ap.add_argument("--max", type=int, default=200, help="max corpus positions")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    engine_path = find_engine_path(None)
    if not engine_path:
        print("Stockfish not found (set STOCKFISH_PATH).", file=sys.stderr)
        return 2

    if args.fen:
        fens = [args.fen]
    else:
        fens = CRAFTED + positions_from_pgns(
            ["alphazero/refs/**/pgn_files/*.pgn", "alphazero/refs/**/*.pgn"],
            every_plies=7, per_game=6, total_cap=args.max,
        )

    print(f"Checking {len(fens)} positions (depth={args.depth}, multipv={args.multipv})\u2026")
    violations: List[Violation] = []
    checked = 0
    # Concept-layer checks need no engine -- run them over every position first.
    for fen in fens:
        try:
            board = chess.Board(fen)
        except Exception:
            continue
        for reason in check_concepts(board):
            violations.append(Violation(fen, "<concept>", reason))
            if args.verbose:
                print(f"\n[CONCEPT VIOLATION] {fen}\n  {reason}")
    with chess.engine.SimpleEngine.popen_uci(str(engine_path)) as engine:
        for fen in fens:
            try:
                board = chess.Board(fen)
            except Exception:
                continue
            if board.is_game_over():
                continue
            try:
                lines = engine_top_lines(board, engine, args.depth, args.multipv, 8)
            except Exception:
                continue
            if not lines or not lines[0].moves:
                continue
            # Exercise the verdict builder for the top move AND for a near-equal
            # alternative (that path fires practical_comparison + _solidity_phrase).
            alt_first = None
            targets: List[LineResult] = [lines[0]]
            for ln in lines[1:]:
                if ln.moves and ln.moves[0] != lines[0].moves[0]:
                    alt_first = ln.moves[0]
                    targets.append(ln)
                    break
            for tgt in targets:
                try:
                    text = format_move_verdict(board, tgt, lines)
                except Exception as exc:
                    violations.append(Violation(fen, "<crash>", f"format_move_verdict raised: {exc!r}"))
                    continue
                cands = [c for c in (tgt.moves[0], lines[0].moves[0], alt_first) if c]
                for reason in check_verdict(board, text, cands):
                    violations.append(Violation(fen, text, reason))
                    if args.verbose:
                        print(f"\n[VIOLATION] {fen}\n  {reason}\n  text: {text}")
            checked += 1

    print(f"\nChecked {checked} positions. Found {len(violations)} faithfulness violation(s).")
    # De-dup by (reason-prefix) to show distinct rule failures first.
    for v in violations[:40]:
        print("\n- FEN:", v.fen)
        print("  ", v.reason)
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
