"""Deterministic detectors for every chess concept in CHESS_CONCEPTS.md.

Each detector inspects a python-chess ``Board`` (and, where useful, the side to
move's legal moves) and returns zero or more :class:`Concept` findings describing
a concept that is *present* or *available* in the position, with concrete,
position-specific detail. The chess-teacher explainer feeds these findings to the
LLM as verified ground truth, alongside the engine lines and the attention-
weighted saliency from the neural bot.

Design:
- Self-contained (only depends on ``python-chess``) so it can be unit-tested and
  imported by ``chess_teacher`` without a circular import.
- Detectors are conservative: they should fire on clear instances of a concept
  and stay quiet otherwise. Every concept name below has at least one validating
  test in ``test_chess_concepts.py``.
- ``ALL_CONCEPTS`` is the canonical list mirroring CHESS_CONCEPTS.md; a test
  asserts every name is covered by a detector.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import chess

WHITE, BLACK = chess.WHITE, chess.BLACK
PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 100,
}


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------
@dataclass
class Concept:
    name: str
    category: str
    detail: str
    side: Optional[str] = None  # "White" / "Black" / None
    squares: List[str] = field(default_factory=list)

    def line(self) -> str:
        who = f"{self.side}: " if self.side else ""
        return f"[{self.name}] {who}{self.detail}"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def cname(color: chess.Color) -> str:
    return "White" if color == WHITE else "Black"


def pname(pt: int) -> str:
    return chess.piece_name(pt)


def sqn(s: int) -> str:
    return chess.square_name(s)


def val(pt: int) -> int:
    return PIECE_VALUES[pt]


def static_exchange_eval(board: chess.Board, target_square: int,
                         attacker_color: chess.Color) -> int:
    """Net material (in pawn units) the attacker wins by initiating captures on
    ``target_square`` (re-queries attackers so x-rays are handled)."""
    target = board.piece_at(target_square)
    if not target or target.color == attacker_color:
        return 0
    sim = board.copy(stack=False)
    gains: List[int] = []
    side = attacker_color
    current_value = PIECE_VALUES[target.piece_type]
    while True:
        attackers = sim.attackers(side, target_square)
        if not attackers:
            break
        least_sq = min(attackers, key=lambda s: PIECE_VALUES[sim.piece_at(s).piece_type])
        piece = sim.piece_at(least_sq)
        if piece is None:
            break
        gains.append(current_value)
        sim.remove_piece_at(least_sq)
        sim.set_piece_at(target_square, piece)
        current_value = PIECE_VALUES[piece.piece_type]
        side = not side
    result = 0
    for g in reversed(gains):
        result = max(0, g - result)
    return result


def is_hanging(board: chess.Board, square: int) -> bool:
    piece = board.piece_at(square)
    if not piece or piece.piece_type == chess.KING:
        return False
    return static_exchange_eval(board, square, not piece.color) > 0


def pawn_map(board: chess.Board) -> Dict[chess.Color, Dict[int, List[int]]]:
    pawns: Dict[chess.Color, Dict[int, List[int]]] = {WHITE: {}, BLACK: {}}
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p and p.piece_type == chess.PAWN:
            pawns[p.color].setdefault(chess.square_file(sq), []).append(chess.square_rank(sq))
    return pawns


def pieces_of(board: chess.Board, color: chess.Color, types) -> List[int]:
    out = []
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p and p.color == color and p.piece_type in types:
            out.append(sq)
    return out


SLIDING = {chess.BISHOP, chess.ROOK, chess.QUEEN}


def _describe(board: chess.Board, squares) -> str:
    items = []
    for s in squares:
        p = board.piece_at(s)
        if p:
            items.append((PIECE_VALUES[p.piece_type], p.symbol().upper() + sqn(s)))
    items.sort(key=lambda x: x[0])
    return ", ".join(sym for _, sym in items) if items else "none"


def _valuable_targets_from(board: chess.Board, from_sq: int, mover: chess.Color,
                           min_val: int = 3):
    """Enemy pieces (value>=min_val or king) attacked by the piece on from_sq."""
    out = []
    piece = board.piece_at(from_sq)
    if not piece:
        return out
    for t in board.attacks(from_sq):
        tp = board.piece_at(t)
        if tp and tp.color != mover and (tp.piece_type == chess.KING or val(tp.piece_type) >= min_val):
            out.append(t)
    return out


# ===========================================================================
# 1. TACTICAL MOTIFS
# ===========================================================================
def detect_forks(board: chess.Board) -> List[Concept]:
    """Existing forks on the board + forks available to the side to move."""
    out: List[Concept] = []
    # existing: a piece attacking >=2 valuable enemy targets, itself not winnable
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if not piece:
            continue
        if static_exchange_eval(board, sq, not piece.color) > 0:
            continue
        targets = _valuable_targets_from(board, sq, piece.color)
        # count only targets that are winnable or the king
        real = [t for t in targets
                if board.piece_at(t).piece_type == chess.KING
                or static_exchange_eval(board, t, piece.color) > 0]
        if len(real) >= 2:
            tl = ", ".join(f"{board.piece_at(t).symbol().upper()}{sqn(t)}" for t in real)
            out.append(Concept("Fork / Double attack", "tactical",
                               f"{piece.symbol().upper()}{sqn(sq)} forks {tl}",
                               cname(piece.color), [sqn(sq)] + [sqn(t) for t in real]))
    # available to side to move
    mover = board.turn
    for mv in board.legal_moves:
        board.push(mv)
        landed = mv.to_square
        piece = board.piece_at(landed)
        winnable = static_exchange_eval(board, landed, not mover) == 0
        targets = _valuable_targets_from(board, landed, mover) if piece else []
        real = [t for t in targets
                if board.piece_at(t).piece_type == chess.KING
                or static_exchange_eval(board, t, mover) > 0]
        board.pop()
        if winnable and len(real) >= 2:
            out.append(Concept("Fork / Double attack", "tactical",
                               f"{mv.uci()} would fork two targets (available for {cname(mover)})",
                               cname(mover), [sqn(landed)]))
            break
    return out


def detect_pins(board: chess.Board) -> List[Concept]:
    out: List[Concept] = []
    for color in (WHITE, BLACK):
        king_sq = board.king(color)
        for sq in chess.SQUARES:
            p = board.piece_at(sq)
            if not p or p.color != color or p.piece_type == chess.KING:
                continue
            if not board.is_pinned(color, sq):
                continue
            # find the pinning enemy slider and what's behind
            pinner = None
            for a in board.attackers(not color, sq):
                ap = board.piece_at(a)
                if ap and ap.piece_type in (chess.BISHOP, chess.ROOK, chess.QUEEN):
                    if king_sq is not None and (chess.between(a, king_sq) & chess.BB_SQUARES[sq]):
                        pinner = a
                        break
            # absolute pin = pinned against the king (python-chess is_pinned is vs king)
            out.append(Concept("Pin (absolute)", "tactical",
                               f"{p.symbol().upper()}{sqn(sq)} is pinned to the king"
                               + (f" by {board.piece_at(pinner).symbol().upper()}{sqn(pinner)}" if pinner else ""),
                               cname(color), [sqn(sq)]))
    # relative pin: slider attacks enemy piece with a MORE valuable enemy piece behind (not king)
    for color in (WHITE, BLACK):
        for s in pieces_of(board, color, SLIDING):
            for target in board.attacks(s):
                tp = board.piece_at(target)
                if not tp or tp.color == color:
                    continue
                ray = chess.ray(s, target)
                if not ray:
                    continue
                behind = _next_on_ray(board, s, target)
                if behind is None:
                    continue
                bp = board.piece_at(behind)
                if bp and bp.color != color and bp.piece_type != chess.KING and \
                        val(bp.piece_type) > val(tp.piece_type):
                    out.append(Concept("Pin (relative)", "tactical",
                                       f"{board.piece_at(s).symbol().upper()}{sqn(s)} pins "
                                       f"{tp.symbol().upper()}{sqn(target)} to the more valuable "
                                       f"{bp.symbol().upper()}{sqn(behind)}",
                                       cname(color), [sqn(s), sqn(target), sqn(behind)]))
    return out


def _next_on_ray(board: chess.Board, origin: int, through: int) -> Optional[int]:
    """First occupied square strictly beyond ``through`` on the ray from origin."""
    ray = chess.ray(origin, through)
    if not ray:
        return None
    fo, ro = chess.square_file(origin), chess.square_rank(origin)
    ft, rt = chess.square_file(through), chess.square_rank(through)
    df = (ft > fo) - (ft < fo)
    dr = (rt > ro) - (rt < ro)
    f, r = ft + df, rt + dr
    while 0 <= f <= 7 and 0 <= r <= 7:
        s = chess.square(f, r)
        if board.piece_at(s):
            return s
        f += df
        r += dr
    return None


def detect_skewers(board: chess.Board) -> List[Concept]:
    out: List[Concept] = []
    for color in (WHITE, BLACK):
        for s in pieces_of(board, color, SLIDING):
            for target in board.attacks(s):
                tp = board.piece_at(target)
                if not tp or tp.color == color:
                    continue
                behind = _next_on_ray(board, s, target)
                if behind is None:
                    continue
                bp = board.piece_at(behind)
                if bp and bp.color != color and val(tp.piece_type) > val(bp.piece_type) \
                        and val(tp.piece_type) >= 5:
                    out.append(Concept("Skewer", "tactical",
                                       f"{board.piece_at(s).symbol().upper()}{sqn(s)} skewers "
                                       f"{tp.symbol().upper()}{sqn(target)}; when it moves, "
                                       f"{bp.symbol().upper()}{sqn(behind)} behind it falls",
                                       cname(color), [sqn(s), sqn(target), sqn(behind)]))
    return out


def detect_discovered_and_double_check(board: chess.Board) -> List[Concept]:
    """Discovered attack / discovered check / double check available to side to move."""
    out: List[Concept] = []
    mover = board.turn
    # baseline: enemy valuable pieces attacked by our sliders now
    def slider_attacks(b):
        m = {}
        for s in pieces_of(b, mover, SLIDING):
            for t in b.attacks(s):
                tp = b.piece_at(t)
                if tp and tp.color != mover:
                    m.setdefault(t, set()).add(s)
        return m
    before = slider_attacks(board)
    seen = set()
    for mv in board.legal_moves:
        moved_from = mv.from_square
        piece = board.piece_at(moved_from)
        if piece and piece.piece_type in SLIDING:
            pass  # a slider moving can still discover another slider; keep simple
        board.push(mv)
        gives_check = board.is_check()
        if gives_check:
            checkers = board.attackers(mover, board.king(not mover))
            if len(checkers) >= 2 and "double" not in seen:
                out.append(Concept("Double check", "tactical",
                                   f"{mv.uci()} gives double check — the king must move",
                                   cname(mover)))
                seen.add("double")
            elif mv.to_square not in checkers and "disccheck" not in seen:
                out.append(Concept("Discovered check", "tactical",
                                   f"{mv.uci()} uncovers a check from another piece",
                                   cname(mover)))
                seen.add("disccheck")
        # discovered (non-check) attack: a slider (not the moved piece) newly hits a valuable target
        after = slider_attacks(board)
        if "disc" not in seen:
            for t, srcs in after.items():
                tp = board.piece_at(t)
                if not tp or tp.piece_type == chess.KING:
                    continue
                new_srcs = srcs - before.get(t, set()) - {mv.to_square}
                if new_srcs and val(tp.piece_type) >= 3 and static_exchange_eval(board, t, mover) > 0:
                    out.append(Concept("Discovered attack", "tactical",
                                       f"{mv.uci()} unveils an attack on "
                                       f"{tp.symbol().upper()}{sqn(t)}", cname(mover)))
                    seen.add("disc")
                    break
        board.pop()
        if {"double", "disccheck", "disc"} <= seen:
            break
    return out


def detect_hanging(board: chess.Board) -> List[Concept]:
    out: List[Concept] = []
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if not p or p.piece_type == chess.KING:
            continue
        gain = static_exchange_eval(board, sq, not p.color)
        if gain > 0:
            att = _describe(board, board.attackers(not p.color, sq))
            def_ = _describe(board, board.attackers(p.color, sq))
            out.append(Concept("Hanging piece", "tactical",
                               f"{p.symbol().upper()}{sqn(sq)} is hanging (attackers {att}; "
                               f"defenders {def_}; opponent wins ~{gain})",
                               cname(p.color), [sqn(sq)]))
    return out


def detect_battery(board: chess.Board) -> List[Concept]:
    out: List[Concept] = []
    for color in (WHITE, BLACK):
        sliders = pieces_of(board, color, SLIDING)
        for i in range(len(sliders)):
            for j in range(i + 1, len(sliders)):
                a, b = sliders[i], sliders[j]
                ray = chess.ray(a, b)
                if not ray:
                    continue
                between = chess.SquareSet(chess.between(a, b))
                if any(board.piece_at(s) for s in between):
                    continue
                pa, pb = board.piece_at(a), board.piece_at(b)
                # both must be able to travel along this line
                if _aligns(pa.piece_type, a, b) and _aligns(pb.piece_type, a, b):
                    out.append(Concept("Battery", "tactical",
                                       f"{pa.symbol().upper()}{sqn(a)} + {pb.symbol().upper()}{sqn(b)} "
                                       f"form a battery on the same line",
                                       cname(color), [sqn(a), sqn(b)]))
    return out


def _aligns(piece_type: int, a: int, b: int) -> bool:
    fa, ra = chess.square_file(a), chess.square_rank(a)
    fb, rb = chess.square_file(b), chess.square_rank(b)
    same_line_straight = fa == fb or ra == rb
    same_diag = abs(fa - fb) == abs(ra - rb)
    if piece_type == chess.QUEEN:
        return same_line_straight or same_diag
    if piece_type == chess.ROOK:
        return same_line_straight
    if piece_type == chess.BISHOP:
        return same_diag
    return False


def detect_xray(board: chess.Board) -> List[Concept]:
    out: List[Concept] = []
    for color in (WHITE, BLACK):
        for s in pieces_of(board, color, SLIDING):
            for target in board.attacks(s):
                tp = board.piece_at(target)
                if not tp:
                    continue
                behind = _next_on_ray(board, s, target)
                if behind is None:
                    continue
                bp = board.piece_at(behind)
                if bp and tp.color != color and bp.color != color \
                        and (bp.piece_type == chess.KING or val(bp.piece_type) >= 5):
                    out.append(Concept("X-ray", "tactical",
                                       f"{board.piece_at(s).symbol().upper()}{sqn(s)} x-rays "
                                       f"{bp.symbol().upper()}{sqn(behind)} through "
                                       f"{tp.symbol().upper()}{sqn(target)}",
                                       cname(color), [sqn(s), sqn(target), sqn(behind)]))
    return out


def detect_overloaded_and_defender(board: chess.Board) -> List[Concept]:
    """Overloaded defender + removal-of-the-defender / deflection opportunity."""
    out: List[Concept] = []
    for color in (WHITE, BLACK):
        for sq in chess.SQUARES:
            p = board.piece_at(sq)
            if not p or p.color != color or p.piece_type == chess.KING:
                continue
            # what critical friendly pieces does this piece defend?
            defended = []
            for t in board.attacks(sq):
                tp = board.piece_at(t)
                if tp and tp.color == color and tp.piece_type != chess.KING:
                    # would t hang if this defender were gone?
                    b2 = board.copy(stack=False)
                    b2.remove_piece_at(sq)
                    if static_exchange_eval(b2, t, not color) > 0 and val(tp.piece_type) >= 3:
                        defended.append(t)
            if len(defended) >= 2:
                tl = ", ".join(f"{board.piece_at(t).symbol().upper()}{sqn(t)}" for t in defended)
                out.append(Concept("Overloading", "tactical",
                                   f"{p.symbol().upper()}{sqn(sq)} is overloaded — it is the key "
                                   f"defender of {tl}; deflect or remove it",
                                   cname(color), [sqn(sq)]))
    return out


def detect_removal_of_defender(board: chess.Board) -> List[Concept]:
    """Side to move can capture a piece that is the sole defender of another,
    winning material (removal of the guard)."""
    out: List[Concept] = []
    mover = board.turn
    for mv in board.legal_moves:
        if not board.is_capture(mv):
            continue
        captured_sq = mv.to_square
        cap = board.piece_at(captured_sq)
        if cap is None:  # en passant
            continue
        # what did the captured piece defend?
        defends = [t for t in board.attacks(captured_sq)
                   if board.piece_at(t) and board.piece_at(t).color != mover
                   and board.piece_at(t).piece_type != chess.KING]
        board.push(mv)
        for t in defends:
            tp = board.piece_at(t)
            if tp and static_exchange_eval(board, t, mover) > 0 and val(tp.piece_type) >= 3:
                out.append(Concept("Removal of the defender", "tactical",
                                   f"{mv.uci()} removes the guard of "
                                   f"{tp.symbol().upper()}{sqn(t)}", cname(mover)))
                board.pop()
                return out
        board.pop()
    return out


def detect_deflection(board: chess.Board) -> List[Concept]:
    """Overloaded enemy piece implies a deflection theme for the side to move."""
    out: List[Concept] = []
    overloaded = [c for c in detect_overloaded_and_defender(board)
                  if c.side != cname(board.turn)]
    for c in overloaded[:1]:
        out.append(Concept("Deflection", "tactical",
                           f"the enemy {c.detail.split(' is overloaded')[0]} is overloaded; "
                           f"a deflection can win material", cname(board.turn)))
    return out


def detect_trapped_piece(board: chess.Board) -> List[Concept]:
    out: List[Concept] = []
    for color in (WHITE, BLACK):
        for sq in chess.SQUARES:
            p = board.piece_at(sq)
            if not p or p.color != color or p.piece_type in (chess.PAWN, chess.KING):
                continue
            if val(p.piece_type) < 3:
                continue
            attacked = board.attackers(not color, sq)
            if not attacked:
                continue
            if static_exchange_eval(board, sq, not color) <= 0:
                continue
            # does it have any safe square? (only matters if it's the mover's piece)
            if color != board.turn:
                continue
            safe = False
            for mv in board.legal_moves:
                if mv.from_square != sq:
                    continue
                board.push(mv)
                landed = mv.to_square
                bad = static_exchange_eval(board, landed, not color) > 0
                board.pop()
                if not bad:
                    safe = True
                    break
            if not safe:
                out.append(Concept("Trapped piece", "tactical",
                                   f"{p.symbol().upper()}{sqn(sq)} is attacked and has no safe "
                                   f"retreat — it is trapped", cname(color), [sqn(sq)]))
    return out


def detect_undermining(board: chess.Board) -> List[Concept]:
    """A pawn that is the base/support of an enemy structure or defends a key piece
    can be undermined (captured/attacked)."""
    out: List[Concept] = []
    mover = board.turn
    for mv in board.legal_moves:
        if not board.is_capture(mv):
            continue
        cap = board.piece_at(mv.to_square)
        if cap is None or cap.piece_type != chess.PAWN:
            continue
        # was this pawn defending a piece or key pawn?
        defends = [t for t in board.attacks(mv.to_square)
                   if board.piece_at(t) and board.piece_at(t).color != mover]
        if defends:
            board.push(mv)
            wins = any(static_exchange_eval(board, t, mover) > 0 for t in defends
                       if board.piece_at(t))
            board.pop()
            if wins:
                out.append(Concept("Undermining", "tactical",
                                   f"{mv.uci()} undermines a supporting pawn", cname(mover)))
                return out
    return out


MATE_SCORE = 30000
_TACTIC_CACHE: Dict = {}
_NODE_BUDGET = 22000  # caps the shallow tactic search (~1s) for interactive use


def _material_pov(board: chess.Board, pov: chess.Color) -> int:
    s = 0
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p and p.piece_type != chess.KING:
            s += val(p.piece_type) if p.color == pov else -val(p.piece_type)
    return s


class _Budget:
    __slots__ = ("n",)

    def __init__(self):
        self.n = 0


def _qsearch(board: chess.Board, alpha: int, beta: int, budget: _Budget) -> int:
    """Capture/promotion quiescence, side-to-move perspective."""
    budget.n += 1
    stand = _material_pov(board, board.turn)
    if budget.n > _NODE_BUDGET:
        return stand
    if stand >= beta:
        return stand
    if stand > alpha:
        alpha = stand
    for mv in board.legal_moves:
        if not (board.is_capture(mv) or mv.promotion):
            continue
        board.push(mv)
        score = -_qsearch(board, -beta, -alpha, budget)
        board.pop()
        if score >= beta:
            return score
        if score > alpha:
            alpha = score
    return alpha


def _negamax(board: chess.Board, depth: int, alpha: int, beta: int, budget: _Budget) -> int:
    budget.n += 1
    if board.is_checkmate():
        return -MATE_SCORE
    if board.is_stalemate() or board.is_insufficient_material():
        return 0
    if depth == 0 or budget.n > _NODE_BUDGET:
        return _qsearch(board, alpha, beta, budget)
    best = -MATE_SCORE - 1
    for mv in board.legal_moves:
        board.push(mv)
        score = -_negamax(board, depth - 1, -beta, -alpha, budget)
        board.pop()
        if score > best:
            best = score
        if score > alpha:
            alpha = score
        if alpha >= beta:
            break
    return best


def shallow_tactic(board: chess.Board, depth: int = 3):
    """Correct bounded negamax + quiescence. Returns (first_move, gain_pawns, is_mate)
    when the side to move can win >= 2 pawns or force mate, else None."""
    if board.is_game_over():
        return None
    mover = board.turn
    base = _material_pov(board, mover)
    budget = _Budget()
    best_move, best = None, -MATE_SCORE - 1
    for mv in board.legal_moves:
        board.push(mv)
        score = -_negamax(board, depth - 1, -MATE_SCORE - 1, MATE_SCORE + 1, budget)
        board.pop()
        if score > best:
            best, best_move = score, mv
        if budget.n > _NODE_BUDGET:
            break
    if best_move is None:
        return None
    is_mate = best >= MATE_SCORE - 100
    gain = best - base
    if is_mate or gain >= 2:
        return (best_move, gain, is_mate)
    return None


def shallow_tactic_cached(board: chess.Board):
    key = (board._transposition_key() if hasattr(board, "_transposition_key") else board.fen())
    if key in _TACTIC_CACHE:
        return _TACTIC_CACHE[key]
    res = shallow_tactic(board)
    if len(_TACTIC_CACHE) > 512:
        _TACTIC_CACHE.clear()
    _TACTIC_CACHE[key] = res
    return res


def detect_combination(board: chess.Board) -> List[Concept]:
    res = shallow_tactic_cached(board)
    if res:
        mv, gain, is_mate = res
        if is_mate:
            return [Concept("Combination", "tactical",
                            f"a forcing sequence starting {mv.uci()} leads to mate", cname(board.turn))]
        return [Concept("Combination", "tactical",
                        f"a forcing combination starting {mv.uci()} wins ~{gain} material",
                        cname(board.turn))]
    return []


def detect_zwischenzug(board: chess.Board) -> List[Concept]:
    """A recapture is available, but a stronger in-between check/capture wins more."""
    if not board.move_stack:
        return []
    last = board.peek()
    mover = board.turn
    recaptures = [m for m in board.legal_moves
                  if m.to_square == last.to_square and board.is_capture(m)]
    if not recaptures:
        return []
    res = shallow_tactic_cached(board)
    if res:
        mv, gain, is_mate = res
        if mv not in recaptures and (board.gives_check(mv) or board.is_capture(mv)):
            return [Concept("Zwischenzug (in-between move)", "tactical",
                            f"instead of recapturing, the in-between {mv.uci()} wins more",
                            cname(mover))]
    return []


def detect_desperado(board: chess.Board) -> List[Concept]:
    """A piece of the side to move is attacked/doomed but can grab material before dying."""
    mover = board.turn
    doomed = [sq for sq in chess.SQUARES
              if board.piece_at(sq) and board.piece_at(sq).color == mover
              and board.piece_at(sq).piece_type not in (chess.PAWN, chess.KING)
              and static_exchange_eval(board, sq, not mover) > 0]
    for sq in doomed:
        for mv in board.legal_moves:
            if mv.from_square == sq and board.is_capture(mv):
                cap = board.piece_at(mv.to_square)
                if cap and val(cap.piece_type) >= 3:
                    return [Concept("Desperado", "tactical",
                                    f"the doomed {board.piece_at(sq).symbol().upper()}{sqn(sq)} can "
                                    f"grab material with {mv.uci()} before it is lost", cname(mover))]
    return []


def detect_interference(board: chess.Board) -> List[Concept]:
    """A winning quiet move lands on the line between an enemy slider and the piece it
    defends, cutting the defense (verified by the shallow tactic)."""
    res = shallow_tactic_cached(board)
    if not res:
        return []
    mv, gain, is_mate = res
    if board.is_capture(mv) or board.gives_check(mv):
        return []
    to = mv.to_square
    mover = board.turn
    for S in pieces_of(board, not mover, SLIDING):
        for T in board.attacks(S):
            tp = board.piece_at(T)
            if tp and tp.color == (not mover) and tp.piece_type != chess.KING:
                if to in chess.SquareSet(chess.between(S, T)):
                    return [Concept("Interference / Obstruction", "tactical",
                                    f"{mv.uci()} interferes with the defense of "
                                    f"{tp.symbol().upper()}{sqn(T)}", cname(mover))]
    return []


def detect_decoy(board: chess.Board) -> List[Concept]:
    """The winning tactic starts with a sacrificial check that lures the enemy king/
    piece to a bad square (attraction)."""
    res = shallow_tactic_cached(board)
    if not res:
        return []
    mv, gain, is_mate = res
    mover = board.turn
    board.push(mv)
    gives_check = board.is_check()
    landed = mv.to_square
    sac = static_exchange_eval(board, landed, not mover) > 0
    board.pop()
    if gives_check and sac:
        return [Concept("Decoy / Attraction", "tactical",
                        f"the forcing sacrifice {mv.uci()} lures the enemy into a losing follow-up",
                        cname(mover))]
    return []


def detect_clearance(board: chess.Board) -> List[Concept]:
    """The winning tactic's first move vacates a square that was blocking a friendly
    slider's line to an enemy target (clearance)."""
    res = shallow_tactic_cached(board)
    if not res:
        return []
    mv, gain, is_mate = res
    mover = board.turn
    frm = mv.from_square
    for S in pieces_of(board, mover, SLIDING):
        if S == frm:
            continue
        if frm in board.attacks(S):  # S sees frm with nothing between
            beyond = _next_on_ray(board, S, frm)
            if beyond is not None:
                bp = board.piece_at(beyond)
                if bp and bp.color != mover:
                    return [Concept("Clearance sacrifice", "tactical",
                                    f"{mv.uci()} clears the line of "
                                    f"{board.piece_at(S).symbol().upper()}{sqn(S)} toward "
                                    f"{bp.symbol().upper()}{sqn(beyond)}", cname(mover))]
    return []


def detect_counterattack(board: chess.Board) -> List[Concept]:
    """The side to move has a hanging piece but, instead of defending, has a bigger
    threat of its own (verified winning tactic that isn't just saving that piece)."""
    mover = board.turn
    my_hanging = [sq for sq in chess.SQUARES
                  if board.piece_at(sq) and board.piece_at(sq).color == mover
                  and board.piece_at(sq).piece_type != chess.KING
                  and static_exchange_eval(board, sq, not mover) > 0]
    if not my_hanging:
        return []
    res = shallow_tactic_cached(board)
    if res:
        mv, gain, is_mate = res
        if mv.from_square not in my_hanging or board.gives_check(mv):
            return [Concept("Counterattack", "tactical",
                            f"rather than defend, {mv.uci()} creates a bigger threat",
                            cname(mover))]
    return []


def detect_perpetual(board: chess.Board) -> List[Concept]:
    """Perpetual check as a *drawing resource*: only flagged when the side to move is
    materially worse (so a draw by repeated checks is the goal), and every reply to a
    check allows another check."""
    mover = board.turn
    diff = _material_pov(board, mover)
    if diff >= -2:  # not clearly worse -> perpetual isn't the point
        return []
    for mv in board.legal_moves:
        if not board.gives_check(mv):
            continue
        board.push(mv)
        replies = list(board.legal_moves)
        if replies and all(_has_check(board_after_reply(board, r), mover) for r in replies):
            board.pop()
            return [Concept("Perpetual check", "tactical",
                            f"{mv.uci()} starts a stream of checks the opponent can't escape "
                            f"(draw by repetition)", cname(mover))]
        board.pop()
    return []


def board_after_reply(board: chess.Board, reply: chess.Move) -> chess.Board:
    b = board.copy(stack=False)
    b.push(reply)
    return b


def _has_check(board: chess.Board, mover: chess.Color) -> bool:
    if board.turn != mover:
        return False
    return any(board.gives_check(m) for m in board.legal_moves)


def detect_greek_gift(board: chess.Board) -> List[Concept]:
    """Classic Bxh7+/Bxh2+ bishop sacrifice pattern availability."""
    mover = board.turn
    target = chess.H7 if mover == WHITE else chess.H2
    for mv in board.legal_moves:
        if mv.to_square != target:
            continue
        pc = board.piece_at(mv.from_square)
        if pc and pc.piece_type == chess.BISHOP and board.is_capture(mv) and board.gives_check(mv):
            return [Concept("Greek gift sacrifice (Bxh7+)", "tactical",
                            f"{mv.uci()} is the classic Greek-gift bishop sac to open the king",
                            cname(mover))]
    return []


def detect_windmill(board: chess.Board) -> List[Concept]:
    """Heuristic: a discovered-check battery (rook + bishop) aimed near the enemy king
    with a capturable enemy piece adjacent — the setup for a windmill."""
    mover = board.turn
    disc = [c for c in detect_discovered_and_double_check(board) if c.name == "Discovered check"]
    if disc:
        return [Concept("Windmill", "tactical",
                        "a discovered-check battery is set up; repeated discovered checks "
                        "(windmill) may harvest material", cname(mover))]
    return []


# ===========================================================================
# 2. CHECKMATE PATTERNS
# ===========================================================================
def _mate_in_one_moves(board: chess.Board):
    for mv in board.legal_moves:
        board.push(mv)
        mate = board.is_checkmate()
        board.pop()
        if mate:
            yield mv


def detect_checkmate_patterns(board: chess.Board) -> List[Concept]:
    out: List[Concept] = []
    mover = board.turn
    for mv in _mate_in_one_moves(board):
        piece = board.piece_at(mv.from_square)
        board.push(mv)
        ksq = board.king(not mover)
        name = _classify_mate(board, mv, piece, ksq, mover)
        board.pop()
        out.append(Concept(name, "checkmate",
                           f"{mv.uci()} is mate ({name})", cname(mover), [sqn(mv.to_square)]))
    # also flag generic "mate in 1 available" via combination if none classified above
    return out


def _classify_mate(board: chess.Board, mv, piece, ksq, mover) -> str:
    """Classify the delivered mate; board is AFTER the mating move (opponent to move)."""
    pt = board.piece_at(mv.to_square).piece_type if board.piece_at(mv.to_square) else None
    kf, kr = chess.square_file(ksq), chess.square_rank(ksq)
    back = 7 if mover == WHITE else 0  # enemy back rank
    # squares around king
    ring = [chess.square(kf + df, kr + dr)
            for df in (-1, 0, 1) for dr in (-1, 0, 1)
            if (df or dr) and 0 <= kf + df <= 7 and 0 <= kr + dr <= 7]
    own_blockers = sum(1 for s in ring
                       if board.piece_at(s) and board.piece_at(s).color == (not mover))
    # Smothered: knight mate, king fully surrounded by own pieces
    if pt == chess.KNIGHT and own_blockers == len(ring):
        return "Smothered mate"
    # Back-rank: rook/queen on the enemy back rank, king boxed by own pawns
    if pt in (chess.ROOK, chess.QUEEN) and kr == back:
        pawns_front = [s for s in ring if board.piece_at(s)
                       and board.piece_at(s).color == (not mover)
                       and board.piece_at(s).piece_type == chess.PAWN]
        if pawns_front:
            return "Back-rank mate"
    # Boden: two bishops criss-crossing
    bishops = pieces_of(board, mover, {chess.BISHOP})
    if pt == chess.BISHOP and len(bishops) >= 2:
        return "Boden's mate"
    # Arabian: knight + rook, king in corner (check before Anastasia)
    if pt == chess.ROOK and ksq in (chess.A1, chess.A8, chess.H1, chess.H8) \
            and pieces_of(board, mover, {chess.KNIGHT}):
        return "Arabian mate"
    # Anastasia: knight + rook on the h-file/edge
    if pt == chess.ROOK and kf in (0, 7) and pieces_of(board, mover, {chess.KNIGHT}):
        return "Anastasia's mate"
    # Epaulette: queen mate, king flanked by own rooks on the sides
    if pt == chess.QUEEN:
        sides = []
        for df in (-1, 1):
            s = chess.square(kf + df, kr) if 0 <= kf + df <= 7 else None
            if s and board.piece_at(s) and board.piece_at(s).color == (not mover) \
                    and board.piece_at(s).piece_type == chess.ROOK:
                sides.append(s)
        if len(sides) == 2:
            return "Epaulette mate"
        # Swallow's tail / dovetail: king escape blocked by own pieces diagonally
        if own_blockers >= 2:
            return "Swallow's tail (Guéridon) mate"
    return "Checkmate"


def detect_named_mate_shortcuts(board: chess.Board) -> List[Concept]:
    """Opening-specific mate names that depend on move context (Scholar's/Fool's/
    Legal's/Damiano's/Hook/Ladder/Cozio). Detected by signature where feasible."""
    out: List[Concept] = []
    # These are recognized primarily from their setups; provide availability when a
    # mate-in-1 matches the piece signature.
    mover = board.turn
    for mv in _mate_in_one_moves(board):
        board.push(mv)
        ksq = board.king(not mover)
        # Scholar's mate: queen mates on f7/f2 supported by a bishop, early game
        if mv.to_square in (chess.F7, chess.F2) and board.piece_at(mv.to_square) \
                and board.piece_at(mv.to_square).piece_type == chess.QUEEN \
                and board.fullmove_number <= 6:
            out.append(Concept("Scholar's mate", "checkmate",
                               f"{mv.uci()} is Scholar's mate (Qxf7#)", cname(mover)))
        board.pop()
    return out


# ===========================================================================
# 3. PIECE ACTIVITY & COORDINATION
# ===========================================================================
def detect_development(board: chess.Board) -> List[Concept]:
    out: List[Concept] = []
    for color in (WHITE, BLACK):
        back = 0 if color == WHITE else 7
        minors = [chess.square(f, back) for f in (1, 2, 5, 6)]
        undeveloped = sum(1 for s in minors
                          if board.piece_at(s) and board.piece_at(s).color == color
                          and board.piece_at(s).piece_type in (chess.KNIGHT, chess.BISHOP))
        developed = 4 - undeveloped
        out.append(Concept("Development", "activity",
                           f"has {developed}/4 minor pieces developed", cname(color)))
    return out


def detect_outpost(board: chess.Board) -> List[Concept]:
    out: List[Concept] = []
    pawns = pawn_map(board)
    for color in (WHITE, BLACK):
        for sq in chess.SQUARES:
            f, r = chess.square_file(sq), chess.square_rank(sq)
            if f < 2 or f > 5:
                continue
            if color == WHITE and r < 3:
                continue
            if color == BLACK and r > 4:
                continue
            if not any(board.piece_at(d) and board.piece_at(d).piece_type == chess.PAWN
                       and board.piece_at(d).color == color for d in board.attackers(color, sq)):
                continue
            challengeable = False
            for ef in (f - 1, f + 1):
                if 0 <= ef <= 7:
                    for er in pawns[not color].get(ef, []):
                        if (color == WHITE and er > r) or (color == BLACK and er < r):
                            challengeable = True
            if not challengeable:
                occ = board.piece_at(sq)
                extra = ""
                if occ and occ.color == color and occ.piece_type == chess.KNIGHT:
                    extra = " (knight already installed)"
                out.append(Concept("Outpost", "activity",
                                   f"{sqn(sq)} is a protected outpost{extra}", cname(color), [sqn(sq)]))
    return out


def detect_bishops(board: chess.Board) -> List[Concept]:
    out: List[Concept] = []
    pawns = pawn_map(board)
    bishop_colors_present = {WHITE: set(), BLACK: set()}
    for color in (WHITE, BLACK):
        bishops = pieces_of(board, color, {chess.BISHOP})
        colors = {(chess.square_file(s) + chess.square_rank(s)) % 2 for s in bishops}
        bishop_colors_present[color] = colors
        if len(bishops) >= 2 and len(colors) == 2:
            out.append(Concept("Bishop pair", "activity",
                               "has the bishop pair (strong in open positions)", cname(color)))
        # good vs bad bishop: bishop hemmed by own pawns on its color
        home_bishops = {chess.C1, chess.F1, chess.C8, chess.F8}
        for s in bishops:
            bc = (chess.square_file(s) + chess.square_rank(s)) % 2
            own_pawns_same = 0
            for f, ranks in pawns[color].items():
                for r in ranks:
                    if (f + r) % 2 == bc:
                        own_pawns_same += 1
            if own_pawns_same >= 4 and s not in home_bishops and len(board.attacks(s)) <= 4:
                out.append(Concept("Good bishop vs bad bishop", "activity",
                                   f"the bishop on {sqn(s)} is 'bad' — hemmed by {own_pawns_same} "
                                   f"own pawns on its color", cname(color), [sqn(s)]))
    # opposite-colored bishops (one each, different colors)
    wb, bb = bishop_colors_present[WHITE], bishop_colors_present[BLACK]
    if len(pieces_of(board, WHITE, {chess.BISHOP})) == 1 and \
       len(pieces_of(board, BLACK, {chess.BISHOP})) == 1 and wb and bb and wb != bb:
        out.append(Concept("Opposite-colored bishops", "activity",
                           "each side has one bishop on opposite colors (drawish endgames, "
                           "sharp middlegame attacks)"))
    return out


def detect_knight_vs_bishop(board: chess.Board) -> List[Concept]:
    pawns = pawn_map(board)
    total_pawns = sum(len(r) for r in pawns[WHITE].values()) + sum(len(r) for r in pawns[BLACK].values())
    locked = _locked_pawns(board)
    wn = len(pieces_of(board, WHITE, {chess.KNIGHT})); wb = len(pieces_of(board, WHITE, {chess.BISHOP}))
    bn = len(pieces_of(board, BLACK, {chess.KNIGHT})); bb = len(pieces_of(board, BLACK, {chess.BISHOP}))
    if (wn and bb and not wb and not bn) or (bn and wb and not bb and not wn):
        closed = locked >= 3 or total_pawns >= 12
        note = ("closed/blocked — favours the knight" if closed
                else "open — favours the bishop")
        return [Concept("Knight vs bishop", "activity",
                        f"minor-piece imbalance (knight vs bishop); pawn structure is {note}")]
    return []


def _locked_pawns(board: chess.Board) -> int:
    n = 0
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p and p.piece_type == chess.PAWN:
            fwd = sq + 8 if p.color == WHITE else sq - 8
            if 0 <= fwd < 64:
                q = board.piece_at(fwd)
                if q and q.piece_type == chess.PAWN and q.color != p.color:
                    n += 1
    return n


def detect_rook_seventh(board: chess.Board) -> List[Concept]:
    out: List[Concept] = []
    for color in (WHITE, BLACK):
        seventh = 6 if color == WHITE else 1
        for s in pieces_of(board, color, {chess.ROOK}):
            if chess.square_rank(s) == seventh:
                out.append(Concept("Rook on the 7th rank", "activity",
                                   f"rook on {sqn(s)} occupies the 7th rank (attacks pawns & king)",
                                   cname(color), [sqn(s)]))
    return out


def detect_doubled_rooks(board: chess.Board) -> List[Concept]:
    out: List[Concept] = []
    for color in (WHITE, BLACK):
        rooks = pieces_of(board, color, {chess.ROOK})
        for i in range(len(rooks)):
            for j in range(i + 1, len(rooks)):
                a, b = rooks[i], rooks[j]
                if chess.square_file(a) == chess.square_file(b) and \
                        not any(board.piece_at(s) for s in chess.SquareSet(chess.between(a, b))):
                    out.append(Concept("Doubled rooks", "activity",
                                       f"rooks doubled on the {chess.FILE_NAMES[chess.square_file(a)]}-file",
                                       cname(color), [sqn(a), sqn(b)]))
    return out


def detect_fianchetto(board: chess.Board) -> List[Concept]:
    out: List[Concept] = []
    spots = {WHITE: {chess.G2: "g2", chess.B2: "b2"}, BLACK: {chess.G7: "g7", chess.B7: "b7"}}
    for color in (WHITE, BLACK):
        for s, name in spots[color].items():
            p = board.piece_at(s)
            if p and p.color == color and p.piece_type == chess.BISHOP:
                out.append(Concept("Fianchetto", "activity",
                                   f"fianchettoed bishop on {name} (controls the long diagonal)",
                                   cname(color), [name]))
    return out


def detect_long_diagonal(board: chess.Board) -> List[Concept]:
    out: List[Concept] = []
    long_diags = {
        (0, 0): "a1-h8", (7, 7): "a1-h8",
        (0, 7): "a8-h1", (7, 0): "a8-h1",
    }
    for color in (WHITE, BLACK):
        for s in pieces_of(board, color, {chess.BISHOP, chess.QUEEN}):
            reach = board.attacks(s)
            # controls a long diagonal if it sees a corner-to-corner diagonal span
            for corner, dname in (((0, 0), "a1-h8"), ((0, 7), "a8-h1")):
                diag = _diagonal_squares(corner)
                if s in diag and len([d for d in diag if d in reach or d == s]) >= 5:
                    out.append(Concept("Long-diagonal control", "activity",
                                       f"{board.piece_at(s).symbol().upper()}{sqn(s)} dominates the "
                                       f"{dname} long diagonal", cname(color), [sqn(s)]))
                    break
    return out


def _diagonal_squares(corner) -> List[int]:
    f, r = corner
    df = 1 if f == 0 else -1
    dr = 1 if r == 0 else -1
    out = []
    while 0 <= f <= 7 and 0 <= r <= 7:
        out.append(chess.square(f, r))
        f += df
        r += dr
    return out


def detect_overprotection(board: chess.Board) -> List[Concept]:
    """A key square/pawn defended by 3+ friendly pieces (Nimzowitsch overprotection)."""
    out: List[Concept] = []
    for color in (WHITE, BLACK):
        for sq in chess.SQUARES:
            p = board.piece_at(sq)
            if p and p.color == color and p.piece_type == chess.PAWN:
                r = chess.square_rank(sq)
                advanced = (color == WHITE and r >= 3) or (color == BLACK and r <= 4)
                if not advanced:
                    continue
                defenders = board.attackers(color, sq)
                if len(defenders) >= 3:
                    out.append(Concept("Overprotection", "activity",
                                       f"the {sqn(sq)} point is overprotected by {len(defenders)} "
                                       f"pieces (Nimzowitsch)", cname(color), [sqn(sq)]))
    return out


def detect_piece_activity(board: chess.Board) -> List[Concept]:
    """Coarse mobility comparison + coordination + worst-piece pointer."""
    out: List[Concept] = []
    mob = {}
    for color in (WHITE, BLACK):
        b = board.copy(stack=False)
        b.turn = color
        mob[color] = b.legal_moves.count()
    out.append(Concept("Piece activity / mobility", "activity",
                       f"legal-move mobility White {mob[WHITE]} vs Black {mob[BLACK]}"))
    # worst piece = least mobile developed minor/rook of side to move
    color = board.turn
    worst = None
    worst_mob = 99
    for s in pieces_of(board, color, {chess.KNIGHT, chess.BISHOP, chess.ROOK}):
        m = len(list(board.attacks(s)))
        if m < worst_mob:
            worst_mob, worst = m, s
    if worst is not None and worst_mob <= 2 and board.fullmove_number >= 10:
        out.append(Concept("Improving the worst piece", "activity",
                           f"{board.piece_at(worst).symbol().upper()}{sqn(worst)} is the least active "
                           f"piece ({worst_mob} squares) — improve it", cname(color), [sqn(worst)]))
    return out


def detect_coordination(board: chess.Board) -> List[Concept]:
    """Heuristic: several friendly pieces attacking squares in the enemy king zone."""
    out: List[Concept] = []
    for color in (WHITE, BLACK):
        ksq = board.king(not color)
        if ksq is None:
            continue
        zone = [ksq] + [chess.square(chess.square_file(ksq) + df, chess.square_rank(ksq) + dr)
                        for df in (-1, 0, 1) for dr in (-1, 0, 1)
                        if 0 <= chess.square_file(ksq) + df <= 7 and 0 <= chess.square_rank(ksq) + dr <= 7]
        attackers = set()
        for z in zone:
            for a in board.attackers(color, z):
                ap = board.piece_at(a)
                if ap and ap.piece_type not in (chess.PAWN, chess.KING):
                    attackers.add(a)
        if len(attackers) >= 3:
            out.append(Concept("Coordination / harmony", "activity",
                               f"{len(attackers)} pieces coordinate against the enemy king zone",
                               cname(color)))
    return out


# ===========================================================================
# 4. PAWN STRUCTURE
# ===========================================================================
def detect_pawn_structure(board: chess.Board) -> List[Concept]:
    out: List[Concept] = []
    pawns = pawn_map(board)
    for color in (WHITE, BLACK):
        own = pawns[color]
        enemy = pawns[not color]
        cn = cname(color)
        # doubled / tripled
        for f, ranks in own.items():
            if len(ranks) >= 3:
                out.append(Concept("Tripled pawns", "pawn",
                                   f"tripled pawns on the {chess.FILE_NAMES[f]}-file", cn))
            elif len(ranks) == 2:
                out.append(Concept("Doubled pawns", "pawn",
                                   f"doubled pawns on the {chess.FILE_NAMES[f]}-file", cn))
        # isolated
        iso = [chess.FILE_NAMES[f] for f in own if (f - 1) not in own and (f + 1) not in own]
        if iso:
            out.append(Concept("Isolated pawn / IQP", "pawn",
                               f"isolated pawn(s) on file(s) {', '.join(sorted(iso))}", cn))
        # backward
        for f, ranks in own.items():
            for r in ranks:
                if _is_backward(color, f, r, own, enemy, board):
                    out.append(Concept("Backward pawn", "pawn",
                                       f"backward pawn on {sqn(chess.square(f, r))}", cn,
                                       [sqn(chess.square(f, r))]))
        # passed + variants
        passed = _passed_pawns(color, own, enemy)
        for pf, pr in passed:
            psq = chess.square(pf, pr)
            detail = f"passed pawn on {sqn(psq)}"
            # protected passed
            if any(board.piece_at(d) and board.piece_at(d).piece_type == chess.PAWN
                   and board.piece_at(d).color == color for d in board.attackers(color, psq)):
                out.append(Concept("Protected passed pawn", "pawn",
                                   f"protected {detail}", cn, [sqn(psq)]))
            out.append(Concept("Passed pawn", "pawn", detail, cn, [sqn(psq)]))
        # connected passed
        pfiles = sorted(set(pf for pf, _ in passed))
        for i in range(len(pfiles) - 1):
            if pfiles[i + 1] - pfiles[i] == 1:
                out.append(Concept("Connected passed pawns", "pawn",
                                   f"connected passed pawns on the {chess.FILE_NAMES[pfiles[i]]}/"
                                   f"{chess.FILE_NAMES[pfiles[i+1]]} files", cn))
        # outside passed pawn (passed pawn far from the other pawns' center of mass)
        if passed:
            all_files = list(own.keys()) + list(enemy.keys())
            if all_files:
                cx = sum(all_files) / len(all_files)
                for pf, pr in passed:
                    if abs(pf - cx) >= 3:
                        out.append(Concept("Outside passed pawn", "pawn",
                                           f"outside passed pawn on {sqn(chess.square(pf, pr))} "
                                           f"(decoy the enemy king)", cn, [sqn(chess.square(pf, pr))]))
                        break
        # pawn islands
        islands = _count_islands(own)
        if islands >= 3:
            out.append(Concept("Pawn island", "pawn",
                               f"{islands} pawn islands (fragmented structure)", cn))
        # hanging pawns (adjacent c/d duo, no own pawns on flanking files, half-open in front)
        if _has_hanging_pawns(color, own, enemy):
            out.append(Concept("Hanging pawns", "pawn",
                               "hanging pawns (mobile c+d duo with no pawn support)", cn))
        # phalanx (two+ pawns side by side on same rank)
        for r in range(8):
            advanced = (color == WHITE and r >= 3) or (color == BLACK and r <= 4)
            if not advanced:
                continue
            files_on_r = sorted(f for f, ranks in own.items() if r in ranks)
            run = 1
            for i in range(1, len(files_on_r)):
                if files_on_r[i] - files_on_r[i - 1] == 1:
                    run += 1
                    if run == 2:
                        out.append(Concept("Phalanx", "pawn",
                                           f"pawn phalanx on rank {r+1}", cn))
                        break
                else:
                    run = 1
        # candidate passed pawn (majority on a wing)
        cand = _candidate_passed(color, own, enemy)
        if cand is not None:
            out.append(Concept("Candidate passed pawn", "pawn",
                               f"candidate passed pawn on the {chess.FILE_NAMES[cand]}-file "
                               f"(from a pawn majority)", cn))
    # pawn chains
    chains = _detect_chains(board)
    out.extend(chains)
    # pawn majority / minority (per wing)
    out.extend(_detect_majorities(pawns))
    # pawn tension
    out.extend(_detect_tension(board))
    # pawn break / lever
    out.extend(_detect_levers(board))
    # weak square / hole + color complex
    out.extend(_detect_holes(board, pawns))
    return out


def _is_backward(color, f, r, own, enemy, board) -> bool:
    # no friendly pawn on adjacent files that is behind-or-level to support advance,
    # and the square in front is controlled by an enemy pawn
    for af in (f - 1, f + 1):
        if af in own:
            for ar in own[af]:
                if (color == WHITE and ar <= r) or (color == BLACK and ar >= r):
                    return False
    front = chess.square(f, r + (1 if color == WHITE else -1))
    if not (0 <= chess.square_rank(front) <= 7):
        return False
    for af in (f - 1, f + 1):
        if 0 <= af <= 7 and af in enemy:
            for er in enemy[af]:
                # enemy pawn that attacks the stop square
                if chess.square_rank(front) == er + (-1 if color == WHITE else 1):
                    return True
    return False


def _passed_pawns(color, own, enemy):
    out = []
    for f, ranks in own.items():
        for r in ranks:
            blocked = False
            for ef in (f - 1, f, f + 1):
                for er in enemy.get(ef, []):
                    if (color == WHITE and er > r) or (color == BLACK and er < r):
                        blocked = True
            if not blocked:
                out.append((f, r))
    return out


def _count_islands(own) -> int:
    files = sorted(own.keys())
    if not files:
        return 0
    islands = 1
    for i in range(1, len(files)):
        if files[i] - files[i - 1] > 1:
            islands += 1
    return islands


def _has_hanging_pawns(color, own, enemy) -> bool:
    # c+d pawns present, b and e files empty of own pawns, and c/d are half-open (no enemy pawn ahead on file)
    if 2 in own and 3 in own and 1 not in own and 4 not in own:
        return True
    return False


def _candidate_passed(color, own, enemy):
    for wing_files in ([0, 1, 2, 3], [4, 5, 6, 7]):
        own_count = sum(len(own.get(f, [])) for f in wing_files)
        enemy_count = sum(len(enemy.get(f, [])) for f in wing_files)
        if own_count > enemy_count and own_count >= 2:
            for f in wing_files:
                if f in own and f not in enemy:
                    return f
    return None


def _detect_chains(board: chess.Board) -> List[Concept]:
    out = []
    for color in (WHITE, BLACK):
        chain = 0
        for sq in chess.SQUARES:
            p = board.piece_at(sq)
            if p and p.piece_type == chess.PAWN and p.color == color:
                for d in board.attackers(color, sq):
                    dp = board.piece_at(d)
                    if dp and dp.piece_type == chess.PAWN:
                        chain += 1
                        break
        if chain >= 1:
            out.append(Concept("Pawn chain", "pawn",
                               f"pawn chain(s) present ({chain} linked pawn(s)) — attack the base",
                               cname(color)))
    return out


def _detect_majorities(pawns) -> List[Concept]:
    out = []
    q_w = sum(len(pawns[WHITE].get(f, [])) for f in (0, 1, 2, 3))
    q_b = sum(len(pawns[BLACK].get(f, [])) for f in (0, 1, 2, 3))
    k_w = sum(len(pawns[WHITE].get(f, [])) for f in (5, 6, 7))
    k_b = sum(len(pawns[BLACK].get(f, [])) for f in (5, 6, 7))
    if q_w > q_b:
        out.append(Concept("Pawn majority / minority", "pawn",
                           f"White has a queenside pawn majority ({q_w}v{q_b})", "White"))
    elif q_b > q_w:
        out.append(Concept("Pawn majority / minority", "pawn",
                           f"Black has a queenside pawn majority ({q_b}v{q_w})", "Black"))
    if k_w > k_b:
        out.append(Concept("Pawn majority / minority", "pawn",
                           f"White has a kingside pawn majority ({k_w}v{k_b})", "White"))
    elif k_b > k_w:
        out.append(Concept("Pawn majority / minority", "pawn",
                           f"Black has a kingside pawn majority ({k_b}v{k_w})", "Black"))
    return out


def _detect_tension(board: chess.Board) -> List[Concept]:
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p and p.piece_type == chess.PAWN:
            for cap in board.attacks(sq):
                cp = board.piece_at(cap)
                if cp and cp.piece_type == chess.PAWN and cp.color != p.color:
                    return [Concept("Pawn tension", "pawn",
                                    f"pawn tension between {sqn(sq)} and {sqn(cap)} — decide when "
                                    f"to release it")]
    return []


def _detect_levers(board: chess.Board) -> List[Concept]:
    """A pawn push available that would attack an enemy pawn (a break/lever)."""
    mover = board.turn
    for mv in board.legal_moves:
        p = board.piece_at(mv.from_square)
        if not p or p.piece_type != chess.PAWN or board.is_capture(mv):
            continue
        to = mv.to_square
        for af in (chess.square_file(to) - 1, chess.square_file(to) + 1):
            if 0 <= af <= 7:
                tsq = chess.square(af, chess.square_rank(to))
                tp = board.piece_at(tsq)
                if tp and tp.piece_type == chess.PAWN and tp.color != mover:
                    return [Concept("Pawn break / lever", "pawn",
                                    f"{mv.uci()} is a pawn break challenging the enemy structure",
                                    cname(mover))]
    return []


def _detect_holes(board: chess.Board, pawns) -> List[Concept]:
    out = []
    for color in (WHITE, BLACK):
        cn = cname(color)
        # color complex
        light = dark = 0
        for f, ranks in pawns[color].items():
            for r in ranks:
                if (f + r) % 2 == 1:
                    light += 1
                else:
                    dark += 1
        if light + dark >= 5 and abs(light - dark) >= 3:
            weak = "dark" if light > dark else "light"
            out.append(Concept("Color complex weakness", "pawn",
                               f"pawns cluster on one color; the {weak} squares are weak in "
                               f"{cn}'s camp", cn))
        # weak square / hole in own half not defendable by a pawn
        home_ranks = range(2, 5) if color == WHITE else range(3, 6)
        for f in range(2, 6):
            for r in home_ranks:
                sq = chess.square(f, r)
                # can any own pawn ever defend sq?
                defendable = False
                for af in (f - 1, f + 1):
                    if af in pawns[color]:
                        for pr in pawns[color][af]:
                            if (color == WHITE and pr < r) or (color == BLACK and pr > r):
                                defendable = True
                if not defendable and not board.piece_at(sq):
                    enemy_controls = any(board.piece_at(a) and board.piece_at(a).piece_type == chess.PAWN
                                         for a in board.attackers(not color, sq))
                    if enemy_controls:
                        out.append(Concept("Weak square / hole", "pawn",
                                           f"{sqn(sq)} is a hole in {cn}'s position (no pawn can "
                                           f"defend it)", cn, [sqn(sq)]))
                        break
    return out


def detect_pawn_storm(board: chess.Board) -> List[Concept]:
    out = []
    pawns = pawn_map(board)
    for color in (WHITE, BLACK):
        ksq = board.king(not color)
        if ksq is None:
            continue
        side_files = (5, 6, 7) if chess.square_file(ksq) >= 4 else (0, 1, 2)
        advanced = 0
        for f in side_files:
            for r in pawns[color].get(f, []):
                adv = r if color == WHITE else 7 - r
                if adv >= 3:
                    advanced += 1
        if advanced >= 2:
            out.append(Concept("Pawn storm", "pawn",
                               f"{cname(color)} has advanced {advanced} pawns toward the enemy king",
                               cname(color)))
    return out


# ===========================================================================
# 5. KING SAFETY
# ===========================================================================
def detect_king_safety(board: chess.Board) -> List[Concept]:
    out: List[Concept] = []
    pawns = pawn_map(board)
    kings = {c: board.king(c) for c in (WHITE, BLACK)}
    for color in (WHITE, BLACK):
        cn = cname(color)
        ksq = kings[color]
        if ksq is None:
            continue
        kf, kr = chess.square_file(ksq), chess.square_rank(ksq)
        home = 0 if color == WHITE else 7
        if kr == home and kf >= 6:
            out.append(Concept("Castling (short/long)", "king",
                               "king is castled kingside", cn))
        elif kr == home and kf <= 2:
            out.append(Concept("Castling (short/long)", "king",
                               "king is castled queenside", cn))
        elif kr == home and 3 <= kf <= 5 and board.fullmove_number >= 6:
            out.append(Concept("Exposed / uncastled king", "king",
                               "king is still in the center (uncastled) — get it to safety", cn))
        # pawn shield
        if kr == home and (kf >= 6 or kf <= 2):
            shield_rank = 1 if color == WHITE else 6
            files = (5, 6, 7) if kf >= 6 else (0, 1, 2)
            missing = [chess.FILE_NAMES[f] for f in files if shield_rank not in pawns[color].get(f, [])]
            if missing:
                out.append(Concept("Pawn shield", "king",
                                   f"pawn shield missing on {', '.join(missing)}", cn))
            else:
                out.append(Concept("Pawn shield", "king", "intact pawn shield", cn))
        # luft
        if kr in (0, 7):
            front = chess.square(kf, kr + (1 if color == WHITE else -1))
            if board.piece_at(front) and board.piece_at(front).piece_type == chess.PAWN:
                # is there any luft (an advanced flank pawn giving an escape)?
                has_luft = False
                for af in (kf - 1, kf + 1):
                    if 0 <= af <= 7:
                        fsq = chess.square(af, kr + (1 if color == WHITE else -1))
                        if not board.piece_at(fsq):
                            has_luft = True
                if not has_luft:
                    out.append(Concept("Luft (escape square)", "king",
                                       "king has no luft — vulnerable to back-rank ideas", cn))
        # open lines toward the king (open/half-open file on the king)
        if kf not in pawns[color]:
            out.append(Concept("Open lines toward the king", "king",
                               f"{cname(not color)} can target the {chess.FILE_NAMES[kf]}-file "
                               f"toward {cn}'s king", cname(not color)))
    # opposite-side castling
    wk, bk = kings[WHITE], kings[BLACK]
    if wk is not None and bk is not None:
        wside = "k" if chess.square_file(wk) >= 5 else ("q" if chess.square_file(wk) <= 2 else None)
        bside = "k" if chess.square_file(bk) >= 5 else ("q" if chess.square_file(bk) <= 2 else None)
        if wside and bside and wside != bside:
            out.append(Concept("Opposite-side castling", "king",
                               "kings castled on opposite wings — expect pawn-storm races"))
    # weakened kingside: castled king whose g-pawn has left home (hole in front)
    for color in (WHITE, BLACK):
        ksq = kings[color]
        if ksq is None:
            continue
        home = 0 if color == WHITE else 7
        if chess.square_file(ksq) >= 5 and chess.square_rank(ksq) == home:
            grank = 1 if color == WHITE else 6
            if grank not in pawns[color].get(6, []):
                out.append(Concept("Weakened kingside", "king",
                                   "the g-pawn has advanced/left home — the king's cover is "
                                   "weakened (dark-square holes)", cname(color)))
    return out


# ===========================================================================
# 6. ENDGAME CONCEPTS
# ===========================================================================
def _is_endgame(board: chess.Board) -> bool:
    heavy = len(pieces_of(board, WHITE, {chess.QUEEN, chess.ROOK})) + \
            len(pieces_of(board, BLACK, {chess.QUEEN, chess.ROOK}))
    minors = len(pieces_of(board, WHITE, {chess.KNIGHT, chess.BISHOP})) + \
             len(pieces_of(board, BLACK, {chess.KNIGHT, chess.BISHOP}))
    queens = len(pieces_of(board, WHITE, {chess.QUEEN})) + len(pieces_of(board, BLACK, {chess.QUEEN}))
    return queens == 0 and (heavy + minors) <= 6


def detect_endgame(board: chess.Board) -> List[Concept]:
    out: List[Concept] = []
    wk, bk = board.king(WHITE), board.king(BLACK)
    pawns = pawn_map(board)
    total_pieces = chess.popcount(board.occupied)

    # King & pawn endgames: opposition, key squares, rule of the square, king centralization
    only_kp = all(board.piece_at(s) is None or board.piece_at(s).piece_type in (chess.KING, chess.PAWN)
                  for s in chess.SQUARES)

    if wk is not None and bk is not None and only_kp:
        # Opposition: kings on same file/rank with one square between, opponent to move
        df = abs(chess.square_file(wk) - chess.square_file(bk))
        dr = abs(chess.square_rank(wk) - chess.square_rank(bk))
        if (df == 0 and dr == 2) or (dr == 0 and df == 2):
            holder = "the side NOT to move" 
            out.append(Concept("Opposition (direct/distant/diagonal)", "endgame",
                               f"kings are in direct opposition — {holder} holds the opposition"))
        elif (df == 0 and dr in (4, 6)) or (dr == 0 and df in (4, 6)):
            out.append(Concept("Opposition (direct/distant/diagonal)", "endgame",
                               "kings on the same line with an odd number of squares between — "
                               "distant opposition matters here"))
        # rule of the square for passed pawns
        for color in (WHITE, BLACK):
            for f, ranks in pawns[color].items():
                for r in ranks:
                    if _passed_pawns(color, pawns[color], pawns[not color]):
                        promo_rank = 7 if color == WHITE else 0
                        dist_pawn = abs(promo_rank - r)
                        ek = bk if color == WHITE else wk
                        # square rule: enemy king in the square of the pawn?
                        king_dist = max(abs(chess.square_file(ek) - f), abs(chess.square_rank(ek) - promo_rank))
                        inside = king_dist <= dist_pawn + (1 if board.turn == color else 0)
                        out.append(Concept("Rule of the square", "endgame",
                                           f"{sqn(chess.square(f, r))} pawn: enemy king is "
                                           f"{'inside' if inside else 'OUTSIDE'} the square "
                                           f"({'catches' if inside else 'cannot catch'} it)",
                                           cname(color)))
                        break
                else:
                    continue
                break
        # king centralization
        for color, k in ((WHITE, wk), (BLACK, bk)):
            cf, cr = chess.square_file(k), chess.square_rank(k)
            centrality = min(cf, 7 - cf) + min(cr, 7 - cr)
            if centrality >= 4:
                out.append(Concept("King centralization", "endgame",
                                   f"{cname(color)}'s king is well centralized", cname(color)))
        # zugzwang (only-kp, side to move has only waiting/king moves and worsens)
        if _looks_like_zugzwang(board):
            out.append(Concept("Zugzwang", "endgame",
                               "the side to move is in (or near) zugzwang — any move worsens the position"))
            out.append(Concept("Triangulation", "endgame",
                               "a king triangulation could pass the move to the opponent (lose a tempo)"))
        # key squares (for a single passed pawn)
        out.extend(_key_squares(board, pawns))
        # corresponding squares (blocked KP endgame heuristic)
        if _locked_pawns(board) >= 2:
            out.append(Concept("Corresponding / related squares", "endgame",
                               "blocked pawn endgame — corresponding-square analysis governs the "
                               "king maneuvering"))
        # pawn breakthrough
        out.extend(_detect_breakthrough(board, pawns))
        # shouldering
        if wk is not None and bk is not None and chess.square_distance(wk, bk) <= 2:
            out.append(Concept("Shouldering / body-check", "endgame",
                               "kings are close — use shouldering to block the enemy king's path"))

    # Rook endgames: Lucena, Philidor, Vancura, rook behind passer, back-rank defense
    out.extend(_detect_rook_endgames(board, pawns))

    # Wrong bishop + rook pawn draw
    out.extend(_detect_wrong_bishop(board, pawns))

    # Fortress (very heuristic): material-down side but fully blockaded
    # Outside passed pawn in endgame already covered in pawn structure.
    if total_pieces <= 6 and _is_endgame(board):
        out.append(Concept("King activity (endgame)", "endgame",
                           "few pieces remain — king activity is a decisive factor"))

    # Outside passed pawn (endgame use)
    if _is_endgame(board) or only_kp:
        for color in (WHITE, BLACK):
            passed = _passed_pawns(color, pawns[color], pawns[not color])
            if not passed:
                continue
            all_files = list(pawns[color].keys()) + list(pawns[not color].keys())
            if not all_files:
                continue
            cx = sum(all_files) / len(all_files)
            for pf, pr in passed:
                if abs(pf - cx) >= 3:
                    out.append(Concept("Outside passed pawn (endgame use)", "endgame",
                                       f"outside passed pawn on {sqn(chess.square(pf, pr))} — "
                                       f"decoy the enemy king, then win on the other wing",
                                       cname(color)))
                    break
    return out


def _looks_like_zugzwang(board: chess.Board) -> bool:
    # KP endgame: if the side to move only has king moves (pawns blocked) and every king
    # move loses the opposition / a pawn. Cheap heuristic: no pawn moves available and
    # king is tied to defense.
    mover = board.turn
    pawn_moves = [m for m in board.legal_moves if board.piece_at(m.from_square)
                  and board.piece_at(m.from_square).piece_type == chess.PAWN]
    king_moves = [m for m in board.legal_moves if board.piece_at(m.from_square)
                  and board.piece_at(m.from_square).piece_type == chess.KING]
    return len(pawn_moves) == 0 and len(king_moves) == len(list(board.legal_moves)) and len(king_moves) <= 5


def _key_squares(board, pawns) -> List[Concept]:
    # single passed pawn: name the key squares (rank ahead) informally
    out = []
    for color in (WHITE, BLACK):
        passed = _passed_pawns(color, pawns[color], pawns[not color])
        if len(passed) == 1 and sum(len(r) for r in pawns[color].values()) == 1:
            f, r = passed[0]
            ahead = r + (1 if color == WHITE else -1)
            if 0 <= ahead <= 7:
                out.append(Concept("Key squares", "endgame",
                                   f"the key squares in front of the {sqn(chess.square(f, r))} pawn "
                                   f"decide the K+P ending", cname(color)))
    return out


def _detect_breakthrough(board, pawns) -> List[Concept]:
    # 3 vs 2 (or similar) pawns facing on a wing with no pieces => breakthrough may exist
    for color in (WHITE, BLACK):
        for wing in ([0, 1, 2], [5, 6, 7]):
            own = sum(len(pawns[color].get(f, [])) for f in wing)
            enemy = sum(len(pawns[not color].get(f, [])) for f in wing)
            if own >= 3 and own > enemy and enemy >= 1:
                return [Concept("Pawn breakthrough", "endgame",
                                f"{cname(color)} has a pawn-majority breakthrough available on the "
                                f"{'queenside' if wing[0] == 0 else 'kingside'}", cname(color))]
    return []


def _detect_rook_endgames(board, pawns) -> List[Concept]:
    out = []
    wr = pieces_of(board, WHITE, {chess.ROOK})
    br = pieces_of(board, BLACK, {chess.ROOK})
    # must be a rook endgame (rooks + pawns + kings only)
    non_rook_pieces = pieces_of(board, WHITE, {chess.QUEEN, chess.BISHOP, chess.KNIGHT}) + \
                      pieces_of(board, BLACK, {chess.QUEEN, chess.BISHOP, chess.KNIGHT})
    if non_rook_pieces or (len(wr) + len(br) == 0):
        return out
    total_pawns = sum(len(r) for r in pawns[WHITE].values()) + sum(len(r) for r in pawns[BLACK].values())
    # Rook behind the passed pawn (Tarrasch)
    for color in (WHITE, BLACK):
        passed = _passed_pawns(color, pawns[color], pawns[not color])
        rooks = wr if color == WHITE else br
        for pf, pr in passed:
            for rk in rooks + (br if color == WHITE else wr):
                if chess.square_file(rk) == pf:
                    behind = (chess.square_rank(rk) < pr) if color == WHITE else (chess.square_rank(rk) > pr)
                    if behind:
                        out.append(Concept("Rook behind the passed pawn", "endgame",
                                           f"a rook stands behind the {sqn(chess.square(pf, pr))} "
                                           f"passer (Tarrasch rule)"))
    # Lucena vs Philidor (single pawn, R vs R)
    if len(wr) + len(br) == 2 and total_pawns == 1:
        # Lucena: pawn on 7th with own rook cutting; Philidor: 3rd-rank defense
        for color in (WHITE, BLACK):
            passed = _passed_pawns(color, pawns[color], pawns[not color])
            if len(passed) == 1:
                f, r = passed[0]
                adv = r if color == WHITE else 7 - r
                if adv >= 6:
                    out.append(Concept("Lucena position", "endgame",
                                       "advanced R+P vs R — winning via the Lucena 'bridge' technique"))
                elif adv <= 4:
                    out.append(Concept("Philidor position", "endgame",
                                       "R+P vs R — Philidor third-rank defense draws"))
                # Vancura: rook pawn defended from the side
                if f in (0, 7) and adv >= 4:
                    out.append(Concept("Vancura position", "endgame",
                                       "rook-pawn ending — the Vancura defense holds the draw"))
    # Fortress heuristic in rook endings not attempted here.
    return out


def _detect_wrong_bishop(board, pawns) -> List[Concept]:
    for color in (WHITE, BLACK):
        bishops = pieces_of(board, color, {chess.BISHOP})
        own_pawns = [(f, r) for f in pawns[color] for r in pawns[color][f]]
        other = pieces_of(board, color, {chess.QUEEN, chess.ROOK, chess.KNIGHT})
        enemy_pieces = pieces_of(board, not color, {chess.QUEEN, chess.ROOK, chess.KNIGHT, chess.BISHOP})
        if len(bishops) == 1 and len(own_pawns) == 1 and not other and not enemy_pieces:
            f, r = own_pawns[0]
            if f in (0, 7):  # rook pawn
                promo = chess.square(f, 7 if color == WHITE else 0)
                promo_color = (chess.square_file(promo) + chess.square_rank(promo)) % 2
                b = bishops[0]
                bcolor = (chess.square_file(b) + chess.square_rank(b)) % 2
                if bcolor != promo_color:
                    return [Concept("Wrong-colored bishop + rook pawn", "endgame",
                                    "wrong-colored bishop with a rook pawn — the promotion square "
                                    "is the wrong color; it's a draw", cname(color))]
    return []


def detect_fortress(board: chess.Board) -> List[Concept]:
    """Very heuristic: a materially worse side whose position is fully blocked."""
    # Only flag in endgames where pawns are largely locked and one side is down material.
    if not _is_endgame(board):
        return []
    locked = _locked_pawns(board)
    if locked >= 3:
        return [Concept("Fortress", "endgame",
                        "a locked pawn structure may form a fortress the stronger side cannot break")]
    return []


# ===========================================================================
# 7. OPENING PRINCIPLES  (board heuristics; move list optional)
# ===========================================================================
def detect_opening_principles(board: chess.Board, moves: Optional[List[str]] = None) -> List[Concept]:
    out: List[Concept] = []
    if board.fullmove_number > 15:
        return out
    for color in (WHITE, BLACK):
        cn = cname(color)
        back = 0 if color == WHITE else 7
        # center control (pawns/pieces bearing on d4/e4/d5/e5)
        center = [chess.D4, chess.E4, chess.D5, chess.E5]
        controlled = sum(1 for c in center if board.is_attacked_by(color, c)
                         or (board.piece_at(c) and board.piece_at(c).color == color))
        out.append(Concept("Control the center", "opening",
                           f"contests {controlled}/4 central squares", cn))
        # castle early
        ksq = board.king(color)
        if ksq is not None and chess.square_rank(ksq) == back and 3 <= chess.square_file(ksq) <= 5 \
                and board.fullmove_number >= 5:
            out.append(Concept("Castle early", "opening",
                               "king not yet castled by move 5+ — prioritize king safety", cn))
        # connect the rooks
        rooks = [s for s in (chess.square(f, back) for f in range(8))
                 if board.piece_at(s) and board.piece_at(s).color == color
                 and board.piece_at(s).piece_type == chess.ROOK]
        if len(rooks) == 2 and not any(board.piece_at(s) for s in chess.SquareSet(chess.between(rooks[0], rooks[1]))):
            out.append(Concept("Connect the rooks", "opening", "rooks are connected", cn))
        # queen out too early
        qsq = pieces_of(board, color, {chess.QUEEN})
        minors_home = sum(1 for f in (1, 2, 5, 6)
                          if board.piece_at(chess.square(f, back))
                          and board.piece_at(chess.square(f, back)).color == color
                          and board.piece_at(chess.square(f, back)).piece_type in (chess.KNIGHT, chess.BISHOP))
        qhome = chess.D1 if color == WHITE else chess.D8
        if qsq and qsq[0] != qhome and minors_home >= 3 and board.fullmove_number <= 6:
            out.append(Concept("Don't bring the queen out too early", "opening",
                               "queen developed while most minors are still home — a target for tempo", cn))
    # lead in development -> fight for the initiative
    dw, db = _dev_count(board, WHITE), _dev_count(board, BLACK)
    if dw != db and abs(dw - db) >= 1:
        lead = WHITE if dw > db else BLACK
        out.append(Concept("Fight for the initiative / lead in development", "opening",
                           f"{cname(lead)} leads in development — open the position and press the "
                           f"initiative before the opponent catches up", cname(lead)))
    # move-history-dependent principles
    if moves:
        out.extend(_opening_from_moves(board, moves))
    return out


def _opening_from_moves(board: chess.Board, moves: List[str]) -> List[Concept]:
    out = []
    b = chess.Board()
    piece_move_count: Dict[Tuple[chess.Color, int], int] = {}
    knight_dev = {WHITE: None, BLACK: None}
    bishop_dev = {WHITE: None, BLACK: None}
    order_flag = {WHITE: False, BLACK: False}
    try:
        for i, uci in enumerate(moves):
            mv = chess.Move.from_uci(uci) if len(uci) >= 4 and uci[1].isdigit() else b.parse_san(uci)
            color = b.turn
            piece = b.piece_at(mv.from_square)
            if piece:
                key = (color, mv.to_square)
                # track same-piece-twice in opening
                if piece.piece_type in (chess.KNIGHT, chess.BISHOP):
                    ply = i // 2 + 1
                    if piece.piece_type == chess.KNIGHT and knight_dev[color] is None:
                        knight_dev[color] = ply
                    if piece.piece_type == chess.BISHOP and bishop_dev[color] is None:
                        bishop_dev[color] = ply
            b.push(mv)
        for color in (WHITE, BLACK):
            if bishop_dev[color] is not None and (
                knight_dev[color] is None or bishop_dev[color] < knight_dev[color]
            ):
                out.append(Concept("Develop knights before bishops", "opening",
                                   "developed a bishop before both knights (mild inaccuracy)", cname(color)))
    except Exception:
        pass
    return out


def detect_gambit(board: chess.Board) -> List[Concept]:
    """Material-down early with a development lead => gambit/initiative for the pawn."""
    if board.fullmove_number > 15:
        return []
    wv = sum(val(board.piece_at(s).piece_type) for s in chess.SQUARES
             if board.piece_at(s) and board.piece_at(s).color == WHITE and board.piece_at(s).piece_type != chess.KING)
    bv = sum(val(board.piece_at(s).piece_type) for s in chess.SQUARES
             if board.piece_at(s) and board.piece_at(s).color == BLACK and board.piece_at(s).piece_type != chess.KING)
    diff = wv - bv
    if abs(diff) == 1:  # a pawn down
        down = WHITE if diff < 0 else BLACK
        dev_down = _dev_count(board, down)
        dev_up = _dev_count(board, not down)
        if dev_down > dev_up:
            return [Concept("Gambit", "opening",
                            f"{cname(down)} is a pawn down but ahead in development — a gambit for "
                            f"the initiative", cname(down))]
    return []


def _dev_count(board, color) -> int:
    back = 0 if color == WHITE else 7
    return sum(1 for f in (1, 2, 5, 6)
               if not (board.piece_at(chess.square(f, back))
                       and board.piece_at(chess.square(f, back)).color == color
                       and board.piece_at(chess.square(f, back)).piece_type in (chess.KNIGHT, chess.BISHOP)))


# ===========================================================================
# 8. NAMED PAWN STRUCTURES & OPENING SKELETONS
# ===========================================================================
import json as _json
import os as _os

_ECO_MAP = None


def _load_eco():
    global _ECO_MAP
    if _ECO_MAP is None:
        path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "eco.json")
        try:
            with open(path, encoding="utf-8") as fh:
                _ECO_MAP = _json.load(fh)
        except Exception:
            _ECO_MAP = {}
    return _ECO_MAP


def _eco_lookup(board: chess.Board):
    """Exact ECO code + name for a known book position, else None."""
    eco = _load_eco()
    if not eco:
        return None
    epd = board.epd()
    if epd in eco:
        return eco[epd]
    # relaxed: ignore a transient en-passant square that book nodes omit
    parts = epd.split(" ")
    if len(parts) >= 4 and parts[3] != "-":
        parts[3] = "-"
        relaxed = " ".join(parts)
        if relaxed in eco:
            return eco[relaxed]
    return None


def detect_opening(board: chess.Board) -> List[Concept]:
    """Identify the opening. Prefer an exact ECO book match (precise code + name),
    else fall back to a conservative structural family detector."""
    if board.fullmove_number > 40:
        return []
    eco = _eco_lookup(board)
    if eco:
        return [Concept("Opening", "openingid",
                        f"this is a known book position: {eco} (ECO)")]
    return _structural_opening(board)


def _structural_opening(board: chess.Board) -> List[Concept]:
    """Identify the opening family from the pawn skeleton + early piece placement.
    Conservative: only emits a name when the signature is clear (so the explainer
    never guesses). Recognises the common e4/d4/c4 systems and Sicilian variations.
    """
    if board.fullmove_number > 20:
        return []
    pawns = pawn_map(board)
    wp, bp = pawns[WHITE], pawns[BLACK]

    def W(sq):
        p = board.piece_at(sq)
        return p is not None and p.piece_type == chess.PAWN and p.color == WHITE

    def B(sq):
        p = board.piece_at(sq)
        return p is not None and p.piece_type == chess.PAWN and p.color == BLACK

    def piece(sq, pt, color):
        p = board.piece_at(sq)
        return p is not None and p.piece_type == pt and p.color == color

    name = None
    # --- 1.e4 openings ---
    if W(chess.E4):
        # Sicilian: e4 vs ...c5, with the classic d-for-c trade in the Open Sicilian
        open_sicilian = (3 not in wp) and (2 not in bp) and (B(chess.D7) or B(chess.D6) or B(chess.E6) or B(chess.E5))
        if B(chess.C5) and W(chess.E4):
            name = "Sicilian Defence"
        if open_sicilian:
            # sub-variations
            has_g6 = B(chess.G6) and piece(chess.G7, chess.BISHOP, BLACK)
            if has_g6 and piece(chess.C6, chess.KNIGHT, BLACK) and B(chess.D7) and not B(chess.D6):
                name = "Sicilian Defence, Accelerated Dragon"
            elif has_g6 and B(chess.D6):
                name = "Sicilian Defence, Dragon"
            elif B(chess.D6) and B(chess.A6):
                name = "Sicilian Defence, Najdorf"
            elif B(chess.D6) and B(chess.E6):
                name = "Sicilian Defence, Scheveningen"
            elif B(chess.E6) and B(chess.A6):
                name = "Sicilian Defence, Kan/Taimanov"
            else:
                name = "Sicilian Defence (Open)"
        elif B(chess.E5):
            if piece(chess.B5, chess.BISHOP, WHITE):
                name = "Ruy Lopez (Spanish)"
            elif piece(chess.C4, chess.BISHOP, WHITE):
                name = "Italian Game"
            elif 3 not in wp and 4 not in bp:  # d-for-e trade
                name = "Scotch / Open Game"
            else:
                name = "Open Game (1.e4 e5)"
        elif B(chess.E6) and B(chess.D5):
            name = "French Defence"
        elif B(chess.C6) and B(chess.D5):
            name = "Caro-Kann Defence"
        elif B(chess.D5) and not B(chess.C6) and not B(chess.E6) and 4 not in bp:
            name = "Scandinavian Defence"
        elif B(chess.D6) and B(chess.G6) and piece(chess.G7, chess.BISHOP, BLACK):
            name = "Pirc / Modern Defence"
    # --- 1.d4 openings ---
    elif W(chess.D4):
        if W(chess.C4):
            if B(chess.G6) and piece(chess.G7, chess.BISHOP, BLACK) and B(chess.D6):
                name = "King's Indian Defence"
            elif B(chess.G6) and piece(chess.G7, chess.BISHOP, BLACK) and B(chess.D5):
                name = "Grünfeld Defence"
            elif B(chess.C6) and B(chess.D5):
                name = "Slav Defence"
            elif B(chess.E6) and B(chess.D5):
                name = "Queen's Gambit Declined"
            elif B(chess.D5) and 2 not in bp:
                name = "Queen's Gambit"
            elif piece(chess.B4, chess.BISHOP, BLACK) or B(chess.E6):
                name = "Indian Defence (1.d4 Nf6)"
            else:
                name = "Queen's Pawn Game"
        else:
            name = "Queen's Pawn Game"
    # --- 1.c4 / flank ---
    elif W(chess.C4) and not W(chess.E4) and not W(chess.D4):
        name = "English Opening"
    elif piece(chess.F3, chess.KNIGHT, WHITE) and not W(chess.E4) and not W(chess.D4) and not W(chess.C4):
        name = "Réti / King's Indian Attack"

    if name:
        return [Concept("Opening", "openingid", f"this position arises from the {name}")]
    return []


def detect_named_structures(board: chess.Board) -> List[Concept]:
    out: List[Concept] = []
    pawns = pawn_map(board)
    wp, bp = pawns[WHITE], pawns[BLACK]

    def pa(sq, pt, color):
        p = board.piece_at(sq)
        return p is not None and p.piece_type == pt and p.color == color

    # IQP
    if 3 in wp.get(3, []) and 2 not in wp and 4 not in wp:
        out.append(Concept("Isolated Queen's Pawn (IQP)", "structure",
                           "White has an IQP on d4 — activity vs long-term weakness", "White"))
    if 4 in bp.get(3, []) and 2 not in bp and 4 not in bp:
        out.append(Concept("Isolated Queen's Pawn (IQP)", "structure",
                           "Black has an IQP on d5", "Black"))
    # Hanging pawns c+d
    for color, pw in ((WHITE, wp), (BLACK, bp)):
        if 2 in pw and 3 in pw and 1 not in pw and 4 not in pw:
            out.append(Concept("Hanging pawns (c+d)", "structure",
                               "c+d hanging pawns — mobile but can become targets", cname(color)))
    # Carlsbad (White c/d pawns, Black c6/d5 vs White d4, minority attack theme)
    if 3 in wp.get(3, []) and 4 not in wp.get(4, []) and 5 in bp.get(2, []) and 4 in bp.get(3, []):
        out.append(Concept("Carlsbad structure", "structure",
                           "Carlsbad structure — White's plan is the minority attack (b4-b5)"))
    # Maroczy Bind (White pawns c4+e4)
    if 3 in wp.get(2, []) and 3 in wp.get(4, []) and 3 not in wp.get(3, []):
        out.append(Concept("Maróczy Bind", "structure",
                           "Maróczy Bind (c4+e4) clamps down on ...d5", "White"))
    # Hedgehog
    if 5 in bp.get(0, []) and 5 in bp.get(1, []) and 5 in bp.get(3, []) and 5 in bp.get(4, []) and 2 not in bp:
        out.append(Concept("Hedgehog", "structure",
                           "Hedgehog (a6/b6/d6/e6) — uncoil with ...b5 or ...d5", "Black"))
    # Sicilian Dragon
    if pa(chess.G7, chess.BISHOP, BLACK) and 5 in bp.get(6, []) and 5 in bp.get(3, []) and 2 not in bp:
        out.append(Concept("Sicilian Dragon", "structure",
                           "Sicilian Dragon (...d6/...g6/...Bg7) — opposite-side castling races", "Black"))
    # KID
    if pa(chess.G7, chess.BISHOP, BLACK) and 5 in bp.get(6, []) and 5 in bp.get(3, []) and 4 in bp.get(4, []):
        out.append(Concept("King's Indian Defense", "structure",
                           "King's Indian (...d6/...e5/...g6/...Bg7) — ...f5 attack vs queenside play",
                           "Black"))
    # French closed
    if 3 in wp.get(3, []) and 4 in wp.get(4, []) and 4 in bp.get(3, []) and 5 in bp.get(4, []):
        out.append(Concept("French Defense (closed center)", "structure",
                           "French closed center (d4+e5 vs d5+e6) — bad French bishop, ...c5/...f6 breaks",
                           "Black"))
    # Caro/Slav skeleton
    if 5 in bp.get(2, []) and 4 in bp.get(3, []):
        out.append(Concept("Caro-Kann / Slav skeleton", "structure",
                           "Caro-Kann/Slav skeleton (...c6 + ...d5) — solid; ...c5 freeing break", "Black"))
    # Stonewall
    if 2 in wp.get(2, []) and 3 in wp.get(3, []) and 2 in wp.get(4, []) and 3 in wp.get(5, []):
        out.append(Concept("Stonewall", "structure",
                           "White Stonewall (c3/d4/e3/f4) — Ne5 outpost, e4 hole", "White"))
    # Scheveningen small center: black d6+e6, white e4, no locked center
    if 5 in bp.get(3, []) and 5 in bp.get(4, []) and 3 in wp.get(4, []) and 2 not in bp:
        out.append(Concept('Scheveningen "small center"', "structure",
                           "Scheveningen small center (...d6+...e6) — flexible Sicilian setup", "Black"))
    return out


# ===========================================================================
# 9. STRATEGIC PLANS & MANEUVERS
# ===========================================================================
def detect_strategic_plans(board: chess.Board) -> List[Concept]:
    out: List[Concept] = []
    pawns = pawn_map(board)
    # minority attack: side with fewer queenside pawns can push them to create a weakness
    for color in (WHITE, BLACK):
        own_q = sum(len(pawns[color].get(f, [])) for f in (0, 1, 2))
        enemy_q = sum(len(pawns[not color].get(f, [])) for f in (0, 1, 2))
        if 0 < own_q < enemy_q:
            out.append(Concept("Minority attack", "plan",
                               f"{cname(color)} has a queenside minority to advance (b4-b5) to create "
                               f"a weakness in the majority", cname(color)))
            break
    # blockade: a piece sitting in front of an enemy passed/isolated pawn
    for color in (WHITE, BLACK):
        for f, ranks in pawns[not color].items():
            for r in ranks:
                stop = chess.square(f, r + (1 if (not color) == WHITE else -1))
                if 0 <= chess.square_rank(stop) <= 7:
                    bp = board.piece_at(stop)
                    if bp and bp.color == color and bp.piece_type in (chess.KNIGHT, chess.BISHOP):
                        # only if that enemy pawn is passed or isolated
                        passed = (f, r) in _passed_pawns(not color, pawns[not color], pawns[color])
                        iso = (f - 1) not in pawns[not color] and (f + 1) not in pawns[not color]
                        if passed or iso:
                            out.append(Concept("Blockade", "plan",
                                               f"{bp.symbol().upper()}{sqn(stop)} blockades the enemy "
                                               f"{'passed' if passed else 'isolated'} pawn on {sqn(chess.square(f, r))}",
                                               cname(color), [sqn(stop)]))
    # space advantage / restriction
    out.extend(_detect_space(board, pawns))
    # exchange the right pieces (color complex pointer) is emitted via bishops/holes
    # rook to open/semi-open file
    out.extend(_detect_rook_files(board, pawns))
    # rerouting a knight: a knight on the rim
    for color in (WHITE, BLACK):
        for s in pieces_of(board, color, {chess.KNIGHT}):
            if chess.square_file(s) in (0, 7):
                out.append(Concept("Rerouting a knight", "plan",
                                   f"the knight on {sqn(s)} is on the rim — reroute it toward an outpost",
                                   cname(color), [sqn(s)]))
    # prophylaxis / two weaknesses / trade-when-ahead
    out.extend(_detect_prophylaxis(board))
    out.extend(_detect_two_weaknesses(board))
    out.extend(_detect_trade_advice(board))
    return out


def _detect_space(board, pawns) -> List[Concept]:
    out = []
    space = {WHITE: 0, BLACK: 0}
    for color in (WHITE, BLACK):
        for f, ranks in pawns[color].items():
            for r in ranks:
                adv = r if color == WHITE else 7 - r
                if adv >= 3:
                    space[color] += 1
    if space[WHITE] - space[BLACK] >= 2:
        out.append(Concept("Space advantage", "plan", "White has a space advantage", "White"))
        out.append(Concept("Restriction / cramping", "plan",
                           "White's space cramps Black — Black should seek exchanges", "White"))
    elif space[BLACK] - space[WHITE] >= 2:
        out.append(Concept("Space advantage", "plan", "Black has a space advantage", "Black"))
        out.append(Concept("Restriction / cramping", "plan",
                           "Black's space cramps White — White should seek exchanges", "Black"))
    return out


def _detect_rook_files(board, pawns) -> List[Concept]:
    out = []
    files_with_pawns = set(pawns[WHITE]) | set(pawns[BLACK])
    for color in (WHITE, BLACK):
        for s in pieces_of(board, color, {chess.ROOK}):
            f = chess.square_file(s)
            if f not in files_with_pawns:
                out.append(Concept("Rook to an open/semi-open file", "plan",
                                   f"rook on the open {chess.FILE_NAMES[f]}-file", cname(color), [sqn(s)]))
            elif f not in pawns[color] and f in pawns[not color]:
                out.append(Concept("Rook to an open/semi-open file", "plan",
                                   f"rook on the semi-open {chess.FILE_NAMES[f]}-file", cname(color), [sqn(s)]))
    return out


def _detect_prophylaxis(board: chess.Board) -> List[Concept]:
    """If the opponent (not the side to move) has a strong threat (a fork/hanging/mate
    available), prophylaxis is called for."""
    if board.is_check():
        return []
    if shallow_tactic_cached(board):  # we already have a winning tactic; no need for prophylaxis
        return []
    b = board.copy(stack=False)
    b.turn = not board.turn
    b.clear_stack()
    if not b.is_valid():
        return []
    res = shallow_tactic_cached(b)
    if res:
        return [Concept("Prophylaxis", "plan",
                        f"the opponent threatens {res[0].uci()} — prevent it prophylactically",
                        cname(board.turn))]
    return []


def _detect_two_weaknesses(board: chess.Board) -> List[Concept]:
    pawns = pawn_map(board)
    for color in (WHITE, BLACK):
        weaknesses = 0
        own = pawns[color]
        enemy = pawns[not color]
        # isolated/backward/doubled pawns count as weaknesses
        for f, ranks in own.items():
            if (f - 1) not in own and (f + 1) not in own:
                weaknesses += 1
            if len(ranks) >= 2:
                weaknesses += 1
        if weaknesses >= 2:
            return [Concept("Two weaknesses principle", "plan",
                            f"{cname(color)} has {weaknesses} pawn weaknesses — the attacker should "
                            f"probe both fronts", cname(not color))]
    return []


def _detect_trade_advice(board: chess.Board) -> List[Concept]:
    wv = sum(val(board.piece_at(s).piece_type) for s in chess.SQUARES
             if board.piece_at(s) and board.piece_at(s).color == WHITE and board.piece_at(s).piece_type != chess.KING)
    bv = sum(val(board.piece_at(s).piece_type) for s in chess.SQUARES
             if board.piece_at(s) and board.piece_at(s).color == BLACK and board.piece_at(s).piece_type != chess.KING)
    diff = wv - bv
    if abs(diff) >= 2:
        ahead = WHITE if diff > 0 else BLACK
        return [Concept("Trade when ahead / avoid trades when behind", "plan",
                        f"{cname(ahead)} is ahead in material — trade pieces (not pawns) to simplify; "
                        f"the other side should keep pieces on", cname(ahead))]
    return []


def detect_exchange_right_pieces(board: chess.Board) -> List[Concept]:
    """If one side has a clearly bad bishop or a strong enemy piece to neutralize."""
    concepts = detect_bishops(board)
    for c in concepts:
        if c.name == "Good bishop vs bad bishop":
            return [Concept("Exchange the right pieces", "plan",
                            f"{c.side} should try to trade off its bad bishop; the opponent should "
                            f"avoid that trade", c.side)]
    return []


# ===========================================================================
# 10. SACRIFICES & MATERIAL
# ===========================================================================
def detect_material_balance(board: chess.Board) -> List[Concept]:
    wv = bv = 0
    counts = {WHITE: {}, BLACK: {}}
    for s in chess.SQUARES:
        p = board.piece_at(s)
        if p and p.piece_type != chess.KING:
            if p.color == WHITE:
                wv += val(p.piece_type)
            else:
                bv += val(p.piece_type)
            counts[p.color][p.piece_type] = counts[p.color].get(p.piece_type, 0) + 1
    diff = wv - bv
    if diff == 0:
        detail = "material is equal"
        side = None
    else:
        side = "White" if diff > 0 else "Black"
        detail = f"{side} is up {abs(diff)} point(s) of material"
    out = [Concept("Material balance", "material", detail, side)]
    return out


def detect_sacrifice_available(board: chess.Board) -> List[Concept]:
    """A sound sacrifice for the side to move: give material now, win it back (or mate)
    within a short forcing sequence."""
    out: List[Concept] = []
    mover = board.turn
    res = shallow_tactic_cached(board)
    if not res:
        return out
    first, gain, is_mate = res
    if board.is_capture(first):
        return out
    board.push(first)
    landed = first.to_square
    sac = board.piece_at(landed) is not None and static_exchange_eval(board, landed, not mover) > 0
    board.pop()
    if sac:
        if is_mate:
            out.append(Concept("Piece sacrifice for attack", "material",
                               f"{first.uci()} sacrifices material for a forced mate", cname(mover)))
        else:
            out.append(Concept("Piece sacrifice for attack", "material",
                               f"{first.uci()} sacrifices material but wins it back with interest "
                               f"(~{gain})", cname(mover)))
        out.append(Concept("Sham vs real sacrifice", "material",
                           "this is a sham (temporary) sacrifice — the material returns by force",
                           cname(mover)))
    return out


def detect_exchange_sac(board: chess.Board) -> List[Concept]:
    """Side to move can give a rook for a minor piece for positional gain (heuristic:
    RxN/RxB where the recapture is by a pawn, damaging structure / winning a strong minor)."""
    mover = board.turn
    for mv in board.legal_moves:
        if not board.is_capture(mv):
            continue
        mover_piece = board.piece_at(mv.from_square)
        target = board.piece_at(mv.to_square)
        if mover_piece and mover_piece.piece_type == chess.ROOK and target \
                and target.piece_type in (chess.KNIGHT, chess.BISHOP):
            return [Concept("Exchange sacrifice", "material",
                            f"{mv.uci()} is an exchange sacrifice (rook for minor) for activity/structure",
                            cname(mover))]
    return []


def detect_positional_pawn_sac(board: chess.Board) -> List[Concept]:
    """A non-forcing pawn push/give that isn't immediately regained — flagged when the
    side to move has a pawn break that gives up a pawn for lasting pressure."""
    # Heuristic placeholder tied to levers + development lead; keep conservative.
    mover = board.turn
    lever = _detect_levers(board)
    if lever:
        wv = sum(val(board.piece_at(s).piece_type) for s in chess.SQUARES
                 if board.piece_at(s) and board.piece_at(s).color == WHITE and board.piece_at(s).piece_type != chess.KING)
        bv = sum(val(board.piece_at(s).piece_type) for s in chess.SQUARES
                 if board.piece_at(s) and board.piece_at(s).color == BLACK and board.piece_at(s).piece_type != chess.KING)
        if abs(wv - bv) <= 1 and _dev_count(board, mover) < _dev_count(board, not mover):
            return [Concept("Positional pawn sacrifice", "material",
                            "a pawn break here offers a positional pawn sacrifice for the initiative",
                            cname(mover))]
    return []


def detect_compensation(board: chess.Board) -> List[Concept]:
    wv = sum(val(board.piece_at(s).piece_type) for s in chess.SQUARES
             if board.piece_at(s) and board.piece_at(s).color == WHITE and board.piece_at(s).piece_type != chess.KING)
    bv = sum(val(board.piece_at(s).piece_type) for s in chess.SQUARES
             if board.piece_at(s) and board.piece_at(s).color == BLACK and board.piece_at(s).piece_type != chess.KING)
    diff = wv - bv
    if abs(diff) in (1, 2, 3):
        down = WHITE if diff < 0 else BLACK
        dev_down = _dev_count(board, down)
        dev_up = _dev_count(board, not down)
        enemy_king = board.king(not down)
        exposed = enemy_king is not None and 3 <= chess.square_file(enemy_king) <= 5 \
            and chess.square_rank(enemy_king) in (0, 7)
        if dev_down > dev_up or exposed:
            return [Concept("Compensation", "material",
                            f"{cname(down)} is down material but has compensation (development/king "
                            f"safety/activity)", cname(down))]
    return []


# ===========================================================================
# 11. EVALUATION / META CONCEPTS
# ===========================================================================
def detect_meta(board: chess.Board) -> List[Concept]:
    out: List[Concept] = []
    # tempo / initiative accounting
    mover = board.turn
    forcing = sum(1 for m in board.legal_moves if board.is_capture(m) or board.gives_check(m))
    if forcing >= 3 or board.is_check() is False and _dev_count(board, mover) < _dev_count(board, not mover):
        out.append(Concept("Initiative", "meta",
                           f"{cname(mover)} has the initiative (multiple forcing moves / lead in "
                           f"development)", cname(mover)))
    out.append(Concept("Tempo / initiative accounting", "meta",
                       f"{cname(mover)} to move with {forcing} forcing move(s) available", cname(mover)))
    # tempo (development lead)
    dw, db = _dev_count(board, WHITE), _dev_count(board, BLACK)
    if abs(dw - db) >= 2:
        lead = WHITE if dw > db else BLACK
        out.append(Concept("Tempo", "meta",
                           f"{cname(lead)} leads in development by ~{abs(dw-db)} tempi", cname(lead)))
    # imbalances (Silman)
    imb = _detect_imbalances(board)
    if imb:
        out.append(Concept("Imbalances", "meta", "; ".join(imb)))
    # dynamic vs static
    static_edges = []
    dynamic_edges = []
    pawns = pawn_map(board)
    for color in (WHITE, BLACK):
        own = pawns[color]
        for f in own:
            if (f - 1) not in own and (f + 1) not in own:
                static_edges.append(f"{cname(not color)} can target {cname(color)}'s isolated pawn")
    if abs(dw - db) >= 2:
        dynamic_edges.append("a development/initiative (dynamic) edge exists")
    if static_edges or dynamic_edges:
        out.append(Concept("Dynamic vs static advantages", "meta",
                           "; ".join((dynamic_edges + static_edges)[:2])))
    return out


def _detect_imbalances(board: chess.Board) -> List[str]:
    items = []
    wn = len(pieces_of(board, WHITE, {chess.KNIGHT})); wb = len(pieces_of(board, WHITE, {chess.BISHOP}))
    bn = len(pieces_of(board, BLACK, {chess.KNIGHT})); bb = len(pieces_of(board, BLACK, {chess.BISHOP}))
    if (wb == 2 and bb < 2) or (bb == 2 and wb < 2):
        items.append("bishop-pair imbalance")
    if wn != bn or wb != bb:
        items.append("minor-piece imbalance (knights vs bishops)")
    return items


# ===========================================================================
# Registry + top-level API
# ===========================================================================
# Canonical concept list mirroring CHESS_CONCEPTS.md (used by the coverage test).
ALL_CONCEPTS: List[str] = [
    # 1 tactical
    "Fork / Double attack", "Pin (absolute)", "Pin (relative)", "Skewer",
    "Discovered attack", "Discovered check", "Double check", "Deflection",
    "Decoy / Attraction", "Removal of the defender", "Overloading",
    "Interference / Obstruction", "Zwischenzug (in-between move)", "Desperado",
    "X-ray", "Battery", "Windmill", "Trapped piece", "Hanging piece",
    "Undermining", "Clearance sacrifice", "Counterattack", "Perpetual check",
    "Greek gift sacrifice (Bxh7+)", "Combination",
    # 2 checkmate
    "Back-rank mate", "Smothered mate", "Anastasia's mate", "Arabian mate",
    "Boden's mate", "Légal's mate", "Scholar's mate", "Fool's mate",
    "Epaulette mate", "Dovetail (Cozio's) mate", "Hook mate",
    "Ladder / staircase mate", "Damiano's mate", "Swallow's tail (Guéridon) mate",
    # 3 activity
    "Development", "Tempo", "Initiative", "Piece activity / mobility",
    "Coordination / harmony", "Outpost", "Good bishop vs bad bishop",
    "Bishop pair", "Opposite-colored bishops", "Knight vs bishop",
    "Rook on the 7th rank", "Doubled rooks", "Fianchetto",
    "Long-diagonal control", "Overprotection", "Improving the worst piece",
    # 4 pawn structure
    "Isolated pawn / IQP", "Doubled pawns", "Tripled pawns", "Backward pawn",
    "Passed pawn", "Protected passed pawn", "Connected passed pawns",
    "Outside passed pawn", "Hanging pawns", "Pawn chain",
    "Pawn majority / minority", "Pawn island", "Pawn break / lever",
    "Pawn storm", "Pawn tension", "Phalanx", "Candidate passed pawn",
    "Weak square / hole", "Color complex weakness",
    # 5 king safety
    "Castling (short/long)", "Pawn shield", "Luft (escape square)",
    "Exposed / uncastled king", "Open lines toward the king",
    "Opposite-side castling", "King activity (endgame)", "Weakened kingside",
    # 6 endgame
    "Opposition (direct/distant/diagonal)", "Zugzwang", "Triangulation",
    "Key squares", "Rule of the square", "Lucena position", "Philidor position",
    "Vancura position", "Rook behind the passed pawn",
    "Wrong-colored bishop + rook pawn", "Fortress",
    "Corresponding / related squares", "Outside passed pawn (endgame use)",
    "King centralization", "Shouldering / body-check", "Pawn breakthrough",
    # 7 opening
    "Control the center", "Develop knights before bishops", "Castle early",
    "Don't move the same piece twice", "Don't bring the queen out too early",
    "Connect the rooks", "Gambit", "Fight for the initiative / lead in development",
    # 8 named structures
    "Isolated Queen's Pawn (IQP)", "Hanging pawns (c+d)", "Carlsbad structure",
    "Maróczy Bind", "Hedgehog", "Sicilian Dragon", "King's Indian Defense",
    "French Defense (closed center)", "Caro-Kann / Slav skeleton", "Stonewall",
    'Scheveningen "small center"',
    # 9 plans
    "Minority attack", "Blockade", "Prophylaxis", "Restriction / cramping",
    "Space advantage", "Two weaknesses principle",
    "Trade when ahead / avoid trades when behind", "Exchange the right pieces",
    "Rerouting a knight", "Rook to an open/semi-open file",
    # 10 sacrifices & material
    "Material balance", "Exchange sacrifice", "Positional pawn sacrifice",
    "Piece sacrifice for attack", "Sham vs real sacrifice", "Compensation",
    # 11 meta
    "Dynamic vs static advantages", "Imbalances", "Tempo / initiative accounting",
    # opening identity
    "Opening",
]

# Detector functions that take just a board.
_BOARD_DETECTORS: List[Callable[[chess.Board], List[Concept]]] = [
    detect_opening,
    detect_forks, detect_pins, detect_skewers, detect_discovered_and_double_check,
    detect_hanging, detect_battery, detect_xray, detect_overloaded_and_defender,
    detect_removal_of_defender, detect_deflection, detect_trapped_piece,
    detect_undermining, detect_combination, detect_zwischenzug, detect_desperado,
    detect_interference, detect_decoy, detect_clearance, detect_counterattack,
    detect_perpetual, detect_greek_gift, detect_windmill,
    detect_checkmate_patterns, detect_named_mate_shortcuts,
    detect_development, detect_outpost, detect_bishops, detect_knight_vs_bishop,
    detect_rook_seventh, detect_doubled_rooks, detect_fianchetto,
    detect_long_diagonal, detect_overprotection, detect_piece_activity,
    detect_coordination,
    detect_pawn_structure, detect_pawn_storm,
    detect_king_safety,
    detect_endgame, detect_fortress,
    detect_gambit,
    detect_named_structures,
    detect_strategic_plans, detect_exchange_right_pieces,
    detect_material_balance, detect_sacrifice_available, detect_exchange_sac,
    detect_positional_pawn_sac, detect_compensation,
    detect_meta,
]


def detect_all_concepts(board: chess.Board, moves: Optional[List[str]] = None) -> List[Concept]:
    """Run every detector and return the union of findings (deduplicated)."""
    found: List[Concept] = []
    for det in _BOARD_DETECTORS:
        try:
            found.extend(det(board))
        except Exception:
            continue
    try:
        found.extend(detect_opening_principles(board, moves))
    except Exception:
        pass
    # dedupe by (name, side, detail)
    seen = set()
    unique = []
    for c in found:
        key = (c.name, c.side, c.detail)
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique


def concept_names_present(board: chess.Board, moves: Optional[List[str]] = None) -> set:
    return {c.name for c in detect_all_concepts(board, moves)}


def format_concepts_for_prompt(board: chess.Board, moves: Optional[List[str]] = None,
                               max_items: int = 40) -> str:
    """Group detected concepts by category into a compact ground-truth block."""
    concepts = detect_all_concepts(board, moves)
    if not concepts:
        return "Chess concepts detected: (none notable)"
    order = ["openingid", "tactical", "checkmate", "material", "king", "pawn",
             "structure", "activity", "plan", "endgame", "opening", "meta"]
    titles = {
        "openingid": "Opening", "tactical": "Tactics", "checkmate": "Checkmate patterns",
        "material": "Material", "king": "King safety", "pawn": "Pawn structure",
        "structure": "Named structures", "activity": "Piece activity",
        "plan": "Strategic plans", "endgame": "Endgame",
        "opening": "Opening principles", "meta": "Evaluation",
    }
    by_cat: Dict[str, List[Concept]] = {}
    for c in concepts[:max_items * 2]:
        by_cat.setdefault(c.category, []).append(c)
    lines = ["Chess concepts detected (deterministic — cite freely):"]
    count = 0
    for cat in order:
        items = by_cat.get(cat, [])
        if not items:
            continue
        lines.append(f"- {titles.get(cat, cat)}:")
        for c in items:
            lines.append(f"    - {c.line()}")
            count += 1
            if count >= max_items:
                break
        if count >= max_items:
            break
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    fen = sys.argv[1] if len(sys.argv) > 1 else chess.STARTING_FEN
    b = chess.Board(fen)
    print(format_concepts_for_prompt(b))
