import argparse
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chess
import chess.engine

try:
    import chess_concepts as _concepts
except Exception:  # pragma: no cover - concept engine optional
    _concepts = None
try:
    from huggingface_hub import InferenceClient
except ImportError:  # pragma: no cover - optional dependency
    InferenceClient = None

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 100,
}

CENTER_SQUARES = {
    chess.D4,
    chess.E4,
    chess.D5,
    chess.E5,
}


def cname(color: chess.Color) -> str:
    """"White" or "Black" for a python-chess color."""
    return "White" if color == chess.WHITE else "Black"


def enemy_targets(board: chess.Board, square: int, color: chess.Color) -> List[str]:
    """Labels (e.g. 'Nf6') of enemy pieces currently attacked from ``square``."""
    return [f"{tp.symbol().upper()}{chess.square_name(t)}"
            for t in board.attacks(square)
            if (tp := board.piece_at(t)) and tp.color != color]

DEFAULT_MODEL = os.getenv("CHESS_TEACHER_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
TOKEN_FILE_NAME = "api_token.txt"
DEFAULT_ROUTER_TIMEOUT = 60


@dataclass
class LineResult:
    label: str
    moves: List[str]
    score: Dict[str, int]
    tags: List[Dict[str, object]]
    source: str


def find_engine_path(cli_path: Optional[str]) -> Optional[Path]:
    if cli_path:
        path = Path(cli_path)
        return path if path.exists() else None

    env_path = os.getenv("STOCKFISH_PATH")
    if env_path:
        path = Path(env_path)
        if path.exists():
            return path

    base = Path(__file__).resolve().parent
    candidates = [
        base / "stockfish" / "stockfish-windows-x86-64-avx2.exe",
        base / "stockfish-windows-x86-64-avx2.exe",
        base / "stockfish",
        Path("/opt/homebrew/bin/stockfish"),
        Path("/usr/local/bin/stockfish"),
        Path("/usr/bin/stockfish"),
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate

    from shutil import which

    found = which("stockfish")
    if found:
        return Path(found)
    return None


def load_hf_token() -> Optional[str]:
    token = os.getenv("HUGGINGFACE_API_TOKEN") or os.getenv("HF_TOKEN")
    if token:
        return token

    token_path = Path(__file__).resolve().parent / TOKEN_FILE_NAME
    if not token_path.exists():
        return None

    content = token_path.read_text(encoding="utf-8").strip()
    return content or None


def parse_move(board: chess.Board, token: str) -> chess.Move:
    token = token.strip()
    if not token:
        raise ValueError("empty move")

    try:
        move = chess.Move.from_uci(token.lower())
        if move in board.legal_moves:
            return move
    except ValueError:
        pass

    try:
        return board.parse_san(token)
    except ValueError:
        pass

    if token[0] in "nbrqk" and token[0].islower():
        try:
            return board.parse_san(token[0].upper() + token[1:])
        except ValueError:
            pass

    raise ValueError(f"illegal or unknown move: {token}")


def score_to_dict(score: chess.engine.PovScore) -> Dict[str, int]:
    if score.is_mate():
        return {"mate": score.mate()}
    return {"cp": score.score(mate_score=100000)}


def format_score(score: Dict[str, int]) -> str:
    if "mate" in score and score["mate"] is not None:
        return f"mate {score['mate']}"
    return f"{score.get('cp', 0) / 100:.2f}"


def score_key(score: Dict[str, int]) -> int:
    mate = score.get("mate")
    if mate is not None:
        if mate > 0:
            return 100000 - mate
        if mate < 0:
            return -100000 - mate
        return 0
    return int(score.get("cp", 0))


def pv_to_san(board: chess.Board, moves: List[chess.Move]) -> List[str]:
    b = board.copy()
    san_moves = []
    for move in moves:
        san_moves.append(b.san(move))
        b.push(move)
    return san_moves


def pinned_squares(board: chess.Board, color: chess.Color) -> List[int]:
    squares = []
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece and piece.color == color and board.is_pinned(color, square):
            squares.append(square)
    return squares


def static_exchange_eval(
    board: chess.Board, target_square: int, attacker_color: chess.Color
) -> int:
    """Static Exchange Evaluation: optimal net material won (in PIECE_VALUES units) by
    `attacker_color` if they initiate a capture sequence on `target_square`. Returns 0
    when no profitable capture exists — each side stops once continuing would lose
    material. Handles x-ray attackers correctly because `attackers()` is re-queried
    after each capture, exposing pieces that were previously blocked."""
    target = board.piece_at(target_square)
    if not target or target.color == attacker_color:
        return 0

    sim = board.copy()
    gains: List[int] = []
    side = attacker_color
    current_value = PIECE_VALUES[target.piece_type]

    while True:
        attackers = sim.attackers(side, target_square)
        if not attackers:
            break

        least_sq = min(
            attackers,
            key=lambda sq: PIECE_VALUES[sim.piece_at(sq).piece_type],
        )
        attacker_piece = sim.piece_at(least_sq)
        if attacker_piece is None:
            break

        gains.append(current_value)
        sim.remove_piece_at(least_sq)
        sim.set_piece_at(target_square, attacker_piece)
        current_value = PIECE_VALUES[attacker_piece.piece_type]
        side = not side

    result = 0
    for gain in reversed(gains):
        result = max(0, gain - result)
    return result


def creates_fork(board: chess.Board, square: int, mover_color: chess.Color) -> bool:
    piece = board.piece_at(square)
    if not piece:
        return False

    # A "fork" by a piece that itself can be profitably captured isn't a real threat.
    if static_exchange_eval(board, square, not mover_color) > 0:
        return False

    targets = []
    for target_sq in board.attacks(square):
        target = board.piece_at(target_sq)
        if not target or target.color == mover_color:
            continue
        if target.piece_type == chess.KING:
            # Attacking the king is always real (check).
            targets.append(target)
            continue
        # Otherwise only count targets we could profitably grab.
        if static_exchange_eval(board, target_sq, mover_color) > 0:
            targets.append(target)

    if not targets:
        return False

    high_value = [
        t for t in targets if PIECE_VALUES[t.piece_type] >= 3 or t.piece_type == chess.KING
    ]
    return len(high_value) >= 2


def is_hanging(board: chess.Board, square: int, color: chess.Color) -> bool:
    """True iff the opponent can win material by capturing the piece on `square`
    according to Static Exchange Evaluation. A defended piece whose capture is
    unprofitable for the attacker is NOT hanging."""
    piece = board.piece_at(square)
    if not piece or piece.color != color:
        return False
    return static_exchange_eval(board, square, not color) > 0


def is_tactical_position(board: chess.Board) -> bool:
    """Position has tactical character that warrants deeper engine analysis."""
    if board.is_check():
        return True
    capture_count = sum(1 for m in board.legal_moves if board.is_capture(m))
    return capture_count >= 4


def tactical_depth(board: chess.Board, base_depth: int, bonus: int = 4) -> int:
    """Adaptive depth: spend more nodes in tactically volatile positions so the engine
    score doesn't mislead the explainer about hanging pieces or exchange outcomes."""
    return base_depth + (bonus if is_tactical_position(board) else 0)


def position_facts(board: chess.Board) -> List[str]:
    """Plain-language ground truth about which pieces are attacked, defended, or
    hanging in the current position. Anchors the LLM in concrete attacker/defender
    info so it doesn't misjudge a defended piece as hanging."""
    hanging_facts: List[str] = []
    contested_facts: List[str] = []

    for color in (chess.WHITE, chess.BLACK):
        color_name = cname(color)
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if not piece or piece.color != color or piece.piece_type == chess.KING:
                continue
            opponent = not color
            attackers = board.attackers(opponent, square)
            if not attackers:
                continue

            defenders = board.attackers(color, square)
            sq_name = chess.square_name(square)
            piece_name = chess.piece_name(piece.piece_type)

            def describe(squares: chess.SquareSet) -> str:
                items = []
                for s in squares:
                    p = board.piece_at(s)
                    if p is None:
                        continue
                    items.append((PIECE_VALUES[p.piece_type], p.symbol().upper()))
                items.sort(key=lambda x: x[0])
                return "/".join(sym for _, sym in items) if items else "none"

            att_str = describe(attackers)
            def_str = describe(defenders)
            see_gain = static_exchange_eval(board, square, opponent)

            if see_gain > 0:
                hanging_facts.append(
                    f"{color_name} {piece_name} on {sq_name} is HANGING — "
                    f"attackers {att_str}, defenders {def_str}; "
                    f"opponent wins ~{see_gain} by capturing"
                )
            else:
                contested_facts.append(
                    f"{color_name} {piece_name} on {sq_name} is attacked but DEFENDED — "
                    f"attackers {att_str}, defenders {def_str}; "
                    f"capture is not profitable for opponent"
                )

    return hanging_facts + contested_facts


def positional_facts(board: chess.Board) -> List[str]:
    """Strategic/structural ground truth: pawn structure, king safety, color complexes,
    open files, outposts, and recognized opening structures. Complements position_facts
    (which is purely tactical) by giving the explainer concrete positional themes to
    discuss — bishop trades, weak squares, plans tied to the pawn skeleton, etc."""
    facts: List[str] = []

    pawns: Dict[chess.Color, Dict[int, List[int]]] = {chess.WHITE: {}, chess.BLACK: {}}
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece and piece.piece_type == chess.PAWN:
            pawns[piece.color].setdefault(chess.square_file(sq), []).append(
                chess.square_rank(sq)
            )

    def piece_at_is(sq: int, piece_type: int, color: chess.Color) -> bool:
        p = board.piece_at(sq)
        return p is not None and p.piece_type == piece_type and p.color == color

    # --- Pawn structure: doubled, isolated, passed ---
    for color in (chess.WHITE, chess.BLACK):
        color_name = cname(color)
        own = pawns[color]
        enemy = pawns[not color]

        doubled = sorted(
            chess.FILE_NAMES[f] for f, ranks in own.items() if len(ranks) >= 2
        )
        if doubled:
            facts.append(
                f"{color_name} has doubled pawns on file(s) {', '.join(doubled)}"
            )

        isolated = sorted(
            chess.FILE_NAMES[f]
            for f in own
            if (f - 1) not in own and (f + 1) not in own
        )
        if isolated:
            facts.append(
                f"{color_name} has isolated pawn(s) on file(s) {', '.join(isolated)}"
            )

        passed = []
        for f, ranks in own.items():
            for r in ranks:
                blocked = False
                for ef in (f - 1, f, f + 1):
                    if ef not in enemy:
                        continue
                    for er in enemy[ef]:
                        if (color == chess.WHITE and er > r) or (
                            color == chess.BLACK and er < r
                        ):
                            blocked = True
                            break
                    if blocked:
                        break
                if not blocked:
                    passed.append(chess.square_name(chess.square(f, r)))
        if passed:
            facts.append(f"{color_name} has passed pawn(s) on {', '.join(passed)}")

    # --- Open / semi-open files ---
    files_with_pawns = set(pawns[chess.WHITE]) | set(pawns[chess.BLACK])
    open_files = [chess.FILE_NAMES[f] for f in range(8) if f not in files_with_pawns]
    if open_files:
        facts.append(f"Open file(s) (no pawns): {', '.join(open_files)}")
    for color in (chess.WHITE, chess.BLACK):
        color_name = cname(color)
        semi = [
            chess.FILE_NAMES[f]
            for f in range(8)
            if f not in pawns[color] and f in pawns[not color]
        ]
        if semi:
            facts.append(f"{color_name} has semi-open file(s) {', '.join(semi)}")

    # --- Bishops and color complex ---
    for color in (chess.WHITE, chess.BLACK):
        color_name = cname(color)
        bishop_squares = [
            sq
            for sq in chess.SQUARES
            if piece_at_is(sq, chess.BISHOP, color)
        ]
        bishop_colors = {
            (chess.square_file(sq) + chess.square_rank(sq)) % 2
            for sq in bishop_squares
        }
        if len(bishop_squares) >= 2 and len(bishop_colors) == 2:
            facts.append(f"{color_name} has the bishop pair")

        light_pawns = 0
        dark_pawns = 0
        for f, ranks in pawns[color].items():
            for r in ranks:
                if (f + r) % 2 == 1:
                    light_pawns += 1
                else:
                    dark_pawns += 1
        total = light_pawns + dark_pawns
        if total >= 5 and abs(light_pawns - dark_pawns) >= 3:
            heavy = "light" if light_pawns > dark_pawns else "dark"
            weak = "dark" if light_pawns > dark_pawns else "light"
            facts.append(
                f"{color_name} pawns sit mostly on {heavy} squares "
                f"({light_pawns}L/{dark_pawns}D) — {weak} squares are weak in their "
                f"camp; the {weak}-squared bishop is the more valuable minor piece "
                f"(trade theirs, keep yours)"
            )

    # --- King safety ---
    for color in (chess.WHITE, chess.BLACK):
        color_name = cname(color)
        king_sq = board.king(color)
        if king_sq is None:
            continue
        kf = chess.square_file(king_sq)
        kr = chess.square_rank(king_sq)
        home = 0 if color == chess.WHITE else 7
        if kr == home and kf >= 6:
            where = "castled kingside"
        elif kr == home and kf <= 2:
            where = "castled queenside"
        elif kr == home and 3 <= kf <= 5:
            where = "uncastled (center)"
        else:
            where = f"on {chess.square_name(king_sq)} (off back rank)"
        facts.append(f"{color_name} king {where}")

        if kr == home and kf >= 6:
            shield_rank = 1 if color == chess.WHITE else 6
            missing = []
            for sf in (5, 6, 7):
                if shield_rank not in pawns[color].get(sf, []):
                    missing.append(chess.FILE_NAMES[sf])
            if missing:
                facts.append(
                    f"{color_name} kingside pawn shield missing on "
                    f"{', '.join(missing)}"
                )

    # --- Outpost squares (in opponent's half, defended by own pawn,
    #     not challengeable by any enemy pawn) ---
    for color in (chess.WHITE, chess.BLACK):
        color_name = cname(color)
        outposts = []
        for sq in chess.SQUARES:
            f = chess.square_file(sq)
            r = chess.square_rank(sq)
            if f < 2 or f > 5:
                continue
            if color == chess.WHITE and r < 3:
                continue
            if color == chess.BLACK and r > 4:
                continue
            pawn_defends = False
            for d in board.attackers(color, sq):
                dp = board.piece_at(d)
                if dp and dp.piece_type == chess.PAWN:
                    pawn_defends = True
                    break
            if not pawn_defends:
                continue
            challengeable = False
            for ef in (f - 1, f + 1):
                if not 0 <= ef <= 7:
                    continue
                for er in pawns[not color].get(ef, []):
                    if (color == chess.WHITE and er > r) or (
                        color == chess.BLACK and er < r
                    ):
                        challengeable = True
                        break
                if challengeable:
                    break
            if not challengeable:
                outposts.append(chess.square_name(sq))
        if outposts:
            facts.append(
                f"{color_name} outpost square(s): {', '.join(outposts)} (pawn-defended, "
                f"no enemy pawn can challenge — ideal for a knight)"
            )

    # --- Recognized opening structures ---
    bp = pawns[chess.BLACK]
    wp = pawns[chess.WHITE]

    # Sicilian Dragon for Black: ...d6, ...g6, ...Bg7, c-pawn traded
    if (
        piece_at_is(chess.G7, chess.BISHOP, chess.BLACK)
        and 5 in bp.get(6, [])
        and 5 in bp.get(3, [])
        and 2 not in bp
    ):
        facts.append(
            "Structure: Sicilian Dragon (Black has ...d6, ...g6, ...Bg7, c-pawn "
            "traded). Black's plan: keep the g7 dark-squared bishop and trade off "
            "White's dark-squared bishop to monopolize the long h8-a1 diagonal and "
            "the dark-square complex around White's king. White's plan: Yugoslav "
            "attack (Be3, Qd2, O-O-O, h4-h5, Bh6 to trade the Dragon bishop) with "
            "a kingside pawn storm."
        )

    # King's Indian Defense for Black: g7 bishop, ...d6, ...g6, ...e5
    if (
        piece_at_is(chess.G7, chess.BISHOP, chess.BLACK)
        and 5 in bp.get(6, [])
        and 5 in bp.get(3, [])
        and 4 in bp.get(4, [])
    ):
        facts.append(
            "Structure: King's Indian Defense (Black: ...d6, ...e5, ...g6, ...Bg7). "
            "Black aims for ...f5 and a kingside attack while White expands on the "
            "queenside. The g7-bishop is critical to Black's defense and attack; "
            "Black should avoid trading it for a knight."
        )

    # White IQP on d4
    if (
        3 in wp.get(3, [])
        and 2 not in wp
        and 4 not in wp
    ):
        facts.append(
            "Structure: White has an Isolated Queen Pawn (IQP) on d4. White plays "
            "for piece activity, c5/e5 outposts for the knights, and kingside "
            "attacks (Bd3+Qc2 battery, Re1). Black's plan: blockade with a knight "
            "on d5 and head for an endgame where the d4-pawn becomes a target."
        )

    # Black IQP on d5
    if (
        4 in bp.get(3, [])
        and 2 not in bp
        and 4 not in bp
    ):
        facts.append(
            "Structure: Black has an Isolated Queen Pawn (IQP) on d5. Mirror "
            "of White IQP — Black plays for active pieces, White blockades and "
            "targets the IQP in the endgame."
        )

    # French Defense closed center: White d4+e5, Black d5+e6
    if (
        3 in wp.get(3, [])
        and 4 in wp.get(4, [])
        and 4 in bp.get(3, [])
        and 5 in bp.get(4, [])
    ):
        facts.append(
            "Structure: French Defense, closed center (White d4+e5, Black d5+e6). "
            "White has space and a kingside attack base anchored on e5. Black's "
            "freeing breaks are ...c5 (queenside) and ...f6 (kingside). Black's "
            "light-squared bishop on c8 is the classic 'French bad bishop'; "
            "trading it off (often via ...b6/...Ba6 or ...Bd7-e8-h5) is a key plan."
        )

    # Hedgehog for Black: a6, b6, d6, e6, no c-pawn
    if (
        5 in bp.get(0, [])
        and 5 in bp.get(1, [])
        and 5 in bp.get(3, [])
        and 5 in bp.get(4, [])
        and 2 not in bp
    ):
        facts.append(
            "Structure: Hedgehog (Black: a6/b6/d6/e6, c-pawn traded). Black holds "
            "a flexible low-profile setup waiting to uncoil with ...b5 or ...d5. "
            "White has space and must not allow either break under good conditions; "
            "Black must avoid passive piece play."
        )

    # Caro-Kann / Slav skeleton for Black: ...c6 + ...d5
    if 5 in bp.get(2, []) and 4 in bp.get(3, []):
        facts.append(
            "Structure: Black has the Caro-Kann/Slav pawn skeleton (...c6 + ...d5). "
            "Solid but slightly passive; ...c5 is Black's typical freeing break."
        )

    # White Stonewall: c3, d4, e3, f4
    if (
        2 in wp.get(2, [])
        and 3 in wp.get(3, [])
        and 2 in wp.get(4, [])
        and 3 in wp.get(5, [])
    ):
        facts.append(
            "Structure: White Stonewall (c3/d4/e3/f4). White plays for a kingside "
            "attack (Ne5 outpost, Bd3+Qh5). The e4 square is a permanent hole and "
            "the c1-bishop is hard to develop — Black should aim for ...Ne4 and "
            "exchange dark-squared bishops."
        )

    return facts


def piece_placement_summary(board: chess.Board) -> str:
    """Plain-language listing of where each side's pieces stand. Small LLMs cannot
    reliably parse FEN, so this gives them the same information in readable form."""
    order = [chess.KING, chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN]
    singular = {
        chess.KING: "King", chess.QUEEN: "Queen", chess.ROOK: "Rook",
        chess.BISHOP: "Bishop", chess.KNIGHT: "Knight", chess.PAWN: "Pawn",
    }
    plural = {
        chess.KING: "Kings", chess.QUEEN: "Queens", chess.ROOK: "Rooks",
        chess.BISHOP: "Bishops", chess.KNIGHT: "Knights", chess.PAWN: "Pawns",
    }
    out = []
    for color in (chess.WHITE, chess.BLACK):
        color_name = cname(color)
        by_type: Dict[int, List[str]] = {}
        for sq in chess.SQUARES:
            p = board.piece_at(sq)
            if p and p.color == color:
                by_type.setdefault(p.piece_type, []).append(chess.square_name(sq))
        parts = []
        for pt in order:
            squares = sorted(by_type.get(pt, []))
            if not squares:
                continue
            label = plural[pt] if len(squares) > 1 else singular[pt]
            parts.append(f"{label} {'/'.join(squares)}")
        out.append(f"{color_name}: {'; '.join(parts)}")
    return "\n".join(out)


def attack_relations(board: chess.Board) -> List[str]:
    """Enumerate every piece-on-piece attack currently on the board. Anchors the LLM
    in concrete attack relationships so it can't invent ones that don't exist
    (e.g., 'e4 attacks c6' when in fact e4 attacks only the empty squares d5/f5).
    Empty-square attacks are intentionally omitted to keep the listing tight."""
    rels: List[str] = []
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if not p:
            continue
        targets = enemy_targets(board, sq, p.color)
        if not targets:
            continue
        color_name = cname(p.color)
        piece_name = chess.piece_name(p.piece_type)
        rels.append(
            f"{color_name} {piece_name} on {chess.square_name(sq)} attacks "
            f"{', '.join(targets)}"
        )
    return rels


def piece_vision_summary(board: chess.Board) -> List[str]:
    """Exhaustive, literal square-control map: for EVERY piece, the exact squares it
    attacks ('sees'), with any enemy pieces on those squares called out. This is the
    complete vision ground truth, so the model never has to guess which piece sees
    what (e.g. that a rook on a5 does NOT see h7 until it moves to the h-file)."""
    rels: List[str] = []
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if not p:
            continue
        seen = sorted(board.attacks(sq),
                      key=lambda s: (chess.square_rank(s), chess.square_file(s)))
        if not seen:
            continue
        squares = " ".join(chess.square_name(s) for s in seen)
        hits = [f"{board.piece_at(s).symbol().upper()}{chess.square_name(s)}"
                for s in seen if board.piece_at(s) and board.piece_at(s).color != p.color]
        tail = f"  [enemy pieces hit: {', '.join(hits)}]" if hits else ""
        rels.append(
            f"{cname(p.color)} {chess.piece_name(p.piece_type)} on {chess.square_name(sq)} "
            f"attacks: {squares}{tail}"
        )
    return rels


def defense_relations(board: chess.Board) -> List[str]:
    """For each piece that is CURRENTLY ATTACKED by an enemy piece, list which
    friendly pieces defend it. Defense only matters when there is an attack to meet,
    so quietly-defended-but-unattacked pieces (e.g. a pawn the king happens to stand
    next to, like g7) are omitted as noise. Together with attack_relations this forms
    the *contested* attacker/defender graph the position actually turns on."""
    rels: List[str] = []
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if not p:
            continue
        # Skip pieces no enemy attacks -- their defenders are irrelevant to the position.
        if not board.attackers(not p.color, sq):
            continue
        defenders = []
        for d in board.attackers(p.color, sq):
            if d == sq:
                continue
            dp = board.piece_at(d)
            if dp:
                defenders.append(f"{dp.symbol().upper()}{chess.square_name(d)}")
        if not defenders:
            continue
        color_name = cname(p.color)
        piece_name = chess.piece_name(p.piece_type)
        rels.append(
            f"{color_name} {piece_name} on {chess.square_name(sq)} is defended by "
            f"{', '.join(defenders)}"
        )
    return rels


def pin_descriptions(board: chess.Board) -> List[str]:
    """Absolutely-pinned pieces (cannot move without exposing the king). Walks the
    pin line from king through pinned piece to identify the pinner."""
    items: List[str] = []
    for color in (chess.WHITE, chess.BLACK):
        color_name = cname(color)
        king_sq = board.king(color)
        if king_sq is None:
            continue
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if not piece or piece.color != color:
                continue
            if not board.is_pinned(color, square):
                continue
            df = chess.square_file(square) - chess.square_file(king_sq)
            dr = chess.square_rank(square) - chess.square_rank(king_sq)
            if df != 0:
                df //= abs(df)
            if dr != 0:
                dr //= abs(dr)
            cf = chess.square_file(square) + df
            cr = chess.square_rank(square) + dr
            pinner_sq = None
            while 0 <= cf <= 7 and 0 <= cr <= 7:
                cs = chess.square(cf, cr)
                cp = board.piece_at(cs)
                if cp:
                    if cp.color != color:
                        pinner_sq = cs
                    break
                cf += df
                cr += dr
            piece_name = chess.piece_name(piece.piece_type)
            if pinner_sq is not None:
                pinner = board.piece_at(pinner_sq)
                items.append(
                    f"{color_name} {piece_name} on {chess.square_name(square)} is "
                    f"PINNED against the king by "
                    f"{pinner.symbol().upper()}{chess.square_name(pinner_sq)}"
                )
            else:
                items.append(
                    f"{color_name} {piece_name} on {chess.square_name(square)} is PINNED"
                )
    return items


def existing_forks(board: chess.Board) -> List[str]:
    """Pieces already on the board that fork two or more valuable enemy targets.
    Distinct from move_tags' 'fork threat' which flags hypothetical moves."""
    items: List[str] = []
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if not p:
            continue
        if not creates_fork(board, sq, p.color):
            continue
        targets = enemy_targets(board, sq, p.color)
        if len(targets) < 2:
            continue
        color_name = cname(p.color)
        piece_name = chess.piece_name(p.piece_type)
        items.append(
            f"{color_name} {piece_name} on {chess.square_name(sq)} FORKS "
            f"{', '.join(targets)}"
        )
    return items


def king_zone_threats(board: chess.Board) -> List[str]:
    """Squares adjacent to each king that the opponent currently attacks. A quick
    proxy for king-attack pressure."""
    items: List[str] = []
    for color in (chess.WHITE, chess.BLACK):
        color_name = cname(color)
        king_sq = board.king(color)
        if king_sq is None:
            continue
        attacked = []
        kf = chess.square_file(king_sq)
        kr = chess.square_rank(king_sq)
        for df in (-1, 0, 1):
            for dr in (-1, 0, 1):
                if df == 0 and dr == 0:
                    continue
                nf, nr = kf + df, kr + dr
                if not (0 <= nf <= 7 and 0 <= nr <= 7):
                    continue
                ns = chess.square(nf, nr)
                if board.attackers(not color, ns):
                    attacked.append(chess.square_name(ns))
        if attacked:
            items.append(
                f"{color_name} king zone — squares next to "
                f"K{chess.square_name(king_sq)} attacked by opponent: "
                f"{', '.join(attacked)}"
            )
    return items


def position_meta(board: chess.Board) -> List[str]:
    """Position-level state: check, castling rights, en passant, material balance,
    and 50-move clock proximity."""
    items: List[str] = []

    if board.is_check():
        side = cname(board.turn)
        items.append(f"{side} is in CHECK")

    rights = []
    if board.has_kingside_castling_rights(chess.WHITE):
        rights.append("White O-O")
    if board.has_queenside_castling_rights(chess.WHITE):
        rights.append("White O-O-O")
    if board.has_kingside_castling_rights(chess.BLACK):
        rights.append("Black O-O")
    if board.has_queenside_castling_rights(chess.BLACK):
        rights.append("Black O-O-O")
    items.append(
        f"Castling rights: {', '.join(rights)}" if rights else "Castling rights: none"
    )

    if board.ep_square is not None:
        items.append(f"En passant target square: {chess.square_name(board.ep_square)}")

    white_mat = 0
    black_mat = 0
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if not p or p.piece_type == chess.KING:
            continue
        v = PIECE_VALUES[p.piece_type]
        if p.color == chess.WHITE:
            white_mat += v
        else:
            black_mat += v
    diff = white_mat - black_mat
    if diff > 0:
        items.append(f"Material balance: White +{diff} (W{white_mat} vs B{black_mat})")
    elif diff < 0:
        items.append(f"Material balance: Black +{-diff} (W{white_mat} vs B{black_mat})")
    else:
        items.append(f"Material balance: equal ({white_mat} vs {black_mat})")

    if board.halfmove_clock >= 30:
        items.append(
            f"Halfmove clock: {board.halfmove_clock}/50 (approaching 50-move rule)"
        )

    return items


def best_move_anchor(board: chess.Board, engine_lines: List["LineResult"]) -> str:
    """One-line callout naming the engine's #1 move with score, ply notation, and a
    short continuation. Pinned at the top of the prompt to keep the LLM from
    inventing its own 'best move'."""
    side = cname(board.turn)
    if not engine_lines or not engine_lines[0].moves:
        return f"ENGINE'S BEST MOVE for {side}: (no engine output available)"
    best = engine_lines[0]
    move_number = board.fullmove_number
    prefix = f"{move_number}." if board.turn == chess.WHITE else f"{move_number}..."
    continuation = format_moves_with_sides(board, best.moves[:4])
    return (
        f"ENGINE'S BEST MOVE for {side}: {prefix}{best.moves[0]} "
        f"(eval {format_score(best.score)} from {side}'s perspective; "
        f"sample continuation: {continuation})"
    )


def move_tags(board: chess.Board, move: chess.Move) -> List[str]:
    tags = []
    piece = board.piece_at(move.from_square)
    if not piece:
        return tags

    is_capture = board.is_capture(move)
    is_ep = is_capture and board.is_en_passant(move)
    captured_piece = (
        board.piece_at(move.to_square) if is_capture and not is_ep else None
    )

    opponent = not board.turn
    before_pins = set(pinned_squares(board, opponent))

    tmp = board.copy()
    tmp.push(move)

    if is_capture:
        if is_ep:
            captured_value = PIECE_VALUES[chess.PAWN]
            captured_symbol = "P"
        elif captured_piece:
            captured_value = PIECE_VALUES[captured_piece.piece_type]
            captured_symbol = captured_piece.symbol().upper()
        else:
            captured_value = 0
            captured_symbol = None

        # After our capture, what's the opponent's optimal recapture gain?
        opp_gain = static_exchange_eval(tmp, move.to_square, not piece.color)
        net_gain = captured_value - opp_gain

        base = f"captures {captured_symbol}" if captured_symbol else "captures"
        if net_gain < 0:
            tags.append(f"{base} (losing exchange {net_gain})")
        elif opp_gain > 0:
            tags.append(f"{base} (winning exchange +{net_gain})")
        else:
            tags.append(base)

    if board.is_castling(move):
        tags.append("castle")
    if board.gives_check(move):
        tags.append("check")
    if move.promotion:
        tags.append("promotion")

    if piece.piece_type in (chess.KNIGHT, chess.BISHOP):
        if chess.square_rank(move.from_square) in (0, 7):
            tags.append("develops minor")

    if move.to_square in CENTER_SQUARES:
        tags.append("occupies center")

    after_pins = set(pinned_squares(tmp, opponent))
    if len(after_pins) > len(before_pins):
        tags.append("creates pin")

    if creates_fork(tmp, move.to_square, piece.color):
        tags.append("fork threat")

    # Hanging is only reported for non-captures; the capture tag above already
    # surfaces losing exchanges, so we'd otherwise double-flag.
    if not is_capture and is_hanging(tmp, move.to_square, piece.color):
        see_loss = static_exchange_eval(tmp, move.to_square, not piece.color)
        tags.append(f"hangs piece (drops {see_loss})")

    return tags


def line_tags(board: chess.Board, moves: List[str]) -> List[Dict[str, object]]:
    tags = []
    b = board.copy()
    for ply_index, san in enumerate(moves, start=1):
        try:
            move = parse_move(b, san)
        except ValueError:
            break
        entry = {"ply": ply_index, "move": b.san(move), "tags": move_tags(b, move)}
        tags.append(entry)
        b.push(move)
    return tags


def ply_label(board: chess.Board, ply_index: int, san: str) -> str:
    move_number = board.fullmove_number + (ply_index - 1) // 2
    is_white_move = (ply_index % 2 == 1) == (board.turn == chess.WHITE)
    if is_white_move:
        return f"{move_number}.{san}"
    return f"{move_number}...{san}"


def format_tags_for_prompt(board: chess.Board, tags: List[Dict[str, object]]) -> str:
    parts = []
    for entry in tags:
        if not entry["tags"]:
            continue
        label = ply_label(board, entry["ply"], entry["move"])
        parts.append(f"{label}: {', '.join(entry['tags'])}")
    return "; ".join(parts)


def engine_top_lines(
    board: chess.Board,
    engine: chess.engine.SimpleEngine,
    depth: int,
    top: int,
    pv_plies: int,
) -> List[LineResult]:
    actual_depth = tactical_depth(board, depth)
    info_list = engine.analyse(board, chess.engine.Limit(depth=actual_depth), multipv=top)
    if isinstance(info_list, dict):
        info_list = [info_list]

    lines = []
    for idx, info in enumerate(info_list, start=1):
        pv = info.get("pv", [])[:pv_plies]
        san_moves = pv_to_san(board, pv)
        score = score_to_dict(info["score"].pov(board.turn))
        tags = line_tags(board, san_moves)
        lines.append(
            LineResult(
                label=f"engine_{idx}",
                moves=san_moves,
                score=score,
                tags=tags,
                source="engine",
            )
        )
    return lines


def find_prompt_moves(prompt: str, board: chess.Board) -> List[str]:
    lower_prompt = prompt.lower()
    moves = []
    for move in board.legal_moves:
        san = board.san(move)
        uci = move.uci()
        if re.search(rf"\b{re.escape(san.lower())}\b", lower_prompt):
            moves.append(san)
            continue
        if re.search(rf"\b{re.escape(uci)}\b", lower_prompt):
            moves.append(san)
    return moves


def safe_json_loads(text: str) -> Optional[dict]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def _chat_payload(model: str, prompt: str, max_new_tokens: int, temperature: float) -> dict:
    """OpenAI-compatible chat-completion request body shared by the LLM backends."""
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": 0.9,
        "stream": False,
    }


def _ollama_generate(prompt: str, model: str, max_new_tokens: int, temperature: float) -> str:
    """Local LLM via Ollama's OpenAI-compatible endpoint (free, offline)."""
    base = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    url = f"{base}/v1/chat/completions"
    payload = _chat_payload(model, prompt, max_new_tokens, temperature)
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    timeout_seconds = int(os.getenv("OLLAMA_TIMEOUT", "120"))
    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8") if exc.fp else ""
        raise RuntimeError(f"Ollama error {exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"Cannot reach Ollama at {base} ({exc}). Run 'ollama serve'.") from exc
    try:
        parsed = json.loads(body)
        text = parsed["choices"][0]["message"]["content"]
    except (json.JSONDecodeError, KeyError, IndexError) as exc:
        raise RuntimeError("Unexpected Ollama response") from exc
    if not text:
        raise RuntimeError("Empty Ollama response")
    return str(text).strip()


def _llm_backend() -> str:
    return os.getenv("CHESS_TEACHER_LLM", "hf").strip().lower()


def hf_generate(
    prompt: str,
    model: str,
    max_new_tokens: int = 512,
    temperature: float = 0.2,
) -> str:
    # Local backend (Ollama) when configured — free, offline, no HF credits needed.
    if _llm_backend() in ("ollama", "local"):
        return _ollama_generate(prompt, model, max_new_tokens, temperature)
    if InferenceClient is None:
        raise RuntimeError("huggingface_hub is not installed. Run 'pip install huggingface_hub'.")
    token = load_hf_token()
    if not token:
        raise RuntimeError("Missing HUGGINGFACE_API_TOKEN, HF_TOKEN, or api_token.txt")

    client_error = None
    try:
        try:
            client = InferenceClient(api_key=token)
        except TypeError:
            client = InferenceClient(token=token)

        if hasattr(client, "chat") and hasattr(client.chat, "completions"):
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.9,
            )
            message = completion.choices[0].message
            text = getattr(message, "content", None)
            if text is None and isinstance(message, dict):
                text = message.get("content")
            if text:
                return str(text).strip()
        else:
            completion = client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.9,
            )
            choice = completion.choices[0]
            message = getattr(choice, "message", None) or {}
            text = getattr(message, "content", None)
            if text is None and isinstance(message, dict):
                text = message.get("content")
            if text:
                return str(text).strip()
    except Exception as exc:
        client_error = exc

    url = "https://router.huggingface.co/v1/chat/completions"
    timeout_seconds = int(os.getenv("HF_ROUTER_TIMEOUT", str(DEFAULT_ROUTER_TIMEOUT)))
    payload = _chat_payload(model, prompt, max_new_tokens, temperature)
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8") if exc.fp else ""
        raise RuntimeError(f"HF router error {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Network error contacting Hugging Face router: {exc}") from exc
    except TimeoutError as exc:
        raise RuntimeError("Hugging Face router request timed out") from exc
    except Exception as exc:
        raise RuntimeError(f"Hugging Face router error: {exc}") from exc

    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Unexpected Hugging Face router response") from exc

    if isinstance(data, dict) and "error" in data:
        raise RuntimeError(f"HF router error: {data['error']}")

    choices = data.get("choices", []) if isinstance(data, dict) else []
    if not choices:
        if client_error:
            raise RuntimeError(f"HF client failed: {client_error}")
        raise RuntimeError("Unexpected Hugging Face router response")

    message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
    text = message.get("content") if isinstance(message, dict) else None
    if not text:
        raise RuntimeError("Unexpected Hugging Face router response")

    return str(text).strip()


def llm_candidate_lines(
    board: chess.Board,
    prompt: str,
    max_candidates: int,
    model: str,
) -> List[Tuple[str, List[str]]]:
    legal_san = [board.san(m) for m in board.legal_moves]
    legal_list = ", ".join(legal_san)

    llm_prompt = (
        "You are a chess assistant. Return only valid JSON.\n"
        "Provide up to {max_candidates} candidate move lines that directly address the user question.\n"
        "Use SAN notation only, and only moves from the provided legal list.\n"
        "Output schema: {{\"candidates\": [{{\"label\": \"...\", \"moves\": [\"Nf3\", \"Nc6\"]}}]}}.\n\n"
        "Position FEN: {fen}\n"
        "Side to move: {side}\n"
        "Legal moves (SAN): {legal}\n"
        "User question: {question}\n"
    ).format(
        max_candidates=max_candidates,
        fen=board.fen(),
        side=cname(board.turn),
        legal=legal_list,
        question=prompt,
    )

    try:
        raw = hf_generate(llm_prompt, model=model, max_new_tokens=192, temperature=0.3)
    except RuntimeError as exc:
        print(f"LLM candidate generation skipped: {exc}")
        return []
    data = safe_json_loads(raw)
    if not data or "candidates" not in data:
        return []

    results = []
    for entry in data.get("candidates", [])[:max_candidates]:
        label = str(entry.get("label", "candidate"))
        moves = entry.get("moves", [])
        if not isinstance(moves, list) or not moves:
            continue
        results.append((label, [str(m) for m in moves]))

    return results


def fallback_explain(
    board: chess.Board,
    prompt: str,
    engine_lines: List[LineResult],
    candidate_lines: List[LineResult],
    error: str,
) -> str:
    parts = [
        "LLM unavailable; showing engine-only guidance.",
        f"Reason: {error}",
        f"Prompt: {prompt}",
    ]

    if engine_lines:
        best = engine_lines[0]
        parts.append(
            "Best engine line: "
            f"{format_score(best.score)} | {' '.join(best.moves)}"
        )

    if candidate_lines:
        parts.append("Candidate line notes:")
        for line in candidate_lines:
            tags = format_tags_for_prompt(board, line.tags)
            line_text = f"{line.label}: {format_score(line.score)} | {' '.join(line.moves)}"
            if tags:
                line_text += f" | tags: {tags}"
            parts.append(line_text)

    return "\n".join(parts)


def extend_line_with_engine(
    board: chess.Board,
    engine: chess.engine.SimpleEngine,
    depth: int,
    pv_plies: int,
    moves: List[str],
) -> List[str]:
    b = board.copy()
    san_moves = []
    for move_str in moves:
        try:
            move = parse_move(b, move_str)
        except ValueError:
            break
        san_moves.append(b.san(move))
        b.push(move)
        if len(san_moves) >= pv_plies:
            return san_moves

    remaining = pv_plies - len(san_moves)
    if remaining <= 0:
        return san_moves

    info = engine.analyse(b, chess.engine.Limit(depth=tactical_depth(b, depth)))
    pv = info.get("pv", [])[:remaining]
    san_moves.extend(pv_to_san(b, pv))
    return san_moves


def evaluate_line(
    board: chess.Board,
    engine: chess.engine.SimpleEngine,
    depth: int,
    moves: List[str],
) -> Dict[str, int]:
    b = board.copy()
    for move_str in moves:
        try:
            move = parse_move(b, move_str)
        except ValueError:
            break
        b.push(move)
    info = engine.analyse(b, chess.engine.Limit(depth=tactical_depth(b, depth)))
    return score_to_dict(info["score"].pov(board.turn))


def build_candidate_lines(
    board: chess.Board,
    engine: chess.engine.SimpleEngine,
    prompt: str,
    depth: int,
    pv_plies: int,
    max_candidates: int,
    model: str,
    enable_llm: bool,
) -> List[LineResult]:
    candidates = []
    seen = set()

    prompt_moves = find_prompt_moves(prompt, board)
    for move in prompt_moves:
        if (move,) not in seen:
            candidates.append(("prompt", [move]))
            seen.add((move,))

    if enable_llm:
        for label, moves in llm_candidate_lines(board, prompt, max_candidates, model):
            key = tuple(moves)
            if key in seen:
                continue
            candidates.append((label, moves))
            seen.add(key)

    results = []
    for label, moves in candidates[:max_candidates]:
        line_moves = extend_line_with_engine(board, engine, depth, pv_plies, moves)
        score = evaluate_line(board, engine, depth, line_moves)
        tags = line_tags(board, line_moves)
        results.append(
            LineResult(
                label=label,
                moves=line_moves,
                score=score,
                tags=tags,
                source="candidate",
            )
        )

    results.sort(key=lambda line: score_key(line.score), reverse=True)
    return results


def extract_hypothetical_candidates(
    text: str,
    board: chess.Board,
    engine_lines: List[LineResult],
    max_lookahead: int = 2,
) -> List[Tuple[List[str], str]]:
    """Find moves mentioned in `text` that could be analyzed as hypotheticals.
    Returns (prefix_moves, san_move) tuples. Considers the current position and
    up to `max_lookahead` plies down the engine's #1 line so questions like
    'what about e5?' (asked when White's e5 only becomes legal after Black moves
    the f-pawn knight) still resolve to the right branch."""
    candidates: List[Tuple[List[str], str]] = []
    seen: set = set()

    for san in find_prompt_moves(text, board):
        key = ((), san)
        if key not in seen:
            candidates.append(([], san))
            seen.add(key)

    if engine_lines and engine_lines[0].moves:
        b = board.copy()
        prefix: List[str] = []
        for mv_san in engine_lines[0].moves[:max_lookahead]:
            try:
                mv = parse_move(b, mv_san)
            except ValueError:
                break
            prefix.append(b.san(mv))
            b.push(mv)
            for san in find_prompt_moves(text, b):
                key = (tuple(prefix), san)
                if key not in seen:
                    candidates.append((list(prefix), san))
                    seen.add(key)

    return candidates


def hypothetical_line(
    board: chess.Board,
    engine: chess.engine.SimpleEngine,
    prefix_moves: List[str],
    user_move: str,
    depth: int,
    pv_plies: int,
) -> Optional[LineResult]:
    """Play `prefix_moves` + `user_move`, let the engine fill the continuation,
    and return a LineResult scored from the original side-to-move's POV. Returns
    None if any move in the chain is illegal."""
    b = board.copy()
    full_moves: List[str] = []
    for mv_san in list(prefix_moves) + [user_move]:
        try:
            mv = parse_move(b, mv_san)
        except ValueError:
            return None
        full_moves.append(b.san(mv))
        b.push(mv)

    remaining = pv_plies - len(full_moves)
    if remaining > 0:
        info = engine.analyse(b, chess.engine.Limit(depth=tactical_depth(b, depth)))
        pv = info.get("pv", [])[:remaining]
        full_moves.extend(pv_to_san(b, pv))

    score = evaluate_line(board, engine, depth, full_moves)
    tags = line_tags(board, full_moves)

    if prefix_moves:
        prefix_text = " ".join(prefix_moves)
        label = f"What if after {prefix_text} the response is {user_move}"
    else:
        label = f"What if {user_move}"

    return LineResult(
        label=label,
        moves=full_moves,
        score=score,
        tags=tags,
        source="hypothetical",
    )


def build_hypothetical_lines(
    board: chess.Board,
    engine: chess.engine.SimpleEngine,
    question: str,
    engine_lines: List[LineResult],
    depth: int,
    pv_plies: int,
    max_lines: int = 4,
) -> List[LineResult]:
    """For each move mentioned in `question`, build a hypothetical LineResult so
    the LLM has actual engine evaluation of the user's idea rather than
    speculating about lines the engine never showed it."""
    candidates = extract_hypothetical_candidates(question, board, engine_lines)
    if not candidates:
        return []

    existing: set = set()
    for line in engine_lines:
        for i in range(len(line.moves)):
            existing.add((tuple(line.moves[:i]), line.moves[i]))

    results: List[LineResult] = []
    for prefix, mv_san in candidates:
        if (tuple(prefix), mv_san) in existing:
            continue
        line = hypothetical_line(board, engine, prefix, mv_san, depth, pv_plies)
        if line is None:
            continue
        results.append(line)
        if len(results) >= max_lines:
            break
    return results


def format_moves_with_sides(board: chess.Board, moves: List[str]) -> str:
    """Annotate each move with the side that plays it, using PGN numbering.
    Example: 'Black: 6...Nf6 -> White: 7.Bc4 -> Black: 7...O-O -> White: 8.Bb3'.
    This is critical so the LLM never confuses which side plays a given move
    in an engine continuation."""

    turn = board.turn
    fullmove = board.fullmove_number
    parts: List[str] = []
    for mv in moves:
        side = cname(turn)
        if turn == chess.WHITE:
            parts.append(f"{side}: {fullmove}.{mv}")
        else:
            parts.append(f"{side}: {fullmove}...{mv}")
            fullmove += 1
        turn = not turn
    return " -> ".join(parts)


def format_line_for_prompt(board: chess.Board, line: LineResult) -> str:
    annotated = format_moves_with_sides(board, line.moves)
    tags = format_tags_for_prompt(board, line.tags)
    return f"{line.label}: {format_score(line.score)} | {annotated} | tags: {tags}"


def format_history_for_prompt(history: List[Dict[str, str]], max_items: int = 6) -> str:
    if not history:
        return "(none)"
    trimmed = history[-max_items:]
    lines = []
    for entry in trimmed:
        role = (entry.get("role") or "user").strip().lower()
        content = (entry.get("content") or "").strip()
        if not content:
            continue
        label = "Coach" if role in {"assistant", "coach"} else "User"
        lines.append(f"{label}: {content}")
    return "\n".join(lines) if lines else "(none)"


# --- Attention-model (AlphaZero-style bot) integration -----------------------
# The bot lives in ./alphazero with its own (torch) venv. We shell out to its
# explain_position.py to get the attention-weighted board state: the model's
# value, its top moves, and the squares its attention says most influenced the
# evaluation / the chosen move. This is model-grounded saliency that complements
# the deterministic facts below (idea from HEX-RL, arXiv:2112.08907).
_AZ_DIR = Path(__file__).resolve().parent / "alphazero"
_AZ_PYTHON = _AZ_DIR / ".venv" / "bin" / "python"
_AZ_SCRIPT = _AZ_DIR / "explain_position.py"
_AZ_ENABLED = os.getenv("CHESS_TEACHER_ATTENTION", "1") != "0"


def attention_report_json(board: chess.Board, timeout: float = 30.0) -> Optional[dict]:
    """Run the alphazero explainer subprocess and return its full JSON report
    (value, top moves, saliency lists, and per-square saliency maps), or None on
    any failure (graceful degradation)."""
    if not _AZ_ENABLED or not _AZ_PYTHON.exists() or not _AZ_SCRIPT.exists():
        return None
    try:
        proc = subprocess.run(
            [str(_AZ_PYTHON), str(_AZ_SCRIPT), "--fen", board.fen(), "--device", "cpu"],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(_AZ_DIR),
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    try:
        data = json.loads(proc.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return None
    if "error" in data:
        return None
    return data


def attention_model_block(board: chess.Board, timeout: float = 30.0) -> Optional[str]:
    """Return the attention model's ground-truth text block for ``board``, or None."""
    data = attention_report_json(board, timeout=timeout)
    if not data or "prompt_block" not in data:
        return None
    return data["prompt_block"]


def build_ground_truth_block(board: chess.Board, include_attention: bool = True) -> str:
    """Assemble every deterministic board-state section into one block. Order is
    chosen to put the most concrete (placement, attacks, defenses) before the more
    interpretive (tactical motifs, structural assessment)."""

    def bullets(items: List[str], empty_msg: str) -> str:
        if not items:
            return f"- {empty_msg}"
        return "\n".join(f"- {x}" for x in items)

    placement = piece_placement_summary(board)
    meta_text = bullets(position_meta(board), "(no special state)")
    attacks_text = bullets(attack_relations(board), "no pieces currently attack any enemy piece")
    vision_text = bullets(piece_vision_summary(board), "(no pieces)")
    defenses_text = bullets(defense_relations(board), "no pieces currently defend a friendly piece")
    pins_text = bullets(pin_descriptions(board), "no pieces are pinned")
    forks_text = bullets(existing_forks(board), "no pieces currently fork enemy targets")
    king_zone_text = bullets(king_zone_threats(board), "neither king has attacked squares adjacent to it")
    position_facts_text = bullets(position_facts(board), "no pieces under attack")
    strategic_text = bullets(positional_facts(board), "no notable structural features")

    return (
        f"Board state:\n{placement}\n\n"
        f"Position meta (check, castling, en passant, material):\n{meta_text}\n\n"
        f"Piece attacks (each piece → enemy pieces it currently attacks):\n{attacks_text}\n\n"
        f"Square control (each piece → EVERY square it attacks; this is the complete, "
        f"literal list — a piece sees ONLY these squares):\n{vision_text}\n\n"
        f"Piece defenses (each piece → friendly pieces defending it):\n{defenses_text}\n\n"
        f"Pinned pieces:\n{pins_text}\n\n"
        f"Existing forks on the board:\n{forks_text}\n\n"
        f"King-zone threats:\n{king_zone_text}\n\n"
        f"Position facts (tactical: hanging vs defended):\n{position_facts_text}\n\n"
        f"Strategic facts (positional structure):\n{strategic_text}"
        + _concepts_section(board)
        + (_attention_section(board) if include_attention else "")
    )


def _concepts_section(board: chess.Board) -> str:
    """Deterministic detected chess concepts (tactics, structures, plans, endgame,
    mates) from chess_concepts.py — verified from the board, safe for the LLM to cite."""
    if _concepts is None:
        return ""
    try:
        block = _concepts.format_concepts_for_prompt(board)
    except Exception:
        return ""
    if not block:
        return ""
    return "\n\n" + block


def _attention_section(board: chess.Board) -> str:
    """Optional trailing section with the attention model's read of the position."""
    block = attention_model_block(board)
    if not block:
        return ""
    return (
        "\n\nAttention model (neural bot) read — model-grounded, use to support the "
        "explanation but defer to the engine lines for best play:\n" + block
    )


def response_needs_rewrite(text: str) -> bool:
    if not text:
        return True
    sample = text.strip()
    if len(sample) < 60:
        return True
    lowered = sample.lower()
    if "candidate line" in lowered or "engine line" in lowered or "tags:" in lowered:
        return True
    list_lines = sum(
        1
        for line in sample.splitlines()
        if re.match(r"^\s*(\d+[\).]|[-*])\s+", line)
    )
    return list_lines >= 2


def basic_explain(
    board: chess.Board,
    prompt: str,
    engine_lines: List[LineResult],
    candidate_lines: List[LineResult],
) -> str:
    if not engine_lines:
        return "No engine lines available to explain the position."

    best = engine_lines[0]
    parts = []
    if best.moves:
        parts.append(f"Best move is {best.moves[0]} ({format_score(best.score)}).")
    else:
        parts.append(f"Best line score is {format_score(best.score)}.")

    if best.tags:
        first_tags = best.tags[0].get("tags", [])
        if first_tags:
            parts.append(f"Key idea: {', '.join(first_tags[:3])}.")

    if len(best.moves) > 1:
        line = " ".join(best.moves[:4])
        parts.append(f"Sample line: {line}.")

    prompt_moves = find_prompt_moves(prompt, board)
    if prompt_moves:
        parts.append(
            f"About {prompt_moves[0]}: it scores worse than the best line and"
            " allows counterplay."
        )

    return " ".join(parts)


def _cp_of(score: Dict[str, int]) -> int:
    m = score.get("mate")
    if m is not None:
        # Encode mate distance so a faster mate ranks strictly above a slower one
        # (mate in 1 must NOT read as equal to mate in 15), while staying far above
        # any centipawn evaluation.
        return (100000 - m) if m > 0 else (-100000 - m)
    return score.get("cp", 0)


def _landing_features(board: chess.Board, san: str):
    """Deterministic practical features of the position AFTER playing ``san``:
    whether the moved piece lands defended/loose, central, shields the king, develops,
    and whether it was under attack before (a rescue)."""
    b = board.copy()
    try:
        mv = parse_move(b, san)
    except ValueError:
        return None
    color = b.turn
    from_sq = mv.from_square
    moved = b.piece_at(from_sq)
    was_hanging = (
        moved is not None and moved.piece_type != chess.KING
        and static_exchange_eval(board, from_sq, not color) > 0
    )
    is_capture = board.is_capture(mv)
    captured_name = None
    if is_capture:
        if board.is_en_passant(mv):
            captured_name = "pawn"
        else:
            cap_piece = board.piece_at(mv.to_square)
            captured_name = chess.piece_name(cap_piece.piece_type) if cap_piece else None
    b.push(mv)
    to = mv.to_square
    piece = b.piece_at(to)
    if piece is None:
        return None
    # Can the opponent actually attack the landing square? Being undefended only
    # matters when the square is under (or can come under) enemy fire.
    opp_attackers = [s for s in b.attackers(not color, to) if b.piece_at(s)]
    attacked = len(opp_attackers) > 0
    hangs = static_exchange_eval(b, to, not color) > 0
    defenders = [s for s in b.attackers(color, to) if b.piece_at(s)]
    defender_types = {b.piece_at(s).piece_type for s in defenders}
    defender_names = sorted({chess.piece_name(pt) for pt in defender_types})
    solid_defenders = sorted({chess.piece_name(pt) for pt in defender_types
                              if pt in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK)})
    # 'solid' = held by a pawn/minor/rook (cheap, ideal); queen/king-only defence is weak.
    solid_defense = len(solid_defenders) > 0
    queen_only = bool(defenders) and not solid_defense
    f, r = chess.square_file(to), chess.square_rank(to)
    ksq = b.king(color)
    king_ring = set()
    if ksq is not None:
        kf, kr = chess.square_file(ksq), chess.square_rank(ksq)
        for df in (-1, 0, 1):
            for dr in (-1, 0, 1):
                nf, nr = kf + df, kr + dr
                if 0 <= nf <= 7 and 0 <= nr <= 7:
                    king_ring.add(chess.square(nf, nr))
    feats = {
        "from": from_sq,
        "to": to,
        "piece": chess.piece_name(piece.piece_type),
        "defended": len(defenders) > 0,
        "defender_names": defender_names,
        "solid_defenders": solid_defenders,
        "solid_defense": solid_defense,
        "queen_only": queen_only,
        "loose": len(defenders) == 0,
        "attacked": attacked,
        "hangs": hangs,
        "safe": not attacked,
        "is_capture": is_capture,
        "captured_name": captured_name,
        "central": 2 <= f <= 5 and 2 <= r <= 5,
        "shields_king": bool(set(b.attacks(to)) & king_ring),
        "develops": chess.square_rank(from_sq) in (0, 7)
        and moved is not None and moved.piece_type in (chess.KNIGHT, chess.BISHOP),
        "was_hanging": was_hanging,
    }
    # practicality score: solid defence and king shelter make a move easier/safer to play.
    score = 0.0
    # Safety is only a factor when the opponent can actually attack the landing square.
    # A piece that lands where nothing attacks it is perfectly safe whether or not it is
    # defended, so "undefended" must NOT be treated as a downside in that case.
    if attacked:
        if hangs:
            score -= 3            # the move leaves the piece en prise
        elif feats["solid_defense"]:
            score += 2            # attacked, but cheaply and solidly held
        elif feats["queen_only"]:
            score += 0.5          # held, but ties the queen to its defence
        else:
            score -= 1            # attacked and only loosely defended
    if feats["shields_king"]:
        score += 1
    if feats["develops"]:
        score += 1
    if feats["central"]:
        score += 0.5
    feats["practicality"] = score
    return feats


def _solidity_phrase(f) -> str:
    # Undefended is only a talking point when the opponent can attack the square.
    if not f["attacked"]:
        return f"safe on {chess.square_name(f['to'])} (nothing attacks it)"
    if f["hangs"]:
        return f"left hanging on {chess.square_name(f['to'])}, where it can be captured"
    if f["solid_defense"]:
        return f"defended by {' and '.join('the ' + n for n in f['solid_defenders'])}"
    if f["queen_only"]:
        return "held only by the queen (which ties the queen to its defence)"
    return f"attacked and only loosely defended on {chess.square_name(f['to'])}"


def practical_comparison(board: chess.Board, best_san: str, best_score: Dict[str, int],
                         alt_san: str, alt_score: Dict[str, int]) -> Optional[str]:
    """Practical, human-oriented comparison of two near-equal moves from deterministic
    board features. Scores each for solidity/king-shelter/development and recommends
    whichever is easier to play — which may be the engine's move OR the alternative."""
    if best_san == alt_san:
        return None
    fb = _landing_features(board, best_san)
    fa = _landing_features(board, alt_san)
    if not fb or not fa:
        return None

    parts: List[str] = []
    if fb["from"] == fa["from"] and fb["was_hanging"]:
        parts.append(
            f"Both {best_san} and {alt_san} rescue the {fb['piece']}, and the engine "
            f"rates them about equal ({format_score(best_score)} vs {format_score(alt_score)})."
        )
    else:
        parts.append(
            f"{alt_san} ({format_score(alt_score)}) is essentially as good as "
            f"{best_san} ({format_score(best_score)})."
        )

    # Recommend the more practical move (higher solidity/king-safety score).
    if fa["practicality"] > fb["practicality"] + 0.75:
        prac, prac_san, other, other_san = fa, alt_san, fb, best_san
    elif fb["practicality"] > fa["practicality"] + 0.75:
        prac, prac_san, other, other_san = fb, best_san, fa, alt_san
    else:
        parts.append("Both are sound and about equally practical — play whichever you understand better.")
        return " ".join(parts)

    virtues = [_solidity_phrase(prac)]
    if prac["shields_king"] and not other["shields_king"]:
        virtues.append("it also helps guard your king")
    if prac["develops"] and not other["develops"]:
        virtues.append("it's a natural developing square")
    parts.append(
        f"{prac_san} is the more practical, solid choice: the {prac['piece']} is "
        + ", ".join(virtues) + "."
    )

    if other["safe"]:
        # No real safety downside: the recommendation is only a mild practical lean,
        # so don't invent an "undefended" drawback for a piece nothing can touch.
        if other["is_capture"] and other["captured_name"]:
            parts.append(
                f"{other_san} is also strong \u2014 it wins the {other['captured_name']} and the "
                f"{other['piece']} is safe on {chess.square_name(other['to'])}; {prac_san} is just "
                f"a touch easier to follow up."
            )
        else:
            parts.append(
                f"{other_san} is also fine and perfectly safe; {prac_san} is just a touch easier "
                f"to follow up."
            )
    else:
        downside = _solidity_phrase(other)
        extra = " though it sits on a slightly more active square" if other["central"] and not prac["central"] else ""
        parts.append(f"{other_san} is fine too, but there the {other['piece']} is {downside}{extra}.")
    parts.append(f"With near-equal evaluations, {prac_san} is the easier move to play for a human.")
    return " ".join(parts)


def _near_equal_alt(engine_lines: List[LineResult], margin_cp: int = 60):
    """Return the best alternative first-move line within margin of the top line.
    Returns None when the top line is a forced mate: there is no 'practical choice'
    to weigh once a mate is on the board, and a slower mate is not an equal option."""
    if not engine_lines or not engine_lines[0].moves:
        return None
    best = engine_lines[0]
    if best.score.get("mate") is not None:
        return None
    best_cp = _cp_of(best.score)
    top_move = best.moves[0]
    for line in engine_lines[1:]:
        if not line.moves or line.moves[0] == top_move:
            continue
        if line.score.get("mate") is not None:
            continue
        if best_cp - _cp_of(line.score) <= margin_cp:
            return line
    return None


def _pin_on_square(board: chess.Board, square: int) -> Optional[str]:
    """If the piece on ``square`` is pinned (absolute to the king, or relative to a more
    valuable piece), return a natural phrase naming the pinned piece, what it is pinned
    to, and the pinner — using the verified concept detector. None if not pinned."""
    if _concepts is None:
        return None
    name = chess.square_name(square)

    def pn(sq_name: str) -> str:
        p = board.piece_at(chess.parse_square(sq_name))
        return chess.piece_name(p.piece_type) if p else "piece"

    try:
        for c in _concepts.detect_pins(board):
            if c.name == "Pin (relative)" and len(c.squares) == 3 and c.squares[1] == name:
                slider, pinned, behind = c.squares
                return (f"the {pn(pinned)} on {pinned} is pinned to the more valuable "
                        f"{pn(behind)} on {behind} by the {pn(slider)} on {slider}")
            if c.name == "Pin (absolute)" and c.squares and c.squares[0] == name:
                return f"the {pn(name)} on {name} is pinned to the king"
    except Exception:
        return None
    return None


def format_move_verdict(
    board: chess.Board,
    move_line: LineResult,
    engine_lines: List[LineResult],
) -> str:
    """Deterministic, correct plain-English verdict for a specific move, built ONLY
    from labeled engine moves + verified per-move tags + the evaluation. Avoids the
    LLM misreading squares/targets when explaining 'why not X'."""
    side = cname(board.turn)
    opp = cname(not board.turn)
    labeled = format_moves_with_sides(board, move_line.moves[:6])
    h_score = format_score(move_line.score)
    h_cp = _cp_of(move_line.score)
    first = move_line.moves[0] if move_line.moves else "the move"
    reply = move_line.moves[1] if len(move_line.moves) > 1 else None

    # Immediate refutation, drawn straight from the labeled line + verified tags.
    cap_tag = None
    for t in (move_line.tags or []):
        if t.get("ply") == 2:
            caps = [x for x in t.get("tags", []) if "captures" in x]
            cap_tag = caps[0] if caps else None
            break

    parts: List[str] = []
    best = engine_lines[0] if engine_lines else None
    is_best = bool(best and best.moves and best.moves[0] == first)
    if is_best:
        parts.append(f"{first} is actually the engine's top move here ({h_score} for {side}).")
        parts.append(f"Main line: {labeled}.")
        alt = _near_equal_alt(engine_lines)
        if alt is not None:
            pc = practical_comparison(board, first, move_line.score, alt.moves[0], alt.score)
            if pc:
                parts.append(pc)
    elif best and best.moves:
        b_score = format_score(best.score)
        delta = _cp_of(best.score) - h_cp  # >0 => the move is worse for the side to move
        if delta >= 150:
            verdict = f"is clearly worse for {side}"
        elif delta >= 50:
            verdict = f"is a bit worse for {side} than the top move"
        elif delta <= -50:
            verdict = "is at least as good as the engine's top move"
        else:
            verdict = "is about as good as the top move"
        parts.append(f"{first} {verdict}.")
        # Name the concept behind the refutation: if the moved piece was pinned,
        # moving it is what drops material (don't just recite the line).
        pin_reason = None
        try:
            pin_reason = _pin_on_square(board, parse_move(board, first).from_square)
        except Exception:
            pin_reason = None
        if reply and cap_tag and delta >= 50:
            if pin_reason:
                parts.append(
                    f"{first} moves a pinned piece — {pin_reason} — so after {first}, "
                    f"{opp} plays {reply} ({cap_tag}) and {side} comes out worse."
                )
            else:
                parts.append(
                    f"The point: after {first}, {opp} plays {reply} ({cap_tag}), so {side} "
                    f"comes out worse."
                )
        elif reply and cap_tag:
            parts.append(f"After {first}, {opp} replies {reply} ({cap_tag}).")
        parts.append(
            f"The engine puts this at {h_score} for {side}, versus {b_score} after the "
            f"best move {best.moves[0]}. Full line: {labeled}."
        )
        # If the asked move is essentially as good, add the practical comparison
        # (never for mates -- a slower/other mate is not a 'practical' equal choice).
        if delta <= 60 and best.score.get("mate") is None and move_line.score.get("mate") is None:
            pc = practical_comparison(board, best.moves[0], best.score, first, move_line.score)
            if pc:
                parts.append(pc)
    else:
        # No reference line supplied — judge from the move's own evaluation.
        if h_cp <= -150:
            verdict = f"is bad for {side} — it drops material"
        elif h_cp >= 150:
            verdict = f"is strong for {side}"
        else:
            verdict = f"keeps roughly balanced play for {side}"
        parts.append(f"{first} {verdict} ({h_score} for {side}).")
        if reply and cap_tag:
            parts.append(f"The point: after {first}, {opp} plays {reply} ({cap_tag}).")
        parts.append(f"Full line: {labeled}.")
    return " ".join(parts)


_MOVE_TOKEN_RE = re.compile(
    r"\b(O-O-O|O-O|[NBRQK][a-h]?[1-8]?x?[a-h][1-8](?:=[NBRQ])?[+#]?|"
    r"[a-h]x[a-h][1-8](?:=[NBRQ])?[+#]?|[a-h][1-8][a-h][1-8][nbrq]?)\b"
)


def _illegal_move_note(board: chess.Board, question: str) -> Optional[str]:
    """If the question references a move-like token that is NOT legal here, return a
    short clarification (e.g. a self-capture like 'Nxe4' onto your own piece), instead
    of letting the LLM invent an explanation for an impossible move."""
    for m in _MOVE_TOKEN_RE.finditer(question):
        tok = m.group(0)
        try:
            parse_move(board, tok)
            continue  # it's legal; handled elsewhere
        except ValueError:
            pass
        sqm = re.search(r"([a-h][1-8])(?:=[NBRQ])?[+#]?$", tok)
        if sqm and "x" in tok:
            target = chess.parse_square(sqm.group(1))
            p = board.piece_at(target)
            if p is not None and p.color == board.turn:
                return (
                    f"{tok} isn't possible — {sqm.group(1)} has your own "
                    f"{chess.piece_name(p.piece_type)}, so there is nothing to capture there."
                )
            if p is None:
                return f"{tok} isn't legal here — there is no enemy piece on {sqm.group(1)} to capture."
        return f"{tok} isn't a legal move in this position."
    return None


def _best_move_question(q: str) -> bool:
    ql = q.lower()
    keys = (
        "best move", "best continuation", "strongest move", "what should i play",
        "what should black play", "what should white play", "what to play",
        "what move should", "what do i play", "what is the best", "what's the best",
        "how should i continue", "what should i do", "what's best",
    )
    return any(k in ql for k in keys)


def _named_mate_for_move(board: chess.Board, mv: chess.Move) -> Optional[str]:
    """If ``mv`` delivers a checkmate the concept engine recognises by name
    (Anastasia's, Arabian, Boden's, smothered, back-rank, ...), return that name;
    otherwise None. Verified from the board, so it's safe to state outright."""
    if _concepts is None:
        return None
    try:
        prefix = mv.uci() + " "
        for c in _concepts.detect_checkmate_patterns(board):
            if c.detail.startswith(prefix) and c.name and c.name != "Checkmate":
                return c.name
    except Exception:
        return None
    return None


def describe_best_move(board: chess.Board, engine_lines: List[LineResult]) -> Optional[str]:
    """Deterministic, correct description of the engine's best move: mechanics are read
    from the board (capture / retreat-to-safety / develop / castle / check), never from
    the LLM — so it can't invent 'captures' or the wrong owner."""
    if not engine_lines or not engine_lines[0].moves:
        return None
    best = engine_lines[0]
    san = best.moves[0]
    b = board.copy()
    try:
        mv = parse_move(b, san)
    except ValueError:
        return None
    side = cname(board.turn)
    piece = b.piece_at(mv.from_square)
    pn = chess.piece_name(piece.piece_type) if piece else "piece"
    from_sq = chess.square_name(mv.from_square)
    to_sq = chess.square_name(mv.to_square)
    parts = [f"The best move is {san} ({format_score(best.score)} for {side})."]

    if piece and piece.piece_type == chess.KING and \
            abs(chess.square_file(mv.from_square) - chess.square_file(mv.to_square)) >= 2:
        parts.append(f"{side} castles — getting the king to safety and connecting the rooks.")
    elif mv.promotion:
        parts.append(f"The pawn promotes on {to_sq}.")
    elif b.is_capture(mv):
        if b.is_en_passant(mv):
            parts.append(f"It captures a pawn en passant on {to_sq}.")
        else:
            cap = b.piece_at(mv.to_square)
            capn = chess.piece_name(cap.piece_type) if cap else "piece"
            gain = static_exchange_eval(board, mv.to_square, board.turn)
            tail = ", winning material" if gain > 0 else ""
            parts.append(f"It captures the {capn} on {to_sq}{tail}.")
    else:
        # non-capture: was the moved piece itself under attack (a save)?
        was_hanging = piece is not None and piece.piece_type != chess.KING and \
            static_exchange_eval(board, mv.from_square, not board.turn) > 0
        if was_hanging:
            parts.append(
                f"It moves the {pn} from {from_sq} — which was under attack — to safety "
                f"on {to_sq} (this is a retreat, not a capture)."
            )
        elif piece and piece.piece_type == chess.PAWN:
            parts.append(f"It advances the pawn to {to_sq}.")
        elif piece and piece.piece_type in (chess.KNIGHT, chess.BISHOP) and \
                chess.square_rank(mv.from_square) in (0, 7):
            parts.append(f"It develops the {pn} to {to_sq}.")
        else:
            parts.append(f"It repositions the {pn} to {to_sq}.")
    if b.gives_check(mv):
        b_after = board.copy()
        b_after.push(mv)
        if b_after.is_checkmate():
            named = _named_mate_for_move(board, mv)
            parts.append(f"It's checkmate \u2014 {named}." if named else "It's checkmate.")
        else:
            parts.append("It comes with check.")
    if len(best.moves) > 1:
        parts.append(f"Main line: {format_moves_with_sides(board, best.moves[:4])}.")
    # Note: no automatic practicality/near-equal-alternative commentary here \u2014 a
    # 'best move' answer should stay focused. That comparison is reserved for explicit
    # 'why not X' verdict questions (format_move_verdict).
    return " ".join(parts)


def _why_move_question(board: chess.Board, q: str) -> bool:
    """True for 'why is this move good / why trade / what's the idea' questions
    (as opposed to 'why NOT X' or 'is X bad', which are verdict questions)."""
    ql = q.lower()
    if "why not" in ql:
        return False
    # 'why can't I play X', 'can I play X', 'is X bad/losing' etc. are VERDICT
    # questions about a (usually non-best) move -- never assume the move is good.
    if any(w in ql for w in (" bad", " worse", "blunder", "mistake", "instead of",
                             "can't", "cant", "cannot", "can i", "unable", "illegal",
                             "lose", "loses", "losing", "hang", "drop")):
        return False
    has_why = ("why" in ql or "idea" in ql or "purpose" in ql or "point of" in ql
               or "reason" in ql)
    if not has_why:
        return False
    move_kw = any(k in ql for k in (
        "trade", "captur", "take", "sacrific", " sac", "give up", "this move",
        "that move", "the move", "best move", "develop", "play"))
    return move_kw or bool(find_prompt_moves(q, board))


def describe_why_move(board: chess.Board, move_san: str,
                      engine_lines: List[LineResult]) -> Optional[str]:
    """Deterministic 'why is this move good' explanation: material/trade outcome,
    bishop-pair swing (computed before vs after the trade), saves/develops/check, and
    the eval + main line. Built only from the board + engine line, so it can't invent
    captures, piece placements, or purposes."""
    b = board.copy()
    try:
        mv = parse_move(b, move_san)
    except ValueError:
        return None
    color = board.turn
    side = cname(color)
    piece = b.piece_at(mv.from_square)
    if piece is None:
        return None
    pn = chess.piece_name(piece.piece_type)
    to = chess.square_name(mv.to_square)
    is_cap = b.is_capture(mv)
    captured_pt = None
    if is_cap and not b.is_en_passant(mv):
        cp = b.piece_at(mv.to_square)
        captured_pt = cp.piece_type if cp else None
    was_hanging = (piece.piece_type != chess.KING
                   and static_exchange_eval(board, mv.from_square, not color) > 0)

    line = None
    for l in engine_lines:
        if l.moves and l.moves[0] == move_san:
            line = l
            break
    score_txt = format_score(line.score) if line else None

    def bcount(bd, c):
        return len(bd.pieces(chess.BISHOP, c))

    my_before, opp_before = bcount(board, color), bcount(board, not color)
    b2 = board.copy()
    b2.push(mv)
    if line and len(line.moves) >= 2:  # apply the forced recapture/reply
        try:
            b2.push(parse_move(b2, line.moves[1]))
        except ValueError:
            pass
    my_after, opp_after = bcount(b2, color), bcount(b2, not color)

    parts: List[str] = []
    if captured_pt is not None:
        capn = chess.piece_name(captured_pt)
        see = static_exchange_eval(board, mv.to_square, color)
        if piece.piece_type == chess.KNIGHT and captured_pt == chess.BISHOP:
            parts.append(f"{move_san} trades the knight for the {capn}.")
        elif piece.piece_type == chess.BISHOP and captured_pt == chess.KNIGHT:
            parts.append(f"{move_san} trades the bishop for the {capn}.")
        elif see > 0:
            parts.append(f"{move_san} captures the {capn} on {to}, winning material.")
        else:
            parts.append(f"{move_san} captures the {capn} on {to} (an even trade).")
    elif was_hanging:
        parts.append(f"{move_san} brings the {pn} to safety on {to} (it was under attack).")
    elif piece.piece_type in (chess.KNIGHT, chess.BISHOP) and chess.square_rank(mv.from_square) in (0, 7):
        parts.append(f"{move_san} develops the {pn} to {to}.")
    else:
        parts.append(f"{move_san} is the engine's top choice here.")

    # The key positional point for many trades: the bishop pair.
    if my_before == 2 and my_after == 2 and opp_before == 2 and opp_after == 1:
        parts.append(
            f"The main point is the bishop pair: after the trade {side} keeps both "
            f"bishops while the opponent is left with just one — a lasting positional plus."
        )
    elif opp_before == 2 and opp_after == 1 and my_after >= my_before and my_after >= 2:
        parts.append(f"It also leaves {side} with two bishops against one.")

    if b2.is_check():
        parts.append("The follow-up also comes with tempo (a check).")

    if line and score_txt:
        cont = format_moves_with_sides(board, line.moves[:4])
        parts.append(f"The engine rates it {score_txt} for {side}. Main line: {cont}.")
    return " ".join(parts)


def _try_move_verdict(
    board: chess.Board,
    question: str,
    engine_lines: List[LineResult],
    extra_lines: List[LineResult],
    model: str,
) -> Optional[str]:
    """If the question is about a specific move we have an engine line for, return a
    deterministic (hallucination-proof) verdict, lightly polished by the LLM with a
    strict 'change no facts' instruction (falls back to the raw verdict)."""
    prompt_moves = find_prompt_moves(question, board)
    if not prompt_moves:
        return None
    target = set(prompt_moves)
    match = None
    # Search supplied candidate/hypothetical lines first, then the engine lines
    # (the asked move may already BE an engine top line, which build_hypothetical_lines
    # skips — that previously fell through to the hallucinating LLM path).
    for line in list(extra_lines) + list(engine_lines):
        if line.moves and line.moves[0] in target:
            match = line
            break
    if match is None:
        return None
    # Return the deterministic verdict directly: it is built only from labeled engine
    # moves + verified tags + evaluation, so it cannot mis-attribute or invent motifs.
    # (An LLM 'polish' pass was tried and reliably re-introduced errors, so it's off.)
    return format_move_verdict(board, match, engine_lines)


def _verified_answer(board: chess.Board, question: str,
                     engine_lines: List[LineResult],
                     extra_lines: List[LineResult]) -> Optional[str]:
    """The deterministic, engine+detector-grounded answer for a move question (why a
    move is good, a 'why not X' verdict, or the best move), or None if the question
    isn't a specific-move question. This is CORRECT by construction and is used both
    as the LLM's authoritative input and as the guaranteed fallback."""
    if _why_move_question(board, question):
        pm = find_prompt_moves(question, board)
        best_san = engine_lines[0].moves[0] if engine_lines and engine_lines[0].moves else None
        mv_san = pm[0] if pm else best_san
        # Positive 'why it's good' only when the named move IS the best move; any other
        # named move defers to the verdict so we never rationalize a blunder.
        if mv_san and (not pm or mv_san == best_san):
            why = describe_why_move(board, mv_san, engine_lines)
            if why:
                return why
    verdict = _try_move_verdict(board, question, engine_lines, extra_lines, "")
    if verdict is not None:
        return verdict
    if _best_move_question(question) and engine_lines:
        bm = describe_best_move(board, engine_lines)
        if bm is not None:
            return bm
    return None


def _allowed_move_tokens(board: chess.Board, lines: List[LineResult],
                         verified: str) -> set:
    """Every move the LLM is allowed to name: legal SANs in the position, plus moves
    that appear in the supplied engine/candidate lines and the verified analysis.
    Normalised (check/mate/annotation suffixes stripped)."""
    def norm(s: str) -> str:
        return s.rstrip("+#!?")
    allowed = {norm(board.san(m)) for m in board.legal_moves}
    for l in lines or []:
        for mv in (l.moves or []):
            allowed.add(norm(mv))
    for tok in _MOVE_TOKEN_RE.findall(verified or ""):
        allowed.add(norm(tok))
    return allowed


_PIECE_WORD_TO_TYPE = {
    "pawn": chess.PAWN, "knight": chess.KNIGHT, "bishop": chess.BISHOP,
    "rook": chess.ROOK, "queen": chess.QUEEN, "king": chess.KING,
}
_PIECE_LETTER_TO_TYPE = {
    "N": chess.KNIGHT, "B": chess.BISHOP, "R": chess.ROOK,
    "Q": chess.QUEEN, "K": chess.KING,
}
_PIECE_ON_SQ_CLAIM = re.compile(
    r"\b(white|black)?\s*(pawn|knight|bishop|rook|queen|king)\s+on\s+([a-h][1-8])\b")
_ATTACK_CLAIM = re.compile(
    r"\bon\s+([a-h][1-8])\b[^.;\n]{0,60}?"
    r"\b(?:attacks?|attacking|controls?|controlling|hits?|targets?|covers?|defends?|defending)\b"
    r"[^.;\n]{0,40}?\b([a-h][1-8])\b")
_SIMPLE_PIECE_MOVE = re.compile(r"^([NBRQK])([a-h][1-8])$")


def _faithfulness_violations(board: chess.Board, text: str,
                             lines: List[LineResult], verified: str) -> List[str]:
    """Specific, human-readable ways ``text`` contradicts the board, for the claim
    classes we can check reliably:
      * invented moves (a move token not legal here and not in the supplied lines),
      * wrong piece type/colour on a square ('White pawn on g7' when g7 is Black's),
      * false 'the piece on A attacks/controls B' relations (A does not see B).
    Empty list => faithful on these classes. Drives the reject/repair guard."""
    if not text or len(text.strip()) < 40:
        return ["the answer is empty or too short"]
    out: List[str] = []
    allowed = _allowed_move_tokens(board, lines, verified)
    for tok in _MOVE_TOKEN_RE.findall(text):
        core = tok.rstrip("+#!?")
        if core in allowed:
            continue
        # 'Bg4' / 'Rd1' in prose usually means the piece STANDING on that square, not a
        # move — allow it when a piece of that type is actually there.
        m = _SIMPLE_PIECE_MOVE.match(core)
        if m:
            pc = board.piece_at(chess.parse_square(m.group(2)))
            if pc and pc.piece_type == _PIECE_LETTER_TO_TYPE[m.group(1)]:
                continue
        out.append(f"it names a move that isn't available here: '{tok}'")
    low = text.lower()
    for m in _PIECE_ON_SQ_CLAIM.finditer(low):
        color_word, ptype, sqname = m.group(1), m.group(2), m.group(3)
        pc = board.piece_at(chess.parse_square(sqname))
        if pc is None:
            out.append(f"it says there is a {ptype} on {sqname}, but that square is empty")
        elif pc.piece_type != _PIECE_WORD_TO_TYPE[ptype]:
            out.append(f"it calls {sqname} a {ptype}, but {sqname} holds a {chess.piece_name(pc.piece_type)}")
        elif color_word and pc.color != (chess.WHITE if color_word == "white" else chess.BLACK):
            out.append(f"it calls the {chess.piece_name(pc.piece_type)} on {sqname} {color_word}, "
                       f"but it is {cname(pc.color).lower()}'s")
    for m in _ATTACK_CLAIM.finditer(low):
        a_sq, b_sq = chess.parse_square(m.group(1)), chess.parse_square(m.group(2))
        if board.piece_at(a_sq) is None:
            continue
        if b_sq not in board.attacks(a_sq):
            out.append(f"it claims the piece on {m.group(1)} attacks/controls "
                       f"{m.group(2)}, but it does not")
    seen = set()
    uniq = []
    for v in out:
        if v not in seen:
            seen.add(v)
            uniq.append(v)
    return uniq


def _guarded_generate(board: chess.Board, base_prompt: str, model: str,
                      lines: List[LineResult], verified: Optional[str] = None,
                      max_new_tokens: int = 384, temperature: float = 0.3,
                      tries: int = 3) -> Optional[str]:
    """Generate, then VERIFY the output against the board. On a checkable false claim,
    re-prompt with the exact false statements quoted as forbidden (a self-repair loop).
    Returns the first faithful answer; if every attempt fails, returns ``verified`` when
    supplied, else None (caller falls back)."""
    prompt = base_prompt
    temp = temperature
    for _ in range(max(1, tries)):
        try:
            out = (hf_generate(prompt, model=model, max_new_tokens=max_new_tokens,
                               temperature=temp) or "").strip()
        except Exception:
            break
        violations = _faithfulness_violations(board, out, lines, verified or "")
        if not violations:
            return out
        vlist = "\n".join(f"- {v}" for v in violations[:6])
        prompt = (
            base_prompt
            + "\n\nYOUR PREVIOUS DRAFT CONTAINED STATEMENTS THAT CONTRADICT THE BOARD:\n"
            + vlist
            + "\n\nRewrite the answer so NONE of these appear and every claim matches the "
            "facts given above. Do not introduce any new move, piece, or relationship."
        )
        temp = max(0.0, temp - 0.1)
    return verified


def _enrich_verified(board: chess.Board, question: str, verified: str,
                     engine_lines: List[LineResult], candidate_lines: List[LineResult],
                     model: str, include_attention: bool = True) -> str:
    """Hand the VERIFIED analysis to the LLM to phrase naturally and add only verified
    positional context. Guard the result; fall back to the verified text on any
    failure so correctness never regresses."""
    try:
        ground_truth = build_ground_truth_block(board, include_attention)
        engine_text = "\n".join(
            f"{i + 1}) {format_score(l.score)}: {format_moves_with_sides(board, l.moves)}"
            for i, l in enumerate(engine_lines)
        )
        enrich_prompt = (
            "You are a chess coach explaining to a student. Below is a VERIFIED, correct "
            "analysis (from a strong engine plus verified pattern detectors). Rewrite it "
            "into a clear, natural explanation that DIRECTLY answers the question.\n"
            "RULES (strict \u2014 you will be checked against the board):\n"
            "- Base every factual claim ONLY on the VERIFIED ANALYSIS and the ground "
            "truth below. If a detail is not stated there, do NOT state it.\n"
            "- A piece's colour, type and square must match 'Board state' exactly. Never "
            "say a pawn/piece belongs to a side unless Board state shows it (e.g. do not "
            "call a Black pawn White's).\n"
            "- Only say a piece attacks/controls/defends a square or piece if that appears "
            "in 'Square control' or 'Piece attacks/defenses'. A piece sees ONLY the "
            "squares listed for it in 'Square control'.\n"
            "- Do NOT introduce any move, capture, pin, fork, skewer, checkmate, passed "
            "pawn, or promotion that is not already in the verified analysis or ground "
            "truth.\n"
            "- Keep every evaluation, move, result, and named pattern exactly as given, "
            "and explain the WHY using the concept named in the verified analysis (e.g. "
            "the pin or the mate pattern), defining that pattern briefly if useful.\n"
            "- Mention at most one short line (<=4 plies). Be concise: 1-3 short "
            "paragraphs, no headings, and never mention 'verified analysis', 'ground "
            "truth', or 'square control'.\n\n"
            f"Side to move: {cname(board.turn)}\n"
            f"User question: {question}\n\n"
            f"=== VERIFIED ANALYSIS (authoritative \u2014 base your answer on this) ===\n{verified}\n\n"
            f"=== Ground truth (added context only) ===\n{ground_truth}\n\n"
            f"Engine top lines:\n{engine_text or '(none)'}\n"
        )
        out = _guarded_generate(board, enrich_prompt, model,
                                list(engine_lines) + list(candidate_lines), verified)
        if out:
            return out
    except Exception:
        pass
    return verified


def llm_explain(
    board: chess.Board,
    prompt: str,
    engine_lines: List[LineResult],
    candidate_lines: List[LineResult],
    model: str,
    include_attention: bool = True,
) -> str:
    # Move questions: get the VERIFIED analysis, then let the LLM phrase it richly with
    # a faithfulness guard + fallback (no more terse deterministic short-circuit).
    verified = _verified_answer(board, prompt, engine_lines, candidate_lines)
    if verified is not None:
        return _enrich_verified(board, prompt, verified, engine_lines,
                                candidate_lines, model, include_attention)
    illegal = _illegal_move_note(board, prompt)
    if illegal is not None:
        return illegal
    engine_text = "\n".join(
        f"{idx + 1}) {format_score(line.score)}: {format_moves_with_sides(board, line.moves)}"
        for idx, line in enumerate(engine_lines)
    )

    candidate_text = "\n".join(format_line_for_prompt(board, line) for line in candidate_lines)
    ground_truth = build_ground_truth_block(board, include_attention)
    anchor = best_move_anchor(board, engine_lines)
    side_name = cname(board.turn)
    opponent_name = cname(not board.turn)

    expl_prompt = (
        "You are a chess coach. ANSWER THE USER'S QUESTION directly and specifically.\n\n"
        "USER QUESTION: {question}\n\n"
        "HOW TO ANSWER: Your FIRST sentence must directly answer the question asked. "
        "Then support it with the verified facts below. Match your focus to the question:\n"
        "- 'what opening / what is this position from' -> name it from the 'Opening:' line "
        "under Chess concepts detected (or say it is not identified); never guess.\n"
        "- 'why can't I play X' / 'is X good/bad' -> evaluate THAT move using the Engine "
        "top lines and Candidate lines, compare scores, and state what refutes it.\n"
        "- 'what should I play' / 'what should I think about' / plans -> give the main "
        "plan(s) from the Strategic facts and Chess concepts detected; you may cite the "
        "engine's top move as the concrete step.\n"
        "- any other positional question -> answer it from the ground truth below.\n"
        "Do NOT default to explaining the engine's best move unless the question asks "
        "what to play or why a move is best.\n"
        "Reference (engine's current top choice — mention only if relevant): {anchor}\n"
        "Do NOT propose moves for the other side first. Do NOT "
        "invent moves; only reference moves that appear in the Engine top lines or "
        "Candidate lines below. Do NOT claim a piece attacks another piece unless "
        "that relationship appears in the Piece attacks section. Do NOT claim a piece "
        "is on a square unless it appears in Board state.\n"
        "SIDES IN ENGINE LINES: each move in every engine/candidate line is explicitly "
        "labeled 'White:' or 'Black:'. The first move belongs to {side} ({side} is to "
        "move); the second move belongs to {opponent}; and so on alternating. Do NOT "
        "claim {side} plays a move that is labeled '{opponent}:' or vice versa. When "
        "you mention a move from a line, attribute it to whichever side the label "
        "shows — for example, a move labeled 'White: Bc4' is WHITE's bishop, not "
        "Black's, even when {side} = Black.\n"
        "When the position has structural character, weave in positional themes from "
        "the Strategic facts: pawn structure, color complexes, bishop trades, open "
        "files, outposts, king safety, and any recognized opening structure (Sicilian "
        "Dragon, French, IQP, King's Indian, Hedgehog, etc.).\n"
        "GROUND TRUTH: Every section below the 'Ground truth' header is computed "
        "exactly. Only call a piece 'hanging' if it appears as HANGING in Position "
        "facts. Pieces listed as DEFENDED are NOT hanging. Do not contradict these "
        "facts.\n"
        "OPENING: Only name the opening or variation if an 'Opening:' line appears in "
        "the Chess concepts detected section, and use that exact name. If none is given, "
        "do NOT guess or name any opening.\n"
        "CONCEPTS: You may cite items in the 'Chess concepts detected' section verbatim; "
        "do not invent concepts, structures, or motifs that are not listed.\n"
        "MOVE PURPOSE: Do NOT invent why a move is played (pin, fork, 'opens lines', "
        "'attacks the queen') unless it appears in the tags, the Piece attacks section, "
        "or the Chess concepts. When you cite a move from a line, attribute it to the "
        "side shown in its 'White:'/'Black:' label — never the wrong side.\n"
        "STYLE: Write naturally, as a coach talking to a student. Do NOT mention "
        "'ground truth', 'Chess concepts detected', 'the section', or say facts were "
        "'detected/verified' — just state them as chess knowledge.\n"
        "Avoid listing full move sequences or tags; mention at most one short line (up to 4 plies).\n"
        "Keep the answer focused and practical in 2-4 short paragraphs.\n\n"
        "Position FEN: {fen}\n"
        "Side to move: {side}\n"
        "User question: {question}\n\n"
        "=== Ground truth ===\n{ground_truth}\n\n"
        "Engine top lines (score from side to move, positive is better):\n{engine}\n\n"
        "Candidate lines with tags:\n{candidates}\n"
    ).format(
        anchor=anchor,
        side=side_name,
        opponent=opponent_name,
        fen=board.fen(),
        question=prompt,
        ground_truth=ground_truth,
        engine=engine_text or "(none)",
        candidates=candidate_text or "(none)",
    )

    # Generate with the faithfulness guard + self-repair loop; fall back to the
    # deterministic factual summary if the model can't produce a faithful answer.
    response = _guarded_generate(board, expl_prompt, model,
                                 list(engine_lines) + list(candidate_lines), verified=None)
    if not response or response_needs_rewrite(response):
        return basic_explain(board, prompt, engine_lines, candidate_lines)
    return response


def llm_followup(
    board: chess.Board,
    prompt: str,
    question: str,
    history: List[Dict[str, str]],
    engine_lines: List[LineResult],
    candidate_lines: List[LineResult],
    model: str,
    hypothetical_lines: Optional[List[LineResult]] = None,
    include_attention: bool = True,
) -> str:
    # Move questions: VERIFIED analysis -> LLM enrichment with guard + fallback.
    extra = list(hypothetical_lines or []) + list(candidate_lines)
    verified = _verified_answer(board, question, engine_lines, extra)
    if verified is not None:
        return _enrich_verified(board, question, verified, engine_lines, extra,
                                model, include_attention)
    illegal = _illegal_move_note(board, question)
    if illegal is not None:
        return illegal

    engine_text = "\n".join(
        f"{idx + 1}) {format_score(line.score)}: {format_moves_with_sides(board, line.moves)}"
        for idx, line in enumerate(engine_lines)
    )
    candidate_text = "\n".join(format_line_for_prompt(board, line) for line in candidate_lines)
    hypothetical_text = "\n".join(
        format_line_for_prompt(board, line) for line in (hypothetical_lines or [])
    )
    history_text = format_history_for_prompt(history)
    ground_truth = build_ground_truth_block(board, include_attention)
    anchor = best_move_anchor(board, engine_lines)
    side_name = cname(board.turn)
    opponent_name = cname(not board.turn)

    follow_prompt = (
        "You are a chess coach continuing a conversation.\n"
        "Answer the follow-up question with explanation only.\n\n"
        "{anchor}\n\n"
        "TASK: Answer the follow-up about THIS position ({side} to move). Do NOT "
        "propose moves for the other side first. Do NOT invent moves; only reference "
        "moves that appear in the Engine top lines, Candidate lines, or Hypothetical "
        "lines below. Do NOT claim a piece attacks another piece unless that "
        "relationship appears in the Piece attacks section. Do NOT claim a piece is "
        "on a square unless it appears in Board state.\n"
        "SIDES IN ENGINE LINES: each move is explicitly labeled 'White:' or 'Black:'. "
        "The first move belongs to {side}; the second to {opponent}; alternating. Do "
        "NOT claim {side} plays a move that is labeled '{opponent}:' or vice versa.\n"
        "HYPOTHETICAL LINES: when the user asks 'what about move X' or 'isn't X bad', "
        "the Hypothetical lines section shows the engine's response IF that move were "
        "played. Use ONLY that line as your factual basis. Every move is labeled "
        "'White:' or 'Black:' with its move number — attribute each move to EXACTLY that "
        "side (e.g. 'Black: 8...d5' means BLACK played d5; never say White played it). "
        "Lead with the IMMEDIATE refutation (usually White's very next reply, e.g. the "
        "recapture) and the final evaluation; that is what makes the move good or bad. "
        "Do NOT narrate the deep tail move-by-move.\n"
        "MOVE PURPOSE: Do NOT invent why a move is played (pin, fork, 'opens lines', "
        "'creates a threat', 'attacks the queen') unless that appears in the tags, the "
        "Piece attacks section, or the Chess concepts. If you don't know a move's "
        "purpose, just state whose move it is and the resulting evaluation.\n"
        "When the question is structural/strategic (plans, piece trades, pawn breaks, "
        "long-term assessment), draw on the Strategic facts.\n"
        "GROUND TRUTH: Every section below the 'Ground truth' header is computed "
        "exactly. Only call a piece 'hanging' if it appears as HANGING in Position "
        "facts. Pieces listed as DEFENDED are NOT hanging. Do not contradict these "
        "facts.\n"
        "OPENING: Only name the opening or variation if an 'Opening:' line appears in "
        "the Chess concepts detected section, and use that exact name. If none is given, "
        "do NOT guess or name any opening.\n"
        "CONCEPTS: You may cite items in the 'Chess concepts detected' section verbatim; "
        "do not invent concepts, structures, or motifs that are not listed.\n"
        "STYLE: Write naturally, as a coach talking to a student. Do NOT mention "
        "'ground truth', 'Chess concepts detected', 'the section', or say facts were "
        "'detected/verified' — just state them as chess knowledge.\n"
        "Avoid listing full move sequences or tags; mention at most one short line (up to 4 plies).\n"
        "Keep the answer focused and practical in 2-4 short paragraphs.\n\n"
        "Position FEN: {fen}\n"
        "Side to move: {side}\n"
        "Original question: {prompt}\n"
        "Conversation so far:\n{history}\n\n"
        "Follow-up question: {question}\n\n"
        "=== Ground truth ===\n{ground_truth}\n\n"
        "Engine top lines (score from side to move, positive is better):\n{engine}\n\n"
        "Candidate lines with tags:\n{candidates}\n\n"
        "Hypothetical lines (engine evaluation if the user's mentioned move were played):\n{hypotheticals}\n"
    ).format(
        anchor=anchor,
        side=side_name,
        opponent=opponent_name,
        fen=board.fen(),
        prompt=prompt,
        history=history_text,
        question=question,
        ground_truth=ground_truth,
        engine=engine_text or "(none)",
        candidates=candidate_text or "(none)",
        hypotheticals=hypothetical_text or "(none)",
    )

    # Generate with the faithfulness guard + self-repair loop; fall back to the
    # deterministic factual summary if the model can't produce a faithful answer.
    allowed = list(hypothetical_lines or []) + list(engine_lines) + list(candidate_lines)
    response = _guarded_generate(board, follow_prompt, model, allowed, verified=None)
    if not response or response_needs_rewrite(response):
        return basic_explain(board, question, engine_lines, candidate_lines)
    return response


def build_board(fen: Optional[str], moves: Optional[List[str]]) -> chess.Board:
    board = chess.Board(fen) if fen else chess.Board()
    if moves:
        for token in moves:
            move = parse_move(board, token)
            board.push(move)
    return board


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM-assisted Stockfish explainer")
    parser.add_argument("--fen", help="FEN string for the position")
    parser.add_argument("--moves", nargs="*", help="Moves from start (SAN or UCI)")
    parser.add_argument("--prompt", help="Question for the coach")
    parser.add_argument("--stockfish", help="Path to Stockfish engine")
    parser.add_argument("--depth", type=int, default=16)
    parser.add_argument("--top", type=int, default=3)
    parser.add_argument("--pv-plies", type=int, default=10)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--llm", choices=["hf", "none"], default="hf")
    parser.add_argument("--model", default=os.getenv("HF_MODEL", DEFAULT_MODEL))
    parser.add_argument("--max-candidates", type=int, default=3)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.prompt:
        args.prompt = input("Prompt: ").strip()

    board = build_board(args.fen, args.moves)

    engine_path = find_engine_path(args.stockfish)
    if not engine_path:
        print("Could not find Stockfish engine. Use --stockfish to set the path.")
        return 2

    engine_lines = []
    candidate_lines = []
    with chess.engine.SimpleEngine.popen_uci(str(engine_path)) as engine:
        engine.configure({"Threads": args.threads})
        engine_lines = engine_top_lines(board, engine, args.depth, args.top, args.pv_plies)
        candidate_lines = build_candidate_lines(
            board,
            engine,
            args.prompt,
            args.depth,
            args.pv_plies,
            args.max_candidates,
            args.model,
            enable_llm=args.llm == "hf",
        )

    print(f"Position FEN: {board.fen()}")
    print(f"Prompt: {args.prompt}")
    print()

    facts = position_facts(board)
    if facts:
        print("Position facts:")
        for fact in facts:
            print(f"- {fact}")
        print()

    strategic = positional_facts(board)
    if strategic:
        print("Strategic facts:")
        for fact in strategic:
            print(f"- {fact}")
        print()

    effective_depth = tactical_depth(board, args.depth)
    depth_note = f"{args.depth}" if effective_depth == args.depth else f"{args.depth} → {effective_depth} (tactical)"
    print(f"Top engine lines (depth {depth_note}, {args.pv_plies} plies):")
    for idx, line in enumerate(engine_lines, start=1):
        print(f"{idx}) {format_score(line.score)}: {' '.join(line.moves)}")

    if candidate_lines:
        print()
        print("Candidate lines:")
        for line in candidate_lines:
            print(format_line_for_prompt(board, line))

    if args.llm == "none":
        return 0

    try:
        answer = llm_explain(board, args.prompt, engine_lines, candidate_lines, args.model)
    except RuntimeError as exc:
        answer = fallback_explain(board, args.prompt, engine_lines, candidate_lines, str(exc))

    print()
    print("LLM answer:")
    print(answer)

    if args.debug:
        payload = {
            "engine_lines": [line.__dict__ for line in engine_lines],
            "candidate_lines": [line.__dict__ for line in candidate_lines],
        }
        print()
        print("Debug:")
        print(json.dumps(payload, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
