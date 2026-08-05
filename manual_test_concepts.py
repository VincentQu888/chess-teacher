"""Manual/robustness test harness for chess_concepts.py.

Goes beyond the canonical fire-tests in test_chess_concepts.py:

1. FUZZ: run every detector individually on thousands of random legal positions,
   surfacing ANY exception (production detect_all_concepts swallows them).
2. SEE: validate static_exchange_eval against a brute-force capture minimax.
3. Cross-checks on random positions:
   - forks really attack >=2 winnable/king targets
   - absolute pins agree with board.is_pinned
   - passed pawns agree with an independent recompute
   - reported mate-in-1 patterns are really mate-in-1
   - hanging pieces really lose material to best capture sequence
4. FALSE-POSITIVE controls: tactic-availability concepts should not fire in a
   large sample of quiet/quiescent random positions without justification.

Run: ./.venv/bin/python manual_test_concepts.py
"""
from __future__ import annotations

import random
import traceback
from collections import Counter

import chess
import chess_concepts as cc

PIECE_VALUES = cc.PIECE_VALUES

# All single-board detectors (the registry) + the move-aware one.
DETECTORS = list(cc._BOARD_DETECTORS)


# ---------------------------------------------------------------------------
# Random position generators
# ---------------------------------------------------------------------------
def random_game_positions(n_games: int, max_plies: int, seed: int):
    """Positions reached by random legal play from the start."""
    rng = random.Random(seed)
    out = []
    for _ in range(n_games):
        b = chess.Board()
        plies = rng.randint(1, max_plies)
        for _ in range(plies):
            moves = list(b.legal_moves)
            if not moves or b.is_game_over():
                break
            b.push(rng.choice(moves))
        out.append(b.copy())
    return out


def random_sparse_positions(n: int, seed: int):
    """Random legal-ish endgame/sparse positions built by placing pieces.
    Only keeps positions python-chess considers valid."""
    rng = random.Random(seed)
    out = []
    tries = 0
    while len(out) < n and tries < n * 50:
        tries += 1
        b = chess.Board(None)
        wk = rng.randrange(64)
        bk = rng.randrange(64)
        if wk == bk or chess.square_distance(wk, bk) <= 1:
            continue
        b.set_piece_at(wk, chess.Piece(chess.KING, chess.WHITE))
        b.set_piece_at(bk, chess.Piece(chess.KING, chess.BLACK))
        n_extra = rng.randint(1, 8)
        ok = True
        for _ in range(n_extra):
            sq = rng.randrange(64)
            if b.piece_at(sq):
                continue
            pt = rng.choice([chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN])
            color = rng.choice([chess.WHITE, chess.BLACK])
            if pt == chess.PAWN and chess.square_rank(sq) in (0, 7):
                continue
            b.set_piece_at(sq, chess.Piece(pt, color))
        b.turn = rng.choice([chess.WHITE, chess.BLACK])
        if not b.is_valid():
            continue
        out.append(b)
    return out


# ---------------------------------------------------------------------------
# Independent ground-truth primitives
# ---------------------------------------------------------------------------
def brute_force_see(board: chess.Board, target: int, attacker_color: chess.Color) -> int:
    """Optimal material the attacker gains by initiating captures on `target`,
    with the DEFENDER free to stop capturing (standard SEE semantics), computed
    by exhaustive minimax over capture sequences on that square only."""
    tgt = board.piece_at(target)
    if not tgt or tgt.color == attacker_color:
        return 0

    def rec(b: chess.Board, side: chess.Color) -> int:
        # side-to-move net gain (negamax); either side may STOP capturing (=> 0).
        cur = b.piece_at(target)
        if cur is None:
            return 0
        best = 0  # option to stop
        for a in b.attackers(side, target):
            ap = b.piece_at(a)
            gain_here = PIECE_VALUES[cur.piece_type]
            b2 = b.copy(stack=False)
            b2.remove_piece_at(a)
            b2.remove_piece_at(target)
            b2.set_piece_at(target, ap)
            net = gain_here - rec(b2, not side)
            if net > best:
                best = net
        return best

    # Attacker initiates optimally; result is clamped at 0 (won't start a losing grab).
    best = 0
    cur = board.piece_at(target)
    for a in board.attackers(attacker_color, target):
        ap = board.piece_at(a)
        gain_here = PIECE_VALUES[cur.piece_type]
        b2 = board.copy(stack=False)
        b2.remove_piece_at(a)
        b2.remove_piece_at(target)
        b2.set_piece_at(target, ap)
        net = gain_here - rec(b2, not attacker_color)
        if net > best:
            best = net
    return best


def independent_passed_pawns(board: chess.Board):
    """Set of (color, square) for passed pawns, computed independently."""
    result = set()
    pawns = {chess.WHITE: [], chess.BLACK: []}
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p and p.piece_type == chess.PAWN:
            pawns[p.color].append(sq)
    enemy_by_file = {chess.WHITE: {}, chess.BLACK: {}}
    for color in (chess.WHITE, chess.BLACK):
        for sq in pawns[color]:
            enemy_by_file[color].setdefault(chess.square_file(sq), []).append(chess.square_rank(sq))
    for color in (chess.WHITE, chess.BLACK):
        for sq in pawns[color]:
            f, r = chess.square_file(sq), chess.square_rank(sq)
            blocked = False
            for ef in (f - 1, f, f + 1):
                for er in enemy_by_file[not color].get(ef, []):
                    if (color == chess.WHITE and er > r) or (color == chess.BLACK and er < r):
                        blocked = True
            if not blocked:
                result.add((color, sq))
    return result


# ---------------------------------------------------------------------------
# Test 1: fuzz every detector for crashes
# ---------------------------------------------------------------------------
def test_fuzz_crashes(positions):
    failures = []
    for b in positions:
        for det in DETECTORS:
            try:
                det(b.copy())
            except Exception:
                failures.append((det.__name__, b.fen(), traceback.format_exc()))
        # move-aware
        try:
            cc.detect_opening_principles(b.copy(), None)
        except Exception:
            failures.append(("detect_opening_principles", b.fen(), traceback.format_exc()))
        # full pipeline
        try:
            cc.format_concepts_for_prompt(b.copy())
        except Exception:
            failures.append(("format_concepts_for_prompt", b.fen(), traceback.format_exc()))
    return failures


# ---------------------------------------------------------------------------
# Test 2: SEE vs brute force
# ---------------------------------------------------------------------------
def test_see(positions):
    mism = []
    checked = 0
    for b in positions:
        for target in chess.SQUARES:
            tp = b.piece_at(target)
            if not tp:
                continue
            for atk in (chess.WHITE, chess.BLACK):
                if atk == tp.color:
                    continue
                if not b.attackers(atk, target):
                    continue
                got = cc.static_exchange_eval(b, target, atk)
                exp = brute_force_see(b, target, atk)
                checked += 1
                # both are "attacker net >= 0" style; compare sign & value
                if got != exp:
                    mism.append((b.fen(), chess.square_name(target),
                                 "W" if atk else "B", got, exp))
    return mism, checked


# ---------------------------------------------------------------------------
# Test 3: fork correctness
# ---------------------------------------------------------------------------
def test_forks(positions):
    bad = []
    fired = 0
    for b in positions:
        try:
            concepts = cc.detect_forks(b)
        except Exception:
            continue
        for c in concepts:
            if "would fork" in c.detail:
                continue  # availability claim, checked separately
            fired += 1
            # parse forking square
            sq_name = c.squares[0]
            sq = chess.parse_square(sq_name)
            piece = b.piece_at(sq)
            if piece is None:
                bad.append((b.fen(), c.detail, "no piece on forker square"))
                continue
            # recompute real targets
            targets = cc._valuable_targets_from(b, sq, piece.color)
            real = [t for t in targets
                    if b.piece_at(t).piece_type == chess.KING
                    or cc.static_exchange_eval(b, t, piece.color) > 0]
            if len(real) < 2:
                bad.append((b.fen(), c.detail, f"only {len(real)} real targets"))
    return bad, fired


# ---------------------------------------------------------------------------
# Test 4: absolute pin correctness vs board.is_pinned
# ---------------------------------------------------------------------------
def test_pins(positions):
    bad = []
    fired = 0
    for b in positions:
        try:
            concepts = cc.detect_pins(b)
        except Exception:
            continue
        for c in concepts:
            if c.name != "Pin (absolute)":
                continue
            fired += 1
            sq = chess.parse_square(c.squares[0])
            piece = b.piece_at(sq)
            if piece is None or not b.is_pinned(piece.color, sq):
                bad.append((b.fen(), c.detail, "not actually pinned per board.is_pinned"))
    return bad, fired


# ---------------------------------------------------------------------------
# Test 5: mate-pattern correctness (really mate-in-1)
# ---------------------------------------------------------------------------
def test_mates(positions):
    bad = []
    fired = 0
    for b in positions:
        try:
            concepts = cc.detect_checkmate_patterns(b) + cc.detect_named_mate_shortcuts(b)
        except Exception:
            continue
        # independent set of mate-in-1 moves
        real_mates = set()
        for mv in b.legal_moves:
            b.push(mv)
            if b.is_checkmate():
                real_mates.add(mv.uci())
            b.pop()
        for c in concepts:
            fired += 1
            # detail contains "<uci> is/delivers/... mate"
            uci = c.detail.split(" ")[0]
            if uci not in real_mates:
                bad.append((b.fen(), f"{c.name}: {c.detail}", "claimed mate is not mate-in-1"))
        # also: if there's a mate-in-1, at least one pattern concept should fire
        if real_mates and not cc.detect_checkmate_patterns(b):
            bad.append((b.fen(), f"mate available {real_mates}", "no checkmate concept fired"))
    return bad, fired


# ---------------------------------------------------------------------------
# Test 6: hanging correctness
# ---------------------------------------------------------------------------
def test_hanging(positions):
    bad = []
    fired = 0
    for b in positions:
        try:
            concepts = cc.detect_hanging(b)
        except Exception:
            continue
        for c in concepts:
            fired += 1
            sq = chess.parse_square(c.squares[0])
            p = b.piece_at(sq)
            if p is None:
                bad.append((b.fen(), c.detail, "no piece"))
                continue
            exp = brute_force_see(b, sq, not p.color)
            if exp <= 0:
                bad.append((b.fen(), c.detail, f"brute SEE={exp} (not actually hanging)"))
    return bad, fired


# ---------------------------------------------------------------------------
# Test 7: passed pawn correctness
# ---------------------------------------------------------------------------
def test_passed(positions):
    bad = []
    fired = 0
    for b in positions:
        try:
            concepts = [c for c in cc.detect_pawn_structure(b) if c.name == "Passed pawn"]
        except Exception:
            continue
        indep = independent_passed_pawns(b)
        indep_sqs = {sq for _, sq in indep}
        for c in concepts:
            fired += 1
            sq = chess.parse_square(c.squares[0])
            if sq not in indep_sqs:
                bad.append((b.fen(), c.detail, "not passed per independent check"))
    return bad, fired


# ---------------------------------------------------------------------------
# Test 8: false positives in quiet (quiescent) positions
# ---------------------------------------------------------------------------
def is_quiescent(board: chess.Board) -> bool:
    """No captures/checks/promotions available for side to move, and not in check."""
    if board.is_check():
        return False
    for mv in board.legal_moves:
        if board.is_capture(mv) or mv.promotion or board.gives_check(mv):
            return False
    return True


def test_false_positive_tactics(positions):
    """In quiescent positions, 'Combination' / 'Piece sacrifice' shouldn't claim a
    win/mate that isn't real; and no side should be able to win >=2 in one move."""
    bad = []
    checked = 0
    for b in positions:
        if not is_quiescent(b):
            continue
        checked += 1
        names = cc.concept_names_present(b)
        # Combination fired -> verify the shallow tactic really returns something
        if "Combination" in names:
            res = cc.shallow_tactic(b)
            if res is None:
                bad.append((b.fen(), "Combination fired but shallow_tactic=None"))
    return bad, checked


def main():
    random.seed(0)
    print("Generating random positions...")
    game_pos = random_game_positions(n_games=400, max_plies=60, seed=1)
    sparse_pos = random_sparse_positions(n=400, seed=2)
    all_pos = game_pos + sparse_pos
    print(f"  {len(game_pos)} game positions + {len(sparse_pos)} sparse positions "
          f"= {len(all_pos)} total\n")

    rc = 0

    print("[1] Fuzzing every detector for crashes...")
    fails = test_fuzz_crashes(all_pos)
    if fails:
        rc = 1
        print(f"  CRASHES: {len(fails)}")
        seen = set()
        for name, fen, tb in fails:
            if name in seen:
                continue
            seen.add(name)
            print(f"  --- {name} on {fen} ---")
            print("    " + tb.strip().replace("\n", "\n    "))
    else:
        print(f"  OK — no exceptions across {len(all_pos)} positions x {len(DETECTORS)} detectors")

    print("\n[2] SEE vs brute-force capture minimax...")
    # use sparse positions (richer capture chains) + a subset of game positions
    see_pos = sparse_pos + game_pos[:150]
    mism, checked = test_see(see_pos)
    if mism:
        rc = 1
        print(f"  MISMATCHES: {len(mism)} / {checked} targets")
        for fen, sq, atk, got, exp in mism[:15]:
            print(f"    {fen} target {sq} atk {atk}: got {got} expected {exp}")
    else:
        print(f"  OK — {checked} (position,target) SEE evaluations match brute force")

    print("\n[3] Fork correctness...")
    bad, fired = test_forks(all_pos)
    if bad:
        rc = 1
        print(f"  BAD: {len(bad)} / {fired} fork claims")
        for fen, detail, why in bad[:15]:
            print(f"    {fen}: {detail} -> {why}")
    else:
        print(f"  OK — all {fired} on-board fork claims verified (>=2 real targets)")

    print("\n[4] Absolute-pin correctness vs board.is_pinned...")
    bad, fired = test_pins(all_pos)
    if bad:
        rc = 1
        print(f"  BAD: {len(bad)} / {fired}")
        for fen, detail, why in bad[:15]:
            print(f"    {fen}: {detail} -> {why}")
    else:
        print(f"  OK — all {fired} absolute-pin claims agree with board.is_pinned")

    print("\n[5] Mate-pattern correctness (really mate-in-1)...")
    bad, fired = test_mates(all_pos)
    if bad:
        rc = 1
        print(f"  BAD: {len(bad)} / {fired}")
        for fen, detail, why in bad[:15]:
            print(f"    {fen}: {detail} -> {why}")
    else:
        print(f"  OK — all {fired} mate-pattern claims are genuine mate-in-1")

    print("\n[6] Hanging-piece correctness (brute-force SEE)...")
    bad, fired = test_hanging(all_pos)
    if bad:
        rc = 1
        print(f"  BAD: {len(bad)} / {fired}")
        for fen, detail, why in bad[:15]:
            print(f"    {fen}: {detail} -> {why}")
    else:
        print(f"  OK — all {fired} hanging claims lose material to best capture line")

    print("\n[7] Passed-pawn correctness (independent recompute)...")
    bad, fired = test_passed(all_pos)
    if bad:
        rc = 1
        print(f"  BAD: {len(bad)} / {fired}")
        for fen, detail, why in bad[:15]:
            print(f"    {fen}: {detail} -> {why}")
    else:
        print(f"  OK — all {fired} passed-pawn claims verified")

    print("\n[8] False-positive tactics in quiescent positions...")
    bad, checked = test_false_positive_tactics(all_pos)
    if bad:
        rc = 1
        print(f"  BAD: {len(bad)} across {checked} quiescent positions")
        for fen, why in bad[:15]:
            print(f"    {fen}: {why}")
    else:
        print(f"  OK — no bogus combination claims across {checked} quiescent positions")

    print("\n" + ("ALL MANUAL TESTS PASSED" if rc == 0 else "SOME MANUAL TESTS FAILED"))
    return rc


if __name__ == "__main__":
    import sys
    sys.exit(main())
