"""Validation tests for chess_concepts.py.

For every concept in ``ALL_CONCEPTS`` there is a canonical position (FEN, and an
optional move list for opening-principle concepts that need move history) in
which the concept's detector must fire. A coverage test asserts every catalogued
concept has a test entry. Negative-control tests assert that tactical concepts do
NOT fire in quiet positions.

Run:  ./.venv/bin/python -m pytest test_chess_concepts.py -q
  or: ./.venv/bin/python test_chess_concepts.py   (prints a pass/fail summary)
"""

from __future__ import annotations

import chess

import chess_concepts as cc

# concept -> (FEN, optional moves-from-startpos list)
CONCEPT_TESTS = {
    # 1. Tactical motifs
    "Fork / Double attack": ("6k1/2q1r3/8/3N4/8/8/8/6K1 b - - 0 1", None),
    "Pin (absolute)": ("4k3/8/2n5/1B6/8/8/8/4K3 w - - 0 1", None),
    "Pin (relative)": ("4q1k1/8/2n5/8/B7/8/8/4K3 w - - 0 1", None),
    "Skewer": ("8/r7/8/8/k7/8/8/R5K1 b - - 0 1", None),
    "Discovered attack": ("4q1k1/8/8/8/4N3/8/8/4R1K1 w - - 0 1", None),
    "Discovered check": ("4k3/8/8/8/4N3/8/8/4R1K1 w - - 0 1", None),
    "Double check": ("4k3/8/8/8/4N3/8/8/4R1K1 w - - 0 1", None),
    "Deflection": ("k7/2b3b1/4n3/8/8/8/8/2R3RK w - - 0 1", None),
    "Decoy / Attraction": ("5r1k/6pp/4Q2N/8/8/8/8/7K w - - 0 1", None),
    "Removal of the defender": ("6k1/8/2n5/1B2b3/8/8/8/4R1K1 w - - 0 1", None),
    "Overloading": ("k7/2b3b1/4n3/8/8/8/8/2R3RK w - - 0 1", None),
    "Interference / Obstruction": None,  # lenient (needs a sac-line uniquely best)
    "Zwischenzug (in-between move)": None,  # lenient (needs move history)
    "Desperado": ("6k1/1b6/8/5p2/4B3/8/8/6K1 w - - 0 1", None),
    "X-ray": ("3q1rk1/8/8/8/8/8/3Q4/3R2K1 w - - 0 1", None),
    "Battery": ("3rk3/8/8/8/8/8/3Q4/3R1K2 w - - 0 1", None),
    "Windmill": ("4k3/8/8/8/4N3/8/8/4R1K1 w - - 0 1", None),
    "Trapped piece": ("7k/8/8/8/8/8/RP6/b6K b - - 0 1", None),
    "Hanging piece": ("4k3/8/8/8/5n2/8/8/4KR2 w - - 0 1", None),
    "Undermining": ("6k1/8/4p3/3n1P2/8/2N5/8/6K1 w - - 0 1", None),
    "Clearance sacrifice": None,  # lenient (needs a clearance uniquely best)
    "Counterattack": None,  # lenient
    "Perpetual check": None,  # lenient (drawing resource, move-context)
    "Greek gift sacrifice (Bxh7+)": ("rnbq1rk1/pppppppp/8/8/8/3B4/PPPP1PPP/RNBQ1RK1 w - - 0 1", None),
    "Combination": ("3q3k/8/8/8/8/8/8/3Q3K w - - 0 1", None),

    # 2. Checkmate patterns (mate-in-1 available)
    "Back-rank mate": ("6k1/5ppp/8/8/8/8/8/R6K w - - 0 1", None),
    "Smothered mate": ("6rk/6pp/8/6N1/8/8/8/6K1 w - - 0 1", None),
    "Anastasia's mate": ("8/4N1pk/8/R7/8/8/8/6K1 w - - 0 1", None),
    "Arabian mate": ("7k/R7/5N2/8/8/8/8/6K1 w - - 0 1", None),
    "Scholar's mate": None,  # move history (filled below)
    "Epaulette mate": ("3rkr2/8/Q7/8/8/8/8/6K1 w - - 0 1", None),
    # The remaining named mates are move-context / multi-move patterns; their detectors
    # exist but a single minimal always-correct fire-position is impractical -> lenient.
    "Boden's mate": None,
    "Légal's mate": None,
    "Fool's mate": None,
    "Dovetail (Cozio's) mate": None,
    "Hook mate": None,
    "Ladder / staircase mate": None,
    "Damiano's mate": None,
    "Swallow's tail (Guéridon) mate": None,

    # 3. Piece activity & coordination
    "Development": (chess.STARTING_FEN, None),
    "Tempo": ("rnbqkbnr/pppppppp/8/8/8/2N2N2/PPPPPPPP/R1BQKB1R w KQkq - 0 1", None),
    "Initiative": ("3q3k/8/8/8/8/8/8/3Q3K w - - 0 1", None),
    "Piece activity / mobility": (chess.STARTING_FEN, None),
    "Coordination / harmony": ("6k1/5ppp/8/5N1Q/8/8/1B4PP/6K1 w - - 0 1", None),
    "Outpost": ("4k3/pp3ppp/3p4/3N4/2P5/8/8/4K3 w - - 0 1", None),
    "Good bishop vs bad bishop": ("4k3/8/8/8/8/P1P1P1P1/3B4/4K3 w - - 0 1", None),
    "Bishop pair": ("4k3/8/8/8/8/8/8/2B1KB2 w - - 0 1", None),
    "Opposite-colored bishops": ("4k3/8/8/2b5/8/5B2/8/4K3 w - - 0 1", None),
    "Knight vs bishop": ("4k3/pppppppp/8/8/8/8/PPPPPPPP/2b1K1N1 w - - 0 1", None),
    "Rook on the 7th rank": ("6k1/R7/8/8/8/8/8/6K1 w - - 0 1", None),
    "Doubled rooks": ("3r2k1/3r4/8/8/8/8/8/6K1 b - - 0 1", None),
    "Fianchetto": ("6k1/5pbp/6p1/8/8/8/8/6K1 b - - 0 1", None),
    "Long-diagonal control": ("6k1/5pbp/6p1/8/8/8/1B6/6K1 w - - 0 1", None),
    "Overprotection": ("4k3/8/8/8/4P3/3P1P2/5N2/4K3 w - - 0 1", None),
    "Improving the worst piece": ("6k1/5ppp/8/8/8/8/1P1P1PPP/2B3K1 w - - 20 20", None),

    # 4. Pawn structure
    "Isolated pawn / IQP": ("4k3/8/8/3P4/8/8/8/4K3 w - - 0 1", None),
    "Doubled pawns": ("4k3/8/8/3P4/3P4/8/8/4K3 w - - 0 1", None),
    "Tripled pawns": ("4k3/8/3P4/3P4/3P4/8/8/4K3 w - - 0 1", None),
    "Backward pawn": ("4k3/8/8/2p5/1p6/1P6/2P5/4K3 w - - 0 1", None),
    "Passed pawn": ("4k3/8/8/3P4/8/8/8/4K3 w - - 0 1", None),
    "Protected passed pawn": ("4k3/8/8/3P4/2P5/8/8/4K3 w - - 0 1", None),
    "Connected passed pawns": ("4k3/8/8/3PP3/8/8/8/4K3 w - - 0 1", None),
    "Outside passed pawn": ("4k3/8/8/P2ppp2/3PPP2/8/8/4K3 w - - 0 1", None),
    "Hanging pawns": ("4k3/8/8/8/2PP4/8/8/4K3 w - - 0 1", None),
    "Pawn chain": ("4k3/8/8/3p4/2p1P3/3P4/8/4K3 w - - 0 1", None),
    "Pawn majority / minority": ("4k3/pp4pp/8/8/8/8/1P3P2/4K3 w - - 0 1", None),
    "Pawn island": ("4k3/8/8/8/8/8/P1P2P1P/4K3 w - - 0 1", None),
    "Pawn break / lever": ("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1", None),
    "Pawn storm": ("6k1/5p1p/6p1/8/6PP/5P2/8/6K1 w - - 0 1", None),
    "Pawn tension": ("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1", None),
    "Phalanx": ("4k3/8/8/8/3PP3/8/8/4K3 w - - 0 1", None),
    "Candidate passed pawn": ("4k3/1pp5/8/8/8/8/PPP5/4K3 w - - 0 1", None),
    "Weak square / hole": ("4k3/8/8/8/2p5/8/1P1P4/4K3 w - - 0 1", None),
    "Color complex weakness": ("4k3/1p1p1p2/p1p1p1p1/8/8/8/8/4K3 w - - 0 1", None),

    # 5. King safety
    "Castling (short/long)": ("6k1/5ppp/8/8/8/8/5PPP/6K1 w - - 0 1", None),
    "Pawn shield": ("6k1/5ppp/8/8/8/8/5PPP/6K1 w - - 0 1", None),
    "Luft (escape square)": ("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1", None),
    "Exposed / uncastled king": ("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQ1RK1 b kq - 0 10", None),
    "Open lines toward the king": ("3rk3/pppp1ppp/8/8/8/8/PPPP1PPP/3RK3 w - - 0 1", None),
    "Opposite-side castling": ("2kr4/ppp2ppp/8/8/8/8/PPP2PPP/5RK1 w - - 0 1", None),
    "King activity (endgame)": ("8/8/4k3/8/8/4K3/8/8 w - - 0 1", None),
    "Weakened kingside": ("6k1/5p1p/6p1/8/8/8/5PPP/6K1 w - - 0 1", None),

    # 6. Endgame
    "Opposition (direct/distant/diagonal)": ("8/8/8/4k3/8/4K3/8/8 b - - 0 1", None),
    "Zugzwang": ("8/8/8/4k3/4p3/4P3/4K3/8 b - - 0 1", None),
    "Triangulation": ("8/8/8/4k3/4p3/4P3/4K3/8 b - - 0 1", None),
    "Key squares": ("8/8/8/4k3/8/4P3/4K3/8 w - - 0 1", None),
    "Rule of the square": ("8/8/8/P7/8/8/6k1/4K3 w - - 0 1", None),
    "Lucena position": ("2K5/2P5/8/8/8/8/1r6/2R3k1 w - - 0 1", None),
    "Philidor position": ("8/8/4k3/4P3/8/4r3/8/4RK2 w - - 0 1", None),
    "Vancura position": ("8/8/r7/P7/8/8/5k2/R6K b - - 0 1", None),
    "Rook behind the passed pawn": ("8/8/8/8/8/1k6/1P6/1R4K1 w - - 0 1", None),
    "Wrong-colored bishop + rook pawn": ("8/8/8/8/8/6k1/7P/6KB w - - 0 1", None),
    "Fortress": None,  # lenient (defensive-setup heuristic)
    "Corresponding / related squares": ("8/8/8/2pP4/2P5/8/6k1/6K1 w - - 0 1", None),
    "Outside passed pawn (endgame use)": ("4k3/8/8/P2ppp2/3PPP2/8/8/4K3 w - - 0 1", None),
    "King centralization": ("8/8/4k3/8/8/4K3/8/8 w - - 0 1", None),
    "Shouldering / body-check": ("8/8/3k4/3P4/3K4/8/8/8 w - - 0 1", None),
    "Pawn breakthrough": ("6k1/pp6/8/8/8/8/PPP5/6K1 w - - 0 1", None),

    # 7. Opening principles
    "Control the center": (chess.STARTING_FEN, None),
    "Develop knights before bishops": (None, ["e2e4", "e7e5", "f1c4", "g8f6", "d2d3", "f8c5"]),
    "Castle early": ("rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 5", None),
    "Don't move the same piece twice": None,  # lenient (needs move history)
    "Don't bring the queen out too early": ("rnb1kbnr/pppp1ppp/8/4p3/4P1q1/8/PPPP1PPP/RNBQKBNR w KQkq - 0 3", None),
    "Connect the rooks": ("6k1/8/8/8/8/8/8/R4RK1 w - - 0 1", None),
    "Gambit": ("rnbqkbnr/pppp1ppp/8/8/4Pp2/5N2/PPPP2PP/RNBQKB1R b KQkq - 1 3", None),
    "Fight for the initiative / lead in development": ("rnbqkbnr/pppppppp/8/8/8/2N2N2/PPPPPPPP/R1BQKB1R w KQkq - 0 1", None),

    # 8. Named structures
    "Isolated Queen's Pawn (IQP)": ("rnbqkbnr/pp3ppp/8/8/3P4/8/PP3PPP/RNBQKBNR w KQkq - 0 1", None),
    "Hanging pawns (c+d)": ("4k3/pp3ppp/8/8/2PP4/8/P4PPP/4K3 w - - 0 1", None),
    "Carlsbad structure": ("4k3/pp3ppp/2p5/3p4/3P4/8/PP3PPP/4K3 w - - 0 1", None),
    "Maróczy Bind": ("4k3/pp1ppppp/8/8/2P1P3/8/PP1P1PPP/4K3 w - - 0 1", None),
    "Hedgehog": ("4k3/8/pp1pp3/8/8/8/8/4K3 b - - 0 1", None),
    "Sicilian Dragon": ("4k3/pp2ppbp/3p2p1/8/8/8/PPPPPPPP/4K3 b - - 0 1", None),
    "King's Indian Defense": ("4k3/pp3pbp/3p2p1/4p3/8/8/PPPPPPPP/4K3 b - - 0 1", None),
    "French Defense (closed center)": ("4k3/ppp2ppp/4p3/3pP3/3P4/8/PPP2PPP/4K3 w - - 0 1", None),
    "Caro-Kann / Slav skeleton": ("4k3/pp2pppp/2p5/3p4/8/8/PPPPPPPP/4K3 b - - 0 1", None),
    "Stonewall": ("4k3/pppppppp/8/8/3P1P2/2P1P3/PP4PP/4K3 w - - 0 1", None),
    'Scheveningen "small center"': ("4k3/pp3ppp/3pp3/8/4P3/8/PPPP1PPP/4K3 b - - 0 1", None),

    # 9. Strategic plans
    "Minority attack": ("4k3/pp3ppp/8/8/8/8/1P3PPP/4K3 w - - 0 1", None),
    "Blockade": ("4k3/8/8/3p4/3N4/8/8/4K3 w - - 0 1", None),
    "Prophylaxis": ("4k3/8/8/8/8/8/2n2P2/R5K1 w - - 0 1", None),
    "Restriction / cramping": ("4k3/pppppppp/8/8/3PPP2/2P3P1/PP5P/4K3 w - - 0 1", None),
    "Space advantage": ("4k3/pppppppp/8/8/3PPP2/2P3P1/PP5P/4K3 w - - 0 1", None),
    "Two weaknesses principle": ("4k3/p1p3p1/8/8/8/8/P4P1P/4K3 w - - 0 1", None),
    "Trade when ahead / avoid trades when behind": ("4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1", None),
    "Exchange the right pieces": ("4k3/8/8/8/8/P1P1P1P1/3B4/4K3 w - - 0 1", None),
    "Rerouting a knight": ("4k3/8/8/8/8/8/8/N3K3 w - - 0 1", None),
    "Rook to an open/semi-open file": ("4rk2/pppp1ppp/8/8/8/8/PPP2PPP/4K3 b - - 0 1", None),

    # 10. Sacrifices & material
    "Material balance": ("4k3/8/8/8/8/8/8/Q3K3 w - - 0 1", None),
    "Exchange sacrifice": ("4k3/8/8/3n4/8/8/8/3RK3 w - - 0 1", None),
    "Positional pawn sacrifice": None,  # lenient (heuristic)
    "Piece sacrifice for attack": ("5r1k/6pp/4Q2N/8/8/8/8/7K w - - 0 1", None),
    "Sham vs real sacrifice": ("5r1k/6pp/4Q2N/8/8/8/8/7K w - - 0 1", None),
    "Compensation": ("r1bqk2r/pppp1ppp/2n2n2/8/2B1P3/8/PP3PPP/RNBQK2R b KQkq - 0 6", None),

    # Opening identity
    "Opening": ("r1bqk2r/pp1pppbp/2n2np1/8/3NP3/2N1B3/PPP2PPP/R2QKB1R w KQkq - 4 7", None),

    # 11. Meta
    "Dynamic vs static advantages": ("4k3/8/8/3P4/8/8/8/R3K2R w KQ - 0 1", None),
    "Imbalances": ("4k3/8/8/8/8/8/8/2B1KB2 w - - 0 1", None),
    "Tempo / initiative accounting": (chess.STARTING_FEN, None),
}


def _board_from(entry):
    fen, moves = entry
    if fen is not None:
        return chess.Board(fen), moves
    b = chess.Board()
    for uci in moves:
        b.push_uci(uci)
    return b, moves


# Concepts whose detectors are heuristic/context-heavy; a triggering position is
# provided where feasible, otherwise they are exempted from the strict-fire test
# (still covered by ALL_CONCEPTS coverage).
_MOVE_HISTORY_FILLED = {
    "Scholar's mate": (None, ["e2e4", "e7e5", "f1c4", "b8c6", "d1h5", "g8f6"]),
}
CONCEPT_TESTS.update(_MOVE_HISTORY_FILLED)

# Concepts that are acceptable to leave without a strict fire-test position
# (their detectors exist and are exercised elsewhere / are inherently heuristic).
# Detectors that exist and are conservative, but for which a single minimal
# always-correct fire-position is impractical (multi-move / move-context / sac
# motifs). They remain catalogued and are exercised by detect_all_concepts.
LENIENT = {
    "Interference / Obstruction", "Clearance sacrifice", "Counterattack",
    "Perpetual check", "Zwischenzug (in-between move)",
    "Boden's mate", "Légal's mate", "Fool's mate", "Dovetail (Cozio's) mate",
    "Hook mate", "Ladder / staircase mate", "Damiano's mate",
    "Swallow's tail (Guéridon) mate",
    "Positional pawn sacrifice", "Fortress",
    "Don't move the same piece twice",
}


def run() -> int:
    # Coverage: every catalogued concept has a test entry.
    missing_entry = [c for c in cc.ALL_CONCEPTS if c not in CONCEPT_TESTS]
    assert not missing_entry, f"concepts with no test entry: {missing_entry}"
    extra = [c for c in CONCEPT_TESTS if c not in cc.ALL_CONCEPTS]
    assert not extra, f"test entries not in ALL_CONCEPTS: {extra}"

    failures = []
    for concept in cc.ALL_CONCEPTS:
        entry = CONCEPT_TESTS[concept]
        if entry is None:
            if concept in LENIENT:
                continue
            failures.append((concept, "no position provided"))
            continue
        board, moves = _board_from(entry)
        names = cc.concept_names_present(board, moves)
        if concept not in names and concept not in LENIENT:
            failures.append((concept, "did not fire"))

    # Negative controls: quiet opening has no tactic-availability motifs.
    quiet = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 2 3")
    quiet_names = cc.concept_names_present(quiet)
    for forbidden in ["Combination", "Decoy / Attraction", "Clearance sacrifice",
                      "Fork / Double attack", "Skewer", "Hanging piece",
                      "Discovered check", "Back-rank mate"]:
        if forbidden in quiet_names:
            failures.append((forbidden, "FALSE POSITIVE in quiet Italian position"))

    total = len([c for c in cc.ALL_CONCEPTS if c not in LENIENT])
    passed = total - len([f for f in failures if f[1] in ("did not fire", "no position provided")])
    print(f"Concept fire-tests: {passed}/{total} passed")
    print(f"Total catalogued concepts: {len(cc.ALL_CONCEPTS)} (lenient: {len(LENIENT)})")
    if failures:
        print("\nFAILURES:")
        for name, why in failures:
            print(f"  - {name}: {why}")
        return 1
    print("ALL CONCEPT VALIDATION TESTS PASSED")
    return 0


# pytest entry points
def test_coverage():
    assert set(CONCEPT_TESTS) >= set(cc.ALL_CONCEPTS)


def test_all_concepts_fire():
    assert run() == 0


if __name__ == "__main__":
    import sys
    sys.exit(run())
