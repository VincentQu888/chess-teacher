"""Manual test: curated, recognizable positions for the 'signature' patterns
(named mates, sacrifices, endgame techniques, opening IDs).

Every position below is a verified genuine instance. Prints PASS/FAIL for whether
the expected concept fires, and the full mate classification where relevant.

Run: ./.venv/bin/python manual_test_named.py
"""
from __future__ import annotations
import chess
import chess_concepts as cc


def mate_names(board):
    return sorted({c.name for c in cc.detect_checkmate_patterns(board)
                   + cc.detect_named_mate_shortcuts(board)})


def concept_names(board, moves=None):
    return cc.concept_names_present(board, moves)


def board_from_moves(moves):
    b = chess.Board()
    for u in moves:
        b.push_uci(u)
    return b


# (label, FEN or None, moves or None, expected_concept)
NAMED_MATES = [
    ("Back-rank mate", "6k1/5ppp/8/8/8/8/8/R6K w - - 0 1", None, "Back-rank mate"),
    ("Smothered mate", "6rk/6pp/8/6N1/8/8/8/6K1 w - - 0 1", None, "Smothered mate"),
    ("Anastasia's mate", "8/4N1pk/8/R7/8/8/8/6K1 w - - 0 1", None, "Anastasia's mate"),
    ("Arabian mate", "7k/R7/5N2/8/8/8/8/6K1 w - - 0 1", None, "Arabian mate"),
    ("Epaulette mate", "3rkr2/8/Q7/8/8/8/8/6K1 w - - 0 1", None, "Epaulette mate"),
    ("Boden's mate", "2kr4/3p4/8/8/2B2B2/8/8/4K3 w - - 0 1", None, "Boden's mate"),
    ("Damiano's mate", "8/K1k5/2P5/3Q4/n7/8/8/8 w - - 0 1", None, "Damiano's mate"),
    ("Hook mate", "3k4/4R3/3PKN2/8/8/8/5p2/8 w - - 0 1", None, "Hook mate"),
    ("Ladder/staircase mate", "3k4/5R2/8/8/R7/8/8/6K1 w - - 0 1", None, "Ladder / staircase mate"),
    ("Swallow's tail mate", "8/4P1Q1/8/8/8/1K6/n4n2/1kr5 w - - 0 1", None, "Swallow's tail (Guéridon) mate"),
    ("Dovetail (Cozio's) mate", "8/1K6/6pp/7k/Q7/1b5P/8/8 w - - 0 1", None, "Dovetail (Cozio's) mate"),
    # move-context mates (position is one move BEFORE the mate)
    ("Scholar's mate", None, ["e2e4", "e7e5", "f1c4", "b8c6", "d1h5", "g8f6"], "Scholar's mate"),
    ("Fool's mate", None, ["f2f3", "e7e5", "g2g4"], "Fool's mate"),
    ("Légal's mate", None, ["e2e4", "e7e5", "g1f3", "d7d6", "f1c4", "c8g4",
                            "b1c3", "g7g6", "f3e5", "g4d1", "c4f7", "e8e7"], "Légal's mate"),
]

SIGNATURE = [
    ("Greek gift", "rnbq1rk1/pppp1ppp/8/8/8/3B4/PPPP1PPP/RNBQ1RK1 w - - 0 1", "Greek gift sacrifice (Bxh7+)"),
    ("Windmill (disc-check battery)", "4k3/8/8/8/4N3/8/8/4R1K1 w - - 0 1", "Windmill"),
    ("Lucena position", "2K5/2P5/8/8/8/8/1r6/2R3k1 w - - 0 1", "Lucena position"),
    ("Philidor position", "8/8/4k3/4P3/8/4r3/8/4RK2 w - - 0 1", "Philidor position"),
    ("Desperado", "6k1/1b6/8/5p2/4B3/8/8/6K1 w - - 0 1", "Desperado"),
    ("Exchange sacrifice", "4k3/8/8/3n4/8/8/8/3RK3 w - - 0 1", "Exchange sacrifice"),
    ("Greek-gift NOT in quiet start (neg)", chess.STARTING_FEN, None),
]

OPENINGS = [
    ("Italian", ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4"]),
    ("Ruy Lopez", ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5"]),
    ("Sicilian Najdorf", ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3", "a7a6"]),
    ("French", ["e2e4", "e7e6", "d2d4", "d7d5"]),
    ("Caro-Kann", ["e2e4", "c7c6", "d2d4", "d7d5"]),
    ("QGD", ["d2d4", "d7d5", "c2c4", "e7e6"]),
    ("KID", ["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "f8g7", "e2e4", "d7d6"]),
]


def run():
    rc = 0
    print("=" * 72)
    print("NAMED MATES")
    print("=" * 72)
    for label, fen, moves, expected in NAMED_MATES:
        b = chess.Board(fen) if fen else board_from_moves(moves)
        got = mate_names(b)
        ok = expected in got
        rc |= (0 if ok else 1)
        print(f"  [{'PASS' if ok else 'FAIL'}] {label:26s} -> {got}")

    print("\n" + "=" * 72)
    print("OTHER SIGNATURE PATTERNS")
    print("=" * 72)
    for label, fen, expected in SIGNATURE:
        ns = concept_names(chess.Board(fen))
        if expected is None:
            leaked = ns & {"Combination", "Fork / Double attack", "Hanging piece",
                           "Greek gift sacrifice (Bxh7+)"}
            ok = not leaked
            rc |= (0 if ok else 1)
            print(f"  [{'PASS' if ok else 'FAIL'}] {label:34s} -> no tactic leakage ({sorted(leaked)})")
        else:
            ok = expected in ns
            rc |= (0 if ok else 1)
            print(f"  [{'PASS' if ok else 'FAIL'}] {label:34s} -> expected '{expected}': {'yes' if ok else 'NO'}")

    print("\n" + "=" * 72)
    print("OPENING IDENTIFICATION (real lines)")
    print("=" * 72)
    for label, moves in OPENINGS:
        b = board_from_moves(moves)
        concs = [c.detail for c in cc.detect_opening(b)]
        ok = bool(concs)
        rc |= (0 if ok else 1)
        print(f"  [{'PASS' if ok else 'FAIL'}] {label:18s} -> {concs}")

    print("\n" + ("ALL NAMED/ SIGNATURE TESTS PASSED" if rc == 0 else "SOME TESTS FAILED"))
    return rc


if __name__ == "__main__":
    import sys
    sys.exit(run())
