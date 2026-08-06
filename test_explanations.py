"""Explanation-faithfulness regression tests (engine-free, fast).

These lock in the fix for the "irrelevant safety commentary" class of bugs and
keep the faithfulness harness honest: it must FLAG planted-false claims and must
NOT flag correct text. See verify_explanations.py for the corpus-scale checker
(which additionally runs Stockfish over hundreds of positions).

Run:  ./.venv/bin/python -m pytest test_explanations.py -q
"""
from __future__ import annotations

import chess

import verify_explanations as v
from chess_teacher import practical_comparison

# The originally reported position: after Qxh8 the queen is on h8 with NO enemy
# attackers and the move wins a rook. "Undefended" is irrelevant here.
REPORTED_FEN = "r2k2nr/ppp2pQp/2npb3/2bNp3/2BqP3/8/PPPP2PP/R1BKNR2 w - - 6 11"


def test_reported_position_no_bogus_undefended_downside():
    b = chess.Board(REPORTED_FEN)
    text = practical_comparison(b, "d3", {"cp": 705}, "Qxh8", {"cp": 694})
    assert text is not None
    low = text.lower()
    # Must NOT invent a safety downside for a queen nothing can attack.
    assert "undefended on h8" not in low
    assert "left undefended" not in low
    # Should recognise the capture wins material and the square is safe.
    assert "wins the rook" in low
    assert "safe on h8" in low
    # And the produced prose must itself pass the faithfulness checker.
    assert v.check_verdict(b, text, ["d3", "Qxh8"]) == []


def test_harness_flags_planted_verdict_violations():
    b = chess.Board(REPORTED_FEN)
    cands = ["d3", "Qxh8"]
    assert v.check_verdict(b, "the queen is left undefended on h8.", cands)      # unattacked
    assert v.check_verdict(b, "d3 is safe on d3 (nothing attacks it).", cands)   # actually attacked
    assert v.check_verdict(b, "the pawn is defended by the rook.", cands)        # no rook defends d3
    assert v.check_verdict(b, "d3 wins the knight.", cands)                      # not a capture


def test_harness_flags_planted_concept_misclassification():
    # e3 is empty and c4 holds a bishop; a detail asserting otherwise must fail.
    b = chess.Board(REPORTED_FEN)
    reasons = []
    for letter, sqname in v._PIECE_ON_SQ_RE.findall("Rc4 threatens Ne3"):
        sq = chess.parse_square(sqname)
        pc = b.piece_at(sq)
        want = v.PIECE_TYPE_BY_LETTER[letter]
        if pc is None or pc.piece_type != want:
            reasons.append(sqname)
    assert reasons == ["c4", "e3"]


def test_concept_layer_faithful_on_sample_positions():
    for fen in [
        REPORTED_FEN,
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 6 5",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    ]:
        assert v.check_concepts(chess.Board(fen)) == [], f"concept violation in {fen}"


if __name__ == "__main__":
    test_reported_position_no_bogus_undefended_downside()
    test_harness_flags_planted_verdict_violations()
    test_harness_flags_planted_concept_misclassification()
    test_concept_layer_faithful_on_sample_positions()
    print("all explanation-faithfulness tests passed")
