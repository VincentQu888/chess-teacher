"""Explanation-faithfulness regression tests (engine-free, fast).

These lock in the fix for the "irrelevant safety commentary" class of bugs and
keep the faithfulness harness honest: it must FLAG planted-false claims and must
NOT flag correct text. See verify_explanations.py for the corpus-scale checker
(which additionally runs Stockfish over hundreds of positions).

Run:  ./.venv/bin/python -m pytest test_explanations.py -q
"""
from __future__ import annotations

import chess

import re

import verify_explanations as v
from chess_teacher import (
    LineResult,
    _near_equal_alt,
    defense_relations,
    describe_best_move,
    practical_comparison,
)

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


def test_defense_relations_only_mentions_attacked_pieces():
    # A quietly-defended-but-unattacked pawn (g7, guarded by the king) is noise
    # and must not be reported; but a defender of an ATTACKED piece is fine.
    b = chess.Board("r2q1rk1/pbpn2pp/1p1ppn2/5pB1/1PPP4/2QBPN2/P4PPP/R4RK1 b - - 0 11")
    rels = defense_relations(b)
    assert not any("on g7 is defended" in r for r in rels), rels
    # g7 may still appear where relevant: as a defender of the attacked f6 knight.
    assert any("knight on f6 is defended" in r and "Pg7" in r for r in rels), rels
    # Invariant: every piece whose defenders are listed is actually attacked.
    for r in rels:
        m = re.search(r" on ([a-h][1-8]) is defended by", r)
        assert m, r
        sq = chess.parse_square(m.group(1))
        p = b.piece_at(sq)
        assert p is not None and b.attackers(not p.color, sq), f"unattacked piece listed: {r}"


def test_best_move_names_mate_and_skips_practicality():
    # Anastasia's mate in 1. The 'best move' answer must (a) name the mate using
    # specific terminology and (b) NOT append practicality/near-equal chatter
    # (a slower mate is never an 'essentially as good' alternative).
    b = chess.Board("8/4N1pk/8/R7/8/8/8/6K1 w - - 0 1")
    lines = [
        LineResult("engine_1", ["Rh5#"], {"mate": 1}, [], "engine"),
        LineResult("engine_2", ["Ra6"], {"mate": 15}, [], "engine"),
    ]
    text = describe_best_move(b, lines)
    assert "checkmate" in text.lower()
    assert "Anastasia's mate" in text
    for noise in ("essentially as good", "more practical", "easier to play",
                  "safe on", "follow up", "perfectly safe"):
        assert noise not in text.lower(), text
    # A forced mate offers no 'near-equal practical alternative'.
    assert _near_equal_alt(lines) is None


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
    test_defense_relations_only_mentions_attacked_pieces()
    test_best_move_names_mate_and_skips_practicality()
    test_concept_layer_faithful_on_sample_positions()
    print("all explanation-faithfulness tests passed")
