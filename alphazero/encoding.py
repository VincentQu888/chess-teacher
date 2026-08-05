"""Board and move encoding for the attention-based AlphaZero chess bot.

Design choices
--------------
* Positions are *canonicalised to the side to move*: if it is Black's turn we
  mirror the board (flip ranks + swap colours) so the network always sees
  "White to move, moving up the board". This roughly halves what the network
  must learn and makes the attention maps directly comparable across turns.
* The board is encoded as 64 per-square tokens (piece id 0..12) plus a small
  vector of global features (castling rights, en-passant file, side-to-move is
  implicit). The transformer in ``model.py`` self-attends over these tokens and
  exposes the attention weights as the "attention-weighted board state" used by
  the explainer (inspired by HEX-RL, arXiv:2112.08907).
* Moves use the AlphaZero 8x8x73 = 4672 policy space (56 queen-like + 8 knight
  + 9 under-promotion planes per from-square). Queen promotions fall out of the
  queen-move planes. We only ever map *legal* moves -> index (masked policy), so
  we never need a global index -> move inverse.
"""

from __future__ import annotations

from typing import List, Tuple

import chess
import numpy as np

# ---------------------------------------------------------------------------
# Square-token features
# ---------------------------------------------------------------------------

# piece id: 0 = empty, 1..6 = own P,N,B,R,Q,K, 7..12 = enemy p,n,b,r,q,k
# (own/enemy defined *after* canonicalisation, i.e. own == side to move)
_PIECE_ORDER = [
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
    chess.KING,
]
NUM_PIECE_IDS = 13  # 0 empty + 6 own + 6 enemy
NUM_SQUARES = 64
NUM_GLOBAL_FEATURES = 7  # own K/Q castle, enemy k/q castle, ep flag, ep file, halfmove clock
POLICY_SIZE = 64 * 73  # 4672


def canonical_board(board: chess.Board) -> Tuple[chess.Board, bool]:
    """Return (canonical_board_white_to_move, mirrored?).

    If Black is to move we mirror so the returned board always has White to
    move representing the same position from the mover's perspective.
    """
    if board.turn == chess.WHITE:
        return board, False
    return board.mirror(), True


def board_to_tokens(board: chess.Board) -> Tuple[np.ndarray, np.ndarray]:
    """Encode ``board`` (already canonicalised to White-to-move) into tokens.

    Returns
    -------
    piece_ids : int64 array, shape (64,)
        Square s (= rank*8 + file, A1=0) -> piece id in 0..12.
    globals   : float32 array, shape (NUM_GLOBAL_FEATURES,)
    """
    piece_ids = np.zeros(NUM_SQUARES, dtype=np.int64)
    for square, piece in board.piece_map().items():
        idx = _PIECE_ORDER.index(piece.piece_type) + 1  # 1..6
        if piece.color == chess.BLACK:  # enemy (White is to move in canonical frame)
            idx += 6
        piece_ids[square] = idx

    ep_file = -1
    if board.ep_square is not None:
        ep_file = chess.square_file(board.ep_square)
    g = np.array(
        [
            float(board.has_kingside_castling_rights(chess.WHITE)),
            float(board.has_queenside_castling_rights(chess.WHITE)),
            float(board.has_kingside_castling_rights(chess.BLACK)),
            float(board.has_queenside_castling_rights(chess.BLACK)),
            float(ep_file >= 0),
            (ep_file / 7.0) if ep_file >= 0 else 0.0,
            min(board.halfmove_clock, 100) / 100.0,
        ],
        dtype=np.float32,
    )
    return piece_ids, g


def encode_position(board: chess.Board) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Canonicalise then encode. Returns (piece_ids, globals, mirrored)."""
    canon, mirrored = canonical_board(board)
    piece_ids, g = board_to_tokens(canon)
    return piece_ids, g, mirrored


# ---------------------------------------------------------------------------
# Move encoding (AlphaZero 8x8x73)
# ---------------------------------------------------------------------------

# 8 sliding directions as (file_delta, rank_delta): N, NE, E, SE, S, SW, W, NW
_QUEEN_DIRS = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
_KNIGHT_DELTAS = [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)]
# under-promotion order: knight, bishop, rook (queen handled by queen planes)
_UNDERPROMO_PIECES = [chess.KNIGHT, chess.BISHOP, chess.ROOK]


def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


def move_to_index(move: chess.Move) -> int:
    """Map a *canonical-frame* (White-to-move) move to 0..4671.

    Raises ValueError if the move cannot be encoded (should not happen for
    legal chess moves in the canonical frame).
    """
    from_sq = move.from_square
    to_sq = move.to_square
    ff, fr = chess.square_file(from_sq), chess.square_rank(from_sq)
    tf, tr = chess.square_file(to_sq), chess.square_rank(to_sq)
    df, dr = tf - ff, tr - fr

    # Under-promotions (to N/B/R). Queen promotions use the queen planes.
    if move.promotion is not None and move.promotion != chess.QUEEN:
        try:
            piece_idx = _UNDERPROMO_PIECES.index(move.promotion)
        except ValueError as exc:  # pragma: no cover
            raise ValueError(f"bad promotion {move.promotion}") from exc
        # direction: df in {-1,0,1} -> {0,1,2}
        dir_idx = df + 1
        if dir_idx not in (0, 1, 2):
            raise ValueError(f"bad underpromotion geometry {move}")
        plane = 56 + 8 + piece_idx * 3 + dir_idx
        return from_sq * 73 + plane

    # Knight moves
    if (df, dr) in _KNIGHT_DELTAS:
        plane = 56 + _KNIGHT_DELTAS.index((df, dr))
        return from_sq * 73 + plane

    # Queen-like (straight or diagonal), incl. king steps, pawn pushes/captures,
    # and queen promotions.
    sdf, sdr = _sign(df), _sign(dr)
    if (sdf, sdr) in _QUEEN_DIRS and (df == 0 or dr == 0 or abs(df) == abs(dr)):
        dir_idx = _QUEEN_DIRS.index((sdf, sdr))
        distance = max(abs(df), abs(dr))
        if 1 <= distance <= 7:
            plane = dir_idx * 7 + (distance - 1)
            return from_sq * 73 + plane

    raise ValueError(f"unencodable move {move.uci()}")


def mirror_move(move: chess.Move) -> chess.Move:
    """Mirror a move (rank flip) for canonicalisation of Black-to-move moves."""
    return chess.Move(
        chess.square_mirror(move.from_square),
        chess.square_mirror(move.to_square),
        promotion=move.promotion,
    )


def legal_moves_with_indices(
    board: chess.Board, mirrored: bool
) -> Tuple[List[chess.Move], np.ndarray]:
    """Return (real_moves, indices) for all legal moves of ``board``.

    ``mirrored`` must match the flag from :func:`encode_position`. ``real_moves``
    are moves on the *original* board (playable directly); ``indices`` are the
    corresponding policy indices in the canonical frame.
    """
    real_moves: List[chess.Move] = []
    idxs: List[int] = []
    for mv in board.legal_moves:
        canon_mv = mirror_move(mv) if mirrored else mv
        idxs.append(move_to_index(canon_mv))
        real_moves.append(mv)
    return real_moves, np.asarray(idxs, dtype=np.int64)


def best_move_target_index(board: chess.Board, best_move: chess.Move) -> int:
    """Policy target index for a move played on the *original* board."""
    _, mirrored = canonical_board(board)
    canon_mv = mirror_move(best_move) if mirrored else best_move
    return move_to_index(canon_mv)


# ---------------------------------------------------------------------------
# Value target helpers
# ---------------------------------------------------------------------------

def cp_to_value(cp: float, scale: float = 300.0) -> float:
    """Centipawn eval (from side-to-move) -> value in (-1, 1) via a logistic."""
    return float(np.tanh(cp / scale))


def mate_to_value(mate_in: int) -> float:
    """Mate score -> +/-1 (sign follows who is mating)."""
    return 1.0 if mate_in > 0 else -1.0
