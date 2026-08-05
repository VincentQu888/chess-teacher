"""Build eco.json: a map from board position (EPD) -> "ECO  Name".

Source: lichess-org/chess-openings TSVs (eco_a.tsv .. eco_e.tsv), columns
eco/name/pgn. For each opening we replay the PGN and record the resulting
position's EPD (placement + turn + castling + en-passant), so the chess-teacher
explainer can identify the exact opening/variation from any position (a pasted
FEN or a played move sequence). Longer (more specific) lines win on EPD clashes.
"""

from __future__ import annotations

import glob
import json
import os

import chess
import chess.pgn
import io


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    mapping = {}          # epd -> (eco, name)
    ply_of = {}           # epd -> ply count (prefer the more specific/longer line)
    files = sorted(glob.glob(os.path.join(here, "eco_*.tsv")))
    rows = 0
    for path in files:
        with open(path, encoding="utf-8") as fh:
            next(fh, None)  # header
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 3:
                    continue
                eco, name, pgn = parts[0], parts[1], parts[2]
                board = chess.Board()
                try:
                    game = chess.pgn.read_game(io.StringIO(pgn))
                    if game is None:
                        continue
                    for mv in game.mainline_moves():
                        board.push(mv)
                except Exception:
                    continue
                epd = board.epd()
                plies = board.ply()
                if epd not in ply_of or plies > ply_of[epd]:
                    ply_of[epd] = plies
                    mapping[epd] = f"{eco}  {name}"
                rows += 1
    out = os.path.join(here, "eco.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(mapping, fh)
    print(f"parsed {rows} rows -> {len(mapping)} unique positions -> {out}")


if __name__ == "__main__":
    main()
