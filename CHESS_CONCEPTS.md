# Chess Concepts

A catalog of well-known chess concepts, grouped by category, each with a one-line
description. Goal: for every concept, design a way to **generate an explanation**
(ideally anchored in deterministic board facts + engine lines, like the existing
`build_ground_truth_block`).

## Status: IMPLEMENTED
Every concept below now has a **deterministic detector** in `chess_concepts.py`
(136 concepts). `detect_all_concepts(board)` runs them all and
`format_concepts_for_prompt(board)` renders a grouped ground-truth block that the
chess-teacher explainer folds into `build_ground_truth_block` alongside the
existing calculation facts and the neural bot's attention-weighted saliency.

Validation: `test_chess_concepts.py` has a canonical fire-position for every
concept (120/120 strict pass) + a coverage assertion + negative controls. 16
hard multi-move / move-context / sacrificial-line motifs (interference, clearance,
counterattack, perpetual, zwischenzug, and the rarer named mates: Boden's,
Légal's, Fool's, Cozio's, Hook, Ladder, Damiano's, Swallow's-tail; plus fortress,
positional-pawn-sac, don't-move-twice) have working conservative detectors but
are marked *lenient* (a single always-correct minimal FEN is impractical).

Run: `./.venv/bin/python test_chess_concepts.py`

(Original legend below is historical.)

## Legend
- ✅ = already detected/derivable in `chess_teacher.py` (has a fact/tag generator)
- 🟡 = partially detectable from current facts
- ⬜ = not yet detected (candidate for a new fact generator)

---

## 1. Tactical Motifs
- **Fork / Double attack** ✅ — one piece attacks two or more targets at once.
- **Pin (absolute)** ✅ — a piece can't move because it shields its king from attack.
- **Pin (relative)** ✅ — a piece is pinned to a more valuable piece behind it (moving is legal but loses material).
- **Skewer** ⬜ — a more valuable piece is attacked and forced to move, exposing a lesser piece behind it (reverse pin).
- **Discovered attack** ⬜ — moving one piece unveils an attack from a piece behind it.
- **Discovered check** ⬜ — a discovered attack where the unveiled attack is a check.
- **Double check** ⬜ — two pieces give check simultaneously; king must move.
- **Deflection** ⬜ — force a defending piece away from a key duty.
- **Decoy / Attraction** ⬜ — lure a piece to a bad square (often for a follow-up tactic).
- **Removal of the defender** ⬜ — capture/chase the piece that guards a key square or piece.
- **Overloading** ⬜ — a piece has too many defensive duties; exploit by attacking one.
- **Interference / Obstruction** ⬜ — block the line between a defender and what it protects.
- **Zwischenzug (in-between move)** ⬜ — insert an unexpected move (often a check/threat) before the "expected" recapture.
- **Desperado** ⬜ — a doomed piece grabs material/creates havoc before it's lost.
- **X-ray** ⬜ — a piece's influence passes through an enemy/friendly piece along a line.
- **Battery** 🟡 — two pieces stacked on the same line/diagonal (e.g., Q+B, R+R) to multiply pressure.
- **Windmill** ⬜ — a repeating discovered-check + capture sequence.
- **Trapped piece** ⬜ — a piece has no safe squares and will be won.
- **Hanging piece** ✅ — an undefended piece that can be captured for free.
- **Undermining** ⬜ — remove a pawn/piece that supports a key square or piece.
- **Clearance sacrifice** ⬜ — vacate a square/line for another piece (with tempo).
- **Counterattack** ⬜ — meet a threat with a bigger threat instead of defending.
- **Perpetual check** ⬜ — repeated checks forcing a draw.
- **Greek gift sacrifice (Bxh7+)** ⬜ — classic bishop sac on h7/h2 to expose the castled king.
- **Combination** ⬜ — a forcing sequence (often with a sacrifice) yielding a concrete gain.

## 2. Checkmate Patterns
- **Back-rank mate** 🟡 — rook/queen mates a king trapped by its own pawns on the back rank.
- **Smothered mate** ⬜ — knight mates a king boxed in entirely by its own pieces.
- **Anastasia's mate** ⬜ — knight + rook mate on the h-file (or edge).
- **Arabian mate** ⬜ — knight + rook mate in the corner.
- **Boden's mate** ⬜ — two bishops on crossing diagonals mate a castled king.
- **Légal's mate** ⬜ — queen sac + minor-piece mate in the opening.
- **Scholar's mate** ⬜ — Qxf7# early using queen + bishop.
- **Fool's mate** ⬜ — fastest possible mate (2 moves) exploiting f/g pawn moves.
- **Epaulette mate** ⬜ — king flanked by its own rooks, mated frontally.
- **Dovetail (Cozio's) mate** ⬜ — queen mate with king's escape blocked by its own pieces diagonally.
- **Hook mate** ⬜ — rook + knight + pawn mate.
- **Ladder / staircase mate** ⬜ — two rooks (or Q+R) drive the king to the edge.
- **Damiano's mate** ⬜ — pawn + queen mate after a rook sac on the h-file.
- **Swallow's tail (Guéridon) mate** ⬜ — queen mate with king's escape squares blocked by own pieces.

## 3. Piece Activity & Coordination
- **Development** ✅ — bringing pieces off their starting squares (tags for developing a minor).
- **Tempo** 🟡 — a unit of time; gaining/losing moves relative to the opponent.
- **Initiative** ⬜ — the ability to make threats and dictate play.
- **Piece activity / mobility** 🟡 — how many good squares and threats a piece has.
- **Coordination / harmony** ⬜ — pieces supporting each other toward a common goal.
- **Outpost** ✅ — a protected square (usually pawn-supported) no enemy pawn can attack, ideal for a knight.
- **Good bishop vs bad bishop** ✅ — bishop free of / hemmed in by its own pawns.
- **Bishop pair** ✅ — owning both bishops; strong in open positions.
- **Opposite-colored bishops** ⬜ — each side's bishop travels different color squares (drawish in endgames, attacking in middlegame).
- **Knight vs bishop** 🟡 — knights favor closed/blocked positions; bishops favor open ones.
- **Rook on the 7th rank** ⬜ — rook infiltrates the enemy's 2nd/7th rank, attacking pawns and king.
- **Doubled rooks** 🟡 — stacking rooks on a file/rank for pressure.
- **Fianchetto** 🟡 — developing a bishop to g2/b2/g7/b7 to control the long diagonal.
- **Long-diagonal control** ✅ — dominating the a1–h8 / a8–h1 diagonal (e.g., Dragon/KID bishop).
- **Overprotection** ⬜ — guarding a key point more than necessary to free the pieces around it.
- **Improving the worst piece** ⬜ — the strategic rule of upgrading your least active piece.

## 4. Pawn Structure
- **Isolated pawn / IQP** ✅ — a pawn with no friendly pawns on adjacent files; dynamic but a long-term target.
- **Doubled pawns** ✅ — two friendly pawns on the same file.
- **Tripled pawns** 🟡 — three friendly pawns on one file.
- **Backward pawn** ✅ — a pawn behind its neighbors that can't safely advance; often on a semi-open file.
- **Passed pawn** ✅ — no enemy pawns can stop it from promoting on its file or adjacent files.
- **Protected passed pawn** ⬜ — a passed pawn defended by another pawn.
- **Connected passed pawns** ⬜ — two adjacent passed pawns supporting each other.
- **Outside passed pawn** ⬜ — a passed pawn far from the main action, used to decoy the king.
- **Hanging pawns** ⬜ — two adjacent friendly pawns (usually c/d) on a half-open file with no pawn support.
- **Pawn chain** 🟡 — a diagonal chain of pawns; attack the base.
- **Pawn majority / minority** 🟡 — more/fewer pawns on one wing.
- **Pawn island** ⬜ — a group of connected friendly pawns separated from others; fewer islands is better.
- **Pawn break / lever** ⬜ — a pawn push that challenges the enemy structure to open lines.
- **Pawn storm** ✅ — advancing a wing of pawns to attack (often the castled king).
- **Pawn tension** ⬜ — adjacent pawns that can capture each other; who releases it and when.
- **Phalanx** ⬜ — two or more pawns side by side on the same rank.
- **Candidate passed pawn** ⬜ — a pawn that can become passed via a majority.
- **Weak square / hole** ✅ — a square that can't be defended by a pawn (often becomes an outpost).
- **Color complex weakness** ✅ — many pawns on one color, weakening the other color's squares.

## 5. King Safety
- **Castling (short/long)** ✅ — king safety + rook activation (tag for castling; king location detected).
- **Pawn shield** ✅ — the pawns in front of the castled king (missing-shield detection exists).
- **Luft (escape square)** ⬜ — a pawn move giving the king a flight square vs back-rank mates.
- **Exposed / uncastled king** ✅ — king stuck in the center or stripped of cover.
- **Open lines toward the king** 🟡 — open files/diagonals aimed at the king (king-zone threats detected).
- **Opposite-side castling** ⬜ — kings on opposite wings → pawn-storm races.
- **King activity (endgame)** ⬜ — the king becomes a strong attacker/defender once queens are off.
- **Weakened kingside** ✅ — holes/missing defenders around the king (e.g., traded fianchetto bishop).

## 6. Endgame Concepts
- **Opposition (direct/distant/diagonal)** ⬜ — kings facing off; the side *not* to move often controls key squares.
- **Zugzwang** ⬜ — any move worsens the position; being forced to move loses.
- **Triangulation** ⬜ — king maneuver to lose a tempo and pass the move to the opponent.
- **Key squares** ⬜ — squares the king must reach to force a pawn's promotion.
- **Rule of the square** ⬜ — geometric check of whether a king can catch a passed pawn.
- **Lucena position** ⬜ — winning R+P vs R technique ("building a bridge").
- **Philidor position** ⬜ — drawing R+P vs R technique (third-rank defense).
- **Vancura position** ⬜ — drawing method vs a rook-pawn from the side.
- **Rook behind the passed pawn** ⬜ — the "Tarrasch rule" for rook endgames.
- **Wrong-colored bishop + rook pawn** 🟡 — draw when the bishop can't control the promotion square.
- **Fortress** ⬜ — a defensive setup the stronger side can't break despite material.
- **Corresponding / related squares** ⬜ — mutual-zugzwang square mapping in blocked endgames.
- **Outside passed pawn (endgame use)** ⬜ — decoy the king, then win on the other wing.
- **King centralization** ⬜ — the endgame principle of activating the king toward the center.
- **Shouldering / body-check** ⬜ — using the king to block the enemy king's path.
- **Pawn breakthrough** ⬜ — a pawn sacrifice to create a runner from a majority.

## 7. Opening Principles
- **Control the center** ✅ — occupy/contest d4-e4-d5-e5 (center-occupation tag exists).
- **Develop knights before bishops** ⬜ — general development order heuristic.
- **Castle early** ⬜ — get the king safe and rooks connected.
- **Don't move the same piece twice** ⬜ — avoid wasting tempi in the opening.
- **Don't bring the queen out too early** ⬜ — she becomes a target for tempo gains.
- **Connect the rooks** ⬜ — clear the back rank so rooks defend each other.
- **Gambit** ⬜ — sacrifice a pawn (rarely more) for development/initiative.
- **Fight for the initiative / lead in development** 🟡 — convert a development edge into threats.

## 8. Named Pawn Structures & Opening Skeletons
- **Isolated Queen's Pawn (IQP)** ✅ — dynamic piece play vs long-term weakness (both colors detected).
- **Hanging pawns (c+d)** ⬜ — mobile duo that can be strong or become targets.
- **Carlsbad structure** ⬜ — QGD-Exchange skeleton; theme is the minority attack.
- **Maróczy Bind** ⬜ — pawns on c4+e4 clamping down on d5/…d5 breaks.
- **Hedgehog** ✅ — a6/b6/d6/e6 low setup uncoiling with …b5/…d5.
- **Sicilian Dragon** ✅ — …d6/…g6/…Bg7 with opposite-side castling races.
- **King's Indian Defense** ✅ — …d6/…e5/…g6/…Bg7; …f5 kingside attack vs queenside expansion.
- **French Defense (closed center)** ✅ — locked d4+e5 vs d5+e6; bad light-squared bishop, …c5/…f6 breaks.
- **Caro-Kann / Slav skeleton** ✅ — …c6 + …d5 solid structure.
- **Stonewall** ✅ — c3/d4/e3/f4 (or …f5/…e6/…d5/…c6); kingside attack + e5/e4 outpost.
- **Scheveningen "small center"** ⬜ — …d6+…e6 flexible Sicilian setup.

## 9. Strategic Plans & Maneuvers
- **Minority attack** ⬜ — advance the pawn minority (usually b4-b5) to create a weakness in the majority.
- **Blockade** ⬜ — stop a passed/isolated pawn by planting a piece in front of it.
- **Prophylaxis** ⬜ — prevent the opponent's plan before pursuing your own.
- **Restriction / cramping** 🟡 — limit enemy piece mobility (often via a space advantage).
- **Space advantage** 🟡 — more territory behind your pawns; more room to maneuver.
- **Two weaknesses principle** ⬜ — create a second front so the defender can't hold both.
- **Trade when ahead / avoid trades when behind** ⬜ — simplify with a material edge; keep pieces when down.
- **Exchange the right pieces** ✅ — e.g., trade the opponent's good bishop, keep your own (structure notes suggest this).
- **Rerouting a knight** ⬜ — maneuver a knight to a better outpost (e.g., Nd2-f1-g3-f5).
- **Rook to an open/semi-open file** ✅ — place rooks where they have scope (open/semi-open files detected).

## 10. Sacrifices & Material
- **Material balance** ✅ — piece/pawn count and value (material detection exists).
- **Exchange sacrifice** ⬜ — give rook for a minor piece to gain activity/structure.
- **Positional pawn sacrifice** ⬜ — give a pawn for long-term positional gains.
- **Piece sacrifice for attack** ⬜ — invest material to open the king.
- **Sham vs real sacrifice** ⬜ — forced-return vs genuine long-term investment.
- **Compensation** ⬜ — non-material factors (activity, king safety, structure) that offset a deficit.

## 11. Evaluation / Meta Concepts
- **Dynamic vs static advantages** ⬜ — temporary (initiative, activity) vs permanent (structure, material).
- **Imbalances** ⬜ — Silman's list: material, minor-piece imbalance, structure, space, development, initiative, king safety.
- **Compensation** ⬜ — see above; how much is "enough" for the material.
- **Tempo / initiative accounting** 🟡 — who is making threats and forcing responses.

---

## Notes for the explanation-generation design
- Prefer deriving each concept from **deterministic facts** (like the existing attack/defense/pin/fork/outpost/structure generators) so the LLM only narrates verified truth.
- Tactics (fork, pin, skewer, discovered attack, deflection…) are good candidates for **pattern detectors** run on the position + engine PV, emitting a structured tag the LLM explains.
- Endgame concepts (opposition, Lucena, Philidor, rule of the square) may need **specialized recognizers** keyed on material signatures + king/pawn geometry.
- Each concept ideally gets: (1) a detector/trigger, (2) a template of the facts to cite, (3) an LLM prompt fragment for the plain-English "why it matters."
