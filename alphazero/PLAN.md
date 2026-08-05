# AlphaZero-style Chess Bot with Attention-Weighted Board States

## ✅ GOAL ACHIEVED (verified)
**v5 attention net @ 3600 MCTS sims (batch 4) vs Stockfish@2000 (UCI_Elo=2000):**
**100 games → 49W 23D 28L → score 0.605 → est 2074 Elo, 95% CI [2006, 2148]**
(two independent 50-game runs: 2070 and 2078; both `TARGET MET`; CI lower bound > 2000).
The bot is the attention architecture (self-attention over 64 square tokens); its
attention-weighted board states feed the chess-teacher explainer. Winning play config:
`evaluate.py --ckpt checkpoints_v5/best.pt --sims 3600 --batch-size 4 --c-puct 1.5`.


Goal: an AlphaZero-style chess bot (policy/value net + PUCT MCTS) that reaches
**2000+ Elo**, tested against Stockfish limited to Elo 2000, using an
**attention** architecture so the per-move **attention-weighted board state**
can drive a better move explainer (chess-teacher).

Inspiration:
- UTMIST **fix-my-elo** Team2 bot (SL policy/value ResNet + MCTS + Stockfish
  testing) — see `refs/fix-my-elo/Team2`. We reuse its conventions (per-square
  planes, canonical side-to-move view, PUCT MCTS, pit-vs-Stockfish testing).
- **HEX-RL**, arXiv:2112.08907 — "Inherently Explainable RL in Natural
  Language". Core transferable idea: a (graph/self-)**attention** mechanism over
  a structured state exposes *which state elements most influenced the action*.
  We put self-attention over 64 square-tokens; the attention the value/CLS token
  and the move's from/to squares place on other squares = the explainable
  "attention-weighted board state".

## Why distillation (not pure self-play)
Pure AlphaZero self-play to 2000 Elo needs datacenter compute. On a laptop
(M3 Pro) the reliable, fast path is to **distil Stockfish**: label positions
with Stockfish's MultiPV top moves (soft policy target) + eval (value target),
train the net, then use PUCT MCTS at play time. Architecture stays AlphaZero-
style (policy/value net + MCTS, self-play capable); Stockfish is the teacher and
the yardstick, both explicitly allowed by the objective.

## Components
| File | Role |
|---|---|
| `encoding.py` | Canonicalise to side-to-move; 64 piece-id tokens + 7 globals; AlphaZero 4672 move index; legal-move → index masking. Unit-tested. |
| `model.py` | `AttentionChessNet`: piece+square embeddings, global & CLS tokens, N transformer blocks (custom attention that returns weights), policy head (per-square×73) + tanh value head. ~4.85M params. |
| `stockfish_data.py` | Multiprocessed Stockfish self-play data generator; MultiPV soft policy + value targets; writes `.npz` shards incrementally. |
| `train.py` | Supervised distillation on MPS: soft-target policy CE + value MSE; checkpoints `best.pt`/`latest.pt`. |
| `mcts.py` | PUCT MCTS over the net (edge stats, alternating-sign backup, FEN eval cache, policy-only mode). |
| `evaluate.py` | Match vs Stockfish `UCI_Elo` (default 2000), alternating colours + varied openings; prints score + Elo estimate w/ CI. |
| `attention_explain.py` | Extracts value-saliency and move-saliency squares from attention → ground-truth block for the LLM coach. |

## Workflow
```
# 1. Generate data (background)
python stockfish_data.py --games 3000 --depth 10 --multipv 5 --workers 10 \
    --out data --shard-size 5000

# 2. Train
python train.py --data-dir data --epochs 25 --batch-size 768 --out checkpoints

# 3. Evaluate vs Stockfish 2000
python evaluate.py --ckpt checkpoints/best.pt --games 40 --sims 160 --sf-elo 2000

# 4. Explainer signal for a position
python attention_explain.py checkpoints/best.pt
```

## Status log
- [x] Environment: py3.13 venv + torch 2.13 (MPS), python-chess 1.11.
- [x] Encoding + move index: unit-tested (promotions, en-passant, mirroring, no collisions).
- [x] Attention net forward/backward on MPS; attention weights exposed.
- [x] MCTS end-to-end on untrained net; policy-only + PUCT modes.
- [x] Data generator validated (~65 pos/s, 10 workers, depth 10); incremental shards.
- [x] Trainer validated (value loss drops; checkpoints saved).
- [x] Attention explainer produces value/move saliency in real board coords.
- [x] Perf: switched training/play attention to fused SDPA fast path (manual
      weight-returning path kept for the explainer); vectorised in-RAM batching.
      MPS is kernel-launch bound (~300s/epoch @ bs1024 on 200k); fp16 ~+10%.
- [x] Dataset: 200k positions (run 1). Run 2 generating concurrently (tag `r2_`)
      toward ~400-500k for a stronger retrain (CPU-bound; coexists with GPU training).
- [x] **v1 trained** (200k, 18 epochs): val policy 3.07 / value 0.067 / acc 0.33.
      Saved to `checkpoints_v1/`.
- [x] **First Elo measurements vs Stockfish@2000** (evaluate.py, verified real match):
      - policy-only (sims=0): 0W 2D 28L / 30 -> score 0.033 -> ~1415 Elo.
      - MCTS sims=400: 0W 4D 16L / 20 -> score 0.100 -> **~1618 Elo**.
- [x] MCTS correctness spot-check: finds free queen & free rook captures; misses a
      specific mate-in-1 and a smaller free-piece grab -> no gross bug; limits are a
      weak policy (acc 0.33) + value saturation in already-winning positions.
- [x] Corpus grown to **549,736 positions** (run1 200k + run2 350k).
- [~] **v2 training in progress**: full 550k, 16 epochs, bs1024 (background, ~3.2 hr;
      epoch1 already ahead of v1: val policy 3.75 vs 3.99). -> `checkpoints_v2/`.
- [ ] **Verify 2000+ Elo vs Stockfish@2000** (primary acceptance gate) -- still open.
- [ ] Elo levers to close ~400-Elo gap: full-corpus retrain (in progress), higher
      MCTS sims (400->1200+) + c_puct/FPU tuning, more epochs, deeper SF labels,
      less-saturating value target.
- [x] **Wired attention saliency into chess-teacher's explainer**: `explain_position.py`
      emits the attention-weighted board state as JSON; `chess_teacher.py`
      (`attention_model_block` / `_attention_section`) shells out to the alphazero
      venv and folds model value + top moves + value/move-saliency squares into
      `build_ground_truth_block`. Verified end-to-end; auto-selects newest checkpoint.
      (Still gated on the model reaching 2000+ so the saliency comes from a strong net.)

### Measured state
| model | search | vs SF@2000 | est. Elo |
|---|---|---|---|
| v1 (200k, 4.8M) | policy-only | 0.033 (0W2D28L/30) | ~1415 |
| v1 (200k, 4.8M) | MCTS 400 | 0.100 (0W4D16L/20) | ~1618 |
| v2 (550k, 4.8M) | MCTS 800 | 0.267 (2W12D16L/30) | **~1824** |
| v2 (550k, 4.8M) | MCTS 1600 | 0.233 (3W8D19L/30) | ~1793 |
| v3 (809k, 7.55M) | -- | training (16 ep, ~6.6h) | TBD |

**Key finding:** more MCTS sims do NOT help (800->1600 flat/slightly down) -- the
**net quality is the ceiling** (~1800), not search. v2's val loss had plateaued
(capacity/data-limited). So v3 widens the model (d_model 256->320, 4.8M->7.55M) and
trains on the full 809k corpus (incl. depth-12 labels).

**Batched MCTS** added (virtual loss, `batch_size`): 3.5x faster (800 sims 1.74s->0.50s
@ bs32), verified correct (finds free queen/rook; same top move as sequential).

**Perf:** MPS is compute-bound ~1.35ms/sample (bs doesn't help; bs4096 falls off a
cliff). Widening is more MPS-efficient than deepening (fewer sequential kernel launches).
v3: ~24.6 min/epoch on 809k. Runs span turns as background jobs.

**Known limiter for a future data regen:** value target `tanh(cp/300)` saturates in
winning positions (bot fails to grab a free knight when already winning); shards store
only the tanh value, not raw cp, so rescaling needs regenerating data (store cp then).

**Next levers if v3 < 2000:** deeper/cleaner data, less-saturating value target,
c_puct/fpu tune at bs<=8 (preserve search quality), more epochs, or a larger model.

### Iteration 3 result + diagnosis (KEY)
- **v3** (d320, 6.57M, 809k, 16ep) finished: val policy 2.607 / acc 0.40 / value 0.043
  (much better than v2's 2.80/0.36/0.055) BUT eval vs SF@2000 (800 sims, 30g) =
  **~1824 Elo -- identical to v2**. Better supervised metrics did NOT raise strength.
- **Diagnosis** (`diag.py`, strong-SF eval of each bot move over 4 games): bot blunders
  on **8.8% of moves**, and its **own value is badly over-optimistic** at blunders
  (plays a move it rates +0.14 that actually loses; throws a +410 winning position
  rating it +0.75). Failure mode = **tactical value blindness**, not capacity -- which is
  why bigger model (v3) and more sims (v2@1600) didn't help (garbage value at leaves).
- **Conclusion:** the lever is a better *value* signal, not more capacity/search.

### BREAKTHROUGH: search scales when batch is small
- Earlier "more sims doesn't help" was an ARTIFACT of aggressive batch=32 virtual
  loss degrading search quality. With **batch_size=4** (small vloss):
  - v3 @ 2400 sims: 3W5D6L/14 -> score 0.393 -> ~1924.
  - **v3 @ 3600 sims: 10W 11D 9L / 30 -> score 0.517 -> ~2012 Elo** (CI [1883, 2143]).
    Evaluator flagged TARGET MET (point estimate >= 2000). +~190 Elo over 800-sim
    baseline, no retraining. **Use batch_size<=4 for play/eval.**
- Status: point estimate clears 2000 but the 95% CI lower bound (1883) does not, so
  a *robust* >=2000 claim needs clear margin + a larger match. -> v5 + high-sims
  confirmation.

### Iteration 4 -- value fix (done)
- **v4**: clean depth-12 HQ corpus (`data_hq/`, 576k, no random-junk positions) +
  **less-saturating value target** `tanh(cp/500)` (via stored `best_cp`, `--value-scale 500`)
  + value_weight 1.5. d256, 18 epochs (~3.7h). -> `checkpoints_v4/`. Verified: value
  recompute engaged. RESULT: v4 = 1W7D16L/24 -> ~1745 (worse than v2/v3!). Blunders
  down (8.8%->6.2%) but clean data had too little coverage + only 576k -> weaker.
  Lesson: need diverse coverage AND realistic data together.

### Iteration 5 (in progress) -- combined corpus + high-sims
- **data_v5** = data_hq (576k realistic) + data_div (368k diverse/losing, explore-prob
  0.35) = **944k**, all with best_cp. Fixes value coverage while keeping policy quality.
- **v5**: d256 on data_v5, value-scale 500 (less saturating), value_weight 1.5, 16 ep
  (~5.5h). -> `checkpoints_v5/`. Then confirm at 3600 sims / batch 4 over 40-60 games
  targeting est clearly >2000 with the CI lower bound at/above 2000.
- Winning eval config so far: `evaluate.py --sims 3600 --batch-size 4 --c-puct 1.5`.

### Iteration 3 (bigger-model run details)
- **v3**: wider model (d_model 320, 6.57M params) on 809k, 16 epochs (~6.3h, ~25min/epoch).
  Tracking clearly ahead of v2 at matched epochs (ep3 val policy 3.09 vs v2 3.23,
  acc 0.327 vs 0.311). -> `checkpoints_v3/`.
- **Data-quality upgrade** (stockfish_data.py): now stores raw `best_cp` (int16) so the
  value target can be re-scaled at train time (`train.py --value-scale`, e.g. tanh(cp/500)
  to fight the saturation that made the bot miss a free knight when already winning);
  and drops the 15% pure-random mid-game moves (temperature-sample Stockfish top moves
  instead) to keep positions realistic. Verified new shard schema.
- **HQ corpus**: depth-12, clean, cp-stored, generating into `data_hq/` (background, CPU)
  for a v4 with a better value target if v3 falls short of 2000.

### Perf notes (MPS)
- Training is GPU-bound but data-gen is CPU-bound, so both run together with only
  ~10% training slowdown. This is the intended steady state between turns.
- Elo measurement will use `evaluate.py` (opponent config already verified:
  Stockfish `UCI_LimitStrength=true, UCI_Elo=2000`, plays legal moves).

## Elo measurement
`elo(bot) ≈ 2000 - 400·log10(1/score - 1)` from an alternating-colour match vs
Stockfish `UCI_LimitStrength=true, UCI_Elo=2000`. Target: estimated Elo ≥ 2000
(ideally lower CI bound ≥ 2000 over a sufficient number of games).
