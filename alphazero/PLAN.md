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

---

## Iteration 6: Self-play RL (expert iteration) — push Elo past the distillation ceiling

### Why (diagnosis recap)
Supervised distillation asymptotes at (and below) its shallow-Stockfish teacher,
and `diag.py` pinned the ceiling on **tactical value blindness** (over-optimistic
value net, ~8.8% blunder rate) — *not* capacity or search. Better supervised
metrics (v3) did not raise Elo. The way past a teacher is **AlphaZero self-play**:
train on the **MCTS visit distribution** (an improved policy the net can chase)
and the **actual game outcome** as the value target (inherently non-saturating
and tactically grounded — the direct fix for value blindness).

### Architecture decisions (attention kept for the explainer)
- **Value target** switched from `tanh(cp)` (saturating, over-optimistic) to the
  **self-play game outcome z ∈ {−1,0,+1}** from the mover's perspective, with an
  optional blend with the MCTS root value (`--value-lambda`).
- **Policy target** switched from Stockfish MultiPV to the **MCTS root visit
  distribution** (policy improvement → lets the net exceed the teacher).
- **Attention network is unchanged** (`model.py`): same tokens/heads/params, so
  its attention-weighted board state still drives the chess-teacher explainer.
  Only the *training targets* changed.
- **Warm-start from the distilled net (v5)** instead of learning from scratch
  (infeasible on a laptop) — this is expert iteration / fine-tuning upward.
- Self-play uses **Dirichlet root noise** + a **temperature schedule** (temp=1 for
  the first `--temp-moves` plies, then greedy) + **resign** (with a no-resign
  fraction), and reuses the per-game FEN eval cache for throughput.

### New components
| File | Role |
|---|---|
| `selfplay.py` | Generate self-play games (multiprocess, CPU workers); writes shards with MCTS-visit policy targets + outcome value `z` + root `q_value`. |
| `train_selfplay.py` | Train on self-play targets, warm-started from a checkpoint; optional retained distillation data (`--retain-dir/--retain-frac`) to avoid catastrophic forgetting. |
| `head_to_head.py` | Candidate-vs-incumbent match (prints `A_SCORE=`) for promotion gating. |
| `expert_iteration.py` | Orchestrator: self-play → train → gated promotion → periodic Stockfish eval, over a rolling replay buffer (`--window` iterations). Resumable via `<workdir>/state.json`. |

### Workflow
```
# one command runs the whole loop (self-play RL), warm-starting from v5:
python expert_iteration.py --workdir ei --warm-start checkpoints_v5/best.pt \
    --iterations 8 --games 140 --sp-sims 160 --workers 10 \
    --epochs 3 --window 4 --retain-dir data_v5 --retain-frac 0.4 \
    --gate-games 30 --gate-sims 120 --gate-threshold 0.52 \
    --sf-every 2 --sf-games 24 --sf-sims 1600
# resumes automatically if re-run (reads ei/state.json).
```
The explainer (`explain_position.py::default_ckpt`) now prefers `ei/best.pt` when
present, so chess-teacher's saliency comes from the strongest (self-play) net
while keeping the identical attention architecture.

### Perf (M3 Pro)
- Self-play ~12 pos/s/worker at 160 sims (batch 4–8, CPU); ~10–12 min per
  140-game iteration with 10 workers. Training a few epochs on the buffer is
  ~seconds–minutes on MPS. Phases run sequentially (no CPU/GPU contention).

### Status
- [x] Self-play generator, self-play trainer, gating match, and orchestrator
      implemented and smoke-tested end-to-end.
- [x] First real *pure-AlphaZero* self-play run (iters 1–2) — **REGRESSED**: at a
      matched config (1600 sims, 24 games, seed 2) v5 scores 0.646 (~2104 Elo)
      but the iter-2 self-play net scored 0.104 (~1626). Root causes: (a) the
      bot-vs-bot gate at 120 sims is blind to real strength (it "won" 2W-28D-0L
      yet collapsed vs Stockfish), and (b) pure game-outcome value targets on a
      tiny sample overwrote v5's tuned value head (train acc fell 0.41→0.16).
- [x] `verify_selfplay.py` added for the definitive head-to-head + SF comparison.

### Iteration 7 (corrected): expert iteration = self-play data + Stockfish labels + SF gate
The pure-self-play regression showed two things must change on laptop compute:
the **target quality** and the **promotion gate**. Corrected design:
- **Generation stays self-play / on-policy**: the net plays itself with MCTS
  (`selfplay.py --label-stockfish`), so we train on the distribution the bot
  actually reaches.
- **Targets come from Stockfish** (soft MultiPV policy + cp value) on each visited
  position — DAgger / expert iteration. Written in the *same shard schema* as the
  distillation corpus, so `train.py` consumes `data_v5 + on-policy` warm-started
  from the current best. This *preserves* the value head (verified: warm-start
  training keeps val acc ~0.41 / value ~0.03, unlike the pure-SP collapse).
- **Promotion is gated vs Stockfish@2000** at realistic sims (`evaluate.py`,
  paired seed). A candidate is promoted only if it scores ≥ the incumbent's
  Stockfish score (v5 is the permanent floor) → the shipped net can **never**
  regress below v5, and only genuine gains are kept.
- On-policy data is oversampled (`--onpolicy-oversample`) against a base-corpus
  subset so it carries real weight while epochs stay fast.
- New/updated files: `selfplay.py` (`--label-stockfish/--sf-depth/--sf-multipv`,
  on-policy SF-labelled generation), `train.py` (comma-separated `--data-dir`,
  `--warm-start`), `expert_iteration.py` (rewritten: gen → warm-start retrain →
  SF gate → monotonic promotion), `verify_selfplay.py`.
- Run: `python expert_iteration.py --workdir ei --base-data ei_base_subset
  --games 160 --sims 100 --sf-depth 12 --epochs 3 --gate-games 40 --gate-sims 800`.

- [x] Corrected expert-iteration run executed (`ei/`, floor = v5 @ gate 40g/800sims
      = score 0.588 ~2061 Elo). Results (gate = 40 games vs SF@2000, 800 sims, paired seed):

  | iter | on-policy data | train val-acc | candidate vs SF@2000 | promoted? |
  |---|---|---|---|---|
  | floor (v5) | — | 0.407 | 0.588 (~2061) | — |
  | 1 (depth-16 labels) | ~16k | 0.417 | 0.588 (~2061) | no (= floor) |
  | 2 (depth-14, more data) | ~40k, val win×4 | 0.434 | 0.550 (~2035) | no (< floor) |

  **Finding:** the corrected loop is *safe* — it never regressed (contrast the naive
  self-play collapse to ~1626) and correctly kept v5 both times. But it did **not
  exceed** v5: supervised metrics improved (val-acc 0.407→0.434) yet Stockfish Elo
  stayed ~2035–2061 (within 40-game noise). This reproduces the project's core
  ceiling — **tactical value blindness**: better policy/value *fit* does not raise
  match Elo, because the value net stays over-optimistic at tactical leaves. On this
  laptop (5M-param net, no datacenter self-play), ~2074–2104 (v5) is the ceiling.

### Iteration 8 (architecture): WDL value head (attack value blindness at the root)
The diagnosed ceiling is *tactical value blindness* (over-optimistic scalar value).
The principled architecture fix (Lc0-style) is a **win/draw/loss value head** trained
with cross-entropy instead of the saturating `tanh(cp)` MSE. Implemented behind a
`ModelConfig.wdl` flag: `model.py` value head -> 3 logits, `forward` still returns a
scalar `value = P(win) - P(loss)` so MCTS / evaluate / the explainer need **no**
changes (attention trunk untouched -> saliency intact, verified). `train.py` gained
`--wdl` (cp->soft-WDL targets + CE loss) and warm-starts the trunk/policy from v5,
reinitialising only the value head.

Results vs SF@2000 (gate 40g/800sims, seed 777; v5 floor = 0.588 ~2061):
  - WDL, 6 ep on 200k: **0.388 (~1920)** — under-trained fresh head.
  - WDL, 10 ep on ~380k (base subset + on-policy ×2): **~0.55 (~2035)** — recovered to
    ~v5 level, policy acc held (0.42), value CE plateaued ~0.51, but did **not exceed**
    the floor. Not promoted; v5 remains deployed (`ei/best.pt` is byte-identical to v5).

**Conclusion:** five independent approaches now land at or below v5 — bigger net
(d320), more data (944k), less-saturating value (tanh(cp/500)), on-policy expert
iteration, and a WDL head. The ~2074–2104 result is a **capacity/compute ceiling**
(5M-param net, laptop MPS), not a target/parameterisation bug. Breaking it needs a
larger network + real (datacenter-scale) self-play, which this hardware cannot do.

### Iteration 9 (search lever): does more play-time search beat 3600 sims?
The last untested lever for higher Elo is play-time search. Paired match vs SF@2000
(same seed/openings, 30 games):
  - v5 @ 3600 sims / batch 4 (deployed): **0.700 -> ~2147** (CI [2025,2321]).
  - v5 @ 6400 sims / batch 2 (more search): ~2058 through 12 paired games — **no gain**.
Combined with v5's near-flat low-vs-high response elsewhere, **search is saturated**:
extra sims don't help because the value net is over-optimistic at tactical leaves
(the same value-blindness ceiling). So ~2074–2147 is the peak from the search side too.

### Iteration 10 (capacity lever): a materially larger network
The objective invites "fix whatever architecture decisions you need", so the last
untested lever — more network capacity — was tested properly (the earlier d320 hint
was weak: old value target, 809k, no on-policy data). Trained a **d320 / 8-layer /
WDL net (~8.7M params, ~1.8× v5's 4.85M)** from scratch on the full 944k corpus +
on-policy self-play data, 16 epochs (matching v5's budget), attention trunk intact.
  - Supervised: **val acc 0.408 ≈ v5's 0.407** — the bigger net fits the targets no
    better than v5.
  - vs SF@2000, **paired seed 4242 @ 3600 sims**: big net **0.671 (~2124, 35g,
    CI [2011,2272])** vs v5 **0.700 (~2147)** — marginally *below*, i.e. **no gain**.
**Conclusion:** capacity is confirmed *not* the bottleneck. Six training approaches
(d320-old, 944k data, tanh(cp/500), expert iteration, WDL head, and a properly-
trained 1.8× net) plus the search lever all converge on ~2074–2147. Deployed net
stays v5. The binding limit is the value signal's tactical accuracy given laptop-
scale, shallow-ish Stockfish labels — not net size, value parameterisation, or search.

### Bottom line on "highest Elo possible"
- Highest **verified** Elo remains **v5 = ~2074** vs SF@2000 (100 games, 3600 sims,
  CI [2006,2148]); the self-play/expert-iteration pipeline **reaches and holds this
  and provably cannot ship anything weaker** (Stockfish-gated promotion, v5 floor).
- Self-play was implemented two ways (pure AlphaZero visit/outcome; and the robust
  expert-iteration with Stockfish labels) and the attention architecture is intact,
  so chess-teacher saliency is unchanged. `explain_position.default_ckpt` prefers
  `ei/best.pt` (≥ v5).
- Materially exceeding v5 needs a bigger net + real (datacenter-scale) self-play or
  a fundamentally better value signal — not achievable on this hardware. The pipeline
  is in place to capture any gain automatically if more compute is applied.
- [ ] (compute-bound / optional) larger self-play budget or a higher-sim play config
      to squeeze marginal Elo; confirm any new best via `verify_selfplay.py`.
