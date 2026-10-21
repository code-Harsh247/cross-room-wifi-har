# Task 3: Ideas to Improve the Results

**Status:** proposal list (2026-10-21). Nothing here is implemented yet.
**Context:** [tasks_task3.md](tasks_task3.md) Results log (U6–U9) · [PRD](PRD_Task3_Few_Shot_Cross_Room.md)

---

## Where we stand

| | A → B (K = 20) | B → A (K = 20) |
|---|---|---|
| Default M1, overall macro-F1 | 0.335 | 0.336 |
| Room-only (exclusive) macro-F1 | 0.664 | 0.592 |
| Shared (sit/stand/walk) macro-F1 | 0.115 | 0.081 |
| Best ablation (M5, logistic regression), overall | 0.438 | 0.409 |
| Upper bound (all target labels; not valid) | 0.901 | 0.692 |

- **What works:** room-only activities, learned from the K support captures, improve steadily with K.
- **What fails:** **sit / stand / walk do not transfer between rooms.** They are not just noisy, they are *systematically swapped*. In Room B, sit and stand are mostly predicted as walk, and walk as empty.
- **Cause:** the frozen encoder organises windows **by room first** (Task 2 UMAP; U6 diagnostics). Source-room prototypes land on the wrong target clusters.

## Ground rules for every idea

1. **The pre-registered default (D1 = M1) stays the headline.** Everything below is reported as a **post-hoc improvement attempt** (an ablation). It is clearly labelled as designed after seeing U6–U8 results.
2. **Fix each idea's settings before running it,** and log them in the Decisions log of `tasks_task3.md`. Never tune on target validation scores.
3. **Labels allowed:** source dev labels + K support captures, via `visible_labels`. Unlabelled target **dev** windows may be used. Target **validation** windows are scored only.
4. **Report positive and negative outcomes alike.** A clear negative result ("this physically motivated fix does not transfer, because …") is still a finding for the report.

---

## Ideas (most promising first)

### 1. Room-relative motion-intensity features
- **Idea:** walking makes the WiFi signal fluctuate much more over time than sitting, standing or an empty room, in *any* room. Measure "how much does this window move?" relative to its own room, and use that as a room-independent activity clue.
- **What we do:**
  - Per window, compute simple temporal-variation statistics on the per-row normalized amplitude (e.g. mean over subcarriers of the std over time, and the mean absolute row-to-row change).
  - Standardize them **within each room** using that room's unlabelled dev windows (or convert to within-room percentiles). This cancels the format difference: Room B rows are noisier for non-physical reasons.
  - Append them to the M1 embedding (weighted) or use them as a separate vote.
- **Quick diagnostic first (~10 min):** train on Room A's sit/stand/walk motion features, test on Room B's, 3-class. Above chance (0.33) means the signal exists.
- **Expected:** walk vs still may transfer. Sit vs stand probably will not.
- **Cost:** about ½ day. **Risk:** low. **Rules:** ✅ unlabelled data only.
- **Diagnostic result (2026-10-21, `scripts/task3_motion_diagnostic.py` → `outputs/task3/motion_diagnostic.json`): ❌ not viable. The motion ordering is *reversed* between the rooms.**

  | Mean temporal std (per-row normalized, dev windows) | sit | stand | walk |
  |---|---|---|---|
  | Room A | 0.025 | 0.050 | **0.229** (most motion, as expected) |
  | Room B | 0.999 | 0.952 | **0.572** (least motion) |
  | Room B after time smoothing, w = 15 | 0.229 | 0.223 | **0.196** (still least) |

  - **Transfer** (train on source sit/stand/walk, test on target, labels for scoring only): every variant is at or below chance:
    - 3-class 0.00–0.20 (chance 0.33);
    - walk vs still 0.01–0.25 (chance 0.5).
  - **Room-relative z-scores make it worse**, not better (walk vs still 0.01–0.04, i.e. near-perfectly inverted). The signal exists in both rooms, but with the opposite sign.
  - **Second pass (post hoc):** time-smoothing both rooms (w = 5, 15) narrows Room B's gap but never flips the order. So Room B's row noise is not the whole explanation; in this data, Room B "walk" windows really are steadier than its "sit"/"stand" windows.
  - **Consequence:** this also explains the systematic swaps in the U9 confusion matrices (B sit/stand → walk). No method can transfer a cue whose meaning is reversed between rooms without target labels for those classes. Ideas 2–4 cannot fix identity either: they can only rebalance or align distributions.

### 2. Balanced assignment (optimal transport / Sinkhorn)
- **Idea:** predictions currently collapse into one or two classes (e.g. everything → walk). Force the predicted class proportions on the unlabelled target data to be roughly balanced, so every class gets its share.
- **What we do:** compute similarities between unlabelled target dev embeddings and the class prototypes. Solve a Sinkhorn optimal-transport problem with a uniform class marginal, then turn the transport plan into refined prototypes or a calibrated decision rule. Apply it to target val.
- **Expected:** a direct boost to macro-F1 by stopping class collapse. It does not fix swapped identities on its own.
- **Assumption to state:** target classes are roughly equally frequent. That is a prior, not a label. It holds in our data by design, and the report must say so.
- **Cost:** about ½ day. **Risk:** medium. **Rules:** ✅ unlabelled target dev only.
- **Result (2026-10-21; settings pre-registered in `configs/task3_posthoc.yaml` before running; methods in `xroom.task3_methods`; run with `python scripts/task3_run.py --config configs/task3_posthoc.yaml --methods M7 M8 M9`): ✅ M9 is the best Task 3 method so far (post-hoc ablation).** Three variants:
  - **M7:** M1 + class offsets learned by Sinkhorn on target dev.
  - **M8:** balanced re-estimated prototypes.
  - **M9:** M5 (logistic regression) + Sinkhorn class offsets.

  | Overall macro-F1 (10 draws) | M1 default | M5 | M7 | M8 | **M9** |
  |---|---|---|---|---|---|
  | A→B K=1 / 5 / 10 / 20 | 0.181 / 0.272 / 0.303 / 0.335 | 0.191 / 0.344 / 0.411 / 0.438 | 0.181 / 0.272 / 0.299 / 0.330 | 0.173 / 0.243 / 0.275 / 0.300 | **0.314 / 0.398 / 0.472 / 0.492** |
  | B→A K=1 / 5 / 10 / 20 | 0.234 / 0.287 / 0.306 / 0.336 | 0.214 / 0.338 / 0.376 / 0.409 | 0.250 / 0.324 / 0.345 / 0.390 | 0.228 / 0.279 / 0.285 / 0.329 | **0.241 / 0.360 / 0.396 / 0.439** |

  - **Consistent:**
    - M9 beats M5 in 39 of 40 paired draws in A→B and 30 of 40 in B→A;
    - M9 beats M1 in 40 of 40 (A→B) and 36 of 40 (B→A).
  - **Shared classes:** A→B shared F1 rises from 0.115 (M1, K = 20) to 0.317.
  - **Where the gain comes from (K = 5, per-class F1):**
    - Balancing spreads predictions over all classes. M5 A→B puts 57% of target val windows into "sit"; M9 puts 29%.
    - **Sit genuinely transfers A→B** (F1 0.52 → 0.61).
    - Stand and walk rise only to ≈ 0.14, about what a balanced random assignment gives per class. Part of the gain is fewer collapsed errors, not new recognition.
  - **M7** helps only in B→A; **M8** (re-estimated prototypes) hurts in both directions.
  - **Status:** post-hoc ablation. It cannot replace the pre-registered default M1 (D1), because it was designed after seeing U6–U8. It assumes roughly equal target class frequencies (true by design in this data; stated as a prior).

### 3. Combine ideas 1 + 2
- **Idea:** motion features fix *which* cluster is walk; balanced assignment stops collapse. Together they target both failure modes.
- **What we do:** run idea 2 on the embedding extended with idea 1's features.
- **Cost:** small once 1 and 2 exist.
- **Status:** dropped. Idea 1's motion signal is reversed between the rooms, so adding it would push the balanced assignment the wrong way.

### 4. Domain-adversarial alignment (DANN) on frozen embeddings
- **Idea:** train a small network on top of the 128-d embeddings that classifies source activities well but **cannot tell the rooms apart**, so its features become room-invariant.
- **What we do:**
  - Small MLP + activity head, trained on source labels + support.
  - Domain head (room A vs B) behind a gradient-reversal layer, trained on unlabelled dev windows of both rooms.
  - Fixed epochs and λ, set in advance.
- **Expected:** standard fix for domain shift. It may fail when classes are swapped, because aligning the overall distributions does not force sit to align with sit.
- **Cost:** about 1 day. **Risk:** medium–high (training instability). **Rules:** ✅.

### 5. Report logistic regression (M5) more prominently
- **Idea:** M5 is already our best method (K = 20: 0.438 / 0.409 vs M1 0.335 / 0.336).
- **What we do:** nothing new. Discuss it in the report as the strongest ablation and explain why a discriminative classifier beats prototypes when classes overlap.
- **Rules:** ⚠️ it cannot replace D1, because it was chosen after seeing validation scores.

### 6. MAE encoder as the feature source
- **Idea:** Task 2 found the MAE encoder the most robust to format perturbations. It may transfer activities better.
- **What we do:** retrain one MAE encoder with Task 2's ablation config (~4 min), embed, and run M1 / M5 on it as a feature-source ablation.
- **Cost:** about 1 hour. **Rules:** ⚠️ ablation only (MAE was compared with activity-labelled probes in Task 2).

### 7. Short-slice averaging of embeddings
- **Idea:** in Task 1, averaging predictions over short slices of a window transferred much better to unseen activities (+13 points). The same may hold for activity embeddings.
- **What we do:** embed 25- or 50-row slices of each 150-row window, average the embeddings (or the slice predictions), then run M1.
- **Cost:** about ½ day. **Risk:** low–medium. **Rules:** ✅.

### 8. Support augmentation for small K
- **Idea:** at K = 1 the room-only prototypes rest on very few windows. Augment the support windows (crop, noise, subcarrier masking: the Task 2 augmentations) and embed several views per support capture.
- **What we do:** for each support capture, embed N augmented views and average them into the prototype.
- **Expected:** a more stable K = 1–5 room-only score. It does not help shared classes.
- **Cost:** about ½ day. **Risk:** low. **Rules:** ✅ (support labels only).

### 9. Retrain the encoder to ignore the room (root-cause fix)
- **Idea:** retrain the SSL encoder with a room-invariance objective, e.g. a domain-adversarial term, per-room instance normalization inside the network, or cross-room positive pairs.
- **What we do:** a new Task 2-style training run, a label-free selection rule, then a full Task 3 re-run.
- **Cost:** several days. **Risk:** high, not realistic before Oct 31. **Rules:** ✅ if selected label-free.
- **Use:** "future work" in the report.

---

## Batch 2 results (2026-10-21, post-hoc; settings pre-registered in `configs/task3_posthoc.yaml` before each run)

Commands:
- `python scripts/task3_embed_posthoc.py`
- `python scripts/task3_run.py --config configs/task3_posthoc.yaml --methods M1 M9 M10 --sources <source>`

Overall macro-F1 at K = 20 (mean of 10 draws):

| Source | Method | A → B | B → A | Note |
|---|---|---|---|---|
| default (Task 2 SimCLR) | M1 (pre-registered default) | 0.335 | 0.336 | headline |
| default | M9 (batch 1 best) | 0.492 | 0.439 | |
| default | **M10** balanced self-training | 0.488 | 0.427 | ❌ no gain over M9 |
| slice25 / slice50 | M9 | 0.481 / 0.483 | 0.449 / 0.438 | ❌ slice averaging: no gain (idea 7) |
| handcrafted | M9 | 0.381 | 0.339 | ❌ worse (room-only F1 high in A→B, 0.85; shared ≈ 0.07) |
| random_init_s0 | M9 | 0.327 | **0.535** | direction-dependent: best single B→A source, poor A→B |
| default + random_init_s0 | M9 | 0.434 | 0.485 | compromise; neither direction best |
| **MAE, 3 seeds (mean, min–max)** | **M9** | **0.499** (0.417–0.625) | **0.515** (0.476–0.555) | idea 6 |
| MAE, 3 seeds | M10 | 0.485 (0.363–0.648) | 0.525 (0.470–0.563) | |
| MAE, 3 seeds | M1 | 0.396 (0.352–0.425) | 0.328 (0.313–0.336) | |

### Findings
1. **MAE features help B→A consistently.** M9 on MAE beats M9 on the default encoder for every MAE seed at K = 20 (0.476–0.555 vs 0.439), and by +0.07 to +0.08 on average at K ≥ 5.
2. **In A→B, MAE is level with the default on average** (0.499 vs 0.492), with a very large spread between encoder seeds (0.417–0.625). The seed-0 result (0.625, with stand F1 0.14 → 0.64–0.73) was a lucky seed, not a reliable gain. The robustness check (pre-registered before seeds 1 and 2 were trained) caught this.
3. **Encoder seed variance is now the biggest source of uncertainty** for anything built on the encoder. The Task 3 default uses a single SimCLR seed, so a fair MAE-vs-SimCLR comparison would also need SimCLR seeds 1–2 run through Task 3.
4. **Self-training (M10) and slice averaging do not improve on M9.** Hand-crafted features are worse.
5. **The feature source that works best depends on the direction** (random-init: B→A 0.535, A→B 0.327). Choosing a source per direction would be validation tuning, so this is reported as an observation only.

## Batch 3 results: encoder ensembles and a fair SimCLR-vs-MAE comparison (2026-10-21, post-hoc)

**Setup** (all pre-registered in `configs/task3_posthoc.yaml` before training or running):
- **SimCLR seeds 1 and 2** were trained label-free with Task 2 defaults (`t3_simclr_s1/s2`). Seed 0 is the committed default encoder.
- **M11 = probability ensemble:** one balanced logistic regression per encoder, equal-weight mean of their probabilities, then the M9 balanced class offsets. No member is chosen by a score.
- **Commands:**
  - `python scripts/task3_embed_posthoc.py`
  - `python scripts/task3_run.py --config configs/task3_posthoc.yaml --methods M11 --sources ens_mae ens_simclr ens_all` (plus the batch-3 runs listed in the config)

Overall macro-F1 (mean of 10 draws):

| System | A→B K=1 / 5 / 10 / 20 | B→A K=1 / 5 / 10 / 20 |
|---|---|---|
| M1 default (pre-registered headline) | 0.181 / 0.272 / 0.303 / 0.335 | 0.234 / 0.287 / 0.306 / 0.336 |
| M9 on the default encoder (batch 1 best) | 0.314 / 0.398 / 0.472 / 0.492 | 0.241 / 0.360 / 0.396 / 0.439 |
| M9, SimCLR 3-seed mean | 0.261 / 0.355 / 0.421 / 0.445 | 0.246 / 0.372 / 0.408 / 0.458 |
| M9, MAE 3-seed mean | 0.293 / 0.399 / 0.472 / 0.499 | 0.282 / 0.428 / 0.479 / 0.515 |
| M11 ens_simclr | 0.264 / 0.360 / 0.420 / 0.441 | 0.252 / 0.365 / 0.406 / 0.460 |
| M11 ens_all (3 SimCLR + 3 MAE) | 0.342 / 0.421 / 0.491 / 0.503 | 0.263 / 0.407 / 0.459 / 0.515 |
| **M11 ens_mae (3 MAE seeds)** | **0.383 / 0.486 / 0.545 / 0.562** | **0.282 / 0.455 / 0.520 / 0.574** |
| Concatenated MAE embeddings + M9 | 0.297 / 0.407 / 0.458 / 0.490 | 0.299 / 0.461 / 0.515 / 0.557 |

### Findings
1. **M11 ens_mae is the best Task 3 system** (post-hoc).
   - It beats M9 on the default encoder in **40 / 40** paired draws in A→B (by +0.07 to +0.09) and **36 / 40** in B→A (by +0.04 to +0.14).
   - It beats the pre-registered M1 in 79 / 80 draws, by +0.21 to +0.24 at K ≥ 5.
   - In B→A it beats even the best single MAE seed (0.574 vs 0.555 at K = 20).
2. **Ensembling removes the seed luck.** The MAE ensemble sits well above the 3-seed mean in both directions (+0.06 A→B at K = 20). Averaging probabilities works better than concatenating embeddings (0.562 vs 0.490 A→B).
3. **Fair encoder comparison (3 seeds each): MAE beats SimCLR** by about +0.05 at K ≥ 10 in both directions. Ensembling helps MAE a lot but SimCLR not at all (0.441 vs 0.445 mean). Mixing all six encoders dilutes the MAE ensemble.
4. **Which classes transfer now** (K = 20, per-class F1):
   - A→B: sit 0.54, stand 0.44, walk 0.15;
   - B→A: stand 0.48, sit 0.21, walk 0.20.

   **Walk remains the hardest shared class.** This matches the reversed motion cue from idea 1.
5. **Status:** post-hoc ablation (designed after U6–U8). The pre-registered default stays M1, and the report shows both. The SimCLR seed 0 encoder is Abhinav's committed one; seeds 1 and 2 were trained on this machine (PyTorch 2.9.1). MAE checkpoints are in `outputs/task2/t3_*` (not committed).

## Suggested order
1. **Idea 1 diagnostic** (10 min). Decide whether motion features are worth building.
2. **Ideas 1 + 2 (+ 3)** as one post-hoc experiment, with settings fixed in advance (~1 day).
3. **Idea 6 (MAE)** if time allows (~1 hour).
4. Everything else → "future work" in the Task 3 report.
5. Then **U10** (report + final deliverables). The report includes whatever these attempts show, positive or negative.

## How to report any of these
- A new row in the ablation table (`ablations_all.csv`), produced by a committed script.
- Labelled **"post-hoc (designed after U6–U8)"**, with its pre-fixed settings listed.
- Compared against M1 (default) and M5 (best ablation) at every K, both directions, mean ± std over 10 draws.
