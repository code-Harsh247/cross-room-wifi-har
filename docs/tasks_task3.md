# Task 3: "New Room, Few Labels" Task Tracker

**Goal:** Recognise sit/stand/walk in a target room **with no labels for them**, and its room-only activities from **K captures** (K = 1, 5, 10, 20). Both directions (A → B, B → A). Metric: macro-F1 on target validation, plus target-exclusive and shared macro-F1.
**Spec:** [PRD_Task3_Few_Shot_Cross_Room.md](PRD_Task3_Few_Shot_Cross_Room.md) (read it first, especially §4 and §6)
**Builds on:** Task 1 router ([../reports/task1_report_section.md](../reports/task1_report_section.md) §9) · Task 2 encoder ([../reports/task2_report_section.md](../reports/task2_report_section.md) §9)
**Deadline:** Oct 31, 2026 · **Presentation:** Nov 2/3, 2026

Legend: `[x]` done · `[ ]` to do · 🔒 **Gate**: every check must pass before the next milestone. If a gate fails, fix it first and record what was wrong in the Decisions log.

---

## How to work through this file (with Claude Code)
1. Run `git pull --rebase` first. Then do **one milestone at a time**. Example prompt:
   > Read CLAUDE.md, docs/PRD_Task3_Few_Shot_Cross_Room.md and docs/tasks_task3.md. Do milestone U1 only, then run its gate and show me the output.
2. A milestone is done only when its gate passes. Tick its boxes and paste the key numbers into the **Results log**.
3. **Commit after each passed gate** (e.g. `Add Task 3 support sampler (U2)`), with no AI attribution lines. Push small commits often; never force-push.
4. **Two non-negotiables:**
   - Target validation windows are only ever scored.
   - Target labels outside the K support captures are never seen by any method, and that includes choosing the method.

---

## Status at a glance

| # | Milestone | Status |
|---|---|---|
| U0 | Sync + setup check | ✅ |
| U1 | Config + protocol + access guard | ✅ |
| U2 | Support sampling | ✅ |
| U3 | Embedding caches (+ AdaBN) | ✅ |
| U4 | Adaptation + classifiers | ✅ |
| U5 | Metrics + episode runner | ✅ |
| U6 | Main results (default M1 vs M0, bounds) | ✅ (check 4 re-specified, see Decisions log) |
| U7 | Routing: oracle vs predicted | ✅ |
| U8 | Ablations + controls | ✅ |
| U9 | Figures | ✅ |
| U10 | Report + final deliverables | ⬜ |

---

## U0: Sync + setup check
- [x] `git pull --rebase` (Task 2 work, the committed encoder `models/task2_default/` and `results/`)
- [x] `python scripts/build_windows.py --rebuild` (if `data_cache/` is missing or stale)
- [x] Task 1 router present at `outputs/task1/router/room_router.joblib`. If missing, run `python scripts/task1_router_variants.py` (~75 s).
- [x] Skim the Task 2 report §1, §8 and §9 (cross-room transfer at chance; embedding organised by room).

🔒 **Gate U0**
| Check | Command | Pass if |
|---|---|---|
| Tests green | `python -m pytest -q` | `86 passed` (0 skipped) |
| Cache | `python scripts/build_windows.py --rebuild` | `wrote 4680 windows`, `excluded captures: 7` |
| Encoder loads | `python -c "import sys; sys.path.insert(0,'src'); from xroom.config import load_task2_config; from xroom.dataset import load_windows; from xroom.encoder import load_encoder, embed; c=load_task2_config(); x,m=load_windows(c); print(embed(load_encoder(c), x[:8]).shape)"` | `(8, 128)` |
| Router loads | `python -c "import sys; sys.path.insert(0,'src'); from xroom.config import load_config; from xroom.router import load_router; print(type(load_router(load_config())).__name__)"` | `RoomClassifier` |

---

## U1: Config + protocol + access guard (PRD FR-1, §4)
- [x] `configs/task3.yaml` (`base_config: task2.yaml`):
  - `task3.k_values: [1, 5, 10, 20]`, `task3.draws: 10`, `task3.seed_base: 0`;
  - method list;
  - **default method `M1`** (decision D1), committed **now**, before any result exists;
  - output paths under `outputs/task3`.
- [x] `src/xroom/fewshot.py`:
  - [x] `Protocol(direction)`: source/target room, shared = {sit, stand, walk}, source-only and target-only activity sets, target label space;
  - [x] masks: source-dev, target-dev, target-val, source-val;
  - [x] `visible_labels(protocol, meta, support_ids) -> pd.Series` (the access guard): returns source-dev activity labels and support-capture labels only; every other label is NA.
- [x] `tests/test_fewshot.py` (protocol + guard)

🔒 **Gate U1**
| Check | How | Pass if |
|---|---|---|
| Label sets | test, both directions | A → B target space = {empty, lying, sit, stand, walk}; B → A = {fall, sit, sitdown, stand, standup, walk}; target-only = {empty, lying} / {fall, sitdown, standup} |
| Guard: no target leakage | test | for random support sets, every non-NA visible target label belongs to a support capture |
| Guard: no validation labels | test | every visible label has `split == "dev"` |
| Guard: sentinel | test: overwrite one non-support target-dev label with `"SENTINEL"` | `"SENTINEL"` never appears in `visible_labels` |
| Default fixed early | `git log -1 -- configs/task3.yaml` | D1 (`default: M1`) is committed before any U6 result file exists |
| Suite | `python -m pytest -q` | all pass |

---

## U2: Support sampling (PRD FR-2, D3)
- [x] `sample_support(meta, protocol, k, seed) -> list[capture_id]`: K captures per target-only class, from **target dev** only, deterministic, **nested** across K for a fixed seed (draw 20 once, take prefixes).
- [x] `scripts/task3_support.py` writes `outputs/task3/support/{AtoB,BtoA}_seed{r}.json` for r = 0..9.
- [x] Tests

🔒 **Gate U2**
| Check | Pass if |
|---|---|
| Counts | exactly K **captures** per target-only class (Room B support has 3K windows per class, Room A K windows) |
| Dev only | no support capture has `split == "val"` |
| Nested | support(K=1) ⊂ support(K=5) ⊂ support(K=10) ⊂ support(K=20) for every seed |
| Deterministic | same seed gives an identical manifest; different seeds differ |
| Coverage | across 10 seeds at K = 20, ≥ 100 distinct captures per class are used (no tiny pool) |
| Suite | all tests pass |

---

## U3: Embedding caches + AdaBN (PRD FR-3, M3)
- [x] `scripts/task3_embed.py` caches (n = 4680, d) arrays in `outputs/task3/emb/`:
  - `default.npy` (Task 2 default encoder);
  - `handcrafted.npy` (`xroom.probe.handcrafted_features`);
  - `random_init_s0.npy`.
- [x] `xroom.adapt.adabn(encoder, windows)`: copy the encoder and re-estimate BatchNorm running stats in train mode, no grad, cumulative average over **unlabelled dev windows of one room**. Produce `adabn_A.npy` / `adabn_B.npy` (each room's windows embedded with its own BN stats).
- [x] Tests

🔒 **Gate U3**
| Check | Pass if |
|---|---|
| Cache consistency | `default.npy` equals `embed(load_encoder(cfg), x)` (max abs diff < 1e-5) |
| AdaBN data | BN stats re-estimated from that room's `split == "dev"` windows only (assert in code + test) |
| AdaBN sanity | AdaBN embeddings finite, per-dim std > 1e-2; within-room probe (`xroom.probe.linear_probe`, dev → dev-holdout) not lower than the default by > 0.05. If it is, BN re-estimation is broken |
| Original untouched | the default encoder's weights and BN buffers are unchanged after `adabn` (it works on a copy) |
| Shapes / dtype | all caches (4680, d) float32, finite |
| Suite | all tests pass |

---

## U4: Adaptation + classifiers (PRD FR-4, §6)
- [x] `xroom.adapt`:
  - `per_room_standardize(f, rooms, fit_mask)` (stats from `fit_mask` rows of each room = dev rows);
  - `coral(source, target)`;
  - `refine_prototypes(protos, unlabeled, fixed=support, iters, temp)` (soft k-means; iterations and temperature fixed in config).
- [x] `xroom.fewshot`:
  - `prototypes(f, labels)`;
  - `nearest_centroid(f, protos, metric="cosine")`;
  - `logreg_classifier(...)` (M5).
- [x] Tests on toy data

🔒 **Gate U4**
| Check | Pass if |
|---|---|
| Per-room standardization | each room's dev rows have mean ≈ 0 and std ≈ 1 per dim (tol 1e-5); val rows are transformed with dev stats only (test: changing a val row does not change the stats) |
| CORAL | on toy Gaussians, the source covariance after CORAL matches the target within 1e-3 |
| Nearest centroid | toy separable clusters are classified 100%; cosine metric is scale-invariant |
| Refinement | on toy data with shifted clusters, refined prototypes move toward the true centres; support-pinned prototypes keep their class; deterministic |
| No labels used | refinement and normalization signatures take no label argument except support labels (code review + test) |
| Suite | all tests pass |

---

## U5: Metrics + episode runner (PRD FR-5, §4)
- [x] `src/xroom/task3_metrics.py`:
  - `episode_scores(y_true, y_pred, protocol)` → overall / exclusive / shared macro-F1, per-class F1, confusion matrix (reuse `xroom.probe.macro_f1`, `per_class_f1`);
  - `aggregate(draws)` → mean ± std, plus a capture bootstrap CI of the mean.
- [x] `scripts/task3_run.py --methods M0 M1 ... --sources default ... --directions AtoB BtoA` → `outputs/task3/runs/<direction>/<source>/<method>/K{k}_seed{r}.json` + `summary.json`.
- [x] Tests

🔒 **Gate U5**
| Check | Pass if |
|---|---|
| Metric correctness | toy check: overall = sklearn macro-F1 over the target labels; exclusive = mean of the room-only per-class F1s; shared = mean of the sit/stand/walk F1s |
| Deterministic | two runs of `task3_run.py --methods M0 --sources default` give byte-identical JSON |
| Guard enforced | the runner obtains labels only through `visible_labels` (grep + test with the sentinel label) |
| Smoke run | `--methods M0 M1 --sources default --k 5 --draws 2` finishes; all scores in [0, 1]; confusion rows sum to the per-class val counts |
| Suite | all tests pass |

---

## U6: Main results (PRD §6, D1)
- [x] Run the default **M1** and the lower bound **M0** with the default encoder: both directions, K ∈ {1, 5, 10, 20}, 10 draws.
- [x] Reference rows: **target-supervised upper bound** (prototypes from all target dev labels; labelled "not a valid Task 3 result") and **chance**.
- [x] Record the commit of `configs/task3.yaml` that fixed D1 next to the results (`2503cfd`, U1).

🔒 **Gate U6**
| Check | Pass if |
|---|---|
| Complete | 2 directions × 4 K × 10 draws × {M0, M1} episode files exist; `summary.json` has mean ± std for overall / exclusive / shared |
| Sanity: bounds | for every K, M1 overall ≤ upper bound + 0.02; all scores > 0 |
| Sanity: K trend | mean exclusive macro-F1 at K = 20 ≥ at K = 1 in both directions (if not, inspect support sampling before going on) |
| Consistency with Task 2 | M0 shared macro-F1 is near the Task 2 cross-room preview (≈ 0.2–0.35). A very different value means a bug in label spaces or prototypes |
| Default untouched | D1 is unchanged since U1 (`git diff <U1 commit> -- configs/task3.yaml` shows no change to `default`) |
| Results log | numbers filled in |

---

## U7: Routing: oracle vs predicted (PRD FR-7, D6)
- [x] `scripts/task3_routing.py`: pool the val windows of both rooms. Route with `route(mode="oracle")` and `route(mode="predicted", router=load_router(cfg))` (strict). Target-routed windows → episode model (M1); source-routed → source nearest-centroid model. Every K, both directions, 10 draws.
- [x] Report target-window macro-F1 and pooled macro-F1 for each routing.

🔒 **Gate U7**
| Check | Pass if |
|---|---|
| Oracle = main result | the oracle-routing target-window macro-F1 equals the U6 M1 numbers exactly |
| Router accuracy | predicted routing room accuracy on pooled val = 1.000 (as in Task 1); if lower, list the misrouted windows |
| Difference reported | predicted − oracle macro-F1 per K and direction is saved (expected ≈ 0) |
| Strict router | the default run uses `kind="strict"`; the activity-tuned router only appears in a clearly labelled stress test |

---

## U8: Ablations + controls (PRD FR-6, FR-9)
- [x] Method ladder on the default encoder: M2 (CORAL), M3 (AdaBN + M1), M4 (refinement), M5 (logreg), M6 (input-level per-room norm). All K, both directions, 10 draws.
- [x] Feature sources with M1: hand-crafted, random-init.
- [x] Controls:
  - **shuffled support** (labels permuted across target-only classes);
  - **contiguous Room B check** for A → B (support outside a contiguous val block; reuse `contiguous_split`).
- [x] Table at K = 5 (`outputs/task3/ablations_K5.csv`) + full table.

🔒 **Gate U8**
| Check | Pass if |
|---|---|
| Complete | every ablation row has all 4 K × 2 directions × 10 draws |
| Shuffled support | exclusive macro-F1 drops to about chance (≤ 1 / n_target_only + 0.1) |
| Default unchanged | the M1 row equals U6 exactly |
| No re-selection | if an ablation beats M1, it is marked "ablation, not selected (would need target labels)"; D1 stays |
| Takeaways | Results log: 3–5 bullets on what helps / what does not |

---

## U9: Figures (PRD FR-8)
- [x] `scripts/task3_figures.py`:
  - `task3_label_efficiency.png`: overall + exclusive macro-F1 vs K, both directions, M1 vs M0, upper bound and chance lines, ± std bands;
  - `task3_confusion_K5.png`: M1, both directions, row-normalized with counts, plus a per-class P/R/F1 table in `outputs/task3/per_class_K5.csv`;
  - `task3_routing.png`: oracle vs predicted bars;
  - optional `task3_umap_alignment.png`: before/after per-room normalization.
- [x] Copy to `reports/figures/`.

🔒 **Gate U9**
| Check | Pass if |
|---|---|
| Files exist | the three required PNGs in `reports/figures/` |
| Correct data | the plotted means equal `summary.json` values (spot-check 3 points) |
| Readable | axes labelled, legends not covering data, both directions shown, K on the x-axis as 1/5/10/20 |
| Interpretation | 2–3 sentences per figure in the Results log |

---

## U10: Report + final deliverables (PRD FR-10)
- [ ] `scripts/task3_report_numbers.py` → `outputs/task3/report_numbers.md`. `scripts/task3_check_report.py` checks the report against it.
- [ ] `reports/task3_report_section.md` (Task 1/2 structure):
  - task + summary;
  - protocol (episode table, access rules);
  - methods (D1 + ladder);
  - main results table (all K, both directions, overall / exclusive / shared);
  - label-efficiency figure;
  - confusion @ K = 5 + per-class table;
  - oracle vs predicted;
  - ablations + controls;
  - limitations;
  - reproduction.
- [ ] **Final deliverables check:** a table mapping every slide-12 item to its section and figure across Tasks 1–3. Put it in the Task 3 report or a short `reports/deliverables_checklist.md`.
- [ ] `python scripts/snapshot_results.py` → commit `results/task3/`.
- [ ] Update `README.md` (Task 3 commands, expected numbers, test count), `CLAUDE.md` (Task 3 status) and `docs/ai_use_and_contributions.md`.

🔒 **Gate U10**
| Check | Pass if |
|---|---|
| Numbers verified | `python scripts/task3_check_report.py` reports every number found |
| Deliverables | every slide-12 item has a ✔ and a link that resolves on GitHub |
| Reproducible | a fresh clone + README commands rebuild every Task 3 number (or the README says which saved results to use) |
| Tests | `python -m pytest -q` all pass; README states the new test count |
| Rules checklist | all ticked |

---

## Rules checklist (check before reporting any result)
- [ ] Target validation windows only scored (no normalization, refinement, AdaBN or selection on them)
- [ ] Target labels only from the K support captures (access guard + sentinel test pass)
- [ ] K counts captures; support drawn from target dev only
- [ ] Default method (D1) fixed before results and never re-selected
- [ ] The Task 2 encoder and Task 1 router are used as published (label-clean)
- [ ] Python only; no authors' MATLAB code; no other team's code
- [ ] Small commits, no AI attribution lines, no force-push

## Decisions log
| Date | Decision | Why |
|---|---|---|
| 2026-10-21 | D1: default method = M1 (per-room embedding standardization + nearest-centroid prototypes, Task 2 default encoder, cosine) | Follows the brief's hints and the Task 2 hand-off; no tunable hyperparameters; no target labels beyond K |
| 2026-10-21 | D2: no transductive use of target validation windows; test-time adaptation uses unlabelled target dev only | "Validation is for scoring only", kept literal |
| 2026-10-21 | D3: 10 draws (seeds 0–9), nested support across K | Smooth, comparable label-efficiency curves; variance estimate |
| 2026-10-21 | D4: room-only prototypes = mean over all windows of the K support captures | Room B captures have 3 windows; K counts captures |
| 2026-10-21 | D5: source-only classes excluded from the target label space | The target room's activity set is known |
| 2026-10-21 | D6: headline uses oracle routing (target val windows); predicted routing reported alongside | The brief asks to compare with the true room; the router is 1.000 on val |
| 2026-10-21 | AdaBN caches are full arrays per room (`adabn_A.npy`, `adabn_B.npy`, all 4680 rows each); M3 takes each window from its own room's array | Simple aligned indexing; the other room's rows are unused |
| 2026-10-21 | M6 (input-level per-room normalization) needs its own embedding cache; it is added in U8 with the other ablations, not in the U5 runner | U5 only needs the method harness; M0–M5 + UB + CHANCE are implemented |
| 2026-10-21 | U6 gate check 4 re-specified as a like-for-like comparison: 3-class sit/stand/walk transfer (source prototypes → target) vs the Task 2 cross-room preview. The original wording compared Task 2's 3-class probe with shared F1 inside the 5/6-class target space, where target-only prototypes also compete | Different quantities; the like-for-like value (0.11–0.23) matches Task 2 (≤ chance) and the extra drop is explained by diagnostic (c). Nothing about the method or D1 was changed |
| 2026-10-21 | M6 caches fit each room's input normalization on its dev 150-row windows (`room_input_stats`), not on full sequences | Same windows as every other Task 3 feature; dev only |
| 2026-10-21 | M5 (logistic regression) beats the default M1 in 7 of 8 cells but is reported as an ablation, not promoted | Promoting it would select with target validation labels (D1); the report states the gap |
| | | |

## Results log (fill in at each gate)
| Gate | Run / commit | Key numbers |
|---|---|---|
| U0 | `f2c1aee` | 4680 windows, 7 excluded captures (rebuild 54 s); 86 passed, 0 skipped; `embed(...)` → (8, 128); router `RoomClassifier` (strict, from `outputs/task1/router/`); torch 2.9.1+cu126, CUDA True |
| U1 | U1 commit ("Add Task 3 config, protocol and label access guard (U1)") | D1 (`default_method: M1`) and every ablation hyperparameter (CORAL eps 1.0, AdaBN batch 256, refinement 10 iters / τ 0.1, logreg C 1.0) committed in `configs/task3.yaml` while `outputs/task3/` did not exist (no Task 3 result yet). 13 new tests: label sets both directions; masks partition rows; guard shows only source-dev + all windows of support captures (20 random supports per direction, toy data) and on real data; no val label visible; sentinel label in hidden target-dev and val rows never visible; invalid support (val, shared class, source room, unknown ID) raises. Config activity sets match the data. Suite 99 passed. Added `load_task3_config` to `xroom.config`. |
| U2 | U2 commit | 20 manifests (`outputs/task3/support/{AtoB,BtoA}_seed{0..9}.json`), byte-identical on re-run. Per class: K captures (B targets 3K windows, A targets K windows); all target dev; nested K=1 ⊂ 5 ⊂ 10 ⊂ 20; same seed identical, different seeds differ. Distinct captures used at K=20 over 10 draws (pool): empty 135 (242), lying 145 (326), fall 128 (177), sitdown 114 (155), standup 120 (155), all ≥ 100. Per-class generator = (seed, crc32(direction/class)). Suite 105 passed. |
| U3 | U3 commit | Caches (4680 rows, float32, finite) in 16 s: default 128-d, handcrafted 484-d, random_init_s0 128-d, adabn_A / adabn_B 128-d (each room's BN stats from its own dev windows; M3 uses room-matched rows). default.npy vs fresh `embed` max diff 0.0. AdaBN per-dim std min 0.37. Dev-holdout within-room probe default A 0.918 / B 0.987 vs AdaBN A 0.920 / B 0.979 (drop −0.003 / 0.008 ≤ 0.05). random_init_s0 dev-holdout reproduces the Task 2 saved value exactly (A 0.951 / B 0.804). (Default dev-holdout 0.918 / 0.987 vs Task 2 saved 0.913 / 0.985: same small cross-machine probe-tuning difference seen before; val numbers matched exactly.) Tests: original encoder untouched; only BN running stats change; val windows and other room's windows have no effect; deterministic; no-dev room raises; cache = encoder. Suite 111 passed. |
| U4 | U4 commit | Toy-data gate: per-room standardization gives dev mean 0 / std 1 per room (tol 1e-5) and val rows never change the stats; CORAL (`coral_fit` / `coral_apply`) matches the target covariance and mean within 1e-3; separable clusters 100% with cosine and euclidean nearest centroid, cosine scale-invariant; refinement (soft k-means on the unit sphere, support pinned) moves biased prototypes to cosine > 0.98 with the true centres, keeps class identity, deterministic; `per_room_standardize`, `coral_fit`, `adabn` take no label argument and `refine_prototypes` only support indices; `logreg_classifier` ignores unlabelled rows. Suite 117 passed. |
| U5 | U5 commit | `episode_scores` overall = sklearn macro-F1 over target labels; exclusive / shared = means of the per-class F1s (toy); confusion-based F1 = `per_class_f1`; draw-mean bootstrap (captures resampled within class, same resample for all draws) brackets the mean and collapses to [1, 1] for perfect predictions. Methods M0–M5 in `xroom.task3_methods` read labels only via `visible_labels`: sentinel test on real data (every non-support target label incl. all val set to `SENTINEL`) leaves predictions of all six methods unchanged in both directions; the only direct `meta["activity"]` read is the separate UB reference. Two full `--methods M0` runs (2 dirs × 4 K × 10 draws) byte-identical. Smoke `--methods M0 M1 --k 5 --draws 2`: scores in [0, 1], confusion rows = val counts per class (B 75 each; A 25, sit 24), support 10 / 15 captures. Suite 123 passed. |
| U6 | U6 commit (D1 fixed at `2503cfd`) | Gate checks 1–3 pass: 40 episodes per (direction, method); M1 ≤ UB + 0.02 at every K; exclusive rises K1 → K20 for M0 and M1 in both directions. **Overall / exclusive / shared macro-F1 (mean of 10 draws):** A→B M1: K1 0.181/0.342/0.074; K5 0.272/0.532/0.098; K10 0.303/0.602/0.104; K20 0.335/0.664/0.115. A→B M0: K1 0.171/0.313/0.077; K5 0.222/0.459/0.064; K10 0.224/0.501/0.040; K20 0.245/0.552/0.041. B→A M1: K1 0.234/0.362/0.105; K5 0.287/0.500/0.073; K10 0.306/0.545/0.067; K20 0.336/0.592/0.081. B→A M0: K1 0.212/0.359/0.065; K5 0.236/0.462/0.011; K10 0.247/0.492/0.003; K20 0.270/0.540/0.000. UB (all target dev labels, not valid) A→B 0.901, B→A 0.692; CHANCE A→B 0.201, B→A 0.180. M1 beats M0 overall at every K in both directions. **Check 4 (as written) fails:** M0 shared is 0.00–0.08, not ≈ 0.2–0.35. Diagnosis (`scripts/task3_diagnostics.py`): (a) source prototypes classify source val at 0.625 (A, 6-class) / 0.893 (B, 5-class), so labels and prototypes work; (b) like-for-like with the Task 2 preview (3-class sit/stand/walk, source prototypes → target) M0 0.155 / 0.135, M1 0.114 / 0.231: below chance (0.333), consistent with Task 2 (0.226 / 0.228, linear probe); (c) in the 5/6-class episodes most target sit/stand/walk windows are absorbed by the same-room target-only prototypes at K = 5 (M0 56% A→B, 94% B→A; M1 24% / 60%), which is why shared F1 is below the 3-class value. No bug; check 4 compared different quantities (see Decisions log). D1 unchanged since U1 (`git diff 2503cfd -- configs/task3.yaml` empty). |
| U7 | U7 commit | Pooled val of both rooms (524 windows); strict router accuracy 1.0000, 0 misrouted. Oracle-routing target macro-F1 equals the U6 M1 overall score for every draw, K and direction. Predicted − oracle = 0.000 for target and pooled macro-F1 at every K (saved in `routing/summary.json`). Pooled 8-activity macro-F1 (oracle = predicted): AtoB K1 0.429, K5 0.484, K10 0.504, K20 0.523; BtoA K1 0.631, K5 0.688, K10 0.704, K20 0.725. Activity-tuned router not used (no stress test needed at 1.000). New sentinel test: the source model (`predict_source`) never sees target labels. Suite 125 passed. |
| U8 | U8 commit | Gate: 24 rows (2 directions × default ladder M0–M6 + UB / CHANCE / SHUF, + M1 on handcrafted and random_init_s0), each 4 K × 10 draws; SHUF exclusive 0.198–0.344 ≤ 1/n + 0.1 (0.600 A→B, 0.433 B→A); M1 summary byte-identical to U6 after re-running; tables `ablations_all.csv`, `ablations_K5.csv`. **Takeaways (overall macro-F1, mean of 10 draws):** (1) per-room embedding normalization helps: M1 > M0 at every K (K20 A→B 0.335 vs 0.245, B→A 0.336 vs 0.270); (2) **logistic regression on the same features (M5) is the strongest method** and the gap grows with K (K20 A→B 0.438, B→A 0.409); it beats M1 in 7 of 8 (direction, K) cells — **ablation, not selected (choosing it would use target validation labels); D1 stays M1**; (3) CORAL (M2) helps only B→A (K20 0.351) and hurts A→B; AdaBN (M3), prototype refinement (M4) and input-level per-room normalization (M6) do not help and mostly hurt (M4 A→B K20 0.210, shared F1 ≈ 0: refinement pulls target sit/stand/walk into the target-only clusters); (4) feature source: the Task 2 encoder beats hand-crafted features at every K in both directions (K20 A→B 0.335 vs 0.211); the random-init encoder is level with it in A→B at K ≤ 5 (K1 0.284 vs 0.181) but behind at K ≥ 10 and in B→A; (5) **contiguous Room B check** (A→B, M1, support and val not adjacent): exclusive drops -0.124 / -0.128 / -0.107 / -0.136 at K = 1/5/10/20 (K20 0.664 → 0.528; overall 0.335 → 0.259): the official Room B-as-target numbers are optimistic because support captures neighbour validation captures in the original recordings. |
| U9 | U9 commit | Files: `reports/figures/task3_label_efficiency.png`, `task3_confusion_K5.png`, `task3_routing.png`; per-class table `outputs/task3/per_class_K5.csv`. Spot-check: 4 plotted values (A→B M1 overall K5, B→A M5 exclusive K20, A→B UB, B→A predicted routing K10) equal `summary.json` exactly. Optional UMAP skipped (umap-learn not installed on this machine; not required). **Interpretation.** *Label efficiency:* room-only (exclusive) F1 climbs steadily with K in both directions and M1 stays above M0 from K = 5; overall F1 rises much less because the shared classes stay near zero; the M5 ablation separates from M1 from K = 5 on, and the upper bound shows how much is lost by having no target labels for sit/stand/walk. *Confusion @ K = 5:* the room-only classes are recognised (F1 A→B empty 0.49 / lying 0.57; B→A fall 0.52 / sitdown 0.47 / standup 0.51), while shared classes are **systematically swapped**, not just noisy: in A→B target sit and stand are mostly predicted as walk (53% / 70%) and walk as empty (45%); in B→A sit goes to sitdown, stand to fall and walk to stand. Source prototypes land on the wrong target clusters, so activity identity does not transfer between rooms in this embedding. *Routing:* oracle and predicted bars are identical at every K (router accuracy 1.000), so the room router adds no error on this data. |
| U10 | | |
