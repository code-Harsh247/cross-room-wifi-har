# PRD — Task 3: "New Room, Few Labels" (Cross-Room Few-Shot Activity Recognition)

**Course:** CS60055 Ubiquitous Computing — Hackathon Challenge 2
**Project:** Cross-Room Few-Shot Adaptation for WiFi Activity Recognition
**Scope:** Task 3, plus the final cross-task deliverables check
**Status:** Draft v1 (2026-10-21)
**Deadline (whole project):** October 31, 2026 · **Presentation:** November 2/3, 2026
**Step-by-step checklist:** [tasks_task3.md](tasks_task3.md)

---

## 1. Summary

Move an activity recogniser from a **source room**, where every activity is labelled, to a **target room**:
- The target room's **shared activities** (sit, stand, walk) have **no labels**.
- Each of its **room-only activities** has labels for only **K captures**, with K ∈ {1, 5, 10, 20}.

The metric is **macro-F1** on the target room's validation windows. It is reported for both directions (A → B and B → A) and every K. Two splits of it are also reported: over the room-only ("target-exclusive") classes, and over the shared classes.

The system is built from the two earlier tasks:
- the frozen **Task 2 encoder** (`models/task2_default`) for embeddings;
- the **Task 1 router** (`outputs/task1/router/room_router.joblib`) to decide which room a window comes from, compared against the true room ("oracle vs predicted room").

**The central question** (slide 8): *can you get useful activity transfer across rooms with far fewer target labels?*

## 2. What the brief says

| Source | Requirement |
|---|---|
| Slide 8 | "Recognize SIT / STAND / WALK in a room with no labels for them, and learn room-only activities from just K captures." Metric: **Macro-F1**, **K = 1, 5, 10, 20**. Task 1's classifier "also routes windows; compare it with the true room". |
| Slide 9 | 1 capture = 1 unit. **K = number of captures, not windows.** Never train on validation or pick support from it. |
| Slide 10 | Labels allowed in Task 3: **source-room labels + the K support captures**. Not allowed: **target labels beyond the K support captures**. Data: development captures of both rooms. Normalization statistics fitted on development data only. Features only from the CSI signal. |
| Slide 11 hint | **Class prototypes or nearest centroid on frozen embeddings; per-room normalization; test-time adaptation.** Read up: **ProtoNets, AdaBN**. |
| Slide 12 report | **Task 3 macro-F1 + target-exclusive macro-F1, all K, A→B and B→A.** **Label-efficiency curve.** **Oracle vs predicted room.** **Confusion matrices at K = 5 with per-class metrics.** Ablation: what helps and what does not. AI-use and contribution disclosure. |

## 3. Starting point

| Asset | How to use it | Facts that matter |
|---|---|---|
| 150-row windows | `xroom.dataset.load_windows(cfg)` | Room A: 1 window per capture. Room B: 3 per capture. Columns `split` (dev/val) and `dev_fold` (train/holdout). |
| Task 2 encoder (label-clean) | `xroom.encoder.load_encoder(load_task2_config())`, `embed(enc, x)` → (n, 128) | Within-room probes: A 0.905, B 0.992. **Cross-room probe at or below chance (≈ 0.23 vs 0.33)**: the embedding is organised by room first (Task 2 report §1, §9). `FrozenEncoder.model` is a CNN with BatchNorm, so AdaBN is possible; `load_run` gives the module and its `NormStats`. |
| Task 1 router (strict) | `xroom.router.load_router(cfg)`, `route(windows, meta, mode="oracle"/"predicted", router=...)` | Routes validation windows with **1.000** balanced accuracy. Settings chosen with room labels only. |
| Metrics | `xroom.probe.macro_f1`, `per_class_f1`, `bootstrap_macro_f1`, `shuffle_by_capture` | Reuse; do not re-implement. |
| Config layering | `configs/task3.yaml` with `base_config: task2.yaml` (which layers on `task1.yaml`) | A layered config may not redefine base sections (enforced). |
| Report tooling pattern | `scripts/task2_report_numbers.py` + `scripts/task2_check_report.py`; `scripts/snapshot_results.py` → `results/` | Reuse the pattern: every report number is generated and checked. |

### Label sets and target data

| | Room A | Room B |
|---|---|---|
| Shared (no target labels) | sit, stand, walk | sit, stand, walk |
| Room-only (K-shot in target) | fall, sitdown, standup | empty, lying |
| Dev captures per room-only class | fall 177, sitdown 155, standup 155 | empty 242, lying 326 |
| Validation windows (scoring) | 149 (6 classes, ~25 each) | 375 (5 classes, 75 each) |

### Inherited pitfalls (must be handled)
1. **The format confound makes the rooms trivially separable** (Task 1 §6). Embeddings carry room identity strongly, which is why naive cross-room transfer fails.
2. **Room B consecutive chunks.** Room B support captures may sit next to validation captures in the original recording, so Room B-as-target scores may be optimistic. Report a contiguous-split check (`contiguous_split` in `scripts/task1_ablations.py`).
3. **Label hygiene.** Anything Task 3 uses must be chosen **without target activity labels beyond the K support captures**, and that includes method and hyperparameter *selection* (Task 1 router §5; Task 2 PROVENANCE). The Task 2 default encoder and the strict router already satisfy this.

## 4. Task definition (one "episode")

For a direction **S → T** (source → target), a value of K and a draw seed r:

| Data | Labels available? | Allowed use |
|---|---|---|
| Source dev windows | ✅ all source activity labels | training, prototypes, tuning |
| Source val windows | (scoring only) | only in the pooled routing evaluation (FR-7) |
| Target dev, **support**: K captures per **room-only** class, drawn with seed r | ✅ labels of those captures only | prototypes / training |
| Target dev, everything else (incl. all shared-class windows) | ❌ **unlabelled** | per-room normalization statistics, AdaBN, transductive refinement |
| Target val windows | ❌ never seen before scoring | **scoring only**; no transductive use (decision D2) |

- **Prediction label space:** the target room's activity set (shared + target room-only). It is known from the task definition and is not learnt.
- **Scores per episode,** on target val windows:
  - **overall macro-F1:** mean F1 over all target classes; the headline;
  - **target-exclusive macro-F1:** mean F1 over the target's room-only classes, computed from the same predictions;
  - **shared macro-F1:** mean F1 over sit, stand and walk;
  - per-class F1 and the confusion matrix.
- **Draws:** R = 10 draws per (direction, K), seeds 0–9.
  - **Nested support:** for a given seed, the K = 1 captures ⊂ K = 5 ⊂ K = 10 ⊂ K = 20. This makes the label-efficiency curve smooth and comparable.
  - Support manifests are saved as JSON.
- **Reporting:** mean ± std over draws, plus a capture-level bootstrap interval of the mean.

## 5. Goals and non-goals

### Goals
1. A reproducible episode harness (directions × K × draws × methods) that enforces the data-access table in code.
2. A **pre-registered default method**, fixed before any target-labelled score is seen (D1). It gets the headline numbers for every K and both directions.
3. The required figures: **label-efficiency curve**, **confusion matrices at K = 5**, **oracle vs predicted room**.
4. An **ablation ladder** showing what helps: per-room normalization, AdaBN, transductive refinement, classifier type, feature source.
5. Honest controls: a no-adaptation lower bound, a target-supervised upper bound (labelled as not a valid result), a shuffled-support control and the contiguous Room B check.
6. Report section + final deliverables check across all three tasks.

### Non-goals
- Training a new encoder. The Task 2 default stays frozen; AdaBN only re-estimates BatchNorm statistics.
- Using target validation windows for anything but scoring.
- Beating published results.

## 6. Methods

All methods work on frozen 128-d embeddings unless stated otherwise.

| ID | Method | What it adds | Uses |
|---|---|---|---|
| **M0** | **No adaptation.** Nearest centroid (cosine). Shared-class prototypes = mean of source dev embeddings; room-only prototypes = mean of the K target support captures. | Lower bound | source labels, support |
| **M1** | **+ per-room embedding normalization.** Standardize each room's embeddings with that room's mean and std, estimated on its **unlabelled dev** windows, before building prototypes. | "per-room normalization" (slide 11); Task 2 hand-off advice | unlabelled target dev |
| M2 | M1 + **CORAL**: whiten the source and re-colour it with the target covariance | Second-order alignment | unlabelled target dev |
| M3 | **AdaBN**: re-estimate the encoder's BatchNorm statistics on unlabelled target dev windows (separately per room), then M1 | "test-time adaptation / AdaBN" (slide 11) | unlabelled target dev |
| M4 | M1 + **transductive prototype refinement**: soft k-means on unlabelled target dev embeddings, starting from the M1 prototypes; a few iterations; support captures stay pinned to their labels | Test-time adaptation without val | unlabelled target dev |
| M5 | M1 features + **logistic regression** on source labelled + support embeddings (class-balanced) instead of nearest centroid | Classifier ablation | source labels, support |
| M6 | **Input-level per-room normalization**: encoder `NormStats` re-fitted per room on unlabelled dev rows | Normalization at the input instead of the embedding | unlabelled target dev |

- **Feature sources** (ablation): default Task 2 encoder (main), hand-crafted mean+std (`xroom.probe.handcrafted_features`), random-init encoder.
- **Reference bounds** (clearly labelled, not results):
  - **Target-supervised upper bound:** nearest centroid with prototypes from *all* target dev labels.
  - **Chance:** 1 / (number of target classes).

### D1 — Pre-registered default (decide before running)
**Default = M1** (per-room embedding normalization + nearest-centroid prototypes on the Task 2 default encoder).
- It follows directly from the brief's hints and the Task 2 hand-off, has no tunable hyperparameters, and uses no target labels beyond the support.
- **M2–M6 are ablations.** If one of them is better, it is reported as such, but it does **not** replace the default. Choosing it would use target validation labels or labels beyond K.
- If a hyperparameter is unavoidable (e.g. refinement iterations), fix it in `configs/task3.yaml` before running, or choose it with a **label-free** criterion only.

## 7. Functional requirements

### FR-1 Protocol and access guard
- `xroom.fewshot.Protocol`: the direction, label sets (shared / source-only / target-only), and masks for source-dev, target-dev-unlabelled, target-support and target-val.
- **Access guard:** a single function returns the labels a method may see for an episode. Every method gets labels only through it. Tests prove that no target label outside the support, and no validation label, reaches any method.

### FR-2 Support sampling
- `sample_support(meta, direction, K, seed)`: K **captures** per target room-only class, from **target dev** only. Nested across K for a fixed seed. Deterministic.
- Save manifests to `outputs/task3/support/<direction>_seed<r>.json`, and commit them under `results/` with the snapshot.
- Room B support windows = all 3 windows of each chosen capture. Room A = 1 window each.

### FR-3 Embedding cache
- Embed every 150-row window once per feature source, and cache to `outputs/task3/emb/<source>.npy`.
- AdaBN variants: one cache per (encoder, room-adapted BN) combination.
- Check: cached default embeddings equal `embed(load_encoder(...), x)` (max abs diff < 1e-5).

### FR-4 Adaptation and classifiers
- Pure functions in `xroom/adapt.py`: `per_room_standardize`, `coral`, `adabn(encoder, windows_by_room)`, `refine_prototypes`.
- Pure functions in `xroom/fewshot.py`: `prototypes`, `nearest_centroid` (cosine), and the logistic-regression variant.
- All statistics are fitted on dev rows only, with the room taken from metadata (oracle) or from the router (FR-7).

### FR-5 Episode runner and metrics
- `scripts/task3_run.py`: for each direction, method, feature source, K and draw seed, run the episode. Save per-episode metrics (overall, exclusive and shared macro-F1, per-class F1, confusion matrix) and an aggregate `summary.json` (mean ± std over draws, bootstrap CI).
- Deterministic: re-running gives byte-identical metrics.

### FR-6 Controls
- **Shuffled support:** K support captures with labels permuted across the room-only classes. Exclusive macro-F1 should fall to about chance.
- **Upper bound and chance** rows (section 6).
- **Contiguous Room B check** for A → B: support drawn from dev captures outside a contiguous validation block. Report the change in exclusive macro-F1.

### FR-7 Routing: oracle vs predicted room
- Pool the validation windows of both rooms. Route each window with `route(..., mode="oracle")` and with `mode="predicted"` (strict router), then classify it with that room's model:
  - target-routed windows → the adapted few-shot model of the episode (target label space);
  - source-routed windows → a source model (nearest centroid with source prototypes; source label space).
- Report target-window macro-F1 and pooled macro-F1 under both routings, for every K and both directions.
- Expected: nearly identical, because the router scores 1.000. Report it anyway (required).
- **Optional stress test:** the activity-tuned router, or a router restricted to the source room.

### FR-8 Figures
- **Label-efficiency curve:** overall and exclusive macro-F1 vs K (1, 5, 10, 20), one panel per direction, with mean ± std bands. Show the default vs M0, plus the upper bound and chance as lines.
- **Confusion matrices at K = 5:** default method, both directions, row-normalized, with counts, plus a per-class precision/recall/F1 table.
- **Oracle vs predicted room:** grouped bars per K and direction.
- Optional: UMAP of target-val embeddings before and after per-room normalization, coloured by activity, with shared classes highlighted.
- Copy final figures to `reports/figures/task3_*.png`.

### FR-9 Ablations
Report on target val, mean ± std over draws, at every K:
- **Method ladder:** M0 → M1 (default) → M2, M3, M4, M5, M6.
- **Feature source:** default encoder vs hand-crafted vs random-init, all with M1.
- One summary table at K = 5 and the full table in `results/`.

### FR-10 Report and final deliverables
- `reports/task3_report_section.md`, structured like the Task 1 and Task 2 sections, with every number produced by `scripts/task3_report_numbers.py` and checked by `scripts/task3_check_report.py`.
- **Final deliverables check:** a table mapping every slide-12 report item to the section and figure that answers it (Tasks 1–3).
- Update `docs/ai_use_and_contributions.md`, `README.md` (Task 3 commands, expected numbers, test count) and `CLAUDE.md` (status).

## 8. Non-functional requirements

| Area | Requirement |
|---|---|
| Rules | No validation data in fitting, normalization or selection. No target labels beyond K. Enforced by the access guard and tests. |
| Reproducibility | `configs/task3.yaml` (layered on `task2.yaml`). Every random choice is seeded. Support manifests are saved. Re-runs are byte-identical. |
| Compute | Embedding cache ≤ ~2 min. The full grid (2 directions × 4 K × 10 draws × ~7 methods × 3 sources) ≤ ~15 min on CPU. AdaBN embeddings ≤ ~2 min on GPU. |
| Code | Matches `src/xroom/` style. Tests for every module. `python -m pytest -q` stays green (86 tests today). |
| Results | `outputs/task3/` (ignored) → snapshot into `results/task3/` with `scripts/snapshot_results.py`. Figures in `reports/figures/`. |
| Git | Small commits, one per passed gate, no AI attribution lines. Pull with `--rebase` before work and before pushing. Never force-push. |

## 9. Proposed code layout

```
configs/task3.yaml                 # base_config: task2.yaml; K list, draws, methods, defaults (D1)
src/xroom/fewshot.py               # Protocol, access guard, support sampling, prototypes, classifiers
src/xroom/adapt.py                 # per-room standardize, CORAL, AdaBN, prototype refinement
src/xroom/task3_metrics.py         # overall/exclusive/shared macro-F1, confusion, aggregation (reuses xroom.probe)
scripts/task3_embed.py             # embedding caches (default, hand-crafted, random-init, AdaBN)
scripts/task3_run.py               # episode grid -> per-episode metrics + summary
scripts/task3_routing.py           # oracle vs predicted room
scripts/task3_figures.py           # label-efficiency, confusion @K=5, routing bars
scripts/task3_report_numbers.py, scripts/task3_check_report.py
tests/test_fewshot.py, test_adapt.py, test_task3_metrics.py
reports/task3_report_section.md, reports/figures/task3_*.png
```

## 10. Milestones

| # | Milestone | Done when |
|---|---|---|
| U0 | Sync + setup check | Pulled; 86 tests pass; encoder and router load; caches rebuilt |
| U1 | Config + protocol + access guard | Label sets and masks correct; guard tests prove no leakage |
| U2 | Support sampling | Nested, deterministic, dev-only K-capture support; manifests saved |
| U3 | Embedding caches | Default / hand-crafted / random-init caches; AdaBN embeddings; consistency check |
| U4 | Adaptation + classifiers | M0–M6 building blocks tested on toy data |
| U5 | Metrics + episode runner | Overall / exclusive / shared macro-F1; deterministic grid run |
| U6 | Main results | Default (M1) and M0 for both directions, all K, 10 draws; upper bound and chance |
| U7 | Routing | Oracle vs predicted for all K and both directions |
| U8 | Ablations + controls | Method ladder, feature sources, shuffled support, contiguous check |
| U9 | Figures | Label-efficiency, confusion @ K = 5, routing bars (+ optional UMAP) |
| U10 | Report + final deliverables | Report section with checked numbers; deliverables table; README / CLAUDE.md / AI-use updated |

## 11. Success criteria

- **Must:**
  - every number in section 4 is reported for both directions and all K, with mean ± std over 10 draws;
  - the three required figures exist;
  - the access-guard tests pass.
- **Must:** the default method is the one fixed in D1, unchanged after seeing results. Any better ablation is reported as an ablation.
- **Should:**
  - the default beats M0 on shared-class macro-F1 (the transfer part), in at least one direction, beyond draw noise;
  - exclusive macro-F1 rises with K.
- **Should:** the ablation table makes clear which adaptation step matters.
- **Could:** the UMAP before/after figure; a router stress test; the contiguous check in both directions.

No absolute macro-F1 target is set. Task 2 suggests shared-class transfer starts near chance, so any clear gain is the finding.

## 12. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Shared-class transfer stays near chance even with M1 | Report it honestly. Try M2–M4 as ablations. Show the per-class confusion to explain which classes fail. This is a legitimate answer to the brief's "open question". |
| Choosing the default after seeing validation | D1 is fixed in `configs/task3.yaml` and committed **before** U6. The report states the commit. |
| Target labels leak through normalization or refinement | The access guard, plus tests that inject a sentinel label and check it never reaches a method |
| AdaBN breaks the encoder (tiny batches, BN collapse) | Re-estimate on all target dev windows in large batches; check embedding std and the within-room probe before use |
| Room B-as-target optimism (consecutive chunks) | Contiguous check (FR-6); stated in limitations |
| K = 1 high variance | 10 draws, report std, nested support |
| Small Room A validation (25 windows per class) | Capture bootstrap CI; avoid over-reading differences < ~0.03 |

## 13. Decisions (record in tasks_task3.md)

- **D1 — Default method:** M1 (per-room embedding standardization + nearest centroid, Task 2 default encoder, cosine distance). Fixed before U6.
- **D2 — No transductive use of target validation windows.** Test-time adaptation uses only unlabelled target *dev* windows, which keeps "validation is for scoring only" literal.
- **D3 — Draws:** R = 10, seeds 0–9, nested support across K.
- **D4 — Room-only prototypes from support:** mean over all windows of the K support captures.
- **D5 — Source-only classes** are excluded from the target label space. The target activity set is known.
- **D6 — Routing evaluation** pools the validation windows of both rooms. The headline metric uses oracle routing (the target val windows), and predicted routing is reported alongside.
