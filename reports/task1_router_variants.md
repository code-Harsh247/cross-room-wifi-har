# Task 1 → Task 3: Router Variants (Milestone M7)

**Date:** 2026-10-18
**Command:** `python scripts/task1_router_variants.py` (~75 s)
**Outputs:**
- `outputs/task1/router_variants/` (`results.csv`, `summary.json`, `paired_vs_baseline_val.csv`, `router_variants_val.png`)
- `outputs/task1/router/`
  - `room_router.joblib` (strict, **Task 3 default**)
  - `room_router_activity_tuned.joblib` (ablation)
  - `*_predictions_val.csv` (validation predictions for each)
- Loading in Task 3: `xroom.router.load_router(cfg, "strict")` and `xroom.router.route(windows, meta, mode="oracle" | "predicted", router=...)`

---

## 1. Why
In Task 3 the room classifier becomes a **router**: it decides which room a window comes from, before activity recognition. It will see target-room activities it may never have been trained on. M6 found that the M5 model (150-row windows, C = 1000) transfers poorly to an unseen activity: trained on sit+stand, it gets only 0.86 on walking. M6 suggested two cheap fixes:

1. **Stronger regularization** (smaller C).
2. **Slice averaging:** cut each 150-row window into short slices, score every slice with a model trained on slices of that length, and average the slice probabilities.

## 2. Setup
- **Grid:** slice length s ∈ {150 (no averaging), 75, 50, 25, 15, 10} × C ∈ {0.01, 0.1, 1, 10, 1000}. Same per-row normalized spectrum features as M5.
- **The test unit is always a 150-row window**, which is what Task 3 will route.
- **Scenarios** (train activities → test activities):

  | Scenario | Train | Test |
  |---|---|---|
  | standard | all | all |
  | sit+stand → walk | still only | unseen walking |
  | shared → room-only | sit/stand/walk | unseen empty, lying, fall, sitdown, standup |
  | room-only → shared | room-specific activities | unseen sit/stand/walk |

- **Selection (development data only):**
  - Train on `dev_fold = train`.
  - The standard scenario is scored on `dev_fold = holdout`.
  - Transfer scenarios are scored on **all** dev captures of the test activities. Those activities are never trained on, so every one of those captures is unseen.
  - **Rule:** highest mean balanced accuracy over the three transfer scenarios, among variants with standard ≥ 0.995.
- **Validation** is reported for every variant, with a **paired capture-level bootstrap** (2,000 resamples) of each variant against the baseline.

**Process note:** the first run scored transfer scenarios on the holdout fold only. There, the walking test scored 1.000 for 27 of 30 variants, so it could not tell them apart. The rule was changed to use all unseen dev captures *after* that run's validation numbers had been printed. Both versions of the rule select the **same variant** (25 rows, C = 1), so the change did not alter the outcome.

## 3. Results

![router variants](figures/task1_router_variants_val.png)

### Selected vs baseline

| | Standard | sit+stand → walk | shared → room-only | room-only → shared | Transfer mean |
|---|---|---|---|---|---|
| **Dev** baseline (150 rows, C = 1000) | 0.999 | 0.973 | 0.918 | 0.989 | 0.960 |
| **Dev** selected (25 rows, C = 1) | 1.000 | 0.982 | 0.962 | 0.992 | **0.978** |
| **Val** baseline | 1.000 | 0.860 | 0.930 | 0.989 | 0.926 |
| **Val** selected | 1.000 | 0.893 | 0.953 | 0.998 | **0.948** |

### Paired bootstrap on validation (difference vs baseline, 95% CI)

| Variant | sit+stand → walk | shared → room-only | room-only → shared |
|---|---|---|---|
| Regularization only (150 rows, C = 0.1) | **+0.073** [+0.033, +0.127] | −0.010 [−0.033, +0.010] | −0.005 [−0.029, +0.015] |
| **Selected** (25 rows, C = 1) | +0.033 [+0.000, +0.067] | +0.023 [−0.003, +0.057] | +0.009 [+0.000, +0.020] |
| 15 rows, C = 1 | **+0.127** [+0.060, +0.200] | +0.013 [−0.027, +0.057] | +0.002 [−0.016, +0.018] |
| 10 rows, C = 1 | **+0.127** [+0.060, +0.200] | +0.013 [−0.027, +0.053] | −0.005 [−0.029, +0.013] |

Test sizes on validation:
- walking: 100 windows (25 A + 75 B)
- room-only: 225 windows
- shared: 299 windows

## 4. Findings
1. **Both fixes help, and neither costs anything on the standard task.** Every variant keeps standard validation accuracy at 1.000, except the most regularized one with no slicing (0.996).
2. **Slice averaging with short slices helps most on unseen walking.** With 10–15-row slices, walking goes from 0.86 to **0.987** (+12.7 points, CI clearly above 0). Room B walking windows called A drop from 28% to 3%.
3. **Regularization alone also helps on walking** (+7.3 points at C = 0.1), but does nothing for the other two scenarios.
4. **The selected variant (25 rows, C = 1) is a modest, consistent improvement on validation:** transfer mean 0.926 → 0.948, with small positive differences in all three scenarios (two at the edge of significance).
5. **Dev cannot separate the top variants.** On dev, 25-row and 15-row slices at C = 1 are a tie (transfer mean 0.97848 vs 0.97841). Validation shows a clearly better walking score for 15 rows, but choosing 15 rows on that basis would mean selecting on validation. We keep the dev choice and report the plateau instead.
6. **Recall A stays ≥ 0.96 everywhere.** The remaining errors are still mostly Room B windows called A.

## 5. Rule check: what may the Task 3 router use?

The rules slide allows **development captures of both rooms** as data. For Task 3 it allows **source-room labels + the K support captures**, and forbids **target labels beyond the K support captures**. Task 1's slide says the room classifier "also routes windows" in Task 3. The speaker notes are empty, so the slides are the only source.

| Part of the router | Uses | Allowed in Task 3? |
|---|---|---|
| Training data: all dev windows of both rooms | CSI + **room labels** (which room's archive a capture came from) | ✅ Room identity is not an activity label, and dev captures of both rooms are allowed data |
| Normalization (StandardScaler) | Dev windows only | ✅ |
| Choosing settings, **activity-tuned** (25 rows, C = 1) | **Activity labels of all dev captures of both rooms**, to build the unseen-activity scenarios | ❌ Uses target-room activity labels beyond K, even if only to pick 2 settings |
| Choosing settings, **strict** (below) | Room labels only | ✅ |

**Strict selection:**
1. Hold out a random 20% of dev captures per room, drawn without any activity information.
2. Train every variant on the rest, with all activities pooled.
3. Pick the lowest holdout log-loss.

The holdout log-loss is lowest for 50–75-row slices at large C. The winner is **50-row slices, C = 1000**. Holdout balanced accuracy is ≥ 0.995 for every variant.

| Validation | Standard | sit+stand → walk | shared → room-only | room-only → shared | Transfer mean |
|---|---|---|---|---|---|
| Baseline (150 rows, C = 1000) | 1.000 | 0.860 | 0.930 | 0.989 | 0.926 |
| **Strict** (50 rows, C = 1000) | 1.000 | 0.853 | 0.947 | 0.991 | 0.930 |
| Activity-tuned (25 rows, C = 1) | 1.000 | 0.893 | 0.953 | 0.998 | 0.948 |

The strict router's differences from the baseline are not significant: walking −0.007 [−0.033, +0.013], room-only +0.017 [−0.007, +0.043], shared +0.002 [−0.016, +0.018]. This is expected, because a room-label-only criterion rewards confidence on familiar data, not robustness to unseen activities.

**Why this costs Task 3 nothing in practice:** both routers are trained on all dev windows of both rooms, so **every activity is familiar to them**. On the actual Task 3 routing job (validation windows), both score **1.000**. The unseen-activity robustness studied above would only matter if the router's training data were restricted.

## 6. Decision (M7 closed)
- **Task 3 default router = strict** (`room_router.joblib`): `RoomClassifier(variant="row", C=1000, slice_rows=50)`, trained on all dev 150-row windows of both rooms with room labels only. Validation routing accuracy is 1.000.
- **Activity-tuned router** (`room_router_activity_tuned.joblib`): kept as a Task 1 analysis and possible Task 3 ablation. It must be labelled as using dev activity labels to pick two settings.
- **Oracle vs predicted routing:** Task 3 reports both via `route(..., mode="oracle")` and `route(..., mode="predicted", router=load_router(cfg))`.
- **Task 1 headline result:** unchanged, still the M5 model.
- **If an instructor rules that room labels of target-room dev data are also off-limits,** the router would need retraining on source data plus the K support captures only. Only the training set would change; `RoomClassifier` and `route` stay the same.

## 7. Caveats
- Validation transfer tests are small (100–299 windows). The bootstrap intervals are wide.
- These scenarios imitate "unseen activity". Task 3's real situation (a new room's activities with only K labeled captures) may behave differently.
- Format-confound caveats from M5 apply unchanged.
