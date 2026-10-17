# Task 1: "Where Am I?" Task Tracker

**Goal:** Given one CSI window (150 × 256), predict **Room A** or **Room B**. Metric: **accuracy** on held-out validation captures.
**Spec:** [PRD_Task1_Room_Identification.md](PRD_Task1_Room_Identification.md)
**Deadline:** Oct 31, 2026 (whole project) · **Presentation:** Nov 2/3, 2026
**Last updated:** 2026-10-18

Legend: `[x]` done · `[ ]` to do · `[~]` partly done · **Owner** = who is doing it (fill in)

---

## Status at a glance

| # | Milestone | Status | Owner |
|---|---|---|---|
| M1 | Data inspection | ✅ Done | |
| M2 | Room A decoding | ✅ Done | |
| M3 | Common format + format-leak audit | ✅ Done | |
| M4 | Splits + windows | ✅ Done | |
| M5 | Baseline room classifier | ✅ Done | |
| M6 | Window-length study + still/moving breakdown | ✅ Done | |
| M7 | Predictions export + router API (for Task 3) | ✅ Done | |
| M8 | Stretch models + ablations | ✅ Done | |
| M9 | Report section for Task 1 | ✅ Done | |

---

## M1: Data inspection ✅
- [x] List captures per room/class, packet counts, durations
- [x] Identify PCAP structure (Ethernet/IP/UDP → 18 B Nexmon header + 1024 B CSI)
- [x] Identify Room B CSV structure (500 × 256 integers, one header row, no timestamps)
- [x] Find duplicates and junk files (`stand65(1).pcap`, `.DS_Store`, `model_plot_1.png`)
- [x] Discover mixed 20 MHz / 80 MHz frames in Room A
- [x] Discover that Room B files are consecutive chunks of longer recordings
- [x] Write [reports/data_inspection.md](../reports/data_inspection.md)

## M2: Room A decoding ✅
- [x] Pure-Python pcap reader (`src/xroom/io_pcap.py`)
- [x] Nexmon packed-float CSI decoder (from the Nexmon docs; no authors' MATLAB)
- [x] Sanity check: smooth spectra, chanspec = ch 42 / 80 MHz
- [x] Unit tests for decoder (autoscale, signs, exponents, real file)

## M3: Common format + format-leak audit ✅
- [x] Keep only full-band 80 MHz packets (`src/xroom/clean.py`)
- [x] Drop captures with < 20 full-band packets (7 sit captures)
- [x] Zero DC/guard subcarriers in both rooms (0, 1, 255, 123–133)
- [x] Room B loader with shape/NaN checks (`src/xroom/io_csv.py`)
- [x] Format-leak audit (`scripts/audit_format.py`) → every trivial statistic separates the rooms (AUC ≈ 1.0)
- [x] Write [reports/preprocessing_pipeline.md](../reports/preprocessing_pipeline.md)

## M4: Splits + windows ✅
- [x] Room A validation = supplied folder
- [x] Room B validation = 25/class, seed 42, frozen in `splits/room_b_validation.json`
- [x] Dev train/holdout folds (80/20 by capture, stratified, seed 0)
- [x] Room A: resample full-band packets in time → 1 window per capture
- [x] Room B: non-overlapping 150-row windows → 3 per capture
- [x] Cache: `data_cache/windows_L150.npy` + meta CSV (`scripts/build_windows.py`)
- [x] Tests: windows never cross captures; splits disjoint and deterministic

---

## M5: Baseline room classifier ✅
**Results:** [reports/task1_baseline_results.md](../reports/task1_baseline_results.md). Validation balanced accuracy **1.000** (all activities, shared activities, capture level), but the **format-only control is also 1.000**.

- [x] `src/xroom/features.py`: time-averaged spectrum, variants `none` / `row` (per-row normalized, main) / `log`; format statistics
- [x] Normalization: per-row gain removal inside the features + StandardScaler fitted on **dev only** (in the model pipeline)
- [x] Logistic regression, class-balanced (`src/xroom/room_classifier.py`)
- [x] Feature variant fixed by design (`row`); `C` selected on `dev_fold = holdout` by log-loss → C = 1000
- [x] Final fit on all dev → scored once on validation
- [x] `src/xroom/eval.py`: accuracy, balanced accuracy, per-room recall, log-loss, confusion matrix, capture level, per room/activity, shared activities
- [x] `scripts/task1_train_eval.py` (single entry point)
- [x] **Format-only control** → 1.000 (rooms separable by format alone)
- [x] **Shuffled labels** → 0.44 ± 0.10 (chance, no pipeline leak)
- [x] **Leave-activity-out**: sit+stand→walk 0.86 · shared→room-only 0.93 · room-only→shared 0.99
- [x] Ablation over feature variants (none 1.000 / row 1.000 / log 0.993)
- [x] Robustness checks: train = val = 1.000 · learning curve (1 capture/room 0.58 → 50 captures 0.97) · nearest centroid 0.88 · no edge-subcarrier reliance (1.000 without them) · leave-one-day-out Room A recall 0.991
- [x] Outputs in `outputs/task1/L150/` (metrics.json, model_selection.csv, predictions_val.csv, confusion_matrix_val.png)
- [x] Tests (`tests/test_room_classifier.py`)

## M6: Window-length study + still/moving breakdown ✅
**Results:** [reports/task1_window_length.md](../reports/task1_window_length.md). 5 rows → 0.988, 25 rows → 0.998, ≥ 100 rows → 1.000. Format-only 1.000 at every length. Transfer to unseen walking *drops* with length (0.95 → 0.86).

- [x] Lengths {5, 10, 25, 50, 100, 150} rows, built in memory from cached capture sequences (`data_cache/sequences.joblib`)
  - [x] Room A windows = contiguous L-row slices of the 150-row resampled capture (option B), so shorter = less time in both rooms
- [x] Same protocol at every length (C re-tuned on dev holdout). Spread = 95% capture-level bootstrap interval (the model is deterministic, so seeds would add nothing)
- [x] Figure with all / shared / format-only / sit+stand→walk curves + still vs moving panel → `reports/figures/task1_room_acc_vs_length.png`
- [x] Still vs moving: no practical difference (moving slightly easier at 5 rows)
- [x] Extra check: sit+stand→walk drop is caused by both window length and weak regularization (fixed-C rerun)
- [x] Tests for slicing and bootstrap

## M7: Predictions export + router API ✅
**Done when:** Task 3 can call the room classifier and read saved predictions. **Report:** [reports/task1_router_variants.md](../reports/task1_router_variants.md)

- [x] `RoomClassifier` class with `fit`, `predict(windows) -> {A,B}`, `predict_proba`, `save`, `load` (computes its own features from raw windows)
- [x] Export `outputs/task1/L150/predictions_val.csv`: `capture_id, start, activity, true_room, pred_room, prob_B`
- [x] Save the trained model to `outputs/task1/L150/room_classifier.joblib`
- [x] Rules question settled from the slides: the router uses **room labels only** on dev captures of both rooms (allowed). Choosing its settings with dev **activity** labels of the target room is not allowed in Task 3, so the default router's settings are chosen with room labels only (strict)
- [x] Router variants (from M6): stronger regularization and slice averaging
  - Activity-tuned on dev: 25-row slices, C = 1. Val transfer mean 0.926 → 0.948 (10–15-row slices lift unseen walking 0.86 → 0.987, tied on dev)
  - Strict (room labels only): 50-row slices, C = 1000. Val transfer 0.930 (≈ baseline)
  - Both route the real validation windows at 1.000, because the router has seen every activity in training
- [x] `RoomClassifier(slice_rows=...)`: slice averaging built in
- [x] Routers saved: `outputs/task1/router/room_router.joblib` (strict, **Task 3 default**) and `room_router_activity_tuned.joblib` (ablation), with validation predictions
- [x] `src/xroom/router.py`: `load_router(cfg, "strict" | "activity_tuned")` and `route(windows, meta, mode="oracle" | "predicted", router=...)` for Task 3's oracle vs predicted comparison (tests in `tests/test_router.py`)

## M8: Stretch models + ablations ✅
**Report:** [reports/task1_ablations.md](../reports/task1_ablations.md) · `python scripts/task1_ablations.py`

- [x] Richer statistics (mean+std, percentiles per subcarrier) + gradient boosting (MLP skipped: boosting and the CNN already cover non-linear models)
- [x] Small 1-D CNN on the full window (GPU, 10 epochs)
- [x] Ablation table: features × model → standard, unseen-activity transfer, and a **shortcut probe** (score on smoothed validation windows). Std/percentile features collapse under the probe (recall B 0.005–0.17); percentiles "win" transfer (1.000) only by reading Room B's row noise. The mean-spectrum baseline is the most robust (0.969) → **M5 choice stays**
- [x] Harmonization: smoothing both rooms does not close the format gap (format-only control still 1.000); room accuracy unaffected. Rounding Room A to integers not done (the integer cue is an artifact the baseline's features ignore)
- [x] Contiguous Room B split: standard 0.999 vs 1.000, so consecutive chunks do not inflate Task 1

## M9: Report section for Task 1 ✅
**Section:** [reports/task1_report_section.md](../reports/task1_report_section.md)

- [x] Preprocessing pipeline diagram (Mermaid; renders on GitHub)
- [x] Why each preprocessing choice
- [x] How validation and dev splits were built
- [x] Task 1 room accuracy (+ balanced, per-room, capture-level) and confusion matrix
- [x] Figure: room accuracy vs window length
- [x] Format-leak audit table + format-only accuracy (100% presented as an upper bound)
- [x] Controls and robustness (shuffled labels, learning curve, nearest centroid, edge subcarriers, leave-one-day-out, leave-activity-out)
- [x] Ablation: what helps and what does not (normalization; models × features with the shortcut probe)
- [x] Limitations: format/session confound, Room B consecutive chunks, unknown Room B rate, 7 dropped sit captures
- [x] Hand-off note for Task 3: strict router + oracle vs predicted routing API (the figure itself is produced in Task 3)
- [ ] Project-level, still open: AI-use and per-member contribution disclosure (needs team input)

---

## Rules checklist (check before every result we report)
- [ ] Validation captures never used for training, normalization stats, model selection or support
- [ ] Splits are by capture; all windows of a capture are on the same side
- [ ] No features from filenames, folders or capture length
- [ ] No authors' MATLAB code/notebook or ports; no other team's code
- [ ] Python only
- [ ] Commit as we go (history counts)
- [ ] AI-use and contribution disclosure kept up to date

## Decisions log
| Date | Decision | Why |
|---|---|---|
| 2026-10-18 | Keep only 80 MHz packets in Room A | 20 MHz frames have no CSI on 192/256 subcarriers; Room B has almost none |
| 2026-10-18 | Resample each Room A capture to 150 rows | ~59 valid packets per capture, never 150 |
| 2026-10-18 | Room B: non-overlapping 150-row windows | Deterministic, no capture crossing |
| 2026-10-18 | Zero DC/guard subcarriers in both rooms | No channel info, only format differences |
| 2026-10-18 | Drop captures with < 20 full-band packets | Too few frames to represent the capture |
| 2026-10-18 | Private GitHub repo | Rules ban sharing code between teams |
| 2026-10-18 | Room classifier feature variant fixed to per-row normalized spectrum; only C tuned | Removes absolute scale (a format cue) by design |
| 2026-10-18 | Shorter windows = contiguous slices of each capture's sequence | Shorter must mean less observation time in both rooms |
| 2026-10-18 | Task 3 default router = strict (50-row slices, C = 1000, settings chosen with room labels only); activity-tuned router (25 rows, C = 1) kept as ablation | Task 3 forbids target activity labels beyond K, even for tuning; both route validation at 1.000 |

## Open questions
- [x] Room B sampling rate is unknown (no timestamps). Does it matter for comparing window lengths across rooms? → Lengths are reported in rows; seconds only for Room A (~26 ms/row). Stated as a caveat in the M6 report.
- [x] Should Room B validation stay a random draw (rule) while we also report a contiguous-block check? → Yes: the official split stays; the contiguous check (M8) gives the same Task 1 result
- [x] Which Task 1 number goes in the headline? → 100% balanced accuracy, always stated next to the format-only control (also 100%) and framed as an upper bound (report section §1)

## How to run
```bash
python scripts/build_windows.py      # decode + window both rooms (~40 s)
python scripts/audit_format.py       # format-leak audit
python -m pytest -q                  # tests
```
