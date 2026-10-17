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
| M5 | Baseline room classifier | ⬜ Next | |
| M6 | Window-length study + still/moving breakdown | ⬜ | |
| M7 | Predictions export + router API (for Task 3) | ⬜ | |
| M8 | Stretch models + ablations | ⬜ | |
| M9 | Report section for Task 1 | ⬜ | |

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

## M5: Baseline room classifier ⬜ (next)
**Done when:** validation accuracy, per-room accuracy, balanced accuracy, capture-level accuracy and a confusion matrix are produced by one command.

- [ ] `src/xroom/features.py`: mean spectrum per subcarrier over time (+ optional std), null subcarriers removed
- [ ] `src/xroom/normalize.py`: options, all fitted on **dev only**
  - [ ] none / log amplitude
  - [ ] global standardization (scaler fitted on dev windows of both rooms)
  - [ ] per-window normalization (divide by the window's own mean → removes scale)
- [ ] Baseline model: logistic regression (scikit-learn)
- [ ] Model selection on `dev_fold = holdout` only (never on validation)
- [ ] Final fit on all dev → score once on validation
- [ ] `src/xroom/eval.py`: accuracy, balanced accuracy, per-room recall, confusion matrix, capture-level majority vote
- [ ] `scripts/task1_train_eval.py --config configs/task1.yaml` (single entry point)
- [ ] **Leak check:** train the same classifier on the audit statistics only → "format-only" accuracy, reported next to the main result
- [ ] **Sanity check:** shuffled labels → accuracy should drop to ~50%
- [ ] Save metrics to `outputs/task1_metrics.json`

## M6: Window-length study + still/moving breakdown ⬜
**Done when:** the "room accuracy vs window length" figure exists (required by the report).

- [ ] Build caches for lengths {10, 25, 50, 100, 150} (`build_windows.py --length L`)
  - [ ] Decide how Room A windows of length L are made (resample the full capture to L rows, or take an L-row slice of the 150-row resampled capture). Write the choice down.
- [ ] Same training/eval protocol at every length; repeat over ≥ 3 seeds
- [ ] Plot accuracy (± spread) vs window length → `outputs/figures/room_acc_vs_length.png`
- [ ] Accuracy split by still activities (empty, sit, stand, lying) vs moving ones (walk, sitdown, standup, fall)

## M7: Predictions export + router API ⬜
**Done when:** Task 3 can call the room classifier and read saved predictions.

- [ ] `RoomClassifier` class with `fit`, `predict(windows) -> {A,B}`, `predict_proba`, `save`, `load`
- [ ] Export `outputs/task1_predictions.csv`: `capture_id, start, true_room, pred_room, prob_B`
- [ ] Save the trained model to `outputs/models/room_classifier.joblib`

## M8: Stretch models + ablations ⬜
Only after M5–M7 work end to end.

- [ ] Richer statistics (std, percentiles per subcarrier) + gradient boosting / MLP
- [ ] Small 1-D CNN on the full 150 × 256 window (watch out: it can exploit time-smoothness differences)
- [ ] Ablation table: feature set × normalization × model → accuracy and format-only accuracy
- [ ] Optional: frequency smoothing applied to both rooms; round Room A to integers; check whether the format gaps shrink
- [ ] Optional: "contiguous block" Room B split as a robustness check (consecutive chunks problem)

## M9: Report section for Task 1 ⬜
- [ ] Preprocessing pipeline diagram
- [ ] Why each preprocessing choice (copy from `reports/preprocessing_pipeline.md` §2)
- [ ] How validation and dev splits were built
- [ ] Task 1 room accuracy (+ balanced, per-room, capture-level)
- [ ] Figure: room accuracy vs window length
- [ ] Format-leak audit table + format-only accuracy (honest framing of the result)
- [ ] Ablation: what helps and what does not
- [ ] Limitations: format/session confound, Room B consecutive chunks, 7 dropped sit captures
- [ ] Hand-off note for Task 3: oracle vs predicted room (figure produced in Task 3)

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

## Open questions
- [ ] Room B sampling rate is unknown (no timestamps). Does it matter for comparing window lengths across rooms?
- [ ] Should Room B validation stay a random draw (rule) while we also report a contiguous-block check?
- [ ] Which Task 1 number goes in the headline: the main model's accuracy, or that number next to the format-only accuracy?

## How to run
```bash
python scripts/build_windows.py      # decode + window both rooms (~40 s)
python scripts/audit_format.py       # format-leak audit
python -m pytest -q                  # tests
```
