# PRD — Task 1: "Where Am I?" (Room Identification from WiFi CSI)

**Course:** CS60055 Ubiquitous Computing — Hackathon Challenge 2
**Project:** Cross-Room Few-Shot Adaptation for WiFi Activity Recognition
**Scope of this document:** Task 1 only (Tasks 2 and 3 are out of scope, except where Task 1 must leave hooks for them)
**Status:** Draft v1
**Deadline (whole project):** October 31, 2026 · **Presentation:** November 2/3, 2026

---

## 1. Summary

Given a single CSI window of shape **150 × 256** (150 time steps × 256 subcarriers), predict whether it was recorded in **Room A** (Experiment 2) or **Room B** (Experiment 3). Score: **accuracy** on held-out validation captures.

Task 1 is also the foundation of the whole project. It forces us to build the complete data pipeline (decode → amplitude → clean → common format → window → normalize) that Tasks 2 and 3 reuse. In Task 3 the same room classifier is used as a **router** that sends each window to the right room-specific logic, and its predictions are compared against the true room ("oracle vs predicted room").

## 2. Goals and non-goals

### Goals
1. A working, reproducible, end-to-end pipeline from raw files (Room A `.pcap`, Room B `.csv`) to `150 × 256` windows.
2. A room classifier with high accuracy on held-out validation captures, with a **leak-free** evaluation protocol.
3. An analysis of **room accuracy vs window length** (required figure).
4. A classifier interface reusable as the Task 3 router, plus the data needed for the "oracle vs predicted room" comparison.
5. Every report item for Task 1 is producible directly from the repo (see §10).

### Non-goals
- Activity recognition (Tasks 2 and 3).
- Self-supervised encoder training.
- Beating state of the art. The slides say: *a pipeline that runs end to end beats a clever model that doesn't.*

## 3. Background and the key difficulty

Walls and furniture shape the WiFi channel, so the **average amplitude per subcarrier** acts like a fingerprint of the room. That makes room identification the easy-looking task, and the slides hint at starting simple (use still windows, a simple classifier).

The real risks are not the model. They are:

| Risk | Why it matters |
|---|---|
| **Format leakage** | Room A is decoded from raw complex CSI in `.pcap`; Room B is pre-computed amplitude `.csv`. If the two pipelines produce data with different scale, null subcarriers, dtype, or sampling artifacts, a classifier can "identify the room" by identifying the *file format*, which would score ~100% and be meaningless. |
| **Capture-level leakage** | Windows from the same capture are highly correlated. Splitting by window instead of by capture inflates accuracy. |
| **Session confound** | Each room was recorded in one session with one setup, so room identity is partly confounded with recording session and time. We cannot fully remove this, but we must not make it worse and should state it as a limitation. |
| **Banned shortcuts** | Filenames, folder names, and capture length are explicitly not allowed as features. |
| **Normalization destroys the fingerprint** | Per-room / per-capture normalization (useful in Task 3, e.g. AdaBN) can remove exactly the per-subcarrier mean that identifies the room. Task 1 normalization must be chosen with this in mind. |

## 4. Users and consumers

- **The team** (2–3 people): needs a clear module boundary so work on Task 1 does not block Tasks 2/3.
- **Task 3 pipeline:** consumes the room classifier as a router and the shared windowing/normalization code.
- **Graders:** read the report, inspect the GitHub history, and check that rules (leakage, banned resources) were followed.

## 5. Data

### 5.1 Inputs

| | Room A (Exp 2) | Room B (Exp 3) |
|---|---|---|
| Size | 3.5 × 4.5 m | 4.5 × 5 m |
| Format | raw `.pcap` (complex CSI must be decoded) | amplitude `.csv` |
| Classes | sit, stand, walk, sitdown, standup, fall | empty, lying, sit, stand, walk |
| Path | `Experiment-2/realdata/input_data/30ms/csidata/<class>/*.pcap` | `Experiment-3/Data/{Empty_Amp,Lying_Amp,Sitting_Amp,Standing_Amp,Walking}_*.csv` |
| Validation | supplied `validation/` folder, 25 per class | **we create it**: 25 captures per class, **seed 42** |

Hardware is identical (Asus RT-AC86U, BCM4366c0, Nexmon, 5 GHz, 80 MHz, 256 CSI subcarriers).

### 5.2 Label for Task 1
Binary: `A` or `B`, determined by *which archive the capture came from*. This label is used only as the training target and evaluation reference, never as an input feature.

### 5.3 Splits
- **Unit of splitting = capture.** All windows from one capture stay in the same split.
- **Validation:** Room A uses the supplied folder; Room B uses our own 25-per-class sample (seed 42), written once to a versioned manifest file (`splits/room_b_validation.json`) and never regenerated.
- **Development set** = everything not in validation. We may further split development into train / dev-validation by capture for model selection (stratified by room and class, fixed seed).
- **Validation captures are for scoring only.** No training, no normalization statistics, no hyperparameter selection, no support-set selection from them.

## 6. Functional requirements

### FR-1 Data inspection
- Script/notebook that lists, for each room: number of captures per class, capture lengths (packets), sampling characteristics, value ranges, and the CSI column layout.
- Output saved to `reports/data_inspection.md` (informs preprocessing choices and the report).

### FR-2 Room A decoding (PCAP → complex CSI)
- Parse Nexmon CSI payloads from `.pcap` in Python (any PCAP parser allowed; **no** port of the authors' MATLAB code or notebook).
- Recover complex CSI per packet for 256 subcarriers, handling the Nexmon/BCM4366c0 format documented in the Nexmon CSI docs (allowed resource).
- Unit tests on a small fixture: packet count, shape `(n_packets, 256)`, dtype, no NaN.

### FR-3 Amplitude conversion
- Room A: `|CSI|` from the complex values.
- Room B: read amplitude directly from CSV; verify it matches Room A's column order and semantics (document any reorder).

### FR-4 Cleaning
- Detect and handle invalid packets (all-zero or missing CSI), NaN/inf, and obviously corrupted rows.
- Handle structurally null subcarriers (guard bands / DC). Decision (keep as-is vs. mask) must be applied **identically** to both rooms and justified in the report. Removing a subcarrier only in one room is a leakage source.
- Optional filtering (e.g. outlier clipping, smoothing) allowed, applied identically to both rooms.

### FR-5 Common format
- Both rooms become float32 arrays of shape `(n_packets, 256)` with the same column ordering, units/scale handling, and dtype.
- **Format-leak audit (mandatory):** compare simple statistics (global min/max/mean, per-subcarrier zero rate, dtype, quantization/value granularity) between rooms. If a trivial statistic separates rooms almost perfectly *for non-physical reasons*, fix the pipeline or document it.

### FR-6 Windowing
- Deterministic windows of **150 × 256** (150 consecutive packets).
- Fixed, documented stride (default: non-overlapping for validation; overlap allowed for training windows only, never crossing capture boundaries).
- Captures shorter than 150 packets: documented policy (drop or pad), applied identically to both rooms.
- Each window keeps metadata: `capture_id`, `room`, `activity` (kept for later tasks, **not** a Task 1 feature), `start_idx`.

### FR-7 Normalization (leakage-safe)
- All statistics fitted **on development data only**.
- Task 1 must evaluate at least these options and report the effect:
  1. none (raw amplitude, optionally log or dB scaling),
  2. global (dataset-level) standardization fitted on development windows from both rooms,
  3. per-window normalization (e.g. divide by window mean / z-score over time only, preserving the *relative* per-subcarrier shape).
- Per-room statistics are **not** used for the main Task 1 model (they would be fitted knowing the room, which is the label). They may appear as an ablation clearly marked as an oracle.

### FR-8 Room classifier
- **Baseline (must ship first):** hand-crafted fingerprint = per-subcarrier mean amplitude over the window (and optionally std), fed to logistic regression or a small SVM/kNN.
- **Stretch models (only after baseline is done and logged):**
  - gradient-boosted trees or MLP on richer statistics (mean, std, percentiles per subcarrier),
  - small 1-D CNN on the full `150 × 256` window.
- Inputs allowed: only what is in the CSI signal. **Not allowed:** filenames, folder names, capture length, capture index/order.
- Classifier exposes `predict(window) -> {A,B}` and `predict_proba(window)` so Task 3 can use it as a router.

### FR-9 Evaluation
- **Primary metric:** accuracy over all validation windows (A and B pooled).
- Also report: per-room accuracy / recall, confusion matrix, balanced accuracy (windows per room differ), and capture-level accuracy (majority vote over windows of one capture).
- **Window-length study (required figure):** room accuracy vs window length. Evaluate at e.g. {10, 25, 50, 100, 150, (and longer if captures allow)} packets. For each length, retrain or re-evaluate under the same protocol and plot accuracy (with variability across seeds).
- **Still vs. moving windows:** report accuracy separately for windows from still activities (empty, sit, stand, lying) vs. dynamic ones (walk, sitdown, standup, fall) to test the "still windows give the cleanest fingerprint" hint.
- **Oracle vs predicted room hook:** write per-window predictions (`capture_id`, `start_idx`, true room, predicted room, probability) to `outputs/task1_predictions.csv`, so Task 3 can run both with oracle routing and with predicted routing.

### FR-10 Reproducibility
- Single entry point, e.g. `python -m task1.run --config configs/task1.yaml`, regenerates: windows, splits, model, metrics, figures.
- All randomness seeded (global seed + the fixed Room B validation seed 42).
- Config file records every preprocessing choice.

## 7. Non-functional requirements

| Area | Requirement |
|---|---|
| Language/libs | Python only; NumPy, SciPy, scikit-learn, PyTorch, a PCAP parser are allowed |
| Compliance | No authors' MATLAB code/notebook or ports; no other team's code; no validation data in training/normalization |
| Performance | Full preprocessing + baseline training should run on a laptop in minutes; cache decoded arrays to disk (`.npy`/`.npz`) so PCAP decoding runs once |
| Code quality | Modular (`io/`, `preprocess/`, `models/`, `eval/`), type hints, docstrings on public functions, tests for decoding/windowing/splitting |
| Git | Small, frequent commits with meaningful messages (history counts toward grading) |
| Integrity | AI-use and per-member contribution disclosure kept up to date from day one |

## 8. Proposed repo layout (Task 1 portion)

```
repo/
├── docs/
│   └── PRD_Task1_Room_Identification.md
├── configs/task1.yaml
├── data/                      # git-ignored raw + cache
├── splits/
│   └── room_b_validation.json # fixed, committed
├── src/
│   ├── io/                    # pcap_reader.py, csv_reader.py
│   ├── preprocess/            # amplitude.py, clean.py, window.py, normalize.py
│   ├── models/                # fingerprint_baseline.py, cnn.py (stretch)
│   └── eval/                  # metrics.py, plots.py
├── tests/
├── outputs/                   # predictions, figures (git-ignored or small)
└── reports/data_inspection.md
```

## 9. Milestones (working backwards from Oct 31)

| # | Milestone | Done when |
|---|---|---|
| M1 | Data inspection | `reports/data_inspection.md` exists; both archives load |
| M2 | Room A decoding | PCAP → `(n,256)` complex → amplitude, with tests |
| M3 | Common format + format-leak audit | Both rooms in identical format; audit table written |
| M4 | Splits + windows | Room B validation manifest committed; deterministic `150×256` windows with metadata |
| M5 | Baseline classifier | Mean-spectrum + logistic regression; accuracy, confusion matrix on validation |
| M6 | Window-length study + still/moving breakdown | Required figure produced |
| M7 | Predictions export + router API | `task1_predictions.csv` consumable by Task 3 |
| M8 | Stretch models + ablations (normalization options) | Documented table of what helps / what does not |
| M9 | Report section for Task 1 | All Task 1 report items satisfied |

Rule of thumb: finish M1–M5 end to end before touching M8.

## 10. Mapping to the report deliverables

| Report item (from the slides) | Task 1 contribution |
|---|---|
| Preprocessing pipeline diagram | Diagram of FR-2 → FR-7, with the choices made |
| Why each preprocessing choice | Rationale per step, incl. null-subcarrier decision, normalization choice |
| How validation sets and dev splits were built | §5.3 and the committed split manifest |
| Task 1 room accuracy | FR-9 primary metric |
| Room accuracy vs window length (figure) | FR-9 window-length study |
| Oracle vs predicted room (figure) | Built from `task1_predictions.csv` in Task 3 |
| Ablation: what helps / what does not | Normalization options, feature sets, model types |
| AI-use and contribution disclosure | Maintained in repo root |

## 11. Success criteria

- **Must:** end-to-end run produces room accuracy on held-out, capture-split validation data with no rule violations.
- **Must:** format-leak audit shows the classifier is not separating rooms via a pipeline artifact (or the artifact is fixed).
- **Should:** high accuracy (target ≥ 95% at window length 150; this is a goal, not a requirement, since the actual ceiling depends on the data).
- **Should:** clear degradation curve as window length shrinks, with an explanation.
- **Could:** stretch models show a measurable gain over the baseline, or an honest "no gain" ablation.

## 12. Open questions / assumptions to verify early

1. **Sampling interval and window duration.** The Room A path says `30ms`. If one packet ≈ 30 ms, a 150-packet window is ≈ 4.5 s. Verify for both rooms; if Room B's rate differs, window duration is not comparable even though shape is.
2. **Room B CSV layout:** are rows = packets and columns = 256 subcarriers, one file per capture? Is there a header/index column? Confirm during FR-1.
3. **What is "one capture" in Room B?** The slides call it a capture, but CSV files are named by class (`Empty_Amp_*.csv`). Confirm whether each file is one capture or a long recording that must be cut. This determines the 25-per-class validation sampling.
4. **Amplitude scale compatibility.** Room A amplitude comes from our own decoding; Room B's comes from the authors' processing. They may differ in scale (linear vs dB, AGC normalization). Resolve in FR-5.
5. **Null/guard subcarriers** in 80 MHz Nexmon CSI: which indices, and are they zero in both rooms?
6. **Class imbalance** between rooms: validation is 150 captures (A, 6 classes) vs. 125 (B, 5 classes). Pooled accuracy can be dominated by one room, so balanced accuracy is reported alongside.
7. **Session confound** cannot be eliminated with this data; state it as a limitation in the report.

## 13. Risks and mitigations

| Risk | Mitigation |
|---|---|
| PCAP decoding takes longer than expected | Start M2 first; cache results; time-box and ask for Nexmon-format help early |
| Suspiciously perfect accuracy (≈100%) | Run the format-leak audit; test with shuffled labels and with per-window statistics only; check feature importances |
| Breaking Task 3 by over-tuning normalization for Task 1 | Keep normalization pluggable via config; keep the Task 1 default global/per-window, with per-room as a separate documented option |
| Rule violation (e.g. validation used in normalization) | Fit-on-dev-only enforced in code: scaler fit function takes only development capture IDs and asserts no overlap with validation |
| Team blocked on one person | Split work: decoding (M2), splits + windows (M4), models + eval (M5–M6) |
