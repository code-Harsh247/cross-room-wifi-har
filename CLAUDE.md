# CLAUDE.md

CS60055 Hackathon 2: cross-room few-shot WiFi activity recognition from CSI, in two rooms.
- Room A: Experiment 2, `.pcap` files.
- Room B: Experiment 3, `.csv` files.

Three tasks:
1. Room identification (**done**)
2. Self-supervised encoder + linear probe (**in progress**)
3. Few-shot activity transfer across rooms

## Where to start
- **Task 2:** read `docs/PRD_Task2_Self_Supervised_Encoder.md`, then follow `docs/tasks_task2.md` **one milestone at a time**.
  - A milestone is done only when its 🔒 gate passes.
  - Record gate numbers in its Results log.
- **Context from Task 1:** `reports/task1_report_section.md`, especially §3 (pipeline), §4 (splits), §6 (format confound) and §9 (router).
- **Task 1 tracker and decisions:** `docs/task.md`.

## Layout
- `src/xroom/`: library code.
  - Decoding: `io_pcap`, `io_csv`, `clean`, `resample`, `windows`
  - Data: `catalog`, `splits`, `dataset`
  - Task 1: `features`, `room_classifier`, `eval`, `router`
- `scripts/`: one entry point per stage. `tests/`: pytest. `configs/task1.yaml`: all settings.
- `reports/`: write-ups. `reports/figures/`: committed figures.
- `outputs/` and `data_cache/` are git-ignored, so copy final figures to `reports/figures/`.
- Raw data lives at the repo root in `csidata/` (A, with `validation/`) and `csvdata/` (B). It is git-ignored and never committed.

## Key commands
```bash
python scripts/build_windows.py --rebuild   # decode raw data -> data_cache/ (~30 s); expect 4680 windows, 7 excluded captures
python -m pytest -q                         # all tests must pass before every commit
python scripts/task1_train_eval.py          # Task 1 sanity check: balanced accuracy 1.0000
```

Data access in code:
```python
from xroom.config import load_config
from xroom.dataset import load_windows, load_sequences, windows_from_sequences
cfg = load_config()
x, meta = load_windows(cfg)        # (4680, 150, 256) float16; meta: capture_id, room, activity, split, dev_fold, start
table, seqs = load_sequences(cfg)  # full per-capture sequences (A: 150 rows, B: 500 rows)
```

## Hard rules (from the brief; never break them)
- **Validation is for scoring only** (`meta.split == "val"`). Never use it for training (including SSL pre-training), normalization statistics, hyperparameter or checkpoint selection, or few-shot support.
- **Tune on development data only**, via `dev_fold` (`train` → `holdout`).
- **Split by capture.** All windows of a capture stay on one side. K counts captures, not windows.
- **Features come only from the CSI signal.** Never use filenames, folders, capture length or recording time.
- **Task 3 labels:** source-room labels plus the K target support captures only. That includes model or encoder *selection*: anything Task 3 reuses must be chosen without target activity labels (see `reports/task1_router_variants.md` §5).
- **Python only.** Do not use the dataset authors' MATLAB code or notebook, or ports of it. Do not use other teams' code.

## Known pitfalls
- **Format confound.** Rooms A and B differ in scale, noise and smoothness for non-physical reasons, so a model can tell them apart from format alone.
  - Features describing variation over time (std, percentiles, full time series) tend to read Room B's row noise.
  - Use per-row normalization and check with the smoothing "shortcut probe" (`harmonize` in `scripts/task1_ablations.py`).
- **Room B consecutive chunks.** Consecutive Room B files are chunks of one recording. Report a contiguous-split check (`contiguous_split` in `scripts/task1_ablations.py`) when results might be inflated.
- **Activities differ per room.** Only sit, stand and walk exist in both rooms.

## Conventions
- **Code:** match `src/xroom/` (type hints, short docstrings, small pure functions). New settings go in a YAML config, not hard-coded. Seed all randomness.
- **Testing:** every new module gets tests in `tests/`.
- **Reporting:** report uncertainty (capture-level bootstrap, or mean ± std over seeds). State limitations honestly.
- **Commits:**
  - small and frequent, one per passed gate or milestone (the history is graded);
  - imperative subject line;
  - **no AI attribution or `Co-Authored-By` lines** (`.claude/settings.json` disables them).
- **Never commit** `data_cache/`, `outputs/`, checkpoints or raw data.
- **Windows + Git Bash:** backslashes in heredoc-generated Python can get mangled. Prefer writing files with the editor tools over shell heredocs.
