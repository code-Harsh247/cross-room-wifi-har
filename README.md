# Cross-Room Few-Shot WiFi Activity Recognition

CS60055 Hackathon Challenge 2: activity recognition from WiFi CSI (channel state information) recorded in two rooms. This README explains how to **reproduce every reported result** from a fresh clone.

| Task | Status | Report |
|---|---|---|
| 1. Room identification | Done | [reports/task1_report_section.md](reports/task1_report_section.md) |
| 2. Self-supervised encoder + linear probe | Done | [reports/task2_report_section.md](reports/task2_report_section.md) |
| 3. Few-shot activity transfer across rooms | In progress (plan: [PRD](docs/PRD_Task3_Few_Shot_Cross_Room.md), [tasks](docs/tasks_task3.md)) | — |

- **Room A:** Experiment 2, `.pcap` files in `csidata/` (its validation set is in `csidata/validation/`).
- **Room B:** Experiment 3, `.csv` files in `csvdata/` (validation captures are listed in `splits/room_b_validation.json`).

The raw dataset (~800 MB) is committed to the repo, so there is nothing to download.

---

## 1. Requirements

| | Tested with | Notes |
|---|---|---|
| OS | Linux (Ubuntu-based) | Windows + Git Bash also used for Task 1 |
| Python | 3.12 | `pyproject.toml` allows ≥ 3.10 |
| GPU | NVIDIA RTX 3050 Laptop, 4 GB | Task 2 only; a CPU works but is much slower |
| PyTorch | 2.14.1 + CUDA 12.6 | |
| Disk | ~6 GB free | CUDA PyTorch ~4 GB, data cache ~0.7 GB, outputs ~0.3 GB |
| RAM | 8 GB | |

## 2. Setup

```bash
git clone https://github.com/code-Harsh247/cross-room-wifi-har.git
cd cross-room-wifi-har

python3 -m venv .venv
source .venv/bin/activate            # Windows Git Bash: source .venv/Scripts/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu126   # CPU only: drop --index-url ...
pip install numpy scipy pandas pyyaml scikit-learn matplotlib joblib pytest umap-learn

python -c "import torch, umap; print(torch.__version__, torch.cuda.is_available())"
```

**Short on disk?** The large folders can live on another drive:
- Create the venv there (on exFAT/FAT drives, use `python3 -m venv --copies` and create its `lib64/` folder first).
- Replace `data_cache/` and `outputs/` in the repo with symlinks to folders on that drive. `.gitignore` already ignores the symlinks.
- If pip fails with "disk quota exceeded", point its temporary folder at the big drive: `TMPDIR=/path/on/drive pip install ...`.

## 3. Build the data cache (needed once)

```bash
python scripts/build_windows.py --rebuild   # ~30 s
python -m pytest -q
```

| Check | Expected |
|---|---|
| `build_windows.py` | prints `wrote 4680 windows` and `excluded captures: 7` |
| `pytest` | all tests pass (86 at the end of Task 2; 2 router tests are skipped until §4 builds the routers) |

## 4. Reproduce Task 1 (room identification)

```bash
python scripts/audit_format.py              # format-leak audit
python scripts/task1_train_eval.py          # main result + controls
python scripts/task1_window_length.py       # window-length study
python scripts/task1_router_variants.py     # room routers used by Task 3
python scripts/task1_ablations.py           # models x features, harmonization, contiguous split
```

**Expected:** `task1_train_eval.py` prints `all activities ... balanced 1.0000`. Every other number is in [reports/task1_report_section.md](reports/task1_report_section.md).

## 5. Reproduce Task 2 (self-supervised encoder)

Run these in order. Times are for the RTX 3050.

```bash
# 5.1 Sanity figure for the augmentations (optional)
python scripts/task2_augment_preview.py                     # outputs/task2/augment_preview.png

# 5.2 Pre-train the main encoder, 3 seeds (~4-6 min each)
for s in 0 1 2; do python scripts/task2_pretrain.py --run simclr_s$s --seed $s; done

# 5.3 Baselines and the shuffled-label control (~1.5 min each)
python scripts/task2_probe.py --baselines-only
python scripts/task2_probe.py --baselines-only --shuffle-labels

# 5.4 Probe the 3 seeds (~30 s)
python scripts/task2_probe.py --encoder simclr_s0 simclr_s1 simclr_s2 --summary simclr_default

# 5.5 Publish the label-clean default encoder for Task 3
python scripts/task2_publish_default.py                     # rewrites the committed models/task2_default/ (see note below)

# 5.6 Figures (copied to reports/figures/task2_*.png)
python scripts/task2_figures.py

# 5.7 Ablations + contiguous Room B check (~2 h; existing runs are reused if interrupted)
python scripts/task2_ablations.py
python scripts/task2_ablations.py --only tau_0.5 method_mae --seeds 2 --skip-contiguous
for v in tau_0.5 method_mae; do
  python scripts/task2_probe.py --encoder abl_${v}_s0 abl_${v}_s1 abl_${v}_s2 --summary $v
done
python scripts/task2_probe.py --encoder abl_aug_no_format_s0 abl_aug_no_format_s1 --summary aug_no_format

# 5.8 Collect every reported number and check the report against it
python scripts/task2_report_numbers.py                      # outputs/task2/report_numbers.md
python scripts/task2_check_report.py                        # expect "161 matched"
```

**Expected results** (validation macro-F1, mean ± std over 3 seeds):

| | Room A | Room B | Headline (mean of rooms) |
|---|---|---|---|
| SimCLR encoder (default) | 0.905 ± 0.027 | 0.992 ± 0.005 | 0.949 |
| Random-init encoder | 0.935 ± 0.011 | 0.856 ± 0.013 | 0.895 |
| Hand-crafted mean + std | 0.940 | 0.989 | 0.965 |

**How exact is reproduction?**
- **Probes are deterministic.** Given the same encoders, the probe scripts give byte-identical `metrics.json` files (one BLAS thread, cuDNN deterministic mode).
- **Pre-training is seeded but may not match exactly on another GPU, driver or PyTorch version.** Expect small differences (about ±0.01 macro-F1) in the SSL rows. If they appear, run 5.8: `task2_check_report.py` will list any report number that no longer matches.

**The default encoder ships with the repo.** `models/task2_default/` holds the exact encoder (`simclr_s0`, epoch 54) behind the Task 2 numbers, so Task 3 needs no training. Step 5.5 overwrites it. Run 5.5 only when you mean to republish, because a retrained encoder on other hardware will differ slightly.

**Using the encoder (Task 3):**
```python
from xroom.config import load_task2_config
from xroom.dataset import load_windows
from xroom.encoder import load_encoder, embed

cfg = load_task2_config()
x, meta = load_windows(cfg)
z = embed(load_encoder(cfg), x)      # (4680, 128) embeddings from the committed models/task2_default/
```

## 6. Saved results (no rerun needed)

`results/` is a committed copy of `outputs/` with every model checkpoint left out (2.4 MB). It holds the saved files behind every reported number:

| Path | Contents |
|---|---|
| `results/task1/` | Task 1 metrics, predictions, model selection, router files |
| `results/task2/probe/<name>/metrics.json`, `summary.json` | probe scores, confusion matrices, bootstrap intervals, controls, dev-holdout scores |
| `results/task2/ablations/` | `results.csv`, `summary.csv` (+ `_partial` for third seeds), `run.log` |
| `results/task2/<run>/` | `config.yaml`, `history.json` (loss and collapse metrics per epoch), `run.json`, `norm_stats.npz` per SSL run |
| `results/task2/report_numbers.md` | every number in the Task 2 report |

The scripts always write to `outputs/`. After rerunning them, refresh the snapshot with `python scripts/snapshot_results.py`.

## 7. Repository layout

```
csidata/, csvdata/      raw data (read-only, committed)
splits/                 fixed Room B validation manifest (never edit)
configs/                task1.yaml (data + Task 1), task2.yaml (layered on task1.yaml)
src/xroom/              library code (decoding, splits, features, encoder, SSL, probes)
scripts/                one entry point per stage (task1_*, task2_*)
tests/                  pytest suite
reports/                write-ups; reports/figures/ holds committed figures
docs/                   PRDs, gated task trackers, AI-use and contribution disclosure
models/task2_default/   committed Task 2 default encoder (used by Task 3)
results/                committed snapshot of outputs/ without checkpoints
data_cache/, outputs/   generated, git-ignored (rebuild with the scripts above)
```

## 8. Rules the results depend on

- **Validation is used only for scoring.** It is never used for training (including self-supervised pre-training), normalization statistics, tuning, checkpoint selection or few-shot support. Tuning uses `dev_fold` train → holdout.
- **Splits are by capture**, and the validation sets are fixed.
- **Features come only from the CSI signal.**
- **Every reported number comes from a committed script.**

Full rules: [CLAUDE.md](CLAUDE.md). AI-tool use and member contributions: [docs/ai_use_and_contributions.md](docs/ai_use_and_contributions.md).

## 9. Troubleshooting

| Symptom | Fix |
|---|---|
| `FileNotFoundError ... windows_L150.npy` | Run `python scripts/build_windows.py --rebuild`. |
| `no encoder at models/task2_default` | It is committed; run `git pull`. To rebuild it: steps 5.2 (seed 0) and 5.5. |
| `no encoder at outputs/task2/simclr_s0` (or another run) | Training runs are git-ignored; run step 5.2 for that seed. |
| 2 tests skipped: "routers not built" | Run `python scripts/task1_router_variants.py`. |
| `torch.cuda.is_available()` is `False` | Install the CUDA build of PyTorch (§2); everything still runs on CPU, slowly. |
| Probes take many minutes | Make sure you run the committed scripts; they pin BLAS to one thread, without which logistic regression is ~60× slower. |
| pip "disk quota exceeded" | Set `TMPDIR` to a folder with free space (§2). |
