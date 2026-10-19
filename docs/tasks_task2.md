# Task 2: "Learn Without Labels" Task Tracker

**Goal:** Train a CSI encoder **without activity labels**, freeze it, and evaluate it with a **linear probe** (macro-F1 on validation). Provide a frozen-encoder API for Task 3.
**Spec:** [PRD_Task2_Self_Supervised_Encoder.md](PRD_Task2_Self_Supervised_Encoder.md) (read it first)
**Task 1 tracker (done):** [task.md](task.md) · **Task 1 summary:** [../reports/task1_report_section.md](../reports/task1_report_section.md)
**Owner:** Abhinav · **Deadline:** Oct 31, 2026 · **Presentation:** Nov 2/3, 2026

Legend: `[x]` done · `[ ]` to do · 🔒 **Gate**: every check must pass before starting the next milestone. If a gate fails, fix it first, and write down what was wrong in the Decisions log.

---

## How to work through this file (with Claude Code)

1. Work on **one milestone at a time**, in order. Example prompt:
   > Read docs/PRD_Task2_Self_Supervised_Encoder.md and docs/tasks_task2.md. Do milestone T1 only, then run its gate and show me the output.
2. A milestone is done only when **its gate passes**. Tick the boxes and paste the key gate numbers into the **Results log** at the bottom.
3. **Commit after each passed gate.** Use a short message such as `Add SSL crop sampler (T1)`, with no AI attribution lines (`.claude/settings.json` already disables them).
4. **Never touch validation data** for training, normalization, tuning or checkpoint selection. Validation is scored once, by the final scripts.

---

## Status at a glance

| # | Milestone | Status |
|---|---|---|
| T0 | Hand-over setup | ✅ |
| T1 | Config + SSL data pipeline | ✅ |
| T2 | Augmentations | ✅ |
| T3 | Encoder + heads | ✅ |
| T4 | SSL objectives (NT-Xent, MAE) | ✅ |
| T5 | Training loop + checkpoints | ⬜ |
| T6 | Probe pipeline + baselines | ⬜ |
| T7 | Main SSL result (≥ 3 seeds) | ⬜ |
| T8 | Figures (UMAP, curves, bars) | ⬜ |
| T9 | Controls + ablations | ⬜ |
| T10 | Task 3 hand-off API | ⬜ |
| T11 | Report section | ⬜ |

---

## T0: Hand-over setup
- [x] Clone `code-Harsh247/cross-room-wifi-har`. Ask Harsh for collaborator access, since the repo is private.
- [x] The dataset comes with the clone: `csidata/` (Room A `.pcap`, with `validation/`) and `csvdata/` (Room B `.csv`), about 800 MB. Nothing to download; the first clone just takes a while. Treat it as read-only.
- [x] Python 3.13 with numpy, scipy, pandas, scikit-learn, PyTorch (CUDA if available), pyyaml, joblib and matplotlib.
- [x] Install UMAP: `pip install umap-learn`. Add `torch` and `umap-learn` to `pyproject.toml` dependencies.
- [x] Rebuild the Task 1 caches: `python scripts/build_windows.py --rebuild`
- [x] Skim `reports/task1_report_section.md` §3–4 (pipeline, splits) and PRD §3 (pitfalls).

🔒 **Gate T0**
| Check | Command | Pass if |
|---|---|---|
| Cache rebuilt | `python scripts/build_windows.py --rebuild` | prints `wrote 4680 windows` and `excluded captures: 7` |
| Tests green | `python -m pytest -q` | `29 passed` |
| Task 1 reproduces | `python scripts/task1_train_eval.py` | `all activities ... balanced 1.0000` |
| GPU + UMAP | `python -c "import torch, umap; print(torch.cuda.is_available())"` | no import error (`True` if a GPU is present) |

---

## T1: Config + SSL data pipeline (PRD FR-1)
- [x] `configs/task2.yaml` with sections `ssl_data` (crop range 96–150, batch size, room balancing, seed), `augment`, `encoder`, `ssl`, `probe`.
- [x] `src/xroom/ssl_data.py`:
  - [x] Crop sampler over **dev** capture sequences (`load_sequences`). Crops are contiguous and never cross captures.
  - [x] Room-balanced batches (≈ 50/50 A/B).
  - [x] Input normalization: per-row mean normalization, then optional log, then per-subcarrier standardization with **dev-only** statistics (saved for reuse). Keep the 242 data subcarriers.
  - [x] Hard assertion: no capture with `split == "val"` ever enters the SSL dataset.
- [x] `tests/test_ssl_data.py`

🔒 **Gate T1**
| Check | How | Pass if |
|---|---|---|
| No validation leakage | test: collect capture IDs from 200 batches, intersect with val IDs | empty intersection; the assertion fires on a deliberately injected val ID |
| Shapes | test | batch shape `(B, T, 242)` with 96 ≤ T ≤ 150; float32; finite |
| Room balance | test over 200 batches | share of Room A crops in [0.45, 0.55] |
| Dev-only stats | test | normalization stats computed from dev rows only (compare against a manual dev-only computation, tolerance 1e-5) |
| Determinism | test | same seed gives identical batches; a different seed gives different batches |
| Suite | `python -m pytest -q` | all pass |

---

## T2: Augmentations (PRD FR-2)
- [x] `src/xroom/augment.py`: temporal crop+resize, Gaussian noise, subcarrier block masking, time masking, per-row gain jitter, format-robustness (random time smoothing / added row noise). Each has a probability and a strength in `configs/task2.yaml`.
- [x] `make_views(x, rng) -> (view1, view2)` for SimCLR.
- [x] Preview figure: one window per room, original plus 4 augmented views, saved to `outputs/task2/augment_preview.png` (`scripts/task2_augment_preview.py`).
- [x] `tests/test_augment.py`

🔒 **Gate T2**
| Check | How | Pass if |
|---|---|---|
| Shape / dtype / finite | test, every augmentation | unchanged shape and dtype, no NaN/inf |
| Identity at p = 0 | test | output equals input exactly |
| Masking amount | test | masked fraction within the configured range |
| Determinism | test | same seed gives the same output |
| Visual sanity | open `augment_preview.png` | views of the same window look alike but not identical; nothing looks like pure noise. If they do, the augmentations are too strong |
| Suite | `python -m pytest -q` | all pass |

---

## T3: Encoder + heads (PRD FR-3)
- [x] `src/xroom/encoder.py`: 1-D CNN (242 input channels → 3–4 conv blocks → global average pool → linear to d = 128), projection head 128 → 128 → 64, and an MAE decoder head (only if T4's MAE is done).
- [x] `tests/test_encoder.py`

🔒 **Gate T3**
| Check | How | Pass if |
|---|---|---|
| Variable length | test with T = 32, 96, 150 | output `(B, 128)` for each |
| Gradients | test: one backward pass | every parameter gets a finite, non-all-zero gradient |
| Size | print parameter count | < 1.5 M parameters (fits a 4 GB GPU at batch 256) |
| Device | test (skip if no CUDA) | forward pass on GPU matches CPU within 1e-4 |
| Eval determinism | test | two forward passes in `eval()` are identical |
| Suite | `python -m pytest -q` | all pass |

---

## T4: SSL objectives (PRD FR-4)
- [x] `src/xroom/ssl.py`: NT-Xent loss (temperature τ); MAE-style masked reconstruction loss (MSE on masked rows only).
- [x] `tests/test_ssl.py`

🔒 **Gate T4**
| Check | How | Pass if |
|---|---|---|
| NT-Xent orders cases correctly | test | loss(identical views) < loss(random pairs) |
| NT-Xent value | test on a hand-computable case (B = 2, orthogonal unit vectors) | matches the analytic value within 1e-5 |
| Symmetry | test | swapping view1 and view2 gives the same loss |
| MAE loss | test | 0 when prediction == target on the masked rows; ignores unmasked rows; mask ratio within ±2% of the configured value |
| Suite | `python -m pytest -q` | all pass |

---

## T5: Training loop + checkpoints (PRD FR-4)
- [ ] `ssl.train(...)`: AdamW, fixed epochs, room-balanced crops, label-free validation = SSL loss on **dev-holdout** crops (`dev_fold == "holdout"`). Log train and holdout loss per epoch.
- [ ] Monitor collapse: per-dimension embedding std, plus the effective rank of a batch of embeddings.
- [ ] Checkpoint to `outputs/task2/<run>/`: encoder weights, config, normalization stats, git hash, loss history.
- [ ] `scripts/task2_pretrain.py --run NAME --seed S [--method simclr|mae] [--epochs N] [--max-windows N]`

🔒 **Gate T5**
| Check | Command | Pass if |
|---|---|---|
| Can overfit a tiny set | `python scripts/task2_pretrain.py --run debug --max-windows 256 --epochs 50` | final train loss ≤ 50% of epoch-1 loss |
| Full run learns | `python scripts/task2_pretrain.py --run simclr_s0 --seed 0` | train loss decreases; holdout SSL loss decreases then flattens (not rising steadily) |
| No collapse | from the log | mean per-dimension embedding std > 1e-2; effective rank > 10 |
| Reload is exact | test or small script | reloaded encoder gives identical embeddings (max abs diff < 1e-5) |
| No labels used | code review + grep | `ssl_data.py`, `ssl.py` and `task2_pretrain.py` never read the `activity` column |
| Runtime | log | one full run ≤ ~15 min on the RTX 3050 (note the actual time) |

---

## T6: Probe pipeline + baselines (PRD FR-5, FR-6)
- [ ] `src/xroom/probe.py`: StandardScaler + LogisticRegression(class_weight="balanced"). C tuned on `dev_fold` train → holdout by macro-F1, refit on all dev, score val. Macro-F1, per-class F1, confusion matrix, capture-level macro-F1, capture bootstrap CI.
- [ ] Probe sets: per-room A (6 classes), per-room B (5 classes), joint (8 classes), cross-room preview (A → B and B → A on sit/stand/walk).
- [ ] Baselines: hand-crafted (per-row normalized mean + std, 484 features), random-init encoder, PCA-128 (dev-fitted).
- [ ] `scripts/task2_probe.py --encoder RUN` and `--baselines-only` → `outputs/task2/probe/<name>/metrics.json`
- [ ] `tests/test_probe.py`

🔒 **Gate T6**
| Check | How | Pass if |
|---|---|---|
| Macro-F1 correct | test | equals `sklearn.metrics.f1_score(average="macro")` on toy data |
| No validation in tuning | test + assertion in code | probe tuning only sees `dev_fold` rows; the val mask is used exactly once, for the final score |
| Baselines above chance | `python scripts/task2_probe.py --baselines-only` | per-room macro-F1 > 1/6 (A) and > 1/5 (B) for the hand-crafted baseline |
| Shuffled-label probe | flag `--shuffle-labels` | macro-F1 at chance level (±0.05) |
| Numbers recorded | Results log | hand-crafted, random-init and PCA rows filled in |
| Suite | `python -m pytest -q` | all pass |

---

## T7: Main SSL result (PRD FR-5, success criteria)
- [ ] Pre-train SimCLR with the **default config** for seeds 0, 1, 2. Do not tune with activity labels.
- [ ] Probe each seed. Aggregate mean ± std per probe set.
- [ ] Save `outputs/task2/probe/simclr_default/summary.json`.

🔒 **Gate T7**
| Check | Pass if |
|---|---|
| SSL beats random-init | per-room A and B macro-F1: the minimum over SSL seeds > the maximum over random-init seeds. **If this fails, stop and debug:** augmentation strength (T2 preview), temperature, learning rate, batch size, collapse metrics (T5). Do not move on until it passes. |
| Stable across seeds | std over seeds ≤ 0.05 macro-F1 (otherwise note it and consider more seeds) |
| Cross-room preview | reported for A → B and B → A (any value; note it against chance = 0.33) |
| Results log | table filled in, with run names and commit hash |

---

## T8: Figures (PRD FR-8)
- [ ] `scripts/task2_figures.py`:
  - UMAP (fit on dev embeddings, transform val; seed 0), two panels: colour = room, colour = activity (marker = room);
  - SSL loss curves;
  - probe bar chart (SSL vs baselines, per room, error bars over seeds).
- [ ] Copy the final PNGs to `reports/figures/task2_*.png`.

🔒 **Gate T8**
| Check | Pass if |
|---|---|
| Files exist | `reports/figures/task2_umap.png`, `task2_loss_curves.png`, `task2_probe_bars.png` |
| UMAP is honest | fitted on dev only (stated in the caption); both rooms visible; legend readable; same seed gives the same picture |
| Visual review | open each figure: no overlapping labels, axes labelled, error bars present |
| Interpretation | 2–3 sentences in the Results log: does the embedding cluster by room, by activity, or both? |

---

## T9: Controls + ablations (PRD FR-7, FR-9)
- [ ] Room probe on embeddings (report accuracy).
- [ ] Format-shift probe: activity probe scored on validation windows smoothed in time and frequency (reuse `harmonize` from `scripts/task1_ablations.py`).
- [ ] Contiguous Room B split for the per-room B probe (reuse `contiguous_split` from `scripts/task1_ablations.py`).
- [ ] Ablations, in priority order, ≥ 2 seeds each: augmentation set → input normalization → SimCLR vs MAE → embedding size → crop range.
- [ ] Optional: supervised upper bound (same encoder, trained with dev labels; clearly labelled).
- [ ] `scripts/task2_ablations.py` → `outputs/task2/ablations/results.csv`

🔒 **Gate T9**
| Check | Pass if |
|---|---|
| Controls reported | room probe, format-shift probe, contiguous split and cross-room preview are all in `results.csv` / `metrics.json` |
| Ablation rows complete | each row has ≥ 2 seeds, per-room macro-F1 mean ± std, and the cross-room preview |
| Baseline row unchanged | the default-config row matches T7 within seed noise |
| Takeaways written | Results log: "what helps / what does not", 3–5 bullets |

---

## T10: Task 3 hand-off API (PRD FR-10)
- [ ] `xroom.encoder.load_encoder(cfg, run="default")` and `embed(encoder, windows) -> np.ndarray (n, d)`: batched, GPU if available, applies the saved dev normalization, eval mode.
- [ ] Publish the **label-clean default encoder**: the default config (or selection by SSL holdout loss only). Write `outputs/task2/default/PROVENANCE.md` stating how it was chosen. Label-tuned variants get other names.
- [ ] Add `paths.encoder_dir` to the config.
- [ ] `tests/test_encoder_api.py`

🔒 **Gate T10**
| Check | Pass if |
|---|---|
| API works | `embed(load_encoder(cfg), x[:10])` returns shape `(10, 128)`, finite |
| Deterministic | two calls give identical output |
| Consistent | matches the embeddings `task2_probe.py` used (max abs diff < 1e-5) |
| Label-clean | `PROVENANCE.md` says no activity labels were used for training or selection |
| Suite | `python -m pytest -q` | all pass |

---

## T11: Report section
- [ ] `reports/task2_report_section.md`, same structure as `reports/task1_report_section.md`:
  - task + summary;
  - method (data, augmentations, encoder, objective);
  - evaluation protocol;
  - results table with CIs;
  - UMAP;
  - controls;
  - ablations;
  - limitations;
  - Task 3 hand-off;
  - reproduction commands.
- [ ] Update the AI-use and contribution disclosure (project-level).

🔒 **Gate T11**
| Check | Pass if |
|---|---|
| Numbers verified | every number in the section matches `metrics.json` / `results.csv` (spot-check script or manual table) |
| Links resolve | all figure and report links open on GitHub |
| Rules checklist | every item below is ticked |

---

## Rules checklist (check before reporting any result)
- [ ] No validation captures in SSL training, normalization statistics, checkpoint selection or probe tuning
- [ ] No activity labels in SSL training or default-encoder selection
- [ ] Splits are by capture; windows of a capture are never split
- [ ] Features come only from the CSI signal (no filenames, folders or capture length)
- [ ] Python only; no authors' MATLAB code; no other team's code
- [ ] Commits are small and frequent, with no AI attribution lines

## Decisions log
| Date | Decision | Why |
|---|---|---|
| 2026-10-18 | Probe headline = mean of per-room macro-F1 (joint 8-class also reported) | Activities differ per room; per-room probes are cleaner (PRD §12) |
| 2026-10-18 | UMAP fitted on dev embeddings, val transformed | Keeps validation out of every fit |
| 2026-10-19 | Normalization statistics weight the two rooms equally (mean of per-room moments), fitted on all dev rows | Room B has ~3.5x more rows; matches the room-balanced batches |
| 2026-10-19 | SSL trains on `dev_fold == train` captures; `holdout` captures give the label-free SSL validation loss | The holdout loss is only meaningful on captures the encoder never trained on |
| 2026-10-19 | `configs/task2.yaml` layers on `task1.yaml` via `base_config` and may not redefine its sections | Task 1 preprocessing defaults stay untouched |
| 2026-10-19 | Augmentations are batched torch functions applied on the GPU after normalization (z-space; masking writes 0 = dev mean) | Fast, and one generator makes every view reproducible |
| 2026-10-19 | Local disk is nearly full: the venv, `data_cache/` and `outputs/` live on an external drive; `data_cache` and `outputs` in the repo are symlinks to it (`.gitignore` entries without trailing slash so the symlinks stay ignored) | CUDA PyTorch alone is ~4 GB; paths in the configs are unchanged |
| | | |

## Results log (fill in at each gate)
| Gate | Run / commit | Key numbers |
|---|---|---|
| T0 | `ed058cf` + T0 commit | 4680 windows, 7 excluded captures; 29 passed; Task 1 balanced 1.0000; torch 2.14.1+cu126, CUDA True (RTX 3050 4 GB), umap-learn OK. Python 3.12 venv. |
| T1 | T1 commit | 9 tests: no val id in 200 batches (toy + real cache); injected val id raises; batch (B, 96–150, 242) float32 finite; Room A share in [0.45, 0.55]; dev-only stats match manual within 1e-5; deterministic per seed |
| T2 | T2 commit | 23 tests: shape/dtype/finite, exact identity at p = 0, masked fraction in range, deterministic per seed. Preview: views alike but not identical, none pure noise; Room B rows visibly noisier than Room A (the format confound the `format` augmentation targets) |
| T3 | T3 commit | 7 tests: output (B, 128) for T = 32/96/150; all params get finite non-zero grads; GPU = CPU within 1e-4; eval deterministic. Encoder 824,704 params (< 1.5 M); peak GPU memory 291 MB for 2 x 256 crops (fits 4 GB) |
| T4 | T4 commit | loss(identical) < loss(random); analytic B = 2 case within 1e-5; symmetric; MAE = 0 on perfect masked rows and ignores unmasked rows; masked share 0.5 ± 0.02 |
| T5 | | train loss epoch 1 → last: ; holdout loss: ; runtime: |
| T6 | | hand-crafted A / B: ; random-init A / B: ; PCA A / B: |
| T7 | | SSL A / B (mean ± std): ; joint: ; cross-room A→B / B→A: |
| T8 | | UMAP interpretation: |
| T9 | | takeaways: |
