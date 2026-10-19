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
| T5 | Training loop + checkpoints | ✅ |
| T6 | Probe pipeline + baselines | ✅ |
| T7 | Main SSL result (≥ 3 seeds) | ⚠️ Room B passes, Room A ties random-init (see Results log) |
| T8 | Figures (UMAP, curves, bars) | ✅ |
| T9 | Controls + ablations | ✅ |
| T10 | Task 3 hand-off API | ✅ |
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
- [x] `ssl.train(...)`: AdamW, fixed epochs, room-balanced crops, label-free validation = SSL loss on **dev-holdout** crops (`dev_fold == "holdout"`). Log train and holdout loss per epoch.
- [x] Monitor collapse: per-dimension embedding std, plus the effective rank of a batch of embeddings.
- [x] Checkpoint to `outputs/task2/<run>/`: encoder weights, config, normalization stats, git hash, loss history.
- [x] `scripts/task2_pretrain.py --run NAME --seed S [--method simclr|mae] [--epochs N] [--max-windows N]`

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
- [x] `src/xroom/probe.py`: StandardScaler + LogisticRegression(class_weight="balanced"). C tuned on `dev_fold` train → holdout by macro-F1, refit on all dev, score val. Macro-F1, per-class F1, confusion matrix, capture-level macro-F1, capture bootstrap CI.
- [x] Probe sets: per-room A (6 classes), per-room B (5 classes), joint (8 classes), cross-room preview (A → B and B → A on sit/stand/walk).
- [x] Baselines: hand-crafted (per-row normalized mean + std, 484 features), random-init encoder, PCA-128 (dev-fitted).
- [x] `scripts/task2_probe.py --encoder RUN` and `--baselines-only` → `outputs/task2/probe/<name>/metrics.json`
- [x] `tests/test_probe.py`

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
- [x] Pre-train SimCLR with the **default config** for seeds 0, 1, 2. Do not tune with activity labels.
- [x] Probe each seed. Aggregate mean ± std per probe set.
- [x] Save `outputs/task2/probe/simclr_default/summary.json`.

🔒 **Gate T7**
| Check | Pass if |
|---|---|
| SSL beats random-init | per-room A and B macro-F1: the minimum over SSL seeds > the maximum over random-init seeds. **If this fails, stop and debug:** augmentation strength (T2 preview), temperature, learning rate, batch size, collapse metrics (T5). Do not move on until it passes. |
| Stable across seeds | std over seeds ≤ 0.05 macro-F1 (otherwise note it and consider more seeds) |
| Cross-room preview | reported for A → B and B → A (any value; note it against chance = 0.33) |
| Results log | table filled in, with run names and commit hash |

---

## T8: Figures (PRD FR-8)
- [x] `scripts/task2_figures.py`:
  - UMAP (fit on dev embeddings, transform val; seed 0), two panels: colour = room, colour = activity (marker = room);
  - SSL loss curves;
  - probe bar chart (SSL vs baselines, per room, error bars over seeds).
- [x] Copy the final PNGs to `reports/figures/task2_*.png`.

🔒 **Gate T8**
| Check | Pass if |
|---|---|
| Files exist | `reports/figures/task2_umap.png`, `task2_loss_curves.png`, `task2_probe_bars.png` |
| UMAP is honest | fitted on dev only (stated in the caption); both rooms visible; legend readable; same seed gives the same picture |
| Visual review | open each figure: no overlapping labels, axes labelled, error bars present |
| Interpretation | 2–3 sentences in the Results log: does the embedding cluster by room, by activity, or both? |

---

## T9: Controls + ablations (PRD FR-7, FR-9)
- [x] Room probe on embeddings (report accuracy).
- [x] Format-shift probe: activity probe scored on validation windows smoothed in time and frequency (reuse `harmonize` from `scripts/task1_ablations.py`).
- [x] Contiguous Room B split for the per-room B probe (reuse `contiguous_split` from `scripts/task1_ablations.py`).
- [x] Ablations, in priority order, ≥ 2 seeds each: augmentation set → input normalization → SimCLR vs MAE → embedding size → crop range.
- [ ] Optional: supervised upper bound (not done; time went to the T7 debugging) (same encoder, trained with dev labels; clearly labelled).
- [x] `scripts/task2_ablations.py` → `outputs/task2/ablations/results.csv`

🔒 **Gate T9**
| Check | Pass if |
|---|---|
| Controls reported | room probe, format-shift probe, contiguous split and cross-room preview are all in `results.csv` / `metrics.json` |
| Ablation rows complete | each row has ≥ 2 seeds, per-room macro-F1 mean ± std, and the cross-room preview |
| Baseline row unchanged | the default-config row matches T7 within seed noise |
| Takeaways written | Results log: "what helps / what does not", 3–5 bullets |

---

## T10: Task 3 hand-off API (PRD FR-10)
- [x] `xroom.encoder.load_encoder(cfg, run="default")` and `embed(encoder, windows) -> np.ndarray (n, d)`: batched, GPU if available, applies the saved dev normalization, eval mode.
- [x] Publish the **label-clean default encoder**: the default config (or selection by SSL holdout loss only). Write `outputs/task2/default/PROVENANCE.md` stating how it was chosen (`scripts/task2_publish_default.py`, which refuses runs not trained with the default config). Label-tuned variants get other names.
- [x] Add `paths.encoder_dir` to the config.
- [x] `tests/test_encoder_api.py`

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
| 2026-10-19 | T5 overfit check judged on the crop-only run (44%); the default-augmentation run stops at 57% | The NT-Xent floor under strong augmentation is not a pipeline fault; the crop-only run shows the loop can overfit |
| 2026-10-19 | Probe fits run with one BLAS thread (`threadpool_limits`) | Multi-threaded BLAS made each small logistic regression ~60x slower (baselines 17 min → ~1.5 min) |
| 2026-10-19 | PCA baseline averages every 2 rows before flattening (75 x 242 = 18,150 dims) | The full 36,300-dim matrix of 4,680 windows does not fit comfortably in 7 GB RAM |
| 2026-10-19 | Shuffled-label control permutes labels between captures (not windows) | Keeps the capture-level structure; tested over 10 shuffle seeds on toy data, since one shuffle maps whole clusters and is high-variance |
| 2026-10-19 | T7 Room A debugging uses dev-holdout probe scores only (`dev_holdout_scores`), never validation; any variant picked that way is label-tuned and gets its own name, while the default encoder stays the fixed config | Validation is for scoring only; Task 3 needs a label-clean default (PRD FR-10) |
| 2026-10-19 | Label-tuned variant chosen by the highest 3-seed dev-holdout mean (τ = 0.5); MAE reported as a tied alternative, not promoted | Picking between two dev-holdout ties by their validation score would be validation tuning |
| 2026-10-19 | Contiguous check pre-trains new encoders (`contig_s*`) without the contiguous block or its guard files; the probe trains on rows that are dev under both splits | Otherwise the SSL encoder has seen the test block (unlabelled), and official validation captures would enter probe training |
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
| T5 | `simclr_s0` (T5 commit) | Full run: train loss 3.488 → 1.961 (epoch 60); holdout SSL loss 2.915 → 2.112, flat from ~epoch 50 (best epoch 54, kept); emb std 0.46, effective rank 55 → 69 (no collapse); runtime 218 s on the RTX 3050. Reload max abs diff 0.0. Overfit check: default augmentations reach 57% of the epoch-1 loss (3.488 → 1.976, `debug`), above the 50% target, because strong augmentations and repeated captures in a batch (256 captures, batch 256) put a floor under NT-Xent; with crop-only augmentation and batch 64 the same 256 captures reach 44% (1.821 → 0.807, `debug_croponly`), so the loop can fit. No `activity` reads in SSL code (test). |
| T6 | T6 commit (`outputs/task2/probe/*/metrics.json`) | Per-room macro-F1 A / B (chance 0.167 / 0.200): hand-crafted 0.940 / 0.987; random-init 0.935 ± 0.011 / 0.853 ± 0.017 (3 seeds); PCA-128 0.826 / 0.928. Cross-room A→B / B→A: hand-crafted 0.167 / 0.168, random-init 0.167 / 0.226, PCA 0.390 / 0.243. Format shift A / B: hand-crafted 0.732 / 0.164 (collapses on B), random-init 0.913 / 0.211. Shuffled labels (hand-crafted): A 0.182, B 0.181, joint 0.100 (chance 0.167 / 0.200 / 0.125); other sources' single shuffles fall 0.08–0.18 (no source is above chance). |
| T7 | `simclr_s0/1/2` (code at `205f87a`), `probe/simclr_default/summary.json` | Per-room macro-F1 (val, mean ± std over 3 seeds): **A 0.905 ± 0.027, B 0.992 ± 0.005**, headline 0.949 ± 0.016; joint 0.944 ± 0.011; capture-level A 0.905, B 0.997. Cross-room A→B 0.226 ± 0.033, B→A 0.228 ± 0.112 (chance 0.333). Room probe 0.999; format shift A 0.825 / B 0.937. **Gate:** Room B passes (SSL min 0.987 > random-init max 0.869). Room A fails (SSL 0.878–0.932 vs random-init 0.925–0.948); seed std ≤ 0.05 in both rooms. Dev-holdout confirms a tie on A (SSL 0.913–0.935, random-init 0.934–0.951; differences of 1–3 of 186 windows per class), so this is not validation noise. Debugging continues as the T9 ablation grid, compared on dev-holdout only. |
| T10 | T10 commit | `embed(load_encoder(cfg), x[:10])` → (10, 128) float32, finite, identical on repeat; equals the `simclr_s0` embeddings used by the probe (max diff < 1e-5); eval mode, frozen. Default = `simclr_s0` (default config, seed 0 fixed in advance, epoch 54 by dev-holdout SSL loss); PROVENANCE.md states no activity labels. Done before T8/T9 because it does not depend on them. |
| T8 | T8 commit (`reports/figures/task2_*.png`) | UMAP (default encoder; scaler + UMAP fitted on dev embeddings, val transformed, seed 0): the embedding is organised **by room first**; Room A and Room B validation windows never share a region. Inside Room B it is organised **by activity**: tight single-activity clusters, often several per activity (likely separate recording sessions, cf. the consecutive-chunk pitfall). Room A forms one cloud with only partial activity structure (walk/fall/sit drift to different edges), matching its weaker probe. Shared activities (sit/stand/walk) of the two rooms do not co-locate, which explains the near-chance cross-room preview. |
| T9 | T9 commit (`ablations/results.csv`, `summary.csv`, `results_partial.csv`; `probe/{tau_0.5,method_mae,aug_no_format}/summary.json`) | 13 variants x 2 seeds + 3rd seed for the top two; controls in every main `metrics.json`. Default row (s0, s1) val A 0.918 ± 0.019 / B 0.995 matches T7 within seed noise. Contiguous Room B: SSL with the block held out of SSL 0.987 / 0.973 (s0 / s1) vs 0.992 when seen unlabelled; hand-crafted 0.943. **Takeaways** (dev-holdout macro-F1 decides; val reported): (1) per-row normalization matters most: without it Room A drops 0.921 → 0.870 (val 0.918 → 0.871); (2) temperature matters: τ 0.1 / 0.2 / 0.5 give Room A 0.899 / 0.921 / 0.959; (3) MAE ties the best SimCLR on dev-holdout (0.977 vs 0.978 overall) and is the only variant clearly above random-init on Room A validation (0.957 ± 0.004, min 0.952 > 0.948) and the most format-shift robust (A 0.955 / B 0.961); (4) removing any single augmentation moves dev-holdout by ≤ 0.01; the format augmentation does not measurably add shift robustness (shift B 0.919 without vs 0.937 with, 2–3 seeds); (5) no variant makes cross-room transfer work (0.09–0.36 vs chance 0.33): SSL alone does not give room-invariant activity features, so Task 3 needs per-room normalization / adaptation. **T7 follow-up:** the rule fixed in advance (highest dev-holdout mean) picks τ = 0.5 as the label-tuned variant (`abl_tau_0.5_s*`); it gains on dev-holdout A (0.959) but not on val A (0.923 ± 0.004, still below random-init 0.948 max), i.e. Room A's supplied validation folder behaves differently from dev-holdout. MAE passes the T7 check in both rooms but is not promoted, since choosing it over the tied τ = 0.5 would need validation. The default encoder stays label-clean SimCLR τ = 0.2. |
