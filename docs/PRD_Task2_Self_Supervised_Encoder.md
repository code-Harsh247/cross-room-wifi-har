# PRD — Task 2: "Learn Without Labels" (Self-Supervised CSI Encoder)

**Course:** CS60055 Ubiquitous Computing — Hackathon Challenge 2
**Project:** Cross-Room Few-Shot Adaptation for WiFi Activity Recognition
**Scope:** Task 2 only. It must leave a clean hand-off for Task 3.
**Status:** Draft v1 (2026-10-18), written for hand-over to Abhinav
**Deadline (whole project):** October 31, 2026 · **Presentation:** November 2/3, 2026
**Step-by-step checklist:** [tasks_task2.md](tasks_task2.md)

---

## 1. Summary

Train a **CSI encoder without activity labels** (self-supervised learning, SSL) on development windows of both rooms. Then **freeze** it and test it with a **linear classifier** (a "linear probe") trained on its embeddings. The metric is **macro-F1** on validation.

The encoder is the backbone of Task 3: few-shot activity recognition in a new room uses its frozen embeddings (prototypes, nearest centroid). Task 2 is therefore judged both by its own probe score and by how useful its embeddings are across rooms.

## 2. What the brief says (slides 8, 10, 11, 12)

| Source | Requirement |
|---|---|
| Slide 8 | "Learn a CSI encoder without activity labels. Freeze it and test it with a linear classifier." Metric: **Macro-F1**. |
| Slide 11 hint | Contrastive learning with augmentations (**crop, noise, subcarrier masking**), or **masked reconstruction**. Read up: **SimCLR, MAE**. |
| Slide 10 rules | Use only **development captures** for training. **Validation captures are never used for training** (this includes SSL pre-training). Normalization statistics are fitted on dev only. Features may only come from the CSI signal (no filenames, folders or capture length). Python only. No other team's code. |
| Slide 12 report | Task 2 linear-probe macro-F1. **UMAP figure by room and by activity.** Ablation of what helps and what does not. AI-use disclosure. |
| Slide 11 (Task 3) | Task 3 will use "class prototypes or nearest centroid on frozen embeddings; per-room normalization; test-time adaptation". |

## 3. Starting point (what already exists, from Task 1)

Read `reports/task1_report_section.md` first. It is the short version of everything below.

| Asset | Where | Use in Task 2 |
|---|---|---|
| Decoded capture sequences | `data_cache/sequences.joblib`, via `xroom.dataset.load_sequences(cfg)` | SSL training crops. Room A: 150 rows per capture (~3.9 s). Room B: 500 rows. |
| 150-row windows | `data_cache/windows_L150.npy` + `_meta.csv`, via `xroom.dataset.load_windows(cfg)` | Probe training and evaluation (same windows as Task 1). |
| Windows of any length | `xroom.dataset.windows_from_sequences(table, seqs, length, stride)` | Crops and length ablations. |
| Splits | `split` (`dev`/`val`) and `dev_fold` (`train`/`holdout`) columns in the meta | Train SSL on dev only. Tune the probe on dev holdout. Score once on val. |
| Null subcarriers | `cfg["common"]["null_subcarriers"]`; `xroom.features.data_subcarriers(...)` gives the 242 data subcarriers | Encoder input channels. |
| Row normalization | `xroom.features.spectrum_features(..., variant="row")` (divide each row by its mean) | Recommended input normalization (removes the format scale gap). |
| Metrics helpers | `xroom.eval` (balanced accuracy, bootstrap CI) | Add a macro-F1 helper. |
| Task 1 router | `xroom.router.load_router(cfg)` / `route(...)` | Not needed for Task 2. Task 3 combines both. |

### Data available (150-row windows)

| Activity | A dev | A val | B dev | B val |
|---|---|---|---|---|
| empty | – | – | 726 | 75 |
| lying | – | – | 978 | 75 |
| sit | 172 | 24 | 384 | 75 |
| stand | 115 | 25 | 588 | 75 |
| walk | 154 | 25 | 552 | 75 |
| fall | 177 | 25 | – | – |
| sitdown | 155 | 25 | – | – |
| standup | 155 | 25 | – | – |
| **Total** | **928** | **149** | **3,228** | **375** |

- Room A has 1 window per capture; Room B has 3 (non-overlapping).
- Dev holdout (for tuning): 186 A + 215 B captures.

### Known pitfalls inherited from Task 1 (must be handled)

1. **Format confound.** The rooms differ in format: scale, noise and smoothness. Room B rows are much noisier, and Room A rows are interpolated, so smooth in time. An encoder can trivially separate the rooms by format. Task 1/M8 showed that features describing **variation over time** (std, percentiles) mostly read Room B's row noise. An SSL encoder can fall into the same trap and organise its embedding by room and format instead of activity.
2. **Room-specific activities.** Five of the eight activities exist in one room only.
3. **Room B consecutive chunks.** Consecutive Room B files are chunks of one recording, so a random validation draw has neighbours in dev. This may inflate Room B activity scores, so report a contiguous-split check (reuse `contiguous_split` in `scripts/task1_ablations.py`).
4. **Label hygiene for Task 3.** The Task 3 rules forbid target-room activity labels beyond the K support captures, *even for tuning*. If the encoder's settings are chosen using dev activity labels of both rooms, that encoder is not clean for Task 3 (the same issue as the Task 1 router; see `reports/task1_router_variants.md` §5).

## 4. Goals and non-goals

### Goals
1. A self-supervised encoder trained on **dev windows of both rooms with no activity labels**.
2. A reproducible **linear-probe evaluation** reporting macro-F1 on validation, with clear baselines.
3. The required **UMAP figures** (colour by room; colour by activity).
4. An **ablation** of what helps: augmentations, input normalization, SSL method, embedding size.
5. A frozen-encoder **API for Task 3** (`load_encoder`, `embed`), with a **label-clean default encoder** whose settings were chosen without activity labels.
6. Controls that show whether the embeddings capture **activity** or just **room/format**.

### Non-goals
- Few-shot adaptation itself (Task 3).
- Beating the state of the art. A simple encoder that clearly beats its baselines is enough.
- Fine-tuning the encoder with labels as the main result. A supervised encoder is allowed only as a clearly labelled upper-bound reference.

## 5. Definitions

- **Window:** a 150 × 256 amplitude array. The encoder uses the 242 data subcarriers.
- **Encoder f:** maps a window to an embedding z ∈ R^d (default d = 128). Used *frozen* after pre-training.
- **Projection head g:** used only during SSL training (SimCLR) and discarded afterwards.
- **Linear probe:** StandardScaler + multinomial logistic regression (class-balanced) on frozen embeddings.
- **Macro-F1:** the unweighted mean of per-class F1 scores. Every activity counts equally regardless of its size.

## 6. Functional requirements

### FR-1 SSL data pipeline
- Training examples are **random contiguous crops** from **dev** capture sequences:
  - crop length sampled in [96, 150] rows (configurable);
  - Room A sequences are 150 rows, Room B 500 rows;
  - no crop crosses a capture.
- **No validation capture may appear in SSL training.** Enforce this with an assertion in code and a unit test.
- **Sampling:** balance the rooms per batch (≈ 50/50), so Room B's 3.5× more windows do not dominate. This uses the room identity only, never activity.
- **Input normalization**, applied inside the dataset:
  - per-row normalization (each row divided by its mean, as in Task 1);
  - optionally log(1 + x);
  - then per-subcarrier standardization with statistics fitted on **dev only**, saved with the checkpoint.
- **Determinism:** global seed + worker seeds. The same seed gives the same crops.

### FR-2 Augmentations (for contrastive SSL)
Implement as pure functions on a (T, C) tensor, each with a probability and a strength in config:

| Augmentation | What it does | Why |
|---|---|---|
| Temporal crop + resize | random crop, linearly resized to a fixed length | from the brief; invariance to timing |
| Gaussian noise | additive noise, σ relative to the window's std | from the brief |
| Subcarrier masking | zero 1–3 contiguous blocks of subcarriers | from the brief |
| Time masking | zero 1–2 contiguous row blocks | robustness |
| Per-row gain jitter | multiply each row by 1 + ε | removes per-packet gain cues |
| Time smoothing / un-smoothing *(format-robustness)* | random moving average over rows, or added row noise | breaks the "smooth = Room A, noisy = Room B" format shortcut |

Each augmentation must keep shape and dtype, keep values finite, and be reproducible with a seed.

### FR-3 Encoder
- **Default:** 1-D CNN over time with the 242 subcarriers as input channels:
  - 3–4 conv blocks (Conv1d → BatchNorm → ReLU, kernel 5–7, some stride or pooling);
  - global average pooling over time;
  - linear layer to d = 128.
- Must accept **any length ≥ 32 rows** (thanks to global pooling).
- **Projection head** (SimCLR): 2-layer MLP, 128 → 128 → 64.
- **Alternative (ablation):** a small Transformer over time patches. Optional.

### FR-4 SSL objectives
- **Primary: SimCLR-style contrastive learning** (NT-Xent loss, temperature τ ≈ 0.1–0.5). Two augmented views of the same crop are positives; all other examples in the batch are negatives.
- **Secondary (ablation): masked reconstruction (MAE-style).** Mask ~50% of time patches, reconstruct the masked rows from the visible ones (MSE on masked positions), and use the encoder output as the embedding.
- **Training:**
  - Adam/AdamW;
  - fixed epoch budget;
  - **label-free early stopping/selection:** the SSL loss on dev-holdout crops. **No activity labels in any SSL training or checkpoint selection.**
- Every run saves a checkpoint (encoder weights + config + normalization stats + git hash).

### FR-5 Linear-probe evaluation
- **Embeddings:** frozen encoder on the fixed 150-row windows (`load_windows`).
- **Probe:** StandardScaler + LogisticRegression(class_weight="balanced"). C tuned on `dev_fold=train → dev_fold=holdout` by macro-F1, then refit on all dev, then scored **once** on val.
- **Report all of these:**

| Name | Train → test | Classes |
|---|---|---|
| **Per-room probe, Room A** (primary) | A dev → A val | 6 |
| **Per-room probe, Room B** (primary) | B dev → B val | 5 |
| **Headline** | mean of the two per-room macro-F1 | — |
| Joint probe | both rooms dev → both val | 8 (union) |
| Cross-room preview (for Task 3) | A dev sit/stand/walk → B val sit/stand/walk, and B → A | 3 |

- **Also report:**
  - per-class F1;
  - confusion matrices for the per-room probes;
  - capture-level macro-F1 (Room B: mean of its 3 windows' probabilities);
  - 95% capture-level bootstrap intervals.
- **SSL runs are stochastic: ≥ 3 seeds**, reported as mean ± std.

### FR-6 Baselines (same probe, same splits)

| Baseline | Purpose |
|---|---|
| **Random-init encoder** (same architecture, untrained) | SSL must beat it clearly, or the training is not doing anything |
| **Hand-crafted features**: per-row normalized mean + std over time (484 features) | A strong simple reference |
| **PCA of row-normalized windows** (flattened, dev-fitted, 128-d) | A linear unsupervised reference |
| **Supervised encoder** (same architecture, trained with dev activity labels), *optional* | Upper bound. Label it clearly; not a Task 2 result |

### FR-7 Shortcut and room-information controls
SSL embeddings may encode room and format more than activity. Report:
- **Room probe on embeddings:** linear probe for room A/B (expected high; just report).
- **Format-shift probe:** score the activity probe on validation windows smoothed in time and frequency (as in M8). A large drop means the embedding relies on format texture.
- **Cross-room preview** (FR-5): the most relevant number for Task 3.
- **Contiguous Room B split** check for the per-room B probe.

### FR-8 Figures
- **UMAP of validation embeddings**, two panels: (a) coloured by room, (b) coloured by activity, with marker shape for room. Required by the brief.
  - Fit UMAP on dev embeddings and transform val (or fit on val only); state which.
  - Fixed seed.
  - Fallback if `umap-learn` is unavailable: t-SNE (scikit-learn), stated in the caption.
- SSL training-loss curves.
- Probe macro-F1 bars: SSL vs baselines, per room, with error bars over seeds.
- Copy final figures to `reports/figures/` (git-ignored `outputs/` are not visible on GitHub).

### FR-9 Ablations (pick what time allows, in this order)
1. **Augmentation set:** all; without format-robustness; without subcarrier masking; crop only.
2. **Input normalization:** per-row + standardize (default) vs standardize only vs log.
3. **Method:** SimCLR vs MAE.
4. **Embedding size:** 64 / 128 / 256.
5. **Crop length range** for SSL training.

Each ablation reports per-room probe macro-F1 (mean ± std over ≥ 2 seeds) and the cross-room preview.

### FR-10 Hand-off API for Task 3
- `xroom.encoder.load_encoder(cfg, run="default") -> Encoder` and `embed(encoder, windows) -> (n, d) np.ndarray`. Batched, GPU if available, deterministic in eval mode. It applies the saved dev normalization stats internally.
- The **default encoder** for Task 3 must have its settings chosen **without activity labels**: either config defaults fixed in advance, or selection by SSL holdout loss. Encoders tuned with activity-labelled probes are saved under a different name and documented as ablations (the same rule as the Task 1 strict router).

## 7. Non-functional requirements

| Area | Requirement |
|---|---|
| Reproducibility | One command per stage (`scripts/task2_*.py`). All randomness seeded. Config in `configs/task2.yaml`. Checkpoints record config + git hash. |
| Compute | One full SSL run ≤ ~15 min on the RTX 3050 laptop GPU (4 GB VRAM). Batch size fits in 4 GB. CPU fallback works (slower). |
| Code style | Match `src/xroom/` (type hints, short docstrings, pure functions where possible). New modules under `src/xroom/`, scripts under `scripts/`, tests under `tests/`. |
| Tests | Every module gets unit tests. `python -m pytest -q` stays green (29 tests today). |
| Data rules | Validation is never in SSL training, normalization fitting, model selection or probe training. This is enforced by assertions. |
| Git | Small, frequent commits (history counts). **No AI attribution lines in commit messages** (`.claude/settings.json` sets this). Do not commit `data_cache/`, `outputs/`, raw data or checkpoints. |
| Docs | Results reports in `reports/task2_*.md`. Final report section in `reports/task2_report_section.md`. |

## 8. Proposed code layout

```
configs/task2.yaml                  # SSL, augmentation, encoder, probe settings
src/xroom/ssl_data.py               # crop sampler, room-balanced batches, input normalization (FR-1)
src/xroom/augment.py                # augmentations (FR-2)
src/xroom/encoder.py                # CNN encoder, projection head, load_encoder/embed (FR-3, FR-10)
src/xroom/ssl.py                    # NT-Xent, MAE objective, training loop, checkpointing (FR-4)
src/xroom/probe.py                  # linear probe + macro-F1 + bootstrap (FR-5)
scripts/task2_pretrain.py           # train one SSL run (seed, config overrides)
scripts/task2_probe.py              # embeddings + probes + baselines + controls -> metrics.json
scripts/task2_figures.py            # UMAP, loss curves, bar charts
scripts/task2_ablations.py          # ablation grid
tests/test_ssl_data.py, test_augment.py, test_encoder.py, test_ssl.py, test_probe.py
outputs/task2/<run_name>/           # checkpoints, metrics (git-ignored)
reports/task2_*.md, reports/figures/task2_*.png
```

## 9. Milestones

| # | Milestone | Done when |
|---|---|---|
| T0 | Hand-over setup | Data in place, Task 1 caches rebuilt, 29 tests pass, `umap-learn` installed |
| T1 | Config + SSL data pipeline | Crops from dev only, room-balanced, normalized. Tests prove no val leakage. |
| T2 | Augmentations | All augmentations tested. Visual sanity figure. |
| T3 | Encoder + heads | Variable-length forward pass, gradients flow, runs on GPU |
| T4 | SSL objectives | NT-Xent (+ MAE) tested on toy inputs |
| T5 | Training loop | Tiny-subset overfit works. A full run's loss decreases. Checkpoint reloads identically. |
| T6 | Probe pipeline + baselines | Probes, macro-F1, bootstrap. Hand-crafted and random-init baselines produce sane numbers. |
| T7 | Main SSL result | ≥ 3 seeds. SSL clearly beats random-init. Metrics saved. |
| T8 | Figures | UMAP (room / activity), loss curves, bar chart |
| T9 | Controls + ablations | Room probe, format-shift probe, cross-room preview, contiguous split, ablation table |
| T10 | Task 3 hand-off API | `load_encoder` / `embed`, label-clean default encoder, tests |
| T11 | Report section | `reports/task2_report_section.md` with all numbers checked against outputs |

## 10. Success criteria

- **Must:**
  - the whole pipeline runs end to end from one config;
  - no validation data in training, normalization or selection;
  - per-room probe macro-F1 is reported with intervals over ≥ 3 seeds;
  - the UMAP figures exist.
- **Must:** the SSL encoder beats the **random-init encoder** on the per-room probes by a clear margin (non-overlapping seed ranges). If not, debug before moving on.
- **Should:**
  - SSL is competitive with the hand-crafted mean+std baseline;
  - the cross-room preview is clearly above chance (1/3 for 3 classes).
- **Should:** the ablation table shows which augmentations matter.
- **Could:** MAE comparison; supervised upper bound.

No absolute macro-F1 target is set. Calibrate expectations after the T6 baselines.

## 11. Risks and mitigations

| Risk | Mitigation |
|---|---|
| The SSL embedding organises by room and format, not activity | Format-robustness augmentations, per-row normalization, room-balanced batches. Measure with the format-shift probe and UMAP. |
| The encoder learns nothing (random-init level) | T5 overfit check. Check that the loss decreases. Check that augmentations are not too strong (positives must still look alike). Lower τ or raise the batch size. |
| Dimensional collapse (all embeddings identical) | Monitor embedding std and rank during training. NT-Xent with a projection head usually avoids it. |
| Room A is small (928 dev windows) | Random crops from each 150-row sequence, plus room-balanced sampling. |
| Probe tuned on validation by accident | Only `dev_fold` is used for tuning, enforced by assertions. Validation is scored once by the final script. |
| GPU memory (4 GB) | Batch ≤ 256 crops of ≤ 150 × 242 floats. Mixed precision optional. |
| Task 3 label-hygiene violation | Default encoder selected without activity labels (FR-10). Label-tuned variants named and documented separately. |

## 12. Open questions (decide early, record in tasks_task2.md)

1. **Probe headline:** per-room mean (recommended) or joint 8-class? Report both, and pick one for the abstract.
2. **UMAP fitting:** fit on dev and transform val (recommended, no val in fitting), or fit on val only.
3. **Room-balanced sampling** uses room identity during SSL. This is not an activity label (same reasoning as the Task 1 router), but state it in the report.
4. **Crop length for SSL vs probe windows:** probing on 150-row windows is the default. A probe-length ablation is optional.
