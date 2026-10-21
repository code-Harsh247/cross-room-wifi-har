# Task 3: Ideas to Improve the Results

**Status:** proposal list (2026-10-21). Nothing here is implemented yet.
**Context:** [tasks_task3.md](tasks_task3.md) Results log (U6–U9) · [PRD](PRD_Task3_Few_Shot_Cross_Room.md)

---

## Where we stand

| | A → B (K = 20) | B → A (K = 20) |
|---|---|---|
| Default M1, overall macro-F1 | 0.335 | 0.336 |
| Room-only (exclusive) macro-F1 | 0.664 | 0.592 |
| Shared (sit/stand/walk) macro-F1 | 0.115 | 0.081 |
| Best ablation (M5, logistic regression), overall | 0.438 | 0.409 |
| Upper bound (all target labels; not valid) | 0.901 | 0.692 |

- **What works:** room-only activities, learned from the K support captures, improve steadily with K.
- **What fails:** **sit / stand / walk do not transfer between rooms.** They are not just noisy, they are *systematically swapped*. In Room B, sit and stand are mostly predicted as walk, and walk as empty.
- **Cause:** the frozen encoder organises windows **by room first** (Task 2 UMAP; U6 diagnostics). Source-room prototypes land on the wrong target clusters.

## Ground rules for every idea

1. **The pre-registered default (D1 = M1) stays the headline.** Everything below is reported as a **post-hoc improvement attempt** (an ablation). It is clearly labelled as designed after seeing U6–U8 results.
2. **Fix each idea's settings before running it,** and log them in the Decisions log of `tasks_task3.md`. Never tune on target validation scores.
3. **Labels allowed:** source dev labels + K support captures, via `visible_labels`. Unlabelled target **dev** windows may be used. Target **validation** windows are scored only.
4. **Report positive and negative outcomes alike.** A clear negative result ("this physically motivated fix does not transfer, because …") is still a finding for the report.

---

## Ideas (most promising first)

### 1. Room-relative motion-intensity features
- **Idea:** walking makes the WiFi signal fluctuate much more over time than sitting, standing or an empty room, in *any* room. Measure "how much does this window move?" relative to its own room, and use that as a room-independent activity clue.
- **What we do:**
  - Per window, compute simple temporal-variation statistics on the per-row normalized amplitude (e.g. mean over subcarriers of the std over time, and the mean absolute row-to-row change).
  - Standardize them **within each room** using that room's unlabelled dev windows (or convert to within-room percentiles). This cancels the format difference: Room B rows are noisier for non-physical reasons.
  - Append them to the M1 embedding (weighted) or use them as a separate vote.
- **Quick diagnostic first (~10 min):** train on Room A's sit/stand/walk motion features, test on Room B's, 3-class. Above chance (0.33) means the signal exists.
- **Expected:** walk vs still may transfer. Sit vs stand probably will not.
- **Cost:** about ½ day. **Risk:** low. **Rules:** ✅ unlabelled data only.
- **Diagnostic result (2026-10-21, `scripts/task3_motion_diagnostic.py` → `outputs/task3/motion_diagnostic.json`): ❌ not viable. The motion ordering is *reversed* between the rooms.**

  | Mean temporal std (per-row normalized, dev windows) | sit | stand | walk |
  |---|---|---|---|
  | Room A | 0.025 | 0.050 | **0.229** (most motion, as expected) |
  | Room B | 0.999 | 0.952 | **0.572** (least motion) |
  | Room B after time smoothing, w = 15 | 0.229 | 0.223 | **0.196** (still least) |

  - **Transfer** (train on source sit/stand/walk, test on target, labels for scoring only): every variant is at or below chance:
    - 3-class 0.00–0.20 (chance 0.33);
    - walk vs still 0.01–0.25 (chance 0.5).
  - **Room-relative z-scores make it worse**, not better (walk vs still 0.01–0.04, i.e. near-perfectly inverted). The signal exists in both rooms, but with the opposite sign.
  - **Second pass (post hoc):** time-smoothing both rooms (w = 5, 15) narrows Room B's gap but never flips the order. So Room B's row noise is not the whole explanation; in this data, Room B "walk" windows really are steadier than its "sit"/"stand" windows.
  - **Consequence:** this also explains the systematic swaps in the U9 confusion matrices (B sit/stand → walk). No method can transfer a cue whose meaning is reversed between rooms without target labels for those classes. Ideas 2–4 cannot fix identity either: they can only rebalance or align distributions.

### 2. Balanced assignment (optimal transport / Sinkhorn)
- **Idea:** predictions currently collapse into one or two classes (e.g. everything → walk). Force the predicted class proportions on the unlabelled target data to be roughly balanced, so every class gets its share.
- **What we do:** compute similarities between unlabelled target dev embeddings and the class prototypes. Solve a Sinkhorn optimal-transport problem with a uniform class marginal, then turn the transport plan into refined prototypes or a calibrated decision rule. Apply it to target val.
- **Expected:** a direct boost to macro-F1 by stopping class collapse. It does not fix swapped identities on its own.
- **Assumption to state:** target classes are roughly equally frequent. That is a prior, not a label. It holds in our data by design, and the report must say so.
- **Cost:** about ½ day. **Risk:** medium. **Rules:** ✅ unlabelled target dev only.

### 3. Combine ideas 1 + 2
- **Idea:** motion features fix *which* cluster is walk; balanced assignment stops collapse. Together they target both failure modes.
- **What we do:** run idea 2 on the embedding extended with idea 1's features.
- **Cost:** small once 1 and 2 exist.

### 4. Domain-adversarial alignment (DANN) on frozen embeddings
- **Idea:** train a small network on top of the 128-d embeddings that classifies source activities well but **cannot tell the rooms apart**, so its features become room-invariant.
- **What we do:**
  - Small MLP + activity head, trained on source labels + support.
  - Domain head (room A vs B) behind a gradient-reversal layer, trained on unlabelled dev windows of both rooms.
  - Fixed epochs and λ, set in advance.
- **Expected:** standard fix for domain shift. It may fail when classes are swapped, because aligning the overall distributions does not force sit to align with sit.
- **Cost:** about 1 day. **Risk:** medium–high (training instability). **Rules:** ✅.

### 5. Report logistic regression (M5) more prominently
- **Idea:** M5 is already our best method (K = 20: 0.438 / 0.409 vs M1 0.335 / 0.336).
- **What we do:** nothing new. Discuss it in the report as the strongest ablation and explain why a discriminative classifier beats prototypes when classes overlap.
- **Rules:** ⚠️ it cannot replace D1, because it was chosen after seeing validation scores.

### 6. MAE encoder as the feature source
- **Idea:** Task 2 found the MAE encoder the most robust to format perturbations. It may transfer activities better.
- **What we do:** retrain one MAE encoder with Task 2's ablation config (~4 min), embed, and run M1 / M5 on it as a feature-source ablation.
- **Cost:** about 1 hour. **Rules:** ⚠️ ablation only (MAE was compared with activity-labelled probes in Task 2).

### 7. Short-slice averaging of embeddings
- **Idea:** in Task 1, averaging predictions over short slices of a window transferred much better to unseen activities (+13 points). The same may hold for activity embeddings.
- **What we do:** embed 25- or 50-row slices of each 150-row window, average the embeddings (or the slice predictions), then run M1.
- **Cost:** about ½ day. **Risk:** low–medium. **Rules:** ✅.

### 8. Support augmentation for small K
- **Idea:** at K = 1 the room-only prototypes rest on very few windows. Augment the support windows (crop, noise, subcarrier masking: the Task 2 augmentations) and embed several views per support capture.
- **What we do:** for each support capture, embed N augmented views and average them into the prototype.
- **Expected:** a more stable K = 1–5 room-only score. It does not help shared classes.
- **Cost:** about ½ day. **Risk:** low. **Rules:** ✅ (support labels only).

### 9. Retrain the encoder to ignore the room (root-cause fix)
- **Idea:** retrain the SSL encoder with a room-invariance objective, e.g. a domain-adversarial term, per-room instance normalization inside the network, or cross-room positive pairs.
- **What we do:** a new Task 2-style training run, a label-free selection rule, then a full Task 3 re-run.
- **Cost:** several days. **Risk:** high, not realistic before Oct 31. **Rules:** ✅ if selected label-free.
- **Use:** "future work" in the report.

---

## Suggested order
1. **Idea 1 diagnostic** (10 min). Decide whether motion features are worth building.
2. **Ideas 1 + 2 (+ 3)** as one post-hoc experiment, with settings fixed in advance (~1 day).
3. **Idea 6 (MAE)** if time allows (~1 hour).
4. Everything else → "future work" in the Task 3 report.
5. Then **U10** (report + final deliverables). The report includes whatever these attempts show, positive or negative.

## How to report any of these
- A new row in the ablation table (`ablations_all.csv`), produced by a committed script.
- Labelled **"post-hoc (designed after U6–U8)"**, with its pre-fixed settings listed.
- Compared against M1 (default) and M5 (best ablation) at every K, both directions, mean ± std over 10 draws.
