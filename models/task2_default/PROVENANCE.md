# Default Task 2 encoder (label-clean)

- Source run: `simclr_s0` (simclr, seed 0), trained at git `b3ae6474f1ea2347b455090a00218013ac574b90`. This ID is not in the published history: commits made before the first push were rewritten (author e-mail only), so the same code has a different ID; match it by commit subject and milestone in `docs/tasks_task2.md`.
- Published at git `1eea34e206070401a8ca8393309d4537ad0dcf8e` by `scripts/task2_publish_default.py`.
- Settings: the default `configs/task2.yaml`, fixed before any probe was run and not changed
  afterwards. The seed was fixed in advance (0) and not chosen by any score.
- Training data: development captures of both rooms with `dev_fold == "train"`
  (1603 captures); no validation capture; no activity labels.
  Room identity was used only to balance batches 50/50.
- Normalization statistics: fitted on development rows only (`norm_stats.npz`).
- Checkpoint selection: lowest SSL loss on dev-holdout crops (label-free); epoch 54.
- **No activity labels were used for training or for selecting this encoder or its settings.**
  Variants compared with activity-labelled probes (`outputs/task2/abl_*`, not committed) are ablations and
  must not be used as the Task 3 default.
