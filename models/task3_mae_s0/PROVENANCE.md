# Task 3 post-hoc MAE encoder, seed 0 (label-free)

- Source run: `t3_mae_s0` (mae, seed 0), trained at git `e6e198f9a2a7142c2aba2e508c8f5ca908d3e9ba-dirty` with
  `python scripts/task2_pretrain.py --run t3_mae_s0 --seed 0 --method mae`
  ("-dirty" = uncommitted config edits at the time; the training code was committed).
- Published at git `793abc6c9a82b241a2ab2c4cad79ef65f55d9eb7` by `scripts/task3_publish_mae.py`.
- Settings: the Task 2 defaults (`configs/task2.yaml`) with `ssl.method = mae`; the seed list was
  fixed in `configs/task3_posthoc.yaml` before training, and no seed was chosen by a score.
- Training data: development captures of both rooms with `dev_fold == "train"`
  (1603 captures); no validation capture; no activity labels.
- Normalization statistics: fitted on development rows only (`norm_stats.npz`).
- Checkpoint selection: lowest SSL loss on dev-holdout crops (label-free); epoch 59.
- Use: one member of the post-hoc ensemble `ens_mae5` (M11). Not the Task 3 default, which stays
  the pre-registered M1 on `models/task2_default`.
