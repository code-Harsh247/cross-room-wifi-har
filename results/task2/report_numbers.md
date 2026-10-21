## Main probe results (validation macro-F1)

| Features | Room A (6 cl.) | Room B (5 cl.) | Joint (8 cl.) | Cross A→B (3 cl.) | Cross B→A (3 cl.) | Headline (mean of rooms) |
|---|---|---|---|---|---|---|
| SimCLR (3 seeds) | 0.905 ± 0.027 | 0.992 ± 0.005 | 0.944 ± 0.011 | 0.226 ± 0.033 | 0.228 ± 0.112 | 0.949 ± 0.016 |
| Random-init encoder (3 seeds) | 0.935 ± 0.011 | 0.856 ± 0.013 | 0.847 ± 0.013 | 0.167 ± 0.000 | 0.226 ± 0.063 | 0.895 ± 0.011 |
| Hand-crafted mean+std (484-d) | 0.940 [0.90, 0.97] | 0.989 [0.98, 1.00] | 0.954 [0.93, 0.97] | 0.167 [0.17, 0.17] | 0.168 [0.17, 0.17] | 0.965 |
| PCA-128 | 0.822 [0.76, 0.88] | 0.928 [0.90, 0.95] | 0.814 [0.78, 0.85] | 0.390 [0.32, 0.46] | 0.243 [0.17, 0.32] | 0.875 |

## SimCLR per seed (validation macro-F1 [95% capture bootstrap])

| Run | Room A | Room B | Joint | Capture-level A / B | Cross A→B / B→A | Room probe acc. | Format shift A / B |
|---|---|---|---|---|---|---|---|
| simclr_s0 | 0.905 [0.86, 0.95] | 0.995 [0.99, 1.00] | 0.936 [0.91, 0.96] | 0.905 / 1.000 | 0.262 / 0.125 | 1.000 | 0.810 / 0.965 |
| simclr_s1 | 0.932 [0.89, 0.97] | 0.995 [0.99, 1.00] | 0.957 [0.93, 0.98] | 0.932 / 1.000 | 0.199 / 0.211 | 1.000 | 0.799 / 0.933 |
| simclr_s2 | 0.878 [0.82, 0.93] | 0.987 [0.97, 1.00] | 0.939 [0.91, 0.96] | 0.878 / 0.992 | 0.217 / 0.347 | 0.996 | 0.865 / 0.914 |

## Dev-holdout per-room macro-F1 per run (probe fit on dev-train; development data only)

- simclr_s0: A 0.913, B 0.985
- simclr_s1: A 0.928, B 0.991
- simclr_s2: A 0.935, B 0.989
- random_init_s0: A 0.951, B 0.804
- random_init_s1: A 0.934, B 0.782
- random_init_s2: A 0.948, B 0.833
- handcrafted: A 0.939, B 0.986
- pca: A 0.811, B 0.892

## Per-class F1 and Room A confusion, SimCLR seed 0 (validation)

- room_A: fall 0.94, sit 0.87, sitdown 0.83, stand 0.92, standup 0.92, walk 0.94
- room_B: empty 0.99, lying 0.99, sit 1.00, stand 0.99, walk 1.00
- room_A confusion, labels ['fall', 'sit', 'sitdown', 'stand', 'standup', 'walk'] (rows true, cols predicted): [[23, 0, 0, 2, 0, 0], [1, 20, 2, 0, 0, 1], [0, 1, 20, 1, 2, 1], [0, 0, 0, 24, 1, 0], [0, 0, 1, 0, 24, 0], [0, 1, 0, 0, 0, 24]]

## Controls: room probe and format shift (validation)

| Features | Room probe acc. | Activity A clean → shifted | Activity B clean → shifted |
|---|---|---|---|
| SimCLR (3 seeds) | 0.999 ± 0.002 | 0.905 → 0.825 | 0.992 → 0.937 |
| Random-init (3 seeds) | 1.000 ± 0.000 | 0.935 → 0.913 | 0.856 → 0.208 |
| Hand-crafted | 1.000 | 0.940 → 0.732 | 0.989 → 0.260 |

## Shuffled-label control (validation macro-F1)

- handcrafted: A 0.182, B 0.181, joint 0.100
- pca: A 0.178, B 0.125, joint 0.133
- random_init_s0: A 0.077, B 0.163, joint 0.133
- random_init_s1: A 0.085, B 0.172, joint 0.102
- random_init_s2: A 0.133, B 0.152, joint 0.095

## SSL training (default config)

- simclr_s0: train loss 3.488 → 1.961; holdout 2.915 → best 2.111 (epoch 54); eff. rank 69.2; emb std 0.459; 218 s; 1603 train / 401 holdout captures
- simclr_s1: train loss 3.522 → 1.949; holdout 2.893 → best 2.107 (epoch 58); eff. rank 69.2; emb std 0.456; 347 s; 1603 train / 401 holdout captures
- simclr_s2: train loss 3.574 → 1.958; holdout 2.913 → best 2.120 (epoch 57); eff. rank 67.7; emb std 0.473; 324 s; 1603 train / 401 holdout captures

## Ablations (mean ± std over seeds; dev-holdout = selection score, val = reported)

| Variant | n | Dev-holdout A | Dev-holdout B | Val A | Val B | Val cross A→B | Val cross B→A |
|---|---|---|---|---|---|---|---|
| default | 2 | 0.921 ± 0.011 | 0.988 ± 0.004 | 0.918 ± 0.019 | 0.995 ± 0.000 | 0.231 ± 0.045 | 0.168 ± 0.061 |
| aug_no_format | 2 | 0.919 ± 0.000 | 0.989 ± 0.001 | 0.905 ± 0.011 | 0.992 ± 0.007 | 0.140 ± 0.018 | 0.205 ± 0.089 |
| aug_no_subcarrier_mask | 2 | 0.925 ± 0.001 | 0.991 ± 0.003 | 0.953 ± 0.000 | 0.995 ± 0.004 | 0.094 ± 0.054 | 0.266 ± 0.139 |
| aug_no_crop_resize | 2 | 0.918 ± 0.015 | 0.986 ± 0.002 | 0.912 ± 0.009 | 0.992 ± 0.004 | 0.147 ± 0.007 | 0.205 ± 0.017 |
| aug_crop_only | 2 | 0.912 ± 0.002 | 0.977 ± 0.005 | 0.946 ± 0.001 | 0.989 ± 0.000 | 0.179 ± 0.074 | 0.334 ± 0.156 |
| norm_standardize_only | 2 | 0.870 ± 0.002 | 0.987 ± 0.003 | 0.871 ± 0.002 | 0.981 ± 0.011 | 0.191 ± 0.025 | 0.358 ± 0.023 |
| norm_row_log | 2 | 0.915 ± 0.012 | 0.988 ± 0.001 | 0.935 ± 0.014 | 0.989 ± 0.004 | 0.217 ± 0.011 | 0.231 ± 0.056 |
| method_mae | 2 | 0.956 ± 0.004 | 0.995 ± 0.002 | 0.956 ± 0.005 | 0.996 ± 0.006 | 0.208 ± 0.023 | 0.238 ± 0.053 |
| embed_64 | 2 | 0.897 ± 0.007 | 0.982 ± 0.004 | 0.914 ± 0.014 | 0.993 ± 0.002 | 0.293 ± 0.058 | 0.185 ± 0.026 |
| embed_256 | 2 | 0.951 ± 0.006 | 0.988 ± 0.005 | 0.916 ± 0.015 | 0.995 ± 0.000 | 0.186 ± 0.088 | 0.232 ± 0.043 |
| crop_150_fixed | 2 | 0.929 ± 0.001 | 0.990 ± 0.005 | 0.925 ± 0.028 | 0.996 ± 0.002 | 0.180 ± 0.009 | 0.274 ± 0.026 |
| crop_64_150 | 2 | 0.899 ± 0.002 | 0.987 ± 0.005 | 0.914 ± 0.004 | 0.987 ± 0.007 | 0.194 ± 0.027 | 0.317 ± 0.075 |
| tau_0.1 | 2 | 0.899 ± 0.004 | 0.979 ± 0.005 | 0.909 ± 0.034 | 0.987 ± 0.004 | 0.138 ± 0.085 | 0.223 ± 0.027 |
| tau_0.5 | 2 | 0.960 ± 0.003 | 0.997 ± 0.000 | 0.922 ± 0.005 | 1.000 ± 0.000 | 0.311 ± 0.179 | 0.288 ± 0.048 |

## Top variants with 3 seeds (dev-holdout chooses; validation reports)

| Variant | Seeds | Dev-holdout A | Dev-holdout B | Dev-holdout mean | Val A | Val B | Val cross A→B / B→A | Format shift A / B | Room probe |
|---|---|---|---|---|---|---|---|---|---|
| default | 3 (dev-holdout: 2) | 0.921 | 0.988 | 0.9543 | 0.905 ± 0.027 | 0.992 ± 0.005 | 0.226 / 0.228 | 0.825 / 0.937 | 0.999 |
|  | val A per seed: 0.905, 0.932, 0.878 | | | | | | | | |
| tau_0.5 | 3 (dev-holdout: 3) | 0.959 | 0.996 | 0.9776 | 0.923 ± 0.004 | 0.998 ± 0.003 | 0.270 / 0.265 | 0.877 / 0.971 | 0.999 |
|  | val A per seed: 0.926, 0.919, 0.926 | | | | | | | | |
| method_mae | 3 (dev-holdout: 3) | 0.958 | 0.996 | 0.9768 | 0.957 ± 0.004 | 0.993 ± 0.007 | 0.257 / 0.288 | 0.955 / 0.961 | 1.000 |
|  | val A per seed: 0.960, 0.952, 0.959 | | | | | | | | |
| aug_no_format | 2 (dev-holdout: 2) | 0.919 | 0.989 | 0.9540 | 0.905 ± 0.011 | 0.992 ± 0.008 | 0.140 / 0.205 | 0.868 / 0.919 | 0.999 |
|  | val A per seed: 0.898, 0.913 | | | | | | | | |

## Contiguous Room B split (Room B macro-F1 [95% capture bootstrap])

- contiguous_B: handcrafted (run handcrafted): 0.943 [0.91, 0.97]
- contiguous_B: SSL (block held out of SSL) (run contig_s0): 0.987 [0.97, 1.00]
- contiguous_B: SSL default (block seen unlabelled) (run simclr_s0): 0.992 [0.98, 1.00]
- contiguous_B: SSL (block held out of SSL) (run contig_s1): 0.973 [0.95, 0.99]
- contiguous_B: SSL default (block seen unlabelled) (run simclr_s1): 0.992 [0.98, 1.00]
