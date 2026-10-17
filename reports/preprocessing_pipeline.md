# Preprocessing Pipeline (Milestones M2–M4)

**Date:** 2026-10-18
**Code:** `src/xroom/` · **Config:** `configs/task1.yaml` · **Entry point:** `python scripts/build_windows.py`

---

## 1. Pipeline at a glance

```
Room A  .pcap ──► parse Ethernet/IP/UDP ──► Nexmon header + 256 × uint32
                 ──► unpack packed-float CSI (complex) ──► |CSI| amplitude
                 ──► keep 80 MHz packets only (drop 20 MHz frames)
                 ──► resample capture (~3.9 s) onto 150 evenly spaced rows
                 ──► zero DC/guard subcarriers ──► 1 window (150 × 256) per capture

Room B  .csv  ──► read 500 × 256 integer amplitude
                 ──► zero DC/guard subcarriers
                 ──► non-overlapping 150-row windows ──► 3 windows (150 × 256) per capture

Both    ──► capture-level splits (val / dev, dev → train / holdout)
        ──► data_cache/windows_L150.npy (float16) + windows_L150_meta.csv
```

| Module | Job |
|---|---|
| `io_pcap.py` | pcap reader, Nexmon payload parsing, packed-float decoder |
| `io_csv.py` | Room B CSV reader with shape and NaN checks |
| `clean.py` | full-band (80 MHz) packet detection, DC/guard masking |
| `resample.py` | time-based linear resampling (averages duplicate timestamps) |
| `windows.py` | per-capture windowing for each room |
| `catalog.py` | lists captures and maps labels to shared names (`sit`, `stand`, `walk`, ...) |
| `splits.py` | validation manifest (Room B) and dev train/holdout folds |
| `dataset.py` | builds and loads the cache |

## 2. Choices and why

| Step | Choice | Why |
|---|---|---|
| Decoding | Nexmon packed-float (12-bit mantissa, 6-bit exponent, 10-bit autoscale), written from the Nexmon CSI format docs | The BCM4366c0 does not store plain int16 I/Q. Decoded spectra are smooth across frequency (lag-1 corr ≈ 0.99), as a real channel should be. |
| Packet filter | Drop packets with ≥ 90% of their energy in subcarriers 128–191 | These are 20 MHz frames (ACKs etc.) with no CSI on the other 192 subcarriers. Room B contains almost none, so keeping them would be a free room cue. |
| Too few packets | Drop captures with < 20 full-band packets | You cannot reconstruct 4 s of motion from 1–15 frames. |
| Room A length | Linear resampling in **time** of the remaining ~59 packets onto 150 rows | Gives the required 150 × 256 shape and keeps the real timing (packets arrive irregularly). |
| Room B windows | Non-overlapping, stride 150 → rows 0–149, 150–299, 300–449 (last 50 rows unused) | Deterministic; windows never cross captures. Overlap can be added for training later via `--stride-b`. |
| Null subcarriers | Zero indices 0, 1, 255 (DC) and 123–133 (guard bands) in both rooms | They carry no channel information, only format differences. All 256 columns are kept, as the spec requires. |
| Storage | float16 on disk | 4,680 × 150 × 256 windows = 344 MB instead of 690 MB. Precision ≈ 0.1% of the value, far below the signal noise. |

## 3. Splits

- **Room A validation:** the supplied `validation/` folder (25 per class).
- **Room B validation:** 25 per class drawn with `numpy.random.default_rng(42)` from naturally sorted capture IDs, frozen in `splits/room_b_validation.json`. The code refuses to silently redraw it.
- **Dev folds** (for model selection, never for scoring): 20% of dev captures per (room, activity) → `holdout`, the rest → `train`, seed 0.
- Every split is by capture. All windows of a capture share its `split` and `dev_fold`.

## 4. Output (window length 150)

| Room | Split | Captures | Windows |
|---|---|---|---|
| A | dev | 928 | 928 |
| A | val | 149 | 149 |
| B | dev | 1,076 | 3,228 |
| B | val | 125 | 375 |
| **Total** | | **2,278** | **4,680** |

Per class, Room A has one window per capture. Room B has 3 per capture.

**Dropped:**
- 1 duplicate file (`stand65(1).pcap`)
- 2 truncated packets
- 7 Room A `sit` captures with < 20 full-band packets: `sit5`, `sit71`, `sit117`, `sit118`, `sit126`, `sit138`, and validation `sit201`. So Room A validation has **24** sit captures instead of 25.

Room A full-band packets per capture: median 59, 5th percentile 32, max 129.

## 5. Format-leak audit (`python scripts/audit_format.py`)

How well a **single trivial statistic** separates the rooms on development windows (AUC: 0.5 = no information, 1.0 = perfect):

| Statistic | Room A | Room B | Separability |
|---|---|---|---|
| Mean amplitude | 1007 | 351 | 0.998 |
| 99th percentile | 1824 | 1370 | 0.935 |
| Exact-zero fraction | 0.000 | 0.002 | 1.000 |
| Integer-valued fraction | 0.69 | 1.00 | 1.000 |
| Frequency smoothness | 0.99 | 0.64 | 1.000 |
| Time smoothness | 0.90 | 0.04 | 1.000 |
| Variation over time (CV) | 0.18 | 0.71 | 1.000 |
| Peak / mean | 2.0 | 5.8 | 0.998 |

**What this means:** after filtering, the windows still differ in ways that are not room physics:
- **Scale:** Room A is autoscaled per packet by our decoder. Room B was exported with different scaling that we cannot see.
- **Noise and roughness:** Room B rows are much noisier, both across frequency and from row to row.
- **Smoothing from resampling:** interpolating ~59 packets up to 150 rows makes Room A smooth in time.

**Any room classifier on these windows can reach ~100% without learning anything about rooms.** This cannot be fully fixed, because we only have Room B's amplitudes and not its raw CSI. For Task 1 (M5) it means:

1. Normalize each window (or row) by its own mean, so absolute scale is gone.
2. Prefer **time-averaged spectral shape** features. Averaging over time removes the time-smoothness and per-row noise differences, leaving mostly the frequency profile.
3. Report the audit table next to the accuracy, and describe the accuracy as an upper bound inflated by format differences.
4. Ablation: compare feature sets by how much they rely on these cues (e.g. accuracy of a model trained only on the audit statistics).

## 6. Correction to the data inspection report

The inspection prototype used the bit length of the mantissa in autoscaling. The Nexmon format uses the position of the highest set bit, which is one less. The pipeline uses the correct version, so Room A amplitudes are **2× larger** than in `data_inspection.md` §5. Every conclusion there still holds.

## 7. How to run

```bash
python scripts/build_windows.py            # ~40 s, writes data_cache/
python scripts/audit_format.py             # format-leak table -> outputs/
python -m pytest -q                        # 11 tests
```

```python
from xroom.config import load_config
from xroom.dataset import load_windows
x, meta = load_windows(load_config())   # x: (4680, 150, 256) float16, meta: capture_id/room/activity/split/dev_fold/start
```
