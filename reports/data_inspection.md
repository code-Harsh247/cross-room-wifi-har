# Data Inspection Report (Task 1, Milestone M1)

**Date:** 2026-10-18
**Scope:** What is actually in the two data folders, and what it means for Task 1.
**Status:** Findings from exploratory scripts. The decoder will be turned into proper `src/` code in M2.

---

## TL;DR

1. **Room A (PCAP) decodes cleanly in Python.** Each capture is about 220 packets over about 3.9 s. Each packet carries 256 complex CSI values in Nexmon's packed-float format.
2. **About 71% of Room A packets are not full-band 80 MHz frames.** They only have energy in one 20 MHz slice (64 of the 256 subcarriers). Only about 59 full-band packets exist per capture, and **no capture has 150 of them.**
3. **Room B (CSV) is almost entirely full-band rows,** with a different amplitude scale and much rougher spectra.
4. ⚠️ So if we window the raw packets naively, a room classifier can reach ~100% by detecting **frame type and file format, not the room.** Fixing this is the main preprocessing decision for Task 1 (see §6).
5. Room B files with consecutive numbers are **consecutive chunks of longer recordings.**

---

## 1. Folder layout (as delivered)

```
csidata/            Room A (Experiment 2), .pcap
  fall/ sit/ sitdown/ stand/ standup/ walk/
  validation/<same 6 classes>/   (25 per class)
csvdata/            Room B (Experiment 3), .csv (flat folder)
```

This differs slightly from the slides (`Experiment-2/realdata/...`, `Experiment-3/Data/`), but the content matches. Junk files to ignore: `.DS_Store` (8×) and `csvdata/model_plot_1.png`.

## 2. Inventory

### Room A
| Class | Dev captures | Validation captures |
|---|---|---|
| fall | 177 | 25 |
| sit | 178 | 25 |
| sitdown | 155 | 25 |
| stand | 115 (+1 exact duplicate) | 25 |
| standup | 155 | 25 |
| walk | 154 | 25 |
| **Total** | **934 unique** | **150** |

- `csidata/stand/stand65(1).pcap` is byte-identical to `stand65.pcap`. **Drop it.**
- No exact duplicates between dev and validation.
- Recording timestamps span 2021-01-10 to 2021-02-04.

### Room B
| Class | Files | Rows per file |
|---|---|---|
| Empty | 267 | 500 |
| Lying | 351 | 500 |
| Sitting | 153 | 500 |
| Standing | 221 | 500 |
| Walking | 209 | 500 |
| **Total** | **1201** | |

- Naming: `<Class>_Amp_<n>.csv`, except Walking (`Walking_<n>.csv`).
- There are 4 extra files `Empty_11.csv`, `Lying_0.csv`, `Sitting_0.csv`, `Standing_0.csv`. Their header is `0..255` instead of `Subcarrier_1..256`. They are **not** duplicates of the `_Amp_` files, so they look like additional captures.
- No duplicate files (byte-level or content-level).
- There is no validation folder. We create one (25 per class, seed 42).

## 3. Room A PCAP structure

- Classic pcap, Ethernet link type. Each packet is Ethernet (14 B) + IPv4 (20 B) + UDP (8 B, port 5500) + **Nexmon CSI payload (1042 B)**.
- Payload = 18-byte header + **1024 bytes of CSI = 256 subcarriers × 4 bytes**.
- Header fields are constant across the whole dataset:
  - one transmitter MAC
  - core/stream = 0 (a single antenna stream)
  - chanspec `0xe02a` (channel 42, 80 MHz, 5 GHz)
  - chip field `0x6a`
- 2 truncated packets (726 B payload) in the whole set. **Drop them.**
- Per capture: median 220 packets (range 112–251), duration ≈ 3.92 s. Inter-packet time is irregular, with a median of ~1.3 ms.
- **12 captures have fewer than 150 packets in total.**

### Decoding
The CSI values are in Nexmon's **packed floating-point format** for BCM4366c0 (12-bit mantissas for I and Q, plus a shared 6-bit exponent, autoscaled to 10 bits), as described in the Nexmon CSI documentation. I implemented it in NumPy, without using the authors' MATLAB code. Sanity checks:

- Amplitudes come out in the range 0–~1000, with each packet's peak near 2¹⁰ as expected.
- Spectra are **smooth across frequency**: neighbouring-subcarrier correlation is ≈ 0.96, which is what a real indoor channel looks like.
- Near-zero values at subcarriers ~123–133 are guard bands. Subcarriers are in **FFT order** (index 0 = DC, 128–255 = negative frequencies).

### ⚠️ Mixed frame types
Packets fall into two groups:

| Packet type | Share of all packets | What it looks like |
|---|---|---|
| **Full-band** (80 MHz) | 29% (68,810) | energy across all data subcarriers |
| **Narrow** (20 MHz only) | 71% (165,209) | energy only in subcarriers **128–191**, the primary 20 MHz channel; the other 192 subcarriers are ~0 |

- Half of all packets have sequence number `0xFFFF`. These are all narrow, which suggests control frames such as ACKs. The rest of the narrow packets are 20 MHz data frames. (The frame types are my inference; the band occupancy is measured.)
- Full-band packets per capture: **median 59, 5th percentile 32, max 129. No capture reaches 150.**
- The folder name `30ms` and the max of ~129 ≈ 3.9 s / 30 ms suggest one 80 MHz frame per 30 ms ping, with many arriving as 20 MHz or being lost.

## 4. Room B CSV structure

- 500 rows × 256 columns, **integer** amplitudes, one header row. No timestamps, so **the sampling rate is unknown.**
- Only 0.7% of rows look like 20 MHz-only frames. The authors' export seems to keep (almost) only full-band frames.
- Same FFT ordering: there is a dip at subcarriers ~123–133, though it is much less clean than in Room A.

## 5. Room A vs Room B: format differences a classifier could exploit

> **Correction (pipeline, M2):** the numbers below were made with a prototype decoder whose autoscaling was off by one bit. With the correct Nexmon autoscaling, Room A amplitudes are 2× larger, and the scale gap reverses (Room A > Room B). See `reports/preprocessing_pipeline.md` §5 for the audit on the final windows.

| Statistic (per file, median) | Room A (decoded, all packets) | Room B (CSV) |
|---|---|---|
| Mean amplitude | 226 | 354 |
| 99th percentile | 932 | 1359 |
| Fraction of exact zeros | 6.3% | 1.3% |
| Energy in guard band (relative) | 0.013 | 0.109 |
| Neighbour-subcarrier correlation (frequency smoothness) | **0.96** | **0.57–0.68** |
| Values are integers | no (floats) | yes |
| Narrow 20 MHz rows | 71% | 0.7% |

Every one of these can separate the rooms almost perfectly on its own, **without any room physics.** Room B's spectra are much rougher than a physical 80 MHz channel should be. This suggests the authors' CSV export used different processing. We only have Room B's amplitudes, so we can reduce these gaps but **cannot fully remove them.** This must be stated as a limitation in the report.

## 6. Room B: files are chunks of longer recordings

Compare the last 50 rows of file *n* with the first 50 rows of file *n+1*. They are **as similar as two parts of the same file**, and much more similar than a random file of the same class:

| Class | tail → next file | tail → own head | tail → random file |
|---|---|---|---|
| Empty | 0.965 | 0.949 | 0.605 |
| Lying | 0.931 | 0.853 | 0.448 |
| Walking | 0.903 | 0.836 | 0.504 |

So a random 25-per-class validation draw (as the rules require) will place validation files next to their neighbours in development. This makes Room B scores optimistic, especially for the activity tasks. We follow the rule (seed 42) and report this as a caveat. Optionally, we could also report a "contiguous block" split as a robustness check.

## 7. Decisions needed for Task 1 (proposed)

| # | Decision | Proposal |
|---|---|---|
| D1 | Room A packets | **Keep only full-band 80 MHz packets.** Drop narrow 20 MHz frames and truncated packets. This matches Room B's content and is a legitimate validity check (the slides mention "missing/zero CSI"). |
| D2 | Room A window length (only ~59 valid packets per capture) | **Resample each capture's valid packets to 150 rows** (linear interpolation over time), giving one 150×256 window per capture. Alternatives: zero-pad, or use shorter native windows. |
| D3 | Room B windows | Non-overlapping 150-row windows → 3 per file (rows 0–149, 150–299, 300–449). Allow overlap only in training. |
| D4 | Scale and roughness gap | Main model: per-row or per-window normalization (removes overall scale). Ablations: frequency smoothing applied to both rooms, and log amplitude. Always run the "trivial-statistic" leak check (§5) on the final features. |
| D5 | Guard/DC subcarriers | Mask the same fixed set (DC + ~123–133) in both rooms, since they carry no room information but do carry format differences. |
| D6 | Extra Room B files (`*_0.csv`, `Empty_11.csv`) | Treat as normal captures of their class. |
| D7 | Duplicate `stand65(1).pcap` | Drop. |

## 8. Updates to the PRD's open questions

| PRD §12 question | Answer |
|---|---|
| Sampling interval | Room A: ~3.9 s per capture, ~15 valid frames/s after filtering. Room B: unknown, no timestamps. |
| Room B layout | Rows = frames, columns = 256 subcarriers, one header row, no index column. |
| One capture in Room B | Each CSV = one capture (500 rows), but consecutive files are contiguous chunks of the same recording. |
| Amplitude scale | Different between rooms (§5). Needs normalization. |
| Null subcarriers | Guard band/DC around 123–133 in FFT order. Room A also has ~6% exact zeros, mostly from narrow frames. |
