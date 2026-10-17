"""Room A: read Nexmon CSI from .pcap files in pure Python/NumPy.

Each packet is Ethernet + IPv4 + UDP (port 5500) + Nexmon payload:
18-byte header followed by ``nfft * 4`` bytes of CSI (one uint32 per subcarrier).
On the BCM4366c0 the uint32 holds Nexmon's packed floating-point format
(I and Q mantissas sharing one exponent), decoded by ``unpack_nexmon_float``.
Format reference: Nexmon CSI documentation (seemoo-lab/nexmon_csi).
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

NEXMON_HEADER_BYTES = 18
NEXMON_MAGIC = 0x1111

_PCAP_MAGICS = {
    0xA1B2C3D4: ("<", 1e-6),  # little endian, microseconds
    0xA1B23C4D: ("<", 1e-9),  # little endian, nanoseconds
    0xD4C3B2A1: (">", 1e-6),
    0x4D3CB2A1: (">", 1e-9),
}


@dataclass
class NexmonCapture:
    """Decoded packets of one capture (only packets with a complete CSI block)."""

    timestamps: np.ndarray  # (n,) float64 seconds
    csi: np.ndarray  # (n, nfft) complex64
    seq: np.ndarray  # (n,) uint16 raw sequence-control field
    chanspec: np.ndarray  # (n,) uint16
    core_stream: np.ndarray  # (n,) uint16
    n_dropped: int  # packets skipped (truncated / not Nexmon)


def read_pcap(path: str | Path) -> tuple[np.ndarray, list[bytes]]:
    """Return (timestamps, raw link-layer frames) of a classic libpcap file."""
    buf = Path(path).read_bytes()
    (magic,) = struct.unpack("<I", buf[:4])
    if magic not in _PCAP_MAGICS:
        raise ValueError(f"{path}: not a classic pcap file (magic {magic:#x})")
    endian, ts_unit = _PCAP_MAGICS[magic]
    rec = struct.Struct(endian + "IIII")
    off, ts, frames = 24, [], []
    while off + rec.size <= len(buf):
        sec, frac, incl, _orig = rec.unpack_from(buf, off)
        off += rec.size
        frames.append(buf[off : off + incl])
        ts.append(sec + frac * ts_unit)
        off += incl
    return np.asarray(ts, dtype=np.float64), frames


def _udp_payload(frame: bytes) -> bytes | None:
    """Strip Ethernet/IPv4/UDP headers; None if the frame is not IPv4/UDP."""
    if len(frame) < 14 + 20 + 8 or frame[12:14] != b"\x08\x00":
        return None
    ihl = (frame[14] & 0x0F) * 4
    if frame[14 + 9] != 17:  # protocol != UDP
        return None
    return frame[14 + ihl + 8 :]


def unpack_nexmon_float(
    raw: np.ndarray, nbits: int = 10, nman: int = 12, nexp: int = 6
) -> np.ndarray:
    """Decode Nexmon packed-float CSI words.

    Args:
        raw: (n_packets, nfft) uint32 words.
        nbits, nman, nexp: format parameters for BCM4358/BCM4366c0
            (10-bit autoscale target, 12-bit mantissas, 6-bit exponent).

    Returns:
        (n_packets, nfft) complex64 CSI. Autoscaling is per packet, so the
        largest component of every packet has its top bit at position ``nbits``.
    """
    h = raw.astype(np.int64)
    iq_mask = (1 << (nman - 1)) - 1
    e_mask = (1 << nexp) - 1
    e_p = 1 << (nexp - 1)
    sgnr_mask = 1 << (nexp + 2 * nman - 1)
    sgni_mask = sgnr_mask >> nman

    vi = (h >> (nexp + nman)) & iq_mask
    vq = (h >> nexp) & iq_mask
    e = h & e_mask
    e = np.where(e >= e_p, e - 2 * e_p, e)

    # Autoscale: position of the highest set bit of (vi | vq), added to the exponent.
    x = vi | vq
    top_bit = np.frexp(x.astype(np.float64))[1].astype(np.int64) - 1  # floor(log2 x) for x > 0
    maxbit = np.where(x > 0, e + top_bit, -e_p).max(axis=1, keepdims=True)
    shift = e + (nbits - maxbit)

    def scale(v: np.ndarray) -> np.ndarray:
        left = v << np.clip(shift, 0, 62)
        right = v >> np.clip(-shift, 0, 62)
        out = np.where(shift >= 0, left, right)
        return np.where(shift < -nman, 0, out)

    re = scale(vi) * np.where(h & sgnr_mask, -1, 1)
    im = scale(vq) * np.where(h & sgni_mask, -1, 1)
    return (re + 1j * im).astype(np.complex64)


def load_nexmon_pcap(path: str | Path, nfft: int = 256) -> NexmonCapture:
    """Read one .pcap and decode every complete Nexmon CSI packet."""
    ts, frames = read_pcap(path)
    csi_bytes = nfft * 4
    keep_ts, words, seq, chan, core = [], [], [], [], []
    dropped = 0
    for t, frame in zip(ts, frames):
        p = _udp_payload(frame)
        if p is None or len(p) < NEXMON_HEADER_BYTES + csi_bytes:
            dropped += 1
            continue
        magic, = struct.unpack_from("<H", p, 0)
        if magic != NEXMON_MAGIC:
            dropped += 1
            continue
        s, cs, ch = struct.unpack_from("<HHH", p, 10)
        keep_ts.append(t)
        seq.append(s)
        core.append(cs)
        chan.append(ch)
        words.append(np.frombuffer(p, dtype="<u4", count=nfft, offset=NEXMON_HEADER_BYTES))
    raw = np.stack(words) if words else np.zeros((0, nfft), dtype=np.uint32)
    return NexmonCapture(
        timestamps=np.asarray(keep_ts, dtype=np.float64),
        csi=unpack_nexmon_float(raw) if len(raw) else np.zeros((0, nfft), np.complex64),
        seq=np.asarray(seq, dtype=np.uint16),
        chanspec=np.asarray(chan, dtype=np.uint16),
        core_stream=np.asarray(core, dtype=np.uint16),
        n_dropped=dropped,
    )
