"""Unit tests for decoding, cleaning, resampling, windowing and splitting."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from xroom.catalog import build_catalog
from xroom.clean import fullband_mask, zero_null_subcarriers
from xroom.config import PROJECT_ROOT, load_config
from xroom.io_pcap import load_nexmon_pcap, unpack_nexmon_float
from xroom.resample import resample_in_time
from xroom.splits import assign_splits, draw_room_b_validation, room_b_validation
from xroom.windows import room_b_windows

CFG = load_config()
HAVE_DATA = CFG["paths"]["room_a_dir"].exists() and CFG["paths"]["room_b_dir"].exists()
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="raw dataset not present")


def pack_word(vi: int, vq: int, e: int, neg_i: bool = False, neg_q: bool = False, nman=12, nexp=6) -> int:
    """Inverse of the Nexmon packed-float layout, for one subcarrier."""
    w = (vi << (nexp + nman)) | (vq << nexp) | (e & ((1 << nexp) - 1))
    if neg_i:
        w |= 1 << (nexp + 2 * nman - 1)
    if neg_q:
        w |= 1 << (nexp + nman - 1)
    return w


# ---------- decoding ----------

def test_unpack_autoscale_and_signs():
    # x = 5|3 = 7 -> top bit 2 -> every value shifted left by 10 - 2 = 8
    raw = np.array([[pack_word(5, 3, 0), pack_word(1, 2, 0, neg_i=True, neg_q=True)]], dtype=np.uint32)
    csi = unpack_nexmon_float(raw)
    assert csi[0, 0] == complex(5 << 8, 3 << 8)
    assert csi[0, 1] == complex(-(1 << 8), -(2 << 8))


def test_unpack_exponent_relative_scaling():
    # second subcarrier has exponent -3 -> 8x smaller than the first, same mantissa
    raw = np.array([[pack_word(4, 0, 0), pack_word(4, 0, -3)]], dtype=np.uint32)
    a = np.abs(unpack_nexmon_float(raw))[0]
    assert a[0] == 1 << 10 and a[1] == a[0] / 8


def test_unpack_autoscale_is_per_packet():
    raw = np.array([[pack_word(1, 0, 0)], [pack_word(1, 0, 5)]], dtype=np.uint32)
    a = np.abs(unpack_nexmon_float(raw))
    assert a[0, 0] == a[1, 0] == 1 << 10


@needs_data
def test_real_pcap_decodes_to_smooth_80mhz_csi():
    cap = load_nexmon_pcap(CFG["paths"]["room_a_dir"] / "sit" / "sit1.pcap")
    assert cap.csi.shape[1] == 256 and len(cap.csi) > 150
    assert set(np.unique(cap.chanspec)) == {0xE02A}  # channel 42, 80 MHz
    amp = np.abs(cap.csi)
    full = fullband_mask(amp, tuple(CFG["room_a"]["primary_band"]), CFG["room_a"]["narrow_energy_frac"])
    assert 0 < full.sum() < len(amp)  # both frame types present
    data_sc = np.r_[6:123, 134:250]
    x = amp[full][:, data_sc]
    a, b = x[:, :-1] - x.mean(1, keepdims=True), x[:, 1:] - x.mean(1, keepdims=True)
    corr = (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1))
    assert np.median(corr) > 0.9  # a real channel is smooth across frequency


# ---------- cleaning ----------

def test_fullband_mask_flags_narrow_and_empty_rows():
    wide = np.ones((1, 256))
    narrow = np.zeros((1, 256))
    narrow[0, 128:192] = 5
    empty = np.zeros((1, 256))
    m = fullband_mask(np.vstack([wide, narrow, empty]), (128, 192), 0.9)
    assert m.tolist() == [True, False, False]


def test_zero_null_subcarriers_copies():
    x = np.ones((2, 3, 256), np.float32)
    y = zero_null_subcarriers(x, CFG["common"]["null_subcarriers"])
    assert x.sum() == 2 * 3 * 256
    assert (y[..., CFG["common"]["null_subcarriers"]] == 0).all()
    assert y.sum() == 2 * 3 * (256 - len(CFG["common"]["null_subcarriers"]))


# ---------- resampling ----------

def test_resample_reproduces_linear_signal():
    t = np.array([0.0, 0.1, 0.35, 0.4, 1.0])
    x = np.stack([2 * t + 1, -t], axis=1)
    y = resample_in_time(t, x, 11)
    grid = np.linspace(0, 1, 11)
    np.testing.assert_allclose(y, np.stack([2 * grid + 1, -grid], 1), atol=1e-6)


def test_resample_handles_duplicate_timestamps():
    t = np.array([0.0, 0.5, 0.5, 1.0])
    x = np.array([[0.0], [1.0], [3.0], [4.0]])
    y = resample_in_time(t, x, 3)
    np.testing.assert_allclose(y[:, 0], [0.0, 2.0, 4.0])


# ---------- windowing ----------

def test_room_b_windows_do_not_cross_capture(tmp_path: Path):
    rows = np.arange(500 * 256, dtype=np.float32).reshape(500, 256)
    f = tmp_path / "Empty_Amp_1.csv"
    pd.DataFrame(rows, columns=[f"Subcarrier_{i + 1}" for i in range(256)]).to_csv(f, index=False)
    cw = room_b_windows(str(f), CFG, length=150, stride=150)
    assert cw.starts == [0, 150, 300]
    assert cw.windows.shape == (3, 150, 256)
    assert cw.windows[2, -1, 2] == rows[449, 2]


# ---------- splits ----------

def _fake_catalog_b(n_per_class: int = 40) -> pd.DataFrame:
    rows = [
        {"capture_id": f"B/{act}_Amp_{i}", "room": "B", "activity": act, "path": "", "provided_split": None}
        for act in ["empty", "lying", "sit"]
        for i in range(1, n_per_class + 1)
    ]
    return pd.DataFrame(rows)


def test_room_b_validation_is_deterministic_and_balanced(tmp_path: Path):
    cat = _fake_catalog_b()
    a = draw_room_b_validation(cat, 25, 42)
    assert a == draw_room_b_validation(cat, 25, 42)
    assert pd.Series([c.split("_")[0] for c in a]).value_counts().eq(25).all()
    manifest = tmp_path / "val.json"
    assert room_b_validation(cat, manifest, 25, 42) == a
    assert room_b_validation(cat, manifest, 25, 42) == a  # reloaded, not redrawn
    with pytest.raises(ValueError):
        room_b_validation(cat, manifest, 25, 7)


@needs_data
def test_real_splits_are_disjoint_and_complete():
    cat = assign_splits(build_catalog(CFG), CFG)
    assert cat["capture_id"].is_unique
    assert not cat["capture_id"].str.contains(r"\(1\)").any()  # duplicate pcap excluded
    val = cat[cat["split"] == "val"]
    assert val.groupby(["room", "activity"]).size().eq(25).all()
    assert val["dev_fold"].isna().all()
    assert set(cat.loc[cat["split"] == "dev", "dev_fold"]) == {"train", "holdout"}
    manifest = PROJECT_ROOT / "splits" / "room_b_validation.json"
    assert manifest.exists()
