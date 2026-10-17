"""List every capture of both rooms with its activity label and provided split."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

# Room B file prefixes -> shared activity names (sit/stand/walk match Room A folder names).
ROOM_B_ACTIVITIES = {
    "Empty": "empty",
    "Lying": "lying",
    "Sitting": "sit",
    "Standing": "stand",
    "Walking": "walk",
}
_ROOM_B_FILE = re.compile(r"^(Empty|Lying|Sitting|Standing|Walking)_(?:Amp_)?(\d+)\.csv$")


def natural_key(s: str) -> list:
    """Sort key so that 'x2' < 'x10'."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def catalog_room_a(root: Path, exclude: list[str]) -> pd.DataFrame:
    """One row per .pcap; ``provided_split`` is 'val' for files under validation/."""
    excluded = {Path(e).as_posix() for e in exclude}
    rows = []
    for path in root.rglob("*.pcap"):
        rel = path.relative_to(root).as_posix()
        if rel in excluded:
            continue
        parts = rel.split("/")
        is_val = parts[0] == "validation"
        activity = parts[1] if is_val else parts[0]
        rows.append(
            {
                "capture_id": "A/" + rel.removesuffix(".pcap"),
                "room": "A",
                "activity": activity,
                "path": str(path),
                "provided_split": "val" if is_val else None,
            }
        )
    return _sorted(pd.DataFrame(rows))


def catalog_room_b(root: Path) -> pd.DataFrame:
    """One row per Room B .csv (non-matching files such as .png are ignored)."""
    rows = []
    for path in root.glob("*.csv"):
        m = _ROOM_B_FILE.match(path.name)
        if not m:
            continue
        rows.append(
            {
                "capture_id": "B/" + path.stem,
                "room": "B",
                "activity": ROOM_B_ACTIVITIES[m.group(1)],
                "path": str(path),
                "provided_split": None,
            }
        )
    return _sorted(pd.DataFrame(rows))


def build_catalog(cfg: dict) -> pd.DataFrame:
    a = catalog_room_a(cfg["paths"]["room_a_dir"], cfg["room_a"]["exclude_files"])
    b = catalog_room_b(cfg["paths"]["room_b_dir"])
    return pd.concat([a, b], ignore_index=True)


def _sorted(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values("capture_id", key=lambda s: s.map(natural_key)).reset_index(drop=True)
