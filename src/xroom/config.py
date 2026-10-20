"""Load the YAML configuration and resolve paths against the project root."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "task1.yaml"


TASK2_CONFIG = PROJECT_ROOT / "configs" / "task2.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Read the config; every entry under ``paths`` becomes an absolute ``Path``.

    A config with ``base_config`` is layered on top of that file: its sections are added,
    and its ``paths`` entries are added to the base paths (base sections are not edited)."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    base_name = cfg.pop("base_config", None)
    if base_name is not None:
        base = load_config(path.parent / base_name)
        clash = (set(cfg) - {"paths"}) & set(base)
        if clash:
            raise ValueError(f"{path.name} must not redefine base sections: {sorted(clash)}")
        paths = base.pop("paths")
        paths.update({k: (PROJECT_ROOT / v).resolve() for k, v in cfg.pop("paths", {}).items()})
        return {**base, **cfg, "paths": paths}
    cfg["paths"] = {k: (PROJECT_ROOT / v).resolve() for k, v in cfg["paths"].items()}
    return cfg


def load_task2_config(path: str | Path = TASK2_CONFIG) -> dict[str, Any]:
    """Task 1 data settings + Task 2 sections (``configs/task2.yaml``)."""
    return load_config(path)


def apply_overrides(cfg: dict[str, Any], items: list[str] | None) -> dict[str, Any]:
    """Return a copy of ``cfg`` with ``"section.key=value"`` overrides (value parsed as YAML;
    ``null`` removes the key, e.g. ``augment.format=null`` switches that augmentation off)."""
    out = copy.deepcopy(cfg)
    for item in items or []:
        key, _, raw = item.partition("=")
        *parents, leaf = key.split(".")
        node = out
        for p in parents:
            node = node.setdefault(p, {})
        value = yaml.safe_load(raw)
        if value is None:
            node.pop(leaf, None)
        else:
            node[leaf] = value
    return out
