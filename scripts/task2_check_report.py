"""Check the Task 2 report against the generated numbers (T11 gate).

Every number with two or more decimals in reports/task2_report_section.md must appear verbatim
in outputs/task2/report_numbers.md (run scripts/task2_report_numbers.py first), or in the small
list of config/spec constants below. Prints the numbers it could not match and exits 1 if any.

Usage:  python scripts/task2_check_report.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xroom.config import load_task2_config  # noqa: E402

# Constants of the setup itself (chance levels, config values), not results.
CONSTANTS = {"0.17", "0.20", "0.33", "0.167", "0.200", "0.125", "0.5", "0.1", "0.2", "0.01"}


def main() -> None:
    cfg = load_task2_config()
    numbers = (Path(cfg["paths"]["encoder_dir"]) / "report_numbers.md").read_text(encoding="utf-8")
    report = (ROOT / "reports" / "task2_report_section.md").read_text(encoding="utf-8")
    found = sorted(set(re.findall(r"(?<![\w.])\d+\.\d{2,}(?![\w.])", report)))
    missing = [n for n in found if n not in CONSTANTS and n not in numbers]
    print(f"{len(found)} distinct numbers in the report; {len(found) - len(missing)} matched")
    if missing:
        print("not found in report_numbers.md:", ", ".join(missing))
        sys.exit(1)


if __name__ == "__main__":
    main()
