#!/usr/bin/env python3
"""Recompute LVBench table counts from per-question dumps. No VLM."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROWS = [
    ("held-out", "AKS", "dumps/table_heldout/aks/held882.json", 411, 882),
    (
        "held-out",
        "StatedTimeAddressDecode",
        "dumps/table_heldout/stated_time_address_decode_iter9/held882.json",
        432,
        882,
    ),
    (
        "held-out",
        "CardinalityLedger",
        "dumps/table_heldout/cardinality_ledger/held882.json",
        483,
        882,
    ),
    ("held-out", "AKS-90", "dumps/table_heldout/aks_k90/held882.json", 434, 882),
    ("dev", "AKS", "dumps/table_dev350/aks/val.json", 174, 350),
    (
        "dev",
        "StatedTimeAddressDecode",
        "dumps/table_dev350/stated_time_address_decode_iter9/val.json",
        178,
        350,
    ),
    (
        "dev",
        "CardinalityLedger",
        "dumps/table_dev350/cardinality_ledger/val.json",
        204,
        350,
    ),
    ("dev", "AKS-90", "dumps/table_dev350/aks_k90/val.json", 186, 350),
]


def archive_root(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).resolve()
    here = Path(__file__).resolve().parent.parent
    if (here / "dumps" / "table_heldout").is_dir():
        return here
    raise SystemExit("pass --archive pointing at the supplementary pack")


def load_dump(path: Path) -> tuple[int, int]:
    data = json.loads(path.read_text())
    results = data.get("results") or []
    correct = sum(1 for r in results if r.get("was_correct"))
    total = len(results)
    header_c, header_t = data.get("correct"), data.get("total")
    if header_c is not None and int(header_c) != correct:
        raise SystemExit(f"{path}: header correct={header_c} != vector {correct}")
    if header_t is not None and int(header_t) != total:
        raise SystemExit(f"{path}: header total={header_t} != vector {total}")
    return correct, total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", default=None)
    args = parser.parse_args()
    root = archive_root(args.archive)

    print("LVBench held-out")
    failed = False
    last_split = None
    for split, label, rel, exp_c, exp_t in ROWS:
        if split != last_split and last_split == "held-out":
            print()
            print("LVBench dev")
        last_split = split
        path = root / rel
        if not path.is_file():
            print(f"{label:<24} MISSING {rel}")
            failed = True
            continue
        got_c, got_t = load_dump(path)
        acc = 100.0 * got_c / got_t if got_t else 0.0
        print(f"{label:<24} {got_c:3d} / {got_t} = {acc:5.2f}")
        if (got_c, got_t) != (exp_c, exp_t):
            print(f"  expected {exp_c} / {exp_t}", file=sys.stderr)
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
