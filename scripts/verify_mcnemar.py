#!/usr/bin/env python3
"""Exact McNemar + paired bootstrap ΔAcc 95% CI from dump vectors. No VLM."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Sequence

HELD = {
    "uniform": "dumps/table_heldout/pilot_uniform_k/held882.json",
    "weakft": "dumps/table_heldout/stated_time_address_decode_iter9/held882.json",
    "aks": "dumps/table_heldout/aks/held882.json",
    "ledger": "dumps/table_heldout/cardinality_ledger/held882.json",
    "aks90": "dumps/table_heldout/aks_k90/held882.json",
}

# Frozen AKS → CardinalityLedger held-out (paper / scores.json).
EXPECTED_AKS_LEDGER = {"ref_only": 63, "cand_only": 135, "p": 3.39e-7}

PAIRS = [
    ("Uniform", "uniform", "StatedTimeAddressDecode", "weakft"),
    ("AKS", "aks", "CardinalityLedger", "ledger"),
    ("AKS-90", "aks90", "CardinalityLedger", "ledger"),
]


def archive_root(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).resolve()
    here = Path(__file__).resolve().parent.parent
    if (here / "dumps" / "table_heldout").is_dir():
        return here
    raise SystemExit("pass --archive pointing at the supplementary pack")


def load_flags(path: Path) -> tuple[list[bool], list[object]]:
    data = json.loads(path.read_text())
    results = data.get("results") or []
    flags = [bool(r.get("was_correct")) for r in results]
    keys = [(r.get("target"), r.get("question_id")) for r in results]
    return flags, keys


def mcnemar_exact(ref: Sequence[bool], cand: Sequence[bool]) -> dict:
    n = min(len(ref), len(cand))
    ref_only = sum(1 for i in range(n) if ref[i] and not cand[i])
    cand_only = sum(1 for i in range(n) if cand[i] and not ref[i])
    m = ref_only + cand_only
    if m == 0:
        p = 1.0
    else:
        k = min(ref_only, cand_only)
        p = min(1.0, sum(math.comb(m, i) for i in range(k + 1)) / (2**m) * 2)
    return {
        "n": n,
        "ref_only": ref_only,
        "cand_only": cand_only,
        "discordant": m,
        "p": p,
        "acc_ref": 100.0 * sum(ref[:n]) / n,
        "acc_cand": 100.0 * sum(cand[:n]) / n,
        "delta": 100.0 * (sum(cand[:n]) - sum(ref[:n])) / n,
    }


def paired_bootstrap_ci(
    ref: Sequence[bool],
    cand: Sequence[bool],
    n_boot: int,
    seed: int,
) -> dict:
    n = min(len(ref), len(cand))
    d = [int(cand[i]) - int(ref[i]) for i in range(n)]
    point = 100.0 * sum(d) / n
    rng = random.Random(seed)
    boots = [0.0] * n_boot
    for b in range(n_boot):
        s = 0
        for i in rng.choices(range(n), k=n):
            s += d[i]
        boots[b] = 100.0 * s / n
    boots.sort()

    def pct(q: float) -> float:
        if n_boot == 1:
            return boots[0]
        k = (n_boot - 1) * q
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return boots[int(k)]
        return boots[f] * (c - k) + boots[c] * (k - f)

    return {
        "n": n,
        "n_boot": n_boot,
        "seed": seed,
        "delta_pp": point,
        "ci95_pp": [pct(0.025), pct(0.975)],
    }


def fmt_p(p: float) -> str:
    if p < 1e-4:
        return f"{p:.2e}".replace("e-0", "e-")
    return f"{p:.6g}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", default=None)
    parser.add_argument("--n-boot", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write paper/bootstrap.json next to the dumps",
    )
    args = parser.parse_args()
    root = archive_root(args.archive)

    flags: dict[str, list[bool]] = {}
    keys: dict[str, list[object]] = {}
    for name, rel in HELD.items():
        path = root / rel
        if not path.is_file():
            raise SystemExit(f"missing {path}")
        flags[name], keys[name] = load_flags(path)

    n = flags["aks"]
    if len(n) != 882:
        raise SystemExit(f"AKS held-out n={len(n)}, expected 882")
    for name in HELD:
        if len(flags[name]) != 882:
            raise SystemExit(f"{name} n={len(flags[name])}, expected 882")
        if keys[name] != keys["aks"]:
            raise SystemExit(f"{name} target/id alignment differs from AKS")

    aks_ledger = mcnemar_exact(flags["aks"], flags["ledger"])
    print("AKS-only:               {0}".format(aks_ledger["ref_only"]))
    print("CardinalityLedger-only: {0}".format(aks_ledger["cand_only"]))
    print("Exact two-sided p:      {0}".format(fmt_p(aks_ledger["p"])))

    failed = False
    if (
        aks_ledger["ref_only"] != EXPECTED_AKS_LEDGER["ref_only"]
        or aks_ledger["cand_only"] != EXPECTED_AKS_LEDGER["cand_only"]
        or abs(aks_ledger["p"] - EXPECTED_AKS_LEDGER["p"])
        > 0.05 * EXPECTED_AKS_LEDGER["p"]
    ):
        print("frozen AKS→Ledger McNemar mismatch", file=sys.stderr)
        failed = True

    print()
    print("Paired bootstrap ΔAcc (percentage points), 95% percentile CI")
    print(f"n_boot={args.n_boot}  seed={args.seed}  n={aks_ledger['n']}")
    boot_rows = []
    for ref_label, ref_key, cand_label, cand_key in PAIRS:
        m = mcnemar_exact(flags[ref_key], flags[cand_key])
        b = paired_bootstrap_ci(
            flags[ref_key], flags[cand_key], args.n_boot, args.seed
        )
        lo, hi = b["ci95_pp"]
        print(
            f"{ref_label} → {cand_label}:  "
            f"Δ = {b['delta_pp']:+.2f}  "
            f"95% CI [{lo:+.2f}, {hi:+.2f}]  "
            f"(McNemar {m['cand_only']}/{m['ref_only']}  p={fmt_p(m['p'])})"
        )
        boot_rows.append(
            {
                "ref": ref_label,
                "cand": cand_label,
                "n": b["n"],
                "delta_pp": round(b["delta_pp"], 4),
                "ci95_pp": [round(lo, 4), round(hi, 4)],
                "mcnemar_ref_only": m["ref_only"],
                "mcnemar_cand_only": m["cand_only"],
                "mcnemar_p": m["p"],
            }
        )

    if args.write:
        out = {
            "protocol": {
                "split": "lvbench_held882_seed42",
                "n_boot": args.n_boot,
                "seed": args.seed,
                "method": (
                    "paired percentile bootstrap of accuracy difference "
                    "in percentage points; questions resampled with replacement"
                ),
            },
            "pairs": boot_rows,
        }
        dest = root / "paper" / "bootstrap.json"
        dest.write_text(json.dumps(out, indent=2) + "\n")
        print()
        print(f"wrote {dest}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
