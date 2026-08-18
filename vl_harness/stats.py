"""Significance testing and selection-bias correction for harness comparisons.

Harness-evolution results are usually reported as a single accuracy delta with no
interval and no correction for the fact that the champion was chosen as the best
of K candidates on the same split. Both omissions matter at the sample sizes this
field runs at: on n=150 with candidates that agree on ~76% of questions, the
best of 16 *equivalent* candidates is expected to look 5-7 pp better than the
baseline. This module makes the two corrections part of the framework rather
than something a reader has to do afterwards.

CLI:

    python -m vl_harness.stats compare  A/val.json B/val.json
    python -m vl_harness.stats sweep    logs/<run>            # champion vs baseline
    python -m vl_harness.stats partition FULL/test.json --val 200:350
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Sequence

# --------------------------------------------------------------------------
# primitives
# --------------------------------------------------------------------------


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion, in percent."""
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return ((c - h) * 100, (c + h) * 100)


def mcnemar_exact(a: Sequence[bool], b: Sequence[bool]) -> dict[str, Any]:
    """Two-sided exact McNemar test for two systems on the SAME questions.

    ``a`` is the reference (baseline), ``b`` the candidate. Only discordant pairs
    carry information, which is why a 6 pp gap on 150 questions can still be
    indistinguishable from noise: it may rest on ~36 informative items.
    """
    n = min(len(a), len(b))
    b01 = sum(1 for i in range(n) if a[i] and not b[i])  # baseline only
    b10 = sum(1 for i in range(n) if b[i] and not a[i])  # candidate only
    m = b01 + b10
    if m == 0:
        p = 1.0
    else:
        k = min(b01, b10)
        p = min(1.0, sum(math.comb(m, i) for i in range(k + 1)) / (2**m) * 2)
    acc_a = sum(a[:n]) / n * 100 if n else 0.0
    acc_b = sum(b[:n]) / n * 100 if n else 0.0
    return {
        "n": n,
        "acc_ref": acc_a,
        "acc_cand": acc_b,
        "delta": acc_b - acc_a,
        "discordant": m,
        "ref_only": b01,
        "cand_only": b10,
        "p": p,
    }


def paired_mde(n: int, discordance: float = 0.24, power: float = 0.80) -> float:
    """Smallest paired delta (pp) detectable at 5% two-sided, given power."""
    z_a, z_b = 1.96, 0.84 if power == 0.80 else 1.28
    return (z_a + z_b) * math.sqrt(discordance / max(n, 1)) * 100


def selection_bias_band(
    n: int,
    k_candidates: int,
    p_true: float = 0.52,
    trials: int = 4000,
    seed: int = 0,
) -> dict[str, float]:
    """Expected best-of-K accuracy when every candidate truly equals baseline.

    Independent candidates, so this is the PESSIMISTIC end of the band: real
    candidates share code lineage and are correlated, which shrinks the effect.
    Report a champion delta against this band, not against zero.
    """
    rng = random.Random(seed)
    best = []
    for _ in range(trials):
        best.append(
            max(
                sum(rng.random() < p_true for _ in range(n)) / n
                for _ in range(max(1, k_candidates))
            )
        )
    best.sort()
    mean = sum(best) / len(best) * 100
    return {
        "expected_best": mean,
        "apparent_gain": mean - p_true * 100,
        "p95": best[int(0.95 * (len(best) - 1))] * 100,
    }


def holm_bonferroni(pvals: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values (family-wise error control)."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj: dict[str, float] = {}
    running = 0.0
    for i, (name, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adj[name] = running
    return adj


# --------------------------------------------------------------------------
# result loading
# --------------------------------------------------------------------------


def load_correct(path: str | Path) -> list[bool]:
    """Per-question correctness vector, in dataset order."""
    d = json.loads(Path(path).read_text())
    return [bool(r.get("was_correct")) for r in (d.get("results") or [])]


def _fmt_p(p: float) -> str:
    stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
    return f"{p:.4f}{stars}"


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def cmd_compare(args) -> None:
    a, b = load_correct(args.reference), load_correct(args.candidate)
    r = mcnemar_exact(a, b)
    lo, hi = wilson_ci(sum(b[: r["n"]]), r["n"])
    print(f"  reference : {r['acc_ref']:5.1f}%   ({Path(args.reference).parts[-3]})")
    print(f"  candidate : {r['acc_cand']:5.1f}%   [95% CI {lo:.1f}-{hi:.1f}]")
    print(f"  delta     : {r['delta']:+5.1f} pp")
    print(
        f"  McNemar   : {r['cand_only']}/{r['ref_only']} discordant={r['discordant']}"
        f"  p={_fmt_p(r['p'])}"
    )
    print(f"  MDE(80%)  : {paired_mde(r['n']):.1f} pp at n={r['n']}")
    if args.candidates_evaluated:
        band = selection_bias_band(
            r["n"], args.candidates_evaluated, r["acc_ref"] / 100
        )
        print(
            f"  best-of-{args.candidates_evaluated} null band: "
            f"+{band['apparent_gain']:.1f} pp expected from selection alone "
            f"(p95 {band['p95'] - r['acc_ref']:+.1f})"
        )
        if r["delta"] <= band["apparent_gain"]:
            print("  -> delta is INSIDE the selection-bias band; not evidence of a real gain")


def cmd_sweep(args) -> None:
    root = Path(args.run)
    files = sorted(root.glob("lvbench/*/*/val.json"))
    if not files:
        print(f"no val.json under {root}")
        return
    scores = {}
    for f in files:
        c = load_correct(f)
        if c:
            scores[f.parts[-3]] = c
    if args.baseline not in scores:
        print(f"baseline '{args.baseline}' not found; have: {sorted(scores)[:5]}")
        return
    ref = scores[args.baseline]
    rows = []
    for name, c in scores.items():
        if name == args.baseline:
            continue
        rows.append((name, mcnemar_exact(ref, c)))
    rows.sort(key=lambda x: -x[1]["delta"])
    adj = holm_bonferroni({n: r["p"] for n, r in rows})
    n = rows[0][1]["n"] if rows else 0
    band = selection_bias_band(n, len(rows) + 1, sum(ref) / max(len(ref), 1))
    print(f"baseline {args.baseline}: {sum(ref) / len(ref) * 100:.1f}%  n={n}")
    print(
        f"selection-bias band for best-of-{len(rows) + 1}: "
        f"+{band['apparent_gain']:.1f} pp expected under the null\n"
    )
    print(f"  {'delta':>7} {'p':>10} {'p_holm':>8}  system")
    for name, r in rows:
        # The band bounds how much a WINNER can gain by chance; it says nothing
        # about candidates that lost, so only positive deltas are judged by it.
        if r["delta"] <= 0:
            flag = ""
        elif r["delta"] <= band["apparent_gain"]:
            flag = "  (within selection noise)"
        else:
            flag = "  (exceeds selection band)"
        print(
            f"  {r['delta']:+7.1f} {_fmt_p(r['p']):>10} {adj[name]:8.3f}  {name}{flag}"
        )


def cmd_budget(args) -> None:
    """The pre-registered budget-sweep analysis.

    One paired test per budget (retrieved-K vs uniform-K at the SAME K), Holm
    corrected across budgets, plus the H1 monotonicity check. Effects below the
    study's minimum detectable effect are reported as inconclusive rather than
    as absence of an effect, as committed to before any of this was run.
    """
    root = Path(args.logs)
    budgets: dict[int, tuple[list[bool], list[bool]]] = {}
    oracles: dict[int, list[bool]] = {}
    for d in sorted(root.glob(f"{args.prefix}*")):
        try:
            k = int(d.name.rsplit("_", 1)[-1])
        except ValueError:
            continue
        u = d / f"lvbench/{args.uniform}/{args.model}/val.json"
        r = d / f"lvbench/{args.retrieved}/{args.model}/val.json"
        o = d / f"lvbench/{args.oracle}/{args.model}/val.json"
        if u.exists() and r.exists():
            budgets[k] = (load_correct(u), load_correct(r))
        if u.exists() and o.exists():
            oracles[k] = load_correct(o)
    if not budgets and not oracles:
        print(f"[budget] no completed arms under {root}/{args.prefix}*")
        return

    if oracles:
        print("  ORACLE CEILING (perfect selection vs uniform, same budget)")
        print(f"  {'budget':>7} {'uniform':>8} {'oracle':>8} {'delta':>7} {'p':>9}")
        for k in sorted(oracles):
            u = load_correct(
                root / f"{args.prefix}{k}/lvbench/{args.uniform}/{args.model}/val.json"
            )
            r = mcnemar_exact(u, oracles[k])
            print(
                f"  {k:>7} {r['acc_ref']:7.1f}% {r['acc_cand']:7.1f}% "
                f"{r['delta']:+7.1f} {r['p']:9.4f}"
            )
        print("  A retriever cannot beat the oracle: if this delta is ~0, no")
        print("  frame-selection axis has headroom on this benchmark/backbone.\n")

    if not budgets:
        return
    results = {k: mcnemar_exact(u, r) for k, (u, r) in sorted(budgets.items())}
    adj = holm_bonferroni({str(k): v["p"] for k, v in results.items()})

    print(f"  {'budget':>7} {'uniform':>8} {'retrieved':>10} {'delta':>7} "
          f"{'p':>9} {'p_holm':>8} {'verdict':>14}")
    for k, r in results.items():
        mde = paired_mde(r["n"])
        if adj[str(k)] < 0.05 and r["delta"] > 0:
            verdict = "SIGNIFICANT+"
        elif abs(r["delta"]) < mde:
            verdict = "inconclusive"
        elif r["delta"] < 0:
            verdict = "negative"
        else:
            verdict = "n.s."
        print(
            f"  {k:>7} {r['acc_ref']:7.1f}% {r['acc_cand']:9.1f}% "
            f"{r['delta']:+7.1f} {r['p']:9.4f} {adj[str(k)]:8.3f} {verdict:>14}"
        )

    ks = sorted(results)
    print(f"\n  MDE at n={results[ks[0]]['n']}: {paired_mde(results[ks[0]]['n']):.1f} pp "
          "(smaller effects are inconclusive, not null)")
    if len(ks) >= 2:
        lo, hi = results[ks[0]]["delta"], results[ks[-1]]["delta"]
        sig_low = adj[str(ks[0])] < 0.05 and lo > 0
        print(f"  H1 monotonicity: delta at K={ks[0]} is {lo:+.1f}, "
              f"at K={ks[-1]} is {hi:+.1f} -> "
              f"{'shrinks with budget as predicted' if hi < lo else 'does NOT shrink'}")
        print(f"  H1 verdict: "
              f"{'SUPPORTED' if (sig_low and hi < lo) else 'not supported (see pre-registration 4)'}")


def cmd_partition(args) -> None:
    """Split one full-pool evaluation into the original train/val/test slices.

    Results are stored in dataset order, so a run over the whole pool can be cut
    back into the historical splits after the fact. The held-out number is the
    one that was never used for model selection.
    """
    d = json.loads(Path(args.results).read_text())
    rs = d.get("results") or []
    lo, hi = (int(x) for x in args.val.split(":"))
    parts = {
        f"train[0:{lo}]": rs[:lo],
        f"val[{lo}:{hi}]  (SELECTION SET)": rs[lo:hi],
        f"test[{hi}:]  (HELD OUT)": rs[hi:],
        "full[0:]": rs,
    }
    print(f"{Path(args.results).parts[-3]}  total={len(rs)}\n")
    for label, sub in parts.items():
        if not sub:
            continue
        k = sum(1 for r in sub if r.get("was_correct"))
        ci = wilson_ci(k, len(sub))
        print(
            f"  {label:<32} n={len(sub):>4}  acc={k / len(sub) * 100:5.1f}%"
            f"  [95% CI {ci[0]:.1f}-{ci[1]:.1f}]"
        )
    if len(rs) > hi:
        full = sum(1 for r in rs if r.get("was_correct")) / len(rs) * 100
        held = sum(1 for r in rs[hi:] if r.get("was_correct")) / len(rs[hi:]) * 100
        print(
            f"\n  contamination check: full - held_out = {full - held:+.2f} pp"
            "  (the bias re-imported by scoring the selection set)"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("compare", help="paired test between two val/test result files")
    c.add_argument("reference")
    c.add_argument("candidate")
    c.add_argument(
        "--candidates-evaluated",
        type=int,
        default=0,
        help="K for the best-of-K selection-bias band",
    )
    c.set_defaults(func=cmd_compare)

    s = sub.add_parser("sweep", help="all systems in a run vs its baseline")
    s.add_argument("run")
    s.add_argument("--baseline", default="uniform_frames_no_memory")
    s.set_defaults(func=cmd_sweep)

    b = sub.add_parser("budget", help="pre-registered budget-sweep analysis")
    b.add_argument("--logs", default="logs")
    b.add_argument("--prefix", default="pilot_budget_")
    b.add_argument("--uniform", default="pilot_uniform_k")
    b.add_argument("--retrieved", default="pilot_retrieved_k")
    b.add_argument("--oracle", default="oracle_evidence_k")
    b.add_argument("--model", default="qwen3-vl-8b-instruct")
    b.set_defaults(func=cmd_budget)

    p = sub.add_parser("partition", help="cut a full-pool evaluation into splits")
    p.add_argument("results")
    p.add_argument("--val", default="200:350", help="val slice as lo:hi")
    p.set_defaults(func=cmd_partition)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
