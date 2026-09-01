"""Compare two judges over the same (sitrep, slot) pairs.

The primary judge shares a model family with the generators, which is the standing threat to the
completeness numbers: a same-family judge may be systematically generous to its own family's
output. Running a **cross-family** judge over the identical pairs turns that limitation into a
measurement — either the two judges agree, and the numbers are robust to judge choice, or they do
not, and the paper needs to say so.

Note what this does and does not establish. Agreement shows the judges are *consistent*; it does
not show either is *correct*. Correctness still requires the hand-scored validation sample
(`judge_slots --sample-for-validation`), which PLAN sets at Cohen's kappa >= 0.7.

Inputs are two `slot_judgements.csv` files (see `src.judge_slots`), joined on
(event, sitrep_id, slot_id).

Writes `results/tables/judge_agreement_xfam.csv` (per-slot) and prints a summary including how
much each arm's operational completeness moves between judges.

CLI:
    python -m src.judge_agreement --a results/tables/slot_judgements.csv \\
                                  --b results/tables/xfam/slot_judgements.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.metrics import VERDICT_SCORE, judge_agreement
from src.util import cfg_path, ensure_dir, get_logger, load_config

log = get_logger("sitrep.judge_agreement")

KEY = ("event", "sitrep_id", "slot_id")


def load(path: Path) -> dict[tuple[str, ...], dict[str, Any]]:
    """slot_judgements.csv -> {(event, sitrep_id, slot_id): row} keeping usable verdicts only."""
    if not path.is_file():
        raise FileNotFoundError(f"no judgement table at {path}")
    out: dict[tuple[str, ...], dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if (row.get("verdict") or "").strip().lower() in VERDICT_SCORE:
                out[tuple(row.get(k, "") for k in KEY)] = row
    return out


def completeness_by_arm(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Mean per-sitrep completeness for each arm, so judges can be compared on the paper's metric."""
    per_sitrep: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in rows:
        arm = "human" if (r.get("kind") or "").lower() == "human" else (r.get("arm") or "machine")
        per_sitrep[(arm, r.get("sitrep_id", ""))].append(VERDICT_SCORE[r["verdict"].strip().lower()])
    by_arm: dict[str, list[float]] = defaultdict(list)
    for (arm, _sid), scores in per_sitrep.items():
        by_arm[arm].append(sum(scores) / len(scores))
    return {arm: sum(v) / len(v) for arm, v in sorted(by_arm.items()) if v}


def compare(a: dict[tuple[str, ...], dict[str, Any]],
            b: dict[tuple[str, ...], dict[str, Any]]) -> dict[str, Any]:
    """Overall + per-slot agreement between two judges on their shared pairs."""
    shared = sorted(set(a) & set(b))
    if not shared:
        raise ValueError("the two judgement tables share no (event, sitrep_id, slot_id) keys")
    va = [a[k]["verdict"].strip().lower() for k in shared]
    vb = [b[k]["verdict"].strip().lower() for k in shared]
    overall = judge_agreement(va, vb)

    per_slot: list[dict[str, Any]] = []
    by_slot: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for k in shared:
        by_slot[k[2]].append((a[k]["verdict"].strip().lower(), b[k]["verdict"].strip().lower()))
    for slot_id, pairs in sorted(by_slot.items()):
        ag = judge_agreement([p[0] for p in pairs], [p[1] for p in pairs])
        per_slot.append({
            "slot_id": slot_id, "n": ag.n,
            "agreement": round(ag.agreement, 4),
            "kappa": None if ag.kappa is None else round(ag.kappa, 4),
            "mean_score_a": round(sum(VERDICT_SCORE[p[0]] for p in pairs) / len(pairs), 4),
            "mean_score_b": round(sum(VERDICT_SCORE[p[1]] for p in pairs) / len(pairs), 4),
        })
    per_slot.sort(key=lambda r: (r["kappa"] if r["kappa"] is not None else 1.0))
    return {
        "n_shared": len(shared), "n_only_a": len(set(a) - set(b)), "n_only_b": len(set(b) - set(a)),
        "overall": overall, "per_slot": per_slot,
        "completeness_a": completeness_by_arm([a[k] for k in shared]),
        "completeness_b": completeness_by_arm([b[k] for k in shared]),
        "judge_a": next(iter(a.values())).get("judge_model", "?"),
        "judge_b": next(iter(b.values())).get("judge_model", "?"),
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description="Agreement between two judges over the same pairs")
    ap.add_argument("--a", type=Path, required=True, help="first slot_judgements.csv")
    ap.add_argument("--b", type=Path, required=True, help="second slot_judgements.csv")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):  # pragma: no cover
        pass
    try:
        res = compare(load(args.a), load(args.b))
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc))
        return 2

    ov = res["overall"]
    print(f"judge A: {res['judge_a']}\njudge B: {res['judge_b']}\n")
    print(f"shared pairs: {res['n_shared']}  (only in A: {res['n_only_a']}, only in B: {res['n_only_b']})")
    print(f"raw agreement       {ov.agreement:.4f}")
    print(f"Cohen's kappa       {ov.kappa if ov.kappa is None else round(ov.kappa, 4)}")
    print(f"linear-weighted     {ov.kappa_linear if ov.kappa_linear is None else round(ov.kappa_linear, 4)}")
    verdict = ("judges agree substantially — the completeness numbers are robust to judge choice"
               if (ov.kappa or 0) >= 0.6 else
               "judges disagree materially — report this and prefer the cross-family judge")
    print(f"  -> {verdict}")

    print("\noperational completeness under each judge:")
    arms = sorted(set(res["completeness_a"]) | set(res["completeness_b"]))
    print(f"  {'arm':28s} {'judge A':>9s} {'judge B':>9s} {'delta':>7s}")
    for arm in arms:
        ca, cb = res["completeness_a"].get(arm), res["completeness_b"].get(arm)
        if ca is None or cb is None:
            continue
        print(f"  {arm:28s} {ca:9.4f} {cb:9.4f} {cb - ca:+7.4f}")

    print("\nlowest-agreement slots:")
    for r in res["per_slot"][:5]:
        print(f"  {r['slot_id']:26s} n={r['n']:4d} agree={r['agreement']:.3f} kappa={r['kappa']} "
              f"(mean score {r['mean_score_a']:.2f} vs {r['mean_score_b']:.2f})")

    out = args.out or (cfg_path(load_config(), "results", "results") / "tables" / "judge_agreement_xfam.csv")
    ensure_dir(out.parent)
    with open(out, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["slot_id", "n", "agreement", "kappa",
                                           "mean_score_a", "mean_score_b"], extrasaction="ignore")
        w.writeheader()
        w.writerows(res["per_slot"])
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
