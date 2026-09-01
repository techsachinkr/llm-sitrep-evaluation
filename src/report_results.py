"""Turn slot judgements into the paper's Phase 4 result tables.

`src/judge_slots.py` writes per-(sitrep, slot) verdicts; `src/metrics.py` holds the scoring
functions. Nothing joined the two into the tables PLAN's Phase 4 acceptance asks for — this
module is that join, and it is the producer of the paper's headline finding:

  results/tables/completeness.csv      completeness per arm (machine models/arms + the human
                                       ceiling), with bootstrap 95% CIs
  results/tables/slot_coverage.csv     per-slot coverage for every arm — the heatmap in Fig. (a)
  results/tables/visibility_ceiling.csv  per-slot ceiling verdict: slots the machine sitreps
                                       systematically cannot fill while humans do

Reads `results/tables/slot_judgements.csv` (written by judge_slots). Human sitreps carry
`kind == "human"` and set the ceiling; everything else is a machine arm labelled `<model>/<arm>`.

CLI: python -m src.report_results [--event cyclone_idai_2019] [--machine-max 0.1] [--human-min 0.5]
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from src.metrics import (
    VERDICT_SCORE,
    bootstrap_ci,
    coverage_by_slot,
    operational_completeness,
    visibility_ceiling,
)
from src.util import cfg_path, ensure_dir, get_logger, load_config

log = get_logger("sitrep.report")

JUDGEMENTS_CSV = "slot_judgements.csv"
HUMAN_KIND = "human"
HUMAN_ARM = "human"


def load_judgements(path: Path) -> list[dict[str, Any]]:
    """Read slot_judgements.csv, keeping only rows with a usable verdict."""
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    good = []
    for r in rows:
        verdict = (r.get("verdict") or "").strip().lower()
        if verdict in VERDICT_SCORE:
            good.append(r)
    dropped = len(rows) - len(good)
    if dropped:
        log.warning("%d/%d judgement row(s) had no usable verdict and were skipped", dropped, len(rows))
    return good


def arm_of(row: dict[str, Any]) -> str:
    """The arm a judgement belongs to: 'human' for the ceiling, '<model>/<arm>' otherwise."""
    if (row.get("kind") or "").strip().lower() == HUMAN_KIND:
        return HUMAN_ARM
    return (row.get("arm") or "").strip() or "machine"


def group_by_arm(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        out[arm_of(r)].append(r)
    return dict(out)


def completeness_table(by_arm: dict[str, list[dict[str, Any]]], *, seed: int,
                       weights: dict[str, float] | None = None) -> list[dict[str, Any]]:
    """Per-arm operational completeness with bootstrap CIs, computed per sitrep then averaged."""
    rows: list[dict[str, Any]] = []
    for arm, judgements in sorted(by_arm.items()):
        per_sitrep: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for j in judgements:
            per_sitrep[j.get("sitrep_id", "")].append(j)
        scores = [operational_completeness(js, weights, default_weight=1.0 if weights else None)
                  for js in per_sitrep.values() if js]
        if not scores:
            continue
        ci = bootstrap_ci(scores, seed=seed) if len(scores) > 1 else None
        rows.append({
            "arm": arm,
            "is_human_ceiling": arm == HUMAN_ARM,
            "n_sitreps": len(scores),
            "n_judgements": len(judgements),
            "completeness": round(sum(scores) / len(scores), 4),
            "ci_lo": round(ci.lo, 4) if ci else "",
            "ci_hi": round(ci.hi, 4) if ci else "",
        })
    rows.sort(key=lambda r: (not r["is_human_ceiling"], -float(r["completeness"])))
    return rows


def slot_coverage_table(by_arm: dict[str, list[dict[str, Any]]]) -> tuple[list[dict[str, Any]],
                                                                         dict[str, dict[str, float]],
                                                                         dict[str, float]]:
    """Per-(arm, slot) coverage rows, plus the machine/human maps the ceiling analysis needs."""
    rows: list[dict[str, Any]] = []
    machine: dict[str, dict[str, float]] = defaultdict(dict)
    human: dict[str, float] = {}
    for arm, judgements in sorted(by_arm.items()):
        for slot_id, cov in coverage_by_slot(judgements).items():
            rows.append({"arm": arm, "is_human_ceiling": arm == HUMAN_ARM, **cov.as_row()})
            if arm == HUMAN_ARM:
                human[slot_id] = cov.coverage
            else:
                machine[slot_id][arm] = cov.coverage
    rows.sort(key=lambda r: (r["slot_id"], not r["is_human_ceiling"], r["arm"]))
    return rows, {k: dict(v) for k, v in machine.items()}, human


def write_csv(path: Path, rows: Sequence[dict[str, Any]], cols: Sequence[str]) -> Path:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cols), extrasaction="ignore", restval="")
        w.writeheader()
        w.writerows(rows)
    return path


def build(tables_dir: Path, *, seed: int, machine_max: float, human_min: float,
          event: str | None = None, weights: dict[str, float] | None = None) -> dict[str, Any]:
    """Produce the three Phase 4 tables; returns a summary dict."""
    rows = load_judgements(tables_dir / JUDGEMENTS_CSV)
    if event:
        rows = [r for r in rows if (r.get("event") or "") == event]
    if not rows:
        return {"ok": False, "reason": "no usable judgements"}
    by_arm = group_by_arm(rows)
    comp = completeness_table(by_arm, seed=seed, weights=weights)
    cov_rows, machine, human = slot_coverage_table(by_arm)
    paths = {
        "completeness": write_csv(tables_dir / "completeness.csv", comp,
                                  ["arm", "is_human_ceiling", "n_sitreps", "n_judgements",
                                   "completeness", "ci_lo", "ci_hi"]),
        "slot_coverage": write_csv(tables_dir / "slot_coverage.csv", cov_rows,
                                   ["slot_id", "arm", "is_human_ceiling", "n", "coverage",
                                    "n_present", "n_partial", "n_absent"]),
    }
    ceilings: list[Any] = []
    if human and machine:
        ceilings = list(visibility_ceiling(machine, human, machine_max=machine_max,
                                           human_min=human_min))
        ceilings.sort(key=lambda c: (not c.is_ceiling, -(c.gap or 0)))
        paths["visibility_ceiling"] = write_csv(
            tables_dir / "visibility_ceiling.csv", [c.as_row() for c in ceilings],
            ["slot_id", "is_ceiling", "human_coverage", "machine_coverage_max", "machine_best_arm",
             "machine_coverage_by_arm", "gap", "reason"])
    else:
        log.warning("no %s judgements yet — the ceiling analysis needs human sitreps scored too",
                    "human" if not human else "machine")
    return {"ok": True, "paths": paths, "completeness": comp, "ceilings": ceilings,
            "n_judgements": len(rows), "arms": sorted(by_arm)}


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description="Build the Phase 4 completeness / ceiling tables")
    ap.add_argument("--event", default=None, help="limit to one event")
    ap.add_argument("--machine-max", type=float, default=None,
                    help="machine coverage at/below this = invisible (default: config ceiling.machine_max)")
    ap.add_argument("--human-min", type=float, default=None,
                    help="human coverage at/above this = expected (default: config ceiling.human_min)")
    ap.add_argument("--tables-dir", type=Path, default=None)
    args = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):  # pragma: no cover
        pass
    cfg = load_config()
    tables = args.tables_dir or (cfg_path(cfg, "results", "results") / "tables")
    ceiling_cfg = cfg.get("ceiling") or {}
    machine_max = args.machine_max if args.machine_max is not None else float(ceiling_cfg.get("machine_max", 0.6))
    human_min = args.human_min if args.human_min is not None else float(ceiling_cfg.get("human_min", 0.8))
    out = build(tables, seed=int(cfg.get("seed", 17)), machine_max=machine_max,
                human_min=human_min, event=args.event)
    if not out["ok"]:
        print(f"No judgements to report ({out['reason']}).\n"
              f"Expected {tables / JUDGEMENTS_CSV} — produce it with:\n"
              "  python -m src.judge_slots --event <event>")
        return 2
    print(f"operational completeness ({out['n_judgements']} judgements, {len(out['arms'])} arm(s))\n")
    print(f"  {'arm':28s} {'n':>4} {'completeness':>13}  95% CI")
    for r in out["completeness"]:
        ci = f"[{r['ci_lo']}, {r['ci_hi']}]" if r["ci_lo"] != "" else ""
        mark = "  <- human ceiling" if r["is_human_ceiling"] else ""
        print(f"  {r['arm']:28s} {r['n_sitreps']:4d} {r['completeness']:13.4f}  {ci}{mark}")
    ceil = [c for c in out["ceilings"] if c.is_ceiling]
    if out["ceilings"]:
        print(f"\nvisibility ceiling: {len(ceil)} of {len(out['ceilings'])} slot(s) that human sitreps "
              f"fill but machine sitreps cannot")
        for c in ceil:
            print(f"  {c.slot_id:24s} human={c.human_coverage:.2f} machine_max={c.machine_coverage_max:.2f} "
                  f"gap={c.gap:.2f}")
    print("\ntables:")
    for name, p in out["paths"].items():
        print(f"  {name:20s} {p}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
