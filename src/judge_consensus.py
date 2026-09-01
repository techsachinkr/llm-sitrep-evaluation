"""Three-judge consensus: Fleiss' kappa, a majority-vote judge, and the ceiling under it.

`src.judge_agreement` compares two judges. With three we can do better than pairwise agreement:

* **Fleiss' kappa** over all three raters at once, on the items every judge scored;
* a **majority-vote consensus judge** (2 of 3; ties fall back to the primary judge), which is a
  more defensible instrument than any single model because no one family's leniency decides a
  verdict alone;
* the **visibility ceiling recomputed under that consensus**, which is the number the paper
  should report.

Why this matters here: the per-judge binary ceiling flag turned out to be fragile at the
threshold (one judge missed `displacement` by 0.004 of machine coverage) even though all three
judges agreed on the *effect size* to within 0.02. Reporting the consensus, and reporting gaps
continuously rather than as a flag, is the honest way to present that.

Missingness is not ignorable: a judge that fails to return parsable output on hard slots shrinks
that slot's denominator non-randomly. This module prints per-judge, per-slot missingness so the
bias is visible rather than silently averaged away.

Writes `results/tables/judge_three_way.csv` (per-slot, per-judge coverage + consensus) and
`results/tables/judge_consensus_ceiling.csv`.

CLI:
    python -m src.judge_consensus --judges deepseek=results/tables/slot_judgements.csv \\
                                           luna=results/tables/xfam/slot_judgements.csv \\
                                           qwen=results/tables/qwen/slot_judgements.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.metrics import VERDICT_SCORE
from src.util import cfg_path, ensure_dir, get_logger, load_config

log = get_logger("sitrep.consensus")

KEY = ("event", "sitrep_id", "slot_id")
CATS = ("absent", "partial", "present")


def load(path: Path) -> dict[tuple[str, ...], dict[str, Any]]:
    """slot_judgements.csv -> {(event, sitrep_id, slot_id): row}, usable verdicts only."""
    if not path.is_file():
        raise FileNotFoundError(f"no judgement table at {path}")
    out: dict[tuple[str, ...], dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if (row.get("verdict") or "").strip().lower() in VERDICT_SCORE:
                out[tuple(row.get(k, "") for k in KEY)] = row
    return out


def fleiss_kappa(assignments: list[list[str]]) -> dict[str, float]:
    """Fleiss' kappa for N items x fixed number of raters over CATS.

    `assignments[i]` is the list of verdicts the raters gave item i.
    """
    n = len(assignments)
    if not n:
        raise ValueError("no items")
    r = len(assignments[0])
    if r < 2:
        raise ValueError("need at least two raters")
    idx = {c: i for i, c in enumerate(CATS)}
    col = [0] * len(CATS)
    agree = []
    for row in assignments:
        counts = [0] * len(CATS)
        for v in row:
            counts[idx[v]] += 1
        for i, c in enumerate(counts):
            col[i] += c
        agree.append((sum(c * c for c in counts) - r) / (r * (r - 1)))
    p_bar = sum(agree) / n
    p_e = sum((c / (n * r)) ** 2 for c in col)
    kappa = (p_bar - p_e) / (1 - p_e) if p_e < 1 else float("nan")
    return {"kappa": kappa, "observed": p_bar, "chance": p_e, "n": n, "raters": r}


def arm_of(row: dict[str, Any]) -> str:
    """Human sitreps form the ceiling; machine sitreps are grouped by generating arm."""
    if (row.get("kind") or "").lower() == "human":
        return "human"
    return f"{row.get('model', '')}/{row.get('arm', '')}"


def coverage(keys: list[tuple[str, ...]], verdict_of, row_of) -> dict[tuple[str, str], float]:
    """{(slot, arm): mean score} over the given keys."""
    acc: dict[tuple[str, str], list[float]] = defaultdict(list)
    for k in keys:
        acc[(k[2], arm_of(row_of(k)))].append(VERDICT_SCORE[verdict_of(k)])
    return {k: sum(v) / len(v) for k, v in acc.items()}


def ceiling(cov: dict[tuple[str, str], float], *, machine_max: float,
            human_min: float) -> list[dict[str, Any]]:
    """Slots where human coverage clears `human_min` and every machine arm sits at/below `machine_max`."""
    out = []
    for slot in sorted({s for s, _ in cov}):
        human = cov.get((slot, "human"))
        machine = [v for (s, a), v in cov.items() if s == slot and a != "human"]
        if human is None or not machine:
            continue
        m_max = max(machine)
        out.append({"slot_id": slot, "human_coverage": round(human, 4),
                    "machine_coverage_max": round(m_max, 4), "gap": round(human - m_max, 4),
                    "is_ceiling": bool(m_max <= machine_max and human >= human_min)})
    return out


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description="Three-judge consensus and Fleiss' kappa")
    ap.add_argument("--judges", nargs="+", required=True, metavar="NAME=CSV",
                    help="two or more judges as name=path/to/slot_judgements.csv; the first is "
                         "the primary and breaks ties")
    ap.add_argument("--machine-max", type=float, default=None)
    ap.add_argument("--human-min", type=float, default=None)
    ap.add_argument("--tables-dir", type=Path, default=None)
    args = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):  # pragma: no cover
        pass

    cfg = load_config()
    cl = cfg.get("ceiling") or {}
    machine_max = args.machine_max if args.machine_max is not None else float(cl.get("machine_max", 0.6))
    human_min = args.human_min if args.human_min is not None else float(cl.get("human_min", 0.8))
    tables = args.tables_dir or (cfg_path(cfg, "results", "results") / "tables")

    judges: dict[str, dict[tuple[str, ...], dict[str, Any]]] = {}
    for spec in args.judges:
        if "=" not in spec:
            ap.error(f"expected NAME=CSV, got {spec!r}")
        name, path = spec.split("=", 1)
        judges[name] = load(Path(path))
    if len(judges) < 2:
        ap.error("need at least two judges")
    names = list(judges)
    primary = names[0]

    common = sorted(set.intersection(*(set(j) for j in judges.values())))
    if not common:
        print("the judgement tables share no keys")
        return 2
    union = set().union(*(set(j) for j in judges.values()))
    print(f"judges: {', '.join(names)}   (primary/tiebreak: {primary})")
    print(f"complete cases: {len(common)} of {len(union)} keys seen ({len(common)/len(union):.1%})\n")

    print("missingness per judge (keys absent from that judge's table):")
    for n in names:
        miss = union - set(judges[n])
        by_slot = Counter(k[2] for k in miss)
        worst = ", ".join(f"{s}={c}" for s, c in by_slot.most_common(3)) or "-"
        print(f"  {n:10s} {len(miss):4d} missing   worst slots: {worst}")

    verdict = lambda n, k: judges[n][k]["verdict"].strip().lower()  # noqa: E731
    fk = fleiss_kappa([[verdict(n, k) for n in names] for k in common])
    print(f"\nFleiss' kappa ({fk['raters']} raters, {fk['n']} items): {fk['kappa']:.4f}"
          f"   observed {fk['observed']:.4f} / chance {fk['chance']:.4f}")
    unanimous = sum(1 for k in common if len({verdict(n, k) for n in names}) == 1)
    print(f"  unanimous on {unanimous}/{len(common)} ({unanimous/len(common):.1%})")

    # Majority vote; ties fall back to the primary judge so the rule is deterministic.
    consensus: dict[tuple[str, ...], str] = {}
    ties = 0
    for k in common:
        c = Counter(verdict(n, k) for n in names)
        top, cnt = c.most_common(1)[0]
        if cnt * 2 <= len(names):  # no strict majority
            ties += 1
            consensus[k] = verdict(primary, k)
        else:
            consensus[k] = top
    print(f"  ties broken by {primary}: {ties}")

    row_of = lambda k: judges[primary][k]  # noqa: E731
    per_judge = {n: coverage(common, lambda k, j=n: verdict(j, k), row_of) for n in names}
    cons_cov = coverage(common, lambda k: consensus[k], row_of)

    slots = sorted({s for s, _ in cons_cov})
    arms = sorted({a for _, a in cons_cov})
    out_rows = []
    for s in slots:
        for a in arms:
            if (s, a) not in cons_cov:
                continue
            row = {"slot_id": s, "arm": a, "consensus": round(cons_cov[(s, a)], 4)}
            for n in names:
                row[n] = round(per_judge[n].get((s, a), float("nan")), 4)
            out_rows.append(row)
    p1 = ensure_dir(tables) / "judge_three_way.csv"
    with open(p1, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["slot_id", "arm", *names, "consensus"], extrasaction="ignore")
        w.writeheader()
        w.writerows(out_rows)

    cons_ceiling = ceiling(cons_cov, machine_max=machine_max, human_min=human_min)
    p2 = tables / "judge_consensus_ceiling.csv"
    with open(p2, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["slot_id", "human_coverage", "machine_coverage_max",
                                           "gap", "is_ceiling"])
        w.writeheader()
        w.writerows(cons_ceiling)

    print(f"\nvisibility ceiling under consensus (machine<={machine_max}, human>={human_min}):")
    flagged = [c for c in cons_ceiling if c["is_ceiling"]]
    for c in flagged:
        print(f"  FLAGGED {c['slot_id']:24s} human={c['human_coverage']:.3f} "
              f"machineMax={c['machine_coverage_max']:.3f} gap={c['gap']:+.3f}")
    if not flagged:
        print("  (none)")

    print(f"\nper-judge gap on the flagged slots (effect size is the robust quantity, not the flag):")
    for c in flagged:
        s = c["slot_id"]
        parts = []
        for n in names:
            h = per_judge[n].get((s, "human"))
            m = [v for (sl, a), v in per_judge[n].items() if sl == s and a != "human"]
            parts.append(f"{n}={h - max(m):+.3f}" if h is not None and m else f"{n}=n/a")
        print(f"  {s:24s} {'  '.join(parts)}")

    print(f"\nwrote {p1}\nwrote {p2}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
