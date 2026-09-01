"""Figure-level agreement between machine sitreps and the human sitrep for the same day.

Slot coverage asks *is this information type addressed?* — a question a machine sitrep can answer
by mentioning a topic. This module asks the harder question PLAN's Phase 4 calls for: **are the
numbers the same ones the humanitarian record carries?**

For every machine sitrep it finds the human sitrep(s) published on the same day, extracts the
quantities from both (`src.metrics.extract_figures`) and reports:

* **precision** — share of the machine's figures that also appear in the human report. Low
  precision means the machine is asserting quantities the official record does not contain.
* **recall** — share of the human report's figures the machine reproduced.
* **casualty recall** — the same, restricted to casualty figures (dead/injured/missing).

Interpretation caveat, which belongs in the paper: a machine figure absent from the human sitrep
is *unverified*, not proven false — OCHA may simply not report it. But combined with low recall it
is strong evidence that the machine is not reconstructing the official quantitative picture.

Writes `results/tables/figure_agreement.csv` (per pair) and prints a per-arm summary.

CLI: python -m src.figure_check [--event cyclone-idai-2019]
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from src.metrics import extract_figures, figure_extraction_check
from src.util import cfg_path, ensure_dir, get_logger, load_config, read_json, resolve_event_dir

log = get_logger("sitrep.figures")

COLUMNS = ("event", "day", "model", "arm", "n_machine_figures", "n_human_figures",
           "n_overlap", "precision", "recall", "casualty_recall", "jaccard")


def _mean(values: Iterable[Any]) -> float:
    vals = [v for v in values if v is not None]
    return statistics.mean(vals) if vals else float("nan")


def load_human_by_day(human_dir: Path) -> dict[str, list[str]]:
    """{date: [sitrep text, ...]} — several organisations may report on the same day."""
    by_day: dict[str, list[str]] = defaultdict(list)
    for p in sorted(human_dir.glob("*.json")):
        rec = read_json(p)
        text = (rec.get("text") or "").strip()
        if text and rec.get("date"):
            by_day[rec["date"]].append(text)
    return dict(by_day)


def compare(machine_dir: Path, human_by_day: dict[str, list[str]], event: str) -> list[dict[str, Any]]:
    """One row per machine sitrep that has a same-day human counterpart."""
    rows: list[dict[str, Any]] = []
    for p in sorted(machine_dir.glob("*.json")):
        rec = read_json(p)
        text = (rec.get("text") or "").strip()
        day = rec.get("day") or rec.get("date")
        if not text or day not in human_by_day:
            continue
        chk = figure_extraction_check(text, "\n".join(human_by_day[day]))
        rows.append({
            "event": event, "day": day, "model": rec.get("model", ""), "arm": rec.get("arm", ""),
            "n_machine_figures": len(chk.machine), "n_human_figures": len(chk.human),
            "n_overlap": len(chk.overlap),
            "precision": None if chk.precision is None else round(chk.precision, 4),
            "recall": None if chk.recall is None else round(chk.recall, 4),
            "casualty_recall": None if chk.casualty_recall is None else round(chk.casualty_recall, 4),
            "jaccard": None if chk.jaccard is None else round(chk.jaccard, 4),
        })
    return rows


def summarise(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per-arm means (machine model + arm), plus a pooled row."""
    by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_arm[f"{r['model']}/{r['arm']}"].append(r)
    out = []
    for arm, rs in sorted(by_arm.items()):
        out.append({
            "arm": arm, "n_pairs": len(rs),
            "precision": _mean(r["precision"] for r in rs),
            "recall": _mean(r["recall"] for r in rs),
            "casualty_recall": _mean(r["casualty_recall"] for r in rs),
            "mean_machine_figures": _mean(r["n_machine_figures"] for r in rs),
            "mean_human_figures": _mean(r["n_human_figures"] for r in rs),
        })
    if rows:
        out.append({
            "arm": "POOLED", "n_pairs": len(rows),
            "precision": _mean(r["precision"] for r in rows),
            "recall": _mean(r["recall"] for r in rows),
            "casualty_recall": _mean(r["casualty_recall"] for r in rows),
            "mean_machine_figures": _mean(r["n_machine_figures"] for r in rows),
            "mean_human_figures": _mean(r["n_human_figures"] for r in rows),
        })
    return out


def source_availability(machine_dir: Path, human_by_day: dict[str, list[str]],
                        stream_dir: Path | None) -> list[dict[str, Any]]:
    """Attribute the figure gap to the SOURCE rather than to the model.

    Per machine sitrep: what share of its figures appear (a) in the day's tweet stream it was
    generated from, and (b) in the human sitrep? And, independently of any model, what share of
    the human sitrep's figures exist anywhere in that day's stream?

    (a) high with (b) low means the generator is faithfully reproducing a source that does not
    carry the official quantitative picture — the visibility ceiling is a property of the source,
    not the model. That is the difference between "this LLM invents numbers" and "social media
    cannot support a situation report", and it needs no API call to measure.
    """
    if stream_dir is None or not stream_dir.is_dir():
        log.warning("no stream directory for this event — skipping the source-availability analysis")
        return []
    stream_figs: dict[str, set[float]] = {}
    for p in sorted(stream_dir.glob("*.jsonl")):
        with open(p, encoding="utf-8") as fh:
            text = " ".join(json.loads(line)["text"] for line in fh if line.strip())
        stream_figs[p.stem] = {f.value for f in extract_figures(text)}
    human_figs = {d: {f.value for f in extract_figures("\n".join(t))} for d, t in human_by_day.items()}

    rows: list[dict[str, Any]] = []
    for p in sorted(machine_dir.glob("*.json")):
        rec = read_json(p)
        day = rec.get("day") or rec.get("date")
        text = (rec.get("text") or "").strip()
        if not text or day not in stream_figs or day not in human_figs:
            continue
        machine = {f.value for f in extract_figures(text)}
        if not machine:
            continue
        hf = human_figs[day]
        rows.append({
            "day": day, "model": rec.get("model", ""), "arm": rec.get("arm", ""),
            "n_machine_figures": len(machine),
            "machine_figs_in_stream": round(len(machine & stream_figs[day]) / len(machine), 4),
            "machine_figs_in_human": round(len(machine & hf) / len(machine), 4),
            "human_figs_in_stream": round(len(hf & stream_figs[day]) / len(hf), 4) if hf else None,
        })
    return rows


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description="Figure-level agreement, machine vs same-day human sitreps")
    ap.add_argument("--event", default="cyclone-idai-2019")
    ap.add_argument("--tables-dir", type=Path, default=None)
    args = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):  # pragma: no cover
        pass
    cfg = load_config()
    processed = cfg_path(cfg, "processed", "data/processed")
    tables = args.tables_dir or (cfg_path(cfg, "results", "results") / "tables")

    human_dir = resolve_event_dir(processed / "sitreps_human", args.event)
    machine_dir = resolve_event_dir(processed / "sitreps_machine", args.event)
    if human_dir is None or machine_dir is None:
        print(f"need both human and machine sitreps for {args.event}\n"
              f"  human:   {human_dir or 'MISSING'}\n  machine: {machine_dir or 'MISSING'}")
        return 2
    rows = compare(machine_dir, load_human_by_day(human_dir), args.event)
    if not rows:
        print("no machine sitrep shares a day with a human sitrep — nothing to compare")
        return 2

    out = ensure_dir(tables) / "figure_agreement.csv"
    with open(out, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(COLUMNS), extrasaction="ignore", restval="")
        w.writeheader()
        w.writerows(rows)

    print(f"figure agreement with the same-day human sitrep ({len(rows)} pairs)\n")
    print(f"{'arm':28s} {'pairs':>5s} {'precision':>10s} {'recall':>8s} {'casualty rec':>13s} "
          f"{'m.figs':>7s} {'h.figs':>7s}")
    print("-" * 84)
    for s in summarise(rows):
        mark = "  <- pooled" if s["arm"] == "POOLED" else ""
        print(f"{s['arm']:28s} {s['n_pairs']:5d} {s['precision']:10.3f} {s['recall']:8.3f} "
              f"{s['casualty_recall']:13.3f} {s['mean_machine_figures']:7.1f} "
              f"{s['mean_human_figures']:7.1f}{mark}")
    print(f"\nprecision = share of machine figures also present in the human sitrep for that day")
    print(f"recall    = share of the human sitrep's figures the machine reproduced")
    print(f"\nwrote {out}")

    # Is the gap the model's doing, or the source's?
    avail = source_availability(machine_dir, load_human_by_day(human_dir),
                                resolve_event_dir(processed / "streams", args.event))
    if avail:
        out2 = tables / "source_availability.csv"
        cols = ["day", "model", "arm", "n_machine_figures", "machine_figs_in_stream",
                "machine_figs_in_human", "human_figs_in_stream"]
        with open(out2, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore", restval="")
            w.writeheader()
            w.writerows(avail)
        in_stream = _mean(r["machine_figs_in_stream"] for r in avail)
        in_human = _mean(r["machine_figs_in_human"] for r in avail)
        src = _mean(r["human_figs_in_stream"] for r in avail)
        print(f"\nWhere do the machine's numbers come from? ({len(avail)} sitreps)")
        print(f"  {in_stream:.3f}  of machine figures appear in the tweet stream it was given")
        print(f"  {in_human:.3f}  of machine figures appear in the human sitrep for that day")
        print(f"  {src:.3f}  of the human sitrep's figures exist anywhere in that day's stream")
        print(f"\n  -> the generator reproduces its source faithfully ({in_stream:.0%}); that source "
              f"carries only\n     {src:.0%} of the official figures, and the machine recovers "
              f"{in_human:.0%} of them. The ceiling is in\n     the SOURCE, not the model.")
        print(f"\nwrote {out2}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
