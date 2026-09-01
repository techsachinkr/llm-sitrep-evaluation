"""Phase 6 figures, built only from the committed result tables.

PLAN's Phase 6 asks for three figures:

* **(a) slot-coverage heatmap** — slots x {human ceiling, model x arm}. Shows at a glance that
  machine sitreps address nearly every information type as well as the official record.
* **(b) completeness bars with CIs** — per arm, with the human ceiling marked. The bootstrap CIs
  come from the Phase 4 tables, not recomputed here.
* **(c) correlation scatter** — each reference metric against operational completeness, with the
  Spearman rho annotated. This is the figure that carries the paper's third contribution.

Everything is read from `results/tables/`; nothing is recomputed from raw judgements, so a figure
can never disagree with the number in RESULTS.md. Matplotlib only (no seaborn), default colours,
and no network access.

Figures are written to `results/figures/` as both PDF (for LaTeX) and PNG (for quick viewing).

CLI:
    python -m src.figures                       # all three, from results/tables
    python -m src.figures --tables-dir results/tables/idai/gemini --suffix _gemini
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.util import cfg_path, ensure_dir, get_logger, load_config

log = get_logger("sitrep.figures")

HUMAN = "human"
#: Reference metrics get friendly axis labels; anything else falls back to its column name.
METRIC_LABELS = {
    "rouge_l_f": "ROUGE-L F1",
    "bertscore_f1": "BERTScore F1",
    "llm_judge_score": "Generic LLM judge (1-10)",
}


def read_rows(path: Path) -> list[dict[str, Any]]:
    """CSV -> list of dicts; empty list when the file is absent."""
    if not path.is_file():
        log.warning("missing %s", path)
        return []
    with open(path, "r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _f(row: dict[str, Any], key: str) -> float | None:
    """Parse a float cell, tolerating blanks."""
    raw = (row.get(key) or "").strip()
    try:
        return float(raw)
    except ValueError:
        return None


def _spread_duplicate_points(xs: list[float], ys: list[float], *,
                             x_step: float = 0.08,
                             y_step: float = 0.012) -> tuple[list[float], list[float]]:
    """Separate exact collisions without moving any marker below its measured value."""
    groups: dict[tuple[float, float], list[int]] = defaultdict(list)
    for i, point in enumerate(zip(xs, ys)):
        groups[point].append(i)

    spread_xs, spread_ys = list(xs), list(ys)
    for indices in groups.values():
        if len(indices) < 2:
            continue
        columns = min(5, math.ceil(math.sqrt(len(indices))))
        for position, index in enumerate(indices):
            row, column = divmod(position, columns)
            row_size = min(columns, len(indices) - row * columns)
            spread_xs[index] += (column - (row_size - 1) / 2) * x_step
            spread_ys[index] += row * y_step
    return spread_xs, spread_ys


# ---------------------------------------------------------------------------
# (a) slot-coverage heatmap
# ---------------------------------------------------------------------------
def figure_slot_heatmap(tables: Path, out: Path) -> Path | None:
    """Slots (rows) x arms (columns), cell = coverage. Human column drawn first."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    rows = read_rows(tables / "slot_coverage.csv")
    if not rows:
        return None
    cov: dict[tuple[str, str], float] = {}
    for r in rows:
        v = _f(r, "coverage")
        if v is not None:
            cov[(r["slot_id"], r["arm"])] = v
    slots = sorted({s for s, _ in cov})
    arms = sorted({a for _, a in cov if a != HUMAN})
    cols = [HUMAN] + arms
    # Order slots by the human-machine gap so the ceiling slots sit at the top.
    def gap(s: str) -> float:
        h = cov.get((s, HUMAN), 0.0)
        m = [cov[(s, a)] for a in arms if (s, a) in cov]
        return h - max(m) if m else 0.0
    slots.sort(key=gap, reverse=True)

    grid = np.full((len(slots), len(cols)), np.nan)
    for i, s in enumerate(slots):
        for j, a in enumerate(cols):
            if (s, a) in cov:
                grid[i, j] = cov[(s, a)]

    fig, ax = plt.subplots(figsize=(1.6 + 1.15 * len(cols), 0.42 * len(slots) + 1.8))
    im = ax.imshow(grid, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(["human\n(ceiling)" if c == HUMAN else c.replace("/", "\n") for c in cols],
                       fontsize=8)
    ax.set_yticks(range(len(slots)))
    ax.set_yticklabels([s.replace("_", " ") for s in slots], fontsize=8)
    for i in range(len(slots)):
        for j in range(len(cols)):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if grid[i, j] < 0.6 else "black")
    ax.axvline(0.5, color="white", linewidth=2)  # separate the ceiling from the machine arms
    ax.set_title("Slot coverage: human ceiling vs machine arms\n"
                 "(rows ordered by human-machine gap)", fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.7, label="coverage (present=1, partial=0.5)")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out.with_suffix(".pdf")


# ---------------------------------------------------------------------------
# (b) completeness bars with CIs
# ---------------------------------------------------------------------------
def figure_completeness_bars(tables: Path, out: Path) -> Path | None:
    """One bar per arm with its bootstrap CI; the human ceiling drawn as a rule."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = read_rows(tables / "completeness.csv")
    if not rows:
        return None
    recs = []
    for r in rows:
        m = _f(r, "completeness")
        if m is None:
            continue
        recs.append({"arm": r.get("arm", "?"), "mean": m,
                     "lo": _f(r, "ci_lo"), "hi": _f(r, "ci_hi"),
                     "human": (r.get("arm") == HUMAN
                               or str(r.get("is_human_ceiling", "")).lower() in ("true", "1"))})
    if not recs:
        return None
    recs.sort(key=lambda d: (not d["human"], -d["mean"]))
    labels = [r["arm"] for r in recs]
    means = [r["mean"] for r in recs]
    lo = [r["mean"] - (r["lo"] if r["lo"] is not None else r["mean"]) for r in recs]
    hi = [(r["hi"] if r["hi"] is not None else r["mean"]) - r["mean"] for r in recs]
    colors = ["#444444" if r["human"] else "#4C72B0" for r in recs]

    fig, ax = plt.subplots(figsize=(1.4 + 1.05 * len(recs), 4.2))
    ax.bar(range(len(recs)), means, yerr=[lo, hi], capsize=4, color=colors)
    ceiling = next((r["mean"] for r in recs if r["human"]), None)
    if ceiling is not None:
        ax.axhline(ceiling, color="#444444", linestyle="--", linewidth=1,
                   label=f"human ceiling = {ceiling:.3f}")
        ax.legend(fontsize=8, loc="lower right")
    ax.set_xticks(range(len(recs)))
    ax.set_xticklabels([l.replace("/", "\n") for l in labels], fontsize=8)
    ax.set_ylabel("operational completeness")
    ax.set_ylim(0, 1.0)
    ax.set_title("Operational completeness by arm (95% bootstrap CI)", fontsize=10)
    for i, (m, rec) in enumerate(zip(means, recs)):
        # Place values above the confidence interval, not on its upper cap.
        interval_top = rec["hi"] if rec["hi"] is not None else m
        ax.text(i, max(m + 0.02, interval_top + 0.015), f"{m:.3f}", ha="center", fontsize=7)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out.with_suffix(".pdf")


# ---------------------------------------------------------------------------
# (c) correlation scatter
# ---------------------------------------------------------------------------
def figure_correlation_scatter(tables: Path, out: Path, *, completeness: Path | None = None,
                               reference: Path | None = None,
                               correlations: Path | None = None) -> Path | None:
    """One panel per reference metric: metric (x) vs completeness (y), rho annotated."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ref_rows = read_rows(reference or (tables / "reference_metrics.csv"))
    comp_rows = read_rows(completeness or (tables / "completeness_by_sitrep.csv"))
    if not ref_rows or not comp_rows:
        return None
    comp = {r["sitrep_id"]: _f(r, "completeness") for r in comp_rows if r.get("sitrep_id")}
    rho = {}
    for r in read_rows(correlations or (tables / "correlations.csv")):
        if r.get("scope") == "pooled" and r.get("coefficient") == "spearman":
            rho[r["metric"]] = (_f(r, "rho"), _f(r, "p_value"))

    metrics = [m for m in METRIC_LABELS if any((r.get(m) or "").strip() for r in ref_rows)]
    if not metrics:
        return None
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.1 * len(metrics), 3.9), squeeze=False)
    for ax, m in zip(axes[0], metrics):
        xs, ys = [], []
        for r in ref_rows:
            x = _f(r, m)
            y = comp.get(r.get("sitrep_id", ""))
            if x is not None and y is not None:
                xs.append(x)
                ys.append(y)
        plot_xs, plot_ys = _spread_duplicate_points(xs, ys)
        ax.scatter(plot_xs, plot_ys, s=16, alpha=0.65, edgecolors="white",
                   linewidths=0.25)
        ax.set_xlabel(METRIC_LABELS.get(m, m), fontsize=9)
        ax.set_ylabel("operational completeness", fontsize=9)
        if m in rho and rho[m][0] is not None:
            r_, p_ = rho[m]
            ax.set_title(f"rho = {r_:+.3f}   p = {p_:.3g}"
                         + ("  (n.s.)" if (p_ or 1) >= 0.05 else ""), fontsize=9)
        ax.grid(alpha=0.25, linewidth=0.5)
    fig.suptitle("Do reference metrics track operational completeness?", fontsize=11)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out.with_suffix(".pdf")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description="Phase 6 figures, from the committed result tables")
    ap.add_argument("--tables-dir", type=Path, default=None)
    ap.add_argument("--figures-dir", type=Path, default=None)
    ap.add_argument("--suffix", default="", help="appended to each figure filename")
    ap.add_argument("--completeness", type=Path, default=None)
    ap.add_argument("--reference-metrics", type=Path, default=None)
    ap.add_argument("--correlations", type=Path, default=None)
    args = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):  # pragma: no cover
        pass

    cfg = load_config()
    tables = args.tables_dir or (cfg_path(cfg, "results", "results") / "tables")
    figs = ensure_dir(args.figures_dir or (cfg_path(cfg, "results", "results") / "figures"))
    sfx = args.suffix

    made = []
    a = figure_slot_heatmap(tables, figs / f"fig_a_slot_coverage{sfx}")
    b = figure_completeness_bars(tables, figs / f"fig_b_completeness{sfx}")
    c = figure_correlation_scatter(tables, figs / f"fig_c_correlation{sfx}",
                                   completeness=args.completeness,
                                   reference=args.reference_metrics,
                                   correlations=args.correlations)
    for name, p in (("(a) slot-coverage heatmap", a), ("(b) completeness bars", b),
                    ("(c) correlation scatter", c)):
        if p:
            made.append(p)
            print(f"  {name:28s} -> {p}")
        else:
            print(f"  {name:28s} -- skipped (inputs missing under {tables})")
    if not made:
        print("no figures produced; check that the Phase 4/5 tables exist")
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
