"""Phase 5 (PLAN §Phase 5) — does any standard metric track operational completeness?

Joins `results/tables/completeness_by_sitrep.csv` (Phase 4, `src/judge_slots.py` +
`src/metrics.py`) with
`results/tables/reference_metrics.csv` (Phase 5) on `sitrep_id` and reports, for every reference
metric, **Spearman rho and Kendall tau** at the sitrep level — per event and pooled — each with a
seeded percentile bootstrap CI (n=1000 by default, `config.yaml: seed`).

It then **mines the disagreements**: sitreps where a reference metric ranks high while operational
completeness ranks low (and the reverse). Those are the qualitative examples the paper needs, and
they are exported as **ids and scores only** — this module never opens a sitrep file, so no verbatim
sitrep or social-media text can reach `results/` (CLAUDE.md rule 7); `assert_no_free_text` enforces
that structurally at write time. The paper paraphrases the cases from the ids.

No LLM calls happen here, so there is nothing to cost-gate (`src.reference_metrics` does the gating).

CLI:
    python -m src.correlate --smoke 8      # tiny sample, 200 bootstraps, prints both tables
    python -m src.correlate                # full run -> correlations.csv + disagreements.csv
    python -m src.correlate --metrics rouge_l_f --top-k 5
"""
from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

from src.reference_metrics import assert_no_free_text, smoke_path, write_table
from src.util import cfg_path, get_logger, load_config

log = get_logger("sitrep.correlate")

# Reference-metric columns produced by src.reference_metrics, in report order.
DEFAULT_METRIC_COLUMNS = ("rouge_l_f", "bertscore_f1", "llm_judge_score")
# Completeness column candidates, most specific first (Phase 4 may name it either way).
COMPLETENESS_CANDIDATES = ("completeness", "operational_completeness", "completeness_weighted",
                           "mean_slot_score")
JOIN_KEY = "sitrep_id"
# Phase 4 writes completeness_by_sitrep.csv; `completeness.csv` is accepted as an older/manual name.
COMPLETENESS_TABLES = ("completeness_by_sitrep.csv", "completeness.csv")
DEFAULT_N_BOOTSTRAP = 1000
DEFAULT_TOP_K = 5
MIN_N = 3  # below this a rank correlation is meaningless

CORRELATION_COLUMNS = [
    "scope", "event", "metric", "coefficient", "rho", "p_value", "ci_lo", "ci_hi",
    "n", "n_bootstrap", "seed", "completeness_column",
]
DISAGREEMENT_COLUMNS = [
    "rank", "direction", "metric", "sitrep_id", "event", "date", "model", "arm",
    "metric_value", "metric_pct_rank", "completeness", "completeness_pct_rank", "gap", "n",
]
HIGH_REF_LOW_COMP = "high_reference_low_completeness"
LOW_REF_HIGH_COMP = "low_reference_high_completeness"


# ---------------------------------------------------------------------------
# correlation coefficients (thin, tested-against-scipy wrappers)
# ---------------------------------------------------------------------------
def _clean(x: Sequence[float], y: Sequence[float]) -> tuple[list[float], list[float]]:
    """Drop pairs where either side is missing/NaN (a skipped metric leaves blanks)."""
    xs: list[float] = []
    ys: list[float] = []
    for a, b in zip(x, y):
        if a is None or b is None:
            continue
        fa, fb = float(a), float(b)
        if math.isnan(fa) or math.isnan(fb):
            continue
        xs.append(fa)
        ys.append(fb)
    return xs, ys


def _degenerate(xs: Sequence[float], ys: Sequence[float]) -> bool:
    """True when a coefficient is undefined: too few pairs, or one side is constant."""
    return len(xs) < MIN_N or len(set(xs)) < 2 or len(set(ys)) < 2


def spearman(x: Sequence[float], y: Sequence[float]) -> tuple[float, float]:
    """Spearman rho and its two-sided p-value; (nan, nan) when undefined."""
    xs, ys = _clean(x, y)
    if _degenerate(xs, ys):
        return math.nan, math.nan
    from scipy import stats

    res = stats.spearmanr(xs, ys)
    return float(res.statistic), float(res.pvalue)


def kendall(x: Sequence[float], y: Sequence[float]) -> tuple[float, float]:
    """Kendall tau-b and its two-sided p-value; (nan, nan) when undefined."""
    xs, ys = _clean(x, y)
    if _degenerate(xs, ys):
        return math.nan, math.nan
    from scipy import stats

    res = stats.kendalltau(xs, ys)
    return float(res.statistic), float(res.pvalue)


COEFFICIENTS: dict[str, Callable[[Sequence[float], Sequence[float]], tuple[float, float]]] = {
    "spearman": spearman,
    "kendall": kendall,
}


def bootstrap_ci(x: Sequence[float], y: Sequence[float],
                 fn: Callable[[Sequence[float], Sequence[float]], tuple[float, float]],
                 *, n_bootstrap: int = DEFAULT_N_BOOTSTRAP, seed: int = 17,
                 alpha: float = 0.05) -> tuple[float, float]:
    """Seeded percentile bootstrap CI for a rank correlation (**pairs** resampled with replacement).

    Deterministic for a given (x, y, fn, n_bootstrap, seed). Uses `random.Random` rather than a
    numpy generator for the same reason `src.metrics.bootstrap_ci` does — no RNG version drift
    across machines — and resamples index tuples, since a rank correlation is a paired statistic
    and `metrics.bootstrap_ci` bootstraps a single sample of values. Resamples that come out
    degenerate (every x or y tied) are dropped; (nan, nan) if too few usable ones remain.
    """
    xs, ys = _clean(x, y)
    if _degenerate(xs, ys) or n_bootstrap < 1:
        return math.nan, math.nan
    rng = random.Random(seed)
    n = len(xs)
    stats_: list[float] = []
    for _ in range(int(n_bootstrap)):
        idx = [rng.randrange(n) for _ in range(n)]
        r, _p = fn([xs[i] for i in idx], [ys[i] for i in idx])
        if not math.isnan(r):
            stats_.append(r)
    if len(stats_) < max(2, n_bootstrap // 10):
        log.warning("bootstrap CI unavailable: only %d/%d resamples were non-degenerate",
                    len(stats_), n_bootstrap)
        return math.nan, math.nan
    stats_.sort()
    return _percentile(stats_, alpha / 2), _percentile(stats_, 1 - alpha / 2)


def _percentile(sorted_vals: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile of an already-sorted sample (same rule as src.metrics)."""
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = q * (len(sorted_vals) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return float(sorted_vals[int(pos)])
    return float(sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo))


def percentile_ranks(values: Sequence[float]) -> list[float]:
    """Average-rank percentiles in [0, 1] (ties share a rank); a single value maps to 0.5."""
    if not values:
        return []
    if len(values) == 1:
        return [0.5]
    from scipy import stats

    ranks = stats.rankdata(list(values), method="average")
    return [float((r - 1.0) / (len(values) - 1.0)) for r in ranks]


# ---------------------------------------------------------------------------
# loading + joining the two Phase 4/5 tables (never the sitreps themselves)
# ---------------------------------------------------------------------------
def _default_completeness_path(tables_dir: Path) -> Path:
    """First existing Phase 4 completeness table (falls back to the canonical name for the error)."""
    for name in COMPLETENESS_TABLES:
        if (tables_dir / name).exists():
            return tables_dir / name
    return tables_dir / COMPLETENESS_TABLES[0]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read a results CSV into dicts; missing file -> empty list (logged)."""
    if not Path(path).exists():
        log.warning("table not found: %s", path)
        return []
    with open(path, "r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def pick_completeness_column(rows: Sequence[dict[str, Any]], requested: str | None = None) -> str | None:
    """Resolve which column holds operational completeness (Phase 4 may name it several ways)."""
    if not rows:
        return requested
    cols = set(rows[0])
    if requested:
        return requested if requested in cols else None
    for cand in COMPLETENESS_CANDIDATES:
        if cand in cols:
            return cand
    return None


def load_completeness(path: Path) -> list[dict[str, str]]:
    """Load the Phase 4 completeness table, preferring `src.metrics` if it exposes a loader.

    Phase 4 owns the definition of operational completeness; this keeps the coupling to the
    committed CSV while letting `src/metrics.py` take over the moment it lands.
    """
    try:
        from src import metrics as _metrics  # type: ignore[attr-defined]
    except ImportError:
        _metrics = None  # Phase 4 not built yet
    loader = getattr(_metrics, "load_completeness", None) if _metrics is not None else None
    if callable(loader):
        log.info("using src.metrics.load_completeness for %s", path)
        return list(loader(path))
    return read_csv_rows(path)


def join_tables(completeness_rows: Sequence[dict[str, Any]], reference_rows: Sequence[dict[str, Any]],
                *, completeness_column: str, metric_columns: Sequence[str]) -> list[dict[str, Any]]:
    """Inner-join on `sitrep_id`; returns numeric rows carrying ids/labels and scores only."""
    by_id = {str(r.get(JOIN_KEY, "")): r for r in completeness_rows if r.get(JOIN_KEY)}
    out: list[dict[str, Any]] = []
    for ref in reference_rows:
        sid = str(ref.get(JOIN_KEY, ""))
        comp_row = by_id.get(sid)
        if comp_row is None:
            continue
        comp = _to_float(comp_row.get(completeness_column))
        if comp is None:
            continue
        row: dict[str, Any] = {
            JOIN_KEY: sid,
            "event": str(ref.get("event") or comp_row.get("event") or ""),
            "date": str(ref.get("date") or ""),
            "model": str(ref.get("model") or comp_row.get("model") or ""),
            "arm": str(ref.get("arm") or comp_row.get("arm") or ""),
            "completeness": comp,
        }
        for col in metric_columns:
            row[col] = _to_float(ref.get(col))
        out.append(row)
    dropped = len(reference_rows) - len(out)
    if dropped:
        log.warning("%d reference row(s) had no usable completeness score and were dropped", dropped)
    return out


def available_metric_columns(reference_rows: Sequence[dict[str, Any]],
                             requested: Sequence[str] | None = None) -> list[str]:
    """Metric columns that exist in the reference table (a skipped metric is simply absent)."""
    cols = set(reference_rows[0]) if reference_rows else set()
    wanted = list(requested) if requested else list(DEFAULT_METRIC_COLUMNS)
    present = [c for c in wanted if c in cols]
    for missing in [c for c in wanted if c not in cols]:
        log.warning("metric column %r is not in the reference table; skipping it", missing)
    return present


# ---------------------------------------------------------------------------
# the correlation table
# ---------------------------------------------------------------------------
def _fmt(x: float, nd: int = 4) -> str | float:
    return "" if x is None or math.isnan(x) else round(float(x), nd)


def correlate(rows: Sequence[dict[str, Any]], *, metric_columns: Sequence[str],
              n_bootstrap: int = DEFAULT_N_BOOTSTRAP, seed: int = 17,
              completeness_column: str = "completeness",
              per_event: bool = True) -> list[dict[str, Any]]:
    """Spearman + Kendall of completeness vs each reference metric, pooled and per event."""
    scopes: list[tuple[str, str, list[dict[str, Any]]]] = [("pooled", "", list(rows))]
    if per_event:
        events = sorted({str(r.get("event") or "") for r in rows})
        scopes += [("event", ev, [r for r in rows if str(r.get("event") or "") == ev]) for ev in events]
    out: list[dict[str, Any]] = []
    for scope, event, subset in scopes:
        comp = [r["completeness"] for r in subset]
        for metric in metric_columns:
            vals = [r.get(metric) for r in subset]
            xs, ys = _clean(comp, vals)
            for name, fn in COEFFICIENTS.items():
                rho, p = fn(comp, vals)
                lo, hi = bootstrap_ci(comp, vals, fn, n_bootstrap=n_bootstrap, seed=seed)
                out.append({
                    "scope": scope, "event": event, "metric": metric, "coefficient": name,
                    "rho": _fmt(rho), "p_value": _fmt(p, 6), "ci_lo": _fmt(lo), "ci_hi": _fmt(hi),
                    "n": len(xs), "n_bootstrap": n_bootstrap, "seed": seed,
                    "completeness_column": completeness_column,
                })
    return out


# ---------------------------------------------------------------------------
# disagreement mining (ids and scores only — never text)
# ---------------------------------------------------------------------------
def mine_disagreements(rows: Sequence[dict[str, Any]], *, metric_columns: Sequence[str],
                       top_k: int = DEFAULT_TOP_K) -> list[dict[str, Any]]:
    """Largest rank disagreements per metric: high reference/low completeness, and the reverse.

    Ranking is on percentile ranks, so metrics on different scales are comparable. Emits at most
    `top_k` rows per (metric, direction), ordered by the size of the gap.
    """
    out: list[dict[str, Any]] = []
    for metric in metric_columns:
        usable = [r for r in rows if r.get(metric) is not None and r.get("completeness") is not None]
        if len(usable) < 2:
            log.warning("not enough scored sitreps to mine disagreements for %r (n=%d)", metric, len(usable))
            continue
        m_pct = percentile_ranks([float(r[metric]) for r in usable])
        c_pct = percentile_ranks([float(r["completeness"]) for r in usable])
        scored = [
            {"row": r, "m_pct": m, "c_pct": c, "gap": m - c}
            for r, m, c in zip(usable, m_pct, c_pct)
        ]
        # The sign of the gap *is* the direction, so each list is filtered before it is truncated:
        # with a small n and a generous top_k, an unfiltered "top k" would otherwise pad the
        # high-reference list with rows that actually disagree the other way (or not at all).
        for direction, ordered in (
            (HIGH_REF_LOW_COMP, sorted((s for s in scored if s["gap"] > 0),
                                       key=lambda s: (-s["gap"], str(s["row"][JOIN_KEY])))),
            (LOW_REF_HIGH_COMP, sorted((s for s in scored if s["gap"] < 0),
                                       key=lambda s: (s["gap"], str(s["row"][JOIN_KEY])))),
        ):
            for rank, s in enumerate(ordered[:top_k], start=1):
                r = s["row"]
                out.append({
                    "rank": rank, "direction": direction, "metric": metric,
                    JOIN_KEY: str(r[JOIN_KEY]), "event": str(r.get("event") or ""),
                    "date": str(r.get("date") or ""), "model": str(r.get("model") or ""),
                    "arm": str(r.get("arm") or ""),
                    "metric_value": _fmt(float(r[metric])), "metric_pct_rank": _fmt(s["m_pct"], 3),
                    "completeness": _fmt(float(r["completeness"])),
                    "completeness_pct_rank": _fmt(s["c_pct"], 3), "gap": _fmt(s["gap"], 3),
                    "n": len(usable),
                })
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _print_table(title: str, rows: Sequence[dict[str, Any]], columns: Sequence[str]) -> None:
    print(f"\n{title}")
    print(" | ".join(columns))
    for r in rows:
        print(" | ".join(str(r.get(c, "")) for c in columns))


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description="Phase 5: correlate reference metrics with completeness")
    ap.add_argument("--completeness", type=Path, default=None,
                    help="Phase 4 table (default: the first of "
                         f"{list(COMPLETENESS_TABLES)} that exists in results/tables/)")
    ap.add_argument("--reference-metrics", type=Path, default=None,
                    help="Phase 5 table (default results/tables/reference_metrics.csv)")
    ap.add_argument("--completeness-column", default=None,
                    help=f"column holding operational completeness (default: first of {list(COMPLETENESS_CANDIDATES)})")
    ap.add_argument("--metrics", default=None,
                    help=f"comma-separated reference-metric columns (default {list(DEFAULT_METRIC_COLUMNS)})")
    ap.add_argument("--n-bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP, help="bootstrap resamples")
    ap.add_argument("--seed", type=int, default=None, help="override config seed")
    ap.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="disagreements per metric per direction")
    ap.add_argument("--smoke", type=int, default=0, metavar="N",
                    help="use only the first N joined sitreps, 200 bootstraps, and print both tables")
    ap.add_argument("--out-correlations", type=Path, default=None)
    ap.add_argument("--out-disagreements", type=Path, default=None)
    args = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):  # pragma: no cover
        pass

    cfg = load_config()
    seed = args.seed if args.seed is not None else int(cfg.get("seed", 17))
    tables = cfg_path(cfg, "results", "results") / "tables"
    comp_path = args.completeness or _default_completeness_path(tables)
    ref_path = args.reference_metrics or tables / "reference_metrics.csv"
    corr_out = args.out_correlations or tables / "correlations.csv"
    dis_out = args.out_disagreements or tables / "disagreements.csv"

    # Print the resolved inputs. The default picks whichever completeness table exists first in
    # results/tables/, which is not necessarily the judge the paper reports — a silent
    # misattribution is exactly the failure this project has already hit three times.
    print(f"completeness table: {comp_path}")
    print(f"reference metrics : {ref_path}")

    comp_rows = load_completeness(comp_path)
    ref_rows = read_csv_rows(ref_path)
    if not comp_rows:
        print(f"No completeness scores at {comp_path} — run Phase 4 "
              f"(`python -m src.judge_slots`) first. Nothing to correlate.")
        return 2
    if not ref_rows:
        print(f"No reference metrics at {ref_path} — run `python -m src.reference_metrics` first.")
        return 2
    comp_col = pick_completeness_column(comp_rows, args.completeness_column)
    if not comp_col:
        print(f"{comp_path} has no completeness column (looked for "
              f"{list(COMPLETENESS_CANDIDATES)}; use --completeness-column)")
        return 2

    metric_cols = available_metric_columns(
        ref_rows, [m.strip() for m in args.metrics.split(",") if m.strip()] if args.metrics else None)
    if not metric_cols:
        print(f"{ref_path} carries none of the expected reference-metric columns")
        return 2
    rows = join_tables(comp_rows, ref_rows, completeness_column=comp_col, metric_columns=metric_cols)
    if len(rows) < MIN_N:
        print(f"only {len(rows)} sitrep(s) have both completeness and reference scores — "
              f"a rank correlation needs at least {MIN_N}.")
        return 2

    n_boot = args.n_bootstrap
    if args.smoke:
        rows = rows[: args.smoke]
        n_boot = min(n_boot, 200)
        if args.out_correlations is None:
            corr_out = smoke_path(corr_out)
        if args.out_disagreements is None:
            dis_out = smoke_path(dis_out)

    corr_rows = correlate(rows, metric_columns=metric_cols, n_bootstrap=n_boot, seed=seed,
                          completeness_column=comp_col)
    dis_rows = mine_disagreements(rows, metric_columns=metric_cols, top_k=args.top_k)
    assert_no_free_text(dis_rows, DISAGREEMENT_COLUMNS)  # belt and braces before either write
    write_table(corr_out, corr_rows, CORRELATION_COLUMNS)
    write_table(dis_out, dis_rows, DISAGREEMENT_COLUMNS)

    print(f"joined {len(rows)} sitrep(s) on {JOIN_KEY}; completeness column = {comp_col!r}")
    print(f"metrics: {metric_cols}; bootstrap n={n_boot}, seed={seed}")
    print(f"correlations:  {corr_out} ({len(corr_rows)} rows)")
    print(f"disagreements: {dis_out} ({len(dis_rows)} rows, ids and scores only)")
    if args.smoke:
        _print_table("correlations", corr_rows,
                     ["scope", "event", "metric", "coefficient", "rho", "p_value", "ci_lo", "ci_hi", "n"])
        _print_table("disagreements", dis_rows,
                     ["direction", "metric", "sitrep_id", "metric_value", "completeness", "gap"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
