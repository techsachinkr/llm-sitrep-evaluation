"""Figures must be built from the committed tables, never recomputed.

A figure that recomputes its own numbers can silently disagree with RESULTS.md. These tests pin
the contract: given only result CSVs, each figure function writes a PDF; given nothing, it returns
None rather than raising or writing an empty file.
"""
from __future__ import annotations

import csv
from pathlib import Path

from src.figures import (_spread_duplicate_points, figure_completeness_bars,
                         figure_correlation_scatter, figure_slot_heatmap, read_rows)


def _write(path: Path, rows: list[dict], cols: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def test_read_rows_missing_file_is_empty(tmp_path):
    assert read_rows(tmp_path / "nope.csv") == []


def test_duplicate_scatter_points_spread_upward_only():
    xs, ys = _spread_duplicate_points([8.0] * 6, [0.7] * 6)
    assert len(set(zip(xs, ys))) == 6
    assert min(ys) == 0.7
    assert all(y >= 0.7 for y in ys)


def test_heatmap_written_from_slot_coverage(tmp_path):
    _write(tmp_path / "slot_coverage.csv",
           [{"slot_id": "casualties", "arm": "human", "coverage": "0.90"},
            {"slot_id": "casualties", "arm": "api-fast/generic", "coverage": "0.40"},
            {"slot_id": "shelter", "arm": "human", "coverage": "0.60"},
            {"slot_id": "shelter", "arm": "api-fast/generic", "coverage": "0.70"}],
           ["slot_id", "arm", "coverage"])
    out = figure_slot_heatmap(tmp_path, tmp_path / "fig_a")
    assert out is not None and out.is_file() and out.stat().st_size > 0
    assert (tmp_path / "fig_a.png").is_file()


def test_completeness_bars_written_with_cis(tmp_path):
    _write(tmp_path / "completeness.csv",
           [{"arm": "human", "completeness": "0.82", "ci_lo": "0.78", "ci_hi": "0.86",
             "is_human_ceiling": "True"},
            {"arm": "api-fast/generic", "completeness": "0.72", "ci_lo": "0.68", "ci_hi": "0.76",
             "is_human_ceiling": "False"}],
           ["arm", "completeness", "ci_lo", "ci_hi", "is_human_ceiling"])
    out = figure_completeness_bars(tmp_path, tmp_path / "fig_b")
    assert out is not None and out.is_file() and out.stat().st_size > 0


def test_correlation_scatter_written(tmp_path):
    _write(tmp_path / "reference_metrics.csv",
           [{"sitrep_id": f"s{i}", "rouge_l_f": f"0.1{i}", "bertscore_f1": f"0.8{i}",
             "llm_judge_score": str(5 + i % 3)} for i in range(6)],
           ["sitrep_id", "rouge_l_f", "bertscore_f1", "llm_judge_score"])
    _write(tmp_path / "completeness_by_sitrep.csv",
           [{"sitrep_id": f"s{i}", "completeness": f"0.7{i}"} for i in range(6)],
           ["sitrep_id", "completeness"])
    _write(tmp_path / "correlations.csv",
           [{"scope": "pooled", "metric": "rouge_l_f", "coefficient": "spearman",
             "rho": "0.07", "p_value": "0.49"}],
           ["scope", "metric", "coefficient", "rho", "p_value"])
    out = figure_correlation_scatter(tmp_path, tmp_path / "fig_c")
    assert out is not None and out.is_file() and out.stat().st_size > 0


def test_figures_return_none_when_tables_absent(tmp_path):
    assert figure_slot_heatmap(tmp_path, tmp_path / "a") is None
    assert figure_completeness_bars(tmp_path, tmp_path / "b") is None
    assert figure_correlation_scatter(tmp_path, tmp_path / "c") is None
