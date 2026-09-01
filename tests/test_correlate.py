"""Offline tests for src/correlate.py (Phase 5 meta-evaluation).

Correlation maths is checked against scipy on known vectors; the bootstrap is checked for
determinism under the seed; and the end-to-end path asserts that a sentinel planted in a machine
sitrep's text never reaches either committed results table (CLAUDE.md rule 7).
"""
from __future__ import annotations

import csv
import json
import math
import types
from pathlib import Path

import pytest
from scipy import stats

from src import correlate as C
from src.correlate import (
    DISAGREEMENT_COLUMNS,
    HIGH_REF_LOW_COMP,
    LOW_REF_HIGH_COMP,
    bootstrap_ci,
    correlate,
    join_tables,
    kendall,
    load_completeness,
    main,
    mine_disagreements,
    percentile_ranks,
    pick_completeness_column,
    read_csv_rows,
    spearman,
)
from src.reference_metrics import ROUGE_L
from src.reference_metrics import main as refmetrics_main

SENTINEL = "OKAPI-SENTINEL-4419"
EVENT = "cyclone_idai_2019"

PERFECT_POS = ([1.0, 2.0, 3.0, 4.0, 5.0], [10.0, 20.0, 30.0, 40.0, 50.0])
PERFECT_NEG = ([1.0, 2.0, 3.0, 4.0, 5.0], [50.0, 40.0, 30.0, 20.0, 10.0])
ZERO_CORR = ([1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 1.0, 3.0])  # sum d^2 = 10 -> rho = 0; C = D = 3 -> tau = 0
MIXED = ([0.10, 0.42, 0.33, 0.71, 0.55, 0.24, 0.88, 0.61],
         [0.30, 0.11, 0.52, 0.49, 0.90, 0.21, 0.44, 0.70])


# ---------------------------------------------------------------------------
# correlation coefficients
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("x,y,expected", [(*PERFECT_POS, 1.0), (*PERFECT_NEG, -1.0), (*ZERO_CORR, 0.0)])
def test_spearman_and_kendall_match_the_known_answer(x, y, expected):
    assert spearman(x, y)[0] == pytest.approx(expected)
    assert kendall(x, y)[0] == pytest.approx(expected)


@pytest.mark.parametrize("x,y", [PERFECT_POS, PERFECT_NEG, ZERO_CORR, MIXED])
def test_spearman_and_kendall_match_scipy(x, y):
    rho, p = spearman(x, y)
    ref = stats.spearmanr(x, y)
    assert rho == pytest.approx(float(ref.statistic)) and p == pytest.approx(float(ref.pvalue))
    tau, pt = kendall(x, y)
    reft = stats.kendalltau(x, y)
    assert tau == pytest.approx(float(reft.statistic)) and pt == pytest.approx(float(reft.pvalue))


def test_coefficients_are_nan_when_undefined():
    for fn in (spearman, kendall):
        assert math.isnan(fn([1.0, 2.0], [1.0, 2.0])[0])            # n < 3
        assert math.isnan(fn([1.0, 1.0, 1.0], [1.0, 2.0, 3.0])[0])  # constant x
        assert math.isnan(fn([1.0, 2.0, 3.0], [7.0, 7.0, 7.0])[0])  # constant y


def test_coefficients_drop_missing_pairs():
    # the None pair is dropped, leaving a perfectly increasing pair of length 3
    assert spearman([1.0, 2.0, 3.0, 9.0], [1.0, 2.0, 3.0, None])[0] == pytest.approx(1.0)
    assert kendall([1.0, 2.0, 3.0, 9.0], [1.0, 2.0, 3.0, math.nan])[0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# bootstrap
# ---------------------------------------------------------------------------
def test_bootstrap_ci_is_deterministic_under_the_seed():
    x, y = MIXED
    a = bootstrap_ci(x, y, spearman, n_bootstrap=200, seed=17)
    b = bootstrap_ci(x, y, spearman, n_bootstrap=200, seed=17)
    assert a == b
    assert bootstrap_ci(x, y, spearman, n_bootstrap=200, seed=18) != a
    assert a[0] <= a[1]


def test_bootstrap_ci_brackets_a_perfect_correlation():
    lo, hi = bootstrap_ci(*PERFECT_POS, spearman, n_bootstrap=100, seed=17)
    assert lo == pytest.approx(1.0) and hi == pytest.approx(1.0)
    lo, hi = bootstrap_ci(*PERFECT_NEG, kendall, n_bootstrap=100, seed=17)
    assert lo == pytest.approx(-1.0) and hi == pytest.approx(-1.0)


def test_bootstrap_ci_nan_when_undefined_or_no_resamples():
    assert all(math.isnan(v) for v in bootstrap_ci([1.0, 2.0], [1.0, 2.0], spearman, n_bootstrap=10))
    assert all(math.isnan(v) for v in bootstrap_ci(*PERFECT_POS, spearman, n_bootstrap=0))


def test_percentile_ranks_handles_ties_and_singletons():
    assert percentile_ranks([]) == []
    assert percentile_ranks([4.2]) == [0.5]
    assert percentile_ranks([10.0, 20.0, 30.0]) == [0.0, 0.5, 1.0]
    assert percentile_ranks([5.0, 5.0, 9.0]) == [0.25, 0.25, 1.0]


# ---------------------------------------------------------------------------
# loading / joining (text can never enter, by construction)
# ---------------------------------------------------------------------------
def _write_csv(path: Path, rows: list[dict], columns: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)
    return path


def test_pick_completeness_column_prefers_known_aliases():
    assert pick_completeness_column([{"sitrep_id": "a", "completeness": "0.5"}]) == "completeness"
    assert pick_completeness_column([{"operational_completeness": "0.5"}]) == "operational_completeness"
    assert pick_completeness_column([{"mean_slot_score": "0.5"}]) == "mean_slot_score"
    assert pick_completeness_column([{"rouge": "0.5"}]) is None
    assert pick_completeness_column([{"custom": "0.5"}], "custom") == "custom"
    assert pick_completeness_column([{"custom": "0.5"}], "missing") is None


def test_read_csv_rows_missing_file_is_empty(tmp_path):
    assert read_csv_rows(tmp_path / "nope.csv") == []


def test_load_completeness_prefers_src_metrics_when_it_lands(tmp_path, monkeypatch):
    import src

    stub = types.ModuleType("src.metrics")
    stub.load_completeness = lambda path: [{"sitrep_id": "from-metrics", "completeness": "0.9"}]
    monkeypatch.setitem(__import__("sys").modules, "src.metrics", stub)
    monkeypatch.setattr(src, "metrics", stub, raising=False)
    assert load_completeness(tmp_path / "x.csv")[0]["sitrep_id"] == "from-metrics"


def test_join_tables_drops_unmatched_rows_and_stray_columns():
    comp = [{"sitrep_id": "m-1", "completeness": "0.8"}, {"sitrep_id": "m-2", "completeness": ""},
            {"sitrep_id": "m-4", "completeness": "0.1"}]
    ref = [{"sitrep_id": "m-1", "event": EVENT, "date": "2019-03-20", "model": "api-strong",
            "arm": "generic", "rouge_l_f": "0.4", "bertscore_f1": "", "text": SENTINEL},
           {"sitrep_id": "m-2", "event": EVENT, "rouge_l_f": "0.9", "text": SENTINEL},
           {"sitrep_id": "m-3", "event": EVENT, "rouge_l_f": "0.2", "text": SENTINEL}]
    rows = join_tables(comp, ref, completeness_column="completeness",
                       metric_columns=["rouge_l_f", "bertscore_f1"])
    assert [r["sitrep_id"] for r in rows] == ["m-1"]  # m-2 has no score, m-3 no completeness row
    assert rows[0]["completeness"] == 0.8 and rows[0]["rouge_l_f"] == 0.4
    assert rows[0]["bertscore_f1"] is None
    assert SENTINEL not in json.dumps(rows)  # stray columns are never carried across


# ---------------------------------------------------------------------------
# correlation table
# ---------------------------------------------------------------------------
def _rows(n: int = 6, event: str = EVENT) -> list[dict]:
    return [{"sitrep_id": f"m-{i}", "event": event, "date": f"2019-03-{20 + i:02d}",
             "model": "api-strong", "arm": "generic",
             "completeness": i / (n - 1), "rouge_l_f": 1.0 - i / (n - 1),
             "llm_judge_score": float(i)} for i in range(n)]


def test_correlate_emits_pooled_and_per_event_rows_with_cis():
    rows = _rows(6) + [dict(r, sitrep_id=r["sitrep_id"] + "b", event="hurricane_maria_2017")
                       for r in _rows(6)]
    out = correlate(rows, metric_columns=["rouge_l_f", "llm_judge_score"], n_bootstrap=50, seed=17)
    scopes = {(r["scope"], r["event"], r["metric"], r["coefficient"]) for r in out}
    assert ("pooled", "", "rouge_l_f", "spearman") in scopes
    assert ("event", EVENT, "llm_judge_score", "kendall") in scopes
    assert len(out) == 3 * 2 * 2  # (pooled + 2 events) x 2 metrics x 2 coefficients
    pooled_rouge = next(r for r in out if r["scope"] == "pooled" and r["metric"] == "rouge_l_f"
                        and r["coefficient"] == "spearman")
    assert pooled_rouge["rho"] == pytest.approx(-1.0) and pooled_rouge["n"] == 12
    assert pooled_rouge["ci_lo"] <= pooled_rouge["rho"] <= pooled_rouge["ci_hi"]
    judge = next(r for r in out if r["scope"] == "pooled" and r["metric"] == "llm_judge_score"
                 and r["coefficient"] == "spearman")
    assert judge["rho"] == pytest.approx(1.0) and judge["seed"] == 17 and judge["n_bootstrap"] == 50


def test_correlate_reports_blank_rho_when_a_metric_is_missing():
    rows = [dict(r, bertscore_f1=None) for r in _rows(5)]
    out = correlate(rows, metric_columns=["bertscore_f1"], n_bootstrap=10, per_event=False)
    assert all(r["rho"] == "" and r["ci_lo"] == "" and r["n"] == 0 for r in out)


# ---------------------------------------------------------------------------
# disagreement mining
# ---------------------------------------------------------------------------
def test_mine_disagreements_picks_the_right_rows():
    rows = _rows(5)
    # m-0: rouge 1.0 (top) but completeness 0.0 (bottom); m-4 is the exact mirror.
    out = mine_disagreements(rows, metric_columns=["rouge_l_f"], top_k=2)
    high = [r for r in out if r["direction"] == HIGH_REF_LOW_COMP]
    low = [r for r in out if r["direction"] == LOW_REF_HIGH_COMP]
    assert [r["sitrep_id"] for r in high] == ["m-0", "m-1"]
    assert [r["sitrep_id"] for r in low] == ["m-4", "m-3"]
    assert high[0]["gap"] == pytest.approx(1.0) and low[0]["gap"] == pytest.approx(-1.0)
    assert high[0]["rank"] == 1 and high[1]["rank"] == 2 and high[0]["n"] == 5
    assert high[0]["metric_value"] == pytest.approx(1.0)
    assert high[0]["completeness"] == pytest.approx(0.0)
    assert set(out[0]) <= set(DISAGREEMENT_COLUMNS)
    # a generous top_k must not pad a direction with rows that disagree the other way (or not at all)
    generous = mine_disagreements(rows, metric_columns=["rouge_l_f"], top_k=99)
    assert all(r["gap"] > 0 for r in generous if r["direction"] == HIGH_REF_LOW_COMP)
    assert all(r["gap"] < 0 for r in generous if r["direction"] == LOW_REF_HIGH_COMP)
    assert len(generous) == 4  # m-2 sits at gap 0 and appears in neither direction


def test_mine_disagreements_skips_agreeing_pairs_and_thin_metrics():
    agreeing = [dict(r, rouge_l_f=r["completeness"]) for r in _rows(4)]
    assert mine_disagreements(agreeing, metric_columns=["rouge_l_f"], top_k=3) == []
    thin = [dict(r, bertscore_f1=None) for r in _rows(4)]
    assert mine_disagreements(thin, metric_columns=["bertscore_f1"], top_k=3) == []


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _tables(tmp_path: Path, n: int = 6) -> tuple[Path, Path]:
    rows = _rows(n)
    comp = _write_csv(tmp_path / "completeness.csv",
                      [{"sitrep_id": r["sitrep_id"], "event": r["event"],
                        "completeness": r["completeness"]} for r in rows],
                      ["sitrep_id", "event", "completeness"])
    ref = _write_csv(tmp_path / "reference_metrics.csv",
                     [{"sitrep_id": r["sitrep_id"], "event": r["event"], "date": r["date"],
                       "model": r["model"], "arm": r["arm"], "rouge_l_f": r["rouge_l_f"],
                       "bertscore_f1": "", "llm_judge_score": r["llm_judge_score"]} for r in rows],
                     ["sitrep_id", "event", "date", "model", "arm", "rouge_l_f", "bertscore_f1",
                      "llm_judge_score"])
    return comp, ref


def test_cli_degrades_when_phase4_or_phase5_tables_are_missing(tmp_path, capsys):
    comp, ref = _tables(tmp_path)
    assert main(["--completeness", str(tmp_path / "nope.csv"), "--reference-metrics", str(ref)]) == 2
    assert "run Phase 4" in capsys.readouterr().out
    assert main(["--completeness", str(comp), "--reference-metrics", str(tmp_path / "nope.csv")]) == 2
    assert "src.reference_metrics" in capsys.readouterr().out


def test_cli_reports_an_unusable_completeness_column(tmp_path, capsys):
    comp, ref = _tables(tmp_path)
    assert main(["--completeness", str(comp), "--reference-metrics", str(ref),
                 "--completeness-column", "nope"]) == 2
    assert "no completeness column" in capsys.readouterr().out


def test_cli_writes_both_tables(tmp_path, capsys):
    comp, ref = _tables(tmp_path)
    corr_out, dis_out = tmp_path / "correlations.csv", tmp_path / "disagreements.csv"
    rc = main(["--completeness", str(comp), "--reference-metrics", str(ref), "--n-bootstrap", "50",
               "--top-k", "2", "--out-correlations", str(corr_out), "--out-disagreements", str(dis_out)])
    assert rc == 0
    corr = list(csv.DictReader(corr_out.open(encoding="utf-8")))
    # bertscore_f1 is present but blank: the column is reported with an empty rho, never dropped
    assert {r["metric"] for r in corr} == {"rouge_l_f", "bertscore_f1", "llm_judge_score"}
    assert all(r["rho"] == "" for r in corr if r["metric"] == "bertscore_f1")
    assert {r["coefficient"] for r in corr} == {"spearman", "kendall"}
    rouge = next(r for r in corr if r["metric"] == "rouge_l_f" and r["coefficient"] == "spearman")
    assert float(rouge["rho"]) == pytest.approx(-1.0) and int(rouge["n"]) == 6
    dis = list(csv.DictReader(dis_out.open(encoding="utf-8")))
    assert set(dis[0]) == set(DISAGREEMENT_COLUMNS)
    assert "correlations:" in capsys.readouterr().out


def test_cli_smoke_uses_suffixed_paths_and_prints(tmp_path, monkeypatch, capsys):
    comp, ref = _tables(tmp_path)
    monkeypatch.setattr(C, "cfg_path", lambda cfg, key, default: tmp_path / "results")
    rc = main(["--smoke", "4", "--completeness", str(comp), "--reference-metrics", str(ref)])
    assert rc == 0
    assert (tmp_path / "results" / "tables" / "correlations_smoke.csv").exists()
    assert (tmp_path / "results" / "tables" / "disagreements_smoke.csv").exists()
    assert not (tmp_path / "results" / "tables" / "correlations.csv").exists()
    out = capsys.readouterr().out
    assert "joined 4 sitrep(s)" in out and "disagreements" in out


# ---------------------------------------------------------------------------
# rule 7: no sitrep text in either committed table, end to end
# ---------------------------------------------------------------------------
def test_no_sitrep_text_reaches_the_results_tables(tmp_path):
    mdir = tmp_path / "sitreps_machine" / EVENT
    hdir = tmp_path / "sitreps_human" / EVENT
    mdir.mkdir(parents=True)
    hdir.mkdir(parents=True)
    for i in range(4):
        date = f"2019-03-{20 + i:02d}"
        (mdir / f"{date}.json").write_text(json.dumps({
            "id": f"m-{i}", "event": EVENT, "date": date, "model": "api-strong", "arm": "generic",
            "text": f"{SENTINEL} flooding displaced roughly {i * 100} people near Beira."}),
            encoding="utf-8")
        (hdir / f"{date}.json").write_text(json.dumps({
            "id": f"ocha-{i}", "event": EVENT, "date": date, "source": "OCHA",
            "text": f"{SENTINEL} OCHA reports {i * 120} displaced people in Beira district."}),
            encoding="utf-8")
    ref_csv = tmp_path / "reference_metrics.csv"
    assert refmetrics_main(["--machine-dir", str(tmp_path / "sitreps_machine"),
                            "--human-dir", str(tmp_path / "sitreps_human"),
                            "--metrics", ROUGE_L, "--out", str(ref_csv)]) == 0
    assert SENTINEL not in ref_csv.read_text(encoding="utf-8")

    comp = _write_csv(tmp_path / "completeness.csv",
                      [{"sitrep_id": f"m-{i}", "completeness": 1.0 - i / 3} for i in range(4)],
                      ["sitrep_id", "completeness"])
    corr_out, dis_out = tmp_path / "correlations.csv", tmp_path / "disagreements.csv"
    assert main(["--completeness", str(comp), "--reference-metrics", str(ref_csv),
                 "--n-bootstrap", "50", "--out-correlations", str(corr_out),
                 "--out-disagreements", str(dis_out)]) == 0
    for path in (corr_out, dis_out):
        body = path.read_text(encoding="utf-8")
        assert SENTINEL not in body and "Beira" not in body and "displaced" not in body
    dis = list(csv.DictReader(dis_out.open(encoding="utf-8")))
    assert dis and all(r["sitrep_id"].startswith("m-") for r in dis)
