import csv
from pathlib import Path

import src.paper_analysis as paper_analysis
from src.paper_analysis import (
    VALIDATION_JUDGES,
    _normalise_heading,
    figure_grounding_summary,
    gated_present_score,
    holm_adjust,
    human_validation_tables,
)


ROOT = Path(__file__).resolve().parents[1]


def test_holm_adjust_is_monotone_in_rank_order():
    raw = [0.02, 0.001, 0.2]
    adjusted = holm_adjust(raw)
    assert adjusted == [0.04, 0.003, 0.2]


def test_content_gate_rejects_empty_or_negative_heading_body():
    assert gated_present_score("") == 0.0
    assert gated_present_score("No information.") == 0.0
    assert gated_present_score("Not reported in the supplied source material.") == 0.0


def test_content_gate_requires_a_substantive_clause():
    assert gated_present_score("Limited access.") == 0.5
    assert gated_present_score("Flooding washed out the main bridge and isolated three districts overnight.") == 1.0


def test_heading_normalisation_accepts_bracketed_slot_ids():
    assert _normalise_heading("**Sourcing and attribution [sourcing]**") == "sourcing and attribution"


def test_released_system_level_bridge_has_four_arms_and_three_metrics_per_event():
    folder = ROOT / "results" / "tables" / "paper"
    with (folder / "system_level_arm_scores.csv").open(encoding="utf-8", newline="") as stream:
        arms = list(csv.DictReader(stream))
    with (folder / "system_level_correlations.csv").open(encoding="utf-8", newline="") as stream:
        correlations = list(csv.DictReader(stream))
    assert len(arms) == 12
    assert len(correlations) == 9
    assert {row["n_arms"] for row in correlations} == {"4"}
    bert = [row["rho"] for row in correlations if row["metric"] == "bertscore_f1"]
    assert bert == ["-1.0", "-1.0", "-0.8"]


def test_figure_grounding_summary_covers_every_table_2_cell():
    rows = figure_grounding_summary()
    assert len(rows) == 5
    assert [row["n_same_day_pairs"] for row in rows] == [83, 28, 20, 8, 8]
    assert rows[0]["official_in_stream"] == 0.142619
    assert rows[0]["official_in_stream_ci_lo"] == 0.095138
    assert rows[0]["official_in_stream_ci_hi"] == 0.197216
    assert rows[0]["official_in_stream_n"] == 21
    assert rows[1]["machine_in_stream"] == 0.901346
    assert rows[2]["figure_precision"] == 0.153425
    assert rows[2]["figure_recall"] == 0.088788
    assert rows[2]["casualty_recall"] == 0.125
    for row in rows:
        for metric in ("official_in_stream", "machine_in_stream"):
            assert row[f"{metric}_ci_lo"] <= row[metric] <= row[f"{metric}_ci_hi"]
        assert row["official_in_stream_bootstrap_unit"] == "event-day"
        assert row["machine_in_stream_bootstrap_unit"] == "same-day-machine-official-pair"


def test_blind_human_validation_recomputes_every_judge_and_scope():
    pairs, summary = human_validation_tables()
    assert len(pairs) == 100
    assert len(summary) == 12
    assert {row["human_verdict"] for row in pairs} == {"absent", "partial", "present"}
    overall = {row["judge"]: row for row in summary if row["scope"] == "all"}
    assert overall["deepseek-v4-pro"]["cohens_kappa"] == 0.592154
    assert overall["gpt-5.6-luna"]["cohens_kappa"] == 0.616803
    assert overall["qwen3.8-flash"]["cohens_kappa"] == 0.681025
    assert overall["gemini-3.7-flash"]["cohens_kappa"] == 0.798658
    assert overall["gemini-3.7-flash"]["support_human_absent"] == 9
    assert overall["gemini-3.7-flash"]["f1_absent"] == 0.823529
    assert overall["gemini-3.7-flash"]["f1_partial"] == 0.829268
    assert overall["gemini-3.7-flash"]["f1_present"] == 0.943662
    assert [overall[judge]["n_missing"] for judge, _prefix in VALIDATION_JUDGES] == [0, 1, 10, 0]
    assert sum(row["passes_preregistered_raw_agreement_threshold"] for row in overall.values()) == 4
    assert sum(row["kappa_at_least_0_7_descriptive"] for row in overall.values()) == 1


def test_human_validation_can_recompute_from_anonymous_release(monkeypatch):
    monkeypatch.setattr(paper_analysis, "VALIDATION_SAMPLE", ROOT / "missing-validation-sample.csv")
    monkeypatch.setattr(paper_analysis, "VALIDATION_KEY", ROOT / "missing-validation-key.csv")
    pairs, summary = paper_analysis.human_validation_tables()
    assert len(pairs) == 100
    assert next(row for row in summary
                if row["judge"] == "gemini-3.7-flash" and row["scope"] == "all")[
                    "cohens_kappa"] == 0.798658
