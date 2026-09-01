"""Exact-arithmetic tests for src/metrics.py (pure functions, no I/O, no network)."""
from __future__ import annotations

import math
from statistics import median

import pytest

from src.metrics import (
    Figure,
    bootstrap_ci,
    cohens_kappa,
    coverage_by_slot,
    extract_figures,
    figure_extraction_check,
    judge_agreement,
    operational_completeness,
    verdict_score,
    visibility_ceiling,
)


def _j(slot_id: str, verdict: str, **extra):
    return {"slot_id": slot_id, "verdict": verdict, "sitrep_id": "s1", **extra}


# ---------------------------------------------------------------------------
# operational completeness
# ---------------------------------------------------------------------------
def test_verdict_scores_are_the_documented_constants():
    assert (verdict_score("present"), verdict_score("partial"), verdict_score("absent")) == (1.0, 0.5, 0.0)
    assert verdict_score(" Present ") == 1.0  # normalised
    with pytest.raises(ValueError):
        verdict_score("maybe")


def test_completeness_of_three_slot_toy_is_exact_fraction():
    # present + partial + absent = (1.0 + 0.5 + 0.0) / 3 = 0.5 exactly
    rows = [_j("casualties", "present"), _j("needs", "partial"), _j("funding", "absent")]
    assert operational_completeness(rows) == 0.5
    # four slots: (1 + 1 + 0.5 + 0) / 4 = 0.625
    rows4 = rows + [_j("access", "present")]
    assert operational_completeness(rows4) == pytest.approx(0.625)
    # mapping form and the 'slot' alias are equivalent to the row form
    assert operational_completeness({"casualties": "present", "needs": "partial", "funding": "absent"}) == 0.5
    assert operational_completeness([{"slot": "a", "verdict": "present"}, {"slot": "b", "verdict": "absent"}]) == 0.5


def test_completeness_rejects_empty_and_malformed_input():
    with pytest.raises(ValueError):
        operational_completeness([])
    with pytest.raises(ValueError):
        operational_completeness([_j("a", "unknown-verdict")])
    with pytest.raises(KeyError):
        operational_completeness([{"slot_id": "a"}])
    with pytest.raises(KeyError):
        operational_completeness([{"verdict": "present"}])
    with pytest.raises(TypeError):
        operational_completeness(["present"])


def test_prevalence_weighting_is_a_weighted_mean():
    rows = [_j("a", "present"), _j("b", "absent"), _j("c", "partial")]
    # (3*1.0 + 1*0.0 + 2*0.5) / 6 = 4/6
    assert operational_completeness(rows, {"a": 3.0, "b": 1.0, "c": 2.0}) == pytest.approx(4 / 6)
    # a zero-weight slot drops out entirely: (1*1.0 + 1*0.5) / 2 = 0.75
    assert operational_completeness(rows, {"a": 1.0, "b": 0.0, "c": 1.0}) == pytest.approx(0.75)
    # uniform weights == the unweighted mean
    assert operational_completeness(rows, {"a": 2.0, "b": 2.0, "c": 2.0}) == pytest.approx(0.5)


def test_weighting_fails_loudly_on_missing_negative_and_zero_weights():
    rows = [_j("a", "present"), _j("b", "absent")]
    with pytest.raises(KeyError):
        operational_completeness(rows, {"a": 1.0})
    assert operational_completeness(rows, {"a": 1.0}, default_weight=1.0) == 0.5
    with pytest.raises(ValueError):
        operational_completeness(rows, {"a": -1.0, "b": 1.0})
    with pytest.raises(ValueError):
        operational_completeness(rows, {"a": 0.0, "b": 0.0})


def test_coverage_by_slot_counts_and_averages_across_sitreps():
    rows = [_j("a", "present"), _j("a", "absent"), _j("a", "partial"), _j("b", "present")]
    cov = coverage_by_slot(rows)
    assert cov["a"].coverage == pytest.approx(0.5) and cov["a"].n == 3
    assert (cov["a"].n_present, cov["a"].n_partial, cov["a"].n_absent) == (1, 1, 1)
    assert cov["b"].coverage == 1.0 and cov["b"].n == 1
    assert cov["a"].as_row()["slot_id"] == "a"


# ---------------------------------------------------------------------------
# visibility ceiling
# ---------------------------------------------------------------------------
def test_ceiling_needs_every_machine_arm_to_be_at_zero():
    machine = {
        "funding": {"opus/generic": 0.0, "sonnet/generic": 0.0, "opus/schema_guided": 0.0},
        "coordination": {"opus/generic": 0.0, "sonnet/generic": 0.6},   # one arm covers it
        "casualties": {"opus/generic": 0.9, "sonnet/generic": 0.8},
    }
    human = {"funding": 0.9, "coordination": 0.8, "casualties": 0.95}
    detail = {s.slot_id: s for s in visibility_ceiling(machine, human)}
    assert detail["funding"].is_ceiling is True
    assert detail["funding"].machine_coverage_max == 0.0
    assert detail["funding"].gap == pytest.approx(0.9)
    assert detail["coordination"].is_ceiling is False and detail["coordination"].machine_best_arm == "sonnet/generic"
    assert detail["casualties"].is_ceiling is False
    # per-slot detail for every slot, ceilings first
    result = visibility_ceiling(machine, human)
    assert [s.slot_id for s in result][0] == "funding" and len(result) == 3


def test_ceiling_thresholds_are_inclusive_at_the_boundary():
    on_boundary = visibility_ceiling({"s": 0.1}, {"s": 0.5})[0]
    assert on_boundary.is_ceiling is True
    assert visibility_ceiling({"s": 0.100001}, {"s": 0.5})[0].is_ceiling is False
    assert visibility_ceiling({"s": 0.1}, {"s": 0.499999})[0].is_ceiling is False
    # custom thresholds
    assert visibility_ceiling({"s": 0.2}, {"s": 0.6}, machine_max=0.2, human_min=0.6)[0].is_ceiling is True
    with pytest.raises(ValueError):
        visibility_ceiling({"s": 0.0}, {"s": 1.0}, machine_max=1.5)


def test_slot_absent_in_humans_too_is_not_a_ceiling_slot():
    detail = {s.slot_id: s for s in visibility_ceiling({"weather_forecast": 0.0}, {"weather_forecast": 0.05})}
    slot = detail["weather_forecast"]
    assert slot.is_ceiling is False
    assert "thin in humans" in slot.reason
    assert slot.gap == pytest.approx(0.05)


def test_ceiling_handles_slots_judged_on_only_one_side():
    result = {s.slot_id: s for s in visibility_ceiling({"only_machine": 0.0}, {"only_human": 0.9})}
    assert result["only_machine"].is_ceiling is False and "no human judgements" in result["only_machine"].reason
    assert result["only_human"].is_ceiling is False and "no machine judgements" in result["only_human"].reason
    assert result["only_human"].machine_coverage_max is None
    assert result["only_human"].gap is None
    row = result["only_machine"].as_row()
    assert row["machine_coverage_by_arm"] == "all=0.0000" and row["is_ceiling"] is False


def test_scalar_and_per_arm_machine_scores_agree():
    scalar = visibility_ceiling({"s": 0.0}, {"s": 0.9})[0]
    per_arm = visibility_ceiling({"s": {"only-arm": 0.0}}, {"s": 0.9})[0]
    assert scalar.is_ceiling == per_arm.is_ceiling is True
    assert scalar.machine_coverage_by_arm == {"all": 0.0}
    assert per_arm.machine_best_arm == "only-arm"


# ---------------------------------------------------------------------------
# figure extraction
# ---------------------------------------------------------------------------
def test_extract_figures_parses_separators_scales_and_percentages():
    figs = extract_figures("At least 1,200 people were killed; 1.4 million people affected; "
                           "45% of homes damaged; 12 000 households displaced.")
    got = {(f.value, f.unit) for f in figs}
    assert (1200.0, "count") in got
    assert (1400000.0, "count") in got
    assert (45.0, "percent") in got
    assert (12000.0, "count") in got


def test_extract_figures_drops_dates_and_bare_years():
    figs = extract_figures("As of 20 March 2019 (report 2019-03-22, issued 22/03/2019), "
                           "1,200 people were killed.")
    assert [f.value for f in figs] == [1200.0]


def test_extract_figures_needs_a_quantity_signal():
    assert extract_figures("Section 3 of the plan. Annex 4.") == []
    assert [f.value for f in extract_figures("3 people were killed.")] == [3.0]
    assert extract_figures("") == []


def test_figure_kind_uses_the_casualty_lexicon():
    figs = extract_figures("57 people died. 400 houses were damaged.")
    kinds = {f.value: (f.kind, f.keyword) for f in figs}
    assert kinds[57.0][0] == "casualty" and kinds[57.0][1] == "died"
    assert kinds[400.0][0] == "quantity"


def test_extracted_figures_retain_exact_character_offsets():
    text = "The first 6 was a label; 6 people were injured."
    figures = extract_figures(text)
    assert len(figures) == 1
    assert [text[row.start:row.end] for row in figures] == [row.raw for row in figures]


def test_figure_extraction_check_reports_overlap_recall_and_precision():
    human = "Government reports 1,200 deaths, 45% of homes damaged and 1.4 million people affected."
    machine = "Posts mention 1,200 deaths and 300 people displaced."
    chk = figure_extraction_check(machine, human)
    human_values = {f.value for f in chk.human}
    assert human_values == {1200.0, 45.0, 1400000.0}
    assert [f.value for f in chk.overlap] == [1200.0]
    assert {f.value for f in chk.only_human} == {45.0, 1400000.0}
    assert {f.value for f in chk.only_machine} == {300.0}
    assert chk.recall == pytest.approx(1 / 3)
    assert chk.precision == pytest.approx(1 / 2)
    assert chk.jaccard == pytest.approx(1 / 4)
    row = chk.as_row()
    assert row["n_overlap"] == 1 and row["n_human_figures"] == 3
    assert "evidence" not in row and all(not isinstance(v, str) for v in row.values())


def test_figure_check_rates_are_none_when_a_side_has_no_figures():
    chk = figure_extraction_check("No numbers here at all.", "Also nothing quantitative.")
    assert chk.recall is None and chk.precision is None and chk.jaccard is None
    assert chk.casualty_recall is None


def test_percent_and_count_with_the_same_number_do_not_match():
    chk = figure_extraction_check("45% of homes damaged.", "45 people killed.")
    assert chk.overlap == [] and chk.recall == 0.0


def test_figure_dataclass_carries_no_prose():
    fig = extract_figures("1,200 people were killed in the district capital.")[0]
    assert isinstance(fig, Figure)
    assert fig.raw == "1,200" and fig.keyword == "killed"
    assert "district" not in repr(fig)


# ---------------------------------------------------------------------------
# bootstrap CIs
# ---------------------------------------------------------------------------
def test_bootstrap_is_deterministic_under_a_seed():
    vals = [0.1, 0.4, 0.5, 0.55, 0.6, 0.8, 0.9, 1.0]
    a = bootstrap_ci(vals, n_boot=1000, seed=17)
    b = bootstrap_ci(vals, n_boot=1000, seed=17)
    assert (a.lo, a.point, a.hi) == (b.lo, b.point, b.hi)
    assert a.point == pytest.approx(sum(vals) / len(vals))
    assert a.lo <= a.point <= a.hi
    assert (a.n, a.n_boot, a.seed, a.alpha) == (8, 1000, 17, 0.05)
    other = bootstrap_ci(vals, n_boot=1000, seed=18)
    assert (other.lo, other.hi) != (a.lo, a.hi)
    assert other.point == a.point  # the point estimate never depends on the seed


def test_bootstrap_edge_cases_and_custom_statistic():
    one = bootstrap_ci([0.5], n_boot=50, seed=17)
    assert one.lo == one.hi == one.point == 0.5
    med = bootstrap_ci([0.0, 0.0, 1.0, 1.0, 1.0], n_boot=200, seed=17, statistic=median)
    assert med.point == 1.0
    with pytest.raises(ValueError):
        bootstrap_ci([])
    with pytest.raises(ValueError):
        bootstrap_ci([0.5], n_boot=0)
    with pytest.raises(ValueError):
        bootstrap_ci([0.5], alpha=1.5)


def test_bootstrap_of_completeness_scores_narrows_with_more_data():
    small = bootstrap_ci([0.2, 0.8] * 3, n_boot=500, seed=17)
    large = bootstrap_ci([0.2, 0.8] * 30, n_boot=500, seed=17)
    assert (large.hi - large.lo) < (small.hi - small.lo)


# ---------------------------------------------------------------------------
# judge agreement / kappa
# ---------------------------------------------------------------------------
def test_kappa_on_a_known_confusion_matrix():
    # 4 present/present, 1 present/absent, 4 absent/absent, 1 absent/present  -> po=0.8, pe=0.5
    judge = ["present"] * 5 + ["absent"] * 5
    human = ["present"] * 4 + ["absent"] + ["absent"] * 4 + ["present"]
    agr = judge_agreement(judge, human)
    assert agr.n == 10
    assert agr.agreement == pytest.approx(0.8)
    assert agr.kappa == pytest.approx(0.6)
    assert agr.kappa_linear == pytest.approx(0.6)
    # confusion[i][j] = judge said labels[i], human said labels[j]; labels = (absent, partial, present)
    assert agr.labels == ["absent", "partial", "present"]
    assert agr.confusion == [[4, 0, 1], [0, 0, 0], [1, 0, 4]]
    assert agr.per_label_agreement["present"] == pytest.approx(4 / 6)
    assert math.isnan(agr.per_label_agreement["partial"])
    row = agr.as_row()
    assert row["raw_agreement"] == pytest.approx(0.8) and row["n_judge_present_human_absent"] == 1


def test_perfect_and_chance_level_agreement():
    assert judge_agreement(["present", "absent"], ["present", "absent"]).kappa == pytest.approx(1.0)
    # both raters always say the same single label -> kappa undefined (nan), agreement 1.0
    degenerate = judge_agreement(["present"] * 5, ["present"] * 5)
    assert degenerate.agreement == 1.0 and math.isnan(degenerate.kappa)


def test_linear_weighting_forgives_adjacent_disagreements():
    # judge/human differ by one step (partial vs present) in half the pairs, and by two steps
    # (absent vs present) in the other half; linear weighting must score higher than unweighted.
    judge = ["present", "present", "absent", "absent", "partial", "partial", "absent", "present"]
    human = ["partial", "present", "present", "absent", "partial", "present", "absent", "present"]
    k = cohens_kappa(judge, human)
    k_lin = cohens_kappa(judge, human, weighting="linear")
    k_quad = cohens_kappa(judge, human, weighting="quadratic")
    assert k_lin > k
    assert k_quad > k_lin


def test_agreement_input_validation():
    with pytest.raises(ValueError):
        judge_agreement(["present"], ["present", "absent"])
    with pytest.raises(ValueError):
        judge_agreement([], [])
    with pytest.raises(ValueError):
        judge_agreement(["yes"], ["present"])
    with pytest.raises(ValueError):
        cohens_kappa(["present"], ["absent"], weighting="cubic")
