"""Offline tests for the Phase 4 report aggregator and the schema-freeze checkpoint."""
from __future__ import annotations

import csv

import pytest
import yaml

from src.freeze_schema import approve, normalise, validate, write_review
from src.report_results import HUMAN_ARM, arm_of, build, completeness_table, group_by_arm, slot_coverage_table

SLOTS = ["casualties", "displacement", "access_constraints", "funding"]


def _judgement(sitrep_id, slot_id, verdict, *, kind="machine", arm="api-strong/generic", event="e1"):
    return {"event": event, "sitrep_id": sitrep_id, "kind": kind, "model": arm.split("/")[0],
            "arm": arm, "date": "2019-04-02", "slot_id": slot_id, "verdict": verdict}


def _write_judgements(tables, rows):
    tables.mkdir(parents=True, exist_ok=True)
    cols = ["event", "sitrep_id", "kind", "model", "arm", "date", "slot_id", "verdict"]
    with open(tables / "slot_judgements.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def test_arm_of_distinguishes_human_ceiling_from_machine_arms():
    assert arm_of(_judgement("s", "x", "present", kind="human", arm="")) == HUMAN_ARM
    assert arm_of(_judgement("s", "x", "present", arm="api-fast/schema_guided")) == "api-fast/schema_guided"
    assert arm_of({"kind": "machine", "arm": ""}) == "machine"


def test_completeness_arithmetic_is_exact():
    rows = [_judgement("m1", "a", "present"), _judgement("m1", "b", "partial"),
            _judgement("m1", "c", "absent"), _judgement("m1", "d", "absent")]
    table = completeness_table(group_by_arm(rows), seed=17)
    assert len(table) == 1
    # (1.0 + 0.5 + 0 + 0) / 4 = 0.375
    assert table[0]["completeness"] == pytest.approx(0.375)
    assert table[0]["n_sitreps"] == 1 and table[0]["n_judgements"] == 4
    assert table[0]["ci_lo"] == "" and table[0]["is_human_ceiling"] is False  # no CI from one sitrep


def test_human_ceiling_sorts_first_and_gets_a_ci():
    rows = []
    for i in (1, 2, 3):
        rows += [_judgement(f"h{i}", s, "present", kind="human", arm="") for s in SLOTS]
        rows += [_judgement(f"m{i}", s, "absent") for s in SLOTS]
    table = completeness_table(group_by_arm(rows), seed=17)
    assert table[0]["arm"] == HUMAN_ARM and table[0]["is_human_ceiling"] is True
    assert table[0]["completeness"] == pytest.approx(1.0)
    assert table[0]["ci_lo"] != ""  # 3 sitreps -> bootstrap CI
    assert table[1]["completeness"] == pytest.approx(0.0)


def test_build_writes_all_three_tables_and_finds_the_ceiling(tmp_path):
    tables = tmp_path / "tables"
    rows = []
    for i in (1, 2, 3):
        # humans cover everything; machines cover casualties only -> 3 ceiling slots
        rows += [_judgement(f"h{i}", s, "present", kind="human", arm="") for s in SLOTS]
        for arm in ("api-strong/generic", "api-fast/generic"):
            rows.append(_judgement(f"m{i}-{arm}", "casualties", "present", arm=arm))
            rows += [_judgement(f"m{i}-{arm}", s, "absent", arm=arm)
                     for s in SLOTS if s != "casualties"]
    _write_judgements(tables, rows)

    out = build(tables, seed=17, machine_max=0.1, human_min=0.5)
    assert out["ok"] and set(out["arms"]) == {HUMAN_ARM, "api-strong/generic", "api-fast/generic"}
    for name in ("completeness", "slot_coverage", "visibility_ceiling"):
        assert out["paths"][name].exists(), name

    ceiling = {c.slot_id: c for c in out["ceilings"]}
    assert {s for s, c in ceiling.items() if c.is_ceiling} == {"displacement", "access_constraints", "funding"}
    assert ceiling["casualties"].is_ceiling is False  # machines do fill this one
    assert ceiling["displacement"].human_coverage == pytest.approx(1.0)
    assert ceiling["displacement"].machine_coverage_max == pytest.approx(0.0)

    with open(out["paths"]["visibility_ceiling"], encoding="utf-8", newline="") as fh:
        csv_rows = list(csv.DictReader(fh))
    assert len(csv_rows) == 4 and csv_rows[0]["is_ceiling"] == "True"  # ceiling slots sort first
    with open(out["paths"]["completeness"], encoding="utf-8", newline="") as fh:
        comp = list(csv.DictReader(fh))
    assert comp[0]["arm"] == HUMAN_ARM and float(comp[0]["completeness"]) == pytest.approx(1.0)
    assert float(comp[1]["completeness"]) == pytest.approx(0.25)  # 1 of 4 slots present


def test_build_without_human_judgements_skips_ceiling_but_still_reports(tmp_path):
    tables = tmp_path / "tables"
    _write_judgements(tables, [_judgement("m1", s, "present") for s in SLOTS])
    out = build(tables, seed=17, machine_max=0.1, human_min=0.5)
    assert out["ok"] and out["ceilings"] == []
    assert "visibility_ceiling" not in out["paths"]  # honest: no ceiling claim without a ceiling
    assert out["paths"]["completeness"].exists()


def test_build_filters_by_event_and_skips_unusable_verdicts(tmp_path):
    tables = tmp_path / "tables"
    _write_judgements(tables, [
        _judgement("a", "casualties", "present", event="e1"),
        _judgement("b", "casualties", "present", event="e2"),
        _judgement("c", "casualties", "", event="e1"),          # unusable
        _judgement("d", "casualties", "error", event="e1"),      # unusable
    ])
    out = build(tables, seed=17, machine_max=0.1, human_min=0.5, event="e1")
    assert out["n_judgements"] == 1


def test_build_with_no_judgements_is_not_an_error(tmp_path):
    out = build(tmp_path / "nope", seed=17, machine_max=0.1, human_min=0.5)
    assert out["ok"] is False and "no usable" in out["reason"]


# ---------------------------------------------------------------------------
# freeze_schema — the mandatory human checkpoint
# ---------------------------------------------------------------------------
def _candidates(n=2, examples=2):
    return {"slots": [{"slot_id": f"S{i:02d}", "name": f"slot {i}",
                       "definition": "a sufficiently long definition of this information slot",
                       "examples": [f"example {j}" for j in range(examples)], "prevalence": 0.5}
                      for i in range(1, n + 1)]}


def test_validate_enforces_plan_contract():
    ok = normalise(_candidates()["slots"])
    assert validate(ok, max_slots=15) == []
    assert any("max_slots" in p for p in validate(ok * 9, max_slots=15))
    thin = normalise(_candidates(examples=1)["slots"])
    assert any("synthetic examples" in p for p in validate(thin, max_slots=15))
    nodef = normalise([{"slot_id": "S1", "name": "x", "definition": "short", "examples": ["a", "b"]}])
    assert any("definition" in p for p in validate(nodef, max_slots=15))
    dup = normalise(_candidates()["slots"] + _candidates()["slots"])
    assert any("duplicate" in p for p in validate(dup, max_slots=15))
    assert any("no slots" in p for p in validate([], max_slots=15))


def test_review_then_approve_freezes_with_provenance(tmp_path):
    (tmp_path / "schema").mkdir(parents=True)
    (tmp_path / "schema" / "candidate_slots.yaml").write_text(
        yaml.safe_dump(_candidates(3)), encoding="utf-8")
    review = write_review(tmp_path)
    assert review.exists() and "REVIEW THIS FILE BY HAND" in review.read_text(encoding="utf-8")

    frozen, problems = approve(tmp_path, "Test Researcher", max_slots=15)
    assert problems == [] and frozen.name == "schema.yaml"
    data = yaml.safe_load(frozen.read_text(encoding="utf-8"))
    assert data["approved_by"] == "Test Researcher" and data["n_slots"] == 3
    assert data["validation_overridden"] is False and data["approved_at"].endswith("+00:00")
    assert [s["id"] for s in data["slots"]] == ["S01", "S02", "S03"]


def test_approve_refuses_invalid_schema_unless_forced(tmp_path):
    (tmp_path / "schema").mkdir(parents=True)
    (tmp_path / "schema" / "candidate_slots.yaml").write_text(
        yaml.safe_dump(_candidates(2, examples=0)), encoding="utf-8")
    src, problems = approve(tmp_path, "Someone", max_slots=15)
    assert problems and not (tmp_path / "schema.yaml").exists()  # nothing frozen
    frozen, problems = approve(tmp_path, "Someone", force=True, max_slots=15)
    assert frozen.exists()
    data = yaml.safe_load(frozen.read_text(encoding="utf-8"))
    assert data["validation_overridden"] is True  # the override is recorded, not hidden


def test_approve_without_candidates_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        approve(tmp_path, "Someone")
