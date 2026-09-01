"""Offline tests for src/judge_slots.py (FakeProvider via provider_factory; tmp_path everywhere).

No network, no repo writes: the data root, tables dir, LLM cache and LLM log all live in tmp_path.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

import src.judge_slots as js
from src.judge_slots import (
    Judgement,
    MissingInput,
    Sitrep,
    Slot,
    build_requests,
    build_system,
    compute_agreement,
    judge_pairs,
    load_judgement_index,
    load_schema,
    load_sitreps,
    select_pairs,
    slot_weights,
    usable_judgements,
    write_judgements,
    write_tables,
    write_validation_sample,
)
from src.llm import LLM, CostLimitExceeded, FakeProvider
from src.util import write_json

EVENT = "cyclone_idai_2019"
EVIDENCE_SENTINEL = "ZEBRA-EVIDENCE-4711"
TEXT_SENTINEL = "QUOKKA-BODY-TEXT-8823"

SLOT_YAML = """\
version: v1
slots:
  - id: casualties
    name: Casualties and figures with sourcing
    definition: Dead/injured/missing counts with an attributed source and an as-of time.
    prevalence: 0.9
    examples: ["The NDMA reports 12 deaths as of 20 March.", "At least 3 people are missing (IFRC)."]
  - id: needs
    name: Needs by cluster
    definition: Sector needs (WASH, shelter, health, food) with population figures.
    prevalence: 0.6
    examples: ["Shelter needs for 400 households.", "WASH: 2 boreholes damaged."]
  - id: funding
    name: Funding
    definition: Appeal amounts, funding received, and gaps.
    prevalence: 0.3
    examples: ["The appeal is 40 per cent funded.", "USD 2 million requested."]
"""


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    """Tests must not depend on the developer's shell / repo .env."""
    import src.util as util_mod

    monkeypatch.delenv("LLM_ALLOW_OVER_LIMIT", raising=False)
    monkeypatch.delenv("LLM_FAKE", raising=False)
    monkeypatch.setattr(util_mod, "_ENV_LOADED", True)


def make_cfg(**llm_overrides):
    llm = {"cache_dir": "unused", "log_file": "unused", "max_concurrency": 2,
           "cost_limit_usd_per_run": 25, "retry_wait_initial_s": 0.0, "retry_wait_max_s": 0.0,
           "batch": {"poll_interval_s": 0}}
    llm.update(llm_overrides)
    return {"seed": 17, "paths": {"processed": "data/processed", "results": "results"},
            "llm": llm, "models": {"judge": {"name": "judge", "provider": "fake", "id": "fake-judge"}}}


def make_data_root(tmp_path: Path, *, machine: bool = True, refusing: bool = False) -> Path:
    root = tmp_path / "processed"
    (root).mkdir(parents=True, exist_ok=True)
    (root / "schema.yaml").write_text(SLOT_YAML, encoding="utf-8")
    write_json(root / "sitreps_human" / EVENT / "2019-03-20.json", {
        "id": "ocha-sitrep-01", "title": "Cyclone Idai Situation Report No. 1", "source": "OCHA",
        "date": "2019-03-20", "url": "https://reliefweb.int/report/x",
        "text": f"OCHA reports 12 deaths as of 20 March. {TEXT_SENTINEL}. Shelter needs are growing.",
        "collection": "manual", "file": "x.pdf", "ingested_at": "2026-08-16T00:00:00+00:00"})
    write_json(root / "sitreps_human" / EVENT / "2019-03-21.json", {
        "id": "ocha-sitrep-02", "title": "Situation Report No. 2", "source": "OCHA",
        "date": "2019-03-21", "text": "IFRC reports 20 deaths. The appeal is 40 per cent funded."})
    if machine:
        body = "Posts describe flooding and rescues."
        if refusing:
            body += " [[REFUSE]]"
        write_json(root / "sitreps_machine" / EVENT / "2019-03-20__api-strong__generic.json", {
            "id": "mach-01", "event": EVENT, "date": "2019-03-20", "model": "api-strong",
            "arm": "generic", "text": body})
    return root


def responder(req: dict) -> str:
    """Deterministic fake judge: verdict decided by the slot id in the user message."""
    user = req["user"]
    slot = user.split("Score exactly this slot:\n- ", 1)[1].split(" |", 1)[0]
    if slot == "casualties":
        return json.dumps({"verdict": "present", "evidence": EVIDENCE_SENTINEL, "confidence": 0.9})
    if slot == "needs":
        return json.dumps({"verdict": "partial", "evidence": "shelter needs", "confidence": 0.5})
    if slot == "funding":
        return json.dumps({"verdict": "absent", "evidence": "", "confidence": 0.8})
    return json.dumps({"verdict": "present", "evidence": "", "confidence": 0.1})


def make_llm(tmp_path: Path, provider: FakeProvider | None = None, cfg: dict | None = None) -> tuple[LLM, FakeProvider]:
    prov = provider or FakeProvider(responder=responder)
    llm = LLM(cfg or make_cfg(), cache_dir=tmp_path / "cache", log_file=tmp_path / "log.jsonl",
              provider_factory=lambda spec: prov)
    return llm, prov


def patch_cli(monkeypatch, tmp_path: Path, provider: FakeProvider | None = None, cfg: dict | None = None):
    """Point the CLI at a tmp-path LLM (fake provider, tmp cache/log) and a test config."""
    prov = provider or FakeProvider(responder=responder)
    cfg = cfg or make_cfg()
    monkeypatch.setattr(js, "load_config", lambda: cfg)
    monkeypatch.setattr(js, "LLM", lambda c: LLM(c, cache_dir=tmp_path / "cache",
                                                 log_file=tmp_path / "log.jsonl",
                                                 provider_factory=lambda spec: prov))
    return prov


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------
def test_load_schema_reads_slots_and_prevalence(tmp_path):
    root = make_data_root(tmp_path)
    slots = load_schema(root / "schema.yaml")
    assert [s.id for s in slots] == ["casualties", "needs", "funding"]
    assert slots[0].name.startswith("Casualties") and len(slots[0].examples) == 2
    assert slot_weights(slots) == {"casualties": 0.9, "needs": 0.6, "funding": 0.3}
    assert slot_weights([Slot("a", "A", "d")]) is None  # partial prevalence -> no weighting


def test_missing_or_empty_schema_reports_the_missing_phase(tmp_path):
    with pytest.raises(MissingInput, match="Phase 2"):
        load_schema(tmp_path / "schema.yaml")
    (tmp_path / "empty.yaml").write_text("slots: []\n", encoding="utf-8")
    with pytest.raises(MissingInput, match="no slots"):
        load_schema(tmp_path / "empty.yaml")
    (tmp_path / "dupe.yaml").write_text("slots:\n  - id: a\n  - id: a\n", encoding="utf-8")
    with pytest.raises(MissingInput, match="duplicate"):
        load_schema(tmp_path / "dupe.yaml")


def test_load_sitreps_is_tolerant_and_degrades_when_nothing_collected(tmp_path):
    root = make_data_root(tmp_path)
    write_json(root / "sitreps_human" / EVENT / "empty.json", {"id": "empty", "text": "   "})
    (root / "sitreps_human" / EVENT / "broken.json").write_text("{not json", encoding="utf-8")
    human = load_sitreps(root, EVENT, "human")
    assert [s.id for s in human] == ["ocha-sitrep-01", "ocha-sitrep-02"]  # bad files skipped
    assert human[0].kind == "human" and human[0].arm_key == "human"
    machine = load_sitreps(root, EVENT, "machine")
    assert machine[0].arm_key == "api-strong/generic"
    # nothing collected yet for another event -> empty list, warning, no exception
    assert load_sitreps(root, "no_such_event", "human") == []
    with pytest.raises(ValueError):
        load_sitreps(root, EVENT, "sideways")


def test_reads_the_shapes_phase_2_and_phase_3_actually_write(tmp_path):
    """`src/consolidate.py` writes slots keyed `slot_id`; `src/generate_sitreps.py` writes
    `{event, day, model, arm, text}` files plus a `_manifest.json` that is not a sitrep."""
    root = tmp_path / "processed"
    (root).mkdir()
    (root / "schema.yaml").write_text(
        "version: v1\nslots:\n  - slot_id: s1\n    name: Casualties\n"
        "    definition: Dead and injured with sourcing.\n    prevalence: 0.8\n"
        "    examples: ['a', 'b']\n", encoding="utf-8")
    assert [s.id for s in load_schema(root / "schema.yaml")] == ["s1"]

    write_json(root / "sitreps_machine" / EVENT / "2019-03-20__api-fast__generic.json",
               {"event": EVENT, "day": "2019-03-20", "model": "api-fast", "arm": "generic",
                "n_posts": 40, "text": "Machine sitrep body.", "request_hash": "abc"})
    write_json(root / "sitreps_machine" / EVENT / "_manifest.json",
               {"event": EVENT, "rows": [{"model": "api-fast"}]})
    machine = load_sitreps(root, EVENT, "machine")
    assert len(machine) == 1                                  # the manifest is not a sitrep
    assert machine[0].id == "2019-03-20__api-fast__generic"   # id from the filename stem
    assert machine[0].date == "2019-03-20" and machine[0].arm_key == "api-fast/generic"


# ---------------------------------------------------------------------------
# prompting
# ---------------------------------------------------------------------------
def test_requests_share_one_cacheable_system_prompt_and_carry_ids_only(tmp_path):
    root = make_data_root(tmp_path)
    slots = load_schema(root / "schema.yaml")
    sitreps = load_sitreps(root, EVENT, "human")
    reqs = build_requests(select_pairs(sitreps, slots), slots)
    assert len(reqs) == len(sitreps) * len(slots)
    assert len({r["system"] for r in reqs}) == 1                 # constant -> prompt cache hits
    assert all(r["cache_system_prompt"] for r in reqs)
    assert all(r["json_schema"]["properties"]["verdict"]["enum"] == ["absent", "partial", "present"]
               for r in reqs)
    for r in reqs:  # meta carries ids/labels only (CLAUDE.md rule 7)
        assert set(r["meta"]) == {"event", "sitrep_id", "kind", "slot_id", "arm", "date"}
        assert TEXT_SENTINEL not in json.dumps(r["meta"])
    system = build_system(slots)
    assert "casualties" in system and "funding" in system      # whole catalogue, so slots are separable
    assert TEXT_SENTINEL not in system


def test_long_sitreps_are_truncated_in_the_prompt(tmp_path, caplog):
    slots = [Slot("casualties", "Casualties", "d")]
    long = Sitrep(id="long", kind="human", event=EVENT, text="x" * 500)
    with caplog.at_level("WARNING"):
        reqs = build_requests([(long, slots[0])], slots, max_chars=100)
    assert "truncated" in reqs[0]["user"] and len(reqs[0]["user"]) < 500
    assert "truncated" in caplog.text and "long" in caplog.text


def test_smoke_selection_spreads_over_kinds_and_slots(tmp_path):
    root = make_data_root(tmp_path)
    slots = load_schema(root / "schema.yaml")
    sitreps = load_sitreps(root, EVENT, "human") + load_sitreps(root, EVENT, "machine")
    pairs = select_pairs(sitreps, slots, smoke=4)
    assert len(pairs) == 4
    assert {s.kind for s, _ in pairs} == {"human", "machine"}
    assert len({sl.id for _, sl in pairs}) == 2                  # slot-major: two slots, two kinds
    assert len(select_pairs(sitreps, slots)) == len(sitreps) * len(slots)
    with pytest.raises(ValueError):
        select_pairs(sitreps, slots, smoke=0)


# ---------------------------------------------------------------------------
# judging
# ---------------------------------------------------------------------------
def test_judging_scores_every_pair_and_gates_cost(tmp_path):
    root = make_data_root(tmp_path)
    slots = load_schema(root / "schema.yaml")
    sitreps = load_sitreps(root, EVENT, "human")
    llm, prov = make_llm(tmp_path)
    pairs = select_pairs(sitreps, slots)
    judgements = judge_pairs(llm, pairs, slots, batch=False)
    assert len(judgements) == 6 and prov.calls == 6
    assert prov.last_kwargs["cache_system_prompt"] is True
    by_slot = {(j.sitrep_id, j.slot_id): j for j in judgements}
    assert by_slot[("ocha-sitrep-01", "casualties")].verdict == "present"
    assert by_slot[("ocha-sitrep-01", "casualties")].score == 1.0
    assert by_slot[("ocha-sitrep-01", "needs")].verdict == "partial"
    assert by_slot[("ocha-sitrep-01", "funding")].score == 0.0
    assert all(j.status == "ok" and j.judge_model == "fake-judge" for j in judgements)
    # (1.0 + 0.5 + 0.0) / 3 exactly, per sitrep
    from src.metrics import operational_completeness
    assert operational_completeness(usable_judgements(judgements)) == 0.5


def test_cost_gate_runs_before_any_call(tmp_path):
    root = make_data_root(tmp_path)
    slots = load_schema(root / "schema.yaml")
    sitreps = load_sitreps(root, EVENT, "human")
    llm, prov = make_llm(tmp_path, cfg=make_cfg(cost_limit_usd_per_run=1e-9))
    with pytest.raises(CostLimitExceeded):
        judge_pairs(llm, select_pairs(sitreps, slots), slots, batch=False)
    assert prov.calls == 0


def test_batch_mode_submits_one_batch_at_the_discount(tmp_path):
    root = make_data_root(tmp_path)
    slots = load_schema(root / "schema.yaml")
    sitreps = load_sitreps(root, EVENT, "human")
    llm, prov = make_llm(tmp_path)
    judgements = judge_pairs(llm, select_pairs(sitreps, slots), slots, batch=True)
    assert prov.batch_submissions == 1 and prov.batch_items == 6
    assert all(j.mode == "batch" and j.status == "ok" for j in judgements)


def test_refusal_and_malformed_verdicts_are_recorded_not_scored(tmp_path):
    root = make_data_root(tmp_path, refusing=True)
    slots = load_schema(root / "schema.yaml")
    sitreps = load_sitreps(root, EVENT, "machine")

    def bad_responder(req):
        slot = req["user"].split("Score exactly this slot:\n- ", 1)[1].split(" |", 1)[0]
        if slot == "needs":
            return json.dumps({"verdict": "maybe", "evidence": "", "confidence": 0.5})
        if slot == "funding":
            return "I am not able to produce a verdict here."
        return responder(req)

    llm, _ = make_llm(tmp_path, provider=FakeProvider(responder=bad_responder))
    judgements = judge_pairs(llm, select_pairs(sitreps, slots), slots, batch=False)
    # the machine sitrep body carries [[REFUSE]] -> the fake provider refuses every call for it
    assert {j.status for j in judgements} == {"refused"}
    assert all(j.verdict is None and j.score is None for j in judgements)
    assert usable_judgements(judgements) == []

    # same responder, a sitrep the provider does not refuse: malformed outputs stay unscored
    clean = load_sitreps(make_data_root(tmp_path / "clean"), EVENT, "human")[:1]
    llm2, _ = make_llm(tmp_path / "b", provider=FakeProvider(responder=bad_responder))
    js2 = judge_pairs(llm2, select_pairs(clean, slots), slots, batch=False)
    statuses = {j.slot_id: j.status for j in js2}
    assert statuses == {"casualties": "ok", "needs": "invalid", "funding": "invalid"}
    notes = {j.slot_id: j.note for j in js2}
    assert "maybe" in notes["needs"] and "JSON" in notes["funding"]
    assert len(usable_judgements(js2)) == 1


# ---------------------------------------------------------------------------
# outputs
# ---------------------------------------------------------------------------
def test_outputs_keep_evidence_out_of_results_tables(tmp_path):
    root = make_data_root(tmp_path)
    slots = load_schema(root / "schema.yaml")
    sitreps = load_sitreps(root, EVENT, "human")
    llm, _ = make_llm(tmp_path)
    judgements = judge_pairs(llm, select_pairs(sitreps, slots), slots, batch=False)
    written = write_judgements(judgements, root / "judgements", EVENT, sitreps)
    tables = write_tables(judgements, tmp_path / "tables", weights=slot_weights(slots))

    assert len(written) == 2
    rec = json.loads((root / "judgements" / EVENT / "ocha-sitrep-01.json").read_text(encoding="utf-8"))
    assert rec["completeness"] == 0.5 and rec["n_scored"] == 3 and rec["kind"] == "human"
    assert EVIDENCE_SENTINEL in json.dumps(rec)          # evidence lives under data/ only

    csv_text = tables["slot_judgements"].read_text(encoding="utf-8")
    assert EVIDENCE_SENTINEL not in csv_text and TEXT_SENTINEL not in csv_text
    rows = list(csv.DictReader(tables["slot_judgements"].open(encoding="utf-8")))
    assert len(rows) == 6
    row = next(r for r in rows if r["slot_id"] == "casualties" and r["sitrep_id"] == "ocha-sitrep-01")
    assert row["verdict"] == "present" and row["score"] == "1.0" and row["evidence_chars"] == str(len(EVIDENCE_SENTINEL))
    assert len(row["evidence_sha8"]) == 8
    comp = list(csv.DictReader(tables["completeness_by_sitrep"].open(encoding="utf-8")))
    assert {r["sitrep_id"] for r in comp} == {"ocha-sitrep-01", "ocha-sitrep-02"}
    assert comp[0]["completeness"] == "0.5"
    # prevalence-weighted: (0.9*1 + 0.6*0.5 + 0.3*0) / 1.8 = 0.666...
    assert float(comp[0]["completeness_weighted"]) == pytest.approx((0.9 + 0.3) / 1.8)


def test_flat_tables_accumulate_across_events_and_can_be_reset(tmp_path):
    root = make_data_root(tmp_path)
    slots = load_schema(root / "schema.yaml")
    sitreps = load_sitreps(root, EVENT, "human")
    llm, _ = make_llm(tmp_path)
    first = judge_pairs(llm, select_pairs(sitreps, slots), slots, batch=False)
    write_tables(first, tmp_path / "tables")

    other = [Judgement(event="nepal_2015", sitrep_id="n-01", kind="human", slot_id="casualties",
                       slot_name="Casualties", verdict="present", score=1.0, confidence=0.9,
                       evidence="x", status="ok")]
    tables = write_tables(other, tmp_path / "tables")
    rows = list(csv.DictReader(tables["slot_judgements"].open(encoding="utf-8")))
    assert {r["event"] for r in rows} == {EVENT, "nepal_2015"} and len(rows) == 7

    # re-judging the same (event, sitrep, slot) replaces its row rather than duplicating it
    tables = write_tables(first, tmp_path / "tables")
    rows = list(csv.DictReader(tables["slot_judgements"].open(encoding="utf-8")))
    assert len(rows) == 7
    # merge=False starts a fresh table
    tables = write_tables(other, tmp_path / "tables", merge=False)
    rows = list(csv.DictReader(tables["slot_judgements"].open(encoding="utf-8")))
    assert {r["event"] for r in rows} == {"nepal_2015"} and len(rows) == 1


def test_call_log_never_contains_sitrep_text(tmp_path):
    root = make_data_root(tmp_path)
    slots = load_schema(root / "schema.yaml")
    llm, _ = make_llm(tmp_path)
    judge_pairs(llm, select_pairs(load_sitreps(root, EVENT, "human"), slots), slots, batch=False)
    log_text = (tmp_path / "log.jsonl").read_text(encoding="utf-8")
    assert TEXT_SENTINEL not in log_text and EVIDENCE_SENTINEL not in log_text
    assert '"tag": "judge-slots-v1"' in log_text


# ---------------------------------------------------------------------------
# judge validation
# ---------------------------------------------------------------------------
def _judge_and_store(tmp_path: Path) -> tuple[Path, list[Judgement], list[Sitrep], list[Slot]]:
    root = make_data_root(tmp_path)
    slots = load_schema(root / "schema.yaml")
    sitreps = load_sitreps(root, EVENT, "human") + load_sitreps(root, EVENT, "machine")
    llm, _ = make_llm(tmp_path)
    judgements = judge_pairs(llm, select_pairs(sitreps, slots), slots, batch=False)
    write_judgements(judgements, root / "judgements", EVENT, sitreps)
    return root, judgements, sitreps, slots


def test_validation_sheet_is_blind_stratified_and_deterministic(tmp_path):
    root, judgements, sitreps, slots = _judge_and_store(tmp_path)
    out = write_validation_sample(judgements, slots, root / "judgements" / "validation_sample.csv",
                                  6, seed=17, sitreps=sitreps)
    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert len(rows) == 6
    assert all(r["human_verdict"] == "" for r in rows)               # blank column for the human
    # blind: the sheet carries no judge verdict/confidence that could anchor the hand scorer
    assert set(rows[0]) == {"event", "sitrep_id", "kind", "slot_id", "slot_name", "slot_definition",
                            "sitrep_path", "human_verdict", "notes"}
    assert len({r["slot_id"] for r in rows}) == 3                    # stratified across slots
    assert {r["kind"] for r in rows} == {"human", "machine"}
    assert rows[0]["sitrep_path"].endswith(".json")
    # deterministic under the seed
    again = write_validation_sample(judgements, slots, tmp_path / "again.csv", 6, seed=17, sitreps=sitreps)
    assert again.read_text(encoding="utf-8") == out.read_text(encoding="utf-8")
    with pytest.raises(MissingInput):
        write_validation_sample([], slots, tmp_path / "x.csv", 5)


def test_agreement_joins_hand_scores_to_stored_judgements(tmp_path):
    root, judgements, sitreps, slots = _judge_and_store(tmp_path)
    index = load_judgement_index(root / "judgements", EVENT)
    assert index[("ocha-sitrep-01", "casualties")] == "present"

    sheet = tmp_path / "scored.csv"
    with sheet.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["sitrep_id", "slot_id", "human_verdict"])
        w.writeheader()
        w.writerow({"sitrep_id": "ocha-sitrep-01", "slot_id": "casualties", "human_verdict": "present"})
        w.writerow({"sitrep_id": "ocha-sitrep-01", "slot_id": "needs", "human_verdict": "partial"})
        w.writerow({"sitrep_id": "ocha-sitrep-01", "slot_id": "funding", "human_verdict": "present"})
        w.writerow({"sitrep_id": "ocha-sitrep-02", "slot_id": "casualties", "human_verdict": ""})
        w.writerow({"sitrep_id": "ghost", "slot_id": "casualties", "human_verdict": "present"})
    agreement, counts = compute_agreement(sheet, index)
    assert counts == {"rows": 5, "blank": 1, "unmatched": 1, "used": 3}
    assert agreement.n == 3 and agreement.agreement == pytest.approx(2 / 3)


def test_agreement_input_errors_are_missing_input(tmp_path):
    root, judgements, sitreps, slots = _judge_and_store(tmp_path)
    index = load_judgement_index(root / "judgements", EVENT)
    with pytest.raises(MissingInput, match="does not exist"):
        compute_agreement(tmp_path / "nope.csv", index)
    bad = tmp_path / "bad.csv"
    bad.write_text("sitrep_id,slot_id\na,b\n", encoding="utf-8")
    with pytest.raises(MissingInput, match="missing column"):
        compute_agreement(bad, index)
    empty = tmp_path / "empty.csv"
    empty.write_text("sitrep_id,slot_id,human_verdict\nocha-sitrep-01,casualties,\n", encoding="utf-8")
    with pytest.raises(MissingInput, match="no filled-in"):
        compute_agreement(empty, index)
    typo = tmp_path / "typo.csv"
    typo.write_text("sitrep_id,slot_id,human_verdict\nocha-sitrep-01,casualties,pressent\n", encoding="utf-8")
    with pytest.raises(MissingInput, match="unknown verdict"):
        compute_agreement(typo, index)
    with pytest.raises(MissingInput, match="no stored judgements"):
        load_judgement_index(root / "judgements", "other_event")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def test_cli_smoke_runs_offline_prints_and_writes_tables(tmp_path, monkeypatch, capsys):
    root = make_data_root(tmp_path)
    prov = patch_cli(monkeypatch, tmp_path)
    rc = js.main(["--event", EVENT, "--data-root", str(root), "--tables-dir", str(tmp_path / "tables"),
                  "--smoke", "4"])
    out = capsys.readouterr().out
    assert rc == 0
    assert prov.calls == 4 and prov.batch_submissions == 0        # smoke is synchronous
    assert "smoke judgement" in out and "casualties" in out
    assert "operational completeness" in out
    assert (tmp_path / "tables" / "slot_judgements.csv").exists()
    assert (root / "judgements" / EVENT / "ocha-sitrep-01.json").exists()


def test_cli_full_run_uses_batch_and_reports_completeness(tmp_path, monkeypatch, capsys):
    root = make_data_root(tmp_path)
    prov = patch_cli(monkeypatch, tmp_path)
    rc = js.main(["--event", EVENT, "--kind", "human", "--data-root", str(root),
                  "--tables-dir", str(tmp_path / "tables"), "--batch"])
    out = capsys.readouterr().out
    assert rc == 0 and prov.batch_submissions == 1 and prov.batch_items == 6
    assert "judged 6 (sitrep, slot) pairs: 6 scored" in out
    rows = list(csv.DictReader((tmp_path / "tables" / "slot_judgements.csv").open(encoding="utf-8")))
    assert len(rows) == 6 and {r["mode"] for r in rows} == {"batch"}


def test_cli_reports_missing_phase_2_and_phase_3_inputs(tmp_path, monkeypatch, capsys):
    patch_cli(monkeypatch, tmp_path)
    empty = tmp_path / "empty_root"
    empty.mkdir()
    assert js.main(["--event", EVENT, "--data-root", str(empty)]) == 2
    root = make_data_root(tmp_path, machine=False)
    assert js.main(["--event", EVENT, "--kind", "machine", "--data-root", str(root),
                    "--tables-dir", str(tmp_path / "t")]) == 2
    assert js.main(["--event", "no_such_event", "--data-root", str(root),
                    "--tables-dir", str(tmp_path / "t")]) == 2


def test_cli_cost_gate_and_smoke_batch_conflict(tmp_path, monkeypatch):
    root = make_data_root(tmp_path)
    prov = patch_cli(monkeypatch, tmp_path, cfg=make_cfg(cost_limit_usd_per_run=1e-9))
    assert js.main(["--event", EVENT, "--data-root", str(root),
                    "--tables-dir", str(tmp_path / "t")]) == 2
    assert prov.calls == 0
    with pytest.raises(SystemExit):
        js.main(["--event", EVENT, "--data-root", str(root), "--smoke", "2", "--batch"])


def test_cli_validation_and_agreement_round_trip(tmp_path, monkeypatch, capsys):
    root = make_data_root(tmp_path)
    patch_cli(monkeypatch, tmp_path)
    tables = tmp_path / "tables"
    assert js.main(["--event", EVENT, "--data-root", str(root), "--tables-dir", str(tables)]) == 0

    assert js.main(["--event", EVENT, "--data-root", str(root), "--tables-dir", str(tables),
                    "--sample-for-validation", "5"]) == 0
    sheet = root / "judgements" / "validation_sample.csv"
    assert sheet.exists()

    rows = list(csv.DictReader(sheet.open(encoding="utf-8")))
    truth = {"casualties": "present", "needs": "partial", "funding": "absent"}
    for r in rows:  # hand-score them the way the fake judge did, except one deliberate disagreement
        r["human_verdict"] = truth[r["slot_id"]]
    rows[0]["human_verdict"] = "absent" if rows[0]["human_verdict"] != "absent" else "present"
    with sheet.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    assert js.main(["--event", EVENT, "--data-root", str(root), "--tables-dir", str(tables),
                    "--agreement", str(sheet)]) == 0
    out = capsys.readouterr().out
    assert "raw agreement" in out and "Cohen's kappa" in out
    report = list(csv.DictReader((tables / "judge_agreement.csv").open(encoding="utf-8")))
    assert len(report) == 1
    assert report[0]["n"] == str(len(rows)) and float(report[0]["raw_agreement"]) == pytest.approx(
        (len(rows) - 1) / len(rows))
