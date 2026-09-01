"""Offline tests for src/reference_metrics.py (Phase 5).

No network, no model downloads: BERTScore is exercised through stubs / a poisoned `sys.modules`
entry, and the generic LLM judge runs on FakeProvider injected via `provider_factory`.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

from src.llm import LLM, FakeProvider
from src.reference_metrics import (
    BERTSCORE,
    PAIR_EVENT,
    PAIR_NONE,
    PAIR_SAME_DAY,
    PAIR_SAME_WINDOW,
    ROUGE_L,
    TABLE_COLUMNS,
    JUDGE_SYSTEM,
    Sitrep,
    assert_no_free_text,
    bertscore_many,
    judge_prompt,
    judge_requests,
    load_bertscorer,
    load_rouge_scorer,
    load_sitreps,
    main,
    pair_references,
    parse_date,
    parse_judge_score,
    rouge_l,
    run_judge,
    score_corpus,
    write_table,
)
from src.util import read_jsonl
from tests.test_llm import _isolate_env, make_cfg  # noqa: F401  (helpers + autouse env isolation)

SENTINEL = "OKAPI-SENTINEL-4419"
EVENT = "cyclone_idai_2019"


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------
def _human(date: str, text: str = "Official situation report body.", sid: str | None = None) -> Sitrep:
    return Sitrep(id=sid or f"ocha-{date}", event=EVENT, date=date, text=text, source="OCHA")


def _machine(date: str, text: str = "Machine sitrep body.", sid: str | None = None,
             model: str = "api-strong", arm: str = "generic") -> Sitrep:
    return Sitrep(id=sid or f"m-{date}", event=EVENT, date=date, text=text, model=model, arm=arm)


def _corpus(tmp_path: Path, machine_texts: dict[str, str] | None = None,
            human_texts: dict[str, str] | None = None) -> tuple[Path, Path]:
    """Write a tiny two-sided corpus in the project's on-disk layout; returns (machine, human)."""
    machine_texts = machine_texts or {"2019-03-20": "Heavy rain and flooding reported near Beira."}
    human_texts = human_texts or {"2019-03-20": "Heavy rain and flooding reported near Beira city."}
    mdir = tmp_path / "sitreps_machine" / EVENT
    hdir = tmp_path / "sitreps_human" / EVENT
    mdir.mkdir(parents=True)
    hdir.mkdir(parents=True)
    for date, text in machine_texts.items():
        (mdir / f"{date}__api-strong__generic.json").write_text(
            json.dumps({"id": f"m-{date}", "event": EVENT, "date": date, "model": "api-strong",
                        "arm": "generic", "text": text}), encoding="utf-8")
    for date, text in human_texts.items():
        (hdir / f"{date}.json").write_text(
            json.dumps({"id": f"ocha-{date}", "event": EVENT, "date": date, "source": "OCHA",
                        "title": "Situation Report", "text": text}), encoding="utf-8")
    return tmp_path / "sitreps_machine", tmp_path / "sitreps_human"


class _StubBert:
    """Minimal stand-in for bert_score.BERTScorer.score (no weights, no torch)."""

    def __init__(self, value: float = 0.8, fail: bool = False) -> None:
        self.value = value
        self.fail = fail
        self.seen: list[tuple[str, list[str]]] = []

    def score(self, cands, refs):
        if self.fail:
            raise RuntimeError("no weights")
        self.seen = list(zip(cands, refs))
        n = len(cands)
        return [self.value] * n, [self.value] * n, [self.value] * n


# ---------------------------------------------------------------------------
# corpus loading (degrades gracefully — nothing is collected yet)
# ---------------------------------------------------------------------------
def test_load_sitreps_missing_directory_returns_empty(tmp_path):
    assert load_sitreps(tmp_path / "nope", kind="human") == []
    assert load_sitreps(tmp_path / "nope", EVENT, kind="machine") == []


def test_load_sitreps_reads_json_and_jsonl_and_recovers_fields(tmp_path):
    ev = tmp_path / EVENT
    ev.mkdir(parents=True)
    (ev / "2019-03-21.json").write_text(json.dumps({"text": "a report"}), encoding="utf-8")
    (ev / "2019-03-22.json").write_text(json.dumps({"text": "  "}), encoding="utf-8")  # empty -> skipped
    # src.generate_sitreps writes `day`, not `date`, and no `id` at all
    (ev / "2019-03-24__api-fast__generic.json").write_text(
        json.dumps({"event": EVENT, "day": "2019-03-24", "model": "api-fast", "arm": "generic",
                    "n_posts": 12, "text": "c"}), encoding="utf-8")
    (ev / "batch.jsonl").write_text(
        json.dumps({"id": "x1", "date": "2019-03-23", "text": "b", "model": "api-fast", "arm": "schema_guided"})
        + "\n", encoding="utf-8")
    out = load_sitreps(tmp_path, kind="machine")
    # the id fallback is the file stem, exactly as in src.judge_slots.load_sitreps (the join key)
    assert [s.id for s in out] == ["2019-03-21", "2019-03-24__api-fast__generic", "x1"]
    assert out[0].event == EVENT and out[0].date == "2019-03-21"    # recovered from the path
    assert out[1].date == "2019-03-24" and out[1].arm == "generic"  # `day` alias
    assert out[2].model == "api-fast" and out[2].arm == "schema_guided"


def test_parse_date_handles_iso_datetimes_and_junk():
    assert parse_date("2019-03-21") is not None
    assert parse_date("2019-03-21T09:00:00+00:00").isoformat() == "2019-03-21"
    assert parse_date("not a date") is None and parse_date(None) is None and parse_date("2019") is None


# ---------------------------------------------------------------------------
# pairing rules (which rule was used matters for the paper)
# ---------------------------------------------------------------------------
def test_pairing_prefers_the_exact_event_day():
    humans = [_human("2019-03-19"), _human("2019-03-20"), _human("2019-03-20", sid="ifrc-2019-03-20")]
    p = pair_references(_machine("2019-03-20"), humans)
    assert p.rule == PAIR_SAME_DAY and p.day_gap == 0
    assert [h.id for h in p.refs] == ["ifrc-2019-03-20", "ocha-2019-03-20"]


def test_pairing_falls_back_to_the_window_nearest_first():
    humans = [_human("2019-03-17"), _human("2019-03-22")]
    p = pair_references(_machine("2019-03-20"), humans, window_days=3)
    assert p.rule == PAIR_SAME_WINDOW
    assert [h.id for h in p.refs] == ["ocha-2019-03-22", "ocha-2019-03-17"]  # gap 2 before gap 3
    assert p.day_gap == 2


def test_pairing_respects_window_max_refs_and_event_fallback():
    humans = [_human("2019-05-01")]
    machine = _machine("2019-03-20")
    assert pair_references(machine, humans, window_days=3).rule == PAIR_NONE
    p = pair_references(machine, humans, window_days=3, event_fallback=True)
    assert p.rule == PAIR_EVENT and p.day_gap == 42
    assert pair_references(machine, [], window_days=3, event_fallback=True).rule == PAIR_NONE
    many = [_human(f"2019-03-2{i}") for i in range(4)]  # 20..23, all within 4 days of the 24th
    capped = pair_references(_machine("2019-03-24"), many, window_days=4, max_refs=2)
    assert capped.rule == PAIR_SAME_WINDOW and [h.id for h in capped.refs] == [
        "ocha-2019-03-23", "ocha-2019-03-22"]


def test_pairing_ignores_other_events():
    other = Sitrep(id="h1", event="hurricane_maria_2017", date="2019-03-20", text="x")
    assert pair_references(_machine("2019-03-20"), [other]).rule == PAIR_NONE


# ---------------------------------------------------------------------------
# ROUGE-L
# ---------------------------------------------------------------------------
def test_rouge_l_on_a_tiny_known_pair():
    scorer = load_rouge_scorer(use_stemmer=False)
    assert rouge_l(scorer, "the cat sat on the mat", ["the cat sat on the mat"]) == {
        "p": 1.0, "r": 1.0, "f": 1.0}
    # LCS of "a b c d" vs "a b" is 2 -> p = 2/2, r = 2/4, f = 2*1*0.5/1.5
    got = rouge_l(scorer, "alpha beta", ["alpha beta gamma delta"])
    assert got["p"] == pytest.approx(1.0) and got["r"] == pytest.approx(0.5)
    assert got["f"] == pytest.approx(2 / 3)
    assert rouge_l(scorer, "alpha beta", ["zulu yankee", "alpha beta gamma"])["f"] > 0.5  # max over refs
    assert rouge_l(scorer, "alpha", []) is None and rouge_l(None, "alpha", ["alpha"]) is None
    assert rouge_l(scorer, "   ", ["alpha"]) is None


# ---------------------------------------------------------------------------
# BERTScore — skipped gracefully, never downloaded
# ---------------------------------------------------------------------------
def test_load_bertscorer_returns_none_when_package_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "bert_score", None)  # makes `from bert_score import ...` raise
    assert load_bertscorer("roberta-large") is None


def test_load_bertscorer_returns_none_when_weights_unavailable(monkeypatch):
    import types

    stub = types.ModuleType("bert_score")

    def _boom(**kwargs):
        raise OSError("could not download roberta-large")

    stub.BERTScorer = _boom  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "bert_score", stub)
    assert load_bertscorer("roberta-large") is None


def test_bertscore_many_handles_missing_scorer_refs_and_failures():
    assert bertscore_many(None, ["a", "b"], [["r"], ["r"]]) == [None, None]
    assert bertscore_many(_StubBert(), ["a"], [[]]) == [None]
    assert bertscore_many(_StubBert(fail=True), ["a"], [["r"]]) == [None]
    stub = _StubBert(0.9)
    out = bertscore_many(stub, ["a", "b"], [["r1", "r2"], []])
    assert out[0] == {"p": 0.9, "r": 0.9, "f": 0.9} and out[1] is None
    assert stub.seen == [("a", ["r1", "r2"])]  # the unpaired candidate is never sent


# ---------------------------------------------------------------------------
# generic LLM judge (no schema mention)
# ---------------------------------------------------------------------------
def test_judge_prompt_never_mentions_the_schema():
    text = (JUDGE_SYSTEM + " " + judge_prompt(_machine("2019-03-20"))).lower()
    for banned in ("schema", "slot", "operational completeness", "coverage"):
        assert banned not in text


def test_judge_requests_carry_ids_only_in_meta():
    reqs = judge_requests([_machine("2019-03-20", text=SENTINEL)])
    assert reqs[0]["meta"] == {"sitrep_id": "m-2019-03-20", "event": EVENT, "date": "2019-03-20",
                               "model": "api-strong", "arm": "generic"}
    assert SENTINEL not in json.dumps(reqs[0]["meta"])
    assert reqs[0]["json_schema"]["required"] == ["score", "justification"]


def test_parse_judge_score_rejects_garbage_and_out_of_range(tmp_path):
    llm = LLM(make_cfg(), cache_dir=tmp_path / "c", log_file=tmp_path / "l.jsonl",
              provider_factory=lambda spec: FakeProvider(responder=lambda req: '{"score": 42}'))
    resp = llm.complete("coder", None, "x", json_schema={"type": "object"})
    assert parse_judge_score(resp) == (None, "")
    llm2 = LLM(make_cfg(), cache_dir=tmp_path / "c2", log_file=tmp_path / "l2.jsonl",
               provider_factory=lambda spec: FakeProvider(responder=lambda req: "not json"))
    assert parse_judge_score(llm2.complete("coder", None, "x", json_schema={"type": "object"})) == (None, "")


def test_run_judge_scores_offline_and_logs_no_text(tmp_path):
    def responder(req):
        return json.dumps({"score": 7, "justification": "Clear and well sourced."})

    llm = LLM(make_cfg(), cache_dir=tmp_path / "c", log_file=tmp_path / "log.jsonl",
              provider_factory=lambda spec: FakeProvider(responder=responder))
    sitreps = [_machine("2019-03-20", text=SENTINEL + " flooding"), _machine("2019-03-21")]
    scores, notes = run_judge(llm, sitreps, batch=False, judge_role="coder")
    assert scores == {"m-2019-03-20": 7.0, "m-2019-03-21": 7.0}
    assert [n["sitrep_id"] for n in notes] == ["m-2019-03-20", "m-2019-03-21"]
    assert SENTINEL not in (tmp_path / "log.jsonl").read_text(encoding="utf-8")
    assert all("justification" not in row for row in read_jsonl(tmp_path / "log.jsonl"))


def test_run_judge_skips_refusals_and_gates_cost(tmp_path):
    from src.llm import CostLimitExceeded

    llm = LLM(make_cfg(), cache_dir=tmp_path / "c", log_file=tmp_path / "log.jsonl",
              provider_factory=lambda spec: FakeProvider())
    refuser = _machine("2019-03-20", text="[[REFUSE]] please")
    scores, notes = run_judge(llm, [refuser], batch=False, judge_role="coder")
    assert scores == {} and notes == []
    tight = LLM(make_cfg(cost_limit_usd_per_run=1e-9), cache_dir=tmp_path / "c2",
                log_file=tmp_path / "l2.jsonl", provider_factory=lambda spec: FakeProvider())
    with pytest.raises(CostLimitExceeded):
        run_judge(tight, [_machine("2019-03-21", text="x" * 4000)], batch=False, judge_role="coder")


# ---------------------------------------------------------------------------
# the committed table: ids and numbers only
# ---------------------------------------------------------------------------
def test_assert_no_free_text_rejects_prose_and_unknown_columns():
    assert_no_free_text([{"sitrep_id": "m-1", "rouge_l_f": 0.4}], ["sitrep_id", "rouge_l_f"])
    with pytest.raises(ValueError, match="unexpected column"):
        assert_no_free_text([{"sitrep_id": "m-1", "text": "hi"}], ["sitrep_id"])
    with pytest.raises(ValueError, match="free text"):
        assert_no_free_text([{"sitrep_id": "Cyclone Idai made landfall near Beira overnight, "
                                           "flooding the city"}], ["sitrep_id"])


def test_write_table_refuses_to_emit_sitrep_text(tmp_path):
    prose = f"{SENTINEL} the cyclone destroyed many homes across the district overnight"
    with pytest.raises(ValueError):
        write_table(tmp_path / "t.csv", [{"sitrep_id": prose}], ["sitrep_id"])
    assert not (tmp_path / "t.csv").exists()


def test_score_corpus_rows_carry_pairing_and_metrics(tmp_path):
    machine = [_machine("2019-03-20", text="alpha beta gamma"), _machine("2019-04-30", text="delta")]
    humans = [_human("2019-03-20", text="alpha beta gamma delta")]
    rows = score_corpus(machine, humans, metrics=[ROUGE_L, BERTSCORE],
                        rouge_scorer_obj=load_rouge_scorer(use_stemmer=False),
                        bert_scorer_obj=_StubBert(0.77), judge_scores={"m-2019-03-20": 6.0},
                        judge_model="claude-opus-5")
    assert [r["pairing_rule"] for r in rows] == [PAIR_SAME_DAY, PAIR_NONE]
    assert rows[0]["n_refs"] == 1 and rows[0]["ref_ids"] == "ocha-2019-03-20"
    assert rows[0]["rouge_l_f"] > 0 and rows[0]["bertscore_f1"] == 0.77
    assert rows[0]["llm_judge_score"] == 6.0 and rows[0]["judge_model"] == "claude-opus-5"
    assert rows[1]["rouge_l_f"] == "" and rows[1]["bertscore_f1"] == "" and rows[1]["judge_model"] == ""
    write_table(tmp_path / "reference_metrics.csv", rows, TABLE_COLUMNS)  # guard must pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def test_cli_reports_a_missing_corpus_instead_of_crashing(tmp_path, capsys):
    (tmp_path / "sitreps_machine").mkdir()
    (tmp_path / "sitreps_human").mkdir()
    rc = main(["--machine-dir", str(tmp_path / "sitreps_machine"),
               "--human-dir", str(tmp_path / "sitreps_human"),
               "--out", str(tmp_path / "out.csv"), "--metrics", ROUGE_L])
    assert rc == 2 and "No machine sitreps" in capsys.readouterr().out
    _corpus(tmp_path / "m")
    rc = main(["--machine-dir", str(tmp_path / "m" / "sitreps_machine"),
               "--human-dir", str(tmp_path / "sitreps_human"),
               "--out", str(tmp_path / "out.csv"), "--metrics", ROUGE_L])
    assert rc == 2 and "not collected yet" in capsys.readouterr().out


def test_cli_smoke_runs_offline_and_skips_bertscore(tmp_path, monkeypatch, capsys):
    mdir, hdir = _corpus(tmp_path)
    monkeypatch.setattr("src.reference_metrics.LLM", lambda cfg: LLM(
        cfg, cache_dir=tmp_path / "cache", log_file=tmp_path / "log.jsonl",
        provider_factory=lambda spec: FakeProvider(
            responder=lambda req: json.dumps({"score": 5, "justification": "ok"}))))
    monkeypatch.setattr("src.reference_metrics.load_bertscorer",
                        lambda *a, **k: pytest.fail("BERTScore must not load in --smoke"))
    out = tmp_path / "reference_metrics.csv"
    rc = main(["--smoke", "2", "--event", EVENT, "--machine-dir", str(mdir), "--human-dir", str(hdir),
               "--out", str(out), "--judge-model", "judge_gemini"])
    assert rc == 0
    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert len(rows) == 1 and rows[0]["pairing_rule"] == PAIR_SAME_DAY
    assert float(rows[0]["rouge_l_f"]) > 0.5 and rows[0]["bertscore_f1"] == ""
    assert rows[0]["llm_judge_score"] == "5.0"
    assert rows[0]["judge_model"] == "google/gemini-3.7-flash"
    printed = capsys.readouterr().out
    assert "scored 1 machine sitrep" in printed and "Beira" not in printed


def test_cli_saves_justifications_outside_results(tmp_path, monkeypatch, capsys):
    mdir, hdir = _corpus(tmp_path)
    monkeypatch.setattr("src.reference_metrics.LLM", lambda cfg: LLM(
        cfg, cache_dir=tmp_path / "cache", log_file=tmp_path / "log.jsonl",
        provider_factory=lambda spec: FakeProvider(
            responder=lambda req: json.dumps({"score": 8, "justification": "Well sourced."}))))
    monkeypatch.setattr("src.reference_metrics.cfg_path",
                        lambda cfg, key, default: tmp_path / "processed")
    out = tmp_path / "rm.csv"
    rc = main(["--smoke", "1", "--machine-dir", str(mdir), "--human-dir", str(hdir),
               "--out", str(out), "--save-justifications"])
    assert rc == 0
    notes = list(read_jsonl(tmp_path / "processed" / "judge_justifications.jsonl"))
    assert notes and notes[0]["justification"] == "Well sourced."
    assert "Well sourced." not in out.read_text(encoding="utf-8")  # never in the committed table


def test_cli_rejects_unknown_metric(tmp_path, capsys):
    rc = main(["--metrics", "bleu", "--machine-dir", str(tmp_path), "--human-dir", str(tmp_path)])
    assert rc == 2 and "unknown metric" in capsys.readouterr().out


def test_cli_smoke_without_llm_makes_no_calls(tmp_path, monkeypatch):
    mdir, hdir = _corpus(tmp_path)
    monkeypatch.setattr("src.reference_metrics.LLM",
                        lambda cfg: pytest.fail("no LLM may be constructed for rouge_l only"))
    out = tmp_path / "rm.csv"
    assert main(["--smoke", "1", "--machine-dir", str(mdir), "--human-dir", str(hdir),
                 "--out", str(out), "--metrics", ROUGE_L]) == 0
    row = next(csv.DictReader(out.open(encoding="utf-8")))
    assert row["llm_judge_score"] == "" and row["judge_model"] == "" and float(row["rouge_l_f"]) > 0


def test_pairing_tolerates_event_key_spelling():
    """Machine and human sitreps carry different spellings of the same event key.

    Machine sitreps inherit the HumAID stream folder (`cyclone_idai_2019`); human sitreps inherit
    the manual ReliefWeb folder (`cyclone-idai-2019`). A raw string comparison silently paired
    nothing — every sitrep fell through to rule 'none', so ROUGE-L and BERTScore were never
    computed against any reference.
    """
    from src.reference_metrics import DEFAULT_MAX_REFS, PAIR_SAME_DAY, Sitrep, pair_references

    human = Sitrep(id="h1", event="cyclone-idai-2019", date="2019-03-16", text="human text")
    machine = Sitrep(id="m1", event="cyclone_idai_2019", date="2019-03-16", text="machine text",
                     model="api-fast", arm="generic")
    p = pair_references(machine, [human], window_days=3, max_refs=DEFAULT_MAX_REFS)
    assert p.rule == PAIR_SAME_DAY, f"expected same-day pairing, got {p.rule!r}"
    assert [h.id for h in p.refs] == ["h1"]


def test_id_list_column_allows_long_ids_but_never_prose():
    """`ref_ids` is a '|'-joined id list: long is fine, whitespace is not.

    The length limit is the wrong test for an id column — reference ids are verbose, so a genuine
    pairing trips it (229 chars for 3 refs). Whitespace-freeness is a strictly stronger guard than
    length, because prose cannot be whitespace-free.
    """
    import pytest

    from src.reference_metrics import assert_no_free_text

    cols = ["sitrep_id", "ref_ids"]
    long_ids = "|".join(f"manual-cyclone-idai-2019-situation-report-number-{i}-pdf" for i in range(4))
    assert len(long_ids) > 200
    assert_no_free_text([{"sitrep_id": "m1", "ref_ids": long_ids}], cols)  # must not raise

    with pytest.raises(ValueError, match="whitespace"):
        assert_no_free_text([{"sitrep_id": "m1", "ref_ids": "some prose slipped in here"}], cols)
