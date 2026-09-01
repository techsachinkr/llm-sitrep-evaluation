"""Offline tests for src/open_code.py (Phase 2a open coding).

No network: every LLM call goes through a FakeProvider injected into `src.llm.LLM`.
Everything is written under tmp_path — nothing lands in the repo.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from src.llm import FakeProvider
from src.open_code import (
    CODE_JSON_SCHEMA,
    assign_phases,
    build_user_prompt,
    load_sitreps,
    main,
    normalise_codes,
    open_code_sitreps,
    sample_sitreps,
)
from src.util import read_json, write_json
from tests.test_llm import make_cfg, make_llm  # noqa: F401  (shared helpers + autouse env isolation)

SENTINEL = "OKAPI-SENTINEL-4417"

# 12 report dates spanning 2019-03-15 .. 2019-04-16 (the Idai tweet window).
DATES = ["2019-03-15", "2019-03-17", "2019-03-19", "2019-03-22", "2019-03-25", "2019-03-27",
         "2019-03-30", "2019-04-02", "2019-04-05", "2019-04-09", "2019-04-12", "2019-04-16"]
SOURCES = ["OCHA", "OCHA", "OCHA", "IFRC", "OCHA", "OCHA", "OCHA", "IFRC", "OCHA", "UNICEF", "OCHA", "OCHA"]

GOOD_REPLY = json.dumps(
    {
        "codes": [
            {"name": "casualty-figures", "definition": "Counts of dead and injured.",
             "evidence_quote": "446 deaths have been confirmed", "location": "Highlights",
             "confidence": "high"},
            {"name": "Access constraints", "definition": "What blocks responders from reaching people.",
             "evidence_quote": "roads to Buzi remain impassable", "location": "Situation Overview",
             "confidence": "medium"},
            {"name": "figure-sourcing", "definition": "Attribution attached to a reported figure.",
             "evidence_quote": "according to INGC", "location": "body", "confidence": "low"},
        ]
    }
)


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------
def make_corpus(tmp_path: Path, *, event: str = "cyclone_idai_2019", n: int = 12,
                marker_text: str = "") -> Path:
    """Write a synthetic human-sitrep corpus in the real on-disk format."""
    root = tmp_path / "sitreps_human"
    ev = root / event
    ev.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        date, source = DATES[i % len(DATES)], SOURCES[i % len(SOURCES)]
        rec = {
            "id": f"manual-{event}-{source.lower()}-{date}-{i}",
            "title": f"{source} {event} Situation Report No. {i + 1}",
            "source": source,
            "date": date,
            "url": f"https://reliefweb.int/report/x/{event}-{i}",
            "text": (f"{source} SITUATION REPORT No. {i + 1} ({date}).\nHighlights: "
                     f"446 deaths confirmed according to INGC. Roads to Buzi remain impassable. "
                     f"{marker_text} " + "Cluster updates follow. " * 20),
            "collection": "manual",
            "file": f"raw/{event}-{i}.html",
            "ingested_at": "2026-08-16T00:00:00+00:00",
        }
        write_json(ev / f"{date}_{i}.json", rec)
    return root


def cli_args(tmp_path: Path, corpus: Path, *extra: str) -> list[str]:
    cfg = {"seed": 17, "paths": {"raw": "data/raw", "processed": "data/processed", "results": "results"},
           "llm": {"cost_limit_usd_per_run": 25}, "schema": {"induction_sample": 60}}
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return [
        "--config", str(cfg_path),
        "--sitrep-dir", str(corpus),
        "--schema-dir", str(tmp_path / "schema"),
        "--tables-dir", str(tmp_path / "tables"),
        *extra,
    ]


# ---------------------------------------------------------------------------
# loading + phases
# ---------------------------------------------------------------------------
def test_load_assigns_event_and_thirds_of_the_date_range(tmp_path):
    records = load_sitreps(make_corpus(tmp_path) / "cyclone_idai_2019")
    assert len(records) == 12
    assert {r["event"] for r in records} == {"cyclone_idai_2019"}
    phases = {r["date"]: r["phase"] for r in records}
    assert phases["2019-03-15"] == "early"      # first day
    assert phases["2019-04-16"] == "late"       # last day
    assert set(phases.values()) == {"early", "mid", "late"}
    # undated records survive as their own stratum rather than vanishing
    rows = [{"event": "e", "date": "undated"}, {"event": "e", "date": "2020-01-01"}]
    assign_phases(rows)
    assert rows[0]["phase"] == "unknown" and rows[1]["phase"] == "early"


def test_corpus_root_and_event_dir_both_load(tmp_path):
    root = make_corpus(tmp_path)
    assert len(load_sitreps(root)) == len(load_sitreps(root / "cyclone_idai_2019")) == 12
    assert load_sitreps(tmp_path / "does_not_exist") == []


# ---------------------------------------------------------------------------
# sampling
# ---------------------------------------------------------------------------
def test_sampling_is_deterministic_under_seed_and_writes_the_audit_trail(tmp_path):
    corpus = make_corpus(tmp_path) / "cyclone_idai_2019"
    audit = tmp_path / "schema" / "induction_sample.json"
    a = sample_sitreps(corpus, 6, 17, out_path=audit)
    b = sample_sitreps(corpus, 6, 17, out_path=audit)
    assert [r["id"] for r in a] == [r["id"] for r in b]
    assert len(a) == 6

    c = sample_sitreps(corpus, 6, 999, write_audit=False)
    assert [r["id"] for r in c] != [r["id"] for r in a]  # a different seed draws differently

    saved = read_json(audit)
    assert saved["seed"] == 17 and saved["n_sampled"] == 6 and saved["n_available"] == 12
    assert saved["sitrep_ids"] == [r["id"] for r in a]
    assert saved["stratify_by"] == ["source", "phase"]
    assert sum(s["n_sampled"] for s in saved["strata"]) == 6
    # the audit trail carries ids and metadata, never report text
    assert SENTINEL not in json.dumps(saved)
    assert all("text" not in s for s in saved["sitreps"])


def test_sampling_spreads_across_sources_and_phases(tmp_path):
    corpus = make_corpus(tmp_path) / "cyclone_idai_2019"
    picked = sample_sitreps(corpus, 6, 17, write_audit=False)
    assert {r["source"] for r in picked} == {"OCHA", "IFRC", "UNICEF"}  # incl. the 1-report source
    assert {r["phase"] for r in picked} == {"early", "mid", "late"}
    # the dominant source cannot crowd out the rare ones: 8/12 reports are OCHA
    assert sum(1 for r in picked if r["source"] == "OCHA") <= 4
    assert len(sample_sitreps(corpus, 999, 17, write_audit=False)) == 12  # n > corpus is capped


def test_event_is_prepended_to_the_strata_when_the_corpus_spans_events(tmp_path):
    root = make_corpus(tmp_path)
    make_corpus(tmp_path, event="hurricane_matthew_2016", n=3)
    audit = tmp_path / "schema" / "induction_sample.json"
    picked = sample_sitreps(root, 4, 17, out_path=audit)
    assert {r["event"] for r in picked} == {"cyclone_idai_2019", "hurricane_matthew_2016"}
    assert read_json(audit)["stratify_by"] == ["event", "source", "phase"]


# ---------------------------------------------------------------------------
# prompting + hygiene
# ---------------------------------------------------------------------------
def test_prompt_carries_the_report_text_but_the_log_row_never_does(tmp_path):
    corpus = make_corpus(tmp_path, n=2, marker_text=SENTINEL) / "cyclone_idai_2019"
    records = load_sitreps(corpus)
    seen: list[str] = []

    def responder(req):
        seen.append(req["user"])
        return GOOD_REPLY

    llm, prov = make_llm(tmp_path, provider=FakeProvider(responder=responder))
    results = open_code_sitreps(llm, records, batch=False, out_dir=tmp_path / "codes", progress=False)

    assert len(seen) == 2 and all(SENTINEL in p for p in seen)   # the model sees the report
    log_text = (tmp_path / "log.jsonl").read_text(encoding="utf-8")
    assert SENTINEL not in log_text                              # ... the log never does
    rows = [json.loads(line) for line in log_text.splitlines() if line.strip()]
    assert len(rows) == 2
    assert all(set(r["meta"]) == {"stage", "event", "sitrep_id", "source", "phase"} for r in rows)
    assert all(r["tag"] == "open-code-v1" and r["json_schema"] for r in rows)
    assert all(r["output_chars"] > 0 and "codes" not in json.dumps(r) for r in rows)
    assert all(r["status"] == "ok" for r in results)


def test_long_reports_are_truncated_head_and_tail(tmp_path):
    rec = {"event": "e", "source": "OCHA", "date": "2019-03-15", "phase": "early", "title": "t",
           "text": "HEAD-MARKER " + ("x" * 5000) + " TAIL-MARKER"}
    prompt, truncated = build_user_prompt(rec, max_chars=500)
    assert truncated and "HEAD-MARKER" in prompt and "TAIL-MARKER" in prompt
    assert "characters omitted from the middle" in prompt
    short, not_truncated = build_user_prompt(rec, max_chars=10_000)
    assert not not_truncated and "omitted" not in short


# ---------------------------------------------------------------------------
# structured-output parsing
# ---------------------------------------------------------------------------
def test_json_schema_is_strict_enough_for_structured_outputs():
    item = CODE_JSON_SCHEMA["properties"]["codes"]["items"]
    assert CODE_JSON_SCHEMA["additionalProperties"] is False and item["additionalProperties"] is False
    assert set(item["required"]) == set(item["properties"])   # every field required (API requirement)
    assert item["properties"]["confidence"]["enum"] == ["high", "medium", "low"]


def test_normalise_codes_rejects_bad_shapes_and_cleans_good_ones():
    codes, err = normalise_codes(json.loads(GOOD_REPLY))
    assert err is None and [c["name"] for c in codes] == ["casualty-figures", "access-constraints",
                                                          "figure-sourcing"]
    assert codes[1]["raw_name"] == "Access constraints"  # original preserved for the audit trail
    assert codes[2]["confidence"] == "low"
    assert normalise_codes({"items": []})[1] == "missing 'codes' array"
    assert normalise_codes([])[1].startswith("expected a JSON object")
    assert normalise_codes({"codes": [{"definition": "no name"}]})[1] == "no usable codes in reply"
    long_quote = {"codes": [{"name": "n", "definition": "d", "evidence_quote": "q" * 900,
                             "location": "", "confidence": "MEDIUM"}]}
    cleaned = normalise_codes(long_quote)[0][0]
    assert len(cleaned["evidence_quote"]) == 400 and cleaned["location"] == "unspecified"
    assert cleaned["confidence"] == "medium"


def test_malformed_and_refused_replies_are_recorded_without_crashing_the_run(tmp_path):
    corpus = make_corpus(tmp_path, n=4) / "cyclone_idai_2019"
    records = sorted(load_sitreps(corpus), key=lambda r: r["id"])
    bad_id, wrong_shape_id, refuse_id = records[0]["id"], records[1]["id"], records[2]["id"]
    records[2]["text"] += " [[REFUSE]]"  # FakeProvider turns this into a real refusal

    def responder(req):  # the title is what identifies a report inside the prompt
        if records[0]["title"] in req["user"]:
            return "sorry, I cannot produce that"        # not JSON at all
        if records[1]["title"] in req["user"]:
            return json.dumps({"items": ["wrong shape"]})  # JSON, wrong schema
        return GOOD_REPLY

    llm, prov = make_llm(tmp_path, provider=FakeProvider(responder=responder))
    results = open_code_sitreps(llm, records, batch=False, out_dir=tmp_path / "codes", progress=False)

    by_id = {r["sitrep_id"]: r for r in results}
    assert len(results) == 4                                   # nothing was dropped
    assert by_id[bad_id]["status"] == "parse_error" and "json parse failed" in by_id[bad_id]["error"]
    assert by_id[wrong_shape_id]["status"] == "parse_error"
    assert by_id[refuse_id]["status"] == "refused" and by_id[refuse_id]["n_codes"] == 0
    assert by_id[records[3]["id"]]["status"] == "ok"
    # every sitrep gets a file, including the failures (the audit trail must be complete)
    assert len(list((tmp_path / "codes").glob("*.json"))) == 4
    assert read_json(next((tmp_path / "codes").glob("*.json")))["tag"] == "open-code-v1"


def test_evidence_quotes_are_stored_on_disk_for_the_audit_trail(tmp_path):
    corpus = make_corpus(tmp_path, n=1) / "cyclone_idai_2019"
    llm, _ = make_llm(tmp_path, provider=FakeProvider(responder=lambda req: GOOD_REPLY))
    results = open_code_sitreps(llm, load_sitreps(corpus), batch=False,
                                out_dir=tmp_path / "codes", progress=False)
    stored = read_json(next((tmp_path / "codes").glob("*.json")))
    assert stored["codes"][0]["evidence_quote"] == "446 deaths have been confirmed"
    assert stored["sitrep_id"] == results[0]["sitrep_id"] and stored["status"] == "ok"


# ---------------------------------------------------------------------------
# resume / cost gate / batch
# ---------------------------------------------------------------------------
def test_resume_makes_no_new_provider_calls(tmp_path):
    corpus = make_corpus(tmp_path, n=3) / "cyclone_idai_2019"
    records = load_sitreps(corpus)
    out = tmp_path / "codes"

    llm1, prov1 = make_llm(tmp_path / "run1", provider=FakeProvider(responder=lambda req: GOOD_REPLY))
    first = open_code_sitreps(llm1, records, batch=False, out_dir=out, progress=False)
    assert prov1.calls == 3 and all(r["status"] == "ok" and not r["resumed"] for r in first)

    # a fresh process with a COLD cache: the on-disk results alone must prevent every call
    llm2, prov2 = make_llm(tmp_path / "run2", provider=FakeProvider(responder=lambda req: GOOD_REPLY))
    second = open_code_sitreps(llm2, records, batch=False, out_dir=out, progress=False)
    assert prov2.calls == 0
    assert [r["sitrep_id"] for r in second] == [r["sitrep_id"] for r in first]
    assert all(r["resumed"] and r["n_codes"] == 3 for r in second)

    # --force equivalent: resume=False re-codes everything
    llm3, prov3 = make_llm(tmp_path / "run3", provider=FakeProvider(responder=lambda req: GOOD_REPLY))
    open_code_sitreps(llm3, records, batch=False, out_dir=out, resume=False, progress=False)
    assert prov3.calls == 3


def test_failed_sitreps_are_retried_on_the_next_run(tmp_path):
    corpus = make_corpus(tmp_path, n=2) / "cyclone_idai_2019"
    records = sorted(load_sitreps(corpus), key=lambda r: r["id"])

    def flaky(req):
        return "not json" if records[0]["title"] in req["user"] else GOOD_REPLY

    llm1, prov1 = make_llm(tmp_path / "r1", provider=FakeProvider(responder=flaky))
    open_code_sitreps(llm1, records, batch=False, out_dir=tmp_path / "codes", progress=False)
    llm2, prov2 = make_llm(tmp_path / "r2", provider=FakeProvider(responder=lambda req: GOOD_REPLY))
    again = open_code_sitreps(llm2, records, batch=False, out_dir=tmp_path / "codes", progress=False)
    assert prov2.calls == 1                                   # only the parse_error is resent
    assert all(r["status"] == "ok" for r in again)


def test_cost_gate_runs_before_anything_is_sent(tmp_path):
    from src.llm import CostLimitExceeded

    corpus = make_corpus(tmp_path, n=3) / "cyclone_idai_2019"
    llm, prov = make_llm(tmp_path, provider=FakeProvider(responder=lambda req: GOOD_REPLY),
                         cost_limit_usd_per_run=1e-9)
    with pytest.raises(CostLimitExceeded):
        open_code_sitreps(llm, load_sitreps(corpus), batch=False, out_dir=tmp_path / "codes",
                          progress=False)
    assert prov.calls == 0 and not list((tmp_path / "codes").glob("*.json"))


def test_batch_dispatch_uses_the_batches_api(tmp_path):
    corpus = make_corpus(tmp_path, n=3) / "cyclone_idai_2019"
    llm, prov = make_llm(tmp_path, provider=FakeProvider(responder=lambda req: GOOD_REPLY))
    results = open_code_sitreps(llm, load_sitreps(corpus), batch=True, out_dir=tmp_path / "codes",
                                progress=False)
    assert prov.batch_submissions == 1 and prov.batch_items == 3
    assert all(r["status"] == "ok" for r in results)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def test_cli_exits_2_with_a_pointer_to_the_worklists_when_no_sitreps_exist(tmp_path, capsys):
    empty = tmp_path / "sitreps_human"
    empty.mkdir()
    assert main(cli_args(tmp_path, empty, "--event", "cyclone_idai_2019", "--smoke", "2")) == 2
    out = capsys.readouterr().out
    assert "No human sitreps found" in out and "docs/worklists/" in out
    assert "src.ingest_manual_reliefweb" in out


def test_cli_smoke_run_is_synchronous_and_prints_the_codes(tmp_path, capsys):
    corpus = make_corpus(tmp_path, n=6)
    llm, prov = make_llm(tmp_path, provider=FakeProvider(responder=lambda req: GOOD_REPLY))
    rc = main(cli_args(tmp_path, corpus, "--event", "cyclone_idai_2019", "--smoke", "2"), llm=llm)
    out = capsys.readouterr().out

    assert rc == 0
    assert prov.calls == 2 and prov.batch_submissions == 0     # smoke is never batched
    assert "SMOKE run: 2 of 6 sitrep(s)" in out
    assert "projected cost" in out and "casualty-figures" in out
    assert "Smoke run complete" in out and "RESULTS.md row" not in out  # provenance is for full runs
    assert '"446 deaths' not in out                            # quotes stay internal by default
    assert (tmp_path / "schema" / "induction_sample.json").exists()
    assert len(list((tmp_path / "schema" / "open_codes").glob("*.json"))) == 2
    table = (tmp_path / "tables" / "open_codes_summary.csv").read_text(encoding="utf-8")
    assert table.splitlines()[0].startswith("sitrep_id,event,source,date,phase,status,n_codes")
    assert len(table.splitlines()) == 3


def test_cli_dry_run_sends_nothing_and_show_quotes_opts_in(tmp_path, capsys):
    corpus = make_corpus(tmp_path, n=4)
    llm, prov = make_llm(tmp_path, provider=FakeProvider(responder=lambda req: GOOD_REPLY))
    assert main(cli_args(tmp_path, corpus, "--n", "3", "--dry-run"), llm=llm) == 0
    assert prov.calls == 0 and "nothing sent" in capsys.readouterr().out

    assert main(cli_args(tmp_path, corpus, "--smoke", "1", "--show-quotes"), llm=llm) == 0
    assert "446 deaths have been confirmed" in capsys.readouterr().out


def test_cli_full_run_prints_a_provenance_row(tmp_path, capsys):
    corpus = make_corpus(tmp_path, n=4)
    llm, prov = make_llm(tmp_path, provider=FakeProvider(responder=lambda req: GOOD_REPLY))
    assert main(cli_args(tmp_path, corpus, "--n", "2"), llm=llm) == 0
    out = capsys.readouterr().out
    assert "FULL run: 2 of 4" in out
    assert "RESULTS.md row | §Methods/schema | open-coded 2/2 sitreps, 6 codes" in out
    assert "seed 17, model coder" in out
