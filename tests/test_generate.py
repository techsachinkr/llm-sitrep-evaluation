"""Offline tests for src/generate_sitreps.py (Phase 3).

No network: every model call goes through FakeProvider injected into `LLM`, and every path
(streams, outputs, cache, log, tables) lives under `tmp_path`.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from src.generate_sitreps import (
    ARMS,
    CostLimitExceeded,
    DayFile,
    SchemaNotAvailable,
    Slot,
    build_map_prompt,
    build_prompt,
    build_reduce_prompt,
    chunk_posts,
    format_plan,
    generate,
    load_slots,
    main,
    plan_jobs,
    posts_tokens,
    sample_posts,
    select_event_days,
)
from src.llm import LLM, FakeProvider
from src.util import read_json, write_jsonl
from tests.test_llm import make_llm  # noqa: F401  (shared helper + autouse env isolation)

EVENT = "cyclone_idai_2019"
SENTINEL = "OKAPI-SENTINEL-4412"
LABELS = ("injured_or_dead_people", "rescue_volunteering_or_donation_effort",
          "infrastructure_and_utility_damage", "displaced_people_and_evacuations",
          "requests_or_urgent_needs")
SLOTS = [
    Slot("casualties", "Casualties and figures", "Verified counts of dead, injured and missing."),
    Slot("access", "Access constraints", "Roads, bridges and airstrips blocking humanitarian access."),
    Slot("funding", "Funding", "Appeal requirements and pledges."),
]


# ---------------------------------------------------------------------------
# fixtures / builders
# ---------------------------------------------------------------------------
def make_posts(n: int, *, day: str = "2019-03-20", start_id: int = 1, text: str = "post") -> list[dict]:
    """Synthetic stream rows in the `data/processed/streams` schema."""
    return [
        {"tweet_id": str(10**18 + start_id + i), "text": f"{text} {i} " + "detail " * 8,
         "class_label": LABELS[i % len(LABELS)], "created_at": f"{day}T{i % 24:02d}:{i % 60:02d}:00+00:00",
         "event": EVENT, "split": "train"}
        for i in range(n)
    ]


def make_stream(tmp_path: Path, days: dict[str, int], *, event: str = EVENT) -> Path:
    """Write `{day: n_posts}` as stream files; returns the streams root."""
    root = tmp_path / "processed" / "streams"
    ev = root / event
    ev.mkdir(parents=True, exist_ok=True)
    for i, (day, n) in enumerate(sorted(days.items())):
        write_jsonl(ev / f"{day}.jsonl", make_posts(n, day=day, start_id=i * 10_000))
    (ev / "_stats.json").write_text(json.dumps({"event": event}), encoding="utf-8")
    return root


def run_generate(tmp_path: Path, llm: LLM, *, streams: Path, **kw):
    """generate() with every output path pinned inside tmp_path."""
    kw.setdefault("arms", ["generic"])
    kw.setdefault("models", ["api-fast"])
    return generate(llm, EVENT, stream_dir=streams, out_dir=tmp_path / "out",
                    table_path=tmp_path / "tables" / "machine_sitreps.csv", progress=False, **kw)


# ---------------------------------------------------------------------------
# day selection
# ---------------------------------------------------------------------------
def test_select_event_days_applies_the_volume_filter(tmp_path):
    streams = make_stream(tmp_path, {"2019-03-15": 5, "2019-03-16": 40, "2019-03-17": 120})
    days = select_event_days(EVENT, 30, stream_dir=streams)
    assert [d.day for d in days] == ["2019-03-16", "2019-03-17"]  # ascending, thin day dropped
    assert [d.n_posts for d in days] == [40, 120]
    assert all(isinstance(d, DayFile) and d.path.is_file() for d in days)
    assert [d.day for d in select_event_days(EVENT, 200, stream_dir=streams)] == []
    assert len(select_event_days(EVENT, 1, stream_dir=streams)) == 3  # _stats.json is not a day


def test_select_event_days_missing_stream_is_a_clear_error(tmp_path):
    streams = make_stream(tmp_path, {"2019-03-16": 40})
    with pytest.raises(FileNotFoundError, match="no social stream"):
        select_event_days("nepal_earthquake_2015", 1, stream_dir=streams)


# ---------------------------------------------------------------------------
# sampling
# ---------------------------------------------------------------------------
def test_sample_posts_is_deterministic_capped_and_stratified(tmp_path):
    streams = make_stream(tmp_path, {"2019-03-20": 200})
    day_file = streams / EVENT / "2019-03-20.jsonl"
    a = sample_posts(day_file, 20, 17)
    b = sample_posts(day_file, 20, 17)
    assert len(a) == 20
    assert [p["tweet_id"] for p in a] == [p["tweet_id"] for p in b]  # deterministic under the seed
    assert {p["class_label"] for p in a} == set(LABELS)              # every label survives the cap
    assert [p["tweet_id"] for p in a] == [p["tweet_id"] for p in sorted(a, key=lambda p: p["created_at"])]
    c = sample_posts(day_file, 20, 99)
    assert [p["tweet_id"] for p in c] != [p["tweet_id"] for p in a]  # the seed actually drives the draw
    assert {p["class_label"] for p in c} == set(LABELS)


def test_sample_posts_edge_cases(tmp_path):
    streams = make_stream(tmp_path, {"2019-03-20": 12})
    day_file = streams / EVENT / "2019-03-20.jsonl"
    assert len(sample_posts(day_file, 400, 17)) == 12          # cap above volume -> everything
    tiny = sample_posts(day_file, 3, 17)                        # cap below the label count
    assert len(tiny) == 3 and len({p["class_label"] for p in tiny}) == 3
    with pytest.raises(ValueError):
        sample_posts(day_file, 0, 17)


# ---------------------------------------------------------------------------
# prompts
# ---------------------------------------------------------------------------
def _slot_words(slots=SLOTS) -> list[str]:
    return [w for s in slots for w in (s.id, s.name, s.definition)]


def test_generic_prompt_mentions_no_slots_and_schema_guided_does():
    posts = make_posts(4)
    generic = build_prompt(posts, "generic", SLOTS, event=EVENT, day="2019-03-20")
    guided = build_prompt(posts, "schema_guided", SLOTS, event=EVENT, day="2019-03-20")
    assert "situation report" in generic.user
    for word in _slot_words():
        assert word not in generic.text, f"generic arm leaked slot vocabulary: {word!r}"
    for word in _slot_words():
        assert word in guided.user
    assert "Cyclone Idai 2019" in generic.user and "2019-03-20" in generic.user
    body = posts[0]["text"].strip()
    assert body in generic.user and body in guided.user
    assert len(guided.user) > len(generic.user)
    assert generic.system == guided.system  # only the instruction differs between arms


def test_schema_guided_without_slots_and_unknown_arm_are_refused():
    posts = make_posts(2)
    with pytest.raises(ValueError, match="slot list"):
        build_prompt(posts, "schema_guided")
    with pytest.raises(ValueError, match="unknown arm"):
        build_prompt(posts, "extractive")
    with pytest.raises(ValueError, match="slot list"):
        build_map_prompt(posts, "schema_guided", None, chunk=1, n_chunks=2)


def test_map_and_reduce_prompts_carry_the_arm():
    posts = make_posts(4)
    m_generic = build_map_prompt(posts, "generic", SLOTS, event=EVENT, day="2019-03-20", chunk=2, n_chunks=3)
    m_guided = build_map_prompt(posts, "schema_guided", SLOTS, event=EVENT, day="2019-03-20", chunk=2, n_chunks=3)
    assert "part 2 of 3" in m_generic.user and "part 2 of 3" in m_guided.user
    for word in _slot_words():
        assert word not in m_generic.text and word in m_guided.user
    digests = ["- three bridges out", "- 12 people missing"]
    r_generic = build_reduce_prompt(digests, "generic", SLOTS, event=EVENT, day="2019-03-20")
    r_guided = build_reduce_prompt(digests, "schema_guided", SLOTS, event=EVENT, day="2019-03-20")
    assert all(d in r_generic.user and d in r_guided.user for d in digests)
    for word in _slot_words():
        assert word not in r_generic.text and word in r_guided.user


# ---------------------------------------------------------------------------
# chunking / map-reduce trigger
# ---------------------------------------------------------------------------
def test_chunking_triggers_past_the_threshold_and_is_deterministic():
    posts = make_posts(60)
    assert len(chunk_posts(posts, 10_000)) == 1                     # under budget -> single shot
    budget = posts_tokens(posts) // 4
    chunks = chunk_posts(posts, budget)
    assert len(chunks) >= 4
    assert chunks == chunk_posts(posts, budget)                     # deterministic
    flat = [p for c in chunks for p in c]
    assert [p["tweet_id"] for p in flat] == [p["tweet_id"] for p in posts]  # order preserved, nothing lost
    assert all(posts_tokens(c) <= budget for c in chunks)
    assert chunk_posts([], 100) == []
    assert len(chunk_posts(posts[:1], 1)) == 1                      # an oversized post gets its own chunk
    with pytest.raises(ValueError):
        chunk_posts(posts, 0)


# ---------------------------------------------------------------------------
# generate()
# ---------------------------------------------------------------------------
def test_generate_writes_one_file_per_day_model_arm(tmp_path):
    streams = make_stream(tmp_path, {"2019-03-15": 5, "2019-03-16": 40, "2019-03-17": 60})
    llm, prov = make_llm(tmp_path)
    out = run_generate(tmp_path, llm, streams=streams, models=["api-fast", "coder"],
                       arms=["generic", "schema_guided"], slots=SLOTS, min_posts=30, cap=25)
    assert out["n_days"] == 2 and out["n_jobs"] == 8 and out["n_written"] == 8 and out["n_failed"] == 0
    names = sorted(p.name for p in (tmp_path / "out" / EVENT).glob("*.json"))
    assert names == sorted(
        [f"{d}__{m}__{a}.json" for d in ("2019-03-16", "2019-03-17")
         for m in ("api-fast", "coder") for a in ("generic", "schema_guided")] + ["_manifest.json"])
    rec = read_json(tmp_path / "out" / EVENT / "2019-03-16__api-fast__generic.json")
    assert set(rec) == {"event", "day", "model", "arm", "n_posts", "text", "generated_at", "request_hash"}
    assert rec["event"] == EVENT and rec["day"] == "2019-03-16" and rec["model"] == "api-fast"
    assert rec["arm"] == "generic" and rec["n_posts"] == 25 and rec["text"].startswith("FAKE:")
    assert len(rec["request_hash"]) == 64
    assert prov.calls == 8  # one call per sitrep; the two arms/models never share a prompt
    with open(tmp_path / "tables" / "machine_sitreps.csv", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 8 and {r["status"] for r in rows} == {"ok"}
    assert {r["map_reduce"] for r in rows} == {"False"} and {r["model_id"] for r in rows} == {"fake-fast", "fake-coder"}
    manifest = read_json(tmp_path / "out" / EVENT / "_manifest.json")
    assert manifest["seed"] == 17 and manifest["cap"] == 25 and len(manifest["rows"]) == 8


def test_generate_map_reduce_past_the_budget(tmp_path):
    streams = make_stream(tmp_path, {"2019-03-16": 60})
    llm, prov = make_llm(tmp_path)
    posts = sample_posts(streams / EVENT / "2019-03-16.jsonl", 60, 17)
    budget = posts_tokens(posts) // 3
    out = run_generate(tmp_path, llm, streams=streams, min_posts=30, cap=60, budget_tokens=budget)
    n_chunks = len(chunk_posts(posts, budget))
    assert n_chunks >= 3
    row = out["rows"][0]
    assert row["map_reduce"] is True and row["n_chunks"] == n_chunks and row["status"] == "ok"
    assert prov.calls == n_chunks + 1  # one digest per chunk, then one compose call
    rec = read_json(tmp_path / "out" / EVENT / "2019-03-16__api-fast__generic.json")
    assert rec["n_posts"] == 60 and rec["text"].startswith("FAKE:")
    # deterministic: a second run from a cold cache reproduces the same request hash and text
    llm2, prov2 = make_llm(tmp_path / "second")
    out2 = generate(llm2, EVENT, arms=["generic"], models=["api-fast"], stream_dir=streams,
                    out_dir=tmp_path / "second" / "out", table_path=tmp_path / "second" / "t.csv",
                    min_posts=30, cap=60, budget_tokens=budget, progress=False)
    assert prov2.calls == n_chunks + 1
    rec2 = read_json(tmp_path / "second" / "out" / EVENT / "2019-03-16__api-fast__generic.json")
    assert rec2["request_hash"] == rec["request_hash"] and rec2["text"] == rec["text"]


def test_rerun_is_fully_cached(tmp_path):
    streams = make_stream(tmp_path, {"2019-03-16": 40, "2019-03-17": 45})
    llm, prov = make_llm(tmp_path)
    first = run_generate(tmp_path, llm, streams=streams, min_posts=30, cap=20)
    assert first["n_written"] == 2 and prov.calls == 2
    spent = llm.run_cost_usd
    # a fresh LLM (new process) over the same cache dir must not call the provider again
    llm2, prov2 = make_llm(tmp_path, provider=prov)
    second = run_generate(tmp_path, llm2, streams=streams, min_posts=30, cap=20)
    assert second["n_written"] == 2 and prov.calls == 2 and prov2 is prov
    assert llm2.run_cost_usd == 0.0 and llm2.summary()["cached"] == 2
    assert [r["request_hash"] for r in second["rows"]] == [r["request_hash"] for r in first["rows"]]
    assert spent > 0


def test_cost_gate_blocks_an_over_budget_run(tmp_path):
    streams = make_stream(tmp_path, {"2019-03-16": 40})
    llm, prov = make_llm(tmp_path, cost_limit_usd_per_run=1e-9)
    with pytest.raises(CostLimitExceeded):
        run_generate(tmp_path, llm, streams=streams, min_posts=30, cap=20, max_tokens=4000)
    assert prov.calls == 0                                    # gated before anything was sent
    assert not (tmp_path / "out").exists()                    # and nothing was written
    llm2, prov2 = make_llm(tmp_path / "ok", provider=prov)
    out = generate(llm2, EVENT, arms=["generic"], models=["api-fast"], stream_dir=streams,
                   out_dir=tmp_path / "ok" / "out", table_path=tmp_path / "ok" / "t.csv",
                   min_posts=30, cap=20, progress=False)
    assert out["n_written"] == 1 and prov.calls == 1


def test_batch_mode_runs_through_the_batches_api(tmp_path):
    streams = make_stream(tmp_path, {"2019-03-16": 40, "2019-03-17": 45})
    llm, prov = make_llm(tmp_path)
    out = generate(llm, EVENT, arms=["generic"], models=["api-fast"], batch=True, stream_dir=streams,
                   out_dir=tmp_path / "out", table_path=tmp_path / "t.csv", min_posts=30, cap=20,
                   progress=False)
    assert out["n_written"] == 2 and prov.batch_submissions == 1 and prov.batch_items == 2
    assert llm.run_cost_usd > 0  # booked at the batch discount by src.llm


def test_dry_run_plans_without_sending(tmp_path):
    streams = make_stream(tmp_path, {"2019-03-16": 40, "2019-03-17": 45})
    llm, prov = make_llm(tmp_path)
    out = run_generate(tmp_path, llm, streams=streams, min_posts=30, cap=20, dry_run=True)
    assert prov.calls == 0 and out["n_written"] == 0 and not (tmp_path / "out").exists()
    assert out["n_jobs"] == 2 and "2019-03-16" in out["plan"] and "single" in out["plan"]
    assert out["estimate_usd"] > 0 and out["estimate_over_limit"] is False
    assert f"${out['estimate_usd']:.2f} projected" in out["plan"]
    assert not (tmp_path / "log.jsonl").exists()  # estimating is not calling


def test_generate_records_refusals_without_writing_a_file(tmp_path):
    streams = make_stream(tmp_path, {"2019-03-16": 40})
    prov = FakeProvider(responder=lambda req: "unused")
    llm, _ = make_llm(tmp_path, provider=prov)
    posts = make_posts(40, day="2019-03-16")
    posts[0]["text"] = "[[REFUSE]] " + posts[0]["text"]
    write_jsonl(streams / EVENT / "2019-03-16.jsonl", posts)
    out = run_generate(tmp_path, llm, streams=streams, min_posts=30, cap=40)
    assert out["n_written"] == 0 and out["n_failed"] == 1
    assert out["rows"][0]["status"] == "refused"
    assert list((tmp_path / "out" / EVENT).glob("2019-*.json")) == []


def test_generate_refuses_the_schema_guided_arm_without_a_schema(tmp_path):
    streams = make_stream(tmp_path, {"2019-03-16": 40})
    llm, prov = make_llm(tmp_path)
    with pytest.raises(ValueError, match="slot list"):
        run_generate(tmp_path, llm, streams=streams, arms=["schema_guided"], min_posts=30, cap=20)
    assert prov.calls == 0


# ---------------------------------------------------------------------------
# hygiene: no social text anywhere near the log (CLAUDE.md rule 7)
# ---------------------------------------------------------------------------
def test_social_text_never_reaches_the_llm_log(tmp_path):
    streams = make_stream(tmp_path, {"2019-03-16": 40})
    posts = make_posts(40, day="2019-03-16", text=f"{SENTINEL} flooding")
    write_jsonl(streams / EVENT / "2019-03-16.jsonl", posts)
    llm, _ = make_llm(tmp_path)
    budget = posts_tokens(posts) // 3
    out = run_generate(tmp_path, llm, streams=streams, arms=["generic", "schema_guided"], slots=SLOTS,
                       min_posts=30, cap=40, budget_tokens=budget)
    assert out["n_written"] == 2
    raw = (tmp_path / "log.jsonl").read_text(encoding="utf-8")
    assert SENTINEL not in raw and "flooding" not in raw
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    assert rows and all(set(r["meta"]) <= {"event", "day", "model", "arm", "stage", "chunk", "n_chunks"}
                        for r in rows)
    assert all(r["output_chars"] >= 0 and "text" not in r for r in rows)
    # the run table carries counts and ids only
    assert SENTINEL not in (tmp_path / "tables" / "machine_sitreps.csv").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# schema loading
# ---------------------------------------------------------------------------
def test_load_slots_accepts_both_layouts_and_reports_absence(tmp_path):
    (tmp_path / "schema").mkdir()
    plain = tmp_path / "schema" / "schema.yaml"
    plain.write_text("- id: casualties\n  name: Casualties\n  definition: Dead and injured.\n",
                     encoding="utf-8")
    slots = load_slots(processed_dir=tmp_path)
    assert slots == [Slot("casualties", "Casualties", "Dead and injured.")]
    plain.unlink()
    (tmp_path / "schema" / "candidate_slots.yaml").write_text(
        "slots:\n  - id: access\n    name: Access\n  - id: funding\n", encoding="utf-8")
    # CLAUDE.md rule 4: un-adjudicated candidates must NOT silently drive the schema_guided arm.
    with pytest.raises(SchemaNotAvailable, match="UN-ADJUDICATED"):
        load_slots(processed_dir=tmp_path)
    cands = load_slots(processed_dir=tmp_path, allow_candidates=True)  # deliberate override
    assert [s.id for s in cands] == ["access", "funding"] and cands[1].name == "funding"
    with pytest.raises(SchemaNotAvailable, match="no Phase 2 schema"):
        load_slots(processed_dir=tmp_path / "empty")


def test_approved_schema_is_used_without_an_override(tmp_path):
    """A frozen schema (approved_by present) loads normally — the guard targets candidates only."""
    (tmp_path / "schema").mkdir()
    (tmp_path / "schema.yaml").write_text(
        "approved_by: Test Human\napproved_at: '2026-08-30T00:00:00+00:00'\n"
        "slots:\n  - id: casualties\n    name: Casualties\n    definition: Dead and injured.\n",
        encoding="utf-8")
    slots = load_slots(processed_dir=tmp_path)
    assert [s.id for s in slots] == ["casualties"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def test_cli_smoke_runs_offline(tmp_path, monkeypatch, capsys):
    import src.generate_sitreps as gen

    streams = make_stream(tmp_path, {"2019-03-15": 5, "2019-03-16": 40, "2019-03-17": 45})
    monkeypatch.setenv("LLM_FAKE", "1")
    monkeypatch.setattr(LLM, "_abs", staticmethod(lambda p: tmp_path / Path(p).name))
    monkeypatch.setattr(gen, "_default_processed_dir", lambda: tmp_path / "processed")
    monkeypatch.setattr(gen, "_default_table_path", lambda: tmp_path / "tables" / "machine.csv")
    assert streams.exists()
    rc = main(["--event", EVENT, "--models", "api-fast", "--smoke", "1", "--cap", "15"])
    printed = capsys.readouterr().out
    assert rc == 0
    assert "2019-03-16" in printed and "single" in printed          # the plan table
    assert "| 2019-03-16 | api-fast | generic | 15 posts ===" in printed.replace("=== ", "| ")
    assert "FAKE:" in printed and "RESULTS.md row" in printed
    files = sorted(p.name for p in (tmp_path / "processed" / "sitreps_machine" / EVENT).glob("*.json"))
    assert files == ["2019-03-16__api-fast__generic.json", "_manifest.json"]
    assert (tmp_path / "tables" / "machine.csv").is_file()


def test_cli_reports_missing_stream_and_missing_schema(tmp_path, monkeypatch, capsys):
    import src.generate_sitreps as gen

    make_stream(tmp_path, {"2019-03-16": 40})
    monkeypatch.setenv("LLM_FAKE", "1")
    monkeypatch.setattr(LLM, "_abs", staticmethod(lambda p: tmp_path / Path(p).name))
    monkeypatch.setattr(gen, "_default_processed_dir", lambda: tmp_path / "processed")
    monkeypatch.setattr(gen, "_default_table_path", lambda: tmp_path / "tables" / "machine.csv")
    assert main(["--event", "nepal_earthquake_2015", "--smoke", "1"]) == 2
    assert "no social stream" in capsys.readouterr().out
    assert main(["--event", EVENT, "--arms", "schema_guided", "--smoke", "1"]) == 2
    assert "no Phase 2 schema" in capsys.readouterr().out
    assert main(["--event", EVENT, "--smoke", "1", "--min-posts", "9999"]) == 2
    assert "no event-day" in capsys.readouterr().out


def test_cli_dry_run_sends_nothing(tmp_path, monkeypatch, capsys):
    import src.generate_sitreps as gen

    make_stream(tmp_path, {"2019-03-16": 40, "2019-03-17": 45})
    monkeypatch.setenv("LLM_FAKE", "1")
    monkeypatch.setattr(LLM, "_abs", staticmethod(lambda p: tmp_path / Path(p).name))
    monkeypatch.setattr(gen, "_default_processed_dir", lambda: tmp_path / "processed")
    monkeypatch.setattr(gen, "_default_table_path", lambda: tmp_path / "tables" / "machine.csv")
    assert main(["--event", EVENT, "--models", "api-fast", "--dry-run"]) == 0
    printed = capsys.readouterr().out
    assert "2 sitrep(s) to generate" in printed and "projected" in printed
    assert not (tmp_path / "processed" / "sitreps_machine").exists()
    assert not (tmp_path / "log.jsonl").exists()


def test_format_plan_and_arms_constant(tmp_path):
    streams = make_stream(tmp_path, {"2019-03-16": 40})
    days = select_event_days(EVENT, 30, stream_dir=streams)
    jobs = plan_jobs(EVENT, days, arms=["generic"], models=["api-fast"], cap=10, seed=17,
                     budget_tokens=20)
    plan = format_plan(jobs, batch=True, estimate_usd=0.42)
    assert "map-reduce" in plan and "Message Batches API" in plan and "$0.42" in plan
    assert ARMS == ("generic", "schema_guided")
