"""Offline tests for src/llm.py (FakeProvider via provider_factory; tmp_path cache/log)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.llm import (
    LLM,
    ConfigError,
    CostLimitExceeded,
    FakeProvider,
    LLMResponse,
    ModelSpec,
    TransientProviderError,
    build_anthropic_kwargs,
    estimate_tokens,
)
from src.util import read_jsonl

SENTINEL = "ZEBRA-SENTINEL-7731"


def make_cfg(**llm_overrides):
    llm = {
        "cache_dir": "unused",
        "log_file": "unused",
        "max_concurrency": 2,
        "cost_limit_usd_per_run": 25,
        "retry_wait_initial_s": 0.0,
        "retry_wait_max_s": 0.0,
    }
    llm.update(llm_overrides)
    return {
        "llm": llm,
        "models": {
            "generation": [
                {"name": "api-strong", "provider": "anthropic", "id": "SET_ME"},
                {"name": "api-fast", "provider": "fake", "id": "fake-fast"},
            ],
            "coder": {"name": "coder", "provider": "fake", "id": "fake-coder"},
            "judge": {"name": "judge", "provider": "anthropic", "id": "SET_ME"},
        },
    }


def make_llm(tmp_path: Path, provider: FakeProvider | None = None, **kw) -> tuple[LLM, FakeProvider]:
    prov = provider or FakeProvider()
    cfg = kw.pop("cfg", None) or make_cfg(**kw.pop("llm_cfg", {}))
    llm = LLM(
        cfg,
        cache_dir=tmp_path / "cache",
        log_file=tmp_path / "log.jsonl",
        provider_factory=lambda spec: prov,
        **kw,
    )
    return llm, prov


# ---------------------------------------------------------------------------
# model resolution
# ---------------------------------------------------------------------------
def test_resolve_roles_names_and_raw_ids(tmp_path):
    llm, _ = make_llm(tmp_path)
    assert llm.resolve_model("coder") == ModelSpec("coder", "fake", "fake-coder")
    assert llm.resolve_model("api-fast") == ModelSpec("api-fast", "fake", "fake-fast")
    assert llm.resolve_model("claude-haiku-4-5") == ModelSpec("claude-haiku-4-5", "anthropic", "claude-haiku-4-5")
    assert llm.resolve_model("gpt-5").provider == "openai"
    assert llm.resolve_model("o3-mini").provider == "openai"
    assert llm.resolve_model("fake").provider == "fake"
    assert llm.resolve_model("fake-x").provider == "fake"
    with pytest.raises(ConfigError):
        llm.resolve_model("some-unknown-model")


def test_set_me_raises_config_error_at_resolve_time(tmp_path):
    llm, _ = make_llm(tmp_path)  # constructing must not raise
    with pytest.raises(ConfigError, match="SET_ME"):
        llm.resolve_model("judge")
    with pytest.raises(ConfigError, match="api-strong"):
        llm.resolve_model("api-strong")


# ---------------------------------------------------------------------------
# complete + cache
# ---------------------------------------------------------------------------
def test_complete_returns_text_and_tokens(tmp_path):
    llm, prov = make_llm(tmp_path)
    r = llm.complete("coder", "sys", "hello world")
    assert isinstance(r, LLMResponse)
    assert r.ok and r.text.startswith("FAKE:")
    assert r.provider == "fake" and r.model_id == "fake-coder" and r.served_model == "fake-coder"
    assert r.input_tokens == estimate_tokens("sys") + estimate_tokens("hello world")
    assert r.output_tokens == estimate_tokens(r.text)
    assert r.attempts == 1 and not r.cached and prov.calls == 1
    assert r.cost_usd == pytest.approx((r.input_tokens * 1.0 + r.output_tokens * 5.0) / 1e6)
    assert llm.run_cost_usd == pytest.approx(r.cost_usd)


def test_cache_hit_on_identical_call(tmp_path):
    llm, prov = make_llm(tmp_path)
    r1 = llm.complete("coder", "sys", "hello", tag="a", meta={"i": 1})
    cost_after_first = llm.run_cost_usd
    r2 = llm.complete("coder", "sys", "hello", tag="b", meta={"i": 2})  # tag/meta differ -> still a hit
    assert prov.calls == 1
    assert r2.cached and not r1.cached
    assert r2.text == r1.text and r2.request_hash == r1.request_hash
    assert r2.cost_usd == r1.cost_usd  # informational
    assert llm.run_cost_usd == pytest.approx(cost_after_first)  # not incremented
    assert r2.latency_s < 0.5
    # cache=False bypasses the read but still writes
    r3 = llm.complete("coder", "sys", "hello", cache=False)
    assert prov.calls == 2 and not r3.cached
    # a fresh LLM instance sees the disk cache
    llm2, prov2 = make_llm(tmp_path)
    r4 = llm2.complete("coder", "sys", "hello")
    assert r4.cached and prov2.calls == 0


def test_cache_key_sensitivity(tmp_path):
    llm, _ = make_llm(tmp_path)
    base = dict(model="coder", system="sys", user="u", max_tokens=100)

    def h(**over):
        kw = dict(base, **over)
        return llm.complete(**kw).request_hash

    h0 = h()
    assert h(system="other") != h0
    assert h(user="other") != h0
    assert h(max_tokens=101) != h0
    assert h(json_schema={"type": "object"}) != h0
    assert h(effort="low") != h0
    assert h(temperature=0.3) != h0
    assert h(thinking="adaptive") != h0
    assert h(tag="x", meta={"k": 1}) == h0


def test_refused_and_error_not_cached(tmp_path):
    llm, prov = make_llm(tmp_path)
    r = llm.complete("coder", None, "please [[REFUSE]] this")
    assert r.refused and not r.ok and r.text == "" and r.stop_reason == "refusal"
    assert r.refusal_category == "fake" and r.error is None
    r2 = llm.complete("coder", None, "please [[REFUSE]] this")
    assert prov.calls == 2 and not r2.cached


# ---------------------------------------------------------------------------
# prices + cost
# ---------------------------------------------------------------------------
def test_cost_math_with_fake_prices(tmp_path):
    llm, _ = make_llm(tmp_path)
    assert llm.cost_of("fake-coder", 1000, 100) == pytest.approx(0.0015)
    # cache read 0.1x, cache write 1.25x of input price
    assert llm.cost_of("fake-x", 0, 0, cache_read=1_000_000) == pytest.approx(0.10)
    assert llm.cost_of("fake-x", 0, 0, cache_write=1_000_000) == pytest.approx(1.25)


def test_price_lookup_prefix_override_unknown(tmp_path):
    llm, _ = make_llm(tmp_path)
    assert llm.price_for("claude-haiku-4-5-20251001") == (1.0, 5.0)
    assert llm.price_for("claude-opus-5") == (5.0, 25.0)
    assert llm.price_for("claude-opus-4-8") == (5.0, 25.0)
    assert llm.price_for("claude-sonnet-5") == (3.0, 15.0)
    assert llm.price_for("gpt-99-turbo") is None
    assert llm.cost_of("gpt-99-turbo", 10, 10) is None
    llm2, _ = make_llm(tmp_path, llm_cfg={"prices": {"gpt-99": [2.0, 8.0], "fake-coder": [0.5, 0.5]}})
    assert llm2.price_for("gpt-99-turbo") == (2.0, 8.0)
    assert llm2.price_for("fake-coder") == (0.5, 0.5)  # longer prefix wins over 'fake'
    assert llm2.price_for("fake-other") == (1.0, 5.0)


def test_unknown_price_call_records_none_cost_but_gate_assumes(tmp_path):
    llm, _ = make_llm(tmp_path, cfg=make_cfg(cost_limit_usd_per_run=0.04))
    with pytest.raises(CostLimitExceeded):
        llm.complete("gpt-99", None, "hi")  # 0.05 assumed > 0.04


# ---------------------------------------------------------------------------
# cost gate
# ---------------------------------------------------------------------------
def test_cost_gate_raises_before_provider_called(tmp_path):
    llm, prov = make_llm(tmp_path, cost_limit_usd_per_run=0.0001)
    with pytest.raises(CostLimitExceeded, match="limit"):
        llm.complete("coder", None, "x" * 4000, max_tokens=4000)
    assert prov.calls == 0
    assert llm.run_cost_usd == 0.0
    # the failed attempt is still logged
    rows = list(read_jsonl(tmp_path / "log.jsonl"))
    assert len(rows) == 1 and rows[0]["error"].startswith("CostLimitExceeded")


def test_allow_over_limit_bypasses_gate(tmp_path, monkeypatch):
    llm, prov = make_llm(tmp_path, cost_limit_usd_per_run=0.0001, allow_over_limit=True)
    r = llm.complete("coder", None, "x" * 4000, max_tokens=4000)
    assert r.ok and prov.calls == 1
    monkeypatch.setenv("LLM_ALLOW_OVER_LIMIT", "1")
    llm2, prov2 = make_llm(tmp_path / "b", cost_limit_usd_per_run=0.0001)
    assert llm2.allow_over_limit
    assert llm2.complete("coder", None, "y" * 4000, max_tokens=4000).ok


def test_estimate_cost_heuristic_and_gate(tmp_path):
    llm, _ = make_llm(tmp_path, cost_limit_usd_per_run=1.0)
    reqs = [{"model": "coder", "system": "s" * 350, "user": f"{i}" * 350, "max_tokens": 1000} for i in range(3)]
    est = llm.estimate_cost(reqs)
    assert est.method == "heuristic" and est.n_requests == 3 and est.n_cached == 0
    assert est.input_tokens == 3 * 200 and est.output_tokens == 3000
    assert est.usd == pytest.approx(3 * (200 * 1.0 + 1000 * 5.0) / 1e6)
    assert est.usd_per_model == {"coder": pytest.approx(est.usd)}
    assert not est.over_limit
    llm.gate(est)  # no raise
    llm.complete(**reqs[0])
    est2 = llm.estimate_cost(reqs)
    assert est2.n_cached == 1 and est2.usd < est.usd
    est3 = llm.estimate_cost(reqs, expected_output_tokens=10)
    assert est3.output_tokens == 20
    tight, _ = make_llm(tmp_path / "t", cost_limit_usd_per_run=0.001)
    est4 = tight.estimate_cost(reqs)
    assert est4.over_limit
    with pytest.raises(CostLimitExceeded):
        tight.gate(est4)


# ---------------------------------------------------------------------------
# complete_many
# ---------------------------------------------------------------------------
def test_complete_many_order_and_concurrency(tmp_path):
    prov = FakeProvider(latency_s=0.05)
    llm, _ = make_llm(tmp_path, provider=prov, max_concurrency=2)
    reqs = [{"model": "coder", "system": None, "user": f"item {i}", "meta": {"i": i}} for i in range(8)]
    out = llm.complete_many(reqs, progress=False)
    assert [r.meta["i"] for r in out] == list(range(8))
    assert all(r.ok for r in out)
    assert prov.calls == 8
    assert 1 <= prov.max_concurrent <= 2


def test_complete_many_captures_errors_and_stops_on_cost_limit(tmp_path):
    def boom(req):
        if "bad" in req["user"]:
            raise ValueError("boom")
        return "fine"

    llm, prov = make_llm(tmp_path, provider=FakeProvider(responder=boom))
    out = llm.complete_many([{"model": "coder", "system": None, "user": "good"},
                             {"model": "coder", "system": None, "user": "bad one"}], progress=False)
    assert out[0].ok and out[0].text == "fine"
    assert not out[1].ok and out[1].error.startswith("ValueError: boom")
    with pytest.raises(ValueError):
        llm.complete("coder", None, "bad two")
    tight, _ = make_llm(tmp_path / "t", cost_limit_usd_per_run=0.0001)
    with pytest.raises(CostLimitExceeded):
        tight.complete_many([{"model": "coder", "system": None, "user": "x" * 4000, "max_tokens": 4000}] * 4,
                            progress=False)
    out2 = tight.complete_many([{"model": "coder", "system": None, "user": "x" * 4000, "max_tokens": 4000}] * 2,
                               progress=False, stop_on_cost_limit=False)
    assert len(out2) == 2 and all(r.error.startswith("CostLimitExceeded") for r in out2)


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------
def test_json_schema_parse_and_malformed(tmp_path):
    llm, _ = make_llm(tmp_path)
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}
    r = llm.complete("coder", None, "give json", json_schema=schema)
    assert r.json == {"ok": True, "echo_len": len("give json")} and r.json_error is None
    bad, _ = make_llm(tmp_path / "b", provider=FakeProvider(responder=lambda req: "{not json"))
    r2 = bad.complete("coder", None, "give json", json_schema=schema)
    assert r2.json is None and r2.json_error and r2.ok  # parse failure is not a call error
    fenced, _ = make_llm(tmp_path / "c", provider=FakeProvider(responder=lambda req: 'Sure:\n```json\n{"a": 1}\n```'))
    assert fenced.complete("coder", None, "x", parse_json=True).json == {"a": 1}
    r3 = llm.complete("coder", None, "plain")
    assert r3.json is None and r3.json_error is None  # no parse requested


# ---------------------------------------------------------------------------
# retries / errors / refusal
# ---------------------------------------------------------------------------
def test_retries_on_transient_errors(tmp_path):
    prov = FakeProvider(fail_times=2)
    llm, _ = make_llm(tmp_path, provider=prov)
    r = llm.complete("coder", None, "retry me")
    assert r.ok and r.attempts == 3 and prov.calls == 3


def test_retries_exhausted_raise_transient(tmp_path):
    prov = FakeProvider(fail_times=10)
    llm, _ = make_llm(tmp_path, provider=prov, cfg=make_cfg(max_attempts=2, retry_wait_initial_s=0, retry_wait_max_s=0))
    with pytest.raises(TransientProviderError):
        llm.complete("coder", None, "never")
    assert prov.calls == 2


def test_refusal_handling(tmp_path):
    llm, _ = make_llm(tmp_path)
    r = llm.complete("coder", None, "[[REFUSE]]", json_schema={"type": "object"})
    assert r.refused and r.refusal_category == "fake" and r.json is None and r.json_error is None
    assert r.error is None and not r.ok
    rows = list(read_jsonl(tmp_path / "log.jsonl"))
    assert rows[-1]["refused"] is True and rows[-1]["refusal_category"] == "fake"


# ---------------------------------------------------------------------------
# anthropic kwargs builder (pure)
# ---------------------------------------------------------------------------
def test_build_anthropic_kwargs_temperature_policy():
    kw, dropped = build_anthropic_kwargs("claude-opus-5", "s", "u", 100, temperature=0.2)
    assert dropped and "temperature" not in kw
    for mid in ("claude-sonnet-5", "claude-fable-5", "claude-mythos-5", "claude-opus-4-8", "claude-opus-4-7"):
        assert build_anthropic_kwargs(mid, None, "u", 10, temperature=0.0)[1] is True
    kw, dropped = build_anthropic_kwargs("claude-haiku-4-5", "s", "u", 100, temperature=0.2)
    assert not dropped and kw["temperature"] == 0.2
    for mid in ("claude-sonnet-4-6", "claude-opus-4-6", "claude-sonnet-4-5"):
        assert build_anthropic_kwargs(mid, None, "u", 10, temperature=0.5)[1] is False
    kw, dropped = build_anthropic_kwargs("claude-opus-5", "s", "u", 100)
    assert not dropped and "temperature" not in kw


def test_build_anthropic_kwargs_shape():
    schema = {"type": "object", "properties": {"x": {"type": "string"}}}
    kw, _ = build_anthropic_kwargs("claude-sonnet-4-6", "SYS", "USER", 512, json_schema=schema, effort="high",
                                   thinking="adaptive", cache_system_prompt=True)
    assert kw["model"] == "claude-sonnet-4-6" and kw["max_tokens"] == 512
    assert kw["messages"] == [{"role": "user", "content": "USER"}]
    assert kw["system"] == [{"type": "text", "text": "SYS", "cache_control": {"type": "ephemeral"}}]
    assert kw["output_config"] == {"effort": "high", "format": {"type": "json_schema", "schema": schema}}
    assert kw["thinking"] == {"type": "adaptive"}
    for forbidden in ("budget_tokens", "top_p", "top_k", "output_format", "temperature"):
        assert forbidden not in kw and forbidden not in json.dumps(kw)
    kw2, _ = build_anthropic_kwargs("claude-haiku-4-5", "SYS", "USER", 512)
    assert kw2["system"] == "SYS" and "output_config" not in kw2 and "thinking" not in kw2
    kw3, _ = build_anthropic_kwargs("claude-haiku-4-5", None, "USER", 512, thinking="disabled")
    assert "system" not in kw3 and kw3["thinking"] == {"type": "disabled"}
    with pytest.raises(ValueError):
        build_anthropic_kwargs("claude-haiku-4-5", None, "u", 10, effort="ultra")
    with pytest.raises(ValueError):
        build_anthropic_kwargs("claude-haiku-4-5", None, "u", 10, thinking="enabled")


def test_build_anthropic_kwargs_model_guards():
    """Parameter combinations documented to 400 are dropped/downgraded per model."""
    kw, _ = build_anthropic_kwargs("claude-haiku-4-5", None, "u", 10, effort="high")
    assert "output_config" not in kw  # effort unsupported on Haiku 4.5
    kw, _ = build_anthropic_kwargs("claude-opus-4-6", None, "u", 10, effort="xhigh")
    assert kw["output_config"] == {"effort": "high"}  # xhigh arrived with Opus 4.7
    kw, _ = build_anthropic_kwargs("claude-fable-5", None, "u", 10, thinking="disabled")
    assert "thinking" not in kw  # thinking always on
    kw, _ = build_anthropic_kwargs("claude-opus-5", None, "u", 10, thinking="disabled", effort="xhigh")
    assert "thinking" not in kw and kw["output_config"] == {"effort": "xhigh"}
    kw, _ = build_anthropic_kwargs("claude-opus-5", None, "u", 10, thinking="disabled", effort="high")
    assert kw["thinking"] == {"type": "disabled"}
    kw, dropped = build_anthropic_kwargs("claude-opus-5", None, "u", 10, temperature=0.0, effort="low")
    assert dropped and "temperature" not in kw and kw["output_config"] == {"effort": "low"}


# ---------------------------------------------------------------------------
# log + summaries
# ---------------------------------------------------------------------------
def test_log_row_fields_and_no_text_leak(tmp_path):
    llm, _ = make_llm(tmp_path)
    r = llm.complete("coder", f"system {SENTINEL}", f"user {SENTINEL} text", tag="unit", meta={"k": "v"},
                     temperature=0.1, effort="low")
    llm.complete("coder", f"system {SENTINEL}", f"user {SENTINEL} text", tag="unit")  # different hash (no temp)
    log_path = tmp_path / "log.jsonl"
    raw = log_path.read_text(encoding="utf-8")
    assert SENTINEL not in raw
    assert r.text not in raw
    rows = list(read_jsonl(log_path))
    assert len(rows) == 2
    row = rows[0]
    for k in ("ts", "run_id", "tag", "meta", "provider", "model_name", "model_id", "served_model", "request_hash",
              "cached", "attempts", "latency_s", "input_tokens", "output_tokens", "cache_read_tokens",
              "cache_write_tokens", "cost_usd", "run_cost_usd_after", "stop_reason", "refused", "refusal_category",
              "max_tokens", "json_schema", "effort", "temperature", "temperature_dropped", "system_chars",
              "user_chars", "output_chars", "error"):
        assert k in row, k
    assert row["tag"] == "unit" and row["meta"] == {"k": "v"} and row["json_schema"] is False
    assert row["system_chars"] == len(f"system {SENTINEL}") and row["output_chars"] == len(r.text)
    assert row["cost_usd"] == pytest.approx(r.cost_usd) and row["run_cost_usd_after"] == pytest.approx(r.cost_usd)
    assert row["effort"] == "low" and row["temperature"] == 0.1 and row["error"] is None
    s = llm.summary()
    assert s["calls"] == 2 and s["cached"] == 0 and s["cost_usd"] == pytest.approx(llm.run_cost_usd)
    assert s["per_model"]["coder"]["calls"] == 2


def test_summarize_log_aggregates(tmp_path):
    llm, _ = make_llm(tmp_path)
    llm.complete("coder", None, "a", tag="t1")
    llm.complete("coder", None, "a", tag="t1")  # cached
    llm.complete("api-fast", None, "b", tag="t2")
    s = LLM.summarize_log(tmp_path / "log.jsonl")
    assert s["total"]["calls"] == 3 and s["total"]["cached"] == 1
    assert s["total"]["cost_usd"] == pytest.approx(llm.run_cost_usd)
    assert set(s["per_model"]) == {"coder", "api-fast"} and s["per_model"]["coder"]["calls"] == 2
    assert set(s["per_tag"]) == {"t1", "t2"} and s["per_tag"]["t2"]["calls"] == 1
    assert s["runs"] == 1
    assert LLM.summarize_log(tmp_path / "missing.jsonl")["total"]["calls"] == 0


def test_estimate_tokens():
    assert estimate_tokens(None) == 0 and estimate_tokens("") == 0
    assert estimate_tokens("abcdefg") == 2  # ceil(7/3.5)
    assert estimate_tokens("x" * 35) == 10


# ---------------------------------------------------------------------------
# additions from the Phase 0 review (env isolation, dedupe, accounting, CLI)
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    """Tests must not depend on the developer's shell / repo .env (LLM_FAKE, LLM_ALLOW_OVER_LIMIT)."""
    import src.util as util_mod

    monkeypatch.delenv("LLM_ALLOW_OVER_LIMIT", raising=False)
    monkeypatch.delenv("LLM_FAKE", raising=False)
    monkeypatch.setattr(util_mod, "_ENV_LOADED", True)


def test_concurrent_identical_requests_call_provider_once(tmp_path):
    prov = FakeProvider(latency_s=0.05)
    llm, _ = make_llm(tmp_path, provider=prov, max_concurrency=8)
    reqs = [{"model": "coder", "system": None, "user": "same prompt", "meta": {"i": i}} for i in range(20)]
    out = llm.complete_many(reqs, progress=False)
    assert all(r.ok for r in out) and [r.meta["i"] for r in out] == list(range(20))
    assert prov.calls == 1  # in-flight dedupe: the first pays, the rest read its cache entry
    assert sum(1 for r in out if r.cached) == 19
    assert llm.run_cost_usd == pytest.approx(out[0].cost_usd)


def test_cache_key_includes_model_and_serving_provider(tmp_path, monkeypatch):
    llm, prov = make_llm(tmp_path)
    r1 = llm.complete("coder", None, "hello")
    r2 = llm.complete("api-fast", None, "hello")
    assert r1.request_hash != r2.request_hash and prov.calls == 2
    # LLM_FAKE=1 serving an anthropic-spec model must not write into the anthropic cache namespace
    monkeypatch.setenv("LLM_FAKE", "1")
    fake_llm = LLM(make_cfg(), cache_dir=tmp_path / "cache", log_file=tmp_path / "log.jsonl")
    r3 = fake_llm.complete("claude-haiku-4-5", None, "hello")
    assert r3.provider == "fake" and r3.served_model == "claude-haiku-4-5"
    real_req = LLM.request_hash({"provider": "anthropic", "model_id": "claude-haiku-4-5", "system": None,
                                 "user": "hello", "max_tokens": 4096, "json_schema": None, "effort": None,
                                 "temperature": None, "thinking": None, "fallbacks": "none"})
    assert r3.request_hash != real_req
    assert not (tmp_path / "cache" / real_req[:2] / f"{real_req}.json").exists()


def test_temperature_dropped_models_share_cache_key(tmp_path):
    llm, _ = make_llm(tmp_path)
    spec = ModelSpec("x", "anthropic", "claude-opus-5")
    a = LLM.request_hash(llm._build_req(spec, None, "u", 10, None, None, 0.0, None, False))
    b = LLM.request_hash(llm._build_req(spec, "", "u", 10, None, None, None, None, False))
    assert a == b  # temperature nulled for a rejecting model; '' system == None system
    hk = ModelSpec("y", "anthropic", "claude-haiku-4-5")
    c = LLM.request_hash(llm._build_req(hk, None, "u", 10, None, None, 0.0, None, False))
    d = LLM.request_hash(llm._build_req(hk, None, "u", 10, None, None, None, None, False))
    assert c != d  # haiku honours temperature, so it stays in the key


def test_unknown_price_calls_accumulate_assumed_cost_and_trip_gate(tmp_path):
    llm, prov = make_llm(tmp_path, cost_limit_usd_per_run=0.12, llm_cfg={"unknown_price_assumed_usd_per_call": 0.05})
    r = llm.complete("gpt-99", None, "u1")  # unknown price -> cost None, assumed $0.05
    assert r.ok and r.cost_usd is None and llm.run_cost_usd == 0.0
    assert llm.run_assumed_usd == pytest.approx(0.05)
    llm.complete("gpt-99", None, "u2")
    assert llm.run_assumed_usd == pytest.approx(0.10)
    with pytest.raises(CostLimitExceeded):
        llm.complete("gpt-99", None, "u3")  # 0.10 + 0.05 > 0.12, before calling the provider
    assert prov.calls == 2
    assert llm.summary()["assumed_cost_usd"] == pytest.approx(0.10)
    rows = list(read_jsonl(tmp_path / "log.jsonl"))
    assert rows[0]["cost_usd"] is None and rows[-1]["error"].startswith("CostLimitExceeded")


def test_attempts_recorded_on_failure_paths(tmp_path):
    llm, prov = make_llm(tmp_path, provider=FakeProvider(fail_times=99), llm_cfg={"max_attempts": 3})
    with pytest.raises(TransientProviderError):
        llm.complete("coder", None, "u")
    rows = list(read_jsonl(tmp_path / "log.jsonl"))
    assert rows[-1]["attempts"] == 3 and prov.calls == 3

    def boom(req):
        raise ValueError("boom")

    llm2, prov2 = make_llm(tmp_path / "b", provider=FakeProvider(responder=boom))
    with pytest.raises(ValueError):
        llm2.complete("coder", None, "u")
    rows2 = list(read_jsonl(tmp_path / "b" / "log.jsonl"))
    assert rows2[-1]["attempts"] == 1 and prov2.calls == 1  # non-retryable: no retry


def test_errors_are_not_cached(tmp_path):
    def boom(req):
        raise ValueError("boom")

    llm, prov = make_llm(tmp_path, provider=FakeProvider(responder=boom))
    for _ in range(2):
        with pytest.raises(ValueError):
            llm.complete("coder", None, "same")
    assert prov.calls == 2 and not (tmp_path / "cache").exists()


def test_cache_false_bypasses_read_but_still_writes(tmp_path):
    llm, prov = make_llm(tmp_path)
    r1 = llm.complete("coder", None, "fresh prompt", cache=False)
    assert not r1.cached and prov.calls == 1
    r2 = llm.complete("coder", None, "fresh prompt")
    assert r2.cached and prov.calls == 1
    r3 = llm.complete("coder", None, "fresh prompt", cache=False)
    assert not r3.cached and prov.calls == 2


def test_estimate_cost_cached_requests_are_free_exactly(tmp_path):
    llm, _ = make_llm(tmp_path)
    reqs = [{"model": "coder", "system": "s" * 350, "user": f"{i}" * 350, "max_tokens": 1000} for i in range(3)]
    est = llm.estimate_cost(reqs)
    llm.complete(**reqs[0])
    est2 = llm.estimate_cost(reqs)
    assert est2.n_cached == 1 and est2.input_tokens == 400 and est2.output_tokens == 2000
    assert est2.usd == pytest.approx(est.usd * 2 / 3)


def test_complete_many_respects_kwarg_concurrency_override(tmp_path):
    import time as _t

    prov = FakeProvider(latency_s=0.05)
    llm, _ = make_llm(tmp_path, provider=prov, max_concurrency=2, llm_cfg={"max_concurrency": 8})
    assert llm.max_concurrency == 2
    reqs = [{"model": "coder", "system": None, "user": f"item {i}", "meta": {"i": i}} for i in range(8)]
    t0 = _t.perf_counter()
    out = llm.complete_many(reqs, progress=False)
    elapsed = _t.perf_counter() - t0
    assert prov.max_concurrent == 2 and elapsed < 8 * 0.05  # parallel, but capped at 2
    assert [r.meta["i"] for r in out] == list(range(8))


def test_non_serialisable_meta_does_not_lose_response(tmp_path):
    llm, prov = make_llm(tmp_path)
    r = llm.complete("coder", None, "hello", meta={"p": Path("x") / "y", "n": {1, 2}})
    assert r.ok and prov.calls == 1
    rows = list(read_jsonl(tmp_path / "log.jsonl"))
    assert isinstance(rows[-1]["meta"]["p"], str)


def test_config_validation_errors(tmp_path):
    with pytest.raises(ConfigError, match="max_concurrency"):
        make_llm(tmp_path, max_concurrency=0)
    with pytest.raises(ConfigError, match="fallbacks"):
        make_llm(tmp_path, llm_cfg={"fallbacks": "sometimes"})
    with pytest.raises(ConfigError, match="prices"):
        make_llm(tmp_path, llm_cfg={"prices": {"m": "cheap"}})
    llm, _ = make_llm(tmp_path, llm_cfg={"fallbacks": None})
    assert llm.fallbacks == "none"


def test_cli_summary_prints_totals(tmp_path, capsys):
    from src.llm import main

    llm, _ = make_llm(tmp_path)
    llm.complete("coder", None, "a", tag="t1")
    llm.complete("coder", None, "a", tag="t1")
    assert main(["--summary", "--log", str(tmp_path / "log.jsonl")]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["total"]["calls"] == 2 and out["total"]["cached"] == 1 and "t1" in out["per_tag"]
