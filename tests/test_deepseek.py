"""Offline tests for the DeepSeek provider (judge back-end) — no network."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.llm import LLM, PRICES_USD_PER_MTOK, DeepSeekProvider, FakeProvider


# ---------------------------------------------------------------------------
# wiring
# ---------------------------------------------------------------------------
def test_provider_inferred_from_model_id():
    assert LLM.infer_provider("deepseek-v4-pro") == "deepseek"
    assert LLM.infer_provider("deepseek-v4-flash") == "deepseek"
    assert LLM.infer_provider("claude-opus-5") == "anthropic"
    assert LLM.infer_provider("gpt-5") == "openai"


def test_prices_are_the_conservative_peak_rates():
    """The gate must over-estimate: peak rates, never the off-peak half price."""
    assert PRICES_USD_PER_MTOK["deepseek-v4-pro"] == (1.32, 3.96)
    assert PRICES_USD_PER_MTOK["deepseek-v4-flash"] == (0.28, 0.42)


def test_deepseek_is_cheaper_than_a_batched_sonnet_judge():
    """The reason for the switch: even un-batched DeepSeek beats batched Sonnet 5."""
    ds_in, ds_out = PRICES_USD_PER_MTOK["deepseek-v4-pro"]          # peak, direct
    s_in, s_out = PRICES_USD_PER_MTOK["claude-sonnet-5"]
    assert ds_in < s_in * 0.5 and ds_out < s_out * 0.5              # vs Sonnet at the batch discount


def test_config_resolves_judge_to_deepseek():
    llm = LLM()
    spec = llm.resolve_model("judge")
    assert spec.id == "deepseek-v4-pro" and spec.provider == "deepseek"


def test_no_batch_support_so_batch_runs_fall_back_to_direct_calls(tmp_path):
    """`complete_batch` must transparently run DeepSeek work as concurrent direct calls."""
    assert DeepSeekProvider.supports_batch is False
    prov = FakeProvider()
    prov.supports_batch = False                                     # mimic DeepSeek
    cfg = {
        "llm": {"cache_dir": str(tmp_path / "c"), "log_file": str(tmp_path / "l.jsonl"),
                "max_concurrency": 4, "cost_limit_usd_per_run": 25},
        "models": {"judge": {"name": "judge", "provider": "fake", "id": "fake-judge"}},
    }
    llm = LLM(cfg, provider_factory=lambda spec: prov)
    reqs = [{"model": "judge", "system": None, "user": f"score {i}", "tag": "j"} for i in range(5)]
    out = llm.complete_batch(reqs, poll_interval_s=0)
    assert len(out) == 5 and all(r.ok for r in out)
    assert prov.batch_submissions == 0 and prov.calls == 5          # direct, not batched
    assert all(r.mode == "sync" for r in out)


# ---------------------------------------------------------------------------
# provider behaviour (stubbed SDK, no network)
# ---------------------------------------------------------------------------
class _StubBadRequest(Exception):
    pass


def _provider(monkeypatch, *, responses, raise_first=None):
    """A DeepSeekProvider whose OpenAI client is a stub recording the kwargs it receives."""
    calls: list[dict] = []

    class _Completions:
        def create(self, **kw):
            calls.append(kw)
            if raise_first and len(calls) == 1:
                raise raise_first
            r = responses[min(len(calls) - 1, len(responses) - 1)]
            return r

    client = SimpleNamespace(chat=SimpleNamespace(completions=_Completions()))
    prov = DeepSeekProvider.__new__(DeepSeekProvider)
    import threading

    import openai
    prov._sdk = openai
    prov._lock = threading.Lock()
    prov._client = client
    prov._warned_temperature = set()
    prov.max_retries, prov.timeout_s = 4, 600.0
    prov.base_url = "https://api.deepseek.com"
    prov.transient_exceptions = ()
    prov._no_json_schema = set()      # learned-once cache of models rejecting strict schemas
    return prov, calls


def _resp(text="{\"verdict\": \"present\"}", finish="stop", prompt=100, completion=20):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text, refusal=None),
                                 finish_reason=finish)],
        usage=SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion,
                              prompt_tokens_details=None),
        model="deepseek-v4-pro", id="resp_1")


def test_call_sends_openai_shape_with_json_schema(monkeypatch):
    prov, calls = _provider(monkeypatch, responses=[_resp()])
    out = prov.call({"model_id": "deepseek-v4-pro", "system": "SYS", "user": "U", "max_tokens": 400,
                     "json_schema": {"type": "object"}, "temperature": None})
    kw = calls[0]
    assert kw["model"] == "deepseek-v4-pro"
    assert kw["messages"][0]["role"] == "system" and kw["messages"][1]["content"] == "U"
    # DeepSeek reads `max_tokens`; it silently ignores OpenAI's `max_completion_tokens`,
    # so sending the latter means the cap is never applied (see max_tokens_param).
    assert kw["max_tokens"] == 400
    assert "max_completion_tokens" not in kw
    assert kw["response_format"]["type"] == "json_schema"
    assert out["text"] == "{\"verdict\": \"present\"}"
    assert out["input_tokens"] == 100 and out["output_tokens"] == 20
    assert out["served_model"] == "deepseek-v4-pro"


def test_strict_schema_rejection_falls_back_to_json_object(monkeypatch):
    import openai

    bad = openai.BadRequestError.__new__(openai.BadRequestError)
    Exception.__init__(bad, "Invalid response_format: json_schema is not supported")
    prov, calls = _provider(monkeypatch, responses=[_resp(), _resp()], raise_first=bad)
    out = prov.call({"model_id": "deepseek-v4-pro", "system": None, "user": "U", "max_tokens": 400,
                     "json_schema": {"type": "object"}})
    assert len(calls) == 2, "should retry once"
    assert calls[0]["response_format"]["type"] == "json_schema"
    assert calls[1]["response_format"] == {"type": "json_object"}   # degraded, still JSON
    assert out["text"].startswith("{")


def test_unrelated_bad_request_is_not_swallowed(monkeypatch):
    import openai

    bad = openai.BadRequestError.__new__(openai.BadRequestError)
    Exception.__init__(bad, "context length exceeded")
    prov, calls = _provider(monkeypatch, responses=[_resp()], raise_first=bad)
    with pytest.raises(openai.BadRequestError):
        prov.call({"model_id": "deepseek-v4-pro", "system": None, "user": "U", "max_tokens": 400,
                   "json_schema": {"type": "object"}})
    assert len(calls) == 1  # no pointless retry


def test_has_credentials_reads_deepseek_key(monkeypatch):
    import src.util as util_mod

    monkeypatch.setattr(util_mod, "_ENV_LOADED", True)
    prov = DeepSeekProvider.__new__(DeepSeekProvider)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_TOKEN", raising=False)
    assert prov.has_credentials() is False
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek-xxx")
    assert prov.has_credentials() is True


def test_json_schema_rejection_is_learned_once(monkeypatch):
    """The strict attempt must be skipped after the first rejection, not retried every call.

    Without this, judging 1,824 (sitrep, slot) pairs would issue 3,648 HTTP requests.
    """
    import openai

    bad = openai.BadRequestError.__new__(openai.BadRequestError)
    Exception.__init__(bad, "Invalid response_format: json_schema is not supported")
    prov, calls = _provider(monkeypatch, responses=[_resp()] * 8, raise_first=bad)
    req = {"model_id": "deepseek-v4-pro", "system": None, "user": "u", "max_tokens": 100,
           "json_schema": {"type": "object"}}
    for _ in range(4):
        prov.call(dict(req))
    kinds = [c["response_format"]["type"] for c in calls]
    assert kinds == ["json_schema", "json_object", "json_object", "json_object", "json_object"]
    assert len(calls) == 5, "4 logical calls should cost 5 requests, not 8"
    assert "deepseek-v4-pro" in prov._no_json_schema


# ---------------------------------------------------------------------------
# OpenRouter (cross-family judge)
# ---------------------------------------------------------------------------
def test_openrouter_inferred_from_namespaced_id():
    """OpenRouter ids are vendor-namespaced; that is how the provider is recognised."""
    assert LLM.infer_provider("openai/gpt-5.6-luna") == "openrouter"
    assert LLM.infer_provider("anthropic/claude-opus-5") == "openrouter"
    assert LLM.infer_provider("deepseek-v4-pro") == "deepseek"     # bare id stays direct
    assert LLM.infer_provider("claude-opus-5") == "anthropic"


def test_openrouter_prices_and_cache_multipliers():
    from src.llm import OpenRouterProvider

    assert PRICES_USD_PER_MTOK["openai/gpt-5.6-luna"] == (0.20, 1.20)
    assert OpenRouterProvider.supports_batch is False              # direct calls
    assert OpenRouterProvider.cache_read_mult == 0.10              # $0.02 read vs $0.20 input
    assert OpenRouterProvider.cache_write_mult == 1.25             # $0.25 write


def test_cross_family_judge_is_cheaper_than_the_deepseek_judge():
    ds_in, ds_out = PRICES_USD_PER_MTOK["deepseek-v4-pro"]
    or_in, or_out = PRICES_USD_PER_MTOK["openai/gpt-5.6-luna"]
    assert or_in < ds_in and or_out < ds_out


def test_openrouter_credentials_and_attribution_headers(monkeypatch):
    import src.util as util_mod
    from src.llm import OpenRouterProvider

    monkeypatch.setattr(util_mod, "_ENV_LOADED", True)
    prov = OpenRouterProvider.__new__(OpenRouterProvider)
    for var in ("OPENROUTER_API_KEY", "OPENROUTER_SITE_URL", "OPENROUTER_APP_NAME"):
        monkeypatch.delenv(var, raising=False)
    assert prov.has_credentials() is False
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-xxx")
    assert prov.has_credentials() is True


def test_config_exposes_a_cross_family_judge_role():
    llm = LLM()
    primary, xfam = llm.resolve_model("judge"), llm.resolve_model("judge_xfam")
    assert primary.provider == "deepseek"
    assert xfam.provider == "openrouter" and xfam.id == "openai/gpt-5.6-luna"
    # the point of the second judge: a different model family from the generators
    assert llm.resolve_model("api-strong").provider != xfam.provider
