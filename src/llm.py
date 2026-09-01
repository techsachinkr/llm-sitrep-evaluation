"""The single LLM entry point for this project (CLAUDE.md rule 2).

Everything that talks to a model goes through `LLM.complete` / `LLM.complete_many`:

* disk cache keyed on the full request (never on tag/meta),
* JSONL log with tokens + cost per call (NEVER prompt/response text — only lengths),
* tenacity retries on transient provider errors,
* a concurrency cap for batch runs,
* a cost meter with a hard, fail-closed gate at `llm.cost_limit_usd_per_run`.

Provider back-ends live in `src/llm_providers.py` (Anthropic, OpenAI, Fake); the pure
`build_anthropic_kwargs` builder and `FakeProvider` are re-exported here for tests.

Bulk work (Phases 2-5) should go through `LLM.complete_batch` — the Anthropic Message Batches
API (50% price, asynchronous, up to 24 h) — with the same request dicts as `complete_many`,
the same cache/log/cost gate, and an on-disk manifest that makes an interrupted batch run
resumable (re-running with the same requests re-attaches to the submitted batch instead of
paying twice).

CLI:  python -m src.llm --summary [--log path]
      python -m src.llm --smoke [--model coder] [--batch]   (LLM_FAKE=1 for an offline smoke)
      python -m src.llm --batch-status <batch_id>
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import threading
import time
import uuid
from concurrent.futures import FIRST_EXCEPTION, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

from src.llm_providers import (  # noqa: F401  (re-exported public names)
    AnthropicProvider,
    BatchUnavailableError,
    DeepSeekProvider,
    OpenRouterProvider,
    FakeProvider,
    OpenAIProvider,
    TransientProviderError,
    build_anthropic_kwargs,
    estimate_tokens,
    model_rejects_temperature,
    openai_model_rejects_temperature,
)
from src.util import (
    ROOT, append_jsonl, env, get_logger, load_config, read_json, read_jsonl, sha256_of, stable_json, utc_now_iso,
)

log = get_logger("sitrep.llm")

__all__ = [
    "LLM", "LLMResponse", "ModelSpec", "CostEstimate", "CostLimitExceeded", "ConfigError",
    "TransientProviderError", "OutputTruncated", "BatchUnavailableError", "estimate_tokens",
    "PRICES_USD_PER_MTOK",
    "build_anthropic_kwargs", "FakeProvider", "AnthropicProvider", "OpenAIProvider",
    "DeepSeekProvider", "OpenRouterProvider",
]

# ---------------------------------------------------------------------------
# Prices (USD per million tokens: (input, output)); cache read = 0.10x input,
# cache write (5-minute) = 1.25x input. Lookup = longest key that prefixes the id.
# ---------------------------------------------------------------------------
PRICES_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-fable-5": (10.0, 50.0), "claude-mythos-5": (10.0, 50.0),
    "claude-opus-5": (5.0, 25.0), "claude-opus-4-8": (5.0, 25.0), "claude-opus-4-7": (5.0, 25.0),
    "claude-opus-4-6": (5.0, 25.0), "claude-opus-4-5": (5.0, 25.0),
    "claude-sonnet-5": (3.0, 15.0),  # list price; intro $2/$10 through 2026-08-31 -- gate at list
    "claude-sonnet-4-6": (3.0, 15.0), "claude-sonnet-4-5": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
    # DeepSeek (https://api.deepseek.com). Peak rates are used deliberately: the cost gate must
    # over-estimate, never under-estimate. Off-peak (most of the day) is half of these.
    "deepseek-v4-pro": (1.32, 3.96),
    # OpenRouter (namespaced ids). gpt-5.6-luna: $0.20 in / $1.20 out, cache read $0.02.
    "openai/gpt-5.6-luna": (0.20, 1.20),
    "openai/gpt-5.6-terra": (0.20, 1.20),
    "openai/gpt-5.6": (1.25, 10.0),
    # qwen3.8-flash: $0.15 in / $0.47 out, cache read $0.016. Cheap third judge — tests whether
    # judge *capability* changes the completeness numbers, not just judge *family*.
    "qwen/qwen3.8-flash": (0.15, 0.47),
    "qwen/qwen3.8-27b": (0.425, 2.55),
    "qwen/qwen3.8-max": (2.00, 6.00),
    # gemini-3.7-flash: $0.75 in / $3.75 out; the ":batch" slug is the same model at 75% off
    # (OpenRouter exposes batch pricing as a separate model id, selected here by config).
    "google/gemini-3.7-flash": (0.75, 3.75),
    "google/gemini-3.7-flash:batch": (0.1875, 0.9375),
    "deepseek-v4-flash": (0.28, 0.42),
    "deepseek": (1.32, 3.96),
    "fake": (1.0, 5.0),
}
CACHE_READ_MULT = 0.10
CACHE_WRITE_MULT = 1.25
DEFAULT_UNKNOWN_PRICE_USD_PER_CALL = 0.05


class ConfigError(ValueError):
    """Bad or unset configuration (e.g. a model id still equal to SET_ME)."""


class CostLimitExceeded(RuntimeError):
    """Projected or actual spend would exceed `llm.cost_limit_usd_per_run`."""


class OutputTruncated(RuntimeError):
    """stop_reason == max_tokens and the caller asked for `max_tokens_hard=True`."""


@dataclass(frozen=True)
class ModelSpec:
    """A resolved model: config `name` (role/entry), `provider`, and the API model `id`."""

    name: str
    provider: str
    id: str


@dataclass
class CostEstimate:
    """Projected cost of a batch of requests (see `LLM.estimate_cost`)."""

    n_requests: int
    n_cached: int
    input_tokens: int
    output_tokens: int
    usd: float
    usd_per_model: dict[str, float]
    method: str  # 'api' | 'heuristic' | 'mixed'
    over_limit: bool
    limit_usd: float


@dataclass
class LLMResponse:
    """Normalised result of one `LLM.complete` call (never logged with its text)."""

    text: str = ""
    json: Any | None = None
    json_error: str | None = None
    model_name: str = ""
    model_id: str = ""
    served_model: str | None = None
    provider: str = ""
    stop_reason: str | None = None
    refused: bool = False
    refusal_category: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float | None = None
    cached: bool = False
    request_hash: str = ""
    latency_s: float = 0.0
    attempts: int = 0
    error: str | None = None
    tag: str = ""
    meta: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] | None = None
    mode: str = "sync"          # 'sync' | 'batch' | 'cache'
    batch_id: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and not self.refused


_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def parse_json_text(text: str) -> tuple[Any | None, str | None]:
    """Parse model output as JSON; tolerate ```json fences and leading prose."""
    if not text or not text.strip():
        return None, "empty output"
    candidates = [text.strip()]
    m = _JSON_FENCE.search(text)
    if m:
        candidates.append(m.group(1).strip())
    for opener, closer in (("{", "}"), ("[", "]")):
        i, j = text.find(opener), text.rfind(closer)
        if i != -1 and j > i:
            candidates.append(text[i : j + 1])
    last_err = "no JSON found"
    for cand in candidates:
        try:
            return json.loads(cand), None
        except json.JSONDecodeError as exc:
            last_err = f"{exc.msg} at pos {exc.pos}"
    return None, last_err


def _parse_iso(value: Any) -> float:
    """ISO-8601 timestamp -> POSIX seconds (0.0 when unparsable, i.e. "very old")."""
    try:
        from datetime import datetime

        return datetime.fromisoformat(str(value)).timestamp()
    except (TypeError, ValueError):
        return 0.0


def _atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.stem, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, ensure_ascii=False, indent=1, sort_keys=True)
        try:
            os.replace(tmp, path)
        except OSError:
            # On Windows, replacing a file another thread has open fails; if the target already
            # exists (an identical writer won the race) treat that as success.
            if path.exists():
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                return
            raise
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# LLM wrapper
# ---------------------------------------------------------------------------
class LLM:
    """Cached, logged, cost-gated, retrying, concurrency-capped model client."""

    def __init__(
        self,
        cfg: dict[str, Any] | None = None,
        *,
        config_path: str | Path | None = None,
        allow_over_limit: bool = False,
        run_id: str | None = None,
        cache_dir: str | Path | None = None,
        log_file: str | Path | None = None,
        max_concurrency: int | None = None,
        cost_limit_usd_per_run: float | None = None,
        provider_factory: Callable[[ModelSpec], Any] | None = None,
    ) -> None:
        self.cfg = cfg if cfg is not None else load_config(config_path)
        lcfg: dict[str, Any] = dict(self.cfg.get("llm") or {})
        self.lcfg = lcfg
        self.run_id = run_id or f"{utc_now_iso().replace(':', '').replace('+0000', 'Z')}-{uuid.uuid4().hex[:6]}"
        self.cache_dir = self._abs(cache_dir or lcfg.get("cache_dir", ".llm_cache"))
        self.log_file = self._abs(log_file or lcfg.get("log_file", "results/llm_calls.jsonl"))
        mc = max_concurrency if max_concurrency is not None else lcfg.get("max_concurrency", 8)
        try:
            self.max_concurrency = int(mc)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"llm.max_concurrency must be an integer >= 1, got {mc!r}") from exc
        if self.max_concurrency < 1:
            raise ConfigError(f"llm.max_concurrency must be >= 1, got {self.max_concurrency}")
        limit = cost_limit_usd_per_run if cost_limit_usd_per_run is not None else lcfg.get("cost_limit_usd_per_run")
        if limit is None:
            raise ConfigError("llm.cost_limit_usd_per_run must be set in config.yaml (the cost gate is mandatory)")
        self.cost_limit_usd = float(limit)
        self.allow_over_limit = bool(allow_over_limit or env("LLM_ALLOW_OVER_LIMIT") == "1")
        self.unknown_price_usd = float(lcfg.get("unknown_price_assumed_usd_per_call", DEFAULT_UNKNOWN_PRICE_USD_PER_CALL))
        self.max_attempts = int(lcfg.get("max_attempts", 4))
        self.retry_wait_initial_s = float(lcfg.get("retry_wait_initial_s", 2.0))
        self.retry_wait_max_s = float(lcfg.get("retry_wait_max_s", 60.0))
        self.fallbacks = str(lcfg.get("fallbacks") or "none").strip().lower()
        if self.fallbacks not in ("none", "default"):
            raise ConfigError(f"llm.fallbacks must be 'none' or 'default', got {lcfg.get('fallbacks')!r}")
        self.prices: dict[str, tuple[float, float]] = dict(PRICES_USD_PER_MTOK)
        for k, v in (lcfg.get("prices") or {}).items():
            try:
                self.prices[str(k)] = (float(v[0]), float(v[1]))
            except (TypeError, ValueError, IndexError, KeyError) as exc:
                raise ConfigError(f"llm.prices[{k!r}] must be [input_usd_per_mtok, output_usd_per_mtok], got {v!r}") from exc
        bcfg: dict[str, Any] = dict(lcfg.get("batch") or {})
        self.batch_poll_interval_s = float(bcfg.get("poll_interval_s", 30.0))
        self.batch_max_wait_s = float(bcfg.get("max_wait_h", 26.0)) * 3600.0
        self.batch_chunk_size = int(bcfg.get("chunk_size", 10000))
        self._provider_factory = provider_factory
        self._fake_all = env("LLM_FAKE") == "1"
        self._providers: dict[tuple[str, str], Any] = {}
        self._lock = threading.RLock()
        self._batch_lock = threading.RLock()          # manifest lookup + submission is one atomic step
        self._pending_batch_keys: set[str] = set()    # request hashes currently waited on in a batch
        self._inflight: dict[str, threading.Lock] = {}
        self._warned_price: set[str] = set()
        self.run_cost_usd = 0.0        # actual USD spent this process on uncached, priced calls
        self.run_assumed_usd = 0.0     # assumed USD for uncached calls whose model price is unknown
        self._reserved_usd = 0.0       # projections of in-flight calls (released on settle)
        self._stats: dict[str, Any] = {
            "calls": 0, "cached": 0, "errors": 0, "refused": 0, "input_tokens": 0, "output_tokens": 0, "per_model": {},
        }

    # -- config helpers -------------------------------------------------------
    @staticmethod
    def _abs(p: str | Path) -> Path:
        p = Path(p)
        return p if p.is_absolute() else ROOT / p

    @staticmethod
    def infer_provider(model_id: str) -> str:
        m = model_id.lower()
        if m.startswith("claude"):
            return "anthropic"
        if "/" in m:            # OpenRouter ids are namespaced: openai/gpt-5.6-luna
            return "openrouter"
        if m.startswith("deepseek"):
            return "deepseek"
        if m.startswith("gpt-") or re.match(r"^o[0-9]", m):
            return "openai"
        if m == "fake" or m.startswith("fake-"):
            return "fake"
        raise ConfigError(f"cannot infer provider for model id {model_id!r}; use a config name or set provider")

    def resolve_model(self, model: str | ModelSpec) -> ModelSpec:
        """Resolve a config role ('coder'/'judge'), a generation entry name, or a raw id."""
        if isinstance(model, ModelSpec):
            spec = model
        else:
            models = self.cfg.get("models") or {}
            entry: dict[str, Any] | None = None
            if model in models and isinstance(models[model], dict):
                entry = dict(models[model])
                entry.setdefault("name", model)
            else:
                for gen in models.get("generation") or []:
                    if isinstance(gen, dict) and gen.get("name") == model:
                        entry = dict(gen)
                        break
            if entry is None:
                spec = ModelSpec(name=model, provider=self.infer_provider(model), id=model)
            else:
                mid = str(entry.get("id", "SET_ME"))
                provider = entry.get("provider") or (self.infer_provider(mid) if mid != "SET_ME" else "anthropic")
                spec = ModelSpec(name=str(entry.get("name", model)), provider=str(provider), id=mid)
        if spec.id == "SET_ME" or not spec.id:
            raise ConfigError(f"model '{spec.name}' id is SET_ME — set models.{spec.name}.id in config.yaml")
        return spec

    def _provider(self, spec: ModelSpec) -> Any:
        key = (spec.provider, spec.id)
        with self._lock:
            if key not in self._providers:
                if self._provider_factory is not None:
                    prov = self._provider_factory(spec)
                elif self._fake_all or spec.provider == "fake":
                    prov = FakeProvider()
                elif spec.provider == "anthropic":
                    prov = AnthropicProvider(
                        max_retries=int(self.lcfg.get("sdk_max_retries", 4)),
                        timeout_s=float(self.lcfg.get("timeout_s", 600)),
                        fallbacks=self.fallbacks,
                    )
                elif spec.provider == "openrouter":
                    prov = OpenRouterProvider(
                        max_retries=int(self.lcfg.get("sdk_max_retries", 4)),
                        timeout_s=float(self.lcfg.get("timeout_s", 600)),
                    )
                elif spec.provider == "deepseek":
                    prov = DeepSeekProvider(
                        max_retries=int(self.lcfg.get("sdk_max_retries", 4)),
                        timeout_s=float(self.lcfg.get("timeout_s", 600)),
                    )
                elif spec.provider == "openai":
                    prov = OpenAIProvider(
                        max_retries=int(self.lcfg.get("sdk_max_retries", 4)), timeout_s=float(self.lcfg.get("timeout_s", 600))
                    )
                else:
                    raise ConfigError(f"unknown provider {spec.provider!r} for model {spec.name}")
                self._providers[key] = prov
            return self._providers[key]

    # -- prices / cost --------------------------------------------------------
    def price_for(self, model_id: str) -> tuple[float, float] | None:
        """(input, output) USD/MTok via longest-prefix match; None if unknown (warns once)."""
        best: str | None = None
        for k in self.prices:
            if model_id.startswith(k) and (best is None or len(k) > len(best)):
                best = k
        if best is None:
            with self._lock:
                first = model_id not in self._warned_price
                self._warned_price.add(model_id)
            if first:
                log.warning("no price known for model %s; cost recorded as None, gate assumes $%.2f/call",
                            model_id, self.unknown_price_usd)
            return None
        return self.prices[best]

    def cost_of(self, model_id: str, input_tokens: int, output_tokens: int,
                cache_read: int = 0, cache_write: int = 0, *,
                cache_read_mult: float | None = None, cache_write_mult: float | None = None) -> float | None:
        """Anthropic semantics: input_tokens EXCLUDES cached tokens (uncached remainder).

        Cache multipliers default to Anthropic's (0.10x read, 1.25x write); providers may
        override via `provider.cache_read_mult` / `provider.cache_write_mult`."""
        p = self.price_for(model_id)
        if p is None:
            return None
        pin, pout = p
        rm = CACHE_READ_MULT if cache_read_mult is None else cache_read_mult
        wm = CACHE_WRITE_MULT if cache_write_mult is None else cache_write_mult
        return (input_tokens * pin + cache_read * rm * pin + cache_write * wm * pin + output_tokens * pout) / 1e6

    def committed_usd(self) -> float:
        """Spent + assumed (unknown-price) + reserved in-flight USD — what the gate compares."""
        with self._lock:
            return self.run_cost_usd + self.run_assumed_usd + self._reserved_usd

    def _projected_cost(self, model_id: str, input_tokens: int, output_tokens: int) -> float:
        c = self.cost_of(model_id, input_tokens, output_tokens)
        return self.unknown_price_usd if c is None else c

    # -- cache ----------------------------------------------------------------
    @staticmethod
    def request_hash(req: dict[str, Any]) -> str:
        keys = ("provider", "model_id", "system", "user", "max_tokens", "json_schema", "effort",
                "temperature", "thinking", "fallbacks")
        return sha256_of({k: req.get(k) for k in keys})

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / key[:2] / f"{key}.json"

    def _cache_read(self, key: str) -> dict[str, Any] | None:
        p = self._cache_path(key)
        if not p.exists():
            return None
        try:
            obj = read_json(p)
        except (OSError, ValueError) as exc:
            log.warning("unreadable cache entry %s (%s); ignoring", p.name, exc)
            return None
        if not isinstance(obj, dict) or not isinstance(obj.get("response"), dict):
            log.warning("malformed cache entry %s; ignoring", p.name)
            return None
        return obj

    def _cache_write(self, key: str, req: dict[str, Any], resp: dict[str, Any], cost: float | None) -> None:
        """Best-effort: a cache-write failure must never discard a paid-for response."""
        try:
            _atomic_write_json(self._cache_path(key), {
                "request": req, "response": resp, "cost_usd": cost, "created": utc_now_iso(),
                "served_model": resp.get("served_model"),
            })
        except OSError as exc:
            log.warning("cache write failed for %s (%s); continuing without caching", key[:12], exc)

    def _inflight_lock(self, key: str) -> threading.Lock:
        with self._lock:
            lk = self._inflight.get(key)
            if lk is None:
                lk = self._inflight[key] = threading.Lock()
            return lk

    # -- request normalisation ---------------------------------------------
    def _build_req(self, spec: ModelSpec, system: str | None, user: str, max_tokens: int, json_schema: dict | None,
                   effort: str | None, temperature: float | None, thinking: str | None,
                   cache_system_prompt: bool, provider_name: str | None = None,
                   fallbacks: str | None = None) -> dict[str, Any]:
        """Normalise a request so equivalent calls hash identically.

        `provider` is the provider that will actually serve the call (e.g. 'fake' under
        LLM_FAKE=1), so fake responses can never poison the real cache. Parameters a model
        rejects (temperature on Opus 5 / Sonnet 5 / ...) are nulled before hashing."""
        pname = provider_name or spec.provider
        if temperature is not None and (
            (pname == "anthropic" and model_rejects_temperature(spec.id))
            or (pname == "openai" and openai_model_rejects_temperature(spec.id))
        ):
            temperature = None
        return {
            "provider": pname, "model_id": spec.id, "system": (system or None), "user": user,
            "max_tokens": int(max_tokens), "json_schema": (json_schema or None), "effort": (effort or None),
            "temperature": temperature, "thinking": (thinking or None),
            "cache_system_prompt": bool(cache_system_prompt),
            "fallbacks": self.fallbacks if fallbacks is None else fallbacks,
        }

    # -- the call -------------------------------------------------------------
    def complete(
        self,
        model: str | ModelSpec,
        system: str | None,
        user: str,
        *,
        max_tokens: int = 4096,
        json_schema: dict[str, Any] | None = None,
        parse_json: bool | None = None,
        effort: str | None = None,
        temperature: float | None = None,
        thinking: str | None = None,
        cache: bool = True,
        cache_system_prompt: bool = False,
        tag: str = "",
        meta: dict[str, Any] | None = None,
        max_tokens_hard: bool = False,
    ) -> LLMResponse:
        """One completion. Raises on non-retryable provider errors and cost-limit breaches;
        refusals and truncation are returned (not raised) unless `max_tokens_hard`."""
        t0 = time.perf_counter()
        spec = self.resolve_model(model)
        provider = self._provider(spec)
        provider_name = str(getattr(provider, "name", spec.provider))
        req = self._build_req(spec, system, user, max_tokens, json_schema, effort, temperature, thinking,
                              cache_system_prompt, provider_name)
        key = self.request_hash(req)
        meta = dict(meta or {})
        resp = LLMResponse(model_name=spec.name, model_id=spec.id, provider=provider_name, request_hash=key,
                           tag=tag, meta=meta)
        do_parse = json_schema is not None if parse_json is None else bool(parse_json)
        provider_dict: dict[str, Any] | None = None
        projected = 0.0
        reserved = False
        key_lock: threading.Lock | None = None
        try:
            hit = self._cache_read(key) if cache else None
            if hit is None and key in self._pending_batch_keys:
                log.warning("request %s is already pending in a batch; the sync call will pay for it again "
                            "(tag=%s)", key[:12], tag)
            if hit is None:
                # Serialise identical concurrent requests: the first pays, the rest read its cache entry.
                key_lock = self._inflight_lock(key)
                key_lock.acquire()
                hit = self._cache_read(key) if cache else None
            if hit is not None:
                provider_dict = hit["response"]
                resp.cached = True
                resp.mode = "cache"
                resp.cost_usd = hit.get("cost_usd")
            else:
                projected = self._projected_cost(spec.id, estimate_tokens(req["system"]) + estimate_tokens(user),
                                                 max_tokens)
                self._reserve(projected)
                reserved = True
                provider_dict = self._call_with_retry(provider, req, resp)
                resp.cost_usd = self.cost_of(
                    spec.id, provider_dict["input_tokens"], provider_dict["output_tokens"],
                    provider_dict["cache_read_tokens"], provider_dict["cache_write_tokens"],
                    cache_read_mult=getattr(provider, "cache_read_mult", None),
                    cache_write_mult=getattr(provider, "cache_write_mult", None),
                )
                self._settle(projected, resp.cost_usd, succeeded=True)
                reserved = False
                if not (provider_dict.get("stop_reason") == "refusal"):
                    self._cache_write(key, req, provider_dict, resp.cost_usd)
            self._fill(resp, provider_dict)
            if resp.stop_reason in ("max_tokens", "model_context_window_exceeded"):
                log.warning("output truncated (%s) at max_tokens=%s for %s (tag=%s)",
                            resp.stop_reason, max_tokens, spec.id, tag)
                if max_tokens_hard:
                    raise OutputTruncated(f"{spec.id} hit {resp.stop_reason} (max_tokens={max_tokens}, tag={tag!r})")
            if resp.refused:
                log.warning("model %s refused (category=%s, tag=%s)", spec.id, resp.refusal_category, tag)
            if do_parse and not resp.refused:
                resp.json, resp.json_error = parse_json_text(resp.text)
        except Exception as exc:
            if reserved:
                self._settle(projected, None, succeeded=False)
            resp.error = f"{type(exc).__name__}: {exc}"
            resp.latency_s = time.perf_counter() - t0
            self._record(resp, req, provider_dict)
            raise
        finally:
            if key_lock is not None:
                key_lock.release()
        resp.latency_s = time.perf_counter() - t0
        self._record(resp, req, provider_dict)
        return resp

    def _fill(self, resp: LLMResponse, d: dict[str, Any]) -> None:
        resp.text = d.get("text", "") or ""
        resp.stop_reason = d.get("stop_reason")
        resp.served_model = d.get("served_model")
        resp.refused = d.get("stop_reason") == "refusal"
        resp.refusal_category = d.get("refusal_category")
        resp.input_tokens = int(d.get("input_tokens", 0) or 0)
        resp.output_tokens = int(d.get("output_tokens", 0) or 0)
        resp.cache_read_tokens = int(d.get("cache_read_tokens", 0) or 0)
        resp.cache_write_tokens = int(d.get("cache_write_tokens", 0) or 0)
        resp.raw = {k: v for k, v in d.items() if k != "text"}

    def _call_with_retry(self, provider: Any, req: dict[str, Any], resp: LLMResponse) -> dict[str, Any]:
        """Call the provider with tenacity retries on transient errors; `resp.attempts` is
        updated on every attempt so failures are logged with the true attempt count."""
        transient = (TransientProviderError, *getattr(provider, "transient_exceptions", ()))
        retrying = Retrying(
            retry=retry_if_exception_type(transient),
            wait=wait_exponential_jitter(initial=self.retry_wait_initial_s, max=self.retry_wait_max_s),
            stop=stop_after_attempt(self.max_attempts),
            reraise=True,
        )
        try:
            for attempt in retrying:
                with attempt:
                    resp.attempts = attempt.retry_state.attempt_number
                    result = provider.call(req)
                if attempt.retry_state.outcome is not None and not attempt.retry_state.outcome.failed:
                    return result
        except transient as exc:
            raise TransientProviderError(f"{type(exc).__name__} after {resp.attempts} attempt(s): {exc}") from exc
        raise RuntimeError("unreachable")  # pragma: no cover

    # -- cost meter (thread-safe, fail-closed) --------------------------------
    def _reserve(self, projected: float) -> None:
        with self._lock:
            total = self.run_cost_usd + self.run_assumed_usd + self._reserved_usd + projected
            if total > self.cost_limit_usd and not self.allow_over_limit:
                raise CostLimitExceeded(
                    f"cost gate: spent ${self.run_cost_usd:.2f} + assumed ${self.run_assumed_usd:.2f} + in-flight "
                    f"${self._reserved_usd:.2f} + projected ${projected:.2f} > limit ${self.cost_limit_usd:.2f}; "
                    f"re-run with a smaller sample or raise llm.cost_limit_usd_per_run after user approval "
                    f"(or LLM_ALLOW_OVER_LIMIT=1)"
                )
            self._reserved_usd += projected

    def _settle(self, projected: float, actual: float | None, *, succeeded: bool) -> None:
        """Release the reservation; book actual cost (or the assumed cost for unknown-price models)."""
        with self._lock:
            self._reserved_usd = max(self._reserved_usd - projected, 0.0)
            if actual is not None:
                self.run_cost_usd += actual
            elif succeeded:
                self.run_assumed_usd += self.unknown_price_usd

    def _record(self, resp: LLMResponse, req: dict[str, Any], d: dict[str, Any] | None) -> None:
        """Update in-memory stats and append the JSONL log row (no text, ever).

        `meta` must hold only ids/labels (never prompt, response or social-media text); it is
        coerced to plain JSON here (non-serialisable values become strings) and a logging
        failure is reported but never raised — a paid-for response is never discarded."""
        try:
            meta_plain = json.loads(stable_json(resp.meta))
        except (TypeError, ValueError):
            meta_plain = {"_unserialisable_meta": True}
        row = {
            "ts": utc_now_iso(), "run_id": self.run_id, "tag": resp.tag, "meta": meta_plain,
            "provider": resp.provider, "model_name": resp.model_name, "model_id": resp.model_id,
            "served_model": resp.served_model, "request_hash": resp.request_hash, "cached": resp.cached,
            "attempts": resp.attempts, "latency_s": round(resp.latency_s, 4),
            "input_tokens": resp.input_tokens, "output_tokens": resp.output_tokens,
            "cache_read_tokens": resp.cache_read_tokens, "cache_write_tokens": resp.cache_write_tokens,
            "cost_usd": resp.cost_usd, "stop_reason": resp.stop_reason, "refused": resp.refused,
            "refusal_category": resp.refusal_category, "max_tokens": req["max_tokens"],
            "json_schema": req.get("json_schema") is not None, "effort": req.get("effort"),
            "temperature": req.get("temperature"),
            "temperature_dropped": bool((d or {}).get("temperature_dropped", False)),
            "system_chars": len(req.get("system") or ""), "user_chars": len(req.get("user") or ""),
            "output_chars": len(resp.text or ""),
            "error": (resp.error[:300] if resp.error else None),
            "mode": ("cache" if resp.cached else resp.mode), "batch_id": resp.batch_id,
        }
        with self._lock:
            row["run_cost_usd_after"] = round(self.run_cost_usd, 6)
            row["run_assumed_usd_after"] = round(self.run_assumed_usd, 6)
            st = self._stats
            st["calls"] += 1
            st["cached"] += int(resp.cached)
            st["errors"] += int(resp.error is not None)
            st["refused"] += int(resp.refused)
            st["input_tokens"] += resp.input_tokens
            st["output_tokens"] += resp.output_tokens
            pm = st["per_model"].setdefault(resp.model_name, {"calls": 0, "cached": 0, "input_tokens": 0,
                                                             "output_tokens": 0, "cost_usd": 0.0})
            pm["calls"] += 1
            pm["cached"] += int(resp.cached)
            pm["input_tokens"] += resp.input_tokens
            pm["output_tokens"] += resp.output_tokens
            if not resp.cached and resp.cost_usd:
                pm["cost_usd"] += resp.cost_usd
            try:
                append_jsonl(self.log_file, row)
            except OSError as exc:  # never lose a response because the log could not be written
                log.error("could not append to %s: %s", self.log_file, exc)

    # -- batch ------------------------------------------------------------------
    def complete_many(self, requests: list[dict[str, Any]], *, progress: bool = True,
                      stop_on_cost_limit: bool = True) -> list[LLMResponse]:
        """Run many `complete()` calls concurrently (<= max_concurrency); results in input order.

        Per-item exceptions become `LLMResponse(error=...)`; CostLimitExceeded cancels
        pending work and is re-raised when `stop_on_cost_limit`."""
        results: list[LLMResponse | None] = [None] * len(requests)
        if not requests:
            return []
        bar = None
        if progress:
            try:
                from tqdm import tqdm
                bar = tqdm(total=len(requests), desc="llm", unit="call", leave=False)
            except ImportError:  # pragma: no cover
                bar = None
        limit_exc: CostLimitExceeded | None = None
        pool = ThreadPoolExecutor(max_workers=self.max_concurrency)
        futures: dict[Future, int] = {pool.submit(self._complete_safe, r): i for i, r in enumerate(requests)}
        try:
            pending = set(futures)
            while pending:
                done, pending = wait(pending, return_when=FIRST_EXCEPTION)
                for fut in done:
                    i = futures[fut]
                    exc = fut.exception()
                    if isinstance(exc, CostLimitExceeded):
                        results[i] = self._error_response(requests[i], exc)
                        if stop_on_cost_limit and limit_exc is None:
                            limit_exc = exc
                            for f in pending:
                                f.cancel()
                    elif exc is not None:  # pragma: no cover - _complete_safe captures everything else
                        results[i] = self._error_response(requests[i], exc)
                    else:
                        results[i] = fut.result()
                    if bar:
                        bar.update(1)
                if limit_exc is not None:
                    break
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
            if bar:
                bar.close()
        if limit_exc is not None:
            raise limit_exc
        for i, r in enumerate(results):
            if r is None:  # cancelled (only possible when stop_on_cost_limit hit)
                results[i] = self._error_response(requests[i], RuntimeError("cancelled"))
        return [r for r in results if r is not None]

    def _complete_safe(self, r: dict[str, Any]) -> LLMResponse:
        try:
            return self.complete(**r)
        except CostLimitExceeded:
            raise
        except Exception as exc:  # noqa: BLE001 - captured per item
            return self._error_response(r, exc)

    def _error_response(self, r: dict[str, Any], exc: BaseException) -> LLMResponse:
        try:
            spec = self.resolve_model(r.get("model", ""))
            name, mid, prov = spec.name, spec.id, spec.provider
        except Exception:  # noqa: BLE001
            name, mid, prov = str(r.get("model", "")), "", ""
        return LLMResponse(model_name=name, model_id=mid, provider=prov, tag=r.get("tag", ""),
                           meta=dict(r.get("meta") or {}), error=f"{type(exc).__name__}: {exc}")

    # -- Message Batches ------------------------------------------------------------
    _REQUEST_KEYS = frozenset({
        "model", "system", "user", "max_tokens", "json_schema", "parse_json", "effort", "temperature",
        "thinking", "cache", "cache_system_prompt", "tag", "meta", "max_tokens_hard",
    })
    BATCH_MAX_BYTES = 200 * 1024 * 1024  # stay under the API's 256 MB per-batch cap

    def _batches_dir(self) -> Path:
        return self.cache_dir / "batches"

    def _prepare(self, r: dict[str, Any], *, batch: bool = False) -> tuple[ModelSpec, Any, dict[str, Any], str]:
        """Validate + resolve one request dict → (spec, provider, normalised req, cache key).

        Unknown keys raise TypeError (same contract as `complete(**r)`). In batch mode the
        request is hashed with `fallbacks='none'` because the Batches API has no fallbacks."""
        unknown = set(r) - self._REQUEST_KEYS
        if unknown:
            raise TypeError(f"unknown request key(s) {sorted(unknown)}; allowed: {sorted(self._REQUEST_KEYS)}")
        if "model" not in r or "user" not in r:
            raise TypeError("request needs at least 'model' and 'user'")
        spec = self.resolve_model(r["model"])
        provider = self._provider(spec)
        pname = str(getattr(provider, "name", spec.provider))
        req = self._build_req(spec, r.get("system"), r["user"], int(r.get("max_tokens", 4096)),
                              r.get("json_schema"), r.get("effort"), r.get("temperature"), r.get("thinking"),
                              bool(r.get("cache_system_prompt", False)), pname,
                              fallbacks="none" if batch else None)
        return spec, provider, req, self.request_hash(req)

    def _response_from_dict(self, r: dict[str, Any], spec: ModelSpec, provider_name: str, req: dict[str, Any],
                            key: str, d: dict[str, Any], *, cached: bool, cost: float | None,
                            mode: str, batch_id: str | None, latency_s: float = 0.0,
                            record: bool = True) -> LLMResponse:
        """Build (+ record) an LLMResponse for request `r` from a normalised provider dict."""
        resp = LLMResponse(model_name=spec.name, model_id=spec.id, provider=provider_name, request_hash=key,
                           tag=r.get("tag", ""), meta=dict(r.get("meta") or {}), cached=cached, cost_usd=cost,
                           mode=mode, batch_id=batch_id, attempts=0 if cached else 1, latency_s=latency_s)
        self._fill(resp, d)
        parse = r.get("parse_json")
        do_parse = (r.get("json_schema") is not None) if parse is None else bool(parse)
        if resp.stop_reason in ("max_tokens", "model_context_window_exceeded"):
            log.warning("output truncated (%s) at max_tokens=%s for %s (tag=%s, mode=%s)", resp.stop_reason,
                        req["max_tokens"], spec.id, resp.tag, mode)
            if r.get("max_tokens_hard"):
                resp.error = f"OutputTruncated: {spec.id} hit {resp.stop_reason} (max_tokens={req['max_tokens']})"
        if resp.refused:
            log.warning("model %s refused (category=%s, tag=%s, mode=%s)", spec.id, resp.refusal_category,
                        resp.tag, mode)
        if do_parse and not resp.refused:
            resp.json, resp.json_error = parse_json_text(resp.text)
        if record:
            self._record(resp, req, d)
        return resp

    def _error_response_for(self, r: dict[str, Any], spec: ModelSpec, provider_name: str, req: dict[str, Any],
                            key: str, error: str, batch_id: str | None, *, mode: str = "batch",
                            record: bool = True) -> LLMResponse:
        resp = LLMResponse(model_name=spec.name, model_id=spec.id, provider=provider_name, request_hash=key,
                           tag=r.get("tag", ""), meta=dict(r.get("meta") or {}), error=error, mode=mode,
                           batch_id=batch_id)
        if record:
            self._record(resp, req, None)
        return resp

    # manifests: one JSON per submitted batch under <cache_dir>/batches/<batch_id>.json
    def _manifest_path(self, batch_id: str) -> Path:
        return self._batches_dir() / f"{batch_id}.json"

    def _save_manifest(self, m: dict[str, Any]) -> None:
        try:
            _atomic_write_json(self._manifest_path(m["batch_id"]), m)
        except OSError as exc:  # pragma: no cover
            log.warning("could not write batch manifest %s: %s", m.get("batch_id"), exc)

    def open_batch_manifests(self, provider_name: str | None = None) -> list[dict[str, Any]]:
        """Manifests of batches that were submitted but not fully collected (any provider or one)."""
        out: list[dict[str, Any]] = []
        d = self._batches_dir()
        if not d.exists():
            return out
        for p in sorted(d.glob("*.json")):
            try:
                m = read_json(p)
            except (OSError, ValueError):
                continue
            if not isinstance(m, dict) or m.get("status") != "submitted":
                continue
            if provider_name is not None and m.get("provider") != provider_name:
                continue
            out.append(m)
        return out

    def _wait_for_batches(self, provider: Any, manifests: dict[str, dict[str, Any]], *, poll_interval_s: float,
                          max_wait_s: float) -> dict[str, str]:
        """Poll until every batch has ended. Returns {batch_id: 'ended' | 'unavailable: <err>'}.

        Transient provider errors are retried; a batch that no longer exists (NotFound, archived,
        unknown to the provider) is reported as unavailable, not raised. TimeoutError after
        `max_wait_s` measured from each batch's creation."""
        transient = (TransientProviderError, *getattr(provider, "transient_exceptions", ()))
        retrying = Retrying(retry=retry_if_exception_type(transient),
                            wait=wait_exponential_jitter(initial=self.retry_wait_initial_s,
                                                         max=self.retry_wait_max_s),
                            stop=stop_after_attempt(self.max_attempts), reraise=True)
        outcome: dict[str, str] = {}
        pending = list(manifests)
        while pending:
            still: list[str] = []
            for bid in pending:
                try:
                    st = retrying(provider.batch_status, bid)
                except BatchUnavailableError as exc:
                    outcome[bid] = f"unavailable: {exc}"
                    log.error("batch %s is unavailable (%s); manifest %s marked failed", bid, exc,
                              self._manifest_path(bid))
                    continue
                if st.get("status") == "ended":
                    outcome[bid] = "ended"
                    log.info("batch %s ended: %s", bid, st.get("counts"))
                else:
                    still.append(bid)
                    log.info("batch %s %s: %s", bid, st.get("status"), st.get("counts"))
            pending = still
            if not pending:
                break
            oldest = min(_parse_iso(manifests[b].get("created")) for b in pending)
            if time.time() - oldest > max_wait_s:
                raise TimeoutError(f"batches still not ended after {max_wait_s / 3600:.1f} h since creation: "
                                   f"{pending} (manifests stay open; re-run to resume)")
            time.sleep(poll_interval_s)
        return outcome

    def _submit_batches(self, provider: Any, pname: str, items: list[tuple[str, dict[str, Any]]],
                        model_by_key: dict[str, str], proj_by_key: dict[str, float]) -> list[dict[str, Any]]:
        """Submit `items` in chunks (count + byte guards); one manifest per created batch,
        written immediately after each successful submission so a later failure orphans nothing."""
        manifests: list[dict[str, Any]] = []
        chunk: list[tuple[str, dict[str, Any]]] = []
        size = 0
        def flush() -> None:
            nonlocal chunk, size
            if not chunk:
                return
            bid = provider.submit_batch(chunk)
            m = {"batch_id": bid, "provider": pname, "custom_ids": [cid for cid, _ in chunk],
                 "models": {cid: model_by_key[cid] for cid, _ in chunk},
                 "projected_usd": round(sum(proj_by_key.get(cid, 0.0) for cid, _ in chunk), 6),
                 "created": utc_now_iso(), "run_id": self.run_id, "status": "submitted",
                 "errored": {}, "actual_usd": 0.0}
            self._save_manifest(m)
            manifests.append(m)
            log.info("submitted batch %s: %d requests (projected $%.2f at batch price)", bid, len(chunk),
                     m["projected_usd"])
            chunk, size = [], 0
        for cid, req in items:
            n = len(req.get("system") or "") + len(req.get("user") or "") + len(stable_json(req.get("json_schema"))) + 256
            if chunk and (len(chunk) >= self.batch_chunk_size or size + n > self.BATCH_MAX_BYTES):
                flush()
            chunk.append((cid, req))
            size += n
        flush()
        return manifests

    def complete_batch(self, requests: list[dict[str, Any]], *, poll_interval_s: float | None = None,
                       max_wait_s: float | None = None, wait: bool = True) -> list[LLMResponse]:
        """Run `requests` through the provider's Message Batches API (50% price, async).

        Same request dicts as `complete_many`; results in input order. Cache hits are served
        from disk; identical requests are submitted once (custom_id = request hash); providers
        without batch support fall back to `complete_many`. The projected cost of everything that
        will be waited on is reserved through the fail-closed gate before submission and released
        item by item as results are booked. Each submitted batch gets a manifest under
        `<cache_dir>/batches/`; a request whose hash is pending in an open manifest is re-attached
        instead of resubmitted (crash / Ctrl-C / `wait=False` safe). With `wait=False` the call
        returns right after submission (responses `mode='pending'`, not logged); call again later.
        """
        if not requests:
            return []
        t0 = time.monotonic()
        poll_interval_s = self.batch_poll_interval_s if poll_interval_s is None else float(poll_interval_s)
        max_wait_s = self.batch_max_wait_s if max_wait_s is None else float(max_wait_s)
        if self.fallbacks == "default":
            log.warning("llm.fallbacks=default is ignored in batch mode (Batches API has no fallbacks); "
                        "batch requests are hashed with fallbacks='none'")
        results: list[LLMResponse | None] = [None] * len(requests)
        prepared = [self._prepare(r, batch=True) for r in requests]

        # 1) cache hits + grouping of the rest by provider and key
        by_provider: dict[str, dict[str, list[int]]] = {}
        provider_objs: dict[str, Any] = {}
        for i, (spec, provider, req, key) in enumerate(prepared):
            pname = str(getattr(provider, "name", spec.provider))
            hit = self._cache_read(key) if requests[i].get("cache", True) else None
            if hit is not None:
                results[i] = self._response_from_dict(requests[i], spec, pname, req, key, hit["response"],
                                                      cached=True, cost=hit.get("cost_usd"), mode="cache",
                                                      batch_id=None)
                continue
            provider_objs[pname] = provider
            by_provider.setdefault(pname, {}).setdefault(key, []).append(i)

        # 2) per provider
        for pname, keys in by_provider.items():
            provider = provider_objs[pname]
            idx_all = [i for lst in keys.values() for i in lst]
            if not getattr(provider, "supports_batch", False):
                log.warning("provider %s has no batch support; running %d requests synchronously", pname,
                            len(idx_all))
                for i, resp in zip(idx_all, self.complete_many([requests[i] for i in idx_all], progress=False,
                                                                stop_on_cost_limit=True)):
                    results[i] = resp
                continue
            discount = float(getattr(provider, "batch_discount", 1.0))
            first_idx = {key: lst[0] for key, lst in keys.items()}
            model_by_key = {key: prepared[i][0].id for key, i in first_idx.items()}
            proj_by_key: dict[str, float] = {}
            for key, i in first_idx.items():
                spec, _, req, _ = prepared[i]
                proj_by_key[key] = discount * self._projected_cost(
                    spec.id, estimate_tokens(req["system"]) + estimate_tokens(req["user"]), req["max_tokens"])

            with self._batch_lock:  # manifest lookup + submission are one atomic step per process
                manifests: dict[str, dict[str, Any]] = {}
                pending_bid: dict[str, str] = {}
                for m in self.open_batch_manifests(pname):
                    for cid in m.get("custom_ids", []):
                        if cid in keys and cid not in pending_bid:
                            pending_bid[cid] = m["batch_id"]
                            manifests[m["batch_id"]] = m
                to_submit = [key for key in keys if key not in pending_bid]
                if pending_bid:
                    log.info("re-attaching %d request(s) to open batch(es) %s", len(pending_bid),
                             sorted(set(pending_bid.values())))
                reserved = sum(proj_by_key[k] for k in keys)  # everything we will wait on
                self._reserve(reserved)
                booked: set[str] = set()
                try:
                    if to_submit:
                        items = [(key, prepared[first_idx[key]][2]) for key in to_submit]
                        for m in self._submit_batches(provider, pname, items, model_by_key, proj_by_key):
                            manifests[m["batch_id"]] = m
                            for cid in m["custom_ids"]:
                                pending_bid[cid] = m["batch_id"]
                    with self._lock:
                        self._pending_batch_keys.update(keys)
                except BaseException:
                    self._release(reserved)  # submitted chunks have manifests and are resumable
                    raise
                if not wait:
                    self._release(reserved)
                    log.info("batch(es) %s submitted; not waiting (%d request(s) pending, ~$%.2f committed at "
                             "batch price)", sorted(manifests), len(keys), reserved)
                    for i in idx_all:
                        spec, _, req, key = prepared[i]
                        results[i] = self._error_response_for(
                            requests[i], spec, pname, req, key, f"batch pending {pending_bid.get(key)}",
                            pending_bid.get(key), mode="pending", record=False)
                    continue

            # 3) poll, then collect (reservation released item by item; leftovers in `finally`)
            try:
                try:
                    outcome = self._wait_for_batches(provider, manifests, poll_interval_s=poll_interval_s,
                                                     max_wait_s=max_wait_s)
                except TimeoutError:
                    raise
                for bid, m in manifests.items():
                    wanted = {cid for cid in m["custom_ids"] if cid in keys}
                    if outcome.get(bid) != "ended":
                        m.update({"status": "failed", "error": outcome.get(bid, "unknown"),
                                  "finished": utc_now_iso()})
                        self._save_manifest(m)
                        for cid in wanted:
                            spec, _, req, key = prepared[first_idx[cid]]
                            for i in keys[cid]:
                                results[i] = self._error_response_for(
                                    requests[i], spec, pname, req, key,
                                    f"batch {bid} {outcome.get(bid)}; re-run to resubmit", bid)
                            with self._lock:
                                self._release(proj_by_key[cid])
                            booked.add(cid)
                        continue
                    seen: set[str] = set()
                    try:
                        results_iter = provider.batch_results(bid)
                    except BatchUnavailableError as exc:
                        m.update({"status": "failed", "error": f"results unavailable: {exc}",
                                  "finished": utc_now_iso()})
                        self._save_manifest(m)
                        for cid in wanted:
                            spec, _, req, key = prepared[first_idx[cid]]
                            for i in keys[cid]:
                                results[i] = self._error_response_for(
                                    requests[i], spec, pname, req, key,
                                    f"batch {bid} results unavailable ({exc}); re-run to resubmit", bid)
                            self._release(proj_by_key[cid])
                            booked.add(cid)
                        continue
                    for cid, d, err in results_iter:
                        seen.add(cid)
                        model_id = m.get("models", {}).get(cid) or model_by_key.get(cid)
                        if d is not None and cid not in keys and model_id:
                            # belongs to another (earlier) request set: still cache it so no one pays twice
                            c = self.cost_of(model_id, d["input_tokens"], d["output_tokens"], d["cache_read_tokens"],
                                             d["cache_write_tokens"],
                                             cache_read_mult=getattr(provider, "cache_read_mult", None),
                                             cache_write_mult=getattr(provider, "cache_write_mult", None))
                            if d.get("stop_reason") != "refusal":
                                self._cache_write(cid, {"provider": pname, "model_id": model_id, "custom_id": cid,
                                                        "note": "collected from batch"}, d,
                                                  None if c is None else c * discount)
                            continue
                        if cid not in keys or cid in booked:
                            continue
                        idxs = keys[cid]
                        spec, _, req, key = prepared[idxs[0]]
                        if d is None:
                            m["errored"][cid] = str(err)
                            with self._lock:
                                self._release(proj_by_key[cid])
                            booked.add(cid)
                            for i in idxs:
                                results[i] = self._error_response_for(requests[i], spec, pname, req, key,
                                                                      f"batch {err}", bid)
                            continue
                        cost = self.cost_of(spec.id, d["input_tokens"], d["output_tokens"], d["cache_read_tokens"],
                                            d["cache_write_tokens"],
                                            cache_read_mult=getattr(provider, "cache_read_mult", None),
                                            cache_write_mult=getattr(provider, "cache_write_mult", None))
                        if cost is not None:
                            cost *= discount
                        with self._lock:  # book this item, release its share of the reservation
                            self._reserved_usd = max(self._reserved_usd - proj_by_key[cid], 0.0)
                            if cost is not None:
                                self.run_cost_usd += cost
                                m["actual_usd"] = round(m.get("actual_usd", 0.0) + cost, 6)
                            else:
                                self.run_assumed_usd += self.unknown_price_usd * discount
                        booked.add(cid)
                        if d.get("stop_reason") != "refusal":
                            self._cache_write(key, req, d, cost)
                        for k, i in enumerate(idxs):  # duplicates: first carries the cost, the rest are 'cache'
                            results[i] = self._response_from_dict(
                                requests[i], spec, pname, req, key, d, cached=(k > 0), cost=cost,
                                mode="batch" if k == 0 else "cache", batch_id=bid,
                                latency_s=time.monotonic() - t0)
                    missing = [cid for cid in wanted if cid not in seen]
                    for cid in missing:
                        spec, _, req, key = prepared[first_idx[cid]]
                        m["errored"][cid] = "missing from batch results"
                        with self._lock:
                            self._release(proj_by_key[cid])
                        booked.add(cid)
                        for i in keys[cid]:
                            results[i] = self._error_response_for(requests[i], spec, pname, req, key,
                                                                  "batch result missing for request", bid)
                    # a manifest is done once every custom_id is cached or errored
                    remaining = [cid for cid in m["custom_ids"]
                                 if cid not in m["errored"] and not self._cache_path(cid).exists()]
                    m.update({"status": "submitted" if remaining else "done", "finished": utc_now_iso(),
                              "n_remaining": len(remaining)})
                    self._save_manifest(m)
                    log.info("batch %s collected: %d ok, %d errored, %d missing, $%.4f actual%s", bid,
                             len(seen) - len([c for c in seen if c in m["errored"]]), len(m["errored"]),
                             len(missing), m.get("actual_usd", 0.0), "" if not remaining else
                             f" ({len(remaining)} item(s) still pending)")
            finally:
                leftover = sum(proj_by_key[k] for k in keys if k not in booked)
                if leftover:
                    self._release(leftover)
                with self._lock:
                    self._pending_batch_keys.difference_update(keys)

        return [r if r is not None else self._error_response(requests[i], RuntimeError("no result"))
                for i, r in enumerate(results)]

    def _release(self, amount: float) -> None:
        with self._lock:
            self._reserved_usd = max(self._reserved_usd - amount, 0.0)

    def run(self, requests: list[dict[str, Any]], *, batch: bool, progress: bool = True,
            stop_on_cost_limit: bool = True, poll_interval_s: float | None = None,
            max_wait_s: float | None = None, wait: bool = True) -> list[LLMResponse]:
        """Dispatch: `complete_batch` (full runs, 50% price) or `complete_many` (smoke / interactive)."""
        if batch:
            return self.complete_batch(requests, poll_interval_s=poll_interval_s, max_wait_s=max_wait_s, wait=wait)
        return self.complete_many(requests, progress=progress, stop_on_cost_limit=stop_on_cost_limit)

    # -- estimation & gate ---------------------------------------------------------
    def estimate_cost(self, requests: list[dict[str, Any]], *, expected_output_tokens: int | None = None,
                      use_api: bool | None = None, batch: bool = False) -> CostEstimate:
        """Project the cost of `requests` BEFORE running them (cached ones count as $0).
        `batch=True` applies the provider's batch discount (Anthropic Message Batches: 50%)."""
        n_cached = in_tok = out_tok = 0
        usd = 0.0
        per_model: dict[str, float] = {}
        methods: set[str] = set()
        for r in requests:
            spec = self.resolve_model(r["model"])
            system, user = r.get("system"), r["user"]
            max_tokens = int(r.get("max_tokens", 4096))
            pname = str(getattr(self._provider(spec), "name", spec.provider))
            req = self._build_req(spec, system, user, max_tokens, r.get("json_schema"), r.get("effort"),
                                  r.get("temperature"), r.get("thinking"), bool(r.get("cache_system_prompt", False)),
                                  pname)
            if r.get("cache", True) and self._cache_read(self.request_hash(req)) is not None:
                n_cached += 1
                continue
            n_in: int | None = None
            if use_api is not False:
                prov = self._provider(spec)
                if getattr(prov, "has_credentials", lambda: False)():
                    n_in = prov.count_tokens(system, user, spec.id)
            if n_in is None:
                n_in = estimate_tokens(system) + estimate_tokens(user)
                methods.add("heuristic")
            else:
                methods.add("api")
            n_out = int(expected_output_tokens) if expected_output_tokens is not None else max_tokens
            c = self._projected_cost(spec.id, n_in, n_out)
            if batch:
                c *= float(getattr(self._provider(spec), "batch_discount", 1.0)) if getattr(
                    self._provider(spec), "supports_batch", False) else 1.0
            in_tok += n_in
            out_tok += n_out
            usd += c
            per_model[spec.name] = per_model.get(spec.name, 0.0) + c
        method = "mixed" if len(methods) > 1 else (methods.pop() if methods else "heuristic")
        over = (self.committed_usd() + usd > self.cost_limit_usd) and not self.allow_over_limit
        return CostEstimate(n_requests=len(requests), n_cached=n_cached, input_tokens=in_tok, output_tokens=out_tok,
                            usd=round(usd, 6), usd_per_model={k: round(v, 6) for k, v in per_model.items()},
                            method=method, over_limit=over, limit_usd=self.cost_limit_usd)

    def gate(self, est: CostEstimate) -> CostEstimate:
        """Log a one-line summary; raise CostLimitExceeded if the projection breaks the limit."""
        committed = self.committed_usd()
        msg = (f"cost projection: {est.n_requests} requests ({est.n_cached} cached), ~{est.input_tokens} in / "
               f"{est.output_tokens} out tokens, ${est.usd:.2f} projected ({est.method}); committed so far "
               f"${committed:.2f}; limit ${est.limit_usd:.2f}")
        if committed + est.usd > self.cost_limit_usd and not self.allow_over_limit:
            log.error("%s -> OVER LIMIT", msg)
            raise CostLimitExceeded(
                f"projected ${committed + est.usd:.2f} > limit ${self.cost_limit_usd:.2f}; re-run with a "
                f"smaller sample or raise llm.cost_limit_usd_per_run after user approval")
        log.info("%s -> OK", msg)
        return est

    # -- reporting -----------------------------------------------------------
    def summary(self) -> dict[str, Any]:
        """In-memory totals for this process (uncached cost only)."""
        with self._lock:
            st = self._stats
            return {"run_id": self.run_id, "calls": st["calls"], "cached": st["cached"], "errors": st["errors"],
                    "refused": st["refused"], "input_tokens": st["input_tokens"], "output_tokens": st["output_tokens"],
                    "cost_usd": round(self.run_cost_usd, 6), "assumed_cost_usd": round(self.run_assumed_usd, 6),
                    "limit_usd": self.cost_limit_usd,
                    "per_model": {k: dict(v, cost_usd=round(v["cost_usd"], 6)) for k, v in st["per_model"].items()}}

    @staticmethod
    def summarize_log(log_file: str | Path | None = None) -> dict[str, Any]:
        """Aggregate an existing JSONL log: totals, per model, per tag (uncached cost only)."""
        p = Path(log_file) if log_file else ROOT / "results" / "llm_calls.jsonl"

        def fresh() -> dict[str, Any]:
            return {"calls": 0, "cached": 0, "errors": 0, "refused": 0, "input_tokens": 0, "output_tokens": 0,
                    "cache_read_tokens": 0, "cost_usd": 0.0}

        total = fresh()
        per_model: dict[str, dict[str, Any]] = {}
        per_tag: dict[str, dict[str, Any]] = {}
        run_ids: set[str] = set()

        def bump(d: dict[str, Any], row: dict[str, Any]) -> None:
            d["calls"] += 1
            d["cached"] += int(bool(row.get("cached")))
            d["errors"] += int(row.get("error") is not None)
            d["refused"] += int(bool(row.get("refused")))
            d["input_tokens"] += int(row.get("input_tokens") or 0)
            d["output_tokens"] += int(row.get("output_tokens") or 0)
            d["cache_read_tokens"] += int(row.get("cache_read_tokens") or 0)
            if not row.get("cached") and row.get("cost_usd"):
                d["cost_usd"] += float(row["cost_usd"])

        for row in read_jsonl(p):
            if row.get("mode") == "pending":  # submission placeholders are not calls
                continue
            bump(total, row)
            run_ids.add(str(row.get("run_id")))
            bump(per_model.setdefault(str(row.get("model_name") or row.get("model_id")), fresh()), row)
            bump(per_tag.setdefault(str(row.get("tag") or ""), fresh()), row)
        for d in (total, *per_model.values(), *per_tag.values()):
            d["cost_usd"] = round(d["cost_usd"], 6)
        return {"log_file": str(p), "runs": len(run_ids), "total": total, "per_model": per_model, "per_tag": per_tag}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _cli_summary(args: argparse.Namespace) -> int:
    """Print aggregate totals of the JSONL call log (default path from config llm.log_file)."""
    log_path = args.log or (load_config().get("llm") or {}).get("log_file")
    s = LLM.summarize_log(LLM._abs(log_path) if log_path else None)
    print(json.dumps(s, indent=2))
    return 0


def _cli_smoke(args: argparse.Namespace) -> int:
    """One tiny live (or LLM_FAKE=1 offline) call through the wrapper; prints tokens/cost."""
    fake = env("LLM_FAKE") == "1"
    model = args.model or ("fake-smoke" if fake else "coder")
    llm = LLM()
    try:
        spec = llm.resolve_model(model)
    except ConfigError as exc:
        print(f"config error: {exc}")
        return 2
    prov = llm._provider(spec)
    if not getattr(prov, "has_credentials", lambda: False)():
        print("no API key found; set ANTHROPIC_API_KEY in .env (or run with LLM_FAKE=1 for a fake-provider smoke)")
        return 2
    if getattr(args, "batch", False):
        if not getattr(prov, "supports_batch", False):
            print(f"provider {getattr(prov, 'name', spec.provider)} has no batch support")
            return 2
        print(f"submitting a 1-request Message Batch on {spec.id} and polling every {args.poll}s ...")
        resp = llm.complete_batch([{"model": spec, "system": None, "user": "Reply with the single word OK.",
                                    "max_tokens": 16, "cache": False, "tag": "smoke-batch"}],
                                  poll_interval_s=args.poll)[0]
    else:
        resp = llm.complete(spec, None, "Reply with the single word OK.", max_tokens=16, cache=False, tag="smoke")
    print(json.dumps({
        "mode": resp.mode, "batch_id": resp.batch_id,
        "provider": resp.provider, "model": spec.name, "model_id": spec.id, "served_model": resp.served_model,
        "text": resp.text[:80], "stop_reason": resp.stop_reason, "refused": resp.refused,
        "input_tokens": resp.input_tokens, "output_tokens": resp.output_tokens, "cost_usd": resp.cost_usd,
        "latency_s": round(resp.latency_s, 3), "attempts": resp.attempts, "log_file": str(llm.log_file),
    }, indent=2))
    print("SMOKE OK" if resp.ok else "SMOKE FAILED")
    return 0 if resp.ok else 1


def _cli_batch_status(args: argparse.Namespace) -> int:
    """Print the status of one batch id (and any open manifests for that provider)."""
    llm = LLM()
    try:
        spec = llm.resolve_model(args.model or "coder")
    except ConfigError as exc:
        print(f"config error: {exc}")
        return 2
    prov = llm._provider(spec)
    if not getattr(prov, "supports_batch", False):
        print(f"provider {getattr(prov, 'name', spec.provider)} has no batch support")
        return 2
    if not getattr(prov, "has_credentials", lambda: False)():
        print("no API key found; set ANTHROPIC_API_KEY in .env (or run with LLM_FAKE=1)")
        return 2
    try:
        status = prov.batch_status(args.batch_status)
    except BatchUnavailableError as exc:
        print(f"batch unavailable: {exc}")
        return 2
    except Exception as exc:  # noqa: BLE001 - CLI must not traceback
        print(f"could not retrieve batch {args.batch_status}: {type(exc).__name__}: {exc}")
        return 1
    open_manifests = [{k: m.get(k) for k in ("batch_id", "created", "n_requests", "projected_usd", "status")}
                      for m in llm.open_batch_manifests(str(getattr(prov, "name", spec.provider)))]
    print(json.dumps({"batch": status, "open_manifests": open_manifests,
                      "manifest_dir": str(llm._batches_dir())}, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point (`python -m src.llm --summary | --smoke`)."""
    ap = argparse.ArgumentParser(prog="python -m src.llm", description="LLM wrapper utilities")
    ap.add_argument("--summary", action="store_true", help="aggregate the JSONL call log")
    ap.add_argument("--log", default=None, help="log file for --summary (default: llm.log_file)")
    ap.add_argument("--smoke", action="store_true", help="one tiny live call through the wrapper")
    ap.add_argument("--model", default=None, help="config role/name or raw id for --smoke (default: coder)")
    ap.add_argument("--batch", action="store_true", help="with --smoke: go through the Message Batches API")
    ap.add_argument("--poll", type=float, default=15.0, help="poll interval (s) for --smoke --batch")
    ap.add_argument("--batch-status", default=None, metavar="BATCH_ID", help="print the status of a batch")
    args = ap.parse_args(argv)
    try:  # Windows consoles default to a legacy code page; keep em-dashes etc. printable
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):  # pragma: no cover
        pass
    if args.smoke:
        return _cli_smoke(args)
    if args.summary:
        return _cli_summary(args)
    if args.batch_status:
        return _cli_batch_status(args)
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
