"""Provider back-ends for `src/llm.py` (Anthropic, OpenAI, Fake).

Every provider exposes the same tiny interface so `LLM` stays provider-agnostic:

    provider.name                     -> "anthropic" | "openai" | "deepseek" | "openrouter" | "fake"
    provider.call(req: dict) -> dict  -> normalized response dict (see `_normalize`)
    provider.count_tokens(system, user, model_id) -> int | None
    provider.has_credentials() -> bool
    provider.transient_exceptions     -> tuple of exception types worth retrying
    provider.cache_read_mult / cache_write_mult -> price multipliers for cached tokens

Optional batch protocol (Anthropic Message Batches API — 50% price, async, <=24h):
    provider.supports_batch          -> bool
    provider.batch_discount          -> float multiplier on prices (0.5)
    provider.submit_batch(items)     -> batch_id      (items = [(custom_id, req), ...])
    provider.batch_status(batch_id)  -> {"status": "in_progress"|"ended"|"canceling", "counts": {...}}
    provider.batch_results(batch_id) -> iterator of (custom_id, normalized_dict | None, error | None)
    provider.cancel_batch(batch_id)

`req` is the normalized request dict built by `LLM.complete`:
    {model_id, system, user, max_tokens, json_schema, effort, temperature,
     thinking, cache_system_prompt, fallbacks}

Nothing in this module logs prompt or response *text*; only lengths/counts.
"""
from __future__ import annotations

import json
import math
import threading
import time
from typing import Any, Callable, Iterator

from src.util import env, get_logger, sha256_of

log = get_logger("sitrep.llm")


# ---------------------------------------------------------------------------
# Shared helpers / exceptions
# ---------------------------------------------------------------------------
class TransientProviderError(RuntimeError):
    """A provider failure that is worth retrying (rate limit, overload, network)."""


class BatchUnavailableError(RuntimeError):
    """A submitted batch (or its results) can no longer be retrieved — unknown id, archived, or
    past the provider's 29-day result retention. Recoverable by resubmitting the requests."""


def estimate_tokens(text: str | None) -> int:
    """Cheap offline token estimate: ceil(len(text) / 3.5). None/empty -> 0."""
    if not text:
        return 0
    return int(math.ceil(len(text) / 3.5))


def _normalize(
    *,
    text: str,
    stop_reason: str | None,
    served_model: str | None,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    refusal_category: str | None = None,
    temperature_dropped: bool = False,
    truncated: bool = False,
    raw: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the normalized provider response dict every provider returns."""
    return {
        "text": text or "",
        "stop_reason": stop_reason,
        "served_model": served_model,
        "input_tokens": int(input_tokens or 0),
        "output_tokens": int(output_tokens or 0),
        "cache_read_tokens": int(cache_read_tokens or 0),
        "cache_write_tokens": int(cache_write_tokens or 0),
        "refusal_category": refusal_category,
        "temperature_dropped": bool(temperature_dropped),
        "truncated": bool(truncated),
        "raw": raw or {},
    }


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------
# Model ids (prefixes) that reject `temperature` with a 400.
TEMPERATURE_REJECTING_PREFIXES: tuple[str, ...] = (
    "claude-opus-5",
    "claude-sonnet-5",
    "claude-fable-5",
    "claude-mythos",
    "claude-opus-4-8",
    "claude-opus-4-7",
)
ALLOWED_EFFORT = ("low", "medium", "high", "xhigh", "max")
ALLOWED_THINKING = ("adaptive", "disabled")
# Models on which `output_config.effort` is not accepted at all.
EFFORT_UNSUPPORTED_PREFIXES: tuple[str, ...] = (
    "claude-haiku-4-5", "claude-sonnet-4-5", "claude-opus-4-1", "claude-opus-4-0", "claude-sonnet-4-0",
)
# Models that accept effort but not the `xhigh` level (added with Opus 4.7).
XHIGH_UNSUPPORTED_PREFIXES: tuple[str, ...] = ("claude-opus-4-6", "claude-sonnet-4-6", "claude-opus-4-5")
# Models where an explicit `thinking: {type: "disabled"}` returns 400 (thinking is always on).
THINKING_ALWAYS_ON_PREFIXES: tuple[str, ...] = ("claude-fable-5", "claude-mythos")
# Models where disabled thinking is only accepted at effort <= high.
DISABLED_THINKING_EFFORT_CAP_PREFIXES: tuple[str, ...] = ("claude-opus-5",)
STREAM_ABOVE_MAX_TOKENS = 16000
SERVER_FALLBACK_BETA = "server-side-fallback-2026-07-01"


def model_rejects_temperature(model_id: str) -> bool:
    """True for models where sending `temperature` returns HTTP 400."""
    return any(model_id.startswith(p) for p in TEMPERATURE_REJECTING_PREFIXES)


def adjust_params_for_model(
    model_id: str, effort: str | None, thinking: str | None
) -> tuple[str | None, str | None, list[str]]:
    """Drop/downgrade `effort`/`thinking` combinations a given model rejects with HTTP 400.

    Returns (effort, thinking, notes); `notes` lists what changed (for one-time warnings).
    """
    notes: list[str] = []
    if effort and any(model_id.startswith(p) for p in EFFORT_UNSUPPORTED_PREFIXES):
        notes.append(f"effort={effort!r} dropped (unsupported on {model_id})")
        effort = None
    elif effort == "xhigh" and any(model_id.startswith(p) for p in XHIGH_UNSUPPORTED_PREFIXES):
        notes.append(f"effort 'xhigh' downgraded to 'high' (unsupported on {model_id})")
        effort = "high"
    if thinking == "disabled":
        if any(model_id.startswith(p) for p in THINKING_ALWAYS_ON_PREFIXES):
            notes.append(f"thinking='disabled' dropped (always on for {model_id})")
            thinking = None
        elif effort in ("xhigh", "max") and any(model_id.startswith(p) for p in DISABLED_THINKING_EFFORT_CAP_PREFIXES):
            notes.append(f"thinking='disabled' dropped (not allowed with effort={effort!r} on {model_id})")
            thinking = None
    return effort, thinking, notes


def build_anthropic_kwargs(
    model_id: str,
    system: str | None,
    user: str,
    max_tokens: int,
    json_schema: dict[str, Any] | None = None,
    effort: str | None = None,
    temperature: float | None = None,
    thinking: str | None = None,
    cache_system_prompt: bool = False,
) -> tuple[dict[str, Any], bool]:
    """Pure builder for `client.messages.create(**kwargs)`.

    Returns (kwargs, temperature_dropped). Never emits budget_tokens/top_p/top_k;
    structured output goes in `output_config.format`, effort in `output_config.effort`.
    """
    if effort is not None and effort not in ALLOWED_EFFORT:
        raise ValueError(f"effort must be one of {ALLOWED_EFFORT}, got {effort!r}")
    if thinking is not None and thinking not in ALLOWED_THINKING:
        raise ValueError(f"thinking must be one of {ALLOWED_THINKING} or None, got {thinking!r}")
    effort, thinking, _notes = adjust_params_for_model(model_id, effort, thinking)
    kw: dict[str, Any] = {
        "model": model_id,
        "max_tokens": int(max_tokens),
        "messages": [{"role": "user", "content": user}],
    }
    if system:
        if cache_system_prompt:
            kw["system"] = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
        else:
            kw["system"] = system
    output_config: dict[str, Any] = {}
    if effort:
        output_config["effort"] = effort
    if json_schema:
        output_config["format"] = {"type": "json_schema", "schema": json_schema}
    if output_config:
        kw["output_config"] = output_config
    if thinking:
        kw["thinking"] = {"type": thinking}
    temperature_dropped = False
    if temperature is not None:
        if model_rejects_temperature(model_id):
            temperature_dropped = True
        else:
            kw["temperature"] = float(temperature)
    return kw, temperature_dropped


class AnthropicProvider:
    """Anthropic Messages API back-end (client constructed lazily on first use)."""

    name = "anthropic"
    cache_read_mult = 0.10   # cache-read tokens cost 0.10x input price
    cache_write_mult = 1.25  # 5-minute cache writes cost 1.25x input price
    supports_batch = True
    batch_discount = 0.5     # Message Batches API: 50% of standard prices on all token usage

    def __init__(self, *, max_retries: int = 4, timeout_s: float = 600.0, fallbacks: str = "none") -> None:
        if fallbacks not in ("none", "default"):
            raise ValueError("llm.fallbacks must be 'none' or 'default'")
        self.max_retries = max_retries
        self.timeout_s = timeout_s
        self.fallbacks = fallbacks
        self._client: Any = None
        self._lock = threading.Lock()
        self._warned_temperature: set[str] = set()
        self._warned_notes: set[str] = set()
        import anthropic  # installed; imported here to keep module import cheap for tests

        self._sdk = anthropic
        self.transient_exceptions: tuple[type[BaseException], ...] = (
            anthropic.APIConnectionError,
            anthropic.APITimeoutError,
            anthropic.RateLimitError,
            anthropic.InternalServerError,
            anthropic.OverloadedError,
        )

    # -- infrastructure -----------------------------------------------------
    def has_credentials(self) -> bool:
        """True when the SDK will find a credential (API key or auth token) in the environment."""
        return bool(env("ANTHROPIC_API_KEY") or env("ANTHROPIC_AUTH_TOKEN"))

    def _get_client(self) -> Any:
        with self._lock:
            if self._client is None:
                env("ANTHROPIC_API_KEY")  # ensures .env is loaded before the SDK reads os.environ
                env("ANTHROPIC_AUTH_TOKEN")
                self._client = self._sdk.Anthropic(max_retries=self.max_retries, timeout=self.timeout_s)
            return self._client

    # -- calls ----------------------------------------------------------------
    def call(self, req: dict[str, Any]) -> dict[str, Any]:
        model_id = req["model_id"]
        kw, temperature_dropped = build_anthropic_kwargs(
            model_id,
            req.get("system"),
            req["user"],
            req["max_tokens"],
            json_schema=req.get("json_schema"),
            effort=req.get("effort"),
            temperature=req.get("temperature"),
            thinking=req.get("thinking"),
            cache_system_prompt=bool(req.get("cache_system_prompt")),
        )
        if temperature_dropped:
            with self._lock:
                first = model_id not in self._warned_temperature
                self._warned_temperature.add(model_id)
            if first:
                log.warning("model %s rejects `temperature`; dropping it (warned once per model)", model_id)
        _, _, notes = adjust_params_for_model(model_id, req.get("effort"), req.get("thinking"))
        for note in notes:
            with self._lock:
                first = note not in self._warned_notes
                self._warned_notes.add(note)
            if first:
                log.warning("%s (warned once)", note)
        client = self._get_client()
        use_fallbacks = (req.get("fallbacks") or self.fallbacks) == "default"
        stream = kw["max_tokens"] > STREAM_ABOVE_MAX_TOKENS
        if use_fallbacks:
            kw = dict(kw, betas=[SERVER_FALLBACK_BETA], fallbacks="default")
            api = client.beta.messages
        else:
            api = client.messages
        if stream:
            with api.stream(**kw) as s:
                response = s.get_final_message()
        else:
            response = api.create(**kw)
        return self._parse(response, temperature_dropped)

    @staticmethod
    def _parse(response: Any, temperature_dropped: bool) -> dict[str, Any]:
        stop_reason = getattr(response, "stop_reason", None)
        blocks = getattr(response, "content", None) or []
        text = "".join(getattr(b, "text", "") for b in blocks if getattr(b, "type", None) == "text")
        usage = getattr(response, "usage", None)
        refusal_category = None
        if stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            refusal_category = getattr(details, "category", None) if details is not None else None
        return _normalize(
            text=text,
            stop_reason=stop_reason,
            served_model=getattr(response, "model", None),
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
            refusal_category=refusal_category,
            temperature_dropped=temperature_dropped,
            truncated=stop_reason in ("max_tokens", "model_context_window_exceeded"),
            raw={
                "id": getattr(response, "id", None),
                "n_content_blocks": len(blocks),
                "request_id": getattr(response, "_request_id", None),
            },
        )

    def count_tokens(self, system: str | None, user: str, model_id: str) -> int | None:
        try:
            kw: dict[str, Any] = {"model": model_id, "messages": [{"role": "user", "content": user}]}
            if system:
                kw["system"] = system
            return int(self._get_client().messages.count_tokens(**kw).input_tokens)
        except Exception as exc:  # noqa: BLE001 - best-effort helper
            log.warning("count_tokens failed for %s: %s: %s", model_id, type(exc).__name__, exc)
            return None

    # -- Message Batches API ---------------------------------------------------
    def batch_params(self, req: dict[str, Any]) -> dict[str, Any]:
        """Per-request params for a batch item (same builder as sync calls; no server-side
        fallbacks — the Batches API rejects that parameter; no streaming needed)."""
        kw, _ = build_anthropic_kwargs(
            req["model_id"], req.get("system"), req["user"], req["max_tokens"],
            json_schema=req.get("json_schema"), effort=req.get("effort"), temperature=req.get("temperature"),
            thinking=req.get("thinking"), cache_system_prompt=bool(req.get("cache_system_prompt")),
        )
        return kw

    def submit_batch(self, items: list[tuple[str, dict[str, Any]]]) -> str:
        """Create one Message Batch from (custom_id, req) items; returns the batch id."""
        from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
        from anthropic.types.messages.batch_create_params import Request

        requests = [
            Request(custom_id=cid, params=MessageCreateParamsNonStreaming(**self.batch_params(req)))
            for cid, req in items
        ]
        batch = self._get_client().messages.batches.create(requests=requests)
        log.info("submitted Anthropic batch %s with %d requests", batch.id, len(requests))
        return str(batch.id)

    def batch_status(self, batch_id: str) -> dict[str, Any]:
        """Retrieve one batch's processing status + per-item counts.

        `processing_status` is 'in_progress' | 'canceling' | 'ended'; only 'ended' means results
        are available (a canceling batch still ends). A batch the API no longer knows raises
        BatchUnavailableError so the caller can resubmit instead of hanging."""
        try:
            b = self._get_client().messages.batches.retrieve(batch_id)
        except self._sdk.NotFoundError as exc:
            raise BatchUnavailableError(f"batch {batch_id} not found: {exc}") from exc
        counts = getattr(b, "request_counts", None)
        return {
            "status": str(getattr(b, "processing_status", "unknown")),
            "counts": {k: int(getattr(counts, k, 0) or 0)
                       for k in ("processing", "succeeded", "errored", "canceled", "expired")},
            "ended_at": str(getattr(b, "ended_at", None) or ""),
            "expires_at": str(getattr(b, "expires_at", None) or ""),
            "results_url": getattr(b, "results_url", None),
        }

    def batch_results(self, batch_id: str) -> Iterator[tuple[str, dict[str, Any] | None, str | None]]:
        """Yield (custom_id, normalized response | None, error | None) for every item of an ended batch.

        `result` is a union: succeeded(message) | errored(error) | canceled | expired. On an errored
        item the payload is an ErrorResponse whose real type/message live one level deeper
        (`result.error.error.type` / `.message`)."""
        try:
            stream = self._get_client().messages.batches.results(batch_id)
        except self._sdk.NotFoundError as exc:
            raise BatchUnavailableError(f"results for batch {batch_id} not found: {exc}") from exc
        except self._sdk.AnthropicError as exc:  # e.g. "No `results_url`" for an archived batch
            raise BatchUnavailableError(f"results for batch {batch_id} unavailable: {exc}") from exc
        for item in stream:
            cid = str(item.custom_id)
            res = item.result
            rtype = getattr(res, "type", None)
            if rtype == "succeeded":
                yield cid, self._parse(res.message, False), None
            elif rtype == "errored":
                envelope = getattr(res, "error", None)                 # ErrorResponse
                inner = getattr(envelope, "error", None) or envelope   # ErrorObject
                etype = getattr(inner, "type", None) or "unknown"
                emsg = getattr(inner, "message", None) or ""
                rid = getattr(envelope, "request_id", None)
                yield cid, None, f"errored ({etype}): {emsg}" + (f" [request_id={rid}]" if rid else "")
            else:  # canceled | expired
                yield cid, None, str(rtype or "unknown result type")

    def cancel_batch(self, batch_id: str) -> None:
        self._get_client().messages.batches.cancel(batch_id)


# ---------------------------------------------------------------------------
# OpenAI (optional)
# ---------------------------------------------------------------------------
OPENAI_TEMPERATURE_REJECTING_PREFIXES: tuple[str, ...] = ("o1", "o3", "o4", "gpt-5")


def openai_model_rejects_temperature(model_id: str) -> bool:
    """True for OpenAI reasoning models that reject a non-default `temperature`."""
    return any(model_id.lower().startswith(p) for p in OPENAI_TEMPERATURE_REJECTING_PREFIXES)


class OpenAIProvider:
    """OpenAI chat.completions back-end (optional; `openai` imported lazily)."""

    name = "openai"
    #: Wire name for the output cap. OpenAI renamed `max_tokens` -> `max_completion_tokens`, but
    #: OpenAI-*compatible* servers did not all follow. An unknown field is silently DROPPED rather
    #: than rejected, so a wrong name here means no cap at all — not a visible error.
    max_tokens_param = "max_completion_tokens"
    cache_read_mult = 0.50   # OpenAI cached prompt tokens are billed at ~0.5x (0.25x on some models)
    cache_write_mult = 0.0   # no explicit cache-write charge on OpenAI
    supports_batch = False   # (OpenAI has its own Batch API; not wired — falls back to sync calls)

    def __init__(self, *, max_retries: int = 4, timeout_s: float = 600.0) -> None:
        self.max_retries = max_retries
        self.timeout_s = timeout_s
        self._client: Any = None
        self._lock = threading.Lock()
        self._warned_temperature: set[str] = set()
        try:
            import openai
        except ImportError as exc:  # pragma: no cover
            raise ImportError("openai provider requested but the `openai` package is not installed") from exc
        self._sdk = openai
        self.transient_exceptions: tuple[type[BaseException], ...] = (
            openai.APIConnectionError,
            openai.APITimeoutError,
            openai.RateLimitError,
            openai.InternalServerError,
        )

    def has_credentials(self) -> bool:
        return bool(env("OPENAI_API_KEY"))

    def _get_client(self) -> Any:
        with self._lock:
            if self._client is None:
                env("OPENAI_API_KEY")
                self._client = self._sdk.OpenAI(max_retries=self.max_retries, timeout=self.timeout_s)
            return self._client

    def call(self, req: dict[str, Any]) -> dict[str, Any]:
        messages: list[dict[str, str]] = []
        if req.get("system"):
            messages.append({"role": "system", "content": req["system"]})
        messages.append({"role": "user", "content": req["user"]})
        kw: dict[str, Any] = {
            "model": req["model_id"],
            "messages": messages,
            self.max_tokens_param: int(req["max_tokens"]),
        }
        temperature_dropped = False
        if req.get("temperature") is not None:
            if openai_model_rejects_temperature(req["model_id"]):
                temperature_dropped = True
                with self._lock:
                    first = req["model_id"] not in self._warned_temperature
                    self._warned_temperature.add(req["model_id"])
                if first:
                    log.warning("OpenAI model %s rejects `temperature`; dropping it (warned once per model)",
                                req["model_id"])
            else:
                kw["temperature"] = float(req["temperature"])
        if req.get("json_schema"):
            kw["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "out", "schema": req["json_schema"], "strict": False},
            }
        elif req.get("_json_object"):  # DeepSeek fallback when strict schemas are rejected
            kw["response_format"] = {"type": "json_object"}
        response = self._get_client().chat.completions.create(**kw)
        choice = response.choices[0] if getattr(response, "choices", None) else None
        message = getattr(choice, "message", None) if choice else None
        text = (getattr(message, "content", None) or "") if message is not None else ""
        refusal_msg = getattr(message, "refusal", None) if message is not None else None
        finish = getattr(choice, "finish_reason", None) if choice else None
        usage = getattr(response, "usage", None)
        details = getattr(usage, "prompt_tokens_details", None)
        cached = getattr(details, "cached_tokens", 0) or 0
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        stop_reason = {"stop": "end_turn", "length": "max_tokens", "content_filter": "refusal"}.get(finish, finish)
        refusal_category = None
        if refusal_msg:  # structured-output refusal: content=None, refusal=<text>, finish_reason='stop'
            stop_reason, refusal_category, text = "refusal", "openai_refusal", ""
        elif finish == "content_filter":
            refusal_category = "content_filter"
        return _normalize(
            text=text,
            stop_reason=stop_reason,
            served_model=getattr(response, "model", None),
            input_tokens=max(prompt_tokens - cached, 0),
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            cache_read_tokens=cached,
            refusal_category=refusal_category,
            temperature_dropped=temperature_dropped,
            truncated=finish == "length",
            raw={"id": getattr(response, "id", None), "finish_reason": finish},
        )

    def count_tokens(self, system: str | None, user: str, model_id: str) -> int | None:
        return None  # heuristic estimate is used instead


# ---------------------------------------------------------------------------
# DeepSeek (OpenAI-compatible wire format, different host + key)
# ---------------------------------------------------------------------------
DEEPSEEK_BASE_URL = "https://api.deepseek.com"


class DeepSeekProvider(OpenAIProvider):
    """DeepSeek back-end, reached through the OpenAI SDK with a different base URL.

    DeepSeek speaks the OpenAI chat-completions format, so the request/response handling is
    inherited wholesale. Three differences matter:

    * credentials come from ``DEEPSEEK_API_KEY`` (falling back to ``DEEPSEEK_TOKEN``);
    * there is **no batch API**, so `LLM.complete_batch` falls back to concurrent sync calls for
      this provider. That costs nothing extra here: DeepSeek's list price is already below
      Anthropic's *batched* price, so the cheaper route is simply to call it directly;
    * strict `json_schema` response formats are newer than plain JSON mode, so a rejection is
      retried once in `{"type": "json_object"}` mode rather than failing the call. `LLM` parses
      the text either way.
    """

    name = "deepseek"
    supports_batch = False
    # DeepSeek's API takes `max_tokens`; it ignores `max_completion_tokens` silently, which let
    # completions run to the model's own 64k ceiling (75% of calls overran their configured cap).
    max_tokens_param = "max_tokens"
    cache_read_mult = 0.10   # DeepSeek prices a cache hit far below a miss; 0.1x is the safe floor
    cache_write_mult = 1.0

    def __init__(self, *, max_retries: int = 4, timeout_s: float = 600.0,
                 base_url: str | None = None) -> None:
        super().__init__(max_retries=max_retries, timeout_s=timeout_s)
        self.base_url = base_url or env("DEEPSEEK_BASE_URL") or DEEPSEEK_BASE_URL
        # Model ids known to reject a strict json_schema response_format. Learned on the first
        # rejection so later calls skip the doomed attempt instead of paying two round trips each.
        self._no_json_schema: set[str] = set()

    def has_credentials(self) -> bool:
        """True when a DeepSeek key is present in the environment."""
        return bool(env("DEEPSEEK_API_KEY") or env("DEEPSEEK_TOKEN"))

    def _get_client(self) -> Any:
        with self._lock:
            if self._client is None:
                key = env("DEEPSEEK_API_KEY") or env("DEEPSEEK_TOKEN")
                if not key:
                    raise RuntimeError(
                        "DEEPSEEK_API_KEY is not set — add it to .env to use a deepseek-* model")
                self._client = self._sdk.OpenAI(api_key=key, base_url=self.base_url,
                                                max_retries=self.max_retries, timeout=self.timeout_s)
            return self._client

    def call(self, req: dict[str, Any]) -> dict[str, Any]:
        """Same as the OpenAI path, but degrade a strict json_schema to plain JSON mode on 400.

        The rejection is a property of the model, not the request, so it is remembered: only the
        first call pays the extra round trip.
        """
        model_id = req.get("model_id", "")
        if req.get("json_schema") and model_id in self._no_json_schema:
            return super().call(dict(req, json_schema=None, _json_object=True))
        try:
            return super().call(req)
        except self._sdk.BadRequestError as exc:
            msg = str(exc).lower()
            if not req.get("json_schema") or ("response_format" not in msg and "json_schema" not in msg):
                raise
            with self._lock:
                first = model_id not in self._no_json_schema
                self._no_json_schema.add(model_id)
            if first:
                log.warning("%s rejects a strict json_schema response_format; using json_object mode "
                            "for the rest of this run (output is still parsed and validated)", model_id)
            return super().call(dict(req, json_schema=None, _json_object=True))


# ---------------------------------------------------------------------------
# OpenRouter (OpenAI-compatible gateway; namespaced model ids like openai/gpt-5.6-luna)
# ---------------------------------------------------------------------------
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterProvider(OpenAIProvider):
    """OpenRouter back-end — the OpenAI wire format against a gateway that fronts many vendors.

    Used here to run a **cross-family judge**: scoring DeepSeek-generated sitreps with an OpenAI
    model removes the same-family circularity that a DeepSeek judge carries.

    Differences from the plain OpenAI provider:

    * credentials from ``OPENROUTER_API_KEY``;
    * model ids are namespaced (``openai/gpt-5.6-luna``), which is also how the provider is
      inferred when config does not name it;
    * optional ``HTTP-Referer`` / ``X-Title`` attribution headers, set from
      ``OPENROUTER_SITE_URL`` / ``OPENROUTER_APP_NAME`` when present;
    * no batch API, so `LLM.complete_batch` runs these as concurrent direct calls. (OpenRouter
      exposes separate ``:batch`` model slugs; select one by model id if you want that pricing.)
    """

    name = "openrouter"
    supports_batch = False
    cache_read_mult = 0.10   # gpt-5.6-luna: $0.02 read vs $0.20 input
    cache_write_mult = 1.25  # $0.25 write vs $0.20 input

    def __init__(self, *, max_retries: int = 4, timeout_s: float = 600.0,
                 base_url: str | None = None) -> None:
        super().__init__(max_retries=max_retries, timeout_s=timeout_s)
        self.base_url = base_url or env("OPENROUTER_BASE_URL") or OPENROUTER_BASE_URL

    def has_credentials(self) -> bool:
        """True when an OpenRouter key is present in the environment."""
        return bool(env("OPENROUTER_API_KEY"))

    def _get_client(self) -> Any:
        with self._lock:
            if self._client is None:
                key = env("OPENROUTER_API_KEY")
                if not key:
                    raise RuntimeError(
                        "OPENROUTER_API_KEY is not set — add it to .env to use an OpenRouter model")
                headers = {}
                if env("OPENROUTER_SITE_URL"):
                    headers["HTTP-Referer"] = env("OPENROUTER_SITE_URL")
                if env("OPENROUTER_APP_NAME"):
                    headers["X-Title"] = env("OPENROUTER_APP_NAME")
                self._client = self._sdk.OpenAI(
                    api_key=key, base_url=self.base_url, max_retries=self.max_retries,
                    timeout=self.timeout_s, default_headers=headers or None)
            return self._client


# ---------------------------------------------------------------------------
# Fake (tests + LLM_FAKE=1)
# ---------------------------------------------------------------------------
class FakeProvider:
    """Deterministic offline provider used by tests and `LLM_FAKE=1` smoke runs.

    - `responder(req) -> str` overrides the reply; default `FAKE:<sha256(user)[:8]>`
      (or a small JSON object when `req['json_schema']` is set).
    - `fail_times`: raise TransientProviderError for the first N calls.
    - user text containing `[[REFUSE]]` -> stop_reason "refusal", empty text.
    - Tracks `calls`, `max_concurrent`, `last_kwargs` for assertions.
    """

    name = "fake"
    cache_read_mult = 0.10
    cache_write_mult = 1.25
    supports_batch = True
    batch_discount = 0.5
    transient_exceptions: tuple[type[BaseException], ...] = ()

    def __init__(
        self,
        responder: Callable[[dict[str, Any]], str] | None = None,
        fail_times: int = 0,
        latency_s: float = 0.0,
        *,
        batch_polls_until_done: int = 0,
        batch_fail_ids: set[str] | None = None,
        batch_result_kinds: dict[str, str] | None = None,
        batch_missing_ids: set[str] | None = None,
        batch_unavailable_ids: set[str] | None = None,
        batch_results_fail_after: int | None = None,
    ) -> None:
        self.responder = responder
        self.fail_times = fail_times
        self.latency_s = latency_s
        self.calls = 0
        self.max_concurrent = 0
        self.last_kwargs: dict[str, Any] | None = None
        self._active = 0
        self._lock = threading.Lock()
        # batch simulation
        self.batch_polls_until_done = batch_polls_until_done
        self.batch_fail_ids: set[str] = set(batch_fail_ids or ())
        # custom_id -> 'canceled' | 'expired' (anything else behaves like a success)
        self.batch_result_kinds: dict[str, str] = dict(batch_result_kinds or {})
        self.batch_missing_ids: set[str] = set(batch_missing_ids or ())      # omitted from results
        self.batch_unavailable_ids: set[str] = set(batch_unavailable_ids or ())  # batch ids that 404
        self.batch_results_fail_after = batch_results_fail_after            # raise mid-stream
        self.batches: dict[str, dict[str, Any]] = {}
        self.batch_submissions = 0
        self.batch_items = 0

    def has_credentials(self) -> bool:
        return True

    def call(self, req: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self.calls += 1
            self._active += 1
            self.max_concurrent = max(self.max_concurrent, self._active)
            self.last_kwargs = dict(req)
            n = self.calls
        try:
            if self.latency_s:
                time.sleep(self.latency_s)
            if n <= self.fail_times:
                raise TransientProviderError(f"fake transient failure {n}/{self.fail_times}")
            user = req["user"]
            system = req.get("system")
            if "[[REFUSE]]" in user:
                return _normalize(
                    text="",
                    stop_reason="refusal",
                    served_model=req["model_id"],
                    input_tokens=estimate_tokens(system) + estimate_tokens(user),
                    output_tokens=0,
                    refusal_category="fake",
                    raw={"fake": True},
                )
            if self.responder is not None:
                text = self.responder(req)
            elif req.get("json_schema"):
                text = json.dumps({"ok": True, "echo_len": len(user)})
            else:
                text = f"FAKE:{sha256_of(user)[:8]}"
            return _normalize(
                text=text,
                stop_reason="end_turn",
                served_model=req["model_id"],
                input_tokens=estimate_tokens(system) + estimate_tokens(user),
                output_tokens=estimate_tokens(text),
                raw={"fake": True},
            )
        finally:
            with self._lock:
                self._active -= 1

    def count_tokens(self, system: str | None, user: str, model_id: str) -> int | None:
        return None

    # -- batch simulation --------------------------------------------------------
    def submit_batch(self, items: list[tuple[str, dict[str, Any]]]) -> str:
        with self._lock:
            self.batch_submissions += 1
            bid = f"msgbatch_fake_{self.batch_submissions:04d}"
            self.batches[bid] = {"items": list(items), "polls": 0}
        return bid

    def batch_status(self, batch_id: str) -> dict[str, Any]:
        if batch_id in self.batch_unavailable_ids or batch_id not in self.batches:
            raise BatchUnavailableError(f"batch {batch_id} not found (fake)")
        b = self.batches[batch_id]
        b["polls"] += 1
        done = b["polls"] > self.batch_polls_until_done
        n = len(b["items"])
        return {"status": "ended" if done else "in_progress",
                "counts": {"processing": 0 if done else n, "succeeded": n if done else 0,
                           "errored": 0, "canceled": 0, "expired": 0}}

    def batch_results(self, batch_id: str) -> Iterator[tuple[str, dict[str, Any] | None, str | None]]:
        if batch_id in self.batch_unavailable_ids or batch_id not in self.batches:
            raise BatchUnavailableError(f"results for batch {batch_id} not found (fake)")
        yielded = 0
        for cid, req in self.batches[batch_id]["items"]:
            if self.batch_results_fail_after is not None and yielded >= self.batch_results_fail_after:
                raise TransientProviderError("simulated failure while streaming batch results")
            if cid in self.batch_missing_ids:
                continue
            with self._lock:
                self.batch_items += 1
            yielded += 1
            if cid in self.batch_fail_ids:
                yield cid, None, "errored (invalid_request_error): simulated batch failure"
            elif cid in self.batch_result_kinds:
                yield cid, None, self.batch_result_kinds[cid]
            else:
                yield cid, self.call(req), None

    def cancel_batch(self, batch_id: str) -> None:
        self.batches.pop(batch_id, None)
