# `src/llm.py` — the project's only door to a language model

Every model call in this repo goes through `LLM` (CLAUDE.md rule 2). It gives you a disk cache,
a JSONL call log with tokens and cost, retries, a concurrency cap and a **fail-closed cost gate**.
Provider back-ends (Anthropic, OpenAI, Fake) live in `src/llm_providers.py`.

## Calling it

```python
from src.llm import LLM, CostLimitExceeded

llm = LLM()                                   # reads config.yaml (llm.* + models.*)
spec = llm.resolve_model("coder")             # config role: coder | judge
spec = llm.resolve_model("api-strong")        # a models.generation entry by name
spec = llm.resolve_model("claude-haiku-4-5")  # raw id (provider inferred from the prefix)

r = llm.complete("judge", system="You score sitreps.", user=prompt,
                 max_tokens=1024, json_schema=SCHEMA, effort="low", tag="judge-v1",
                 meta={"event": "nepal-2015", "slot": "casualties"})
r.text, r.json, r.json_error, r.ok, r.refused, r.cached, r.cost_usd, r.input_tokens, r.output_tokens

# many requests, synchronous (ThreadPoolExecutor, <= llm.max_concurrency, results in input order)
est = llm.estimate_cost(requests)   # heuristic ceil(len/3.5) or count_tokens API when a key is present
llm.gate(est)                       # raises CostLimitExceeded before anything is sent  <- smoke-first rule
resps = llm.complete_many(requests) # per-item errors -> LLMResponse(error=...); CostLimitExceeded re-raised

# many requests, Anthropic Message Batches API (50% price, async, <= 24 h) — the default for FULL runs
llm.gate(llm.estimate_cost(requests, batch=True))
resps = llm.complete_batch(requests)               # same dicts, same order; polls every llm.batch.poll_interval_s
resps = llm.run(requests, batch=not smoke)         # dispatcher used by the phase scripts
print(llm.summary())                # {'run_id', 'calls', 'cached', 'input_tokens', 'output_tokens', 'cost_usd', ...}
```

## Batch mode (`complete_batch`) — how bulk phases run

Project policy: **smoke runs are synchronous, full runs go through the Message Batches API**
(50% of list price, asynchronous, results usually in well under an hour, hard limit 24 h).
`complete_batch(requests, poll_interval_s=None, max_wait_s=None, wait=True)`:

1. **Normalise + validate.** Requests are the same dicts as `complete_many` (unknown keys now raise
   `TypeError` in both paths). Cache hits are served from disk (`mode="cache"`), identical requests
   collapse to one batch item (`custom_id` = the 64-char request hash), and providers without batch
   support fall back to `complete_many`. Batch requests are hashed with `fallbacks="none"` (the
   Batches API has no fallbacks) so a batch result is reusable by later sync calls.
2. **Gate.** The projected cost of everything that will be waited on, priced at the batch discount,
   is reserved through the same fail-closed gate *before* submission.
3. **Submit.** Chunked by `llm.batch.chunk_size` (default 10 000; also split at ~200 MB to stay under
   the API's 256 MB cap). **One manifest per created batch**, written immediately after each chunk
   succeeds: `<cache_dir>/batches/<batch_id>.json` with its `custom_ids`, models, projection and
   `status: submitted`. A failure on a later chunk therefore never orphans an earlier one.
4. **Resume.** On entry, any request whose hash is listed in an open manifest is **re-attached** to
   that batch instead of resubmitted — after a crash, a Ctrl-C, a timeout, or a deliberate
   `wait=False` submission. Partial collection is safe too: already-collected items are cache hits,
   the rest re-attach, and a manifest is only marked `done` once every one of its ids is cached or
   errored. Nothing is ever paid for twice.
5. **Poll.** `batches.retrieve` every `llm.batch.poll_interval_s` (default 30 s), transient errors
   retried like normal calls, until `ended`. The deadline (`llm.batch.max_wait_h`, default 26 h) is
   measured from *batch creation*, and a `TimeoutError` leaves the manifest open for a later resume.
   A batch the API no longer knows (`NotFound`, archived, past the 29-day result retention) is
   reported as `BatchUnavailableError`, its manifest marked `failed`, and the next run resubmits it.
6. **Collect.** Succeeded → cost booked at 50% of list price, written to the same disk cache (so a
   later sync call is a hit), one log row per input request (`mode: batch`, duplicates `cache`).
   Errored / canceled / expired / missing → `LLMResponse(error=...)`, **not cached**, so a re-run
   resubmits only those; the error carries the API's real error type and `request_id`. Refusals are
   returned, never cached. Each item releases its own slice of the reservation as it is booked, so
   `committed_usd() == run_cost_usd + run_assumed_usd` once a run settles — including on exceptions.

`wait=False` returns straight after submission with `mode="pending"` placeholders (deliberately
**not** written to the call log, so `--summary` counts stay honest); call again later to collect.
`python -m src.llm --smoke --batch` submits a one-request batch and waits;
`python -m src.llm --batch-status msgbatch_...` prints its counts plus any open manifests.

## Where things live

| What | Where | Notes |
|---|---|---|
| disk cache | `llm.cache_dir` (default `.llm_cache/`, gitignored) | `<key[:2]>/<key>.json`; key = sha256 of {provider, model_id, system, user, max_tokens, json_schema, effort, temperature, thinking, fallbacks}. Refusals and errors are never cached. A hit costs $0 against the run meter (`cost_usd` is copied for information). |
| call log | `llm.log_file` (default `results/llm_calls.jsonl`, gitignored) | one row per `complete()` — cache hits and errors included. Columns: tokens, cost, `run_cost_usd_after`, latency, attempts, stop_reason, refusal, `temperature_dropped`, and **only the lengths** of system/user/output. Prompt or response text is never written (rule 7). |
| summary | `python -m src.llm --summary [--log path]` | totals + per model + per tag; `LLM.summarize_log(path)` in code |
| smoke | `python -m src.llm --smoke [--model coder]` | one tiny live call ("Reply with the single word OK."), `cache=False`, `tag=smoke`; prints tokens/cost/served_model |

## Cost gate (fail closed)

* Prices are `PRICES_USD_PER_MTOK` (below) plus any `llm.prices` overrides in `config.yaml`
  (`{model_prefix: [in, out]}`), matched by **longest id prefix**.
* `cost_usd = input_tokens*in + cache_read*0.10*in + cache_write*1.25*in + output_tokens*out` (all /1e6).
  Anthropic `input_tokens` already excludes cached tokens, so this is exact.
* Before **every uncached call** the wrapper reserves a projection (heuristic input tokens + `max_tokens`
  worth of output, or `llm.unknown_price_assumed_usd_per_call` = $0.05 when the model has no price) and
  raises `CostLimitExceeded` if `spent + assumed + in-flight + projection > llm.cost_limit_usd_per_run`
  (config default $25). Concurrent threads cannot slip past the gate together. Calls to models with no
  known price book the assumed $0.05 into `run_assumed_usd` (reported in `summary()`), so a long run of
  unpriced calls still trips the gate.
* `estimate_cost()` + `gate()` project a whole batch before it starts; cached requests count as $0.
* Bypass only deliberately: `LLM(allow_over_limit=True)` or `LLM_ALLOW_OVER_LIMIT=1`, after the user has
  approved the projected spend.

| model id prefix | $/MTok in | $/MTok out |
|---|---|---|
| claude-fable-5, claude-mythos-5 | 10 | 50 |
| claude-opus-5, claude-opus-4-8, claude-opus-4-7, claude-opus-4-6, claude-opus-4-5 | 5 | 25 |
| claude-sonnet-5 (list; intro $2/$10 until 2026-08-31 — we gate at list) | 3 | 15 |
| claude-sonnet-4-6, claude-sonnet-4-5 | 3 | 15 |
| claude-haiku-4-5 | 1 | 5 |
| fake* (tests / LLM_FAKE) | 1 | 5 |

Unknown ids: `cost_usd=None` in the log (warned once), $0.05/call assumed by the gate.

## Retries, refusals, truncation, fallbacks

* Transient failures (`APIConnectionError`, `APITimeoutError`, `RateLimitError`, `InternalServerError`,
  `OverloadedError`, `TransientProviderError`) are retried with exponential jitter
  (`llm.max_attempts` default 4, on top of the SDK's own `llm.sdk_max_retries` default 4).
  400/401/403/404 propagate immediately. `attempts` is recorded per call.
* **Refusals are data, not errors.** `stop_reason == "refusal"` (Opus 5 / Fable 5 safety classifiers,
  HTTP 200 with possibly empty content) returns `refused=True`, `refusal_category` from
  `stop_details`, `ok=False`; nothing is raised or cached.
* `stop_reason == "max_tokens"` is warned and returned (`raw["truncated"]=True`); pass
  `max_tokens_hard=True` to raise instead. Requests with `max_tokens > 16000` are streamed.
* **Server-side fallbacks default to off** (`llm.fallbacks: none`). The Anthropic API can re-run a
  refused request on another model (`fallbacks="default"`, beta `server-side-fallback-2026-07-01`);
  the wrapper supports it, but this is a *benchmarking* pipeline: silently substituting the model would
  corrupt the "which model wrote this sitrep" variable. If you turn it on (`llm.fallbacks: default`),
  `served_model` in the response and log tells you which model actually answered — filter on it.

## Temperature caveat

`claude-opus-5`, `claude-sonnet-5`, `claude-fable-5`, `claude-mythos-*`, `claude-opus-4-8` and
`claude-opus-4-7` reject `temperature` (HTTP 400). `build_anthropic_kwargs` drops it for those ids
(warning once per model, `temperature_dropped: true` in the log row) and keeps it for
`claude-haiku-4-5`, `claude-sonnet-4-6`, `claude-sonnet-4-5`, `claude-opus-4-6`, `claude-opus-4-5`.
`budget_tokens`, `top_p`, `top_k` and the deprecated top-level `output_format` are never sent;
structured output goes in `output_config.format`, effort in `output_config.effort`. A dropped
temperature is also nulled **before hashing**, so the same request with and without it shares one
cache entry on those models (and stays distinct on models that honour temperature).

Other model-specific guards (`adjust_params_for_model`, warned once each): `effort` is dropped on
Haiku 4.5 / Sonnet 4.5 / Opus ≤4.1 (unsupported), `xhigh` is downgraded to `high` on Opus 4.6 /
Sonnet 4.6 / Opus 4.5, `thinking="disabled"` is dropped on Fable 5 / Mythos (always on) and on Opus 5
when combined with `xhigh`/`max` effort. Practical consequence for this project: **do not rely on
`temperature=0` for determinism** on the Claude 5 family — determinism comes from the disk cache
(identical request → identical cached response) and from low `effort`; say so in the methods section.

## `LLM_FAKE=1` and tests

`LLM_FAKE=1` makes every model resolve to `FakeProvider` (deterministic `FAKE:<sha256(user)[:8]>`
replies, a small JSON object when a `json_schema` is given, `[[REFUSE]]` in the prompt simulates a
refusal, `$1/$5` per MTok), so `python -m src.llm --smoke` and whole-pipeline dry runs work offline —
including cache, log and cost gate. Tests inject the same class through
`LLM(provider_factory=lambda spec: FakeProvider(...))` with `tmp_path` cache/log; the pure builder
`build_anthropic_kwargs(...)` is unit-tested directly. Run: `.venv/Scripts/python.exe -m pytest -q tests/test_llm.py`.
