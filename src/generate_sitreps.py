"""Phase 3 - machine sitreps from social streams (PLAN §Phase 3).

For every event-day whose stream carries enough volume, take a capped, label-stratified
sample of posts and ask a model to write a situation report. Two arms:

* ``generic`` - a plain "write a situation report" instruction. The prompt says nothing
  about what practitioners actually record; it is the control condition.
* ``schema_guided`` - the same, plus the Phase 2 slot list (names + definitions). This
  tests whether *telling* the model what practitioners need closes the completeness gap.

Days whose sampled posts exceed a token budget are handled map-reduce style: the posts are
chunked deterministically, each chunk is digested, then the digests are composed into one
sitrep. Both rounds are cost-gated and go through `src.llm` like every other model call.

Outputs: ``data/processed/sitreps_machine/<event>/<day>__<model>__<arm>.json`` holding
``{event, day, model, arm, n_posts, text, generated_at, request_hash}``, a per-run manifest
beside them, and a run table at ``results/tables/machine_sitreps.csv``.

Data hygiene (CLAUDE.md rule 7): post text goes into prompts only. `meta=` on every LLM
call carries ids and labels - never text - and everything written lands under `data/`
(gitignored) or as counts in `results/`.

The ``schema_guided`` arm needs Phase 2 to have run: it reads the slot list from
``data/processed/schema/schema.yaml`` (or ``schema.yaml`` / ``candidate_slots.yaml``).
If no schema file exists yet the arm is refused with a clear message rather than guessed at.

CLI:
    python -m src.generate_sitreps --event cyclone_idai_2019 --smoke 3
    python -m src.generate_sitreps --event cyclone_idai_2019 --arms generic schema_guided \
        --models api-strong api-fast --batch
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple, Sequence

from src.llm import LLM, CostLimitExceeded, ConfigError, LLMResponse, estimate_tokens
from src.util import (
    cfg_path,
    resolve_event_dir,
    ensure_dir,
    get_logger,
    load_config,
    read_jsonl,
    utc_now_iso,
    write_json,
)

log = get_logger("sitrep.generate")

ARMS: tuple[str, ...] = ("generic", "schema_guided")
DEFAULT_MIN_POSTS = 30          # an event-day thinner than this is not worth reporting on
DEFAULT_CAP = 400               # sampling.posts_per_event_day
DEFAULT_BUDGET_TOKENS = 6000    # posts block above this -> map-reduce
DEFAULT_MAX_TOKENS = 2000       # sitrep length ceiling
SCHEMA_CANDIDATES = ("schema/schema.yaml", "schema.yaml", "schema/candidate_slots.yaml",
                     "candidate_slots.yaml")
TABLE_COLUMNS = ("event", "day", "model", "model_id", "arm", "n_posts", "n_chunks", "map_reduce",
                 "n_chars", "status", "request_hash")


# ---------------------------------------------------------------------------
# data types
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DayFile:
    """One event-day stream file that passed the volume filter."""

    day: str
    path: Path
    n_posts: int


@dataclass(frozen=True)
class Slot:
    """One induced information slot (Phase 2 schema)."""

    id: str
    name: str
    definition: str


class Prompt(NamedTuple):
    """A system/user prompt pair. `text` is both halves, for tests and token estimates."""

    system: str
    user: str

    @property
    def text(self) -> str:
        return f"{self.system}\n\n{self.user}"


@dataclass
class Job:
    """One (day, model, arm) generation, with its sampled posts and - if long - its chunks."""

    event: str
    day: str
    model: str
    arm: str
    posts: list[dict[str, Any]]
    chunks: list[list[dict[str, Any]]] = field(default_factory=list)
    model_id: str = ""
    digests: list[str] = field(default_factory=list)
    text: str = ""
    request_hash: str = ""
    status: str = "planned"

    @property
    def map_reduce(self) -> bool:
        return len(self.chunks) > 1

    @property
    def stem(self) -> str:
        return f"{self.day}__{self.model}__{self.arm}"

    def row(self) -> dict[str, Any]:
        """The provenance row for `results/tables/machine_sitreps.csv`."""
        return {"event": self.event, "day": self.day, "model": self.model, "model_id": self.model_id,
                "arm": self.arm, "n_posts": len(self.posts), "n_chunks": len(self.chunks),
                "map_reduce": self.map_reduce, "n_chars": len(self.text), "status": self.status,
                "request_hash": self.request_hash}


class SchemaNotAvailable(FileNotFoundError):
    """The Phase 2 slot list the schema_guided arm needs does not exist yet."""


def _default_processed_dir() -> Path:
    return cfg_path(load_config(), "processed", "data/processed")


def _default_table_path() -> Path:
    return cfg_path(load_config(), "results", "results") / "tables" / "machine_sitreps.csv"


# ---------------------------------------------------------------------------
# day selection + sampling
# ---------------------------------------------------------------------------
def select_event_days(event: str, min_posts: int = DEFAULT_MIN_POSTS, *,
                      stream_dir: str | Path | None = None) -> list[DayFile]:
    """Event-days holding at least `min_posts` posts, ascending by date.

    Reads `data/processed/streams/<event>/<YYYY-MM-DD>.jsonl`; helper files (`_stats.json`
    and anything else starting with `_`) are ignored. Raises FileNotFoundError when the
    event has no collected stream at all.
    """
    base = Path(stream_dir) if stream_dir else _default_processed_dir() / "streams"
    ev_dir = resolve_event_dir(base, event)
    if ev_dir is None:
        raise FileNotFoundError(
            f"no social stream for event {event!r} at {base / event} - run "
            f"`python -m src.collect_social --events {event}` first")
    out: list[DayFile] = []
    for p in sorted(ev_dir.glob("*.jsonl")):
        if p.name.startswith("_"):
            continue
        n = sum(1 for _ in read_jsonl(p))
        if n >= min_posts:
            out.append(DayFile(day=p.stem, path=p, n_posts=n))
        else:
            log.debug("%s %s: %d posts < min_posts=%d - skipped", event, p.stem, n, min_posts)
    log.info("%s: %d event-day(s) with >= %d posts", event, len(out), min_posts)
    return out


def _post_sort_key(p: dict[str, Any]) -> tuple[str, str]:
    return (str(p.get("created_at", "")), str(p.get("tweet_id", "")))


def _allocate(counts: dict[str, int], cap: int) -> dict[str, int]:
    """Largest-remainder quota per label; every present label keeps >= 1 when `cap` allows."""
    labels = sorted(counts)
    if cap >= sum(counts.values()):
        return {label: counts[label] for label in labels}
    quota = {label: (1 if cap >= len(labels) else 0) for label in labels}
    left = cap - sum(quota.values())
    residual = {label: counts[label] - quota[label] for label in labels}
    pool = sum(residual.values())
    exact = {label: (residual[label] * left / pool if pool else 0.0) for label in labels}
    for label in labels:
        take = min(int(exact[label]), residual[label])
        quota[label] += take
        left -= take
    order = sorted(labels, key=lambda label: (-(exact[label] % 1.0), -counts[label], label))
    while left > 0:
        progressed = False
        for label in order:
            if left == 0:
                break
            if quota[label] < counts[label]:
                quota[label] += 1
                left -= 1
                progressed = True
        if not progressed:  # pragma: no cover - cap < total guarantees progress
            break
    return quota


def sample_posts(day_file: str | Path, cap: int = DEFAULT_CAP, seed: int = 17) -> list[dict[str, Any]]:
    """A capped sample of one day's posts, stratified by `class_label`.

    Every label present keeps at least one post (when `cap` allows), so no information type
    is sampled away. Deterministic: the draw is seeded from `seed` plus the file name, and
    the result is returned in (created_at, tweet_id) order.
    """
    if cap <= 0:
        raise ValueError(f"cap must be >= 1, got {cap}")
    path = Path(day_file)
    posts = [p for p in read_jsonl(path) if str(p.get("text", "")).strip()]
    if len(posts) <= cap:
        return sorted(posts, key=_post_sort_key)
    by_label: dict[str, list[dict[str, Any]]] = {}
    for p in posts:
        by_label.setdefault(str(p.get("class_label") or "unlabelled"), []).append(p)
    quota = _allocate({label: len(v) for label, v in by_label.items()}, cap)
    rnd = random.Random(f"{seed}:{path.name}")
    chosen: list[dict[str, Any]] = []
    for label in sorted(by_label):
        group = sorted(by_label[label], key=_post_sort_key)
        idx = list(range(len(group)))
        rnd.shuffle(idx)
        chosen.extend(group[i] for i in sorted(idx[: quota[label]]))
    return sorted(chosen, key=_post_sort_key)


# ---------------------------------------------------------------------------
# schema (the Phase 2 slot list)
# ---------------------------------------------------------------------------
def load_slots(path: str | Path | None = None, *, processed_dir: str | Path | None = None,
               allow_candidates: bool = False) -> list[Slot]:
    """Load the Phase 2 slot list; raises SchemaNotAvailable when Phase 2 has not run."""
    import yaml

    if path is not None:
        candidates = [Path(path)]
    else:
        base = Path(processed_dir) if processed_dir else _default_processed_dir()
        candidates = [base / c for c in SCHEMA_CANDIDATES]
    for p in candidates:
        if not p.is_file():
            continue
        with open(p, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or []
        rows = data.get("slots") or data.get("candidate_slots") or [] if isinstance(data, dict) else data
        if not rows:
            raise SchemaNotAvailable(f"{p} contains no slots")
        slots: list[Slot] = []
        for r in rows:
            sid = str(r.get("id") or r.get("name") or "").strip()
            if not sid:
                raise ValueError(f"{p}: every slot needs an 'id' (or a 'name')")
            slots.append(Slot(id=sid, name=str(r.get("name") or sid).strip(),
                              definition=str(r.get("definition") or "").strip()))
        # CLAUDE.md rule 4: only an ADJUDICATED schema may drive the schema_guided arm.
        # candidate_slots.yaml is raw Phase 2 output that no human has approved yet.
        if "candidate" in p.name and not allow_candidates:
            raise SchemaNotAvailable(
                f"{p} holds UN-ADJUDICATED candidate slots. The schema_guided arm needs a schema a "
                "human approved (CLAUDE.md rule 4). Freeze one first:\n"
                "  python -m src.freeze_schema --review\n"
                '  python -m src.freeze_schema --approve --by "Your Name"\n'
                "or pass --allow-candidate-schema to override deliberately "
                "(must be disclosed in the paper).")
        if isinstance(data, dict) and data.get("approved_by"):
            log.info("using schema approved by %s at %s", data["approved_by"], data.get("approved_at"))
        log.info("loaded %d slots from %s", len(slots), p)
        return slots
    raise SchemaNotAvailable(
        "no Phase 2 schema found (looked for " + ", ".join(str(c) for c in candidates) +
        ") - run Phase 2 first, or generate with `--arms generic` only")


# ---------------------------------------------------------------------------
# prompts (pure functions)
# ---------------------------------------------------------------------------
SYSTEM = "You are an experienced humanitarian information officer writing situation reports."
_NO_INVENT = ("Use only what the material below contains; add no outside knowledge and invent "
              "no details.")


def event_label(event: str) -> str:
    """`cyclone_idai_2019` -> `Cyclone Idai 2019`."""
    return " ".join(part if part.isdigit() else part.capitalize() for part in str(event).split("_"))


def render_posts(posts: Sequence[dict[str, Any]]) -> str:
    """One line per post: `- (label) text`, whitespace collapsed. Deterministic."""
    lines = []
    for p in posts:
        text = " ".join(str(p.get("text", "")).split())
        label = str(p.get("class_label") or "unlabelled").replace("_", " ")
        lines.append(f"- ({label}) {text}")
    return "\n".join(lines)


def _slot_block(slots: Sequence[Slot]) -> str:
    return "\n".join(f"- {s.name} [{s.id}]: {s.definition}" if s.definition else f"- {s.name} [{s.id}]"
                     for s in slots)


def _check_arm(arm: str, slots: Sequence[Slot] | None) -> None:
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")
    if arm == "schema_guided" and not slots:
        raise ValueError("the schema_guided arm needs the Phase 2 slot list (slots=...)")


def _arm_instruction(arm: str, slots: Sequence[Slot] | None) -> str:
    """The only text that differs between the two arms."""
    _check_arm(arm, slots)
    base = "Write a situation report for this day.\n" + _NO_INVENT
    if arm == "generic":
        return base
    return (
        base + "\n\nHumanitarian situation reports record the following information:\n\n"
        f"{_slot_block(slots or [])}\n\n"
        "Use these names as section headings, in this order. Where the material holds nothing "
        'for a heading, write "No information" under it rather than filling it in.'
    )


def _header(n: int, event: str | None, day: str | None, unit: str) -> str:
    where = f" about {event_label(event)}" if event else ""
    when = f" posted on {day}" if day else ""
    return f"Below are {n} {unit}{where}{when}."


def build_prompt(posts: Sequence[dict[str, Any]], arm: str, slots: Sequence[Slot] | None = None, *,
                 event: str | None = None, day: str | None = None) -> Prompt:
    """The single-shot prompt: posts in, one sitrep out. Pure function."""
    instruction = _arm_instruction(arm, slots)
    user = (f"{_header(len(posts), event, day, 'social media posts')}\n\n{instruction}\n\n"
            f"Posts:\n{render_posts(posts)}")
    return Prompt(system=SYSTEM, user=user)


def build_map_prompt(posts: Sequence[dict[str, Any]], arm: str, slots: Sequence[Slot] | None = None, *,
                     event: str | None = None, day: str | None = None, chunk: int = 1,
                     n_chunks: int = 1) -> Prompt:
    """Map step: digest one chunk of posts, preserving every reportable detail."""
    _check_arm(arm, slots)
    organise = ("Group related points together." if arm == "generic" else
                "Group the digest under these headings, omitting any with nothing to report:\n\n"
                f"{_slot_block(slots or [])}")
    user = (f"{_header(len(posts), event, day, 'social media posts')} "
            f"This is part {chunk} of {n_chunks}.\n\n"
            "Write a compact factual digest of these posts as bullet points. Keep every figure, "
            "place name, organisation, time and attribution exactly as stated. " + _NO_INVENT +
            f" Do not write the report yet.\n\n{organise}\n\nPosts:\n{render_posts(posts)}")
    return Prompt(system=SYSTEM, user=user)


def build_reduce_prompt(digests: Sequence[str], arm: str, slots: Sequence[Slot] | None = None, *,
                        event: str | None = None, day: str | None = None) -> Prompt:
    """Reduce step: compose the chunk digests into one sitrep."""
    instruction = _arm_instruction(arm, slots)
    body = "\n\n".join(f"Digest {i}:\n{d.strip()}" for i, d in enumerate(digests, 1))
    user = (f"{_header(len(digests), event, day, 'factual digests of social media posts')}\n\n"
            f"{instruction}\n\n{body}")
    return Prompt(system=SYSTEM, user=user)


# ---------------------------------------------------------------------------
# chunking
# ---------------------------------------------------------------------------
def posts_tokens(posts: Sequence[dict[str, Any]]) -> int:
    """Heuristic token count of the rendered posts block (the cost gate's estimator)."""
    return estimate_tokens(render_posts(posts))


def chunk_posts(posts: Sequence[dict[str, Any]],
                budget_tokens: int = DEFAULT_BUDGET_TOKENS) -> list[list[dict[str, Any]]]:
    """Greedily pack posts, in order, into chunks of <= `budget_tokens`. Deterministic.

    A single post larger than the budget gets a chunk of its own. The whole block fitting
    returns one chunk, so `len(result) > 1` is exactly the map-reduce trigger.
    """
    if budget_tokens <= 0:
        raise ValueError(f"budget_tokens must be >= 1, got {budget_tokens}")
    if not posts:
        return []
    if posts_tokens(posts) <= budget_tokens:
        return [list(posts)]
    chunks: list[list[dict[str, Any]]] = []
    cur: list[dict[str, Any]] = []
    cur_tokens = 0
    for p in posts:
        cost = estimate_tokens(render_posts([p])) + 1  # +1 for the joining newline
        if cur and cur_tokens + cost > budget_tokens:
            chunks.append(cur)
            cur, cur_tokens = [], 0
        cur.append(p)
        cur_tokens += cost
    if cur:
        chunks.append(cur)
    return chunks


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------
def _request(prompt: Prompt, model: str, *, tag: str, meta: dict[str, Any], max_tokens: int,
             effort: str | None) -> dict[str, Any]:
    """One `src.llm` request dict. `meta` carries ids only - never post text (rule 7)."""
    req = {"model": model, "system": prompt.system, "user": prompt.user, "max_tokens": max_tokens,
           "tag": tag, "meta": dict(meta)}
    if effort:
        req["effort"] = effort
    return req


def plan_jobs(event: str, day_files: Sequence[DayFile], *, arms: Sequence[str], models: Sequence[str],
              cap: int, seed: int, budget_tokens: int) -> list[Job]:
    """Sample each day once and fan it out over models x arms (deterministic order)."""
    jobs: list[Job] = []
    for df in day_files:
        posts = sample_posts(df.path, cap, seed)
        chunks = chunk_posts(posts, budget_tokens)
        for model in models:
            for arm in arms:
                jobs.append(Job(event=event, day=df.day, model=model, arm=arm, posts=posts,
                                chunks=chunks))
    return jobs


def stage1_requests(jobs: Sequence[Job], slots: Sequence[Slot] | None, *, max_tokens: int,
                    effort: str | None) -> tuple[list[dict[str, Any]], list[tuple[Job, int]]]:
    """Round-1 requests (one per short job, one per chunk of a long job) and their owners.

    `owners[i]` is `(job, chunk_no)` for `requests[i]`; `chunk_no == 0` means the request is
    the job's whole sitrep rather than a chunk digest.
    """
    requests: list[dict[str, Any]] = []
    owners: list[tuple[Job, int]] = []
    for j in jobs:
        meta = {"event": j.event, "day": j.day, "model": j.model, "arm": j.arm}
        if j.map_reduce:
            for i, chunk in enumerate(j.chunks, 1):
                p = build_map_prompt(chunk, j.arm, slots, event=j.event, day=j.day, chunk=i,
                                     n_chunks=len(j.chunks))
                requests.append(_request(p, j.model, tag=f"gen-map-{j.arm}",
                                         meta=dict(meta, stage="map", chunk=i),
                                         max_tokens=max_tokens, effort=effort))
                owners.append((j, i))
        else:
            p = build_prompt(j.posts, j.arm, slots, event=j.event, day=j.day)
            requests.append(_request(p, j.model, tag=f"gen-{j.arm}", meta=dict(meta, stage="single"),
                                     max_tokens=max_tokens, effort=effort))
            owners.append((j, 0))
    return requests, owners


def format_plan(jobs: Sequence[Job], *, batch: bool, estimate_usd: float | None = None) -> str:
    """The human-readable table printed before anything is sent."""
    lines = [f"{'day':<12} {'model':<12} {'arm':<14} {'posts':>6} {'chunks':>7}  mode",
             "-" * 62]
    for j in jobs:
        mode = "map-reduce" if j.map_reduce else "single"
        lines.append(f"{j.day:<12} {j.model:<12} {j.arm:<14} {len(j.posts):>6} {len(j.chunks):>7}  {mode}")
    n_mr = sum(1 for j in jobs if j.map_reduce)
    tail = (f"{len(jobs)} sitrep(s) to generate ({n_mr} map-reduce), "
            f"{'Message Batches API' if batch else 'synchronous'}")
    if estimate_usd is not None:
        tail += (f", ~${estimate_usd:.2f} projected"
                 + (" for round 1 (the compose round is gated separately)" if n_mr else ""))
    return "\n".join(lines + ["-" * 62, tail])


def _run_stage(llm: LLM, requests: list[dict[str, Any]], *, batch: bool, progress: bool,
               stage: str) -> list[LLMResponse]:
    """Gate then dispatch one round of requests (batch for full runs, sync for smoke)."""
    if not requests:
        return []
    est = llm.gate(llm.estimate_cost(requests, batch=batch))
    log.info("stage %s: %d request(s), ~$%.3f projected (%s)", stage, len(requests), est.usd,
             "batch" if batch else "sync")
    return llm.run(requests, batch=batch, progress=progress)


def generate(llm: LLM, event: str, *, arms: Sequence[str] = ("generic",),
             models: Sequence[str] = ("api-strong",), batch: bool = False,
             min_posts: int = DEFAULT_MIN_POSTS, cap: int = DEFAULT_CAP, seed: int = 17,
             days: Sequence[str] | None = None, slots: Sequence[Slot] | None = None,
             budget_tokens: int = DEFAULT_BUDGET_TOKENS, max_tokens: int = DEFAULT_MAX_TOKENS,
             effort: str | None = None, stream_dir: str | Path | None = None,
             out_dir: str | Path | None = None, table_path: str | Path | None = None,
             progress: bool = True, dry_run: bool = False) -> dict[str, Any]:
    """Generate one sitrep per (event-day x model x arm) and write them to disk.

    Long days go map-reduce: round 1 digests each chunk, round 2 composes the digests.
    Every round is cost-gated before dispatch, and re-runs are free (identical prompts hit
    the `src.llm` disk cache).
    """
    for arm in arms:
        _check_arm(arm, slots)
    if not models:
        raise ValueError("no models given (config models.generation names, e.g. api-strong)")
    day_files = select_event_days(event, min_posts, stream_dir=stream_dir)
    if days is not None:
        wanted = set(days)
        day_files = [d for d in day_files if d.day in wanted]
    jobs = plan_jobs(event, day_files, arms=arms, models=models, cap=cap, seed=seed,
                     budget_tokens=budget_tokens)
    for j in jobs:
        j.model_id = llm.resolve_model(j.model).id
    summary: dict[str, Any] = {"event": event, "n_days": len(day_files), "n_jobs": len(jobs),
                               "batch": batch, "arms": list(arms), "models": list(models),
                               "plan": format_plan(jobs, batch=batch), "rows": [r.row() for r in jobs],
                               "written": [], "n_written": 0, "n_failed": 0, "table": None}
    if not jobs:
        log.warning("nothing to generate for %s (no event-day passed min_posts=%d)", event, min_posts)
        return summary

    # -- round 1: single-shot prompts + map prompts for the long days --------------
    stage1, owners = stage1_requests(jobs, slots, max_tokens=max_tokens, effort=effort)
    if dry_run:
        est = llm.estimate_cost(stage1, batch=batch)
        summary.update(estimate_usd=est.usd, estimate_over_limit=est.over_limit,
                       plan=format_plan(jobs, batch=batch, estimate_usd=est.usd))
        return summary
    responses = _run_stage(llm, stage1, batch=batch, progress=progress, stage="1 (single + map)")
    for (j, chunk_no), resp in zip(owners, responses):
        if not resp.ok:
            j.status = "refused" if resp.refused else "error"
            log.warning("%s %s: %s", j.stem, j.status, resp.error or resp.refusal_category)
            continue
        if chunk_no:
            j.digests.append(resp.text)
        else:
            j.text, j.request_hash, j.status = resp.text, resp.request_hash, "ok"

    # -- round 2: compose the digests ---------------------------------------------
    stage2: list[dict[str, Any]] = []
    reducers = [j for j in jobs if j.map_reduce and j.status == "planned" and j.digests]
    for j in reducers:
        p = build_reduce_prompt(j.digests, j.arm, slots, event=j.event, day=j.day)
        stage2.append(_request(p, j.model, tag=f"gen-reduce-{j.arm}",
                               meta={"event": j.event, "day": j.day, "model": j.model, "arm": j.arm,
                                     "stage": "reduce", "n_chunks": len(j.chunks)},
                               max_tokens=max_tokens, effort=effort))
    for j, resp in zip(reducers, _run_stage(llm, stage2, batch=batch, progress=progress,
                                            stage="2 (reduce)")):
        if resp.ok:
            j.text, j.request_hash, j.status = resp.text, resp.request_hash, "ok"
        else:
            j.status = "refused" if resp.refused else "error"
            log.warning("%s %s: %s", j.stem, j.status, resp.error or resp.refusal_category)

    # -- write ---------------------------------------------------------------------
    base = Path(out_dir) if out_dir else _default_processed_dir() / "sitreps_machine"
    ev_dir = ensure_dir(Path(base) / event)
    written: list[str] = []
    for j in jobs:
        if j.status != "ok":
            continue
        path = ev_dir / f"{j.stem}.json"
        write_json(path, {"event": j.event, "day": j.day, "model": j.model, "arm": j.arm,
                          "n_posts": len(j.posts), "text": j.text, "generated_at": utc_now_iso(),
                          "request_hash": j.request_hash})
        written.append(str(path))
    rows = [j.row() for j in jobs]
    write_json(ev_dir / "_manifest.json",
               {"event": event, "generated_at": utc_now_iso(), "run_id": llm.run_id, "seed": seed,
                "cap": cap, "min_posts": min_posts, "budget_tokens": budget_tokens,
                "max_tokens": max_tokens, "effort": effort, "batch": batch, "rows": rows})
    table = Path(table_path) if table_path else _default_table_path()
    write_table(rows, table)
    summary.update(rows=rows, written=written, n_written=len(written),
                   n_failed=sum(1 for j in jobs if j.status != "ok"), table=str(table),
                   out_dir=str(ev_dir), llm=llm.summary())
    log.info("%s: wrote %d/%d sitrep(s) to %s", event, len(written), len(jobs), ev_dir)
    return summary


def write_table(rows: Sequence[dict[str, Any]], path: str | Path) -> Path:
    """Write the run table (one row per generated sitrep) as CSV."""
    p = Path(path)
    ensure_dir(p.parent)
    with open(p, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(TABLE_COLUMNS))
        w.writeheader()
        w.writerows([{c: r.get(c) for c in TABLE_COLUMNS} for r in rows])
    return p


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _print_smoke(summary: dict[str, Any], limit_chars: int = 1200) -> None:
    """Print the generated sitreps so a human can eyeball them (smoke-first, rule 1)."""
    from src.util import read_json

    for path in summary["written"]:
        rec = read_json(path)
        text = rec["text"]
        clipped = text[:limit_chars] + ("\n... [clipped]" if len(text) > limit_chars else "")
        print(f"\n=== {rec['day']} | {rec['model']} | {rec['arm']} | {rec['n_posts']} posts ===")
        print(clipped)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description="Phase 3: generate machine sitreps from social streams")
    ap.add_argument("--event", required=True, help="event id, e.g. cyclone_idai_2019")
    ap.add_argument("--arms", nargs="+", default=["generic"], choices=list(ARMS),
                    help="prompt arms (default: generic)")
    ap.add_argument("--models", nargs="+", default=None,
                    help="config model names (default: every models.generation entry)")
    ap.add_argument("--smoke", type=int, default=0, metavar="N",
                    help="tiny synchronous run: first N event-days, sampling.smoke_posts per day")
    ap.add_argument("--batch", action="store_true", help="use the Message Batches API (full runs)")
    ap.add_argument("--days", nargs="+", default=None, help="restrict to these YYYY-MM-DD days")
    ap.add_argument("--min-posts", type=int, default=None,
                    help="volume filter (default sampling.min_posts_per_event_day)")
    ap.add_argument("--cap", type=int, default=None, help="posts per event-day (default from config)")
    ap.add_argument("--budget-tokens", type=int, default=None,
                    help="posts-block token budget above which map-reduce kicks in (default from config)")
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="sitrep length ceiling (default generation.max_tokens)")
    ap.add_argument("--effort", default=None, help="model effort (low|medium|high); default from config")
    ap.add_argument("--schema", type=Path, default=None, help="slot list for the schema_guided arm")
    ap.add_argument("--allow-candidate-schema", action="store_true",
                    help="use un-adjudicated candidate slots (skips the rule-4 human checkpoint)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and the projection, send nothing")
    args = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):  # pragma: no cover
        pass

    cfg = load_config()
    sampling = cfg.get("sampling") or {}
    gcfg = cfg.get("generation") or {}
    smoke = max(int(args.smoke), 0)
    cap = args.cap or int(sampling.get("smoke_posts", 20) if smoke else
                          sampling.get("posts_per_event_day", DEFAULT_CAP))
    min_posts = int(args.min_posts if args.min_posts is not None else
                    sampling.get("min_posts_per_event_day", DEFAULT_MIN_POSTS))
    budget_tokens = int(args.budget_tokens if args.budget_tokens is not None else
                        gcfg.get("map_reduce_token_budget", DEFAULT_BUDGET_TOKENS))
    max_tokens = int(args.max_tokens if args.max_tokens is not None else
                     gcfg.get("max_tokens", DEFAULT_MAX_TOKENS))
    effort = args.effort or gcfg.get("effort") or None
    models = args.models or [m["name"] for m in (cfg.get("models") or {}).get("generation") or []]
    batch = bool(args.batch) and not smoke
    if args.batch and smoke:
        log.warning("--smoke forces synchronous calls; ignoring --batch")

    slots: list[Slot] | None = None
    try:
        if "schema_guided" in args.arms or args.schema:
            slots = load_slots(args.schema, allow_candidates=args.allow_candidate_schema)
        llm = LLM(cfg)
        day_files = select_event_days(args.event, min_posts)
    except (SchemaNotAvailable, FileNotFoundError, ConfigError, ValueError) as exc:
        print(f"error: {exc}")
        return 2
    if args.days:
        day_files = [d for d in day_files if d.day in set(args.days)]
    if smoke:
        day_files = day_files[:smoke]
    if not day_files:
        print(f"error: no event-day of {args.event} passed min_posts={min_posts}")
        return 2

    jobs = plan_jobs(args.event, day_files, arms=args.arms, models=models, cap=cap, seed=int(cfg.get("seed", 17)),
                     budget_tokens=budget_tokens)
    for j in jobs:
        try:
            j.model_id = llm.resolve_model(j.model).id
        except ConfigError as exc:
            print(f"error: {exc}")
            return 2
    if args.dry_run:
        plan = generate(llm, args.event, arms=args.arms, models=models, batch=batch,
                        min_posts=min_posts, cap=cap, seed=int(cfg.get("seed", 17)),
                        days=[d.day for d in day_files], slots=slots, budget_tokens=budget_tokens,
                        max_tokens=max_tokens, effort=effort, dry_run=True)
        print(plan["plan"])
        if plan.get("estimate_over_limit"):
            print(f"WARNING: the projection exceeds llm.cost_limit_usd_per_run "
                  f"(${llm.cost_limit_usd:.2f}) — the run would be blocked by the cost gate.")
        return 0
    print(format_plan(jobs, batch=batch))

    try:
        summary = generate(llm, args.event, arms=args.arms, models=models, batch=batch,
                           min_posts=min_posts, cap=cap, seed=int(cfg.get("seed", 17)),
                           days=[d.day for d in day_files], slots=slots,
                           budget_tokens=budget_tokens, max_tokens=max_tokens,
                           effort=effort, progress=not smoke)
    except CostLimitExceeded as exc:
        print(f"COST LIMIT: {exc}")
        return 1
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}")
        return 2

    if smoke:
        _print_smoke(summary)
    s = summary["llm"]
    print(f"\nwrote {summary['n_written']}/{summary['n_jobs']} sitrep(s) to {summary['out_dir']}"
          f" ({summary['n_failed']} failed)")
    print(f"table:  {summary['table']}")
    print(f"spend:  ${s['cost_usd']:.4f} over {s['calls']} call(s) ({s['cached']} cached), "
          f"limit ${s['limit_usd']:.2f}")
    print("\nRESULTS.md row (paste after review):")
    print(f"| §Results | machine sitreps, {args.event} | {summary['n_written']} sitreps "
          f"({len(models)} models x {len(args.arms)} arms x {summary['n_days']} days) | "
          f"`python -m src.generate_sitreps --event {args.event} --arms {' '.join(args.arms)}"
          f"{' --batch' if batch else ''}` | seed {cfg.get('seed', 17)}, cap {cap}, "
          f"min_posts {min_posts} | {utc_now_iso()[:10]} |")
    return 0 if summary["n_failed"] == 0 else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
