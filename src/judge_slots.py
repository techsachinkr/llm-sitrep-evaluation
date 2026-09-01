"""Phase 4 slot scoring: a judge model decides {absent, partial, present} per (sitrep, slot).

Runs over BOTH machine sitreps (`data/processed/sitreps_machine/<event>/*.json`) and held-out
human sitreps (`data/processed/sitreps_human/<event>/*.json`). Human sitreps do double duty:
they are the achievable ceiling for the completeness metric and the judge validation set.

One LLM call per (sitrep, slot) with a strict JSON schema `{verdict, evidence, confidence}`.
The system prompt is the *whole* slot catalogue and is identical across every call of a run, so
it is sent with `cache_system_prompt=True` (hundreds of calls -> real prompt-cache savings).
Full runs go through the Message Batches API (`--batch`, 50% price); `--smoke N` is synchronous
and prints the judgements for a human to eyeball first (CLAUDE.md rule 1).

Outputs
  data/processed/judgements/<event>/<sitrep_id>.json   full records incl. evidence spans
  results/tables/slot_judgements.csv                   flat table, **no evidence text**
  results/tables/completeness_by_sitrep.csv            per-sitrep operational completeness
  data/processed/judgements/validation_sample.csv      --sample-for-validation N (blind)
  results/tables/judge_agreement.csv                   --agreement <hand-scored csv>

`results/` is committed, so evidence spans (which may quote a machine sitrep derived from
social-media posts) stay under `data/processed/judgements/` and only their length + hash reach
the CSV (CLAUDE.md rule 7).

CLI:
  python -m src.judge_slots --event cyclone_idai_2019 --smoke 5
  python -m src.judge_slots --event cyclone_idai_2019 --batch
  python -m src.judge_slots --event cyclone_idai_2019 --sample-for-validation 100
  python -m src.judge_slots --event cyclone_idai_2019 --agreement data/processed/judgements/validation_sample.csv
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml

from src.llm import LLM, ConfigError, CostLimitExceeded, LLMResponse
from src.metrics import (VERDICTS, Agreement, bootstrap_ci, judge_agreement, normalise_verdict,
                         operational_completeness, verdict_score)
from src.util import (
    resolve_event_dir,cfg_path, ensure_dir, get_logger, load_config, read_json, sha256_of,
                      utc_now_iso, write_json)

log = get_logger("sitrep.judge")

TAG = "judge-slots-v1"
KINDS = ("human", "machine")
DEFAULT_MAX_CHARS = 20_000        # sitrep characters sent per call (~6k tokens)
DEFAULT_MAX_TOKENS = 400          # verdict + short evidence span
AGREEMENT_TARGET = 0.7            # PLAN Phase 4 acceptance criterion

JUDGE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": list(VERDICTS)},
        "evidence": {"type": "string",
                     "description": "verbatim span from the report (<=200 chars); empty when absent"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["verdict", "evidence", "confidence"],
    "additionalProperties": False,
}


class MissingInput(RuntimeError):
    """A Phase 4 prerequisite (schema, sitreps, judgements) is not on disk yet."""


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Slot:
    """One information slot of the induced schema (Phase 2, `data/processed/schema.yaml`)."""

    id: str
    name: str
    definition: str
    examples: tuple[str, ...] = ()
    prevalence: float | None = None


@dataclass(frozen=True)
class Sitrep:
    """A sitrep to be judged (human-authored or machine-generated)."""

    id: str
    kind: str          # 'human' | 'machine'
    event: str
    text: str
    date: str = ""
    title: str = ""
    source: str = ""
    model: str = ""    # machine only
    arm: str = ""      # machine only ('generic' | 'schema_guided')
    path: str = ""

    @property
    def arm_key(self) -> str:
        """Model/arm label used as the ceiling analysis's 'arm' dimension."""
        if self.kind == "human":
            return "human"
        return "/".join(p for p in (self.model, self.arm) if p) or "machine"


def load_schema(path: Path) -> list[Slot]:
    """Load the frozen slot schema (`slots:` list, or a bare list) from `schema.yaml`."""
    if not path.exists():
        raise MissingInput(
            f"no schema at {path} - Phase 2 (schema induction) has not been frozen yet; "
            f"run the Phase 2 pipeline and have the user adjudicate the slot list first")
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    entries = raw.get("slots") if isinstance(raw, dict) else raw
    if not entries:
        raise MissingInput(f"schema at {path} defines no slots")
    slots: list[Slot] = []
    seen: set[str] = set()
    for i, e in enumerate(entries):
        # 'slot_id' is what Phase 2 (`src/consolidate.py`) writes; 'id' is accepted too.
        if not isinstance(e, dict) or not (e.get("id") or e.get("slot_id")):
            raise MissingInput(f"schema entry {i} in {path} has no 'id'/'slot_id'")
        sid = str(e.get("id") or e.get("slot_id")).strip()
        if sid in seen:
            raise MissingInput(f"duplicate slot id {sid!r} in {path}")
        seen.add(sid)
        slots.append(Slot(
            id=sid,
            name=str(e.get("name", sid)).strip(),
            definition=str(e.get("definition", "")).strip(),
            examples=tuple(str(x).strip() for x in (e.get("examples") or [])),
            prevalence=float(e["prevalence"]) if e.get("prevalence") is not None else None,
        ))
    log.info("loaded %d slots from %s", len(slots), path)
    return slots


def slot_weights(slots: Sequence[Slot]) -> dict[str, float] | None:
    """Prevalence weights for `operational_completeness`, or None when the schema has none."""
    weights = {s.id: s.prevalence for s in slots if s.prevalence is not None}
    return {k: float(v) for k, v in weights.items()} if len(weights) == len(slots) else None


def load_sitreps(data_root: Path, event: str, kind: str) -> list[Sitrep]:
    """Read `<data_root>/sitreps_<kind>/<event>/*.json` into `Sitrep`s (tolerant of extra keys)."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    folder = resolve_event_dir(data_root / f"sitreps_{kind}", event)
    if folder is None:
        log.warning("no %s sitreps for %s (looked for %s with '-' and '_' spellings)",
                    kind, event, data_root / f"sitreps_{kind}" / event)
        return []
    out: list[Sitrep] = []
    for path in sorted(folder.glob("*.json")):
        if path.name.startswith("_"):
            continue  # bookkeeping, e.g. the generator's _manifest.json
        try:
            rec = read_json(path)
        except (ValueError, OSError) as exc:
            log.warning("skipping unreadable sitrep %s: %s", path.name, exc)
            continue
        if not isinstance(rec, dict):
            log.warning("skipping %s: not a JSON object", path.name)
            continue
        text = str(rec.get("text") or "").strip()
        if not text:
            log.warning("skipping %s: empty 'text'", path.name)
            continue
        out.append(Sitrep(
            id=str(rec.get("id") or path.stem),
            kind=kind,
            event=str(rec.get("event") or event),
            text=text,
            date=str(rec.get("date") or rec.get("day") or ""),   # 'day' is the generator's field
            title=str(rec.get("title") or ""),
            source=str(rec.get("source") or ""),
            model=str(rec.get("model") or rec.get("model_name") or ""),
            arm=str(rec.get("arm") or ""),
            path=path.as_posix(),
        ))
    log.info("loaded %d %s sitreps for %s", len(out), kind, event)
    return out


# ---------------------------------------------------------------------------
# Prompting
# ---------------------------------------------------------------------------
_SYSTEM_HEAD = """\
You are an evaluator for humanitarian situation reports (sitreps). A schema of information slots
was induced from professional OCHA/IFRC sitreps. For ONE slot named in the user message, decide
how well the report given in the user message fills that slot.

Verdicts:
- present: the report substantively fills the slot (concrete information a responder could act on;
  for figure-bearing slots that means figures with a source or an explicit as-of time).
- partial: the slot is touched but thin - vague, unsourced, undated, or only one aspect of it.
- absent: the report does not address the slot at all.

Rules:
- Judge only the slot you are asked about; other slots have their own calls.
- Judge only what the report says. Do not reward plausible-sounding prose with no content.
- evidence: copy at most 200 characters verbatim from the report to support the verdict; use an
  empty string when the verdict is absent.
- confidence: 0.0-1.0, your confidence in the verdict.
- Reply with JSON only: {"verdict": ..., "evidence": ..., "confidence": ...}

The full slot catalogue follows so you can tell neighbouring slots apart.
"""


def build_system(slots: Sequence[Slot]) -> str:
    """The constant system prompt: instructions + the whole slot catalogue (prompt-cacheable)."""
    parts = [_SYSTEM_HEAD, "\nSLOT CATALOGUE\n"]
    for s in slots:
        parts.append(f"- {s.id} | {s.name}: {s.definition}")
        for ex in s.examples[:2]:
            parts.append(f"    example: {ex}")
    return "\n".join(parts)


def build_user(sitrep: Sitrep, slot: Slot, max_chars: int = DEFAULT_MAX_CHARS) -> str:
    """The per-call user message: the report text plus the single slot to score."""
    text = sitrep.text
    if len(text) > max_chars:
        text = text[:max_chars] + "\n[...truncated...]"
    header = f"REPORT (id={sitrep.id}, kind={sitrep.kind}, date={sitrep.date or 'unknown'})"
    return (f"{header}\n--- BEGIN REPORT ---\n{text}\n--- END REPORT ---\n\n"
            f"Score exactly this slot:\n- {slot.id} | {slot.name}: {slot.definition}\n\n"
            f"Return JSON only.")


def select_pairs(sitreps: Sequence[Sitrep], slots: Sequence[Slot],
                 smoke: int | None = None) -> list[tuple[Sitrep, Slot]]:
    """All (sitrep, slot) pairs; for `--smoke N` a tiny slice spread over kinds and slots."""
    if smoke is None:
        return [(s, sl) for s in sitreps for sl in slots]
    if smoke < 1:
        raise ValueError("--smoke needs a positive sample size")
    first: list[Sitrep] = []
    for kind in KINDS:
        first.extend([s for s in sitreps if s.kind == kind][:1])
    first = first or list(sitreps[:1])
    pairs: list[tuple[Sitrep, Slot]] = []
    for slot in slots:                      # slot-major so a smoke run shows several slots
        for s in first:
            if len(pairs) >= smoke:
                return pairs
            pairs.append((s, slot))
    return pairs


def build_requests(pairs: Sequence[tuple[Sitrep, Slot]], slots: Sequence[Slot], *,
                   model: str = "judge", max_chars: int = DEFAULT_MAX_CHARS,
                   max_tokens: int = DEFAULT_MAX_TOKENS) -> list[dict[str, Any]]:
    """One `LLM` request dict per pair. `meta` carries ids/labels only - never text (rule 7)."""
    system = build_system(slots)
    long_ones = {s.id for s, _ in pairs if len(s.text) > max_chars}
    if long_ones:
        log.warning("%d sitrep(s) truncated to %d chars for judging: %s",
                    len(long_ones), max_chars, ", ".join(sorted(long_ones)[:5]))
    return [{
        "model": model,
        "system": system,
        "user": build_user(sitrep, slot, max_chars=max_chars),
        "max_tokens": max_tokens,
        "json_schema": JUDGE_JSON_SCHEMA,
        "effort": "low",
        "cache_system_prompt": True,
        "tag": TAG,
        "meta": {"event": sitrep.event, "sitrep_id": sitrep.id, "kind": sitrep.kind,
                 "slot_id": slot.id, "arm": sitrep.arm_key, "date": sitrep.date},
    } for sitrep, slot in pairs]


# ---------------------------------------------------------------------------
# Judging
# ---------------------------------------------------------------------------
@dataclass
class Judgement:
    """One (sitrep, slot) verdict plus the provenance needed to reproduce or audit it."""

    event: str
    sitrep_id: str
    kind: str
    slot_id: str
    slot_name: str
    verdict: str | None
    score: float | None
    confidence: float | None
    evidence: str
    status: str            # 'ok' | 'invalid' | 'refused' | 'error'
    note: str = ""
    model: str = ""
    arm: str = ""
    date: str = ""
    judge_model: str = ""
    request_hash: str = ""
    mode: str = ""
    cached: bool = False
    judged_at: str = field(default_factory=utc_now_iso)

    def as_record(self) -> dict[str, Any]:
        return asdict(self)

    def as_row(self) -> dict[str, Any]:
        """Flat CSV row for `results/` - evidence is reduced to a length and a hash prefix."""
        row = asdict(self)
        evidence = row.pop("evidence") or ""
        row["evidence_chars"] = len(evidence)
        row["evidence_sha8"] = sha256_of(evidence)[:8] if evidence else ""
        return row


def parse_response(resp: LLMResponse, sitrep: Sitrep, slot: Slot) -> Judgement:
    """Turn one `LLMResponse` into a `Judgement`, recording refusals/malformed output as data."""
    j = Judgement(event=sitrep.event, sitrep_id=sitrep.id, kind=sitrep.kind, slot_id=slot.id,
                  slot_name=slot.name, verdict=None, score=None, confidence=None, evidence="",
                  status="error", model=sitrep.model, arm=sitrep.arm_key, date=sitrep.date,
                  judge_model=resp.model_id or resp.model_name, request_hash=resp.request_hash,
                  mode=resp.mode, cached=resp.cached)
    if resp.error:
        j.note = resp.error[:300]
        return j
    if resp.refused:
        j.status = "refused"
        j.note = f"refusal ({resp.refusal_category or 'unspecified'})"
        return j
    payload = resp.json
    if not isinstance(payload, dict):
        j.status = "invalid"
        j.note = f"unparsable JSON ({resp.json_error or 'not an object'})"
        return j
    try:
        j.verdict = normalise_verdict(payload.get("verdict"))
    except ValueError as exc:
        j.status = "invalid"
        j.note = str(exc)[:300]
        return j
    j.score = verdict_score(j.verdict)
    j.evidence = str(payload.get("evidence") or "")[:400]
    try:
        conf = payload.get("confidence")
        j.confidence = None if conf is None else float(conf)
    except (TypeError, ValueError):
        j.confidence = None
        j.note = "unparsable confidence"
    j.status = "ok"
    return j


def usable_judgements(judgements: Iterable[Judgement | dict[str, Any]]) -> list[dict[str, Any]]:
    """Rows with a real verdict - what the metrics may be computed over."""
    rows = [j.as_record() if isinstance(j, Judgement) else dict(j) for j in judgements]
    return [r for r in rows if r.get("status") == "ok" and r.get("verdict")]


def judge_pairs(llm: LLM, pairs: Sequence[tuple[Sitrep, Slot]], slots: Sequence[Slot], *,
                batch: bool, model: str = "judge", max_chars: int = DEFAULT_MAX_CHARS,
                max_tokens: int = DEFAULT_MAX_TOKENS, gate: bool = True) -> list[Judgement]:
    """Cost-gate, then run every (sitrep, slot) call and parse the results."""
    requests = build_requests(pairs, slots, model=model, max_chars=max_chars, max_tokens=max_tokens)
    if gate:
        llm.gate(llm.estimate_cost(requests, batch=batch))
    responses = llm.run(requests, batch=batch)
    return [parse_response(resp, sitrep, slot) for resp, (sitrep, slot) in zip(responses, pairs)]


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------
def _write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str], *,
               merge_keys: Sequence[str] = ()) -> Path:
    """Write `rows` to `path`; with `merge_keys`, keep existing rows whose key is not re-judged.

    The study runs event by event, so the flat tables accumulate across runs instead of the
    latest event silently replacing the previous one. Re-judging the same (event, sitrep, slot)
    overwrites its row.
    """
    ensure_dir(path.parent)
    out = list(rows)
    if merge_keys and path.exists():
        def key(row: dict[str, Any]) -> tuple[str, ...]:
            return tuple(str(row.get(k, "")) for k in merge_keys)

        fresh = {key(r) for r in out}
        with open(path, "r", encoding="utf-8", newline="") as fh:
            old = [r for r in csv.DictReader(fh) if key(r) not in fresh]
        out = sorted(old + out, key=key)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fieldnames), extrasaction="ignore", restval="")
        writer.writeheader()
        writer.writerows(out)
    return path


def write_judgements(judgements: Sequence[Judgement], out_root: Path, event: str,
                     sitreps: Sequence[Sitrep]) -> list[Path]:
    """One JSON file per sitrep under `<out_root>/<event>/` (evidence spans live here only)."""
    by_sitrep: dict[str, list[Judgement]] = {}
    for j in judgements:
        by_sitrep.setdefault(j.sitrep_id, []).append(j)
    index = {s.id: s for s in sitreps}
    written: list[Path] = []
    for sitrep_id, js in by_sitrep.items():
        s = index.get(sitrep_id)
        usable = usable_judgements(js)
        payload = {
            "sitrep_id": sitrep_id,
            "event": event,
            "kind": js[0].kind,
            "date": js[0].date,
            "model": js[0].model,
            "arm": js[0].arm,
            "source": s.source if s else "",
            "sitrep_path": s.path if s else "",
            "judge_model": js[0].judge_model,
            "judged_at": utc_now_iso(),
            "n_slots": len(js),
            "n_scored": len(usable),
            "completeness": operational_completeness(usable) if usable else None,
            "judgements": [j.as_record() for j in js],
        }
        written.append(write_json(out_root / event / f"{sitrep_id}.json", payload))
    return written


JUDGEMENT_COLUMNS = ("event", "sitrep_id", "kind", "model", "arm", "date", "slot_id", "slot_name",
                     "verdict", "score", "confidence", "status", "note", "evidence_chars",
                     "evidence_sha8", "judge_model", "request_hash", "mode", "cached", "judged_at")
COMPLETENESS_COLUMNS = ("event", "sitrep_id", "kind", "model", "arm", "date", "n_slots", "n_scored",
                        "n_failed", "completeness", "completeness_weighted")


def write_tables(judgements: Sequence[Judgement], tables_dir: Path,
                 weights: dict[str, float] | None = None, *, merge: bool = True) -> dict[str, Path]:
    """`slot_judgements.csv` (no evidence text) and `completeness_by_sitrep.csv`.

    Both tables merge with what is already on disk (keyed on event/sitrep/slot) so judging a
    second event adds to them instead of replacing the first; `merge=False` starts fresh.
    """
    paths = {"slot_judgements": _write_csv(
        tables_dir / "slot_judgements.csv", [j.as_row() for j in judgements], JUDGEMENT_COLUMNS,
        merge_keys=("event", "sitrep_id", "slot_id") if merge else ())}
    by_sitrep: dict[str, list[Judgement]] = {}
    for j in judgements:
        by_sitrep.setdefault(j.sitrep_id, []).append(j)
    rows: list[dict[str, Any]] = []
    for sitrep_id, js in by_sitrep.items():
        usable = usable_judgements(js)
        weighted = None
        if usable and weights:
            weighted = operational_completeness(usable, weights, default_weight=1.0)
        rows.append({"event": js[0].event, "sitrep_id": sitrep_id, "kind": js[0].kind,
                     "model": js[0].model, "arm": js[0].arm, "date": js[0].date,
                     "n_slots": len(js), "n_scored": len(usable), "n_failed": len(js) - len(usable),
                     "completeness": operational_completeness(usable) if usable else None,
                     "completeness_weighted": weighted})
    paths["completeness_by_sitrep"] = _write_csv(
        tables_dir / "completeness_by_sitrep.csv", rows, COMPLETENESS_COLUMNS,
        merge_keys=("event", "sitrep_id") if merge else ())
    return paths


VALIDATION_COLUMNS = ("event", "sitrep_id", "kind", "slot_id", "slot_name", "slot_definition",
                      "sitrep_path", "human_verdict", "notes")


def write_validation_sample(judgements: Sequence[Judgement], slots: Sequence[Slot], path: Path,
                            n: int, *, seed: int = 17,
                            sitreps: Sequence[Sitrep] = ()) -> Path:
    """Write a blind, slot-stratified hand-scoring sheet (`human_verdict` left blank).

    The judge's own verdict is deliberately **not** in the sheet: seeing it would anchor the
    human scorer and inflate the agreement number that validates the judge.
    """
    usable = [j for j in judgements if j.status == "ok"]
    if not usable:
        raise MissingInput("no usable judgements to sample for validation")
    defs = {s.id: (s.name, s.definition) for s in slots}
    paths = {s.id: s.path for s in sitreps}
    rng = random.Random(seed)
    strata: dict[tuple[str, str], list[Judgement]] = {}
    for j in usable:
        strata.setdefault((j.kind, j.slot_id), []).append(j)
    for items in strata.values():
        rng.shuffle(items)
    keys = sorted(strata, key=lambda k: (k[1], k[0]))   # slot-major: kinds alternate in small samples
    picked: list[Judgement] = []
    i = 0
    while len(picked) < min(n, len(usable)):
        bucket = strata[keys[i % len(keys)]]
        if bucket:
            picked.append(bucket.pop())
        i += 1
    picked.sort(key=lambda j: (j.kind, j.sitrep_id, j.slot_id))
    rows = [{"event": j.event, "sitrep_id": j.sitrep_id, "kind": j.kind, "slot_id": j.slot_id,
             "slot_name": defs.get(j.slot_id, (j.slot_name, ""))[0],
             "slot_definition": defs.get(j.slot_id, ("", ""))[1],
             "sitrep_path": paths.get(j.sitrep_id, ""), "human_verdict": "", "notes": ""}
            for j in picked]
    log.info("validation sample: %d rows over %d strata -> %s", len(rows), len(keys), path)
    return _write_csv(path, rows, VALIDATION_COLUMNS)


# ---------------------------------------------------------------------------
# Judge validation (agreement)
# ---------------------------------------------------------------------------
def load_judgement_index(out_root: Path, event: str) -> dict[tuple[str, str], str]:
    """`{(sitrep_id, slot_id): verdict}` from the stored per-sitrep judgement files."""
    folder = out_root / event
    if not folder.is_dir():
        raise MissingInput(f"no stored judgements at {folder}; run the judging pass first")
    index: dict[tuple[str, str], str] = {}
    for path in sorted(folder.glob("*.json")):
        rec = read_json(path)
        for j in rec.get("judgements", []):
            if j.get("status") == "ok" and j.get("verdict"):
                index[(str(rec.get("sitrep_id") or path.stem), str(j["slot_id"]))] = str(j["verdict"])
    if not index:
        raise MissingInput(f"no usable stored judgements under {folder}")
    return index


def compute_agreement(scored_csv: Path, index: dict[tuple[str, str], str]) -> tuple[Agreement, dict[str, int]]:
    """Join a hand-scored sheet to the stored judge verdicts -> agreement + kappa."""
    if not scored_csv.exists():
        raise MissingInput(f"hand-scored file {scored_csv} does not exist")
    counts = {"rows": 0, "blank": 0, "unmatched": 0, "used": 0}
    judge_labels: list[str] = []
    human_labels: list[str] = []
    with open(scored_csv, "r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        missing = {"sitrep_id", "slot_id", "human_verdict"} - set(reader.fieldnames or ())
        if missing:
            raise MissingInput(f"{scored_csv} is missing column(s): {sorted(missing)}")
        for row in reader:
            counts["rows"] += 1
            human = (row.get("human_verdict") or "").strip()
            if not human:
                counts["blank"] += 1
                continue
            key = (str(row["sitrep_id"]).strip(), str(row["slot_id"]).strip())
            if key not in index:
                counts["unmatched"] += 1
                log.warning("no stored judgement for %s / %s", *key)
                continue
            try:
                human_labels.append(normalise_verdict(human))
            except ValueError as exc:
                raise MissingInput(f"row {counts['rows']} of {scored_csv}: {exc}") from exc
            judge_labels.append(index[key])
            counts["used"] += 1
    if not judge_labels:
        raise MissingInput(f"{scored_csv} has no filled-in, matchable human_verdict rows")
    return judge_agreement(judge_labels, human_labels), counts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _print_smoke(judgements: Sequence[Judgement]) -> None:
    """Print smoke judgements for a human to eyeball (terminal only - never written to a log)."""
    print(f"\n--- {len(judgements)} smoke judgement(s) ---")
    for j in judgements:
        conf = "n/a" if j.confidence is None else f"{j.confidence:.2f}"
        print(f"[{j.kind}] {j.sitrep_id} | {j.slot_id}: {j.verdict or j.status.upper()} "
              f"(conf {conf}, {j.mode}{', cached' if j.cached else ''})")
        if j.evidence:
            print(f"    evidence: {j.evidence[:160]!r}")
        if j.note:
            print(f"    note: {j.note}")


def _summarise(judgements: Sequence[Judgement], seed: int) -> None:
    usable = usable_judgements(judgements)
    failed = [j for j in judgements if j.status != "ok"]
    print(f"\njudged {len(judgements)} (sitrep, slot) pairs: {len(usable)} scored, {len(failed)} not "
          f"({', '.join(sorted({j.status for j in failed})) or 'none'})")
    for kind in KINDS:
        rows = [r for r in usable if r["kind"] == kind]
        if not rows:
            continue
        per_sitrep: dict[str, list[dict[str, Any]]] = {}
        for r in rows:
            per_sitrep.setdefault(r["sitrep_id"], []).append(r)
        scores = [operational_completeness(v) for v in per_sitrep.values()]
        ci = bootstrap_ci(scores, seed=seed) if len(scores) > 1 else None
        detail = f" 95% CI [{ci.lo:.3f}, {ci.hi:.3f}]" if ci else ""
        print(f"  {kind}: operational completeness {sum(scores) / len(scores):.3f}{detail} "
              f"over {len(scores)} sitrep(s)")


def _judge_run(args: argparse.Namespace, cfg: dict[str, Any], data_root: Path, tables: Path) -> int:
    slots = load_schema(Path(args.schema) if args.schema else data_root / "schema.yaml")
    kinds = KINDS if args.kind == "both" else (args.kind,)
    sitreps = [s for k in kinds for s in load_sitreps(data_root, args.event, k)]
    if not sitreps:
        raise MissingInput(
            f"no sitreps for event {args.event!r} under {data_root}/sitreps_{{{','.join(kinds)}}}/ - "
            f"Phase 1 (human sitreps) / Phase 3 (machine sitreps) have produced nothing yet")
    pairs = select_pairs(sitreps, slots, smoke=args.smoke)
    log.info("judging %d (sitrep, slot) pairs (%d sitreps x %d slots, mode=%s)",
             len(pairs), len(sitreps), len(slots), "batch" if args.batch else "sync")
    llm = LLM(cfg)
    judgements = judge_pairs(llm, pairs, slots, batch=args.batch, model=args.model,
                             max_chars=args.max_chars, max_tokens=args.max_tokens)
    out_root = data_root / "judgements"
    written = write_judgements(judgements, out_root, args.event, sitreps)
    paths = write_tables(judgements, tables, weights=slot_weights(slots), merge=not args.fresh_tables)
    if args.smoke:
        _print_smoke(judgements)
    _summarise(judgements, int(cfg.get("seed", 17)))
    print(f"\nwrote {len(written)} judgement file(s) under {(out_root / args.event).as_posix()}")
    for name, path in paths.items():
        print(f"wrote {name}: {path.as_posix()}")
    print(f"llm: {llm.summary()}")
    return 0


def _validation_run(args: argparse.Namespace, cfg: dict[str, Any], data_root: Path) -> int:
    slots = load_schema(Path(args.schema) if args.schema else data_root / "schema.yaml")
    kinds = KINDS if args.kind == "both" else (args.kind,)
    sitreps = [s for k in kinds for s in load_sitreps(data_root, args.event, k)]
    out_root = data_root / "judgements"
    folder = out_root / args.event
    if not folder.is_dir():
        raise MissingInput(f"no stored judgements at {folder}; run the judging pass first")
    judgements: list[Judgement] = []
    for path in sorted(folder.glob("*.json")):
        rec = read_json(path)
        for j in rec.get("judgements", []):
            judgements.append(Judgement(**{k: v for k, v in j.items()
                                           if k in Judgement.__dataclass_fields__}))
    out = write_validation_sample(judgements, slots, out_root / "validation_sample.csv",
                                  args.sample_for_validation, seed=int(cfg.get("seed", 17)),
                                  sitreps=sitreps)
    print(f"wrote blind validation sheet: {out.as_posix()}\n"
          f"fill in the 'human_verdict' column ({'/'.join(VERDICTS)}) and re-run with "
          f"--agreement {out.as_posix()}")
    return 0


def _agreement_run(args: argparse.Namespace, data_root: Path, tables: Path) -> int:
    index = load_judgement_index(data_root / "judgements", args.event)
    agreement, counts = compute_agreement(Path(args.agreement), index)
    row = {"event": args.event, "source_file": Path(args.agreement).as_posix(),
           "rows_in_file": counts["rows"], "blank": counts["blank"], "unmatched": counts["unmatched"],
           **agreement.as_row()}
    path = _write_csv(tables / "judge_agreement.csv", [row], list(row))
    print(f"judge validation on {agreement.n} hand-scored pairs "
          f"({counts['blank']} blank, {counts['unmatched']} unmatched)")
    print(f"  raw agreement    {agreement.agreement:.3f}  (PLAN target >= {AGREEMENT_TARGET})")
    print(f"  Cohen's kappa    {agreement.kappa:.3f}   (linear-weighted {agreement.kappa_linear:.3f})")
    print(f"  confusion (rows = judge {agreement.labels}, cols = human): {agreement.confusion}")
    print(f"wrote {path.as_posix()}")
    if agreement.agreement < AGREEMENT_TARGET:
        log.warning("raw agreement %.3f is below the PLAN acceptance criterion %.2f",
                    agreement.agreement, AGREEMENT_TARGET)
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point (`python -m src.judge_slots --event <event> [--smoke N] [--batch]`)."""
    ap = argparse.ArgumentParser(prog="python -m src.judge_slots",
                                 description="Phase 4: judge each (sitrep, slot) pair")
    ap.add_argument("--event", required=True, help="event id, e.g. cyclone_idai_2019")
    ap.add_argument("--kind", choices=(*KINDS, "both"), default="both", help="sitreps to judge")
    ap.add_argument("--smoke", type=int, default=None, metavar="N",
                    help="judge only N (sitrep, slot) pairs synchronously and print them")
    ap.add_argument("--batch", action="store_true",
                    help="use the Message Batches API (50%% price) - the default for full runs")
    ap.add_argument("--model", default="judge", help="config role/name for the judge (default: judge)")
    ap.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS,
                    help=f"sitrep characters per call (default {DEFAULT_MAX_CHARS})")
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, metavar="N",
                    help=f"output budget per judgement (default {DEFAULT_MAX_TOKENS}). Reasoning "
                         "models spend this budget thinking before they emit anything, so they "
                         "need a larger cap or every response comes back empty.")
    ap.add_argument("--schema", default=None, help="slot schema (default <data-root>/schema.yaml)")
    ap.add_argument("--data-root", default=None, help="processed-data root (default paths.processed)")
    ap.add_argument("--tables-dir", default=None, help="output tables dir (default <results>/tables)")
    ap.add_argument("--fresh-tables", action="store_true",
                    help="overwrite the flat tables instead of merging them with earlier events")
    ap.add_argument("--sample-for-validation", type=int, default=None, metavar="N",
                    help="write a blind hand-scoring sheet of N stored judgements and exit")
    ap.add_argument("--agreement", default=None, metavar="CSV",
                    help="hand-scored sheet: compute raw agreement + Cohen's kappa and exit")
    args = ap.parse_args(argv)
    try:  # Windows consoles default to a legacy code page
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):  # pragma: no cover
        pass
    if args.smoke is not None and args.batch:
        ap.error("--smoke is synchronous by design; drop --batch")
    cfg = load_config()
    data_root = Path(args.data_root) if args.data_root else cfg_path(cfg, "processed", "data/processed")
    tables = Path(args.tables_dir) if args.tables_dir else cfg_path(cfg, "results", "results") / "tables"
    try:
        if args.agreement:
            return _agreement_run(args, data_root, tables)
        if args.sample_for_validation is not None:
            return _validation_run(args, cfg, data_root)
        return _judge_run(args, cfg, data_root, tables)
    except MissingInput as exc:
        log.error("cannot proceed: %s", exc)
        return 2
    except ConfigError as exc:
        log.error("config error: %s", exc)
        return 2
    except CostLimitExceeded as exc:
        log.error("cost gate: %s", exc)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
