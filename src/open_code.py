"""Phase 2a — LLM open coding of sampled human situation reports (PLAN §Phase 2).

This is the scholarly heart of the paper: a qualitative content analysis, run as an
open-coding pass over a stratified sample of professional sitreps. For each report the
coder model lists every distinct *information type* it carries, each with a short
verbatim evidence quote and a location hint. `src/consolidate.py` (Phase 2b) clusters
those codes into the candidate slot list the user adjudicates.

Audit trail (CLAUDE.md rule 4) — everything lands under ``data/processed/schema/``:

* ``induction_sample.json``  — the sampled sitrep ids, the strata, the seed (no text).
* ``open_codes/<sitrep_id>.json`` — the per-sitrep codes, **including evidence quotes**.

Evidence quotes are INTERNAL ONLY: they stay in ``data/``, are never written to the
call log (`src/llm.py` logs lengths, never text), and never appear verbatim in the
paper — the paper paraphrases or uses synthetic examples.

CLI::

    python -m src.open_code --event cyclone_idai_2019 --smoke 2      # eyeball first
    python -m src.open_code --event cyclone_idai_2019 --n 60 --batch # full run
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterator, Sequence

from src.llm import LLM, ConfigError, CostLimitExceeded, LLMResponse
from src.util import (resolve_event_dir, cfg_path, ensure_dir, get_logger, load_config, read_json, sha256_of,
                      slugify, utc_now_iso, write_json)

log = get_logger("sitrep.open_code")

PHASES: tuple[str, ...] = ("early", "mid", "late")
CONFIDENCES: frozenset[str] = frozenset({"high", "medium", "low"})
DEFAULT_MAX_CHARS = 30_000          # ~8.5k tokens of report text per prompt
DEFAULT_MAX_TOKENS = 6_000          # room for ~20 codes with definitions + quotes
MAX_QUOTE_CHARS = 400
TAG = "open-code-v1"

# -- Prompt + structured-output schema --------------------------------------
SYSTEM_PROMPT = """You are a qualitative content analyst coding professional humanitarian \
situation reports (OCHA, IFRC and UN agency products) for a research study.

Your task is OPEN CODING, the first pass of a grounded content analysis: read ONE report and \
list every distinct type of information it carries. Code what this report actually contains. Do \
not import a checklist of what you think a sitrep ought to contain, and never invent a code for \
information that is absent.

For each code:
- name: a short kebab-case noun phrase, e.g. casualty-figures, access-constraints, \
funding-appeal-status, needs-by-cluster-wash.
- definition: one sentence stating what this information type IS, generalised beyond this \
report so it could serve as a slot definition for other reports.
- evidence_quote: a VERBATIM span of at most 200 characters copied from the report that \
demonstrates the code. Copy exactly; never paraphrase or elide.
- location: where it appears — the section heading if the report has one (e.g. "Highlights", \
"Situation Overview", "Health", "Funding"), otherwise one of: header metadata, opening, body, \
closing, footer.
- confidence: high, medium or low — how sure you are that this is a distinct, well-evidenced \
information type in THIS report.

Guidance:
- One code per distinct information type. Split two things that serve different operational \
purposes; merge restatements of the same type.
- Typically 8-20 codes. Prefer completeness over brevity.
- Also code reporting CONVENTIONS the report uses: attribution phrases ("according to ..."), \
hedged or unconfirmed figures, "as of <date>" currency markers, revision language, \
figures-with-sources.
- Output only the JSON object described by the schema."""

CODE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "codes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "definition": {"type": "string"},
                    "evidence_quote": {"type": "string"},
                    "location": {"type": "string"},
                    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                },
                "required": ["name", "definition", "evidence_quote", "location", "confidence"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["codes"],
    "additionalProperties": False,
}


def build_user_prompt(record: dict[str, Any], *, max_chars: int = DEFAULT_MAX_CHARS) -> tuple[str, bool]:
    """Render the per-sitrep user prompt; returns (prompt, text_was_truncated).

    Long reports keep their head and tail (funding, contacts and gaps live at the end),
    with the omitted middle marked so the model knows text is missing.
    """
    text = str(record.get("text") or "")
    truncated = len(text) > max_chars
    if truncated:
        head = int(max_chars * 0.8)
        tail = max_chars - head
        omitted = len(text) - max_chars
        text = f"{text[:head]}\n\n[... {omitted} characters omitted from the middle ...]\n\n{text[-tail:]}"
    header = "\n".join(
        f"{label}: {record.get(key) or 'unknown'}"
        for label, key in (("Event", "event"), ("Source", "source"), ("Date", "date"),
                           ("Response phase", "phase"), ("Title", "title"))
    )
    prompt = (
        "Open-code the situation report below.\n\n"
        f"<report_metadata>\n{header}\n</report_metadata>\n\n"
        f"<report_text>\n{text}\n</report_text>\n\n"
        "List every distinct information type present, following the schema."
    )
    return prompt, truncated


# -- Corpus loading + phase assignment --------------------------------------
def _iter_sitrep_files(root: Path) -> Iterator[Path]:
    """Yield sitrep JSON files under an event directory OR the corpus root."""
    if not root.exists() or not root.is_dir():
        return
    yield from (p for p in sorted(root.glob("*.json")) if not p.name.startswith("_"))
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        yield from (p for p in sorted(sub.glob("*.json")) if not p.name.startswith("_"))


def _parse_date(value: Any) -> date | None:
    try:
        return datetime.strptime(str(value), "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return None


def assign_phases(records: Sequence[dict[str, Any]]) -> None:
    """Add a `phase` field (early/mid/late thirds of that event's date range).

    Undated records get phase "unknown" so they still form their own stratum rather
    than silently disappearing from the sample.
    """
    by_event: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_event[str(r.get("event") or "")].append(r)
    for rows in by_event.values():
        dates = [d for d in (_parse_date(r.get("date")) for r in rows) if d is not None]
        lo, hi = (min(dates), max(dates)) if dates else (None, None)
        span = ((hi - lo).days + 1) if (lo and hi) else 1
        for r in rows:
            d = _parse_date(r.get("date"))
            if d is None or lo is None:
                r["phase"] = "unknown"
            else:
                r["phase"] = PHASES[min(2, ((d - lo).days * 3) // span)]


def load_sitreps(event_dir_or_root: str | Path) -> list[dict[str, Any]]:
    """Load every human sitrep under a path, adding `event` and `phase`.

    Accepts either ``.../sitreps_human`` (all events) or ``.../sitreps_human/<event>``.
    Malformed files are warned about and skipped — never crash a whole run.
    """
    root = Path(event_dir_or_root)
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in _iter_sitrep_files(root):
        try:
            rec = read_json(path)
        except (OSError, ValueError) as exc:
            log.warning("skipping unreadable sitrep %s: %s", path.name, exc)
            continue
        if not isinstance(rec, dict) or not str(rec.get("text") or "").strip():
            log.warning("skipping %s: not a sitrep record with text", path.name)
            continue
        rec = dict(rec)
        rec.setdefault("id", path.stem)
        rec["event"] = rec.get("event") or path.parent.name
        rec["path"] = str(path)
        if rec["id"] in seen:
            log.warning("duplicate sitrep id %s (%s) — keeping the first", rec["id"], path.name)
            continue
        seen.add(str(rec["id"]))
        records.append(rec)
    assign_phases(records)
    return records


# -- Stratified sampling (deterministic under the config seed) --------------
def _sample_key(record: dict[str, Any]) -> tuple[str, str, str]:
    return (str(record.get("event") or ""), str(record.get("date") or ""), str(record.get("id") or ""))


def _stratum_order(groups: dict[tuple[str, ...], list[dict[str, Any]]]) -> list[tuple[str, ...]]:
    """Visit order for the round-robin: interleaved by the LEADING strata field.

    Strata are grouped by their first field (``event`` for a multi-event corpus, else
    ``source``), each group ordered biggest-first, and the groups are then interleaved.
    So every value of the leading field is reached before any of them is visited twice —
    a 4-sitrep sample over two events takes 2 from each rather than 4 from the larger.
    """
    by_lead: dict[str, list[tuple[str, ...]]] = defaultdict(list)
    for key in sorted(groups, key=lambda k: (-len(groups[k]), k)):
        by_lead[key[0]].append(key)
    leads = sorted(by_lead, key=lambda lead: (-sum(len(groups[k]) for k in by_lead[lead]), lead))
    order: list[tuple[str, ...]] = []
    for depth in range(max((len(v) for v in by_lead.values()), default=0)):
        order.extend(by_lead[lead][depth] for lead in leads if depth < len(by_lead[lead]))
    return order


def sample_sitreps(event_dir_or_root: str | Path, n: int, seed: int,
                   stratify_by: Sequence[str] = ("source", "phase"), *,
                   records: Sequence[dict[str, Any]] | None = None,
                   out_path: str | Path | None = None,
                   write_audit: bool = True) -> list[dict[str, Any]]:
    """Draw a deterministic stratified sample of sitreps and write the audit trail.

    Strata are the cross-product of `stratify_by` (``event`` is prepended automatically
    when the corpus spans several events, per PLAN §Phase 2). Allocation is **balanced
    round-robin** over strata rather than proportional: for schema induction the goal is
    to observe the full range of information types, so a rare source or a thin late-phase
    tail must be represented even when a single event-source dominates the corpus. Strata
    are visited interleaved by the leading field (see `_stratum_order`), so every event —
    then every source, then every phase — is reached before any is drawn from twice.

    Deterministic: within-stratum order is shuffled with ``Random(f"{seed}|{key}")``, so
    a stratum's draw does not depend on what else is in the corpus.

    Returns the sampled records (with text) and writes the sampled-id list to
    ``data/processed/schema/induction_sample.json`` unless `write_audit` is False.
    """
    recs = list(records) if records is not None else load_sitreps(event_dir_or_root)
    fields = tuple(stratify_by)
    if len({r.get("event") for r in recs}) > 1 and "event" not in fields:
        fields = ("event", *fields)

    groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for rec in sorted(recs, key=_sample_key):
        key = tuple(str(rec.get(f) or "unknown") for f in fields)
        groups.setdefault(key, []).append(rec)
    for key, items in groups.items():
        random.Random(f"{seed}|" + "|".join(key)).shuffle(items)

    order = _stratum_order(groups)
    picked: list[dict[str, Any]] = []
    depth = 0
    while len(picked) < n:
        progressed = False
        for key in order:
            if len(picked) >= n:
                break
            if depth < len(groups[key]):
                picked.append(groups[key][depth])
                progressed = True
        if not progressed:
            break
        depth += 1
    picked.sort(key=_sample_key)

    chosen = {id(r) for r in picked}
    audit = {
        "created_at": utc_now_iso(), "source_path": str(event_dir_or_root), "seed": seed,
        "n_requested": int(n), "n_available": len(recs), "n_sampled": len(picked),
        "stratify_by": list(fields), "allocation": "balanced round-robin over strata",
        "strata": [
            {"key": list(key), "n_available": len(items),
             "n_sampled": sum(1 for r in items if id(r) in chosen)}
            for key, items in sorted(groups.items())
        ],
        "sitrep_ids": [str(r["id"]) for r in picked],
        "sitreps": [
            {k: r.get(k) for k in ("id", "event", "source", "date", "phase", "title", "url")}
            | {"text_chars": len(str(r.get("text") or ""))}
            for r in picked
        ],
    }
    if write_audit:
        path = Path(out_path) if out_path else default_schema_dir() / "induction_sample.json"
        write_json(path, audit)
        log.info("sample: %d/%d sitreps over %d strata -> %s", len(picked), len(recs), len(groups), path)
    return picked


def default_schema_dir(cfg: dict[str, Any] | None = None) -> Path:
    """``data/processed/schema`` (config-driven)."""
    cfg = cfg if cfg is not None else load_config()
    return cfg_path(cfg, "processed", "data/processed") / "schema"


# -- Open coding ------------------------------------------------------------
def _safe_name(sitrep_id: str) -> str:
    slug = slugify(sitrep_id, max_len=90)
    return slug if slug == sitrep_id else f"{slug}-{sha256_of(sitrep_id)[:8]}"


def normalise_codes(payload: Any) -> tuple[list[dict[str, Any]], str | None]:
    """Validate a model reply against the open-coding shape.

    Returns (codes, error). Malformed replies are reported, never raised — one bad
    sitrep must not kill a 60-sitrep run.
    """
    if not isinstance(payload, dict):
        return [], f"expected a JSON object, got {type(payload).__name__}"
    raw = payload.get("codes")
    if not isinstance(raw, list):
        return [], "missing 'codes' array"
    codes: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        conf = str(item.get("confidence") or "").strip().lower()
        codes.append(
            {
                "name": slugify(name, max_len=60),
                "raw_name": name,
                "definition": str(item.get("definition") or "").strip(),
                "evidence_quote": str(item.get("evidence_quote") or "").strip()[:MAX_QUOTE_CHARS],
                "location": str(item.get("location") or "").strip() or "unspecified",
                "confidence": conf if conf in CONFIDENCES else "unknown",
            }
        )
    if not codes:
        return [], "no usable codes in reply"
    return codes, None


def _request_for(record: dict[str, Any], *, model: str, max_chars: int, max_tokens: int,
                 effort: str | None) -> tuple[dict[str, Any], bool]:
    prompt, truncated = build_user_prompt(record, max_chars=max_chars)
    req: dict[str, Any] = {
        "model": model,
        "system": SYSTEM_PROMPT,
        "user": prompt,
        "max_tokens": max_tokens,
        "json_schema": CODE_JSON_SCHEMA,
        "cache_system_prompt": True,
        "tag": TAG,
        # meta carries ids/labels ONLY — never prompt or report text (CLAUDE.md rule 7).
        "meta": {
            "stage": "open_code",
            "event": record.get("event"),
            "sitrep_id": record.get("id"),
            "source": record.get("source"),
            "phase": record.get("phase"),
        },
    }
    if effort:
        req["effort"] = effort
    return req, truncated


def _result_from_response(record: dict[str, Any], resp: LLMResponse, *, truncated: bool,
                          prompt_chars: int) -> dict[str, Any]:
    """One per-sitrep result record — the unit of the audit trail, written to disk as-is."""
    out: dict[str, Any] = {
        "sitrep_id": str(record.get("id") or ""), "event": record.get("event"),
        "source": record.get("source"), "date": record.get("date"), "phase": record.get("phase"),
        "title": record.get("title"), "url": record.get("url"), "tag": TAG,
        "text_chars": len(str(record.get("text") or "")), "prompt_chars": prompt_chars,
        "truncated_input": bool(truncated), "model_name": resp.model_name, "model_id": resp.model_id,
        "request_hash": resp.request_hash, "cached": bool(resp.cached), "coded_at": utc_now_iso(),
        "status": "ok", "codes": [], "n_codes": 0, "error": None,
    }
    codes: list[dict[str, Any]] = []
    err: str | None = None
    if resp.error:
        out["status"], err = "error", resp.error[:300]
    elif resp.refused:
        out["status"], err = "refused", f"refusal ({resp.refusal_category or 'unspecified'})"
    elif resp.json_error and resp.json is None:
        out["status"], err = "parse_error", f"json parse failed: {resp.json_error}"
    else:
        codes, err = normalise_codes(resp.json)
        out["status"] = "parse_error" if err else "ok"
    out["error"] = err
    out["codes"], out["n_codes"] = codes, len(codes)
    if out["status"] == "parse_error":
        # Deliberately NOT storing raw model output: it echoes sitrep text and this directory
        # is committed as the Phase 2 audit trail. Lengths are enough to debug a parse failure.
        out["raw_output_chars"] = len(resp.text or "")
    return out


def open_code_sitreps(llm: LLM, records: Sequence[dict[str, Any]], *, batch: bool,
                      model: str = "coder", out_dir: str | Path | None = None,
                      max_chars: int = DEFAULT_MAX_CHARS, max_tokens: int = DEFAULT_MAX_TOKENS,
                      effort: str | None = "medium", resume: bool = True, gate: bool = True,
                      progress: bool = True) -> list[dict[str, Any]]:
    """Open-code every record; returns one result dict per input record, in order.

    Results are written to ``<out_dir>/<sitrep_id>.json`` as they are produced. With
    `resume` (the default) a sitrep that already has a successful file on disk is not
    re-sent — a re-run costs nothing and makes no provider call.

    Refusals, JSON parse failures and per-item transport errors become results with a
    non-``ok`` status; they never abort the run. A `CostLimitExceeded` from the gate does
    propagate — that is the fail-closed spend guard and needs a human decision.
    """
    out_path = Path(out_dir) if out_dir is not None else default_schema_dir() / "open_codes"
    ensure_dir(out_path)
    results: list[dict[str, Any] | None] = [None] * len(records)
    pending: list[tuple[int, dict[str, Any], dict[str, Any], bool]] = []

    for i, rec in enumerate(records):
        target = out_path / f"{_safe_name(str(rec.get('id') or ''))}.json"
        if resume and target.exists():
            try:
                prior = read_json(target)
            except (OSError, ValueError):
                prior = None
            if isinstance(prior, dict) and prior.get("status") == "ok":
                results[i] = dict(prior, resumed=True)
                continue
        req, truncated = _request_for(rec, model=model, max_chars=max_chars,
                                      max_tokens=max_tokens, effort=effort)
        pending.append((i, rec, req, truncated))

    if pending:
        requests = [p[2] for p in pending]
        est = llm.estimate_cost(requests, batch=batch)
        log.info("open coding %d sitrep(s) (%d resumed): ~$%.2f projected (%s, batch=%s)",
                 len(pending), len(records) - len(pending), est.usd, est.method, batch)
        if gate:
            llm.gate(est)
        responses = llm.run(requests, batch=batch, progress=progress)
        for (i, rec, req, truncated), resp in zip(pending, responses):
            result = _result_from_response(rec, resp, truncated=truncated,
                                           prompt_chars=len(req["user"]))
            result["resumed"] = False
            write_json(out_path / f"{_safe_name(result['sitrep_id'])}.json", result)
            results[i] = result
            if result["status"] != "ok":
                log.warning("sitrep %s -> %s (%s)", result["sitrep_id"], result["status"], result["error"])

    return [r for r in results if r is not None]


# -- Reporting --------------------------------------------------------------
SUMMARY_COLUMNS = ["sitrep_id", "event", "source", "date", "phase", "status", "n_codes",
                   "model_id", "cached", "resumed", "text_chars", "truncated_input"]


def write_summary_table(results: Sequence[dict[str, Any]], path: str | Path) -> Path:
    """Write ``results/tables/open_codes_summary.csv`` (one row per sitrep, no text)."""
    p = Path(path)
    ensure_dir(p.parent)
    with open(p, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for r in results:
            writer.writerow({c: r.get(c, "") for c in SUMMARY_COLUMNS})
    return p


def print_report(results: Sequence[dict[str, Any]], *, show_quotes: bool = False,
                 top_codes: int = 15) -> None:
    """Per-sitrep code-count table + a code-name frequency preview (no quotes by default)."""
    print(f"\n{'sitrep':<34} {'src':<8} {'date':<10} {'phase':<7} {'status':<11} codes")
    print("-" * 80)
    for r in results:
        print(f"{str(r.get('sitrep_id'))[:34]:<34} {str(r.get('source') or '?')[:8]:<8} "
              f"{str(r.get('date') or '?'):<10} {str(r.get('phase') or '?'):<7} "
              f"{str(r.get('status')):<11} {r.get('n_codes', 0)}")
    counts = Counter(r.get("status") for r in results)
    total_codes = sum(int(r.get("n_codes") or 0) for r in results)
    print(f"\n{len(results)} sitrep(s): " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items(), key=str)))
    print(f"{total_codes} codes total"
          + (f", {total_codes / max(1, counts.get('ok', 0)):.1f} per successfully coded sitrep" if counts.get("ok") else ""))

    freq = Counter(c["name"] for r in results for c in r.get("codes") or [])
    if freq:
        print(f"\nmost frequent code names (of {len(freq)} distinct):")
        for name, k in freq.most_common(top_codes):
            print(f"  {k:>3}x  {name}")
    for r in results:
        if r.get("status") == "ok" and r.get("codes"):
            print(f"\nexample — {r['sitrep_id']} ({r.get('source')}, {r.get('date')}):")
            for c in r["codes"][:8]:
                line = f"  - {c['name']}  [{c['location']}, {c['confidence']}]"
                if show_quotes:
                    line += f'\n      "{c["evidence_quote"][:100]}"'
                print(line)
            break


# -- CLI --------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Phase 2a: LLM open coding of human sitreps (PLAN §Phase 2)")
    ap.add_argument("--event", default=None, help="event id, e.g. cyclone_idai_2019 (default: all events)")
    ap.add_argument("--n", type=int, default=None, help="sample size (default: schema.induction_sample)")
    ap.add_argument("--smoke", type=int, default=None, metavar="N",
                    help="tiny synchronous run over N sitreps; prints the codes for review")
    ap.add_argument("--batch", action="store_true", help="use the Message Batches API (50%% price; full runs)")
    ap.add_argument("--model", default="coder", help="config role or model id (default: coder)")
    ap.add_argument("--effort", default=None, help="reasoning effort (default: medium; low for --smoke)")
    ap.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS, help="report characters per prompt")
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="output tokens per sitrep")
    ap.add_argument("--stratify-by", nargs="*", default=["source", "phase"], help="strata fields")
    ap.add_argument("--seed", type=int, default=None, help="override config seed")
    ap.add_argument("--force", action="store_true", help="re-code sitreps that already have results")
    ap.add_argument("--dry-run", action="store_true", help="sample + project cost, send nothing")
    ap.add_argument("--show-quotes", action="store_true",
                    help="print evidence quotes (internal only — never paste into the paper)")
    ap.add_argument("--config", default=None, help="config.yaml path")
    ap.add_argument("--sitrep-dir", type=Path, default=None, help="override data/processed/sitreps_human")
    ap.add_argument("--schema-dir", type=Path, default=None, help="override data/processed/schema")
    ap.add_argument("--tables-dir", type=Path, default=None, help="override results/tables")
    return ap


def _no_corpus_message(searched: Path, root: Path) -> str:
    return (
        f"No human sitreps found under {searched}.\n"
        "Phase 2 needs the ReliefWeb side of the corpus first (PLAN §2a). Next steps:\n"
        "  1. pick reports from docs/worklists/<event>.md (or docs/idai_worklist.md),\n"
        "  2. save the pages by hand per docs/manual_collection.md,\n"
        "  3. run: .venv/Scripts/python.exe -m src.ingest_manual_reliefweb --scan <folder>\n"
        f"     which writes {root.as_posix()}/<event>/<date>.json."
    )


def main(argv: list[str] | None = None, *, llm: LLM | None = None) -> int:
    """CLI entry point. 0 = ok, 2 = nothing to do / bad input, 1 = run aborted."""
    args = _build_parser().parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):  # pragma: no cover
        pass

    cfg = load_config(args.config)
    processed = cfg_path(cfg, "processed", "data/processed")
    sitrep_root = Path(args.sitrep_dir) if args.sitrep_dir else processed / "sitreps_human"
    sitrep_dir = ((resolve_event_dir(sitrep_root, args.event) or sitrep_root / args.event)
                  if args.event else sitrep_root)
    schema_dir = Path(args.schema_dir) if args.schema_dir else processed / "schema"
    tables_dir = Path(args.tables_dir) if args.tables_dir else cfg_path(cfg, "results", "results") / "tables"
    seed = args.seed if args.seed is not None else int(cfg.get("seed", 17))
    smoke = args.smoke is not None
    n = args.smoke if smoke else (args.n or int((cfg.get("schema") or {}).get("induction_sample", 60)))
    batch = bool(args.batch) and not smoke
    effort = args.effort or ("low" if smoke else "medium")

    records = load_sitreps(sitrep_dir)
    if not records:
        print(_no_corpus_message(sitrep_dir, sitrep_root))
        return 2
    sample = sample_sitreps(sitrep_dir, n, seed, tuple(args.stratify_by), records=records,
                            out_path=schema_dir / "induction_sample.json")
    if not sample:
        print(f"Sampling returned 0 of {len(records)} sitreps — check --n and --stratify-by.")
        return 2
    print(f"{'SMOKE' if smoke else 'FULL'} run: {len(sample)} of {len(records)} sitrep(s) from {sitrep_dir}")
    print(f"  seed={seed}  strata={args.stratify_by}  model={args.model}  batch={batch}  effort={effort}")
    print(f"  audit trail: {schema_dir / 'induction_sample.json'}")

    if llm is None:
        try:
            llm = LLM(cfg)
        except ConfigError as exc:
            print(f"LLM config error: {exc}")
            return 2
    try:
        requests = [_request_for(r, model=args.model, max_chars=args.max_chars,
                                 max_tokens=args.max_tokens, effort=effort)[0] for r in sample]
        est = llm.estimate_cost(requests, batch=batch)
        print(f"  projected cost: ${est.usd:.2f} for {est.n_requests} request(s) "
              f"({est.n_cached} cached, ~{est.input_tokens} in / {est.output_tokens} out, {est.method})")
        if args.dry_run:
            print("--dry-run: nothing sent.")
            return 0
        results = open_code_sitreps(llm, sample, batch=batch, model=args.model,
                                    out_dir=schema_dir / "open_codes", max_chars=args.max_chars,
                                    max_tokens=args.max_tokens, effort=effort,
                                    resume=not args.force, progress=not smoke)
    except CostLimitExceeded as exc:
        print(f"COST LIMIT: {exc}\nRaise llm.cost_limit_usd_per_run only after the user approves the spend.")
        return 1
    except (ConfigError, TimeoutError) as exc:
        print(f"run aborted: {type(exc).__name__}: {exc}")
        return 1

    print_report(results, show_quotes=args.show_quotes)
    table = write_summary_table(results, tables_dir / "open_codes_summary.csv")
    print(f"\ncodes:   {schema_dir / 'open_codes'}  (evidence quotes are INTERNAL — never paste into the paper)")
    print(f"table:   {table}")
    print(f"llm:     {json.dumps(llm.summary(), sort_keys=True)}")
    ok = sum(1 for r in results if r.get("status") == "ok")
    if smoke:
        print(f"\nSmoke run complete — eyeball the {ok} coded sitrep(s) above, then run the full "
              f"sample:\n  .venv/Scripts/python.exe -m src.open_code "
              f"--event {args.event or 'ALL'} --n {args.n or 60} --batch")
    else:
        # Provenance row for the user to paste into RESULTS.md (this script never edits it).
        print(f"\nRESULTS.md row | §Methods/schema | open-coded {ok}/{len(results)} sitreps, "
              f"{sum(int(r.get('n_codes') or 0) for r in results)} codes | "
              f"`python -m src.open_code --event {args.event or 'ALL'} --n {n}"
              f"{' --batch' if batch else ''}` | seed {seed}, model {args.model} | {utc_now_iso()[:10]}")
    return 0 if ok else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
