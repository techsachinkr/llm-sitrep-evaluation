"""Phase 5 (PLAN §Phase 5) — reference metrics: the standard scores this paper puts on trial.

Machine sitreps are scored against the human sitrep(s) of the **same event-day** with the metrics
the summarisation literature reaches for by default:

* **ROUGE-L** (`rouge-score`) — lexical overlap, max over references.
* **BERTScore** (`bert-score`, CPU) — embedding similarity; the model is configurable and loaded
  lazily, so tests and `--smoke` never pull weights.
* **generic LLM judge** — a 1-10 quality score with **no mention of the induced schema**, so it
  stands in for the "ask a big model if it's good" evaluation everyone runs. Goes through
  `src.llm` (judge role), cost-gated, `complete_batch` for full runs.

Pairing rule (recorded per row, because the paper has to state it):
    `same_day`      the human sitrep(s) published on the machine sitrep's date
    `same_window`   nearest human sitreps within +/- `--window-days` when that day has none
    `event_window`  any human sitrep for the event (only with `--event-fallback`)
    `none`          no human sitrep available -> reference metrics are left blank, row still emitted

Output: `results/tables/reference_metrics.csv` — **ids and numbers only**. That table is committed,
so `assert_no_free_text` refuses to write anything that looks like sitrep or social-media prose
(CLAUDE.md rule 7). Judge justifications are model text and therefore never enter it; pass
`--save-justifications` to keep them under `data/` (gitignored) for the audit trail.

CLI:
    python -m src.reference_metrics --smoke 3                     # sync, tiny sample, prints rows
    python -m src.reference_metrics --event cyclone_idai_2019     # full run (batch LLM judge)
    python -m src.reference_metrics --metrics rouge_l,bertscore   # no LLM calls at all
"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, field
from datetime import date as _date, datetime
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from src.llm import LLM, CostLimitExceeded, LLMResponse
from src.util import (
    normalize_event,
    resolve_event_dir,
    cfg_path,
    ensure_dir,
    append_jsonl,
    get_logger,
    load_config,
    set_seed,
    utc_now_iso,
)

log = get_logger("sitrep.refmetrics")

# -- metric ids -------------------------------------------------------------------------------
ROUGE_L = "rouge_l"
BERTSCORE = "bertscore"
LLM_JUDGE = "generic_llm_judge"
ALL_METRICS = (ROUGE_L, BERTSCORE, LLM_JUDGE)

# -- pairing rules ----------------------------------------------------------------------------
PAIR_SAME_DAY = "same_day"
PAIR_SAME_WINDOW = "same_window"
PAIR_EVENT = "event_window"
PAIR_NONE = "none"

DEFAULT_WINDOW_DAYS = 3
DEFAULT_MAX_REFS = 3
DEFAULT_BERTSCORE_MODEL = "roberta-large"
TABLE_VERSION = "refmetrics-v1"

TABLE_COLUMNS = [
    "sitrep_id", "event", "date", "model", "arm", "pairing_rule", "n_refs", "ref_ids",
    "ref_date", "day_gap", "rouge_l_f", "rouge_l_p", "rouge_l_r",
    "bertscore_f1", "bertscore_p", "bertscore_r", "llm_judge_score", "judge_model", "version",
]

# The generic judge deliberately knows nothing about the induced schema (PLAN §Phase 5).
JUDGE_SYSTEM = (
    "You are an experienced humanitarian information officer who reviews situation reports. "
    "You rate the overall quality of a report on a 1-10 scale and reply with JSON only."
)
JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "score": {"type": "integer", "minimum": 1, "maximum": 10},
        "justification": {"type": "string"},
    },
    "required": ["score", "justification"],
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# corpus
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Sitrep:
    """One sitrep, human or machine. `text` never leaves this process in a written table."""

    id: str
    event: str
    date: str
    text: str
    model: str = ""
    arm: str = ""
    source: str = ""
    title: str = ""

    @property
    def day(self) -> _date | None:
        return parse_date(self.date)


def parse_date(value: Any) -> _date | None:
    """Parse an ISO-ish date ('2019-03-21', '2019-03-21T09:00:00+00:00') -> date, else None."""
    if isinstance(value, _date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if not isinstance(value, str) or len(value) < 10:
        return None
    try:
        return _date.fromisoformat(value[:10])
    except ValueError:
        return None


def _iter_records(path: Path) -> Iterator[dict[str, Any]]:
    """Yield dict records from a .json (object or list) or .jsonl file."""
    import json

    with open(path, "r", encoding="utf-8") as fh:
        if path.suffix == ".jsonl":
            for line in fh:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        yield obj
            return
        obj = json.load(fh)
    if isinstance(obj, dict):
        yield obj
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict):
                yield item


def load_sitreps(root: Path, event: str | None = None, *, kind: str = "human") -> list[Sitrep]:
    """Load `<root>/<event>/*.json[l]` into Sitreps (empty list if nothing is collected yet).

    Missing fields are recovered from the path: the event from the parent directory, the date from
    a leading YYYY-MM-DD in the file name, the id from the file stem. The id fallback matches
    `src.judge_slots.load_sitreps` exactly — `sitrep_id` is what Phase 4 and Phase 5 join on — and
    `day` is accepted as a date alias because that is what `src.generate_sitreps` writes.
    """
    if not root.exists():
        log.warning("%s sitrep directory does not exist: %s", kind, root)
        return []
    out: list[Sitrep] = []
    # '-' and '_' spellings of the same event both occur in this corpus; resolve rather
    # than trust the caller (see src.util.resolve_event_dir).
    event_dirs = ([resolve_event_dir(root, event) or root / event] if event
                  else sorted(p for p in root.iterdir() if p.is_dir()))
    for ev_dir in event_dirs:
        if not ev_dir.is_dir():
            log.warning("no %s sitreps for event %r under %s", kind, ev_dir.name, root)
            continue
        for path in sorted(list(ev_dir.glob("*.json")) + list(ev_dir.glob("*.jsonl"))):
            for i, rec in enumerate(_iter_records(path)):
                text = str(rec.get("text") or "").strip()
                if not text:
                    log.warning("skipping %s (record %d): empty text", path.name, i)
                    continue
                stem = path.stem if path.suffix == ".json" else f"{path.stem}-{i}"
                out.append(Sitrep(
                    id=str(rec.get("id") or stem),
                    event=str(rec.get("event") or ev_dir.name),
                    date=str(rec.get("date") or rec.get("day") or stem[:10]),
                    text=text,
                    model=str(rec.get("model") or ""),
                    arm=str(rec.get("arm") or ""),
                    source=str(rec.get("source") or ""),
                    title=str(rec.get("title") or ""),
                ))
    log.info("loaded %d %s sitrep(s) from %s", len(out), kind, root)
    return out


# ---------------------------------------------------------------------------
# pairing
# ---------------------------------------------------------------------------
@dataclass
class Pairing:
    """Which human sitreps a machine sitrep is scored against, and by which rule."""

    refs: list[Sitrep] = field(default_factory=list)
    rule: str = PAIR_NONE
    day_gap: int | None = None


def pair_references(machine: Sitrep, humans: Sequence[Sitrep], *,
                    window_days: int = DEFAULT_WINDOW_DAYS, max_refs: int = DEFAULT_MAX_REFS,
                    event_fallback: bool = False) -> Pairing:
    """Pick the human reference(s) for one machine sitrep, preferring the exact event-day.

    Order of rules: same day -> within +/- `window_days` (nearest first) -> whole event (only when
    `event_fallback`) -> none. Ties break on the earlier date so the choice is deterministic.
    """
    # Compare canonical keys: machine sitreps inherit the stream folder's underscored
    # spelling and human sitreps the ReliefWeb folder's hyphenated one, so a raw string
    # equality here silently paired nothing at all (all 102 sitreps -> rule 'none').
    m_event = normalize_event(machine.event)
    same_event = [h for h in humans if normalize_event(h.event) == m_event]
    if not same_event:
        return Pairing([], PAIR_NONE, None)
    m_day = machine.day
    if m_day is None:
        log.warning("machine sitrep %s has an unparseable date %r; falling back to event pairing",
                    machine.id, machine.date)
        if not event_fallback:
            return Pairing([], PAIR_NONE, None)
        refs = sorted(same_event, key=lambda h: (h.date, h.id))[:max_refs]
        return Pairing(refs, PAIR_EVENT, None)

    dated = [(h, (h.day - m_day).days) for h in same_event if h.day is not None]
    same_day = [h for h, d in dated if d == 0]
    if same_day:
        refs = sorted(same_day, key=lambda h: h.id)[:max_refs]
        return Pairing(refs, PAIR_SAME_DAY, 0)
    in_window = sorted((x for x in dated if abs(x[1]) <= window_days),
                       key=lambda x: (abs(x[1]), x[1], x[0].id))
    if in_window:
        refs = [h for h, _ in in_window[:max_refs]]
        return Pairing(refs, PAIR_SAME_WINDOW, in_window[0][1])
    if event_fallback and dated:
        ordered = sorted(dated, key=lambda x: (abs(x[1]), x[1], x[0].id))
        return Pairing([h for h, _ in ordered[:max_refs]], PAIR_EVENT, ordered[0][1])
    return Pairing([], PAIR_NONE, None)


# ---------------------------------------------------------------------------
# ROUGE-L
# ---------------------------------------------------------------------------
def load_rouge_scorer(use_stemmer: bool = True) -> Any | None:
    """Build a `rouge_score` ROUGE-L scorer, or None (logged) when the package is unavailable."""
    try:
        from rouge_score import rouge_scorer
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
        log.warning("ROUGE-L skipped: rouge-score is not installed (%s)", exc)
        return None
    return rouge_scorer.RougeScorer(["rougeL"], use_stemmer=use_stemmer)


def rouge_l(scorer: Any, candidate: str, references: Sequence[str]) -> dict[str, float] | None:
    """Best-over-references ROUGE-L (the standard multi-reference protocol), keyed p/r/f."""
    if scorer is None or not references or not candidate.strip():
        return None
    best: dict[str, float] | None = None
    for ref in references:
        s = scorer.score(ref, candidate)["rougeL"]
        cand = {"p": float(s.precision), "r": float(s.recall), "f": float(s.fmeasure)}
        if best is None or cand["f"] > best["f"]:
            best = cand
    return best


# ---------------------------------------------------------------------------
# BERTScore (lazy; never loaded by tests or --smoke)
# ---------------------------------------------------------------------------
def load_bertscorer(model_type: str = DEFAULT_BERTSCORE_MODEL, *, device: str = "cpu",
                    batch_size: int = 8) -> Any | None:
    """Instantiate a `bert_score.BERTScorer`, or None (logged) if the package/weights are missing.

    This is the only place that can trigger a model download, so it is called lazily and only when
    BERTScore is actually requested for a non-empty set of pairs.
    """
    try:
        from bert_score import BERTScorer
    except ImportError as exc:
        log.warning("BERTScore skipped: bert-score is not installed (%s)", exc)
        return None
    try:
        return BERTScorer(model_type=model_type, lang="en", device=device, batch_size=batch_size,
                          rescale_with_baseline=False)
    except Exception as exc:  # noqa: BLE001 - offline / no weights / OOM must not kill the run
        log.warning("BERTScore skipped: could not load %r (%s: %s)", model_type, type(exc).__name__, exc)
        return None


def bertscore_many(scorer: Any, candidates: Sequence[str],
                   references: Sequence[Sequence[str]]) -> list[dict[str, float] | None]:
    """Batched BERTScore; `references[i]` may hold several refs (bert-score takes the max)."""
    if scorer is None or not candidates:
        return [None] * len(candidates)
    idx = [i for i, refs in enumerate(references) if refs]
    if not idx:
        return [None] * len(candidates)
    try:
        p, r, f = scorer.score([candidates[i] for i in idx], [list(references[i]) for i in idx])
    except Exception as exc:  # noqa: BLE001
        log.warning("BERTScore skipped: scoring failed (%s: %s)", type(exc).__name__, exc)
        return [None] * len(candidates)
    out: list[dict[str, float] | None] = [None] * len(candidates)
    for j, i in enumerate(idx):
        out[i] = {"p": float(p[j]), "r": float(r[j]), "f": float(f[j])}
    return out


# ---------------------------------------------------------------------------
# generic LLM judge (no schema mention)
# ---------------------------------------------------------------------------
def judge_prompt(sitrep: Sitrep) -> str:
    """The generic quality prompt. Deliberately says nothing about slots or the induced schema."""
    return (
        f"Rate the overall quality of the situation report below. It concerns the event "
        f"'{sitrep.event}' on {sitrep.date or 'an unspecified date'}.\n\n"
        "Give a single overall score from 1 (very poor) to 10 (excellent), judging how useful, "
        "clear, well organised and credible it is as a situation report.\n\n"
        "SITUATION REPORT\n"
        "----------------\n"
        f"{sitrep.text}\n"
        "----------------\n\n"
        'Reply with JSON only: {"score": <integer 1-10>, "justification": "<one or two sentences>"}'
    )


def judge_requests(sitreps: Sequence[Sitrep], *, model: str = "judge", max_tokens: int = 512,
                   effort: str | None = "low") -> list[dict[str, Any]]:
    """Build `src.llm` request dicts. `meta` carries ids/labels only — never text (rule 7)."""
    return [{
        "model": model,
        "system": JUDGE_SYSTEM,
        "user": judge_prompt(s),
        "max_tokens": max_tokens,
        "json_schema": JUDGE_SCHEMA,
        "effort": effort,
        "tag": "refmetrics-generic-judge",
        "meta": {"sitrep_id": s.id, "event": s.event, "date": s.date, "model": s.model, "arm": s.arm},
    } for s in sitreps]


def parse_judge_score(resp: LLMResponse) -> tuple[float | None, str]:
    """Return (score, justification). Out-of-range or unparseable output -> (None, '')."""
    if resp is None or not resp.ok or not isinstance(resp.json, dict):
        return None, ""
    raw = resp.json.get("score")
    try:
        score = float(raw)
    except (TypeError, ValueError):
        return None, ""
    if not 1.0 <= score <= 10.0:
        log.warning("judge returned an out-of-range score %r for %s", raw, resp.meta.get("sitrep_id"))
        return None, ""
    return score, str(resp.json.get("justification") or "")


# ---------------------------------------------------------------------------
# the committed table: ids and numbers only
# ---------------------------------------------------------------------------
MAX_CELL_CHARS = 200
MAX_CELL_WORDS = 6
#: Columns holding a delimiter-joined list of record ids. These are legitimately long (reference
#: ids are verbose) but must contain NO whitespace — which is a strictly stronger guarantee than
#: the length/word limits, since prose cannot be whitespace-free.
ID_LIST_COLUMNS = frozenset({"ref_ids"})
ID_LIST_SEPARATOR = "|"


def assert_no_free_text(rows: Iterable[dict[str, Any]], columns: Sequence[str]) -> None:
    """Fail closed if a results table would carry prose (sitrep or social-media text).

    `results/tables/*.csv` is committed, so every cell must be an id, a label or a number
    (CLAUDE.md rule 7). Enforced structurally: unknown columns are rejected, and string cells
    must be short and word-poor.
    """
    allowed = set(columns)
    for row in rows:
        extra = set(row) - allowed
        if extra:
            raise ValueError(f"refusing to write unexpected column(s) {sorted(extra)}; "
                             f"results tables carry ids and numbers only")
        for col, val in row.items():
            if not isinstance(val, str):
                continue
            if col in ID_LIST_COLUMNS:
                # Length is fine here; whitespace is not. Any space means this is not an id list.
                if val and any(ch.isspace() for ch in val):
                    raise ValueError(f"refusing to write whitespace into id-list column {col!r}; "
                                     f"expected {ID_LIST_SEPARATOR!r}-joined ids")
                continue
            if len(val) > MAX_CELL_CHARS or len(val.split()) > MAX_CELL_WORDS:
                raise ValueError(f"refusing to write free text into column {col!r} "
                                 f"({len(val)} chars, {len(val.split())} words)")


def write_table(path: Path, rows: Sequence[dict[str, Any]], columns: Sequence[str]) -> Path:
    """Write a results CSV after the no-free-text guard passes."""
    assert_no_free_text(rows, columns)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(columns))
        w.writeheader()
        for row in rows:
            w.writerow({c: row.get(c, "") for c in columns})
    return path


def _round(x: float | None, nd: int = 4) -> str | float:
    return "" if x is None else round(float(x), nd)


def smoke_path(path: Path) -> Path:
    """`results/tables/x.csv` -> `results/tables/x_smoke.csv` (a smoke run never clobbers a full one)."""
    return path.with_name(f"{path.stem}_smoke{path.suffix}")


def score_corpus(machine: Sequence[Sitrep], humans: Sequence[Sitrep], *,
                 metrics: Sequence[str], window_days: int = DEFAULT_WINDOW_DAYS,
                 max_refs: int = DEFAULT_MAX_REFS, event_fallback: bool = False,
                 rouge_scorer_obj: Any | None = None, bert_scorer_obj: Any | None = None,
                 judge_scores: dict[str, float] | None = None,
                 judge_model: str = "") -> list[dict[str, Any]]:
    """Assemble the `reference_metrics.csv` rows (pairing + already-computed metric inputs)."""
    judge_scores = judge_scores or {}
    pairings = [pair_references(m, humans, window_days=window_days, max_refs=max_refs,
                               event_fallback=event_fallback) for m in machine]
    bert = [None] * len(machine)
    if BERTSCORE in metrics and bert_scorer_obj is not None:
        bert = bertscore_many(bert_scorer_obj, [m.text for m in machine],
                              [[r.text for r in p.refs] for p in pairings])
    rows: list[dict[str, Any]] = []
    for m, pairing, bs in zip(machine, pairings, bert):
        rg = None
        if ROUGE_L in metrics and rouge_scorer_obj is not None:
            rg = rouge_l(rouge_scorer_obj, m.text, [r.text for r in pairing.refs])
        rows.append({
            "sitrep_id": m.id, "event": m.event, "date": m.date, "model": m.model, "arm": m.arm,
            "pairing_rule": pairing.rule, "n_refs": len(pairing.refs),
            "ref_ids": "|".join(r.id for r in pairing.refs),
            "ref_date": pairing.refs[0].date if pairing.refs else "",
            "day_gap": "" if pairing.day_gap is None else pairing.day_gap,
            "rouge_l_f": _round(rg["f"] if rg else None), "rouge_l_p": _round(rg["p"] if rg else None),
            "rouge_l_r": _round(rg["r"] if rg else None),
            "bertscore_f1": _round(bs["f"] if bs else None), "bertscore_p": _round(bs["p"] if bs else None),
            "bertscore_r": _round(bs["r"] if bs else None),
            "llm_judge_score": _round(judge_scores.get(m.id), 2),
            "judge_model": judge_model if m.id in judge_scores else "",
            "version": TABLE_VERSION,
        })
    return rows


def run_judge(llm: LLM, sitreps: Sequence[Sitrep], *, batch: bool, judge_role: str = "judge",
              gate: bool = True) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Cost-gated generic-judge pass. Returns ({sitrep_id: score}, justification rows)."""
    if not sitreps:
        return {}, []
    reqs = judge_requests(sitreps, model=judge_role)
    if gate:
        llm.gate(llm.estimate_cost(reqs, batch=batch))
    resps = llm.run(reqs, batch=batch)
    scores: dict[str, float] = {}
    notes: list[dict[str, Any]] = []
    for s, resp in zip(sitreps, resps):
        score, why = parse_judge_score(resp)
        if score is None:
            log.warning("no usable judge score for %s (error=%s refused=%s)", s.id, resp.error, resp.refused)
            continue
        scores[s.id] = score
        notes.append({"sitrep_id": s.id, "event": s.event, "date": s.date, "score": score,
                      "justification": why, "judge_model": resp.model_id, "at": utc_now_iso()})
    return scores, notes


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _resolve_metrics(cfg: dict[str, Any], requested: str | None) -> list[str]:
    """Metric ids from --metrics, else config `reference_metrics` (list or {metrics: [...]})."""
    if requested:
        names = [m.strip() for m in requested.split(",") if m.strip()]
    else:
        raw = cfg.get("reference_metrics") or list(ALL_METRICS)
        names = list(raw.get("metrics", ALL_METRICS)) if isinstance(raw, dict) else list(raw)
    unknown = [m for m in names if m not in ALL_METRICS]
    if unknown:
        raise ValueError(f"unknown metric(s) {unknown}; known: {list(ALL_METRICS)}")
    return names


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description="Phase 5: reference metrics for machine sitreps")
    ap.add_argument("--event", default=None, help="limit to one event key (default: all)")
    ap.add_argument("--smoke", type=int, default=0, metavar="N",
                    help="score only N machine sitreps, synchronously, and print the rows")
    ap.add_argument("--metrics", default=None, help=f"comma-separated subset of {list(ALL_METRICS)}")
    ap.add_argument("--window-days", type=int, default=DEFAULT_WINDOW_DAYS,
                    help="same-window fallback radius in days (default: %(default)s)")
    ap.add_argument("--max-refs", type=int, default=DEFAULT_MAX_REFS, help="references per machine sitrep")
    ap.add_argument("--event-fallback", action="store_true",
                    help="if no reference is within the window, fall back to any sitrep of the event")
    ap.add_argument("--bertscore-model", default=DEFAULT_BERTSCORE_MODEL, help="bert-score model_type")
    ap.add_argument("--bertscore", action="store_true", help="force BERTScore on in --smoke mode")
    ap.add_argument("--machine-dir", type=Path, default=None, help="override data/processed/sitreps_machine")
    ap.add_argument("--human-dir", type=Path, default=None, help="override data/processed/sitreps_human")
    ap.add_argument("--out", type=Path, default=None, help="output CSV (default results/tables/reference_metrics.csv)")
    ap.add_argument("--judge-model", default="judge",
                    help="config role for the generic LLM judge (default: judge). Use a role from "
                         "a different family than the slot judge so the correlation is not "
                         "measuring a model against itself.")
    ap.add_argument("--save-justifications", action="store_true",
                    help="append judge justifications to data/processed/judge_justifications.jsonl (gitignored)")
    ap.add_argument("--no-batch", action="store_true", help="use synchronous calls for the full judge run")
    args = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):  # pragma: no cover
        pass

    cfg = load_config()
    set_seed(int(cfg.get("seed", 17)))
    processed = cfg_path(cfg, "processed", "data/processed")
    human_dir = args.human_dir or processed / "sitreps_human"
    machine_dir = args.machine_dir or processed / "sitreps_machine"
    out_path = args.out or (cfg_path(cfg, "results", "results") / "tables" / "reference_metrics.csv")
    try:
        metrics = _resolve_metrics(cfg, args.metrics)
    except ValueError as exc:
        print(exc)
        return 2

    machine = load_sitreps(machine_dir, args.event, kind="machine")
    humans = load_sitreps(human_dir, args.event, kind="human")
    if not machine:
        print(f"No machine sitreps under {machine_dir} — run Phase 3 (src/generate_sitreps.py) first.")
        return 2
    if not humans:
        print(f"No human sitreps under {human_dir} — the ReliefWeb corpus is not collected yet "
              f"(PLAN §2a). Reference metrics need references; nothing scored.")
        return 2
    if args.smoke:
        machine = machine[: args.smoke]
        if args.out is None:
            out_path = smoke_path(out_path)
        if BERTSCORE in metrics and not args.bertscore:
            metrics = [m for m in metrics if m != BERTSCORE]
            log.info("BERTScore skipped in --smoke (no model download); pass --bertscore to force it")

    rouge_obj = load_rouge_scorer() if ROUGE_L in metrics else None
    bert_obj = load_bertscorer(args.bertscore_model) if BERTSCORE in metrics else None

    judge_scores: dict[str, float] = {}
    notes: list[dict[str, Any]] = []
    judge_model = ""
    if LLM_JUDGE in metrics:
        llm = LLM(cfg)
        try:
            judge_model = llm.resolve_model(args.judge_model).id
            judge_scores, notes = run_judge(llm, machine, judge_role=args.judge_model,
                                            batch=not (args.smoke or args.no_batch))
        except CostLimitExceeded as exc:
            print(f"Cost gate stopped the judge run: {exc}")
            return 3
        except Exception as exc:  # noqa: BLE001 - a judge outage must not lose ROUGE/BERTScore
            log.warning("generic LLM judge skipped (%s: %s)", type(exc).__name__, exc)
        log.info("llm summary: %s", llm.summary())

    rows = score_corpus(machine, humans, metrics=metrics, window_days=args.window_days,
                        max_refs=args.max_refs, event_fallback=args.event_fallback,
                        rouge_scorer_obj=rouge_obj, bert_scorer_obj=bert_obj,
                        judge_scores=judge_scores, judge_model=judge_model)
    write_table(out_path, rows, TABLE_COLUMNS)
    if notes and args.save_justifications:
        note_path = processed / "judge_justifications.jsonl"
        append_jsonl(note_path, notes)
        print(f"justifications: {note_path} (gitignored; model text stays out of results/)")

    by_rule: dict[str, int] = {}
    for r in rows:
        by_rule[r["pairing_rule"]] = by_rule.get(r["pairing_rule"], 0) + 1
    print(f"scored {len(rows)} machine sitrep(s) against {len(humans)} human sitrep(s)")
    print(f"metrics: {metrics}")
    print(f"pairing: {by_rule}")
    print(f"table:   {out_path}")
    if args.smoke:
        cols = ["sitrep_id", "date", "model", "arm", "pairing_rule", "n_refs", "day_gap",
                "rouge_l_f", "bertscore_f1", "llm_judge_score"]
        print("\n" + " | ".join(cols))
        for r in rows:
            print(" | ".join(str(r.get(c, "")) for c in cols))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
