"""Sitrep writing conventions: sourcing, hedging, update markers, figure-with-source rate.

PLAN Phase 2, second half. Pure lexicon + regex - **no LLM anywhere in this module** - run
over the FULL human sitrep corpus (`data/processed/sitreps_human/<event>/<date>.json`).

The four convention families (every counter is a paper number, so every pattern is unit-tested
against hand-written strings in `tests/test_conventions.py`):

* **sourcing**   attribution phrases ("according to", "reports indicate", "as reported by")
                 plus mentions of named agencies (OCHA, IFRC, WHO, INGC, ministries, ...).
                 Acronyms are matched CASE-SENSITIVELY so "who" the pronoun is not the WHO.
* **hedging**    "unconfirmed", "estimated", "approximately", "reportedly", "preliminary",
                 modal "may"/"could" (lower-case only, so the month May is not a hedge), ...
* **updates**    "as of <date>" / "as at <date>" markers and revision language ("revised",
                 "previously reported", "since the last report", "supersedes", ...).
* **figures**    a figure is a number that is NOT part of a date, a time or a report number;
                 it counts as *sourced* when a sourcing cue sits within `--window` tokens
                 (default 12) of it. figure_with_source_rate = sourced / figures.

Outputs: `results/tables/convention_stats.csv` (one row per sitrep + a TOTAL row) and
`results/tables/convention_terms.csv` (corpus frequency per lexicon entry).

CLI: python -m src.conventions [--smoke [N]] [--corpus DIR] [--window 12]
"""
from __future__ import annotations

import argparse
import bisect
import csv
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from src.util import cfg_path, ensure_dir, get_logger, load_config, read_json

log = get_logger("sitrep.conventions")

#: how many tokens may sit between a figure and a sourcing cue for the figure to count as sourced
FIGURE_SOURCE_WINDOW_TOKENS = 12

Pattern = tuple[str, str, bool]  # (label, regex, case_sensitive)


# ---------------------------------------------------------------------------
# Lexicons
# ---------------------------------------------------------------------------
_MONTHS = ("January|February|March|April|May|June|July|August|September|October|November|December"
           "|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept|Sep|Oct|Nov|Dec")
_DAY = r"\d{1,2}(?:st|nd|rd|th)?"

#: date/time constructs - the numbers inside these are NOT operational figures
DATE_RE = re.compile("|".join((
    r"\d{4}-\d{2}-\d{2}",
    r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}",
    rf"\d{{1,2}}\s*[-–]\s*{_DAY}\s+(?:{_MONTHS})\.?(?:,?\s+\d{{4}})?",   # "1-7 April" ranges
    rf"{_DAY}\s+(?:{_MONTHS})\.?(?:,?\s+\d{{4}})?",
    rf"(?:{_MONTHS})\.?\s+{_DAY}(?:,?\s+\d{{4}})?",
    rf"(?:{_MONTHS})\.?\s+\d{{4}}",
    r"\d{1,2}:\d{2}(?:\s*(?:a\.?m\.?|p\.?m\.?|hrs|hours|GMT|UTC))?",
    r"\b(?:in|since|of|for|during|from|until|by|through)\s+(?:19|20)\d{2}\b",
)), re.IGNORECASE)

#: report/page numbering - also not operational figures
NON_FIGURE_RE = re.compile("|".join((
    r"(?:No\.|Nos\.|#)\s*\d+",
    r"(?:sitrep|situation\s+report|flash\s+update|issue|bulletin|version)\s+\d{1,3}\b",
    r"(?:page|pp?\.)\s*\d+(?:\s*(?:of|/)\s*\d+)?",
)), re.IGNORECASE)

#: a number, optionally with grouped thousands or decimals ("598", "1,200", "3.5", "45")
FIGURE_RE = re.compile(r"(?<![\w.,\-])(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)")

ATTRIBUTION_PATTERNS: tuple[Pattern, ...] = (
    ("according_to", r"according\s+to", False),
    ("as_per", r"\bas\s+per\b", False),
    ("reports_indicate", r"\breports?\s+(?:indicate|suggest|show|state)\w*", False),
    ("reported_by", r"\b(?:as\s+)?reported\s+by\b", False),
    ("data_from", r"\b(?:data|figures?|estimates?|information|statistics|casualty\s+figures?)"
                  r"\s+(?:from|released\s+by|provided\s+by|compiled\s+by|shared\s+by)\b", False),
    ("citing", r"\bcit(?:es|ed|ing)\b", False),
    ("source_label", r"\bsources?\s*:", False),
    ("sources_say", r"\bsources?\s+(?:say|said|report|reported|indicate)\w*", False),
    ("authorities_report", r"\b(?:local|national|government|district|provincial|state|municipal)\s+"
                           r"authorities\s+(?:report|say|estimate|indicate|confirm)\w*", False),
)

#: acronyms - CASE SENSITIVE on purpose (WHO vs. "who", CARE vs. "care", IPC vs. "ipc")
AGENCY_ACRONYM_PATTERNS: tuple[Pattern, ...] = (
    ("agency_acronym", r"\b(?:UN\s+)?OCHA\b", True),
    ("agency_acronym", r"\b(?:UNICEF|WFP|WHO|UNHCR|IOM|UNDP|UNFPA|FAO|UNESCO|UNRWA)\b", True),
    ("agency_acronym", r"\b(?:IFRC|ICRC|MSF|NRC|IRC|CARE)\b", True),
    ("agency_acronym", r"\b(?:INGC|NDMA|NDRRMC|BNPB|DREF)\b", True),
    ("agency_acronym", r"\b(?:ECHO|USAID|FEMA|DFID|FCDO|GDACS|WMO|IPC|ACAPS|IASC|CERF|REACH)\b", True),
)

AGENCY_NAME_PATTERNS: tuple[Pattern, ...] = (
    ("agency_name", r"\bunited\s+nations\b", False),
    ("agency_name", r"\bworld\s+health\s+organi[sz]ation\b", False),
    ("agency_name", r"\bworld\s+food\s+program(?:me)?\b", False),
    ("agency_name", r"\bred\s+(?:cross|crescent)\b", False),
    ("agency_name", r"\bsave\s+the\s+children\b|\bworld\s+vision\b|\boxfam\b|\bplan\s+international\b", False),
    ("agency_name", r"\bm[eé]decins\s+sans\s+fronti[eè]res\b|\bdoctors\s+without\s+borders\b", False),
    ("agency_name", r"\bministry\s+of\s+\w+", False),
    ("agency_name", r"\bgovernment\s+of\s+[A-Z]\w+", False),
    ("agency_name", r"\bnational\s+disaster\s+(?:management|risk)\s+\w+", False),
    ("agency_name", r"\bcivil\s+protection\b|\bmeteorolog\w+\s+(?:agency|service|department|institute|office)\b",
     False),
    ("agency_name", r"\b(?:humanitarian\s+country\s+team|cluster\s+lead)\b", False),
)

HEDGE_PATTERNS: tuple[Pattern, ...] = (
    ("unconfirmed", r"\bunconfirmed\b", False),
    ("unverified", r"\bunverified\b", False),
    ("not_yet_confirmed", r"\b(?:not\s+yet|yet\s+to\s+be)\s+(?:confirmed|verified|established)\b", False),
    ("to_be_confirmed", r"\bto\s+be\s+confirmed\b|\bTBC\b", False),
    ("subject_to_change", r"\bsubject\s+to\s+(?:change|verification|revision|confirmation)\b", False),
    ("estimated", r"\bestimat(?:e|es|ed|ing|ion|ions)\b", False),
    ("approximately", r"\bapproximately\b|\bapprox\.?(?!\w)", False),
    ("reportedly", r"\breportedly\b", False),
    ("allegedly", r"\ballegedly\b", False),
    ("apparently", r"\bapparently\b", False),
    ("preliminary", r"\bpreliminary\b", False),
    ("provisional", r"\bprovisional\w*\b", False),
    ("tentative", r"\btentative\w*\b", False),
    ("indicative", r"\bindicative\b", False),
    ("likely", r"\blikely\b", False),
    ("possibly", r"\bpossibl[ey]\b|\bpotentially\b", False),
    ("modal_may", r"(?<!\w)may(?!\w)", True),      # lower-case only: the month "May" is not a hedge
    ("modal_could", r"(?<!\w)could(?!\w)", True),
    ("appears_to", r"\bappears?\s+to\b|\bseems?\s+to\b", False),
    ("roughly", r"\broughly\b", False),
    ("no_confirmation", r"\bno\s+(?:official\s+)?confirmation\b", False),
)

#: "as of <date|today|this morning|14:00>" / "as at <date>"
AS_OF_RE = re.compile(
    r"\bas\s+(?:of|at)\s+(?:"
    rf"{_DAY}\s+(?:{_MONTHS})\.?(?:,?\s+\d{{4}})?"
    rf"|(?:{_MONTHS})\.?\s+{_DAY}(?:,?\s+\d{{4}})?"
    r"|\d{4}-\d{2}-\d{2}"
    r"|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"
    r"|\d{1,2}:\d{2}(?:\s*(?:hrs|hours|GMT|UTC|a\.?m\.?|p\.?m\.?))?"
    r"|today|yesterday|this\s+(?:morning|afternoon|evening|week)"
    r"|the\s+time\s+of\s+(?:writing|reporting|publication)"
    r")", re.IGNORECASE)

REVISION_PATTERNS: tuple[Pattern, ...] = (
    # the specific multi-word markers come first: within a lexicon the earlier pattern wins an overlap
    ("figures_updated", r"\b(?:figures?|numbers?|totals?|estimates?|data|casualty\s+figures?)\s+"
                        r"(?:have\s+been\s+|has\s+been\s+|were\s+|was\s+)?(?:revised|updated|corrected)\b", False),
    ("revised", r"\brevis(?:ed|ion|ions|es)\b", False),
    ("previously_reported", r"\bpreviously\s+(?:reported|stated|published)\b", False),
    ("compared_to_previous", r"\bcompared\s+(?:to|with)\s+(?:the\s+)?(?:previous|last|earlier)\b", False),
    ("since_last_report", r"\bsince\s+the\s+(?:last|previous)\s+(?:report|update|sitrep)\b", False),
    ("this_update", r"\bthis\s+(?:update|report|sitrep|situation\s+report)\b", False),
    ("supersedes", r"\bsupersedes?\b|\breplaces\s+the\s+previous\b", False),
    ("no_change", r"\b(?:no\s+change|unchanged)\s+(?:since|from)\b", False),
    ("new_since", r"\bnew\s+since\b", False),
    ("correction", r"\bcorrection\b", False),
    ("reporting_period", r"\breporting\s+period\b|\bcovers?\s+the\s+period\b", False),
    ("update_number", r"\b(?:update|sitrep|situation\s+report|flash\s+update)\s+(?:no\.?\s*|#)\s*\d+\b", False),
)


def _compile(patterns: Sequence[Pattern]) -> tuple[tuple[str, re.Pattern[str]], ...]:
    """Compile a lexicon, honouring each entry's case-sensitivity flag."""
    return tuple((label, re.compile(rx, 0 if cs else re.IGNORECASE)) for label, rx, cs in patterns)


_ATTRIBUTION = _compile(ATTRIBUTION_PATTERNS)
_AGENCIES = _compile(AGENCY_ACRONYM_PATTERNS + AGENCY_NAME_PATTERNS)
_HEDGES = _compile(HEDGE_PATTERNS)
_REVISIONS = _compile(REVISION_PATTERNS)


# ---------------------------------------------------------------------------
# Span helpers
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Span:
    """A lexicon hit: `label` plus the [start, end) character offsets."""

    label: str
    start: int
    end: int


def find_spans(text: str, lexicon: Iterable[tuple[str, re.Pattern[str]]]) -> list[Span]:
    """All non-overlapping hits of `lexicon` in `text`, left to right.

    Within one lexicon an earlier pattern wins an overlap, so a phrase is never
    double-counted by two entries that share a prefix.
    """
    spans: list[Span] = []
    for label, rx in lexicon:
        for m in rx.finditer(text):
            if not _overlaps(m.start(), m.end(), spans):
                spans.append(Span(label, m.start(), m.end()))
    return sorted(spans, key=lambda s: (s.start, s.end))


def _overlaps(start: int, end: int, spans: Iterable[Span]) -> bool:
    return any(start < s.end and s.start < end for s in spans)


def find_sourcing(text: str) -> tuple[list[Span], list[Span]]:
    """(attribution phrases, named-agency mentions)."""
    return find_spans(text, _ATTRIBUTION), find_spans(text, _AGENCIES)


def find_hedges(text: str) -> list[Span]:
    """Hedging / uncertainty markers."""
    return find_spans(text, _HEDGES)


def find_as_of(text: str) -> list[Span]:
    """"as of <date>" / "as at <date>" currency markers."""
    return [Span("as_of_date", m.start(), m.end()) for m in AS_OF_RE.finditer(text)]


def find_revisions(text: str) -> list[Span]:
    """Revision / previous-report language."""
    return find_spans(text, _REVISIONS)


def find_figures(text: str) -> list[Span]:
    """Numbers that are real quantities: dates, times and report numbers are excluded."""
    blocked = [(m.start(), m.end()) for m in DATE_RE.finditer(text)]
    blocked += [(m.start(), m.end()) for m in NON_FIGURE_RE.finditer(text)]
    out: list[Span] = []
    for m in FIGURE_RE.finditer(text):
        if any(m.start() < e and s < m.end() for s, e in blocked):
            continue
        out.append(Span("figure", m.start(), m.end()))
    return out


# ---------------------------------------------------------------------------
# Token distance
# ---------------------------------------------------------------------------
def token_spans(text: str) -> list[tuple[int, int]]:
    """Whitespace tokenisation as [start, end) character offsets."""
    return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]


def _tok_index(starts: list[int], pos: int) -> int:
    return max(0, bisect.bisect_right(starts, pos) - 1)


def token_gap(a: Span, b: Span, toks: list[tuple[int, int]]) -> int:
    """Number of whole tokens strictly between spans `a` and `b` (0 if they touch or overlap)."""
    if not toks:
        return 0
    starts = [t[0] for t in toks]
    a0, a1 = _tok_index(starts, a.start), _tok_index(starts, max(a.start, a.end - 1))
    b0, b1 = _tok_index(starts, b.start), _tok_index(starts, max(b.start, b.end - 1))
    if a1 < b0:
        return b0 - a1 - 1
    if b1 < a0:
        return a0 - b1 - 1
    return 0


def figures_with_source(text: str, figures: Sequence[Span], cues: Sequence[Span],
                        window: int = FIGURE_SOURCE_WINDOW_TOKENS) -> list[bool]:
    """For each figure: is there a sourcing cue within `window` tokens of it?"""
    toks = token_spans(text)
    return [any(token_gap(f, c, toks) <= window for c in cues) for f in figures]


# ---------------------------------------------------------------------------
# Per-text statistics
# ---------------------------------------------------------------------------
@dataclass
class TextStats:
    """Convention counters for one piece of text (all counts are raw match counts)."""

    n_chars: int = 0
    n_tokens: int = 0
    sourcing_phrases: int = 0
    agency_mentions: int = 0
    sourcing_cues: int = 0
    hedges: int = 0
    hedge_types: int = 0
    as_of_dates: int = 0
    revision_markers: int = 0
    update_markers: int = 0
    n_figures: int = 0
    n_figures_with_source: int = 0
    terms: Counter = field(default_factory=Counter)

    @property
    def figure_with_source_rate(self) -> float | None:
        """Sourced figures / figures, or None when the text contains no figure."""
        return (self.n_figures_with_source / self.n_figures) if self.n_figures else None

    def per_1k(self, count: int) -> float:
        """`count` normalised per 1,000 whitespace tokens."""
        return round(1000.0 * count / self.n_tokens, 3) if self.n_tokens else 0.0


def analyze_text(text: str, window: int = FIGURE_SOURCE_WINDOW_TOKENS) -> TextStats:
    """Count every convention family in `text`. Pure: no I/O, no model calls."""
    text = text or ""
    attribution, agencies = find_sourcing(text)
    hedges = find_hedges(text)
    as_of = find_as_of(text)
    revisions = find_revisions(text)
    figures = find_figures(text)
    cues = attribution + agencies
    sourced = figures_with_source(text, figures, cues, window)
    terms: Counter = Counter()
    for cat, spans in (("sourcing", attribution), ("agency", agencies), ("hedge", hedges),
                       ("update", as_of), ("update", revisions)):
        for sp in spans:
            terms[f"{cat}:{sp.label}"] += 1
    return TextStats(
        n_chars=len(text),
        n_tokens=len(token_spans(text)),
        sourcing_phrases=len(attribution),
        agency_mentions=len(agencies),
        sourcing_cues=len(cues),
        hedges=len(hedges),
        hedge_types=len({s.label for s in hedges}),
        as_of_dates=len(as_of),
        revision_markers=len(revisions),
        update_markers=len(as_of) + len(revisions),
        n_figures=len(figures),
        n_figures_with_source=sum(sourced),
        terms=terms,
    )


# ---------------------------------------------------------------------------
# Corpus pass
# ---------------------------------------------------------------------------
CSV_COLUMNS = ["event", "sitrep_id", "source", "date", "n_chars", "n_tokens",
               "sourcing_phrases", "agency_mentions", "sourcing_cues", "hedges", "hedge_types",
               "as_of_dates", "revision_markers", "update_markers",
               "n_figures", "n_figures_with_source", "figure_with_source_rate",
               "sourcing_per_1k", "hedges_per_1k", "update_markers_per_1k", "figures_per_1k"]

SUM_FIELDS = ("n_chars", "n_tokens", "sourcing_phrases", "agency_mentions", "sourcing_cues",
              "hedges", "as_of_dates", "revision_markers", "update_markers",
              "n_figures", "n_figures_with_source")


def load_corpus(sitrep_dir: str | Path) -> list[dict[str, Any]]:
    """Every `<event>/<date>.json` sitrep record under `sitrep_dir`, sorted by (event, file)."""
    out: list[dict[str, Any]] = []
    for path in sorted(Path(sitrep_dir).glob("*/*.json")):
        if path.name.startswith("_"):
            continue
        try:
            rec = read_json(path)
        except (ValueError, OSError) as exc:
            log.warning("skipping unreadable sitrep %s (%s)", path, exc)
            continue
        if not isinstance(rec, dict) or not str(rec.get("text") or "").strip():
            log.warning("skipping sitrep with no text: %s", path)
            continue
        rec.setdefault("id", path.stem)
        rec.setdefault("event", path.parent.name)
        out.append(rec)
    return out


def _row(rec: dict[str, Any], st: TextStats) -> dict[str, Any]:
    rate = st.figure_with_source_rate
    return {
        "event": rec.get("event", ""), "sitrep_id": rec.get("id", ""),
        "source": rec.get("source", ""), "date": rec.get("date", ""),
        "n_chars": st.n_chars, "n_tokens": st.n_tokens,
        "sourcing_phrases": st.sourcing_phrases, "agency_mentions": st.agency_mentions,
        "sourcing_cues": st.sourcing_cues, "hedges": st.hedges, "hedge_types": st.hedge_types,
        "as_of_dates": st.as_of_dates, "revision_markers": st.revision_markers,
        "update_markers": st.update_markers, "n_figures": st.n_figures,
        "n_figures_with_source": st.n_figures_with_source,
        "figure_with_source_rate": "" if rate is None else round(rate, 4),
        "sourcing_per_1k": st.per_1k(st.sourcing_cues), "hedges_per_1k": st.per_1k(st.hedges),
        "update_markers_per_1k": st.per_1k(st.update_markers), "figures_per_1k": st.per_1k(st.n_figures),
    }


def analyze_corpus(records: Sequence[dict[str, Any]],
                   window: int = FIGURE_SOURCE_WINDOW_TOKENS) -> tuple[list[dict[str, Any]], Counter]:
    """Per-sitrep CSV rows plus a final TOTAL row, and the corpus-wide term counter."""
    rows: list[dict[str, Any]] = []
    total = TextStats()
    terms: Counter = Counter()
    for rec in records:
        st = analyze_text(str(rec.get("text") or ""), window)
        rows.append(_row(rec, st))
        terms.update(st.terms)
        for fld in SUM_FIELDS:
            setattr(total, fld, getattr(total, fld) + getattr(st, fld))
    total.hedge_types = len({k.split(":", 1)[1] for k in terms if k.startswith("hedge:")})
    rows.append(_row({"event": "ALL", "id": "TOTAL", "source": f"{len(records)} sitreps", "date": ""}, total))
    return rows, terms


def write_tables(rows: Sequence[dict[str, Any]], terms: Counter, tables_dir: str | Path,
                 suffix: str = "") -> tuple[Path, Path]:
    """Write convention_stats.csv + convention_terms.csv; returns both paths."""
    out_dir = ensure_dir(tables_dir)
    stats_path = out_dir / f"convention_stats{suffix}.csv"
    with open(stats_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    terms_path = out_dir / f"convention_terms{suffix}.csv"
    with open(terms_path, "w", encoding="utf-8", newline="") as fh:
        tw = csv.writer(fh)
        tw.writerow(["category", "label", "count"])
        for key, count in sorted(terms.items(), key=lambda kv: (-kv[1], kv[0])):
            cat, label = key.split(":", 1)
            tw.writerow([cat, label, count])
    return stats_path, terms_path


NO_CORPUS_MESSAGE = """
No human sitreps found - the convention statistics need the ReliefWeb corpus.

Expected: data/processed/sitreps_human/<event>/<date>.json
          ({id,title,source,date,url,text,collection,file,ingested_at})

Collect it first (PLAN 2a / Phase 1):
  python -m src.ingest_manual_reliefweb --scan data/raw/reliefweb_manual/<event>
  python -m src.ingest_manual_reliefweb
or, once RELIEFWEB_APPNAME is approved, python -m src.collect_reliefweb.
"""

_HEAD = ["sitrep_id", "n_tokens", "sourcing_cues", "hedges", "update_markers",
         "n_figures", "n_figures_with_source", "figure_with_source_rate"]


def _fmt(row: dict[str, Any]) -> str:
    return "  ".join(f"{str(row[h])[:22]:>22}" if i == 0 else f"{str(row[h]):>12}"
                     for i, h in enumerate(_HEAD))


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description="Sitrep convention statistics (PLAN Phase 2, no LLM)")
    ap.add_argument("--config", type=Path, default=None, help="config.yaml to use")
    ap.add_argument("--corpus", type=Path, default=None, help="sitreps_human directory")
    ap.add_argument("--tables-dir", type=Path, default=None, help="where the CSVs go")
    ap.add_argument("--window", type=int, default=FIGURE_SOURCE_WINDOW_TOKENS,
                    help="token window for figure-with-source (default 12)")
    ap.add_argument("--smoke", nargs="?", type=int, const=5, default=None,
                    help="only analyse N sitreps and print them (default 5)")
    args = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):  # pragma: no cover
        pass
    if args.window < 0:
        print("--window must be >= 0")
        return 2

    cfg = load_config(args.config)
    corpus_dir = args.corpus or (cfg_path(cfg, "processed", "data/processed") / "sitreps_human")
    tables_dir = args.tables_dir or (cfg_path(cfg, "results", "results") / "tables")
    records = load_corpus(corpus_dir)
    if not records:
        log.warning("no sitreps under %s", corpus_dir)
        print(NO_CORPUS_MESSAGE)
        return 2

    smoke = args.smoke is not None
    if smoke:
        records = records[: max(1, args.smoke)]
    rows, terms = analyze_corpus(records, args.window)
    stats_path, terms_path = write_tables(rows, terms, tables_dir, suffix="_smoke" if smoke else "")

    total = rows[-1]
    print(f"\nconvention stats over {len(records)} sitrep(s), figure window = {args.window} tokens"
          f"{'  [SMOKE]' if smoke else ''}\n")
    print("  ".join(f"{h:>22}" if i == 0 else f"{h:>12}" for i, h in enumerate(_HEAD)))
    for row in rows[:-1][:20]:
        print(_fmt(row))
    print("-" * 118)
    print(_fmt(total))
    print(f"\ncorpus totals: {total['sourcing_cues']} sourcing cues "
          f"({total['sourcing_phrases']} phrases + {total['agency_mentions']} agency mentions), "
          f"{total['hedges']} hedges in {total['hedge_types']} forms, "
          f"{total['update_markers']} update markers ({total['as_of_dates']} 'as of' + "
          f"{total['revision_markers']} revision), {total['n_figures']} figures of which "
          f"{total['n_figures_with_source']} sourced (rate {total['figure_with_source_rate']}).")
    print(f"\nwrote {stats_path}\nwrote {terms_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
