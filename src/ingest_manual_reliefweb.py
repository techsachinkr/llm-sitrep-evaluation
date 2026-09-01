"""Manual-collection fallback (PLAN §2a): ingest browser-saved ReliefWeb
report pages / PDFs / text files into the unified human-sitrep format.

This script NEVER fetches anything from the network — reliefweb.int answers
scripts with HTTP 202/403 (bot challenge) and the API needs an approved appname.
It only reads files the user saved by hand and listed in a manifest CSV
(``event,title,source,date,url,file``; ``file`` relative to the manifest dir).

Output: ``data/processed/sitreps_human/<event>/<date>.json`` with
``{id, title, source, date, url, text, collection: "manual", file, ingested_at}``
plus ``<event>/_manifest_ingested.csv`` per event.

CLI::

    python -m src.ingest_manual_reliefweb --scan data/raw/reliefweb_manual/<event>  # build the manifest
    python -m src.ingest_manual_reliefweb                                           # ingest it
"""
from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from src.reliefweb_api import parse_date, soup_text
from src.util import (
    resolve_event_dir,
    sha256_of,
    ROOT,
    cfg_path,
    ensure_dir,
    get_logger,
    load_config,
    slugify,
    utc_now_iso,
    write_json,
)

log = get_logger("ingest_manual")

MANIFEST_COLUMNS = ["event", "title", "source", "date", "url", "file"]
TEMPLATE_PATH = ROOT / "docs" / "manifest_template.csv"
MIN_TEXT_CHARS = 200

BODY_SELECTORS = (
    "article",
    '[class*="rw-report__content"]',
    '[class*="report__content"]',
    '[class*="rw-article__content"]',
    "main",
    "#main-content",
    "[role=main]",
)
NOISE_TAGS = ("script", "style", "nav", "header", "footer", "aside", "form", "noscript")

README_TEXT = """# Manual ReliefWeb collection

Put browser-saved ReliefWeb report artifacts here (PDF preferred; else the
report page saved as "Webpage, HTML only"), organised as `<event>/<file>`, and
list each one as a row in `manifest.csv` (columns: event,title,source,date,url,file;
`file` is relative to this directory, `date` is YYYY-MM-DD).

Then run `python -m src.ingest_manual_reliefweb`.
Full instructions: docs/manual_collection.md.
"""


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------
def _clean_title(title: str | None) -> str | None:
    """Strip ' - Nepal | ReliefWeb'-style suffixes from a page title."""
    if not title:
        return None
    t = re.sub(r"\s+", " ", title).strip()
    had_suffix = bool(re.search(r"\|\s*ReliefWeb\s*$", t))
    t = re.sub(r"\s*\|\s*ReliefWeb\s*$", "", t).strip()
    if had_suffix:  # ReliefWeb page titles end with ' - <Country>' before '| ReliefWeb'
        m = re.match(r"^(.*\S)\s+[-–]\s+([A-Za-z][A-Za-z ,'()-]{0,40})$", t)
        if m and len(m.group(2).split()) <= 4:
            t = m.group(1).strip()
    return t or None


def _collapse(text: str) -> str:
    text = text.replace("\r", "")
    text = re.sub(r"[ \t\f\v]+\n", "\n", text)
    text = re.sub(r"[ \t\f\v]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _iso_date(value: str | None) -> str | None:
    """Normalise a date string to YYYY-MM-DD (ISO, 'YYYY-MM-DD', '30 April 2015', ...) or None."""
    if not value:
        return None
    return parse_date(value)


# Organisation names as they appear in ReliefWeb reports -> the short name used in the manifest.
SOURCE_PATTERNS: tuple[tuple[str, str], ...] = (
    ("UN Office for the Coordination of Humanitarian Affairs", "OCHA"),
    ("United Nations Office for the Coordination of Humanitarian Affairs", "OCHA"),
    ("Office for the Coordination of Humanitarian Affairs", "OCHA"),
    ("OCHA", "OCHA"),
    ("International Federation of Red Cross", "IFRC"),
    ("IFRC", "IFRC"),
    ("World Food Programme", "WFP"),
    ("UNICEF", "UNICEF"),
    ("World Health Organization", "WHO"),
    ("International Organization for Migration", "IOM"),
    ("UNHCR", "UNHCR"),
)

# "(as of 2 April 2019)", "as of 02 Apr 2019", "12 December 2019"
_DATE_IN_TEXT = re.compile(
    r"(?:as of\s+)?(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4})",
    re.IGNORECASE)


# Publisher hints that appear in ReliefWeb PDF filenames. Checked before the document text,
# because the first organisation named inside a report is usually a donor, not the publisher.
FILENAME_SOURCE_HINTS: tuple[tuple[str, str], ...] = (
    ("unicef", "UNICEF"), ("who", "WHO"), ("iom", "IOM"), ("wfp", "WFP"), ("unhcr", "UNHCR"),
    ("mdr", "IFRC"), ("ifrc", "IFRC"), ("dref", "IFRC"),
    ("rosea", "OCHA"), ("ocha", "OCHA"),
    ("etc ", "WFP"), ("logistics", "WFP"),
)

# Lines that are template furniture rather than a report title.
BOILERPLATE_MARKERS: tuple[str, ...] = (
    "the mission of the united nations office",
    "united nations office for the coordination of humanitarian affairs",
    "coordination saves lives",
    "www.unocha.org", "reliefweb.int", "microsoft word",
)

# "as of 6 May 2019" / "6 May 2019" but NOT the "next report" footer.
_AS_OF_RE = re.compile(
    r"as\s+of\s+(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4})",
    re.IGNORECASE)
_NEXT_REPORT_RE = re.compile(r"next\s+(?:report|update|sitrep)", re.IGNORECASE)
# ROSEA_20190506_..., ..._190330.pdf, ..._12032019.pdf (DDMMYYYY), "19 April 2019" in the name
_FN_YYYYMMDD = re.compile(r"(?<!\d)(20\d{2})(\d{2})(\d{2})(?!\d)")
_FN_DDMMYYYY = re.compile(r"(?<!\d)(\d{2})(\d{2})(20\d{2})(?!\d)")
_FN_YYMMDD = re.compile(r"_(\d{2})(\d{2})(\d{2})(?:[._]|$)")
# "19April2019" / "6 May 2019" without separators, as seen in WHO/UNICEF filenames
_FN_DMY_WORD = re.compile(
    r"(\d{1,2})\s*(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s*(20\d{2})", re.IGNORECASE)
# "Sep 7, 2017" / "September 21 2017" — month first, as agencies write it in report titles.
_FN_MDY_WORD = re.compile(
    r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+(\d{1,2})\s*,?\s+(20\d{2})",
    re.IGNORECASE)
# "2017 09 20" / "2017-09-20" — ISO with separators (the no-separator form is _FN_YYYYMMDD).
_FN_ISO_SEP = re.compile(r"(?<!\d)(20\d{2})[ _.-](\d{2})[ _.-](\d{2})(?!\d)")
# "Cuba 080917" — six-digit DDMMYY bounded by a space or the string edge, not only by "_".
_FN_DDMMYY_LOOSE = re.compile(r"(?<![\d])(\d{2})(\d{2})(\d{2})(?![\d\w])")
_MONTH_NUM = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}


def date_from_filename(name: str) -> str | None:
    """Publication date encoded in a ReliefWeb PDF filename (the most reliable signal).

    Handles `ROSEA_20190506_...`, `..._190330.pdf`, `..._12032019.pdf` and a spelled-out
    `... 19 April 2019 ...`. Returns YYYY-MM-DD, or None when the name carries no date.
    """
    stem = Path(name).stem
    m = _FN_YYYYMMDD.search(stem)
    if m:
        iso = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
        return iso if _plausible(iso) else None
    m = _FN_DDMMYYYY.search(stem)
    if m:
        iso = f"{m.group(3)}-{m.group(2)}-{m.group(1)}"
        if _plausible(iso):
            return iso
    m = _FN_YYMMDD.search(stem)
    if m:
        iso = f"20{m.group(1)}-{m.group(2)}-{m.group(3)}"
        if _plausible(iso):
            return iso
    m = _FN_DMY_WORD.search(re.sub(r"[_\-]+", " ", stem))  # '10_may_2019' -> '10 may 2019'
    if m:
        iso = parse_date(f"{m.group(1)} {m.group(2)} {m.group(3)}")
        if iso and _plausible(iso):
            return iso
    m = _FN_ISO_SEP.search(stem)
    if m:
        iso = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
        if _plausible(iso):
            return iso
    m = _FN_MDY_WORD.search(re.sub(r"[_\-]+", " ", stem))  # 'Sep 7, 2017'
    if m:
        mo = _MONTH_NUM[m.group(1)[:3].lower()]
        iso = f"{m.group(3)}-{mo:02d}-{int(m.group(2)):02d}"
        if _plausible(iso):
            return iso
    m = _FN_DDMMYY_LOOSE.search(stem)
    if m:
        a, b, c = m.group(1), m.group(2), m.group(3)
        yymmdd = f"20{a}-{b}-{c}"   # 171011 -> 2017-10-11
        ddmmyy = f"20{c}-{b}-{a}"   # 080917 -> 2017-09-08
        for iso in (yymmdd, ddmmyy):
            if _plausible(iso) and _modern(iso):
                return iso
        for iso in (yymmdd, ddmmyy):
            if _plausible(iso):
                return iso
    return parse_date_in(stem)


def _modern(iso: str) -> bool:
    """Year falls in the era these corpora come from (HumAID spans 2016-2019).

    Used only to disambiguate a six-digit filename date, where YYMMDD and DDMMYY are both
    syntactically valid and one of them silently produces a date years off the event.
    """
    try:
        return 2015 <= int(iso.split("-")[0]) <= 2030
    except (ValueError, IndexError):
        return False


def _plausible(iso: str) -> bool:
    """Guard against nonsense produced by a number that is not a date."""
    try:
        y, mo, d = (int(x) for x in iso.split("-"))
    except ValueError:
        return False
    return 2000 <= y <= 2100 and 1 <= mo <= 12 and 1 <= d <= 31


def parse_date_in(text: str) -> str | None:
    """First `<d> <Month> <yyyy>` anywhere in `text`, normalised (None when absent)."""
    m = _DATE_IN_TEXT.search(text or "")
    return parse_date(m.group(1).replace(".", "")) if m else None


EVENT_DATE_SLACK_DAYS = 180


def event_date_window(event: str, streams_root: Path | None = None) -> tuple[str, str] | None:
    """Plausible publication window for `event`, from the tweet stream we hold for it.

    A date parsed out of a report's *body* is unreliable — the body may quote an earlier
    disaster, a funding-appeal year, or a photo caption. Dates from the filename are trusted;
    everything else is checked against this window so a year-level error surfaces as `undated`
    (which the scan report lists for a human) instead of silently landing in the corpus.

    The window is generous (`EVENT_DATE_SLACK_DAYS` either side) because late situation reports
    are legitimate and valuable for schema induction — we are excluding *wrong years*, not late
    reports. Returns None when we hold no stream for the event, in which case no check is made.
    """
    from datetime import date as _date, timedelta

    root = streams_root or (ROOT / "data" / "processed" / "streams")
    d = resolve_event_dir(root, event)
    if d is None:
        return None
    days = sorted(p.stem for p in d.glob("*.jsonl"))
    if not days:
        return None
    try:
        lo = _date.fromisoformat(days[0]) - timedelta(days=EVENT_DATE_SLACK_DAYS)
        hi = _date.fromisoformat(days[-1]) + timedelta(days=EVENT_DATE_SLACK_DAYS)
    except ValueError:
        return None
    return lo.isoformat(), hi.isoformat()


def publication_date(text: str, filename: str, manifest_date: str | None = None) -> str | None:
    """Best publication date: manifest > filename > 'as of' in the body > first date in the body.

    The 'next report' footer is excluded explicitly — taking the first date in an OCHA sitrep
    otherwise dates the report several days late.
    """
    if manifest_date:
        iso = parse_date(manifest_date)
        if iso:
            return iso
    iso = date_from_filename(filename)
    if iso:
        return iso
    for m in _AS_OF_RE.finditer(text or ""):
        window = (text or "")[max(0, m.start() - 60):m.start()]
        if _NEXT_REPORT_RE.search(window):
            continue
        iso = parse_date(m.group(1).replace(".", ""))
        if iso:
            return iso
    return date_from_text(text)


def title_from_filename(name: str) -> str:
    """Turn a ReliefWeb PDF filename into a readable title (last-resort but usually decent)."""
    stem = Path(name).stem
    stem = re.sub(r"(?<!\d)(20\d{6}|\d{2}\d{2}20\d{2})(?!\d)", " ", stem)
    stem = re.sub(r"\b(final|for upload|compressed|v\d+)\b", " ", stem, flags=re.IGNORECASE)
    stem = re.sub(r"[_\-]+", " ", stem)
    stem = re.sub(r"\s*\(\d+\)\s*$", "", stem)      # browser "(1)" duplicate marker
    stem = re.sub(r"[.\s]{2,}", " ", stem).strip(" .-")
    return re.sub(r"\s{2,}", " ", stem)


REPORT_WORDS = ("sitrep", "situation", "flash", "update", "snapshot", "report", "appeal",
                "bulletin", "dref", "epoa", "response")


def informative_title(candidate: str) -> bool:
    """True when a filename-derived title actually names the report (not e.g. 'MDRMZ014do')."""
    low = candidate.lower()
    return len(candidate) >= 12 and (len(candidate.split()) >= 3 or any(w in low for w in REPORT_WORDS))


def is_boilerplate(line: str) -> bool:
    """True for template furniture or a content line that must not be used as a report title.

    Covers three cases seen in real OCHA/UN PDFs: the mission-statement footer, a source-file
    artifact left in the PDF metadata (`....ai`, `Microsoft Word - ...`), and a bullet from the
    Highlights box (the first text line of every OCHA Flash Update).
    """
    line = line.strip()
    low = line.lower()
    if any(marker in low for marker in BOILERPLATE_MARKERS):
        return True
    if line[:1] in {"•", "-", "–", "*", "▪", "●"}:  # a highlights bullet
        return True
    return low.endswith((".ai", ".indd", ".docx", ".doc", ".pdf", ".pptx"))


def infer_source(text: str, title: str | None = None, url: str | None = None,
                 filename: str | None = None) -> str:
    """Best-effort publisher short name from a report's text/title/url ('' when unknown)."""
    if filename:
        low_name = Path(filename).name.lower()
        for needle, short in FILENAME_SOURCE_HINTS:
            if needle in low_name:
                return short
    # The publisher names itself in the header/footer; a donor list sits in the body, so weight
    # the first and last chunk of the document rather than scanning straight through.
    body = text or ""
    haystack = " ".join(x for x in (title or "", url or "", body[:1200], body[-1200:]) if x).lower()
    for needle, short in SOURCE_PATTERNS:
        if needle.lower() in haystack:
            return short
    return ""


def date_from_text(text: str) -> str | None:
    """First 'as of <date>'-style date in the text, normalised to YYYY-MM-DD."""
    for m in _DATE_IN_TEXT.finditer(text or ""):
        iso = parse_date(m.group(1).replace(".", ""))
        if iso:
            return iso
    return None


def url_from_html(html: str) -> str | None:
    """Canonical ReliefWeb URL of a saved page (og:url / <link rel=canonical>)."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    og = soup.find("meta", attrs={"property": "og:url"})
    if og and og.get("content"):
        return str(og["content"]).strip()
    link = soup.find("link", attrs={"rel": ["canonical", "Canonical"]})
    if link and link.get("href"):
        return str(link["href"]).strip()
    return None


def scan_folder(folder: Path, base: Path, *, event: str | None = None,
                min_chars: int = MIN_TEXT_CHARS) -> list[dict[str, str]]:
    """Infer a manifest row for every saved report in `folder` (recursively).

    `event` defaults to the folder name. Title/date/source/url are read from the file itself:
    HTML pages carry og:title / article:published_time / og:url; PDFs and text files are parsed
    for the title line and an 'as of <date>' string. Files that yield too little text are skipped
    with a warning (usually a mis-saved page)."""
    rows: list[dict[str, str]] = []
    ev = event or folder.name
    for path in sorted(folder.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in (".html", ".htm", ".pdf", ".txt", ".md"):
            continue
        try:
            ext = extract_file(path)
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop the scan
            log.warning("scan: cannot read %s (%s) — skipped", path.name, exc)
            continue
        text = ext.get("text") or ""
        if len(text) < min_chars:
            log.warning("scan: %s has only %d chars of text (<%d) — skipped; re-save it as PDF",
                        path.name, len(text), min_chars)
            continue
        url = ""
        if path.suffix.lower() in (".html", ".htm"):
            url = url_from_html(path.read_text(encoding="utf-8", errors="replace")) or ""
        # Titles: for PDFs the filename is the most reliable source — multi-column text extraction
        # scrambles line order, so "first non-boilerplate line" lands on an arbitrary sentence.
        # HTML saves carry a real <title>/og:title, so those win when present.
        title = ext.get("title") or ""
        fn_title = title_from_filename(path.name)
        if path.suffix.lower() == ".pdf" and informative_title(fn_title):
            title = fn_title
        elif not title or is_boilerplate(title):
            candidate = _title_from_text(text)
            title = candidate if candidate and not is_boilerplate(candidate) else fn_title
        date = publication_date(text, path.name, ext.get("date")) or ""
        rows.append({"event": ev, "title": title, "source": infer_source(text, title, url, path.name),
                     "date": date, "url": url,
                     "file": path.relative_to(base).as_posix()})
        log.info("scan: %s -> %s | %s | %s", path.name, (date or "?"), (rows[-1]["source"] or "?"), title[:60])
    return rows


def _title_from_text(text: str) -> str | None:
    """First non-trivial line of a PDF/text report, used as its title."""
    for line in (text or "").splitlines():
        line = line.strip()
        if not (15 <= len(line) <= 200):
            continue
        if line.lower().startswith(("http", "page ")) or is_boilerplate(line):
            continue
        return line
    return None


def write_manifest(manifest_path: Path, rows: list[dict[str, str]], *, merge: bool = True) -> int:
    """Write manifest rows, keeping any existing rows for other files (match on `file`)."""
    existing: list[dict[str, str]] = []
    if merge and manifest_path.exists():
        try:
            existing = load_manifest(manifest_path)
        except (OSError, ValueError):
            existing = []
    by_file = {r["file"]: r for r in existing}
    for r in rows:  # a rescan refreshes rows for the same file, keeps manual edits to others
        by_file[r["file"]] = r
    ordered = sorted(by_file.values(), key=lambda r: (r.get("event", ""), r.get("date", ""), r.get("file", "")))
    ensure_dir(manifest_path.parent)
    with open(manifest_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_COLUMNS)
        w.writeheader()
        w.writerows([{c: r.get(c, "") for c in MANIFEST_COLUMNS} for r in ordered])
    return len(ordered)


def extract_html(html: str) -> dict[str, Any]:
    """Extract {title, date, text} from a saved ReliefWeb report page."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")

    # Metadata that lives in <head> is read first (unaffected by noise removal).
    title = None
    og = soup.find("meta", attrs={"property": "og:title"})
    if og and og.get("content"):
        title = og["content"]
    elif soup.title and soup.title.string:
        title = soup.title.string
    date = None
    pub = soup.find("meta", attrs={"property": "article:published_time"})
    if pub and pub.get("content"):
        date = _iso_date(pub["content"])

    # Drop site chrome BEFORE any body-level fallback so a header <h1>/<time> is never picked up.
    for tag in soup(list(NOISE_TAGS)):
        tag.decompose()

    if not title and soup.find("h1"):
        title = soup.find("h1").get_text(" ")
    title = _clean_title(title)
    if not date:
        t = soup.find("time", attrs={"datetime": True})
        if t:
            date = _iso_date(t["datetime"])

    body = None
    for sel in BODY_SELECTORS:
        try:
            body = soup.select_one(sel)
        except Exception:  # invalid selector for this parser — skip
            body = None
        if body is not None:
            break
    if body is None:
        body = _densest_block(soup)
    text = soup_text(body) if body is not None else ""
    return {"title": title, "date": date, "text": text}


def _densest_block(soup: Any) -> Any:
    """Fallback: block element with the most direct-paragraph text; else <body>."""
    best, best_len = None, 0
    for el in soup.find_all(["div", "section", "article", "main", "td"]):
        n = sum(len(p.get_text()) for p in el.find_all("p", recursive=False))
        if n > best_len:
            best, best_len = el, n
    return best if best is not None else (soup.body or soup)


def extract_pdf(path: Path) -> dict[str, Any]:
    """Extract text from a PDF via pypdf (imported lazily)."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pypdf is required for PDF ingest: pip install pypdf") from exc
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception as exc:  # pragma: no cover - malformed page
            log.warning("pypdf failed on a page of %s: %s", path.name, exc)
    meta_title = None
    try:
        meta_title = (reader.metadata or {}).get("/Title") if reader.metadata else None
    except Exception:  # pragma: no cover
        meta_title = None
    return {"title": str(meta_title) if meta_title else None, "date": None, "text": _collapse("\n\n".join(pages))}


def extract_text_file(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return {"title": None, "date": None, "text": _collapse(text)}


def extract_file(path: Path) -> dict[str, Any]:
    """Dispatch on suffix (.html/.htm, .pdf, .txt/.md)."""
    suf = path.suffix.lower()
    if suf in (".html", ".htm"):
        return extract_html(path.read_text(encoding="utf-8", errors="replace"))
    if suf == ".pdf":
        return extract_pdf(path)
    if suf in (".txt", ".md"):
        return extract_text_file(path)
    raise ValueError(f"unsupported file type {suf!r} for {path.name} (use .html/.htm, .pdf, .txt/.md)")


# ---------------------------------------------------------------------------
# Manifest + ingest
# ---------------------------------------------------------------------------
def ensure_manifest(manifest_path: Path) -> bool:
    """Create the manifest (from docs template) + README if missing. Returns True if created."""
    if manifest_path.exists():
        return False
    ensure_dir(manifest_path.parent)
    if TEMPLATE_PATH.exists():
        shutil.copyfile(TEMPLATE_PATH, manifest_path)
    else:  # pragma: no cover - template ships with the repo
        manifest_path.write_text(",".join(MANIFEST_COLUMNS) + "\n", encoding="utf-8")
    readme = manifest_path.parent / "README.md"
    if not readme.exists():
        readme.write_text(README_TEXT, encoding="utf-8")
    return True


def load_manifest(manifest_path: Path) -> list[dict[str, str]]:
    """Read manifest rows (skips blank / comment rows); validates the header."""
    with open(manifest_path, "r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        header = [h.strip() for h in (reader.fieldnames or [])]
        missing = [c for c in MANIFEST_COLUMNS if c not in header]
        if missing:
            raise ValueError(f"manifest {manifest_path} is missing columns {missing}; expected {MANIFEST_COLUMNS}")
        rows = []
        for raw in reader:
            row = {k.strip(): (v or "").strip() for k, v in raw.items() if k}
            if not any(row.values()) or row.get("event", "").startswith("#"):
                continue
            rows.append(row)
    return rows


def _unique_path(directory: Path, stem: str, taken: set[Path]) -> Path:
    """Deterministic name: <date>.json for the first record of that date IN THIS RUN, then
    <date>_2.json, <date>_3.json … Re-running with the same manifest overwrites the same
    files (idempotent); files on disk from earlier runs are not treated as collisions."""
    cand = directory / f"{stem}.json"
    k = 1
    while cand in taken:
        k += 1
        cand = directory / f"{stem}_{k}.json"
    return cand


def _previously_written(event_dir: Path) -> set[str]:
    """Output file names recorded by a previous run (from _manifest_ingested.csv)."""
    mpath = event_dir / "_manifest_ingested.csv"
    if not mpath.exists():
        return set()
    try:
        with open(mpath, "r", encoding="utf-8", newline="") as fh:
            return {row.get("out_file", "") for row in csv.DictReader(fh) if row.get("out_file")}
    except (OSError, ValueError):
        return set()


_MISSING = object()


def ingest(manifest_path: Path, out_dir: Path, *, min_chars: int = MIN_TEXT_CHARS) -> dict[str, Any]:
    """Ingest every manifest row; returns a summary dict (counts, skipped, written)."""
    rows = load_manifest(manifest_path)
    base = manifest_path.parent
    counts: Counter[str] = Counter()
    skipped: list[dict[str, str]] = []
    written: list[Path] = []
    taken: set[Path] = set()
    seen_ids: set[str] = set()
    seen_text: dict[str, str] = {}
    windows: dict[str, tuple[str, str] | None] = {}
    per_event_rows: dict[str, list[dict[str, Any]]] = {}

    for i, row in enumerate(rows, start=2):  # line numbers for messages (header = 1)
        event = slugify(row.get("event") or "") if row.get("event") else ""
        rel = row.get("file", "")
        if not event or not rel:
            log.warning("manifest line %d: missing event or file — skipped", i)
            skipped.append({**row, "reason": "missing event/file"})
            continue
        path = Path(rel)
        if not path.is_absolute():
            path = base / rel
        if not path.exists():
            log.warning("manifest line %d: file not found %s — skipped", i, path)
            skipped.append({**row, "reason": f"file not found: {path}"})
            continue
        try:
            ext = extract_file(path)
        except Exception as exc:
            log.warning("manifest line %d: extraction failed for %s: %s — skipped", i, path.name, exc)
            skipped.append({**row, "reason": f"extract error: {exc}"})
            continue
        text = ext.get("text") or ""
        if len(text) < min_chars:
            log.warning("manifest line %d: %s has only %d chars of text (<%d) — skipped", i, path.name, len(text), min_chars)
            skipped.append({**row, "reason": f"text too short ({len(text)} chars)"})
            continue

        manifest_date = row.get("date") or ""
        date = _iso_date(manifest_date)
        if manifest_date and not date:
            log.warning("manifest line %d: could not parse date %r (use YYYY-MM-DD); falling back to the page date",
                        i, manifest_date)
        date = date or ext.get("date") or "undated"
        title = row.get("title") or ext.get("title") or path.stem
        url = row.get("url") or ""
        rec_id = "manual-" + slugify(url or rel, max_len=80)
        if rec_id in seen_ids:
            log.warning("manifest line %d: duplicate record id %s (same url/file listed twice) — skipped", i, rec_id)
            skipped.append({**row, "reason": f"duplicate id {rec_id}"})
            continue
        # Content dedupe: browsers save a second copy as "name (1).pdf". Two files with identical
        # extracted text are the same report and must not be counted twice in prevalence stats.
        # Key on date + text: a browser "(1)" copy repeats both, whereas two genuinely different
        # reports in a series can share boilerplate-heavy text but never share a publication date.
        # Reject a date that is years away from the event: it came from the document body and
        # describes something other than this report's publication (see event_date_window).
        if date and date != "undated":
            win = windows.get(event, _MISSING)
            if win is _MISSING:
                win = windows[event] = event_date_window(event)
            if win and not (win[0] <= date <= win[1]):
                log.warning("manifest line %d: %s date %s is outside the %s window %s..%s — "
                            "treating as undated (correct it in the manifest to keep it)",
                            i, path.name, date, event, win[0], win[1])
                date = ""

        text_hash = sha256_of(date + "|" + re.sub(r"\s+", " ", text).strip().lower())
        if text_hash in seen_text:
            first = seen_text[text_hash]
            log.warning("manifest line %d: %s has the same text as %s — skipped as a duplicate copy",
                        i, path.name, first)
            skipped.append({**row, "reason": f"duplicate content of {first}"})
            continue
        seen_text[text_hash] = path.name
        seen_ids.add(rec_id)
        event_dir = ensure_dir(out_dir / event)
        out_path = _unique_path(event_dir, date, taken)
        taken.add(out_path)
        record = {
            "id": rec_id,
            "title": title,
            "source": row.get("source") or "unknown",
            "date": date,
            "url": url,
            "text": text,
            "collection": "manual",
            "file": path.relative_to(base).as_posix() if path.is_relative_to(base) else path.as_posix(),
            "ingested_at": utc_now_iso(),
        }
        write_json(out_path, record)
        written.append(out_path)
        counts[event] += 1
        per_event_rows.setdefault(event, []).append(
            {**{c: row.get(c, "") for c in MANIFEST_COLUMNS}, "id": rec_id, "out_file": out_path.name, "chars": len(text)}
        )
        log.info("ingested %s → %s (%d chars)", path.name, out_path.relative_to(out_dir), len(text))

    for event, erows in per_event_rows.items():
        # Remove files this script wrote in an earlier run that the current manifest no longer produces.
        current = {r["out_file"] for r in erows}
        for stale in _previously_written(out_dir / event) - current:
            sp = out_dir / event / stale
            if sp.exists() and sp.suffix == ".json":
                sp.unlink()
                log.info("removed stale output %s/%s (no longer in manifest)", event, stale)
        mpath = out_dir / event / "_manifest_ingested.csv"
        with open(mpath, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=MANIFEST_COLUMNS + ["id", "out_file", "chars"])
            w.writeheader()
            w.writerows(erows)

    return {"rows": len(rows), "counts": dict(counts), "skipped": skipped, "written": [str(p) for p in written]}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Ingest manually saved ReliefWeb reports into data/processed/sitreps_human/")
    ap.add_argument("--manifest", type=Path, default=None, help="manifest CSV (default data/raw/reliefweb_manual/manifest.csv)")
    ap.add_argument("--out", type=Path, default=None, help="output dir (default data/processed/sitreps_human)")
    ap.add_argument("--min-chars", type=int, default=MIN_TEXT_CHARS, help="skip rows with less extracted text")
    ap.add_argument("--scan", type=Path, default=None, metavar="DIR",
                    help="build/refresh manifest rows from the saved reports in DIR, then exit")
    ap.add_argument("--event", type=str, default=None, help="event key for --scan (default: the folder name)")
    args = ap.parse_args(argv)
    try:  # Windows consoles/pipes may use a legacy code page
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):  # pragma: no cover
        pass

    cfg = load_config()
    manifest = args.manifest or (cfg_path(cfg, "raw", "data/raw") / "reliefweb_manual" / "manifest.csv")
    out_dir = args.out or (cfg_path(cfg, "processed", "data/processed") / "sitreps_human")

    if args.scan is not None:
        folder = args.scan if args.scan.is_absolute() else (Path.cwd() / args.scan)
        if not folder.is_dir():
            print(f"--scan: {folder} is not a directory")
            return 2
        ensure_manifest(manifest)
        rows = scan_folder(folder, manifest.parent, event=args.event, min_chars=args.min_chars)
        if not rows:
            print(f"No readable reports found under {folder}.\n"
                  f"Save ReliefWeb reports there as PDF (preferred) or 'Webpage, HTML only'.")
            return 2
        total = write_manifest(manifest, rows)
        missing_date = [r for r in rows if not r["date"]]
        missing_src = [r for r in rows if not r["source"]]
        missing_url = [r for r in rows if not r["url"]]
        print(f"Scanned {len(rows)} report(s) into {manifest} ({total} row(s) total).")
        for label, bad in (("date", missing_date), ("source", missing_src), ("url", missing_url)):
            if bad:
                print(f"  {len(bad)} row(s) need a {label} filled in by hand: "
                      + ", ".join(Path(r["file"]).name for r in bad[:5])
                      + (" ..." if len(bad) > 5 else ""))
        print("Review the manifest, then run: python -m src.ingest_manual_reliefweb")
        return 0

    if ensure_manifest(manifest):
        print(f"Created empty manifest at {manifest} (and README.md next to it).")
    try:
        summary = ingest(manifest, out_dir, min_chars=args.min_chars)
    except (ValueError, OSError) as exc:
        print(f"error: {exc}")
        return 1
    if summary["rows"] == 0:
        print(
            f"Manifest {manifest} has no rows yet — nothing to ingest.\n"
            "Save ReliefWeb reports (PDF or 'HTML only') under "
            f"{manifest.parent} and add one manifest row per report "
            "(event,title,source,date,url,file). See docs/manual_collection.md."
        )
        return 0
    print("Per-event ingested counts:")
    for event, n in sorted(summary["counts"].items()):
        print(f"  {event:<40} {n:>4}")
    total = sum(summary["counts"].values())
    print(f"Total: {total} sitreps written to {out_dir}; skipped {len(summary['skipped'])} row(s).")
    for s in summary["skipped"]:
        print(f"  skipped: {s.get('file', '?')} — {s.get('reason')}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
