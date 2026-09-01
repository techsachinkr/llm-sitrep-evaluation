"""ReliefWeb API v2 client (reports / disasters) with polite rate limiting,
raw-response disk cache, retry/backoff and a persisted daily call counter.

Usage (library)::

    from src.reliefweb_api import ReliefWebClient, normalize_report
    rw = ReliefWebClient()                       # appname from RELIEFWEB_APPNAME
    hits = rw.search_disasters("Nepal earthquake 2015")
    reports = rw.fetch_reports(disaster_id=hits[0]["id"], max_total=50)
    records = [normalize_report(r) for r in reports]

Usage (CLI)::

    python -m src.reliefweb_api --smoke --query "Nepal earthquake 2015" --n 10

Policy (CLAUDE.md rule 8): the appname must be pre-approved by ReliefWeb; if it
is not set we do NOT call the API and instead print the manual-collection
fallback message. Quotas: <=1000 entries/call, <=1000 calls/day, <=1 req/s.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import re
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.parse import quote

import httpx
from tenacity import (
    RetryCallState,
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from src.util import (
    append_jsonl,
    cfg_path,
    ensure_dir,
    env,
    get_logger,
    load_config,
    read_json,
    read_jsonl,
    sha256_of,
    utc_now_iso,
    write_json,
)

log = get_logger("reliefweb")

MANUAL_FALLBACK_MESSAGE = """RELIEFWEB_APPNAME is not set (or not yet approved). ReliefWeb API v2 requires a pre-approved appname (since 2025-11-01) — do not query with an unregistered one. Request one via https://apidoc.reliefweb.int/ (or feedback@reliefweb.int) and put it in .env as RELIEFWEB_APPNAME=... Meanwhile use the manual-collection fallback: see docs/manual_collection.md and run `python -m src.ingest_manual_reliefweb`."""

DEFAULT_REPORT_FIELDS: list[str] = [
    "id", "title", "url", "url_alias", "date.original", "date.created",
    "source.name", "source.shortname", "format.name", "disaster.id",
    "disaster.name", "disaster.glide", "primary_country.iso3",
    "primary_country.name", "language.code", "body", "body-html",
    "file.url", "file.filename",
]

MAX_LIMIT = 1000  # ReliefWeb hard cap on entries per call


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class ReliefWebError(Exception):
    """Base class for ReliefWeb client errors."""


class ReliefWebAccessError(ReliefWebError):
    """Missing / rejected appname (the API refuses unregistered appnames)."""


class ReliefWebQuotaError(ReliefWebError):
    """Daily network-call quota (default 1000/day) would be exceeded."""


class ReliefWebRetryableError(ReliefWebError):
    """HTTP 429 / 5xx — retried with backoff (honouring Retry-After)."""

    def __init__(self, status: int, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _valid_appname(value: str | None) -> str | None:
    """Return a plausible appname or None.

    Guards against `.env` lines like ``RELIEFWEB_APPNAME=   # comment`` which
    python-dotenv can parse as the literal comment text.
    """
    if not value:
        return None
    v = value.strip()
    if not v or v.startswith("#") or not re.fullmatch(r"[A-Za-z0-9._@:/-]+", v):
        return None
    return v


def _parse_retry_after(value: str | None) -> float | None:
    """Parse a Retry-After header (seconds or HTTP-date) into seconds."""
    if value is None:
        return None
    value = value.strip()
    if re.fullmatch(r"\d+(\.\d+)?", value):
        return float(value)
    try:  # HTTP-date
        from email.utils import parsedate_to_datetime

        dt = parsedate_to_datetime(value)
        return max(0.0, (dt - datetime.now(timezone.utc)).total_seconds())
    except Exception:
        return None


def _today_utc() -> str:
    """Today's date in UTC (the quota day)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _atomic_write_json(path: Path, obj: Any) -> None:
    """Write JSON via a unique temp file + os.replace; tolerate a concurrent identical writer."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.stem, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, ensure_ascii=False, indent=1, sort_keys=True)
        try:
            os.replace(tmp, path)
        except OSError:
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


def _default_cache_dir() -> Path:
    try:
        cfg = load_config()
    except Exception:  # pragma: no cover - config missing is unusual
        cfg = {}
    return cfg_path(cfg, "raw", "data/raw") / "reliefweb"


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------
class ReliefWebClient:
    """Small, polite ReliefWeb API v2 client.

    * ``>= min_interval_s`` between network calls (monotonic clock + lock)
    * raw responses cached under ``<cache_dir>/<endpoint>/<sha256(body)>.json``
    * retries on 429/5xx with exponential backoff (<=4 attempts, Retry-After honoured)
    * network calls counted per UTC day in ``<cache_dir>/_calls.jsonl``; refuses
      to exceed ``max_calls_per_day`` (ReliefWebQuotaError)
    """

    BASE = "https://api.reliefweb.int/v2"

    def __init__(
        self,
        appname: str | None = None,
        *,
        cache_dir: str | Path | None = None,
        min_interval_s: float = 1.0,
        transport: httpx.BaseTransport | None = None,
        max_calls_per_day: int = 1000,
        timeout_s: float = 60.0,
        max_attempts: int = 4,
    ) -> None:
        self.appname = _valid_appname(appname if appname is not None else env("RELIEFWEB_APPNAME"))
        if not self.appname:
            raise ReliefWebAccessError(MANUAL_FALLBACK_MESSAGE)
        self.cache_dir = ensure_dir(Path(cache_dir) if cache_dir else _default_cache_dir())
        self.min_interval_s = float(min_interval_s)
        self.max_calls_per_day = int(max_calls_per_day)
        self.max_attempts = max(1, int(max_attempts))
        self._counter_file = self.cache_dir / "_calls.jsonl"
        self._lock = threading.Lock()
        self._last_request_at: float | None = None
        self._counter_day = _today_utc()
        self._calls_today = self._load_calls_today()
        self.network_calls = 0  # network calls made by this process
        self._client = httpx.Client(
            transport=transport,
            timeout=timeout_s,
            headers={
                "User-Agent": f"sitrep-gap/0.1 ({self.appname})",
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
        )

    # -- housekeeping ------------------------------------------------------
    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "ReliefWebClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _load_calls_today(self) -> int:
        """Count today's (UTC) network calls in the persisted counter file (corrupt lines skipped)."""
        today = _today_utc()
        self._counter_day = today
        n = 0
        if not self._counter_file.exists():
            return 0
        with open(self._counter_file, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    log.warning("skipping corrupt line in %s", self._counter_file)
                    continue
                if isinstance(row, dict) and not row.get("cached") and str(row.get("ts", ""))[:10] == today:
                    n += 1
        return n

    def _record_call(self, endpoint: str, status: int | None, cached: bool) -> None:
        append_jsonl(
            self._counter_file,
            {"ts": utc_now_iso(), "endpoint": endpoint, "status": status, "cached": cached},
        )

    def _cache_file(self, endpoint: str, body: dict[str, Any]) -> Path:
        return self.cache_dir / endpoint.strip("/") / f"{sha256_of(body)}.json"

    def _url(self, endpoint: str) -> str:
        """Endpoint URL with the (percent-encoded) appname query parameter."""
        return f"{self.BASE}/{endpoint.strip('/')}?appname={quote(self.appname, safe='')}"

    # -- network -----------------------------------------------------------
    def _wait_for_slot(self) -> None:
        """Block until >= min_interval_s has elapsed since the last request."""
        if self._last_request_at is not None:
            elapsed = time.monotonic() - self._last_request_at
            remaining = self.min_interval_s - elapsed
            if remaining > 0:
                time.sleep(remaining)

    def _send_once(self, endpoint: str, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        """One network attempt (quota check, rate limit, request, counter row)."""
        with self._lock:
            if _today_utc() != self._counter_day:  # UTC midnight rollover for long-lived processes
                self._calls_today = self._load_calls_today()
            if self._calls_today >= self.max_calls_per_day:
                raise ReliefWebQuotaError(
                    f"ReliefWeb daily quota reached ({self._calls_today}/{self.max_calls_per_day} "
                    f"network calls today, counter at {self._counter_file}); try again tomorrow "
                    "or rely on the cache."
                )
            self._wait_for_slot()
            self._calls_today += 1
            self.network_calls += 1
            status: int | None = None
            try:
                resp = self._client.post(self._url(endpoint), json=body)
                status = resp.status_code
            finally:
                self._last_request_at = time.monotonic()
                self._record_call(endpoint, status, cached=False)

        if status == 429 or status >= 500:
            raise ReliefWebRetryableError(
                status,
                f"ReliefWeb {endpoint} returned HTTP {status}",
                retry_after=_parse_retry_after(resp.headers.get("Retry-After")),
            )
        if status in (401, 403):
            raise ReliefWebAccessError(
                f"ReliefWeb rejected the request (HTTP {status}) — is RELIEFWEB_APPNAME "
                f"'{self.appname}' registered and approved? Body: {resp.text[:300]}\n\n"
                + MANUAL_FALLBACK_MESSAGE
            )
        if status >= 400:
            raise ReliefWebError(f"ReliefWeb {endpoint} HTTP {status}: {resp.text[:500]}")
        try:
            data = resp.json()
        except ValueError as exc:
            raise ReliefWebError(f"ReliefWeb {endpoint}: non-JSON response ({exc})") from exc
        if not isinstance(data, dict):
            raise ReliefWebError(f"ReliefWeb {endpoint}: unexpected response type {type(data).__name__}")
        return status, data

    @staticmethod
    def _retry_wait(retry_state: RetryCallState) -> float:
        """Honour Retry-After when present, else exponential backoff with jitter."""
        exc = retry_state.outcome.exception() if retry_state.outcome else None
        ra = getattr(exc, "retry_after", None)
        if ra is not None:
            return min(float(ra), 120.0)
        return wait_exponential_jitter(initial=1.0, max=30.0)(retry_state)

    def _send_with_retry(self, endpoint: str, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        retrying = Retrying(
            retry=retry_if_exception_type((ReliefWebRetryableError, httpx.TransportError)),
            wait=self._retry_wait,
            stop=stop_after_attempt(self.max_attempts),
            reraise=True,
            before_sleep=lambda rs: log.warning(
                "ReliefWeb %s attempt %d failed (%s); retrying",
                endpoint, rs.attempt_number, rs.outcome.exception() if rs.outcome else "?",
            ),
        )
        try:
            return retrying(self._send_once, endpoint, body)
        except httpx.TransportError as exc:
            raise ReliefWebError(
                f"ReliefWeb {endpoint}: network failure after {self.max_attempts} attempt(s): "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    # -- public ------------------------------------------------------------
    def post(self, endpoint: str, body: dict[str, Any], *, use_cache: bool = True) -> dict[str, Any]:
        """POST a JSON query to ``/v2/<endpoint>`` and return the parsed response.

        Cache hits (same endpoint + identical body) never touch the network.
        """
        cache_file = self._cache_file(endpoint, body)
        if use_cache and cache_file.exists():
            try:
                cached = read_json(cache_file)
                response = cached["response"]
                self._record_call(endpoint, cached.get("status"), cached=True)
                log.debug("cache hit %s %s", endpoint, cache_file.name)
                return response
            except (ValueError, KeyError, OSError):
                log.warning("corrupt cache file %s — refetching", cache_file)
        status, data = self._send_with_retry(endpoint, body)
        try:
            _atomic_write_json(
                cache_file,
                {"endpoint": endpoint, "body": body, "fetched_at": utc_now_iso(), "status": status, "response": data},
            )
        except OSError as exc:  # never lose a fetched response because the cache could not be written
            log.warning("could not write ReliefWeb cache file %s: %s", cache_file, exc)
        return data

    def search_disasters(self, query: str, *, limit: int = 20) -> list[dict[str, Any]]:
        """Search the disasters endpoint by name; returns flattened fields dicts."""
        body = {
            "query": {"value": query, "fields": ["name"]},
            "fields": {"include": ["id", "name", "glide", "status", "date.event", "country.iso3", "type.name"]},
            "limit": min(int(limit), MAX_LIMIT),
            "preset": "analysis",
        }
        data = self.post("disasters", body)
        return [_flatten_item(it) for it in data.get("data", [])]

    def fetch_reports(
        self,
        *,
        disaster_id: int | str | None = None,
        query: str | None = None,
        formats: Sequence[str] | None = ("Situation Report",),
        sources: Sequence[str] | None = ("OCHA", "IFRC"),
        date_from: str | None = None,
        date_to: str | None = None,
        limit: int = MAX_LIMIT,
        max_total: int | None = None,
        fields: Sequence[str] | None = None,
        sort: Sequence[str] = ("date.original:asc",),
        language: str | None = "en",
    ) -> list[dict[str, Any]]:
        """Fetch reports matching the filters, paginating by offset.

        Returns a list of ``item['fields']`` dicts augmented with ``id``/``href``.
        """
        limit = max(1, min(int(limit), MAX_LIMIT))
        if max_total is not None:
            limit = min(limit, max(1, int(max_total)))
        conditions: list[dict[str, Any]] = []
        if formats:
            conditions.append({"field": "format.name", "value": list(formats), "operator": "OR"})
        if sources:
            conditions.append({"field": "source.shortname", "value": list(sources), "operator": "OR"})
        if disaster_id is not None:
            conditions.append({"field": "disaster.id", "value": int(disaster_id)})
        if language:
            conditions.append({"field": "language.code", "value": language})
        if date_from or date_to:
            rng: dict[str, str] = {}
            if date_from:
                rng["from"] = _iso_datetime(date_from)
            if date_to:
                rng["to"] = _iso_datetime(date_to, end_of_day=True)
            conditions.append({"field": "date.original", "value": rng})

        out: list[dict[str, Any]] = []
        offset = 0
        while True:
            body: dict[str, Any] = {
                "filter": {"operator": "AND", "conditions": conditions},
                "fields": {"include": list(fields) if fields is not None else list(DEFAULT_REPORT_FIELDS)},
                "sort": list(sort),
                "limit": limit,
                "offset": offset,
            }
            if query:
                body["query"] = {"value": query}
            data = self.post("reports", body)
            items = data.get("data") or []
            if not items:
                break
            out.extend(_flatten_item(it) for it in items)
            offset += len(items)
            total = data.get("totalCount")
            if max_total is not None and len(out) >= max_total:
                out = out[:max_total]
                break
            if len(items) < limit or (isinstance(total, int) and offset >= total):
                break
        log.info("fetch_reports: %d reports (disaster_id=%s query=%r)", len(out), disaster_id, query)
        return out


def _iso_datetime(value: str, *, end_of_day: bool = False) -> str:
    """Accept 'YYYY-MM-DD' or a full ISO timestamp; return full ISO (UTC).

    A bare date used as a range END expands to 23:59:59 so that day is included."""
    v = value.strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", v):
        return f"{v}T23:59:59+00:00" if end_of_day else f"{v}T00:00:00+00:00"
    return v


def _flatten_item(item: dict[str, Any]) -> dict[str, Any]:
    """``{'id','href','fields':{...}}`` → fields dict augmented with id/href."""
    fields = dict(item.get("fields") or {})
    if item.get("id") is not None:
        fields["id"] = item["id"]
    if "href" in item:
        fields["href"] = item["href"]
    return fields


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
def _as_list(x: Any) -> list[Any]:
    if x is None:
        return []
    return list(x) if isinstance(x, (list, tuple)) else [x]


BLOCK_TAGS = (
    "p", "div", "br", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "table",
    "section", "article", "blockquote", "dt", "dd", "hr", "pre", "figure", "figcaption", "header", "footer",
)


def soup_text(element: Any) -> str:
    """Block-aware text extraction: newlines between block elements, inline text
    kept together (``<p>Over <b>8,000</b> killed.</p>`` -> 'Over 8,000 killed.').

    Mutates ``element`` (inserts newline strings) - pass a throwaway soup.
    """
    for tag in element.find_all(BLOCK_TAGS):
        if tag.name == "br":
            tag.replace_with("\n")
        else:
            tag.insert_before("\n")
            tag.insert_after("\n")
    text = element.get_text("")
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _html_to_text(html: str) -> str:
    """Strip HTML to plain text (scripts/styles removed, block-aware newlines)."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return soup_text(soup)


def parse_date(value: Any) -> str | None:
    """ISO timestamp / 'YYYY-MM-DD' / '25 Apr 2015' / '25 April 2015' → 'YYYY-MM-DD' (or None)."""
    if not value:
        return None
    s = str(value).strip()
    m = re.match(r"(\d{4}-\d{2}-\d{2})", s)
    if m:
        return m.group(1)
    for fmt in ("%d %b %Y", "%d %B %Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def normalize_report(item: dict[str, Any]) -> dict[str, Any]:
    """Convert a ReliefWeb report item into the unified sitrep record.

    Accepts either the raw ``{'id','href','fields':{...}}`` item or the
    flattened fields dict returned by :meth:`ReliefWebClient.fetch_reports`.
    """
    f = _flatten_item(item) if isinstance(item.get("fields"), dict) else item
    sources = [s for s in _as_list(f.get("source")) if isinstance(s, dict)]
    shortnames = [str(s.get("shortname") or s.get("name") or "") for s in sources]
    names = [str(s.get("name") or s.get("shortname") or "") for s in sources]
    formats = [str(x.get("name")) for x in _as_list(f.get("format")) if isinstance(x, dict) and x.get("name")]
    disasters = [d for d in _as_list(f.get("disaster")) if isinstance(d, dict)]
    country = f.get("primary_country") if isinstance(f.get("primary_country"), dict) else {}
    langs = [x.get("code") for x in _as_list(f.get("language")) if isinstance(x, dict) and x.get("code")]
    date_obj = f.get("date") if isinstance(f.get("date"), dict) else {}
    body = f.get("body")
    body_html = f.get("body-html") or f.get("body_html")
    if isinstance(body, str) and body.strip():
        text = body.strip()
    elif isinstance(body_html, str) and body_html.strip():
        text = _html_to_text(body_html)
    else:
        text = ""
    files = [
        {"url": x.get("url"), "filename": x.get("filename")}
        for x in _as_list(f.get("file"))
        if isinstance(x, dict)
    ]
    return {
        "id": str(f.get("id")) if f.get("id") is not None else None,
        "title": f.get("title"),
        "source": "/".join(s for s in shortnames if s),
        "source_names": [n for n in names if n],
        "date": parse_date(date_obj.get("original") or date_obj.get("created")),
        "url": f.get("url"),
        "url_alias": f.get("url_alias"),
        "format": "/".join(formats),
        "disaster_ids": [str(d.get("id")) for d in disasters if d.get("id") is not None],
        "disaster_names": [str(d.get("name")) for d in disasters if d.get("name")],
        "glide": next((str(d["glide"]) for d in disasters if d.get("glide")), None),
        "country_iso3": country.get("iso3"),
        "country_name": country.get("name"),
        "language": "/".join(str(c) for c in langs) or None,
        "text": text,
        "body_html": body_html if isinstance(body_html, str) else None,
        "files": files,
        "collection": "api",
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _best_disaster(candidates: Iterable[dict[str, Any]], query: str) -> dict[str, Any] | None:
    """Pick the candidate whose name shares the most tokens with the query."""
    q = set(re.findall(r"[a-z0-9]+", query.lower()))
    best, best_score = None, -1.0
    for c in candidates:
        name = str(c.get("name") or "").lower()
        toks = set(re.findall(r"[a-z0-9]+", name))
        score = len(q & toks) / (len(q) or 1)
        if score > best_score:
            best, best_score = c, score
    return best


def _smoke(args: argparse.Namespace) -> int:
    if not _valid_appname(env("RELIEFWEB_APPNAME")):
        print(MANUAL_FALLBACK_MESSAGE)
        return 2
    try:
        rw = ReliefWebClient()
    except ReliefWebAccessError as exc:
        print(str(exc))
        return 2
    try:
        disaster_id = args.disaster_id
        if disaster_id is None:
            if not args.query:
                print("provide --disaster-id N or --query 'Nepal earthquake 2015'")
                return 1
            cands = rw.search_disasters(args.query, limit=20)
            if not cands:
                print(f"no disasters matched {args.query!r}")
                return 1
            print("Disaster candidates:")
            for c in cands:
                d = c.get("date") if isinstance(c.get("date"), dict) else {}
                print(
                    f"  {str(c.get('id')):>8}  {str(c.get('name'))[:60]:<60}  "
                    f"{str(c.get('glide') or '-'):<22} {str(d.get('event') or '')[:10]}"
                )
            best = _best_disaster(cands, args.query) or cands[0]
            disaster_id = int(best["id"])
            print(f"-> using disaster {disaster_id}: {best.get('name')}")
        reports = rw.fetch_reports(disaster_id=disaster_id, max_total=args.n, limit=args.n)
        records = [normalize_report(r) for r in reports]
        print(f"{'date':<10} | {'source':<10} | {'format':<18} | {'title':<70} | id")
        for r in records:
            print(
                f"{r['date'] or '':<10} | {(r['source'] or '')[:10]:<10} | "
                f"{(r['format'] or '')[:18]:<18} | {(r['title'] or '')[:70]:<70} | {r['id']}"
            )
        out = rw.cache_dir / f"smoke_{disaster_id}.json"
        write_json(out, records)
        print(f"wrote {out}")
        print(f"SMOKE OK: {len(records)} reports")
        return 0
    except ReliefWebError as exc:
        print(f"ReliefWeb error: {exc}")
        return 1
    finally:
        rw.close()


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="ReliefWeb API v2 client smoke test")
    ap.add_argument("--smoke", action="store_true", help="run a tiny live query (needs RELIEFWEB_APPNAME)")
    ap.add_argument("--disaster-id", type=int, default=None)
    ap.add_argument("--query", type=str, default=None, help="disaster name query, e.g. 'Nepal earthquake 2015'")
    ap.add_argument("--n", type=int, default=10, help="number of reports to fetch")
    args = ap.parse_args(argv)
    try:  # Windows consoles/pipes may use a legacy code page; keep em-dashes and titles printable
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):  # pragma: no cover
        pass
    if args.smoke:
        return _smoke(args)
    ap.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
