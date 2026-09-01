"""Verify (and correct) the work-list links against the ReliefWeb API — run once the appname lands.

The work lists in `docs/worklists/` were assembled from web search while the API was unavailable,
so every link in them is a *candidate*: it may 404. This script settles the question deterministically.

For each event it does two things:

1. **Checks** every candidate URL by querying `url_alias` in bulk (a handful of API calls for the
   whole corpus, well inside the 1,000-calls/day quota) and marks each link ok / missing.
2. **Completes** the list: for events whose hub disaster id is known it pulls the *authoritative* set
   of Situation Report / Flash Update items from the API, so reports that search never surfaced are
   added and the page stops depending on search coverage at all.

The corrected JSON is written back and `src/build_worklists.py` regenerates the pages, this time
without the "UNVERIFIED" banner for the events that were checked.

CLI:
    python -m src.verify_worklists --input data/raw/worklists_raw.json          # check + complete
    python -m src.verify_worklists --input ... --events cyclone_idai_2019       # one event
    python -m src.verify_worklists --input ... --check-only                     # no API completion
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Iterable

from src.reliefweb_api import (
    MANUAL_FALLBACK_MESSAGE,
    ReliefWebAccessError,
    ReliefWebClient,
    ReliefWebError,
    normalize_report,
)
from src.util import get_logger, read_json, write_json

log = get_logger("sitrep.verify")

URL_PREFIX = "https://reliefweb.int/report/"
SITREP_FORMATS = ("Situation Report",)
CHUNK = 100  # url_alias values per API call


def alias_of(url: str) -> str:
    """'https://reliefweb.int/report/mozambique/foo' -> 'report/mozambique/foo' (the API's url_alias)."""
    return url.split("https://reliefweb.int/", 1)[-1].strip("/")


def chunks(seq: list[Any], n: int) -> Iterable[list[Any]]:
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def check_urls(rw: ReliefWebClient, urls: list[str]) -> dict[str, dict[str, Any]]:
    """Return {url: {'ok': bool, 'title', 'date', 'source'}} by querying url_alias in bulk."""
    found: dict[str, dict[str, Any]] = {}
    aliases = [alias_of(u) for u in urls]
    for batch in chunks(aliases, CHUNK):
        body = {
            "filter": {"field": "url_alias", "value": batch, "operator": "OR"},
            "fields": {"include": ["id", "title", "url", "url_alias", "date.original",
                                   "source.shortname", "format.name"]},
            "limit": min(len(batch) * 2, 1000),
        }
        data = rw.post("reports", body)
        for item in data.get("data", []):
            f = item.get("fields", {}) or {}
            url = f.get("url") or ""
            rec = normalize_report(item)
            found[url] = {"ok": True, "title": rec.get("title"), "date": rec.get("date"),
                          "source": rec.get("source")}
            found[f"https://reliefweb.int/{(f.get('url_alias') or '').strip('/')}"] = found[url]
    return {u: found.get(u, {"ok": False}) for u in urls}


def authoritative_reports(rw: ReliefWebClient, disaster_id: int, window: tuple[str, str] | None,
                          max_total: int = 300) -> list[dict[str, Any]]:
    """Every Situation Report the API holds for a disaster (the ground truth for a work list)."""
    items = rw.fetch_reports(disaster_id=disaster_id, formats=SITREP_FORMATS, sources=None,
                             max_total=max_total, limit=200)
    out: list[dict[str, Any]] = []
    for it in items:
        rec = normalize_report({"id": it.get("id"), "fields": it})
        date = rec.get("date") or ""
        out.append({"title": rec.get("title"), "date": date, "url": rec.get("url"),
                    "source": (rec.get("source") or "").split("/")[0] or "other",
                    "kind": "situation_report",
                    "in_tweet_window": bool(window and window[0] <= date <= window[1])})
    return out


def verify(payload: dict[str, Any], rw: ReliefWebClient, *, only: set[str] | None = None,
           complete: bool = True, windows: dict[str, tuple[str, str]] | None = None) -> dict[str, Any]:
    """Check every candidate link, optionally add API-authoritative ones; returns a summary."""
    summary: dict[str, Any] = {"events": [], "n_ok": 0, "n_missing": 0, "n_added": 0}
    for ev in payload.get("results", []):
        key = ev.get("event", "")
        if only and key not in only:
            continue
        urls = [r["url"] for r in ev.get("reports", []) if str(r.get("url", "")).startswith(URL_PREFIX)]
        status = check_urls(rw, urls) if urls else {}
        kept, missing = [], []
        for r in ev.get("reports", []):
            st = status.get(r.get("url", ""), {"ok": False})
            if st.get("ok"):
                kept.append({**r, "title": st.get("title") or r.get("title"),
                             "date": st.get("date") or r.get("date"),
                             "source": st.get("source") or r.get("source")})
            else:
                missing.append(r)
                log.warning("%s: link does not resolve, dropping: %s", key, r.get("url"))
        added = 0
        if complete and ev.get("hub_disaster_id"):
            have = {r["url"] for r in kept}
            for rec in authoritative_reports(rw, int(ev["hub_disaster_id"]),
                                             (windows or {}).get(key)):
                if rec["url"] not in have:
                    kept.append(rec)
                    added += 1
            ev["source_of_truth"] = "reliefweb-api"
        ev["reports"] = sorted(kept, key=lambda r: (r.get("date") or "9999"))
        ev["verified"] = True
        ev["n_missing"] = len(missing)
        summary["events"].append({"event": key, "ok": len(kept) - added, "missing": len(missing), "added": added})
        summary["n_ok"] += len(kept) - added
        summary["n_missing"] += len(missing)
        summary["n_added"] += added
        log.info("%s: %d ok, %d dropped, %d added from the API", key, len(kept) - added, len(missing), added)
    return summary


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description="Verify work-list links against the ReliefWeb API")
    ap.add_argument("--input", type=Path, required=True, help="work-list JSON to verify (updated in place)")
    ap.add_argument("--events", nargs="*", default=None, help="limit to these event keys")
    ap.add_argument("--check-only", action="store_true", help="do not add API-authoritative reports")
    args = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):  # pragma: no cover
        pass
    try:
        rw = ReliefWebClient()
    except ReliefWebAccessError:
        print(MANUAL_FALLBACK_MESSAGE)
        print("\nVerification needs the API. Until then the work lists stay marked UNVERIFIED.")
        return 2
    payload = read_json(args.input)
    try:
        summary = verify(payload, rw, only=set(args.events) if args.events else None,
                         complete=not args.check_only)
    except ReliefWebError as exc:
        print(f"ReliefWeb error: {exc}")
        return 1
    finally:
        rw.close()
    write_json(args.input, payload)
    print(f"verified {len(summary['events'])} event(s): {summary['n_ok']} links ok, "
          f"{summary['n_missing']} dropped, {summary['n_added']} added from the API")
    print(f"updated {args.input}\nnow run: python -m src.build_worklists --input {args.input}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
