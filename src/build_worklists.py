"""Turn researched ReliefWeb link sets into per-event manual-collection work lists.

Input: a JSON file (list of per-event objects, see `EVENT_KEYS`) produced by the link-research
step, plus `results/tables/social_stats.csv` for the tweet windows.

Output: `docs/worklists/<event>.md` per event and `docs/worklists/README.md` (index + viability
table). These are the click-lists a human works through — the fetching itself cannot be automated
(reliefweb.int answers scripts with an empty HTTP 202 bot challenge and the API needs the
pre-approved appname; see docs/manual_collection.md).

CLI: python -m src.build_worklists --input data/raw/worklists_raw.json
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

from src.util import ROOT, cfg_path, ensure_dir, get_logger, load_config, read_json

log = get_logger("sitrep.worklists")

EVENT_KEYS = ("event", "has_coverage", "coverage_note", "hub_url", "reports")
KIND_ORDER = {"flash_update": 0, "snapshot": 1, "situation_report": 2, "dref_update": 3, "other": 4}
KIND_LABEL = {"flash_update": "Flash Update", "snapshot": "Snapshot", "situation_report": "Situation Report",
              "dref_update": "IFRC DREF / operations update", "other": "Other"}
# An event is a usable study case only if enough human sitreps fall inside the tweet window.
MIN_IN_WINDOW = 5

# Links come from web search, not from the ReliefWeb API, so none of them is confirmed to resolve.
# `src/verify_worklists.py` turns this banner off event-by-event once the appname lands.
UNVERIFIED_BANNER = "\n".join((
    "> ⚠️ **Links are UNVERIFIED.** They were gathered by web search while the ReliefWeb API is",
    "> unavailable (appname pending), and could not be machine-confirmed — the audit pass ran out",
    "> of search budget before it could corroborate any of them. Structurally malformed and",
    "> audit-flagged links have already been removed, but a link here may still 404. If one does,",
    "> use the event hub below. Run `python -m src.verify_worklists` once `RELIEFWEB_APPNAME` is",
    "> approved to confirm every link against the API and rewrite these pages authoritatively.",
))


def load_windows(stats_csv: Path) -> dict[str, dict[str, str]]:
    """event -> {n_tweets, first_day, last_day, n_days} from the social stats table."""
    if not stats_csv.exists():
        return {}
    with open(stats_csv, "r", encoding="utf-8", newline="") as fh:
        return {r["event"]: r for r in csv.DictReader(fh)}


def dedupe_reports(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop duplicate URLs and anything that is not a reliefweb.int report/disaster link."""
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for r in reports:
        url = (r.get("url") or "").strip()
        if not url.startswith("https://reliefweb.int/report/"):
            log.warning("dropping non-report url: %s", url[:100])
            continue
        if url in seen:
            continue
        seen.add(url)
        out.append(r)
    return sorted(out, key=lambda r: (r.get("date") or "9999", KIND_ORDER.get(r.get("kind", "other"), 9)))


def viability(n_in_window: int, has_coverage: bool) -> tuple[str, str]:
    """(verdict, why) for the index table."""
    if not has_coverage:
        return "excluded", "no OCHA/IFRC situation reports (domestic response; FEMA/EU-ERCC route only)"
    if n_in_window >= MIN_IN_WINDOW:
        return "usable", f"{n_in_window} candidate reports inside the tweet window"
    if n_in_window > 0:
        return "marginal", f"only {n_in_window} candidate report(s) inside the tweet window"
    return "marginal", "coverage exists but no candidate report falls inside the tweet window"


def render_event(ev: dict[str, Any], win: dict[str, str] | None) -> str:
    """Markdown work list for one event."""
    event = ev["event"]
    reports = dedupe_reports(ev.get("reports") or [])
    inw = [r for r in reports if r.get("in_tweet_window")]
    outw = [r for r in reports if not r.get("in_tweet_window")]
    folder = event.replace("_", "-")
    lines: list[str] = [f"# Work list — {event}", ""]
    if win:
        lines += [f"**Tweet stream we hold:** {int(win['n_tweets']):,} tweets, "
                  f"{win['first_day']} → {win['last_day']} ({win['n_days']} days).", ""]
    lines += [ev.get("coverage_note", "").strip(), ""]
    if reports:
        lines += [UNVERIFIED_BANNER, ""]
    if not ev.get("has_coverage"):
        lines += ["> **This event is excluded from the study.** No OCHA/IFRC situation-report series to",
                  "> compare against — the human side would have to come from FEMA / state EOC / EU-ERCC",
                  "> briefings (PLAN §2b's alternative route), which is a separate manual collection effort.",
                  ""]
        if reports:
            lines += ["Anything relevant that did turn up:", ""]
            lines += ["| Date | Source | Title | Link |", "|---|---|---|---|"]
            lines += [f"| {r.get('date') or '?'} | {r.get('source', '?')} | {r.get('title', '')[:70]} | "
                      f"[open]({r['url']}) |" for r in reports]
            lines += [""]
        return "\n".join(lines)

    lines += ["## How to collect", "",
              f"Save each report into `data/raw/reliefweb_manual/{folder}/` "
              "(PDF via *Download report* if offered, else Ctrl+S → \"Webpage, HTML only\"; any filename), then:", "",
              "```",
              f"python -m src.ingest_manual_reliefweb --scan data/raw/reliefweb_manual/{folder}",
              "python -m src.ingest_manual_reliefweb",
              "```", "",
              "`--scan` reads title/date/source/URL back out of the files, so you type nothing.", ""]
    if ev.get("hub_url"):
        lines += [f"**Event hub (lists everything, use it to fill gaps):** {ev['hub_url']}", "",
                  "On the hub: *Updates* tab → left sidebar → **Format = Situation Report**, "
                  "**Source = OCHA** (repeat for **IFRC**), sort oldest first.", ""]

    def table(rows: list[dict[str, Any]]) -> list[str]:
        out = ["| Date | Source | Kind | Title | Link |", "|---|---|---|---|---|"]
        for r in rows:
            title = (r.get("title") or "").replace("|", "-")[:80]
            out.append(f"| {r.get('date') or '?'} | {r.get('source', '?')} | "
                       f"{KIND_LABEL.get(r.get('kind', 'other'), 'Other')} | {title} | [open]({r['url']}) |")
        return out

    lines += [f"## Priority A — inside the tweet window ({len(inw)} reports)", ""]
    lines += table(inw) if inw else ["*(none verified inside the window — see the hub)*"]
    lines += ["", f"## Priority B — outside the window ({len(outw)} reports)", "",
              "Still valuable: these give the schema induction its later, consolidated end of the spectrum.", ""]
    lines += table(outw) if outw else ["*(none)*"]
    lines += ["", "---", "",
              "If a link 404s it was a search artifact — use the hub above; and please tell me so I can drop it.", ""]
    return "\n".join(lines)


def render_index(events: list[dict[str, Any]], windows: dict[str, dict[str, str]]) -> str:
    """Index page with the viability verdict per event."""
    rows: list[tuple[str, int, int, str, str, str]] = []
    for ev in events:
        reports = dedupe_reports(ev.get("reports") or [])
        n_in = len([r for r in reports if r.get("in_tweet_window")])
        verdict, why = viability(n_in, bool(ev.get("has_coverage")))
        w = windows.get(ev["event"], {})
        rows.append((ev["event"], len(reports), n_in, w.get("n_tweets", "?"), verdict, why))
    order = {"usable": 0, "marginal": 1, "excluded": 2}
    rows.sort(key=lambda r: (order.get(r[4], 3), -r[2]))
    lines = [
        "# ReliefWeb work lists — one per HumAID event", "",
        "The social side of all 19 events is already collected (`results/tables/social_stats.csv`).",
        "The human side must be saved by hand: reliefweb.int answers scripted clients with an empty",
        "HTTP 202 bot challenge and the API needs the pre-approved appname, so these pages are click-lists.",
        "All bookkeeping after the click is automated (`--scan`, see `docs/manual_collection.md`).", "",
        f"An event counts as **usable** when at least {MIN_IN_WINDOW} candidate human reports fall inside its",
        "tweet window — that overlap is what the completeness comparison is computed on.", "",
        UNVERIFIED_BANNER, "",
        "| Event | Tweets | Candidate links | In window | Verdict | Why |", "|---|---|---|---|---|---|",
    ]
    for event, n_all, n_in, n_tw, verdict, why in rows:
        tw = f"{int(n_tw):,}" if str(n_tw).isdigit() else n_tw
        mark = {"usable": "**usable**", "marginal": "marginal", "excluded": "excluded"}[verdict]
        lines.append(f"| [{event}]({event}.md) | {tw} | {n_all} | {n_in} | {mark} | {why} |")
    lines += ["", "## Current plan", "",
              "**Cyclone Idai 2019 is the pilot** — collected deeply first (see `docs/idai_worklist.md`,",
              "which is the hand-curated version of `cyclone_idai_2019.md`). The other usable events are",
              "ready to add as events 2-5 once the pilot validates the pipeline end to end.", ""]
    return "\n".join(lines)


def build(input_json: Path, out_dir: Path, stats_csv: Path) -> dict[str, Any]:
    """Write every work list + the index; returns a small summary."""
    data = read_json(input_json)
    events = data["results"] if isinstance(data, dict) and "results" in data else data
    if not isinstance(events, list):
        raise ValueError(f"{input_json} must hold a list of event objects (or {{'results': [...]}})")
    windows = load_windows(stats_csv)
    ensure_dir(out_dir)
    written: list[str] = []
    for ev in events:
        missing = [k for k in EVENT_KEYS if k not in ev]
        if missing:
            log.warning("event %s missing keys %s — skipped", ev.get("event", "?"), missing)
            continue
        path = out_dir / f"{ev['event']}.md"
        path.write_text(render_event(ev, windows.get(ev["event"])), encoding="utf-8")
        written.append(path.name)
    (out_dir / "README.md").write_text(render_index(events, windows), encoding="utf-8")
    n_links = sum(len(dedupe_reports(e.get("reports") or [])) for e in events)
    log.info("wrote %d work lists (%d candidate links) to %s", len(written), n_links, out_dir)
    return {"files": written, "n_events": len(written), "n_links": n_links, "out_dir": str(out_dir)}


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description="Build per-event ReliefWeb manual-collection work lists")
    ap.add_argument("--input", type=Path, required=True, help="JSON produced by the link-research step")
    ap.add_argument("--out", type=Path, default=None, help="output dir (default docs/worklists)")
    args = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):  # pragma: no cover
        pass
    cfg = load_config()
    out = args.out or (ROOT / "docs" / "worklists")
    summary = build(args.input, out, cfg_path(cfg, "results", "results") / "tables" / "social_stats.csv")
    print(f"wrote {summary['n_events']} work lists ({summary['n_links']} candidate links) to {summary['out_dir']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
