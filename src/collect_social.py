"""Collect the social-media side of the corpus (PLAN §2b/§2c).

Currently supports **HumAID** (QCRI): 19 disaster events, ~76k human-labelled tweets that ship
with `tweet_text` — no hydration required, which is why it satisfies the PLAN §2b entry rule
("an event enters the study only if the social side has retrievable TEXT").

Tweets carry no timestamp column, but a Twitter snowflake id encodes its creation time, so the
per-day stream files required by PLAN §2c are derived from `tweet_id` (see `snowflake_to_utc`).

Data hygiene (CLAUDE.md rule 7): everything written here lands under `data/`, which is gitignored;
tweet text must never be committed or quoted verbatim in the paper.

CLI:
    python -m src.collect_social --list
    python -m src.collect_social --events cyclone_idai_2019 hurricane_dorian_2019
    python -m src.collect_social --all
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from src.util import cfg_path, ensure_dir, get_logger, load_config, utc_now_iso, write_json, write_jsonl

log = get_logger("sitrep.social")

HUMAID_DATASET = "QCRI/HumAID-events"
TWITTER_EPOCH_MS = 1288834974657  # 2010-11-04T01:42:54.657Z — snowflake epoch

# The 19 HumAID events (config name -> human label). Sizes are in results/tables/social_stats.csv
# after a run; which ones enter the study is a user decision (PLAN §2b, RUN.md checkpoint).
HUMAID_EVENTS: tuple[str, ...] = (
    "california_wildfires_2018", "canada_wildfires_2016", "cyclone_idai_2019", "ecuador_earthquake_2016",
    "greece_wildfires_2018", "hurricane_dorian_2019", "hurricane_florence_2018", "hurricane_harvey_2017",
    "hurricane_irma_2017", "hurricane_maria_2017", "hurricane_matthew_2016", "italy_earthquake_aug_2016",
    "kaikoura_earthquake_2016", "kerala_floods_2018", "maryland_floods_2018", "midwestern_us_floods_2019",
    "pakistan_earthquake_2019", "puebla_mexico_earthquake_2017", "srilanka_floods_2017",
)


def snowflake_to_utc(tweet_id: int | str) -> datetime | None:
    """Decode a Twitter snowflake id into its UTC creation time.

    Ids are 64-bit: the top 41 bits are milliseconds since the Twitter epoch. Ids minted before
    snowflake (pre-2010, < 2**32) carry no timestamp and return None.
    """
    try:
        i = int(tweet_id)
    except (TypeError, ValueError):
        return None
    if i < 2**32:  # pre-snowflake id
        return None
    ms = (i >> 22) + TWITTER_EPOCH_MS
    try:
        return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None


def load_humaid_event(event: str) -> list[dict[str, Any]]:
    """Load every split of one HumAID event config into a list of records."""
    from datasets import load_dataset

    out: list[dict[str, Any]] = []
    ds = load_dataset(HUMAID_DATASET, event)
    for split, rows in ds.items():
        for r in rows:
            out.append({"tweet_id": str(r["tweet_id"]), "text": r["tweet_text"],
                        "class_label": r["class_label"], "split": split, "event": event})
    return out


def bucket_by_day(records: Iterable[dict[str, Any]]) -> tuple[dict[str, list[dict[str, Any]]], int]:
    """Group records into {YYYY-MM-DD: [record, ...]} using the snowflake timestamp.

    Returns (buckets, n_undated). Each record gains a `created_at` ISO field.
    """
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    undated = 0
    for r in records:
        dt = snowflake_to_utc(r["tweet_id"])
        if dt is None:
            undated += 1
            continue
        r = dict(r, created_at=dt.isoformat(timespec="seconds"))
        buckets[dt.strftime("%Y-%m-%d")].append(r)
    return dict(buckets), undated


def collect_event(event: str, raw_dir: Path, stream_dir: Path) -> dict[str, Any]:
    """Download one HumAID event, write the raw dump and the per-day stream files."""
    records = load_humaid_event(event)
    ensure_dir(raw_dir)
    write_jsonl(raw_dir / f"{event}.jsonl", records)
    buckets, undated = bucket_by_day(records)
    ev_dir = ensure_dir(stream_dir / event)
    for old in ev_dir.glob("*.jsonl"):  # idempotent: rebuild the event's stream from scratch
        old.unlink()
    for day, rows in sorted(buckets.items()):
        write_jsonl(ev_dir / f"{day}.jsonl", sorted(rows, key=lambda r: r["created_at"]))
    days = sorted(buckets)
    labels = Counter(r["class_label"] for r in records)
    stats = {
        "event": event, "n_tweets": len(records), "n_undated": undated, "n_days": len(days),
        "first_day": days[0] if days else None, "last_day": days[-1] if days else None,
        "median_tweets_per_day": (sorted(len(v) for v in buckets.values())[len(buckets) // 2] if buckets else 0),
        "max_tweets_per_day": max((len(v) for v in buckets.values()), default=0),
        "top_labels": dict(labels.most_common(5)), "source": HUMAID_DATASET, "collected_at": utc_now_iso(),
    }
    write_json(ev_dir / "_stats.json", stats)
    log.info("%s: %d tweets over %d days (%s..%s), %d undated", event, len(records), len(days),
             stats["first_day"], stats["last_day"], undated)
    return stats


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description="Collect HumAID social-media streams (PLAN §2b)")
    ap.add_argument("--events", nargs="*", default=None, help="HumAID event configs (default: none; use --all)")
    ap.add_argument("--all", action="store_true", help="collect all 19 HumAID events")
    ap.add_argument("--list", action="store_true", help="list the known HumAID events and exit")
    args = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):  # pragma: no cover
        pass

    if args.list:
        for e in HUMAID_EVENTS:
            print(e)
        return 0
    events = list(HUMAID_EVENTS) if args.all else list(args.events or [])
    if not events:
        ap.print_help()
        return 2
    unknown = [e for e in events if e not in HUMAID_EVENTS]
    if unknown:
        print(f"unknown event(s): {unknown}\nknown: {list(HUMAID_EVENTS)}")
        return 2

    cfg = load_config()
    raw_dir = cfg_path(cfg, "raw", "data/raw") / "humaid"
    stream_dir = cfg_path(cfg, "processed", "data/processed") / "streams"
    all_stats = [collect_event(e, raw_dir, stream_dir) for e in events]

    tables = ensure_dir(cfg_path(cfg, "results", "results") / "tables")
    import csv

    cols = ["event", "n_tweets", "n_days", "first_day", "last_day", "median_tweets_per_day",
            "max_tweets_per_day", "n_undated", "source"]
    with open(tables / "social_stats.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows([{c: s[c] for c in cols} for s in all_stats])
    print(f"\ncollected {len(all_stats)} event(s); {sum(s['n_tweets'] for s in all_stats):,} tweets")
    print(f"raw:     {raw_dir}")
    print(f"streams: {stream_dir}")
    print(f"stats:   {tables / 'social_stats.csv'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
