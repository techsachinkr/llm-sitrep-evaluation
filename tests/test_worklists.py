"""Offline tests for the work-list builder and the API verifier."""
from __future__ import annotations

import json

import httpx
import pytest

from src.build_worklists import UNVERIFIED_BANNER, build, dedupe_reports, render_event, viability
from src.reliefweb_api import ReliefWebClient
from src.verify_worklists import alias_of, chunks, verify

REAL = "https://reliefweb.int/report/mozambique/real-one"
FAKE = "https://reliefweb.int/report/mozambique/hallucinated-one"


def _report(url=REAL, date="2019-04-02", in_window=True, **kw):
    base = {"title": "Situation Report No. 1", "date": date, "url": url, "source": "OCHA",
            "kind": "situation_report", "in_tweet_window": in_window}
    base.update(kw)
    return base


# ---------------------------------------------------------------------------
# build_worklists
# ---------------------------------------------------------------------------
def test_dedupe_drops_duplicates_and_non_report_urls():
    rows = [_report(), _report(), _report(url="https://reliefweb.int/map/x/y"),
            _report(url="https://example.com/evil"),
            _report(url="https://reliefweb.int/report/mozambique/other", date="2019-03-20")]
    out = dedupe_reports(rows)
    assert [r["url"] for r in out] == ["https://reliefweb.int/report/mozambique/other", REAL]  # date-sorted
    assert all(r["url"].startswith("https://reliefweb.int/report/") for r in out)


def test_viability_verdicts():
    assert viability(9, True)[0] == "usable"
    assert viability(5, True)[0] == "usable"
    assert viability(4, True)[0] == "marginal"
    assert viability(0, True)[0] == "marginal"
    assert viability(0, False)[0] == "excluded"


def test_render_event_marks_unverified_and_splits_priorities():
    ev = {"event": "cyclone_idai_2019", "has_coverage": True, "coverage_note": "Dense OCHA series.",
          "hub_url": "https://reliefweb.int/disaster/tc-2019-000021-moz",
          "reports": [_report(), _report(url=REAL + "-b", date="2019-05-06", in_window=False)]}
    md = render_event(ev, {"n_tweets": "3933", "first_day": "2019-03-15", "last_day": "2019-04-16", "n_days": "29"})
    assert UNVERIFIED_BANNER in md  # candidate links must never look confirmed
    assert "Priority A — inside the tweet window (1 reports)" in md
    assert "Priority B — outside the window (1 reports)" in md
    assert "tc-2019-000021-moz" in md
    assert "--scan data/raw/reliefweb_manual/cyclone-idai-2019" in md
    assert "3,933 tweets" in md


def test_render_event_excluded_has_no_collection_instructions():
    ev = {"event": "hurricane_harvey_2017", "has_coverage": False,
          "coverage_note": "US domestic response; FEMA rather than OCHA.", "hub_url": "", "reports": []}
    md = render_event(ev, None)
    assert "excluded from the study" in md
    assert "--scan" not in md and UNVERIFIED_BANNER not in md


def test_build_writes_pages_and_index(tmp_path):
    payload = {"results": [
        {"event": "cyclone_idai_2019", "has_coverage": True, "coverage_note": "x", "hub_url": "",
         "reports": [_report(url=f"{REAL}-{i}", date=f"2019-04-0{i}") for i in range(1, 7)]},
        {"event": "hurricane_harvey_2017", "has_coverage": False, "coverage_note": "FEMA.",
         "hub_url": "", "reports": []}]}
    src = tmp_path / "raw.json"
    src.write_text(json.dumps(payload), encoding="utf-8")
    stats = tmp_path / "social_stats.csv"
    stats.write_text("event,n_tweets,n_days,first_day,last_day\ncyclone_idai_2019,3933,29,2019-03-15,2019-04-16\n",
                     encoding="utf-8")
    out = tmp_path / "wl"
    summary = build(src, out, stats)
    assert summary["n_events"] == 2 and summary["n_links"] == 6
    index = (out / "README.md").read_text(encoding="utf-8")
    assert "**usable**" in index and "excluded" in index
    assert "| [cyclone_idai_2019](cyclone_idai_2019.md) | 3,933 | 6 | 6 |" in index
    assert (out / "cyclone_idai_2019.md").exists() and (out / "hurricane_harvey_2017.md").exists()


def test_build_skips_malformed_event_objects(tmp_path):
    src = tmp_path / "raw.json"
    src.write_text(json.dumps({"results": [{"event": "x"}, {"event": "ok", "has_coverage": False,
                                                            "coverage_note": "", "hub_url": "", "reports": []}]}),
                   encoding="utf-8")
    summary = build(src, tmp_path / "wl", tmp_path / "missing.csv")
    assert summary["n_events"] == 1  # the one missing required keys is skipped, not fatal


# ---------------------------------------------------------------------------
# verify_worklists
# ---------------------------------------------------------------------------
def test_alias_and_chunks():
    assert alias_of("https://reliefweb.int/report/mozambique/foo") == "report/mozambique/foo"
    assert alias_of("https://reliefweb.int/report/x/y/") == "report/x/y"
    assert [len(c) for c in chunks(list(range(250)), 100)] == [100, 100, 50]
    assert list(chunks([], 10)) == []


def _client(tmp_path, handler):
    return ReliefWebClient("test-app", cache_dir=tmp_path / "rw",
                           transport=httpx.MockTransport(handler), min_interval_s=0.0)


def test_verify_drops_unresolvable_links_and_corrects_metadata(tmp_path):
    def handler(request):
        body = json.loads(request.content)
        assert body["filter"]["field"] == "url_alias"
        return httpx.Response(200, json={"data": [{"id": "1", "fields": {
            "title": "Mozambique: Cyclone Idai & Floods Situation Report No. 1", "url": REAL,
            "url_alias": "report/mozambique/real-one", "date": {"original": "2019-04-02T00:00:00+00:00"},
            "source": [{"shortname": "OCHA"}], "format": [{"name": "Situation Report"}]}}], "totalCount": 1})

    payload = {"results": [{"event": "cyclone_idai_2019", "has_coverage": True, "coverage_note": "",
                            "hub_url": "", "reports": [_report(url=REAL, date="", source="?"),
                                                       _report(url=FAKE, date="2019-04-03")]}]}
    rw = _client(tmp_path, handler)
    summary = verify(payload, rw, complete=False)
    rw.close()
    ev = payload["results"][0]
    assert len(ev["reports"]) == 1 and ev["reports"][0]["url"] == REAL
    assert ev["reports"][0]["date"] == "2019-04-02" and ev["reports"][0]["source"] == "OCHA"
    assert ev["verified"] is True and ev["n_missing"] == 1
    assert summary["n_ok"] == 1 and summary["n_missing"] == 1


def test_verify_completes_from_api_when_hub_id_known(tmp_path):
    extra = "https://reliefweb.int/report/mozambique/api-only-report"

    def handler(request):
        body = json.loads(request.content)
        if body.get("filter", {}).get("field") == "url_alias":
            return httpx.Response(200, json={"data": [{"id": "1", "fields": {
                "title": "known", "url": REAL, "url_alias": "report/mozambique/real-one",
                "date": {"original": "2019-04-02T00:00:00+00:00"}, "source": [{"shortname": "OCHA"}]}}]})
        if body.get("offset", 0) == 0:  # fetch_reports page 1
            return httpx.Response(200, json={"data": [{"id": "2", "fields": {
                "title": "API-only sitrep", "url": extra, "date": {"original": "2019-04-05T00:00:00+00:00"},
                "source": [{"shortname": "OCHA"}], "format": [{"name": "Situation Report"}]}}], "totalCount": 1})
        return httpx.Response(200, json={"data": [], "totalCount": 1})

    payload = {"results": [{"event": "cyclone_idai_2019", "has_coverage": True, "coverage_note": "",
                            "hub_url": "https://reliefweb.int/disaster/tc-2019-000021-moz",
                            "hub_disaster_id": 12345, "reports": [_report(url=REAL)]}]}
    rw = _client(tmp_path, handler)
    summary = verify(payload, rw, complete=True, windows={"cyclone_idai_2019": ("2019-03-15", "2019-04-16")})
    rw.close()
    ev = payload["results"][0]
    urls = {r["url"] for r in ev["reports"]}
    assert urls == {REAL, extra} and summary["n_added"] == 1
    assert ev["source_of_truth"] == "reliefweb-api"
    added = [r for r in ev["reports"] if r["url"] == extra][0]
    assert added["in_tweet_window"] is True and added["date"] == "2019-04-05"


def test_verify_respects_event_filter(tmp_path):
    def handler(request):
        return httpx.Response(200, json={"data": [], "totalCount": 0})

    payload = {"results": [
        {"event": "a", "has_coverage": True, "coverage_note": "", "hub_url": "", "reports": [_report()]},
        {"event": "b", "has_coverage": True, "coverage_note": "", "hub_url": "", "reports": [_report()]}]}
    rw = _client(tmp_path, handler)
    summary = verify(payload, rw, only={"a"}, complete=False)
    rw.close()
    assert [e["event"] for e in summary["events"]] == ["a"]
    assert payload["results"][0].get("verified") and not payload["results"][1].get("verified")
    assert payload["results"][1]["reports"]  # untouched event keeps its candidate links
