"""Offline tests for src.reliefweb_api and src.ingest_manual_reliefweb.

All HTTP goes through httpx.MockTransport — no network access.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import httpx
import pytest

from src.ingest_manual_reliefweb import extract_html, ingest, main as ingest_main
from src.reliefweb_api import (
    MANUAL_FALLBACK_MESSAGE,
    ReliefWebError,
    ReliefWebAccessError,
    ReliefWebClient,
    ReliefWebQuotaError,
    normalize_report,
)

APP = "test-app"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _item(i: int, **fields):
    base = {"title": f"Report {i}", "date": {"original": "2015-04-25T00:00:00+00:00"}}
    base.update(fields)
    return {"id": str(i), "href": f"https://api.reliefweb.int/v2/reports/{i}", "fields": base}


class Recorder:
    """MockTransport handler that records requests and serves scripted responses."""

    def __init__(self, responses=None):
        self.requests: list[httpx.Request] = []
        self.times: list[float] = []
        self.responses = list(responses or [])

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        self.times.append(time.monotonic())
        if self.responses:
            nxt = self.responses.pop(0)
            if callable(nxt):
                return nxt(request)
            return nxt
        return httpx.Response(200, json={"data": [], "totalCount": 0})

    def body(self, k: int = 0) -> dict:
        return json.loads(self.requests[k].content.decode("utf-8"))


def make_client(tmp_path, handler, **kw):
    kw.setdefault("min_interval_s", 0.0)
    return ReliefWebClient(APP, cache_dir=tmp_path / "rw", transport=httpx.MockTransport(handler), **kw)


# ---------------------------------------------------------------------------
# client
# ---------------------------------------------------------------------------
def test_missing_appname_raises(monkeypatch, tmp_path):
    monkeypatch.delenv("RELIEFWEB_APPNAME", raising=False)
    # explicit empty string → treated as missing (do NOT fall back to .env)
    with pytest.raises(ReliefWebAccessError) as ei:
        ReliefWebClient("", cache_dir=tmp_path)
    assert "RELIEFWEB_APPNAME" in str(ei.value)
    assert str(ei.value) == MANUAL_FALLBACK_MESSAGE
    # a comment-looking value (dotenv misparse) is also rejected
    with pytest.raises(ReliefWebAccessError):
        ReliefWebClient("# REQUIRED: pre-approved appname", cache_dir=tmp_path)


def test_fetch_reports_body_and_pagination(tmp_path):
    page1 = httpx.Response(200, json={"data": [_item(1), _item(2)], "totalCount": 5})
    page2 = httpx.Response(200, json={"data": [_item(3), _item(4)], "totalCount": 5})
    empty = httpx.Response(200, json={"data": [], "totalCount": 5})
    rec = Recorder([page1, page2, empty])
    rw = make_client(tmp_path, rec)
    out = rw.fetch_reports(disaster_id=15082, limit=2, date_from="2015-04-25", date_to="2015-06-30")

    assert [r["id"] for r in out] == ["1", "2", "3", "4"]
    assert all("href" in r for r in out)
    assert len(rec.requests) == 3
    req = rec.requests[0]
    assert req.method == "POST"
    assert req.url.host == "api.reliefweb.int"
    assert req.url.path == "/v2/reports"
    assert req.url.params["appname"] == APP
    assert req.headers["User-Agent"].startswith("sitrep-gap/0.1")

    body = rec.body(0)
    conds = body["filter"]["conditions"]
    assert body["filter"]["operator"] == "AND"
    by_field = {c["field"]: c for c in conds}
    assert by_field["format.name"]["value"] == ["Situation Report"]
    assert by_field["format.name"]["operator"] == "OR"
    assert by_field["source.shortname"]["value"] == ["OCHA", "IFRC"]
    assert by_field["disaster.id"]["value"] == 15082
    assert by_field["language.code"]["value"] == "en"
    assert by_field["date.original"]["value"]["from"].startswith("2015-04-25")
    assert by_field["date.original"]["value"]["to"].startswith("2015-06-30")
    assert "body" in body["fields"]["include"] and "body-html" in body["fields"]["include"]
    assert body["sort"] == ["date.original:asc"]
    assert body["limit"] == 2 and body["limit"] <= 1000
    assert body["offset"] == 0
    assert rec.body(1)["offset"] == 2
    assert rec.body(2)["offset"] == 4


def test_fetch_reports_limit_capped_and_max_total(tmp_path):
    rec = Recorder([httpx.Response(200, json={"data": [_item(i) for i in range(1, 6)], "totalCount": 50})])
    rw = make_client(tmp_path, rec)
    out = rw.fetch_reports(query="Nepal", limit=5000, max_total=3)
    assert len(out) == 3
    body = rec.body(0)
    assert body["limit"] <= 1000
    assert body["limit"] == 3
    assert body["query"] == {"value": "Nepal"}
    assert len(rec.requests) == 1


def test_rate_limit_between_network_calls(tmp_path, monkeypatch):
    """Deterministic: a fake clock records the sleep the client requests between network calls."""
    import src.reliefweb_api as rw_mod

    class FakeClock:
        def __init__(self):
            self.now = 1000.0
            self.sleeps: list[float] = []

        def monotonic(self):
            return self.now

        def sleep(self, secs):
            self.sleeps.append(secs)
            self.now += secs

    clock = FakeClock()
    monkeypatch.setattr(rw_mod, "time", clock)
    rec = Recorder([
        httpx.Response(200, json={"data": [_item(1)], "totalCount": 1}),
        httpx.Response(200, json={"data": [_item(2)], "totalCount": 1}),
        httpx.Response(200, json={"data": [_item(3)], "totalCount": 1}),
    ])
    rw = make_client(tmp_path, rec, min_interval_s=1.0)
    rw.post("reports", {"a": 1})
    rw.post("reports", {"a": 2})            # immediately after -> must sleep ~1.0s
    rw.post("reports", {"a": 2})            # cache hit -> no sleep, no network
    clock.now += 5.0
    rw.post("reports", {"a": 3})            # >1s elapsed -> no sleep
    assert len(rec.requests) == 3
    assert len(clock.sleeps) == 1 and clock.sleeps[0] == pytest.approx(1.0, abs=1e-6)


def test_cache_prevents_second_network_call(tmp_path):
    rec = Recorder([httpx.Response(200, json={"data": [_item(1)], "totalCount": 1})])
    rw = make_client(tmp_path, rec)
    body = {"filter": {"field": "id", "value": 1}, "limit": 1}
    r1 = rw.post("reports", body)
    r2 = rw.post("reports", body)
    assert r1 == r2
    assert len(rec.requests) == 1
    files = list((tmp_path / "rw" / "reports").glob("*.json"))
    assert len(files) == 1
    cached = json.loads(files[0].read_text(encoding="utf-8"))
    assert cached["endpoint"] == "reports" and cached["body"] == body and cached["status"] == 200
    assert cached["response"]["data"][0]["id"] == "1"
    # a second client on the same cache dir also hits the cache
    rw2 = make_client(tmp_path, rec)
    rw2.post("reports", body)
    assert len(rec.requests) == 1
    # counter file logs both cached and network calls
    rows = [json.loads(l) for l in (tmp_path / "rw" / "_calls.jsonl").read_text(encoding="utf-8").splitlines()]
    assert sum(1 for r in rows if not r["cached"]) == 1
    assert sum(1 for r in rows if r["cached"]) == 2


def test_daily_quota(tmp_path):
    rec = Recorder([
        httpx.Response(200, json={"data": [], "totalCount": 0}),
        httpx.Response(200, json={"data": [], "totalCount": 0}),
    ])
    rw = make_client(tmp_path, rec, max_calls_per_day=1)
    rw.post("reports", {"q": 1})
    with pytest.raises(ReliefWebQuotaError):
        rw.post("reports", {"q": 2})
    assert len(rec.requests) == 1
    # quota persists across client instances (counter file)
    rw2 = make_client(tmp_path, rec, max_calls_per_day=1)
    with pytest.raises(ReliefWebQuotaError):
        rw2.post("reports", {"q": 3})


def test_retry_on_429_then_200(tmp_path):
    rec = Recorder([
        httpx.Response(429, headers={"Retry-After": "0"}, json={"error": "slow down"}),
        httpx.Response(200, json={"data": [_item(7)], "totalCount": 1}),
    ])
    rw = make_client(tmp_path, rec)
    out = rw.post("reports", {"q": "x"})
    assert out["data"][0]["id"] == "7"
    assert len(rec.requests) == 2
    assert rw.network_calls == 2


def test_access_error_on_403(tmp_path):
    rec = Recorder([httpx.Response(403, text="appname not approved")])
    rw = make_client(tmp_path, rec)
    with pytest.raises(ReliefWebAccessError) as ei:
        rw.post("reports", {"q": "x"})
    assert "RELIEFWEB_APPNAME" in str(ei.value)


def test_search_disasters_body(tmp_path):
    rec = Recorder([httpx.Response(200, json={"data": [
        {"id": "15082", "fields": {"id": 15082, "name": "Nepal: Earthquake - Apr 2015", "glide": "EQ-2015-000048-NPL"}}
    ], "totalCount": 1})])
    rw = make_client(tmp_path, rec)
    hits = rw.search_disasters("Nepal earthquake 2015", limit=5)
    assert hits[0]["name"].startswith("Nepal")
    assert rec.requests[0].url.path == "/v2/disasters"
    body = rec.body(0)
    assert body["query"] == {"value": "Nepal earthquake 2015", "fields": ["name"]}
    assert body["preset"] == "analysis" and body["limit"] == 5
    assert "glide" in body["fields"]["include"]


# ---------------------------------------------------------------------------
# normalize_report
# ---------------------------------------------------------------------------
def test_normalize_report_body_present():
    item = _item(
        42,
        body="Plain body text.",
        **{"body-html": "<p>Ignored html</p>"},
        source=[{"name": "UN Office for the Coordination of Humanitarian Affairs", "shortname": "OCHA"},
                {"name": "International Federation of Red Cross And Red Crescent Societies", "shortname": "IFRC"}],
        format=[{"name": "Situation Report"}],
        disaster=[{"id": 15082, "name": "Nepal: Earthquake - Apr 2015", "glide": "EQ-2015-000048-NPL"}],
        primary_country={"iso3": "npl", "name": "Nepal"},
        language=[{"code": "en"}],
        url="https://reliefweb.int/node/42",
        file=[{"url": "https://reliefweb.int/f.pdf", "filename": "f.pdf", "mimetype": "application/pdf"}],
    )
    rec = normalize_report(item)
    assert rec["id"] == "42"
    assert rec["text"] == "Plain body text."
    assert rec["source"] == "OCHA/IFRC"
    assert rec["source_names"][0].startswith("UN Office")
    assert rec["date"] == "2015-04-25"
    assert rec["format"] == "Situation Report"
    assert rec["disaster_ids"] == ["15082"] and rec["glide"] == "EQ-2015-000048-NPL"
    assert rec["country_iso3"] == "npl" and rec["language"] == "en"
    assert rec["files"] == [{"url": "https://reliefweb.int/f.pdf", "filename": "f.pdf"}]
    assert rec["collection"] == "api" and rec["body_html"] == "<p>Ignored html</p>"


def test_normalize_report_html_fallback_and_flat_input():
    html = "<div><h2>Highlights</h2><p>Over 8,000 people <b>killed</b>.</p><script>x()</script><p>Second para.</p></div>"
    flat = {"id": "9", "title": "T", "date": {"original": "2015-05-01T12:34:56+00:00"},
            "body-html": html, "source": [{"shortname": "OCHA", "name": "OCHA"}]}
    rec = normalize_report(flat)
    assert rec["date"] == "2015-05-01"
    assert "Over 8,000 people killed." in rec["text"]
    assert "x()" not in rec["text"]
    assert "Highlights" in rec["text"] and "Second para." in rec["text"]
    assert rec["source"] == "OCHA"
    # missing everything → graceful
    empty = normalize_report({"id": 1, "fields": {}})
    assert empty["text"] == "" and empty["date"] is None and empty["source"] == ""


# ---------------------------------------------------------------------------
# manual ingest
# ---------------------------------------------------------------------------
LONG_PARA = ("Over 8,000 people have been confirmed dead and thousands more injured following the "
             "7.8 magnitude earthquake, according to the Government. Search and rescue teams are deployed. ") * 3

FIXTURE_HTML = f"""<!doctype html>
<html><head>
<meta property="og:title" content="Nepal: Earthquake Situation Report No. 5 (as of 30 April 2015) - Nepal | ReliefWeb">
<meta property="article:published_time" content="2015-04-30T10:00:00+00:00">
<title>Nepal: Earthquake Situation Report No. 5 - Nepal | ReliefWeb</title>
<style>.x{{color:red}}</style>
</head><body>
<nav><ul><li>Home</li><li>Updates</li><li>Countries</li><li>NAVNOISE</li></ul></nav>
<header><h1>Header noise</h1></header>
<main>
<article class="rw-report__content">
<h2>Highlights</h2>
<p>{LONG_PARA}</p>
<p>Shelter remains the priority need in the 14 most affected districts.</p>
</article>
</main>
<footer><p>FOOTERNOISE Terms of use</p></footer>
<script>window.noise = 1;</script>
</body></html>
"""


def test_extract_html_fixture():
    ext = extract_html(FIXTURE_HTML)
    assert ext["title"] == "Nepal: Earthquake Situation Report No. 5 (as of 30 April 2015)"
    assert ext["date"] == "2015-04-30"
    assert "Highlights" in ext["text"] and "Shelter remains" in ext["text"]
    assert "NAVNOISE" not in ext["text"] and "FOOTERNOISE" not in ext["text"] and "window.noise" not in ext["text"]


def _write_manifest(path: Path, rows: list[dict]):
    import csv
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["event", "title", "source", "date", "url", "file"])
        w.writeheader()
        w.writerows(rows)


def test_ingest_manual(tmp_path):
    raw = tmp_path / "reliefweb_manual"
    (raw / "nepal-eq-2015").mkdir(parents=True)
    (raw / "nepal-eq-2015" / "sitrep5.html").write_text(FIXTURE_HTML, encoding="utf-8")
    (raw / "nepal-eq-2015" / "sitrep5b.txt").write_text("IFRC operations update.\n\n" + LONG_PARA, encoding="utf-8")
    (raw / "nepal-eq-2015" / "short.txt").write_text("too short", encoding="utf-8")
    manifest = raw / "manifest.csv"
    _write_manifest(manifest, [
        {"event": "Nepal EQ 2015", "title": "", "source": "OCHA", "date": "",
         "url": "https://reliefweb.int/report/nepal/sitrep-5", "file": "nepal-eq-2015/sitrep5.html"},
        {"event": "Nepal EQ 2015", "title": "IFRC Ops Update 1", "source": "IFRC", "date": "2015-04-30",
         "url": "https://reliefweb.int/report/nepal/ifrc-1", "file": "nepal-eq-2015/sitrep5b.txt"},
        {"event": "Nepal EQ 2015", "title": "Short", "source": "OCHA", "date": "2015-05-01",
         "url": "", "file": "nepal-eq-2015/short.txt"},
        {"event": "Nepal EQ 2015", "title": "Missing", "source": "OCHA", "date": "2015-05-02",
         "url": "", "file": "nepal-eq-2015/missing.pdf"},
    ])
    out = tmp_path / "sitreps_human"
    summary = ingest(manifest, out)

    ev = out / "nepal-eq-2015"
    files = sorted(p.name for p in ev.glob("*.json"))
    assert files == ["2015-04-30.json", "2015-04-30_2.json"]  # date from HTML meta + manifest → collision suffix
    assert summary["counts"] == {"nepal-eq-2015": 2}
    assert len(summary["skipped"]) == 2
    reasons = " ".join(s["reason"] for s in summary["skipped"])
    assert "too short" in reasons and "not found" in reasons

    rec1 = json.loads((ev / "2015-04-30.json").read_text(encoding="utf-8"))
    assert rec1["title"] == "Nepal: Earthquake Situation Report No. 5 (as of 30 April 2015)"
    assert rec1["source"] == "OCHA" and rec1["collection"] == "manual"
    assert rec1["id"].startswith("manual-")
    assert rec1["url"] == "https://reliefweb.int/report/nepal/sitrep-5"
    assert "Shelter remains" in rec1["text"] and "NAVNOISE" not in rec1["text"]
    assert rec1["file"] == "nepal-eq-2015/sitrep5.html" or rec1["file"].endswith("sitrep5.html")
    rec2 = json.loads((ev / "2015-04-30_2.json").read_text(encoding="utf-8"))
    assert rec2["title"] == "IFRC Ops Update 1" and rec2["source"] == "IFRC"
    assert rec2["text"].startswith("IFRC operations update.")
    assert (ev / "_manifest_ingested.csv").exists()
    assert (ev / "_manifest_ingested.csv").read_text(encoding="utf-8").count("\n") == 3  # header + 2 rows


def test_ingest_cli_empty_manifest_creates_template(tmp_path, capsys):
    manifest = tmp_path / "manual" / "manifest.csv"
    rc = ingest_main(["--manifest", str(manifest), "--out", str(tmp_path / "out")])
    assert rc == 0
    assert manifest.exists()
    assert manifest.read_text(encoding="utf-8").strip() == "event,title,source,date,url,file"
    assert (manifest.parent / "README.md").exists()
    out = capsys.readouterr().out
    assert "no rows" in out and "manual_collection.md" in out


# ---------------------------------------------------------------------------
# additions from the Phase 0 review
# ---------------------------------------------------------------------------
def test_ingest_is_idempotent_and_cleans_stale_outputs(tmp_path):
    from src.ingest_manual_reliefweb import _clean_title

    raw = tmp_path / "reliefweb_manual"
    (raw / "ev").mkdir(parents=True)
    (raw / "ev" / "a.txt").write_text("A report.\n\n" + LONG_PARA, encoding="utf-8")
    (raw / "ev" / "b.txt").write_text("B report.\n\n" + LONG_PARA, encoding="utf-8")
    manifest = raw / "manifest.csv"
    rows = [
        {"event": "Ev", "title": "A", "source": "OCHA", "date": "30 April 2015", "url": "http://x/a", "file": "ev/a.txt"},
        {"event": "Ev", "title": "B", "source": "IFRC", "date": "2015-04-30", "url": "http://x/b", "file": "ev/b.txt"},
        {"event": "Ev", "title": "B again", "source": "IFRC", "date": "2015-04-30", "url": "http://x/b", "file": "ev/b.txt"},
    ]
    _write_manifest(manifest, rows)
    out = tmp_path / "out"
    s1 = ingest(manifest, out)
    names = sorted(p.name for p in (out / "ev").glob("*.json"))
    assert names == ["2015-04-30.json", "2015-04-30_2.json"]  # '30 April 2015' parsed; duplicate id skipped
    assert any("duplicate" in sk["reason"] for sk in s1["skipped"])
    rec = json.loads((out / "ev" / "2015-04-30.json").read_text(encoding="utf-8"))
    assert rec["file"] == "ev/a.txt"  # forward slashes on every platform

    # Re-run: same files, no new suffixes
    ingest(manifest, out)
    assert sorted(p.name for p in (out / "ev").glob("*.json")) == names

    # Drop a row: the stale output written earlier is removed
    _write_manifest(manifest, rows[:1])
    ingest(manifest, out)
    assert sorted(p.name for p in (out / "ev").glob("*.json")) == ["2015-04-30.json"]

    assert _clean_title("Nepal: Earthquake 2015 Situation Report No. 7 - Nepal | ReliefWeb") == \
        "Nepal: Earthquake 2015 Situation Report No. 7"
    assert _clean_title("Nepal Earthquake - Flash Update 1 | ReliefWeb") == "Nepal Earthquake - Flash Update 1"
    assert _clean_title("Türkiye/Syria: Earthquakes - Flash Update No. 3 - Syrian Arab Republic | ReliefWeb") == \
        "Türkiye/Syria: Earthquakes - Flash Update No. 3"


def test_extract_html_ignores_header_h1_and_time_when_meta_missing():
    html = """<html><head><title>Real Title - Nepal | ReliefWeb</title></head><body>
    <header><h1>Site header</h1><time datetime="2001-01-01T00:00:00Z">old</time></header>
    <article><h1>Body heading</h1><time datetime="2015-05-03T10:00:00+00:00">3 May</time>
    <p>%s</p></article></body></html>""" % LONG_PARA
    ext = extract_html(html)
    assert ext["title"] == "Real Title"
    assert ext["date"] == "2015-05-03"
    assert "Site header" not in ext["text"]


def test_date_to_bare_date_covers_whole_day(tmp_path):
    h = Recorder()
    c = make_client(tmp_path, h)
    c.fetch_reports(disaster_id=1, date_from="2015-04-25", date_to="2015-05-31", limit=10)
    body = h.body(0)
    rng = [cnd for cnd in body["filter"]["conditions"] if cnd.get("field") == "date.original"][0]["value"]
    assert rng == {"from": "2015-04-25T00:00:00+00:00", "to": "2015-05-31T23:59:59+00:00"}


def test_corrupt_counter_line_is_skipped(tmp_path):
    (tmp_path / "rw").mkdir()
    (tmp_path / "rw" / "_calls.jsonl").write_text("not json\n", encoding="utf-8")
    h = Recorder()
    c = make_client(tmp_path, h)  # must not raise
    assert c.post("reports", {"limit": 1})["totalCount"] == 0


def test_transport_error_wrapped_after_retries(tmp_path):
    def boom(request):
        raise httpx.ConnectError("no route", request=request)

    c = make_client(tmp_path, boom, max_attempts=2, min_interval_s=0.0)
    with pytest.raises(ReliefWebError, match="network failure after 2 attempt"):
        c.post("reports", {"limit": 1})


def test_appname_validation_rejects_bad_values():
    from src.reliefweb_api import _valid_appname

    assert _valid_appname("my-app.example.org") == "my-app.example.org"
    assert _valid_appname("# REQUIRED: pre-approved appname") is None
    assert _valid_appname("bad name&x=1") is None
    assert _valid_appname("   ") is None


# ---------------------------------------------------------------------------
# --scan / auto-manifest (the download cannot be automated; the bookkeeping is)
# ---------------------------------------------------------------------------
SITREP_BODY = (
    "Cyclone Idai made landfall near Beira city on 14 March. An estimated 1.85 million people are in need of "
    "humanitarian assistance across Sofala, Manica, Zambezia and Tete. According to the National Institute for "
    "Disaster Management, the death toll has risen to 598 people, with more than 1,600 injured. More than "
    "130,000 people are displaced in 136 accommodation sites. Access constraints persist; Buzi district is "
    "reachable only by air or boat. "
)


def _saved_html(url: str, title: str, published: str, body: str) -> str:
    return (f'<!doctype html><html><head><title>{title} | ReliefWeb</title>'
            f'<meta property="og:title" content="{title}">'
            f'<meta property="og:url" content="{url}">'
            f'<meta property="article:published_time" content="{published}">'
            f'</head><body><header><h1>ReliefWeb</h1><nav>NAVNOISE</nav></header>'
            f'<article><p>Source: UN Office for the Coordination of Humanitarian Affairs</p>'
            f'<p>{body}</p></article><footer>FOOTERNOISE</footer></body></html>')


def test_scan_builds_manifest_from_saved_files(tmp_path):
    from src.ingest_manual_reliefweb import scan_folder, write_manifest, load_manifest

    base = tmp_path / "reliefweb_manual"
    ev = base / "cyclone-idai-2019"
    ev.mkdir(parents=True)
    (ev / "sitrep-01.html").write_text(_saved_html(
        "https://reliefweb.int/report/mozambique/sitrep-no-1",
        "Mozambique: Cyclone Idai &amp; Floods Situation Report No. 1 (as of 2 April 2019)",
        "2019-04-02T00:00:00+00:00", SITREP_BODY), encoding="utf-8")
    (ev / "ocha-sitrep-05.txt").write_text(
        "Mozambique: Cyclone Idai & Floods Situation Report No. 5 (as of 6 April 2019)\n\n"
        "United Nations Office for the Coordination of Humanitarian Affairs\n\n" + SITREP_BODY * 2, encoding="utf-8")
    (ev / "ifrc-dref.txt").write_text(
        "Mozambique: Tropical Cyclone Idai DREF Operation Update (as of 12 April 2019)\n\n"
        "International Federation of Red Cross and Red Crescent Societies\n\n" + SITREP_BODY * 2, encoding="utf-8")
    (ev / "badsave.html").write_text("<html><body><p>Verifying you are human.</p></body></html>", encoding="utf-8")

    rows = scan_folder(ev, base)
    assert len(rows) == 3  # the bot-challenge save is rejected as too short
    by_file = {Path(r["file"]).name: r for r in rows}
    assert set(by_file) == {"sitrep-01.html", "ocha-sitrep-05.txt", "ifrc-dref.txt"}

    html_row = by_file["sitrep-01.html"]
    assert html_row["event"] == "cyclone-idai-2019"
    assert html_row["date"] == "2019-04-02"
    assert html_row["source"] == "OCHA"
    assert html_row["url"] == "https://reliefweb.int/report/mozambique/sitrep-no-1"
    assert "Situation Report No. 1" in html_row["title"]
    assert html_row["file"] == "cyclone-idai-2019/sitrep-01.html"  # posix, relative to the manifest dir

    pdf_row = by_file["ocha-sitrep-05.txt"]  # PDF-style: date + source from the text, url unknown
    assert pdf_row["date"] == "2019-04-06" and pdf_row["source"] == "OCHA" and pdf_row["url"] == ""
    assert "Situation Report No. 5" in pdf_row["title"]
    assert by_file["ifrc-dref.txt"]["source"] == "IFRC"

    mpath = base / "manifest.csv"
    assert write_manifest(mpath, rows) == 3
    loaded = load_manifest(mpath)
    assert len(loaded) == 3 and {r["source"] for r in loaded} == {"OCHA", "IFRC"}

    # rescanning refreshes rows in place and preserves a hand-filled field on an untouched row
    hand = [dict(r, url="https://reliefweb.int/report/mozambique/hand-filled") if
            Path(r["file"]).name == "ifrc-dref.txt" else r for r in loaded]
    write_manifest(mpath, hand, merge=False)
    (ev / "ifrc-dref.txt").rename(ev / "ifrc-dref-renamed.txt")  # a different file -> a new row
    write_manifest(mpath, scan_folder(ev, base))
    final = {Path(r["file"]).name: r for r in load_manifest(mpath)}
    assert final["ifrc-dref.txt"]["url"].endswith("hand-filled")  # untouched row kept
    assert "ifrc-dref-renamed.txt" in final


def test_scan_then_ingest_end_to_end(tmp_path):
    from src.ingest_manual_reliefweb import main as ingest_main

    base = tmp_path / "reliefweb_manual"
    ev = base / "cyclone-idai-2019"
    ev.mkdir(parents=True)
    (ev / "s1.html").write_text(_saved_html(
        "https://reliefweb.int/report/mozambique/s1", "Cyclone Idai Situation Report No. 1 (as of 2 April 2019)",
        "2019-04-02T00:00:00+00:00", SITREP_BODY), encoding="utf-8")
    (ev / "s2.html").write_text(_saved_html(
        "https://reliefweb.int/report/mozambique/s2", "Cyclone Idai Situation Report No. 2 (as of 3 April 2019)",
        "2019-04-03T00:00:00+00:00", SITREP_BODY), encoding="utf-8")
    manifest = base / "manifest.csv"
    out = tmp_path / "sitreps_human"

    assert ingest_main(["--scan", str(ev), "--manifest", str(manifest)]) == 0
    assert ingest_main(["--manifest", str(manifest), "--out", str(out)]) == 0
    files = sorted(p.name for p in (out / "cyclone-idai-2019").glob("*.json"))
    assert files == ["2019-04-02.json", "2019-04-03.json"]
    rec = json.loads((out / "cyclone-idai-2019" / "2019-04-02.json").read_text(encoding="utf-8"))
    assert rec["source"] == "OCHA" and rec["collection"] == "manual"
    assert "1.85 million" in rec["text"] and "NAVNOISE" not in rec["text"]


def test_scan_on_empty_or_missing_dir(tmp_path, capsys):
    from src.ingest_manual_reliefweb import main as ingest_main

    empty = tmp_path / "empty"
    empty.mkdir()
    assert ingest_main(["--scan", str(empty), "--manifest", str(tmp_path / "m.csv")]) == 2
    assert "No readable reports" in capsys.readouterr().out
    assert ingest_main(["--scan", str(tmp_path / "nope"), "--manifest", str(tmp_path / "m.csv")]) == 2
    assert "not a directory" in capsys.readouterr().out


def test_source_and_date_inference_helpers():
    from src.ingest_manual_reliefweb import date_from_text, infer_source, url_from_html

    assert infer_source("... published by OCHA on behalf of partners") == "OCHA"
    assert infer_source("International Federation of Red Cross and Red Crescent Societies") == "IFRC"
    assert infer_source("World Food Programme country brief") == "WFP"
    assert infer_source("no organisation named here") == ""
    assert date_from_text("Situation Report No. 3 (as of 17 March 2019)") == "2019-03-17"
    assert date_from_text("Flash Update, 2 Apr 2019") == "2019-04-02"
    assert date_from_text("no date at all") is None
    assert url_from_html('<html><head><link rel="canonical" href="https://reliefweb.int/report/x"></head></html>') \
        == "https://reliefweb.int/report/x"
    assert url_from_html("<html><head></head><body>x</body></html>") is None
