# Manual collection of ReliefWeb situation reports (fallback for PLAN §2a)

Use this when `RELIEFWEB_APPNAME` is not yet set / approved. The ReliefWeb API v2
requires a **pre-approved appname** (enforced since 2025-11-01); we never query it
with an unregistered one. The website itself answers scripts with HTTP 202/403
(bot challenge), so the pages must be saved **by hand in a browser** — nobody
should scrape reliefweb.int. All ingestion code (`src/ingest_manual_reliefweb.py`)
reads only local files and never fetches.

Target volume (PLAN §2): ~60–150 sitreps across ≥3 events; aim for 15–30 per
event, spread across the response timeline (roughly the first 8 weeks after
onset — first days, weeks 1–2, weeks 3–4, weeks 5–8), so the schema induction
sees early flash-style reports as well as later consolidated ones.

## 1. Find the reports (browser)

1. Open https://reliefweb.int/updates
2. Filters (left sidebar): **Format → Situation Report**; **Source → OCHA** (then repeat
   for **IFRC**); **Disaster →** the event, e.g. "Nepal: Earthquake - Apr 2015",
   "Türkiye/Syria: Earthquakes - Feb 2023", "Pakistan: Floods - Jun 2022".
   Optionally **Language → English**. Sort by date, oldest first.
3. Bookmark the filtered search URL. Open https://reliefweb.int/search/converter,
   paste the search URL, and save the equivalent API query it produces (into
   `data/raw/reliefweb_manual/<event>/api_query.json`) — once the appname is
   approved you can replay it through `src/reliefweb_api.py` for citation-grade
   metadata.

## 2. Save each report

For each report page in the list:

- **(a) preferred:** if the page has a **"Download report"** (PDF) button, save the
  PDF under `data/raw/reliefweb_manual/<event>/` (e.g. `nepal-eq-2015/ocha-sitrep-05.pdf`).
- **(b) otherwise:** in the browser use *Save page as… → "Webpage, HTML only"* and
  save it under the same folder (`.html`). (Do not use "complete" — it stores assets we don't need.)
- **(c)** add a row to `data/raw/reliefweb_manual/manifest.csv`:

  | column | value |
  |---|---|
  | `event`  | short event key, same for all reports of the event, e.g. `nepal-eq-2015` |
  | `title`  | report title as shown on ReliefWeb |
  | `source` | `OCHA`, `IFRC`, … (short name) |
  | `date`   | "Originally published" date, `YYYY-MM-DD` |
  | `url`    | the report page URL (for citation) |
  | `file`   | path relative to `data/raw/reliefweb_manual/`, e.g. `nepal-eq-2015/ocha-sitrep-05.pdf` |

  A header-only template lives at `docs/manifest_template.csv`; the ingest script
  copies it to `data/raw/reliefweb_manual/manifest.csv` on first run.

Record **every** report you save in the manifest — the manifest is the citation
record for the paper (title + URL) and the audit trail for the corpus.

## 3. Ingest (the manifest is built for you)

**You do not fill in the manifest by hand.** Save the files, then let the scanner read the
metadata back out of them:

```
python -m src.ingest_manual_reliefweb --scan data/raw/reliefweb_manual/<event>
```

For each saved report this infers **title** (og:title / the PDF's first heading), **date**
(article:published_time, or an "as of <date>" line in the text), **source** (OCHA / IFRC / WFP /
UNICEF / WHO / IOM / UNHCR, from the publisher line) and **url** (og:url or the canonical link —
HTML saves only; PDFs have no URL inside them, so those rows are listed for you to paste one in,
or leave blank). It reports exactly which rows still need a human. Re-running `--scan` refreshes
rows for files it sees and leaves your hand-edits to other rows alone. Mis-saved pages (a
bot-challenge page instead of the report) yield too little text and are skipped with a warning
naming the file.

Then ingest:



```
python -m src.ingest_manual_reliefweb            # default manifest + output dir
python -m src.ingest_manual_reliefweb --manifest path/to/manifest.csv --out data/processed/sitreps_human
```

Accepted inputs: `.pdf` (preferred), `.html`/`.htm` (browser "HTML only" save), and `.txt`/`.md`
(if you simply select-all/copy the report body into a text file — the most robust option when a
page saves badly). Dates in the manifest may be `YYYY-MM-DD` or `30 April 2015`.

The ingest is idempotent: re-running with the same manifest rewrites the same files
(`<date>.json`, then `<date>_2.json`, … in manifest order for reports sharing a date), removes
outputs from rows you deleted, and skips duplicate URLs.

Output: `data/processed/sitreps_human/<event>/<date>.json`
(`{id, title, source, date, url, text, collection: "manual", file, ingested_at}`),
`<date>_2.json` etc. when two reports share a date, and
`<event>/_manifest_ingested.csv`. Rows whose extracted text is shorter than 200
characters are skipped with a warning (usually a mis-saved page — re-save as PDF).
The script prints per-event counts; check them against your target volume.

## 4. API etiquette (once the appname is approved)

The client (`src/reliefweb_api.py`) enforces ReliefWeb's published limits: ≤ 1,000 entries per
call, ≤ 1,000 calls per day (persisted counter in `data/raw/reliefweb/_calls.jsonl`, UTC day), and
≥ 1 second between network calls; every raw response is cached under `data/raw/reliefweb/<endpoint>/`
so re-runs never hit the API. Never query with an unregistered appname and never scrape
reliefweb.int pages programmatically.

## 5. Later: replay through the API

When `RELIEFWEB_APPNAME` is approved, run
`python -m src.reliefweb_api --smoke --query "Nepal earthquake 2015" --n 10`
and then the Phase 1 collector; API records carry `collection: "api"` and can
supersede or complement the manual ones (match on URL).
