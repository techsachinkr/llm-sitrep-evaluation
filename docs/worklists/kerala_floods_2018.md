# Work list — kerala_floods_2018

**Tweet stream we hold:** 7,984 tweets, 2018-08-17 → 2018-09-12 (27 days).

Coverage exists but is thin and fragmented: there is NO dedicated OCHA "Kerala Floods Situation Report No. N" series (OCHA has no country office in India), so the only OCHA products are the regional "Asia and the Pacific: Weekly Regional Humanitarian Snapshot" issues that carry a Kerala paragraph. The recurring, dated series that do exist are Sphere India's "Humanitarian Snapshot Report" (numbered; only No. 10 of 20 Aug surfaced in search, implying ~10+ siblings not indexed by the search engine) and Humanity Road's numbered "2018 India Kerala Flooding Situation Report" (Nos. 1-2 found, both social-media-derived, so weak as independent ground truth). IFRC has one DREF operation (MDRIN020) with an EPoA Update and a Final Report, but those are undated in search results and mostly post-date the tweet window. Overlap with the 2018-08-17 to 2018-09-12 tweet stream is decent for the snapshot products (20 Aug, 27 Aug, 3 Sep) but the series is far too sparse for a dense day-by-day sitrep alignment; the user should open the disaster hub https://reliefweb.int/disaster/fl-2018-000134-ind and filter by source=Sphere India / OCHA to recover the missing Humanitarian Snapshot Report numbers 1-9 and 11+.

> ⚠️ **Links are UNVERIFIED.** They were gathered by web search while the ReliefWeb API is
> unavailable (appname pending), and could not be machine-confirmed — the audit pass ran out
> of search budget before it could corroborate any of them. Structurally malformed and
> audit-flagged links have already been removed, but a link here may still 404. If one does,
> use the event hub below. Run `python -m src.verify_worklists` once `RELIEFWEB_APPNAME` is
> approved to confirm every link against the API and rewrite these pages authoritatively.

## How to collect

Save each report into `data/raw/reliefweb_manual/kerala-floods-2018/` (PDF via *Download report* if offered, else Ctrl+S → "Webpage, HTML only"; any filename), then:

```
python -m src.ingest_manual_reliefweb --scan data/raw/reliefweb_manual/kerala-floods-2018
python -m src.ingest_manual_reliefweb
```

`--scan` reads title/date/source/URL back out of the files, so you type nothing.

**Event hub (lists everything, use it to fill gaps):** https://reliefweb.int/disaster/fl-2018-000134-ind

On the hub: *Updates* tab → left sidebar → **Format = Situation Report**, **Source = OCHA** (repeat for **IFRC**), sort oldest first.

## Priority A — inside the tweet window (5 reports)

| Date | Source | Kind | Title | Link |
|---|---|---|---|---|
| 2018-08-18 | other | Situation Report | 2018 India Kerala Flooding Situation Report 2 – period covered: August 17 - 18,  | [open](https://reliefweb.int/report/india/2018-india-kerala-flooding-situation-report-2-period-covered-august-17-18-2018) |
| 2018-08-20 | other | Snapshot | Humanitarian Snapshot Report 10: Flood Situation in Kerala (20 August 2018) | [open](https://reliefweb.int/report/india/humanitarian-snapshot-report-10-flood-situation-kerala-20-august-2018) |
| 2018-08-20 | OCHA | Snapshot | Asia and the Pacific: Weekly Regional Humanitarian Snapshot (14 - 20 August 2018 | [open](https://reliefweb.int/report/india/asia-and-pacific-weekly-regional-humanitarian-snapshot-14-20-august-2018) |
| 2018-08-27 | OCHA | Snapshot | Asia and the Pacific: Weekly Regional Humanitarian Snapshot (21 - 27 August 2018 | [open](https://reliefweb.int/report/indonesia/asia-and-pacific-weekly-regional-humanitarian-snapshot-21-27-august-2018) |
| 2018-09-03 | IFRC | Situation Report | Overall Update on Kerala Flood Relief 2018 (3rd September 2018) | [open](https://reliefweb.int/report/india/overall-update-kerala-flood-relief-2018-3rd-september-2018) |

## Priority B — outside the window (4 reports)

Still valuable: these give the schema induction its later, consolidated end of the spectrum.

| Date | Source | Kind | Title | Link |
|---|---|---|---|---|
| 2018-08-16 | other | Situation Report | 2018 India Kerala Flooding Situation Report 1 – period covered: August 16, 2018 | [open](https://reliefweb.int/report/india/2018-india-kerala-flooding-situation-report-1-period-covered-august-16-2018) |
| ? | IFRC | IFRC DREF / operations update | India: Kerala Floods DREF n° MDRIN020 Emergency Plan of Action (EPoA) Update | [open](https://reliefweb.int/report/india/india-kerala-floods-dref-n-mdrin020-emergency-plan-action-epoa-update) |
| ? | IFRC | IFRC DREF / operations update | India: Kerala Floods DREF n° MDRIN020 Emergency Plan of Action Final Report | [open](https://reliefweb.int/report/india/india-kerala-floods-dref-n-mdrin020-emergency-plan-action-final-report) |
| ? | other | Other | Kerala Floods 2018 Joint Detailed Needs Assessment Report | [open](https://reliefweb.int/report/india/kerala-floods-2018-joint-detailed-needs-assessment-report) |

---

If a link 404s it was a search artifact — use the hub above; and please tell me so I can drop it.
