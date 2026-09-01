# Work list — hurricane_maria_2017

**Tweet stream we hold:** 7,278 tweets, 2017-09-20 → 2017-10-02 (12 days).

Coverage is dense and multi-agency. OCHA ROLAC ran two parallel series (a regional "Caribbean: Hurricane Season Situation Report" numbered series plus "Caribbean: Hurricane Maria Flash Update" Nos. 1-2), then a country-level "Dominica: Hurricane Maria Situation Report" series (No. 1 on 25 Sep through No. 9 on 2 Nov, roughly weekly; No. 3 exists in the numbering but did not surface in any search, so leave it as a gap to fill from the hub). Alongside these, CDEMA (regional lead agency), PAHO/WHO, WFP/Logistics Sector, UNICEF Eastern Caribbean and IFRC (MDRDM003) all published their own series. Overlap with the 2017-09-20 → 2017-10-02 tweet window is good: 12 of the 28 verified reports fall inside it, including the earliest OCHA, CDEMA, PAHO and WFP products, though the country-level OCHA sitrep series only starts on 25 Sep (day 5 of the window).

> ⚠️ **Links are UNVERIFIED.** They were gathered by web search while the ReliefWeb API is
> unavailable (appname pending), and could not be machine-confirmed — the audit pass ran out
> of search budget before it could corroborate any of them. Structurally malformed and
> audit-flagged links have already been removed, but a link here may still 404. If one does,
> use the event hub below. Run `python -m src.verify_worklists` once `RELIEFWEB_APPNAME` is
> approved to confirm every link against the API and rewrite these pages authoritatively.

## How to collect

Save each report into `data/raw/reliefweb_manual/hurricane-maria-2017/` (PDF via *Download report* if offered, else Ctrl+S → "Webpage, HTML only"; any filename), then:

```
python -m src.ingest_manual_reliefweb --scan data/raw/reliefweb_manual/hurricane-maria-2017
python -m src.ingest_manual_reliefweb
```

`--scan` reads title/date/source/URL back out of the files, so you type nothing.

**Event hub (lists everything, use it to fill gaps):** https://reliefweb.int/disaster/tc-2017-000136-atg

On the hub: *Updates* tab → left sidebar → **Format = Situation Report**, **Source = OCHA** (repeat for **IFRC**), sort oldest first.

## Priority A — inside the tweet window (12 reports)

| Date | Source | Kind | Title | Link |
|---|---|---|---|---|
| 2017-09-20 | OCHA | Situation Report | The Caribbean: Hurricane Season Situation Report No. 08 (as of 20 September 2017 | [open](https://reliefweb.int/report/dominica/caribbean-hurricane-season-situation-report-no-08-20-september-2017) |
| 2017-09-20 | WHO | Situation Report | Hurricanes Irma and Maria Situation Report No. 7, 20 September 2017 – 19:00 EST  | [open](https://reliefweb.int/report/dominica/hurricanes-irma-and-maria-situation-report-no-7-20-september-2017-1900-est) |
| 2017-09-20 | other | Situation Report | CDEMA Situation Report #1 - Hurricane Maria - as of 9:00pm AST on September 20,  | [open](https://reliefweb.int/report/dominica/cdema-situation-report-1-hurricane-maria-900pm-ast-september-20-2017) |
| 2017-09-20 | other | Situation Report | Hurricane Maria Situation Report 1 (September 20, 2017) | [open](https://reliefweb.int/report/dominica/hurricane-maria-situation-report-1-september-20-2017) |
| 2017-09-21 | OCHA | Flash Update | Caribbean: Hurricane Maria Flash Update No.2 21 September, 2017 | [open](https://reliefweb.int/report/dominica/caribbean-hurricane-maria-flash-update-no2-21-september-2017) |
| 2017-09-21 | WFP | Situation Report | WFP Hurricane Irma/Maria Situation Report #8 - 21 September 2017 | [open](https://reliefweb.int/report/dominica/wfp-hurricane-irmamaria-situation-report-8-21-september-2017) |
| 2017-09-22 | other | Situation Report | CDEMA Situation Report #2 - Hurricane Maria - as of 9:00pm AST on September 22,  | [open](https://reliefweb.int/report/dominica/cdema-situation-report-2-hurricane-maria-900pm-ast-september-22-2017) |
| 2017-09-22 | other | Situation Report | Hurricane Maria Situation Report 2 (September 22, 2017) | [open](https://reliefweb.int/report/dominica/hurricane-maria-situation-report-2-september-22-2017) |
| 2017-09-25 | OCHA | Situation Report | Dominica: Hurricane Maria Situation Report No. 1 (as of 25 September 2017) | [open](https://reliefweb.int/report/dominica/dominica-hurricane-maria-situation-report-no-1-25-september-2017) |
| 2017-09-26 | OCHA | Snapshot | Dominica: Hurricane Maria Snapshot (as of 26 Sep 2017) | [open](https://reliefweb.int/report/dominica/dominica-hurricane-maria-snapshot-26-sep-2017) |
| 2017-09-28 | WFP | Situation Report | Hurricanes Irma and Maria Situation Update - 28 September 2017 (Logistics Sector | [open](https://reliefweb.int/report/antigua-and-barbuda/hurricanes-irma-and-maria-situation-update-28-september-2017) |
| 2017-10-02 | OCHA | Situation Report | Dominica: Hurricane Maria Situation Report No. 2 (as of 2 October, 2017) | [open](https://reliefweb.int/report/dominica/dominica-hurricane-maria-situation-report-no-2-2-october-2017) |

## Priority B — outside the window (21 reports)

Still valuable: these give the schema induction its later, consolidated end of the spectrum.

| Date | Source | Kind | Title | Link |
|---|---|---|---|---|
| 2017-09-18 | OCHA | Situation Report | The Caribbean: Hurricane Season Situation Report No. 7 (as of 18 September 2017) | [open](https://reliefweb.int/report/antigua-and-barbuda/caribbean-hurricane-season-situation-report-no-7-18-september-2017) |
| 2017-09-19 | OCHA | Flash Update | Caribbean: Hurricane Maria Flash Update No.1 19 September, 2017 – 17:00 (GMT-5) | [open](https://reliefweb.int/report/dominica/caribbean-hurricane-maria-flash-update-no1-19-september-2017-1700-gmt-5) |
| 2017-10-04 | other | Situation Report | CDEMA Situation Report #8 - Hurricane Maria, October 4, 2017 | [open](https://reliefweb.int/report/dominica/cdema-situation-report-8-hurricane-maria-october-4-2017) |
| 2017-10-05 | WFP | Situation Report | Hurricanes Irma and Maria Situation Update - 5 October 2017 (Logistics Sector) | [open](https://reliefweb.int/report/antigua-and-barbuda/hurricanes-irma-and-maria-situation-update-5-october-2017) |
| 2017-10-06 | OCHA | Situation Report | The Caribbean: Hurricane Season Situation Report No. 11 (as of 6 October 2017) | [open](https://reliefweb.int/report/dominica/caribbean-hurricane-season-situation-report-no-11-6-october-2017) |
| 2017-10-06 | other | Situation Report | CDEMA Situation Report #9 - Hurricane Maria, October 6, 2017 | [open](https://reliefweb.int/report/dominica/cdema-situation-report-9-hurricane-maria-october-6-2017) |
| 2017-10-07 | OCHA | Situation Report | Dominica: Hurricane Maria Situation Report No. 4 (as of 7 October, 2017) | [open](https://reliefweb.int/report/dominica/dominica-hurricane-maria-situation-report-no-4-7-october-2017) |
| 2017-10-11 | OCHA | Situation Report | Dominica: Hurricane Maria Situation Report No. 5 (as of 11 October, 2017) | [open](https://reliefweb.int/report/dominica/dominica-hurricane-maria-situation-report-no-5-11-october-2017) |
| 2017-10-14 | OCHA | Situation Report | Dominica: Hurricane Maria Situation Report No. 6 (as of 14 October 2017) | [open](https://reliefweb.int/report/dominica/dominica-hurricane-maria-situation-report-no-6-14-october-2017) |
| 2017-10-15 | WFP | Situation Report | Hurricanes Irma and Maria Situation Update - 15 October 2017 (Logistics Sector) | [open](https://reliefweb.int/report/antigua-and-barbuda/hurricanes-irma-and-maria-situation-update-15-october-2017) |
| 2017-10-18 | UNICEF | Situation Report | UNICEF Eastern Caribbean Humanitarian Situation Report, 18 October 2017 | [open](https://reliefweb.int/report/antigua-and-barbuda/unicef-eastern-caribbean-humanitarian-situation-report-18-october-2017) |
| 2017-10-19 | OCHA | Situation Report | Dominica: Hurricane Maria Situation Report No. 7 (as of 19 October, 2017) | [open](https://reliefweb.int/report/dominica/dominica-hurricane-maria-situation-report-no-7-19-october-2017) |
| 2017-10-26 | OCHA | Situation Report | Dominica: Hurricane Maria Situation Report No. 8 (as of 26 October 2017) | [open](https://reliefweb.int/report/dominica/dominica-hurricane-maria-situation-report-no-8-26-october-2017) |
| 2017-11-02 | OCHA | Situation Report | Dominica: Hurricane Maria Situation Report No. 9 (as of 02 November 2017) | [open](https://reliefweb.int/report/dominica/dominica-hurricane-maria-situation-report-no-9-02-november-2017) |
| 2017-11-22 | UNICEF | Situation Report | UNICEF LACRO Caribbean Hurricanes Sitrep No. 8, 22 November 2017 | [open](https://reliefweb.int/report/dominica/unicef-lacro-caribbean-hurricanes-sitrep-no-8-22-november-2017) |
| 2018-01-17 | UNICEF | Situation Report | UNICEF Eastern Caribbean Humanitarian Situation Report, 17 January 2018 | [open](https://reliefweb.int/report/dominica/unicef-eastern-caribbean-humanitarian-situation-report-17-january-2018) |
| ? | UNICEF | Situation Report | UNICEF Eastern Caribbean Humanitarian Situation Report No. 9 | [open](https://reliefweb.int/report/dominica/unicef-eastern-caribbean-humanitarian-situation-report-no-9) |
| ? | IFRC | IFRC DREF / operations update | Dominica: Hurricane Maria - Emergency Plan of Action (MDRDM003) | [open](https://reliefweb.int/report/dominica/dominica-hurricane-maria-emergency-plan-action-mdrdm003) |
| ? | IFRC | IFRC DREF / operations update | Dominica: Hurricane Maria - Emergency Plan of Action operation update n°3 (MDRDM | [open](https://reliefweb.int/report/dominica/dominica-hurricane-maria-emergency-plan-action-operation-update-n-3-mdrdm003) |
| ? | IFRC | IFRC DREF / operations update | Dominica: Hurricane Maria - Emergency Appeal Operations Update - 12-month operat | [open](https://reliefweb.int/report/dominica/dominica-hurricane-maria-emergency-appeal-operations-update-12-month-operations) |
| ? | IFRC | IFRC DREF / operations update | Dominica: Hurricane Maria - Final Report MDRDM003 | [open](https://reliefweb.int/report/dominica/dominica-hurricane-maria-final-report-mdrdm003) |

---

If a link 404s it was a search artifact — use the hub above; and please tell me so I can drop it.
