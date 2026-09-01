# Work list — cyclone_idai_2019

**Tweet stream we hold:** 3,933 tweets, 2019-03-15 → 2019-04-16 (29 days).

OCHA (ROSEA + Mozambique/Zimbabwe RC offices) published a near-daily series for Idai: Mozambique Flash Updates No. 1-15 (15 Mar - 1 Apr 2019) followed immediately by Mozambique Situation Reports No. 1-22 (2 Apr - 20 May 2019), plus a parallel Zimbabwe series (Floods Flash Updates No. 1-6, 17-26 Mar; Floods/Cyclone Situation Reports No. 1-4, 27 Mar - 24 Apr) and regional Cyclone Idai Snapshots. Overlap with the 2019-03-15 to 2019-04-16 tweet window is excellent - roughly one OCHA product per day for the entire window across Mozambique and Zimbabwe, supplemented by UNICEF/IOM/WFP-ETC agency sitreps and IFRC DREF/EPoA documents. Malawi has no standalone OCHA Idai sitrep series (only regional snapshots and ACAPS/IFRC products), so Malawi is the thin leg.

> ⚠️ **Links are UNVERIFIED.** They were gathered by web search while the ReliefWeb API is
> unavailable (appname pending), and could not be machine-confirmed — the audit pass ran out
> of search budget before it could corroborate any of them. Structurally malformed and
> audit-flagged links have already been removed, but a link here may still 404. If one does,
> use the event hub below. Run `python -m src.verify_worklists` once `RELIEFWEB_APPNAME` is
> approved to confirm every link against the API and rewrite these pages authoritatively.

## How to collect

Save each report into `data/raw/reliefweb_manual/cyclone-idai-2019/` (PDF via *Download report* if offered, else Ctrl+S → "Webpage, HTML only"; any filename), then:

```
python -m src.ingest_manual_reliefweb --scan data/raw/reliefweb_manual/cyclone-idai-2019
python -m src.ingest_manual_reliefweb
```

`--scan` reads title/date/source/URL back out of the files, so you type nothing.

**Event hub (lists everything, use it to fill gaps):** https://reliefweb.int/disaster/tc-2019-000021-moz

On the hub: *Updates* tab → left sidebar → **Format = Situation Report**, **Source = OCHA** (repeat for **IFRC**), sort oldest first.

## Priority A — inside the tweet window (40 reports)

| Date | Source | Kind | Title | Link |
|---|---|---|---|---|
| 2019-03-15 | OCHA | Flash Update | Mozambique: Cyclone Idai Flash Update No. 1, 15 March 2019 | [open](https://reliefweb.int/report/mozambique/mozambique-cyclone-idai-flash-update-no-1-15-march-2019) |
| 2019-03-16 | OCHA | Flash Update | Mozambique: Cyclone Idai Flash Update No. 2, 16 March 2019 | [open](https://reliefweb.int/report/mozambique/mozambique-cyclone-idai-flash-update-no-2-16-march-2019) |
| 2019-03-17 | OCHA | Flash Update | Zimbabwe: Floods Flash Update No. 1, 17 March 2019 | [open](https://reliefweb.int/report/zimbabwe/zimbabwe-floods-flash-update-no-1-17-march-2019) |
| 2019-03-17 | OCHA | Flash Update | Mozambique: Cyclone Idai Flash Update No. 3, 17 March 2019 | [open](https://reliefweb.int/report/mozambique/mozambique-cyclone-idai-flash-update-no-3-17-march-2019) |
| 2019-03-18 | OCHA | Flash Update | Mozambique: Cyclone Idai Flash Update No. 4, 18 March 2019 | [open](https://reliefweb.int/report/mozambique/mozambique-cyclone-idai-flash-update-no-4-18-march-2019) |
| 2019-03-18 | OCHA | Flash Update | Zimbabwe: Floods Flash Update No. 2, 18 March 2019 | [open](https://reliefweb.int/report/zimbabwe/zimbabwe-floods-flash-update-no-2-18-march-2019) |
| 2019-03-19 | OCHA | Flash Update | Zimbabwe: Floods Flash Update No. 3, 19 March 2019 | [open](https://reliefweb.int/report/zimbabwe/zimbabwe-floods-flash-update-no-3-19-march-2019) |
| 2019-03-19 | OCHA | Flash Update | Mozambique: Cyclone Idai & Floods Flash Update No. 5, 19 March 2019 | [open](https://reliefweb.int/report/mozambique/mozambique-cyclone-idai-floods-flash-update-no-5-19-march-2019) |
| 2019-03-21 | OCHA | Flash Update | Mozambique: Cyclone Idai & Floods Flash Update No. 6, 21 March 2019 | [open](https://reliefweb.int/report/mozambique/mozambique-cyclone-idai-floods-flash-update-no-6-21-march-2019) |
| 2019-03-21 | OCHA | Flash Update | Zimbabwe: Floods Flash Update No. 4, 21 March 2019 | [open](https://reliefweb.int/report/zimbabwe/zimbabwe-floods-flash-update-no-4-21-march-2019) |
| 2019-03-23 | OCHA | Flash Update | Mozambique: Cyclone Idai & Floods Flash Update No. 7, 23 March 2019 | [open](https://reliefweb.int/report/mozambique/mozambique-cyclone-idai-floods-flash-update-no-7-23-march-2019) |
| 2019-03-23 | OCHA | Flash Update | Zimbabwe: Floods Flash Update No. 5, 23 March 2019 | [open](https://reliefweb.int/report/zimbabwe/zimbabwe-floods-flash-update-no-5-23-march-2019) |
| 2019-03-24 | OCHA | Flash Update | Mozambique: Cyclone Idai & Floods Flash Update No. 8, 24 March 2019 | [open](https://reliefweb.int/report/mozambique/mozambique-cyclone-idai-floods-flash-update-no-8-24-march-2019) |
| 2019-03-25 | OCHA | Flash Update | Mozambique: Cyclone Idai & Floods Flash Update No. 9, 25 March 2019 | [open](https://reliefweb.int/report/mozambique/mozambique-cyclone-idai-floods-flash-update-no-9-25-march-2019) |
| 2019-03-26 | OCHA | Flash Update | Mozambique: Cyclone Idai & Floods Flash Update No. 10, 26 March 2019 | [open](https://reliefweb.int/report/mozambique/mozambique-cyclone-idai-floods-flash-update-no-10-26-march-2019) |
| 2019-03-26 | OCHA | Flash Update | Zimbabwe: Floods Flash Update No. 6, 26 March 2019 | [open](https://reliefweb.int/report/zimbabwe/zimbabwe-floods-flash-update-no-6-26-march-2019) |
| 2019-03-26 | OCHA | Snapshot | Southern Africa: Cyclone Idai Snapshot (as of 26 March 2019) | [open](https://reliefweb.int/report/mozambique/southern-africa-cyclone-idai-snapshot-26-march-2019) |
| 2019-03-27 | OCHA | Flash Update | Mozambique: Cyclone Idai & Floods Flash Update No. 11, 27 March 2019 | [open](https://reliefweb.int/report/mozambique/mozambique-cyclone-idai-floods-flash-update-no-11-27-march-2019) |
| 2019-03-27 | OCHA | Situation Report | Zimbabwe: Floods Situation Report No. 1, As of 27 March 2019 | [open](https://reliefweb.int/report/zimbabwe/zimbabwe-floods-situation-report-no-1-27-march-2019) |
| 2019-03-28 | OCHA | Flash Update | Mozambique: Cyclone Idai & Floods Flash Update No. 12, 28 March 2019 | [open](https://reliefweb.int/report/mozambique/mozambique-cyclone-idai-floods-flash-update-no-12-28-march-2019) |
| 2019-03-29 | OCHA | Flash Update | Mozambique: Cyclone Idai & Floods Flash Update No. 13, 29 March 2019 | [open](https://reliefweb.int/report/mozambique/mozambique-cyclone-idai-floods-flash-update-no-13-29-march-2019) |
| 2019-03-30 | OCHA | Flash Update | Mozambique: Cyclone Idai & Floods Flash Update No. 14, 30 March 2019 | [open](https://reliefweb.int/report/mozambique/mozambique-cyclone-idai-floods-flash-update-no-14-30-march-2019) |
| 2019-03-30 | OCHA | Other | Zimbabwe: Situation Update, 30 March 2019 | [open](https://reliefweb.int/report/zimbabwe/zimbabwe-situation-update-30-march-2019) |
| 2019-04-01 | OCHA | Flash Update | Mozambique: Cyclone Idai & Floods Flash Update No. 15, 01 April 2019 | [open](https://reliefweb.int/report/mozambique/mozambique-cyclone-idai-floods-flash-update-no-15-01-april-2019) |
| 2019-04-02 | OCHA | Situation Report | Mozambique: Cyclone Idai & Floods Situation Report No. 1 (as of 2 April 2019) | [open](https://reliefweb.int/report/mozambique/mozambique-cyclone-idai-floods-situation-report-no-1-2-april-2019) |
| 2019-04-03 | OCHA | Situation Report | Mozambique: Cyclone Idai & Floods Situation Report No. 2 (as of 3 April 2019) | [open](https://reliefweb.int/report/mozambique/mozambique-cyclone-idai-floods-situation-report-no-2-3-april-2019) |
| 2019-04-04 | OCHA | Situation Report | Mozambique: Cyclone Idai & Floods Situation Report No. 3 (as of 4 April 2019) | [open](https://reliefweb.int/report/mozambique/mozambique-cyclone-idai-floods-situation-report-no-3-4-april-2019) |
| 2019-04-05 | OCHA | Situation Report | Mozambique: Cyclone Idai & Floods Situation Report No. 4 (as of 5 April 2019) | [open](https://reliefweb.int/report/mozambique/mozambique-cyclone-idai-floods-situation-report-no-4-5-april-2019) |
| 2019-04-06 | OCHA | Situation Report | Mozambique: Cyclone Idai & Floods Situation Report No. 5 (as of 6 April 2019) | [open](https://reliefweb.int/report/mozambique/mozambique-cyclone-idai-floods-situation-report-no-5-6-april-2019) |
| 2019-04-07 | OCHA | Situation Report | Mozambique: Cyclone Idai & Floods Situation Report No. 6 (as of 7 April 2019) | [open](https://reliefweb.int/report/mozambique/mozambique-cyclone-idai-floods-situation-report-no-6-7-april-2019) |
| 2019-04-08 | OCHA | Situation Report | Mozambique: Cyclone Idai & Floods Situation Report No. 7 (as of 8 April 2019) | [open](https://reliefweb.int/report/mozambique/mozambique-cyclone-idai-floods-situation-report-no-7-8-april-2019) |
| 2019-04-08 | other | Situation Report | Mozambique Cyclone Idai Response: Situation Report No. 3, 04-08 April 2019 | [open](https://reliefweb.int/report/mozambique/mozambique-cyclone-idai-response-situation-report-no-3-04-08-april-2019) |
| 2019-04-09 | OCHA | Situation Report | Mozambique: Cyclone Idai & Floods Situation Report No. 8 (as of 9 April 2019) | [open](https://reliefweb.int/report/mozambique/mozambique-cyclone-idai-floods-situation-report-no-8-9-april-2019) |
| 2019-04-10 | OCHA | Situation Report | Zimbabwe: Floods Situation Report No. 2, As of 10 April 2019 | [open](https://reliefweb.int/report/zimbabwe/zimbabwe-floods-situation-report-no-2-10-april-2019) |
| 2019-04-10 | OCHA | Situation Report | Mozambique: Cyclone Idai & Floods Situation Report No. 9 (as of 10 April 2019) | [open](https://reliefweb.int/report/mozambique/mozambique-cyclone-idai-floods-situation-report-no-9-10-april-2019) |
| 2019-04-12 | OCHA | Situation Report | Mozambique: Cyclone Idai & Floods Situation Report No. 11 (as of 12 April 2019) | [open](https://reliefweb.int/report/mozambique/mozambique-cyclone-idai-floods-situation-report-no-11-12-april-2019) |
| 2019-04-12 | UNICEF | Situation Report | UNICEF Mozambique Cyclone Idai Situation Report #5 (08 - 12 April 2019) | [open](https://reliefweb.int/report/mozambique/unicef-mozambique-cyclone-idai-situation-report-5-08-12-april-2019) |
| 2019-04-13 | OCHA | Situation Report | Mozambique: Cyclone Idai & Floods Situation Report No. 12 (as of 13 April 2019) | [open](https://reliefweb.int/report/mozambique/mozambique-cyclone-idai-floods-situation-report-no-12-13-april-2019) |
| 2019-04-14 | WFP | Situation Report | Mozambique: Cyclone Idai - ETC Situation Report #10 (Reporting Period: 11/04/19  | [open](https://reliefweb.int/report/mozambique/mozambique-cyclone-idai-etc-situation-report-10-reporting-period-110419-140419) |
| 2019-04-15 | OCHA | Situation Report | Mozambique: Cyclone Idai & Floods Situation Report No. 14 (as of 15 April 2019) | [open](https://reliefweb.int/report/mozambique/mozambique-cyclone-idai-floods-situation-report-no-14-15-april-2019) |

## Priority B — outside the window (19 reports)

Still valuable: these give the schema induction its later, consolidated end of the spectrum.

| Date | Source | Kind | Title | Link |
|---|---|---|---|---|
| 2019-03-10 | OCHA | Snapshot | Southern Africa: Cyclone Idai Snapshot (as of 10 March 2019) | [open](https://reliefweb.int/report/malawi/southern-africa-cyclone-idai-snapshot-10-march-2019) |
| 2019-03-12 | OCHA | Snapshot | Southern Africa: Cyclone Idai Snapshot (as of 12 March 2019) | [open](https://reliefweb.int/report/malawi/southern-africa-cyclone-idai-snapshot-12-march-2019) |
| 2019-04-18 | OCHA | Situation Report | Mozambique: Cyclone Idai & Floods Situation Report No. 16 (as of 18 April 2019) | [open](https://reliefweb.int/report/mozambique/mozambique-cyclone-idai-floods-situation-report-no-16-18-april-2019) |
| 2019-04-19 | UNICEF | Situation Report | UNICEF Mozambique Cyclone Idai Situation Report #6 (15 - 19 April 2019) | [open](https://reliefweb.int/report/mozambique/unicef-mozambique-cyclone-idai-situation-report-6-15-19-april-2019) |
| 2019-04-19 | other | Situation Report | Mozambique: Tropical Cyclone Idai, Situation Report 03 (April 19 2019) | [open](https://reliefweb.int/report/mozambique/mozambique-tropical-cyclone-idai-situation-report-03-april-19-2019) |
| 2019-04-20 | OCHA | Situation Report | Mozambique: Cyclone Idai & Floods Situation Report No. 17 (as of 20 April 2019) | [open](https://reliefweb.int/report/mozambique/mozambique-cyclone-idai-floods-situation-report-no-17-20-april-2019) |
| 2019-04-22 | OCHA | Situation Report | Mozambique: Cyclone Idai & Floods Situation Report No. 18 (as of 22 April 2019) | [open](https://reliefweb.int/report/mozambique/mozambique-cyclone-idai-floods-situation-report-no-18-22-april-2019) |
| 2019-04-24 | OCHA | Situation Report | Zimbabwe: Cyclone & Floods Situation Report No. 4, As of 24 April 2019 | [open](https://reliefweb.int/report/zimbabwe/zimbabwe-cyclone-floods-situation-report-no-4-24-april-2019) |
| 2019-04-29 | OCHA | Situation Report | Mozambique: Cyclone Idai & Floods Situation Report No. 19 (As of 29 April 2019) | [open](https://reliefweb.int/report/mozambique/mozambique-cyclone-idai-floods-situation-report-no-19-29-april-2019) |
| 2019-05-01 | UNICEF | Situation Report | UNICEF Mozambique Cyclone Idai Situation Report #7 (22 April - 01 May 2019) | [open](https://reliefweb.int/report/mozambique/unicef-mozambique-cyclone-idai-situation-report-7-22-april-01-may-2019) |
| 2019-05-06 | OCHA | Situation Report | Mozambique: Cyclone Idai & Floods Situation Report No. 20 (as of 6 May 2019) | [open](https://reliefweb.int/report/mozambique/mozambique-cyclone-idai-floods-situation-report-no-20-6-may-2019) |
| 2019-05-10 | other | Situation Report | Mozambique: Tropical Cyclones Idai and Kenneth, Situation Report 06 (10 May 2019 | [open](https://reliefweb.int/report/mozambique/mozambique-tropical-cyclones-idai-and-kenneth-situation-report-06-10-may-2019) |
| 2019-05-13 | OCHA | Situation Report | Mozambique: Cyclone Idai & Floods Situation Report No. 21 (As of 13 May 2019) | [open](https://reliefweb.int/report/mozambique/mozambique-cyclone-idai-floods-situation-report-no-21-13-may-2019) |
| 2019-05-20 | OCHA | Situation Report | Mozambique: Cyclone Idai & Floods Situation Report No. 22 (As of 20 May 2019) | [open](https://reliefweb.int/report/mozambique/mozambique-cyclone-idai-floods-situation-report-no-22-20-may-2019) |
| 2019-07-10 | OCHA | Snapshot | Southern Africa: Cyclones Idai and Kenneth Snapshot, as of 10 July 2019 | [open](https://reliefweb.int/report/mozambique/southern-africa-cyclones-idai-and-kenneth-snapshot-10-july-2019) |
| ? | IFRC | IFRC DREF / operations update | Mozambique: Tropical Cyclone Idai - Emergency Plan of Action (EPoA), DREF n°: MD | [open](https://reliefweb.int/report/mozambique/mozambique-tropical-cyclone-idai-emergency-plan-action-epoa-dref-n-mdrmz014-pmz045) |
| ? | IFRC | IFRC DREF / operations update | Mozambique: Tropical Cyclones Idai and Kenneth - Emergency Appeal n° MDRMZ014 Op | [open](https://reliefweb.int/report/mozambique/mozambique-tropical-cyclones-idai-and-kenneth-emergency-appeal-n-mdrmz014) |
| ? | IFRC | IFRC DREF / operations update | Mozambique: Tropical Cyclones Idai and Kenneth - Emergency Appeal n° MDRMZ014 Op | [open](https://reliefweb.int/report/mozambique/mozambique-tropical-cyclones-idai-and-kenneth-emergency-appeal-n-mdrmz014-0) |
| ? | IFRC | IFRC DREF / operations update | Zimbabwe: Tropical Cyclone Idai Final Report, DREF Operation n°: MDRZW014 | [open](https://reliefweb.int/report/zimbabwe/zimbabwe-tropical-cyclone-idai-final-report-dref-operation-n-mdrzw014) |

---

If a link 404s it was a search artifact — use the hub above; and please tell me so I can drop it.
