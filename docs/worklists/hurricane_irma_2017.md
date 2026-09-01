# Work list — hurricane_irma_2017

**Tweet stream we hold:** 9,399 tweets, 2017-09-06 → 2017-09-17 (12 days).

Coverage is dense and multi-agency. The spine is the OCHA ROLAC regional series "The Caribbean: Hurricane Irma Situation Report No. 01-06" (6-15 Sep, roughly every 1-2 days, continuing as "Caribbean: Hurricane Season Situation Report No. 7-8" once Maria arrived), plus a near-daily UN Resident Coordinator's Office series for Cuba (No. 1 through 19+, 7 Sep to mid-Oct) and parallel UNICEF Eastern Caribbean, WFP, WHO/PAHO and CDEMA sitrep series. Overlap with the 2017-09-06 to 2017-09-17 tweet window is excellent: the great majority of the acute-phase reports fall inside it, with the OCHA regional series alone giving 6+ in-window documents.

> ⚠️ **Links are UNVERIFIED.** They were gathered by web search while the ReliefWeb API is
> unavailable (appname pending), and could not be machine-confirmed — the audit pass ran out
> of search budget before it could corroborate any of them. Structurally malformed and
> audit-flagged links have already been removed, but a link here may still 404. If one does,
> use the event hub below. Run `python -m src.verify_worklists` once `RELIEFWEB_APPNAME` is
> approved to confirm every link against the API and rewrite these pages authoritatively.

## How to collect

Save each report into `data/raw/reliefweb_manual/hurricane-irma-2017/` (PDF via *Download report* if offered, else Ctrl+S → "Webpage, HTML only"; any filename), then:

```
python -m src.ingest_manual_reliefweb --scan data/raw/reliefweb_manual/hurricane-irma-2017
python -m src.ingest_manual_reliefweb
```

`--scan` reads title/date/source/URL back out of the files, so you type nothing.

**Event hub (lists everything, use it to fill gaps):** https://reliefweb.int/disaster/tc-2017-000125-dom

On the hub: *Updates* tab → left sidebar → **Format = Situation Report**, **Source = OCHA** (repeat for **IFRC**), sort oldest first.

## Priority A — inside the tweet window (21 reports)

| Date | Source | Kind | Title | Link |
|---|---|---|---|---|
| 2017-09-06 | OCHA | Situation Report | The Caribbean: Hurricane Irma Situation Report No. 01 (as of 6 of September 2017 | [open](https://reliefweb.int/report/dominican-republic/caribbean-hurricane-irma-situation-report-no-01-6-september-2017) |
| 2017-09-06 | UNICEF | Situation Report | UNICEF Eastern Caribbean Humanitarian Situation Report #1, 6 September 2017 | [open](https://reliefweb.int/report/antigua-and-barbuda/unicef-eastern-caribbean-humanitarian-situation-report-1-6-september-2017) |
| 2017-09-07 | WHO | Flash Update | Hurricane IRMA 2017 EMT Situation Update # 1, September 7 2017 | [open](https://reliefweb.int/report/antigua-and-barbuda/hurricane-irma-2017-emt-situation-update-1-september-7-2017) |
| 2017-09-07 | other | Situation Report | Response to Hurricane Irma: Cuba Situation Report No. 1 - Office of the Resident | [open](https://reliefweb.int/report/cuba/response-hurricane-irma-cuba-situation-report-no-1-office-resident-coordinator-07092017) |
| 2017-09-07 | other | Situation Report | Hurricane Irma Situation Report 2 (September 7, 2017) | [open](https://reliefweb.int/report/anguilla/hurricane-irma-situation-report-2-september-7-2017) |
| 2017-09-08 | OCHA | Situation Report | El Caribe: Huracan Irma Informe de Situacion No. 2 (al 08 Septiembre 2017) [Span | [open](https://reliefweb.int/report/cuba/el-caribe-hurac-n-irma-informe-de-situaci-n-no-2-al-08-septiembre-2017) |
| 2017-09-08 | OCHA | Situation Report | The Caribbean: Hurricane Irma Situation Report No. 03 (as of 8 of September 2017 | [open](https://reliefweb.int/report/antigua-and-barbuda/caribbean-hurricane-irma-situation-report-no-03-8-september-2017) |
| 2017-09-08 | OCHA | Situation Report | El Caribe: Huracan Irma Informe de Situacion No. 3 (al 08 Septiembre 2017) [Span | [open](https://reliefweb.int/report/cuba/el-caribe-hurac-n-irma-informe-de-situaci-n-no-3-al-08-septiembre-2017) |
| 2017-09-08 | UNICEF | Situation Report | UNICEF Eastern Caribbean Humanitarian Situation Report, 8 September 2017 | [open](https://reliefweb.int/report/antigua-and-barbuda/unicef-eastern-caribbean-humanitarian-situation-report-8-september-2017) |
| 2017-09-08 | WFP | Situation Report | WFP Hurricane Irma Situation Report #3 - 8 September 2017 | [open](https://reliefweb.int/report/haiti/wfp-hurricane-irma-situation-report-3-8-september-2017) |
| 2017-09-08 | other | Situation Report | Hurricane Irma Situation Report 3 (September 8, 2017) | [open](https://reliefweb.int/report/antigua-and-barbuda/hurricane-irma-situation-report-3-september-8-2017) |
| 2017-09-09 | other | Situation Report | Response to Hurricane Irma: Cuba Situation Report No. 3 - Office of the Resident | [open](https://reliefweb.int/report/cuba/response-hurricane-irma-cuba-situation-report-no-3-office-resident-coordinator-09092017) |
| 2017-09-10 | OCHA | Situation Report | The Caribbean: Hurricane Irma Situation Report No. 04 (as of 10 of September 201 | [open](https://reliefweb.int/report/antigua-and-barbuda/caribbean-hurricane-irma-situation-report-no-04-10-september-2017) |
| 2017-09-11 | other | Situation Report | Respuesta al huracan Irma - Cuba Reporte de Situacion No. 04 de la Oficina de la | [open](https://reliefweb.int/report/cuba/respuesta-al-hurac-n-irma-cuba-reporte-de-situaci-n-no-04-de-la-oficina-de-la) |
| 2017-09-11 | WHO | Situation Report | WHO/PAHO Hurricane IRMA 2017 Situation Report No. 3, September 11 2017 | [open](https://reliefweb.int/report/turks-and-caicos-islands/whopaho-hurricane-irma-2017-situation-report-no-3-september-11-2017) |
| 2017-09-13 | OCHA | Situation Report | El Caribe: Huracan Irma Reporte de Situacion No. 5 (al 13 de Septiembre 2017) [S | [open](https://reliefweb.int/report/antigua-and-barbuda/el-caribe-hurac-n-irma-reporte-de-situaci-n-no-5-al-13-de-septiembre-2017) |
| 2017-09-13 | other | Situation Report | CDEMA Situation Report #6 - Hurricane Irma - as of 9:00pm on September 13th, 201 | [open](https://reliefweb.int/report/antigua-and-barbuda/cdema-situation-report-6-hurricane-irma-900pm-september-13th-2017) |
| 2017-09-15 | OCHA | Snapshot | Cuba: Impact from Hurricane Irma (15 Sep 2017) | [open](https://reliefweb.int/report/cuba/cuba-impact-hurricane-irma-15-sep-2017) |
| 2017-09-15 | OCHA | Situation Report | The Caribbean: Hurricane Irma Situation Report No. 6 (as of 15 September 2017) | [open](https://reliefweb.int/report/antigua-and-barbuda/caribbean-hurricane-irma-situation-report-no-6-15-september-2017) |
| 2017-09-16 | OCHA | Flash Update | Cuba: Hurricane Irma update (as of 16 September 2017) | [open](https://reliefweb.int/report/cuba/cuba-hurricane-irma-update-16-september-2017) |
| ? | other | Situation Report | Respuesta al huracan Irma - Cuba Reporte de Situacion No. 02 de la Oficina de la | [open](https://reliefweb.int/report/cuba/respuesta-al-hurac-n-irma-cuba-reporte-de-situaci-n-no-02-de-la-oficina-de-la) |

## Priority B — outside the window (17 reports)

Still valuable: these give the schema induction its later, consolidated end of the spectrum.

| Date | Source | Kind | Title | Link |
|---|---|---|---|---|
| 2017-09-05 | WFP | Situation Report | WFP Hurricane Irma Situation Report #1 - 5 September 2017 | [open](https://reliefweb.int/report/haiti/wfp-hurricane-irma-situation-report-1-5-september-2017) |
| 2017-09-18 | OCHA | Situation Report | The Caribbean: Hurricane Season Situation Report No. 7 (as of 18 September 2017) | [open](https://reliefweb.int/report/antigua-and-barbuda/caribbean-hurricane-season-situation-report-no-7-18-september-2017) |
| 2017-09-19 | OCHA | Flash Update | Cuba: Hurricane Irma update (as of 19 September 2017) | [open](https://reliefweb.int/report/cuba/cuba-hurricane-irma-update-19-september-2017) |
| 2017-09-20 | OCHA | Situation Report | The Caribbean: Hurricane Season Situation Report No. 08 (as of 20 September 2017 | [open](https://reliefweb.int/report/dominica/caribbean-hurricane-season-situation-report-no-08-20-september-2017) |
| 2017-09-20 | WHO | Situation Report | Hurricanes Irma and Maria Situation Report No. 7, 20 September 2017 - 19:00 EST | [open](https://reliefweb.int/report/dominica/hurricanes-irma-and-maria-situation-report-no-7-20-september-2017-1900-est) |
| 2017-09-20 | IOM | Situation Report | The Caribbean: Hurricanes Irma and Jose Response Situation Report #2, 20 Septemb | [open](https://reliefweb.int/report/dominica/caribbean-hurricanes-irma-and-jose-response-situation-report-2-20-september-2017) |
| 2017-09-21 | WFP | Situation Report | WFP Hurricane Irma/Maria Situation Report #8 - 21 September 2017 | [open](https://reliefweb.int/report/dominica/wfp-hurricane-irmamaria-situation-report-8-21-september-2017) |
| 2017-09-22 | other | Situation Report | Hurricane Devastation in the Caribbean - Situation Report No. 1, September 22, 2 | [open](https://reliefweb.int/report/cuba/hurricane-devastation-caribbean-situation-report-no-1-september-22-2017) |
| 2017-09-28 | other | Situation Report | Response to Hurricane Irma: Cuba Situation Report No. 16 - Office of the Residen | [open](https://reliefweb.int/report/cuba/response-hurricane-irma-cuba-situation-report-no-16-office-resident-coordinator-28092017) |
| 2017-10-02 | other | Situation Report | Response to Hurricane Irma: Cuba Situation Report No. 17 - Office of the Residen | [open](https://reliefweb.int/report/cuba/response-hurricane-irma-cuba-situation-report-no-17-office-resident-coordinator-02102017) |
| 2017-10-02 | other | Situation Report | Hurricane Devastation in the Caribbean - Situation Report No. 2, October 2, 2017 | [open](https://reliefweb.int/report/cuba/hurricane-devastation-caribbean-situation-report-no-2-october-2-2017) |
| 2017-10-12 | other | Situation Report | Response to Hurricane Irma: Cuba Situation Report No. 19 - Office of the Residen | [open](https://reliefweb.int/report/cuba/response-hurricane-irma-cuba-situation-report-no-19-office-resident-coordinator-12102017) |
| ? | UNICEF | Situation Report | UNICEF Eastern Caribbean Humanitarian Situation Report No. 9 [date not shown in  | [open](https://reliefweb.int/report/dominica/unicef-eastern-caribbean-humanitarian-situation-report-no-9) |
| ? | IFRC | IFRC DREF / operations update | Antigua and Barbuda: Hurricane Irma - Emergency Plan of Action (EPoA) n MDRAG003 | [open](https://reliefweb.int/report/antigua-and-barbuda/antigua-and-barbuda-hurricane-irma-emergency-plan-action-epoa-n-mdrag003) |
| ? | IFRC | IFRC DREF / operations update | Cuba: Hurricane Irma - Emergency appeal operation update no.2 (MDRCU004) | [open](https://reliefweb.int/report/cuba/cuba-hurricane-irma-emergency-appeal-operation-update-no2-mdrcu004) |
| ? | IFRC | IFRC DREF / operations update | Haiti: Hurricane Irma (MDRHT014) DREF Final Report | [open](https://reliefweb.int/report/haiti/haiti-hurricane-irma-mdrht014-dref-final-report) |
| ? | IFRC | Other | Americas: Hurricane Irma - Information Bulletin N 3 [date not shown in search re | [open](https://reliefweb.int/report/antigua-and-barbuda/americas-hurricane-irma-information-bulletin-n-3) |

---

If a link 404s it was a search artifact — use the hub above; and please tell me so I can drop it.
