# Work list — puebla_mexico_earthquake_2017

**Tweet stream we hold:** 2,015 tweets, 2017-09-20 → 2017-10-06 (15 days).

Coverage is dense and well-suited to the tweet window: OCHA (Flash Update, humanitarian snapshots EN/ES, and a full set of per-state "Earthquake 7.1 ... snapshot (As of 02nd October 2017)" products for Puebla, Morelos-area states, Mexico City, Guerrero, Oaxaca and USAR teams), the UN Resident Coordinator's office in Mexico (Reporte de Situación No. 01, 21 Sep; Situation Report No. 02, 26 Sep), and UNICEF's numbered "Mexico Earthquake Humanitarian Situation Report" series (No. 1 on 20 Sep through No. 9 in November plus a three-month review). At least 14 of the verified documents fall inside 2017-09-20 → 2017-10-06, with the OCHA/RC products clustering exactly on 20-22 Sep and 02 Oct. I could not verify UNICEF Nos. 2, 4, 5, 7 or any IFRC DREF/EPoA for MDRMX before the search budget ran out — use the disaster hub page to fill those gaps.

> ⚠️ **Links are UNVERIFIED.** They were gathered by web search while the ReliefWeb API is
> unavailable (appname pending), and could not be machine-confirmed — the audit pass ran out
> of search budget before it could corroborate any of them. Structurally malformed and
> audit-flagged links have already been removed, but a link here may still 404. If one does,
> use the event hub below. Run `python -m src.verify_worklists` once `RELIEFWEB_APPNAME` is
> approved to confirm every link against the API and rewrite these pages authoritatively.

## How to collect

Save each report into `data/raw/reliefweb_manual/puebla-mexico-earthquake-2017/` (PDF via *Download report* if offered, else Ctrl+S → "Webpage, HTML only"; any filename), then:

```
python -m src.ingest_manual_reliefweb --scan data/raw/reliefweb_manual/puebla-mexico-earthquake-2017
python -m src.ingest_manual_reliefweb
```

`--scan` reads title/date/source/URL back out of the files, so you type nothing.

**Event hub (lists everything, use it to fill gaps):** https://reliefweb.int/disaster/eq-2017-000138-mex

On the hub: *Updates* tab → left sidebar → **Format = Situation Report**, **Source = OCHA** (repeat for **IFRC**), sort oldest first.

## Priority A — inside the tweet window (16 reports)

| Date | Source | Kind | Title | Link |
|---|---|---|---|---|
| 2017-09-20 | OCHA | Snapshot | Mexico: Earthquake Humanitarian Snapshot (as of 20 Sep 2017) | [open](https://reliefweb.int/report/mexico/mexico-earthquake-humanitarian-snapshot-20-sep-2017) |
| 2017-09-20 | OCHA | Snapshot | México: Snapshot Humanitario - Terremoto (al 20 de septiembre 2017) | [open](https://reliefweb.int/report/mexico/m-xico-snapshot-humanitario-terremoto-al-20-de-septiembre-2017) |
| 2017-09-20 | UNICEF | Situation Report | Mexico Earthquake Humanitarian Situation Report No.1 - 20 September 2017 | [open](https://reliefweb.int/report/mexico/mexico-earthquake-humanitarian-situation-report-no1-20-september-2017) |
| 2017-09-20 | other | Situation Report | Reporte preliminar por sismo magnitud 7.1, 20 de septiembre de 2017 | [open](https://reliefweb.int/report/mexico/reporte-preliminar-por-sismo-magnitud-71-20-de-septiembre-de-2017) |
| 2017-09-21 | OCHA | Situation Report | México: Sismo 7.1 grados - Reporte de Situación No. 01 de la Oficina del Coordin | [open](https://reliefweb.int/report/mexico/m-xico-sismo-71-grados-reporte-de-situaci-n-no-01-de-la-oficina-del-coordinador) |
| 2017-09-21 | other | Situation Report | Puebla, Mexico M7.1 Earthquake Situation Report 1 (September 21, 2017) | [open](https://reliefweb.int/report/mexico/puebla-mexico-m71-earthquake-situation-report-1-september-21-2017) |
| 2017-09-22 | OCHA | Snapshot | Central America: Earthquakes September 2017 (as of 22 Sep 2017) | [open](https://reliefweb.int/report/mexico/central-america-earthquakes-september-2017-22-sep-2017-0) |
| 2017-09-26 | OCHA | Situation Report | Mexico: Earthquake magnitude 7.1 Situation Report No. 02 from the United Nations | [open](https://reliefweb.int/report/mexico/mexico-earthquake-magnitude-71-situation-report-no-02-united-nations-mexico-26th) |
| 2017-10-02 | OCHA | Snapshot | Mexico: Earthquake 7.1 Puebla snapshot (As of 02nd October 2017, 14:00 hrs.) | [open](https://reliefweb.int/report/mexico/mexico-earthquake-71-puebla-snapshot-02nd-october-2017-1400-hrs) |
| 2017-10-02 | OCHA | Snapshot | Mexico: Earthquake 7.1 Mexico City snapshot (As of 02nd October 2017, 14:00 hrs. | [open](https://reliefweb.int/report/mexico/mexico-earthquake-71-mexico-city-snapshot-02nd-october-2017-1400-hrs) |
| 2017-10-02 | OCHA | Snapshot | Mexico: Earthquake 7.1 Guerrero snapshot (As of 02nd October 2017, 14:00 hrs.) | [open](https://reliefweb.int/report/mexico/mexico-earthquake-71-guerrero-snapshot-02nd-october-2017-1400-hrs) |
| 2017-10-02 | OCHA | Snapshot | Mexico: Earthquake 7.1 Oaxaca snapshot (As of 02nd October 2017, 14:00 hrs.) | [open](https://reliefweb.int/report/mexico/mexico-earthquake-71-oaxaca-snapshot-02nd-october-2017-1400-hrs) |
| 2017-10-02 | OCHA | Snapshot | Mexico: Earthquake 7.1 USAR teams snapshot (As of 02nd October 2017) | [open](https://reliefweb.int/report/mexico/mexico-earthquake-71-usar-teams-snapshot-02nd-october-2017) |
| 2017-10-02 | OCHA | Snapshot | México: Sismo 7.1 Infografía de Equipos USAR (Al 02 de octubre de 2017) | [open](https://reliefweb.int/report/mexico/m-xico-sismo-71-infograf-de-equipos-usar-al-02-de-octubre-de-2017) |
| 2017-10-02 | OCHA | Snapshot | México: Sismo 7.1 Infografía de la Ciudad de México (Actualizado al 02 de Octubr | [open](https://reliefweb.int/report/mexico/m-xico-sismo-71-infograf-de-la-ciudad-de-m-xico-actualizado-al-02-de-octubre-de-2017) |
| 2017-10-02 | OCHA | Snapshot | México: Sismo 7.1 Infografía del estado de Oaxaca (Actualizado al 02 de Octubre  | [open](https://reliefweb.int/report/mexico/m-xico-sismo-71-infograf-del-estado-de-oaxaca-actualizado-al-02-de-octubre-de-2017) |

## Priority B — outside the window (8 reports)

Still valuable: these give the schema induction its later, consolidated end of the spectrum.

| Date | Source | Kind | Title | Link |
|---|---|---|---|---|
| 2017-09-19 | OCHA | Flash Update | Mexico: Earthquake Flash Update No.1, 19 September, 2017 | [open](https://reliefweb.int/report/mexico/mexico-earthquake-flash-update-no1-19-september-2017) |
| 2017-10-13 | UNICEF | Situation Report | Mexico Earthquake Humanitarian Situation Report No. 6 - 13 October 2017 | [open](https://reliefweb.int/report/mexico/mexico-earthquake-humanitarian-situation-report-no-6-13-october-2017) |
| 2017-11-10 | UNICEF | Situation Report | Mexico Earthquake Humanitarian Situation Report No. 8 - 10 November 2017 | [open](https://reliefweb.int/report/mexico/mexico-earthquake-humanitarian-situation-report-no-8-10-november-2017) |
| 2017-11-24 | UNICEF | Situation Report | Mexico Earthquake Humanitarian Situation Report No. 9 - 24 November 2017 | [open](https://reliefweb.int/report/mexico/mexico-earthquake-humanitarian-situation-report-no-9-24-november-2017) |
| 2017-12-19 | UNICEF | Situation Report | Mexico Earthquakes Humanitarian Situation Report - Three Month Review, 19 Decemb | [open](https://reliefweb.int/report/mexico/mexico-earthquakes-humanitarian-situation-report-three-month-review-19-december-2017) |
| ? | OCHA | Snapshot | Overview of the state of Morelos - Earthquake 7.1 (19/09/2017) | [open](https://reliefweb.int/report/mexico/overview-state-morelos-earthquake-71-19092017) |
| ? | OCHA | Snapshot | México: Infografía de situación Sismo 7.1 (19/09/2017) | [open](https://reliefweb.int/report/mexico/m-xico-infograf-de-situaci-n-sismo-71-19092017) |
| ? | other | Situation Report | Reporte de acciones de la Coordinación Nacional de Protección Civil, tras el sis | [open](https://reliefweb.int/report/mexico/reporte-de-acciones-de-la-coordinaci-n-nacional-de-protecci-n-civil-tras-el-sismo-del) |

---

If a link 404s it was a search artifact — use the hub above; and please tell me so I can drop it.
