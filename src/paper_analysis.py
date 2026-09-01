"""Publication-facing audit tables and sensitivity analyses.

The experiment tables preserve every run, including failed judgements and a few stale duplicate
rows left by manifest filename repairs.  This module constructs the exact analysis population used
in the paper: manifest-admitted documents, one row per (judge, document, slot), and usable verdicts
only.  It also writes the inferential and sensitivity tables needed to recompute the manuscript.

No report or social-media text is written below ``results/``.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Sequence

import yaml

from src.judge_consensus import fleiss_kappa
from src.metrics import bootstrap_ci, judge_agreement
from src.util import ensure_dir, read_json

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "tables"
PAPER_TABLES = RESULTS / "paper"
DATA = ROOT / "data" / "processed"
SEED = 17
SLOTS = 15

VALIDATION_SAMPLE = DATA / "judgements" / "validation_sample.csv"
VALIDATION_KEY = DATA / "judgements" / "DO_NOT_OPEN_UNTIL_SCORED__judge_key.csv"
VALIDATION_JUDGES = (
    ("deepseek-v4-pro", "deepseek"),
    ("gpt-5.6-luna", "luna"),
    ("qwen3.8-flash", "qwen"),
    ("gemini-3.7-flash", "gemini"),
)
VALIDATION_LABELS = {"absent", "partial", "present"}


@dataclass(frozen=True)
class SchemaSlot:
    id: str
    name: str


def schema_slots() -> list[SchemaSlot]:
    with open(DATA / "schema.yaml", "r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    return [SchemaSlot(str(row["id"]), str(row["name"])) for row in data["slots"]]

EVENTS = (
    "cyclone-idai-2019",
    "hurricane-irma-2017",
    "hurricane-maria-2017",
    "puebla-mexico-earthquake-2017",
    "kerala-floods-2018",
)
PRIMARY_EVENTS = EVENTS[:3]

JUDGE_FILES: dict[str, dict[str, Path]] = {
    "cyclone-idai-2019": {
        "deepseek-v4-pro": RESULTS / "slot_judgements.csv",
        "gpt-5.6-luna": RESULTS / "xfam" / "slot_judgements.csv",
        "gemini-3.7-flash": RESULTS / "idai" / "gemini" / "slot_judgements.csv",
        "qwen3.8-flash": RESULTS / "qwen" / "slot_judgements.csv",
    },
    **{
        event: {
            "gpt-5.6-luna": RESULTS / event / "luna" / "slot_judgements.csv",
            "gemini-3.7-flash": RESULTS / event / "gemini" / "slot_judgements.csv",
        }
        for event in EVENTS[1:]
    },
}

CORRELATION_FILES = {
    "cyclone-idai-2019": RESULTS / "correlations_gemini.csv",
    "hurricane-irma-2017": RESULTS / "hurricane-irma-2017" / "correlations_luna.csv",
    "hurricane-maria-2017": RESULTS / "hurricane-maria-2017" / "correlations_luna.csv",
}

REFERENCE_FILES = {
    "cyclone-idai-2019": RESULTS / "reference_metrics.csv",
    "hurricane-irma-2017": RESULTS / "hurricane-irma-2017" / "reference_metrics.csv",
    "hurricane-maria-2017": RESULTS / "hurricane-maria-2017" / "reference_metrics.csv",
}

FIGURE_FILES = {
    "cyclone-idai-2019": RESULTS / "figure_agreement.csv",
    "hurricane-irma-2017": RESULTS / "hurricane-irma-2017" / "figure_agreement.csv",
    "hurricane-maria-2017": RESULTS / "hurricane-maria-2017" / "figure_agreement.csv",
    "puebla-mexico-earthquake-2017": RESULTS / "puebla-mexico-earthquake-2017" / "figure_agreement.csv",
    "kerala-floods-2018": RESULTS / "kerala-floods-2018" / "figure_agreement.csv",
}

SOURCE_AVAILABILITY_FILES = {
    "cyclone-idai-2019": RESULTS / "source_availability.csv",
    "hurricane-irma-2017": RESULTS / "hurricane-irma-2017" / "source_availability.csv",
    "hurricane-maria-2017": RESULTS / "hurricane-maria-2017" / "source_availability.csv",
    "puebla-mexico-earthquake-2017": RESULTS / "puebla-mexico-earthquake-2017" / "source_availability.csv",
    "kerala-floods-2018": RESULTS / "kerala-floods-2018" / "source_availability.csv",
}

SYSTEM_LEVEL_JUDGES = {
    "cyclone-idai-2019": "gemini-3.7-flash",
    "hurricane-irma-2017": "gpt-5.6-luna",
    "hurricane-maria-2017": "gpt-5.6-luna",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], columns: Sequence[str]) -> Path:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(columns), extrasaction="ignore", restval="")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _usable(row: dict[str, str]) -> bool:
    return row.get("status") == "ok" and row.get("verdict") in {"absent", "partial", "present"}


def load_judge_rows(path: Path) -> tuple[list[dict[str, str]], int]:
    """Load usable rows, keeping the last row for a duplicate document-slot key."""
    rows = read_csv(path)
    keyed: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in rows:
        keyed[(row["event"], row["sitrep_id"], row["slot_id"])] = row
    deduplicated = len(rows) - len(keyed)
    return [row for row in keyed.values() if _usable(row)], deduplicated


def all_judges() -> dict[tuple[str, str], list[dict[str, str]]]:
    out: dict[tuple[str, str], list[dict[str, str]]] = {}
    for event, judges in JUDGE_FILES.items():
        for judge, path in judges.items():
            out[(event, judge)] = load_judge_rows(path)[0]
    return out


def arm_of(row: dict[str, str]) -> str:
    return "human" if row.get("kind") == "human" else row.get("arm", "machine")


def per_document_scores(rows: Iterable[dict[str, str]]) -> dict[tuple[str, str], float]:
    scores: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        scores[(arm_of(row), row["sitrep_id"])].append(float(row["score"]))
    return {key: fmean(values) for key, values in scores.items()}


def admitted_human_ids(event: str) -> set[str]:
    manifest = DATA / "sitreps_human" / event / "_manifest_ingested.csv"
    existing = {path.name for path in (DATA / "sitreps_human" / event).glob("*.json")
                if not path.name.startswith("_")}
    return {row["id"] for row in read_csv(manifest) if row.get("out_file") in existing}


def usable_machine_ids(event: str) -> set[str]:
    candidates = [DATA / "sitreps_machine" / event, DATA / "sitreps_machine" / event.replace("-", "_")]
    folder = next(path for path in candidates if path.is_dir())
    ids = set()
    for path in folder.glob("*.json"):
        if path.name.startswith("_"):
            continue
        record = read_json(path)
        if str(record.get("text") or "").strip():
            ids.add(str(record.get("id") or path.stem))
    return ids


def corpus_table(judges: dict[tuple[str, str], list[dict[str, str]]]) -> list[dict[str, Any]]:
    raw_manifest = read_csv(ROOT / "data" / "raw" / "reliefweb_manual" / "manifest.csv")
    collected = {event: sum(row.get("event") == event for row in raw_manifest) for event in EVENTS}
    machine_generated = {
        event: len([p for base in (DATA / "sitreps_machine" / event,
                                  DATA / "sitreps_machine" / event.replace("-", "_"))
                    if base.is_dir() for p in base.glob("*.json") if not p.name.startswith("_")])
        for event in EVENTS
    }
    rows = []
    for event in EVENTS:
        human_ids = admitted_human_ids(event)
        machine_ids = usable_machine_ids(event)
        n_pairs = sum(len(event_rows) for (ev, _judge), event_rows in judges.items() if ev == event)
        judge_names = ";".join(JUDGE_FILES[event])
        rows.append({
            "event": event,
            "human_collected": collected[event],
            "human_judged": len(human_ids),
            "human_excluded": collected[event] - len(human_ids),
            "machine_generated": machine_generated[event],
            "machine_judged": len(machine_ids),
            "machine_excluded": machine_generated[event] - len(machine_ids),
            "n_judges": len(JUDGE_FILES[event]),
            "judges": judge_names,
            "usable_judged_pairs": n_pairs,
        })
    return rows


def slot_tables(judges: dict[tuple[str, str], list[dict[str, str]]]) -> tuple[list[dict[str, Any]],
                                                                              list[dict[str, Any]]]:
    coverage_rows: list[dict[str, Any]] = []
    gap_rows: list[dict[str, Any]] = []
    for (event, judge), rows in judges.items():
        expected = {"human": len(admitted_human_ids(event))}
        for machine_id in usable_machine_ids(event):
            row = next(r for r in rows if r["sitrep_id"] == machine_id)
            expected[arm_of(row)] = expected.get(arm_of(row), 0) + 1
        grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
        for row in rows:
            grouped[(row["slot_id"], arm_of(row))].append(float(row["score"]))
        for (slot, arm), scores in sorted(grouped.items()):
            coverage_rows.append({
                "event": event, "judge": judge, "slot_id": slot, "arm": arm,
                "n_expected_documents": expected[arm], "n_scored_documents": len(scores),
                "n_missing": expected[arm] - len(scores), "score_sum": round(sum(scores), 4),
                "coverage": round(fmean(scores), 6),
            })
        slots = sorted({slot for slot, _arm in grouped})
        for slot in slots:
            human = fmean(grouped[(slot, "human")])
            machine = {arm: fmean(values) for (s, arm), values in grouped.items()
                       if s == slot and arm != "human"}
            max_arm, machine_max = max(machine.items(), key=lambda item: item[1])
            gap = human - machine_max
            gap_rows.append({
                "event": event, "judge": judge, "slot_id": slot,
                "human_coverage": round(human, 6), "machine_coverage_max": round(machine_max, 6),
                "machine_max_arm": max_arm, "gap": round(gap, 6),
                "binary_flag": int(human >= 0.8 and machine_max <= 0.6),
            })
    return coverage_rows, gap_rows


def completeness_tables(judges: dict[tuple[str, str], list[dict[str, str]]]) -> tuple[list[dict[str, Any]],
                                                                                      list[dict[str, Any]]]:
    by_document: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    for (event, judge), rows in judges.items():
        grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
        for row in rows:
            grouped[(arm_of(row), row["sitrep_id"])].append(float(row["score"]))
        by_arm: dict[str, list[float]] = defaultdict(list)
        for (arm, document), scores in sorted(grouped.items()):
            score = fmean(scores)
            by_arm[arm].append(score)
            by_document.append({
                "event": event, "judge": judge, "document_id": document, "arm": arm,
                "n_scored_slots": len(scores), "n_missing_slots": SLOTS - len(scores),
                "completeness": round(score, 8),
            })
        for arm, values in sorted(by_arm.items()):
            ci = bootstrap_ci(values, n_boot=1000, seed=SEED)
            summary.append({
                "event": event, "judge": judge, "arm": arm, "n_documents": len(values),
                "completeness": round(fmean(values), 6), "ci_lo": round(ci.lo, 6),
                "ci_hi": round(ci.hi, 6), "n_bootstrap": 1000, "seed": SEED,
            })
    return by_document, summary


def bootstrap_difference(machine: Sequence[float], human: Sequence[float], *, n_boot: int = 10000,
                         seed: int = SEED) -> tuple[float, float, float]:
    rng = random.Random(seed)
    diffs = []
    for _ in range(n_boot):
        m = [machine[rng.randrange(len(machine))] for _ in machine]
        h = [human[rng.randrange(len(human))] for _ in human]
        diffs.append(fmean(m) - fmean(h))
    diffs.sort()
    lo = diffs[math.floor(0.025 * (n_boot - 1))]
    hi = diffs[math.ceil(0.975 * (n_boot - 1))]
    return fmean(machine) - fmean(human), lo, hi


def permutation_p(machine: Sequence[float], human: Sequence[float], *, n_perm: int = 100000,
                  seed: int = SEED) -> float:
    observed = abs(fmean(machine) - fmean(human))
    pooled = list(machine) + list(human)
    n_machine = len(machine)
    rng = random.Random(seed)
    extreme = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        diff = abs(fmean(pooled[:n_machine]) - fmean(pooled[n_machine:]))
        extreme += diff >= observed - 1e-15
    return (extreme + 1) / (n_perm + 1)


def inversion_table(by_document: list[dict[str, Any]]) -> list[dict[str, Any]]:
    subset = [row for row in by_document if row["event"] == "cyclone-idai-2019"
              and row["judge"] == "gemini-3.7-flash"]
    machine = [float(row["completeness"]) for row in subset
               if row["arm"] == "api-strong/schema_guided"]
    human = [float(row["completeness"]) for row in subset if row["arm"] == "human"]
    diff, lo, hi = bootstrap_difference(machine, human)
    return [{
        "event": "cyclone-idai-2019", "judge": "gemini-3.7-flash",
        "machine_arm": "api-strong/schema_guided", "n_machine": len(machine),
        "n_human": len(human), "machine_mean": round(fmean(machine), 6),
        "human_mean": round(fmean(human), 6), "difference": round(diff, 6),
        "difference_ci_lo": round(lo, 6), "difference_ci_hi": round(hi, 6),
        "permutation_p_two_sided": round(permutation_p(machine, human), 6),
        "n_bootstrap": 10000, "n_permutations": 100000, "seed": SEED,
    }]


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    adjusted = [0.0] * len(p_values)
    running = 0.0
    m = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def correlations_holm() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event, path in CORRELATION_FILES.items():
        for row in read_csv(path):
            if row["scope"] == "pooled" and row["coefficient"] == "spearman":
                rows.append({"event": event, "metric": row["metric"], "rho": float(row["rho"]),
                             "p_raw": float(row["p_value"]), "n": int(row["n"])})
    adjusted = holm_adjust([row["p_raw"] for row in rows])
    for row, p_adj in zip(rows, adjusted):
        row["p_holm"] = round(p_adj, 8)
        row["significant_holm_05"] = int(p_adj < 0.05)
    return rows


def system_level_tables(by_document: list[dict[str, Any]]) -> tuple[list[dict[str, Any]],
                                                                     list[dict[str, Any]]]:
    """Aggregate report scores to the four generator/prompt arms for each primary event."""
    from scipy.stats import spearmanr

    arm_rows: list[dict[str, Any]] = []
    correlation_rows: list[dict[str, Any]] = []
    metrics = ("rouge_l_f", "bertscore_f1", "llm_judge_score")
    for event in PRIMARY_EVENTS:
        judge = SYSTEM_LEVEL_JUDGES[event]
        completeness = {
            row["document_id"]: float(row["completeness"])
            for row in by_document
            if row["event"] == event and row["judge"] == judge and row["arm"] != "human"
        }
        grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for row in read_csv(REFERENCE_FILES[event]):
            document = row["sitrep_id"]
            if document not in completeness:
                continue
            arm = f"{row['model']}/{row['arm']}"
            grouped[arm]["completeness"].append(completeness[document])
            for metric in metrics:
                if row.get(metric, "") != "":
                    grouped[arm][metric].append(float(row[metric]))
        ordered = sorted(grouped, key=lambda arm: fmean(grouped[arm]["completeness"]), reverse=True)
        for rank, arm in enumerate(ordered, start=1):
            values = grouped[arm]
            arm_rows.append({
                "event": event, "completeness_judge": judge, "arm": arm,
                "n_documents": len(values["completeness"]), "completeness_rank": rank,
                "completeness": round(fmean(values["completeness"]), 6),
                **{metric: round(fmean(values[metric]), 6) for metric in metrics},
                "generic_judge_model": "google/gemini-3.7-flash",
            })
        event_rows = [row for row in arm_rows if row["event"] == event]
        for metric in metrics:
            result = spearmanr([row["completeness"] for row in event_rows],
                               [row[metric] for row in event_rows])
            correlation_rows.append({
                "event": event, "completeness_judge": judge, "metric": metric,
                "unit": "generator-prompt-arm", "n_arms": len(event_rows),
                "rho": round(float(result.statistic), 6),
                "p_two_sided_asymptotic": round(float(result.pvalue), 6),
                "inference_note": "descriptive-small-n",
            })
    return arm_rows, correlation_rows


def figure_grounding_summary() -> list[dict[str, Any]]:
    """Summarise per-pair grounding rates with seeded percentile bootstrap intervals."""
    rows: list[dict[str, Any]] = []
    fields = (
        ("figure_precision", "precision", FIGURE_FILES),
        ("figure_recall", "recall", FIGURE_FILES),
        ("casualty_recall", "casualty_recall", FIGURE_FILES),
        ("official_in_stream", "human_figs_in_stream", SOURCE_AVAILABILITY_FILES),
        ("machine_in_stream", "machine_figs_in_stream", SOURCE_AVAILABILITY_FILES),
    )
    for event in EVENTS:
        file_rows = {path: read_csv(path) for path in {files[event] for _, _, files in fields}}
        row: dict[str, Any] = {
            "event": event,
            "n_same_day_pairs": len(file_rows[FIGURE_FILES[event]]),
            "bootstrap_unit": "metric-specific; see per-metric bootstrap-unit columns",
            "ci_type": "percentile-95",
            "n_bootstrap": 10000,
            "seed": SEED,
        }
        for output_name, input_name, files in fields:
            source_rows = file_rows[files[event]]
            if output_name == "official_in_stream":
                by_day: dict[str, float] = {}
                for item in source_rows:
                    if item.get(input_name, "") == "":
                        continue
                    value = float(item[input_name])
                    day = item["day"]
                    if day in by_day and by_day[day] != value:
                        raise RuntimeError(f"official-in-stream value varies within {event} day {day}")
                    by_day[day] = value
                values = [by_day[day] for day in sorted(by_day)]
                bootstrap_unit = "event-day"
            else:
                values = [float(item[input_name]) for item in source_rows
                          if item.get(input_name, "") != ""]
                bootstrap_unit = "same-day-machine-official-pair"
            ci = bootstrap_ci(values, n_boot=10000, seed=SEED)
            row[output_name] = round(ci.point, 6)
            row[f"{output_name}_ci_lo"] = round(ci.lo, 6)
            row[f"{output_name}_ci_hi"] = round(ci.hi, 6)
            row[f"{output_name}_n"] = len(values)
            row[f"{output_name}_bootstrap_unit"] = bootstrap_unit
        rows.append(row)
    return rows


def agreement_table(judges: dict[tuple[str, str], list[dict[str, str]]]) -> list[dict[str, Any]]:
    out = []
    for event in PRIMARY_EVENTS:
        indexes = {}
        for judge in ("gpt-5.6-luna", "gemini-3.7-flash"):
            indexes[judge] = {(r["event"], r["sitrep_id"], r["slot_id"]): r["verdict"]
                              for r in judges[(event, judge)]}
        common = sorted(set.intersection(*(set(index) for index in indexes.values())))
        result = fleiss_kappa([[indexes[judge][key] for judge in indexes] for key in common])
        out.append({
            "event": event, "judge_1": "gpt-5.6-luna", "judge_2": "gemini-3.7-flash",
            "item_unit": "document-slot", "category_treatment": "nominal-unweighted",
            "missing_rule": "complete-case", "n_items": result["n"], "raters_per_item": 2,
            "observed_agreement": round(result["observed"], 6),
            "chance_agreement": round(result["chance"], 6), "fleiss_kappa": round(result["kappa"], 6),
        })
    return out


def _private_human_validation_pairs() -> list[dict[str, Any]]:
    """Join the private blind scoring sheet to its withheld key."""
    sample = read_csv(VALIDATION_SAMPLE)
    key_rows = read_csv(VALIDATION_KEY)
    if len(sample) != 100 or len(key_rows) != 100:
        raise RuntimeError("human validation requires exactly 100 sample and 100 key rows")

    def item_key(row: dict[str, str]) -> tuple[str, str]:
        return row["sitrep_id"].strip(), row["slot_id"].strip()

    sample_index = {item_key(row): row for row in sample}
    key_index = {item_key(row): row for row in key_rows}
    if len(sample_index) != len(sample) or len(key_index) != len(key_rows):
        raise RuntimeError("human validation contains duplicate document-slot items")
    if set(sample_index) != set(key_index):
        raise RuntimeError("human validation sample and withheld judge key do not match")
    invalid_human = {row.get("human_verdict", "").strip() for row in sample} - VALIDATION_LABELS
    if invalid_human:
        raise RuntimeError(f"human validation contains blank or invalid verdicts: {invalid_human}")

    pairs: list[dict[str, Any]] = []
    for sample_row, human in enumerate(sample, start=1):
        key = key_index[item_key(human)]
        pair: dict[str, Any] = {
            "sample_row": sample_row,
            "event": "cyclone-idai-2019",
            "sitrep_id": human["sitrep_id"].strip(),
            "kind": human["kind"].strip(),
            "slot_id": human["slot_id"].strip(),
            "human_verdict": human["human_verdict"].strip(),
            "judges_unanimous": key["judges_unanimous"].strip(),
        }
        for _judge, prefix in VALIDATION_JUDGES:
            pair[f"{prefix}_verdict"] = key[f"{prefix}_verdict"].strip()
            pair[f"{prefix}_confidence"] = key[f"{prefix}_confidence"].strip()
        pairs.append(pair)
    return pairs


def human_validation_tables() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Recompute human agreement from private inputs or the anonymous released pair table.

    The released item table excludes report paths, definitions, free-text notes, and evidence. It
    retains only identifiers and categorical/numeric annotations needed to recompute agreement.
    """
    if VALIDATION_SAMPLE.is_file() and VALIDATION_KEY.is_file():
        pairs = _private_human_validation_pairs()
    else:
        released = ROOT / "supplementary" / "reproducibility" / "tables" / \
            "human_validation_pairs.csv"
        if not released.is_file():
            raise RuntimeError("human validation inputs and anonymous released pairs are both missing")
        pairs = read_csv(released)

    if len(pairs) != 100:
        raise RuntimeError("human validation requires exactly 100 released pairs")
    required = {"sitrep_id", "kind", "slot_id", "human_verdict"} | {
        f"{prefix}_{suffix}"
        for _judge, prefix in VALIDATION_JUDGES
        for suffix in ("verdict", "confidence")
    }
    missing_columns = required - set(pairs[0])
    if missing_columns:
        raise RuntimeError(f"human validation pairs lack columns: {sorted(missing_columns)}")
    if len({(row["sitrep_id"], row["slot_id"]) for row in pairs}) != len(pairs):
        raise RuntimeError("human validation pairs contain duplicate document-slot items")
    invalid_human = {row["human_verdict"].strip() for row in pairs} - VALIDATION_LABELS
    if invalid_human:
        raise RuntimeError(f"human validation pairs contain invalid verdicts: {invalid_human}")

    summary: list[dict[str, Any]] = []
    for judge, prefix in VALIDATION_JUDGES:
        for scope in ("all", "human", "machine"):
            expected = pairs if scope == "all" else [row for row in pairs if row["kind"] == scope]
            scored = [row for row in expected if row[f"{prefix}_verdict"] in VALIDATION_LABELS]
            agreement = judge_agreement(
                [row[f"{prefix}_verdict"] for row in scored],
                [row["human_verdict"] for row in scored],
            )
            class_metrics: dict[str, Any] = {}
            for i, label in enumerate(agreement.labels):
                true_positive = agreement.confusion[i][i]
                predicted = sum(agreement.confusion[i])
                support = sum(agreement.confusion[j][i] for j in range(len(agreement.labels)))
                precision = true_positive / predicted if predicted else float("nan")
                recall = true_positive / support if support else float("nan")
                f1 = 2 * precision * recall / (precision + recall) if precision + recall else float("nan")
                class_metrics.update({
                    f"support_human_{label}": support,
                    f"precision_{label}": round(precision, 6),
                    f"recall_{label}": round(recall, 6),
                    f"f1_{label}": round(f1, 6),
                })
            summary.append({
                "event": "cyclone-idai-2019",
                "judge": judge,
                "scope": scope,
                "item_unit": "document-slot",
                "category_treatment": "nominal-unweighted",
                "missing_rule": "complete-case-per-judge",
                "n_expected": len(expected),
                "n_scored": agreement.n,
                "n_missing": len(expected) - agreement.n,
                "raw_agreement": round(agreement.agreement, 6),
                "cohens_kappa": round(agreement.kappa, 6),
                "cohens_kappa_linear": round(agreement.kappa_linear, 6),
                "passes_preregistered_raw_agreement_threshold": int(agreement.agreement >= 0.7),
                "kappa_at_least_0_7_descriptive": int(agreement.kappa >= 0.7),
                **class_metrics,
                **{
                    f"n_judge_{judge_label}_human_{human_label}": agreement.confusion[i][j]
                    for i, judge_label in enumerate(agreement.labels)
                    for j, human_label in enumerate(agreement.labels)
                },
            })
    return pairs, summary


def _normalise_heading(value: str) -> str:
    value = re.sub(r"^\s*#{1,6}\s*", "", value.strip())
    value = re.sub(r"^\s*\*\*(.*?)\*\*\s*$", r"\1", value)
    value = re.sub(r"^\s*\d+[.)]\s*", "", value)
    value = re.sub(r"\s*\[[^\]]+\]\s*$", "", value)
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def content_sections(text: str, aliases: dict[str, set[str]]) -> dict[str, str]:
    headings: list[tuple[int, str, str]] = []
    lines = text.splitlines()
    for index, line in enumerate(lines):
        candidate, inline = line, ""
        bracketed = re.match(r"^\s*#{1,6}\s*\[([^\]]+)\]\s*:?[ \t]*(.*)$", line)
        bold = re.match(r"^\s*\*\*(.+?)\*\*\s*:?[ \t]*(.*)$", line)
        if bracketed:
            candidate, inline = bracketed.group(1), bracketed.group(2)
        elif bold:
            candidate, inline = bold.group(1), bold.group(2)
        normalised = _normalise_heading(candidate)
        slot = next((slot_id for slot_id, names in aliases.items() if normalised in names), None)
        if slot:
            headings.append((index, slot, inline))
    sections = {}
    for pos, (start, slot, inline) in enumerate(headings):
        end = headings[pos + 1][0] if pos + 1 < len(headings) else len(lines)
        sections[slot] = "\n".join(([inline] if inline else []) + lines[start + 1:end]).strip()
    return sections


NEGATIVE_ONLY = re.compile(
    r"^(?:no (?:information|data|details|figures|specific|relevant)|not (?:reported|mentioned|available)|"
    r"none (?:reported|mentioned)|the (?:posts|source material) (?:do|does) not)", re.I)


def gated_present_score(body: str) -> float:
    """Maximum score allowed by content beneath a schema-guided heading."""
    lines = [re.sub(r"^[\s>*-]+", "", line).strip() for line in body.splitlines() if line.strip()]
    support = [line for line in lines if not NEGATIVE_ONLY.match(line)]
    if not support:
        return 0.0
    tokens = re.findall(r"[A-Za-z][A-Za-z'-]*", " ".join(support))
    return 1.0 if len(tokens) >= 8 else 0.5


def machine_reports() -> dict[tuple[str, str], str]:
    reports: dict[tuple[str, str], str] = {}
    for event in EVENTS:
        candidates = [DATA / "sitreps_machine" / event, DATA / "sitreps_machine" / event.replace("-", "_")]
        folder = next(path for path in candidates if path.is_dir())
        for path in folder.glob("*.json"):
            if path.name.startswith("_"):
                continue
            record = read_json(path)
            text = str(record.get("text") or "")
            if text.strip():
                reports[(event, str(record.get("id") or path.stem))] = text
    return reports


def content_gate_tables(judges: dict[tuple[str, str], list[dict[str, str]]]) -> tuple[list[dict[str, Any]],
                                                                                     list[dict[str, Any]]]:
    slots = schema_slots()
    aliases = {slot.id: {_normalise_heading(slot.id), _normalise_heading(slot.name)} for slot in slots}
    reports = machine_reports()
    detail: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    for (event, judge), rows in judges.items():
        schema_rows = [row for row in rows if row.get("kind") == "machine"
                       and row.get("arm", "").endswith("/schema_guided")]
        by_doc: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
        for row in schema_rows:
            by_doc[(row["arm"], row["sitrep_id"])].append(row)
        by_arm_original: dict[str, list[float]] = defaultdict(list)
        by_arm_gated: dict[str, list[float]] = defaultdict(list)
        downgraded: dict[str, int] = defaultdict(int)
        for (arm, document), document_rows in by_doc.items():
            sections = content_sections(reports[(event, document)], aliases)
            original_scores = []
            gated_scores = []
            for row in document_rows:
                original = float(row["score"])
                maximum = gated_present_score(sections.get(row["slot_id"], ""))
                gated = min(original, maximum) if original == 1.0 else original
                downgraded[arm] += gated < original
                original_scores.append(original)
                gated_scores.append(gated)
            original_mean, gated_mean = fmean(original_scores), fmean(gated_scores)
            by_arm_original[arm].append(original_mean)
            by_arm_gated[arm].append(gated_mean)
            detail.append({
                "event": event, "judge": judge, "document_id": document, "arm": arm,
                "n_scored_slots": len(original_scores), "original_completeness": round(original_mean, 6),
                "content_gated_completeness": round(gated_mean, 6),
            })
        for arm in sorted(by_arm_original):
            summary.append({
                "event": event, "judge": judge, "arm": arm, "n_documents": len(by_arm_original[arm]),
                "original_completeness": round(fmean(by_arm_original[arm]), 6),
                "content_gated_completeness": round(fmean(by_arm_gated[arm]), 6),
                "present_cells_downgraded": downgraded[arm],
            })
    return detail, summary


def write_main_tables() -> None:
    judges = all_judges()
    corpus = corpus_table(judges)
    coverage, gaps = slot_tables(judges)
    documents, completeness = completeness_tables(judges)
    inversion = inversion_table(documents)
    correlations = correlations_holm()
    system_arms, system_correlations = system_level_tables(documents)
    grounding = figure_grounding_summary()
    agreement = agreement_table(judges)
    validation_pairs, validation_summary = human_validation_tables()
    gated_detail, gated_summary = content_gate_tables(judges)

    write_csv(PAPER_TABLES / "corpus_manifest.csv", corpus, tuple(corpus[0]))
    write_csv(PAPER_TABLES / "slot_coverage_by_judge_arm.csv", coverage, tuple(coverage[0]))
    write_csv(PAPER_TABLES / "structural_gaps.csv", gaps, tuple(gaps[0]))
    write_csv(PAPER_TABLES / "completeness_by_document.csv", documents, tuple(documents[0]))
    write_csv(PAPER_TABLES / "completeness_by_judge_arm.csv", completeness, tuple(completeness[0]))
    write_csv(PAPER_TABLES / "idai_inversion_test.csv", inversion, tuple(inversion[0]))
    write_csv(PAPER_TABLES / "correlations_holm.csv", correlations, tuple(correlations[0]))
    write_csv(PAPER_TABLES / "system_level_arm_scores.csv", system_arms, tuple(system_arms[0]))
    write_csv(PAPER_TABLES / "system_level_correlations.csv", system_correlations,
              tuple(system_correlations[0]))
    write_csv(PAPER_TABLES / "figure_grounding_summary.csv", grounding, tuple(grounding[0]))
    write_csv(PAPER_TABLES / "judge_agreement.csv", agreement, tuple(agreement[0]))
    write_csv(PAPER_TABLES / "human_validation_pairs.csv", validation_pairs,
              tuple(validation_pairs[0]))
    write_csv(PAPER_TABLES / "human_validation_summary.csv", validation_summary,
              tuple(validation_summary[0]))
    write_csv(PAPER_TABLES / "content_gate_by_document.csv", gated_detail, tuple(gated_detail[0]))
    write_csv(PAPER_TABLES / "content_gate_summary.csv", gated_summary, tuple(gated_summary[0]))

    currency = [row for row in gaps if row["slot_id"] == "currency_and_metadata"]
    excluding_qwen = [row for row in currency if row["judge"] != "qwen3.8-flash"]
    sensitivity = [
        {"analysis": "all-observed", "positive_cells": sum(row["gap"] > 0 for row in currency),
         "total_cells": len(currency), "excluded_judge": ""},
        {"analysis": "exclude-high-missingness-judge",
         "positive_cells": sum(row["gap"] > 0 for row in excluding_qwen),
         "total_cells": len(excluding_qwen), "excluded_judge": "qwen3.8-flash"},
    ]
    write_csv(PAPER_TABLES / "structural_gap_sensitivity.csv", sensitivity, tuple(sensitivity[0]))

    truncation_path = PAPER_TABLES / "truncation_sensitivity.csv"
    if truncation_path.is_file():
        truncation = read_csv(truncation_path)
        summary_rows = []
        for scope, subset in [("all", truncation)] + [
                (event, [row for row in truncation if row["event"] == event]) for event in PRIMARY_EVENTS]:
            shifts = [float(row["shift"]) for row in subset]
            ci = bootstrap_ci(shifts, n_boot=10000, seed=SEED)
            summary_rows.append({"scope": scope, "n_documents": len(shifts),
                                 "mean_ceiling_shift": round(fmean(shifts), 6),
                                 "ci_lo": round(ci.lo, 6), "ci_hi": round(ci.hi, 6),
                                 "n_bootstrap": 10000, "seed": SEED})
        write_csv(PAPER_TABLES / "truncation_sensitivity_summary.csv", summary_rows,
                  tuple(summary_rows[0]))

    gemini_dir = ensure_dir(PAPER_TABLES / "idai" / "gemini")
    fig_coverage = [row for row in coverage if row["event"] == "cyclone-idai-2019"
                    and row["judge"] == "gemini-3.7-flash"]
    write_csv(gemini_dir / "slot_coverage.csv", [
        {"slot_id": row["slot_id"], "arm": row["arm"], "coverage": row["coverage"]}
        for row in fig_coverage], ("slot_id", "arm", "coverage"))
    fig_completeness = [row for row in completeness if row["event"] == "cyclone-idai-2019"
                        and row["judge"] == "gemini-3.7-flash"]
    write_csv(gemini_dir / "completeness.csv", fig_completeness,
              ("arm", "n_documents", "completeness", "ci_lo", "ci_hi", "n_bootstrap", "seed"))

    print(f"corpus usable judged pairs: {sum(row['usable_judged_pairs'] for row in corpus)}")
    print(f"currency gap positive: {sum(row['gap'] > 0 for row in currency)}/{len(currency)}")
    print("excluding high-missingness qwen: "
          f"{sum(row['gap'] > 0 for row in excluding_qwen)}/{len(excluding_qwen)}")
    print(f"Idai Gemini inversion: {inversion[0]}")


def _manifest_sitreps(event: str) -> list[Any]:
    from src.judge_slots import Sitrep

    folder = DATA / "sitreps_human" / event
    out = []
    for row in read_csv(folder / "_manifest_ingested.csv"):
        path = folder / row["out_file"]
        if not path.is_file():
            continue
        record = read_json(path)
        text = str(record.get("text") or "").strip()
        if not text:
            continue
        out.append(Sitrep(id=str(record.get("id") or path.stem), kind="human", event=event, text=text,
                          date=str(record.get("date") or ""), title=str(record.get("title") or ""),
                          source=str(record.get("source") or ""), path=path.as_posix()))
    return out


def rejudge_untruncated(sample_per_event: int = 4) -> None:
    """Re-judge a seeded random sample of reports affected by the 20k-character cap."""
    from src.judge_slots import judge_pairs, load_schema, write_judgements, write_tables
    from src.llm import LLM
    from src.util import load_config

    rng = random.Random(SEED)
    selected: list[Sitrep] = []
    for event in PRIMARY_EVENTS:
        candidates = sorted((s for s in _manifest_sitreps(event) if len(s.text) > 20_000), key=lambda s: s.id)
        selected.extend(rng.sample(candidates, min(sample_per_event, len(candidates))))
    slots = load_schema(DATA / "schema.yaml")
    pairs = [(sitrep, slot) for sitrep in selected for slot in slots]
    judgements = judge_pairs(LLM(load_config()), pairs, slots, batch=False, model="judge_gemini",
                             max_chars=max(len(s.text) for s in selected) + 1, max_tokens=1200, gate=True)
    raw_dir = DATA / "judgements_untruncated"
    for event in PRIMARY_EVENTS:
        event_sitreps = [s for s in selected if s.event == event]
        event_judgements = [j for j in judgements if j.event == event]
        write_judgements(event_judgements, raw_dir, event, event_sitreps)
    table_dir = ensure_dir(PAPER_TABLES / "truncation_untruncated")
    write_tables(judgements, table_dir, merge=False)

    original = all_judges()
    original_index = {}
    for event in PRIMARY_EVENTS:
        for (arm, document), score in per_document_scores(original[(event, "gemini-3.7-flash")]).items():
            if arm == "human":
                original_index[(event, document)] = score
    full_index: dict[tuple[str, str], list[float]] = defaultdict(list)
    for judgement in judgements:
        if judgement.status == "ok" and judgement.score is not None:
            full_index[(judgement.event, judgement.sitrep_id)].append(float(judgement.score))
    rows = []
    for sitrep in sorted(selected, key=lambda s: (s.event, s.id)):
        full = fmean(full_index[(sitrep.event, sitrep.id)])
        truncated = original_index[(sitrep.event, sitrep.id)]
        rows.append({"event": sitrep.event, "document_id": sitrep.id, "chars": len(sitrep.text),
                     "truncated_completeness": round(truncated, 6),
                     "untruncated_completeness": round(full, 6), "shift": round(full - truncated, 6)})
    write_csv(PAPER_TABLES / "truncation_sensitivity.csv", rows, tuple(rows[0]))
    print(f"untruncated sample n={len(rows)}; mean ceiling shift={fmean(row['shift'] for row in rows):+.4f}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rejudge-untruncated", action="store_true")
    parser.add_argument("--sample-per-event", type=int, default=4)
    args = parser.parse_args(argv)
    write_main_tables()
    if args.rejudge_untruncated:
        rejudge_untruncated(args.sample_per_event)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
