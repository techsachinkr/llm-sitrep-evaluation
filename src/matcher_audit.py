"""Build a deterministic private worklist for manual auditing of figure matching.

The worklist contains report and social-media excerpts and must remain private. The public audit
table contains only sample identifiers, numeric keys, excerpt hashes, and manual labels.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from scipy.stats import beta

from src.figure_check import load_human_by_day
from src.metrics import Figure, extract_figures
from src.paper_analysis import PRIMARY_EVENTS, SEED
from src.util import ensure_dir, resolve_event_dir

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
DEFAULT_WORKLIST = PROCESSED / "matcher_audit_private.json"
DEFAULT_LABELS = PROCESSED / "matcher_audit_labels.json"
PUBLIC_AUDIT = ROOT / "supplementary" / "reproducibility" / "matcher_audit.csv"
PUBLIC_SUMMARY = ROOT / "supplementary" / "reproducibility" / "matcher_audit_summary.csv"
CONTEXT_CHARS = 180
SAMPLE_PER_STRATUM = 50
_TOKENS = re.compile(r"[a-z][a-z0-9'-]{2,}", re.I)
_STOP = {
    "and", "are", "but", "for", "from", "has", "have", "into", "its", "not", "of", "on",
    "or", "that", "the", "their", "there", "this", "to", "was", "were", "which", "with",
}


@dataclass(frozen=True)
class Occurrence:
    figure: Figure
    context: str
    record_id: str


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _occurrences(text: str, record_id: str) -> list[Occurrence]:
    """Attach privacy-sensitive context to figures while preserving extractor order."""
    occurrences: list[Occurrence] = []
    for figure in extract_figures(text):
        if figure.start < 0 or figure.end < 0:
            continue
        context = re.sub(r"\s+", " ", text[max(0, figure.start - CONTEXT_CHARS):
                                                min(len(text), figure.end + CONTEXT_CHARS)]).strip()
        occurrences.append(Occurrence(figure, context, record_id))
    return occurrences


def _terms(text: str) -> set[str]:
    return {token.lower() for token in _TOKENS.findall(text) if token.lower() not in _STOP}


def _similarity(left: str, right: str) -> float:
    a, b = _terms(left), _terms(right)
    return len(a & b) / len(a | b) if a and b else 0.0


def _deduplicate(occurrences: Iterable[Occurrence]) -> dict[tuple[str, float], list[Occurrence]]:
    grouped: dict[tuple[str, float], list[Occurrence]] = defaultdict(list)
    for occurrence in occurrences:
        grouped[occurrence.figure.key].append(occurrence)
    return grouped


def audit_population() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Return absent official, matched official, and matched machine figure candidates."""
    absent: list[dict[str, Any]] = []
    official_matched: list[dict[str, Any]] = []
    machine_matched: list[dict[str, Any]] = []
    human_root = PROCESSED / "sitreps_human"
    machine_root = PROCESSED / "sitreps_machine"
    stream_root = PROCESSED / "streams"
    for event in PRIMARY_EVENTS:
        human_dir = resolve_event_dir(human_root, event)
        machine_dir = resolve_event_dir(machine_root, event)
        stream_dir = resolve_event_dir(stream_root, event)
        if human_dir is None or machine_dir is None or stream_dir is None:
            raise FileNotFoundError(f"missing human, machine, or stream directory for {event}")
        human_by_day = load_human_by_day(human_dir)
        machine_by_day: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for machine_path in sorted(machine_dir.glob("*.json")):
            record = json.loads(machine_path.read_text(encoding="utf-8"))
            text = str(record.get("text") or "").strip()
            day = str(record.get("day") or record.get("date") or "")
            if text and day:
                machine_by_day[day].append((str(record.get("id") or machine_path.stem), text))
        for stream_path in sorted(stream_dir.glob("*.jsonl")):
            day = stream_path.stem
            if day not in human_by_day:
                continue
            human_text = "\n".join(human_by_day[day])
            human = _deduplicate(_occurrences(human_text, f"{event}/{day}/official"))
            stream_occurrences: list[Occurrence] = []
            stream_records: list[tuple[str, str]] = []
            with stream_path.open(encoding="utf-8") as stream:
                for position, line in enumerate(stream, start=1):
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    text = str(record.get("text") or "")
                    record_id = str(record.get("tweet_id") or f"row-{position}")
                    stream_records.append((record_id, text))
                    stream_occurrences.extend(_occurrences(text, record_id))
            stream = _deduplicate(stream_occurrences)
            for key, official_occurrences in human.items():
                official = official_occurrences[0]
                base = {
                    "event": event, "day": day, "unit": key[0], "value": key[1],
                    "official_raw": official.figure.raw, "official_context": official.context,
                    "official_context_sha256": _sha256(official.context),
                }
                if key in stream:
                    ranked = sorted(stream[key], key=lambda item: (
                        -_similarity(official.context, item.context), item.record_id, item.context
                    ))
                    candidates = ranked[:3]
                    official_matched.append({
                        **base,
                        "stream_candidates": [
                            {"record_id": item.record_id, "raw": item.figure.raw,
                             "context": item.context, "context_sha256": _sha256(item.context)}
                            for item in candidates
                        ],
                    })
                else:
                    ranked_records = sorted(stream_records, key=lambda item: (
                        -_similarity(official.context, item[1]), item[0]
                    ))[:3]
                    absent.append({
                        **base,
                        "stream_candidates": [
                            {"record_id": record_id,
                             "context": re.sub(r"\s+", " ", text).strip(),
                             "context_sha256": _sha256(re.sub(r"\s+", " ", text).strip())}
                            for record_id, text in ranked_records
                        ],
                    })
            for document_id, machine_text in machine_by_day.get(day, []):
                machine = _deduplicate(_occurrences(machine_text, document_id))
                for key in sorted(machine.keys() & stream.keys()):
                    generated = machine[key][0]
                    ranked = sorted(stream[key], key=lambda item: (
                        -_similarity(generated.context, item.context), item.record_id, item.context
                    ))
                    machine_matched.append({
                        "event": event, "day": day, "unit": key[0], "value": key[1],
                        "machine_document_id": document_id, "official_raw": generated.figure.raw,
                        "official_context": generated.context,
                        "official_context_sha256": _sha256(generated.context),
                        "stream_candidates": [
                            {"record_id": item.record_id, "raw": item.figure.raw,
                             "context": item.context, "context_sha256": _sha256(item.context)}
                            for item in ranked[:3]
                        ],
                    })
    return absent, official_matched, machine_matched


def sampled_worklist(*, seed: int = SEED, n_per_stratum: int = SAMPLE_PER_STRATUM) -> dict[str, Any]:
    absent, official_matched, machine_matched = audit_population()
    rng = random.Random(seed)
    selections = {
        "stream_absent": rng.sample(sorted(absent, key=lambda row: (
            row["event"], row["day"], row["unit"], row["value"])), n_per_stratum),
        "official_exact_match": rng.sample(sorted(official_matched, key=lambda row: (
            row["event"], row["day"], row["unit"], row["value"])), n_per_stratum // 2),
        "machine_exact_match": rng.sample(sorted(machine_matched, key=lambda row: (
            row["event"], row["day"], row["machine_document_id"], row["unit"], row["value"])),
            n_per_stratum - n_per_stratum // 2),
    }
    for stratum, rows in selections.items():
        prefix = {"stream_absent": "A", "official_exact_match": "O",
                  "machine_exact_match": "G"}[stratum]
        for index, row in enumerate(rows, start=1):
            row["sample_id"] = f"{prefix}{index:02d}"
            row["stratum"] = stratum
    return {
        "seed": seed,
        "sample_per_stratum": n_per_stratum,
        "population_stream_absent": len(absent),
        "population_official_exact_match": len(official_matched),
        "population_machine_exact_match": len(machine_matched),
        "instructions": {
            "stream_absent": "Confirm the official token is a substantive figure and no candidate supplies the same fact in an equivalent form.",
            "official_exact_match": "Confirm at least one same-number stream occurrence denotes the same fact as the official figure.",
            "machine_exact_match": "Confirm at least one same-number stream occurrence denotes the same fact as the machine figure.",
        },
        "items": (selections["stream_absent"] + selections["official_exact_match"] +
                  selections["machine_exact_match"]),
    }


def _exact_binomial_ci(errors: int, total: int, alpha: float = 0.05) -> tuple[float, float]:
    lo = 0.0 if errors == 0 else float(beta.ppf(alpha / 2, errors, total - errors + 1))
    hi = 1.0 if errors == total else float(beta.ppf(1 - alpha / 2, errors + 1, total - errors))
    return lo, hi


def resolved_labels(spec: dict[str, Any], items: list[dict[str, Any]]) -> dict[str, str]:
    labels: dict[str, str] = {}
    for item in items:
        rules = spec["strata"][item["stratum"]]
        labels[item["sample_id"]] = rules.get("exceptions", {}).get(
            item["sample_id"], rules["default_label"]
        )
    return labels


def write_public_audit(worklist: dict[str, Any], label_spec: dict[str, Any]) -> None:
    """Write text-free item labels and exact-binomial error summaries."""
    items = worklist["items"]
    labels = resolved_labels(label_spec, items)
    rows = []
    for item in items:
        rows.append({
            "sample_id": item["sample_id"], "stratum": item["stratum"],
            "event": item["event"], "day": item["day"], "unit": item["unit"],
            "normalized_value": item["value"],
            "subject_context_sha256": item["official_context_sha256"],
            "candidate_context_sha256s": ";".join(
                candidate["context_sha256"] for candidate in item["stream_candidates"]
            ),
            "manual_label": labels[item["sample_id"]],
        })
    ensure_dir(PUBLIC_AUDIT.parent)
    with PUBLIC_AUDIT.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    error_labels = {
        "stream_absent": {"missed_rounded_equivalent", "ocr_tokenization_error",
                          "non_substantive_extraction", "contact_number_extraction"},
        "official_exact_match": {"coincidental_match"},
        "machine_exact_match": {"coincidental_match"},
    }
    summary = []
    populations = {
        "stream_absent": worklist["population_stream_absent"],
        "official_exact_match": worklist["population_official_exact_match"],
        "machine_exact_match": worklist["population_machine_exact_match"],
    }
    for stratum in ("stream_absent", "official_exact_match", "machine_exact_match"):
        sample = [row for row in rows if row["stratum"] == stratum]
        errors = sum(row["manual_label"] in error_labels[stratum] for row in sample)
        lo, hi = _exact_binomial_ci(errors, len(sample))
        summary.append({
            "stratum": stratum, "population_items": populations[stratum],
            "sample_items": len(sample), "sampling_seed": worklist["seed"],
            "manual_errors": errors, "error_rate": round(errors / len(sample), 6),
            "ci_method": "Clopper-Pearson-exact-95", "error_rate_ci_lo": round(lo, 6),
            "error_rate_ci_hi": round(hi, 6), "reviewer": label_spec["reviewer"],
            "review_date": label_spec["review_date"],
        })
    with PUBLIC_SUMMARY.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create the private figure-matcher audit worklist")
    parser.add_argument("--output", type=Path, default=DEFAULT_WORKLIST)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    args = parser.parse_args(argv)
    worklist = sampled_worklist()
    ensure_dir(args.output.parent)
    args.output.write_text(json.dumps(worklist, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.labels.is_file():
        label_spec = json.loads(args.labels.read_text(encoding="utf-8"))
        write_public_audit(worklist, label_spec)
        print(f"public audit: {PUBLIC_AUDIT}")
    print(f"private audit worklist: {args.output} ({len(worklist['items'])} items)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
