"""Build the anonymized, text-free supplementary reproducibility artifact.

The export contains instructions, public document metadata, hashes, and numeric judgements. It
never copies ReliefWeb report text, social-media text, model evidence spans, or author identity.
"""
from __future__ import annotations

import csv
import hashlib
import json
import platform
import shutil
import sys
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote_plus

import yaml

from src import consolidate, metrics, open_code, reference_metrics
from src.consolidate import Cluster, OpenCode, naming_requests
from src.generate_sitreps import (
    build_map_prompt,
    build_prompt,
    build_reduce_prompt,
    load_slots as load_generation_schema,
)
from src.judge_slots import (
    JUDGE_JSON_SCHEMA,
    Sitrep as JudgeSitrep,
    build_system as build_judge_system,
    build_user as build_judge_user,
    load_schema as load_judge_schema,
)
from src.paper_analysis import EVENTS, JUDGE_FILES, PAPER_TABLES, RESULTS, read_csv
from src.util import ensure_dir

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "supplementary" / "reproducibility"
RAW_MANIFEST = ROOT / "data" / "raw" / "reliefweb_manual" / "manifest.csv"
RAW_REPORTS = RAW_MANIFEST.parent
PROCESSED = ROOT / "data" / "processed"


DATE_OVERRIDES = {
    "cyclone-idai-2019/ETC Mozambique SitRep #10_1.pdf": "2019-04-14",
    "hurricane-irma-2017/HurricanIrma_EMT_SITREP_01.pdf": "2017-09-07",
    "hurricane-irma-2017/SitRep 1 Hurricane Irma UNS Cuba 0709017en.pdf": "2017-09-07",
    "hurricane-irma-2017/SitRep 3 Hurricane Irma UNS Cuba 0909017en.pdf": "2017-09-09",
    "hurricane-irma-2017/cdema_sitrep_6_hurricane_irma.pdf": "2017-09-13",
    "hurricane-maria-2017/CDEMA Situation Report #8 - Hurricane Maria.pdf": "2017-10-04",
    "hurricane-maria-2017/cdema_sitrep_1_hurricane_maria.pdf": "2017-09-20",
    "hurricane-maria-2017/cdema_sitrep_2_hurricane_maria.pdf": "2017-09-22",
    "hurricane-maria-2017/cdema_sitrep_9_hurricane_maria.pdf": "2017-10-06",
    "kerala-floods-2018/jdna-kerala-report_1st-draft.pdf": "2018-10-20",
    "puebla-mexico-earthquake-2017/6.INFOGRAFIACDMX.pdf": "2017-10-02",
    "puebla-mexico-earthquake-2017/8.INFOGRAFIAOAXACA.pdf": "2017-10-02",
    "puebla-mexico-earthquake-2017/MX-Infografía USAR sismo 7.1-OCHA-20161002 Spanish.pdf":
        "2017-10-02",
    "puebla-mexico-earthquake-2017/MX-Snapshot USAR Earthquake 7.1-OCHA-20161002 English.pdf":
        "2017-10-02",
}


def write_csv(path: Path, rows: Iterable[dict[str, Any]], columns: Iterable[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(columns), extrasaction="ignore", restval="")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export_schema() -> None:
    with (PROCESSED / "schema.yaml").open(encoding="utf-8") as stream:
        schema = yaml.safe_load(stream)
    schema["approved_by"] = "anonymous author"
    schema["source_file"] = "schema_review.yaml (private adjudication trail)"
    (OUT / "frozen_schema.yaml").write_text(
        yaml.safe_dump(schema, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )


def export_prompts() -> None:
    generation_slots = load_generation_schema(PROCESSED / "schema.yaml")
    judge_slots = load_judge_schema(PROCESSED / "schema.yaml")
    sentinel_posts = [{"class_label": "POST_CLASS", "text": "<SOCIAL_MEDIA_POST_TEXT>"}]
    open_user, _ = open_code.build_user_prompt({
        "event": "<EVENT>", "source": "<SOURCE>", "date": "<DATE>",
        "phase": "<RESPONSE_PHASE>", "title": "<REPORT_TITLE>",
        "text": "<PROFESSIONAL_REPORT_TEXT>",
    })
    cluster = Cluster(
        members=[OpenCode("<REPORT_ID>", 0, "<CODE_NAME>", "<CODE_DEFINITION>")],
        slot_id="<SLOT_ID>",
    )
    naming = naming_requests([cluster], n_sitreps=56)[0]
    judge_sitrep = JudgeSitrep(
        id="<REPORT_ID>", kind="<human_OR_machine>", event="<EVENT>",
        text="<SITUATION_REPORT_TEXT>", date="<DATE>",
    )
    generic_sitrep = reference_metrics.Sitrep(
        id="<REPORT_ID>", event="<EVENT>", date="<DATE>", text="<SITUATION_REPORT_TEXT>"
    )

    calls = [
        {"role": "open-coding", "stage": "one report", "system": open_code.SYSTEM_PROMPT,
         "user": open_user, "json_schema": open_code.CODE_JSON_SCHEMA},
        {"role": "schema-consolidation", "stage": "name one cluster", "system": naming["system"],
         "user": naming["user"], "json_schema": naming["json_schema"]},
    ]
    for arm in ("generic", "schema_guided"):
        single = build_prompt(sentinel_posts, arm, generation_slots, event="EVENT", day="<DATE>")
        mapped = build_map_prompt(sentinel_posts, arm, generation_slots, event="EVENT", day="<DATE>",
                                  chunk=1, n_chunks=2)
        reduced = build_reduce_prompt(["<FACTUAL_DIGEST_1>", "<FACTUAL_DIGEST_2>"], arm,
                                      generation_slots, event="EVENT", day="<DATE>")
        calls.extend([
            {"role": "generation", "stage": f"single/{arm}", "system": single.system,
             "user": single.user, "json_schema": None},
            {"role": "generation", "stage": f"map/{arm}", "system": mapped.system,
             "user": mapped.user, "json_schema": None},
            {"role": "generation", "stage": f"reduce/{arm}", "system": reduced.system,
             "user": reduced.user, "json_schema": None},
        ])
    calls.extend([
        {"role": "slot-judging", "stage": "one document-slot", "system": build_judge_system(judge_slots),
         "user": build_judge_user(judge_sitrep, judge_slots[0]), "json_schema": JUDGE_JSON_SCHEMA},
        {"role": "generic-quality-judge", "stage": "one machine report",
         "system": reference_metrics.JUDGE_SYSTEM, "user": reference_metrics.judge_prompt(generic_sitrep),
         "json_schema": reference_metrics.JUDGE_SCHEMA},
    ])
    write_json(OUT / "prompts.json", {
        "format": "Exact request strings rendered with angle-bracketed sentinel payload values",
        "replacement_rule": "Replace sentinel values only; retain every other character and instruction.",
        "calls": calls,
    })


def model_rows() -> list[dict[str, str]]:
    return [
        {"paper_name": "api-strong", "role": "generator; open coder; cluster namer",
         "provider": "DeepSeek direct", "public_model_id": "deepseek-v4-pro",
         "version_or_snapshot": "DeepSeek-V4-Pro GA mutable alias updated 2026-08-13",
         "call_dates_utc": "2026-08-30 to 2026-08-31",
         "snapshot_note": "Provider exposed no immutable snapshot suffix; request log served_model matched alias.",
         "documentation_url": "https://api-docs.deepseek.com/updates/"},
        {"paper_name": "api-fast", "role": "generator", "provider": "DeepSeek direct",
         "public_model_id": "deepseek-v4-flash",
         "version_or_snapshot": "DeepSeek-V4-Flash-0731, API update dated 2026-07-31",
         "call_dates_utc": "2026-08-30 to 2026-08-31",
         "snapshot_note": "Provider exposed no immutable snapshot suffix; request log served_model matched alias.",
         "documentation_url": "https://api-docs.deepseek.com/updates/"},
        {"paper_name": "DeepSeek slot judge", "role": "Idai slot judge", "provider": "DeepSeek direct",
         "public_model_id": "deepseek-v4-pro", "version_or_snapshot": "mutable API alias",
         "call_dates_utc": "2026-08-30", "snapshot_note": "served_model matched alias",
         "documentation_url": "https://api-docs.deepseek.com/updates/"},
        {"paper_name": "Luna", "role": "slot judge", "provider": "OpenRouter",
         "public_model_id": "openai/gpt-5.6-luna", "version_or_snapshot": "mutable router alias",
         "call_dates_utc": "2026-08-30 to 2026-08-31", "snapshot_note": "served_model matched alias",
         "documentation_url": "https://openrouter.ai/openai/gpt-5.6-luna"},
        {"paper_name": "Qwen", "role": "Idai slot judge", "provider": "OpenRouter",
         "public_model_id": "qwen/qwen3.8-flash", "version_or_snapshot": "mutable router alias",
         "call_dates_utc": "2026-08-30", "snapshot_note": "served_model matched alias",
         "documentation_url": "https://openrouter.ai/qwen/qwen3.8-flash"},
        {"paper_name": "Gemini", "role": "slot judge; generic quality judge; truncation sensitivity",
         "provider": "OpenRouter", "public_model_id": "google/gemini-3.7-flash",
         "version_or_snapshot": "mutable router alias", "call_dates_utc": "2026-08-31",
         "snapshot_note": "served_model matched alias",
         "documentation_url": "https://openrouter.ai/google/gemini-3.7-flash"},
    ]


def decoding_rows() -> list[dict[str, str]]:
    omitted = "omitted (provider default)"
    base = {"temperature": omitted, "top_p": "not implemented or transmitted (provider default)"}
    return [
        {**base, "role": "open coding", "model_id": "deepseek-v4-pro", "scope": "Idai human reports",
         "max_output_before_fix": "6000 requested; not transmitted", "max_output_after_fix": "6000 transmitted",
         "reported_calls": "pre-fix, effectively provider default", "effort": "medium (54); low (2)",
         "notes": "30,000-character head/tail input rule; reported calls were not rerun."},
        {**base, "role": "cluster naming", "model_id": "deepseek-v4-pro", "scope": "25 candidates",
         "max_output_before_fix": "900 requested; not transmitted", "max_output_after_fix": "900 transmitted",
         "reported_calls": "pre-fix, effectively provider default", "effort": "low",
         "notes": "Reported calls were not rerun."},
        {**base, "role": "generation", "model_id": "deepseek-v4-pro; deepseek-v4-flash",
         "scope": "Idai pilot; four replication events", "max_output_before_fix": "2000 requested; not transmitted",
         "max_output_after_fix": "48000 requested and transmitted",
         "reported_calls": "Idai pre-fix; replication events post-fix", "effort": "omitted",
         "notes": "Probe-only 8,000/32,000 caps were not used for reported replication corpora."},
        {**base, "role": "slot judge", "model_id": "deepseek-v4-pro; openai/gpt-5.6-luna",
         "scope": "Idai", "max_output_before_fix": "DeepSeek: 400 requested, not transmitted; Luna: 400 transmitted",
         "max_output_after_fix": "400 transmitted", "reported_calls": "DeepSeek pre-fix; Luna unaffected",
         "effort": "low", "notes": "Reported DeepSeek judgements were not rerun."},
        {**base, "role": "slot judge", "model_id": "qwen/qwen3.8-flash", "scope": "Idai",
         "max_output_before_fix": "3500 transmitted", "max_output_after_fix": "3500 transmitted",
         "reported_calls": "OpenRouter, unaffected", "effort": "low", "notes": ""},
        {**base, "role": "slot judge", "model_id": "google/gemini-3.7-flash",
         "scope": "Idai; Puebla; Kerala", "max_output_before_fix": "1200",
         "max_output_after_fix": "1200", "reported_calls": "OpenRouter, unaffected", "effort": "low", "notes": ""},
        {**base, "role": "slot judge", "model_id": "openai/gpt-5.6-luna; google/gemini-3.7-flash",
         "scope": "Irma; Maria", "max_output_before_fix": "3500", "max_output_after_fix": "3500",
         "reported_calls": "OpenRouter, unaffected", "effort": "low", "notes": ""},
        {**base, "role": "slot judge", "model_id": "openai/gpt-5.6-luna", "scope": "Puebla; Kerala",
         "max_output_before_fix": "1200", "max_output_after_fix": "1200",
         "reported_calls": "OpenRouter, unaffected", "effort": "low", "notes": ""},
        {**base, "role": "generic quality judge", "model_id": "google/gemini-3.7-flash",
         "scope": "Idai; Irma; Maria", "max_output_before_fix": "512", "max_output_after_fix": "512",
         "reported_calls": "OpenRouter, unaffected", "effort": "low",
         "notes": "Table 3 generic-judge score; schema blind."},
        {**base, "role": "untruncated sensitivity judge", "model_id": "google/gemini-3.7-flash",
         "scope": "12 long human reports", "max_output_before_fix": "1200", "max_output_after_fix": "1200",
         "reported_calls": "OpenRouter, unaffected", "effort": "low",
         "notes": "Full report text, no 20,000-character cap."},
    ]


def export_models_and_decoding() -> None:
    models = model_rows()
    write_csv(OUT / "model_roles.csv", models, models[0])
    decoding = decoding_rows()
    write_csv(OUT / "decoding_parameters.csv", decoding, decoding[0])


def ingested_index() -> dict[str, dict[str, str]]:
    index: dict[str, dict[str, str]] = {}
    for event_dir in (PROCESSED / "sitreps_human").iterdir():
        manifest = event_dir / "_manifest_ingested.csv"
        if manifest.is_file():
            for row in read_csv(manifest):
                index[row["file"].replace("\\", "/")] = row
    return index


def export_report_manifest() -> None:
    raw = read_csv(RAW_MANIFEST)
    ingested = ingested_index()
    hashes = {row["file"]: sha256(RAW_REPORTS / row["file"]) for row in raw}
    duplicate_hashes = {digest for digest, count in Counter(hashes.values()).items() if count > 1}
    rows = []
    for position, row in enumerate(raw, start=1):
        file = row["file"].replace("\\", "/")
        admitted = ingested.get(file)
        out_file = admitted.get("out_file", "") if admitted else ""
        evaluated = bool(admitted and out_file and not Path(out_file).name.startswith("_"))
        if evaluated:
            exclusion = ""
        elif hashes[file] in duplicate_hashes:
            exclusion = "duplicate_sha256"
        elif out_file.startswith("_"):
            exclusion = "reserved_prefix_output_excluded_by_manifest_rule"
        else:
            exclusion = "not_admitted_at_ingest"
        original_date = row.get("date", "")
        document_date = DATE_OVERRIDES.get(file, original_date)
        chars = int(admitted.get("chars", 0) or 0) if admitted else 0
        title_query = f'https://reliefweb.int/search/results?search={quote_plus(chr(34) + row["title"] + chr(34))}'
        rows.append({
            "manifest_row": position, "event": row["event"], "title": row["title"],
            "source": row.get("source", ""), "document_date": document_date,
            "date_in_collection_manifest": original_date, "date_verified_or_corrected": int(
                bool(DATE_OVERRIDES.get(file))
            ), "retrieval_url": row.get("url") or title_query,
            "retrieval_url_type": "direct" if row.get("url") else "ReliefWeb exact-title search",
            "local_filename": file, "sha256": hashes[file], "document_id": admitted.get("id", "") if admitted else "",
            "extracted_characters": chars, "slot_judge_character_cap": 20000,
            "characters_sent_to_slot_judge": min(chars, 20000),
            "truncated_for_original_slot_judging": int(chars > 20000),
            "evaluated": int(evaluated), "exclusion_reason": exclusion,
        })
    if len(rows) != 125 or sum(row["evaluated"] for row in rows) != 121:
        raise RuntimeError("report manifest must contain 125 collected and 121 evaluated documents")
    write_csv(OUT / "reliefweb_report_manifest.csv", rows, rows[0])


def export_document_and_source_manifests() -> None:
    """Release hashes/provenance without copying reports that can repeat social-media PII."""
    model_ids = {"api-strong": "deepseek-v4-pro", "api-fast": "deepseek-v4-flash"}
    documents: list[dict[str, Any]] = []
    for event in EVENTS:
        candidates = [PROCESSED / "sitreps_machine" / event,
                      PROCESSED / "sitreps_machine" / event.replace("-", "_")]
        folder = next(path for path in candidates if path.is_dir())
        for path in sorted(folder.glob("*.json")):
            if path.name.startswith("_"):
                continue
            record = json.loads(path.read_text(encoding="utf-8"))
            report_text = str(record.get("text") or "")
            documents.append({
                "document_id": str(record.get("id") or path.stem), "event": event,
                "day": str(record.get("day") or ""), "paper_model": str(record.get("model") or ""),
                "public_model_id": model_ids.get(str(record.get("model") or ""), ""),
                "arm": str(record.get("arm") or ""), "n_posts": int(record.get("n_posts") or 0),
                "generated_at": str(record.get("generated_at") or ""),
                "request_hash": str(record.get("request_hash") or ""),
                "text_sha256": hashlib.sha256(report_text.encode("utf-8")).hexdigest(),
                "text_characters": len(report_text), "evaluated": int(bool(report_text.strip())),
            })
    if len(documents) != 264 or sum(row["evaluated"] for row in documents) != 261:
        raise RuntimeError("machine document export must contain 264 generated and 261 evaluated reports")
    write_csv(OUT / "machine_report_manifest.csv", documents, documents[0])

    source_rows: list[dict[str, Any]] = []
    streams = PROCESSED / "streams"
    for event_dir in sorted(path for path in streams.iterdir() if path.is_dir()):
        for day_file in sorted(event_dir.glob("*.jsonl")):
            with day_file.open(encoding="utf-8") as stream:
                for position, line in enumerate(stream, start=1):
                    record = json.loads(line)
                    source_text = str(record.get("text") or "")
                    source_rows.append({
                        "event_directory": event_dir.name, "day": day_file.stem,
                        "row_in_day_file": position, "tweet_id": record.get("tweet_id", ""),
                        "created_at": record.get("created_at", ""),
                        "class_label": record.get("class_label", ""), "split": record.get("split", ""),
                        "text_sha256": hashlib.sha256(source_text.encode("utf-8")).hexdigest(),
                    })
    write_csv(OUT / "humaid_source_manifest.csv", source_rows, source_rows[0])


def export_measurement_spec() -> None:
    packages = {
        name: version(name) for name in (
            "bert-score", "rouge-score", "sentence-transformers", "scikit-learn", "scipy", "transformers",
            "numpy", "torch", "tokenizers", "huggingface-hub",
        )
    }
    write_json(OUT / "measurement_spec.json", {
        "figure_grounding": {
            "number_regex": metrics._NUM_RE.pattern,
            "date_exclusion_regex": metrics._DATE_RE.pattern,
            "scales": metrics._SCALES,
            "casualty_lexicon": metrics._CASUALTY_WORDS,
            "quantity_lexicon": metrics._QUANTITY_WORDS,
            "context_characters": metrics._CONTEXT_CHARS,
            "clause_break_characters": metrics._CLAUSE_BREAKS,
            "normalization": "Remove comma, ASCII space, and NBSP; parse float; multiply scale; keep count/percent separate.",
            "matching": "Exact equality of (unit, round(normalized_value, 6)); no relative or absolute tolerance.",
            "deduplication": "Unique within document by the same (unit, rounded value) key.",
            "exclusions": "Recognized dates and bare integer years 1900-2100 are removed.",
            "summary_unit": "Official-in-stream is the arithmetic mean over unique valid event-days, because its official value is duplicated across machine arms. Other rates average same-day machine-official pairs. Undefined denominators are dropped.",
            "uncertainty": "10,000 percentile-bootstrap resamples using random.Random(17): unique event-days for official-in-stream and valid same-day machine-official pairs for all other rates.",
        },
        "manual_matcher_audit": {
            "scope": "Idai, Irma, and Maria only",
            "sampling_seed": 17,
            "stream_absent_sample": "50 simple-random unique (event, day, unit, normalized official value) items from 1,910 regex-declared absences.",
            "matched_sample": "25 simple-random official-stream exact matches from 174 plus 25 simple-random machine-stream exact matches from 2,845.",
            "review": "One anonymous author checked private official/machine context and up to three same-day stream candidates; excerpts are omitted and SHA-256 hashes released.",
            "error_intervals": "Two-sided 95% Clopper-Pearson exact binomial intervals.",
            "released_files": ["matcher_audit.csv", "matcher_audit_summary.csv"],
        },
        "reference_pairing": {
            "priority": "same day, then nearest report within +/-3 days; maximum 3 references",
            "tie_break": "absolute day gap, signed day gap, document id",
            "event_fallback": False,
        },
        "rouge_l": {
            "package": f"rouge-score=={packages['rouge-score']}",
            "implementation": "rouge_score.rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)",
            "tokenization": "Package default: lowercase, replace non-[a-z0-9] with spaces, split, Porter stem tokens >3 chars.",
            "multi_reference": "Return precision/recall/F1 from the reference with maximum ROUGE-L F1.",
        },
        "bertscore": {
            "package": f"bert-score=={packages['bert-score']}",
            "model_type": "roberta-large",
            "huggingface_revision": "722cf37b1afa9454edce342e7895e588b6ff1d59",
            "language": "en", "device": "cpu", "batch_size": 8,
            "rescale_with_baseline": False,
            "multi_reference": "bert-score list-of-references API; maximum match per candidate",
        },
        "schema_induction": {
            "sentence_transformers_package": f"sentence-transformers=={packages['sentence-transformers']}",
            "checkpoint": "sentence-transformers/all-MiniLM-L6-v2",
            "huggingface_revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
            "input": "code name + ' - ' + code definition",
            "encode": {"batch_size": 32, "normalize_embeddings": True, "convert_to_numpy": True},
            "clustering": {"implementation": "sklearn.cluster.AgglomerativeClustering",
                           "metric": "precomputed cosine distance", "linkage": "average",
                           "n_clusters": None, "distance_threshold": 0.35},
        },
        "human_slot_judging_input": {
            "original_rule": "Send the first 20,000 Unicode characters, then append '\\n[...truncated...]'.",
            "unit": "Python string character/code-point count after PDF text extraction",
            "ceiling": "Mean observed-case C_j(d) over the 56 manifest-admitted Idai human reports.",
            "untruncated_sensitivity": "Seed-17 sample of four >20,000-character reports per primary event; full text.",
        },
        "human_judge_validation": {
            "event": "cyclone-idai-2019",
            "sample_design": "100 seed-17 slot-stratified document-slot items; 50 human-report and 50 machine-report items; all 15 slots represented.",
            "annotator": "The anonymous schema author, who was blind to the withheld four-judge key while scoring.",
            "labels": ["absent", "partial", "present"],
            "agreement": "Exact-match accuracy and unweighted nominal Cohen's kappa; linear-weighted Cohen's kappa is a sensitivity statistic. Per-label precision, recall, and F1 use one-vs-rest confusion counts with the human label as ground truth.",
            "missing_rule": "A missing model verdict is excluded only from that judge's denominator; expected, scored, and missing counts are reported by scope.",
            "preregistered_threshold": "Raw judge-versus-human agreement >= 0.7 on the 100-item sample (PLAN Phase 4). The plan did not specify a kappa threshold and did not apply this threshold to inter-judge Fleiss' kappa.",
            "released_files": ["tables/human_validation_pairs.csv", "tables/human_validation_summary.csv"],
        },
        "software_versions": packages,
        "runtime": {"python": platform.python_version(), "implementation": platform.python_implementation(),
                    "platform": platform.platform()},
    })


def export_environment() -> None:
    distributions = sorted(
        {dist.metadata["Name"]: dist.version for dist in __import__("importlib.metadata").metadata.distributions()}.items(),
        key=lambda item: item[0].lower(),
    )
    (OUT / "environment.txt").write_text(
        f"Python=={platform.python_version()}\nPlatform={platform.platform()}\n" +
        "".join(f"{name}=={package_version}\n" for name, package_version in distributions),
        encoding="utf-8",
    )


def export_tables() -> None:
    table_dir = ensure_dir(OUT / "tables")
    for source in PAPER_TABLES.glob("*.csv"):
        shutil.copyfile(source, table_dir / source.name)
    event_table_sources = {
        "idai": RESULTS,
        "irma": RESULTS / "hurricane-irma-2017",
        "maria": RESULTS / "hurricane-maria-2017",
        "puebla": RESULTS / "puebla-mexico-earthquake-2017",
        "kerala": RESULTS / "kerala-floods-2018",
    }
    for event, folder in event_table_sources.items():
        for name in ("figure_agreement.csv", "source_availability.csv", "reference_metrics.csv"):
            source = folder / name
            if source.is_file():
                shutil.copyfile(source, table_dir / f"{event}_{name}")
    for name in ("correlations_gemini.csv", "correlations_luna.csv"):
        source = RESULTS / name
        if source.is_file():
            shutil.copyfile(source, table_dir / f"idai_{name}")
    for event, folder in (("irma", event_table_sources["irma"]),
                          ("maria", event_table_sources["maria"])):
        source = folder / "correlations_luna.csv"
        if source.is_file():
            shutil.copyfile(source, table_dir / f"{event}_correlations_luna.csv")
    judgement_rows: list[dict[str, Any]] = []
    for event, judges in JUDGE_FILES.items():
        for judge, source in judges.items():
            for row in read_csv(source):
                safe = {key: value for key, value in row.items() if key != "evidence"}
                judgement_rows.append({"publication_event": event, "publication_judge": judge, **safe})
    columns = list(judgement_rows[0])
    write_csv(OUT / "per_judgement.csv", judgement_rows, columns)


def export_matcher_audit_spec() -> None:
    source = PROCESSED / "matcher_audit_labels.json"
    if not source.is_file():
        raise FileNotFoundError("run python -m src.matcher_audit before exporting the artifact")
    shutil.copyfile(source, OUT / "matcher_audit_label_spec.json")


def export_readme() -> None:
    text = """# Anonymous reproducibility supplement

This directory is the submission artifact for the paper. It contains no report body, social-media
text, evidence span, API credential, or author identity.

## Re-run order

1. Use `reliefweb_report_manifest.csv` to retrieve the 125 public reports. When a direct landing
   page was not preserved during collection, `retrieval_url` is an exact-title ReliefWeb query;
   title, source, verified date, filename, and SHA-256 disambiguate the document. Preserve each
   `local_filename`, then run `python -m src.ingest_manual_reliefweb`.
2. Obtain HumAID under its licence and run `python -m src.collect_social --events
   cyclone_idai_2019 hurricane_irma_2017 hurricane_maria_2017 puebla_mexico_earthquake_2017
   kerala_floods_2018`. Require every `(tweet_id, text_sha256)` in `humaid_source_manifest.csv` to
   match before making an API call.
3. Copy `frozen_schema.yaml` to `data/processed/schema.yaml`. Re-issue calls using `prompts.json`,
   `model_roles.csv`, and `decoding_parameters.csv`; the repository CLIs are `src.open_code`,
   `src.consolidate`, `src.generate_sitreps`, `src.judge_slots`, and `src.reference_metrics`.
   Exact original generation is checksum-auditable but cannot be promised from mutable aliases.
4. Run `python -m src.paper_analysis`, `python -m src.matcher_audit`, then
   `python -m src.export_reproducibility`.
5. Run `python -m src.figures --tables-dir results/tables/paper/idai/gemini --completeness
   results/tables/idai/gemini/completeness_by_sitrep.csv --reference-metrics
   results/tables/reference_metrics.csv --correlations results/tables/correlations_gemini.csv`.
6. From `paper/`, run `pdflatex main`, `bibtex main`, and two further `pdflatex main` passes.

For a deterministic audit that does not re-call mutable models, copy `tables/*.csv` back to
`results/tables/paper/` and use `per_judgement.csv` with the last-row/observed-case rules implemented
in `src.paper_analysis`. The release tests in `tests/test_export_reproducibility.py` enforce all
manifest counts and model/prompt/schema contracts.

## Files

- `prompts.json`: every system and user instruction, rendered verbatim with sentinel payloads.
- `frozen_schema.yaml`: all 15 definitions and two synthetic examples per slot, author-anonymized.
- `model_roles.csv`: public model identifiers, provider surfaces, call dates, and snapshot limits.
- `decoding_parameters.csv`: temperature, top-p, effort, and every reported output-token cap.
- `reliefweb_report_manifest.csv`: 125 titles, dates, retrieval URLs, hashes, input lengths, and
  uniform inclusion/exclusion reasons. The four exclusions yield the 121-report analysis corpus.
- `machine_report_manifest.csv`: provenance and hashes for all 264 generated outputs; `evaluated`
  identifies the 261 non-empty reports used in analysis. Output text is not public because spot
  checks found that generations can repeat personal contact details from crisis posts.
- `humaid_source_manifest.csv`: non-text tweet identifiers, dates, labels, order, and text hashes.
  Obtain HumAID under its licence and require every hash to match before regeneration.
- `measurement_spec.json`: exact regexes, normalization, pairing, ROUGE, BERTScore, and clustering.
- `matcher_audit.csv` and `matcher_audit_summary.csv`: text-free seed-17 manual labels, context
  hashes, and exact-binomial error bounds for 50 absent and 50 matched primary-event figures.
- `matcher_audit_label_spec.json`: the complete anonymous label decisions used to regenerate the
  public audit tables after inspecting the private worklist.
- `environment.txt`: Python/platform metadata and the complete installed distribution set.
- `per_judgement.csv`: text-free raw document-slot rows, including status and duplicates; the
  publication analysis applies its documented last-row and observed-case rules.
- `tables/human_validation_pairs.csv` and `tables/human_validation_summary.csv`: the 100 blind
  author annotations joined to all four withheld judge labels, plus recomputed all/human/machine
  accuracy, nominal Cohen's kappa, linear-weighted sensitivity, per-label precision/recall/F1, and
  per-judge missingness.
- `tables/`: generated publication tables plus per-report reference metrics and per-day
  figure/source-availability records; these recompute every Table 2 percentage and Table 3 star.

API aliases are not deterministic snapshots. The artifact records the public ID, call date, exact
request parameters, served-model string, and cached response provenance available from each
provider; DeepSeek and OpenRouter did not expose immutable snapshot IDs for these calls.
"""
    (OUT / "README.md").write_text(text, encoding="utf-8")


def main() -> int:
    ensure_dir(OUT)
    export_schema()
    export_prompts()
    export_models_and_decoding()
    export_report_manifest()
    export_document_and_source_manifests()
    export_measurement_spec()
    export_environment()
    export_tables()
    export_matcher_audit_spec()
    export_readme()
    print(f"reproducibility artifact: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
