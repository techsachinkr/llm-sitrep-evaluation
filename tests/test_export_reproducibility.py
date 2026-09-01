import csv
import json
from pathlib import Path

import yaml

from src.export_reproducibility import decoding_rows, model_rows


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "supplementary" / "reproducibility"


def read_csv(name: str) -> list[dict[str, str]]:
    with (ARTIFACT / name).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_every_paper_model_alias_resolves_to_a_public_identifier():
    models = {row["paper_name"]: row for row in model_rows()}
    assert models["api-strong"]["public_model_id"] == "deepseek-v4-pro"
    assert models["api-fast"]["public_model_id"] == "deepseek-v4-flash"
    assert models["Gemini"]["public_model_id"] == "google/gemini-3.7-flash"
    assert all(row["call_dates_utc"] and row["documentation_url"] for row in models.values())


def test_decoding_table_states_every_requested_parameter_and_cap_state():
    rows = decoding_rows()
    assert rows
    required = {
        "temperature", "top_p", "role", "model_id", "max_output_before_fix",
        "max_output_after_fix", "reported_calls", "effort",
    }
    assert all(required <= row.keys() for row in rows)
    generation = next(row for row in rows if row["role"] == "generation")
    assert generation["max_output_before_fix"] == "2000 requested; not transmitted"
    assert generation["max_output_after_fix"] == "48000 requested and transmitted"


def test_checked_in_artifact_reconciles_corpus_schema_prompts_and_documents():
    reports = read_csv("reliefweb_report_manifest.csv")
    machines = read_csv("machine_report_manifest.csv")
    assert len(reports) == 125
    assert sum(int(row["evaluated"]) for row in reports) == 121
    assert len(machines) == 264
    assert sum(int(row["evaluated"]) for row in machines) == 261

    schema = yaml.safe_load((ARTIFACT / "frozen_schema.yaml").read_text(encoding="utf-8"))
    assert len(schema["slots"]) == 15
    assert all(len(slot["examples"]) == 2 for slot in schema["slots"])
    assert schema["approved_by"] == "anonymous author"

    prompts = json.loads((ARTIFACT / "prompts.json").read_text(encoding="utf-8"))
    roles = {call["role"] for call in prompts["calls"]}
    assert roles == {
        "open-coding", "schema-consolidation", "generation", "slot-judging",
        "generic-quality-judge",
    }
    assert len(prompts["calls"]) == 10


def test_public_artifact_contains_no_raw_social_text_or_evidence_column():
    source_columns = read_csv("humaid_source_manifest.csv")[0]
    judgement_columns = read_csv("per_judgement.csv")[0]
    machine_columns = read_csv("machine_report_manifest.csv")[0]
    assert "text" not in source_columns
    assert "evidence" not in judgement_columns
    assert "text" not in machine_columns
    assert "text_sha256" in source_columns
    assert "text_sha256" in machine_columns


def test_matcher_audit_is_complete_and_text_free():
    audit = read_csv("matcher_audit.csv")
    summary = read_csv("matcher_audit_summary.csv")
    assert len(audit) == 100
    assert {row["sample_items"] for row in summary} == {"25", "50"}
    assert {row["manual_errors"] for row in summary} == {"1", "5", "21"}
    assert all("context_sha256" in key for key in audit[0] if "context" in key)
    spec = json.loads((ARTIFACT / "measurement_spec.json").read_text(encoding="utf-8"))
    assert spec["manual_matcher_audit"]["sampling_seed"] == 17
    labels = json.loads((ARTIFACT / "matcher_audit_label_spec.json").read_text(encoding="utf-8"))
    assert labels["review_status"].startswith("all 100")


def test_human_validation_is_released_and_fully_recomputable():
    pairs = read_csv("tables/human_validation_pairs.csv")
    summary = read_csv("tables/human_validation_summary.csv")
    assert len(pairs) == 100
    assert len(summary) == 12
    assert "sitrep_path" not in pairs[0]
    assert "notes" not in pairs[0]
    assert {row["n_scored"] for row in summary if row["scope"] == "all"} == {
        "90", "99", "100"
    }
    gemini = next(row for row in summary
                  if row["judge"] == "gemini-3.7-flash" and row["scope"] == "all")
    assert gemini["raw_agreement"] == "0.91"
    assert gemini["cohens_kappa"] == "0.798658"
    assert gemini["precision_absent"] == "0.875"
    assert gemini["recall_partial"] == "0.894737"
    assert gemini["f1_present"] == "0.943662"
    spec = json.loads((ARTIFACT / "measurement_spec.json").read_text(encoding="utf-8"))
    registered = spec["human_judge_validation"]["preregistered_threshold"]
    assert "Raw judge-versus-human agreement >= 0.7" in registered
    assert "did not specify a kappa threshold" in registered
