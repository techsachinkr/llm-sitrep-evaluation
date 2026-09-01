"""Offline tests for src/consolidate.py.

No network and no model download: sentence-transformers is monkeypatched (or refused, to prove
the lexical fallback), and every LLM call goes through FakeProvider injected into `src.llm.LLM`.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.consolidate import (
    DEFAULT_THRESHOLD,
    Cluster,
    OpenCode,
    apply_naming,
    build_clusters,
    cluster_vectors,
    consolidate,
    cosine_distance_matrix,
    embed_texts,
    load_open_codes,
    main,
    naming_requests,
    tfidf_matrix,
    write_outputs,
)
from src.llm import CostLimitExceeded, FakeProvider
from src.util import write_json
from tests.test_llm import make_cfg, make_llm  # noqa: F401  (shared helpers + autouse env isolation)

#: the identical code in s1 and s3 must always land in one slot, whatever the threshold
ACCESS = ("access constraints", "areas that responders cannot reach")
CODE_SETS = {
    "s1": [("casualty figures", "deaths and injuries reported with a named source"),
           ("displacement figures", "number of people displaced from their homes"),
           ACCESS],
    "s2": [("death toll", "deaths and injuries reported by the authorities"),
           ("displaced people", "number of people displaced by the flooding"),
           ("funding requirements", "money requested by the emergency appeal")],
    "s3": [("casualty numbers", "deaths and injuries reported so far"), ACCESS],
    "s4": [("funding gap", "share of the emergency appeal that is unfunded")],
}


def write_codes(codes_dir: Path, sets: dict[str, list[tuple[str, str]]] | None = None) -> Path:
    """Write one open-code file per sitrep, in the shape `src/open_code.py` emits."""
    for sid, codes in (sets or CODE_SETS).items():
        write_json(codes_dir / f"{sid}.json", {
            "sitrep_id": sid, "event": "cyclone_idai_2019", "source": "OCHA",
            "codes": [{"name": n, "definition": d, "evidence_quote": f"QUOTE for {n}", "location": "body"}
                      for n, d in codes],
        })
    return codes_dir


def slot_responder(req: dict) -> str:
    """A well-behaved naming model: echoes the slot id it was asked about."""
    slot = re.search(r"Cluster (S\d+)", req["user"]).group(1)
    return json.dumps({"slot_id": slot, "name": f"named {slot}", "definition": f"definition of {slot}",
                       "examples": [f"{slot} example one", f"{slot} example two"]})


# ---------------------------------------------------------------------------
# loading open codes
# ---------------------------------------------------------------------------
def test_load_open_codes_reads_every_file(tmp_path):
    codes, sitrep_ids = load_open_codes(write_codes(tmp_path / "oc"))
    assert sitrep_ids == ["s1", "s2", "s3", "s4"]
    assert len(codes) == 9
    assert codes[0] == OpenCode("s1", 0, "casualty figures",
                                "deaths and injuries reported with a named source",
                                "cyclone_idai_2019", "OCHA")
    assert codes[0].text == "casualty figures - deaths and injuries reported with a named source"


def test_load_open_codes_dedupes_repeated_names_within_one_sitrep(tmp_path):
    write_json(tmp_path / "oc" / "s1.json", {"sitrep_id": "s1", "codes": [
        {"name": "casualty figures", "definition": "a"},
        {"name": "Casualty Figures", "definition": "duplicate, different case"},
        {"name": "  ", "definition": "no name at all"},
        "not a dict",
        {"name": "access", "definition": "b"},
    ]})
    codes, sitrep_ids = load_open_codes(tmp_path / "oc")
    assert [c.name for c in codes] == ["casualty figures", "access"]
    assert sitrep_ids == ["s1"]


def test_load_open_codes_accepts_a_bare_list_and_falls_back_to_the_filename(tmp_path):
    write_json(tmp_path / "oc" / "rw-42.json", [{"name": "funding gap", "definition": "d"}])
    codes, sitrep_ids = load_open_codes(tmp_path / "oc")
    assert sitrep_ids == ["rw-42"] and codes[0].sitrep_id == "rw-42"


def test_load_open_codes_skips_unusable_files(tmp_path):
    write_codes(tmp_path / "oc", {"s1": [("a", "b")]})
    (tmp_path / "oc" / "broken.json").write_text("{nope", encoding="utf-8")
    write_json(tmp_path / "oc" / "nocodes.json", {"sitrep_id": "x", "codes": "not a list"})
    write_json(tmp_path / "oc" / "_index.json", {"sitrep_id": "skipme", "codes": [{"name": "n"}]})
    codes, sitrep_ids = load_open_codes(tmp_path / "oc")
    assert sitrep_ids == ["s1"] and len(codes) == 1


def test_load_open_codes_skips_sitreps_open_coding_failed_on(tmp_path):
    """A refused/parse-failed sitrep must not sit in the prevalence denominator."""
    write_codes(tmp_path / "oc", {"s1": [("a", "x"), ("b", "y")]})
    write_json(tmp_path / "oc" / "s9.json",
               {"sitrep_id": "s9", "status": "refused", "codes": [], "error": "refusal"})
    write_json(tmp_path / "oc" / "s8.json",
               {"sitrep_id": "s8", "status": "ok", "codes": [{"name": "c", "definition": "z"}]})
    codes, sitrep_ids = load_open_codes(tmp_path / "oc")
    assert sitrep_ids == ["s1", "s8"]
    assert Cluster(members=codes[:1]).prevalence(len(sitrep_ids)) == 0.5


def test_load_open_codes_keeps_a_successful_sitrep_with_no_codes(tmp_path):
    write_json(tmp_path / "oc" / "s7.json", {"sitrep_id": "s7", "status": "ok", "codes": []})
    codes, sitrep_ids = load_open_codes(tmp_path / "oc")
    assert codes == [] and sitrep_ids == ["s7"]


def test_load_open_codes_of_missing_directory_is_empty(tmp_path):
    assert load_open_codes(tmp_path / "nope") == ([], [])


# ---------------------------------------------------------------------------
# representation + clustering
# ---------------------------------------------------------------------------
def test_tfidf_matrix_is_row_normalised_and_deterministic():
    texts = ["casualty figures with a source", "casualty figures with a source", "funding appeal coverage"]
    x = tfidf_matrix(texts)
    assert np.allclose(np.linalg.norm(x, axis=1), 1.0)
    assert np.allclose(x, tfidf_matrix(texts))          # same input -> byte-identical output
    assert np.allclose(x[0], x[1])                      # identical texts -> identical vectors
    assert cosine_distance_matrix(x)[0, 2] > 0.9        # unrelated texts stay far apart


def test_tfidf_of_stopword_only_text_is_a_zero_row_at_distance_one():
    x = tfidf_matrix(["of the and", "casualty figures"])
    dist = cosine_distance_matrix(x)
    assert np.allclose(x[0], 0.0)
    assert dist[0, 1] == pytest.approx(1.0)


def test_cosine_distance_matrix_properties():
    x = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    dist = cosine_distance_matrix(x)
    assert dist[0, 1] == pytest.approx(0.0)
    assert dist[0, 2] == pytest.approx(1.0)
    assert dist[0, 3] == pytest.approx(2.0)
    assert np.allclose(dist, dist.T) and np.allclose(np.diag(dist), 0.0)


def test_cluster_vectors_groups_neighbours_and_is_deterministic():
    x = np.array([[1.0, 0.0], [0.99, 0.14], [0.0, 1.0], [0.14, 0.99]])
    labels = cluster_vectors(x, 0.35)
    assert labels[0] == labels[1] and labels[2] == labels[3] and labels[0] != labels[2]
    assert np.array_equal(labels, cluster_vectors(x, 0.35))          # deterministic, no seed needed
    assert len(set(cluster_vectors(x, 1.5).tolist())) == 1           # a loose threshold merges everything
    assert len(set(cluster_vectors(x, 0.001).tolist())) == 4         # a tight one splits everything


def test_cluster_vectors_handles_degenerate_inputs():
    assert cluster_vectors(np.zeros((0, 3)), 0.35).tolist() == []
    assert cluster_vectors(np.ones((1, 3)), 0.35).tolist() == [0]


class _FakeST:
    """Stand-in for SentenceTransformer: two-dimensional, deterministic, offline."""

    def __init__(self) -> None:
        self.encoded: list[list[str]] = []

    def encode(self, texts, **kwargs):
        self.encoded.append(list(texts))
        return np.array([[float(len(t)), 1.0] for t in texts])


def test_embed_texts_uses_the_embedding_model_and_caches_the_vectors(tmp_path, monkeypatch):
    fake = _FakeST()
    monkeypatch.setattr("src.consolidate._load_sentence_transformer", lambda name, cache: fake)
    texts = ["casualty figures", "funding gap"]
    vecs, method = embed_texts(texts, vector_cache=tmp_path / "vc")
    assert method == "sentence-transformers:all-MiniLM-L6-v2"
    assert vecs.shape == (2, 2) and fake.encoded == [texts]
    assert list((tmp_path / "vc").glob("*.npy"))
    again, _ = embed_texts(texts, vector_cache=tmp_path / "vc")
    assert np.allclose(again, vecs) and fake.encoded == [texts]      # served from the vector cache


def test_embed_texts_falls_back_to_tfidf_when_the_model_cannot_load(tmp_path, monkeypatch, caplog):
    def boom(name, cache):
        raise OSError("no network and no local copy of the model")

    monkeypatch.setattr("src.consolidate._load_sentence_transformer", boom)
    vecs, method = embed_texts(["casualty figures", "funding gap"], vector_cache=tmp_path / "vc")
    assert method == "lexical-tfidf"
    assert vecs.shape[0] == 2 and np.allclose(np.linalg.norm(vecs, axis=1), 1.0)
    assert not list((tmp_path / "vc").glob("*.npy"))
    assert any("falling back to TF-IDF" in r.getMessage() for r in caplog.records)


def test_embed_texts_falls_back_when_the_model_returns_the_wrong_shape(monkeypatch):
    class _Bad:
        def encode(self, texts, **kwargs):
            return np.zeros((1, 4))

    monkeypatch.setattr("src.consolidate._load_sentence_transformer", lambda name, cache: _Bad())
    _, method = embed_texts(["a", "b"])
    assert method == "lexical-tfidf"


def test_embed_texts_can_be_forced_lexical_without_touching_the_loader(monkeypatch):
    monkeypatch.setattr("src.consolidate._load_sentence_transformer",
                        lambda name, cache: pytest.fail("the loader must not be called"))
    _, method = embed_texts(["a b c", "d e f"], use_embeddings=False)
    assert method == "lexical-tfidf"


def test_embed_texts_of_nothing():
    vecs, method = embed_texts([])
    assert vecs.shape[0] == 0 and method == "lexical-tfidf"


# ---------------------------------------------------------------------------
# clusters, prevalence, naming
# ---------------------------------------------------------------------------
def _codes(spec: list[tuple[str, str]]) -> list[OpenCode]:
    return [OpenCode(sid, i, name, "d") for i, (sid, name) in enumerate(spec)]


def test_build_clusters_orders_by_breadth_then_size_and_assigns_slot_ids():
    codes = _codes([("s1", "wide"), ("s2", "wide"), ("s3", "wide"),        # 3 sitreps, 3 codes
                    ("s1", "big"), ("s1", "big2"), ("s2", "big"), ("s3", "big"),  # 3 sitreps, 4 codes
                    ("s4", "narrow")])                                     # 1 sitrep, 1 code
    labels = [0, 0, 0, 1, 1, 1, 1, 2]
    clusters = build_clusters(codes, labels)
    assert [c.slot_id for c in clusters] == ["S01", "S02", "S03"]
    assert len(clusters[0].members) == 4 and len(clusters[1].members) == 3   # tie on breadth -> size wins
    assert clusters[2].sitrep_ids == ["s4"]


def test_prevalence_is_the_fraction_of_sampled_sitreps():
    cluster = Cluster(members=_codes([("s1", "a"), ("s2", "a"), ("s2", "b")]))
    assert cluster.sitrep_ids == ["s1", "s2"]
    assert cluster.prevalence(4) == 0.5
    assert cluster.prevalence(3) == round(2 / 3, 4)
    assert cluster.prevalence(0) == 0.0


def test_canonical_name_is_the_most_frequent_code_name():
    cluster = Cluster(members=_codes([("s1", "Death toll"), ("s2", "death toll"), ("s3", "casualties")]))
    assert cluster.canonical_name == "death toll"
    assert cluster.name_counts == [("death toll", 2), ("casualties", 1)]


def test_naming_requests_never_put_text_in_meta():
    clusters = build_clusters(_codes([("s1", "death toll"), ("s2", "death toll")]), [0, 0])
    reqs = naming_requests(clusters, n_sitreps=4)
    assert len(reqs) == 1
    req = reqs[0]
    assert set(req["meta"]) == {"phase", "slot_id", "n_codes", "n_sitreps"}
    assert all(isinstance(v, (str, int)) for v in req["meta"].values())
    assert "death toll" not in json.dumps(req["meta"])
    assert req["model"] == "coder" and req["tag"] == "consolidate-name-v1"
    assert req["json_schema"]["required"] == ["slot_id", "name", "definition", "examples"]
    assert "prevalence 0.50" in req["user"] and "death toll" in req["user"]


class _Resp:
    def __init__(self, payload=None, ok=True, error=None):
        self.json, self.ok, self.error, self.json_error = payload, ok, error, None


def test_apply_naming_takes_the_model_output():
    clusters = build_clusters(_codes([("s1", "death toll")]), [0])
    named = apply_naming(clusters, [_Resp({"slot_id": "S01", "name": "casualties",
                                           "definition": "deaths and injuries",
                                           "examples": ["one", "two", "three"]})])
    assert named == 1
    assert clusters[0].name == "casualties" and clusters[0].named_by == "llm"
    assert clusters[0].examples == ["one", "two"]            # exactly two examples are kept


@pytest.mark.parametrize("resp", [_Resp(None, ok=False, error="boom"), _Resp(None), _Resp("not a dict"),
                                  _Resp({"name": "", "definition": "d", "examples": []})])
def test_apply_naming_falls_back_on_unusable_answers(resp):
    clusters = build_clusters(_codes([("s1", "death toll"), ("s2", "death toll")]), [0, 0])
    assert apply_naming(clusters, [resp]) == 0
    assert clusters[0].named_by == "fallback"
    assert clusters[0].name == "death toll"
    assert "2 open code(s) across 2 sitrep(s)" in clusters[0].definition
    assert clusters[0].examples == []


# ---------------------------------------------------------------------------
# consolidate()
# ---------------------------------------------------------------------------
def test_consolidate_end_to_end_with_a_fake_naming_model(tmp_path):
    codes, sitrep_ids = load_open_codes(write_codes(tmp_path / "oc"))
    llm, prov = make_llm(tmp_path, provider=FakeProvider(responder=slot_responder))
    clusters, meta = consolidate(codes, sitrep_ids, use_embeddings=False, llm=llm, batch=False)
    kept = [c for c in clusters if c.kept]
    assert meta["n_codes"] == 9 and meta["n_sitreps"] == 4
    assert meta["method"]["representation"] == "lexical-tfidf"
    assert meta["method"]["threshold"] == DEFAULT_THRESHOLD["lexical"]
    assert meta["n_candidates"] == len(kept) == meta["n_named_by_model"] == prov.calls
    assert all(c.named_by == "llm" and c.examples and c.definition for c in kept)
    assert [c.slot_id for c in kept] == [f"S{i:02d}" for i in range(1, len(kept) + 1)]
    assert meta["sitrep_ids"] == ["s1", "s2", "s3", "s4"]
    assert "PENDING" in meta["user_checkpoint"]
    # the two "access constraints" codes are the same string, so they must land in the same slot
    home = {(c.sitrep_id, c.name): cl.slot_id for cl in clusters for c in cl.members}
    assert home[("s1", "access constraints")] == home[("s3", "access constraints")]


def test_consolidate_without_an_llm_uses_deterministic_names(tmp_path):
    codes, sitrep_ids = load_open_codes(write_codes(tmp_path / "oc"))
    clusters, meta = consolidate(codes, sitrep_ids, use_embeddings=False)
    assert meta["n_named_by_model"] == 0 and meta["method"]["naming_model"] == "none"
    assert all(c.named_by == "fallback" for c in clusters if c.kept)


def test_consolidate_caps_candidates_and_marks_the_rest_dropped(tmp_path):
    codes, sitrep_ids = load_open_codes(write_codes(tmp_path / "oc"))
    clusters, meta = consolidate(codes, sitrep_ids, use_embeddings=False, threshold=0.01, max_candidates=3)
    assert meta["n_clusters"] > 3 and meta["n_candidates"] == 3
    assert [c.kept for c in clusters][:3] == [True, True, True]
    assert not any(c.kept for c in clusters[3:])


def test_consolidate_min_codes_drops_singleton_clusters(tmp_path):
    codes, sitrep_ids = load_open_codes(write_codes(tmp_path / "oc"))
    clusters, meta = consolidate(codes, sitrep_ids, use_embeddings=False, threshold=0.01, min_codes=2)
    # at this threshold only the two identical "access constraints" codes still cluster together
    assert meta["n_candidates"] == 1
    kept = [c for c in clusters if c.kept]
    assert len(kept[0].members) == 2 and kept[0].canonical_name == "access constraints"


def test_consolidate_gates_on_cost_before_calling_the_model(tmp_path):
    codes, sitrep_ids = load_open_codes(write_codes(tmp_path / "oc"))
    llm, prov = make_llm(tmp_path, provider=FakeProvider(responder=slot_responder),
                         cost_limit_usd_per_run=1e-9)
    with pytest.raises(CostLimitExceeded):
        consolidate(codes, sitrep_ids, use_embeddings=False, llm=llm, batch=False)
    assert prov.calls == 0


# ---------------------------------------------------------------------------
# outputs
# ---------------------------------------------------------------------------
def test_write_outputs_writes_yaml_json_and_csv(tmp_path):
    codes, sitrep_ids = load_open_codes(write_codes(tmp_path / "oc"))
    llm, _ = make_llm(tmp_path, provider=FakeProvider(responder=slot_responder))
    clusters, meta = consolidate(codes, sitrep_ids, use_embeddings=False, llm=llm, batch=False,
                                 max_candidates=2, max_slots=15)
    paths = write_outputs(clusters, meta, tmp_path / "schema", tmp_path / "tables")

    doc = yaml.safe_load(paths["yaml"].read_text(encoding="utf-8"))
    assert doc["n_sitreps"] == 4 and doc["max_slots"] == 15 and len(doc["slots"]) == 2
    slot = doc["slots"][0]
    assert set(slot) == {"slot_id", "name", "definition", "examples", "prevalence", "n_sitreps",
                         "n_codes", "named_by", "top_codes"}
    assert slot["slot_id"] == "S01" and len(slot["examples"]) == 2
    assert isinstance(slot["prevalence"], float) and 0.0 < slot["prevalence"] <= 1.0

    decisions = json.loads(paths["decisions"].read_text(encoding="utf-8"))
    assert len(decisions["clusters"]) == 2 and decisions["dropped_clusters"]
    audited = {(m["sitrep_id"], m["code_index"]) for c in decisions["clusters"] + decisions["dropped_clusters"]
               for m in c["members"]}
    assert len(audited) == 9                                   # every open code is accounted for
    assert "evidence_quote" not in paths["decisions"].read_text(encoding="utf-8")
    assert decisions["sitrep_ids"] == ["s1", "s2", "s3", "s4"]

    with open(paths["csv"], encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert [r["slot_id"] for r in rows] == ["S01", "S02"]
    assert rows[0]["named_by"] == "llm" and rows[0]["example_1"] == "S01 example one"
    assert rows[0]["top_codes"]


def test_write_outputs_suffix_keeps_smoke_artefacts_separate(tmp_path):
    codes, sitrep_ids = load_open_codes(write_codes(tmp_path / "oc"))
    clusters, meta = consolidate(codes, sitrep_ids, use_embeddings=False)
    paths = write_outputs(clusters, meta, tmp_path / "schema", tmp_path / "tables", suffix="_smoke")
    assert paths["yaml"].name == "candidate_slots_smoke.yaml"
    assert not (tmp_path / "schema" / "candidate_slots.yaml").exists()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def write_config(tmp_path: Path) -> Path:
    """A config whose paths and models all point somewhere harmless and offline."""
    cfg = {
        "seed": 17,
        "paths": {"raw": str(tmp_path / "data/raw"), "processed": str(tmp_path / "data/processed"),
                  "results": str(tmp_path / "results")},
        "llm": {"cache_dir": str(tmp_path / "cache"), "log_file": str(tmp_path / "log.jsonl"),
                "max_concurrency": 2, "cost_limit_usd_per_run": 25},
        "models": {"generation": [{"name": "api-fast", "provider": "fake", "id": "fake-fast"}],
                   "coder": {"name": "coder", "provider": "fake", "id": "fake-coder"},
                   "judge": {"name": "judge", "provider": "fake", "id": "fake-judge"}},
        "schema": {"max_slots": 15},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def test_main_degrades_gracefully_without_open_codes(tmp_path, capsys):
    rc = main(["--config", str(write_config(tmp_path))])
    assert rc == 2
    out = capsys.readouterr().out
    assert "No open codes found" in out and "src.open_code" in out


def test_main_smoke_runs_offline_and_writes_smoke_artefacts(tmp_path, capsys):
    config = write_config(tmp_path)
    write_codes(tmp_path / "data/processed/schema/open_codes")
    rc = main(["--config", str(config), "--smoke", "2", "--no-embeddings"])
    assert rc == 0
    schema_dir = tmp_path / "data/processed/schema"
    doc = yaml.safe_load((schema_dir / "candidate_slots_smoke.yaml").read_text(encoding="utf-8"))
    assert doc["n_sitreps"] == 2 and doc["smoke"] == 2 and doc["slots"]
    assert (schema_dir / "consolidation_decisions_smoke.json").exists()
    assert (tmp_path / "results/tables/candidate_slots_smoke.csv").exists()
    out = capsys.readouterr().out
    assert "USER CHECKPOINT (mandatory" in out and "candidate slots" in out


def test_main_full_run_writes_the_real_artefacts_with_no_llm(tmp_path, capsys):
    config = write_config(tmp_path)
    write_codes(tmp_path / "data/processed/schema/open_codes")
    rc = main(["--config", str(config), "--no-embeddings", "--no-llm", "--threshold", "0.5"])
    assert rc == 0
    doc = yaml.safe_load((tmp_path / "data/processed/schema/candidate_slots.yaml")
                         .read_text(encoding="utf-8"))
    assert doc["n_sitreps"] == 4 and doc["method"]["threshold"] == 0.5
    assert doc["method"]["naming_model"] == "none" and doc["smoke"] is None
    assert (tmp_path / "results/tables/candidate_slots.csv").exists()
    assert "llm:" not in capsys.readouterr().out
