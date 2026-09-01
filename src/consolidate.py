"""Consolidate per-sitrep open codes into a candidate slot list (PLAN Phase 2).

Pipeline:

1. load the open codes written by `src/open_code.py`
   (`data/processed/schema/open_codes/*.json`, one file per coded sitrep,
   each `{sitrep_id?, codes: [{name, definition, evidence_quote, ...}]}`),
2. embed ``name - definition`` with sentence-transformers (all-MiniLM-L6-v2, CPU, model cached
   under `data/raw/models/`) or, when that model cannot be loaded, a deterministic TF-IDF
   lexical fallback so the pipeline always runs (which one was used is logged and recorded),
3. agglomerative clustering (average linkage, cosine distance, `--threshold`) - deterministic,
4. one LLM call per kept cluster (role `coder`, via `src.llm` only) to NAME and DEFINE the slot
   with two *synthetic* examples. Prevalence is computed from the data, never from the model:
   ``prevalence = |sitreps with >=1 code in the cluster| / |sampled sitreps|``.

Nothing here freezes a schema. The **USER CHECKPOINT is mandatory** (CLAUDE.md rule 4): this
script presents ~15-25 candidates and the user merges/renames them down to `schema.max_slots`
before `data/processed/schema.yaml` is written.

Outputs:
  data/processed/schema/candidate_slots.yaml
  data/processed/schema/consolidation_decisions.json   (audit trail: which code -> which cluster)
  results/tables/candidate_slots.csv

Evidence quotes are never read into the prompts and never leave `data/` (CLAUDE.md rule 7);
only code names and definitions are clustered and shown to the naming model.

CLI: python -m src.consolidate [--smoke [N]] [--threshold 0.35] [--no-embeddings] [--no-llm]
"""
from __future__ import annotations

import argparse
import csv
import math
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml

from src.llm import LLM
from src.util import (
    cfg_path,
    ensure_dir,
    get_logger,
    load_config,
    read_json,
    set_seed,
    sha256_of,
    utc_now_iso,
    write_json,
)

log = get_logger("sitrep.consolidate")

ST_MODEL = "all-MiniLM-L6-v2"
#: distance threshold defaults differ by representation - dense embeddings are far denser than TF-IDF
DEFAULT_THRESHOLD = {"embedding": 0.35, "lexical": 0.60}
#: how many candidates to put in front of the user at the checkpoint (PLAN: ~15-25)
DEFAULT_MAX_CANDIDATES = 25
#: at most this many distinct code names are shown to the naming model (token bound)
MAX_CODES_IN_PROMPT = 40

SLOT_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "slot_id": {"type": "string"},
        "name": {"type": "string"},
        "definition": {"type": "string"},
        "examples": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 2},
    },
    "required": ["slot_id", "name", "definition", "examples"],
    "additionalProperties": False,
}

NAMING_SYSTEM = (
    "You are a qualitative content analyst consolidating open codes from professional humanitarian "
    "situation reports (OCHA/IFRC) into information SLOTS - the recurring kinds of information a "
    "practitioner expects a sitrep to carry.\n"
    "Given one cluster of open codes, return JSON with: a short slot name (2-5 words, lower case "
    "except proper nouns), a one-or-two-sentence definition stating what must be present for the slot "
    "to count as filled, and exactly two SYNTHETIC one-line examples (invented, plausible, about a "
    "fictional disaster - never quote real text). Echo back the slot_id you were given. "
    "Name the cluster as it actually is; do not widen it to a generic catch-all."
)

_STOPWORDS = frozenset("""a an and are as at be by for from has have in into is it its of on or that the
their to with within without number numbers report reports sitrep information data""".split())


# ---------------------------------------------------------------------------
# Open codes
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OpenCode:
    """One open code from one sitrep (evidence quotes are deliberately not carried here)."""

    sitrep_id: str
    code_index: int
    name: str
    definition: str
    event: str = ""
    source: str = ""

    @property
    def text(self) -> str:
        """The string that gets embedded / vectorised."""
        return f"{self.name} - {self.definition}".strip(" -")


def load_open_codes(codes_dir: str | Path) -> tuple[list[OpenCode], list[str]]:
    """Read every open-code file. Returns (codes, sitrep_ids) with per-sitrep name de-duplication.

    Files `src/open_code.py` marked as failed (`status` != "ok": refusal, parse error, provider
    error) are skipped entirely - counting them would silently deflate every prevalence, since
    they contribute no codes to any cluster but would sit in the denominator.
    """
    codes: list[OpenCode] = []
    sitrep_ids: list[str] = []
    n_failed = 0
    for path in sorted(Path(codes_dir).glob("*.json")):
        if path.name.startswith("_"):
            continue
        try:
            payload = read_json(path)
        except (ValueError, OSError) as exc:
            log.warning("skipping unreadable open-code file %s (%s)", path, exc)
            continue
        if isinstance(payload, list):
            payload = {"codes": payload}
        if not isinstance(payload, dict):
            log.warning("skipping malformed open-code file %s", path)
            continue
        sid = str(payload.get("sitrep_id") or payload.get("id") or path.stem)
        status = payload.get("status")
        if status is not None and str(status) != "ok":
            log.warning("skipping sitrep %s: open coding ended with status=%s", sid, status)
            n_failed += 1
            continue
        event = str(payload.get("event") or "")
        source = str(payload.get("source") or "")
        raw = payload.get("codes")
        if not isinstance(raw, list):
            log.warning("skipping open-code file with no 'codes' list: %s", path)
            continue
        sitrep_ids.append(sid)
        seen: set[str] = set()
        for i, item in enumerate(raw):
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            if not name or name.lower() in seen:
                continue
            seen.add(name.lower())
            codes.append(OpenCode(sitrep_id=sid, code_index=i, name=name,
                                  definition=str(item.get("definition") or "").strip(),
                                  event=event or str(item.get("event") or ""), source=source))
    log.info("loaded %d open codes from %d sitrep(s) in %s (%d file(s) skipped as not ok)",
             len(codes), len(sitrep_ids), codes_dir, n_failed)
    return codes, sitrep_ids


# ---------------------------------------------------------------------------
# Representation
# ---------------------------------------------------------------------------
def _lex_tokens(text: str) -> list[str]:
    """Lower-cased word unigrams (stop-words dropped) plus their bigrams."""
    words = [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOPWORDS and len(w) > 1]
    return words + [f"{a}_{b}" for a, b in zip(words, words[1:])]


def tfidf_matrix(texts: Sequence[str]) -> np.ndarray:
    """Deterministic L2-normalised TF-IDF matrix (sublinear tf, smoothed idf). No sklearn needed."""
    docs = [_lex_tokens(t) for t in texts]
    df: Counter = Counter(tok for d in docs for tok in set(d))
    vocab = {tok: i for i, tok in enumerate(sorted(df))}
    x = np.zeros((len(docs), len(vocab)), dtype=float)
    n = len(docs)
    for i, doc in enumerate(docs):
        for tok, count in Counter(doc).items():
            x[i, vocab[tok]] = (1.0 + math.log(count)) * (math.log((1.0 + n) / (1.0 + df[tok])) + 1.0)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def _load_sentence_transformer(model_name: str, cache_folder: str | Path | None):  # pragma: no cover - I/O
    """Import and load the sentence-transformers model (patched out in tests)."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, cache_folder=str(cache_folder) if cache_folder else None)


def embed_texts(texts: Sequence[str], *, use_embeddings: bool = True, model_name: str = ST_MODEL,
                model_cache: str | Path | None = None,
                vector_cache: str | Path | None = None) -> tuple[np.ndarray, str]:
    """Vectorise `texts`; returns (matrix, method).

    Tries sentence-transformers first and falls back to the deterministic TF-IDF representation
    on ANY failure (package missing, no network for the first download, corrupt cache), so the
    pipeline still runs offline. `method` is recorded in every output artefact.
    """
    if not texts:
        return np.zeros((0, 1), dtype=float), "lexical-tfidf"
    if not use_embeddings:
        return tfidf_matrix(texts), "lexical-tfidf"
    key = sha256_of([model_name, list(texts)])[:32]
    npy = Path(vector_cache) / f"{key}.npy" if vector_cache else None
    if npy is not None and npy.exists():
        try:
            log.info("embeddings: reusing cached vectors %s", npy)
            return np.load(npy), f"sentence-transformers:{model_name}"
        except (OSError, ValueError) as exc:
            log.warning("could not read embedding cache %s (%s); recomputing", npy, exc)
    try:
        model = _load_sentence_transformer(model_name, model_cache)
        vecs = np.asarray(model.encode(list(texts), batch_size=32, show_progress_bar=False,
                                       normalize_embeddings=True, convert_to_numpy=True), dtype=float)
        if vecs.ndim != 2 or vecs.shape[0] != len(texts):
            raise ValueError(f"unexpected embedding shape {getattr(vecs, 'shape', None)}")
    except Exception as exc:  # noqa: BLE001 - any failure must degrade, never crash the phase
        log.warning("sentence-transformers unavailable (%s: %s); falling back to TF-IDF lexical clustering",
                    type(exc).__name__, exc)
        return tfidf_matrix(texts), "lexical-tfidf"
    if npy is not None:
        ensure_dir(npy.parent)
        np.save(npy, vecs)
    return vecs, f"sentence-transformers:{model_name}"


def cosine_distance_matrix(x: np.ndarray) -> np.ndarray:
    """Symmetric cosine distances in [0, 2]; all-zero rows sit at distance 1 from everything."""
    mat = np.asarray(x, dtype=float)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = mat / norms
    dist = 1.0 - unit @ unit.T
    dist = (dist + dist.T) / 2.0
    np.fill_diagonal(dist, 0.0)
    return np.clip(dist, 0.0, 2.0)


def cluster_vectors(x: np.ndarray, threshold: float) -> np.ndarray:
    """Agglomerative (average linkage, cosine) cluster labels. Deterministic - no randomness."""
    n = len(x)
    if n == 0:
        return np.zeros(0, dtype=int)
    if n == 1:
        return np.zeros(1, dtype=int)
    from sklearn.cluster import AgglomerativeClustering

    model = AgglomerativeClustering(n_clusters=None, distance_threshold=float(threshold),
                                    metric="precomputed", linkage="average")
    return np.asarray(model.fit_predict(cosine_distance_matrix(x)), dtype=int)


# ---------------------------------------------------------------------------
# Clusters
# ---------------------------------------------------------------------------
@dataclass
class Cluster:
    """A group of open codes that the clustering step judged to be the same information slot."""

    members: list[OpenCode] = field(default_factory=list)
    slot_id: str = ""
    name: str = ""
    definition: str = ""
    examples: list[str] = field(default_factory=list)
    named_by: str = "pending"
    kept: bool = True

    @property
    def sitrep_ids(self) -> list[str]:
        return sorted({c.sitrep_id for c in self.members})

    @property
    def name_counts(self) -> list[tuple[str, int]]:
        """Distinct code names, most frequent first (ties alphabetical)."""
        counts = Counter(c.name.strip().lower() for c in self.members)
        return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))

    @property
    def canonical_name(self) -> str:
        return self.name_counts[0][0] if self.members else ""

    def prevalence(self, n_sitreps: int) -> float:
        return round(len(self.sitrep_ids) / n_sitreps, 4) if n_sitreps else 0.0


def build_clusters(codes: Sequence[OpenCode], labels: Sequence[int]) -> list[Cluster]:
    """Group codes by label and order clusters deterministically (breadth, then size, then name)."""
    by_label: dict[int, Cluster] = {}
    for code, label in zip(codes, labels):
        by_label.setdefault(int(label), Cluster()).members.append(code)
    clusters = sorted(by_label.values(),
                      key=lambda c: (-len(c.sitrep_ids), -len(c.members), c.canonical_name))
    for i, cluster in enumerate(clusters, start=1):
        cluster.slot_id = f"S{i:02d}"
    return clusters


def _fallback_naming(cluster: Cluster) -> None:
    """Deterministic name/definition when the model is unavailable or its answer is unusable."""
    names = [n for n, _ in cluster.name_counts[:5]]
    cluster.name = cluster.canonical_name or "unnamed slot"
    cluster.definition = (f"Consolidated from {len(cluster.members)} open code(s) across "
                          f"{len(cluster.sitrep_ids)} sitrep(s); most frequent codes: {', '.join(names)}.")
    cluster.examples = []
    cluster.named_by = "fallback"


def naming_requests(clusters: Sequence[Cluster], n_sitreps: int, model: str = "coder",
                    max_tokens: int = 900) -> list[dict[str, Any]]:
    """One `src.llm` request per cluster. `meta` carries ids/labels only - never any text."""
    requests: list[dict[str, Any]] = []
    for cluster in clusters:
        lines = [f"- {name} (x{count})" for name, count in cluster.name_counts[:MAX_CODES_IN_PROMPT]]
        defs = [c.definition for c in cluster.members if c.definition][:5]
        user = (f"Cluster {cluster.slot_id}: {len(cluster.members)} open codes from "
                f"{len(cluster.sitrep_ids)} of {n_sitreps} sampled situation reports "
                f"(prevalence {cluster.prevalence(n_sitreps):.2f}).\n\n"
                f"Code names:\n" + "\n".join(lines) +
                ("\n\nSample code definitions:\n" + "\n".join(f"- {d}" for d in defs) if defs else "") +
                f"\n\nReturn the JSON object for this slot, with slot_id = \"{cluster.slot_id}\".")
        requests.append({"model": model, "system": NAMING_SYSTEM, "user": user, "max_tokens": max_tokens,
                         "json_schema": SLOT_JSON_SCHEMA, "effort": "low", "tag": "consolidate-name-v1",
                         "meta": {"phase": "consolidate", "slot_id": cluster.slot_id,
                                  "n_codes": len(cluster.members), "n_sitreps": len(cluster.sitrep_ids)}})
    return requests


def apply_naming(clusters: Sequence[Cluster], responses: Sequence[Any]) -> int:
    """Copy validated model output onto the clusters; returns how many were named by the model."""
    named = 0
    for cluster, resp in zip(clusters, responses):
        payload = getattr(resp, "json", None)
        ok = getattr(resp, "ok", False)
        if not ok or not isinstance(payload, dict):
            log.warning("slot %s: naming failed (%s); using the deterministic fallback name",
                        cluster.slot_id, getattr(resp, "error", None) or getattr(resp, "json_error", None)
                        or "no JSON object")
            _fallback_naming(cluster)
            continue
        # Field-name tolerance: providers without strict json_schema support (DeepSeek falls back
        # to plain json_object mode) are not bound to the schema's key names, and return e.g.
        # `slot_name` / `synthetic_examples`. Accept the obvious aliases rather than discarding
        # a perfectly good slot definition.
        def _first(*keys: str) -> Any:
            for k in keys:
                if payload.get(k):
                    return payload[k]
            return None

        name = str(_first("name", "slot_name", "slot", "title") or "").strip()
        definition = str(_first("definition", "slot_definition", "description") or "").strip()
        raw_examples = _first("examples", "synthetic_examples", "example_sentences", "samples") or []
        if isinstance(raw_examples, str):
            raw_examples = [raw_examples]
        examples = [str(e).strip() for e in raw_examples if str(e).strip()]
        if not name or not definition:
            log.warning("slot %s: model returned an incomplete slot; using the fallback name", cluster.slot_id)
            _fallback_naming(cluster)
            continue
        cluster.name, cluster.definition, cluster.examples = name, definition, examples[:2]
        cluster.named_by = "llm"
        named += 1
    return named


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------
def _slot_dict(cluster: Cluster, n_sitreps: int) -> dict[str, Any]:
    return {
        "slot_id": cluster.slot_id,
        "name": cluster.name,
        "definition": cluster.definition,
        "examples": list(cluster.examples),
        "prevalence": cluster.prevalence(n_sitreps),
        "n_sitreps": len(cluster.sitrep_ids),
        "n_codes": len(cluster.members),
        "named_by": cluster.named_by,
        "top_codes": [name for name, _ in cluster.name_counts[:5]],
    }


def _decision_dict(cluster: Cluster, n_sitreps: int) -> dict[str, Any]:
    return {
        "slot_id": cluster.slot_id,
        "name": cluster.name,
        "kept": cluster.kept,
        "prevalence": cluster.prevalence(n_sitreps),
        "n_codes": len(cluster.members),
        "sitrep_ids": cluster.sitrep_ids,
        "members": [{"sitrep_id": c.sitrep_id, "code_index": c.code_index, "name": c.name,
                     "definition": c.definition} for c in
                    sorted(cluster.members, key=lambda c: (c.sitrep_id, c.code_index))],
    }


CSV_COLUMNS = ["slot_id", "name", "prevalence", "n_sitreps", "n_codes", "named_by",
               "definition", "example_1", "example_2", "top_codes"]


def write_outputs(clusters: Sequence[Cluster], meta: dict[str, Any], out_dir: str | Path,
                  tables_dir: str | Path, suffix: str = "") -> dict[str, Path]:
    """Write candidate_slots.yaml, consolidation_decisions.json and candidate_slots.csv."""
    n_sitreps = int(meta["n_sitreps"])
    kept = [c for c in clusters if c.kept]
    schema_dir = ensure_dir(out_dir)
    table_dir = ensure_dir(tables_dir)

    yaml_path = schema_dir / f"candidate_slots{suffix}.yaml"
    doc = dict(meta)
    doc["slots"] = [_slot_dict(c, n_sitreps) for c in kept]
    with open(yaml_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(doc, fh, sort_keys=False, allow_unicode=True, width=100)

    json_path = schema_dir / f"consolidation_decisions{suffix}.json"
    write_json(json_path, {
        **meta,
        "clusters": [_decision_dict(c, n_sitreps) for c in kept],
        "dropped_clusters": [_decision_dict(c, n_sitreps) for c in clusters if not c.kept],
    })

    csv_path = table_dir / f"candidate_slots{suffix}.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for cluster in kept:
            ex = list(cluster.examples) + ["", ""]
            writer.writerow({"slot_id": cluster.slot_id, "name": cluster.name,
                             "prevalence": cluster.prevalence(n_sitreps),
                             "n_sitreps": len(cluster.sitrep_ids), "n_codes": len(cluster.members),
                             "named_by": cluster.named_by, "definition": cluster.definition,
                             "example_1": ex[0], "example_2": ex[1],
                             "top_codes": "; ".join(n for n, _ in cluster.name_counts[:5])})
    return {"yaml": yaml_path, "decisions": json_path, "csv": csv_path}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def consolidate(codes: Sequence[OpenCode], sitrep_ids: Sequence[str], *, threshold: float | None = None,
                use_embeddings: bool = True, max_candidates: int = DEFAULT_MAX_CANDIDATES,
                min_codes: int = 1, llm: LLM | None = None, model: str = "coder", batch: bool = False,
                model_cache: str | Path | None = None, vector_cache: str | Path | None = None,
                max_slots: int | None = None) -> tuple[list[Cluster], dict[str, Any]]:
    """Cluster `codes`, name the kept clusters, and return (clusters, run metadata)."""
    vectors, method = embed_texts([c.text for c in codes], use_embeddings=use_embeddings,
                                  model_cache=model_cache, vector_cache=vector_cache)
    if threshold is None:
        threshold = DEFAULT_THRESHOLD["lexical" if method.startswith("lexical") else "embedding"]
    labels = cluster_vectors(vectors, threshold)
    clusters = build_clusters(codes, labels)
    n_sitreps = len(set(sitrep_ids))
    n_kept = 0
    for cluster in clusters:  # clusters are already ordered breadth-first, so this keeps the top ones
        cluster.kept = n_kept < max_candidates and len(cluster.members) >= min_codes
        n_kept += int(cluster.kept)
    kept = [c for c in clusters if c.kept]

    if llm is None:
        log.warning("no LLM given: naming every cluster with the deterministic fallback")
        for cluster in kept:
            _fallback_naming(cluster)
        n_named = 0
    else:
        requests = naming_requests(kept, n_sitreps, model=model)
        llm.gate(llm.estimate_cost(requests, batch=batch))
        n_named = apply_naming(kept, llm.run(requests, batch=batch))

    meta = {
        "generated_at": utc_now_iso(),
        "method": {"representation": method, "clustering": "agglomerative/average/cosine",
                   "threshold": round(float(threshold), 4), "naming_model": model if llm else "none"},
        "n_sitreps": n_sitreps,
        "n_codes": len(codes),
        "n_clusters": len(clusters),
        "n_candidates": len(kept),
        "n_named_by_model": n_named,
        "max_slots": max_slots,
        "sitrep_ids": sorted(set(sitrep_ids)),
        "user_checkpoint": ("PENDING - candidates only. The user must merge/rename/approve these into "
                            "data/processed/schema.yaml (CLAUDE.md rule 4)."),
    }
    return clusters, meta


NO_CODES_MESSAGE = """
No open codes found - consolidation needs the Phase 2 open-coding pass first.

Expected: data/processed/schema/open_codes/*.json
          (one file per coded sitrep: {sitrep_id, codes:[{name, definition, evidence_quote}]})

Produce them with `python -m src.open_code` once the human sitrep corpus exists at
data/processed/sitreps_human/<event>/<date>.json (PLAN 2a / Phase 1), or point this script at
another directory with --codes-dir.
"""


def _print_checkpoint(clusters: Sequence[Cluster], meta: dict[str, Any], paths: dict[str, Path]) -> None:
    kept = [c for c in clusters if c.kept]
    n_sitreps = int(meta["n_sitreps"])
    print(f"\n{meta['n_codes']} open codes from {n_sitreps} sitrep(s) -> {meta['n_clusters']} clusters "
          f"-> {len(kept)} candidate slots")
    print(f"representation: {meta['method']['representation']}   "
          f"threshold: {meta['method']['threshold']}   naming: {meta['method']['naming_model']} "
          f"({meta['n_named_by_model']}/{len(kept)} named by the model)\n")
    print(f"{'slot':<5} {'prev':>5} {'#sitrep':>7} {'#codes':>6}  name")
    print("-" * 92)
    for cluster in kept:
        print(f"{cluster.slot_id:<5} {cluster.prevalence(n_sitreps):>5.2f} {len(cluster.sitrep_ids):>7} "
              f"{len(cluster.members):>6}  {cluster.name[:60]}")
    print("\n" + "=" * 92)
    print("USER CHECKPOINT (mandatory, CLAUDE.md rule 4) - these are CANDIDATES, not a schema.")
    print("Read the definitions/examples below, then merge, rename and approve them down to "
          f"<= {meta.get('max_slots')} slots; only then is data/processed/schema.yaml written.")
    print("=" * 92 + "\n")
    for cluster in kept:
        print(f"[{cluster.slot_id}] {cluster.name}  (prevalence {cluster.prevalence(n_sitreps):.2f}, "
              f"{len(cluster.members)} codes, named_by={cluster.named_by})")
        print(f"    {cluster.definition}")
        for ex in cluster.examples:
            print(f"    e.g. {ex}")
        print(f"    top codes: {', '.join(n for n, _ in cluster.name_counts[:5])}\n")
    for key in ("yaml", "decisions", "csv"):
        print(f"wrote {paths[key]}")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description="Cluster open codes into candidate slots (PLAN Phase 2)")
    ap.add_argument("--config", type=Path, default=None, help="config.yaml to use")
    ap.add_argument("--codes-dir", type=Path, default=None, help="directory of open-code JSON files")
    ap.add_argument("--out-dir", type=Path, default=None, help="where the schema artefacts go")
    ap.add_argument("--tables-dir", type=Path, default=None, help="where candidate_slots.csv goes")
    ap.add_argument("--threshold", type=float, default=None,
                    help="cosine distance threshold (default 0.35 embeddings / 0.60 lexical)")
    ap.add_argument("--max-candidates", type=int, default=DEFAULT_MAX_CANDIDATES,
                    help=f"candidates to present at the checkpoint (default {DEFAULT_MAX_CANDIDATES})")
    ap.add_argument("--min-codes", type=int, default=1, help="drop clusters with fewer codes than this")
    ap.add_argument("--model", default="coder", help="config role used to name slots (default coder)")
    ap.add_argument("--no-embeddings", action="store_true", help="force the TF-IDF lexical fallback")
    ap.add_argument("--no-llm", action="store_true", help="skip slot naming (deterministic names only)")
    ap.add_argument("--batch", dest="batch", action="store_true", default=None,
                    help="force the Message Batches API (default: on for full runs, off for --smoke)")
    ap.add_argument("--no-batch", dest="batch", action="store_false", help="force synchronous calls")
    ap.add_argument("--smoke", nargs="?", type=int, const=5, default=None,
                    help="consolidate only N sampled open-code files, synchronously (default 5)")
    args = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):  # pragma: no cover
        pass

    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 17))
    set_seed(seed)
    processed = cfg_path(cfg, "processed", "data/processed")
    schema_dir = args.out_dir or (processed / "schema")
    codes_dir = args.codes_dir or (processed / "schema" / "open_codes")
    tables_dir = args.tables_dir or (cfg_path(cfg, "results", "results") / "tables")
    smoke = args.smoke is not None

    codes, sitrep_ids = load_open_codes(codes_dir)
    if not codes:
        log.warning("no open codes under %s", codes_dir)
        print(NO_CODES_MESSAGE)
        return 2
    if smoke:
        keep = set(random.Random(seed).sample(sitrep_ids, min(max(1, args.smoke), len(sitrep_ids))))
        codes = [c for c in codes if c.sitrep_id in keep]
        sitrep_ids = [s for s in sitrep_ids if s in keep]
        log.info("smoke: %d codes from %d sitrep(s)", len(codes), len(sitrep_ids))

    llm = None if args.no_llm else LLM(config_path=args.config)
    batch = (not smoke) if args.batch is None else bool(args.batch)
    clusters, meta = consolidate(
        codes, sitrep_ids, threshold=args.threshold, use_embeddings=not args.no_embeddings,
        max_candidates=args.max_candidates, min_codes=args.min_codes, llm=llm, model=args.model,
        batch=batch, model_cache=cfg_path(cfg, "raw", "data/raw") / "models",
        vector_cache=Path(schema_dir) / "embedding_cache",
        max_slots=int((cfg.get("schema") or {}).get("max_slots", 15)),
    )
    meta["smoke"] = args.smoke if smoke else None
    paths = write_outputs(clusters, meta, schema_dir, tables_dir, suffix="_smoke" if smoke else "")
    _print_checkpoint(clusters, meta, paths)
    if llm is not None:
        print(f"llm: {llm.summary()}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
