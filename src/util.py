"""Shared utilities: config, env, paths, seeds, JSON/JSONL I/O, hashing, logging.

Every script in this project should import from here rather than re-implementing
config loading or file helpers, so behaviour (seeds, paths) stays uniform.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = ROOT / "config.yaml"

_ENV_LOADED = False


# ---------------------------------------------------------------------------
# Environment / config
# ---------------------------------------------------------------------------
def load_env(dotenv_path: Path | None = None) -> None:
    """Load `.env` from the repo root once (existing OS vars win)."""
    global _ENV_LOADED
    if _ENV_LOADED and dotenv_path is None:
        return
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - dotenv is in requirements
        _ENV_LOADED = True
        return
    load_dotenv(dotenv_path or (ROOT / ".env"), override=False)
    _ENV_LOADED = True


def env(name: str, default: str | None = None) -> str | None:
    """Return an env var, treating empty/whitespace values as unset."""
    load_env()
    val = os.environ.get(name)
    if val is None or not val.strip():
        return default
    return val.strip()


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load config.yaml (or a given YAML file) into a dict."""
    p = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(p, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Config at {p} is not a mapping")
    return cfg


def cfg_path(cfg: dict[str, Any], key: str, default: str) -> Path:
    """Resolve a path from cfg['paths'][key] (relative to repo root)."""
    raw = (cfg.get("paths") or {}).get(key, default)
    p = Path(raw)
    return p if p.is_absolute() else ROOT / p


def set_seed(seed: int) -> None:
    """Fix random seeds (stdlib, numpy if present, torch if present)."""
    random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover
        pass
    try:  # pragma: no cover - torch optional
        import torch

        torch.manual_seed(seed)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Filesystem / JSON helpers
# ---------------------------------------------------------------------------
def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: str | Path, obj: Any, indent: int = 2) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=indent, sort_keys=False)
    os.replace(tmp, p)
    return p


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return iter(())
    return _iter_jsonl(p)


def _iter_jsonl(p: Path) -> Iterator[dict[str, Any]]:
    with open(p, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def append_jsonl(path: str | Path, rows: Iterable[dict[str, Any]] | dict[str, Any]) -> int:
    """Append one row or many rows; returns number of rows written."""
    p = Path(path)
    ensure_dir(p.parent)
    if isinstance(rows, dict):
        rows = [rows]
    n = 0
    with open(p, "a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            n += 1
    return n


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    p = Path(path)
    ensure_dir(p.parent)
    n = 0
    with open(p, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            n += 1
    return n


def stable_json(obj: Any) -> str:
    """Deterministic JSON serialisation (sorted keys, no whitespace)."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def sha256_of(obj: Any) -> str:
    """SHA-256 of the stable JSON form of `obj` (or of a str as-is)."""
    data = obj if isinstance(obj, str) else stable_json(obj)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalize_event(event: str | None) -> str:
    """Canonical form of an event key, for comparing keys that came from different sources.

    The same event is spelled two ways in this corpus: HumAID stream directories use underscores
    (`cyclone_idai_2019`) and the manual ReliefWeb folders use hyphens (`cyclone-idai-2019`).
    Records inherit whichever spelling their directory had, so a raw `a.event == b.event` between
    a machine sitrep and a human sitrep is always False — silently, with no error.
    """
    return (event or "").strip().lower().replace("_", "-")


def resolve_event_dir(base: str | Path, event: str) -> Path | None:
    """Find `<base>/<event>` tolerating hyphen/underscore differences in the event key.

    The corpus carries two spellings of the same event: HumAID stream configs use underscores
    (`cyclone_idai_2019`) while the manual ReliefWeb folder uses hyphens (`cyclone-idai-2019`).
    Loaders must accept either, otherwise a phase silently finds no data and downstream analysis
    is computed on a subset without anyone noticing.

    Returns the matching directory, or None when neither spelling exists.
    """
    base = Path(base)
    for name in (event, event.replace("-", "_"), event.replace("_", "-")):
        cand = base / name
        if cand.is_dir():
            return cand
    return None


def slugify(text: str, max_len: int = 60) -> str:
    """Filesystem/ID-safe slug: lowercase, [a-z0-9-], collapsed dashes."""
    out = []
    prev_dash = False
    for ch in text.lower():
        if ch.isalnum():
            out.append(ch)
            prev_dash = False
        elif not prev_dash:
            out.append("-")
            prev_dash = True
    slug = "".join(out).strip("-")
    return slug[:max_len].rstrip("-") or "item"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def get_logger(name: str = "sitrep") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(os.environ.get("SITREP_LOG_LEVEL", "INFO"))
        logger.propagate = False
    return logger
