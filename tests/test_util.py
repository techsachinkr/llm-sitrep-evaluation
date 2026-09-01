"""Tests for src/util.py (config/env/paths/JSON helpers)."""
from __future__ import annotations

import os
import random

import pytest

from src import util


def test_load_config_reads_repo_config():
    cfg = util.load_config()
    assert cfg["seed"] == 17
    assert "llm" in cfg and "models" in cfg
    assert cfg["llm"]["cost_limit_usd_per_run"] == 25


def test_cfg_path_resolves_relative_to_root(tmp_path):
    cfg = {"paths": {"raw": "data/raw"}}
    p = util.cfg_path(cfg, "raw", "x")
    assert p == util.ROOT / "data" / "raw"
    absolute = tmp_path / "abs"
    assert util.cfg_path({"paths": {"raw": str(absolute)}}, "raw", "x") == absolute
    assert util.cfg_path({}, "missing", "data/default") == util.ROOT / "data" / "default"


def test_env_treats_blank_as_unset(monkeypatch):
    monkeypatch.setenv("SITREP_TEST_VAR", "   ")
    assert util.env("SITREP_TEST_VAR", "dflt") == "dflt"
    monkeypatch.setenv("SITREP_TEST_VAR", " value ")
    assert util.env("SITREP_TEST_VAR") == "value"
    monkeypatch.delenv("SITREP_TEST_VAR", raising=False)
    assert util.env("SITREP_TEST_VAR") is None


def test_json_roundtrip_and_jsonl(tmp_path):
    p = tmp_path / "sub" / "x.json"
    util.write_json(p, {"a": 1, "b": [1, 2]})
    assert util.read_json(p) == {"a": 1, "b": [1, 2]}
    assert not p.with_suffix(".json.tmp").exists()

    jl = tmp_path / "rows.jsonl"
    assert list(util.read_jsonl(jl)) == []  # missing file → empty
    assert util.append_jsonl(jl, {"i": 1}) == 1
    assert util.append_jsonl(jl, [{"i": 2}, {"i": 3}]) == 2
    assert [r["i"] for r in util.read_jsonl(jl)] == [1, 2, 3]
    assert util.write_jsonl(jl, [{"i": 9}]) == 1
    assert [r["i"] for r in util.read_jsonl(jl)] == [9]


def test_sha256_is_order_independent_for_dicts():
    a = util.sha256_of({"x": 1, "y": {"b": 2, "a": 1}})
    b = util.sha256_of({"y": {"a": 1, "b": 2}, "x": 1})
    assert a == b
    assert util.sha256_of("abc") == util.sha256_of("abc")
    assert util.sha256_of("abc") != util.sha256_of({"s": "abc"})


def test_slugify():
    assert util.slugify("Nepal: Earthquake 2015 — Situation Report No. 7") == "nepal-earthquake-2015-situation-report-no-7"
    assert util.slugify("!!!") == "item"
    assert len(util.slugify("a" * 100, max_len=10)) == 10


def test_set_seed_is_reproducible():
    util.set_seed(17)
    a = [random.random() for _ in range(3)]
    util.set_seed(17)
    b = [random.random() for _ in range(3)]
    assert a == b


def test_utc_now_iso_has_timezone():
    assert util.utc_now_iso().endswith("+00:00")


def test_get_logger_singleton_handlers():
    lg1 = util.get_logger("sitrep.test")
    lg2 = util.get_logger("sitrep.test")
    assert lg1 is lg2 and len(lg1.handlers) == 1
