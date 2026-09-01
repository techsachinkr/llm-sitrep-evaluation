"""Offline tests for LLM.complete_batch (FakeProvider batch simulation)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.llm import LLM, CostLimitExceeded, FakeProvider, TransientProviderError
from src.util import read_jsonl
from tests.test_llm import make_cfg, make_llm  # noqa: F401  (shared helpers + autouse env isolation)


def _reqs(n: int, **kw):
    return [{"model": "coder", "system": "S" * 35, "user": f"user {i} " + "x" * 30, "meta": {"i": i},
             "tag": "b", **kw} for i in range(n)]


def test_batch_submits_once_dedupes_and_records_discounted_cost(tmp_path):
    prov = FakeProvider()
    llm, _ = make_llm(tmp_path, provider=prov)
    reqs = _reqs(4) + [dict(_reqs(1)[0], meta={"i": 99})]  # 5 requests, 4 unique
    out = llm.complete_batch(reqs, poll_interval_s=0)
    assert [r.meta["i"] for r in out] == [0, 1, 2, 3, 99]
    assert all(r.ok for r in out)
    assert prov.batch_submissions == 1 and prov.batch_items == 4  # one batch, duplicates collapsed
    assert prov.calls == 4  # the fake serves batch items through call(); no sync calls happened
    assert out[0].mode == "batch" and out[0].batch_id and not out[0].cached
    assert out[4].mode == "cache" and out[4].cached and out[4].request_hash == out[0].request_hash
    # cost = 50% of the sync price (fake: $1/$5 per MTok; tokens = ceil(chars/3.5))
    from src.llm import estimate_tokens
    n_in = estimate_tokens(reqs[0]["system"]) + estimate_tokens(reqs[0]["user"])
    n_out = estimate_tokens(out[0].text)
    sync_cost = (n_in * 1.0 + n_out * 5.0) / 1e6
    assert out[0].cost_usd == pytest.approx(sync_cost * 0.5)
    assert llm.run_cost_usd == pytest.approx(4 * sync_cost * 0.5)
    rows = list(read_jsonl(tmp_path / "log.jsonl"))
    assert len(rows) == 5
    assert {r["mode"] for r in rows} == {"batch", "cache"} and all(r["batch_id"] for r in rows)
    # results are cached: a later sync call is a cache hit
    r2 = llm.complete(**{k: v for k, v in reqs[0].items() if k != "meta"})
    assert r2.cached and prov.calls == 4
    # manifest closed
    mf = list((tmp_path / "cache" / "batches").glob("*.json"))
    assert len(mf) == 1 and json.loads(mf[0].read_text(encoding="utf-8"))["status"] == "done"


def test_batch_serves_cache_hits_without_submitting(tmp_path):
    prov = FakeProvider()
    llm, _ = make_llm(tmp_path, provider=prov)
    reqs = _reqs(3)
    llm.complete(**{k: v for k, v in reqs[0].items()})  # warm one entry via sync
    out = llm.complete_batch(reqs, poll_interval_s=0)
    assert out[0].cached and out[0].mode == "cache"
    assert prov.batch_items == 2 and prov.batch_submissions == 1
    # everything cached now -> no submission at all
    out2 = llm.complete_batch(reqs, poll_interval_s=0)
    assert all(r.cached for r in out2) and prov.batch_submissions == 1


def test_batch_cost_gate_before_submission(tmp_path):
    prov = FakeProvider()
    llm, _ = make_llm(tmp_path, provider=prov, cost_limit_usd_per_run=0.0001)
    with pytest.raises(CostLimitExceeded):
        llm.complete_batch(_reqs(3, max_tokens=4000), poll_interval_s=0)
    assert prov.batch_submissions == 0
    # the same requests fit when priced at the batch discount but not at sync price
    est_sync = llm.estimate_cost(_reqs(3, max_tokens=4000))
    est_batch = llm.estimate_cost(_reqs(3, max_tokens=4000), batch=True)
    assert est_batch.usd == pytest.approx(est_sync.usd * 0.5, abs=2e-6)  # CostEstimate.usd is rounded to 6 dp
    llm2, _ = make_llm(tmp_path / "b", provider=FakeProvider(), cost_limit_usd_per_run=est_batch.usd * 1.2)
    assert all(r.ok for r in llm2.complete_batch(_reqs(3, max_tokens=4000), poll_interval_s=0))
    fresh = [dict(r, user=r["user"] + " v2") for r in _reqs(3, max_tokens=4000)]  # uncached
    with pytest.raises(CostLimitExceeded):
        llm2.gate(llm2.estimate_cost(fresh))  # whole-set projection at sync price exceeds the limit ...
    llm2.gate(llm2.estimate_cost(fresh, batch=True))  # ... but fits at batch price
    assert all(r.ok for r in llm2.complete_batch(fresh, poll_interval_s=0))


def test_batch_polls_until_ended_and_reports_errored_items(tmp_path):
    reqs = _reqs(3)
    llm0, _ = make_llm(tmp_path)
    keys = [llm0._prepare(r)[3] for r in reqs]
    prov = FakeProvider(batch_polls_until_done=2, batch_fail_ids={keys[1]})
    llm, _ = make_llm(tmp_path, provider=prov)
    out = llm.complete_batch(reqs, poll_interval_s=0)
    assert prov.batches[out[0].batch_id]["polls"] == 3  # 2 in_progress + 1 ended
    assert out[0].ok and out[2].ok
    assert not out[1].ok and out[1].error.startswith("batch errored") and out[1].mode == "batch"
    rows = list(read_jsonl(tmp_path / "log.jsonl"))
    assert sum(1 for r in rows if r["error"]) == 1
    # errored item is not cached, so a re-run resubmits just that one
    prov.batch_fail_ids.clear()
    out2 = llm.complete_batch(reqs, poll_interval_s=0)
    assert all(r.ok for r in out2) and prov.batch_submissions == 2 and prov.batch_items == 4


def test_batch_resume_reattaches_to_open_manifest(tmp_path):
    prov = FakeProvider(batch_polls_until_done=1)
    llm, _ = make_llm(tmp_path, provider=prov)
    reqs = _reqs(2)
    pending = llm.complete_batch(reqs, wait=False)
    assert all((not r.ok) and r.error.startswith("batch pending") for r in pending)
    assert prov.batch_submissions == 1
    mf = list((tmp_path / "cache" / "batches").glob("*.json"))
    assert len(mf) == 1 and json.loads(mf[0].read_text(encoding="utf-8"))["status"] == "submitted"
    # a fresh LLM instance (new process) with the same requests re-attaches instead of resubmitting
    llm2, _ = make_llm(tmp_path, provider=prov)
    out = llm2.complete_batch(reqs, poll_interval_s=0)
    assert all(r.ok for r in out) and prov.batch_submissions == 1
    assert json.loads(mf[0].read_text(encoding="utf-8"))["status"] == "done"
    assert llm2.run_cost_usd > 0


def test_batch_falls_back_to_sync_for_unsupported_provider(tmp_path):
    prov = FakeProvider()
    prov.supports_batch = False
    llm, _ = make_llm(tmp_path, provider=prov)
    out = llm.complete_batch(_reqs(2), poll_interval_s=0)
    assert all(r.ok and r.mode == "sync" for r in out) and prov.batch_submissions == 0 and prov.calls == 2


def test_batch_chunking(tmp_path):
    prov = FakeProvider()
    llm, _ = make_llm(tmp_path, provider=prov, llm_cfg={"batch": {"chunk_size": 2}})
    out = llm.complete_batch(_reqs(5), poll_interval_s=0)
    assert all(r.ok for r in out) and prov.batch_submissions == 3
    assert len({r.batch_id for r in out}) == 3


def test_run_dispatch_and_cli_batch_smoke(tmp_path, monkeypatch, capsys):
    from src.llm import main

    prov = FakeProvider()
    llm, _ = make_llm(tmp_path, provider=prov)
    assert llm.run(_reqs(1), batch=True, poll_interval_s=0)[0].mode == "batch"
    assert llm.run(_reqs(1, tag="s"), batch=False, progress=False)[0].mode == "cache"  # same key → cached
    monkeypatch.setenv("LLM_FAKE", "1")
    monkeypatch.chdir(tmp_path)  # keep the smoke's cache/log out of the repo? (paths are absolute → use env)
    monkeypatch.setattr(LLM, "_abs", staticmethod(lambda p: tmp_path / Path(p).name))
    assert main(["--smoke", "--batch", "--poll", "0"]) == 0
    printed = capsys.readouterr().out
    assert "SMOKE OK" in printed and '"mode": "batch"' in printed


# ---------------------------------------------------------------------------
# additions from the batch review
# ---------------------------------------------------------------------------
def _committed_matches(llm):
    """The gate's committed figure must equal actually-booked spend once nothing is in flight."""
    return llm.committed_usd() == pytest.approx(llm.run_cost_usd + llm.run_assumed_usd)


def test_wait_false_then_collect_does_not_double_count_or_log_pending(tmp_path):
    prov = FakeProvider()
    llm, _ = make_llm(tmp_path, provider=prov)
    reqs = _reqs(3)
    pending = llm.complete_batch(reqs, wait=False)
    assert all(r.mode == "pending" and not r.ok for r in pending)
    assert llm.committed_usd() == 0.0  # reservation released on the wait=False return
    assert list(read_jsonl(tmp_path / "log.jsonl")) == []  # placeholders are not calls
    out = llm.complete_batch(reqs, poll_interval_s=0)  # same process re-attaches
    assert all(r.ok for r in out) and prov.batch_submissions == 1
    assert _committed_matches(llm) and llm.run_cost_usd > 0
    rows = list(read_jsonl(tmp_path / "log.jsonl"))
    assert len(rows) == 3 and all(r["mode"] == "batch" for r in rows)
    assert rows[-1]["run_cost_usd_after"] == pytest.approx(llm.run_cost_usd, abs=1e-6)  # booked before logging
    assert all(r["latency_s"] >= 0 for r in rows)


def test_partial_collection_crash_resumes_without_resubmitting(tmp_path):
    reqs = _reqs(4)
    llm0, _ = make_llm(tmp_path)
    keys = [llm0._prepare(r, batch=True)[3] for r in reqs]
    prov = FakeProvider(batch_results_fail_after=2)  # streams 2 results, then dies
    llm, _ = make_llm(tmp_path, provider=prov)
    with pytest.raises(TransientProviderError):
        llm.complete_batch(reqs, poll_interval_s=0)
    assert llm.committed_usd() == pytest.approx(llm.run_cost_usd)  # reservation released on the way out
    assert llm.run_cost_usd > 0  # the two collected items were booked
    mpath = tmp_path / "cache" / "batches" / f"{prov.batch_submissions and list(prov.batches)[0]}.json"
    assert json.loads(mpath.read_text(encoding="utf-8"))["status"] == "submitted"  # still open
    prov.batch_results_fail_after = None
    llm2, _ = make_llm(tmp_path, provider=prov)  # fresh instance == new process
    out = llm2.complete_batch(reqs, poll_interval_s=0)
    assert all(r.ok for r in out)
    assert prov.batch_submissions == 1  # nothing resubmitted: 2 from cache, 2 re-attached
    assert json.loads(mpath.read_text(encoding="utf-8"))["status"] == "done"
    assert sum(1 for r in out if r.cached) == 2 and sum(1 for r in out if r.mode == "batch") == 2


def test_unavailable_batch_is_reported_and_resubmitted(tmp_path):
    prov = FakeProvider()
    llm, _ = make_llm(tmp_path, provider=prov)
    reqs = _reqs(2)
    llm.complete_batch(reqs, wait=False)
    bid = list(prov.batches)[0]
    prov.batch_unavailable_ids.add(bid)  # e.g. archived / past the 29-day retention
    out = llm.complete_batch(reqs, poll_interval_s=0)
    assert all((not r.ok) and "unavailable" in r.error for r in out)
    assert _committed_matches(llm)
    mpath = tmp_path / "cache" / "batches" / f"{bid}.json"
    assert json.loads(mpath.read_text(encoding="utf-8"))["status"] == "failed"
    out2 = llm.complete_batch(reqs, poll_interval_s=0)  # dead manifest no longer re-attached
    assert all(r.ok for r in out2) and prov.batch_submissions == 2


def test_canceled_expired_and_missing_items(tmp_path):
    reqs = _reqs(4)
    llm0, _ = make_llm(tmp_path)
    keys = [llm0._prepare(r, batch=True)[3] for r in reqs]
    prov = FakeProvider(batch_result_kinds={keys[0]: "canceled", keys[1]: "expired"},
                        batch_missing_ids={keys[2]})
    llm, _ = make_llm(tmp_path, provider=prov)
    out = llm.complete_batch(reqs, poll_interval_s=0)
    assert out[0].error.endswith("canceled") and out[1].error.endswith("expired")
    assert "missing" in out[2].error and out[3].ok
    assert _committed_matches(llm)
    rows = [r for r in read_jsonl(tmp_path / "log.jsonl") if r["error"]]
    assert len(rows) == 3


def test_timeout_keeps_manifest_open_and_releases_reservation(tmp_path):
    prov = FakeProvider(batch_polls_until_done=99)
    llm, _ = make_llm(tmp_path, provider=prov)
    with pytest.raises(TimeoutError):
        llm.complete_batch(_reqs(2), poll_interval_s=0, max_wait_s=0)
    assert llm.committed_usd() == 0.0
    mf = list((tmp_path / "cache" / "batches").glob("*.json"))
    assert len(mf) == 1 and json.loads(mf[0].read_text(encoding="utf-8"))["status"] == "submitted"
    prov.batch_polls_until_done = 0
    out = llm.complete_batch(_reqs(2), poll_interval_s=0)  # resumes the same batch
    assert all(r.ok for r in out) and prov.batch_submissions == 1


def test_refusal_and_truncation_in_batch(tmp_path):
    prov = FakeProvider(responder=lambda req: "x" * 40)
    llm, _ = make_llm(tmp_path, provider=prov)
    refused = llm.complete_batch([{"model": "coder", "user": "please [[REFUSE]] this", "tag": "b"}],
                                 poll_interval_s=0)[0]
    assert refused.refused and not refused.ok and refused.mode == "batch"
    cached_entries = [q for q in (tmp_path / "cache").glob("*/*.json") if q.parent.name != "batches"]
    assert not cached_entries  # refusals are never cached
    llm2, prov2 = make_llm(tmp_path / "t", provider=FakeProvider(
        responder=lambda req: "y" * 20), )
    # simulate truncation by patching the fake's stop_reason through a responder + monkey-free path
    trunc_req = {"model": "coder", "user": "u", "max_tokens": 8, "max_tokens_hard": True, "tag": "b"}
    d = prov2.call(llm2._prepare(trunc_req, batch=True)[2])
    d["stop_reason"] = "max_tokens"
    spec, _, req, key = llm2._prepare(trunc_req, batch=True)
    resp = llm2._response_from_dict(trunc_req, spec, "fake", req, key, d, cached=False, cost=0.0,
                                    mode="batch", batch_id="b1")
    assert resp.error and "OutputTruncated" in resp.error


def test_unknown_price_in_batch_books_assumed_cost(tmp_path):
    prov = FakeProvider()
    llm, _ = make_llm(tmp_path, provider=prov, llm_cfg={"unknown_price_assumed_usd_per_call": 0.05})
    out = llm.complete_batch([{"model": "gpt-99", "user": "u1", "tag": "b"}], poll_interval_s=0)
    assert out[0].ok and out[0].cost_usd is None
    assert llm.run_assumed_usd == pytest.approx(0.05 * 0.5)  # assumed cost also gets the batch discount
    assert _committed_matches(llm)


def test_prepare_rejects_unknown_request_keys(tmp_path):
    llm, _ = make_llm(tmp_path)
    with pytest.raises(TypeError, match="unknown request key"):
        llm.complete_batch([{"model": "coder", "user": "u", "max_token": 10}], poll_interval_s=0)
    with pytest.raises(TypeError):
        llm.complete_batch([{"user": "u"}], poll_interval_s=0)


def test_batch_hashes_ignore_fallbacks_setting(tmp_path):
    """A batch result must be reusable by sync calls: fallbacks are hashed as 'none' in batch mode."""
    llm, prov = make_llm(tmp_path, llm_cfg={"fallbacks": "default"})
    req = {"model": "coder", "user": "shared prompt", "tag": "b"}
    b = llm.complete_batch([req], poll_interval_s=0)[0]
    key_sync = llm._prepare(req)[3]
    assert b.request_hash != key_sync  # sync key carries fallbacks=default
    llm2, prov2 = make_llm(tmp_path, llm_cfg={"fallbacks": "none"})  # the project default
    s = llm2.complete(req["model"], None, req["user"], tag=req["tag"])
    assert s.cached and s.request_hash == b.request_hash and prov2.calls == 0


def test_chunk_manifests_survive_a_failed_later_chunk(tmp_path):
    class FlakySubmit(FakeProvider):
        def submit_batch(self, items):
            if self.batch_submissions >= 1:
                raise RuntimeError("submit failed")
            return super().submit_batch(items)

    prov = FlakySubmit()
    llm, _ = make_llm(tmp_path, provider=prov, llm_cfg={"batch": {"chunk_size": 2}})
    with pytest.raises(RuntimeError, match="submit failed"):
        llm.complete_batch(_reqs(5), poll_interval_s=0)
    assert llm.committed_usd() == 0.0
    mf = list((tmp_path / "cache" / "batches").glob("*.json"))
    assert len(mf) == 1  # the first chunk's batch is recorded, not orphaned
    assert json.loads(mf[0].read_text(encoding="utf-8"))["status"] == "submitted"


def test_cli_batch_status_is_guarded(tmp_path, monkeypatch, capsys):
    from src.llm import main

    monkeypatch.setenv("LLM_FAKE", "1")
    monkeypatch.setattr(LLM, "_abs", staticmethod(lambda p: tmp_path / Path(p).name))
    assert main(["--batch-status", "msgbatch_does_not_exist", "--model", "fake-x"]) == 2
    assert "unavailable" in capsys.readouterr().out.lower()
