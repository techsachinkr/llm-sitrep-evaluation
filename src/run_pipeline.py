"""Run Phases 2-5 end to end in batch mode, stopping at the human checkpoint.

The pipeline splits either side of the mandatory schema adjudication (CLAUDE.md rule 4):

    python -m src.run_pipeline --stage induce     # 2a open-code -> 2b consolidate -> STOP
    #   ... you review data/processed/schema/schema_review.yaml and approve it:
    python -m src.freeze_schema --review
    python -m src.freeze_schema --approve --by "Your Name"
    python -m src.run_pipeline --stage analyse    # 3 generate -> 4 judge -> report -> 5 metrics

`--stage all` runs both halves and will refuse to continue past the checkpoint unless a frozen
`schema.yaml` already exists. Every phase is separately cost-gated; `--dry-run` prints the
projections and sends nothing.

Batch mode (default) uses the Anthropic Message Batches API at 50% of list price. Batches are
asynchronous: a phase may take minutes to hours, and if the run is interrupted the manifests under
`.llm_cache/batches/` let the same command re-attach instead of paying twice.

CLI: python -m src.run_pipeline [--stage induce|analyse|all] [--event cyclone-idai-2019]
                                [--no-batch] [--dry-run] [--smoke N]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from src.util import cfg_path, get_logger, load_config

log = get_logger("sitrep.run")

PY = sys.executable


def _run(args: list[str], *, label: str) -> int:
    """Run one pipeline step as a subprocess so a crash cannot take the whole run down."""
    print(f"\n{'='*74}\n{label}\n{'='*74}")
    print("$ " + " ".join(a if " " not in a else f'"{a}"' for a in args[1:]) + "\n", flush=True)
    t0 = time.monotonic()
    rc = subprocess.run(args).returncode
    print(f"\n[{label}] exit={rc} in {time.monotonic() - t0:.0f}s", flush=True)
    return rc


def induce(event: str, *, batch: bool, dry: bool, smoke: int | None) -> int:
    """Phase 2: open-code the sitreps, then consolidate the codes into candidate slots."""
    base = [PY, "-m", "src.open_code", "--event", event]
    if batch:
        base.append("--batch")
    if dry:
        base.append("--dry-run")
    if smoke:
        base += ["--smoke", str(smoke)]
    rc = _run(base, label="PHASE 2a  open-code the human sitreps")
    if rc != 0 or dry:
        return rc
    cons = [PY, "-m", "src.consolidate"]
    if batch:
        cons.append("--batch")
    return _run(cons, label="PHASE 2b  consolidate codes into candidate slots")


def analyse(event: str, *, batch: bool, dry: bool, smoke: int | None) -> int:
    """Phases 3-5: generate, judge, report, then the reference-metric meta-evaluation."""
    # NB: pass the event key through unchanged. The loaders resolve the '-'/'_' spellings
    # themselves (src.util.resolve_event_dir); rewriting it here silently split generation and
    # judging across two directories once already.
    gen = [PY, "-m", "src.generate_sitreps", "--event", event,
           "--arms", "generic", "schema_guided"]
    if batch:
        gen.append("--batch")
    if dry:
        gen.append("--dry-run")
    if smoke:
        gen += ["--smoke", str(smoke)]
    rc = _run(gen, label="PHASE 3  generate machine sitreps (2 arms x 2 models)")
    if rc != 0 or dry:
        return rc

    jud = [PY, "-m", "src.judge_slots", "--event", event]
    if batch:
        jud.append("--batch")
    if smoke:
        jud += ["--smoke", str(smoke)]
    rc = _run(jud, label="PHASE 4  judge every (sitrep, slot) pair")
    if rc != 0:
        return rc

    rc = _run([PY, "-m", "src.report_results"],
              label="PHASE 4  completeness + VISIBILITY CEILING tables")
    if rc != 0:
        return rc

    ref = [PY, "-m", "src.reference_metrics", "--event", event]  # no --batch flag on this one
    rc = _run(ref, label="PHASE 5  ROUGE-L / BERTScore / generic LLM judge")
    if rc != 0:
        log.warning("reference metrics failed (rc=%s) — correlation will be skipped", rc)
        return rc
    return _run([PY, "-m", "src.correlate"], label="PHASE 5  correlation meta-evaluation")


def schema_is_frozen(processed: Path) -> bool:
    """True when a human-approved schema exists (rule 4)."""
    frozen = processed / "schema.yaml"
    if not frozen.is_file():
        return False
    import yaml

    data = yaml.safe_load(frozen.read_text(encoding="utf-8")) or {}
    return bool(isinstance(data, dict) and data.get("approved_by"))


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description="Run Phases 2-5 in batch mode")
    ap.add_argument("--stage", choices=("induce", "analyse", "all"), default="all")
    ap.add_argument("--event", default="cyclone-idai-2019")
    ap.add_argument("--no-batch", action="store_true", help="synchronous calls (2x the price)")
    ap.add_argument("--dry-run", action="store_true", help="print cost projections, send nothing")
    ap.add_argument("--smoke", type=int, default=None, help="tiny sample per phase, for eyeballing")
    args = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):  # pragma: no cover
        pass
    batch = not args.no_batch
    processed = cfg_path(load_config(), "processed", "data/processed")

    if args.stage in ("induce", "all"):
        rc = induce(args.event, batch=batch, dry=args.dry_run, smoke=args.smoke)
        if rc != 0:
            return rc
        if args.stage == "induce" or not schema_is_frozen(processed):
            print(f"\n{'='*74}\nSTOP — human checkpoint (CLAUDE.md rule 4)\n{'='*74}")
            print("Candidate slots are ready. They are NOT a schema until you approve them:\n")
            print("  python -m src.freeze_schema --review")
            print(f"  # edit {processed / 'schema' / 'schema_review.yaml'} — merge, rename, drop,")
            print("  #   and give every slot two synthetic examples")
            print('  python -m src.freeze_schema --approve --by "Your Name"\n')
            print("Then continue with:  python -m src.run_pipeline --stage analyse")
            return 0

    if args.stage in ("analyse", "all"):
        if not schema_is_frozen(processed):
            print("No approved schema yet — run the checkpoint first:\n"
                  "  python -m src.freeze_schema --review\n"
                  '  python -m src.freeze_schema --approve --by "Your Name"')
            return 2
        return analyse(args.event, batch=batch, dry=args.dry_run, smoke=args.smoke)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
