"""Freeze the adjudicated schema — the mandatory human checkpoint (CLAUDE.md rule 4).

Phase 2 produces *candidates* (`data/processed/schema/candidate_slots.yaml`). They are not a
schema until a human merges/renames/approves them. This script is that gate: it turns reviewed
candidates into `data/processed/schema.yaml`, which is the only file the judging and
`schema_guided` generation arms are allowed to consume.

Workflow:

    python -m src.freeze_schema --review      # write a reviewable copy + instructions
    # ... edit data/processed/schema/schema_review.yaml by hand: merge, rename, drop, add examples
    python -m src.freeze_schema --approve --by "Your Name"

The frozen file records who approved it and when, so the paper's methods section can state the
adjudication honestly. Validation enforces PLAN's contract: unique ids, a definition per slot,
two examples per slot, and at most `schema.max_slots` slots.

CLI: python -m src.freeze_schema [--review | --approve --by NAME] [--force]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

from src.util import cfg_path, ensure_dir, get_logger, load_config, utc_now_iso

log = get_logger("sitrep.freeze")

CANDIDATES = "schema/candidate_slots.yaml"
REVIEW = "schema/schema_review.yaml"
FROZEN = "schema.yaml"
REQUIRED_EXAMPLES = 2


def _slots_of(data: Any) -> list[dict[str, Any]]:
    """Accept `{slots: [...]}`, `{candidate_slots: [...]}` or a bare list."""
    if isinstance(data, dict):
        rows = data.get("slots") or data.get("candidate_slots") or []
    else:
        rows = data or []
    return [dict(r) for r in rows if isinstance(r, dict)]


def validate(slots: list[dict[str, Any]], max_slots: int) -> list[str]:
    """Return a list of problems; empty means the schema may be frozen."""
    problems: list[str] = []
    if not slots:
        problems.append("no slots at all")
    if len(slots) > max_slots:
        problems.append(f"{len(slots)} slots exceeds schema.max_slots={max_slots}")
    seen: set[str] = set()
    for i, s in enumerate(slots, start=1):
        sid = str(s.get("id") or s.get("slot_id") or "").strip()
        name = str(s.get("name") or "").strip()
        where = f"slot #{i} ({sid or name or '?'})"
        if not sid:
            problems.append(f"{where}: missing id")
        elif sid in seen:
            problems.append(f"{where}: duplicate id {sid!r}")
        else:
            seen.add(sid)
        if not name:
            problems.append(f"{where}: missing name")
        if len(str(s.get("definition") or "").strip()) < 20:
            problems.append(f"{where}: definition missing or too short (needs a usable definition)")
        ex = s.get("examples") or []
        if not isinstance(ex, list) or len([e for e in ex if str(e).strip()]) < REQUIRED_EXAMPLES:
            problems.append(f"{where}: needs {REQUIRED_EXAMPLES} synthetic examples (PLAN Phase 2)")
    return problems


def normalise(slots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Canonical slot records: id, name, definition, examples, prevalence."""
    out: list[dict[str, Any]] = []
    for s in slots:
        out.append({
            "id": str(s.get("id") or s.get("slot_id") or "").strip(),
            "name": str(s.get("name") or "").strip(),
            "definition": str(s.get("definition") or "").strip(),
            "examples": [str(e).strip() for e in (s.get("examples") or []) if str(e).strip()],
            "prevalence": s.get("prevalence"),
        })
    return out


def write_review(data_root: Path) -> Path:
    """Copy the candidates to a hand-editable review file."""
    src = data_root / CANDIDATES
    if not src.exists():
        raise FileNotFoundError(
            f"no candidate slots at {src}\nRun the Phase 2 pass first:\n"
            "  python -m src.open_code --event <event>\n  python -m src.consolidate")
    slots = normalise(_slots_of(yaml.safe_load(src.read_text(encoding="utf-8"))))
    dest = data_root / REVIEW
    ensure_dir(dest.parent)
    header = (
        "# REVIEW THIS FILE BY HAND, then run: python -m src.freeze_schema --approve --by \"Your Name\"\n"
        "#\n"
        "# This is the mandatory human checkpoint (CLAUDE.md rule 4) and it goes in the paper's\n"
        "# methods section. Merge near-duplicate slots, rename them in practitioners' language,\n"
        "# delete anything that is not a real information type, and give every slot two SYNTHETIC\n"
        f"# examples (never quote a real report). Keep at most {{max_slots}} slots.\n#\n"
        "# prevalence = fraction of sampled sitreps whose open codes fell in this cluster.\n\n")
    cfg = load_config()
    dest.write_text(header.format(max_slots=(cfg.get("schema") or {}).get("max_slots", 15))
                    + yaml.safe_dump({"slots": slots}, sort_keys=False, allow_unicode=True),
                    encoding="utf-8")
    return dest


def approve(data_root: Path, approved_by: str, *, force: bool = False, max_slots: int = 15) -> tuple[Path, list[str]]:
    """Validate the reviewed file and freeze it to `schema.yaml`."""
    review = data_root / REVIEW
    src = review if review.exists() else data_root / CANDIDATES
    if not src.exists():
        raise FileNotFoundError(f"nothing to approve: neither {review} nor {data_root / CANDIDATES} exists")
    slots = normalise(_slots_of(yaml.safe_load(src.read_text(encoding="utf-8"))))
    problems = validate(slots, max_slots)
    if problems and not force:
        return src, problems
    frozen = data_root / FROZEN
    payload = {
        "schema_version": 1,
        "approved_by": approved_by,
        "approved_at": utc_now_iso(),
        "source_file": str(src.relative_to(data_root)) if src.is_relative_to(data_root) else str(src),
        "n_slots": len(slots),
        "validation_overridden": bool(problems and force),
        "slots": slots,
    }
    ensure_dir(frozen.parent)
    frozen.write_text(
        "# FROZEN SCHEMA — adjudicated by a human (CLAUDE.md rule 4). Judging and the\n"
        "# schema_guided generation arm read this file and only this file.\n\n"
        + yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return frozen, problems


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description="Freeze the adjudicated slot schema (human checkpoint)")
    ap.add_argument("--review", action="store_true", help="write a hand-editable copy of the candidates")
    ap.add_argument("--approve", action="store_true", help="validate the reviewed file and freeze it")
    ap.add_argument("--by", default="", help="who adjudicated the schema (recorded in schema.yaml)")
    ap.add_argument("--force", action="store_true", help="freeze despite validation problems (recorded)")
    ap.add_argument("--data-root", type=Path, default=None)
    args = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):  # pragma: no cover
        pass
    cfg = load_config()
    data_root = args.data_root or cfg_path(cfg, "processed", "data/processed")
    max_slots = int((cfg.get("schema") or {}).get("max_slots", 15))

    try:
        if args.review:
            dest = write_review(data_root)
            print(f"wrote {dest}\n\nEdit it by hand, then run:\n"
                  '  python -m src.freeze_schema --approve --by "Your Name"')
            return 0
        if args.approve:
            if not args.by.strip():
                print("--approve needs --by \"Your Name\" (the adjudication is recorded in the schema)")
                return 2
            dest, problems = approve(data_root, args.by.strip(), force=args.force, max_slots=max_slots)
            if problems and not args.force:
                print(f"NOT frozen — {len(problems)} problem(s) in {dest}:")
                for p in problems:
                    print(f"  - {p}")
                print("\nFix them (or re-run with --force to freeze anyway, which is recorded).")
                return 1
            if problems:
                print(f"frozen WITH {len(problems)} unresolved problem(s) (recorded as validation_overridden)")
            print(f"frozen schema -> {dest}")
            print("Judging and the schema_guided arm will now use it.")
            return 0
    except FileNotFoundError as exc:
        print(str(exc))
        return 2
    ap.print_help()
    return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
