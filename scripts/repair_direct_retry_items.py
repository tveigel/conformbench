#!/usr/bin/env python3
"""Repair Direct JSON items affected by an accidental retry-policy change."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from conformbench import benchmark
from conformbench.items import normalize_item
from conformbench.systems import direct
from scripts.run_architecture_comparison import (
    finalize_run_artifacts,
    item_with_model_metadata,
    load_item_packets,
    read_json,
    scan_costs,
    write_json,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rerun selected Direct JSON items with the 3-attempt policy and inject them."
    )
    parser.add_argument("--items-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--repair-dir", type=Path, required=True)
    parser.add_argument("--backup-dir", type=Path, required=True)
    parser.add_argument("--item-id", action="append", required=True)
    parser.add_argument("--generator-model", default="gpt-5.4-mini")
    parser.add_argument("--generator-reasoning", default="none")
    parser.add_argument("--evaluator-model", default="gpt-5.4")
    parser.add_argument("--evaluator-reasoning", default="medium")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--inject", action="store_true")
    args = parser.parse_args()

    if direct._MAX_ATTEMPTS != 3:
        raise SystemExit(f"Direct _MAX_ATTEMPTS is {direct._MAX_ATTEMPTS}, expected 3")

    target_ids = set(args.item_id)
    packets = [
        packet
        for packet in load_item_packets(args.items_dir)
        if normalize_item(packet)["item_id"] in target_ids
    ]
    found_ids = {normalize_item(packet)["item_id"] for packet in packets}
    missing = sorted(target_ids - found_ids)
    if missing:
        raise SystemExit(f"Could not find item ids in {args.items_dir}: {missing}")

    args.repair_dir.mkdir(parents=True, exist_ok=True)
    repair_started_at = datetime.now(timezone.utc).isoformat()
    benchmark.run(
        items=[
            item_with_model_metadata(
                packet,
                model_id=args.generator_model,
                reasoning_effort=args.generator_reasoning,
                direct_invalid_output_policy="prior_state",
            )
            for packet in packets
        ],
        solver=direct.solve,
        output_dir=args.repair_dir,
        run_id=args.repair_dir.name,
        workers=args.workers,
        evaluator_model_id=args.evaluator_model,
        evaluator_reasoning_effort=args.evaluator_reasoning,
        score=True,
    )
    repair_summary = finalize_run_artifacts(
        run_dir=args.repair_dir,
        items_dir=args.items_dir,
        status="completed",
        stop_reason="",
        generate_figures=False,
    )

    injected: list[dict[str, Any]] = []
    if args.inject:
        args.backup_dir.mkdir(parents=True, exist_ok=True)
        for turn_path in sorted(args.repair_dir.rglob("turn_result.json")):
            rel_dir = turn_path.parent.relative_to(args.repair_dir)
            source_dir = turn_path.parent
            dest_dir = args.run_dir / rel_dir
            backup_item_dir = args.backup_dir / rel_dir
            backup_item_dir.mkdir(parents=True, exist_ok=True)
            dest_dir.mkdir(parents=True, exist_ok=True)

            old_turn = dest_dir / "turn_result.json"
            old_eval = dest_dir / "evaluation.json"
            old_failure = dest_dir / "solver_failure.json"
            if old_turn.exists():
                shutil.copy2(old_turn, backup_item_dir / "turn_result.json")
            if old_eval.exists():
                shutil.copy2(old_eval, backup_item_dir / "evaluation.json")
            if old_failure.exists():
                shutil.copy2(old_failure, backup_item_dir / "solver_failure.json")

            shutil.copy2(source_dir / "turn_result.json", dest_dir / "turn_result.json")
            shutil.copy2(source_dir / "evaluation.json", dest_dir / "evaluation.json")
            if old_failure.exists():
                old_failure.unlink()

            turn = read_json(source_dir / "turn_result.json")
            injected.append(
                {
                    "item_id": turn.get("scenario_id"),
                    "relative_dir": str(rel_dir),
                    "repair_attempts": (turn.get("agent_response") or {}).get("attempts"),
                    "invalid_output_policy": (turn.get("agent_response") or {}).get(
                        "invalid_output_policy"
                    ),
                }
            )

        final_summary = finalize_run_artifacts(
            run_dir=args.run_dir,
            items_dir=args.items_dir,
            status="completed_repaired",
            stop_reason="",
            generate_figures=False,
        )
        run_metadata_path = args.run_dir / "run_metadata.json"
        if run_metadata_path.exists():
            run_metadata = read_json(run_metadata_path)
            repairs = run_metadata.setdefault("repairs", [])
            if isinstance(repairs, list):
                repairs.append(
                    {
                        "kind": "direct_retry_policy_repair",
                        "repaired_at": datetime.now(timezone.utc).isoformat(),
                        "repair_dir": str(args.repair_dir),
                        "backup_dir": str(args.backup_dir),
                        "item_ids": sorted(target_ids),
                        "policy": "3 Direct output attempts, then prior-state no-update fallback",
                    }
                )
            write_json(run_metadata_path, run_metadata)
    else:
        final_summary = None

    manifest = {
        "kind": "direct_retry_policy_repair",
        "started_at": repair_started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "items_dir": str(args.items_dir),
        "run_dir": str(args.run_dir),
        "repair_dir": str(args.repair_dir),
        "backup_dir": str(args.backup_dir),
        "item_ids": sorted(target_ids),
        "generator_model": args.generator_model,
        "generator_reasoning": args.generator_reasoning,
        "evaluator_model": args.evaluator_model,
        "evaluator_reasoning": args.evaluator_reasoning,
        "workers": args.workers,
        "policy": "3 Direct output attempts, then prior-state no-update fallback",
        "repair_cost": scan_costs(args.repair_dir),
        "repair_summary": _small_summary(repair_summary),
        "injected": injected,
        "final_run_cost": scan_costs(args.run_dir) if args.inject else None,
        "final_run_summary": _small_summary(final_summary) if final_summary else None,
    }
    write_json(args.repair_dir / "repair_manifest.json", manifest)
    if args.inject:
        write_json(args.run_dir / "direct_retry_policy_repair_manifest.json", manifest)

    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def _small_summary(summary: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(summary, dict):
        return None
    aggregate = summary.get("aggregate") or {}
    changed = ((aggregate.get("metric_views") or {}).get("changed_fields") or {})
    strict = changed.get("strict") or {}
    return {
        "item_count": aggregate.get("item_count"),
        "fully_correct": aggregate.get("fully_correct"),
        "all_field_accuracy": ((aggregate.get("metric_views") or {}).get("all_fields") or {})
        .get("strict", {})
        .get("accuracy"),
        "changed_precision": strict.get("precision"),
        "changed_recall": strict.get("recall"),
        "changed_f1": strict.get("f1"),
        "gold_changed_total": changed.get("gold_changed_total"),
    }


if __name__ == "__main__":
    raise SystemExit(main())
