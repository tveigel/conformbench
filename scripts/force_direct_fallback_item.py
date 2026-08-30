#!/usr/bin/env python3
"""Force selected Direct JSON items to the invalid-output fallback.

This is for paper-run repair when an item should be treated as having exhausted
the allowed Direct JSON output attempts. The script does not call the generator.
It materializes the same prior-state fallback used by Direct, scores it with
the normal evaluator, and injects the resulting turn/evaluation artifacts into
an existing run.
"""

from __future__ import annotations

import argparse
import json
import shutil
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from conformbench import benchmark
from conformbench.items import normalize_item
from conformbench.questionnaires import load_questionnaire
from conformbench.systems.flatagent import (
    _complete_state,
    _materialize_repeat_counts,
    _schema_shape,
)
from scripts.run_architecture_comparison import (
    finalize_run_artifacts,
    load_item_packets,
    read_json,
    scan_costs,
    write_json,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--fallback-dir", type=Path, required=True)
    parser.add_argument("--backup-dir", type=Path, required=True)
    parser.add_argument("--item-id", action="append", required=True)
    parser.add_argument("--source-turn", type=Path)
    parser.add_argument("--allowed-attempts", type=int, default=3)
    parser.add_argument("--evaluator-model", default="gpt-5.4")
    parser.add_argument("--evaluator-reasoning", default="medium")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--inject", action="store_true")
    args = parser.parse_args()

    if args.allowed_attempts < 1:
        raise SystemExit("--allowed-attempts must be >= 1")

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

    source_turn = read_json(args.source_turn) if args.source_turn else {}
    source_generation = (
        ((source_turn.get("provenance") or {}).get("generation") or {})
        if isinstance(source_turn, dict)
        else {}
    )
    source_agent = (
        source_turn.get("agent_response")
        if isinstance(source_turn.get("agent_response"), dict)
        else {}
    )
    allowed_model_calls = _first_model_calls(source_agent, args.allowed_attempts)

    def solver(turn: Any) -> dict[str, Any]:
        schema = turn.schema
        shape = _schema_shape(schema)
        fallback_state = _materialize_repeat_counts(
            _complete_state(turn.prior_state, schema),
            shape,
        )
        model_config = source_generation.get("model") or {
            "model": "gpt-5.4-mini",
            "requested_model": "gpt-5.4-mini",
            "reasoning_effort": "none",
        }
        failure = {
            "agent": "Direct",
            "attempts": args.allowed_attempts,
            "last_error": "Direct item marked failed after exhausting the allowed output attempts.",
            "model_calls": deepcopy(allowed_model_calls),
            "repair_note": (
                "Forced fallback because this item had already exceeded the "
                "paper run's cumulative Direct output-attempt budget."
            ),
        }
        return {
            "resulting_state": fallback_state,
            "agent_response": {
                "raw_response": (
                    "Direct fallback: prior-state no-update candidate after "
                    "exhausting the allowed output attempts."
                ),
                "attempts": args.allowed_attempts,
                "model_calls": deepcopy(allowed_model_calls),
                "invalid_output_policy": "prior_state",
                "invalid_output_error": failure["last_error"],
                "rejected_attempts": [],
            },
            "provenance": {
                "generation": {
                    "agent": "Direct",
                    "model": model_config,
                    "tool_policy": {
                        "mode": "state_json",
                        "invalid_output_policy": "prior_state",
                    },
                    "schema_interface": {
                        "public_projection": (
                            "shared_schema_field_guide_with_schema_completed_prior_state"
                        ),
                    },
                    "fallback": {
                        "reason": "cumulative_output_attempt_budget_exceeded",
                        "scoring_interpretation": "no_update_prior_state",
                        "failure": failure,
                    },
                },
            },
        }

    args.fallback_dir.mkdir(parents=True, exist_ok=True)
    benchmark.run(
        items=packets,
        solver=solver,
        output_dir=args.fallback_dir,
        run_id=args.fallback_dir.name,
        workers=args.workers,
        evaluator_model_id=args.evaluator_model,
        evaluator_reasoning_effort=args.evaluator_reasoning,
        score=True,
    )
    fallback_summary = finalize_run_artifacts(
        run_dir=args.fallback_dir,
        items_dir=args.items_dir,
        status="completed",
        stop_reason="",
        generate_figures=False,
    )

    injected: list[dict[str, Any]] = []
    if args.inject:
        args.backup_dir.mkdir(parents=True, exist_ok=True)
        for turn_path in sorted(args.fallback_dir.rglob("turn_result.json")):
            rel_dir = turn_path.parent.relative_to(args.fallback_dir)
            source_dir = turn_path.parent
            dest_dir = args.run_dir / rel_dir
            backup_item_dir = args.backup_dir / rel_dir
            backup_item_dir.mkdir(parents=True, exist_ok=True)
            dest_dir.mkdir(parents=True, exist_ok=True)

            for name in ("turn_result.json", "evaluation.json", "solver_failure.json"):
                old_path = dest_dir / name
                if old_path.exists():
                    shutil.copy2(old_path, backup_item_dir / name)

            shutil.copy2(source_dir / "turn_result.json", dest_dir / "turn_result.json")
            shutil.copy2(source_dir / "evaluation.json", dest_dir / "evaluation.json")
            stale_failure = dest_dir / "solver_failure.json"
            if stale_failure.exists():
                stale_failure.unlink()

            turn = read_json(source_dir / "turn_result.json")
            evaluation = read_json(source_dir / "evaluation.json")
            injected.append(
                {
                    "item_id": turn.get("scenario_id"),
                    "relative_dir": str(rel_dir),
                    "attempts": (turn.get("agent_response") or {}).get("attempts"),
                    "invalid_output_policy": (turn.get("agent_response") or {}).get(
                        "invalid_output_policy"
                    ),
                    "evaluation_summary": evaluation.get("summary"),
                }
            )

        final_summary = finalize_run_artifacts(
            run_dir=args.run_dir,
            items_dir=args.items_dir,
            status="completed_repaired",
            stop_reason="",
            generate_figures=True,
        )
        _append_run_repair_metadata(
            run_dir=args.run_dir,
            fallback_dir=args.fallback_dir,
            backup_dir=args.backup_dir,
            item_ids=sorted(target_ids),
            allowed_attempts=args.allowed_attempts,
        )
    else:
        final_summary = None

    manifest = {
        "kind": "direct_forced_invalid_output_fallback",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "items_dir": str(args.items_dir),
        "run_dir": str(args.run_dir),
        "fallback_dir": str(args.fallback_dir),
        "backup_dir": str(args.backup_dir),
        "item_ids": sorted(target_ids),
        "allowed_attempts": args.allowed_attempts,
        "source_turn": str(args.source_turn) if args.source_turn else "",
        "fallback_cost": scan_costs(args.fallback_dir),
        "fallback_summary": _small_summary(fallback_summary),
        "injected": injected,
        "final_run_cost": scan_costs(args.run_dir) if args.inject else None,
        "final_run_summary": _small_summary(final_summary) if final_summary else None,
    }
    write_json(args.fallback_dir / "forced_fallback_manifest.json", manifest)
    if args.inject:
        write_json(args.run_dir / "direct_forced_fallback_manifest.json", manifest)

    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def _first_model_calls(agent_response: dict[str, Any], max_count: int) -> list[dict[str, Any]]:
    calls = agent_response.get("model_calls")
    if not isinstance(calls, list):
        return []
    result: list[dict[str, Any]] = []
    for index, call in enumerate(calls[:max_count], start=1):
        if not isinstance(call, dict):
            continue
        item = deepcopy(call)
        item["call_index"] = index
        result.append(item)
    return result


def _append_run_repair_metadata(
    *,
    run_dir: Path,
    fallback_dir: Path,
    backup_dir: Path,
    item_ids: list[str],
    allowed_attempts: int,
) -> None:
    path = run_dir / "run_metadata.json"
    if not path.exists():
        return
    payload = read_json(path)
    repairs = payload.setdefault("repairs", [])
    if isinstance(repairs, list):
        repairs.append(
            {
                "kind": "direct_forced_invalid_output_fallback",
                "repaired_at": datetime.now(timezone.utc).isoformat(),
                "fallback_dir": str(fallback_dir),
                "backup_dir": str(backup_dir),
                "item_ids": item_ids,
                "policy": (
                    f"{allowed_attempts} Direct output attempts, then prior-state "
                    "no-update fallback"
                ),
            }
        )
    write_json(path, payload)


def _small_summary(summary: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(summary, dict):
        return None
    aggregate = summary.get("aggregate") or {}
    views = aggregate.get("metric_views") or {}
    changed = views.get("changed_fields") or {}
    strict = changed.get("strict") or {}
    return {
        "correct": aggregate.get("correct"),
        "incorrect": aggregate.get("incorrect"),
        "partially_correct": aggregate.get("partially_correct"),
        "all_field_accuracy": aggregate.get("accuracy"),
        "changed_precision": strict.get("precision"),
        "changed_recall": strict.get("recall"),
        "changed_f1": strict.get("f1"),
        "gold_changed_total": changed.get("gold_changed_total"),
    }


if __name__ == "__main__":
    raise SystemExit(main())
