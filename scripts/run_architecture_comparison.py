#!/usr/bin/env python3
"""Run the guarded overnight architecture comparison.

The runner is intentionally conservative:

* fixed generation and evaluator model configuration;
* scored batches instead of generate-all-then-score-all;
* early stop per architecture when changed-field F1 falls below a threshold;
* global and per-architecture cost caps using provider-reported costs;
* standard Studio artifacts after every batch and final reports at the end.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import sys
import traceback
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from conformbench import benchmark
from conformbench.items import BENCHMARK_ITEMS_DIR, DATA_ROOT, REPORTS_DIR, normalize_item
from conformbench.llm_accounting import extract_cost, extract_usage


ARCHITECTURES = {
    "direct": "conformbench.systems.direct:solve",
    "diff_then_verify": "conformbench.systems.diff_then_verify:solve",
    "flatagent": "conformbench.systems.flatagent:solve",
    "split_agent": "conformbench.systems.split_agent:solve",
    "split_agent_briefed": "conformbench.systems.split_agent_briefed:solve",
    "no_update": "conformbench.systems.no_update:solve",
}

# Official standard-processing token prices in USD per 1M tokens, checked
# 2026-07-10. Provider responses do not include dollar costs, so these rates
# make the runner's cost caps effective with the official OpenAI endpoint.
# https://developers.openai.com/api/docs/models/gpt-5.4
# https://developers.openai.com/api/docs/models/gpt-5.4-mini
OPENAI_TOKEN_PRICES_USD_PER_MILLION = {
    "gpt-5.4": {"input": 2.50, "cached_input": 0.25, "output": 15.00},
    "gpt-5.4-mini": {"input": 0.75, "cached_input": 0.075, "output": 4.50},
}


def parse_args() -> argparse.Namespace:
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items-dir", type=Path, default=BENCHMARK_ITEMS_DIR)
    parser.add_argument("--runs-dir", type=Path, default=REPORTS_DIR / "runs")
    parser.add_argument("--run-prefix", default=f"archcmp_{timestamp}")
    parser.add_argument(
        "--run-purpose",
        choices=["dev_tuning", "selected_system"],
        help=(
            "Clean Studio/reporting label. Defaults to dev_tuning for dev split "
            "runs and selected_system for test split runs."
        ),
    )
    parser.add_argument(
        "--architectures",
        nargs="+",
        choices=sorted(ARCHITECTURES),
        default=["direct", "flatagent", "split_agent"],
    )
    parser.add_argument("--generator-model", default="gpt-5.4-mini")
    parser.add_argument("--generator-reasoning")
    parser.add_argument(
        "--direct-invalid-output-policy",
        choices=["fail", "prior_state"],
        default="fail",
        help=(
            "Direct JSON behavior after all JSON/shape retries fail. "
            "'fail' marks the architecture failed; 'prior_state' scores the item "
            "as a no-update prior-state candidate."
        ),
    )
    parser.add_argument("--evaluator-model", default="gpt-5.4")
    parser.add_argument(
        "--evaluator-reasoning",
        choices=["low", "medium", "high"],
        default="medium",
    )
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int, help="Debug limit on total item count.")
    parser.add_argument(
        "--changed-f1-floor",
        type=float,
        default=0.60,
        help="Early-stop floor for strict changed-field F1 after each batch. Set 0 to disable.",
    )
    parser.add_argument(
        "--min-gold-changed-for-gate",
        type=int,
        default=20,
        help="Only apply the changed-F1 gate after this many gold-changed fields.",
    )
    parser.add_argument("--max-total-cost-usd", type=float, default=250.0)
    parser.add_argument("--max-architecture-cost-usd", type=float, default=100.0)
    parser.add_argument("--max-batch-cost-usd", type=float, default=40.0)
    parser.add_argument(
        "--cost-tracking-max-tokens",
        type=int,
        default=16000,
        help="Max-token cap used while preserving provider cost metadata.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument(
        "--allow-split-generation",
        action="store_true",
        help="Allow SplitAgent to create missing split files. Off by default to avoid hidden LLM spend.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    if args.resume and args.overwrite:
        raise SystemExit("Use at most one of --resume and --overwrite")

    os.environ.setdefault("CONFORMBENCH_TRACK_LLM_COSTS", "true")
    os.environ.setdefault("CONFORMBENCH_COST_TRACKING_MAX_TOKENS", str(args.cost_tracking_max_tokens))

    args.runs_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.runs_dir / f"{args.run_prefix}__comparison.log"
    manifest_path = args.runs_dir / f"{args.run_prefix}__comparison_manifest.json"

    items = load_item_packets(args.items_dir)
    if args.limit is not None:
        items = items[: args.limit]
    if not items:
        raise SystemExit(f"No ground_truth.json items found under {args.items_dir}")

    plan = build_plan(args, items)
    split_metadata = split_metadata_from_items_dir(args.items_dir)
    run_purpose = run_purpose_from_args(args, split_metadata)
    if split_metadata:
        plan["split"] = split_metadata["split"]
        plan["split_id"] = split_metadata.get("split_id")
    if run_purpose:
        plan["run_purpose"] = run_purpose
    write_json(manifest_path, plan)
    log(log_path, f"Plan written to {manifest_path}")
    log(log_path, json.dumps(plan, indent=2, sort_keys=True))

    preflight(args, items, log_path)
    if args.plan_only:
        log(log_path, "Plan-only mode: no LLM calls made.")
        return 0

    comparison: dict[str, Any] = {
        **plan,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "runs": [],
    }
    write_json(manifest_path, comparison)

    total_cost = 0.0
    for architecture in args.architectures:
        run_dir = args.runs_dir / run_dir_name(args, architecture)
        result = run_architecture(
            args=args,
            architecture=architecture,
            items=items,
            run_dir=run_dir,
            log_path=log_path,
            global_cost_before=total_cost,
        )
        comparison["runs"].append(result)
        total_cost += float(result.get("new_cost_usd", 0.0) or 0.0)
        comparison["total_observed_cost_usd"] = round(total_cost, 6)
        write_json(manifest_path, comparison)
        if total_cost >= args.max_total_cost_usd:
            log(
                log_path,
                (
                    f"Global cost cap reached after {architecture}: "
                    f"${total_cost:.4f} >= ${args.max_total_cost_usd:.4f}. Stopping."
                ),
            )
            break

    comparison["status"] = "completed"
    comparison["completed_at"] = datetime.now(timezone.utc).isoformat()
    comparison["total_observed_cost_usd"] = round(total_cost, 6)
    write_json(manifest_path, comparison)
    write_comparison_markdown(args.runs_dir / f"{args.run_prefix}__comparison.md", comparison)
    log(log_path, "Architecture comparison finished.")
    return 0


def run_architecture(
    *,
    args: argparse.Namespace,
    architecture: str,
    items: list[dict[str, Any]],
    run_dir: Path,
    log_path: Path,
    global_cost_before: float,
) -> dict[str, Any]:
    solver_path = ARCHITECTURES[architecture]
    solver = benchmark.load_solver(solver_path)
    prepare_run_dir(run_dir, args)
    clear_stale_terminal_markers(run_dir)
    split_metadata = split_metadata_from_items_dir(args.items_dir)
    run_purpose = run_purpose_from_args(args, split_metadata)
    display_name = display_name_for_run(run_purpose, architecture)
    if split_metadata:
        write_json(run_dir / "split_view_metadata.json", {
            **split_metadata,
            "mode": "direct-run",
            "origin": "architecture-comparison",
            **({"run_purpose": run_purpose} if run_purpose else {}),
            **({"display_name": display_name} if display_name else {}),
        })

    run_metadata = {
        "run_id": run_dir.name,
        "architecture": architecture,
        "solver": solver_path,
        "items_dir": str(args.items_dir),
        **({"split": split_metadata["split"]} if split_metadata else {}),
        **({"split_id": split_metadata["split_id"]} if split_metadata and split_metadata.get("split_id") else {}),
        **({"run_purpose": run_purpose} if run_purpose else {}),
        **({"display_name": display_name} if display_name else {}),
        "model_id": args.generator_model if architecture != "no_update" else "deterministic",
        "model_reasoning_effort": args.generator_reasoning or "",
        "evaluator_model_id": args.evaluator_model,
        "evaluator_reasoning_effort": args.evaluator_reasoning,
        "direct_invalid_output_policy": args.direct_invalid_output_policy,
        "batch_size": args.batch_size,
        "changed_f1_floor": args.changed_f1_floor,
        "min_gold_changed_for_gate": args.min_gold_changed_for_gate,
        "max_architecture_cost_usd": args.max_architecture_cost_usd,
        "max_batch_cost_usd": args.max_batch_cost_usd,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(run_dir / "run_metadata.json", run_metadata)

    evaluated = evaluated_item_ids(run_dir)
    remaining = [item for item in items if item_id(item) not in evaluated]
    log(log_path, f"[{architecture}] {len(evaluated)} already evaluated, {len(remaining)} remaining.")

    status = "completed"
    stop_reason = ""
    failure_record: dict[str, Any] | None = None
    previous_run_cost = scan_costs(run_dir)["total_cost_usd"]

    for batch_index, batch in enumerate(chunks(remaining, args.batch_size), start=1):
        batch_ids = [item_id(item) for item in batch]
        log(log_path, f"[{architecture}] Batch {batch_index}: {batch_ids[0]} ... {batch_ids[-1]} ({len(batch)} items)")
        before_batch_cost = scan_costs(run_dir)["total_cost_usd"]

        try:
            benchmark.run(
                items=[
                    item_with_model_metadata(
                        item,
                        model_id=args.generator_model,
                        reasoning_effort=args.generator_reasoning,
                        direct_invalid_output_policy=args.direct_invalid_output_policy,
                    )
                    for item in batch
                ],
                solver=solver,
                output_dir=run_dir,
                run_id=run_dir.name,
                workers=args.workers,
                evaluator_model_id=args.evaluator_model,
                evaluator_reasoning_effort=args.evaluator_reasoning,
                score=True,
            )
        except Exception as exc:
            status = "failed"
            stop_reason = f"batch {batch_index} failed: {exc}"
            failure_record = build_architecture_failure_record(
                architecture=architecture,
                batch_index=batch_index,
                batch_ids=batch_ids,
                exc=exc,
                run_dir=run_dir,
            )
            write_json(run_dir / "architecture_failure.json", failure_record)
            log(log_path, f"[{architecture}] ERROR: {stop_reason}")
            log(
                log_path,
                (
                    f"[{architecture}] failure_kind={failure_record['failure_kind']}; "
                    f"artifact={run_dir / 'architecture_failure.json'}; continuing comparison."
                ),
            )
            log(log_path, traceback.format_exc())
            break

        summary = finalize_run_artifacts(
            run_dir=run_dir,
            items_dir=args.items_dir,
            status=status,
            stop_reason=stop_reason,
            generate_figures=False,
        )
        cost_summary = scan_costs(run_dir)
        batch_cost = cost_summary["total_cost_usd"] - before_batch_cost
        arch_new_cost = cost_summary["total_cost_usd"] - previous_run_cost
        total_cost = global_cost_before + arch_new_cost
        changed = ((summary.get("aggregate") or {}).get("metric_views") or {}).get("changed_fields") or {}
        strict = changed.get("strict") or {}
        changed_f1 = float(strict.get("f1") or 0.0)
        gold_changed = int(changed.get("gold_changed_total") or 0)
        evaluated_now = len(evaluated_item_ids(run_dir))
        log(
            log_path,
            (
                f"[{architecture}] after batch {batch_index}: items={evaluated_now}, "
                f"changed_F1={changed_f1:.4f}, gold_changed={gold_changed}, "
                f"batch_cost=${batch_cost:.4f}, arch_cost=${arch_new_cost:.4f}, "
                f"global_cost=${total_cost:.4f}"
            ),
        )

        if batch_cost >= args.max_batch_cost_usd:
            status = "stopped_early"
            stop_reason = (
                f"batch cost ${batch_cost:.4f} exceeded cap ${args.max_batch_cost_usd:.4f}"
            )
            break
        if arch_new_cost >= args.max_architecture_cost_usd:
            status = "stopped_early"
            stop_reason = (
                f"architecture cost ${arch_new_cost:.4f} exceeded cap "
                f"${args.max_architecture_cost_usd:.4f}"
            )
            break
        if total_cost >= args.max_total_cost_usd:
            status = "stopped_early"
            stop_reason = (
                f"global cost ${total_cost:.4f} exceeded cap ${args.max_total_cost_usd:.4f}"
            )
            break
        if gold_changed >= args.min_gold_changed_for_gate and changed_f1 < args.changed_f1_floor:
            status = "stopped_early"
            stop_reason = (
                f"changed-field F1 {changed_f1:.4f} below floor {args.changed_f1_floor:.4f}"
            )
            break

    run_metadata.update(
        {
            "status": status,
            "stop_reason": stop_reason,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            **({"failure": failure_record} if failure_record is not None else {}),
        }
    )
    write_json(run_dir / "run_metadata.json", run_metadata)
    summary = finalize_run_artifacts(
        run_dir=run_dir,
        items_dir=args.items_dir,
        status=status,
        stop_reason=stop_reason,
        generate_figures=True,
    )
    if failure_record is not None:
        patch_failure_metadata(run_dir, failure_record)
    cost_summary = scan_costs(run_dir)
    write_stop_marker(run_dir, status=status, stop_reason=stop_reason)
    write_failure_marker(run_dir, status=status, stop_reason=stop_reason, failure_record=failure_record)
    new_cost = cost_summary["total_cost_usd"] - previous_run_cost
    result = {
        "architecture": architecture,
        "run_dir": str(run_dir),
        "status": status,
        "stop_reason": stop_reason,
        **({"failure_kind": failure_record["failure_kind"]} if failure_record is not None else {}),
        **({"failure_artifact": str(run_dir / "architecture_failure.json")} if failure_record is not None else {}),
        "evaluated_items": len(evaluated_item_ids(run_dir)),
        "new_cost_usd": round(new_cost, 6),
        "cost": cost_summary,
        "changed_f1": (((summary.get("aggregate") or {}).get("metric_views") or {}).get("changed_fields") or {}).get("strict", {}).get("f1"),
    }
    log(log_path, f"[{architecture}] finished with status={status}: {stop_reason or 'complete'}")
    return result


def build_architecture_failure_record(
    *,
    architecture: str,
    batch_index: int,
    batch_ids: list[str],
    exc: BaseException,
    run_dir: Path,
) -> dict[str, Any]:
    solver_failures = collect_solver_failures(run_dir)
    failure_kind = classify_architecture_failure(
        architecture=architecture,
        exc=exc,
        solver_failures=solver_failures,
    )
    return {
        "architecture": architecture,
        "failure_kind": failure_kind,
        "batch_index": batch_index,
        "batch_item_ids": batch_ids,
        "exception_type": type(exc).__name__,
        "error": str(exc),
        "traceback": traceback.format_exc(),
        "solver_failure_artifacts": solver_failures,
        "policy": (
            "Architecture is marked failed and retained as an architecture-level "
            "valid-output/runtime failure. The comparison runner continues with "
            "subsequent architectures."
        ),
    }


def classify_architecture_failure(
    *,
    architecture: str,
    exc: BaseException,
    solver_failures: list[dict[str, Any]],
) -> str:
    message = str(exc)
    if architecture == "direct":
        direct_solver_failure = any(
            str(record.get("agent") or "").lower() == "direct"
            or "Direct baseline failed to produce a valid resulting-state JSON object"
            in str(record.get("error") or "")
            or "Direct baseline failed to produce a valid resulting-state JSON object"
            in str(record.get("last_error") or "")
            for record in solver_failures
        )
        direct_shape_failure = (
            "Direct baseline failed to produce a valid resulting-state JSON object" in message
            or "invalid resulting state" in message
            or "Resulting state shape errors" in message
            or "Response was not a valid JSON object" in message
        )
        if direct_solver_failure or direct_shape_failure:
            return "strict_json_output_failure"
    return "runtime_failure"


def collect_solver_failures(run_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(run_dir.rglob("solver_failure.json")):
        try:
            payload = read_json(path)
        except Exception as exc:
            records.append({
                "path": str(path.relative_to(run_dir)),
                "read_error": str(exc),
            })
            continue
        solver_failure = payload.get("solver_failure")
        if not isinstance(solver_failure, dict):
            solver_failure = {}
        records.append({
            "path": str(path.relative_to(run_dir)),
            "item_id": payload.get("item_id"),
            "questionnaire_id": payload.get("questionnaire_id"),
            "stage": payload.get("stage"),
            "exception_type": payload.get("exception_type"),
            "error": payload.get("error"),
            "agent": solver_failure.get("agent"),
            "attempts": solver_failure.get("attempts"),
            "last_error": solver_failure.get("last_error"),
            "rejected_attempt_count": len(solver_failure.get("rejected_attempts") or []),
        })
    return records


def finalize_run_artifacts(
    *,
    run_dir: Path,
    items_dir: Path,
    status: str,
    stop_reason: str,
    generate_figures: bool,
) -> dict[str, Any]:
    from scripts.compute_run_metrics import compute_run_metrics

    cost_summary = scan_costs(run_dir)
    if not any(run_dir.rglob("turn_result.json")):
        summary = empty_summary_report(
            run_dir=run_dir,
            cost_summary=cost_summary,
            status=status,
            stop_reason=stop_reason,
        )
        write_json(run_dir / "cost_report.json", cost_summary)
        return summary

    summary = compute_run_metrics(
        run_dir,
        items_dir=items_dir,
        generate_figures=generate_figures,
    )
    write_json(run_dir / "cost_report.json", cost_summary)
    patch_summary_cost(run_dir, cost_summary, status=status, stop_reason=stop_reason)
    if not generate_figures:
        return read_json(run_dir / "summary_report.json")
    try:
        from scripts.generate_full_run_report import generate_report_for_run_dir

        generate_report_for_run_dir(run_dir)
        patch_summary_cost(run_dir, cost_summary, status=status, stop_reason=stop_reason)
        append_cost_section(run_dir / "report.md", cost_summary, status, stop_reason)
    except Exception as exc:
        write_json(run_dir / "report_generation_error.json", {"error": str(exc)})
    return read_json(run_dir / "summary_report.json")


def empty_summary_report(
    *,
    run_dir: Path,
    cost_summary: dict[str, Any],
    status: str,
    stop_reason: str,
) -> dict[str, Any]:
    summary = {
        "run_id": run_dir.name,
        "aggregate": {
            "item_count": 0,
            "cost": cost_summary,
            "metric_views": {
                "changed_fields": {
                    "strict": {"precision": 0.0, "recall": 0.0, "f1": None},
                    "gold_changed_total": 0,
                },
            },
        },
        "items": [],
        "provenance": {
            "run_status": status,
            "stop_reason": stop_reason,
            "cost": cost_summary,
        },
    }
    write_json(run_dir / "summary_report.json", summary)
    write_json(run_dir / "metrics.json", summary)
    return summary


def scan_costs(run_dir: Path) -> dict[str, Any]:
    generation = empty_cost_bucket()
    evaluation = empty_cost_bucket()

    for turn_path in sorted(run_dir.rglob("turn_result.json")):
        turn = read_json(turn_path)
        agent_response = turn.get("agent_response") if isinstance(turn, dict) else {}
        calls = (agent_response or {}).get("model_calls") if isinstance(agent_response, dict) else None
        if isinstance(calls, list):
            for call in calls:
                add_cost_record(generation, call)

    for failure_path in sorted(run_dir.rglob("solver_failure.json")):
        failure = read_json(failure_path)
        solver_failure = failure.get("solver_failure") if isinstance(failure, dict) else None
        if not isinstance(solver_failure, dict):
            continue
        calls = solver_failure.get("model_calls")
        if isinstance(calls, list):
            for call in calls:
                add_cost_record(generation, call)

    for eval_path in sorted(run_dir.rglob("evaluation.json")):
        evaluation_payload = read_json(eval_path)
        for trace in evaluation_payload.get("prompt_trace") or []:
            if isinstance(trace, dict):
                add_cost_record(evaluation, trace)

    total = generation["cost_usd"] + evaluation["cost_usd"]
    return {
        "currency": "USD",
        "generation": generation,
        "evaluation": evaluation,
        "total_cost_usd": round(total, 8),
        "total_calls": generation["calls"] + evaluation["calls"],
        "missing_cost_calls": generation["missing_cost_calls"] + evaluation["missing_cost_calls"],
        "total_tokens": generation["total_tokens"] + evaluation["total_tokens"],
    }


def empty_cost_bucket() -> dict[str, Any]:
    return {
        "calls": 0,
        "cost_usd": 0.0,
        "provider_reported_cost_usd": 0.0,
        "estimated_cost_usd": 0.0,
        "provider_cost_calls": 0,
        "estimated_cost_calls": 0,
        "missing_cost_calls": 0,
        "prompt_tokens": 0,
        "cached_prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }


def add_cost_record(bucket: dict[str, Any], payload: Any) -> None:
    bucket["calls"] += 1
    usage = extract_usage(payload)
    cached_prompt_tokens = cached_input_tokens(payload)
    if usage is not None:
        bucket["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
        bucket["cached_prompt_tokens"] += cached_prompt_tokens
        bucket["completion_tokens"] += int(usage.get("completion_tokens") or 0)
        bucket["total_tokens"] += int(usage.get("total_tokens") or 0)

    cost = extract_cost(payload)
    if cost is None:
        estimated_cost = estimate_openai_cost_usd(
            payload,
            usage=usage,
            cached_prompt_tokens=cached_prompt_tokens,
        )
        if estimated_cost is None:
            bucket["missing_cost_calls"] += 1
        else:
            bucket["estimated_cost_calls"] += 1
            bucket["estimated_cost_usd"] = round(
                bucket["estimated_cost_usd"] + estimated_cost,
                8,
            )
            bucket["cost_usd"] = round(bucket["cost_usd"] + estimated_cost, 8)
    elif str(cost.get("currency") or "USD").upper() == "USD":
        provider_cost = float(cost.get("total") or 0.0)
        bucket["provider_cost_calls"] += 1
        bucket["provider_reported_cost_usd"] = round(
            bucket["provider_reported_cost_usd"] + provider_cost,
            8,
        )
        bucket["cost_usd"] = round(bucket["cost_usd"] + provider_cost, 8)
    else:
        bucket.setdefault("non_usd_costs", []).append(cost)


def estimate_openai_cost_usd(
    payload: Any,
    *,
    usage: dict[str, Any] | None = None,
    cached_prompt_tokens: int | None = None,
) -> float | None:
    usage = usage or extract_usage(payload)
    if usage is None:
        return None
    model = model_name_from_payload(payload)
    prices = token_prices_for_model(model)
    if prices is None:
        return None
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    cached_tokens = (
        cached_input_tokens(payload)
        if cached_prompt_tokens is None
        else max(0, int(cached_prompt_tokens))
    )
    cached_tokens = min(cached_tokens, prompt_tokens)
    uncached_tokens = prompt_tokens - cached_tokens
    return (
        uncached_tokens * prices["input"]
        + cached_tokens * prices["cached_input"]
        + completion_tokens * prices["output"]
    ) / 1_000_000.0


def token_prices_for_model(model: str | None) -> dict[str, float] | None:
    if not model:
        return None
    clean = model.strip().lower().split(":", 1)[-1]
    if clean.startswith("openai/"):
        clean = clean.removeprefix("openai/")
    for model_id in sorted(OPENAI_TOKEN_PRICES_USD_PER_MILLION, key=len, reverse=True):
        if clean == model_id or clean.startswith(model_id + "-"):
            return OPENAI_TOKEN_PRICES_USD_PER_MILLION[model_id]
    return None


def model_name_from_payload(payload: Any) -> str | None:
    if isinstance(payload, dict):
        model_config = payload.get("model_config")
        if isinstance(model_config, dict):
            for key in ("model", "resolved_model_name", "requested_model"):
                value = model_config.get(key)
                if isinstance(value, str) and value:
                    return value
        for key in ("model_name", "model"):
            value = payload.get(key)
            if isinstance(value, str) and value:
                return value
        for key in ("response", "response_metadata", "additional_kwargs", "llm_output"):
            found = model_name_from_payload(payload.get(key))
            if found:
                return found
        for child in payload.values():
            found = model_name_from_payload(child)
            if found:
                return found
    elif isinstance(payload, list):
        for child in payload:
            found = model_name_from_payload(child)
            if found:
                return found
    return None


def cached_input_tokens(payload: Any) -> int:
    if isinstance(payload, dict):
        for key in ("cache_read", "cached_tokens"):
            value = payload.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                return max(0, value)
        for key in (
            "input_token_details",
            "prompt_tokens_details",
            "usage",
            "usage_metadata",
            "token_usage",
            "response",
            "response_metadata",
            "raw",
        ):
            if key in payload:
                found = cached_input_tokens(payload.get(key))
                if found:
                    return found
        for child in payload.values():
            found = cached_input_tokens(child)
            if found:
                return found
    elif isinstance(payload, list):
        for child in payload:
            found = cached_input_tokens(child)
            if found:
                return found
    return 0


def load_item_packets(items_dir: Path) -> list[dict[str, Any]]:
    paths = sorted(items_dir.rglob("ground_truth.json"))
    packets = [read_json(path) for path in paths]
    packets.sort(key=lambda packet: (normalize_item(packet)["questionnaire_id"], normalize_item(packet)["item_id"]))
    return packets


def split_metadata_from_items_dir(items_dir: Path) -> dict[str, str] | None:
    parts = items_dir.parts
    if not parts:
        return None
    split = parts[-1]
    if split == "train":
        split = "dev"
    if split not in {"dev", "test"}:
        return None
    metadata = {"split": split}
    if len(parts) >= 3 and parts[-3] == "splits":
        metadata["split_id"] = parts[-2]
    return metadata


def run_purpose_from_args(
    args: argparse.Namespace,
    split_metadata: dict[str, str] | None,
) -> str | None:
    if args.run_purpose:
        return args.run_purpose
    if not split_metadata:
        return None
    if split_metadata.get("split") == "dev":
        return "dev_tuning"
    if split_metadata.get("split") == "test":
        return "selected_system"
    return None


def display_name_for_run(run_purpose: str | None, architecture: str) -> str | None:
    if not run_purpose:
        return None
    return f"{run_purpose} · {architecture}"


def item_with_model_metadata(
    item: dict[str, Any],
    *,
    model_id: str,
    reasoning_effort: str | None,
    direct_invalid_output_policy: str,
) -> dict[str, Any]:
    packet = deepcopy(item)
    packet["model_id"] = model_id
    if should_send_reasoning_effort(reasoning_effort):
        packet["model_reasoning_effort"] = reasoning_effort
    if direct_invalid_output_policy != "fail":
        packet["direct_invalid_output_policy"] = direct_invalid_output_policy
    return packet


def should_send_reasoning_effort(reasoning_effort: str | None) -> bool:
    if not reasoning_effort:
        return False
    return reasoning_effort.strip().lower() not in {"none", "off", "disabled", "false", "0"}


def build_plan(args: argparse.Namespace, items: list[dict[str, Any]]) -> dict[str, Any]:
    questionnaires = sorted({normalize_item(item)["questionnaire_id"] for item in items})
    return {
        "run_prefix": args.run_prefix,
        "items_dir": str(args.items_dir),
        "runs_dir": str(args.runs_dir),
        "item_count": len(items),
        "questionnaires": questionnaires,
        "architectures": args.architectures,
        "generator_model": args.generator_model,
        "generator_reasoning": args.generator_reasoning or "",
        "direct_invalid_output_policy": args.direct_invalid_output_policy,
        "evaluator_model": args.evaluator_model,
        "evaluator_reasoning": args.evaluator_reasoning,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "changed_f1_floor": args.changed_f1_floor,
        "cost_caps": {
            "total_usd": args.max_total_cost_usd,
            "architecture_usd": args.max_architecture_cost_usd,
            "batch_usd": args.max_batch_cost_usd,
        },
        "cost_tracking_max_tokens": args.cost_tracking_max_tokens,
    }


def preflight(args: argparse.Namespace, items: list[dict[str, Any]], log_path: Path) -> None:
    for architecture in args.architectures:
        import_path = ARCHITECTURES[architecture]
        module_name, attr = import_path.split(":", 1)
        module = importlib.import_module(module_name)
        if not callable(getattr(module, attr)):
            raise SystemExit(f"Solver is not callable: {import_path}")

    preflight_model_routing(
        args,
        log_path,
        require_api_keys=not args.plan_only,
    )

    if any(architecture.startswith("split_agent") for architecture in args.architectures) and not args.allow_split_generation:
        split_dir = DATA_ROOT / "splits"
        missing: list[str] = []
        for questionnaire in sorted({normalize_item(item)["questionnaire_id"] for item in items}):
            if not (split_dir / f"{questionnaire}_split.json").exists():
                missing.append(questionnaire)
        if missing:
            raise SystemExit(
                "SplitAgent split files are missing and --allow-split-generation is off: "
                + ", ".join(missing)
            )

    for architecture in args.architectures:
        run_dir = args.runs_dir / run_dir_name(args, architecture)
        if run_dir.exists() and not (args.resume or args.overwrite):
            raise SystemExit(
                f"Run directory already exists: {run_dir}. Use --resume or --overwrite."
            )

    log(log_path, "Preflight passed.")


def preflight_model_routing(
    args: argparse.Namespace,
    log_path: Path,
    *,
    require_api_keys: bool = True,
) -> None:
    from llm import resolve_chat_model_config

    checks = [
        ("generator", args.generator_model),
        ("evaluator", args.evaluator_model),
    ]
    for label, model_name in checks:
        config = resolve_chat_model_config(model=model_name)
        provider = str(config.get("provider") or "")
        resolved = str(config.get("model") or "")
        log(log_path, f"Preflight {label} model routing: {model_name} -> {provider}/{resolved}")
        if model_looks_openai_family(model_name) and provider == "anthropic":
            raise SystemExit(
                f"{label} model {model_name!r} resolved to the Anthropic backend. "
                "Set LLM_BACKEND=openai, or use an explicit provider/model that the backend supports."
            )
        if provider == "openai" and not os.getenv("OPENAI_API_KEY"):
            if require_api_keys:
                raise SystemExit(
                    f"{label} model {model_name!r} resolved to the official OpenAI backend, "
                    "but OPENAI_API_KEY is not set."
                )
            log(
                log_path,
                f"Preflight {label}: OPENAI_API_KEY is absent; allowed for plan-only mode.",
            )


def model_looks_openai_family(model_name: str | None) -> bool:
    if not model_name:
        return False
    clean = model_name.split(":", 1)[-1].split("/", 1)[-1].lower()
    return clean.startswith(("gpt-", "chatgpt-", "o1", "o3", "o4"))


def prepare_run_dir(run_dir: Path, args: argparse.Namespace) -> None:
    if run_dir.exists() and args.overwrite:
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)


def clear_stale_terminal_markers(run_dir: Path) -> None:
    for name in ("FAILED.md", "STOPPED_EARLY.md", "architecture_failure.json"):
        path = run_dir / name
        if path.exists():
            path.unlink()


def evaluated_item_ids(run_dir: Path) -> set[str]:
    ids: set[str] = set()
    for eval_path in run_dir.rglob("evaluation.json"):
        turn_path = eval_path.parent / "turn_result.json"
        if not turn_path.exists():
            continue
        try:
            ids.add(str(read_json(turn_path).get("scenario_id")))
        except Exception:
            continue
    return {item for item in ids if item}


def item_id(item: dict[str, Any]) -> str:
    return normalize_item(item)["item_id"]


def run_dir_name(args: argparse.Namespace, architecture: str) -> str:
    generator = safe_label(args.generator_model if architecture != "no_update" else "deterministic")
    evaluator = safe_label(args.evaluator_model)
    return f"{args.run_prefix}__{architecture}__gen-{generator}__judge-{evaluator}-{args.evaluator_reasoning}"


def safe_label(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in value).strip("-")


def patch_summary_cost(
    run_dir: Path,
    cost_summary: dict[str, Any],
    *,
    status: str,
    stop_reason: str,
) -> None:
    for name in ("summary_report.json", "metrics.json"):
        path = run_dir / name
        if not path.exists():
            continue
        payload = read_json(path)
        payload.setdefault("provenance", {})["cost"] = cost_summary
        payload.setdefault("provenance", {})["run_status"] = status
        if stop_reason:
            payload.setdefault("provenance", {})["stop_reason"] = stop_reason
        if name == "summary_report.json":
            payload.setdefault("aggregate", {})["cost"] = cost_summary
        write_json(path, payload)


def patch_failure_metadata(run_dir: Path, failure_record: dict[str, Any]) -> None:
    for name in ("summary_report.json", "metrics.json"):
        path = run_dir / name
        if not path.exists():
            continue
        payload = read_json(path)
        payload.setdefault("provenance", {})["failure"] = failure_record
        write_json(path, payload)


def append_cost_section(
    report_path: Path,
    cost_summary: dict[str, Any],
    status: str,
    stop_reason: str,
) -> None:
    if not report_path.exists():
        return
    text = report_path.read_text(encoding="utf-8").rstrip()
    section = [
        "",
        "## Run Cost",
        "",
        f"- Status: `{status}`",
        f"- Total observed LLM API cost: ${cost_summary['total_cost_usd']:.6f} USD",
        f"- Generation cost: ${cost_summary['generation']['cost_usd']:.6f} USD",
        f"- Evaluation cost: ${cost_summary['evaluation']['cost_usd']:.6f} USD",
        f"- LLM calls missing cost metadata: {cost_summary['missing_cost_calls']}",
    ]
    if stop_reason:
        section.append(f"- Stop reason: {stop_reason}")
    report_path.write_text(text + "\n" + "\n".join(section) + "\n", encoding="utf-8")


def write_stop_marker(run_dir: Path, *, status: str, stop_reason: str) -> None:
    if status != "stopped_early":
        return
    (run_dir / "STOPPED_EARLY.md").write_text(
        f"# Stopped Early\n\n{stop_reason}\n",
        encoding="utf-8",
    )


def write_failure_marker(
    run_dir: Path,
    *,
    status: str,
    stop_reason: str,
    failure_record: dict[str, Any] | None,
) -> None:
    if status != "failed":
        return
    lines = [
        "# Failed",
        "",
        stop_reason or "Architecture failed.",
    ]
    if failure_record is not None:
        lines.extend(
            [
                "",
                f"- Failure kind: `{failure_record.get('failure_kind')}`",
                f"- Batch: `{failure_record.get('batch_index')}`",
                f"- Failure artifact: `architecture_failure.json`",
            ]
        )
        solver_failures = failure_record.get("solver_failure_artifacts") or []
        if solver_failures:
            lines.append("- Solver failure artifacts:")
            for record in solver_failures[:10]:
                lines.append(f"  - `{record.get('path')}`: {record.get('error')}")
    (run_dir / "FAILED.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_comparison_markdown(path: Path, comparison: dict[str, Any]) -> None:
    lines = [
        f"# Architecture Comparison: {comparison['run_prefix']}",
        "",
        f"- Generator model: `{comparison['generator_model']}`",
        f"- Evaluator model: `{comparison['evaluator_model']}`",
        f"- Evaluator reasoning: `{comparison['evaluator_reasoning']}`",
        f"- Total observed cost: ${comparison.get('total_observed_cost_usd', 0):.6f} USD",
        "",
        "| Architecture | Status | Failure kind | Items | Changed F1 | New Cost | Stop reason |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for run in comparison.get("runs") or []:
        changed_f1 = run.get("changed_f1")
        changed_text = "" if changed_f1 is None else f"{float(changed_f1):.4f}"
        lines.append(
            "| {architecture} | {status} | {failure_kind} | {items} | {changed} | ${cost:.6f} | {reason} |".format(
                architecture=run.get("architecture", ""),
                status=run.get("status", ""),
                failure_kind=run.get("failure_kind", ""),
                items=run.get("evaluated_items", 0),
                changed=changed_text,
                cost=float(run.get("new_cost_usd", 0.0) or 0.0),
                reason=str(run.get("stop_reason") or "").replace("|", "\\|"),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def chunks(items: list[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def log(path: Path, message: str) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{timestamp}] {message}\n")
    print(message, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
