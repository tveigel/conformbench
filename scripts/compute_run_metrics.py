#!/usr/bin/env python3
"""Compute run-level diagnostics, metrics, and figures from scored artifacts.

This script is intentionally downstream of the evaluator.  It reads each
per-item ``turn_result.json`` plus ``evaluation.json`` pair, recomputes
diagnostics/metric views from the raw field verdicts, and writes run-level
derived artifacts:

* ``summary_report.json``
* ``metrics.json``
* ``figures/*.png`` when matplotlib/numpy are available
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from conformbench import benchmark
from conformbench.evaluator.diagnostics import (
    build_diagnostics,
    build_field_verdicts_with_gt_aliases,
    promote_matched_alignment_key_partials,
)
from conformbench.evaluator.metric_views import (
    aggregate_metric_views,
    compute_turn_metric_views,
)
from conformbench.evaluator.models import AlignmentEntry
from conformbench.evaluator.pipeline import (
    _MISSING,
    _ground_truth_entry,
    _rich_ground_truth_contract,
)
from conformbench.evaluator.run_turn import GroundTruthContract
from conformbench.items import DATA_ROOT


_REPEAT_QID_RE = re.compile(r"^(?P<group>\w+)\[(?P<idx>\d+|\?gt\d+)\]\.(?P<field>.+)$")


def _portable_data_path(path: Path) -> str:
    """Return a release-safe path without embedding a local home directory."""
    try:
        relative = path.resolve().relative_to(DATA_ROOT.resolve())
    except ValueError:
        return path.name
    return f"<CONFORMBENCH_DATA_DIR>/{relative.as_posix()}"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = read_json(path)
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def safe_div(numerator: float | int, denominator: float | int) -> float:
    return numerator / denominator if denominator else 0.0


def prf(tp: float, predicted: int, gold: int) -> dict[str, float]:
    precision = safe_div(tp, predicted)
    recall = safe_div(tp, gold)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def state_value(state: Mapping[str, Any] | None, slot: str) -> Any:
    if not isinstance(state, Mapping):
        return None
    if slot in state:
        return state.get(slot)

    match = _REPEAT_QID_RE.match(slot)
    if not match:
        return None

    idx_token = match.group("idx")
    if idx_token.startswith("?gt"):
        idx = int(idx_token[3:])
    else:
        idx = int(idx_token)

    rows = state.get(match.group("group"))
    if not isinstance(rows, list) or idx >= len(rows):
        return None
    row = rows[idx]
    if not isinstance(row, Mapping):
        return None
    return row.get(match.group("field"))


def state_value_or_missing(state: Mapping[str, Any] | None, slot: str) -> Any:
    if not isinstance(state, Mapping):
        return _MISSING
    if slot in state:
        return state.get(slot)

    match = _REPEAT_QID_RE.match(slot)
    if not match:
        return _MISSING
    idx_token = match.group("idx")
    idx = int(idx_token[3:] if idx_token.startswith("?gt") else idx_token)
    rows = state.get(match.group("group"))
    if not isinstance(rows, list) or idx >= len(rows):
        return _MISSING
    row = rows[idx]
    if not isinstance(row, Mapping) or match.group("field") not in row:
        return _MISSING
    return row.get(match.group("field"))


def _normalise_field_results(raw: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(raw, dict):
        return {}
    results: dict[str, dict[str, Any]] = {}
    for qid, verdict in raw.items():
        if isinstance(verdict, dict):
            results[str(qid)] = dict(verdict)
    return results


def _normalise_alignment_log(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    entries: list[dict[str, Any]] = []
    for entry in raw:
        if isinstance(entry, dict):
            entries.append(dict(entry))
    return entries


def _ensure_metric_contract(
    gt: dict[str, Any],
    *,
    field_results: dict[str, dict[str, Any]],
    candidate_state: dict[str, Any],
    questionnaire_path: Path | None,
    alignment_log: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Add lightweight field/repeat metadata for public packets.

    Rich benchmark ground truths already contain field policies and repeat
    contracts, so they pass through unchanged.  Public/studio packets usually
    only contain prior/gold states; for those we synthesize enough metadata for
    the original metric-view code to operate from the raw values.
    """
    if gt.get("fields") or gt.get("repeat_groups"):
        return gt, alignment_log

    prior_state = gt.get("prior_state") or {}
    gold_state = gt.get("gold_resulting_state") or {}
    scalar_fields: dict[str, dict[str, Any]] = {}
    repeat_groups: dict[str, dict[str, Any]] = {}
    repeat_field_names: dict[str, dict[int, set[str]]] = defaultdict(lambda: defaultdict(set))
    extra_agent_indices: dict[str, set[int]] = defaultdict(set)

    for qid in field_results:
        match = _REPEAT_QID_RE.match(qid)
        if not match:
            prior = state_value_or_missing(prior_state, qid)
            expected = state_value(gold_state, qid)
            scalar_fields[qid] = _ground_truth_entry(
                qid,
                expected=expected,
                prior_value=prior,
                questionnaire_path=questionnaire_path,
            )
            continue

        group = match.group("group")
        idx_token = match.group("idx")
        field = match.group("field")
        if idx_token.startswith("?gt"):
            idx = int(idx_token[3:])
        else:
            idx = int(idx_token)
        gold_rows = gold_state.get(group)
        if isinstance(gold_rows, list) and idx < len(gold_rows):
            repeat_field_names[group][idx].add(field)
        else:
            extra_agent_indices[group].add(idx)

    for group, fields_by_index in repeat_field_names.items():
        instances: list[dict[str, Any]] = []
        for idx in sorted(fields_by_index):
            field_entries: dict[str, dict[str, Any]] = {}
            for field in sorted(fields_by_index[idx]):
                qid = f"{group}[{idx}].{field}"
                prior = state_value_or_missing(prior_state, qid)
                expected = state_value(gold_state, qid)
                field_entries[field] = _ground_truth_entry(
                    qid,
                    expected=expected,
                    prior_value=prior,
                    questionnaire_path=questionnaire_path,
                )
            instances.append({
                "ground_truth_index": idx,
                "identity_anchor": None,
                "fields": field_entries,
            })
        repeat_groups[group] = {"alignment_keys": [], "instances": instances}

    if not alignment_log:
        synthesized_alignment: list[dict[str, Any]] = []
        for group, group_contract in repeat_groups.items():
            for instance in group_contract.get("instances") or []:
                gt_index = int(instance["ground_truth_index"])
                synthesized_alignment.append({
                    "group": group,
                    "gt_index": gt_index,
                    "agent_index": gt_index,
                    "status": "matched",
                    "matched_on": "index",
                })
            for agent_index in sorted(extra_agent_indices.get(group, set())):
                synthesized_alignment.append({
                    "group": group,
                    "gt_index": -1,
                    "agent_index": agent_index,
                    "status": "hallucinated",
                    "matched_on": "",
                })
        alignment_log = synthesized_alignment

    gt["fields"] = scalar_fields
    gt["repeat_groups"] = repeat_groups
    return gt, alignment_log


def _gold_value(gt: Mapping[str, Any], qid: str) -> Any:
    value = state_value(gt.get("gold_resulting_state") or {}, qid)
    if value is not None:
        return value
    match = _REPEAT_QID_RE.match(qid)
    if match:
        group = match.group("group")
        field = match.group("field")
        idx_token = match.group("idx")
        gt_idx = int(idx_token[3:] if idx_token.startswith("?gt") else idx_token)
        group_contract = (gt.get("repeat_groups") or {}).get(group) or {}
        for instance in group_contract.get("instances") or []:
            if instance.get("ground_truth_index") != gt_idx:
                continue
            entry = (instance.get("fields") or {}).get(field) or {}
            return entry.get("expected_summary", entry.get("expected"))
    entry = (gt.get("fields") or {}).get(qid) or {}
    return entry.get("expected_summary", entry.get("expected"))


def _field_counts(field_results: Mapping[str, Mapping[str, Any]]) -> tuple[dict[str, int], dict[str, int]]:
    scalar = {"correct": 0, "partial": 0, "incorrect": 0, "total": 0}
    repeat = {"correct": 0, "partial": 0, "incorrect": 0, "total": 0}
    for qid, verdict in field_results.items():
        bucket = repeat if _REPEAT_QID_RE.match(qid) else scalar
        correctness = verdict.get("correctness") or verdict.get("final_correctness")
        if correctness == "partially_correct":
            key = "partial"
        elif correctness in {"correct", "incorrect"}:
            key = str(correctness)
        else:
            key = "incorrect"
        bucket[key] += 1
        bucket["total"] += 1
    return scalar, repeat


def _attempted_scores(field_results: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    counts = Counter()
    for qid, verdict in field_results.items():
        if "[?gt" in qid:
            continue
        correctness = verdict.get("correctness") or verdict.get("final_correctness")
        if correctness == "partially_correct":
            counts["partially_correct"] += 1
        elif correctness == "correct":
            counts["correct"] += 1
        else:
            counts["incorrect"] += 1
    evaluated = counts["correct"] + counts["partially_correct"] + counts["incorrect"]
    return {
        "evaluated": evaluated,
        "correct": counts["correct"],
        "partially_correct": counts["partially_correct"],
        "incorrect": counts["incorrect"],
        "accuracy": round(safe_div(counts["correct"], evaluated), 4),
        "lenient_accuracy": round(
            safe_div(counts["correct"] + counts["partially_correct"], evaluated),
            4,
        ),
    }


def _source_counts(field_results: Mapping[str, Mapping[str, Any]]) -> dict[str, int]:
    counts = Counter()
    for verdict in field_results.values():
        source = verdict.get("support_source") or verdict.get("source") or "unknown"
        if source == "made_up":
            source = "fabricated"
        counts[str(source)] += 1
    return dict(sorted(counts.items()))


def _error_rates(
    field_results: Mapping[str, Mapping[str, Any]],
    *,
    evaluated: int,
) -> dict[str, float]:
    hallucination = 0
    unsupported = 0
    omission = 0
    for qid, verdict in field_results.items():
        correctness = verdict.get("correctness") or verdict.get("final_correctness")
        partial_reason = verdict.get("partial_reason")
        source = verdict.get("support_source") or verdict.get("source")
        if partial_reason == "hallucination" or source == "fabricated":
            hallucination += 1
        if source == "unsupported_inference":
            unsupported += 1
        if partial_reason == "omission" or "[?gt" in qid:
            omission += 1
        if correctness == "incorrect" and "[?gt" in qid:
            omission += 1
    return {
        "hallucination_rate": round(safe_div(hallucination, evaluated), 4),
        "unsupported_inference_rate": round(safe_div(unsupported, evaluated), 4),
        "omission_rate": round(safe_div(omission, evaluated), 4),
    }


def _instance_alignment(alignment_log: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_group: dict[str, Counter[str]] = defaultdict(Counter)
    for entry in alignment_log:
        group = str(entry.get("group") or "_unknown")
        status = str(entry.get("status") or "")
        if status:
            by_group[group][status] += 1
    rows: list[dict[str, Any]] = []
    for group, counts in sorted(by_group.items()):
        matched = counts.get("matched", 0)
        missed = counts.get("missed", 0)
        hallucinated = counts.get("hallucinated", 0)
        precision = safe_div(matched, matched + hallucinated)
        recall = safe_div(matched, matched + missed)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({
            "group": group,
            "matched": matched,
            "missed": missed,
            "hallucinated": hallucinated,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        })
    return rows


def _mismatches(
    gt: Mapping[str, Any],
    candidate_state: Mapping[str, Any],
    field_results: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for qid, verdict in sorted(field_results.items()):
        correctness = verdict.get("correctness") or verdict.get("final_correctness")
        if correctness in {"correct", None, ""}:
            continue
        rows.append({
            "qid": qid,
            "verdict": correctness,
            "source": verdict.get("support_source") or verdict.get("source"),
            "decision_source": verdict.get("decision_source"),
            "partial_reason": verdict.get("partial_reason"),
            "expected": _gold_value(gt, qid),
            "actual": state_value(candidate_state, qid),
            "reasoning": verdict.get("reasoning") or "",
        })
    return rows


def _dimension_data(gt: Mapping[str, Any]) -> dict[str, Any]:
    difficulty_profile = gt.get("difficulty_profile") or {}
    dimensions = difficulty_profile.get("dimensions") or difficulty_profile.get("stressors") or {}
    if not isinstance(dimensions, dict):
        dimensions = {}
    return dict(dimensions)


def _counter_breakdown(
    utterances: list[dict[str, Any]],
    value_for_utterance,
) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for utterance in utterances:
        value = value_for_utterance(utterance)
        if isinstance(value, bool):
            key = "yes" if value else "no"
        elif value is None or value == "":
            key = "unknown"
        else:
            key = str(value)
        buckets[key].append(utterance)
    return {
        "counts": {key: len(rows) for key, rows in sorted(buckets.items())},
        "breakdowns": {
            key: {
                "utterance_count": len(rows),
                "metric_views": aggregate_metric_views([row.get("metric_views") or {} for row in rows]),
            }
            for key, rows in sorted(buckets.items())
        },
    }


def _aggregate_source_counts(utterances: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter()
    for utterance in utterances:
        counts.update(utterance.get("source_counts") or {})
    return dict(sorted(counts.items()))


def _aggregate_partial_reasons(utterances: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter()
    for utterance in utterances:
        for mismatch in utterance.get("mismatches") or []:
            if mismatch.get("verdict") == "partially_correct" and mismatch.get("partial_reason"):
                counts[str(mismatch["partial_reason"])] += 1
    return dict(sorted(counts.items()))


def _aggregate_instance_alignment(utterances: list[dict[str, Any]]) -> dict[str, float]:
    matched = missed = hallucinated = 0
    for utterance in utterances:
        for row in utterance.get("instance_alignment") or []:
            matched += int(row.get("matched", 0) or 0)
            missed += int(row.get("missed", 0) or 0)
            hallucinated += int(row.get("hallucinated", 0) or 0)
    precision = safe_div(matched, matched + hallucinated)
    recall = safe_div(matched, matched + missed)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def _aggregate_error_rates(utterances: list[dict[str, Any]], total_fields: int) -> dict[str, float]:
    numerators = Counter()
    for utterance in utterances:
        evaluated = int((utterance.get("scores") or {}).get("evaluated", 0) or 0)
        rates = utterance.get("error_rates") or {}
        numerators["hallucination"] += round(float(rates.get("hallucination_rate", 0) or 0) * evaluated)
        numerators["unsupported"] += round(float(rates.get("unsupported_inference_rate", 0) or 0) * evaluated)
        numerators["omission"] += round(float(rates.get("omission_rate", 0) or 0) * evaluated)
    return {
        "hallucination_rate": round(safe_div(numerators["hallucination"], total_fields), 4),
        "unsupported_inference_rate": round(safe_div(numerators["unsupported"], total_fields), 4),
        "omission_rate": round(safe_div(numerators["omission"], total_fields), 4),
    }


def _aggregate_agent_runtime(utterances: list[dict[str, Any]]) -> dict[str, Any]:
    totals = Counter()
    for utterance in utterances:
        runtime = utterance.get("agent_runtime") or {}
        totals["operation_count"] += int(runtime.get("operation_count", 0) or 0)
        totals["tool_call_count"] += int(runtime.get("tool_call_count", 0) or 0)
        totals["tool_error_count"] += int(runtime.get("tool_error_count", 0) or 0)
        totals["tool_rejected_update_count"] += int(runtime.get("tool_rejected_update_count", 0) or 0)
        totals["empty_tool_update_call_count"] += int(runtime.get("empty_tool_update_call_count", 0) or 0)
        if runtime.get("zero_operation"):
            totals["zero_operation_item_count"] += 1
        if runtime.get("zero_operation_with_tool_error"):
            totals["zero_operation_with_tool_error_count"] += 1
        if runtime.get("zero_operation_without_tool_call"):
            totals["zero_operation_without_tool_call_count"] += 1
        if runtime.get("tool_error_count"):
            totals["items_with_tool_errors"] += 1

    return {
        "item_count": len(utterances),
        "operation_count": totals["operation_count"],
        "tool_call_count": totals["tool_call_count"],
        "tool_error_count": totals["tool_error_count"],
        "items_with_tool_errors": totals["items_with_tool_errors"],
        "tool_rejected_update_count": totals["tool_rejected_update_count"],
        "empty_tool_update_call_count": totals["empty_tool_update_call_count"],
        "zero_operation_item_count": totals["zero_operation_item_count"],
        "zero_operation_with_tool_error_count": totals["zero_operation_with_tool_error_count"],
        "zero_operation_without_tool_call_count": totals["zero_operation_without_tool_call_count"],
    }


def _build_aggregate(utterances: list[dict[str, Any]]) -> dict[str, Any]:
    metric_views = aggregate_metric_views([utterance.get("metric_views") or {} for utterance in utterances])
    all_fields = metric_views.get("all_fields") or {}
    total_fields = int(all_fields.get("total_evaluated_leaf_fields", 0) or 0)
    correct = int(all_fields.get("correct", 0) or 0)
    partial = int(all_fields.get("partial", 0) or 0)
    incorrect = int(all_fields.get("incorrect", 0) or 0)
    gt_expected = int(all_fields.get("gt_expected_total", total_fields) or 0)

    questionnaires: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for utterance in utterances:
        questionnaires[str(utterance.get("questionnaire") or "unknown")].append(utterance)

    return {
        "total_fields": total_fields,
        "total_gt_expected": gt_expected,
        "correct": correct,
        "partially_correct": partial,
        "incorrect": incorrect,
        "unmatched": sum(1 for u in utterances for m in u.get("mismatches") or [] if "[?gt" in m.get("qid", "")),
        "accuracy": round(safe_div(correct, total_fields), 4),
        "lenient_accuracy": round(safe_div(correct + partial, total_fields), 4),
        "micro_prf": {
            "strict": prf(correct, total_fields, gt_expected),
            "lenient": prf(correct + partial, total_fields, gt_expected),
            "weighted": prf(correct + 0.5 * partial, total_fields, gt_expected),
        },
        "macro_prf": {
            "strict": (all_fields.get("macro") or {}).get("strict") or {"precision": 0.0, "recall": 0.0, "f1": 0.0},
            "lenient": (all_fields.get("macro") or {}).get("lenient") or {"precision": 0.0, "recall": 0.0, "f1": 0.0},
            "weighted": (all_fields.get("macro") or {}).get("weighted") or {"precision": 0.0, "recall": 0.0, "f1": 0.0},
        },
        "instance_alignment": _aggregate_instance_alignment(utterances),
        "error_rates": _aggregate_error_rates(utterances, total_fields),
        "source_counts": _aggregate_source_counts(utterances),
        "partial_reasons": _aggregate_partial_reasons(utterances),
        "scalar_vs_repeat": {
            "scalar": _sum_score_blocks(utterances, "scalar"),
            "repeat_group": _sum_score_blocks(utterances, "repeat_group"),
        },
        "attempted_scores": _sum_attempted_scores(utterances),
        "agent_runtime": _aggregate_agent_runtime(utterances),
        "metric_views": metric_views,
        "questionnaires": {
            "counts": {q: len(rows) for q, rows in sorted(questionnaires.items())},
            "breakdowns": {
                q: {
                    "utterance_count": len(rows),
                    "metric_views": aggregate_metric_views([row.get("metric_views") or {} for row in rows]),
                }
                for q, rows in sorted(questionnaires.items())
            },
        },
        "derived_variables": _derived_variable_counts(utterances),
        "item_stressors": _item_stressor_counts(utterances),
        "human_dimensions": _item_stressor_counts(utterances),
        "targeted_failure_modes": _targeted_failure_mode_counts(utterances),
    }


def _sum_score_blocks(utterances: list[dict[str, Any]], key: str) -> dict[str, int]:
    totals = {"correct": 0, "partial": 0, "incorrect": 0, "total": 0}
    for utterance in utterances:
        block = ((utterance.get("scalar_vs_repeat") or {}).get(key) or {})
        for field in totals:
            totals[field] += int(block.get(field, 0) or 0)
    return totals


def _sum_attempted_scores(utterances: list[dict[str, Any]]) -> dict[str, Any]:
    totals = {
        "evaluated": 0,
        "correct": 0,
        "partially_correct": 0,
        "incorrect": 0,
    }
    for utterance in utterances:
        block = utterance.get("attempted_scores") or {}
        for key in totals:
            totals[key] += int(block.get(key, 0) or 0)
    totals["accuracy"] = round(safe_div(totals["correct"], totals["evaluated"]), 4)
    totals["lenient_accuracy"] = round(
        safe_div(totals["correct"] + totals["partially_correct"], totals["evaluated"]),
        4,
    )
    return totals


def _derived_variable_counts(utterances: list[dict[str, Any]]) -> dict[str, Any]:
    keys = sorted({
        key
        for utterance in utterances
        for key in (utterance.get("derived_variables") or {})
    })
    return {
        key: _counter_breakdown(
            utterances,
            lambda utterance, key=key: (utterance.get("derived_variables") or {}).get(key),
        )
        for key in keys
    }


def _item_stressor_counts(utterances: list[dict[str, Any]]) -> dict[str, Any]:
    keys = sorted({
        key
        for utterance in utterances
        for key in (utterance.get("item_stressors") or {})
    })
    return {
        key: _counter_breakdown(
            utterances,
            lambda utterance, key=key: (utterance.get("item_stressors") or {}).get(key),
        )
        for key in keys
    }


def _targeted_failure_mode_counts(utterances: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter()
    for utterance in utterances:
        for mode in utterance.get("targeted_failure_modes") or []:
            counts[str(mode)] += 1
    return dict(sorted(counts.items()))


def _compact_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    provenance = summary.get("provenance") or {}
    compact = {
        "generated_at": summary.get("generated_at"),
        "run_dir": provenance.get("run_dir"),
        "aggregate": (summary.get("aggregate") or {}).get("metric_views") or {},
        "utterances": [
            {
                "scenario_id": utterance.get("scenario_id"),
                "questionnaire": utterance.get("questionnaire"),
                "state": utterance.get("state"),
                "utterance_id": utterance.get("utterance_id"),
                "scores": utterance.get("scores"),
                "metric_views": utterance.get("metric_views"),
                "diagnostics": utterance.get("diagnostics"),
            }
            for utterance in summary.get("utterances") or []
        ],
    }
    metadata = {
        key: provenance[key]
        for key in ("run_metadata", "generation", "evaluation")
        if provenance.get(key) is not None
    }
    if metadata:
        compact["metadata"] = metadata
    return compact


def _first_artifact_provenance(paths: list[Path], artifact_name: str) -> dict[str, Any]:
    for path in paths:
        if path.name != artifact_name or not path.exists():
            continue
        try:
            payload = read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        provenance = payload.get("provenance")
        if isinstance(provenance, dict):
            return provenance
    return {}


def compute_run_metrics(
    run_dir: str | Path,
    *,
    items_dir: str | Path | None = None,
    generate_figures: bool = True,
) -> dict[str, Any]:
    run_root = Path(run_dir).resolve()
    if not run_root.exists():
        raise FileNotFoundError(f"Run directory not found: {run_root}")

    turn_paths = sorted(run_root.rglob("turn_result.json"))
    if not turn_paths:
        raise FileNotFoundError(f"No turn_result.json files found under {run_root}")

    run_metadata = read_optional_json(run_root / "run_metadata.json")
    first_turn_provenance = _first_artifact_provenance(turn_paths, "turn_result.json")
    eval_paths = [turn_path.parent / "evaluation.json" for turn_path in turn_paths]
    first_eval_provenance = _first_artifact_provenance(eval_paths, "evaluation.json")

    utterances: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    items_root = Path(items_dir).resolve() if items_dir else None

    for turn_path in turn_paths:
        eval_path = turn_path.parent / "evaluation.json"
        if not eval_path.exists():
            errors.append({"path": str(turn_path), "error": "missing evaluation.json"})
            continue

        try:
            utterances.append(
                _compute_utterance_summary(
                    turn_path=turn_path,
                    eval_path=eval_path,
                    run_root=run_root,
                    items_dir=items_root,
                )
            )
        except Exception as exc:
            errors.append({"path": str(turn_path), "error": str(exc)})

    if not utterances:
        raise RuntimeError(f"No scored items could be summarized under {run_root}: {errors}")

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "run_dir": _portable_data_path(run_root),
            "source": "turn_result.json + evaluation.json",
            "metrics_source": "conformbench.evaluator.metric_views",
            "diagnostics_source": "conformbench.evaluator.diagnostics",
        },
        "aggregate": _build_aggregate(utterances),
        "utterances": utterances,
    }
    if run_metadata:
        summary["provenance"]["run_metadata"] = run_metadata
    generation = first_turn_provenance.get("generation")
    if generation is not None:
        summary["provenance"]["generation"] = generation
    evaluation = first_eval_provenance.get("evaluation")
    if evaluation is not None:
        summary["provenance"]["evaluation"] = evaluation
    runner = first_turn_provenance.get("runner") or first_eval_provenance.get("runner")
    if runner is not None:
        summary["provenance"]["runner"] = runner
    if errors:
        summary["errors"] = errors

    summary_path = run_root / "summary_report.json"
    metrics_path = run_root / "metrics.json"
    write_json(summary_path, summary)
    write_json(metrics_path, _compact_metrics(summary))

    if generate_figures:
        try:
            from conformbench.evaluator.visualize import generate_figures as _generate_figures

            figures = _generate_figures(summary, run_root / "figures")
            summary["provenance"]["figures"] = [str(path) for path in figures]
            write_json(summary_path, summary)
        except Exception as exc:
            summary["provenance"]["figure_generation_error"] = str(exc)
            write_json(summary_path, summary)

    return summary


def _compute_utterance_summary(
    *,
    turn_path: Path,
    eval_path: Path,
    run_root: Path,
    items_dir: Path | None,
) -> dict[str, Any]:
    turn_result = read_json(turn_path)
    evaluation = read_json(eval_path)
    ground_truth = benchmark._resolve_ground_truth_for_turn(turn_result, items_dir=items_dir)
    gt = _rich_ground_truth_contract(ground_truth)
    questionnaire = str(
        turn_result.get("questionnaire")
        or gt.get("questionnaire")
        or ground_truth.get("questionnaire_id")
        or ""
    )
    questionnaire_path = benchmark._questionnaire_path(questionnaire)

    field_results = _normalise_field_results(evaluation.get("field_results") or {})
    candidate_state = evaluation.get("candidate_resulting_state") or turn_result.get("answers_after") or {}
    if not isinstance(candidate_state, dict):
        candidate_state = {}
    alignment_log = _normalise_alignment_log(evaluation.get("alignment_log") or [])
    gt, alignment_log = _ensure_metric_contract(
        gt,
        field_results=field_results,
        candidate_state=candidate_state,
        questionnaire_path=questionnaire_path,
        alignment_log=alignment_log,
    )

    promoted_results = promote_matched_alignment_key_partials(
        field_results,
        candidate_state=candidate_state,
        alignment_log=alignment_log,
        gt_repeat_groups=gt.get("repeat_groups") or {},
        questionnaire_path=questionnaire_path,
    )
    field_verdicts = build_field_verdicts_with_gt_aliases(promoted_results, alignment_log)
    gt_contract = GroundTruthContract.model_validate(gt)
    alignment_entries = [AlignmentEntry.model_validate(entry) for entry in alignment_log]
    diagnostics = build_diagnostics(
        gt=gt_contract,
        candidate_state=candidate_state,
        alignment_log=alignment_entries,
        field_verdicts=field_verdicts,
        questionnaire_path=questionnaire_path,
    )
    metric_views = compute_turn_metric_views(
        gt=gt,
        candidate_state=candidate_state,
        field_results=promoted_results,
        alignment_log=alignment_log,
        diagnostics=diagnostics,
        questionnaire_path=questionnaire_path,
    )

    all_fields = metric_views.get("all_fields") or {}
    scalar_scores, repeat_scores = _field_counts(promoted_results)
    evaluated = int(all_fields.get("total_evaluated_leaf_fields", 0) or 0)
    scores = {
        "total": evaluated,
        "evaluated": evaluated,
        "gt_expected": int(all_fields.get("gt_expected_total", evaluated) or 0),
        "correct": int(all_fields.get("correct", 0) or 0),
        "partially_correct": int(all_fields.get("partial", 0) or 0),
        "incorrect": int(all_fields.get("incorrect", 0) or 0),
        "accuracy": float(all_fields.get("accuracy", 0.0) or 0.0),
        "lenient_accuracy": float(all_fields.get("lenient_accuracy", 0.0) or 0.0),
    }
    derived_variables = dict(gt.get("derived_variables") or {})
    if not derived_variables:
        derived_variables = _fallback_derived_variables(gt)

    difficulty_profile = gt.get("difficulty_profile") or {}
    targeted_failure_modes = list(gt.get("targeted_failure_mode") or [])
    for mode in difficulty_profile.get("targeted_failure_modes") or []:
        if mode not in targeted_failure_modes:
            targeted_failure_modes.append(mode)

    return {
        "questionnaire": questionnaire,
        "scenario": turn_result.get("scenario") or gt.get("scenario_label") or gt.get("scenario_id") or "",
        "scenario_id": turn_result.get("scenario_id") or gt.get("scenario_id") or "",
        "state": turn_result.get("state") or "",
        "utterance_id": turn_result.get("utterance_id") or "",
        "difficulty": turn_result.get("difficulty") or gt.get("difficulty") or gt.get("difficulty_tier") or "",
        "artifact_dir": str(turn_path.parent.relative_to(run_root)),
        "user_message": evaluation.get("user_message") or turn_result.get("current_utterance") or "",
        "scores": scores,
        "field_prf": {
            "strict": all_fields.get("strict") or {},
            "lenient": all_fields.get("lenient") or {},
            "weighted": all_fields.get("weighted") or {},
        },
        "error_rates": _error_rates(promoted_results, evaluated=evaluated),
        "source_counts": _source_counts(promoted_results),
        "instance_alignment": _instance_alignment(alignment_log),
        "scalar_vs_repeat": {
            "scalar": scalar_scores,
            "repeat_group": repeat_scores,
        },
        "attempted_scores": _attempted_scores(promoted_results),
        "agent_runtime": _agent_runtime_diagnostics(turn_result),
        "metric_views": metric_views,
        "diagnostics": diagnostics,
        "derived_variables": derived_variables,
        "item_stressors": _dimension_data(gt),
        "item_stressor_notes": (difficulty_profile.get("dimension_notes") or {}),
        "targeted_failure_modes": targeted_failure_modes,
        "human_dimensions": _dimension_data(gt),
        "human_dimension_notes": (difficulty_profile.get("dimension_notes") or {}),
        "mismatches": _mismatches(gt, candidate_state, promoted_results),
    }


def _agent_runtime_diagnostics(turn_result: Mapping[str, Any]) -> dict[str, Any]:
    agent_response = turn_result.get("agent_response") if isinstance(turn_result, Mapping) else {}
    if not isinstance(agent_response, Mapping):
        agent_response = {}
    operations = turn_result.get("operations") if isinstance(turn_result, Mapping) else []
    if not isinstance(operations, list):
        operations = agent_response.get("operations")
    if not isinstance(operations, list):
        updates = turn_result.get("answer_updates") if isinstance(turn_result, Mapping) else {}
        operations = list(updates) if isinstance(updates, Mapping) else []
    operation_count = len(operations)

    tool_updates = agent_response.get("tool_updates")
    if not isinstance(tool_updates, list):
        tool_updates = []

    tool_error_count = 0
    tool_rejected_update_count = 0
    empty_tool_update_call_count = 0
    tool_errors: list[dict[str, Any]] = []

    for index, entry in enumerate(tool_updates):
        if not isinstance(entry, Mapping):
            continue
        updates = entry.get("updates")
        if isinstance(updates, Mapping) and not updates:
            empty_tool_update_call_count += 1
        elif updates in ({}, None):
            empty_tool_update_call_count += 1

        result = entry.get("result")
        if not isinstance(result, Mapping):
            result = {}

        rejected = entry.get("rejected")
        if not isinstance(rejected, list):
            rejected = result.get("rejected")
        rejected_count = len(rejected) if isinstance(rejected, list) else 0
        tool_rejected_update_count += rejected_count

        errors = []
        for key in ("errors", "shape_errors"):
            values = result.get(key)
            if isinstance(values, list):
                errors.extend(str(value) for value in values)
            elif values:
                errors.append(str(values))
        status = result.get("status")
        is_error = status == "error" or bool(errors) or rejected_count > 0
        if is_error:
            tool_error_count += 1
            tool_errors.append({
                "tool_call_index": index,
                "partition": entry.get("partition"),
                "status": status,
                "rejected_count": rejected_count,
                "errors": errors[:5],
            })

    return {
        "operation_count": operation_count,
        "zero_operation": operation_count == 0,
        "tool_call_count": len(tool_updates),
        "tool_error_count": tool_error_count,
        "tool_rejected_update_count": tool_rejected_update_count,
        "empty_tool_update_call_count": empty_tool_update_call_count,
        "zero_operation_with_tool_error": operation_count == 0 and tool_error_count > 0,
        "zero_operation_without_tool_call": operation_count == 0 and not tool_updates,
        "tool_errors": tool_errors,
    }


def _fallback_derived_variables(gt: Mapping[str, Any]) -> dict[str, Any]:
    evidence = gt.get("evidence") or {}
    primary_delta_type = (
        gt.get("primary_delta_type")
        or gt.get("family")
        or gt.get("delta_type")
        or "unknown"
    )
    repeat_names = [
        key
        for key, value in (gt.get("gold_resulting_state") or {}).items()
        if isinstance(value, list) and any(isinstance(row, Mapping) for row in value)
    ]
    return {
        "primary_delta_type": primary_delta_type,
        "prior_state_condition": gt.get("state_condition") or "unknown",
        "history_required": bool(evidence.get("history_required", gt.get("visible_history"))),
        "conflict_present": bool(evidence.get("conflict_present", False)),
        "repeat_group_involved": bool(repeat_names),
        "repeat_group_involvement": "yes" if repeat_names else "none",
        "repeat_group_names": repeat_names,
        "revision_operation": primary_delta_type,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--items-dir", type=Path)
    parser.add_argument("--no-figures", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = compute_run_metrics(
        args.run_dir,
        items_dir=args.items_dir,
        generate_figures=not args.no_figures,
    )
    run_dir = Path(args.run_dir).resolve()
    print(f"Wrote {run_dir / 'summary_report.json'}")
    print(f"Wrote {run_dir / 'metrics.json'}")
    figures = sorted((run_dir / "figures").glob("*.png"))
    if figures:
        print(f"Wrote {len(figures)} figure(s) to {run_dir / 'figures'}")
    elif (summary.get("provenance") or {}).get("figure_generation_error"):
        print(
            "Skipped figures: "
            f"{(summary.get('provenance') or {}).get('figure_generation_error')}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
