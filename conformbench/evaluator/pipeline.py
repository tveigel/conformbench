"""Artifact-level evaluation pipeline for public runner outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .hard import evaluate_hard_fields, get_questionnaire_field_info
from .models import FieldVerdict
from .run_turn import TurnEvaluationResult, derive_gold_state_diff, evaluate_turn_result
from .scoring import (
    infer_scoring_profile,
    is_empty_for_scoring,
    values_equal_for_scoring,
)
from ..items import normalize_item

_MISSING = object()


def evaluate_solver_turn(
    turn_result: dict[str, Any],
    ground_truth: dict[str, Any],
    *,
    model_id: str | None = None,
    reasoning_effort: str | None = None,
    questionnaire_path: Path | None = None,
    runner_version: dict[str, Any] | None = None,
) -> TurnEvaluationResult:
    """Evaluate one solver turn against its ground-truth item packet.

    Rich internal benchmark items are routed through the full evaluator. Public
    packets without per-field policies use deterministic field-level scoring
    over the materialized resulting state.
    """

    rich_ground_truth = _rich_ground_truth_contract(ground_truth)
    if rich_ground_truth.get("fields") or rich_ground_truth.get("repeat_groups"):
        return evaluate_turn_result(
            turn_result,
            rich_ground_truth,
            model_id=model_id,
            reasoning_effort=reasoning_effort,
            questionnaire_path=questionnaire_path,
            runner_version=runner_version,
        )

    return evaluate_public_turn_result(
        turn_result,
        ground_truth,
        model_id=model_id,
        reasoning_effort=reasoning_effort,
        questionnaire_path=questionnaire_path,
        runner_version=runner_version,
    )


def evaluate_public_turn_result(
    turn_result: dict[str, Any],
    ground_truth: dict[str, Any],
    *,
    model_id: str | None = None,
    reasoning_effort: str | None = None,
    questionnaire_path: Path | None = None,
    runner_version: dict[str, Any] | None = None,
) -> TurnEvaluationResult:
    """Deterministically score a public item packet field by field."""

    item = normalize_item(ground_truth)
    _validate_public_turn_pair(turn_result, item)

    gold_state = item.get("gold_resulting_state")
    if not isinstance(gold_state, dict):
        raise ValueError(f"Item {item['item_id']!r} has no gold_resulting_state to score")

    prior_state = item["prior_state"]
    candidate_state = turn_result.get("answers_after") or {}
    if not isinstance(candidate_state, dict):
        raise ValueError("turn_result.answers_after must be an object")

    gold_leaves = _flatten_state(gold_state)
    prior_leaves = _flatten_state(prior_state)
    candidate_leaves = _flatten_state(candidate_state)

    field_results: dict[str, FieldVerdict] = {}
    hard_inputs: list[tuple[str, Any, dict[str, Any]]] = []
    hard_metadata: dict[str, dict[str, Any]] = {}

    for qid, gold_value in gold_leaves.items():
        candidate_value = candidate_leaves.get(qid)
        gt_entry = _ground_truth_entry(
            qid,
            expected=gold_value,
            prior_value=prior_leaves.get(qid, _MISSING),
            questionnaire_path=questionnaire_path,
        )
        hard_inputs.append((qid, candidate_value, gt_entry))
        hard_metadata[qid] = gt_entry

    if hard_inputs:
        field_results.update(
            evaluate_hard_fields(hard_inputs, questionnaire_path=questionnaire_path)
        )

    for qid, verdict in list(field_results.items()):
        if verdict.correctness != "needs_semantic":
            continue
        gt_entry = hard_metadata.get(qid) or {}
        verdict.correctness = "incorrect"
        verdict.decision_source = "deterministic_public_eval"
        verdict.postprocess_reason = (
            "Public packet has no semantic field policy; deterministic scoring "
            "requires the materialized candidate value to match the gold value."
        )
        verdict.reasoning = f"{verdict.postprocess_reason} {verdict.reasoning}".strip()
        verdict.partial_reason = None
        verdict.set_support_source(_source_for_unmatched_candidate(gt_entry))

    unmatched_fields: list[str] = []
    for qid in sorted(set(candidate_leaves) - set(gold_leaves)):
        candidate_value = candidate_leaves[qid]
        if is_empty_for_scoring(candidate_value):
            continue
        unmatched_fields.append(qid)
        field_results[qid] = FieldVerdict(
            correctness="incorrect",
            source="fabricated",
            decision_source="unexpected_field",
            reasoning=f"Candidate populated {qid}, but the gold state has no such field.",
        )

    summary = _summary(field_results, unmatched_count=len(unmatched_fields))

    return TurnEvaluationResult(
        turn=int(turn_result.get("turn_index") or 0),
        user_message=str(turn_result.get("current_utterance") or ""),
        fields_filled=len(candidate_leaves),
        candidate_resulting_state=candidate_state,
        candidate_state_diff=derive_gold_state_diff(prior_state, candidate_state),
        field_results=field_results,
        alignment_log=[],
        unmatched_fields=unmatched_fields,
        derived_variables=dict((ground_truth.get("derived_variables") or {})),
        provenance={
            "runner": runner_version,
            "generation": (turn_result.get("provenance") or {}).get("generation"),
            "evaluation": {
                "runner": runner_version,
                "model": None,
                "models": [],
                "mode": "deterministic_public_eval",
                "requested_model": model_id,
                "reasoning_effort": reasoning_effort,
            },
        },
        prompt_trace=[],
        summary=summary,
    )


def write_evaluation_json(
    item_dir: str | Path,
    *,
    ground_truth: dict[str, Any] | None = None,
    ground_truth_path: str | Path | None = None,
    model_id: str | None = None,
    reasoning_effort: str | None = None,
    questionnaire_path: Path | None = None,
    runner_version: dict[str, Any] | None = None,
) -> TurnEvaluationResult:
    """Read ``turn_result.json``, write ``evaluation.json``, and return it."""

    directory = Path(item_dir)
    turn_path = directory / "turn_result.json"
    if not turn_path.exists():
        raise FileNotFoundError(f"Missing turn_result.json at {turn_path}")

    turn_result = json.loads(turn_path.read_text(encoding="utf-8"))
    if ground_truth is None:
        resolved_gt_path = Path(ground_truth_path) if ground_truth_path else directory / "ground_truth.json"
        if not resolved_gt_path.exists():
            raise FileNotFoundError(
                "Missing ground truth. Pass ground_truth or ground_truth_path; "
                f"no file found at {resolved_gt_path}"
            )
        ground_truth = json.loads(resolved_gt_path.read_text(encoding="utf-8"))
    evaluation = evaluate_solver_turn(
        turn_result,
        ground_truth,
        model_id=model_id,
        reasoning_effort=reasoning_effort,
        questionnaire_path=questionnaire_path,
        runner_version=runner_version,
    )
    output_path = directory / "evaluation.json"
    output_path.write_text(
        json.dumps(evaluation.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return evaluation


def _rich_ground_truth_contract(ground_truth: dict[str, Any]) -> dict[str, Any]:
    item = normalize_item(ground_truth)
    metadata = dict(item.get("metadata") or {})
    if isinstance(ground_truth.get("metadata"), dict):
        metadata.update(ground_truth["metadata"])

    contract = {
        **metadata,
        **{
            key: value
            for key, value in ground_truth.items()
            if key not in {"item_id", "questionnaire_id", "metadata"}
        },
        "scenario_id": ground_truth.get("scenario_id") or item["item_id"],
        "scenario_label": ground_truth.get("scenario_label")
        or metadata.get("scenario_label")
        or item["item_id"],
        "questionnaire": ground_truth.get("questionnaire")
        or item["questionnaire_id"],
        "source_questionnaire": ground_truth.get("source_questionnaire")
        or item["questionnaire_id"],
        "current_utterance": item["current_utterance"],
        "prior_state": item["prior_state"],
        "visible_history": item["visible_history"],
        "gold_resulting_state": item["gold_resulting_state"] or {},
    }
    contract.setdefault(
        "gold_state_diff",
        derive_gold_state_diff(item["prior_state"], item["gold_resulting_state"] or {}),
    )
    contract.setdefault("fields", {})
    contract.setdefault("repeat_groups", {})
    return contract


def _validate_public_turn_pair(turn_result: dict[str, Any], item: dict[str, Any]) -> None:
    if turn_result.get("scenario_id") != item["item_id"]:
        raise ValueError(
            f"scenario_id mismatch: turn_result={turn_result.get('scenario_id')!r}, "
            f"ground_truth={item['item_id']!r}"
        )
    if turn_result.get("questionnaire") != item["questionnaire_id"]:
        raise ValueError(
            f"questionnaire mismatch: turn_result={turn_result.get('questionnaire')!r}, "
            f"ground_truth={item['questionnaire_id']!r}"
        )
    if turn_result.get("current_utterance") != item["current_utterance"]:
        raise ValueError("current_utterance mismatch between turn_result and ground_truth")
    if (turn_result.get("visible_history") or []) != item["visible_history"]:
        raise ValueError("visible_history mismatch between turn_result and ground_truth")


def _flatten_state(state: Any, prefix: str = "") -> dict[str, Any]:
    if not prefix:
        if not isinstance(state, dict):
            return {}
        flattened: dict[str, Any] = {}
        for key in sorted(state):
            flattened.update(_flatten_state(state[key], str(key)))
        return flattened

    if isinstance(state, dict):
        flattened: dict[str, Any] = {}
        for key in sorted(state):
            flattened.update(_flatten_state(state[key], f"{prefix}.{key}"))
        return flattened

    if isinstance(state, list) and all(isinstance(item, dict) for item in state):
        flattened: dict[str, Any] = {}
        for idx, row in enumerate(state):
            flattened.update(_flatten_state(row, f"{prefix}[{idx}]"))
        return flattened

    return {prefix: state}


def _ground_truth_entry(
    qid: str,
    *,
    expected: Any,
    prior_value: Any,
    questionnaire_path: Path | None,
) -> dict[str, Any]:
    strategy = _strategy_for_field(qid, expected, questionnaire_path=questionnaire_path)
    entry: dict[str, Any] = {
        "strategy": strategy,
        "expected": expected,
        "acceptable_alternatives": [],
        "partially_correct_values": [],
        "evidence": "",
        "present_in_utterance": True,
        "extraction_difficulty": "direct",
    }
    if strategy == "set_match":
        entry["required"] = expected if isinstance(expected, list) else [expected]
        entry["acceptable_additions"] = []
        entry["unacceptable"] = []
        entry["debatable"] = []

    if prior_value is not _MISSING:
        scoring_profile = infer_scoring_profile(qid, entry, expected=expected)
        if values_equal_for_scoring(
            prior_value,
            expected,
            scoring_profile=scoring_profile,
        ):
            entry["present_in_utterance"] = False
            entry["evidence_source"] = "prior_state"
    return entry


def _strategy_for_field(
    qid: str,
    expected: Any,
    *,
    questionnaire_path: Path | None,
) -> str:
    info = get_questionnaire_field_info(qid, questionnaire_path)
    if info.get("type") == "multiple_choice" and isinstance(expected, list):
        return "set_match"
    return "exact"


def _source_for_unmatched_candidate(gt_entry: dict[str, Any]) -> str:
    if gt_entry.get("evidence_source") == "prior_state":
        return "prior_state"
    return "unsupported_inference"


def _summary(
    field_results: dict[str, FieldVerdict],
    *,
    unmatched_count: int,
) -> dict[str, int]:
    counts = {
        "total": len(field_results),
        "evaluated": len(field_results),
        "unmatched": unmatched_count,
        "correct": 0,
        "partially_correct": 0,
        "incorrect": 0,
    }
    for verdict in field_results.values():
        counts[verdict.correctness] = counts.get(verdict.correctness, 0) + 1
    return counts
