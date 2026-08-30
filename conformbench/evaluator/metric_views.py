"""Derived metric views for evaluation reports.

Keeps scoring/diagnostic derivation separate from report serialization.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping as MappingABC
from pathlib import Path
from typing import Any, Mapping, Sequence

from .error_groups import (
    aggregate_error_group_views,
    build_error_group_view,
)
from .scoring import (
    SCORING_PROFILE_AUXILIARY_UNSCORED_NOTE,
    infer_scoring_profile,
    is_empty_for_scoring,
    normalize_runtime_empty,
    values_equal_for_scoring,
)

_REPEAT_QID_RE = re.compile(r"^(?P<group>\w+)\[(?P<idx>\d+|\?gt\d+)\]\.(?P<field>.+)$")
INPUT_INCOHERENCE_PROBE = "input_incoherence_overcommit"


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


def _values_equal(
    left: Any,
    right: Any,
    *,
    scoring_profile: str | None = None,
) -> bool:
    return values_equal_for_scoring(
        left,
        right,
        scoring_profile=scoring_profile,
    )


def _is_empty(
    value: Any,
    *,
    scoring_profile: str | None = None,
) -> bool:
    return is_empty_for_scoring(value, scoring_profile=scoring_profile)


def _change_kind(
    prior_value: Any,
    resulting_value: Any,
    *,
    scoring_profile: str | None = None,
) -> str:
    if _is_empty(prior_value, scoring_profile=scoring_profile) and not _is_empty(
        resulting_value,
        scoring_profile=scoring_profile,
    ):
        return "set"
    if not _is_empty(prior_value, scoring_profile=scoring_profile) and _is_empty(
        resulting_value,
        scoring_profile=scoring_profile,
    ):
        return "clear"
    if not _values_equal(
        prior_value,
        resulting_value,
        scoring_profile=scoring_profile,
    ):
        return "changed"
    return "preserved"


def _transition_label(change_kind: str) -> str:
    return {"preserved": "preserve", "changed": "change"}.get(
        change_kind,
        change_kind,
    )


def _item_repeat_involved(gt: Mapping[str, Any]) -> bool:
    """Return item-level repeat involvement from derived variables when present.

    Many materialized ground truths include repeat-group state for context even
    when the item does not exercise repeat routing.  The report row labeled
    "repeat involvement" should therefore follow the item-design metadata, not
    merely the presence of any repeat-group block in the record.
    """
    derived = gt.get("derived_variables") or {}
    if "repeat_group_involved" in derived:
        return bool(derived.get("repeat_group_involved"))

    involvement = derived.get("repeat_group_involvement")
    if involvement is not None:
        return str(involvement).strip().lower() not in {"", "none", "false", "0"}

    if "repeat_group_names" in derived:
        names = derived.get("repeat_group_names")
        return bool(names)

    return bool(gt.get("repeat_groups") or {})


def _prf_bundle(tp: float, predicted: int, gold: int) -> dict[str, float]:
    precision = round(_safe_div(tp, predicted), 4)
    recall = round(_safe_div(tp, gold), 4)
    return {
        "precision": precision,
        "recall": recall,
        "f1": round(_f1(precision, recall), 4),
    }


def _repeat_instance_total_from_state(state: Mapping[str, Any] | None) -> int:
    if not isinstance(state, MappingABC):
        return 0
    return sum(
        1
        for value in state.values()
        if isinstance(value, list)
        for item in value
        if isinstance(item, MappingABC)
    )


def _repeat_group_count_from_state(state: Mapping[str, Any] | None) -> int:
    if not isinstance(state, MappingABC):
        return 0
    return sum(
        1
        for value in state.values()
        if isinstance(value, list) and any(isinstance(item, MappingABC) for item in value)
    )


def _targeted_failure_modes(gt: Mapping[str, Any]) -> set[str]:
    modes: set[str] = set()
    for mode in gt.get("targeted_failure_mode") or []:
        if mode:
            modes.add(str(mode))
    difficulty_profile = gt.get("difficulty_profile") or {}
    for mode in difficulty_profile.get("targeted_failure_modes") or []:
        if mode:
            modes.add(str(mode))
    return modes


def _load_gate_requirements(questionnaire_path: Path | None) -> dict[str, dict[str, Any]]:
    """Map leaf IDs to their gate controller requirement."""
    if questionnaire_path is None or not questionnaire_path.exists():
        return {}

    try:
        data = json.loads(questionnaire_path.read_text())
    except Exception:
        return {}

    requirements: dict[str, dict[str, Any]] = {}

    def _walk(
        questions: list[dict[str, Any]],
        gate_stack: list[dict[str, Any]],
        *,
        repeat_group: str | None = None,
    ) -> None:
        for question in questions:
            qid = question.get("id") or question.get("_id")
            if not qid:
                continue
            structure_type = question.get("structure_type", "regular")

            if structure_type == "regular":
                if gate_stack:
                    requirement = dict(gate_stack[-1])
                    requirement["repeat_group"] = repeat_group
                    requirements[qid] = requirement
                continue

            children = question.get("questions") or question.get("fields") or []
            if structure_type == "gate":
                gate = question.get("gate", {})
                gate_stack.append(
                    {
                        "gate_on": gate.get("gate_on"),
                        "when_values": list(gate.get("when_values") or []),
                    }
                )
                _walk(children, gate_stack, repeat_group=repeat_group)
                gate_stack.pop()
                continue

            child_repeat_group = repeat_group
            if structure_type == "repeat_group":
                child_repeat_group = qid

            _walk(children, gate_stack, repeat_group=child_repeat_group)

    _walk(data.get("questions") or [], [])
    return requirements


def _state_value(state: Mapping[str, Any], slot: str) -> Any:
    if slot in state:
        return state.get(slot)

    match = _REPEAT_QID_RE.match(slot)
    if not match:
        return None

    group_name = match.group("group")
    idx_token = match.group("idx")
    field_name = match.group("field")
    if idx_token.startswith("?gt"):
        idx = int(idx_token[3:])
    else:
        idx = int(idx_token)

    group_value = state.get(group_name)
    if not isinstance(group_value, list) or idx >= len(group_value):
        return None
    row = group_value[idx]
    if not isinstance(row, dict):
        return None
    return row.get(field_name)


def _repeat_alignment_map(
    alignment_log: Sequence[Mapping[str, Any]],
) -> dict[str, dict[int, int]]:
    mapping: dict[str, dict[int, int]] = {}
    for entry in alignment_log:
        if entry.get("status") != "matched":
            continue
        group = entry.get("group")
        gt_index = entry.get("gt_index")
        agent_index = entry.get("agent_index")
        if not group or gt_index is None or agent_index is None:
            continue
        mapping.setdefault(group, {})[int(agent_index)] = int(gt_index)
    return mapping


def _repeat_gt_lookup(gt: Mapping[str, Any]) -> dict[str, dict[int, Mapping[str, Any]]]:
    lookup: dict[str, dict[int, Mapping[str, Any]]] = {}
    for group_name, group in (gt.get("repeat_groups") or {}).items():
        instances = {}
        for instance in group.get("instances") or []:
            gt_index = instance.get("ground_truth_index")
            if gt_index is None:
                continue
            instances[int(gt_index)] = instance
        lookup[group_name] = instances
    return lookup


def _leaf_records(
    *,
    gt: Mapping[str, Any],
    candidate_state: Mapping[str, Any],
    field_results: Mapping[str, Mapping[str, Any]],
    alignment_log: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    repeat_alignment = _repeat_alignment_map(alignment_log)
    repeat_lookup = _repeat_gt_lookup(gt)
    prior_state = gt.get("prior_state") or {}
    gold_state = gt.get("gold_resulting_state") or {}

    for qid, verdict in field_results.items():
        gt_entry: Mapping[str, Any] | None = None
        prior_value = None
        gold_value = None
        candidate_value = None
        present_in_utterance = True
        is_repeat = False
        gt_index: int | None = None
        agent_index: int | None = None
        group_name: str | None = None

        if qid in (gt.get("fields") or {}):
            gt_entry = (gt.get("fields") or {}).get(qid) or {}
            prior_value = _state_value(prior_state, qid)
            gold_value = _state_value(gold_state, qid)
            candidate_value = _state_value(candidate_state, qid)
        else:
            match = _REPEAT_QID_RE.match(qid)
            if not match:
                continue
            is_repeat = True
            group_name = match.group("group")
            idx_token = match.group("idx")
            field_name = match.group("field")

            if idx_token.startswith("?gt"):
                gt_index = int(idx_token[3:])
            else:
                agent_index = int(idx_token)
                gt_index = repeat_alignment.get(group_name, {}).get(agent_index)

            if gt_index is None:
                # Hallucinated instance — agent invented data with no GT
                # counterpart.  Still counts toward field-level metrics so
                # that extra fabricated instances penalise precision. Compare
                # against the prior by agent index so retained prior rows do
                # not look like candidate-side edits.
                prior_rows = prior_state.get(group_name) or []
                candidate_rows = candidate_state.get(group_name) or []
                prior_value = None
                if (
                    agent_index is not None
                    and agent_index < len(candidate_rows)
                    and isinstance(candidate_rows[agent_index], Mapping)
                ):
                    candidate_value = candidate_rows[agent_index].get(field_name)
                if (
                    agent_index is not None
                    and agent_index < len(prior_rows)
                    and isinstance(prior_rows[agent_index], Mapping)
                ):
                    prior_value = prior_rows[agent_index].get(field_name)

                correctness = verdict.get("correctness", "incorrect")
                if correctness == "needs_semantic":
                    correctness = "incorrect"
                candidate_change = _change_kind(
                    prior_value,
                    candidate_value,
                    scoring_profile=None,
                )

                records.append(
                    {
                        "qid": qid,
                        "is_repeat": True,
                        "group": group_name,
                        "gt_index": None,
                        "agent_index": agent_index,
                        "prior_value": prior_value,
                        "gold_value": None,
                        "candidate_value": candidate_value,
                        "gold_change": "preserved",
                        "candidate_change": candidate_change,
                        "candidate_matches_gold": False,
                        "correctness": correctness,
                        "partial_reason": verdict.get("partial_reason"),
                        "present_in_utterance": False,
                        "scoring_profile": None,
                        "is_auxiliary_unscored_note": False,
                        "gt_entry": {},
                    }
                )
                continue

            gt_instance = repeat_lookup.get(group_name, {}).get(gt_index) or {}
            gt_entry = (gt_instance.get("fields") or {}).get(field_name) or {}
            prior_rows = prior_state.get(group_name) or []
            gold_rows = gold_state.get(group_name) or []
            candidate_rows = candidate_state.get(group_name) or []

            prior_index = agent_index if agent_index is not None else gt_index
            if prior_index < len(prior_rows) and isinstance(prior_rows[prior_index], Mapping):
                prior_value = prior_rows[prior_index].get(field_name)
            if gt_index < len(gold_rows) and isinstance(gold_rows[gt_index], Mapping):
                gold_value = gold_rows[gt_index].get(field_name)
            elif gt_entry:
                gold_value = gt_entry.get("expected")

            if (
                agent_index is not None
                and agent_index < len(candidate_rows)
                and isinstance(candidate_rows[agent_index], Mapping)
            ):
                candidate_value = candidate_rows[agent_index].get(field_name)
            elif (
                agent_index is None
                and gt_index is not None
                and gt_index < len(candidate_rows)
                and isinstance(candidate_rows[gt_index], Mapping)
            ):
                candidate_value = candidate_rows[gt_index].get(field_name)

        present_in_utterance = bool((gt_entry or {}).get("present_in_utterance", True))
        scoring_profile = infer_scoring_profile(
            qid,
            gt_entry or {},
            expected=(gt_entry or {}).get("expected", gold_value),
        )
        gold_change = _change_kind(
            prior_value,
            gold_value,
            scoring_profile=scoring_profile,
        )
        candidate_change = _change_kind(
            prior_value,
            candidate_value,
            scoring_profile=scoring_profile,
        )
        correctness = verdict.get("correctness", "incorrect")
        if correctness == "needs_semantic":
            correctness = "incorrect"

        records.append(
            {
                "qid": qid,
                "is_repeat": is_repeat,
                "group": group_name,
                "gt_index": gt_index,
                "agent_index": agent_index,
                "prior_value": prior_value,
                "gold_value": gold_value,
                "candidate_value": candidate_value,
                "gold_change": gold_change,
                "candidate_change": candidate_change,
                "candidate_matches_gold": _values_equal(
                    candidate_value,
                    gold_value,
                    scoring_profile=scoring_profile,
                ),
                "correctness": correctness,
                "partial_reason": verdict.get("partial_reason"),
                "present_in_utterance": present_in_utterance,
                "scoring_profile": scoring_profile,
                "is_auxiliary_unscored_note": scoring_profile == SCORING_PROFILE_AUXILIARY_UNSCORED_NOTE,
                "gt_entry": dict(gt_entry or {}),
            }
        )

    return records


def _whole_form_bundle(
    records: Sequence[Mapping[str, Any]],
    gt_expected: int | None = None,
) -> dict[str, Any]:
    evaluated = len(records)
    correct = sum(1 for record in records if record["correctness"] == "correct")
    partial = sum(1 for record in records if record["correctness"] == "partially_correct")
    incorrect = sum(1 for record in records if record["correctness"] == "incorrect")
    if gt_expected is None:
        gt_expected = evaluated
    return {
        "total_evaluated_leaf_fields": evaluated,
        "gt_expected_total": gt_expected,
        "correct": correct,
        "partial": partial,
        "incorrect": incorrect,
        "accuracy": round(_safe_div(correct, evaluated), 4),
        "lenient_accuracy": round(_safe_div(correct + partial, evaluated), 4),
        "strict": _prf_bundle(correct, evaluated, gt_expected),
        "lenient": _prf_bundle(correct + partial, evaluated, gt_expected),
        "weighted": _prf_bundle(correct + 0.5 * partial, evaluated, gt_expected),
    }


def _auxiliary_unscored_note_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    candidate_non_empty = [
        record
        for record in records
        if not _is_empty(
            record.get("candidate_value"),
            scoring_profile=record.get("scoring_profile"),
        )
    ]
    gold_non_empty = [
        record
        for record in records
        if not _is_empty(
            record.get("gold_value"),
            scoring_profile=record.get("scoring_profile"),
        )
    ]
    gold_empty = [
        record
        for record in records
        if _is_empty(
            record.get("gold_value"),
            scoring_profile=record.get("scoring_profile"),
        )
    ]
    hallucinations = [
        record
        for record in candidate_non_empty
        if _is_empty(
            record.get("gold_value"),
            scoring_profile=record.get("scoring_profile"),
        )
    ]
    grounded_captures = [
        record
        for record in gold_non_empty
        if record.get("correctness") == "correct"
    ]
    abstention_successes = [
        record
        for record in gold_empty
        if _is_empty(
            record.get("candidate_value"),
            scoring_profile=record.get("scoring_profile"),
        )
    ]
    return {
        "total_auxiliary_unscored_note_fields": len(records),
        "candidate_non_empty_total": len(candidate_non_empty),
        "gold_grounded_total": len(gold_non_empty),
        "gold_empty_total": len(gold_empty),
        "hallucination_count": len(hallucinations),
        "grounded_capture_count": len(grounded_captures),
        "abstention_success_count": len(abstention_successes),
        "auxiliary_note_hallucination_rate": round(
            _safe_div(len(hallucinations), len(candidate_non_empty)),
            4,
        ),
        "auxiliary_note_grounded_capture_rate": round(
            _safe_div(len(grounded_captures), len(gold_non_empty)),
            4,
        ),
        "auxiliary_note_abstention_acceptance_rate": round(
            _safe_div(len(abstention_successes), len(gold_empty)),
            4,
        ),
    }


def compute_turn_metric_views(
    *,
    gt: Mapping[str, Any],
    candidate_state: Mapping[str, Any],
    field_results: Mapping[str, Mapping[str, Any]],
    alignment_log: Sequence[Mapping[str, Any]],
    diagnostics: Mapping[str, Any],
    questionnaire_path: Path | None = None,
) -> dict[str, Any]:
    records = _leaf_records(
        gt=gt,
        candidate_state=candidate_state,
        field_results=field_results,
        alignment_log=alignment_log,
    )

    primary_records = [
        record for record in records if not record["is_auxiliary_unscored_note"]
    ]
    auxiliary_note_records = [
        record for record in records if record["is_auxiliary_unscored_note"]
    ]

    # GT expected = fields the GT defines (recall denominator).
    # Hallucinated instances (agent-invented, no GT backing) are identified
    # by having an empty gt_entry dict.
    gt_expected_all = sum(1 for r in primary_records if r.get("gt_entry"))
    gt_expected_primary = sum(1 for r in primary_records if r.get("gt_entry"))

    whole_form = _whole_form_bundle(primary_records, gt_expected=gt_expected_primary)
    all_fields = _whole_form_bundle(primary_records, gt_expected=gt_expected_all)
    error_groups = build_error_group_view(gt=gt, records=primary_records)

    changed_records = [
        record
        for record in primary_records
        if record["gold_change"] in {"set", "clear", "changed"}
    ]
    predicted_changed_records = [
        record
        for record in primary_records
        if record["candidate_change"] in {"set", "clear", "changed"}
    ]
    changed_correct = sum(
        1
        for record in changed_records
        if record["correctness"] == "correct"
        and record["candidate_change"] in {"set", "clear", "changed"}
    )
    changed_partial = sum(
        1
        for record in changed_records
        if record["correctness"] == "partially_correct"
        and record["candidate_change"] in {"set", "clear", "changed"}
    )
    changed_incorrect = sum(1 for record in changed_records if record["correctness"] == "incorrect")

    changed_fields = {
        "gold_changed_total": len(changed_records),
        "predicted_changed_total": len(predicted_changed_records),
        "changed_correct": changed_correct,
        "changed_partial": changed_partial,
        "changed_incorrect": changed_incorrect,
        "strict": _prf_bundle(changed_correct, len(predicted_changed_records), len(changed_records)),
        "lenient": _prf_bundle(changed_correct + changed_partial, len(predicted_changed_records), len(changed_records)),
        "weighted": _prf_bundle(changed_correct + 0.5 * changed_partial, len(predicted_changed_records), len(changed_records)),
    }

    preserved_records = [record for record in primary_records if record["gold_change"] == "preserved"]
    collateral_edit_count = sum(
        1
        for record in preserved_records
        if record["candidate_change"] != "preserved"
    )
    preserved_total = len(preserved_records)
    preserved_kept = max(0, preserved_total - collateral_edit_count)

    preservation = {
        "preserved_total": preserved_total,
        "preserved_kept": preserved_kept,
        "collateral_edit_count": collateral_edit_count,
        "preservation_success_rate": round(_safe_div(preserved_kept, preserved_total), 4),
        "preservation_error_rate": round(_safe_div(collateral_edit_count, preserved_total), 4),
    }

    diag_counts = diagnostics.get("counts") or {}
    unsupported_commit_count = int(diag_counts.get("unsupported_commit", 0) or 0)
    forbidden_commit_count = int(diag_counts.get("forbidden_commit", 0) or 0)
    targeted_failure_modes = _targeted_failure_modes(gt)
    input_incoherence_probe = INPUT_INCOHERENCE_PROBE in targeted_failure_modes
    evidence = gt.get("evidence") or {}

    correction_applicable = gt.get("primary_delta_type") == "correct"
    retraction_applicable = gt.get("primary_delta_type") == "retract"
    history_applicable = bool(evidence.get("history_required"))

    history_changed_records = [
        record
        for record in changed_records
        if not record["present_in_utterance"]
    ]
    history_unsupported = [
        record
        for record in primary_records
        if not record["present_in_utterance"]
        and record["candidate_change"] in {"set", "clear", "changed"}
        and record["gold_change"] == "preserved"
    ]

    gate_requirements = _load_gate_requirements(questionnaire_path)
    gated_records = []
    for record in changed_records:
        match = _REPEAT_QID_RE.match(record["qid"])
        field_lookup = match.group("field") if match else record["qid"]
        requirement = gate_requirements.get(field_lookup)
        if not requirement:
            continue
        gated_records.append((record, requirement))

    gate_success = None
    if gated_records:
        gate_success = True
        for record, requirement in gated_records:
            gate_on = requirement.get("gate_on")
            when_values = requirement.get("when_values") or []
            candidate_gate = _state_value(candidate_state, str(gate_on))
            gold_gate = _state_value(gt.get("gold_resulting_state") or {}, str(gate_on))
            if candidate_gate != gold_gate:
                gate_success = False
                break
            if when_values and gold_gate not in when_values:
                gate_success = False
                break
            if record["correctness"] != "correct":
                gate_success = False
                break

    counts = diagnostics.get("counts") or {}
    changed_repeat_records = [record for record in changed_records if record["is_repeat"]]
    repeat_alignment_errors = int(counts.get("repeat_group_alignment_error", 0) or 0)
    repeat_success = None
    if changed_repeat_records or repeat_alignment_errors:
        repeat_success = (
            repeat_alignment_errors == 0
            and all(record["correctness"] == "correct" for record in changed_repeat_records)
        )

    failed_changed = sum(1 for r in changed_records if r["correctness"] != "correct")

    task_diagnostics = {
        "unsupported_commit_count": unsupported_commit_count,
        "unsupported_commit_rate": round(
            _safe_div(
                unsupported_commit_count,
                whole_form["total_evaluated_leaf_fields"],
            ),
            4,
        ),
        "input_incoherence_probe": input_incoherence_probe,
        "input_incoherence_overcommit_count": (
            unsupported_commit_count if input_incoherence_probe else 0
        ),
        "input_incoherence_forbidden_commit_count": (
            forbidden_commit_count if input_incoherence_probe else 0
        ),
        "input_incoherence_success": (
            unsupported_commit_count == 0 and forbidden_commit_count == 0
            if input_incoherence_probe
            else None
        ),
        "collateral_edit_count": collateral_edit_count,
        "failed_correction_count": failed_changed if correction_applicable else 0,
        "failed_retraction_count": failed_changed if retraction_applicable else 0,
        "repeat_routing_error_count": repeat_alignment_errors,
        "correction_success": (
            bool(correction_applicable)
            and all(record["correctness"] == "correct" for record in changed_records)
            if correction_applicable
            else None
        ),
        "retraction_success": (
            bool(retraction_applicable)
            and all(record["correctness"] == "correct" for record in changed_records)
            if retraction_applicable
            else None
        ),
        "history_recovery_success": (
            bool(history_applicable)
            and all(record["correctness"] == "correct" for record in history_changed_records)
            and not history_unsupported
            if history_applicable
            else None
        ),
        "gate_execution_success": gate_success,
        "repeat_group_execution_success": repeat_success,
    }

    field_classes = {
        "commitment_core": sum(
            1 for record in records if record["scoring_profile"] == "commitment_core"
        ),
        "unknown_equivalent": sum(
            1 for record in records if record["scoring_profile"] == "unknown_equivalent"
        ),
        "auxiliary_unscored_note": len(auxiliary_note_records),
    }

    all_correct = all(record["correctness"] == "correct" for record in primary_records) if primary_records else True
    all_lenient_correct = (
        all(record["correctness"] in {"correct", "partially_correct"} for record in primary_records)
        if primary_records
        else True
    )
    whole_record_exact_match = {
        "exact_match": all_correct,
        "lenient_exact_match": all_lenient_correct,
    }

    transition_map: dict[str, list[dict[str, Any]]] = {}
    for record in primary_records:
        key = _transition_label(str(record["gold_change"]))
        transition_map.setdefault(key, []).append(record)
    # A transition is correct only when the resulting field value is correct.
    # Merely choosing the right operation class (for example, changing a field
    # to the wrong non-empty value) must not receive field-level credit.
    transition_correct = sum(
        1 for record in primary_records if record["correctness"] == "correct"
    )
    by_transition: dict[str, dict[str, int]] = {}
    for key, recs in transition_map.items():
        correct = sum(1 for r in recs if r["correctness"] == "correct")
        by_transition[key] = {
            "gold_total": len(recs),
            "correct": correct,
            "accuracy": round(_safe_div(correct, len(recs)), 4),
        }
    transition_accuracy = {
        "total_evaluated_leaf_fields": whole_form["total_evaluated_leaf_fields"],
        "gt_expected_total": whole_form["gt_expected_total"],
        "correct": transition_correct,
        "partial": 0,
        "incorrect": max(
            0,
            whole_form["total_evaluated_leaf_fields"] - transition_correct,
        ),
        "accuracy": round(
            _safe_div(
                transition_correct,
                whole_form["total_evaluated_leaf_fields"],
            ),
            4,
        ),
        "by_transition": by_transition,
    }

    gt_repeat_groups = gt.get("repeat_groups") or {}
    item_repeat_involved = _item_repeat_involved(gt)
    gold_repeat_instance_total = sum(
        len(rg.get("instances") or []) for rg in gt_repeat_groups.values()
    )
    prior_repeat_instance_total = _repeat_instance_total_from_state(gt.get("prior_state"))
    resulting_repeat_instance_total = _repeat_instance_total_from_state(
        gt.get("gold_resulting_state")
    )
    repeat_instance_coordination_total = max(
        prior_repeat_instance_total,
        gold_repeat_instance_total,
        resulting_repeat_instance_total,
    )
    repeat_group_coordination_count = max(
        _repeat_group_count_from_state(gt.get("prior_state")),
        _repeat_group_count_from_state(gt.get("gold_resulting_state")),
        sum(1 for rg in gt_repeat_groups.values() if rg.get("instances")),
    )
    spurious_count = sum(
        1 for entry in alignment_log if entry.get("status") == "hallucinated"
    )
    repeat_groups_view = {
        "item_repeat_involved": item_repeat_involved,
        "prior_repeat_instance_total": prior_repeat_instance_total,
        "gold_repeat_instance_total": gold_repeat_instance_total,
        "resulting_repeat_instance_total": resulting_repeat_instance_total,
        "repeat_instance_coordination_total": repeat_instance_coordination_total,
        "repeat_group_coordination_count": repeat_group_coordination_count,
        "gold_repeat_instance_operation_total": gold_repeat_instance_total,
        "spurious_instance_count": spurious_count,
        "alignment_failure_count": repeat_alignment_errors,
    }

    return {
        "whole_record_exact_match": whole_record_exact_match,
        "whole_form": whole_form,
        "primary_fields": whole_form,
        "all_fields": all_fields,
        "changed_fields": changed_fields,
        "preservation": preservation,
        "transition_accuracy": transition_accuracy,
        "task_diagnostics": task_diagnostics,
        "repeat_groups": repeat_groups_view,
        "error_groups": error_groups,
        "field_classes": field_classes,
        "auxiliary_unscored_notes": _auxiliary_unscored_note_metrics(auxiliary_note_records),
    }


def aggregate_metric_views(utterance_views: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    whole_views = [view.get("whole_form") or {} for view in utterance_views]
    all_field_views = [view.get("all_fields") or {} for view in utterance_views]
    changed_views = [view.get("changed_fields") or {} for view in utterance_views]
    preservation_views = [view.get("preservation") or {} for view in utterance_views]
    task_views = [view.get("task_diagnostics") or {} for view in utterance_views]
    field_class_views = [view.get("field_classes") or {} for view in utterance_views]
    auxiliary_note_views = [
        view.get("auxiliary_unscored_notes")
        or view.get("optional_notes")
        or {}
        for view in utterance_views
    ]
    exact_match_views = [view.get("whole_record_exact_match") or {} for view in utterance_views]
    transition_views = [view.get("transition_accuracy") or {} for view in utterance_views]
    repeat_views = [view.get("repeat_groups") or {} for view in utterance_views]
    error_group_views = [view.get("error_groups") or {} for view in utterance_views]

    whole_evaluated = sum(int(view.get("total_evaluated_leaf_fields", 0) or 0) for view in whole_views)
    whole_gt_expected = sum(int(view.get("gt_expected_total", 0) or 0) for view in whole_views)
    whole_correct = sum(int(view.get("correct", 0) or 0) for view in whole_views)
    whole_partial = sum(int(view.get("partial", 0) or 0) for view in whole_views)
    whole_incorrect = sum(int(view.get("incorrect", 0) or 0) for view in whole_views)
    all_fields_evaluated = sum(int(view.get("total_evaluated_leaf_fields", 0) or 0) for view in all_field_views)
    all_fields_gt_expected = sum(int(view.get("gt_expected_total", 0) or 0) for view in all_field_views)
    all_fields_correct = sum(int(view.get("correct", 0) or 0) for view in all_field_views)
    all_fields_partial = sum(int(view.get("partial", 0) or 0) for view in all_field_views)
    all_fields_incorrect = sum(int(view.get("incorrect", 0) or 0) for view in all_field_views)

    changed_gold_total = sum(int(view.get("gold_changed_total", 0) or 0) for view in changed_views)
    changed_pred_total = sum(int(view.get("predicted_changed_total", 0) or 0) for view in changed_views)
    changed_correct = sum(int(view.get("changed_correct", 0) or 0) for view in changed_views)
    changed_partial = sum(int(view.get("changed_partial", 0) or 0) for view in changed_views)
    changed_incorrect = sum(int(view.get("changed_incorrect", 0) or 0) for view in changed_views)

    preserved_total = sum(int(view.get("preserved_total", 0) or 0) for view in preservation_views)
    preserved_kept = sum(int(view.get("preserved_kept", 0) or 0) for view in preservation_views)
    collateral_edits = sum(int(view.get("collateral_edit_count", 0) or 0) for view in preservation_views)
    auxiliary_note_total = sum(
        int(
            view.get(
                "total_auxiliary_unscored_note_fields",
                view.get("total_optional_note_fields", 0),
            )
            or 0
        )
        for view in auxiliary_note_views
    )
    auxiliary_note_candidate_non_empty = sum(
        int(view.get("candidate_non_empty_total", 0) or 0)
        for view in auxiliary_note_views
    )
    auxiliary_note_gold_grounded = sum(
        int(view.get("gold_grounded_total", 0) or 0)
        for view in auxiliary_note_views
    )
    auxiliary_note_gold_empty = sum(
        int(view.get("gold_empty_total", 0) or 0)
        for view in auxiliary_note_views
    )
    auxiliary_note_hallucinations = sum(
        int(view.get("hallucination_count", 0) or 0)
        for view in auxiliary_note_views
    )
    auxiliary_note_grounded_captures = sum(
        int(view.get("grounded_capture_count", 0) or 0)
        for view in auxiliary_note_views
    )
    auxiliary_note_abstention_successes = sum(
        int(view.get("abstention_success_count", 0) or 0)
        for view in auxiliary_note_views
    )

    def _macro_prf(source: Sequence[Mapping[str, Any]], key: str) -> dict[str, float]:
        n = len(source) or 1
        return {
            "precision": round(sum(float((view.get(key) or {}).get("precision", 0) or 0) for view in source) / n, 4),
            "recall": round(sum(float((view.get(key) or {}).get("recall", 0) or 0) for view in source) / n, 4),
            "f1": round(sum(float((view.get(key) or {}).get("f1", 0) or 0) for view in source) / n, 4),
        }

    def _success_summary(key: str) -> dict[str, Any]:
        applicable_values = [view.get(key) for view in task_views if view.get(key) is not None]
        success_count = sum(1 for value in applicable_values if bool(value))
        applicable_count = len(applicable_values)
        return {
            "applicable": applicable_count,
            "success": success_count,
            "success_rate": round(_safe_div(success_count, applicable_count), 4),
        }

    # ── Exact-match aggregation ──
    item_count = len(exact_match_views)
    exact_match_count = sum(1 for v in exact_match_views if v.get("exact_match", False))
    lenient_exact_match_count = sum(
        1
        for v in exact_match_views
        if v.get("lenient_exact_match", v.get("exact_match", False))
    )

    # ── Transition-accuracy aggregation ──
    ta_evaluated = sum(int(v.get("total_evaluated_leaf_fields", 0) or 0) for v in transition_views)
    ta_gt_expected = sum(int(v.get("gt_expected_total", 0) or 0) for v in transition_views)
    ta_correct = sum(int(v.get("correct", 0) or 0) for v in transition_views)
    ta_partial = sum(int(v.get("partial", 0) or 0) for v in transition_views)
    ta_incorrect = sum(int(v.get("incorrect", 0) or 0) for v in transition_views)
    agg_by_transition: dict[str, dict[str, int]] = {}
    for v in transition_views:
        for key, data in (v.get("by_transition") or {}).items():
            if key not in agg_by_transition:
                agg_by_transition[key] = {"gold_total": 0, "correct": 0}
            agg_by_transition[key]["gold_total"] += int(data.get("gold_total", 0) or 0)
            agg_by_transition[key]["correct"] += int(data.get("correct", 0) or 0)
    for data in agg_by_transition.values():
        data["accuracy"] = round(
            _safe_div(data.get("correct", 0), data.get("gold_total", 0)),
            4,
        )

    # ── Repeat-groups aggregation ──
    rg_item_involved = sum(1 for v in repeat_views if v.get("item_repeat_involved"))
    rg_prior_total = sum(int(v.get("prior_repeat_instance_total", 0) or 0) for v in repeat_views)
    rg_gt_total = sum(int(v.get("gold_repeat_instance_total", 0) or 0) for v in repeat_views)
    rg_resulting_total = sum(
        int(v.get("resulting_repeat_instance_total", 0) or 0) for v in repeat_views
    )
    rg_coord_total = sum(
        int(
            v.get(
                "repeat_instance_coordination_total",
                v.get("gold_repeat_instance_total", 0),
            )
            or 0
        )
        for v in repeat_views
    )
    rg_group_coord_total = sum(
        int(v.get("repeat_group_coordination_count", 0) or 0) for v in repeat_views
    )
    rg_op_total = sum(int(v.get("gold_repeat_instance_operation_total", 0) or 0) for v in repeat_views)
    rg_spurious = sum(int(v.get("spurious_instance_count", 0) or 0) for v in repeat_views)
    rg_failures = sum(int(v.get("alignment_failure_count", 0) or 0) for v in repeat_views)

    return {
        "whole_record_exact_match": {
            "item_count": item_count,
            "exact_match_count": exact_match_count,
            "exact_match_rate": round(_safe_div(exact_match_count, item_count), 4),
            "lenient_exact_match_count": lenient_exact_match_count,
            "lenient_exact_match_rate": round(
                _safe_div(lenient_exact_match_count, item_count),
                4,
            ),
        },
        "whole_form": {
            "total_evaluated_leaf_fields": whole_evaluated,
            "gt_expected_total": whole_gt_expected,
            "correct": whole_correct,
            "partial": whole_partial,
            "incorrect": whole_incorrect,
            "accuracy": round(_safe_div(whole_correct, whole_evaluated), 4),
            "lenient_accuracy": round(_safe_div(whole_correct + whole_partial, whole_evaluated), 4),
            "strict": _prf_bundle(whole_correct, whole_evaluated, whole_gt_expected),
            "lenient": _prf_bundle(whole_correct + whole_partial, whole_evaluated, whole_gt_expected),
            "weighted": _prf_bundle(whole_correct + 0.5 * whole_partial, whole_evaluated, whole_gt_expected),
            "macro": {
                "strict": _macro_prf(whole_views, "strict"),
                "lenient": _macro_prf(whole_views, "lenient"),
                "weighted": _macro_prf(whole_views, "weighted"),
            },
        },
        "primary_fields": {
            "total_evaluated_leaf_fields": whole_evaluated,
            "gt_expected_total": whole_gt_expected,
            "correct": whole_correct,
            "partial": whole_partial,
            "incorrect": whole_incorrect,
            "accuracy": round(_safe_div(whole_correct, whole_evaluated), 4),
            "lenient_accuracy": round(_safe_div(whole_correct + whole_partial, whole_evaluated), 4),
            "strict": _prf_bundle(whole_correct, whole_evaluated, whole_gt_expected),
            "lenient": _prf_bundle(whole_correct + whole_partial, whole_evaluated, whole_gt_expected),
            "weighted": _prf_bundle(whole_correct + 0.5 * whole_partial, whole_evaluated, whole_gt_expected),
            "macro": {
                "strict": _macro_prf(whole_views, "strict"),
                "lenient": _macro_prf(whole_views, "lenient"),
                "weighted": _macro_prf(whole_views, "weighted"),
            },
        },
        "all_fields": {
            "total_evaluated_leaf_fields": all_fields_evaluated,
            "gt_expected_total": all_fields_gt_expected,
            "correct": all_fields_correct,
            "partial": all_fields_partial,
            "incorrect": all_fields_incorrect,
            "accuracy": round(_safe_div(all_fields_correct, all_fields_evaluated), 4),
            "lenient_accuracy": round(_safe_div(all_fields_correct + all_fields_partial, all_fields_evaluated), 4),
            "strict": _prf_bundle(all_fields_correct, all_fields_evaluated, all_fields_gt_expected),
            "lenient": _prf_bundle(all_fields_correct + all_fields_partial, all_fields_evaluated, all_fields_gt_expected),
            "weighted": _prf_bundle(all_fields_correct + 0.5 * all_fields_partial, all_fields_evaluated, all_fields_gt_expected),
            "macro": {
                "strict": _macro_prf(all_field_views, "strict"),
                "lenient": _macro_prf(all_field_views, "lenient"),
                "weighted": _macro_prf(all_field_views, "weighted"),
            },
        },
        "changed_fields": {
            "gold_changed_total": changed_gold_total,
            "predicted_changed_total": changed_pred_total,
            "changed_correct": changed_correct,
            "changed_partial": changed_partial,
            "changed_incorrect": changed_incorrect,
            "strict": _prf_bundle(changed_correct, changed_pred_total, changed_gold_total),
            "lenient": _prf_bundle(changed_correct + changed_partial, changed_pred_total, changed_gold_total),
            "weighted": _prf_bundle(changed_correct + 0.5 * changed_partial, changed_pred_total, changed_gold_total),
            "macro": {
                "strict": _macro_prf(changed_views, "strict"),
                "lenient": _macro_prf(changed_views, "lenient"),
                "weighted": _macro_prf(changed_views, "weighted"),
            },
        },
        "preservation": {
            "preserved_total": preserved_total,
            "preserved_kept": preserved_kept,
            "collateral_edit_count": collateral_edits,
            "preservation_success_rate": round(_safe_div(preserved_kept, preserved_total), 4),
            "preservation_error_rate": round(_safe_div(collateral_edits, preserved_total), 4),
        },
        "transition_accuracy": {
            "total_evaluated_leaf_fields": ta_evaluated,
            "gt_expected_total": ta_gt_expected,
            "correct": ta_correct,
            "partial": ta_partial,
            "incorrect": ta_incorrect,
            "accuracy": round(_safe_div(ta_correct, ta_evaluated), 4),
            "by_transition": agg_by_transition,
        },
        "task_diagnostics": {
            "unsupported_commit_count": sum(
                int(view.get("unsupported_commit_count", 0) or 0) for view in task_views
            ),
            "unsupported_commit_rate": round(
                _safe_div(
                    sum(int(view.get("unsupported_commit_count", 0) or 0) for view in task_views),
                    whole_evaluated,
                ),
                4,
            ),
            "collateral_edit_count": sum(
                int(view.get("collateral_edit_count", 0) or 0) for view in task_views
            ),
            "failed_correction_count": sum(
                int(view.get("failed_correction_count", 0) or 0) for view in task_views
            ),
            "failed_retraction_count": sum(
                int(view.get("failed_retraction_count", 0) or 0) for view in task_views
            ),
            "repeat_routing_error_count": sum(
                int(view.get("repeat_routing_error_count", 0) or 0) for view in task_views
            ),
            "input_incoherence_overcommit_count": sum(
                int(view.get("input_incoherence_overcommit_count", 0) or 0) for view in task_views
            ),
            "input_incoherence_forbidden_commit_count": sum(
                int(view.get("input_incoherence_forbidden_commit_count", 0) or 0) for view in task_views
            ),
            "correction_success": _success_summary("correction_success"),
            "retraction_success": _success_summary("retraction_success"),
            "history_recovery_success": _success_summary("history_recovery_success"),
            "gate_execution_success": _success_summary("gate_execution_success"),
            "repeat_group_execution_success": _success_summary("repeat_group_execution_success"),
            "input_incoherence_success": _success_summary("input_incoherence_success"),
        },
        "repeat_groups": {
            "item_count": len(repeat_views),
            "item_repeat_involved_count": rg_item_involved,
            "prior_repeat_instance_total": rg_prior_total,
            "gold_repeat_instance_total": rg_gt_total,
            "resulting_repeat_instance_total": rg_resulting_total,
            "repeat_instance_coordination_total": rg_coord_total,
            "repeat_group_coordination_count": rg_group_coord_total,
            "gold_repeat_instance_operation_total": rg_op_total,
            "spurious_instance_count": rg_spurious,
            "alignment_failure_count": rg_failures,
        },
        "error_groups": aggregate_error_group_views(error_group_views),
        "field_classes": {
            "commitment_core": sum(int(view.get("commitment_core", 0) or 0) for view in field_class_views),
            "unknown_equivalent": sum(int(view.get("unknown_equivalent", 0) or 0) for view in field_class_views),
            "auxiliary_unscored_note": sum(
                int(view.get("auxiliary_unscored_note", view.get("optional_note", 0)) or 0)
                for view in field_class_views
            ),
        },
        "auxiliary_unscored_notes": {
            "total_auxiliary_unscored_note_fields": auxiliary_note_total,
            "candidate_non_empty_total": auxiliary_note_candidate_non_empty,
            "gold_grounded_total": auxiliary_note_gold_grounded,
            "gold_empty_total": auxiliary_note_gold_empty,
            "hallucination_count": auxiliary_note_hallucinations,
            "grounded_capture_count": auxiliary_note_grounded_captures,
            "abstention_success_count": auxiliary_note_abstention_successes,
            "auxiliary_note_hallucination_rate": round(
                _safe_div(auxiliary_note_hallucinations, auxiliary_note_candidate_non_empty),
                4,
            ),
            "auxiliary_note_grounded_capture_rate": round(
                _safe_div(auxiliary_note_grounded_captures, auxiliary_note_gold_grounded),
                4,
            ),
            "auxiliary_note_abstention_acceptance_rate": round(
                _safe_div(auxiliary_note_abstention_successes, auxiliary_note_gold_empty),
                4,
            ),
        },
    }
