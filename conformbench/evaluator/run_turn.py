"""
Turn-level evaluator for committed record state after one user utterance.

Evaluates a pilot-runtime ``turn_result.json`` against the expanded pilot
``ground_truth.json`` contract.
"""

from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

AUTO_FILLED_REPEAT_FIELDS: dict[str, set[str]] = {
    "participating_organisations": {"participant_number"},
}

def _is_empty_diff_value(value):
    return value is None or value == "" or value == [] or value == {}


def _diff_change_kind(prior_val, result_val):
    if _is_empty_diff_value(prior_val) and not _is_empty_diff_value(result_val):
        return "set"
    if not _is_empty_diff_value(prior_val) and _is_empty_diff_value(result_val):
        return "clear"
    if prior_val != result_val:
        return "changed"
    return "preserved"


def derive_gold_state_diff(
    prior_state: dict[str, Any] | None,
    gold_resulting_state: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    before = prior_state or {}
    after = gold_resulting_state or {}
    all_keys = sorted(set(before) | set(after))
    diff: list[dict[str, Any]] = []
    for key in all_keys:
        prior_val = before.get(key)
        result_val = after.get(key)
        kind = _diff_change_kind(prior_val, result_val)
        diff.append({"slot": key, "prior_value": prior_val, "resulting_value": result_val, "change_kind": kind})
    return diff

from .scoring import (
    infer_scoring_profile,
    is_empty_for_scoring,
    normalize_runtime_empty,
    values_equal_for_scoring,
)

from .alignment import (
    describe_match,
    extract_agent_instances,
    is_delete_instance_directive,
    match_instances,
    without_delete_instance_marker,
)
from .hard import evaluate_hard_fields, get_questionnaire_field_info
from .models import AlignmentEntry, FieldVerdict

_WS = re.compile(r"\s+")
_REPEAT_FIELD_RE = re.compile(r"^(?P<group>\w+)\[(?P<idx>\d+)\]\.(?P<field>.+)$")
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_MISSING = object()
_STRICT_ALIGNMENT_KEY_FRAGMENTS = (
    "plate",
    "email",
    "identifier",
    "_id",
    " id",
    "number",
    "pic",
)
_PROMOTABLE_ALIGNMENT_KEY_PARTIAL_REASONS = {"over_specified", "ambiguous"}
_ALIAS_CONTRACT_PARTIAL_REASONS = {"wrong_choice", "omission"}
_IDENTITY_ALIAS_CONTRACT_CUES = (
    "clear referent",
    "describing the vehicle",
    "identity",
    "instance",
    "natural language referent",
    "referent/alias",
    "role label",
)
_SEMANTIC_ALIAS_FIELD_FRAGMENTS = (
    "wp_number",
    "participant_number_short_name",
    "participant_short_name",
    "lead_participant_short_name",
    "responsible_participant",
)
_PARTICIPANT_REFERENCE_FIELD_IDS = {
    "wp_lead_participant_number",
    "wp_lead_participant_short_name",
    "deliverable_lead_participant_short_name",
    "participant_number_short_name",
    "subcontracting_participant_number_short_name",
    "equipment_participant_number_short_name",
    "responsible_participant",
}
_SEMANTIC_ALIAS_CONTRACT_CUES = (
    "e.g.",
    "for example",
    "such as",
    "number/code",
    "number / short name",
    "number/short name",
    "short name/acronym",
    "must match a participant",
    "consistent with",
    "separator",
    "delimiter",
    "separated",
)
_SEMANTIC_ALIAS_STRICT_BASE_FIELDS = {
    "call_identifier",
    "contract_or_grant_number",
    "participant_number",
    "participant_pic",
    "pic",
    "proposal_number",
    "topic_identifier",
}
_SEMANTIC_ALIAS_STRICT_FIELD_TYPES = {
    "boolean",
    "date",
    "datetime",
    "integer",
    "number",
    "time",
}
_SEMANTIC_ALIAS_STRICT_FIELD_FRAGMENTS = (
    "amount",
    "budget",
    "cost",
    "deadline",
    "duration",
    "email",
    "eur",
    "identifier",
    "licence_plate",
    "license_plate",
    "number_of_",
    "phone",
    "plate",
    "postal",
    "postcode",
    "url",
    "vat",
)
_GENERIC_IDENTITY_TOKENS = {
    "a",
    "accident",
    "at",
    "car",
    "claimant",
    "driver",
    "fault",
    "fled",
    "hit",
    "me",
    "mine",
    "my",
    "other",
    "our",
    "perpetrator",
    "reporting",
    "that",
    "the",
    "their",
    "them",
    "third",
    "truck",
    "us",
    "vehicle",
    "victim",
}
_SELF_ROLE_TOKENS = {
    "claimant",
    "insured",
    "me",
    "mine",
    "my",
    "our",
    "reporting",
    "us",
    "victim",
}
_OTHER_ROLE_TOKENS = {
    "at",
    "fault",
    "fled",
    "hit",
    "other",
    "perpetrator",
    "third",
}
_SUPPORT_STOPWORDS = {
    "about",
    "after",
    "also",
    "and",
    "because",
    "before",
    "from",
    "into",
    "other",
    "that",
    "the",
    "then",
    "there",
    "this",
    "with",
}


class ConversationMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: str
    content: str


class RepeatGroupInstanceContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ground_truth_index: int
    identity_anchor: str | None = None
    fields: dict[str, dict[str, Any]] = Field(default_factory=dict)


class RepeatGroupContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    alignment_keys: list[str] = Field(default_factory=list)
    instances: list[RepeatGroupInstanceContract] = Field(default_factory=list)


class GroundTruthContract(BaseModel):
    model_config = ConfigDict(extra="allow")

    scenario_id: str
    scenario_label: str
    questionnaire: str
    source_questionnaire: str | None = None
    current_utterance: str
    prior_state: dict[str, Any]
    visible_history: list[ConversationMessage] = Field(default_factory=list)
    gold_resulting_state: dict[str, Any]
    gold_state_diff: list[dict[str, Any]] = Field(default_factory=list)
    fields: dict[str, dict[str, Any]] = Field(default_factory=dict)
    repeat_groups: dict[str, RepeatGroupContract] = Field(default_factory=dict)
    gold_annotations: dict[str, Any] = Field(default_factory=dict)
    semantic_ius: list[dict[str, Any]] = Field(default_factory=list)
    forbidden_commits: list[dict[str, Any]] = Field(default_factory=list)
    primary_delta_type: str | None = None
    state_condition: str | None = None
    evidence: dict[str, Any] = Field(default_factory=dict)
    derived_variables: dict[str, Any] = Field(default_factory=dict)
    targeted_failure_mode: list[str] = Field(default_factory=list)
    evaluation_error_groups: list[dict[str, Any]] = Field(default_factory=list)
    difficulty_profile: dict[str, Any] = Field(default_factory=dict)


class TurnResultContract(BaseModel):
    model_config = ConfigDict(extra="allow")

    scenario_id: str
    questionnaire: str
    scenario: str
    state: str
    utterance_id: str
    turn_index: int
    current_utterance: str
    visible_history: list[ConversationMessage] = Field(default_factory=list)
    answers_before: dict[str, Any] = Field(default_factory=dict)
    answers_after: dict[str, Any] = Field(default_factory=dict)
    answer_updates: dict[str, Any] = Field(default_factory=dict)
    agent_response: Any = None
    is_complete: bool
    difficulty: str | None = None


class TurnEvaluationResult(BaseModel):
    """Evaluation output for a single turn."""

    turn: int = 0
    user_message: str = ""
    fields_filled: int = 0
    candidate_resulting_state: dict[str, Any] = Field(default_factory=dict)
    candidate_state_diff: list[dict[str, Any]] = Field(default_factory=list)
    field_results: dict[str, FieldVerdict] = Field(default_factory=dict)
    alignment_log: list[AlignmentEntry] = Field(default_factory=list)
    unmatched_fields: list[str] = Field(default_factory=list)
    derived_variables: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)
    prompt_trace: list[dict[str, Any]] = Field(default_factory=list)
    summary: dict[str, int] = Field(default_factory=dict)


def _normalise_result_value(value: Any) -> Any:
    """Normalise runtime sentinel values to record-state values."""
    return normalize_runtime_empty(value)


def _is_open_runtime_placeholder(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() in {
        "<open>",
        "open",
    }


def _normalise_scalar_for_compare(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        return _WS.sub(" ", value).strip().lower()
    if isinstance(value, list):
        return [_normalise_scalar_for_compare(item) for item in value]
    if isinstance(value, dict):
        return {
            key: _normalise_scalar_for_compare(item)
            for key, item in sorted(value.items())
        }
    return value


def _values_equal(
    left: Any,
    right: Any,
    *,
    scoring_profile: str | None = None,
) -> bool:
    return values_equal_for_scoring(
        _normalise_result_value(left),
        _normalise_result_value(right),
        scoring_profile=scoring_profile,
    )


def _is_empty_value(
    value: Any,
    *,
    scoring_profile: str | None = None,
) -> bool:
    return is_empty_for_scoring(
        _normalise_result_value(value),
        scoring_profile=scoring_profile,
    )


def _has_raw_prior_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip() not in {"", "<open>"}
    if isinstance(value, (list, dict)):
        return bool(value)
    return True


def _change_kind(
    prior_value: Any,
    resulting_value: Any,
    *,
    scoring_profile: str | None = None,
) -> str:
    if _is_empty_value(prior_value, scoring_profile=scoring_profile) and not _is_empty_value(
        resulting_value,
        scoring_profile=scoring_profile,
    ):
        return "set"
    if not _is_empty_value(prior_value, scoring_profile=scoring_profile) and _is_empty_value(
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


def _history_dump(history: list[ConversationMessage]) -> list[dict[str, Any]]:
    return [message.model_dump() for message in history]


def _validate_runtime_pair(
    turn_result: dict[str, Any],
    ground_truth: dict[str, Any],
) -> tuple[TurnResultContract, GroundTruthContract]:
    turn = TurnResultContract.model_validate(turn_result)
    gt = GroundTruthContract.model_validate(ground_truth)

    if turn.scenario_id != gt.scenario_id:
        raise ValueError(
            f"scenario_id mismatch: turn_result={turn.scenario_id!r}, "
            f"ground_truth={gt.scenario_id!r}"
        )
    if turn.questionnaire != gt.questionnaire:
        raise ValueError(
            f"questionnaire mismatch: turn_result={turn.questionnaire!r}, "
            f"ground_truth={gt.questionnaire!r}"
        )
    if turn.current_utterance != gt.current_utterance:
        raise ValueError("current_utterance mismatch between turn_result and ground_truth")
    if _history_dump(turn.visible_history) != _history_dump(gt.visible_history):
        raise ValueError("visible_history mismatch between turn_result and ground_truth")

    return turn, gt


def _generation_model_id_from_turn(turn_result: dict[str, Any]) -> str | None:
    generation = (turn_result.get("provenance") or {}).get("generation") or {}
    if not isinstance(generation, dict):
        return None
    model = generation.get("model")
    if not isinstance(model, dict):
        return None
    for key in (
        "resolved_model_version",
        "resolved_model_name",
        "model",
        "requested_model",
    ):
        value = model.get(key)
        if value:
            return str(value)
    return None


def _build_candidate_repeat_group(
    answers_after: dict[str, Any],
    answer_updates: dict[str, Any],
    prior_state: dict[str, Any],
    group_name: str,
) -> list[dict[str, Any]]:
    raw_indexed_instances = extract_agent_instances(answers_after, group_name)
    indexed_instances = _non_deleted_instances(raw_indexed_instances)
    raw_explicit_indexed_updates = extract_agent_instances(answer_updates, group_name)
    explicit_indexed_updates = _non_deleted_instances(raw_explicit_indexed_updates)
    deleted_indices = {
        idx
        for indexed in (raw_indexed_instances, raw_explicit_indexed_updates)
        for idx, fields in indexed.items()
        if is_delete_instance_directive(fields)
    }
    rows: list[dict[str, Any]] = []
    prior_rows = _prior_repeat_rows(prior_state, group_name)

    if deleted_indices:
        rows = _drop_deleted_repeat_rows(prior_rows, deleted_indices)
    elif group_name in answers_after and isinstance(answers_after[group_name], list):
        raw_rows = answers_after[group_name]
        if all(isinstance(row, dict) for row in raw_rows):
            rows = [dict(row) for row in raw_rows]

    if rows:
        # Some agents carry both a nested repeat-group row and flat indexed
        # qids.  The flat view often contains untouched '<open>' placeholders;
        # only explicit indexed updates should override the nested row.  When a
        # delete directive is present, indexed answers may be the compacted
        # post-delete rows, so they can safely overlay the prior-row base.
        row_overlays = explicit_indexed_updates or (
            indexed_instances if deleted_indices else {}
        )
        if row_overlays:
            if max(row_overlays) >= len(rows):
                rows.extend(
                    {} for _ in range(max(row_overlays) + 1 - len(rows))
                )
            for idx, fields in row_overlays.items():
                rows[idx] = {**rows[idx], **fields}
        return [_normalise_result_value(row) for row in rows]

    overlay_instances = explicit_indexed_updates or indexed_instances
    if overlay_instances:
        if not rows:
            rows = [dict(row) for row in prior_rows]
        if max(overlay_instances) >= len(rows):
            rows.extend({} for _ in range(max(overlay_instances) + 1 - len(rows)))

        for idx, fields in overlay_instances.items():
            materialized_fields: dict[str, Any] = {}
            for field_name, value in fields.items():
                qid = f"{group_name}[{idx}].{field_name}"
                if _is_open_runtime_placeholder(value) and qid not in answer_updates:
                    continue
                materialized_fields[field_name] = value
            rows[idx] = {**rows[idx], **materialized_fields}

    if rows:
        return [_normalise_result_value(row) for row in rows]

    return [
        _normalise_result_value(indexed_instances[idx])
        for idx in sorted(indexed_instances.keys())
    ]


def _prior_repeat_rows(
    prior_state: dict[str, Any],
    group_name: str,
) -> list[dict[str, Any]]:
    raw_rows = prior_state.get(group_name)
    if not isinstance(raw_rows, list) or not all(isinstance(row, dict) for row in raw_rows):
        return []
    return [dict(row) for row in raw_rows]


def _drop_deleted_repeat_rows(
    rows: list[dict[str, Any]],
    deleted_indices: set[int],
) -> list[dict[str, Any]]:
    return [
        dict(row)
        for idx, row in enumerate(rows)
        if idx not in deleted_indices
    ]


def _non_deleted_instances(
    indexed_instances: dict[int, dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    return {
        idx: without_delete_instance_marker(fields)
        for idx, fields in indexed_instances.items()
        if not is_delete_instance_directive(fields)
    }


def _build_candidate_resulting_state(
    turn: TurnResultContract,
    gt: GroundTruthContract,
) -> dict[str, Any]:
    candidate: dict[str, Any] = {}
    answers_after = turn.answers_after

    for field_id, gold_value in gt.gold_resulting_state.items():
        if isinstance(gold_value, list) and all(isinstance(item, dict) for item in gold_value):
            candidate[field_id] = _build_candidate_repeat_group(
                answers_after,
                turn.answer_updates,
                turn.answers_before or gt.prior_state,
                field_id,
            )
        else:
            candidate[field_id] = _normalise_result_value(
                answers_after.get(field_id, gt.prior_state.get(field_id))
            )

    return candidate


def _count_leaf_fields(state: dict[str, Any]) -> int:
    total = 0
    for value in state.values():
        if isinstance(value, list) and all(isinstance(item, dict) for item in value):
            total += sum(len(item) for item in value)
        else:
            total += 1
    return total


def _get_repeat_prior_value(
    prior_state: dict[str, Any],
    group_name: str,
    gt_index: int,
    field_name: str,
) -> Any:
    group_value = prior_state.get(group_name)
    if not isinstance(group_value, list) or gt_index >= len(group_value):
        return None
    instance = group_value[gt_index]
    if not isinstance(instance, dict):
        return None
    return instance.get(field_name)


def _participant_reference_rows(source: str, rows_value: Any) -> list[dict[str, Any]]:
    if not isinstance(rows_value, list):
        return []
    rows: list[dict[str, Any]] = []
    for idx, participant in enumerate(rows_value):
        if not isinstance(participant, dict):
            continue
        row = {
            "source": source,
            "row": idx + 1,
            "participant_number": participant.get("participant_number"),
            "participant_short_name": participant.get("participant_short_name"),
            "participant_legal_name": participant.get("participant_legal_name"),
            "participant_role": participant.get("participant_role"),
            "participant_pic": participant.get("participant_pic"),
            "participant_country": participant.get("participant_country"),
        }
        if any(
            value not in (None, "", [], {})
            for key, value in row.items()
            if key not in {"source", "row"}
        ):
            rows.append(row)
    return rows


def _gt_repeat_expected_rows(gt_repeat_groups: dict[str, Any] | None, group_name: str) -> list[dict[str, Any]]:
    group = (gt_repeat_groups or {}).get(group_name)
    if not isinstance(group, dict):
        return []
    expected_rows: list[dict[str, Any]] = []
    for fallback_idx, instance in enumerate(group.get("instances") or []):
        if not isinstance(instance, dict):
            continue
        fields = instance.get("fields")
        if not isinstance(fields, dict):
            fields = instance
        row: dict[str, Any] = {}
        for field_name, spec in fields.items():
            if field_name == "ground_truth_index":
                continue
            if isinstance(spec, dict):
                row[field_name] = spec.get("expected_summary", spec.get("expected"))
            else:
                row[field_name] = spec
        if row:
            row.setdefault("_ground_truth_index", instance.get("ground_truth_index", fallback_idx))
            expected_rows.append(row)
    return expected_rows


def _field_leaf_id(qid: str) -> str:
    return qid.rsplit(".", 1)[-1]


def _reference_context_for_qid(
    qid: str,
    *,
    prior_state: dict[str, Any],
    candidate_state: dict[str, Any] | None = None,
    gt_repeat_groups: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if _field_leaf_id(qid) not in _PARTICIPANT_REFERENCE_FIELD_IDS:
        return []

    rows: list[dict[str, Any]] = []
    rows.extend(
        _participant_reference_rows(
            "prior_state",
            prior_state.get("participating_organisations"),
        )
    )
    rows.extend(
        _participant_reference_rows(
            "candidate_resulting_state",
            (candidate_state or {}).get("participating_organisations"),
        )
    )
    rows.extend(
        _participant_reference_rows(
            "gold_resulting_state",
            _gt_repeat_expected_rows(gt_repeat_groups, "participating_organisations"),
        )
    )
    if not rows:
        return []

    return [
        {
            "label": "Participant number / short-name options",
            "source_field": "participating_organisations",
            "columns": [
                "source",
                "row",
                "participant_number",
                "participant_short_name",
                "participant_legal_name",
                "participant_role",
                "participant_pic",
                "participant_country",
            ],
            "rows": rows,
        }
    ]


def _build_soft_payload(
    *,
    qid: str,
    candidate_value: Any,
    prior_value: Any,
    gt_entry: dict[str, Any],
    questionnaire_path: Path | None = None,
    reference_context: list[dict[str, Any]] | None = None,
    semantic_ius_by_field: dict[str, list[dict[str, Any]]] | None = None,
    semantic_field_path: str | None = None,
) -> dict[str, Any]:
    payload = {
        "qid": qid,
        "candidate_value": candidate_value,
        "prior_value": prior_value,
        "gt_entry": gt_entry,
    }
    semantic_field_path = semantic_field_path or qid
    if semantic_ius_by_field:
        semantic_ius = semantic_ius_by_field.get(semantic_field_path)
        if semantic_ius:
            payload["semantic_ius"] = deepcopy(semantic_ius)
            payload["semantic_ius_field_path"] = semantic_field_path
    if reference_context:
        payload["reference_context"] = reference_context
    field_metadata = get_questionnaire_field_info(qid, questionnaire_path)
    if field_metadata:
        payload["field_metadata"] = field_metadata
    return payload


def _semantic_ius_by_field(
    semantic_ius: list[dict[str, Any]] | None,
) -> dict[str, list[dict[str, Any]]]:
    by_field: dict[str, list[dict[str, Any]]] = {}
    for iu in semantic_ius or []:
        if not isinstance(iu, dict):
            continue
        field_path = iu.get("field_path")
        if not field_path:
            continue
        by_field.setdefault(str(field_path), []).append(dict(iu))
    return by_field


def _field_contract_text(
    qid: str,
    questionnaire_path: Path | None = None,
) -> str:
    metadata = get_questionnaire_field_info(qid, questionnaire_path)
    chunks: list[str] = []
    for key in ("question_text", "label", "gold_standard", "normalization_rules"):
        value = metadata.get(key)
        if value:
            chunks.append(str(value))
    information_units = metadata.get("information_units")
    if isinstance(information_units, list):
        for unit in information_units:
            if not isinstance(unit, dict):
                continue
            for key in ("id", "name", "description"):
                value = unit.get(key)
                if value:
                    chunks.append(str(value))
    return " ".join(chunks).lower()


def _field_contract_allows_identity_alias(
    qid: str,
    questionnaire_path: Path | None = None,
) -> bool:
    return any(
        cue in _field_contract_text(qid, questionnaire_path)
        for cue in _IDENTITY_ALIAS_CONTRACT_CUES
    )


def _field_name_for_qid(qid: str) -> str:
    return qid.rsplit(".", 1)[-1].lower()


def _normalise_field_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _is_scalar_semantic_alias_value(value: Any) -> bool:
    return not isinstance(value, (bool, dict, list, set, tuple))


def _candidate_equals_attached_forbidden_commit(
    qid: str,
    candidate_value: Any,
    gt_entry: dict[str, Any],
) -> bool:
    commits = gt_entry.get("_forbidden_commits") or []
    if not isinstance(commits, list):
        return False
    scoring_profile = infer_scoring_profile(
        qid,
        gt_entry,
        expected=gt_entry.get("expected"),
    )
    for commit in commits:
        if not isinstance(commit, dict):
            continue
        wrong_value = commit.get("wrong_value", commit.get("resulting_value"))
        if values_equal_for_scoring(
            candidate_value,
            wrong_value,
            scoring_profile=scoring_profile,
        ):
            return True
    return False


def _is_strict_semantic_alias_field(
    qid: str,
    metadata: dict[str, Any],
    *,
    questionnaire_path: Path | None = None,
) -> bool:
    field_name = _normalise_field_token(_field_name_for_qid(qid))
    if field_name in {"wp_number", "participant_number_short_name"}:
        return False
    if field_name.endswith("_number_short_name"):
        return False
    if "short_name" in field_name or field_name == "responsible_participant":
        return False

    field_type = str(metadata.get("type") or "").lower()
    if field_type in _SEMANTIC_ALIAS_STRICT_FIELD_TYPES:
        return True
    if field_name in _SEMANTIC_ALIAS_STRICT_BASE_FIELDS:
        return True
    if field_name.endswith("_id"):
        return True
    if any(fragment in field_name for fragment in _SEMANTIC_ALIAS_STRICT_FIELD_FRAGMENTS):
        return True

    contract_text = _field_contract_text(qid, questionnaire_path)
    if "exactly as stated" in contract_text:
        return True
    if "exact value" in contract_text:
        return True
    if "numeric " in contract_text and "short name" not in contract_text:
        return True
    return False


def _field_contract_allows_semantic_alias(
    qid: str,
    *,
    questionnaire_path: Path | None = None,
) -> bool:
    field_name = _normalise_field_token(_field_name_for_qid(qid))
    if any(fragment in field_name for fragment in _SEMANTIC_ALIAS_FIELD_FRAGMENTS):
        return True
    contract_text = _field_contract_text(qid, questionnaire_path)
    return any(cue in contract_text for cue in _SEMANTIC_ALIAS_CONTRACT_CUES)


def _should_try_semantic_alias_match(
    *,
    qid: str,
    candidate_value: Any,
    gt_entry: dict[str, Any],
    questionnaire_path: Path | None = None,
) -> bool:
    expected = gt_entry.get("expected_summary") or gt_entry.get("expected")
    scoring_profile = infer_scoring_profile(qid, gt_entry, expected=expected)

    if not _is_scalar_semantic_alias_value(candidate_value):
        return False
    if not _is_scalar_semantic_alias_value(expected):
        return False
    if _is_empty_value(candidate_value, scoring_profile=scoring_profile):
        return False
    if _is_empty_value(expected, scoring_profile=scoring_profile):
        return False
    if values_equal_for_scoring(
        candidate_value,
        expected,
        scoring_profile=scoring_profile,
    ):
        return False
    if _candidate_equals_attached_forbidden_commit(qid, candidate_value, gt_entry):
        return False

    metadata = get_questionnaire_field_info(qid, questionnaire_path)
    if _is_strict_semantic_alias_field(
        qid,
        metadata,
        questionnaire_path=questionnaire_path,
    ):
        return False
    return _field_contract_allows_semantic_alias(
        qid,
        questionnaire_path=questionnaire_path,
    )


def _requires_empty_gold_exactness(qid: str, gt_entry: dict[str, Any]) -> bool:
    """Route null-gold fields through deterministic final-state scoring.

    A semantic field with gold ``null`` is still a null final-state commitment.
    Non-null candidates must not be sent to the semantic judge for rescue.
    """
    expected = gt_entry.get("expected")
    scoring_profile = infer_scoring_profile(qid, gt_entry, expected=expected)
    return is_empty_for_scoring(expected, scoring_profile=scoring_profile)


def _expected_value_from_gt_entry(gt_entry: dict[str, Any]) -> Any:
    if "expected_summary" in gt_entry and gt_entry.get("expected_summary") is not None:
        return gt_entry.get("expected_summary")
    return gt_entry.get("expected")


def _candidate_preserved_prior_when_gold_changed(
    qid: str,
    candidate_value: Any,
    prior_value: Any,
    gt_entry: dict[str, Any],
    *,
    expected_value: Any = _MISSING,
) -> bool:
    """True when the candidate left a field unchanged although gold changed.

    This is a universal pre-routing guard: exact matching, set/list matching,
    semantic mismatch escalation, and semantic IU scoring should never rescue a
    field whose resulting value simply preserved the prior state while gold
    required a different resulting value.
    """
    expected = (
        _expected_value_from_gt_entry(gt_entry)
        if expected_value is _MISSING
        else expected_value
    )
    scoring_profile = infer_scoring_profile(qid, gt_entry, expected=expected)
    return _values_equal(
        candidate_value,
        prior_value,
        scoring_profile=scoring_profile,
    ) and not _values_equal(
        expected,
        prior_value,
        scoring_profile=scoring_profile,
    )


def _preserved_prior_miss_verdict(
    qid: str,
    candidate_value: Any,
    prior_value: Any,
    gt_entry: dict[str, Any],
    *,
    expected_value: Any = _MISSING,
) -> FieldVerdict:
    expected = (
        _expected_value_from_gt_entry(gt_entry)
        if expected_value is _MISSING
        else expected_value
    )
    return FieldVerdict(
        correctness="incorrect",
        source="prior_state",
        decision_source="hard_preserve_miss",
        reasoning=(
            f"Candidate preserved prior value {prior_value!r} for {qid}, but "
            f"gold required resulting value {expected!r}. Scoring stopped "
            "before exact, set/list, semantic, or semantic-IU routes because "
            "this is a missed set/change/clear, not a paraphrase or alias "
            "question."
        ),
    )


def _mark_exact_mismatch_escalation(
    payload: dict[str, Any],
    verdict: FieldVerdict,
) -> dict[str, Any]:
    payload["semantic_exact_mismatch_check"] = True
    payload["hard_eval_reasoning"] = verdict.reasoning
    return payload


def _promote_semantic_alias_matches(
    hard_results: dict[str, FieldVerdict],
    hard_tuples: list[tuple[str, Any, dict[str, Any]]],
    *,
    prior_value_for_qid: Callable[[str], Any],
    current_utterance: str,
    prior_state: dict[str, Any],
    visible_history: list[dict[str, Any]],
    candidate_state: dict[str, Any] | None = None,
    gt_repeat_groups: dict[str, Any] | None = None,
    label: str = "",
    model_id: str | None = None,
    reasoning_effort: str | None = None,
    exclude_judge_model: str | None = None,
    form_title: str | None = None,
    questionnaire_path: Path | None = None,
    semantic_ius_by_field: dict[str, list[dict[str, Any]]] | None = None,
    trace_collector: list[dict[str, Any]] | None = None,
) -> None:
    from .soft import evaluate_soft_fields

    payloads: list[dict[str, Any]] = []
    for qid, agent_value, gt_entry in hard_tuples:
        verdict = hard_results.get(qid)
        if verdict is None or verdict.correctness != "incorrect":
            continue
        if verdict.decision_source == "hard_preserve_miss":
            continue
        if not _should_try_semantic_alias_match(
            qid=qid,
            candidate_value=agent_value,
            gt_entry=gt_entry,
            questionnaire_path=questionnaire_path,
        ):
            continue
        payload = _build_soft_payload(
            qid=qid,
            candidate_value=agent_value,
            prior_value=prior_value_for_qid(qid),
            gt_entry=gt_entry,
            questionnaire_path=questionnaire_path,
            reference_context=_reference_context_for_qid(
                qid,
                prior_state=prior_state,
                candidate_state=candidate_state,
                gt_repeat_groups=gt_repeat_groups,
            ),
        )
        payload["semantic_equivalence_check"] = True
        payload["hard_eval_reasoning"] = verdict.reasoning
        payloads.append(payload)

    if not payloads:
        return

    logger.info(
        f"{label}evaluating {len(payloads)} schema alias candidates with alias judge"
    )
    semantic_results = evaluate_soft_fields(
        payloads,
        current_utterance=current_utterance,
        prior_state=prior_state,
        visible_history=visible_history,
        model_id=model_id,
        reasoning_effort=reasoning_effort,
        exclude_model=exclude_judge_model,
        rotation_key_prefix=f"{label}semantic-alias",
        form_title=form_title,
        trace_collector=trace_collector,
    )

    for qid, semantic_verdict in semantic_results.items():
        if semantic_verdict.correctness != "correct":
            continue
        verdict = hard_results.get(qid)
        if verdict is None or verdict.correctness != "incorrect":
            continue

        previous_reason = verdict.reasoning.strip()
        verdict.correctness = "correct"
        verdict.partial_reason = None
        _set_verdict_support_source(verdict, semantic_verdict.source)
        verdict.decision_source = "semantic_alias_judge"
        verdict.postprocess_reason = (
            "Promoted to correct because the semantic judge accepted the "
            "candidate as a schema-supported alias or formatting equivalent "
            "of the gold value."
        )
        semantic_reason = semantic_verdict.reasoning.strip()
        verdict.reasoning = verdict.postprocess_reason
        if semantic_reason:
            verdict.reasoning += f" Judge note: {semantic_reason}"
        if previous_reason:
            verdict.reasoning += f" Original hard-eval note: {previous_reason}"


def _alignment_key_field_metadata(
    group_name: str,
    alignment_keys: list[str],
    *,
    questionnaire_path: Path | None = None,
) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    for key in alignment_keys:
        info = get_questionnaire_field_info(
            f"{group_name}[0].{key}",
            questionnaire_path,
        )
        if info:
            metadata[key] = info
    return metadata


def _repeat_slot_parts(slot: str) -> tuple[str, int, str] | None:
    match = _REPEAT_FIELD_RE.match(slot)
    if not match:
        return None
    return match.group("group"), int(match.group("idx")), match.group("field")


def _attach_forbidden_commits(
    gt_fields: dict[str, dict[str, Any]],
    gt_repeat_groups: dict[str, Any],
    forbidden_commits: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Copy GT field specs and attach slot-local forbidden commit metadata."""
    fields = deepcopy(gt_fields)
    repeat_groups = deepcopy(gt_repeat_groups)

    for commit in forbidden_commits:
        if not isinstance(commit, dict):
            continue
        slot = commit.get("slot") or commit.get("field")
        if not slot:
            continue
        slot = str(slot)
        repeat_parts = _repeat_slot_parts(slot)
        if repeat_parts is None:
            entry = fields.get(slot)
            if isinstance(entry, dict):
                entry.setdefault("_forbidden_commits", []).append(commit)
            continue

        group_name, gt_idx, field_name = repeat_parts
        group = repeat_groups.get(group_name)
        if not isinstance(group, dict):
            continue
        instances = group.get("instances")
        if not isinstance(instances, list):
            continue
        for fallback_idx, instance in enumerate(instances):
            if not isinstance(instance, dict):
                continue
            if instance.get("ground_truth_index", fallback_idx) != gt_idx:
                continue
            fields_map = instance.get("fields")
            if not isinstance(fields_map, dict):
                fields_map = instance
            field_entry = fields_map.get(field_name)
            if isinstance(field_entry, dict):
                field_entry.setdefault("_forbidden_commits", []).append(commit)
            break

    return fields, repeat_groups


def _normalise_forbidden_commits(commits: list[Any]) -> list[dict[str, Any]]:
    """Return forbidden commits as plain dictionaries."""
    normalised: list[dict[str, Any]] = []
    for commit in commits:
        if isinstance(commit, dict):
            normalised.append(dict(commit))
        elif hasattr(commit, "model_dump"):
            dumped = commit.model_dump()
            if isinstance(dumped, dict):
                normalised.append(dumped)
    return normalised


def _field_matches_forbidden_commit(
    qid: str,
    candidate_value: Any,
    *,
    gt_entry: dict[str, Any] | None,
    alignment_log: list[AlignmentEntry],
    forbidden_commits: list[dict[str, Any]],
) -> bool:
    candidate_slots = {qid}
    repeat_parts = _repeat_slot_parts(qid)
    if repeat_parts is not None:
        group_name, agent_idx, field_name = repeat_parts
        for entry in alignment_log:
            if (
                entry.status == "matched"
                and entry.group == group_name
                and entry.agent_index == agent_idx
            ):
                candidate_slots.add(f"{group_name}[{entry.gt_index}].{field_name}")
                break

    scoring_profile = infer_scoring_profile(
        qid,
        gt_entry or {},
        expected=(gt_entry or {}).get("expected"),
    )
    for commit in forbidden_commits:
        if not isinstance(commit, dict):
            continue
        slot = commit.get("slot") or commit.get("field")
        if str(slot or "") not in candidate_slots:
            continue
        wrong_value = commit.get("wrong_value", commit.get("resulting_value"))
        if values_equal_for_scoring(
            candidate_value,
            wrong_value,
            scoring_profile=scoring_profile,
        ):
            return True
    return False


def _candidate_value_for_qid(
    qid: str,
    candidate_state: dict[str, Any],
    raw_candidate_state: dict[str, Any] | None = None,
) -> Any:
    raw_value: Any = None
    raw_found = False
    if raw_candidate_state:
        if qid in raw_candidate_state:
            raw_value = raw_candidate_state.get(qid)
            raw_found = True
        else:
            repeat_parts = _repeat_slot_parts(qid)
            if repeat_parts is not None:
                group_name, agent_idx, field_name = repeat_parts
                raw_rows = raw_candidate_state.get(group_name)
                if isinstance(raw_rows, list) and agent_idx < len(raw_rows):
                    raw_row = raw_rows[agent_idx]
                    if isinstance(raw_row, dict) and field_name in raw_row:
                        raw_value = raw_row.get(field_name)
                        raw_found = True

    def _prefer_raw_if_normalized_empty(value: Any) -> Any:
        if value is None and raw_found:
            return raw_value
        return value

    if qid in candidate_state:
        return _prefer_raw_if_normalized_empty(candidate_state.get(qid))
    repeat_parts = _repeat_slot_parts(qid)
    if repeat_parts is None:
        return _prefer_raw_if_normalized_empty(None)
    group_name, agent_idx, field_name = repeat_parts
    rows = candidate_state.get(group_name)
    if not isinstance(rows, list) or agent_idx >= len(rows):
        return _prefer_raw_if_normalized_empty(None)
    row = rows[agent_idx]
    value = row.get(field_name) if isinstance(row, dict) else None
    return _prefer_raw_if_normalized_empty(value)


def _visible_text_supports_candidate(
    candidate_value: Any,
    *,
    current_utterance: str,
    visible_history: list[dict[str, Any]],
) -> bool:
    if candidate_value is None or isinstance(candidate_value, (bool, int, float)):
        return False
    candidate_text = str(candidate_value).lower().replace("other:", " ")
    tokens = [
        token
        for token in _TOKEN_RE.findall(candidate_text)
        if len(token) > 2 and token not in _SUPPORT_STOPWORDS
    ]
    if not tokens:
        return False
    visible_text = " ".join(
        [current_utterance]
        + [str(message.get("content") or "") for message in visible_history]
    ).lower()
    hits = sum(1 for token in tokens if token in visible_text)
    return hits >= max(1, len(tokens) // 2)


def _prior_value_for_qid(
    qid: str,
    prior_state: dict[str, Any],
    alignment_log: list[AlignmentEntry],
) -> Any:
    if qid in prior_state:
        return prior_state.get(qid)
    repeat_parts = _repeat_slot_parts(qid)
    if repeat_parts is None:
        return None
    group_name, agent_idx, field_name = repeat_parts
    gt_idx = agent_idx
    for entry in alignment_log:
        if (
            entry.status == "matched"
            and entry.group == group_name
            and entry.agent_index == agent_idx
        ):
            gt_idx = entry.gt_index
            break
    return _get_repeat_prior_value(prior_state, group_name, gt_idx, field_name)


def _set_verdict_support_source(verdict: FieldVerdict, source: str | None) -> None:
    verdict.set_support_source(source)


def _refine_field_result_sources(
    results: dict[str, FieldVerdict],
    *,
    candidate_state: dict[str, Any],
    raw_candidate_state: dict[str, Any] | None = None,
    prior_state: dict[str, Any],
    alignment_log: list[AlignmentEntry],
    gt_fields: dict[str, dict[str, Any]],
    gt_repeat_groups: dict[str, Any],
    forbidden_commits: list[dict[str, Any]],
    current_utterance: str,
    visible_history: list[dict[str, Any]],
) -> None:
    """Make source labels describe candidate support, not just gold support."""

    def _gt_entry_for_qid(qid: str) -> dict[str, Any] | None:
        if qid in gt_fields:
            entry = gt_fields.get(qid)
            return entry if isinstance(entry, dict) else None
        repeat_parts = _repeat_slot_parts(qid)
        if repeat_parts is None:
            return None
        group_name, agent_idx, field_name = repeat_parts
        gt_idx = agent_idx
        for entry in alignment_log:
            if (
                entry.status == "matched"
                and entry.group == group_name
                and entry.agent_index == agent_idx
            ):
                gt_idx = entry.gt_index
                break
        group = gt_repeat_groups.get(group_name)
        if not isinstance(group, dict):
            return None
        for fallback_idx, instance in enumerate(group.get("instances") or []):
            if not isinstance(instance, dict):
                continue
            if instance.get("ground_truth_index", fallback_idx) != gt_idx:
                continue
            fields_map = instance.get("fields")
            if not isinstance(fields_map, dict):
                fields_map = instance
            entry = fields_map.get(field_name)
            return entry if isinstance(entry, dict) else None
        return None

    for qid, verdict in results.items():
        candidate_value = _candidate_value_for_qid(qid, candidate_state, raw_candidate_state)
        gt_entry = _gt_entry_for_qid(qid)
        prior_value = _prior_value_for_qid(qid, prior_state, alignment_log)
        scoring_profile = infer_scoring_profile(
            qid,
            gt_entry or {},
            expected=(gt_entry or {}).get("expected"),
        )
        if _is_empty_value(candidate_value):
            if (
                candidate_value is not None
                and _has_raw_prior_value(prior_value)
                and values_equal_for_scoring(
                    candidate_value,
                    prior_value,
                    scoring_profile=scoring_profile,
                )
            ):
                _set_verdict_support_source(verdict, "prior_state")
            elif verdict.source == "prior_state" and not _has_raw_prior_value(prior_value):
                _set_verdict_support_source(verdict, None)
            continue
        if values_equal_for_scoring(
            candidate_value,
            prior_value,
            scoring_profile=scoring_profile,
        ):
            if not _visible_text_supports_candidate(
                candidate_value,
                current_utterance=current_utterance,
                visible_history=visible_history,
            ) or (gt_entry or {}).get("present_in_utterance") is False:
                _set_verdict_support_source(verdict, "prior_state")
                continue
        if verdict.correctness == "correct":
            continue
        if _field_matches_forbidden_commit(
            qid,
            candidate_value,
            gt_entry=gt_entry,
            alignment_log=alignment_log,
            forbidden_commits=forbidden_commits,
        ):
            _set_verdict_support_source(verdict, "unsupported_inference")
            continue
        if (
            verdict.source == "fabricated"
            and _visible_text_supports_candidate(
                candidate_value,
                current_utterance=current_utterance,
                visible_history=visible_history,
            )
        ):
            _set_verdict_support_source(verdict, "extracted")


def _is_strict_alignment_key(field_name: str) -> bool:
    """Return True for identity keys that should remain exact after alignment."""
    normalised = field_name.replace("-", "_").lower()
    return any(fragment in normalised for fragment in _STRICT_ALIGNMENT_KEY_FRAGMENTS)


def _identity_tokens(value: Any) -> set[str]:
    return set(_TOKEN_RE.findall(str(value or "").lower()))


def _has_descriptive_identity_tokens(value: Any) -> bool:
    tokens = _identity_tokens(value)
    return any(
        token not in _GENERIC_IDENTITY_TOKENS and len(token) > 1
        for token in tokens
    )


def _role_alias_side(tokens: set[str]) -> str | None:
    if tokens & _SELF_ROLE_TOKENS:
        return "self"
    if tokens & _OTHER_ROLE_TOKENS:
        return "other"
    return None


def _identity_alias_looks_compatible(candidate_value: Any, expected_value: Any) -> bool:
    """Guard broad alias rescue against contradictory generic role labels."""
    candidate_tokens = _identity_tokens(candidate_value)
    expected_tokens = _identity_tokens(expected_value)
    if candidate_tokens & expected_tokens:
        return True

    candidate_side = _role_alias_side(candidate_tokens)
    expected_side = _role_alias_side(expected_tokens)
    if candidate_side and expected_side:
        return candidate_side == expected_side

    if _has_descriptive_identity_tokens(candidate_value):
        return True
    if _has_descriptive_identity_tokens(expected_value):
        return True
    return False


def _expected_value_for_matched_alignment_key(
    *,
    group_name: str,
    gt_idx: int,
    field_name: str,
    gt_repeat_groups: dict[str, Any],
) -> Any:
    group = gt_repeat_groups.get(group_name)
    if not isinstance(group, dict):
        return None
    for instance in group.get("instances") or []:
        if not isinstance(instance, dict):
            continue
        if instance.get("ground_truth_index") != gt_idx:
            continue
        fields = instance.get("fields")
        if not isinstance(fields, dict):
            return None
        entry = fields.get(field_name)
        if isinstance(entry, dict):
            return entry.get("expected_summary") or entry.get("expected")
    return None


def _is_promotable_alignment_key_partial(
    *,
    qid: str,
    verdict: FieldVerdict,
    candidate_value: Any,
    expected_value: Any,
) -> bool:
    """Return True when a matched alignment key is only an alias mismatch."""
    if verdict.partial_reason in _PROMOTABLE_ALIGNMENT_KEY_PARTIAL_REASONS:
        return True
    if verdict.partial_reason not in _ALIAS_CONTRACT_PARTIAL_REASONS:
        return False
    if not _field_contract_allows_identity_alias(qid):
        return False
    return _identity_alias_looks_compatible(candidate_value, expected_value)


def _promote_matched_alignment_key_partials(
    results: dict[str, FieldVerdict],
    *,
    candidate_state: dict[str, Any],
    alignment_log: list[AlignmentEntry],
    gt_repeat_groups: dict[str, Any],
) -> None:
    """Treat matched semantic identity aliases as correct.

    Repeat-instance alignment is itself a semantic identity judgement, but it
    only licenses narrow alias/extra-detail fixes. Non-strict alignment keys
    such as ``role`` or ``vehicle_type_make_model`` are promoted when the judge
    already marked the core identity as defensible, or when the schema contract
    says the field stores natural referent aliases and the candidate does not
    contradict the expected identity side. Hallucinations and mixed partials
    remain partial.
    """
    matched_pairs = {
        (entry.group, entry.agent_index): entry.gt_index
        for entry in alignment_log
        if entry.status == "matched" and entry.agent_index is not None
    }

    for qid, verdict in results.items():
        if verdict.correctness != "partially_correct":
            continue
        repeat_parts = _repeat_slot_parts(qid)
        if repeat_parts is None:
            continue
        group_name, agent_idx, field_name = repeat_parts
        gt_idx = matched_pairs.get((group_name, agent_idx))
        if gt_idx is None:
            continue

        group = gt_repeat_groups.get(group_name)
        if not isinstance(group, dict):
            continue
        alignment_keys = set(group.get("alignment_keys") or [])
        if field_name not in alignment_keys or _is_strict_alignment_key(field_name):
            continue

        candidate_value = _candidate_value_for_qid(qid, candidate_state)
        if _is_empty_value(candidate_value):
            continue
        if verdict.source == "fabricated":
            continue
        expected_value = _expected_value_for_matched_alignment_key(
            group_name=group_name,
            gt_idx=gt_idx,
            field_name=field_name,
            gt_repeat_groups=gt_repeat_groups,
        )
        if not _is_promotable_alignment_key_partial(
            qid=qid,
            verdict=verdict,
            candidate_value=candidate_value,
            expected_value=expected_value,
        ):
            continue

        previous_reason = verdict.reasoning.strip()
        if verdict.original_correctness is None:
            verdict.original_correctness = verdict.correctness
        if verdict.original_source is None:
            verdict.original_source = verdict.support_source
        if verdict.original_partial_reason is None:
            verdict.original_partial_reason = verdict.partial_reason
        verdict.correctness = "correct"
        verdict.partial_reason = None
        if verdict.source in {None, "unsupported_inference"}:
            _set_verdict_support_source(verdict, "likely_inferred")
        verdict.decision_source = "repeat_alignment_postprocess"
        verdict.postprocess_reason = (
            "Promoted to correct because the repeat-instance matcher aligned "
            "this row semantically and this non-strict alignment key identifies "
            "the same instance."
        )
        verdict.reasoning = verdict.postprocess_reason
        if previous_reason:
            verdict.reasoning += f" Original judge note: {previous_reason}"


def _evaluate_instance_fields(
    results: dict[str, FieldVerdict],
    group_name: str,
    agent_idx: int,
    agent_fields: dict[str, Any],
    gt_inst: dict[str, Any],
    *,
    current_utterance: str,
    prior_state: dict[str, Any],
    gold_resulting_state: dict[str, Any],
    visible_history: list[dict[str, Any]],
    candidate_state: dict[str, Any] | None = None,
    gt_repeat_groups: dict[str, Any] | None = None,
    model_id: str | None = None,
    reasoning_effort: str | None = None,
    exclude_judge_model: str | None = None,
    label: str = "",
    form_title: str | None = None,
    questionnaire_path: Path | None = None,
    semantic_ius_by_field: dict[str, list[dict[str, Any]]] | None = None,
    trace_collector: list[dict[str, Any]] | None = None,
) -> None:
    from .soft import evaluate_soft_fields

    inst_hard: list[tuple[str, Any, dict[str, Any]]] = []
    inst_soft: list[dict[str, Any]] = []
    gt_idx = gt_inst.get("ground_truth_index", -1)
    _auto_filled = AUTO_FILLED_REPEAT_FIELDS.get(group_name, set())

    for field_name, gt_entry in gt_inst["fields"].items():
        if field_name in _auto_filled:
            continue
        qid = f"{group_name}[{agent_idx}].{field_name}"
        agent_val = agent_fields.get(field_name)

        prior_value = _get_repeat_prior_value(
            prior_state,
            group_name,
            gt_idx,
            field_name,
        )
        gold_value = _get_repeat_prior_value(
            gold_resulting_state,
            group_name,
            gt_idx,
            field_name,
        )
        if _candidate_preserved_prior_when_gold_changed(
            qid,
            agent_val,
            prior_value,
            gt_entry,
            expected_value=gold_value,
        ):
            results[qid] = _preserved_prior_miss_verdict(
                qid,
                agent_val,
                prior_value,
                gt_entry,
                expected_value=gold_value,
            )
            continue
        strategy = gt_entry.get("strategy", "semantic")
        if strategy in {"exact", "set_match"} or _requires_empty_gold_exactness(
            qid,
            gt_entry,
        ):
            inst_hard.append((qid, agent_val, gt_entry))
        else:
            inst_soft.append(
                _build_soft_payload(
                    qid=qid,
                    candidate_value=agent_val,
                    prior_value=prior_value,
                    gt_entry=gt_entry,
                    questionnaire_path=questionnaire_path,
                    reference_context=_reference_context_for_qid(
                        qid,
                        prior_state=prior_state,
                        candidate_state=candidate_state,
                        gt_repeat_groups=gt_repeat_groups,
                    ),
                    semantic_ius_by_field=semantic_ius_by_field,
                    semantic_field_path=f"{group_name}[{gt_idx}].{field_name}",
                )
            )

    if inst_hard:
        hard_results = evaluate_hard_fields(inst_hard, questionnaire_path=questionnaire_path)
        for qid, verdict in list(hard_results.items()):
            if verdict.correctness == "needs_semantic":
                for tuple_qid, agent_val, gt_entry in inst_hard:
                    if tuple_qid != qid:
                        continue
                    field_name = qid.split(".", 1)[1]
                    prior_value = _get_repeat_prior_value(
                        prior_state,
                        group_name,
                        gt_idx,
                        field_name,
                    )
                    payload = _build_soft_payload(
                        qid=qid,
                        candidate_value=agent_val,
                        prior_value=prior_value,
                        gt_entry=gt_entry,
                        questionnaire_path=questionnaire_path,
                        reference_context=_reference_context_for_qid(
                            qid,
                            prior_state=prior_state,
                            candidate_state=candidate_state,
                            gt_repeat_groups=gt_repeat_groups,
                        ),
                        semantic_ius_by_field=semantic_ius_by_field,
                        semantic_field_path=f"{group_name}[{gt_idx}].{field_name}",
                    )
                    if _should_try_semantic_alias_match(
                        qid=qid,
                        candidate_value=agent_val,
                        gt_entry=gt_entry,
                        questionnaire_path=questionnaire_path,
                    ):
                        hard_results[qid] = FieldVerdict(
                            correctness="incorrect",
                            source=verdict.source,
                            decision_source="hard_eval",
                            reasoning=verdict.reasoning,
                        )
                        break
                    else:
                        _mark_exact_mismatch_escalation(payload, verdict)
                    inst_soft.append(payload)
                    del hard_results[qid]
                    break
                else:
                    del hard_results[qid]
        _promote_semantic_alias_matches(
            hard_results,
            inst_hard,
            prior_value_for_qid=lambda qid: _get_repeat_prior_value(
                prior_state,
                group_name,
                gt_idx,
                qid.split(".", 1)[1],
            ),
            current_utterance=current_utterance,
            prior_state=prior_state,
            visible_history=visible_history,
            candidate_state=candidate_state,
            gt_repeat_groups=gt_repeat_groups,
            label=label,
            model_id=model_id,
            reasoning_effort=reasoning_effort,
            exclude_judge_model=exclude_judge_model,
            form_title=form_title,
            questionnaire_path=questionnaire_path,
            trace_collector=trace_collector,
        )
        results.update(hard_results)

    if inst_soft:
        logger.info(
            f"{label}evaluating {len(inst_soft)} soft fields for "
            f"{group_name}[{agent_idx}] (GT #{gt_idx})"
        )
        results.update(
            evaluate_soft_fields(
                inst_soft,
                current_utterance=current_utterance,
                prior_state=prior_state,
                visible_history=visible_history,
                model_id=model_id,
                reasoning_effort=reasoning_effort,
                exclude_model=exclude_judge_model,
                rotation_key_prefix=f"{label}{group_name}[{agent_idx}]",
                form_title=form_title,
                trace_collector=trace_collector,
            )
        )


def _evaluate_fields(
    candidate_state: dict[str, Any],
    gold_resulting_state: dict[str, Any],
    gt_fields: dict[str, dict[str, Any]],
    gt_repeat_groups: dict[str, Any],
    *,
    current_utterance: str,
    prior_state: dict[str, Any],
    visible_history: list[dict[str, Any]],
    label: str = "",
    model_id: str | None = None,
    reasoning_effort: str | None = None,
    exclude_judge_model: str | None = None,
    form_title: str | None = None,
    questionnaire_path: Path | None = None,
    semantic_ius_by_field: dict[str, list[dict[str, Any]]] | None = None,
    trace_collector: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, FieldVerdict], list[AlignmentEntry], dict[str, int]]:
    from .soft import evaluate_soft_fields

    results: dict[str, FieldVerdict] = {}
    alignment_log: list[AlignmentEntry] = []

    hard_tuples: list[tuple[str, Any, dict[str, Any]]] = []
    soft_payloads: list[dict[str, Any]] = []

    for qid, gt_entry in gt_fields.items():
        agent_val = candidate_state.get(qid)
        prior_value = prior_state.get(qid)
        gold_value = gold_resulting_state.get(qid, _MISSING)
        if _candidate_preserved_prior_when_gold_changed(
            qid,
            agent_val,
            prior_value,
            gt_entry,
            expected_value=gold_value,
        ):
            results[qid] = _preserved_prior_miss_verdict(
                qid,
                agent_val,
                prior_value,
                gt_entry,
                expected_value=gold_value,
            )
            continue
        strategy = gt_entry.get("strategy", "semantic")
        if strategy in {"exact", "set_match"} or _requires_empty_gold_exactness(
            qid,
            gt_entry,
        ):
            hard_tuples.append((qid, agent_val, gt_entry))
        else:
            soft_payloads.append(
                _build_soft_payload(
                    qid=qid,
                    candidate_value=agent_val,
                    prior_value=prior_value,
                    gt_entry=gt_entry,
                    questionnaire_path=questionnaire_path,
                    reference_context=_reference_context_for_qid(
                        qid,
                        prior_state=prior_state,
                        candidate_state=candidate_state,
                        gt_repeat_groups=gt_repeat_groups,
                    ),
                    semantic_ius_by_field=semantic_ius_by_field,
                )
            )

    if hard_tuples:
        logger.info(f"{label}evaluating {len(hard_tuples)} top-level hard fields")
        hard_results = evaluate_hard_fields(hard_tuples, questionnaire_path=questionnaire_path)
        for qid, verdict in list(hard_results.items()):
            if verdict.correctness == "needs_semantic":
                for tuple_qid, agent_val, gt_entry in hard_tuples:
                    if tuple_qid != qid:
                        continue
                    prior_value = prior_state.get(qid)
                    payload = _build_soft_payload(
                        qid=qid,
                        candidate_value=agent_val,
                        prior_value=prior_value,
                        gt_entry=gt_entry,
                        questionnaire_path=questionnaire_path,
                        reference_context=_reference_context_for_qid(
                            qid,
                            prior_state=prior_state,
                            candidate_state=candidate_state,
                            gt_repeat_groups=gt_repeat_groups,
                        ),
                        semantic_ius_by_field=semantic_ius_by_field,
                    )
                    if _should_try_semantic_alias_match(
                        qid=qid,
                        candidate_value=agent_val,
                        gt_entry=gt_entry,
                        questionnaire_path=questionnaire_path,
                    ):
                        hard_results[qid] = FieldVerdict(
                            correctness="incorrect",
                            source=verdict.source,
                            decision_source="hard_eval",
                            reasoning=verdict.reasoning,
                        )
                        break
                    else:
                        _mark_exact_mismatch_escalation(payload, verdict)
                    soft_payloads.append(payload)
                    del hard_results[qid]
                    break
                else:
                    del hard_results[qid]
        _promote_semantic_alias_matches(
            hard_results,
            hard_tuples,
            prior_value_for_qid=lambda qid: prior_state.get(qid),
            current_utterance=current_utterance,
            prior_state=prior_state,
            visible_history=visible_history,
            candidate_state=candidate_state,
            gt_repeat_groups=gt_repeat_groups,
            label=label,
            model_id=model_id,
            reasoning_effort=reasoning_effort,
            exclude_judge_model=exclude_judge_model,
            form_title=form_title,
            questionnaire_path=questionnaire_path,
            trace_collector=trace_collector,
        )
        results.update(hard_results)

    if soft_payloads:
        logger.info(f"{label}evaluating {len(soft_payloads)} top-level soft fields")
        results.update(
            evaluate_soft_fields(
                soft_payloads,
                current_utterance=current_utterance,
                prior_state=prior_state,
                visible_history=visible_history,
                model_id=model_id,
                reasoning_effort=reasoning_effort,
                exclude_model=exclude_judge_model,
                rotation_key_prefix=f"{label}top-level",
                form_title=form_title,
                trace_collector=trace_collector,
            )
        )

    for group_name, group_gt in gt_repeat_groups.items():
        _auto_filled = AUTO_FILLED_REPEAT_FIELDS.get(group_name, set())
        alignment_keys = group_gt.get("alignment_keys", [])
        gt_instances = group_gt.get("instances", [])
        agent_rows = candidate_state.get(group_name)
        if not isinstance(agent_rows, list):
            agent_rows = []

        agent_as_indexed = {
            idx: row for idx, row in enumerate(agent_rows) if isinstance(row, dict)
        }
        try:
            alignment_field_metadata = _alignment_key_field_metadata(
                group_name,
                alignment_keys,
                questionnaire_path=questionnaire_path,
            )
            matches = match_instances(
                agent_as_indexed,
                gt_instances,
                alignment_keys,
                group_name=group_name,
                form_title=form_title,
                model_id=model_id,
                reasoning_effort=reasoning_effort,
                exclude_model=exclude_judge_model,
                trace_collector=trace_collector,
                current_utterance=current_utterance,
                visible_history=visible_history,
                prior_state=prior_state,
                alignment_field_metadata=alignment_field_metadata,
            )
        except TypeError as exc:
            if "unexpected keyword argument" not in str(exc):
                raise
            matches = match_instances(
                agent_as_indexed,
                gt_instances,
                alignment_keys,
                group_name=group_name,
                form_title=form_title,
            )
        matched_agent_indices: set[int] = set()

        for gt_inst, agent_idx in matches:
            gt_idx = gt_inst["ground_truth_index"]
            if agent_idx is None:
                logger.warning(
                    f"{label}GT {group_name}[{gt_idx}] unmatched — "
                    f"instance not extracted by agent"
                )
                alignment_log.append(
                    AlignmentEntry(
                        group=group_name,
                        gt_index=gt_idx,
                        status="missed",
                    )
                )
                # Create field verdicts for the missing instance.
                # Fields with non-empty GT expected values are marked incorrect;
                # fields with empty GT values are correct only if the candidate
                # also left the same-index slot empty. This matters for empty
                # placeholder rows that represent clearing a prior instance.
                for field_name, gt_entry in gt_inst["fields"].items():
                    if field_name in _auto_filled:
                        continue
                    qid = f"{group_name}[?gt{gt_idx}].{field_name}"
                    expected = gt_entry.get("expected")
                    scoring_profile = infer_scoring_profile(
                        qid,
                        gt_entry,
                        expected=expected,
                    )
                    candidate_rows = candidate_state.get(group_name)
                    candidate_value = None
                    if isinstance(candidate_rows, list) and gt_idx < len(candidate_rows):
                        candidate_row = candidate_rows[gt_idx]
                        if isinstance(candidate_row, dict):
                            candidate_value = candidate_row.get(field_name)

                    if _is_empty_value(expected, scoring_profile=scoring_profile):
                        if not _is_empty_value(candidate_value, scoring_profile=scoring_profile):
                            prior_value = _get_repeat_prior_value(
                                prior_state,
                                group_name,
                                gt_idx,
                                field_name,
                            )
                            source = (
                                "prior_state"
                                if _values_equal(
                                    candidate_value,
                                    prior_value,
                                    scoring_profile=scoring_profile,
                                )
                                else "fabricated"
                            )
                            results[qid] = FieldVerdict(
                                correctness="incorrect",
                                source=source,
                                decision_source="instance_alignment",
                                reasoning=(
                                    f"Instance not cleared by agent. Expected empty "
                                    f"value for {qid}, but candidate retained "
                                    f"{candidate_value!r}."
                                ),
                            )
                            continue
                        results[qid] = FieldVerdict(
                            correctness="correct",
                            source=None,
                            decision_source="instance_alignment",
                            reasoning=f"Both GT and agent are empty for {qid}.",
                        )
                    else:
                        results[qid] = FieldVerdict(
                            correctness="incorrect",
                            source=None,
                            decision_source="instance_alignment",
                            reasoning=f"Instance not extracted by agent. Expected: '{expected}'.",
                        )
                continue

            matched_agent_indices.add(agent_idx)
            match_key = describe_match(agent_as_indexed[agent_idx], alignment_keys)
            alignment_log.append(
                AlignmentEntry(
                    group=group_name,
                    gt_index=gt_idx,
                    agent_index=agent_idx,
                    status="matched",
                    matched_on=match_key,
                )
            )
            _evaluate_instance_fields(
                results,
                group_name,
                agent_idx,
                agent_as_indexed[agent_idx],
                gt_inst,
                current_utterance=current_utterance,
                prior_state=prior_state,
                gold_resulting_state=gold_resulting_state,
                visible_history=visible_history,
                candidate_state=candidate_state,
                gt_repeat_groups=gt_repeat_groups,
                model_id=model_id,
                reasoning_effort=reasoning_effort,
                exclude_judge_model=exclude_judge_model,
                label=label,
                form_title=form_title,
                questionnaire_path=questionnaire_path,
                semantic_ius_by_field=semantic_ius_by_field,
                trace_collector=trace_collector,
            )

        for agent_idx in sorted(set(agent_as_indexed.keys()) - matched_agent_indices):
            alignment_log.append(
                AlignmentEntry(
                    group=group_name,
                    gt_index=-1,
                    agent_index=agent_idx,
                    status="hallucinated",
                )
            )
            # Penalise each non-empty field in the hallucinated instance:
            # the agent invented data with no ground-truth backing.
            agent_fields = agent_as_indexed[agent_idx]
            for field_name, field_value in agent_fields.items():
                if not _is_empty_value(field_value):
                    qid = f"{group_name}[{agent_idx}].{field_name}"
                    prior_value = _get_repeat_prior_value(
                        prior_state,
                        group_name,
                        agent_idx,
                        field_name,
                    )
                    source = (
                        "prior_state"
                        if _values_equal(field_value, prior_value)
                        else "fabricated"
                    )
                    reasoning = (
                        f"Unmatched retained prior instance: no matching "
                        f"ground-truth instance for {group_name}[{agent_idx}]."
                        if source == "prior_state"
                        else (
                            f"Hallucinated instance: no matching ground-truth "
                            f"instance for {group_name}[{agent_idx}]."
                        )
                    )
                    results[qid] = FieldVerdict(
                        correctness="incorrect",
                        source=source,
                        decision_source="instance_alignment",
                        reasoning=reasoning,
                    )

    summary = {
        "evaluated": len(results),
        "correct": 0,
        "partially_correct": 0,
        "incorrect": 0,
    }
    for verdict in results.values():
        summary[verdict.correctness] = summary.get(verdict.correctness, 0) + 1

    return results, alignment_log, summary


def _collect_unmatched_fields(
    turn: TurnResultContract,
    gt: GroundTruthContract,
) -> list[str]:
    gold_top_level = set(gt.gold_resulting_state.keys())
    repeat_groups = set(gt.repeat_groups.keys())

    unmatched: list[str] = []
    for qid in turn.answer_updates:
        if qid in gold_top_level:
            continue
        if qid in repeat_groups:
            continue
        if any(qid.startswith(f"{group}[") for group in repeat_groups):
            continue
        unmatched.append(qid)
    return sorted(unmatched)


def evaluate_turn_result(
    turn_result: dict[str, Any],
    expected_updates: dict[str, Any],
    *,
    model_id: str | None = None,
    reasoning_effort: str | None = None,
    questionnaire_path: Path | None = None,
    runner_version: dict[str, Any] | None = None,
) -> TurnEvaluationResult:
    """Evaluate a pilot-runtime ``turn_result.json`` against ``ground_truth.json``."""
    turn, gt = _validate_runtime_pair(turn_result, expected_updates)

    if questionnaire_path is not None:
        from . import hard as _hard_mod

        _hard_mod._set_active_questionnaire_path(questionnaire_path)

    form_title: str | None = None
    if questionnaire_path is not None:
        try:
            q_data = json.loads(Path(questionnaire_path).read_text())
            form_title = q_data.get("title")
        except Exception:
            form_title = None

    candidate_state = _build_candidate_resulting_state(turn, gt)
    candidate_state_diff = derive_gold_state_diff(gt.prior_state, candidate_state)
    unmatched_fields = _collect_unmatched_fields(turn, gt)
    generation_model_id = _generation_model_id_from_turn(turn_result)

    label = (
        f"[{turn.scenario}/"
        f"{turn.state}/"
        f"{turn.utterance_id}] "
    )

    prompt_trace: list[dict[str, Any]] = []
    forbidden_commits = _normalise_forbidden_commits(gt.forbidden_commits)
    semantic_ius_by_field = _semantic_ius_by_field(gt.semantic_ius)
    gt_fields, gt_repeat_groups = _attach_forbidden_commits(
        gt.fields,
        {
            group_name: group.model_dump()
            for group_name, group in gt.repeat_groups.items()
        },
        forbidden_commits,
    )

    results, alignment_log, field_summary = _evaluate_fields(
        candidate_state,
        gt.gold_resulting_state,
        gt_fields,
        gt_repeat_groups,
        current_utterance=turn.current_utterance,
        prior_state=gt.prior_state,
        visible_history=_history_dump(gt.visible_history),
        label=label,
        model_id=model_id,
        reasoning_effort=reasoning_effort,
        exclude_judge_model=generation_model_id,
        form_title=form_title,
        questionnaire_path=questionnaire_path,
        semantic_ius_by_field=semantic_ius_by_field,
        trace_collector=prompt_trace,
    )

    _refine_field_result_sources(
        results,
        candidate_state=candidate_state,
        raw_candidate_state=turn.answers_after,
        prior_state=gt.prior_state,
        alignment_log=alignment_log,
        gt_fields=gt_fields,
        gt_repeat_groups=gt_repeat_groups,
        forbidden_commits=forbidden_commits,
        current_utterance=turn.current_utterance,
        visible_history=_history_dump(gt.visible_history),
    )
    _promote_matched_alignment_key_partials(
        results,
        candidate_state=candidate_state,
        alignment_log=alignment_log,
        gt_repeat_groups=gt_repeat_groups,
    )

    correctness_counts = {
        "correct": 0,
        "partially_correct": 0,
        "incorrect": 0,
    }
    for verdict in results.values():
        correctness_counts[verdict.correctness] = (
            correctness_counts.get(verdict.correctness, 0) + 1
        )

    summary: dict[str, int] = {
        "total": _count_leaf_fields(candidate_state),
        "evaluated": len(results),
        "unmatched": len(unmatched_fields),
        "correct": correctness_counts.get("correct", 0),
        "partially_correct": correctness_counts.get("partially_correct", 0),
        "incorrect": correctness_counts.get("incorrect", 0),
    }

    return TurnEvaluationResult(
        turn=turn.turn_index,
        user_message=turn.current_utterance,
        fields_filled=_count_leaf_fields(candidate_state),
        candidate_resulting_state=candidate_state,
        candidate_state_diff=candidate_state_diff,
        field_results=results,
        alignment_log=alignment_log,
        unmatched_fields=unmatched_fields,
        derived_variables=gt.derived_variables,
        provenance={
            "runner": runner_version,
            "generation": (turn_result.get("provenance") or {}).get("generation"),
            "evaluation": {
                "runner": runner_version,
                "model": prompt_trace[0]["model_config"] if prompt_trace else None,
                "models": [
                    trace.get("model_config")
                    for trace in prompt_trace
                    if trace.get("model_config") is not None
                ],
                "excluded_generation_model": generation_model_id,
            },
        },
        prompt_trace=prompt_trace,
        summary=summary,
    )
