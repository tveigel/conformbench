"""Compute diagnostic counts and detail entries for a single evaluated turn.

This is the **single source of truth** for diagnostic metrics such as
``unsupported_commit``, ``missed_supported_update``, ``collateral_edit``,
``failed_correction``, ``failed_retraction``, ``forbidden_commit``,
``other_without_specification``, and ``repeat_group_alignment_error``.

Submission runs call ``build_diagnostics`` from the offline report scripts so
that evaluator output can stay focused on raw field verdicts.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any

from .scoring import (
    SCORING_PROFILE_AUXILIARY_UNSCORED_NOTE,
    infer_scoring_profile,
    is_empty_for_scoring,
    normalize_runtime_empty,
    values_equal_for_scoring,
)

from .models import AlignmentEntry, FieldVerdict
from .hard import get_questionnaire_field_info

# Lazy import to avoid circular dependency — GroundTruthContract lives in
# run_turn.py which imports from this module.
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .run_turn import GroundTruthContract

_REPEAT_FIELD_RE = re.compile(r"^(?P<group>\w+)\[(?P<idx>\d+)\]\.(?P<field>.+)$")
_REPEAT_FIELD_WITH_GT_RE = re.compile(
    r"^(?P<group>\w+)\[(?P<idx>\d+|\?gt\d+)\]\.(?P<field>.+)$"
)
_STRICT_ALIGNMENT_KEY_FRAGMENTS = (
    "plate",
    "email",
    "identifier",
    "_id",
    " id",
    "number",
    "pic",
)
_CHOICE_TYPES = {"single_choice", "multiple_choice"}
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


# ── Helpers (mirrors of run_turn helpers, kept here to avoid coupling) ──


def _normalise(value: Any) -> Any:
    return normalize_runtime_empty(value)


def _values_equal(left: Any, right: Any, *, scoring_profile: str | None = None) -> bool:
    return values_equal_for_scoring(
        _normalise(left), _normalise(right), scoring_profile=scoring_profile,
    )


def _is_empty(value: Any, *, scoring_profile: str | None = None) -> bool:
    return is_empty_for_scoring(_normalise(value), scoring_profile=scoring_profile)


def _change_kind(prior: Any, resulting: Any, *, scoring_profile: str | None = None) -> str:
    if _is_empty(prior, scoring_profile=scoring_profile) and not _is_empty(resulting, scoring_profile=scoring_profile):
        return "set"
    if not _is_empty(prior, scoring_profile=scoring_profile) and _is_empty(resulting, scoring_profile=scoring_profile):
        return "clear"
    if not _values_equal(prior, resulting, scoring_profile=scoring_profile):
        return "changed"
    return "preserved"


def _get_state_value(state: dict[str, Any], slot: str) -> Any:
    if slot in state:
        return state[slot]
    match = _REPEAT_FIELD_RE.match(slot)
    if not match:
        return None
    group_value = state.get(match.group("group"))
    idx = int(match.group("idx"))
    if not isinstance(group_value, list) or idx >= len(group_value):
        return None
    instance = group_value[idx]
    return instance.get(match.group("field")) if isinstance(instance, dict) else None


def _repeat_slot_parts(slot: str) -> tuple[str, int, str] | None:
    match = _REPEAT_FIELD_RE.match(slot)
    if not match:
        return None
    return match.group("group"), int(match.group("idx")), match.group("field")


def _is_bare_other(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() == "other"


def _contains_bare_other(value: Any) -> bool:
    if _is_bare_other(value):
        return True
    if isinstance(value, (list, tuple, set)):
        return any(_is_bare_other(item) for item in value)
    return False


def _iter_candidate_leaf_values(state: dict[str, Any]):
    for slot, value in state.items():
        if isinstance(value, list) and all(isinstance(row, dict) for row in value):
            for index, row in enumerate(value):
                for field, leaf_value in row.items():
                    yield f"{slot}[{index}].{field}", leaf_value
            continue
        yield slot, value


# ── Public API ──────────────────────────────────────────────────────────


def build_diagnostics(
    *,
    gt: GroundTruthContract,
    candidate_state: dict[str, Any],
    alignment_log: list[AlignmentEntry],
    field_verdicts: dict[str, str] | None = None,
    questionnaire_path: Path | None = None,
) -> dict[str, Any]:
    """Compute diagnostic counts and detail entries for one evaluated turn.

    Parameters
    ----------
    gt:
        The ground-truth contract for this turn.
    candidate_state:
        The agent's resulting state (flat dict + repeat-group lists).
    alignment_log:
        Instance-alignment entries produced during evaluation.
    field_verdicts:
        Optional mapping from slot name → correctness string
        (``"correct"``, ``"partially_correct"``, ``"incorrect"``).
        When provided, the judge verdict is used instead of strict
        string equality to decide whether a change is supported.
    questionnaire_path:
        Optional path to questionnaire metadata.  When provided, diagnostics
        can identify bare ``Other`` values on fields with ``other_specify``.
    """
    counts: dict[str, int] = {
        "forbidden_commit": 0,
        "other_without_specification": 0,
        "unsupported_commit": 0,
        "missed_supported_update": 0,
        "collateral_edit": 0,
        "failed_correction": 0,
        "failed_retraction": 0,
        "repeat_group_alignment_error": 0,
        "repeat_missing_instance": 0,
        "repeat_spurious_instance": 0,
        "repeat_wrong_instance_update": 0,
        "repeat_merge_error": 0,
        "repeat_split_error": 0,
    }
    details: dict[str, list[dict[str, Any]]] = {key: [] for key in counts}

    for slot, candidate_value in _iter_candidate_leaf_values(candidate_state):
        if not _contains_bare_other(candidate_value):
            continue
        field_info = get_questionnaire_field_info(slot, questionnaire_path)
        if not bool(field_info.get("other_specify")):
            continue
        if field_info.get("type") not in _CHOICE_TYPES:
            continue
        counts["other_without_specification"] += 1
        details["other_without_specification"].append({
            "slot": slot,
            "candidate_value": candidate_value,
            "field_type": field_info.get("type"),
            "options": field_info.get("options") or [],
            "reason": (
                "Bare 'Other' is incomplete for an other_specify choice field; "
                "the payload must include a free-text specification."
            ),
        })

    repeat_groups_with_gold_operations: set[str] = set()
    for group_name, group in gt.repeat_groups.items():
        prior_rows = gt.prior_state.get(group_name)
        gold_rows = gt.gold_resulting_state.get(group_name)
        if not isinstance(prior_rows, list):
            prior_rows = []
        if not isinstance(gold_rows, list):
            gold_rows = []
        for instance in group.instances:
            gt_index = instance.ground_truth_index
            prior_row = (
                prior_rows[gt_index]
                if gt_index < len(prior_rows) and isinstance(prior_rows[gt_index], dict)
                else {}
            )
            gold_row = (
                gold_rows[gt_index]
                if gt_index < len(gold_rows) and isinstance(gold_rows[gt_index], dict)
                else {}
            )
            for field_name, gt_entry in instance.fields.items():
                scoring_profile = infer_scoring_profile(
                    f"{group_name}[{gt_index}].{field_name}",
                    gt_entry or {},
                    expected=(gt_entry or {}).get("expected"),
                )
                gold_value = gold_row.get(field_name, gt_entry.get("expected"))
                if _change_kind(
                    prior_row.get(field_name),
                    gold_value,
                    scoring_profile=scoring_profile,
                ) != "preserved":
                    repeat_groups_with_gold_operations.add(group_name)
                    break

    def _record_slot(
        slot: str,
        *,
        prior_value: Any,
        gold_value: Any,
        candidate_value: Any,
        gt_entry: dict[str, Any] | None,
    ) -> None:
        scoring_profile = infer_scoring_profile(
            slot, gt_entry or {}, expected=(gt_entry or {}).get("expected", gold_value),
        )
        if scoring_profile == SCORING_PROFILE_AUXILIARY_UNSCORED_NOTE:
            return

        gold_change = _change_kind(prior_value, gold_value, scoring_profile=scoring_profile)
        candidate_change = _change_kind(prior_value, candidate_value, scoring_profile=scoring_profile)
        candidate_matches_gold = _values_equal(candidate_value, gold_value, scoring_profile=scoring_profile)

        # Prefer the LLM judge verdict over strict string equality when available.
        verdict = (field_verdicts or {}).get(slot)
        judged_correct = verdict == "correct" if verdict else candidate_matches_gold

        if gold_change != "preserved" and not judged_correct:
            counts["missed_supported_update"] += 1
            details["missed_supported_update"].append({
                "slot": slot,
                "expected_change": gold_change,
                "prior_value": prior_value,
                "gold_value": gold_value,
                "candidate_value": candidate_value,
            })

        if candidate_change != "preserved" and not judged_correct:
            counts["unsupported_commit"] += 1
            details["unsupported_commit"].append({
                "slot": slot,
                "candidate_change": candidate_change,
                "prior_value": prior_value,
                "gold_value": gold_value,
                "candidate_value": candidate_value,
            })

        if gold_change == "preserved" and candidate_change != "preserved":
            counts["collateral_edit"] += 1
            details["collateral_edit"].append({
                "slot": slot,
                "prior_value": prior_value,
                "candidate_value": candidate_value,
            })
            repeat_parts = _repeat_slot_parts(slot)
            if repeat_parts and repeat_parts[0] in repeat_groups_with_gold_operations:
                counts["repeat_wrong_instance_update"] += 1
                details["repeat_wrong_instance_update"].append({
                    "slot": slot,
                    "prior_value": prior_value,
                    "candidate_value": candidate_value,
                })

        if gt.primary_delta_type == "correct" and gold_change == "changed" and not judged_correct:
            counts["failed_correction"] += 1
            details["failed_correction"].append({
                "slot": slot,
                "prior_value": prior_value,
                "gold_value": gold_value,
                "candidate_value": candidate_value,
            })

        if (
            gt.primary_delta_type == "retract"
            and gold_change == "clear"
            and not _is_empty(candidate_value, scoring_profile=scoring_profile)
        ):
            counts["failed_retraction"] += 1
            details["failed_retraction"].append({
                "slot": slot,
                "prior_value": prior_value,
                "candidate_value": candidate_value,
            })

    # ── Scalar fields ───────────────────────────────────────────────────
    repeat_group_names = set(gt.repeat_groups.keys())
    for slot in sorted(
        (set(gt.prior_state) | set(gt.gold_resulting_state) | set(candidate_state))
        - repeat_group_names
    ):
        _record_slot(
            slot,
            prior_value=gt.prior_state.get(slot),
            gold_value=gt.gold_resulting_state.get(slot),
            candidate_value=candidate_state.get(slot),
            gt_entry=gt.fields.get(slot),
        )

    # ── Repeat-group fields ─────────────────────────────────────────────
    matched_agent_by_gt: dict[str, dict[int, int]] = {}
    for entry in alignment_log:
        if entry.status != "matched" or entry.agent_index is None:
            continue
        matched_agent_by_gt.setdefault(entry.group, {})[entry.gt_index] = entry.agent_index

    for group_name, group in gt.repeat_groups.items():
        prior_rows = gt.prior_state.get(group_name)
        gold_rows = gt.gold_resulting_state.get(group_name)
        candidate_rows = candidate_state.get(group_name)
        if not isinstance(prior_rows, list):
            prior_rows = []
        if not isinstance(gold_rows, list):
            gold_rows = []
        if not isinstance(candidate_rows, list):
            candidate_rows = []

        for instance in group.instances:
            gt_index = instance.ground_truth_index
            agent_index = matched_agent_by_gt.get(group_name, {}).get(gt_index)
            prior_row = (
                prior_rows[gt_index]
                if gt_index < len(prior_rows) and isinstance(prior_rows[gt_index], dict)
                else {}
            )
            gold_row = (
                gold_rows[gt_index]
                if gt_index < len(gold_rows) and isinstance(gold_rows[gt_index], dict)
                else {}
            )
            candidate_row = (
                candidate_rows[agent_index]
                if agent_index is not None and agent_index < len(candidate_rows) and isinstance(candidate_rows[agent_index], dict)
                else {}
            )

            for field_name, gt_entry in instance.fields.items():
                _record_slot(
                    f"{group_name}[{gt_index}].{field_name}",
                    prior_value=prior_row.get(field_name),
                    gold_value=gold_row.get(field_name, gt_entry.get("expected")),
                    candidate_value=candidate_row.get(field_name),
                    gt_entry=gt_entry,
                )

    # ── Forbidden commits ───────────────────────────────────────────────
    for commit in gt.forbidden_commits:
        slot = commit.get("slot") or commit.get("field")
        wrong_value = commit.get("resulting_value", commit.get("wrong_value"))
        if not slot:
            continue
        if _values_equal(_get_state_value(candidate_state, slot), wrong_value):
            counts["forbidden_commit"] += 1
            details["forbidden_commit"].append({
                "slot": slot,
                "wrong_value": wrong_value,
                "reason": commit.get("reason"),
            })

    # ── Repeat-group alignment errors ───────────────────────────────────
    status_by_group: dict[str, Counter[str]] = {}
    for entry in alignment_log:
        status_by_group.setdefault(entry.group, Counter())[entry.status] += 1
        if entry.status == "matched":
            continue
        counts["repeat_group_alignment_error"] += 1
        details["repeat_group_alignment_error"].append(entry.model_dump())
        if entry.status == "missed":
            counts["repeat_missing_instance"] += 1
            details["repeat_missing_instance"].append(entry.model_dump())
        elif entry.status == "hallucinated":
            counts["repeat_spurious_instance"] += 1
            details["repeat_spurious_instance"].append(entry.model_dump())

    for group_name, statuses in status_by_group.items():
        if statuses["missed"] and statuses["matched"]:
            counts["repeat_merge_error"] += 1
            details["repeat_merge_error"].append({
                "group": group_name,
                "missed": statuses["missed"],
                "matched": statuses["matched"],
            })
        if statuses["hallucinated"] and statuses["matched"]:
            counts["repeat_split_error"] += 1
            details["repeat_split_error"].append({
                "group": group_name,
                "hallucinated": statuses["hallucinated"],
                "matched": statuses["matched"],
            })

    return {"counts": counts, "details": details}


def build_field_verdicts_with_gt_aliases(
    field_results: dict[str, dict[str, Any]],
    alignment_log: list[dict[str, Any] | AlignmentEntry],
) -> dict[str, str]:
    """Build a slot→correctness map with gt-index aliases for repeat groups.

    ``_build_diagnostics`` uses gt-index-based slot names
    (``vehicles[0].role``) while field_results uses agent-index-based keys
    (``vehicles[2].role``). This helper creates a mapping that covers both
    naming conventions so the verdict lookup works for all slots.
    """
    verdicts: dict[str, str] = {}
    for qid, v in field_results.items():
        correctness = v.get("correctness", "incorrect") if isinstance(v, dict) else v
        verdicts[qid] = correctness

    for entry in alignment_log:
        if isinstance(entry, dict):
            status = entry.get("status")
            agent_index = entry.get("agent_index")
            gt_index = entry.get("gt_index")
            group = entry.get("group", "")
        else:
            status = entry.status
            agent_index = entry.agent_index
            gt_index = entry.gt_index
            group = entry.group

        if status != "matched" or agent_index is None:
            continue
        agent_prefix = f"{group}[{agent_index}]."
        gt_prefix = f"{group}[{gt_index}]."
        for qid, correctness in list(verdicts.items()):
            if qid.startswith(agent_prefix):
                verdicts[gt_prefix + qid[len(agent_prefix):]] = correctness

    return verdicts


def _is_strict_alignment_key(field_name: str) -> bool:
    normalised = field_name.replace("-", "_").lower()
    return any(fragment in normalised for fragment in _STRICT_ALIGNMENT_KEY_FRAGMENTS)


def _field_contract_text(qid: str, questionnaire_path: Path | None) -> str:
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
    questionnaire_path: Path | None,
) -> bool:
    contract_text = _field_contract_text(qid, questionnaire_path)
    return any(cue in contract_text for cue in _IDENTITY_ALIAS_CONTRACT_CUES)


def _identity_tokens(value: Any) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", str(value or "").lower()))


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
    verdict: dict[str, Any],
    candidate_value: Any,
    expected_value: Any,
    questionnaire_path: Path | None,
) -> bool:
    """True when a matched alignment key is only an alias mismatch."""
    if verdict.get("partial_reason") in _PROMOTABLE_ALIGNMENT_KEY_PARTIAL_REASONS:
        return True
    if verdict.get("partial_reason") not in _ALIAS_CONTRACT_PARTIAL_REASONS:
        return False
    if not _field_contract_allows_identity_alias(qid, questionnaire_path):
        return False
    return _identity_alias_looks_compatible(candidate_value, expected_value)


def _candidate_value_for_repeat_qid(qid: str, candidate_state: dict[str, Any]) -> Any:
    match = _REPEAT_FIELD_WITH_GT_RE.match(qid)
    if not match:
        return None
    idx_token = match.group("idx")
    if idx_token.startswith("?gt"):
        return None
    rows = candidate_state.get(match.group("group"))
    idx = int(idx_token)
    if not isinstance(rows, list) or idx >= len(rows):
        return None
    row = rows[idx]
    if not isinstance(row, dict):
        return None
    return row.get(match.group("field"))


def promote_matched_alignment_key_partials(
    field_results: dict[str, dict[str, Any]],
    *,
    candidate_state: dict[str, Any],
    alignment_log: list[dict[str, Any]],
    gt_repeat_groups: dict[str, Any],
    questionnaire_path: Path | None = None,
) -> dict[str, Any]:
    """Return field verdicts with matched semantic identity aliases corrected.

    This is the read-time companion to the online evaluator's repeat identity
    promotion.  It lets summary and dashboard paths recompute metrics from
    older artifacts without preserving false repeat-key partials as failures.
    """
    normalised: dict[str, Any] = {}
    for qid, verdict in field_results.items():
        if isinstance(verdict, dict):
            normalised[qid] = FieldVerdict.model_validate(verdict).model_dump()
        elif isinstance(verdict, FieldVerdict):
            normalised[qid] = verdict.model_dump()
        else:
            normalised[qid] = verdict
    matched_pairs = {
        (entry.get("group"), entry.get("agent_index")): entry.get("gt_index")
        for entry in alignment_log
        if entry.get("status") == "matched" and entry.get("agent_index") is not None
    }

    for qid, verdict in list(normalised.items()):
        if not isinstance(verdict, dict):
            continue
        if verdict.get("correctness") != "partially_correct":
            continue
        match = _REPEAT_FIELD_WITH_GT_RE.match(qid)
        if not match:
            continue
        idx_token = match.group("idx")
        if idx_token.startswith("?gt"):
            continue
        group_name = match.group("group")
        agent_idx = int(idx_token)
        field_name = match.group("field")
        gt_idx = matched_pairs.get((group_name, agent_idx))
        if gt_idx is None:
            continue
        group = gt_repeat_groups.get(group_name)
        if not isinstance(group, dict):
            continue
        if field_name not in set(group.get("alignment_keys") or []):
            continue
        if _is_strict_alignment_key(field_name):
            continue
        candidate_value = _candidate_value_for_repeat_qid(qid, candidate_state)
        if _is_empty(candidate_value):
            continue
        if verdict.get("source") == "fabricated":
            continue
        expected_value = _expected_value_for_matched_alignment_key(
            group_name=group_name,
            gt_idx=int(gt_idx),
            field_name=field_name,
            gt_repeat_groups=gt_repeat_groups,
        )
        if not _is_promotable_alignment_key_partial(
            qid=qid,
            verdict=verdict,
            candidate_value=candidate_value,
            expected_value=expected_value,
            questionnaire_path=questionnaire_path,
        ):
            continue

        previous_reason = str(verdict.get("reasoning") or "").strip()
        verdict.setdefault("original_correctness", verdict.get("correctness"))
        verdict.setdefault("original_source", verdict.get("support_source") or verdict.get("source"))
        verdict.setdefault("original_partial_reason", verdict.get("partial_reason"))
        verdict["correctness"] = "correct"
        verdict["partial_reason"] = None
        if verdict.get("source") in {None, "unsupported_inference"}:
            verdict["source"] = "likely_inferred"
            verdict["support_source"] = "likely_inferred"
        elif "support_source" not in verdict:
            verdict["support_source"] = verdict.get("source")
        verdict["decision_source"] = "repeat_alignment_postprocess"
        verdict["postprocess_reason"] = (
            "Promoted to correct because the repeat-instance matcher aligned "
            "this row semantically and this non-strict alignment key identifies "
            "the same instance."
        )
        verdict["reasoning"] = verdict["postprocess_reason"]
        verdict["final_correctness"] = "correct"
        if previous_reason:
            verdict["reasoning"] += f" Original judge note: {previous_reason}"

    return normalised
