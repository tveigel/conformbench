"""Deterministic grouped-error accounting for evaluator reports.

The field-level evaluator remains strict: every non-correct field verdict is
kept as-is.  This module only adds a diagnostic view that can collapse declared
cascade-prone failures into root-cause groups for reporting.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any


NON_ERROR_CORRECTNESS = {"", "correct"}


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple | set):
        return list(value)
    return [value]


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    except TypeError:
        return str(value)


def _normalise_text(value: Any) -> str:
    return " ".join(_stringify(value).strip().lower().split())


def _is_emptyish(value: Any) -> bool:
    return value is None or value == "" or value == [] or value == {}


def _iter_leaf_values(value: Any):
    if isinstance(value, Mapping):
        for child in value.values():
            yield from _iter_leaf_values(child)
        return
    if isinstance(value, list | tuple | set):
        for child in value:
            yield from _iter_leaf_values(child)
        return
    yield value


def _field_pattern_matches(qid: str, pattern: str) -> bool:
    """Match simple field-path patterns where ``*`` means any substring.

    We intentionally avoid fnmatch because field paths contain square brackets,
    which fnmatch treats as character classes rather than literal index syntax.
    """
    pattern_text = str(pattern or "").strip()
    if not pattern_text:
        return False
    if "*" not in pattern_text:
        return qid == pattern_text
    regex = "^" + re.escape(pattern_text).replace(r"\*", ".*") + "$"
    return re.match(regex, qid) is not None


def _matches_any_field_selector(
    qid: str,
    *,
    field_paths: Sequence[Any] | None = None,
    field_patterns: Sequence[Any] | None = None,
) -> bool:
    exact = {str(path) for path in _as_list(field_paths) if str(path or "").strip()}
    if qid in exact:
        return True
    return any(_field_pattern_matches(qid, str(pattern)) for pattern in _as_list(field_patterns))


def _value_matches_alias(value: Any, aliases: Sequence[Any]) -> bool:
    if not aliases:
        return False

    for leaf in _iter_leaf_values(value):
        for alias in aliases:
            if isinstance(alias, bool):
                if leaf is alias:
                    return True
                continue
            if isinstance(alias, int | float) and not isinstance(alias, bool):
                if leaf == alias:
                    return True
                if isinstance(leaf, str) and _normalise_text(leaf) == _normalise_text(alias):
                    return True
                continue

            alias_text = _normalise_text(alias)
            leaf_text = _normalise_text(leaf)
            if not alias_text or not leaf_text:
                continue
            if leaf_text == alias_text:
                return True
            boundary_pattern = (
                r"(?<![A-Za-z0-9])"
                + re.escape(str(alias).strip())
                + r"(?![A-Za-z0-9])"
            )
            if re.search(boundary_pattern, _stringify(leaf), flags=re.IGNORECASE):
                return True
    return False


def declared_error_groups(gt: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return normalized declarative error-group specs from ground truth."""
    raw = (
        gt.get("evaluation_error_groups")
        or gt.get("error_groups")
        or gt.get("diagnostic_error_groups")
        or []
    )
    if isinstance(raw, Mapping):
        raw = raw.get("groups") or []
    groups: list[dict[str, Any]] = []
    for idx, item in enumerate(_as_list(raw)):
        if not isinstance(item, Mapping):
            continue
        spec = dict(item)
        group_id = str(spec.get("id") or f"error_group_{idx + 1}").strip()
        if not group_id:
            continue
        spec["id"] = group_id
        groups.append(spec)
    return groups


def _candidate_view(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "qid": record.get("qid"),
        "correctness": record.get("correctness"),
        "candidate_value": record.get("candidate_value"),
        "gold_value": record.get("gold_value"),
    }


def _member_records(
    records: Sequence[Mapping[str, Any]],
    spec: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    field_paths = (
        spec.get("field_paths")
        or spec.get("fields")
        or spec.get("member_field_paths")
        or []
    )
    field_patterns = (
        spec.get("field_patterns")
        or spec.get("member_field_patterns")
        or spec.get("field_path_patterns")
        or []
    )
    return [
        record
        for record in records
        if _matches_any_field_selector(
            str(record.get("qid") or ""),
            field_paths=field_paths,
            field_patterns=field_patterns,
        )
    ]


def _trigger_records(
    records: Sequence[Mapping[str, Any]],
    spec: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    trigger_paths = spec.get("trigger_field_paths") or spec.get("trigger_fields") or []
    trigger_patterns = (
        spec.get("trigger_field_patterns")
        or spec.get("trigger_patterns")
        or []
    )
    if not trigger_paths and not trigger_patterns:
        return _member_records(records, spec)
    return [
        record
        for record in records
        if _matches_any_field_selector(
            str(record.get("qid") or ""),
            field_paths=trigger_paths,
            field_patterns=trigger_patterns,
        )
    ]


def _non_empty_trigger_records(
    records: Sequence[Mapping[str, Any]],
    spec: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    trigger_paths = spec.get("trigger_on_non_empty_field_paths") or []
    trigger_patterns = spec.get("trigger_on_non_empty_field_patterns") or []
    if not trigger_paths and not trigger_patterns:
        return []
    return [
        record
        for record in records
        if _matches_any_field_selector(
            str(record.get("qid") or ""),
            field_paths=trigger_paths,
            field_patterns=trigger_patterns,
        )
    ]


def build_error_group_view(
    *,
    gt: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build grouped/root-cause accounting from strict field verdict records.

    Counts are diagnostic only.  Existing field-level correctness and PR/F1
    metrics should continue to use the original verdicts.
    """
    specs = declared_error_groups(gt)
    failed_records = [
        record
        for record in records
        if str(record.get("correctness") or "") not in NON_ERROR_CORRECTNESS
    ]
    failed_qids = {str(record.get("qid") or "") for record in failed_records}

    active_grouped_qids: set[str] = set()
    groups: list[dict[str, Any]] = []

    for spec in specs:
        members = _member_records(failed_records, spec)
        aliases = _as_list(spec.get("trigger_aliases") or spec.get("invalid_aliases") or [])
        triggered_by: list[dict[str, Any]] = []

        for record in _trigger_records(failed_records, spec):
            if aliases and _value_matches_alias(record.get("candidate_value"), aliases):
                triggered_by.append({
                    **_candidate_view(record),
                    "trigger": "invalid_alias",
                })

        for record in _non_empty_trigger_records(failed_records, spec):
            if not _is_emptyish(record.get("candidate_value")):
                triggered_by.append({
                    **_candidate_view(record),
                    "trigger": "non_empty_declared_trigger",
                })

        activate_on_any = bool(spec.get("activate_on_any_member_error"))
        if not aliases and not (
            spec.get("trigger_field_paths")
            or spec.get("trigger_fields")
            or spec.get("trigger_field_patterns")
            or spec.get("trigger_patterns")
            or spec.get("trigger_on_non_empty_field_paths")
            or spec.get("trigger_on_non_empty_field_patterns")
        ):
            activate_on_any = True

        active = bool(members) and (bool(triggered_by) or activate_on_any)
        member_qids = {str(record.get("qid") or "") for record in members}
        if active:
            active_grouped_qids.update(member_qids)

        groups.append({
            "id": spec["id"],
            "label": spec.get("label") or spec.get("description") or spec["id"],
            "type": spec.get("type") or "declared_error_group",
            "active": active,
            "counting_policy": spec.get("counting_policy") or "one_root_error_plus_field_errors",
            "member_field_error_count": len(members),
            "incorrect_member_count": sum(1 for record in members if record.get("correctness") == "incorrect"),
            "partial_member_count": sum(1 for record in members if record.get("correctness") == "partially_correct"),
            "member_fields": [_candidate_view(record) for record in members],
            "triggered_by": triggered_by,
            "canonical_values": _as_list(spec.get("canonical_values") or spec.get("canonical_value") or []),
            "invalid_aliases": aliases,
        })

    grouped_failed_qids = failed_qids & active_grouped_qids
    ungrouped_failed_qids = failed_qids - grouped_failed_qids
    active_groups = [group for group in groups if group.get("active")]

    return {
        "declared_group_count": len(specs),
        "active_group_count": len(active_groups),
        "strict_field_error_count": len(failed_qids),
        "grouped_field_error_count": len(grouped_failed_qids),
        "ungrouped_field_error_count": len(ungrouped_failed_qids),
        "grouped_root_error_count": len(active_groups) + len(ungrouped_failed_qids),
        "groups": groups,
    }


def aggregate_error_group_views(
    views: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate per-turn ``error_groups`` metric views."""
    active_by_id: dict[str, dict[str, Any]] = {}
    for view in views:
        for group in view.get("groups") or []:
            group_id = str(group.get("id") or "")
            if not group_id:
                continue
            entry = active_by_id.setdefault(
                group_id,
                {
                    "label": group.get("label") or group_id,
                    "type": group.get("type") or "",
                    "active_turn_count": 0,
                    "member_field_error_count": 0,
                },
            )
            if group.get("active"):
                entry["active_turn_count"] += 1
                entry["member_field_error_count"] += int(
                    group.get("member_field_error_count") or 0
                )

    return {
        "declared_group_count": sum(int(view.get("declared_group_count", 0) or 0) for view in views),
        "active_group_count": sum(int(view.get("active_group_count", 0) or 0) for view in views),
        "strict_field_error_count": sum(int(view.get("strict_field_error_count", 0) or 0) for view in views),
        "grouped_field_error_count": sum(int(view.get("grouped_field_error_count", 0) or 0) for view in views),
        "ungrouped_field_error_count": sum(int(view.get("ungrouped_field_error_count", 0) or 0) for view in views),
        "grouped_root_error_count": sum(int(view.get("grouped_root_error_count", 0) or 0) for view in views),
        "active_groups_by_id": {
            group_id: entry
            for group_id, entry in sorted(active_by_id.items())
            if entry["active_turn_count"]
        },
    }
