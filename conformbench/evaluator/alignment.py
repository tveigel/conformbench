"""
Shared repeat-group alignment helpers.

Used by both the form-level judge (judge.py) and the turn-level evaluator
(run_turn.py) to align agent repeat-group instances to ground-truth
instances before field-level comparison.

Uses an LLM to semantically match GT instances to agent instances based on the
full visible turn context and all repeat-row fields, with alignment-key fields
highlighted as useful identity hints.  This replaces the earlier deterministic
greedy approach which could not reliably handle natural-language variation in
field values.
"""

from __future__ import annotations

import json
import hashlib
import re
from typing import Any

from loguru import logger

from .provenance import build_trace_entry

from .llm_retry import invoke_with_retries

# ── Agent instance extraction ────────────────────────────────────────────

DELETE_INSTANCE_SENTINEL = "__DELETE_INSTANCE__"
DELETE_INSTANCE_MARKER = "__delete_instance__"


def extract_agent_instances(
    answers: dict[str, Any], group_prefix: str,
) -> dict[int, dict[str, Any]]:
    """
    Parse flat agent answers into per-instance dicts.

    E.g. {"vehicles[0].role": "x", "vehicles[1].role": "y"}
      -> {0: {"role": "x"}, 1: {"role": "y"}}

    Whole-instance deletion directives are represented as:
      {"vehicles[0]": "__DELETE_INSTANCE__"}
      -> {0: {"__delete_instance__": True}}
    """
    field_pattern = re.compile(rf"^{re.escape(group_prefix)}\[(\d+)\]\.(.+)$")
    delete_pattern = re.compile(rf"^{re.escape(group_prefix)}\[(\d+)\]$")
    instances: dict[int, dict[str, Any]] = {}
    for key, value in answers.items():
        m_delete = delete_pattern.match(key)
        if m_delete and value == DELETE_INSTANCE_SENTINEL:
            idx = int(m_delete.group(1))
            instances.setdefault(idx, {})[DELETE_INSTANCE_MARKER] = True
            continue

        m = field_pattern.match(key)
        if m:
            idx = int(m.group(1))
            field = m.group(2)
            instances.setdefault(idx, {})[field] = value
    return instances


def is_delete_instance_directive(fields: dict[str, Any]) -> bool:
    """Return True when an extracted repeat row represents a delete request."""

    return fields.get(DELETE_INSTANCE_MARKER) is True


def without_delete_instance_marker(fields: dict[str, Any]) -> dict[str, Any]:
    """Return extracted row fields without evaluator-internal delete metadata."""

    return {
        key: value
        for key, value in fields.items()
        if key != DELETE_INSTANCE_MARKER
    }


# ── LLM-based instance matching ──────────────────────────────────────────


def _get_alignment_model(
    model_id: str | None = None,
    *,
    reasoning_effort: str | None = None,
    rotation_key: str | None = None,
    exclude_model: str | None = None,
):
    """Build a chat model for alignment (same config as the judge)."""
    from llm import get_judge_model_with_config

    if model_id:
        overrides: dict[str, Any] = {"model": model_id}
        if reasoning_effort:
            overrides["reasoning_effort"] = reasoning_effort
        return get_judge_model_with_config(**overrides)
    overrides = {
        "rotation_key": rotation_key,
        "exclude_model": exclude_model,
    }
    if reasoning_effort:
        overrides["reasoning_effort"] = reasoning_effort
    return get_judge_model_with_config(
        **overrides,
    )


def _response_text(response: Any) -> str:
    from llm import response_text

    return response_text(response)


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def _alignment_rotation_key(
    *,
    group_name: str,
    gt_instances: list[dict[str, Any]],
    agent_instances: dict[int, dict[str, Any]],
    alignment_keys: list[str],
    current_utterance: str,
    form_title: str | None,
) -> str:
    return _stable_json(
        {
            "phase": "instance_alignment",
            "form_title": form_title,
            "group_name": group_name,
            "alignment_keys": alignment_keys,
            "current_utterance_sha256": hashlib.sha256(
                current_utterance.encode("utf-8")
            ).hexdigest(),
            "gt_instances": gt_instances,
            "agent_instances": agent_instances,
        }
    )


def match_instances(
    agent_instances: dict[int, dict[str, Any]],
    gt_instances: list[dict[str, Any]],
    alignment_keys: list[str],
    group_name: str = "instances",
    *,
    form_title: str | None = None,
    model_id: str | None = None,
    reasoning_effort: str | None = None,
    exclude_model: str | None = None,
    trace_collector: list[dict[str, Any]] | None = None,
    current_utterance: str = "",
    visible_history: list[dict[str, Any]] | None = None,
    prior_state: dict[str, Any] | None = None,
    alignment_field_metadata: dict[str, dict[str, Any]] | None = None,
) -> list[tuple[dict[str, Any], int | None]]:
    """
    For each GT instance, find the best-matching agent instance using an LLM.

    The LLM receives the current utterance, visible history, prior rows for the
    group, full GT rows, full agent rows, and alignment-key hints.  This lets it
    handle aliases, row reordering, missing/extra rows, and natural-language
    variation robustly via semantic matching.

    Returns a list parallel to *gt_instances*:
      [(gt_instance, matched_agent_index | None), ...]
    """
    from .prompts import (
        ALIGNMENT_SYSTEM_PROMPT,
        build_alignment_prompt,
        build_alignment_system_prompt,
    )

    # Fast path: nothing to match
    if not gt_instances:
        return []
    if not agent_instances:
        return [(gt, None) for gt in gt_instances]

    prior_instances: list[dict[str, Any]] = []
    if isinstance(prior_state, dict):
        raw_prior_rows = prior_state.get(group_name)
        if isinstance(raw_prior_rows, list):
            prior_instances = [
                row for row in raw_prior_rows if isinstance(row, dict)
            ]

    user_prompt = build_alignment_prompt(
        gt_instances,
        agent_instances,
        alignment_keys,
        group_name,
        current_utterance=current_utterance,
        visible_history=visible_history or [],
        prior_instances=prior_instances,
        alignment_field_metadata=alignment_field_metadata,
    )

    rotation_key = _alignment_rotation_key(
        group_name=group_name,
        gt_instances=gt_instances,
        agent_instances=agent_instances,
        alignment_keys=alignment_keys,
        current_utterance=current_utterance,
        form_title=form_title,
    )
    model, model_config = _get_alignment_model(
        model_id,
        reasoning_effort=reasoning_effort,
        rotation_key=rotation_key,
        exclude_model=exclude_model,
    )
    alignment_prompt = build_alignment_system_prompt(form_title) if form_title else ALIGNMENT_SYSTEM_PROMPT
    messages = [
        {"role": "system", "content": alignment_prompt},
        {"role": "user", "content": user_prompt},
    ]

    logger.info(
        f"Alignment LLM call: {len(gt_instances)} GT × "
        f"{len(agent_instances)} agent instances for '{group_name}'"
    )

    response = invoke_with_retries(
        model,
        messages,
        description=f"instance alignment for {group_name}",
    )
    raw = _response_text(response)
    if trace_collector is not None:
        trace_collector.append(
            build_trace_entry(
                phase="instance_alignment",
                call_index=len(trace_collector) + 1,
                messages=messages,
                system_prompt=alignment_prompt,
                model_config=model_config,
                response=response,
                tool_calls=[],
                tool_results=[],
                extra={
                    "group_name": group_name,
                    "alignment_keys": list(alignment_keys),
                    "alignment_field_metadata": alignment_field_metadata or {},
                    "gt_instance_count": len(gt_instances),
                    "agent_instance_count": len(agent_instances),
                    "prior_instance_count": len(prior_instances),
                },
            )
        )

    # Strip markdown code fences if present
    if "```" in raw:
        raw = (
            raw.split("```json")[-1].split("```")[0]
            if "```json" in raw
            else raw.split("```")[1].split("```")[0]
        )

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        logger.error(
            f"Alignment LLM returned invalid JSON for '{group_name}':\n"
            f"{raw[:500]}"
        )
        # Cannot align — mark all GT instances as unmatched
        return [(gt, None) for gt in gt_instances]

    # Parse the LLM response into GT → agent index mapping.  The prompt asks
    # for object-shaped unmatched rows with reasons, but older cached prompts
    # and tests may still return bare integer arrays.  Accept both forms.
    matches_raw = parsed.get("matches", [])
    gt_to_agent: dict[int, int] = {}
    used_agent: set[int] = set()

    valid_agent_indices = set(agent_instances.keys())
    valid_gt_indices = {gt["ground_truth_index"] for gt in gt_instances}

    def _coerce_index(value: Any) -> int | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                return None
        return None

    def _unmatched_indices(raw: Any, key: str) -> set[int]:
        result: set[int] = set()
        if not isinstance(raw, list):
            return result
        for item in raw:
            if isinstance(item, dict):
                idx = _coerce_index(item.get(key))
            else:
                idx = _coerce_index(item)
            if idx is not None:
                result.add(idx)
        return result

    for m in matches_raw:
        if not isinstance(m, dict):
            logger.warning(f"Alignment: skipping malformed match entry {m!r}")
            continue
        gt_idx = _coerce_index(m.get("gt_index"))
        a_idx = _coerce_index(m.get("agent_index"))
        # Validate indices and enforce 1-to-1
        if (
            gt_idx in valid_gt_indices
            and a_idx in valid_agent_indices
            and gt_idx not in gt_to_agent
            and a_idx not in used_agent
        ):
            gt_to_agent[gt_idx] = a_idx
            used_agent.add(a_idx)
        else:
            logger.warning(
                f"Alignment: skipping invalid/duplicate match "
                f"gt_index={gt_idx}, agent_index={a_idx}"
            )

    declared_unmatched_gt = _unmatched_indices(parsed.get("unmatched_gt"), "gt_index")
    declared_unmatched_agent = _unmatched_indices(
        parsed.get("unmatched_agent"),
        "agent_index",
    )
    inferred_unmatched_gt = valid_gt_indices - set(gt_to_agent.keys())
    inferred_unmatched_agent = valid_agent_indices - used_agent
    if declared_unmatched_gt and declared_unmatched_gt != inferred_unmatched_gt:
        logger.warning(
            "Alignment unmatched_gt declaration disagrees with matches for '{}': "
            "declared={}, inferred={}",
            group_name,
            sorted(declared_unmatched_gt),
            sorted(inferred_unmatched_gt),
        )
    if declared_unmatched_agent and declared_unmatched_agent != inferred_unmatched_agent:
        logger.warning(
            "Alignment unmatched_agent declaration disagrees with matches for '{}': "
            "declared={}, inferred={}",
            group_name,
            sorted(declared_unmatched_agent),
            sorted(inferred_unmatched_agent),
        )

    # Build result list parallel to gt_instances
    result: list[tuple[dict[str, Any], int | None]] = []
    for gt_inst in gt_instances:
        gt_idx = gt_inst["ground_truth_index"]
        agent_idx = gt_to_agent.get(gt_idx)
        result.append((gt_inst, agent_idx))

    matched_count = sum(1 for _, a in result if a is not None)
    logger.info(
        f"Alignment result for '{group_name}': "
        f"{matched_count}/{len(gt_instances)} GT matched, "
        f"{len(agent_instances) - matched_count} agent unmatched"
    )

    return result


# ── Description helper ───────────────────────────────────────────────────


def describe_match(agent_fields: dict, alignment_keys: list[str]) -> str:
    """Summarise which alignment keys matched for logging / diagnostics."""
    parts = []
    for k in alignment_keys:
        v = agent_fields.get(k, "")
        if v:
            parts.append(f"{k}={str(v)[:40]}")
    return "; ".join(parts) if parts else "best-effort"


# ── Alignment-map builder (for summarize.py) ─────────────────────────────


def build_alignment_map(
    answers: dict[str, Any],
    gt_repeat_groups: dict[str, Any],
) -> dict[str, dict[int, dict[str, Any]]]:
    """Build a mapping from (group, agent_idx) → matched GT instance fields.

    Returns ``{group_name: {agent_idx: gt_instance_fields_dict, ...}, ...}``
    for every repeat group in *gt_repeat_groups*.

    This is used by summarize.py to look up the correct GT expected value
    for a given ``vehicles[N].field`` qid.
    """
    mapping: dict[str, dict[int, dict[str, Any]]] = {}

    for group_name, group_gt in gt_repeat_groups.items():
        alignment_keys = group_gt.get("alignment_keys", [])
        gt_instances: list[dict] = group_gt.get("instances", [])

        if group_name in answers and isinstance(answers.get(group_name), list):
            # Table-style: agent stores a flat list (injury_table, vehicles, …)
            agent_rows = answers[group_name]
            agent_as_indexed = {i: row for i, row in enumerate(agent_rows)}
        else:
            # Indexed: vehicles[0].field, vehicles[1].field, …
            agent_as_indexed = {
                idx: without_delete_instance_marker(fields)
                for idx, fields in extract_agent_instances(answers, group_name).items()
                if not is_delete_instance_directive(fields)
            }

        matches = match_instances(
            agent_as_indexed, gt_instances, alignment_keys,
            group_name=group_name,
        )

        group_map: dict[int, dict[str, Any]] = {}
        for gt_inst, agent_idx in matches:
            if agent_idx is not None:
                group_map[agent_idx] = gt_inst.get("fields", {})
        mapping[group_name] = group_map

    return mapping
