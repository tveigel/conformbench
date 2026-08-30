"""Controlled two-stage sparse-diff then single-call verification baseline."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from time import perf_counter
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from conformbench.benchmark import validate_resulting_state
from conformbench.llm_accounting import message_accounting_snapshot
from llm import get_chat_model_with_config, response_text
from . import flatagent
from .prompt_context import (
    DIFF_THEN_VERIFY_FINAL_INSTRUCTION,
    DIFF_THEN_VERIFY_SYSTEM_PROMPT,
    build_turn_context,
    llm_trace_enabled,
    trace_messages,
    trace_response,
)


def solve(turn: Any) -> dict[str, Any]:
    """Run the existing Update Tool stage, then one fresh verification call."""

    return asyncio.run(_solve_async(turn))


async def _solve_async(turn: Any) -> dict[str, Any]:
    total_started = perf_counter()

    diff_started = perf_counter()
    diff_result = await flatagent._solve_async(turn)
    diff_seconds = perf_counter() - diff_started
    candidate_state = deepcopy(diff_result["resulting_state"])

    verify_started = perf_counter()
    verify_result = await _verify_candidate_async(turn, candidate_state)
    verify_seconds = perf_counter() - verify_started
    total_seconds = perf_counter() - total_started

    final_state = verify_result["resulting_state"]
    validation_errors = validate_resulting_state(final_state, turn.schema)
    if validation_errors:
        raise ValueError(
            "DiffThenVerify produced invalid resulting state: "
            + "; ".join(validation_errors[:12])
        )

    diff_response = deepcopy(diff_result.get("agent_response") or {})
    verifier_response = deepcopy(verify_result.get("agent_response") or {})

    diff_tool_updates = [
        {**deepcopy(entry), "stage": "diff"}
        for entry in diff_response.get("tool_updates") or []
        if isinstance(entry, dict)
    ]
    verify_tool_updates = [
        {**deepcopy(entry), "stage": "verify"}
        for entry in verifier_response.get("tool_updates") or []
        if isinstance(entry, dict)
    ]
    model_calls = [
        *deepcopy(diff_response.get("model_calls") or []),
        *deepcopy(verifier_response.get("model_calls") or []),
    ]

    diff_model = (
        ((diff_result.get("provenance") or {}).get("generation") or {}).get("model")
        or {}
    )
    verify_model = (
        ((verify_result.get("provenance") or {}).get("generation") or {}).get("model")
        or {}
    )
    same_generator = _same_generator_config(diff_model, verify_model)
    if not same_generator:
        raise ValueError(
            "Diff and verification stages resolved to different generators: "
            f"diff={diff_model!r}, verify={verify_model!r}"
        )

    return {
        "resulting_state": final_state,
        "agent_response": {
            "raw_response": verifier_response.get("raw_response", ""),
            "tool_updates": [*diff_tool_updates, *verify_tool_updates],
            "model_calls": model_calls,
            "timing": {
                "diff_stage_seconds": round(diff_seconds, 6),
                "verification_stage_seconds": round(verify_seconds, 6),
                "total_generation_seconds": round(total_seconds, 6),
                "clock": "monotonic_wall",
                "excludes_evaluator": True,
            },
            "stages": {
                "diff": {
                    "candidate_state": candidate_state,
                    "raw_response": diff_response.get("raw_response", ""),
                    "tool_updates": diff_tool_updates,
                    "model_call_count": len(diff_response.get("model_calls") or []),
                },
                "verify": {
                    "raw_response": verifier_response.get("raw_response", ""),
                    "tool_updates": verify_tool_updates,
                    "model_call_count": len(verifier_response.get("model_calls") or []),
                    "emitted_tool_call_count": verifier_response.get(
                        "emitted_tool_call_count", 0
                    ),
                    "expected_tool_call_count": verifier_response.get(
                        "expected_tool_call_count", 0
                    ),
                    "applied_corrective_patch_count": verifier_response.get(
                        "applied_corrective_patch_count", 0
                    ),
                    "multiple_patch_calls_merged": verifier_response.get(
                        "multiple_patch_calls_merged", False
                    ),
                },
            },
            **(
                {
                    "trace": {
                        "diff": diff_response.get("trace") or [],
                        "verify": verifier_response.get("trace") or [],
                    }
                }
                if llm_trace_enabled()
                else {}
            ),
        },
        "provenance": {
            "generation": {
                "agent": "DiffThenVerify",
                "architecture": "sparse_diff_then_single_call_verification",
                "same_generator_both_stages": same_generator,
                "model": verify_model,
                "stages": {
                    "diff": {
                        "agent": "FlatAgent",
                        "model": diff_model,
                        "source": "conformbench.systems.flatagent._solve_async",
                    },
                    "verify": {
                        "agent": "Verify",
                        "model": verify_model,
                        "fresh_context": True,
                        "llm_call_limit": 1,
                        "patch_target": "materialized_diff_candidate",
                    },
                },
                "tool_policy": {
                    "mode": "diff_then_single_call_corrective_patch",
                    "allowed_tools": ["update_questionnaire_answers"],
                    "verification_llm_call_limit": 1,
                },
                "schema_interface": {
                    "repeat_group_representation": "indexed_child_qids",
                    "public_projection": (
                        "shared_schema_field_guide_with_original_prior_and_"
                        "materialized_candidate"
                    ),
                },
            }
        },
    }


async def _verify_candidate_async(
    turn: Any,
    candidate_state: dict[str, Any],
) -> dict[str, Any]:
    schema = turn.schema
    shape = flatagent._schema_shape(schema)
    original_prior = flatagent._materialize_repeat_counts(
        flatagent._complete_state(turn.prior_state, schema),
        shape,
    )
    state = flatagent._materialize_repeat_counts(
        flatagent._complete_state(candidate_state, schema),
        shape,
    )
    tool_updates: list[dict[str, Any]] = []

    @tool("update_questionnaire_answers")
    def update_questionnaire_answers(updates: dict[str, Any]) -> dict[str, Any]:
        """Apply one sparse corrective patch to the candidate post-turn record."""

        before_counts = flatagent._repeat_row_counts(state, shape)
        result = flatagent.apply_answer_updates(
            state=state,
            updates=updates,
            schema=schema,
            shape=shape,
        )
        materialized = flatagent._materialize_repeat_counts(
            flatagent._complete_state(state, schema),
            shape,
        )
        state.clear()
        state.update(materialized)
        after_counts = flatagent._repeat_row_counts(state, shape)
        count_changes = {
            group_id: count
            for group_id, count in after_counts.items()
            if before_counts.get(group_id) != count
        }
        if count_changes:
            result["materialized_repeat_groups"] = count_changes
        tool_updates.append({"updates": deepcopy(updates), "result": deepcopy(result)})
        return result

    metadata = getattr(turn, "metadata", {}) or {}
    overrides: dict[str, Any] = {}
    if metadata.get("model_id"):
        overrides["model"] = metadata["model_id"]
    reasoning_effort = metadata.get("model_reasoning_effort") or metadata.get(
        "reasoning_effort"
    )
    if reasoning_effort:
        overrides["reasoning_effort"] = reasoning_effort

    update_tool = update_questionnaire_answers
    model, model_config = get_chat_model_with_config(tools=[update_tool], **overrides)
    messages: list[Any] = [
        SystemMessage(content=DIFF_THEN_VERIFY_SYSTEM_PROMPT),
        HumanMessage(
            content=build_turn_context(
                schema=schema,
                shape=shape,
                prior_state=original_prior,
                prior_state_label="ORIGINAL PRIOR RECORD STATE",
                candidate_state=state,
                visible_history=turn.visible_history,
                current_utterance=turn.current_utterance,
                final_instruction=DIFF_THEN_VERIFY_FINAL_INSTRUCTION,
            )
        ),
    ]

    trace: list[dict[str, Any]] | None = [] if llm_trace_enabled() else None
    trace_call: dict[str, Any] | None = None
    if trace is not None:
        trace_call = {"call_index": 1, "input_messages": trace_messages(messages)}

    response = await model.ainvoke(messages)
    if trace_call is not None:
        trace_call["response"] = trace_response(response)
        trace.append(trace_call)

    model_calls = [
        message_accounting_snapshot(
            response,
            phase="diff_then_verify_verification",
            call_index=1,
            model_config=model_config,
            extra={"candidate_source": "materialized_diff_stage"},
        )
    ]
    emitted_tool_calls = list(getattr(response, "tool_calls", None) or [])

    corrective_updates: dict[str, Any] = {}
    expected_tool_call_count = 0
    for call_index, tool_call in enumerate(emitted_tool_calls):
        tool_name = tool_call.get("name")
        if tool_name != update_tool.name:
            result = {"status": "error", "error": f"Unknown tool: {tool_name}"}
            tool_updates.append(
                {
                    "tool_name": tool_name,
                    "updates": {},
                    "result": deepcopy(result),
                    "emitted_call_index": call_index,
                }
            )
            continue
        expected_tool_call_count += 1
        corrective_updates.update(
            flatagent._tool_updates(tool_call.get("args") or {})
        )

    if expected_tool_call_count:
        normalized_updates, normalization_notes = _normalize_corrective_updates(
            updates=corrective_updates,
            state=state,
            shape=shape,
        )
        update_tool.invoke({"updates": normalized_updates})
        if tool_updates:
            tool_updates[-1]["emitted_updates"] = deepcopy(corrective_updates)
            tool_updates[-1]["normalization_notes"] = normalization_notes

    resulting_state = flatagent._materialize_repeat_counts(
        flatagent._complete_state(state, schema),
        shape,
    )
    validation_errors = validate_resulting_state(resulting_state, schema)
    if validation_errors:
        raise ValueError(
            "Verification stage produced invalid resulting state: "
            + "; ".join(validation_errors[:12])
        )

    return {
        "resulting_state": resulting_state,
        "agent_response": {
            "raw_response": response_text(response),
            "tool_updates": tool_updates,
            "model_calls": model_calls,
            "emitted_tool_call_count": len(emitted_tool_calls),
            "expected_tool_call_count": expected_tool_call_count,
            "applied_corrective_patch_count": 1 if expected_tool_call_count else 0,
            "multiple_patch_calls_merged": expected_tool_call_count > 1,
            **({"trace": trace} if trace is not None else {}),
        },
        "provenance": {
            "generation": {
                "agent": "Verify",
                "model": model_config,
                "tool_policy": {
                    "mode": "single_call_corrective_patch",
                    "allowed_tools": [update_tool.name],
                    "llm_call_limit": 1,
                },
            }
        },
    }


def _same_generator_config(left: dict[str, Any], right: dict[str, Any]) -> bool:
    keys = ("provider", "model", "requested_model", "reasoning_effort")
    return all((left or {}).get(key) == (right or {}).get(key) for key in keys)


def _normalize_corrective_updates(
    *,
    updates: dict[str, Any],
    state: dict[str, Any],
    shape: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Normalize equivalent repeat patches without inventing commitments.

    The verifier still emits the public sparse-patch interface. This adapter
    makes intuitive row-object patches explicit and treats deletion of an
    already-absent materialized row as an idempotent no-op. Every normalization
    is retained in the trace for auditability.
    """

    normalized: dict[str, Any] = {}
    notes: list[dict[str, Any]] = []
    repeat_groups = shape.get("repeat_groups") or {}

    def row_exists(group_id: str, row_index: int) -> bool:
        rows = state.get(group_id)
        return isinstance(rows, list) and 0 <= row_index < len(rows)

    def add_row_patch(group_id: str, row_index: int, row_patch: dict[str, Any]) -> None:
        if flatagent._DELETE_INSTANCE in row_patch.values():
            delete_path = f"{group_id}[{row_index}]"
            if row_exists(group_id, row_index):
                normalized[delete_path] = flatagent._DELETE_INSTANCE
            else:
                notes.append(
                    {
                        "action": "ignored_idempotent_delete",
                        "path": delete_path,
                        "reason": "row already absent from materialized candidate",
                    }
                )
            return
        for field_id, value in row_patch.items():
            normalized[f"{group_id}[{row_index}].{field_id}"] = value

    for path, value in updates.items():
        if path in repeat_groups:
            if value == flatagent._DELETE_INSTANCE:
                rows = state.get(path)
                if isinstance(rows, list):
                    for row_index in reversed(range(len(rows))):
                        normalized[f"{path}[{row_index}]"] = flatagent._DELETE_INSTANCE
                notes.append(
                    {
                        "action": "expanded_repeat_group_delete",
                        "path": path,
                        "row_count": len(rows) if isinstance(rows, list) else 0,
                    }
                )
                continue
            if isinstance(value, list) and all(isinstance(row, dict) for row in value):
                for row_index, row_patch in enumerate(value):
                    add_row_patch(path, row_index, row_patch)
                notes.append(
                    {
                        "action": "expanded_repeat_group_row_patches",
                        "path": path,
                        "row_patch_count": len(value),
                    }
                )
                continue

        row_match = flatagent._REPEAT_ROW_RE.match(path)
        if row_match:
            group_id = row_match.group("group")
            row_index = int(row_match.group("index"))
            if value == flatagent._DELETE_INSTANCE:
                if row_exists(group_id, row_index):
                    normalized[path] = value
                else:
                    notes.append(
                        {
                            "action": "ignored_idempotent_delete",
                            "path": path,
                            "reason": "row already absent from materialized candidate",
                        }
                    )
                continue
            if isinstance(value, dict) and group_id in repeat_groups:
                add_row_patch(group_id, row_index, value)
                notes.append(
                    {
                        "action": "expanded_repeat_row_object",
                        "path": path,
                        "field_count": len(value),
                    }
                )
                continue

        normalized[path] = value

    valid_updates: dict[str, Any] = {}
    bare_fields = shape.get("bare_fields") or set()
    for path, value in normalized.items():
        valid = path in bare_fields or path in repeat_groups
        if not valid:
            child_match = flatagent._REPEAT_CHILD_RE.match(path)
            if child_match:
                group_id = child_match.group("group")
                field_id = child_match.group("field")
                valid = (
                    group_id in repeat_groups
                    and field_id in repeat_groups[group_id]
                )
        if not valid:
            row_match = flatagent._REPEAT_ROW_RE.match(path)
            valid = bool(
                row_match
                and row_match.group("group") in repeat_groups
                and value == flatagent._DELETE_INSTANCE
            )
        if valid:
            valid_updates[path] = value
        else:
            notes.append(
                {
                    "action": "ignored_unknown_update_path",
                    "path": path,
                    "reason": "path is not present in the public schema",
                }
            )

    return valid_updates, notes
