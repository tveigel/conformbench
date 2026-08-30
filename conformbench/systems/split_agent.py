"""SplitAgent: partition the form once, run parallel sub-agents per partition.

The form is split into independent field groups by an LLM once per
questionnaire.  Every turn broadcasts the public input to one sub-agent per
partition; each sub-agent only sees and updates its assigned fields.  Results
are merged into the final resulting state.

Run:
    python -m conformbench run --items public --solver conformbench.systems.split_agent:solve
"""

from __future__ import annotations

import json
import random
import time
from copy import deepcopy
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from conformbench.benchmark import validate_resulting_state
from conformbench.items import DATA_ROOT
from conformbench.llm_accounting import message_accounting_snapshot
from llm import get_chat_model_with_config

from .split_agent_utils.splitter import load_or_create_split
from .flatagent import (
    _schema_shape,
    _complete_state,
    _materialize_repeat_counts,
    _REPEAT_CHILD_RE,
    _REPEAT_ROW_RE,
    _tool_result_needs_retry,
)
from .prompt_context import (
    SPLIT_PARTITION_FINAL_INSTRUCTION,
    SPLIT_PARTITION_SYSTEM_PROMPT as SUB_AGENT_SYSTEM_PROMPT,
    build_turn_context,
    llm_trace_enabled,
    trace_messages,
    trace_response,
)

_MAX_TOOL_ITERATIONS = 8
_SPLITS_DIR = DATA_ROOT / "splits"
_TRANSPORT_MAX_ATTEMPTS = 5
_TRANSIENT_ERROR_NAMES = {
    "APIConnectionError",
    "APIStatusError",
    "APITimeoutError",
    "InternalServerError",
    "RateLimitError",
    "ServiceUnavailableError",
}
_TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}


def solve(turn: Any) -> dict[str, Any]:
    """Run SplitAgent on one benchmark turn and return the resulting state."""
    return _solve(turn)


def _solve(turn: Any) -> dict[str, Any]:
    schema = turn.schema
    questionnaire_id = turn.questionnaire_id

    metadata = getattr(turn, "metadata", {}) or {}
    overrides: dict[str, Any] = {}
    if metadata.get("model_id"):
        overrides["model"] = metadata["model_id"]
    reasoning_effort = metadata.get("model_reasoning_effort") or metadata.get("reasoning_effort")
    if reasoning_effort:
        overrides["reasoning_effort"] = reasoning_effort

    split = load_or_create_split(
        schema,
        questionnaire_id,
        _SPLITS_DIR,
        model_id=overrides.get("model"),
    )
    partitions = split["partitions"]

    shape = _schema_shape(schema)
    full_state = _materialize_repeat_counts(_complete_state(turn.prior_state, schema), shape)

    partition_results = [
        _run_partition(
            partition=partition,
            schema=schema,
            shape=shape,
            full_state=full_state,
            visible_history=turn.visible_history,
            current_utterance=turn.current_utterance,
            overrides=overrides,
        )
        for partition in partitions
    ]

    merged_state = deepcopy(full_state)
    all_tool_updates: list[dict[str, Any]] = []
    all_model_calls: list[dict[str, Any]] = []
    all_traces: list[dict[str, Any]] = []
    partition_provenance: list[dict[str, Any]] = []

    for partition, result in zip(partitions, partition_results):
        p_state = result["state"]
        for key in partition["fields"]:
            if key in p_state:
                merged_state[key] = p_state[key]
        all_tool_updates.extend(result["tool_updates"])
        all_model_calls.extend(result["model_calls"])
        if isinstance(result.get("trace"), list):
            all_traces.append({
                "partition": partition["name"],
                "trace": result["trace"],
            })
        partition_provenance.append({
            "name": partition["name"],
            "fields": partition["fields"],
            "tool_update_count": len(result["tool_updates"]),
        })

    candidate_state = _materialize_repeat_counts(_complete_state(merged_state, schema), shape)
    validation_errors = validate_resulting_state(candidate_state, schema)
    if validation_errors:
        raise ValueError(
            "SplitAgent produced invalid resulting state: "
            + "; ".join(validation_errors[:12])
        )

    return {
        "resulting_state": candidate_state,
        "agent_response": {
            "raw_response": f"SplitAgent: {len(partitions)} partitions",
            "tool_updates": all_tool_updates,
            "model_calls": all_model_calls,
            **({"partition_traces": all_traces} if all_traces else {}),
        },
        "provenance": {
            "generation": {
                "agent": "SplitAgent",
                "model": overrides.get("model", "default"),
                "tool_policy": {
                    "mode": "partitioned_update_tool",
                    "partitions": len(partitions),
                    "max_tool_iterations": _MAX_TOOL_ITERATIONS,
                    "whole_repeat_group_updates": "rejected_in_partitions",
                },
                "schema_interface": {
                    "public_projection": (
                        "shared_schema_field_guide_with_schema_completed_prior_state"
                    ),
                    "partition_context": (
                        "assigned_fields_only_prior_state_with_global_key_awareness"
                    ),
                },
                "split": {
                    "questionnaire_id": questionnaire_id,
                    "partitions": partition_provenance,
                },
            },
        },
    }


def _run_partition(
    *,
    partition: dict[str, Any],
    schema: dict[str, Any],
    shape: dict[str, Any],
    full_state: dict[str, Any],
    visible_history: list[dict[str, Any]],
    current_utterance: Any,
    overrides: dict[str, Any],
) -> dict[str, Any]:
    """Run a single partition sub-agent and return its updated state."""
    allowed_keys = set(partition["fields"])
    partition_state = {
        k: deepcopy(full_state[k]) for k in allowed_keys if k in full_state
    }
    tool_updates: list[dict[str, Any]] = []

    @tool("update_questionnaire_answers")
    def update_questionnaire_answers(updates: dict[str, Any]) -> dict[str, Any]:
        """Update questionnaire answers for this partition's fields only."""
        filtered, rejected = _filter_updates(updates, allowed_keys, shape)
        result: dict[str, Any] = {}
        if rejected:
            result["rejected"] = rejected
        if filtered:
            from .flatagent import apply_answer_updates
            result.update(
                apply_answer_updates(
                    state=partition_state,
                    updates=filtered,
                    schema=schema,
                    shape=shape,
                )
            )
        else:
            result["status"] = "ok"
            result["applied_update_count"] = 0
        tool_updates.append({
            "partition": partition["name"],
            "updates": deepcopy(filtered),
            "rejected": deepcopy(rejected),
            "result": deepcopy(result),
        })
        return result

    update_tool = update_questionnaire_answers
    model, model_config = get_chat_model_with_config(tools=[update_tool], **overrides)

    partition_schema = _schema_for_partition(schema, allowed_keys)
    partition_shape = _schema_shape(partition_schema)
    partition_prior = {
        k: deepcopy(partition_state[k]) for k in allowed_keys if k in partition_state
    }
    messages: list[Any] = [
        SystemMessage(content=SUB_AGENT_SYSTEM_PROMPT),
        HumanMessage(
            content=_build_partition_message(
                schema=partition_schema,
                shape=partition_shape,
                full_shape=shape,
                partition=partition,
                partition_prior=partition_prior,
                visible_history=visible_history,
                current_utterance=current_utterance,
            )
        ),
    ]

    model_calls: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] | None = [] if llm_trace_enabled() else None
    for iteration in range(_MAX_TOOL_ITERATIONS):
        trace_call: dict[str, Any] | None = None
        if trace is not None:
            trace_call = {
                "call_index": len(model_calls) + 1,
                "loop_iteration": iteration + 1,
                "input_messages": trace_messages(messages),
            }
        response = _invoke_with_transport_retries(model, messages)
        if trace_call is not None:
            trace_call["response"] = trace_response(response)
            trace.append(trace_call)
        model_calls.append(
            message_accounting_snapshot(
                response,
                phase="split_partition_generation",
                call_index=len(model_calls) + 1,
                loop_iteration=iteration + 1,
                model_config=model_config,
                extra={"partition": partition["name"]},
            )
        )
        messages.append(response)

        tool_calls = list(getattr(response, "tool_calls", None) or [])
        if not tool_calls:
            break

        tool_results: list[dict[str, Any]] = []
        for call_index, tool_call in enumerate(tool_calls):
            tool_name = tool_call.get("name")
            tool_id = tool_call.get("id") or f"split-{partition['name']}-{iteration}-{call_index}"
            if tool_name != update_tool.name:
                result = {"status": "error", "error": f"Unknown tool: {tool_name}"}
                tool_updates.append({
                    "partition": partition["name"],
                    "tool_name": tool_name,
                    "updates": {},
                    "rejected": [],
                    "result": deepcopy(result),
                })
            else:
                args = tool_call.get("args") or {}
                updates = args.get("updates")
                if isinstance(updates, dict):
                    result = update_tool.invoke({"updates": updates})
                elif isinstance(args, dict):
                    result = update_tool.invoke({"updates": args})
                else:
                    result = {"status": "error", "error": "updates must be an object"}
                    tool_updates.append({
                        "partition": partition["name"],
                        "updates": {},
                        "rejected": [],
                        "result": deepcopy(result),
                    })
            tool_results.append(result if isinstance(result, dict) else {"status": "error"})
            messages.append(
                ToolMessage(
                    content=json.dumps(result, sort_keys=True, default=str),
                    tool_call_id=tool_id,
                    name=tool_name or update_tool.name,
                )
            )
        if any(_tool_result_needs_retry(result) for result in tool_results):
            messages.append(
                HumanMessage(
                    content=_partition_tool_retry_message(tool_results, allowed_keys)
                )
            )

    return {
        "state": partition_state,
        "tool_updates": tool_updates,
        "model_calls": model_calls,
        **({"trace": trace} if trace is not None else {}),
    }


def _invoke_with_transport_retries(model: Any, messages: list[Any]) -> Any:
    delay_s = 2.0
    for attempt in range(1, _TRANSPORT_MAX_ATTEMPTS + 1):
        try:
            return model.invoke(messages)
        except Exception as exc:
            if not _is_transient_transport_error(exc) or attempt == _TRANSPORT_MAX_ATTEMPTS:
                raise
            jitter_s = random.uniform(0.0, delay_s * 0.25)
            time.sleep(delay_s + jitter_s)
            delay_s *= 2.0
    raise RuntimeError("unreachable")


def _is_transient_transport_error(exc: Exception) -> bool:
    seen: set[int] = set()
    current: Exception | None = exc
    while isinstance(current, Exception) and id(current) not in seen:
        seen.add(id(current))
        if type(current).__name__ in _TRANSIENT_ERROR_NAMES:
            return True
        status_code = getattr(current, "status_code", None)
        if isinstance(status_code, int) and status_code in _TRANSIENT_STATUS_CODES:
            return True
        cause = getattr(current, "__cause__", None)
        context = getattr(current, "__context__", None)
        current = cause if isinstance(cause, Exception) else context
    return False


def _filter_updates(
    updates: dict[str, Any],
    allowed_keys: set[str],
    shape: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Split updates into allowed and rejected based on partition assignment."""
    if not isinstance(updates, dict):
        return {}, ["updates must be an object"]

    filtered: dict[str, Any] = {}
    rejected: list[str] = []

    for path, value in updates.items():
        top_key = _top_level_key(path, shape)
        if path in shape["repeat_groups"]:
            rejected.append(
                f"'{path}' is a whole repeat-group update. SplitAgent partitions "
                "must use repeat_group_id[index].field_id child paths, or "
                "repeat_group_id[index] = \"__DELETE_INSTANCE__\" for row deletion."
            )
            continue
        if top_key and top_key in allowed_keys:
            filtered[path] = value
        else:
            rejected.append(f"'{path}' is not in this partition's assignment")

    return filtered, rejected


def _top_level_key(path: str, shape: dict[str, Any]) -> str | None:
    """Determine which top-level state key a qid path belongs to."""
    if path in shape["bare_fields"]:
        return path
    if path in shape["repeat_groups"]:
        return path
    child_match = _REPEAT_CHILD_RE.match(path)
    if child_match:
        return child_match.group("group")
    row_match = _REPEAT_ROW_RE.match(path)
    if row_match:
        return row_match.group("group")
    return None


def _schema_for_partition(schema: dict[str, Any], allowed_keys: set[str]) -> dict[str, Any]:
    """Return a schema projection containing only assigned top-level keys."""

    projected = deepcopy(schema)
    projected["questions"] = _filter_schema_nodes(
        schema.get("questions") or [],
        allowed_keys=allowed_keys,
        repeat_stack=(),
    )
    return projected


def _filter_schema_nodes(
    nodes: list[dict[str, Any]],
    *,
    allowed_keys: set[str],
    repeat_stack: tuple[str, ...],
) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for node in nodes:
        if not isinstance(node, dict):
            continue
        structure_type = node.get("structure_type", "regular")

        if structure_type == "regular":
            field_id = node.get("id")
            if not field_id:
                continue
            top_key = repeat_stack[-1] if repeat_stack else field_id
            if top_key in allowed_keys:
                filtered.append(deepcopy(node))
            continue

        if structure_type == "repeat_group":
            group_id = node.get("id")
            if group_id and group_id in allowed_keys:
                filtered.append(deepcopy(node))
            continue

        if structure_type in {"group", "gate"}:
            children = _filter_schema_nodes(
                node.get("fields") or [],
                allowed_keys=allowed_keys,
                repeat_stack=repeat_stack,
            )
            if children:
                next_node = deepcopy(node)
                next_node["fields"] = children
                filtered.append(next_node)
            continue

        if structure_type == "branch":
            next_node = deepcopy(node)
            branch = deepcopy(node.get("branch") or {})
            kept = False
            routes = []
            for route in branch.get("routes") or []:
                children = _filter_schema_nodes(
                    route.get("children") or [],
                    allowed_keys=allowed_keys,
                    repeat_stack=repeat_stack,
                )
                if children:
                    next_route = deepcopy(route)
                    next_route["children"] = children
                    routes.append(next_route)
                    kept = True
            branch["routes"] = routes
            default_children = _filter_schema_nodes(
                branch.get("default_children") or [],
                allowed_keys=allowed_keys,
                repeat_stack=repeat_stack,
            )
            branch["default_children"] = default_children
            kept = kept or bool(default_children)
            if kept:
                next_node["branch"] = branch
                filtered.append(next_node)

    return filtered


def _build_partition_message(
    *,
    schema: dict[str, Any],
    shape: dict[str, Any],
    full_shape: dict[str, Any] | None = None,
    partition: dict[str, Any],
    partition_prior: dict[str, Any],
    visible_history: list[dict[str, Any]],
    current_utterance: Any,
) -> str:
    context = build_turn_context(
        schema=schema,
        shape=shape,
        prior_state=partition_prior,
        visible_history=visible_history,
        current_utterance=current_utterance,
        final_instruction=SPLIT_PARTITION_FINAL_INSTRUCTION,
        assignment=partition,
        prior_state_label="PRIOR RECORD STATE (assigned fields only)",
    )
    allowed = json.dumps(sorted(partition.get("fields") or []), ensure_ascii=False)
    all_shape = full_shape or shape
    top_level_keys = sorted(
        set(all_shape.get("bare_fields", set())) | set(all_shape.get("repeat_groups", {}))
    )
    unassigned_keys = [
        key for key in top_level_keys if key not in set(partition.get("fields") or [])
    ]
    unassigned_text = json.dumps(unassigned_keys, ensure_ascii=False)
    return (
        context
        + "\n\n=== GLOBAL FIELD AWARENESS ===\n"
        "Other top-level record fields exist outside this partition: "
        f"{unassigned_text}\n"
        "If a fact belongs more naturally to one of those fields, make no update "
        "from this partition."
        + "\n\n=== PARTITION TOOL PATH RULES ===\n"
        "Only call update_questionnaire_answers for paths whose top-level key is "
        f"in your assigned fields list: {allowed}\n"
        "Unassigned facts belong to other partitions; do not update them from this partition.\n"
        "The partition tool rejects whole repeat-group writes. Use explicit "
        "repeat child paths such as group_id[index].field_id, or explicit row deletion.\n"
        "\n=== CONSERVATIVE COMMITMENT CHECK ===\n"
        "- Silence is not evidence for false, no, none, unknown, other, or a default option.\n"
        "- A related fact is not enough; the assigned field's own question and instruction must ask for it.\n"
        "- Do not convert a concrete event detail into a nearby administrative/default field.\n"
        "- When the evidence is partial or ambiguous for all assigned fields, make no tool call.\n"
        "It is normal for a partition to make no tool call on a turn. Before calling "
        "the tool, verify that at least one assigned path has a specific "
        "evidence-licensed resulting value, correction, or clearing action."
    )


def _partition_tool_retry_message(
    results: list[dict[str, Any]],
    allowed_keys: set[str],
) -> str:
    allowed = json.dumps(sorted(allowed_keys), ensure_ascii=False)
    problems: list[str] = []
    for result in results:
        if not _tool_result_needs_retry(result):
            continue
        for key in ("error", "errors", "shape_errors", "rejected"):
            value = result.get(key)
            if isinstance(value, list):
                problems.extend(str(item) for item in value[:4])
            elif value:
                problems.append(str(value))
    summary = "; ".join(problems[:8]) or "The previous tool call failed."
    return (
        "The previous update_questionnaire_answers tool call failed or rejected "
        f"some updates: {summary}\n\n"
        "Retry now with corrected update paths only. For this SplitAgent partition, "
        f"every top-level key must be in your assigned fields list: {allowed}\n"
        "Use bare field ids for assigned top-level fields and "
        "repeat_group_id[index].field_id for repeat children. Do not use whole "
        "repeat-group ids, section/group/gate/branch ids, or rejected "
        "out-of-partition paths. Omit any update that is not explicitly "
        "evidence-licensed."
    )
