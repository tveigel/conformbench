"""FlatAgent baseline: update-tool calls projected into a final state."""

from __future__ import annotations

import asyncio
from copy import deepcopy
import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from conformbench.benchmark import validate_resulting_state
from conformbench.llm_accounting import message_accounting_snapshot
from llm import get_chat_model_with_config, response_text
from .prompt_context import (
    FLATAGENT_FINAL_INSTRUCTION,
    FLATAGENT_SYSTEM_PROMPT as SYSTEM_PROMPT,
    build_turn_context,
    llm_trace_enabled,
    trace_messages,
    trace_response,
)


_MAX_TOOL_ITERATIONS = 8
_DELETE_INSTANCE = "__DELETE_INSTANCE__"
_OPEN_VALUES = {"<open>", "<OPEN>", "open"}
_REPEAT_CHILD_RE = re.compile(
    r"^(?P<group>[A-Za-z_][A-Za-z0-9_-]*)"
    r"\[(?P<index>\d+)]\."
    r"(?P<field>[A-Za-z_][A-Za-z0-9_-]*)$"
)
_REPEAT_ROW_RE = re.compile(
    r"^(?P<group>[A-Za-z_][A-Za-z0-9_-]*)"
    r"\[(?P<index>\d+)]$"
)


def solve(turn: Any) -> dict[str, Any]:
    """Run FlatAgent on one benchmark turn and return the resulting state."""

    return asyncio.run(_solve_async(turn))


async def _solve_async(turn: Any) -> dict[str, Any]:
    schema = turn.schema
    shape = _schema_shape(schema)
    state = _materialize_repeat_counts(_complete_state(turn.prior_state, schema), shape)
    tool_updates: list[dict[str, Any]] = []

    metadata = getattr(turn, "metadata", {}) or {}
    overrides: dict[str, Any] = {}
    if metadata.get("model_id"):
        overrides["model"] = metadata["model_id"]
    reasoning_effort = metadata.get("model_reasoning_effort") or metadata.get("reasoning_effort")
    if reasoning_effort:
        overrides["reasoning_effort"] = reasoning_effort

    @tool("update_questionnaire_answers")
    def update_questionnaire_answers(updates: dict[str, Any]) -> dict[str, Any]:
        """Apply sparse questionnaire updates by qid/path. Omitted qids are not modified."""

        before_counts = _repeat_row_counts(state, shape)
        result = apply_answer_updates(
            state=state,
            updates=updates,
            schema=schema,
            shape=shape,
        )
        materialized = _materialize_repeat_counts(_complete_state(state, schema), shape)
        state.clear()
        state.update(materialized)
        after_counts = _repeat_row_counts(state, shape)
        count_changes = {
            group_id: count
            for group_id, count in after_counts.items()
            if before_counts.get(group_id) != count
        }
        if count_changes:
            result["materialized_repeat_groups"] = count_changes
        tool_updates.append({"updates": deepcopy(updates), "result": deepcopy(result)})
        return result

    update_tool = update_questionnaire_answers
    model, model_config = get_chat_model_with_config(tools=[update_tool], **overrides)
    messages: list[Any] = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=_build_user_message(
                schema=schema,
                shape=shape,
                prior_state=turn.prior_state,
                visible_history=turn.visible_history,
                current_utterance=turn.current_utterance,
            )
        ),
    ]

    final_text = ""
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
        response = await model.ainvoke(messages)
        if trace_call is not None:
            trace_call["response"] = trace_response(response)
            trace.append(trace_call)
        model_calls.append(
            message_accounting_snapshot(
                response,
                phase="flatagent_generation",
                call_index=len(model_calls) + 1,
                loop_iteration=iteration + 1,
                model_config=model_config,
            )
        )
        final_text = response_text(response)
        messages.append(response)

        tool_calls = list(getattr(response, "tool_calls", None) or [])
        if not tool_calls:
            break

        tool_results: list[dict[str, Any]] = []
        for call_index, tool_call in enumerate(tool_calls):
            tool_name = tool_call.get("name")
            tool_id = tool_call.get("id") or f"flatagent-{iteration}-{call_index}"
            if tool_name != update_tool.name:
                result = {"status": "error", "error": f"Unknown tool: {tool_name}"}
                tool_updates.append({
                    "tool_name": tool_name,
                    "updates": {},
                    "result": deepcopy(result),
                })
            else:
                result = update_tool.invoke(
                    {"updates": _tool_updates(tool_call.get("args") or {})}
                )
            tool_results.append(result if isinstance(result, dict) else {"status": "error"})
            messages.append(
                ToolMessage(
                    content=json.dumps(result, sort_keys=True, default=str),
                    tool_call_id=tool_id,
                    name=tool_name or update_tool.name,
                )
            )
        if any(_tool_result_needs_retry(result) for result in tool_results):
            messages.append(HumanMessage(content=_tool_retry_message(tool_results)))

    candidate_state = _materialize_repeat_counts(_complete_state(state, schema), shape)
    validation_errors = validate_resulting_state(candidate_state, schema)
    if validation_errors:
        raise ValueError(
            "FlatAgent produced invalid resulting state: "
            + "; ".join(validation_errors[:12])
        )

    return {
        "resulting_state": candidate_state,
        "agent_response": {
            "raw_response": final_text,
            "tool_updates": tool_updates,
            "model_calls": model_calls,
            **({"trace": trace} if trace is not None else {}),
        },
        "provenance": {
            "generation": {
                "agent": "FlatAgent",
                "model": model_config,
                "tool_policy": {
                    "mode": "update_tool",
                    "allowed_tools": [update_tool.name],
                    "max_tool_iterations": _MAX_TOOL_ITERATIONS,
                },
                "schema_interface": {
                    "repeat_group_representation": "indexed_child_qids",
                    "public_projection": (
                        "shared_schema_field_guide_with_schema_completed_prior_state"
                    ),
                },
            },
        },
    }


def apply_answer_updates(
    *,
    state: dict[str, Any],
    updates: dict[str, Any],
    schema: dict[str, Any],
    shape: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply FlatAgent qid/path updates to ``state`` in place."""

    shape = shape or _schema_shape(schema)
    if not isinstance(updates, dict):
        return {"status": "error", "error": "updates must be an object"}

    errors: list[str] = []
    deletes: dict[str, set[int]] = {}
    setters: list[tuple[str, Any]] = []

    for path, value in updates.items():
        if not isinstance(path, str) or not path:
            errors.append(f"Invalid update path: {path!r}")
            continue
        row_match = _REPEAT_ROW_RE.match(path)
        if row_match and value == _DELETE_INSTANCE:
            deletes.setdefault(row_match.group("group"), set()).add(
                int(row_match.group("index"))
            )
            continue
        setters.append((path, value))

    for group_id, indices in sorted(deletes.items()):
        if group_id not in shape["repeat_groups"]:
            errors.append(f"Unknown repeat group for delete: {group_id!r}")
            continue
        rows = state.get(group_id)
        if not isinstance(rows, list):
            state[group_id] = []
            rows = state[group_id]
        for row_index in sorted(indices, reverse=True):
            if 0 <= row_index < len(rows):
                rows.pop(row_index)
            else:
                errors.append(f"Delete index out of range: {group_id}[{row_index}]")

    for path, value in setters:
        if _is_open(value):
            continue
        error = _apply_setter(state, path, value, shape)
        if error:
            errors.append(error)

    shape_errors = validate_resulting_state(_complete_state(state, schema), schema)
    status = "ok" if not errors and not shape_errors else "error"
    result: dict[str, Any] = {
        "status": status,
        "applied_update_count": len(setters) + sum(len(v) for v in deletes.values()),
    }
    if errors:
        result["errors"] = errors
    if shape_errors:
        result["shape_errors"] = shape_errors
    return result


def _apply_setter(
    state: dict[str, Any],
    path: str,
    value: Any,
    shape: dict[str, Any],
) -> str | None:
    public_value = _public_value(value)

    if path in shape["repeat_groups"]:
        if not isinstance(public_value, list):
            return f"Repeat group {path!r} must be set to a list of sparse row patch objects"
        return _apply_repeat_group_patch(state, path, public_value, shape)

    if path in shape["bare_fields"]:
        columns = shape["top_level_tables"].get(path)
        state[path] = (
            _complete_table(public_value, columns)
            if columns is not None
            else public_value
        )
        return None

    child_match = _REPEAT_CHILD_RE.match(path)
    if not child_match:
        return f"Unknown update path: {path!r}"

    group_id = child_match.group("group")
    field_id = child_match.group("field")
    if group_id not in shape["repeat_groups"]:
        return f"Unknown repeat group: {group_id!r}"
    if field_id not in shape["repeat_groups"][group_id]:
        return f"Unknown field {field_id!r} for repeat group {group_id!r}"

    row_index = int(child_match.group("index"))
    rows = state.setdefault(group_id, [])
    if not isinstance(rows, list):
        state[group_id] = []
        rows = state[group_id]
    while len(rows) <= row_index:
        rows.append(_empty_repeat_row(group_id, shape))

    columns = shape["repeat_tables"].get(group_id, {}).get(field_id)
    rows[row_index][field_id] = (
        _complete_table(public_value, columns)
        if columns is not None
        else public_value
    )
    return None


def _apply_repeat_group_patch(
    state: dict[str, Any],
    group_id: str,
    row_patches: list[Any],
    shape: dict[str, Any],
) -> str | None:
    """Apply sparse row patches while preserving omitted child fields."""

    rows = state.setdefault(group_id, [])
    if not isinstance(rows, list):
        state[group_id] = []
        rows = state[group_id]

    errors: list[str] = []
    child_fields = shape["repeat_groups"].get(group_id, set())
    tables = shape["repeat_tables"].get(group_id, {})
    for row_index, row_patch in enumerate(row_patches):
        if not isinstance(row_patch, dict):
            errors.append(
                f"Repeat group {group_id!r} row patch at index {row_index} must be an object"
            )
            continue
        while len(rows) <= row_index:
            rows.append(_empty_repeat_row(group_id, shape))
        target_row = rows[row_index]
        if not isinstance(target_row, dict):
            target_row = _empty_repeat_row(group_id, shape)
            rows[row_index] = target_row
        for field_id, raw_value in row_patch.items():
            if field_id not in child_fields:
                errors.append(
                    f"Unknown field {field_id!r} for repeat group {group_id!r}"
                )
                continue
            if raw_value == _DELETE_INSTANCE:
                errors.append(
                    f"Delete repeat rows with {group_id}[{row_index}] = "
                    f'"{_DELETE_INSTANCE}", not inside a row patch field'
                )
                continue
            if _is_open(raw_value):
                continue
            public_value = _public_value(raw_value)
            columns = tables.get(field_id)
            target_row[field_id] = (
                _complete_table(public_value, columns)
                if columns is not None
                else public_value
            )

    return "; ".join(errors) if errors else None


def _tool_result_needs_retry(result: dict[str, Any]) -> bool:
    if not isinstance(result, dict):
        return True
    if result.get("status") == "error":
        return True
    for key in ("error", "errors", "shape_errors", "rejected"):
        value = result.get(key)
        if isinstance(value, list) and value:
            return True
        if value and not isinstance(value, list):
            return True
    return False


def _tool_retry_message(results: list[dict[str, Any]]) -> str:
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
        "Retry now with corrected update paths only. Use bare field ids for "
        "top-level fields, repeat_group_id[index].field_id for repeat children, "
        "or a repeat-group id with a sparse list of row patch objects. Do not use "
        "section/group/gate/branch ids. Omit any update that is not evidence-licensed."
    )


def _complete_state(prior_state: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    shape = _schema_shape(schema)
    source = prior_state if isinstance(prior_state, dict) else {}
    state: dict[str, Any] = {}

    for field_id in sorted(shape["bare_fields"]):
        columns = shape["top_level_tables"].get(field_id)
        if columns is not None:
            state[field_id] = _complete_table(source.get(field_id), columns)
        else:
            state[field_id] = deepcopy(source[field_id]) if field_id in source else None

    for group_id in sorted(shape["repeat_groups"]):
        rows = source.get(group_id)
        if rows is None or group_id not in source:
            state[group_id] = None
            continue
        if not isinstance(rows, list):
            state[group_id] = []
            continue
        state[group_id] = []
        for row in rows:
            state[group_id].append(
                _complete_repeat_row(row if isinstance(row, dict) else {}, group_id, shape)
            )

    return state


def _complete_repeat_row(
    row: dict[str, Any],
    group_id: str,
    shape: dict[str, Any],
) -> dict[str, Any]:
    tables = shape["repeat_tables"].get(group_id, {})
    return {
        field_id: (
            _complete_table(row.get(field_id), tables[field_id])
            if field_id in tables
            else deepcopy(row[field_id]) if field_id in row else None
        )
        for field_id in sorted(shape["repeat_groups"].get(group_id, set()))
    }


def _complete_table(value: Any, columns: set[str]) -> list[dict[str, Any]] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        return []
    rows: list[dict[str, Any]] = []
    for row in value:
        source = row if isinstance(row, dict) else {}
        rows.append(
            {
                column_id: deepcopy(source[column_id]) if column_id in source else None
                for column_id in sorted(columns)
            }
        )
    return rows


def _empty_repeat_row(group_id: str, shape: dict[str, Any]) -> dict[str, Any]:
    tables = shape["repeat_tables"].get(group_id, {})
    return {
        field_id: [] if field_id in tables else None
        for field_id in sorted(shape["repeat_groups"].get(group_id, set()))
    }


def _schema_shape(schema: dict[str, Any]) -> dict[str, Any]:
    bare_fields: set[str] = set()
    repeat_groups: dict[str, set[str]] = {}
    repeat_specs: dict[str, dict[str, Any]] = {}
    top_level_tables: dict[str, set[str]] = {}
    repeat_tables: dict[str, dict[str, set[str]]] = {}

    def walk(nodes: list[dict[str, Any]], repeat_stack: tuple[str, ...] = ()) -> None:
        for node in nodes:
            if not isinstance(node, dict):
                continue
            structure_type = node.get("structure_type", "regular")

            if structure_type == "regular":
                field_id = node.get("id")
                if not field_id:
                    continue
                if repeat_stack:
                    group_id = repeat_stack[-1]
                    repeat_groups.setdefault(group_id, set()).add(field_id)
                    if node.get("type") == "table":
                        repeat_tables.setdefault(group_id, {})[field_id] = _table_columns(node)
                else:
                    bare_fields.add(field_id)
                    if node.get("type") == "table":
                        top_level_tables[field_id] = _table_columns(node)
                continue

            if structure_type == "repeat_group":
                group_id = node.get("id")
                if group_id:
                    repeat_groups.setdefault(group_id, set())
                    repeat_specs[group_id] = _repeat_spec(node)
                    walk(node.get("fields") or [], (*repeat_stack, group_id))
                continue

            if structure_type in {"group", "gate"}:
                walk(node.get("fields") or [], repeat_stack)
                continue

            if structure_type == "branch":
                branch = node.get("branch") or {}
                for route in branch.get("routes") or []:
                    walk(route.get("children") or [], repeat_stack)
                walk(branch.get("default_children") or [], repeat_stack)

    walk(schema.get("questions") or [])
    return {
        "bare_fields": bare_fields,
        "repeat_groups": repeat_groups,
        "repeat_specs": repeat_specs,
        "top_level_tables": top_level_tables,
        "repeat_tables": repeat_tables,
    }


def _repeat_spec(node: dict[str, Any]) -> dict[str, Any]:
    repeat = node.get("repeat") or {}
    spec = {
        "mode": repeat.get("mode"),
        "from_slot": repeat.get("from_slot"),
        "count": repeat.get("count"),
        "label": node.get("label"),
        "item_label": repeat.get("item_label"),
    }
    return {key: value for key, value in spec.items() if value is not None}


def _table_columns(node: dict[str, Any]) -> set[str]:
    return {
        column["id"]
        for column in node.get("columns") or []
        if isinstance(column, dict) and column.get("id")
    }


def _materialize_repeat_counts(
    state: dict[str, Any],
    shape: dict[str, Any],
) -> dict[str, Any]:
    """Project schema count controllers into repeat row shells."""

    materialized = deepcopy(state)
    for group_id, spec in sorted(shape.get("repeat_specs", {}).items()):
        target_count = _repeat_target_count(materialized, spec)
        if target_count is None:
            continue
        rows = materialized.get(group_id)
        if not isinstance(rows, list):
            rows = []
        completed_rows = [
            _complete_repeat_row(row if isinstance(row, dict) else {}, group_id, shape)
            for row in rows[:target_count]
        ]
        while len(completed_rows) < target_count:
            completed_rows.append(_empty_repeat_row(group_id, shape))
        materialized[group_id] = completed_rows
    return materialized


def _repeat_row_counts(state: dict[str, Any], shape: dict[str, Any]) -> dict[str, int]:
    return {
        group_id: len(state.get(group_id) or [])
        for group_id in shape.get("repeat_groups", {})
        if isinstance(state.get(group_id), list)
    }


def _repeat_target_count(state: dict[str, Any], spec: dict[str, Any]) -> int | None:
    mode = spec.get("mode")
    if mode == "from_slot":
        from_slot = spec.get("from_slot")
        if not isinstance(from_slot, str):
            return None
        return _coerce_nonnegative_int(state.get(from_slot))
    return None


def _coerce_nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float):
        return int(value) if value >= 0 and value.is_integer() else None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            return int(stripped)
    return None


def _build_user_message(
    *,
    schema: dict[str, Any],
    shape: dict[str, Any],
    prior_state: dict[str, Any],
    visible_history: list[dict[str, Any]],
    current_utterance: Any,
) -> str:
    prompt_state = _materialize_repeat_counts(
        _complete_state(prior_state, schema),
        shape,
    )
    return build_turn_context(
        schema=schema,
        shape=shape,
        prior_state=prompt_state,
        visible_history=visible_history,
        current_utterance=current_utterance,
        final_instruction=FLATAGENT_FINAL_INSTRUCTION,
    )


def _tool_updates(args: dict[str, Any]) -> dict[str, Any]:
    updates = args.get("updates")
    if isinstance(updates, dict):
        return updates
    return args if isinstance(args, dict) else {}


def _is_open(value: Any) -> bool:
    return isinstance(value, str) and value in _OPEN_VALUES


def _public_value(value: Any) -> Any:
    return None if value == "" else value
