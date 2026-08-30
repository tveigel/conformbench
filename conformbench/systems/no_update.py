"""No-update lower-bound baseline.

This deterministic sanity baseline ignores the current utterance and returns
the schema-completed prior state without deriving any new repeat instances. It
measures how much score can be achieved by preservation alone.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


def solve(turn: Any) -> dict[str, Any]:
    """Return the prior record state as the complete post-turn state."""

    shape = _schema_shape(turn.schema)
    state = _complete_state(turn.prior_state, shape)
    return {
        "resulting_state": state,
        "agent_response": {
            "raw_response": "NoUpdate baseline: copied schema-completed prior state.",
            "operations": [],
        },
        "provenance": {
            "generation": {
                "agent": "NoUpdate",
                "model": "deterministic",
                "tool_policy": {"mode": "copy_prior"},
                "schema_interface": {
                    "public_projection": "schema_completed_prior_state_without_repeat_materialization",
                },
            },
        },
    }


def _complete_state(prior_state: dict[str, Any], shape: dict[str, Any]) -> dict[str, Any]:
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
        state[group_id] = [
            _complete_repeat_row(row if isinstance(row, dict) else {}, group_id, shape)
            for row in rows
        ]

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


def _schema_shape(schema: dict[str, Any]) -> dict[str, Any]:
    bare_fields: set[str] = set()
    repeat_groups: dict[str, set[str]] = {}
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
        "top_level_tables": top_level_tables,
        "repeat_tables": repeat_tables,
    }


def _table_columns(node: dict[str, Any]) -> set[str]:
    return {
        column["id"]
        for column in node.get("columns") or []
        if isinstance(column, dict) and column.get("id")
    }
