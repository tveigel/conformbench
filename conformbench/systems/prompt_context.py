"""Shared prompt and context builders for comparable baseline systems."""

from __future__ import annotations

import json
import os
import re
from typing import Any


COMMON_TASK_POLICY = """\
The public task context contains evidence and incumbent state. The schema field
guide, visible conversation history, and current user utterance are the evidence
used to license commitments. The prior record is the incumbent state to audit
against that evidence.

Your task is to audit the supplied prior record and return the post-turn record
containing all and only commitments licensed by the schema, visible history, and
current utterance.

Shared decision policy:
- Audit the prior state. Preserve a prior value only when it remains licensed
  by the visible history, current utterance, and schema. If a prior value is
  unsupported, withdrawn, contradicted, or schema-inconsistent, clear or repair
  it as licensed by the public packet.
- Treat visible history as evidence, not just background. If the prior record
  is missing, unsupported by, or contradicts a value determined by the schema,
  visible history, and current utterance, repair or clear it as licensed.
- Commit only values supported by public evidence. Do not infer plausible but
  unstated values, do not complete partial addresses or names from world
  knowledge, and do not fabricate identifiers.
- Use null when the evidence clears, retracts, denies, withdraws, or fails to
  license a prior scalar value.
- Put each fact in the most specific schema field whose question and field
  instruction ask for that fact. Do not copy broad narrative details into child
  fields unless the child field itself asks for them.
- Never write "<open>" as a public value. Do not replace a licensed prior value
  with an uncertain guess. If no value is licensed for a scalar field, use null.
- Repeat groups are arrays of entity rows. Keep row identity stable: update an
  existing row when evidence identifies the same entity; create or delete rows
  only when evidence or a schema repeat count licenses it.
- For repeat groups whose schema says mode=from_slot, update the count field
  when evidence licenses a count. If a count is known but row attributes are not,
  leave child fields null rather than inventing labels.
- Table fields are arrays of row objects. Retain existing rows only when they
  remain licensed by the evidence; clear, repair, add, or remove rows when the
  evidence and schema license that row-level change.
"""


DIRECT_OUTPUT_INTERFACE = """\
Output interface for Direct:
- Return exactly one complete JSON object representing the resulting record.
- Include every top-level regular field id and every repeat-group id shown in
  the prior record state. Do not include section/group/gate/branch ids.
- Repeat groups are null when absent/unknown, or arrays of row objects when
  rows are present. Every repeat row must include every child field id.
- Table fields are null when absent/unknown, or arrays of row objects when rows
  are present. Every table row must include every column id.
- Use null for unknown or empty scalar fields. Do not use placeholder strings.
- Output only the JSON object, with no commentary or explanation.
"""


FLATAGENT_OUTPUT_INTERFACE = """\
Output interface for FlatAgent:
- Use the update_questionnaire_answers tool for all committed changes.
- The tool applies sparse updates: omitted paths are not modified by that tool
  call. This does not override the audit policy; explicitly clear or repair any
  prior value that is unsupported, withdrawn, contradicted, or schema-
  inconsistent.
- If auditing the schema, visible history, and current utterance licenses no
  concrete change, do not call the tool.
- Regular fields use their bare schema id.
- Repeat child fields use repeat_group_id[index].field_id.
- Section/group/gate/branch ids are navigation labels only, not update paths.
- Delete a repeat row with repeat_group_id[index] = "__DELETE_INSTANCE__".
- Table fields use the bare schema id and an array of row objects.
- Repeat groups may also be set directly to a sparse array of row patches.
  In a repeat-group patch, omitted child fields are not modified by that patch;
  prefer explicit repeat_group_id[index].field_id child paths when possible,
  especially when clearing or repairing a child value.
- When done, reply briefly. Do not output the full JSON state; the tool-updated
  state is what will be submitted.
"""


DIFF_THEN_VERIFY_OUTPUT_INTERFACE = """\
Output interface for the verification stage:
- The first stage has already materialized a complete candidate post-turn
  record. Treat that candidate as the working state; do not rebuild it from
  scratch and do not repeat updates that it already contains correctly.
- Independently audit the complete candidate against the schema, original
  prior record, visible history, and current utterance.
- If the candidate needs correction, call update_questionnaire_answers once
  with one sparse corrective patch containing all and only needed corrections.
- If the candidate is already fully supported, do not call the tool.
- The patch is applied to the candidate, not to the original prior record.
- Regular fields use their bare schema id.
- Repeat child fields use repeat_group_id[index].field_id.
- Section/group/gate/branch ids are navigation labels only, not update paths.
- Delete a repeat row with repeat_group_id[index] = "__DELETE_INSTANCE__".
- Delete only a row that is present in the candidate. The candidate already
  reflects any count-field materialization, so do not also delete an index that
  disappeared when the count changed.
- To patch several children in one repeat row, use separate
  repeat_group_id[index].field_id entries. Do not use a row-object path such as
  repeat_group_id[index] = {field: value}, and never put
  "__DELETE_INSTANCE__" inside a row object's child field.
- Table fields use the bare schema id and an array of row objects.
- Repeat groups may also be set directly to a sparse array of row patches.
- Never use the tool to explain a judgment. After the optional corrective
  patch, reply briefly and do not output a full JSON state.
"""


SPLIT_PARTITION_OUTPUT_INTERFACE = """\
Output interface for a SplitAgent partition:
- You are responsible only for the fields listed in YOUR FIELD ASSIGNMENT.
- You receive only your assigned slice of the prior state. Fields outside that
  assignment are out of scope for this partition.
- Use the update_questionnaire_answers tool for all committed changes within
  your assigned fields.
- The tool applies sparse updates: omitted assigned paths are not modified by
  that tool call. This does not override the audit policy for assigned fields;
  explicitly clear or repair assigned prior values that are unsupported,
  withdrawn, contradicted, or schema-inconsistent.
- If auditing the schema, visible history, and current utterance licenses no
  concrete change to any assigned field, do not call the tool.
- Most turns will not require updates in most partitions. A no-op is correct
  only when the evidence licenses no change, clearing, or repair for your
  assigned fields.
- Do not fill a nearby assigned field merely because related information appears
  in the utterance. The field's own question and instruction must license the
  commitment.
- The tool rejects out-of-scope updates; do not update fields outside your
  assignment.
- Regular fields use their bare schema id.
- Repeat child fields use repeat_group_id[index].field_id.
- Section/group/gate/branch ids are navigation labels only, not update paths.
- Delete a repeat row with repeat_group_id[index] = "__DELETE_INSTANCE__".
- Table fields use the bare schema id and an array of row objects.
- Do not set a whole repeat-group id directly; this partition tool rejects
  whole repeat-group writes. Use repeat_group_id[index].field_id child paths or
  explicit repeat row deletion.
- Do not infer false/no/none/other/default values from silence or from a field
  being absent. Commit those values only when public evidence states them.
- For closed-choice fields, choose an option only when the evidence supports
  that exact option. Do not choose the nearest option by plausibility; update
  only to an option or null when the evidence licenses it.
- When done, reply briefly.
"""


def _system_prompt(*, agent_name: str, role: str, output_interface: str) -> str:
    return (
        f"You are {agent_name}, {role}.\n\n"
        f"{COMMON_TASK_POLICY}\n"
        f"{output_interface}"
    )


DIRECT_SYSTEM_PROMPT = _system_prompt(
    agent_name="Direct",
    role="a single-call no-tool baseline for structured form updating",
    output_interface=DIRECT_OUTPUT_INTERFACE,
)

FLATAGENT_SYSTEM_PROMPT = _system_prompt(
    agent_name="FlatAgent",
    role="a single-agent tool-using baseline for structured form updating",
    output_interface=FLATAGENT_OUTPUT_INTERFACE,
)

DIFF_THEN_VERIFY_SYSTEM_PROMPT = (
    "You are Verify, the independent second stage of a controlled "
    "Diff-then-Verify form-update baseline.\n\n"
    f"{COMMON_TASK_POLICY}\n"
    "For this stage, ORIGINAL PRIOR RECORD STATE is the incumbent before the "
    "user turn, and CANDIDATE POST-TURN RECORD TO VERIFY is the complete "
    "working result produced by the first-stage sparse diff. Audit the "
    "candidate itself. A corrective tool call mutates the candidate.\n\n"
    f"{DIFF_THEN_VERIFY_OUTPUT_INTERFACE}"
)

SPLIT_PARTITION_SYSTEM_PROMPT = _system_prompt(
    agent_name="a SplitAgent partition agent",
    role="a tool-using updater responsible for one assigned subset of fields",
    output_interface=SPLIT_PARTITION_OUTPUT_INTERFACE,
)

DIRECT_FINAL_INSTRUCTION = (
    "Audit the prior state against the schema, visible history, and current "
    "utterance. Return the full resulting record JSON using all and only "
    "evidence-licensed post-turn values."
)

FLATAGENT_FINAL_INSTRUCTION = (
    "Audit the prior state against the schema, visible history, and current "
    "utterance. Call update_questionnaire_answers with all and only "
    "evidence-licensed updates needed for the post-turn record."
)

DIFF_THEN_VERIFY_FINAL_INSTRUCTION = (
    "Independently audit every commitment in the complete candidate post-turn "
    "record against the schema and public evidence. Check for missed updates, "
    "unsupported retained or added values, incorrect clearing, gate effects, "
    "and repeated-instance routing. If and only if a correction is needed, "
    "call update_questionnaire_answers once with one complete sparse corrective "
    "patch. Otherwise make no tool call."
)

SPLIT_PARTITION_FINAL_INSTRUCTION = (
    "Audit your assigned prior-state fields against the schema, visible history, "
    "and current utterance. Call update_questionnaire_answers with all and only "
    "evidence-licensed updates for your assigned fields. If no assigned field is "
    "licensed to change, make no tool call."
)


def build_turn_context(
    *,
    schema: dict[str, Any],
    shape: dict[str, Any],
    prior_state: dict[str, Any],
    visible_history: list[dict[str, Any]],
    current_utterance: Any,
    final_instruction: str,
    assignment: dict[str, Any] | None = None,
    prior_state_label: str = "PRIOR RECORD STATE",
    candidate_state: dict[str, Any] | None = None,
    candidate_state_label: str = "CANDIDATE POST-TURN RECORD TO VERIFY",
) -> str:
    """Build the shared public context shown to baseline systems."""

    parts: list[str] = []
    if assignment is not None:
        parts.extend(
            [
                "=== YOUR FIELD ASSIGNMENT ===",
                f"Partition: {assignment.get('name', '')}",
                f"Description: {assignment.get('description', '')}",
                "Assigned fields: "
                + json.dumps(assignment.get("fields", []), ensure_ascii=False),
                "",
            ]
        )

    parts.extend(
        [
            "=== SCHEMA FIELD GUIDE ===",
            schema_guide(schema, shape),
            "\n=== STATE SHAPE CONTRACT ===",
            state_shape_contract(shape),
            f"\n=== {prior_state_label} ===",
            json.dumps(prior_state, indent=2, ensure_ascii=False),
        ]
    )
    if candidate_state is not None:
        parts.extend(
            [
                f"\n=== {candidate_state_label} ===",
                json.dumps(candidate_state, indent=2, ensure_ascii=False),
            ]
        )
    parts.extend(
        [
            "\n=== CONVERSATION HISTORY ===",
            "\n".join(history_line(item) for item in visible_history)
            if visible_history
            else "(none)",
            "\n=== CURRENT USER UTTERANCE ===",
            utterance_text(current_utterance),
            "\n=== TASK ===",
            final_instruction,
        ]
    )
    return "\n".join(parts)


def schema_guide(schema: dict[str, Any], shape: dict[str, Any]) -> str:
    lines = [
        "Use bare field ids for top-level regular fields.",
        "Use repeat_group_id[index].field_id for repeat children.",
        "Section/group/gate/branch ids are navigation labels only, not output keys.",
        "Field instructions describe semantic scope; keep values within that scope.",
        "",
    ]

    repeat_specs = shape.get("repeat_specs", {})

    def walk(
        nodes: list[dict[str, Any]],
        repeat_stack: tuple[str, ...] = (),
        indent: int = 0,
    ) -> None:
        prefix = "  " * indent
        for node in nodes:
            if not isinstance(node, dict):
                continue
            structure_type = node.get("structure_type", "regular")

            if structure_type == "regular":
                field_id = node.get("id")
                if not field_id:
                    continue
                qid = f"{repeat_stack[-1]}[i].{field_id}" if repeat_stack else field_id
                lines.append(prefix + _field_guide_line(qid, node))
                continue

            if structure_type == "repeat_group":
                group_id = node.get("id")
                if not group_id:
                    continue
                spec = repeat_specs.get(group_id) or _repeat_spec(node)
                mode = spec.get("mode") or "unknown"
                repeat_bits = [f"mode={mode}"]
                if spec.get("from_slot"):
                    repeat_bits.append(f"from_slot={spec['from_slot']}")
                if spec.get("count") is not None:
                    repeat_bits.append(f"count={spec['count']}")
                label = _compact_text(node.get("label") or group_id)
                repeat_line = (
                    f"{prefix}- REPEAT {group_id}: {label}; "
                    + ", ".join(repeat_bits)
                    + f"; child paths are {group_id}[index].field"
                )
                repeat_line += _metadata_suffix(node)
                lines.append(repeat_line)
                walk(node.get("fields") or [], (*repeat_stack, group_id), indent + 1)
                continue

            if structure_type == "gate":
                gate = node.get("gate") or {}
                label = _compact_text(node.get("label") or node.get("id") or "gate")
                lines.append(
                    f"{prefix}- GATE/CONDITION (not an output key) {node.get('id', '')}: {label}; "
                    f"active when {gate.get('gate_on')} in {gate.get('when_values')}"
                )
                walk(node.get("fields") or [], repeat_stack, indent + 1)
                continue

            if structure_type == "branch":
                branch = node.get("branch") or {}
                lines.append(
                    f"{prefix}- BRANCH/ROUTING (not an output key) {node.get('id', '')}: "
                    f"route on {branch.get('branch_on')}"
                )
                for route in branch.get("routes") or []:
                    lines.append(f"{prefix}  - when {route.get('when_value')}:")
                    walk(route.get("children") or [], repeat_stack, indent + 2)
                if branch.get("default_children"):
                    lines.append(f"{prefix}  - default:")
                    walk(branch.get("default_children") or [], repeat_stack, indent + 2)
                continue

            if structure_type == "group":
                label = _compact_text(node.get("label") or node.get("id") or "group")
                lines.append(
                    f"{prefix}- SECTION (not an output key) {node.get('id', '')}: {label}"
                )
                walk(node.get("fields") or [], repeat_stack, indent + 1)

    walk(schema.get("questions") or [])
    return "\n".join(lines)


def state_shape_contract(shape: dict[str, Any]) -> str:
    """Compact checklist of exact state keys and nested row keys."""

    top_level_keys = sorted(set(shape.get("bare_fields", set())) | set(shape.get("repeat_groups", {})))
    lines = [
        "The resulting record state is a flat JSON object.",
        "Top-level keys, exactly:",
        json.dumps(top_level_keys, ensure_ascii=False),
    ]

    repeat_groups = shape.get("repeat_groups") or {}
    if repeat_groups:
        lines.append("Repeat-group row object keys, exactly:")
        for group_id, fields in sorted(repeat_groups.items()):
            lines.append(f"- {group_id}: {json.dumps(sorted(fields), ensure_ascii=False)}")

    top_level_tables = shape.get("top_level_tables") or {}
    repeat_tables = shape.get("repeat_tables") or {}
    if top_level_tables or repeat_tables:
        lines.append("Table row column keys, exactly:")
        for table_id, columns in sorted(top_level_tables.items()):
            lines.append(f"- {table_id}: {json.dumps(sorted(columns), ensure_ascii=False)}")
        for group_id, tables in sorted(repeat_tables.items()):
            for table_id, columns in sorted(tables.items()):
                lines.append(
                    f"- {group_id}[i].{table_id}: "
                    f"{json.dumps(sorted(columns), ensure_ascii=False)}"
                )

    return "\n".join(lines)


def history_line(item: dict[str, Any]) -> str:
    speaker = item.get("speaker") or item.get("role") or "unknown"
    text = item.get("text") or item.get("content") or ""
    return f"[{speaker}]: {text}"


def utterance_text(current_utterance: Any) -> str:
    if isinstance(current_utterance, dict):
        return str(current_utterance.get("text", current_utterance))
    return str(current_utterance)


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


def _field_guide_line(qid: str, node: dict[str, Any]) -> str:
    bits = [f"- {qid}", f"type={node.get('type', 'unknown')}"]
    if node.get("options"):
        bits.append("options=" + json.dumps(node.get("options"), ensure_ascii=False))
    question = _compact_text(node.get("question_text"))
    if question:
        bits.append(f"question={question}")
    gold_standard = _compact_text(node.get("gold_standard"))
    if gold_standard:
        bits.append(f"instruction={gold_standard}")
    if node.get("type") == "table":
        columns = [
            _table_column_guide(column)
            for column in node.get("columns") or []
            if isinstance(column, dict)
        ]
        bits.append("columns=" + json.dumps(columns, ensure_ascii=False))
    bits.extend(_metadata_bits(node))
    return "; ".join(bits)


def _table_column_guide(column: dict[str, Any]) -> dict[str, str]:
    return {
        key: value
        for key, value in {
            "id": _compact_text(column.get("id")),
            "type": _compact_text(column.get("type")),
            "question": _compact_text(column.get("question_text"), limit=120),
            "instruction": _compact_text(column.get("gold_standard"), limit=160),
            "semantic_units": _information_unit_guide(column),
            "fold_policy": _compact_json(column.get("fold_policy") or column.get("fold"), limit=180),
            "parent_instructions": _compact_json(column.get("parent_gold_standards"), limit=220),
        }.items()
        if value
    }


def _information_unit_guide(node: dict[str, Any]) -> str:
    units: list[str] = []
    for unit in node.get("information_units") or []:
        if not isinstance(unit, dict):
            continue
        unit_id = _compact_text(unit.get("id"), limit=60)
        name = _compact_text(unit.get("name"), limit=80)
        description = _compact_text(unit.get("description"), limit=180)
        if unit_id and description:
            units.append(f"{unit_id}:{name}={description}" if name else f"{unit_id}={description}")
        elif unit_id:
            units.append(unit_id)
    return _compact_text("; ".join(units), limit=520)


def _metadata_suffix(node: dict[str, Any]) -> str:
    bits = _metadata_bits(node)
    return ("; " + "; ".join(bits)) if bits else ""


def _metadata_bits(node: dict[str, Any]) -> list[str]:
    bits: list[str] = []
    ius = _information_unit_guide(node)
    if ius:
        bits.append("semantic_units=" + ius)
    fold_policy = _compact_json(node.get("fold_policy") or node.get("fold"), limit=260)
    if fold_policy:
        bits.append("fold_policy=" + fold_policy)
    parent_gold = _compact_json(node.get("parent_gold_standards"), limit=360)
    if parent_gold:
        bits.append("parent_instructions=" + parent_gold)
    return bits


def _compact_json(value: Any, *, limit: int = 260) -> str:
    if value is None:
        return ""
    return _compact_text(json.dumps(value, ensure_ascii=False), limit=limit)


def _compact_text(value: Any, *, limit: int = 260) -> str:
    if value is None:
        return ""
    text = re.sub(r"\s+", " ", str(value)).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def llm_trace_enabled() -> bool:
    return os.getenv("CONFORMBENCH_TRACE_MESSAGES", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def trace_messages(messages: list[Any]) -> list[dict[str, Any]]:
    return [_trace_message(message) for message in messages]


def trace_response(response: Any) -> dict[str, Any]:
    return _trace_message(response)


def _trace_message(message: Any) -> dict[str, Any]:
    content = getattr(message, "content", "")
    payload: dict[str, Any] = {
        "role": getattr(message, "type", message.__class__.__name__),
        "class": message.__class__.__name__,
        "content": _jsonable(content),
    }
    name = getattr(message, "name", None)
    if name:
        payload["name"] = name
    tool_call_id = getattr(message, "tool_call_id", None)
    if tool_call_id:
        payload["tool_call_id"] = tool_call_id
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        payload["tool_calls"] = _jsonable(tool_calls)
    invalid_tool_calls = getattr(message, "invalid_tool_calls", None)
    if invalid_tool_calls:
        payload["invalid_tool_calls"] = _jsonable(invalid_tool_calls)
    return payload


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value, ensure_ascii=False)
        return value
    except TypeError:
        return json.loads(json.dumps(value, ensure_ascii=False, default=str))
