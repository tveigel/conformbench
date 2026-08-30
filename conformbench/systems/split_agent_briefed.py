"""SplitAgentBriefed: partition triage, then one global updater commits.

The form is split into independent field groups by an LLM once per
questionnaire. A second questionnaire-level LLM step writes a reusable worker
brief for each partition, describing its scope, neighboring traps, and no-op
rules. Every turn broadcasts the public input to one triage agent per
partition. Triage agents do not write the record; they only decide whether
their partition plausibly needs an update. A single global updater then sees
the selected partitions and performs sparse record updates.

Run:
    python -m conformbench run --items public --solver conformbench.systems.split_agent_briefed:solve
"""

from __future__ import annotations

import json
import random
import re
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from conformbench.benchmark import validate_resulting_state
from conformbench.items import DATA_ROOT
from conformbench.llm_accounting import message_accounting_snapshot
from llm import get_chat_model_with_config, response_text

from .split_agent_utils.splitter import load_or_create_split
from .split_agent_utils.schema_analysis import build_splitter_context
from .flatagent import (
    _schema_shape,
    _complete_state,
    _materialize_repeat_counts,
    _REPEAT_CHILD_RE,
    _REPEAT_ROW_RE,
    _repeat_row_counts,
    _tool_result_needs_retry,
    _tool_updates,
    apply_answer_updates,
)
from .prompt_context import (
    COMMON_TASK_POLICY,
    FLATAGENT_OUTPUT_INTERFACE,
    SPLIT_PARTITION_FINAL_INSTRUCTION,
    SPLIT_PARTITION_SYSTEM_PROMPT as SUB_AGENT_SYSTEM_PROMPT,
    build_turn_context,
    llm_trace_enabled,
    trace_messages,
    trace_response,
)

_MAX_TOOL_ITERATIONS = 8
_TRIAGE_MAX_ATTEMPTS = 2
_SPLITS_DIR = DATA_ROOT / "splits"
_BRIEFS_DIR = _SPLITS_DIR / "partition_briefs"
_BRIEF_MAX_ATTEMPTS = 3
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)
_TRANSPORT_MAX_ATTEMPTS = 8
_TRANSIENT_ERROR_NAMES = {
    "APIConnectionError",
    "APIStatusError",
    "APITimeoutError",
    "ConnectError",
    "InternalServerError",
    "ProxyError",
    "ReadError",
    "ReadTimeout",
    "RateLimitError",
    "ServiceUnavailableError",
}

PARTITION_BRIEF_SYSTEM_PROMPT = """\
You are a senior form-architecture assistant. You receive a full questionnaire
schema summary and an already validated partition split. Your job is to write a
short operational worker brief for every partition.

Each partition worker will see the full conversation but may update ONLY its
assigned fields. The brief must help the worker understand:
- what its assigned fields are responsible for;
- what related evidence belongs to neighboring partitions instead;
- when a no-op is correct;
- how to handle repeated instances and row identity for its assigned repeats;
- what overcommit traps the worker should avoid.

Do not move fields between partitions. Do not invent item-specific facts. Do
not refer to gold answers or benchmark labels. Stay schema-level and reusable
across all items for this questionnaire.

Output ONLY a JSON object with this shape:
{
  "briefs": {
    "<partition_name>": {
      "worker_prompt": "120-220 words of direct instructions to this worker.",
      "in_scope_commitments": ["..."],
      "stay_in_lane_rules": ["..."],
      "cross_partition_awareness": ["..."],
      "repeat_identity_rules": ["..."],
      "abstention_rules": ["..."]
    }
  }
}

Every partition name from the supplied split must appear exactly once.
"""

TRIAGE_SYSTEM_PROMPT = f"""\
You are a high-recall partition triage agent for conversational form updating.

{COMMON_TASK_POLICY}

Your job is NOT to update the record. Your only job is to decide whether your
assigned partition should be shown to the global updater.

Use needs_update=true if any assigned field might require an evidence-licensed
addition, correction, refinement, retraction/clearing, repeat-row change, gate
consequence, or silent prior-state/history repair. Use needs_update=false only
when you are confident that no field in your assigned partition can change.
When uncertain, prefer true so the global updater can inspect the partition.

Output ONLY a JSON object:
{{
  "needs_update": true,
  "reason": "one short evidence-grounded sentence",
  "likely_fields": ["assigned_field_or_repeat_group_id"],
  "risk": "optional short note about ambiguity or lane boundary"
}}

Do not include updates, tool calls, or full record JSON.
"""

GLOBAL_UPDATER_SYSTEM_PROMPT = f"""\
You are SplitAgentBriefed's global updater. Partition triage agents have
selected the schema partitions that might need work; you alone make the final
record commitments.

{COMMON_TASK_POLICY}

{FLATAGENT_OUTPUT_INTERFACE}

Additional constraints:
- You receive only the selected partition schemas and selected prior-state
  fields. Unselected partitions are out of scope for this update call.
- A selected partition is only a routing hint. If the evidence does not license
  an update in a selected field, omit that field.
- Do not update outside the selected fields; the tool rejects out-of-scope
  paths.
- Prefer repeat_group_id[index].field_id child paths for repeat-row updates.
  Use whole repeat-group sparse patches only when they are clearer.
- In whole repeat-group sparse patches, every array element must be a row patch
  object. Never use null placeholders to skip rows.
- Delete repeat rows only with repeat_group_id[index] = "__DELETE_INSTANCE__";
  never put "__DELETE_INSTANCE__" inside a row patch object.
- If changing a count/controller field will truncate repeat rows, do not also
  patch rows that the new count removes.
"""

TRIAGE_FINAL_INSTRUCTION = (
    "Decide whether any assigned field might need an evidence-licensed update, "
    "repair, correction, clearing action, gate consequence, or repeat-row "
    "change. Return only the triage JSON object."
)

GLOBAL_UPDATER_FINAL_INSTRUCTION = (
    "Audit the selected prior-state fields against the schema, visible history, "
    "and current utterance. Call update_questionnaire_answers with all and only "
    "evidence-licensed updates for selected fields. If no selected field is "
    "actually licensed to change, clear, or repair, make no tool call."
)
_TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}


def solve(turn: Any) -> dict[str, Any]:
    """Run SplitAgentBriefed on one benchmark turn and return the resulting state."""
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
    brief_payload = load_or_create_partition_briefs(
        schema=schema,
        questionnaire_id=questionnaire_id,
        split=split,
        briefs_dir=_BRIEFS_DIR,
        model_id=overrides.get("model"),
    )
    partitions = _attach_partition_briefs(split["partitions"], brief_payload)

    shape = _schema_shape(schema)
    full_state = _materialize_repeat_counts(_complete_state(turn.prior_state, schema), shape)

    triage_results = [
        _run_partition_triage(
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

    selected_partitions = [
        partition
        for partition, result in zip(partitions, triage_results)
        if result.get("needs_update")
    ]
    selected_triage_results = [
        result for result in triage_results if result.get("needs_update")
    ]

    global_result = _run_global_updater(
        schema=schema,
        shape=shape,
        full_state=full_state,
        selected_partitions=selected_partitions,
        triage_results=selected_triage_results,
        visible_history=turn.visible_history,
        current_utterance=turn.current_utterance,
        overrides=overrides,
    )

    all_tool_updates: list[dict[str, Any]] = list(global_result["tool_updates"])
    all_model_calls: list[dict[str, Any]] = []
    all_traces: list[dict[str, Any]] = []
    partition_provenance: list[dict[str, Any]] = []

    for partition, result in zip(partitions, triage_results):
        all_model_calls.extend(result["model_calls"])
        if isinstance(result.get("trace"), list):
            all_traces.append({
                "partition": partition["name"],
                "phase": "triage",
                "trace": result["trace"],
            })
        partition_provenance.append({
            "name": partition["name"],
            "fields": partition["fields"],
            "needs_update": bool(result.get("needs_update")),
            "reason": result.get("reason", ""),
            "likely_fields": result.get("likely_fields", []),
            "generated_brief": bool(partition.get("generated_brief")),
        })

    all_model_calls.extend(global_result["model_calls"])
    if isinstance(global_result.get("trace"), list):
        all_traces.append({
            "partition": "global_updater",
            "phase": "update",
            "trace": global_result["trace"],
        })

    candidate_state = _materialize_repeat_counts(
        _complete_state(global_result["state"], schema),
        shape,
    )
    validation_errors = validate_resulting_state(candidate_state, schema)
    if validation_errors:
        raise ValueError(
            "SplitAgent produced invalid resulting state: "
            + "; ".join(validation_errors[:12])
        )

    return {
        "resulting_state": candidate_state,
        "agent_response": {
            "raw_response": (
                "SplitAgentBriefed: "
                f"{len(selected_partitions)}/{len(partitions)} partitions selected"
            ),
            "tool_updates": all_tool_updates,
            "model_calls": all_model_calls,
            "partition_triage": partition_provenance,
            **({"partition_traces": all_traces} if all_traces else {}),
        },
        "provenance": {
            "generation": {
                "agent": "SplitAgentBriefed",
                "model": overrides.get("model", "default"),
                "tool_policy": {
                    "mode": "partition_triage_then_global_update_tool",
                    "partitions": len(partitions),
                    "selected_partitions": len(selected_partitions),
                    "max_tool_iterations": _MAX_TOOL_ITERATIONS,
                    "partition_triage": "boolean_high_recall_no_record_writes",
                    "global_updates": "selected_fields_only",
                },
                "schema_interface": {
                    "public_projection": (
                        "triage_partition_schema_slices_then_selected_schema_union"
                    ),
                    "partition_context": (
                        "assigned_fields_only_prior_state_with_global_key_awareness"
                        "_and_generated_partition_brief"
                    ),
                    "partition_briefs": {
                        "path": str(_brief_cache_path(_BRIEFS_DIR, questionnaire_id)),
                        "model": brief_payload.get("model", "unknown"),
                    },
                },
                "split": {
                    "questionnaire_id": questionnaire_id,
                    "partitions": partition_provenance,
                },
            },
        },
    }


def load_or_create_partition_briefs(
    *,
    schema: dict[str, Any],
    questionnaire_id: str,
    split: dict[str, Any],
    briefs_dir: Path,
    model_id: str | None = None,
) -> dict[str, Any]:
    """Load or create reusable generated worker briefs for each partition."""
    briefs_dir.mkdir(parents=True, exist_ok=True)
    path = _brief_cache_path(briefs_dir, questionnaire_id)

    if path.exists():
        with path.open() as f:
            payload = json.load(f)
        normalized, errors = _normalize_partition_briefs(payload, split)
        if not errors:
            return normalized

    payload = create_partition_briefs(
        schema=schema,
        questionnaire_id=questionnaire_id,
        split=split,
        model_id=model_id,
    )
    with path.open("w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return payload


def create_partition_briefs(
    *,
    schema: dict[str, Any],
    questionnaire_id: str,
    split: dict[str, Any],
    model_id: str | None = None,
) -> dict[str, Any]:
    """Ask an LLM to write validated partition-worker briefs."""
    overrides: dict[str, Any] = {}
    if model_id:
        overrides["model"] = model_id
    model, model_config = _get_chat_model_with_transport_retries(**overrides)
    messages: list[Any] = [
        SystemMessage(content=PARTITION_BRIEF_SYSTEM_PROMPT),
        HumanMessage(
            content=_build_partition_brief_user_message(
                schema=schema,
                questionnaire_id=questionnaire_id,
                split=split,
            )
        ),
    ]

    last_error = ""
    for attempt in range(1, _BRIEF_MAX_ATTEMPTS + 1):
        response = _invoke_with_transport_retries(model, messages)
        text = response_text(response)
        try:
            payload = _extract_json_object(text)
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = f"JSON parse error: {exc}"
            if attempt < _BRIEF_MAX_ATTEMPTS:
                messages.append(HumanMessage(content=_brief_retry_message(last_error)))
            continue

        normalized, errors = _normalize_partition_briefs(payload, split)
        if not errors:
            normalized["questionnaire_id"] = questionnaire_id
            normalized["model"] = model_config.get("model", "unknown")
            return normalized

        last_error = "Validation errors: " + "; ".join(errors)
        if attempt < _BRIEF_MAX_ATTEMPTS:
            messages.append(HumanMessage(content=_brief_retry_message(last_error)))

    raise ValueError(
        f"Partition brief generation failed for '{questionnaire_id}' after "
        f"{_BRIEF_MAX_ATTEMPTS} attempts. Last error: {last_error}"
    )


def _brief_cache_path(briefs_dir: Path, questionnaire_id: str) -> Path:
    return briefs_dir / f"{questionnaire_id}_partition_briefs.json"


def _build_partition_brief_user_message(
    *,
    schema: dict[str, Any],
    questionnaire_id: str,
    split: dict[str, Any],
) -> str:
    split_summary = {
        "partitions": [
            {
                "name": partition.get("name"),
                "description": partition.get("description"),
                "fields": partition.get("fields") or [],
            }
            for partition in split.get("partitions") or []
        ]
    }
    return (
        f"Questionnaire: {questionnaire_id}\n\n"
        "=== FULL FORM FIELD INVENTORY AND DEPENDENCIES ===\n"
        f"{build_splitter_context(schema)}\n\n"
        "=== VALIDATED PARTITION SPLIT ===\n"
        f"{json.dumps(split_summary, indent=2, ensure_ascii=False)}\n\n"
        "Write reusable partition-worker briefs now. Each brief should make the "
        "worker's lane clear, including which tempting neighboring facts must be "
        "left to other partitions."
    )


def _normalize_partition_briefs(
    payload: dict[str, Any],
    split: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    if not isinstance(payload, dict):
        return {}, ["brief payload must be an object"]

    raw_briefs = payload.get("briefs")
    if raw_briefs is None and isinstance(payload.get("partitions"), list):
        raw_briefs = {
            item.get("name"): item
            for item in payload["partitions"]
            if isinstance(item, dict) and item.get("name")
        }
    if not isinstance(raw_briefs, dict):
        return {}, ["brief payload must contain a briefs object"]

    partition_names = [
        str(partition.get("name"))
        for partition in split.get("partitions") or []
        if partition.get("name")
    ]
    errors: list[str] = []
    normalized_briefs: dict[str, dict[str, Any]] = {}
    for name in partition_names:
        raw = raw_briefs.get(name)
        if not isinstance(raw, dict):
            errors.append(f"missing brief for partition {name!r}")
            continue
        worker_prompt = _clean_brief_text(raw.get("worker_prompt"))
        if not worker_prompt:
            errors.append(f"brief for {name!r} is missing worker_prompt")
            continue
        normalized_briefs[name] = {
            "worker_prompt": worker_prompt,
            "in_scope_commitments": _clean_brief_list(raw.get("in_scope_commitments")),
            "stay_in_lane_rules": _clean_brief_list(raw.get("stay_in_lane_rules")),
            "cross_partition_awareness": _clean_brief_list(raw.get("cross_partition_awareness")),
            "repeat_identity_rules": _clean_brief_list(raw.get("repeat_identity_rules")),
            "abstention_rules": _clean_brief_list(raw.get("abstention_rules")),
        }

    extras = sorted(str(name) for name in raw_briefs if str(name) not in partition_names)
    if extras:
        errors.append("unknown partition briefs: " + ", ".join(extras[:8]))

    if errors:
        return {}, errors
    return {
        "briefs": normalized_briefs,
        "partition_names": partition_names,
        "model": payload.get("model", "unknown"),
    }, []


def _attach_partition_briefs(
    partitions: list[dict[str, Any]],
    brief_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    briefs = brief_payload.get("briefs") or {}
    attached: list[dict[str, Any]] = []
    for partition in partitions:
        next_partition = deepcopy(partition)
        brief = briefs.get(partition.get("name"))
        if isinstance(brief, dict):
            next_partition["generated_brief"] = brief
        attached.append(next_partition)
    return attached


def _clean_brief_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "; ".join(str(item).strip() for item in value if str(item).strip())
    return ""


def _clean_brief_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _extract_json_object(text: str) -> dict[str, Any]:
    fence = _JSON_FENCE_RE.search(text)
    if fence:
        text = fence.group(1)
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found: {text[:200]}")
    payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("top-level JSON value must be an object")
    return payload


def _brief_retry_message(error: str) -> str:
    return (
        f"Your previous partition-brief JSON was invalid:\n{error}\n\n"
        "Return the corrected JSON object only. Include one brief for every "
        "partition name from the supplied split, each with a non-empty "
        "worker_prompt."
    )


def _run_partition_triage(
    *,
    partition: dict[str, Any],
    schema: dict[str, Any],
    shape: dict[str, Any],
    full_state: dict[str, Any],
    visible_history: list[dict[str, Any]],
    current_utterance: Any,
    overrides: dict[str, Any],
) -> dict[str, Any]:
    """Ask one partition whether its fields should be shown to the updater."""
    allowed_keys = set(partition["fields"])
    partition_schema = _schema_for_partition(schema, allowed_keys)
    partition_shape = _schema_shape(partition_schema)
    partition_prior = {
        k: deepcopy(full_state[k]) for k in allowed_keys if k in full_state
    }
    messages: list[Any] = [
        SystemMessage(content=TRIAGE_SYSTEM_PROMPT),
        HumanMessage(
            content=_build_triage_message(
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
    model, model_config = _get_chat_model_with_transport_retries(**overrides)
    model_calls: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] | None = [] if llm_trace_enabled() else None
    last_error = ""

    for attempt in range(1, _TRIAGE_MAX_ATTEMPTS + 1):
        trace_call: dict[str, Any] | None = None
        if trace is not None:
            trace_call = {
                "call_index": len(model_calls) + 1,
                "attempt": attempt,
                "input_messages": trace_messages(messages),
            }
        response = _invoke_with_transport_retries(model, messages)
        if trace_call is not None:
            trace_call["response"] = trace_response(response)
            trace.append(trace_call)
        model_calls.append(
            message_accounting_snapshot(
                response,
                phase="split_partition_triage",
                call_index=len(model_calls) + 1,
                loop_iteration=attempt,
                model_config=model_config,
                extra={"partition": partition["name"]},
            )
        )
        text = response_text(response)
        try:
            payload = _extract_json_object(text)
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = f"JSON parse error: {exc}"
        else:
            normalized, errors = _normalize_triage_payload(
                payload,
                allowed_keys=allowed_keys,
            )
            if not errors:
                normalized["model_calls"] = model_calls
                if trace is not None:
                    normalized["trace"] = trace
                return normalized
            last_error = "Validation errors: " + "; ".join(errors)

        if attempt < _TRIAGE_MAX_ATTEMPTS:
            messages.append(
                HumanMessage(
                    content=(
                        f"Your previous triage JSON was invalid:\n{last_error}\n\n"
                        "Return only the corrected JSON object with keys "
                        "needs_update, reason, likely_fields, and risk."
                    )
                )
            )

    fallback = {
        "needs_update": True,
        "reason": (
            "Triage JSON could not be validated, so this partition is selected "
            "to avoid a false negative."
        ),
        "likely_fields": sorted(allowed_keys),
        "risk": last_error,
        "parse_error": last_error,
        "model_calls": model_calls,
    }
    if trace is not None:
        fallback["trace"] = trace
    return fallback


def _run_global_updater(
    *,
    schema: dict[str, Any],
    shape: dict[str, Any],
    full_state: dict[str, Any],
    selected_partitions: list[dict[str, Any]],
    triage_results: list[dict[str, Any]],
    visible_history: list[dict[str, Any]],
    current_utterance: Any,
    overrides: dict[str, Any],
) -> dict[str, Any]:
    """Run one updater over the union of selected partitions."""
    state = deepcopy(full_state)
    tool_updates: list[dict[str, Any]] = []
    selected_keys = {
        field
        for partition in selected_partitions
        for field in partition.get("fields", [])
    }
    if not selected_keys:
        return {
            "state": state,
            "tool_updates": tool_updates,
            "model_calls": [],
            **({"trace": []} if llm_trace_enabled() else {}),
        }

    @tool("update_questionnaire_answers")
    def update_questionnaire_answers(updates: dict[str, Any]) -> dict[str, Any]:
        """Apply sparse updates to selected questionnaire fields only."""
        filtered, rejected = _filter_selected_updates(updates, selected_keys, shape)
        before_counts = _repeat_row_counts(state, shape)
        result: dict[str, Any] = {}
        if rejected:
            result["rejected"] = rejected
        if filtered:
            result.update(
                apply_answer_updates(
                    state=state,
                    updates=filtered,
                    schema=schema,
                    shape=shape,
                )
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
        else:
            result["status"] = "ok"
            result["applied_update_count"] = 0
        tool_updates.append({
            "updates": deepcopy(filtered),
            "rejected": deepcopy(rejected),
            "result": deepcopy(result),
        })
        return result

    update_tool = update_questionnaire_answers
    model, model_config = _get_chat_model_with_transport_retries(
        tools=[update_tool],
        **overrides,
    )
    selected_schema = _schema_for_partition(schema, selected_keys)
    selected_shape = _schema_shape(selected_schema)
    selected_prior = {
        k: deepcopy(state[k]) for k in selected_keys if k in state
    }
    messages: list[Any] = [
        SystemMessage(content=GLOBAL_UPDATER_SYSTEM_PROMPT),
        HumanMessage(
            content=_build_global_update_message(
                schema=selected_schema,
                shape=selected_shape,
                full_shape=shape,
                selected_partitions=selected_partitions,
                triage_results=triage_results,
                selected_prior=selected_prior,
                visible_history=visible_history,
                current_utterance=current_utterance,
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
        response = _invoke_with_transport_retries(model, messages)
        if trace_call is not None:
            trace_call["response"] = trace_response(response)
            trace.append(trace_call)
        model_calls.append(
            message_accounting_snapshot(
                response,
                phase="split_global_update",
                call_index=len(model_calls) + 1,
                loop_iteration=iteration + 1,
                model_config=model_config,
                extra={
                    "selected_partitions": [
                        partition.get("name") for partition in selected_partitions
                    ]
                },
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
            tool_id = tool_call.get("id") or f"split-global-{iteration}-{call_index}"
            if tool_name != update_tool.name:
                result = {"status": "error", "error": f"Unknown tool: {tool_name}"}
                tool_updates.append({
                    "tool_name": tool_name,
                    "updates": {},
                    "rejected": [],
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
            messages.append(
                HumanMessage(content=_global_tool_retry_message(tool_results, selected_keys))
            )

    return {
        "state": state,
        "tool_updates": tool_updates,
        "model_calls": model_calls,
        "raw_response": final_text,
        **({"trace": trace} if trace is not None else {}),
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
    model, model_config = _get_chat_model_with_transport_retries(
        tools=[update_tool],
        **overrides,
    )

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
            malformed_tool_call = _is_malformed_tool_call_transport_error(exc)
            if (
                not malformed_tool_call
                and not _is_transient_transport_error(exc)
            ) or attempt == _TRANSPORT_MAX_ATTEMPTS:
                raise
            if malformed_tool_call:
                messages.append(
                    HumanMessage(
                        content=(
                            "The previous assistant response used malformed "
                            "tool-call JSON and could not be parsed. Retry now. "
                            "If you call update_questionnaire_answers, its "
                            "argument must be one valid JSON object with an "
                            "'updates' object. Do not concatenate JSON objects "
                            "or emit any non-JSON tool argument text."
                        )
                    )
                )
            jitter_s = random.uniform(0.0, delay_s * 0.25)
            time.sleep(delay_s + jitter_s)
            delay_s *= 2.0
    raise RuntimeError("unreachable")


def _get_chat_model_with_transport_retries(**overrides: Any) -> Any:
    delay_s = 2.0
    for attempt in range(1, _TRANSPORT_MAX_ATTEMPTS + 1):
        try:
            return get_chat_model_with_config(**overrides)
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


def _is_malformed_tool_call_transport_error(exc: Exception) -> bool:
    seen: set[int] = set()
    current: Exception | None = exc
    while isinstance(current, Exception) and id(current) not in seen:
        seen.add(id(current))
        message = str(current)
        if (
            "tool-call transport error" in message
            and "unparseable JSON" in message
        ):
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


def _filter_selected_updates(
    updates: dict[str, Any],
    allowed_keys: set[str],
    shape: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Allow global updater writes only inside triage-selected partitions."""
    if not isinstance(updates, dict):
        return {}, ["updates must be an object"]

    filtered: dict[str, Any] = {}
    rejected: list[str] = []
    for path, value in updates.items():
        top_key = _top_level_key(path, shape)
        if top_key and top_key in allowed_keys:
            filtered[path] = value
        else:
            rejected.append(f"'{path}' is outside the triage-selected partitions")
    return filtered, rejected


def _normalize_triage_payload(
    payload: dict[str, Any],
    *,
    allowed_keys: set[str],
) -> tuple[dict[str, Any], list[str]]:
    if not isinstance(payload, dict):
        return {}, ["triage payload must be an object"]

    needs_update = _coerce_triage_bool(payload.get("needs_update"))
    if needs_update is None:
        return {}, ["needs_update must be a boolean"]

    reason = _clean_brief_text(payload.get("reason"))[:600]
    risk = _clean_brief_text(payload.get("risk"))[:400]
    likely_fields = _clean_brief_list(payload.get("likely_fields"))
    allowed_sorted = sorted(allowed_keys)
    normalized_fields: list[str] = []
    for field in likely_fields:
        if field in allowed_keys and field not in normalized_fields:
            normalized_fields.append(field)

    if likely_fields and not normalized_fields and needs_update:
        normalized_fields = allowed_sorted
        risk = (risk + " " if risk else "") + "Unrecognized likely_fields were supplied."
    if normalized_fields and not needs_update:
        needs_update = True
        risk = (risk + " " if risk else "") + "likely_fields made triage positive."

    return {
        "needs_update": needs_update,
        "reason": reason,
        "likely_fields": normalized_fields,
        "risk": risk,
    }, []


def _coerce_triage_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "yes", "y", "1", "needs_update"}:
            return True
        if text in {"false", "no", "n", "0", "no_update"}:
            return False
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


def _build_triage_message(
    *,
    schema: dict[str, Any],
    shape: dict[str, Any],
    full_shape: dict[str, Any],
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
        final_instruction=TRIAGE_FINAL_INSTRUCTION,
        assignment=partition,
        prior_state_label="PRIOR RECORD STATE (assigned fields only)",
    )
    generated_brief = _format_generated_partition_brief(partition.get("generated_brief"))
    top_level_keys = sorted(
        set(full_shape.get("bare_fields", set())) | set(full_shape.get("repeat_groups", {}))
    )
    unassigned_keys = [
        key for key in top_level_keys if key not in set(partition.get("fields") or [])
    ]
    return (
        context
        + (
            "\n\n=== PARTITIONER-GENERATED SCOPE BRIEF ===\n"
            + generated_brief
            if generated_brief
            else ""
        )
        + "\n\n=== GLOBAL FIELD AWARENESS ===\n"
        "Other top-level record fields exist outside this partition: "
        f"{json.dumps(unassigned_keys, ensure_ascii=False)}\n"
        "If the evidence belongs only to those fields, return needs_update=false. "
        "If the evidence might affect both your fields and neighboring fields, "
        "return needs_update=true and explain the boundary.\n"
        "\n=== TRIAGE POLICY ===\n"
        "- Prefer false positives over false negatives: true means the global updater "
        "will inspect this partition, not that an update is guaranteed.\n"
        "- Use false only when the visible evidence clearly licenses no addition, "
        "correction, refinement, retraction, clearing action, repeat-row change, "
        "gate consequence, or silent repair for assigned fields.\n"
        "- Do not propose record values here. Only route attention."
    )


def _build_global_update_message(
    *,
    schema: dict[str, Any],
    shape: dict[str, Any],
    full_shape: dict[str, Any],
    selected_partitions: list[dict[str, Any]],
    triage_results: list[dict[str, Any]],
    selected_prior: dict[str, Any],
    visible_history: list[dict[str, Any]],
    current_utterance: Any,
) -> str:
    selected_fields = sorted(
        {
            field
            for partition in selected_partitions
            for field in partition.get("fields", [])
        }
    )
    assignment = {
        "name": "selected_partitions_global_update",
        "description": (
            "Union of partitions whose high-recall triage agents flagged possible "
            "record work"
        ),
        "fields": selected_fields,
    }
    context = build_turn_context(
        schema=schema,
        shape=shape,
        prior_state=selected_prior,
        visible_history=visible_history,
        current_utterance=current_utterance,
        final_instruction=GLOBAL_UPDATER_FINAL_INSTRUCTION,
        assignment=assignment,
        prior_state_label="PRIOR RECORD STATE (selected fields only)",
    )
    triage_lines = []
    for partition, result in zip(selected_partitions, triage_results):
        likely_fields = result.get("likely_fields") or []
        triage_lines.append(
            "- "
            + str(partition.get("name"))
            + ": "
            + str(result.get("reason") or "selected")
            + (
                " Likely fields: "
                + json.dumps(likely_fields, ensure_ascii=False)
                if likely_fields
                else ""
            )
            + (
                " Risk: "
                + str(result.get("risk"))
                if result.get("risk")
                else ""
            )
        )

    all_top_level = sorted(
        set(full_shape.get("bare_fields", set())) | set(full_shape.get("repeat_groups", {}))
    )
    unselected = [field for field in all_top_level if field not in set(selected_fields)]
    return (
        context
        + "\n\n=== SELECTED PARTITION TRIAGE ===\n"
        + ("\n".join(triage_lines) if triage_lines else "(none)")
        + "\n\n=== UNSELECTED PARTITIONS ARE OUT OF SCOPE ===\n"
        "Do not update these unselected top-level fields: "
        f"{json.dumps(unselected, ensure_ascii=False)}\n"
        "A selected partition is only a routing hint. It is correct to make no "
        "tool call if the evidence does not license a concrete selected-field "
        "change, clearing, or repair.\n"
        "\n=== SELECTED TOOL PATH RULES ===\n"
        "Only call update_questionnaire_answers for selected fields: "
        f"{json.dumps(selected_fields, ensure_ascii=False)}\n"
        "Use bare field ids for selected scalar/table fields, "
        "repeat_group_id[index].field_id for repeat children, or, when clearer, "
        "a selected repeat-group id with a sparse array of row patch objects. "
        "Omitted paths are not modified by that tool call. This does not "
        "override the audit policy for selected fields. In sparse repeat-group "
        "arrays, every array element must be an object; never use null placeholders. Delete "
        'repeat rows only with repeat_group_id[index] = "__DELETE_INSTANCE__", '
        "never inside a row patch object. If a count/controller update removes "
        "rows, do not also patch rows that the new count removes."
    )


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
    generated_brief = _format_generated_partition_brief(partition.get("generated_brief"))
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
        + (
            "\n\n=== PARTITIONER-GENERATED WORKER BRIEF ===\n"
            + generated_brief
            if generated_brief
            else ""
        )
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


def _format_generated_partition_brief(brief: Any) -> str:
    if not isinstance(brief, dict):
        return ""
    lines: list[str] = []
    worker_prompt = _clean_brief_text(brief.get("worker_prompt"))
    if worker_prompt:
        lines.append(worker_prompt)
    sections = [
        ("In-scope commitments", brief.get("in_scope_commitments")),
        ("Stay-in-lane rules", brief.get("stay_in_lane_rules")),
        ("Cross-partition awareness", brief.get("cross_partition_awareness")),
        ("Repeat identity rules", brief.get("repeat_identity_rules")),
        ("Abstention rules", brief.get("abstention_rules")),
    ]
    for title, value in sections:
        items = _clean_brief_list(value)
        if not items:
            continue
        lines.append(f"\n{title}:")
        lines.extend(f"- {item}" for item in items)
    return "\n".join(lines).strip()


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


def _global_tool_retry_message(
    results: list[dict[str, Any]],
    selected_keys: set[str],
) -> str:
    allowed = json.dumps(sorted(selected_keys), ensure_ascii=False)
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
        "Retry with corrected update paths only. For this global update pass, "
        f"top-level keys must be in the selected fields list: {allowed}\n"
        "Use bare field ids for selected top-level fields, "
        "repeat_group_id[index].field_id for repeat children, or a selected "
        "repeat-group id with a sparse list of row patch objects. Do not use "
        "null placeholders in sparse repeat-group lists. Delete rows only with "
        'repeat_group_id[index] = "__DELETE_INSTANCE__", never inside row patch '
        "objects. If a count/controller update removes rows, do not patch rows "
        "that the new count removes. Do not use section/group/gate/branch ids. "
        "Omit any update that is not explicitly evidence-licensed."
    )
