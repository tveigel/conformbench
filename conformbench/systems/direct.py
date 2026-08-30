"""Direct JSON baseline: one model emits the full resulting state."""

from __future__ import annotations

import json
import random
import re
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from llm import get_chat_model_with_config, response_text
from conformbench.benchmark import validate_resulting_state
from conformbench.llm_accounting import message_accounting_snapshot
from .flatagent import _complete_state, _materialize_repeat_counts, _schema_shape
from .prompt_context import (
    DIRECT_FINAL_INSTRUCTION,
    DIRECT_SYSTEM_PROMPT as SYSTEM_PROMPT,
    build_turn_context,
    llm_trace_enabled,
    trace_messages,
    trace_response,
)


_JSON_FENCE_RE = re.compile(
    r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL,
)
_MAX_ATTEMPTS = 3
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
    """Run direct JSON baseline on one benchmark turn."""
    return _solve(turn)


def _solve(turn: Any) -> dict[str, Any]:
    schema = turn.schema

    metadata = getattr(turn, "metadata", {}) or {}
    model_id = metadata.get("model_id")
    reasoning_effort = metadata.get("model_reasoning_effort") or metadata.get("reasoning_effort")
    invalid_output_policy = str(
        metadata.get("direct_invalid_output_policy") or "fail"
    ).strip().lower()
    overrides: dict[str, Any] = {}
    if model_id:
        overrides["model"] = model_id
    if reasoning_effort:
        overrides["reasoning_effort"] = reasoning_effort

    model, model_config = get_chat_model_with_config(**overrides)

    user_content = _build_user_message(
        schema=schema,
        prior_state=turn.prior_state,
        visible_history=turn.visible_history,
        current_utterance=turn.current_utterance,
    )

    messages: list[Any] = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ]

    last_error = ""
    last_text = ""
    model_calls: list[dict[str, Any]] = []
    rejected_attempts: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] | None = [] if llm_trace_enabled() else None
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        trace_call: dict[str, Any] | None = None
        if trace is not None:
            trace_call = {
                "call_index": attempt,
                "input_messages": trace_messages(messages),
            }
        response = _invoke_with_transport_retries(model, messages)
        if trace_call is not None:
            trace_call["response"] = trace_response(response)
            trace.append(trace_call)
        model_calls.append(
            message_accounting_snapshot(
                response,
                phase="direct_generation",
                call_index=attempt,
                model_config=model_config,
            )
        )
        last_text = response_text(response)
        try:
            result_state = _extract_json(last_text)
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = f"Response was not a valid JSON object: {exc}"
        else:
            shape_errors = validate_resulting_state(result_state, schema)
            if not shape_errors:
                return {
                    "resulting_state": result_state,
                    "agent_response": {
                        "raw_response": last_text,
                        "attempts": attempt,
                        "model_calls": model_calls,
                        **({"trace": trace} if trace is not None else {}),
                    },
                    "provenance": {
                        "generation": {
                            "agent": "Direct",
                            "model": model_config,
                            "tool_policy": {"mode": "state_json"},
                            "schema_interface": {
                                "public_projection": (
                                    "shared_schema_field_guide_with_schema_completed_prior_state"
                                ),
                            },
                        },
                    },
                }
            last_error = "Resulting state shape errors: " + "; ".join(shape_errors[:12])

        rejected_attempts.append(
            {
                "attempt": attempt,
                "error": last_error,
                "raw_response": last_text,
            }
        )
        if attempt < _MAX_ATTEMPTS:
            messages.append(HumanMessage(content=_retry_message(last_error)))

    failure_payload = {
        "agent": "Direct",
        "attempts": _MAX_ATTEMPTS,
        "last_error": last_error,
        "rejected_attempts": rejected_attempts,
        "model_calls": model_calls,
        **({"trace": trace} if trace is not None else {}),
    }
    if invalid_output_policy == "prior_state":
        shape = _schema_shape(schema)
        fallback_state = _materialize_repeat_counts(
            _complete_state(turn.prior_state, schema),
            shape,
        )
        return {
            "resulting_state": fallback_state,
            "agent_response": {
                "raw_response": last_text,
                "attempts": _MAX_ATTEMPTS,
                "model_calls": model_calls,
                "invalid_output_policy": "prior_state",
                "invalid_output_error": last_error,
                "rejected_attempts": rejected_attempts,
                **({"trace": trace} if trace is not None else {}),
            },
            "provenance": {
                "generation": {
                    "agent": "Direct",
                    "model": model_config,
                    "tool_policy": {
                        "mode": "state_json",
                        "invalid_output_policy": "prior_state",
                    },
                    "schema_interface": {
                        "public_projection": (
                            "shared_schema_field_guide_with_schema_completed_prior_state"
                        ),
                    },
                    "fallback": {
                        "reason": "invalid_resulting_state_json",
                        "scoring_interpretation": "no_update_prior_state",
                        "failure": failure_payload,
                    },
                },
            },
        }

    exc = ValueError(
        "Direct baseline failed to produce a valid resulting-state JSON object "
        f"after {_MAX_ATTEMPTS} attempts. Last error: {last_error}. "
        f"Last response prefix: {last_text[:500]}"
    )
    exc.conformbench_failure = failure_payload
    raise exc


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


def _build_user_message(
    *,
    schema: dict[str, Any],
    prior_state: dict[str, Any],
    visible_history: list[dict[str, Any]],
    current_utterance: Any,
) -> str:
    shape = _schema_shape(schema)
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
        final_instruction=DIRECT_FINAL_INSTRUCTION,
    )


def _retry_message(error: str) -> str:
    return (
        "Your previous answer could not be used as the resulting state.\n"
        f"{error}\n\n"
        "Try again. Output ONLY one complete JSON object matching the schema. "
        "Include every top-level schema field and repeat-group id. Use only schema field ids. "
        "Make repeat groups/table fields null when absent, or arrays of objects when rows are present. "
        "Every repeat row must contain every child field, and every table row every column. "
        "Do not include commentary."
    )


def _extract_json(text: str) -> dict[str, Any]:
    fence = _JSON_FENCE_RE.search(text)
    if fence:
        text = fence.group(1)

    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found in model response: {text[:200]}")

    return json.loads(text[start : end + 1])
