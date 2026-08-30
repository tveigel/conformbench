"""Shared provenance helpers for benchmark evaluation.

This module centralizes:
- request hashing
- prompt-trace serialization
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Mapping, Sequence


def _json_ready(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    if hasattr(value, "model_dump"):
        try:
            return _json_ready(value.model_dump())
        except Exception:
            pass
    return str(value)


def serialize_message(message: Any) -> dict[str, Any]:
    """Convert a LangChain/OpenAI/plain message object into JSON-safe form."""
    if isinstance(message, Mapping):
        return {str(key): _json_ready(value) for key, value in message.items()}

    payload: dict[str, Any] = {}

    role = getattr(message, "type", None)
    if role == "ai":
        role = "assistant"
    if role:
        payload["role"] = role

    content = getattr(message, "content", None)
    if content is not None:
        payload["content"] = _json_ready(content)

    name = getattr(message, "name", None)
    if name:
        payload["name"] = name

    tool_call_id = getattr(message, "tool_call_id", None)
    if tool_call_id:
        payload["tool_call_id"] = tool_call_id

    additional_kwargs = getattr(message, "additional_kwargs", None)
    if additional_kwargs:
        payload["additional_kwargs"] = _json_ready(additional_kwargs)

    response_metadata = getattr(message, "response_metadata", None)
    if response_metadata:
        payload["response_metadata"] = _json_ready(response_metadata)

    usage_metadata = getattr(message, "usage_metadata", None)
    if usage_metadata:
        payload["usage_metadata"] = _json_ready(usage_metadata)

    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        payload["tool_calls"] = _json_ready(tool_calls)

    invalid_tool_calls = getattr(message, "invalid_tool_calls", None)
    if invalid_tool_calls:
        payload["invalid_tool_calls"] = _json_ready(invalid_tool_calls)

    if not payload and hasattr(message, "model_dump"):
        try:
            return _json_ready(message.model_dump())
        except Exception:
            pass

    if not payload:
        payload["value"] = _json_ready(message)

    return payload


def serialize_messages(messages: Iterable[Any]) -> list[dict[str, Any]]:
    return [serialize_message(message) for message in messages]


def compute_request_hash(
    *,
    phase: str,
    call_index: int,
    messages: Sequence[Any],
    model_config: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> str:
    payload = {
        "phase": phase,
        "call_index": call_index,
        "messages": serialize_messages(messages),
        "model_config": _json_ready(model_config or {}),
        "extra": _json_ready(extra or {}),
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def build_trace_entry(
    *,
    phase: str,
    call_index: int,
    messages: Sequence[Any],
    model_config: Mapping[str, Any] | None,
    system_prompt: str | None,
    response: Any | None = None,
    tool_calls: Sequence[Any] | None = None,
    tool_results: Sequence[Any] | None = None,
    loop_iteration: int | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    serialized_messages = serialize_messages(messages)
    serialized_response = serialize_message(response) if response is not None else None
    return {
        "phase": phase,
        "call_index": call_index,
        "loop_iteration": loop_iteration,
        "request_hash": compute_request_hash(
            phase=phase,
            call_index=call_index,
            messages=messages,
            model_config=model_config,
            extra=extra,
        ),
        "messages": serialized_messages,
        "system_prompt": system_prompt,
        "model_config": _json_ready(model_config or {}),
        "tool_calls": _json_ready(tool_calls or []),
        "tool_results": _json_ready(tool_results or []),
        "response": serialized_response,
        "extra": _json_ready(extra or {}),
    }
