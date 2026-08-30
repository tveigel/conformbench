"""Utilities for preserving and aggregating LLM usage/cost metadata."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def json_ready(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(item) for item in value]
    if hasattr(value, "model_dump"):
        try:
            return json_ready(value.model_dump())
        except Exception:
            pass
    return str(value)


def message_accounting_snapshot(
    message: Any,
    *,
    phase: str,
    call_index: int,
    model_config: Mapping[str, Any] | None = None,
    loop_iteration: int | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    response_metadata = json_ready(getattr(message, "response_metadata", None) or {})
    usage_metadata = json_ready(getattr(message, "usage_metadata", None) or {})
    additional_kwargs = json_ready(getattr(message, "additional_kwargs", None) or {})
    payload = {
        "phase": phase,
        "call_index": call_index,
        "loop_iteration": loop_iteration,
        "model_config": json_ready(model_config or {}),
        "response_metadata": response_metadata,
        "usage_metadata": usage_metadata,
        "additional_kwargs": additional_kwargs,
        "extra": json_ready(extra or {}),
    }
    cost = extract_cost(payload)
    if cost is not None:
        payload["cost"] = cost
    usage = extract_usage(payload)
    if usage is not None:
        payload["usage"] = usage
    return payload


def extract_cost(value: Any) -> dict[str, Any] | None:
    """Return the first provider-style cost object found in a payload."""

    if isinstance(value, Mapping):
        if "cost" in value:
            normalised = _normalise_cost(value.get("cost"))
            if normalised is not None:
                return normalised
        if _is_cost_like(value):
            return _normalise_cost(value)
        for key in ("response_metadata", "additional_kwargs", "llm_output"):
            child = value.get(key)
            found = extract_cost(child)
            if found is not None:
                return found
        for child in value.values():
            found = extract_cost(child)
            if found is not None:
                return found
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            found = extract_cost(child)
            if found is not None:
                return found
    return None


def extract_usage(value: Any) -> dict[str, Any] | None:
    if isinstance(value, Mapping):
        for key in ("usage", "usage_metadata", "token_usage"):
            usage = _normalise_usage(value.get(key))
            if usage is not None:
                return usage
        for key in ("response_metadata", "additional_kwargs", "llm_output"):
            found = extract_usage(value.get(key))
            if found is not None:
                return found
        for child in value.values():
            found = extract_usage(child)
            if found is not None:
                return found
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            found = extract_usage(child)
            if found is not None:
                return found
    return None


def _normalise_cost(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    if isinstance(value.get("total"), (int, float)):
        return {
            "total": float(value["total"]),
            "currency": str(value.get("currency") or "USD"),
            "raw": json_ready(value),
        }
    interaction = value.get("interaction")
    if isinstance(interaction, Mapping) and isinstance(interaction.get("total"), (int, float)):
        return {
            "total": float(interaction["total"]),
            "currency": str(interaction.get("currency") or value.get("currency") or "USD"),
            "raw": json_ready(value),
        }
    return None


def _is_cost_like(value: Mapping[str, Any]) -> bool:
    return isinstance(value.get("total"), (int, float)) and bool(value.get("currency"))


def _normalise_usage(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    prompt = _first_number(value, "prompt_tokens", "input_tokens")
    completion = _first_number(value, "completion_tokens", "output_tokens")
    total = _first_number(value, "total_tokens")
    if total is None and (prompt is not None or completion is not None):
        total = int(prompt or 0) + int(completion or 0)
    if prompt is None and completion is None and total is None:
        return None
    return {
        "prompt_tokens": int(prompt or 0),
        "completion_tokens": int(completion or 0),
        "total_tokens": int(total or 0),
        "raw": json_ready(value),
    }


def _first_number(value: Mapping[str, Any], *keys: str) -> int | None:
    for key in keys:
        item = value.get(key)
        if isinstance(item, bool):
            continue
        if isinstance(item, int):
            return item
        if isinstance(item, float) and item.is_integer():
            return int(item)
    return None
