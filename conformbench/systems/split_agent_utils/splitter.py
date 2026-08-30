"""Create a form split via LLM, with deterministic validation and retry."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from llm import get_chat_model_with_config, response_text
from .schema_analysis import build_splitter_context, extract_field_inventory
from .validator import validate_split

_MAX_ATTEMPTS = 3
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)

SPLITTER_SYSTEM_PROMPT = """\
You are a form-splitting assistant. Given a structured questionnaire schema, \
you partition its top-level state keys into independent groups that will each \
be handled by a separate form-filling agent.

Your output is a JSON object with a single "partitions" array. Each partition:
- "name": short snake_case identifier
- "description": one-line description of what this group covers
- "fields": array of top-level state key IDs belonging to this partition

Top-level state keys are:
  - bare field IDs (scalars, even if nested inside groups/gates/branches in the schema)
  - repeat group IDs (the group itself, not its child fields)

Rules:
1. COMPLETE: every top-level key appears in exactly one partition.
2. DEPENDENCY-SAFE: fields listed under DEPENDENCY CONSTRAINTS must all be \
in the same partition. Never split a gate controller from its gated children, \
a branch controller from its route children, or a count field from its repeat group.
3. SEMANTICALLY COHERENT: group fields that are about the same topic or entity.
4. BALANCED: aim for 3-8 partitions. Each should have 3-15 top-level keys \
(a repeat group counts as one key regardless of how many instances it has).
5. Avoid single-field partitions — merge small isolated fields into the most \
semantically related partition.

Output ONLY the JSON object. No commentary.\
"""


def create_split(
    schema: dict[str, Any],
    questionnaire_id: str,
    *,
    model_id: str | None = None,
) -> dict[str, Any]:
    """Call the LLM to produce a validated split for this questionnaire."""

    context = build_splitter_context(schema)
    inventory = extract_field_inventory(schema)

    overrides: dict[str, Any] = {}
    if model_id:
        overrides["model"] = model_id

    model, model_config = get_chat_model_with_config(**overrides)
    messages: list[Any] = [
        SystemMessage(content=SPLITTER_SYSTEM_PROMPT),
        HumanMessage(content=_build_user_message(context, questionnaire_id)),
    ]

    last_error = ""
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        response = model.invoke(messages)
        text = response_text(response)

        try:
            split = _extract_json(text)
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = f"JSON parse error: {exc}"
            if attempt < _MAX_ATTEMPTS:
                messages.append(HumanMessage(content=_retry_message(last_error)))
            continue

        errors = validate_split(split, schema)
        if not errors:
            split["questionnaire_id"] = questionnaire_id
            split["model"] = model_config.get("model", "unknown")
            return split

        last_error = "Validation errors: " + "; ".join(errors)
        if attempt < _MAX_ATTEMPTS:
            messages.append(HumanMessage(content=_retry_message(last_error)))

    raise ValueError(
        f"Splitter failed to produce a valid split for '{questionnaire_id}' "
        f"after {_MAX_ATTEMPTS} attempts. Last error: {last_error}"
    )


def load_or_create_split(
    schema: dict[str, Any],
    questionnaire_id: str,
    splits_dir: Path,
    *,
    model_id: str | None = None,
) -> dict[str, Any]:
    """Load an existing split from disk, or create and save one."""
    splits_dir.mkdir(parents=True, exist_ok=True)
    split_path = splits_dir / f"{questionnaire_id}_split.json"

    if split_path.exists():
        with open(split_path) as f:
            split = json.load(f)
        errors = validate_split(split, schema)
        if not errors:
            return split

    split = create_split(schema, questionnaire_id, model_id=model_id)
    with open(split_path, "w") as f:
        json.dump(split, f, indent=2)
    return split


def _build_user_message(context: str, questionnaire_id: str) -> str:
    return (
        f"Questionnaire: {questionnaire_id}\n\n"
        f"{context}\n\n"
        "Produce the partition JSON now."
    )


def _retry_message(error: str) -> str:
    return (
        f"Your previous split was invalid:\n{error}\n\n"
        "Fix the issues and output the corrected JSON. "
        "Every top-level key must appear in exactly one partition. "
        "Dependency constraints must be respected."
    )


def _extract_json(text: str) -> dict[str, Any]:
    fence = _JSON_FENCE_RE.search(text)
    if fence:
        text = fence.group(1)
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found: {text[:200]}")
    return json.loads(text[start : end + 1])
