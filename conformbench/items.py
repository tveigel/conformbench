"""Load and normalize public benchmark item packets."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(
    os.environ.get("CONFORMBENCH_DATA_DIR", PROJECT_ROOT / "data")
).expanduser().resolve()
SCHEMA_ROOT = DATA_ROOT / "schema"
QUESTIONNAIRES_DIR = SCHEMA_ROOT / "questionnaires"
ITEMS_ROOT = DATA_ROOT / "items"
PUBLIC_ITEMS_DIR = ITEMS_ROOT / "public"
BENCHMARK_ITEMS_DIR = ITEMS_ROOT / "benchmark"
REPORTS_DIR = DATA_ROOT / "reports"


def normalize_item(data: dict[str, Any]) -> dict[str, Any]:
    """Normalize current and legacy item JSON into the public packet shape."""

    item_id = data.get("item_id") or data.get("scenario_id")
    if not item_id:
        raise ValueError("Item is missing item_id/scenario_id")

    questionnaire_id = (
        data.get("questionnaire_id")
        or data.get("source_questionnaire")
        or data.get("questionnaire")
    )
    if not questionnaire_id:
        raise ValueError(f"Item {item_id!r} is missing questionnaire id")
    questionnaire_id = questionnaire_id.removeprefix("pilot_")

    prior_state = data.get("prior_state")
    if not isinstance(prior_state, dict):
        raise ValueError(f"Item {item_id!r} is missing prior_state")

    current_utterance = data.get("current_utterance")
    if not isinstance(current_utterance, str):
        raise ValueError(f"Item {item_id!r} is missing current_utterance")

    visible_history = data.get("visible_history") or []
    if not isinstance(visible_history, list):
        raise ValueError(f"Item {item_id!r} has invalid visible_history")

    public_keys = {
        "item_id",
        "scenario_id",
        "questionnaire",
        "questionnaire_id",
        "source_questionnaire",
        "prior_state",
        "visible_history",
        "current_utterance",
        "gold_resulting_state",
    }

    return {
        **{k: v for k, v in data.items() if k != "metadata"},
        "item_id": item_id,
        "questionnaire_id": questionnaire_id,
        "prior_state": prior_state,
        "visible_history": visible_history,
        "current_utterance": current_utterance,
        "gold_resulting_state": data.get("gold_resulting_state"),
        "metadata": {key: value for key, value in data.items() if key not in public_keys},
    }


def load_item(path: str | Path) -> dict[str, Any]:
    return normalize_item(json.loads(Path(path).read_text(encoding="utf-8")))


def load_items(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    return [load_item(path) for path in paths]


def discover_items(root: str | Path, *, pattern: str = "ground_truth.json") -> list[dict[str, Any]]:
    return load_items(sorted(Path(root).rglob(pattern)))
