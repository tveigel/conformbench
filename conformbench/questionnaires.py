"""Load questionnaire schemas from ``data/schema/questionnaires``."""

from __future__ import annotations

import json
from typing import Any

from .items import QUESTIONNAIRES_DIR


def list_questionnaires() -> list[str]:
    """Return available questionnaire ids."""

    if not QUESTIONNAIRES_DIR.exists():
        return []
    return sorted(path.stem for path in QUESTIONNAIRES_DIR.glob("*.json"))


def load_questionnaire(questionnaire_id: str) -> dict[str, Any]:
    """Load one questionnaire schema by id."""

    if not questionnaire_id or "/" in questionnaire_id or "\\" in questionnaire_id:
        raise ValueError(f"Invalid questionnaire id: {questionnaire_id!r}")

    path = QUESTIONNAIRES_DIR / f"{questionnaire_id}.json"
    if not path.is_file():
        available = ", ".join(list_questionnaires())
        raise KeyError(f"Unknown questionnaire {questionnaire_id!r}. Available: {available}")

    return json.loads(path.read_text(encoding="utf-8"))
