"""
Deterministic evaluation for hard fields (exact match, set match).

No LLM calls — pure string / set comparison with normalisation.

Text/open exact fields are promoted to ``needs_semantic`` when a deterministic
match fails and the candidate is non-empty. The turn-level router applies the
prior-state preserve guard before calling the judge.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from loguru import logger

from .scoring import (
    infer_scoring_profile,
    is_empty_for_scoring,
    normalize_value_for_scoring,
)

from .models import FieldVerdict

# ── Questionnaire type map ───────────────────────────────────────────────

_FREE_TEXT_TYPES = frozenset({"text", "multiline_text"})
_EXACT_MISMATCH_SEMANTIC_TYPES = frozenset({
    "free_text",
    "multiline_text",
    "string",
    "text",
    "textarea",
})
_CHOICE_TYPES = frozenset({"single_choice", "multiple_choice"})

# Thread-local holder so callers can set it before evaluate_hard_fields()
import threading as _threading
_thread_local = _threading.local()

def _get_active_questionnaire_path() -> Path | None:
    return getattr(_thread_local, "questionnaire_path", None)

def _set_active_questionnaire_path(val: Path | None) -> None:
    _thread_local.questionnaire_path = val


_QUESTIONNAIRE_METADATA_KEYS = (
    "type",
    "structure_type",
    "question_text",
    "label",
    "gold_standard",
    "information_units",
    "normalization_rules",
    "other_specify",
    "options",
    "choices",
    "values",
    "scoring_profile",
    "unknown_equivalent",
)


@lru_cache(maxsize=8)
def _load_questionnaire_types(
    questionnaire_path: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Build flat map of field ids and qualified child ids to schema metadata.

    Recursively walks questions, groups, gates, repeat-group fields,
    and table columns so every leaf field is indexed.
    """
    path = questionnaire_path or _get_active_questionnaire_path()
    if path is None:
        logger.warning(
            "No questionnaire path set — field-type lookups will be skipped. "
            "Pass questionnaire_path or set _active_questionnaire_path via _set_active_questionnaire_path()."
        )
        return {}
    if not path.exists():
        logger.warning(f"Questionnaire not found at {path} — field-type lookups unavailable.")
        return {}
    raw = json.loads(path.read_text())
    type_map: dict[str, dict[str, Any]] = {}

    def _metadata_for(item: dict[str, Any]) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        for key in _QUESTIONNAIRE_METADATA_KEYS:
            if key in item:
                metadata[key] = item[key]
        if "options" not in metadata:
            metadata["options"] = item.get("choices") or item.get("values") or []
        metadata.setdefault("other_specify", False)
        return metadata

    def _register(item: dict[str, Any], key: str) -> None:
        if "id" not in item:
            return
        metadata = _metadata_for(item)
        if "type" not in metadata and not any(
            child_key in item for child_key in ("fields", "columns", "questions")
        ):
            return
        type_map[key] = metadata

    def _walk(items: list[dict], parent_path: str | None = None) -> None:
        for item in items:
            item_id = item.get("id")
            current_path = (
                f"{parent_path}.{item_id}" if parent_path and item_id else item_id
            )
            if isinstance(item_id, str):
                _register(item, item_id)
                if isinstance(current_path, str) and current_path != item_id:
                    _register(item, current_path)
                if parent_path:
                    short_path = f"{parent_path.rsplit('.', 1)[-1]}.{item_id}"
                    if short_path not in {item_id, current_path}:
                        _register(item, short_path)
            for key in ("fields", "columns"):
                if key in item and isinstance(item[key], list):
                    _walk(item[key], str(current_path) if current_path else parent_path)
            if "questions" in item and isinstance(item["questions"], list):
                _walk(item["questions"], str(current_path) if current_path else parent_path)

    _walk(raw.get("questions", []))
    return type_map


def _base_field_id(qid: str) -> str:
    """Extract base field name: 'vehicles[0].licence_plate' → 'licence_plate'."""
    return qid.rsplit(".", 1)[-1] if "." in qid else qid


def _field_lookup_keys(qid: str) -> list[str]:
    """Return schema lookup keys from most to least specific."""
    keys = [qid]
    unindexed = re.sub(r"\[\d+\]", "", qid)
    if unindexed != qid:
        keys.append(unindexed)
    base = _base_field_id(qid)
    if base not in keys:
        keys.append(base)
    return keys


def _is_free_text_field(qid: str) -> bool:
    """True if the questionnaire input type is free text (text / multiline_text)."""
    type_map = _load_questionnaire_types(_get_active_questionnaire_path())
    info = next(
        (type_map[key] for key in _field_lookup_keys(qid) if key in type_map),
        None,
    )
    return info is not None and info["type"] in _FREE_TEXT_TYPES


def get_questionnaire_field_info(
    qid: str,
    questionnaire_path: Path | None = None,
) -> dict[str, Any]:
    """Return questionnaire metadata for a field id, if available."""
    type_map = _load_questionnaire_types(questionnaire_path or _get_active_questionnaire_path())
    info = next(
        (type_map[key] for key in _field_lookup_keys(qid) if key in type_map),
        {},
    )
    return dict(info) if isinstance(info, dict) else {}


def _has_other_specification(value: Any) -> bool:
    """True when a value carries an explicit ``Other: ...`` specification."""
    if isinstance(value, str):
        return _normalise(value).startswith("other:")
    if isinstance(value, list):
        return any(_has_other_specification(item) for item in value)
    return False


def _strip_other_specification(value: Any) -> Any:
    """Return the free-text payload from ``Other: ...`` strings."""
    if isinstance(value, str) and _normalise(value).startswith("other:"):
        return value.split(":", 1)[1].strip()
    return value


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _other_specification_reason(
    *,
    qid: str,
    agent_value: Any,
    expected: Any,
    gt_entry: dict[str, Any],
) -> FieldVerdict | None:
    """Promote explicit Other-specify values to the semantic judge."""
    info = get_questionnaire_field_info(qid)
    is_other_specify_field = bool(info.get("other_specify"))
    if not is_other_specify_field and not (
        _has_other_specification(agent_value) or _has_other_specification(expected)
    ):
        return None

    if not (_has_other_specification(agent_value) or _has_other_specification(expected)):
        return None

    verdict = _other_specification_exact_verdict(
        qid=qid,
        agent_value=agent_value,
        expected=expected,
        gt_entry=gt_entry,
    )
    if verdict is not None:
        return verdict

    return FieldVerdict(
        correctness="needs_semantic",
        source=_source_from_gt(gt_entry),
        reasoning=(
            f"Field {qid} uses an explicit Other-specify value "
            f"(candidate={agent_value!r}, gold={expected!r}). "
            "Promoting to semantic evaluation."
        ),
    )


# ── Normalisation ────────────────────────────────────────────────────────

_WS = re.compile(r"\s+")
_CHOICE_ALIAS_PUNCT = re.compile(r"[^a-z0-9]+")
_MORPHOLOGICAL_CHOICE_SUFFIXES = frozenset({"s", "es", "ed", "ing", "y"})
_CHOICE_NEGATION_CUES = frozenset({
    "absence",
    "absent",
    "lack",
    "lacked",
    "lacking",
    "lacks",
    "neither",
    "never",
    "no",
    "none",
    "nor",
    "not",
    "without",
})
_CHOICE_NEGATION_LOOKBACK = 4


def _normalise(value: Any) -> str:
    """Lowercase, collapse whitespace, strip."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return _WS.sub(" ", str(value)).strip().lower()


def _is_choice_field(qid: str) -> bool:
    info = get_questionnaire_field_info(qid)
    return str(info.get("type") or "") in _CHOICE_TYPES


def _choice_options(qid: str) -> list[str]:
    info = get_questionnaire_field_info(qid)
    if str(info.get("type") or "") not in _CHOICE_TYPES:
        return []
    options = info.get("options") or []
    if not isinstance(options, list):
        return []
    return [str(option) for option in options if option is not None]


def _choice_alias_key(
    value: Any,
    *,
    scoring_profile: str | None,
) -> str:
    """Return a stable comparison key for choice labels and simple aliases."""
    normalized = normalize_value_for_scoring(
        value,
        scoring_profile=scoring_profile,
    )
    if normalized is None:
        return ""
    if isinstance(normalized, bool):
        return "true" if normalized else "false"
    if isinstance(normalized, (list, dict)):
        return ""
    text = str(normalized).replace("&", " and ")
    return _WS.sub(" ", _CHOICE_ALIAS_PUNCT.sub(" ", text)).strip()


def _has_choice_negation(value_key: str) -> bool:
    return any(token in _CHOICE_NEGATION_CUES for token in value_key.split())


def _phrase_has_negation_scope(value_tokens: list[str], start_index: int) -> bool:
    window_start = max(0, start_index - _CHOICE_NEGATION_LOOKBACK)
    return any(
        token in _CHOICE_NEGATION_CUES
        for token in value_tokens[window_start:start_index]
    )


def _has_unguarded_token_phrase(value_key: str, option_key: str) -> bool:
    value_tokens = value_key.split()
    option_tokens = option_key.split()
    if not value_tokens or not option_tokens:
        return False
    if len(option_tokens) > len(value_tokens):
        return False
    for index in range(0, len(value_tokens) - len(option_tokens) + 1):
        if value_tokens[index:index + len(option_tokens)] != option_tokens:
            continue
        if not _phrase_has_negation_scope(value_tokens, index):
            return True
    return False


def _has_negated_token_phrase(value_key: str, option_key: str) -> bool:
    value_tokens = value_key.split()
    option_tokens = option_key.split()
    if not value_tokens or not option_tokens:
        return False
    if len(option_tokens) > len(value_tokens):
        return False
    for index in range(0, len(value_tokens) - len(option_tokens) + 1):
        if value_tokens[index:index + len(option_tokens)] != option_tokens:
            continue
        if _phrase_has_negation_scope(value_tokens, index):
            return True
    return False


def _is_morphological_choice_variant(value_compact: str, option_compact: str) -> bool:
    if len(value_compact) < 4 or len(option_compact) < 4:
        return False
    if value_compact.startswith(option_compact):
        return value_compact[len(option_compact):] in _MORPHOLOGICAL_CHOICE_SUFFIXES
    if option_compact.startswith(value_compact):
        return option_compact[len(value_compact):] in _MORPHOLOGICAL_CHOICE_SUFFIXES
    return False


def _canonical_choice_option_key(
    qid: str,
    value: Any,
    *,
    scoring_profile: str | None,
) -> str | None:
    """Map a value to the questionnaire option it names, if unambiguous."""
    if _has_other_specification(value):
        return None

    value_key = _choice_alias_key(value, scoring_profile=scoring_profile)
    value_compact = value_key.replace(" ", "")
    if not value_key:
        return None

    options = _choice_options(qid)
    if not options:
        return None

    option_keys: list[tuple[str, str]] = []
    for option in options:
        option_key = _choice_alias_key(option, scoring_profile=scoring_profile)
        if not option_key:
            continue
        option_keys.append((option_key, option_key.replace(" ", "")))

    for option_key, option_compact in option_keys:
        if value_key == option_key or value_compact == option_compact:
            return option_key

    possible_matches: set[str] = set()
    value_has_negation = _has_choice_negation(value_key)
    for option_key, option_compact in option_keys:
        if option_key == "other":
            continue
        phrase_matches = _has_unguarded_token_phrase(value_key, option_key)
        morphological_matches = (
            not value_has_negation
            and _is_morphological_choice_variant(value_compact, option_compact)
        )
        if phrase_matches or morphological_matches:
            possible_matches.add(option_key)

    if len(possible_matches) == 1:
        return next(iter(possible_matches))
    return None


def _has_negated_choice_option_reference(
    qid: str,
    value: Any,
    *,
    scoring_profile: str | None,
) -> bool:
    """True when a value mentions a listed option under local negation."""
    value_key = _choice_alias_key(value, scoring_profile=scoring_profile)
    if not value_key or not _has_choice_negation(value_key):
        return False

    value_compact = value_key.replace(" ", "")
    value_tokens = value_key.split()
    for option in _choice_options(qid):
        option_key = _choice_alias_key(option, scoring_profile=scoring_profile)
        if not option_key or option_key == "other":
            continue
        option_compact = option_key.replace(" ", "")
        if _has_negated_token_phrase(value_key, option_key):
            return True
        if len(option_key.split()) == 1:
            for token_index, token in enumerate(value_tokens):
                if not _phrase_has_negation_scope(value_tokens, token_index):
                    continue
                if _is_morphological_choice_variant(token, option_compact):
                    return True
        if _is_morphological_choice_variant(value_compact, option_compact):
            return True
    return False


def _choice_values_match(
    qid: str,
    left: Any,
    right: Any,
    *,
    scoring_profile: str | None,
) -> bool:
    left_key = _choice_alias_key(left, scoring_profile=scoring_profile)
    right_key = _choice_alias_key(right, scoring_profile=scoring_profile)
    if left_key and left_key == right_key:
        return True
    left_compact = left_key.replace(" ", "")
    right_compact = right_key.replace(" ", "")
    if left_compact and left_compact == right_compact:
        return True

    left_option = _canonical_choice_option_key(
        qid,
        left,
        scoring_profile=scoring_profile,
    )
    right_option = _canonical_choice_option_key(
        qid,
        right,
        scoring_profile=scoring_profile,
    )
    return left_option is not None and left_option == right_option


_OTHER_SPEC_OPTIONAL_PARTICLES = frozenset({
    "away",
    "back",
    "down",
    "in",
    "into",
    "off",
    "on",
    "onto",
    "out",
    "up",
})


def _other_spec_texts_match(
    left: Any,
    right: Any,
    *,
    scoring_profile: str | None,
) -> bool:
    """Compare Other-specify payload text without requiring wrapper identity."""
    left_key = _choice_alias_key(
        _strip_other_specification(left),
        scoring_profile=scoring_profile,
    )
    right_key = _choice_alias_key(
        _strip_other_specification(right),
        scoring_profile=scoring_profile,
    )
    if not left_key or not right_key:
        return False
    if left_key == right_key:
        return True
    left_tokens = left_key.split()
    right_tokens = right_key.split()
    left_core = [token for token in left_tokens if token not in _OTHER_SPEC_OPTIONAL_PARTICLES]
    right_core = [token for token in right_tokens if token not in _OTHER_SPEC_OPTIONAL_PARTICLES]
    if left_core and left_core == right_core:
        return True
    if len(left_core) == 1 and len(right_core) == 1:
        left_compact = left_core[0]
        right_compact = right_core[0]
        return _is_morphological_choice_variant(left_compact, right_compact)
    return False


def _specified_other_option_key(
    qid: str,
    value: Any,
    *,
    scoring_profile: str | None,
) -> str | None:
    """Return the listed option named by an Other payload, if any."""
    if not _has_other_specification(value):
        return None
    option_key = _canonical_choice_option_key(
        qid,
        _strip_other_specification(value),
        scoring_profile=scoring_profile,
    )
    return None if option_key == "other" else option_key


def _other_specification_exact_verdict(
    *,
    qid: str,
    agent_value: Any,
    expected: Any,
    gt_entry: dict[str, Any],
) -> FieldVerdict | None:
    """Resolve simple Other-specify equivalences without an LLM call."""
    scoring_profile = infer_scoring_profile(qid, gt_entry, expected=expected)
    candidate_other_names_option = _specified_other_option_key(
        qid,
        agent_value,
        scoring_profile=scoring_profile,
    )
    expected_has_other = _has_other_specification(expected)
    expected_option = _canonical_choice_option_key(
        qid,
        _strip_other_specification(expected),
        scoring_profile=scoring_profile,
    )

    if (
        candidate_other_names_option is not None
        and not expected_has_other
        and expected_option == candidate_other_names_option
    ):
        return FieldVerdict(
            correctness="partially_correct",
            partial_reason="wrong_choice",
            source=_source_from_gt(gt_entry),
            reasoning=(
                f"Candidate uses Other-specify for listed option '{expected}'. "
                "The meaning is recoverable, but the wrong form option was selected."
            ),
        )

    if not _other_spec_texts_match(
        agent_value,
        expected,
        scoring_profile=scoring_profile,
    ):
        return None

    return FieldVerdict(
        correctness="correct",
        source=_source_for_exact_match(
            agent_value,
            expected,
            gt_entry,
            scoring_profile=scoring_profile,
        ),
        reasoning=(
            f"Other-specify payload match for {qid}: expected '{expected}', "
            f"got '{agent_value}'."
        ),
    )


def _is_bare_other_choice_value(
    value: Any,
    *,
    scoring_profile: str | None,
) -> bool:
    return _choice_alias_key(value, scoring_profile=scoring_profile) == "other"


def _should_escalate_exact_mismatch(
    qid: str,
    agent_value: Any,
    expected: Any,
    gt_entry: dict[str, Any],
    *,
    scoring_profile: str | None,
) -> bool:
    """Return whether a failed exact comparison needs semantic review.

    This is the runtime equivalent of the Requirement 4 P0/P1/P2 audit policy:
    text/string/open fields get an exact-match fast path, but a non-empty
    mismatch is not terminal until the turn-level router checks whether the
    candidate merely preserved a prior value when gold required a change.
    """
    if is_empty_for_scoring(agent_value, scoring_profile=scoring_profile):
        return False
    if is_empty_for_scoring(expected, scoring_profile=scoring_profile):
        return False

    if _is_choice_field(qid) and _is_bare_other_choice_value(
        agent_value,
        scoring_profile=scoring_profile,
    ):
        return False

    info = get_questionnaire_field_info(qid)
    field_type = str(info.get("type") or "").strip().lower()
    if field_type in _EXACT_MISMATCH_SEMANTIC_TYPES:
        return True
    if bool(info.get("other_specify")):
        return True
    return _has_other_specification(agent_value) or _has_other_specification(expected)


def _choice_item_norm(
    qid: str,
    value: Any,
    *,
    scoring_profile: str | None,
) -> str:
    option_key = _canonical_choice_option_key(
        qid,
        value,
        scoring_profile=scoring_profile,
    )
    if option_key is not None:
        return f"choice:{option_key}"
    alias_key = _choice_alias_key(value, scoring_profile=scoring_profile)
    if alias_key:
        return f"value:{alias_key}"
    return _normalise(value)


def _normalise_bool(value: Any) -> bool | None:
    """Coerce to bool for fields like any_injuries."""
    if isinstance(value, bool):
        return value
    s = _normalise(value)
    if s in ("true", "yes", "1"):
        return True
    if s in ("false", "no", "0"):
        return False
    return None


# ── Source scoring (deterministic from GT metadata) ──────────────────────


def _source_from_gt(gt_entry: dict) -> str:
    """Derive source label from ground-truth metadata."""
    evidence_source = gt_entry.get("evidence_source")
    if evidence_source == "prior_state":
        return "prior_state"
    if evidence_source in {"history", "visible_history"}:
        return "likely_inferred"
    if (
        evidence_source in {None, ""}
        and not gt_entry.get("present_in_utterance", True)
        and str(gt_entry.get("evidence") or "").strip()
    ):
        return "likely_inferred"
    if not gt_entry.get("present_in_utterance", True):
        return "fabricated"
    diff = gt_entry.get("extraction_difficulty", "direct")
    if diff == "not_stated":
        return "fabricated"
    if diff == "requires_inference":
        return "likely_inferred"
    return "extracted"


def _candidate_matches_forbidden_commit(
    agent_value: Any,
    gt_entry: dict,
    *,
    scoring_profile: str | None,
) -> bool:
    commits = gt_entry.get("_forbidden_commits") or gt_entry.get("forbidden_commits") or []
    if not isinstance(commits, list):
        return False
    agent_norm = normalize_value_for_scoring(agent_value, scoring_profile=scoring_profile)
    for commit in commits:
        if not isinstance(commit, dict):
            continue
        if "wrong_value" in commit:
            wrong_value = commit.get("wrong_value")
        else:
            wrong_value = commit.get("resulting_value")
        wrong_norm = normalize_value_for_scoring(wrong_value, scoring_profile=scoring_profile)
        if agent_norm == wrong_norm:
            return True
    return False


def _source_for_candidate_mismatch(
    agent_value: Any,
    gt_entry: dict,
    *,
    scoring_profile: str | None,
) -> str | None:
    """Best-effort source label for an incorrect candidate value.

    Deterministic fields do not have access to open-ended semantic evidence, so
    this function uses item metadata and explicit forbidden commits. It avoids
    calling plausible but unlicensed values "fabricated" merely because they do
    not match gold.
    """
    if is_empty_for_scoring(agent_value, scoring_profile=scoring_profile):
        return None
    if _candidate_matches_forbidden_commit(
        agent_value,
        gt_entry,
        scoring_profile=scoring_profile,
    ):
        return "unsupported_inference"

    evidence_source = gt_entry.get("evidence_source")
    if evidence_source == "prior_state":
        return "prior_state"
    if evidence_source in {"history", "visible_history"}:
        return "likely_inferred"

    if not gt_entry.get("present_in_utterance", True):
        return "fabricated"
    diff = gt_entry.get("extraction_difficulty", "direct")
    if diff == "not_stated":
        return "fabricated"
    if diff == "requires_inference":
        return "likely_inferred"
    return "extracted"


def _source_for_exact_match(
    agent_value: Any,
    expected: Any,
    gt_entry: dict,
    *,
    scoring_profile: str | None,
) -> str | None:
    """Return provenance for an exact match.

    A correctly empty, unsupported field is an abstention, not a fabricated
    value. Keep explicit support for empty values when the GT records evidence
    such as a negative statement or gate consequence.
    """
    evidence_source = gt_entry.get("evidence_source")
    has_recorded_evidence = bool(str(gt_entry.get("evidence") or "").strip())
    if (
        evidence_source in {None, "", "none"}
        and not has_recorded_evidence
        and is_empty_for_scoring(expected, scoring_profile=scoring_profile)
        and is_empty_for_scoring(agent_value, scoring_profile=scoring_profile)
    ):
        return None
    return _source_from_gt(gt_entry)


# ── Exact match ──────────────────────────────────────────────────────────


def _normalised_values_match(
    left_norm: Any,
    right_norm: Any,
) -> bool:
    if left_norm == right_norm:
        return True
    if isinstance(left_norm, str) and isinstance(right_norm, (int, float)) and not isinstance(right_norm, bool):
        try:
            left_numeric = float(left_norm) if "." in left_norm else int(left_norm)
        except ValueError:
            return False
        return left_numeric == right_norm
    if isinstance(right_norm, str) and isinstance(left_norm, (int, float)) and not isinstance(left_norm, bool):
        try:
            right_numeric = float(right_norm) if "." in right_norm else int(right_norm)
        except ValueError:
            return False
        return left_norm == right_numeric
    return False


def _candidate_matches_value(
    qid: str,
    agent_value: Any,
    target_value: Any,
    *,
    scoring_profile: str | None,
) -> bool:
    if _is_choice_field(qid):
        return _choice_values_match(
            qid,
            agent_value,
            target_value,
            scoring_profile=scoring_profile,
        )

    agent_norm = normalize_value_for_scoring(
        agent_value,
        scoring_profile=scoring_profile,
    )
    target_norm = normalize_value_for_scoring(
        target_value,
        scoring_profile=scoring_profile,
    )
    return _normalised_values_match(agent_norm, target_norm)


def _candidate_matches_any_value(
    qid: str,
    agent_value: Any,
    values: list[Any],
    *,
    scoring_profile: str | None,
) -> bool:
    return any(
        _candidate_matches_value(
            qid,
            agent_value,
            value,
            scoring_profile=scoring_profile,
        )
        for value in values
    )


def _candidate_matches_acceptable_alternative(
    qid: str,
    agent_value: Any,
    expected: Any,
    alternatives: list[Any],
    *,
    scoring_profile: str | None,
) -> bool:
    if not alternatives:
        return False

    # Concrete values must not rescue a null/empty gold commitment.  The only
    # empty-gold alternative license is the explicit unknown-equivalent profile,
    # which is already handled by scoring normalization before this point.
    if (
        is_empty_for_scoring(expected, scoring_profile=scoring_profile)
        and not is_empty_for_scoring(agent_value, scoring_profile=scoring_profile)
        and scoring_profile != "unknown_equivalent"
    ):
        return False

    return _candidate_matches_any_value(
        qid,
        agent_value,
        alternatives,
        scoring_profile=scoring_profile,
    )


def _candidate_matches_expected(
    qid: str,
    agent_value: Any,
    expected: Any,
    *,
    scoring_profile: str | None,
) -> bool:
    if isinstance(expected, bool):
        return _normalise_bool(agent_value) == expected
    return _candidate_matches_value(
        qid,
        agent_value,
        expected,
        scoring_profile=scoring_profile,
    )


def _evaluate_exact(
    qid: str, agent_value: Any, gt_entry: dict,
) -> FieldVerdict:
    expected = gt_entry.get("expected")
    alternatives = gt_entry.get("acceptable_alternatives", [])
    if not isinstance(alternatives, list):
        alternatives = []
    scoring_profile = infer_scoring_profile(qid, gt_entry, expected=expected)

    agent_norm = normalize_value_for_scoring(
        agent_value,
        scoring_profile=scoring_profile,
    )

    if (
        is_empty_for_scoring(expected, scoring_profile=scoring_profile)
        and is_empty_for_scoring(agent_value, scoring_profile=scoring_profile)
    ):
        return FieldVerdict(
            correctness="correct",
            source=_source_for_exact_match(
                agent_value,
                expected,
                gt_entry,
                scoring_profile=scoring_profile,
            ),
            reasoning=f"Both {qid} values are empty after scoring normalization.",
        )

    if _candidate_matches_expected(
        qid,
        agent_value,
        expected,
        scoring_profile=scoring_profile,
    ):
        reasoning = f"Exact match for {qid}."
        if _is_choice_field(qid):
            reasoning = (
                f"Choice alias match for {qid}: expected '{expected}', "
                f"got '{agent_value}'."
            )
        return FieldVerdict(
            correctness="correct",
            source=_source_for_exact_match(
                agent_value,
                expected,
                gt_entry,
                scoring_profile=scoring_profile,
            ),
            reasoning=reasoning,
        )

    if _candidate_matches_acceptable_alternative(
        qid,
        agent_value,
        expected,
        alternatives,
        scoring_profile=scoring_profile,
    ):
        return FieldVerdict(
            correctness="correct",
            source=_source_for_exact_match(
                agent_value,
                expected,
                gt_entry,
                scoring_profile=scoring_profile,
            ),
            reasoning=(
                f"Value '{agent_value}' is in acceptable_alternatives for {qid}. "
                f"Expected: '{expected}'."
            ),
        )

    if (
        is_empty_for_scoring(expected, scoring_profile=scoring_profile)
        and not is_empty_for_scoring(agent_value, scoring_profile=scoring_profile)
        and _candidate_matches_forbidden_commit(
            agent_value,
            gt_entry,
            scoring_profile=scoring_profile,
        )
    ):
        return FieldVerdict(
            correctness="incorrect",
            source="unsupported_inference",
            reasoning=(
                f"Candidate value '{agent_value}' matches a forbidden commit "
                f"for empty gold field {qid}."
            ),
        )

    if (
        is_empty_for_scoring(expected, scoring_profile=scoring_profile)
        and not is_empty_for_scoring(agent_value, scoring_profile=scoring_profile)
    ):
        return FieldVerdict(
            correctness="incorrect",
            source=_source_for_candidate_mismatch(
                agent_value,
                gt_entry,
                scoring_profile=scoring_profile,
            ),
            reasoning=(
                f"Agent set {qid} to {agent_value!r} when gold expects "
                "null/empty. Acceptable alternatives cannot override an "
                "empty gold value."
            ),
        )

    other_specified = _other_specification_reason(
        qid=qid,
        agent_value=agent_value,
        expected=expected,
        gt_entry=gt_entry,
    )
    if other_specified is not None:
        return other_specified

    # Handle boolean fields
    if isinstance(expected, bool):
        agent_bool = _normalise_bool(agent_value)
        if agent_bool == expected:
            return FieldVerdict(
                correctness="correct",
                source=_source_for_exact_match(
                    agent_value,
                    expected,
                    gt_entry,
                    scoring_profile=scoring_profile,
                ),
                reasoning=f"Exact match: {agent_value} == {expected}",
            )
        return FieldVerdict(
            correctness="incorrect",
            source=_source_for_candidate_mismatch(
                agent_value,
                gt_entry,
                scoring_profile=scoring_profile,
            ),
            reasoning=f"Expected {expected}, got {agent_value}",
        )

    # Check against expected + acceptable alternatives
    expected_norm = normalize_value_for_scoring(expected, scoring_profile=scoring_profile)
    all_accepted = [expected_norm] + [
        normalize_value_for_scoring(value, scoring_profile=scoring_profile)
        for value in alternatives
    ]

    if _is_choice_field(qid):
        if any(
            _choice_values_match(
                qid,
                agent_value,
                accepted_value,
                scoring_profile=scoring_profile,
            )
            for accepted_value in [expected] + list(alternatives)
        ):
            return FieldVerdict(
                correctness="correct",
                source=_source_for_exact_match(
                    agent_value,
                    expected,
                    gt_entry,
                    scoring_profile=scoring_profile,
                ),
                reasoning=(
                    f"Choice alias match for {qid}: expected '{expected}', "
                    f"got '{agent_value}'."
                ),
            )

    # Handle type coercion: if agent is string and expected is number, try numeric comparison
    if (isinstance(agent_norm, str) and isinstance(expected_norm, (int, float)) and
        not isinstance(expected_norm, bool)):
        try:
            agent_norm = float(agent_norm) if '.' in agent_norm else int(agent_norm)
            all_accepted = [float(v) if isinstance(v, str) and '.' in v else
                           (int(v) if isinstance(v, str) else v) for v in all_accepted]
        except (ValueError, TypeError):
            pass  # Keep original values if conversion fails

    if agent_norm in all_accepted:
        return FieldVerdict(
            correctness="correct",
            source=_source_for_exact_match(
                agent_value,
                expected,
                gt_entry,
                scoring_profile=scoring_profile,
            ),
            reasoning=f"Exact match for {qid}.",
        )

    # Empty agent value
    if is_empty_for_scoring(agent_value, scoring_profile=scoring_profile):
        return FieldVerdict(
            correctness="incorrect",
            source=None,
            reasoning=f"Agent left {qid} empty. Expected: '{expected}'.",
        )

    if _should_escalate_exact_mismatch(
        qid,
        agent_value,
        expected,
        gt_entry,
        scoring_profile=scoring_profile,
    ):
        return FieldVerdict(
            correctness="needs_semantic",
            source=_source_from_gt(gt_entry),
            reasoning=(
                f"Exact text/open field mismatch for {qid}: expected "
                f"'{expected}', got '{agent_value}'. Promoting to semantic "
                "evaluation after turn-level preserve checks."
            ),
        )

    if (
        _is_choice_field(qid)
        and not _is_bare_other_choice_value(agent_value, scoring_profile=scoring_profile)
        and not _is_bare_other_choice_value(expected, scoring_profile=scoring_profile)
    ):
        if (
            _has_negated_choice_option_reference(
                qid,
                agent_value,
                scoring_profile=scoring_profile,
            )
            or _has_negated_choice_option_reference(
                qid,
                expected,
                scoring_profile=scoring_profile,
            )
        ):
            return FieldVerdict(
                correctness="incorrect",
                source=_source_for_candidate_mismatch(
                    agent_value,
                    gt_entry,
                    scoring_profile=scoring_profile,
                ),
                reasoning=(
                    f"Choice alias blocked by negation for {qid}: "
                    f"expected '{expected}', got '{agent_value}'."
                ),
            )

        agent_option = _canonical_choice_option_key(
            qid,
            agent_value,
            scoring_profile=scoring_profile,
        )
        expected_option = _canonical_choice_option_key(
            qid,
            expected,
            scoring_profile=scoring_profile,
        )
        if agent_option is None or expected_option is None:
            return FieldVerdict(
                correctness="needs_semantic",
                source=_source_from_gt(gt_entry),
                reasoning=(
                    f"Choice field mismatch for {qid} includes a non-canonical "
                    f"label (expected '{expected}', got '{agent_value}'). "
                    "Promoting to semantic evaluation."
                ),
            )

    return FieldVerdict(
        correctness="incorrect",
        source=_source_for_candidate_mismatch(
            agent_value,
            gt_entry,
            scoring_profile=scoring_profile,
        ),
        reasoning=f"Expected '{expected}', got '{agent_value}'.",
    )


# ── Set match ────────────────────────────────────────────────────────────


def _evaluate_set_match(
    qid: str, agent_value: Any, gt_entry: dict,
) -> FieldVerdict:
    required_raw = _as_list(gt_entry.get("required", []))
    scoring_profile = infer_scoring_profile(qid, gt_entry, expected=required_raw)
    acceptable_raw = _as_list(gt_entry.get("acceptable_additions", []))
    unacceptable_raw = _as_list(gt_entry.get("unacceptable", []))
    debatable_raw = _as_list(gt_entry.get("debatable", []))

    required_empty = is_empty_for_scoring(
        required_raw,
        scoring_profile=scoring_profile,
    )
    agent_empty = is_empty_for_scoring(agent_value, scoring_profile=scoring_profile)

    if required_empty and agent_empty:
        return FieldVerdict(
            correctness="correct",
            source=_source_for_exact_match(
                agent_value,
                required_raw,
                gt_entry,
                scoring_profile=scoring_profile,
            ),
            reasoning=f"Both {qid} sets are empty after scoring normalization.",
        )

    if required_empty and not agent_empty:
        return FieldVerdict(
            correctness="incorrect",
            source=_source_for_candidate_mismatch(
                agent_value,
                gt_entry,
                scoring_profile=scoring_profile,
            ),
            reasoning=(
                f"Candidate supplies {agent_value!r} while gold requires an "
                f"empty set for {qid}. Acceptable additions and semantic "
                "judging cannot override an empty gold set."
            ),
        )

    # Parse agent value — may be list or comma/semicolon-separated string
    if (
        _has_other_specification(agent_value)
        or _has_other_specification(required_raw)
        or _has_other_specification(acceptable_raw)
        or _has_other_specification(debatable_raw)
        or _has_other_specification(unacceptable_raw)
    ):
        return FieldVerdict(
            correctness="needs_semantic",
            source=_source_from_gt(gt_entry),
            reasoning=(
                f"Field {qid} contains an explicit Other-specify list value. "
                "Promoting to semantic evaluation."
            ),
        )

    def _norm_item(value: Any) -> str:
        if _is_choice_field(qid):
            return _choice_item_norm(qid, value, scoring_profile=scoring_profile)
        return _normalise(value)

    required = {_norm_item(v) for v in required_raw}
    acceptable = {_norm_item(v) for v in acceptable_raw}
    unacceptable = {_norm_item(v) for v in unacceptable_raw}
    debatable = {_norm_item(v) for v in debatable_raw}

    if isinstance(agent_value, list):
        agent_items = {_norm_item(v) for v in agent_value}
    elif isinstance(agent_value, str):
        agent_items = {_norm_item(v) for v in re.split(r"[,;]+", agent_value) if v.strip()}
    else:
        agent_items = set()

    if not agent_items:
        return FieldVerdict(
            correctness="incorrect",
            source=None,
            reasoning=f"Agent left {qid} empty. Required: {gt_entry.get('required')}.",
        )

    # Check for unacceptable items
    bad = agent_items & unacceptable
    # Check required coverage
    missing_required = required - agent_items
    # Items that are fine (required + acceptable + debatable)
    allowed = required | acceptable | debatable

    has_unacceptable = len(bad) > 0
    has_all_required = len(missing_required) == 0
    has_extras_outside_allowed = bool(agent_items - allowed)

    if has_unacceptable:
        if has_all_required:
            # All required items present but also includes unacceptable —
            # partial credit rather than outright failure.
            return FieldVerdict(
                correctness="partially_correct",
                partial_reason="hallucination",
                source=_source_from_gt(gt_entry),
                reasoning=(
                    f"All required items present ({required}) but also "
                    f"contains unacceptable items: {bad}."
                ),
            )
        return FieldVerdict(
            correctness="incorrect",
            source=_source_from_gt(gt_entry),
            reasoning=(
                f"Contains unacceptable items: {bad}. "
                f"Missing required: {missing_required}."
            ),
        )

    if has_all_required and not has_extras_outside_allowed:
        return FieldVerdict(
            correctness="correct",
            source=_source_from_gt(gt_entry),
            reasoning=f"All required items present, no unacceptable additions.",
        )

    if has_all_required and has_extras_outside_allowed:
        extras = agent_items - allowed
        return FieldVerdict(
            correctness="partially_correct",
            partial_reason="hallucination",
            source=_source_from_gt(gt_entry),
            reasoning=(
                f"All required items present but has unknown extras: {extras}."
            ),
        )

    # Missing some required
    return FieldVerdict(
        correctness="partially_correct",
        partial_reason="omission",
        source=_source_from_gt(gt_entry),
        reasoning=f"Missing required items: {missing_required}.",
    )


# ── Public entry point ───────────────────────────────────────────────────


def evaluate_hard_fields(
    fields: list[tuple[str, Any, dict]],
    *,
    questionnaire_path: Path | None = None,
) -> dict[str, FieldVerdict]:
    """
    Evaluate a list of hard fields deterministically.

    Parameters
    ----------
    fields : list of (qid, agent_value, gt_entry) tuples
    questionnaire_path : explicit questionnaire to load type info from.

    Returns
    -------
    dict mapping qid -> FieldVerdict
    """
    if questionnaire_path is not None:
        _set_active_questionnaire_path(questionnaire_path)
    results: dict[str, FieldVerdict] = {}
    for qid, agent_value, gt_entry in fields:
        strategy = gt_entry.get("strategy", "exact")
        if strategy == "set_match":
            results[qid] = _evaluate_set_match(qid, agent_value, gt_entry)
        else:
            results[qid] = _evaluate_exact(qid, agent_value, gt_entry)
        if results[qid].decision_source == "initial_evaluation":
            results[qid].decision_source = "hard_eval"
    return results
