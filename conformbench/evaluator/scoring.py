from __future__ import annotations

import re
from typing import Any, Mapping


SCORING_PROFILE_COMMITMENT_CORE = "commitment_core"
SCORING_PROFILE_UNKNOWN_EQUIVALENT = "unknown_equivalent"
SCORING_PROFILE_AUXILIARY_UNSCORED_NOTE = "auxiliary_unscored_note"
SCORING_PROFILE_OPTIONAL_NOTE = "optional_note"
SCORING_PROFILES = {
    SCORING_PROFILE_COMMITMENT_CORE,
    SCORING_PROFILE_UNKNOWN_EQUIVALENT,
    SCORING_PROFILE_AUXILIARY_UNSCORED_NOTE,
}
LEGACY_SCORING_PROFILE_ALIASES = {
    SCORING_PROFILE_OPTIONAL_NOTE: SCORING_PROFILE_COMMITMENT_CORE,
}

DIFFICULTY_TIERS = {"anchor", "challenge", "hard"}

UNKNOWN_EQUIVALENT_CANONICAL = "__unknown__"

_WS = re.compile(r"\s+")
_RUNTIME_EMPTY_SENTINELS = {
    "",
    "open",
    "<open>",
    "__delete__",
}

_UNKNOWN_EQUIVALENT_TOKENS = {
    "",
    "n/a",
    "na",
    "not applicable",
    "not asked",
    "not available",
    "not known",
    "not obtained",
    "not provided",
    "not recorded",
    "not reported",
    "not specified",
    "null",
    "unknown",
    "unspecified",
}
_DIFFICULTY_TIER_ALIASES = {
    "anchor": "anchor",
    "challenge": "challenge",
    "easy": "anchor",
    "hard": "hard",
    "medium": "challenge",
    "pilot": "challenge",
}
_REPEAT_ROUTING_ALIASES = {
    "none": "none",
    "low": "unique_instance",
    "medium": "same_group_competition",
    "moderate": "same_group_competition",
    "high": "cross_group_or_multi_instance",
    "uniquely_keyed_instance": "unique_instance",
    "multiple_candidate_instances": "same_group_competition",
    "cross_group_routing": "cross_group_or_multi_instance",
}


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return _WS.sub(" ", str(value)).strip().lower()


def _dimension_value(
    dimensions: Mapping[str, Any],
    canonical_key: str,
    *legacy_keys: str,
) -> str:
    for key in (canonical_key, *legacy_keys):
        if key in dimensions:
            return _normalize_text(dimensions.get(key)).replace(" ", "_")
    return ""


def base_field_id(field_id: str) -> str:
    return field_id.rsplit(".", 1)[-1] if "." in field_id else field_id


def is_optional_note_field(field_id: str) -> bool:
    """Return whether a field is a legacy optional-note name.

    Record note fields are commitment-core by default. This helper is kept only
    for compatibility with older callers; it no longer drives scoring-profile
    inference.
    """
    base = base_field_id(field_id)
    return base.endswith("_notes") or base in {"free_text_notes", "additional_comments"}


def normalize_scoring_profile(profile: Any) -> str | None:
    token = _normalize_text(profile).replace(" ", "_")
    if not token:
        return None
    if token in SCORING_PROFILES:
        return token
    return LEGACY_SCORING_PROFILE_ALIASES.get(token)


def is_unknown_equivalent_literal(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return _normalize_text(value) in _UNKNOWN_EQUIVALENT_TOKENS
    return False


def has_unknown_equivalence_license(
    spec: Mapping[str, Any] | None,
    *,
    expected: Any = None,
) -> bool:
    metadata = dict(spec or {})
    if metadata.get("unknown_equivalent") is True:
        return True

    if metadata.get("scoring_profile") == SCORING_PROFILE_UNKNOWN_EQUIVALENT:
        return True

    alternatives = metadata.get("acceptable_alternatives")
    if alternatives is None:
        alternatives = metadata.get("alternatives")
    if not isinstance(alternatives, list):
        return False

    if not any(is_unknown_equivalent_literal(value) for value in alternatives):
        return False

    # The equivalence is only meaningful when the gold itself is epistemic /
    # empty-valued, not for ordinary committed values that merely happen to list
    # some unknown-like alternatives by mistake.
    return is_unknown_equivalent_literal(expected)


def infer_scoring_profile(
    field_id: str,
    spec: Mapping[str, Any] | None,
    *,
    expected: Any = None,
) -> str:
    explicit = normalize_scoring_profile((spec or {}).get("scoring_profile"))
    if explicit in SCORING_PROFILES:
        return explicit

    if has_unknown_equivalence_license(spec, expected=expected):
        return SCORING_PROFILE_UNKNOWN_EQUIVALENT

    return SCORING_PROFILE_COMMITMENT_CORE


def annotate_runtime_field_spec(
    field_id: str,
    spec: Mapping[str, Any] | None,
    *,
    expected: Any = None,
) -> dict[str, Any]:
    normalized = dict(spec or {})
    profile = infer_scoring_profile(field_id, normalized, expected=expected)
    normalized["scoring_profile"] = profile
    if profile == SCORING_PROFILE_UNKNOWN_EQUIVALENT:
        normalized.setdefault("unknown_equivalent", True)
    return normalized


def normalize_runtime_empty(value: Any) -> Any:
    if isinstance(value, str) and value.strip().lower() in _RUNTIME_EMPTY_SENTINELS:
        return None
    if isinstance(value, list):
        return [normalize_runtime_empty(item) for item in value]
    if isinstance(value, dict):
        return {key: normalize_runtime_empty(item) for key, item in value.items()}
    return value


def normalize_value_for_scoring(
    value: Any,
    *,
    scoring_profile: str | None = None,
) -> Any:
    normalized = normalize_runtime_empty(value)

    if scoring_profile == SCORING_PROFILE_UNKNOWN_EQUIVALENT and is_unknown_equivalent_literal(normalized):
        return UNKNOWN_EQUIVALENT_CANONICAL

    if isinstance(normalized, str):
        return _normalize_text(normalized)
    if isinstance(normalized, list):
        return [
            normalize_value_for_scoring(item, scoring_profile=scoring_profile)
            for item in normalized
        ]
    if isinstance(normalized, dict):
        return {
            key: normalize_value_for_scoring(item, scoring_profile=scoring_profile)
            for key, item in sorted(normalized.items())
        }
    return normalized


def values_equal_for_scoring(
    left: Any,
    right: Any,
    *,
    scoring_profile: str | None = None,
) -> bool:
    if (
        is_empty_for_scoring(left, scoring_profile=scoring_profile)
        and is_empty_for_scoring(right, scoring_profile=scoring_profile)
    ):
        return True
    return normalize_value_for_scoring(
        left,
        scoring_profile=scoring_profile,
    ) == normalize_value_for_scoring(right, scoring_profile=scoring_profile)


def is_empty_for_scoring(
    value: Any,
    *,
    scoring_profile: str | None = None,
) -> bool:
    normalized = normalize_value_for_scoring(value, scoring_profile=scoring_profile)
    if normalized in (None, "", UNKNOWN_EQUIVALENT_CANONICAL):
        return True
    if isinstance(normalized, list):
        return all(
            is_empty_for_scoring(item, scoring_profile=scoring_profile)
            for item in normalized
        )
    if isinstance(normalized, dict):
        return all(
            is_empty_for_scoring(item, scoring_profile=scoring_profile)
            for item in normalized.values()
        )
    return False


def canonicalize_difficulty_tier(value: Any) -> str | None:
    token = _normalize_text(value).replace(" ", "_")
    if not token:
        return None
    return _DIFFICULTY_TIER_ALIASES.get(token)


def repeat_instance_routing_pressure(
    dimensions: Mapping[str, Any] | None,
    *,
    failure_modes: list[str] | None = None,
) -> str:
    del failure_modes
    dims = dict(dimensions or {})
    explicit = _dimension_value(
        dims,
        "repeat_instance_routing_pressure",
        "repeat_group_identity_pressure",
    )
    explicit = _REPEAT_ROUTING_ALIASES.get(explicit, explicit)
    if explicit in {
        "none",
        "unique_instance",
        "same_group_competition",
        "cross_group_or_multi_instance",
    }:
        return explicit
    return "none"


def repeat_group_identity_pressure(
    dimensions: Mapping[str, Any] | None,
    *,
    failure_modes: list[str] | None = None,
) -> str:
    """Backward-compatible alias for the renamed repeat-routing stressor."""
    return repeat_instance_routing_pressure(
        dimensions,
        failure_modes=failure_modes,
    )


def is_hard_item(
    *,
    evidence: Mapping[str, Any] | None,
    dimensions: Mapping[str, Any] | None,
    failure_modes: list[str] | None = None,
) -> bool:
    del evidence, dimensions, failure_modes
    return False


def infer_difficulty_tier(
    explicit: Any,
    *,
    contrast_role: str | None = None,
    evidence: Mapping[str, Any] | None = None,
    dimensions: Mapping[str, Any] | None = None,
    failure_modes: list[str] | None = None,
) -> str:
    del evidence, dimensions, failure_modes
    canonical = canonicalize_difficulty_tier(explicit)
    if canonical:
        return canonical
    if contrast_role == "anchor":
        return "anchor"
    return "challenge"
