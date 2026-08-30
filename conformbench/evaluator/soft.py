"""
LLM-based evaluation for semantic-IU fields and schema alias checks.

Paper scoring uses item-level semantic information units. Legacy holistic
field-level semantic judging is intentionally not available.
"""

from __future__ import annotations

import json
import os
import hashlib
from typing import Any

from loguru import logger
from pydantic import BaseModel

from .provenance import build_trace_entry
from .scoring import (
    infer_scoring_profile,
    is_empty_for_scoring,
    normalize_value_for_scoring,
    values_equal_for_scoring,
)

from .llm_retry import invoke_with_retries
from .models import FieldVerdict, normalize_source_label
from .prompts import SYSTEM_PROMPT, build_system_prompt, build_user_prompt

# ── Structured output schema for the LLM ────────────────────────────────


class _VerdictItem(BaseModel):
    correctness: str
    partial_reason: str | None = None
    source: str
    reasoning: str


_TRACEABLE_EXPANSION_FALLBACK_FIELDS = frozenset({
    # Backward-compatible fallback for older artifacts without schema metadata.
    # New decisions should come from questionnaire field contracts.
    "description_of_accident",
    "property_damage_description",
    "witnesses_names_contacts",
})

_TRACEABLE_EXPANSION_CONTRACT_CUES = (
    "additional context",
    "any grounded context",
    "any other relevant",
    "brief note",
    "distance from",
    "grounded context",
    "grounded detail",
    "landmark",
    "location relative",
    "observation if available",
    "precise location",
    "precise positioning",
    "relative to the scene",
    "setting, sequence",
    "supporting detail",
    "vehicles involved",
    "what was damaged",
)


def _base_qid(qid: str) -> str:
    return qid.rsplit(".", 1)[-1] if "." in qid else qid


def _metadata_contract_text(metadata: dict[str, Any]) -> str:
    chunks: list[str] = []
    for key in ("question_text", "label", "gold_standard", "normalization_rules"):
        value = metadata.get(key)
        if value:
            chunks.append(str(value))
    information_units = metadata.get("information_units")
    if isinstance(information_units, list):
        for unit in information_units:
            if not isinstance(unit, dict):
                continue
            for key in ("id", "name", "description"):
                value = unit.get(key)
                if value:
                    chunks.append(str(value))
    return " ".join(chunks).lower()


def _list_metadata_values(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value in (None, ""):
        return []
    return [value]


def _metadata_value_matches(candidate: Any, reference: Any, *, scoring_profile: str | None) -> bool:
    return normalize_value_for_scoring(
        candidate,
        scoring_profile=scoring_profile,
    ) == normalize_value_for_scoring(
        reference,
        scoring_profile=scoring_profile,
    )


def _candidate_matches_field_acceptable_alternative(field: dict[str, Any]) -> Any | None:
    gt_entry = field.get("gt_entry") or {}
    alternatives = _list_metadata_values(gt_entry.get("acceptable_alternatives"))
    if not alternatives:
        return None
    candidate = field.get("candidate_value")
    if candidate is None or str(candidate).strip() in {"", "<open>"}:
        return None
    qid = str(field.get("qid") or "")
    expected = gt_entry.get("expected_summary") or gt_entry.get("expected")
    scoring_profile = infer_scoring_profile(qid, gt_entry, expected=expected)
    for alternative in alternatives:
        if _metadata_value_matches(candidate, alternative, scoring_profile=scoring_profile):
            return alternative
    return None


def _allows_traceable_expansion(
    *,
    qid: str,
    field: dict[str, Any],
) -> bool:
    metadata = field.get("field_metadata")
    if not isinstance(metadata, dict) or not metadata:
        return _base_qid(qid) in _TRACEABLE_EXPANSION_FALLBACK_FIELDS

    if metadata.get("traceable_expansion_ok") is True:
        return True

    field_type = str(metadata.get("type") or "")
    if field_type not in {"text", "multiline_text"}:
        return False

    contract_text = _metadata_contract_text(metadata)
    return any(cue in contract_text for cue in _TRACEABLE_EXPANSION_CONTRACT_CUES)


def _accept_traceable_field_expansion(
    *,
    qid: str,
    field: dict[str, Any],
    verdict: FieldVerdict,
) -> FieldVerdict:
    """Upgrade allowed descriptive expansions from partial to correct.

    The semantic judge can still be useful for detecting whether the extra
    detail is traceable. But for broad narrative fields, a traceable expansion
    should not be penalized merely because the provisional gold is terse.
    """
    if (
        verdict.correctness != "partially_correct"
        or verdict.partial_reason != "over_specified"
        or not _allows_traceable_expansion(qid=qid, field=field)
        or verdict.source in {"unsupported_inference", "fabricated"}
    ):
        return verdict

    candidate_value = field.get("candidate_value")
    if candidate_value is None or str(candidate_value).strip() in {"", "<open>"}:
        return verdict

    reasoning = (
        "Traceable descriptive expansion accepted because this field's schema "
        f"contract allows grounded detail. Judge rationale before upgrade: {verdict.reasoning}"
    )
    return FieldVerdict(
        correctness="correct",
        source=verdict.source,
        decision_source=verdict.decision_source,
        reasoning=reasoning,
    )


# ── LLM factory ──────────────────────────────────────────────────────────


def _get_judge_model(
    model_id: str | None = None,
    *,
    reasoning_effort: str | None = None,
    rotation_key: str | None = None,
    exclude_model: str | None = None,
):
    """
    Build a chat model for the judge.

    Uses the centralised LLM config (utils.llm) with JSON output mode.
    """
    from llm import get_judge_model_with_config

    if model_id:
        overrides: dict[str, Any] = {"model": model_id}
    else:
        overrides = {
            "rotation_key": rotation_key,
            "exclude_model": exclude_model,
        }
    if reasoning_effort:
        overrides["reasoning_effort"] = reasoning_effort
    return get_judge_model_with_config(**overrides)


def _response_text(response: Any) -> str:
    from llm import response_text

    return response_text(response)


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def _semantic_rotation_key(
    *,
    fields: list[dict[str, Any]],
    current_utterance: str,
    form_title: str | None,
    rotation_key_prefix: str | None,
) -> str:
    """Build a reproducible key for judge-model rotation.

    The key is not sent to the model.  It only makes model-pool selection stable
    for the same scored batch, even when evaluation is parallelised.
    """

    payload = {
        "prefix": rotation_key_prefix,
        "form_title": form_title,
        "current_utterance_sha256": hashlib.sha256(
            current_utterance.encode("utf-8")
        ).hexdigest(),
        "fields": [
            {
                "qid": field.get("qid"),
                "candidate_value": field.get("candidate_value"),
                "prior_value": field.get("prior_value"),
                "gold_value": (field.get("gt_entry") or {}).get("expected"),
                "claim_anchor": (field.get("gt_entry") or {}).get("claim_anchor"),
                "entity_role": (field.get("gt_entry") or {}).get("entity_role"),
            }
            for field in fields
        ],
    }
    return _stable_json(payload)


def _strip_json_fences(raw: str) -> str:
    if "```" not in raw:
        return raw
    if "```json" in raw:
        return raw.split("```json")[-1].split("```")[0]
    return raw.split("```")[1].split("```")[0]


def _has_item_level_semantic_ius(field: dict[str, Any]) -> bool:
    semantic_ius = field.get("semantic_ius")
    return isinstance(semantic_ius, list) and any(
        isinstance(iu, dict) for iu in semantic_ius
    )


def _is_empty_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() in {"", "<open>"}
    if isinstance(value, (list, dict, set, tuple)):
        return len(value) == 0
    return False


def _source_from_gt(gt_entry: dict[str, Any]) -> str | None:
    """Derive source label from ground-truth metadata for deterministic soft exits."""
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


def _source_for_soft_exact_match(
    *,
    candidate_value: Any,
    expected: Any,
    gt_entry: dict[str, Any],
    scoring_profile: str | None,
) -> str | None:
    evidence_source = gt_entry.get("evidence_source")
    has_recorded_evidence = bool(str(gt_entry.get("evidence") or "").strip())
    if (
        evidence_source in {None, "", "none"}
        and not has_recorded_evidence
        and is_empty_for_scoring(expected, scoring_profile=scoring_profile)
        and is_empty_for_scoring(candidate_value, scoring_profile=scoring_profile)
    ):
        return None
    return _source_from_gt(gt_entry)


def _source_for_soft_candidate_mismatch(gt_entry: dict[str, Any]) -> str | None:
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


def _expected_value_for_field(field: dict[str, Any]) -> Any:
    gt_entry = field.get("gt_entry") or {}
    return gt_entry.get("expected_summary") or gt_entry.get("expected")


def _candidate_preserved_prior_when_gold_changed(
    field: dict[str, Any],
    *,
    expected: Any,
    scoring_profile: str | None,
) -> bool:
    return values_equal_for_scoring(
        field.get("candidate_value"),
        field.get("prior_value"),
        scoring_profile=scoring_profile,
    ) and not values_equal_for_scoring(
        expected,
        field.get("prior_value"),
        scoring_profile=scoring_profile,
    )


def _preserved_prior_miss_verdict(
    field: dict[str, Any],
    *,
    qid: str,
    expected: Any,
) -> FieldVerdict:
    prior_value = field.get("prior_value")
    return FieldVerdict(
        correctness="incorrect",
        source="prior_state",
        decision_source="hard_preserve_miss",
        reasoning=(
            f"Candidate preserved prior value {prior_value!r} for {qid}, but "
            f"gold required resulting value {expected!r}. Scoring stopped "
            "before semantic or semantic-IU judging because this is a missed "
            "set/change/clear."
        ),
    )


def _deterministic_soft_verdict(field: dict[str, Any]) -> FieldVerdict | None:
    """Accept obvious semantic matches, and keep empty-gold commitments deterministic."""
    qid = str(field.get("qid") or "")
    gt_entry = field.get("gt_entry") or {}
    candidate_value = field.get("candidate_value")
    expected = _expected_value_for_field(field)
    scoring_profile = infer_scoring_profile(qid, gt_entry, expected=expected)

    if _candidate_preserved_prior_when_gold_changed(
        field,
        expected=expected,
        scoring_profile=scoring_profile,
    ):
        return _preserved_prior_miss_verdict(
            field,
            qid=qid,
            expected=expected,
        )

    if is_empty_for_scoring(expected, scoring_profile=scoring_profile):
        if is_empty_for_scoring(candidate_value, scoring_profile=scoring_profile):
            return FieldVerdict(
                correctness="correct",
                source=_source_for_soft_exact_match(
                    candidate_value=candidate_value,
                    expected=expected,
                    gt_entry=gt_entry,
                    scoring_profile=scoring_profile,
                ),
                decision_source="semantic_empty_gold_fast_path",
                reasoning=(
                    f"Semantic field {qid} has empty gold and empty candidate "
                    "after scoring normalization; judge was not called."
                ),
            )
        return FieldVerdict(
            correctness="incorrect",
            source=_source_for_soft_candidate_mismatch(gt_entry),
            decision_source="semantic_empty_gold_fast_path",
            reasoning=(
                f"Semantic field {qid} has empty gold but non-empty candidate "
                f"{candidate_value!r}; judge was not called."
            ),
        )

    if (
        not is_empty_for_scoring(expected, scoring_profile=scoring_profile)
        and is_empty_for_scoring(candidate_value, scoring_profile=scoring_profile)
    ):
        return FieldVerdict(
            correctness="incorrect",
            source=_source_for_soft_candidate_mismatch(gt_entry),
            decision_source="semantic_empty_candidate_fast_path",
            reasoning=(
                f"Semantic field {qid} has non-empty gold but empty candidate "
                "after scoring normalization; judge was not called."
            ),
        )

    if values_equal_for_scoring(
        candidate_value,
        expected,
        scoring_profile=scoring_profile,
    ):
        return FieldVerdict(
            correctness="correct",
            source=_source_for_soft_exact_match(
                candidate_value=candidate_value,
                expected=expected,
                gt_entry=gt_entry,
                scoring_profile=scoring_profile,
            ),
            decision_source="semantic_exact_fast_path",
            reasoning=(
                f"Semantic field {qid} accepted by normalized exact-match "
                "fast path; judge was not called."
            ),
        )

    return None


def _semantic_iu_system_prompt(form_title: str | None = None) -> str:
    form = form_title or "form"
    return f"""\
You are a constrained semantic-IU comparator for a {form}.

You will receive fields with item-level gold information units (IUs). For each
IU, decide only whether the candidate resulting value fully, partially, or does
not cover that IU; whether this turn newly updated the covered content into
alignment with the gold; whether it contradicts that IU; whether the update
made the field more accurate than the prior value; and quote the candidate span
that supports your answer.
Then decide whether the candidate contains any material unsupported extra
commitment for the field; whether it contains grounded but field-irrelevant
extra content; and whether the candidate update improved the field relative to
the prior even if no listed IU has full or partial coverage yet.

Do not assign holistic correct/partial/incorrect labels. The scorer derives
those deterministically from IU coverage, contradictions, unsupported extra
commitments, and field-irrelevant extra content.

The deterministic IU policy is:
- all required IUs are present in the final candidate value, no required IU is
  contradicted, and there is no material unsupported or field-irrelevant extra
  commitment -> correct;
- if all required IUs are present but grounded extra content is irrelevant to
  this field or belongs in a different field -> partially_correct for
  over_specified;
- otherwise, if at least one required IU was fully or partially updated in this
  turn to match the gold, or the candidate update made the field more accurate
  than the prior value, with no contradicted required IU -> partially_correct
  for omission;
- otherwise -> incorrect.
This means that merely preserving an IU that was already correct in the prior
value is not enough for partial credit when other required IUs are still
missing. For partial credit, at least one expected IU must receive new full or
partial matching coverage during this turn, or the update must make the field
closer to the gold than the prior state was. If all required IUs match in the
final candidate, the answer is fully correct even if some IUs were already
present in the prior value. When an answer contains both some updated required
IUs and unsupported or field-irrelevant extra commitments, report those extra
commitments so the scorer can mark the mixed partial case.

Improvement over prior means the candidate changed the field in a way that
removes an unsupported prior commitment, adds useful non-contradictory content,
narrows an ambiguous prior value toward the gold, or otherwise makes the field
more accurate than it would have been if preserved. Do not mark improvement for
mere preservation, formatting-only changes, or changes that add a contradiction.
Any contradicted required IU remains incorrect even when some other part
improved.

IU coverage values:
- "full": the candidate contains all required information for this IU.
- "partial": the candidate contains some meaningful information from this IU,
  but one or more required subcommitments inside the IU are missing.
- "none": the candidate does not contain useful matching information for this
  IU.

Set `present` to true only for full coverage. Use `coverage="partial"` for
single-IU partials such as a witness name without the required contact or
observation detail, or stage evidence without the required validation caveat.

Material unsupported commitments are concrete facts, entities, quantities,
temporal claims, clinical/legal conclusions, causal relations, or repeated-row
bindings that are not licensed by the current utterance, visible history, prior
state, questionnaire rules, or the provided gold/evidence.

Field-irrelevant extra content is licensed by the public state, but does not
belong in this field or makes the field less usable: copied utterance dumps,
audit-trail/meta phrasing, details better stored in a neighboring field, or
wordy context that buries the direct answer. Do not use this channel for
helpful grounded disambiguation that improves this field.

If a field or IU lists acceptable variants, accepted alternatives, allowed
normalizations, or boundary notes, treat those variants as satisfying the
corresponding IU. Do not mark a candidate as missing an IU or containing a
material unsupported extra commitment merely because it uses one of those
allowed formulations. These allowed values are examples of formulations that
are semantically close enough to count as completely correct: minor phrasing
deviations are fine when the main information is present and the candidate does
not add unsupported or field-irrelevant clutter.

Return JSON only:
{{
  "<field id>": {{
    "ius": {{
      "<iu_id>": {{
        "coverage": "full" | "partial" | "none",
        "present": true | false,
        "updated_to_match": true | false,
        "improved_over_prior": true | false,
        "contradicted": true | false,
        "candidate_span": "<short quote or empty string>",
        "reasoning": "<brief comparator note>"
      }}
    }},
    "improvement_over_prior": {{
      "present": true | false,
      "candidate_span": "<short quote or empty string>",
      "reasoning": "<brief note>"
    }},
    "unsupported_extra": {{
      "present": true | false,
      "candidate_span": "<short quote or empty string>",
      "reasoning": "<brief note>"
    }},
    "field_irrelevant_extra": {{
      "present": true | false,
      "candidate_span": "<short quote or empty string>",
      "reasoning": "<brief note>"
    }},
    "source": "extracted" | "likely_inferred" | "prior_state" | "unsupported_inference" | "fabricated" | null
  }}
}}
Nothing else.\
"""


def _format_iu_gold(ius: list[dict[str, Any]]) -> str:
    rows: list[dict[str, Any]] = []
    for iu in ius:
        if not isinstance(iu, dict):
            continue
        rows.append(
            {
                "iu_id": iu.get("iu_id"),
                "schema_iu_id": iu.get("schema_iu_id") or iu.get("schema_id"),
                "gold_content": iu.get("gold_content"),
                "accepted_variants": iu.get("accepted_variants")
                or iu.get("acceptable_alternatives")
                or [],
                "normalization_guidance": iu.get("normalization_guidance")
                or iu.get("notes")
                or "",
                "allowed_extra_commitments": iu.get("allowed_extra_commitments") or [],
                "required": iu.get("required", True),
                "evidence_spans": iu.get("evidence_spans") or [],
            }
        )
    return json.dumps(rows, indent=2, ensure_ascii=False)


def _build_semantic_iu_user_prompt(
    *,
    current_utterance: str,
    visible_history: list[dict[str, str]],
    prior_state: dict[str, Any],
    fields: list[dict[str, Any]],
) -> str:
    blocks = [
        "## Current Utterance\n",
        current_utterance,
        "\n---\n",
        "## Visible History\n",
        json.dumps(visible_history, indent=2, ensure_ascii=False),
        "\n---\n",
        "## Prior State\n",
        json.dumps(prior_state, indent=2, ensure_ascii=False),
        "\n---\n",
        f"## Semantic IU Fields ({len(fields)})\n",
    ]
    for field in fields:
        qid = str(field["qid"])
        gt_entry = dict(field.get("gt_entry") or {})
        expected = gt_entry.get("expected_summary") or gt_entry.get("expected")
        blocks.extend(
            [
                f"### Field: `{qid}`",
                f"**Item IU field_path:** {field.get('semantic_ius_field_path', qid)!r}",
                f"**Prior value:** {field.get('prior_value')!r}",
                f"**Candidate resulting value:** {field.get('candidate_value')!r}",
                f"**Gold resulting value:** {expected!r}",
            ]
        )
        claim_anchor = gt_entry.get("claim_anchor")
        entity_role = gt_entry.get("entity_role")
        if claim_anchor or entity_role:
            bits = []
            if claim_anchor:
                bits.append(f"claim_anchor={claim_anchor}")
            if entity_role:
                bits.append(f"entity_role={entity_role}")
            blocks.append("**Claim target:** " + ", ".join(bits))
        metadata = field.get("field_metadata")
        if isinstance(metadata, dict) and metadata:
            question_text = metadata.get("question_text") or metadata.get("label")
            if question_text:
                blocks.append("**Question text:** " + str(question_text))
            gold_standard = metadata.get("gold_standard")
            if gold_standard:
                blocks.append("**Questionnaire gold standard:** " + str(gold_standard))
            normalization_rules = metadata.get("normalization_rules")
            if normalization_rules:
                blocks.append("**Normalization rules:** " + str(normalization_rules))
            units = metadata.get("information_units")
            if units:
                blocks.append(
                    "**Schema IU rubric:**\n"
                    + json.dumps(units, indent=2, ensure_ascii=False)
                )
        evidence = gt_entry.get("evidence")
        if evidence:
            blocks.append(f'**Field evidence:** "{evidence}"')
        alternatives = gt_entry.get("acceptable_alternatives")
        if alternatives:
            blocks.append(
                "**Field acceptable alternatives:**\n"
                + json.dumps(alternatives, indent=2, ensure_ascii=False)
            )
        notes = gt_entry.get("notes")
        if notes:
            blocks.append("**Field scoring notes:** " + str(notes))
        blocks.append(
            "**Item-level gold IUs:**\n"
            + _format_iu_gold(field.get("semantic_ius") or [])
        )
        blocks.append("")
    blocks.append(
        "For every listed item-level IU, return coverage, present, "
        "updated_to_match, improved_over_prior, contradicted, span, and "
        "reasoning. Also return field-level improvement_over_prior and "
        "unsupported_extra and field_irrelevant_extra. Do not return a "
        "correctness label."
    )
    return "\n".join(blocks)


def _iu_results_by_id(entry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw = entry.get("ius") or entry.get("iu_results") or {}
    if isinstance(raw, dict):
        return {
            str(key): value
            for key, value in raw.items()
            if isinstance(value, dict)
        }
    if isinstance(raw, list):
        by_id: dict[str, dict[str, Any]] = {}
        for value in raw:
            if not isinstance(value, dict):
                continue
            iu_id = value.get("iu_id") or value.get("id")
            if iu_id:
                by_id[str(iu_id)] = value
        return by_id
    return {}


def _iu_coverage(iu_result: dict[str, Any]) -> str:
    raw = str(iu_result.get("coverage") or "").strip().lower()
    aliases = {
        "fully_present": "full",
        "full_coverage": "full",
        "present": "full",
        "partially_present": "partial",
        "partial_coverage": "partial",
        "missing": "none",
        "absent": "none",
        "not_present": "none",
    }
    coverage = aliases.get(raw, raw)
    if coverage in {"full", "partial", "none"}:
        return coverage
    if iu_result.get("present") is True:
        return "full"
    if iu_result.get("partial_present") is True:
        return "partial"
    return "none"


def _iu_updated_to_match(iu_result: dict[str, Any]) -> bool:
    return (
        iu_result.get("updated_to_match") is True
        or iu_result.get("updated_coverage") is True
        or iu_result.get("updated_to_partial_match") is True
    )


def _iu_improved_over_prior(iu_result: dict[str, Any]) -> bool:
    return (
        iu_result.get("improved_over_prior") is True
        or iu_result.get("improvement_over_prior") is True
        or iu_result.get("improved_relative_to_prior") is True
    )


def _entry_improvement_over_prior(entry: dict[str, Any]) -> tuple[bool, str]:
    raw = entry.get("improvement_over_prior")
    if raw is None:
        raw = entry.get("improved_over_prior")

    if isinstance(raw, dict):
        present = (
            raw.get("present") is True
            or raw.get("improved") is True
            or raw.get("value") is True
        )
        detail = str(
            raw.get("reasoning")
            or raw.get("candidate_span")
            or raw.get("span")
            or ""
        ).strip()
        return present, detail

    return raw is True, ""


def _effective_required_ius(field: dict[str, Any]) -> list[dict[str, Any]]:
    ius = [iu for iu in field.get("semantic_ius") or [] if isinstance(iu, dict)]
    required = [iu for iu in ius if iu.get("required", True) is True]
    if required:
        return required
    gt_entry = field.get("gt_entry") or {}
    expected = gt_entry.get("expected_summary") or gt_entry.get("expected")
    if not _is_empty_value(expected):
        return ius
    return required


def _semantic_iu_verdict_from_entry(
    *,
    qid: str,
    field: dict[str, Any],
    entry: dict[str, Any],
) -> FieldVerdict:
    matched_alternative = _candidate_matches_field_acceptable_alternative(field)
    if matched_alternative is not None:
        gt_entry = field.get("gt_entry") or {}
        source = "extracted" if gt_entry.get("present_in_utterance", True) else "likely_inferred"
        return FieldVerdict(
            correctness="correct",
            source=source,
            decision_source="semantic_iu_metadata",
            reasoning=(
                f"Candidate value matches an accepted semantic variant for {qid}: "
                f"{matched_alternative!r}. Required IUs are treated as covered by "
                "field-level accepted-variant metadata."
            ),
        )

    required_ius = _effective_required_ius(field)
    results_by_id = _iu_results_by_id(entry)

    full_coverage: list[str] = []
    partial_coverage: list[str] = []
    updated_to_match: list[str] = []
    updated_partial: list[str] = []
    improved_over_prior: list[str] = []
    missing: list[str] = []
    contradicted: list[str] = []
    for iu in required_ius:
        iu_id = str(iu.get("iu_id") or iu.get("schema_iu_id") or "")
        iu_result = results_by_id.get(iu_id, {})
        if iu_result.get("contradicted") is True:
            contradicted.append(iu_id)
            continue

        coverage = _iu_coverage(iu_result)
        updated = _iu_updated_to_match(iu_result)
        improved = _iu_improved_over_prior(iu_result)
        if improved:
            improved_over_prior.append(iu_id)
        if coverage == "full":
            full_coverage.append(iu_id)
            if updated:
                updated_to_match.append(iu_id)
        elif coverage == "partial":
            partial_coverage.append(iu_id)
            if updated:
                updated_partial.append(iu_id)
        else:
            missing.append(iu_id)

    unsupported_extra = entry.get("unsupported_extra")
    unsupported = (
        isinstance(unsupported_extra, dict)
        and unsupported_extra.get("present") is True
    )
    field_irrelevant_extra = entry.get("field_irrelevant_extra")
    field_irrelevant = (
        isinstance(field_irrelevant_extra, dict)
        and field_irrelevant_extra.get("present") is True
    )
    source = normalize_source_label(entry.get("source"))
    field_improved_over_prior, field_improvement_detail = (
        _entry_improvement_over_prior(entry)
    )
    improvement_signal = field_improved_over_prior or bool(improved_over_prior)

    total_required = len(required_ius)
    if contradicted:
        correctness = "incorrect"
        partial_reason = None
    elif (
        total_required
        and len(full_coverage) == total_required
        and not unsupported
        and not field_irrelevant
    ):
        correctness = "correct"
        partial_reason = None
    elif (
        total_required
        and len(full_coverage) == total_required
        and field_irrelevant
        and not unsupported
    ):
        correctness = "partially_correct"
        partial_reason = "over_specified"
    elif total_required and (updated_to_match or updated_partial or improvement_signal):
        correctness = "partially_correct"
        partial_reason = "mixed" if unsupported or field_irrelevant else "omission"
    elif unsupported:
        correctness = "incorrect"
        partial_reason = None
    elif total_required:
        correctness = "incorrect"
        partial_reason = None
    else:
        correctness = "correct"
        partial_reason = None

    if correctness == "partially_correct" and partial_reason is None:
        partial_reason = "ambiguous"

    reason_bits = [
        (
            f"Semantic IU comparator found {len(full_coverage)}/{total_required} "
            f"required IUs fully present for {qid}."
        )
    ]
    if partial_coverage:
        reason_bits.append(
            "Partial-coverage IUs: " + ", ".join(partial_coverage) + "."
        )
    if missing:
        reason_bits.append("Missing IUs: " + ", ".join(missing) + ".")
    if updated_to_match:
        reason_bits.append(
            "Updated-to-match IUs: " + ", ".join(updated_to_match) + "."
        )
    if updated_partial:
        reason_bits.append(
            "Updated partial-coverage IUs: " + ", ".join(updated_partial) + "."
        )
    if improved_over_prior:
        reason_bits.append(
            "Improved-over-prior IUs: " + ", ".join(improved_over_prior) + "."
        )
    if field_improved_over_prior:
        reason_bits.append(
            "Field improved over prior."
            + (f" {field_improvement_detail}" if field_improvement_detail else "")
        )
    if contradicted:
        reason_bits.append("Contradicted IUs: " + ", ".join(contradicted) + ".")
    if unsupported:
        detail = ""
        if isinstance(unsupported_extra, dict):
            detail = str(unsupported_extra.get("reasoning") or "").strip()
        reason_bits.append(
            "Material unsupported extra commitment present."
            + (f" {detail}" if detail else "")
        )
    if field_irrelevant:
        detail = ""
        if isinstance(field_irrelevant_extra, dict):
            detail = str(field_irrelevant_extra.get("reasoning") or "").strip()
        reason_bits.append(
            "Field-irrelevant extra content present."
            + (f" {detail}" if detail else "")
        )

    return FieldVerdict(
        correctness=correctness,
        source=source,
        decision_source="semantic_iu_judge",
        reasoning=" ".join(reason_bits),
        partial_reason=partial_reason,
    )


def _evaluate_semantic_iu_fields(
    fields: list[dict[str, Any]],
    *,
    current_utterance: str,
    prior_state: dict[str, Any],
    visible_history: list[dict[str, str]],
    model_id: str | None = None,
    reasoning_effort: str | None = None,
    exclude_model: str | None = None,
    rotation_key_prefix: str | None = None,
    form_title: str | None = None,
    trace_collector: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Evaluate a batch of soft fields via LLM.

    Parameters
    ----------
    fields : list of soft-field payloads — max ~8
    model_id : optional model-mapping-id override
    form_title : optional form context for prompt customization

    Returns
    -------
    dict mapping qid -> FieldVerdict
    """

    if not fields:
        return {}

    rotation_key = _semantic_rotation_key(
        fields=fields,
        current_utterance=current_utterance,
        form_title=form_title,
        rotation_key_prefix=rotation_key_prefix,
    )
    built = _get_judge_model(
        model_id,
        reasoning_effort=reasoning_effort,
        rotation_key=rotation_key,
        exclude_model=exclude_model,
    )
    if isinstance(built, tuple) and len(built) == 2:
        model, model_config = built
    else:
        model = built
        model_config = getattr(model, "_benchmark_model_config", None)

    user_prompt = _build_semantic_iu_user_prompt(
        current_utterance=current_utterance,
        visible_history=visible_history,
        prior_state=prior_state,
        fields=fields,
    )

    system_prompt = _semantic_iu_system_prompt(form_title)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    logger.debug(f"Calling IU judge LLM for {len(fields)} semantic IU fields")
    response = invoke_with_retries(
        model,
        messages,
        description=f"semantic IU judge ({len(fields)} fields)",
    )
    raw = _response_text(response)
    if trace_collector is not None and model_config is not None:
        trace_collector.append(
            build_trace_entry(
                phase="semantic_iu_judge",
                call_index=len(trace_collector) + 1,
                messages=messages,
                system_prompt=system_prompt,
                model_config=model_config,
                response=response,
                tool_calls=[],
                tool_results=[],
                extra={"field_qids": [str(field["qid"]) for field in fields]},
            )
        )

    raw = _strip_json_fences(raw)

    try:
        parsed: dict = json.loads(raw)
    except json.JSONDecodeError:
        logger.error(f"Judge returned invalid JSON:\n{raw[:500]}")
        # Fall back: mark all fields as needing manual review
        return {
            str(field["qid"]): FieldVerdict(
                correctness="incorrect",
                source=None,
                decision_source="semantic_iu_judge_error",
                reasoning="Semantic IU judge returned invalid JSON — manual review needed.",
            )
            for field in fields
        }

    results: dict[str, FieldVerdict] = {}
    field_by_qid = {str(field["qid"]): field for field in fields}
    for qid, field in field_by_qid.items():
        entry = parsed.get(qid, {})
        if not isinstance(entry, dict):
            entry = {}
        results[qid] = _semantic_iu_verdict_from_entry(
            qid=qid,
            field=field,
            entry=entry,
        )
    return results


def _evaluate_semantic_alias_fields(
    fields: list[dict[str, Any]],
    *,
    current_utterance: str,
    prior_state: dict[str, Any],
    visible_history: list[dict[str, str]],
    model_id: str | None = None,
    reasoning_effort: str | None = None,
    exclude_model: str | None = None,
    rotation_key_prefix: str | None = None,
    form_title: str | None = None,
    trace_collector: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if not fields:
        return {}
    invalid_fields = [
        str(field.get("qid") or "")
        for field in fields
        if field.get("semantic_equivalence_check") is not True
    ]
    if invalid_fields:
        raise ValueError(
            "Internal error: semantic alias judge received non-alias fields "
            f"without semantic IUs: {', '.join(invalid_fields)}"
        )

    rotation_key = _semantic_rotation_key(
        fields=fields,
        current_utterance=current_utterance,
        form_title=form_title,
        rotation_key_prefix=rotation_key_prefix,
    )
    built = _get_judge_model(
        model_id,
        reasoning_effort=reasoning_effort,
        rotation_key=rotation_key,
        exclude_model=exclude_model,
    )
    if isinstance(built, tuple) and len(built) == 2:
        model, model_config = built
    else:
        model = built
        model_config = getattr(model, "_benchmark_model_config", None)

    user_prompt = build_user_prompt(
        current_utterance=current_utterance,
        visible_history=visible_history,
        prior_state=prior_state,
        fields=fields,
    )

    system_prompt = build_system_prompt(form_title) if form_title else SYSTEM_PROMPT

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    logger.debug(f"Calling schema alias judge LLM for {len(fields)} fields")
    response = invoke_with_retries(
        model,
        messages,
        description=f"schema alias judge ({len(fields)} fields)",
    )
    raw = _response_text(response)
    if trace_collector is not None and model_config is not None:
        trace_collector.append(
            build_trace_entry(
                phase="semantic_alias_judge",
                call_index=len(trace_collector) + 1,
                messages=messages,
                system_prompt=system_prompt,
                model_config=model_config,
                response=response,
                tool_calls=[],
                tool_results=[],
                extra={"field_qids": [str(field["qid"]) for field in fields]},
            )
        )

    raw = _strip_json_fences(raw)

    try:
        parsed: dict = json.loads(raw)
    except json.JSONDecodeError:
        logger.error(f"Judge returned invalid JSON:\n{raw[:500]}")
        return {
            str(field["qid"]): FieldVerdict(
                correctness="incorrect",
                source=None,
                decision_source="semantic_alias_judge_error",
                reasoning=(
                    "Schema alias judge returned invalid JSON — manual review needed."
                ),
            )
            for field in fields
        }

    results: dict[str, FieldVerdict] = {}
    field_qids = {str(field["qid"]) for field in fields}
    field_by_qid = {str(field["qid"]): field for field in fields}

    for qid in field_qids:
        entry = parsed.get(qid, {})
        if not isinstance(entry, dict):
            entry = {}

        correctness = entry.get("correctness", "incorrect")
        if correctness not in ("correct", "partially_correct", "incorrect"):
            correctness = "incorrect"

        source = normalize_source_label(entry.get("source"))

        reasoning = entry.get("reasoning", "")

        # Extract partial_reason for partially_correct verdicts
        partial_reason = None
        if correctness == "partially_correct":
            pr = entry.get("partial_reason")
            valid_reasons = (
                "hallucination", "over_specified", "omission",
                "wrong_choice", "mixed", "ambiguous",
            )
            partial_reason = pr if pr in valid_reasons else "ambiguous"

        verdict = FieldVerdict(
            correctness=correctness,
            source=source,
            decision_source="semantic_alias_candidate_judge",
            reasoning=reasoning,
            partial_reason=partial_reason,
        )
        results[qid] = _accept_traceable_field_expansion(
            qid=qid,
            field=field_by_qid.get(qid, {}),
            verdict=verdict,
        )

    return results


def _evaluate_exact_mismatch_fields(
    fields: list[dict[str, Any]],
    *,
    current_utterance: str,
    prior_state: dict[str, Any],
    visible_history: list[dict[str, str]],
    model_id: str | None = None,
    reasoning_effort: str | None = None,
    exclude_model: str | None = None,
    rotation_key_prefix: str | None = None,
    form_title: str | None = None,
    trace_collector: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if not fields:
        return {}
    invalid_fields = [
        str(field.get("qid") or "")
        for field in fields
        if field.get("semantic_exact_mismatch_check") is not True
    ]
    if invalid_fields:
        raise ValueError(
            "Internal error: exact-mismatch judge received fields without "
            f"semantic_exact_mismatch_check: {', '.join(invalid_fields)}"
        )

    rotation_key = _semantic_rotation_key(
        fields=fields,
        current_utterance=current_utterance,
        form_title=form_title,
        rotation_key_prefix=rotation_key_prefix,
    )
    built = _get_judge_model(
        model_id,
        reasoning_effort=reasoning_effort,
        rotation_key=rotation_key,
        exclude_model=exclude_model,
    )
    if isinstance(built, tuple) and len(built) == 2:
        model, model_config = built
    else:
        model = built
        model_config = getattr(model, "_benchmark_model_config", None)

    user_prompt = build_user_prompt(
        current_utterance=current_utterance,
        visible_history=visible_history,
        prior_state=prior_state,
        fields=fields,
    )

    system_prompt = build_system_prompt(form_title) if form_title else SYSTEM_PROMPT

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    logger.debug(f"Calling exact-mismatch semantic judge for {len(fields)} fields")
    response = invoke_with_retries(
        model,
        messages,
        description=f"exact-mismatch semantic judge ({len(fields)} fields)",
    )
    raw = _response_text(response)
    if trace_collector is not None and model_config is not None:
        trace_collector.append(
            build_trace_entry(
                phase="semantic_exact_mismatch_judge",
                call_index=len(trace_collector) + 1,
                messages=messages,
                system_prompt=system_prompt,
                model_config=model_config,
                response=response,
                tool_calls=[],
                tool_results=[],
                extra={"field_qids": [str(field["qid"]) for field in fields]},
            )
        )

    raw = _strip_json_fences(raw)

    try:
        parsed: dict = json.loads(raw)
    except json.JSONDecodeError:
        logger.error(f"Judge returned invalid JSON:\n{raw[:500]}")
        return {
            str(field["qid"]): FieldVerdict(
                correctness="incorrect",
                source=None,
                decision_source="semantic_exact_mismatch_judge_error",
                reasoning=(
                    "Exact-mismatch semantic judge returned invalid JSON — "
                    "manual review needed."
                ),
            )
            for field in fields
        }

    results: dict[str, FieldVerdict] = {}
    field_qids = {str(field["qid"]) for field in fields}
    field_by_qid = {str(field["qid"]): field for field in fields}

    for qid in field_qids:
        entry = parsed.get(qid, {})
        if not isinstance(entry, dict):
            entry = {}

        correctness = entry.get("correctness", "incorrect")
        if correctness not in ("correct", "partially_correct", "incorrect"):
            correctness = "incorrect"

        source = normalize_source_label(entry.get("source"))
        reasoning = entry.get("reasoning", "")

        partial_reason = None
        if correctness == "partially_correct":
            pr = entry.get("partial_reason")
            valid_reasons = (
                "hallucination", "over_specified", "omission",
                "wrong_choice", "mixed", "ambiguous",
            )
            partial_reason = pr if pr in valid_reasons else "ambiguous"

        verdict = FieldVerdict(
            correctness=correctness,
            source=source,
            decision_source="semantic_exact_mismatch_judge",
            reasoning=reasoning,
            partial_reason=partial_reason,
        )
        results[qid] = _accept_traceable_field_expansion(
            qid=qid,
            field=field_by_qid.get(qid, {}),
            verdict=verdict,
        )

    return results


# ── Public entry point ───────────────────────────────────────────────────


def evaluate_soft_fields(
    fields: list[dict[str, Any]],
    *,
    current_utterance: str,
    prior_state: dict[str, Any],
    visible_history: list[dict[str, str]],
    model_id: str | None = None,
    reasoning_effort: str | None = None,
    exclude_model: str | None = None,
    rotation_key_prefix: str | None = None,
    form_title: str | None = None,
    trace_collector: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Evaluate soft fields.

    Fields with item-level ``semantic_ius`` use constrained IU comparison and a
    deterministic verdict derivation. Fields without semantic IUs must either
    be accepted by the deterministic fast path, be explicit schema alias
    equivalence checks, or be exact-mismatch escalations from the hard scorer.
    """
    if not fields:
        return {}

    results: dict[str, FieldVerdict] = {}
    fields_requiring_judge: list[dict[str, Any]] = []
    for field in fields:
        qid = str(field.get("qid") or "")
        deterministic_verdict = _deterministic_soft_verdict(field)
        if deterministic_verdict is not None:
            results[qid] = deterministic_verdict
        else:
            fields_requiring_judge.append(field)

    if not fields_requiring_judge:
        return results

    alias_fields = [
        field
        for field in fields_requiring_judge
        if field.get("semantic_equivalence_check") is True
    ]
    exact_mismatch_fields = [
        field
        for field in fields_requiring_judge
        if field.get("semantic_exact_mismatch_check") is True
    ]
    iu_fields = [
        field
        for field in fields_requiring_judge
        if _has_item_level_semantic_ius(field)
        and field.get("semantic_equivalence_check") is not True
        and field.get("semantic_exact_mismatch_check") is not True
    ]
    unsupported_fields = [
        str(field.get("qid") or "")
        for field in fields_requiring_judge
        if field.get("semantic_equivalence_check") is not True
        and field.get("semantic_exact_mismatch_check") is not True
        and not _has_item_level_semantic_ius(field)
    ]

    if unsupported_fields:
        raise ValueError(
            "Semantic fields without item-level semantic_ius cannot be scored. "
            "Add item-level semantic IUs, convert the field to deterministic "
            f"scoring, or exclude it explicitly. Fields: {', '.join(unsupported_fields)}"
        )

    if alias_fields:
        results.update(
            _evaluate_semantic_alias_fields(
                alias_fields,
                current_utterance=current_utterance,
                prior_state=prior_state,
                visible_history=visible_history,
                model_id=model_id,
                reasoning_effort=reasoning_effort,
                exclude_model=exclude_model,
                rotation_key_prefix=rotation_key_prefix,
                form_title=form_title,
                trace_collector=trace_collector,
            )
        )
    if exact_mismatch_fields:
        results.update(
            _evaluate_exact_mismatch_fields(
                exact_mismatch_fields,
                current_utterance=current_utterance,
                prior_state=prior_state,
                visible_history=visible_history,
                model_id=model_id,
                reasoning_effort=reasoning_effort,
                exclude_model=exclude_model,
                rotation_key_prefix=rotation_key_prefix,
                form_title=form_title,
                trace_collector=trace_collector,
            )
        )
    if iu_fields:
        results.update(
            _evaluate_semantic_iu_fields(
                iu_fields,
                current_utterance=current_utterance,
                prior_state=prior_state,
                visible_history=visible_history,
                model_id=model_id,
                reasoning_effort=reasoning_effort,
                exclude_model=exclude_model,
                rotation_key_prefix=rotation_key_prefix,
                form_title=form_title,
                trace_collector=trace_collector,
            )
        )
    return results
