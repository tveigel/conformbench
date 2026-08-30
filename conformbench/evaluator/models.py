"""
Pydantic models for the judge framework.

Separated from judge.py to avoid circular imports — hard.py and soft.py
both need FieldVerdict, and judge.py imports hard/soft.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, computed_field, field_validator, model_validator


# Reason codes that explain *why* a field is partially_correct.
# Used to distinguish fundamentally different failure modes that the
# coarse "partially_correct" bucket otherwise hides.
#
#   hallucination   — agent added fabricated details not in the public state
#   over_specified  — agent added correct, public-state-grounded detail that
#                     is unnecessary (but not fabricated) for this field
#   omission        — agent missed one or more required components
#   wrong_choice    — agent chose the wrong item from a constrained set
#   mixed           — combination of issues (e.g. omission + hallucination)
#   ambiguous       — GT itself flags the answer as defensible but not ideal
PartialReason = Literal[
    "hallucination",
    "over_specified",
    "omission",
    "wrong_choice",
    "mixed",
    "ambiguous",
]

CorrectnessLabel = Literal["correct", "partially_correct", "incorrect", "needs_semantic"]

SourceLabel = Literal[
    "extracted",
    "likely_inferred",
    "prior_state",
    "unsupported_inference",
    "fabricated",
]

SOURCE_LABELS: set[str] = {
    "extracted",
    "likely_inferred",
    "prior_state",
    "unsupported_inference",
    "fabricated",
}

LEGACY_SOURCE_ALIASES = {
    "made_up": "fabricated",
    "made-up": "fabricated",
}


def normalize_source_label(source: object) -> SourceLabel | None:
    """Normalize current and legacy source labels.

    ``made_up`` used to mix unsupported inferences, stale prior carry-over, and
    true fabrication. New artifacts use ``fabricated`` only for values with no
    basis in visible inputs; ``made_up`` remains a read-time compatibility
    alias.
    """
    if source is None:
        return None
    token = str(source).strip()
    if not token:
        return None
    token = LEGACY_SOURCE_ALIASES.get(token, token)
    if token in SOURCE_LABELS:
        return token  # type: ignore[return-value]
    return None


class FieldVerdict(BaseModel):
    """Evaluation result for a single field."""
    correctness: CorrectnessLabel
    # Legacy name kept for consumers. It mirrors support_source after any
    # post-processing so a final-correct row cannot still display an old error
    # taxonomy label as its current source.
    source: SourceLabel | None = None
    support_source: SourceLabel | None = None
    original_correctness: CorrectnessLabel | None = None
    original_source: SourceLabel | None = None
    original_partial_reason: PartialReason | None = None
    decision_source: str | None = None
    postprocess_reason: str | None = None
    reasoning: str = ""
    partial_reason: PartialReason | None = None

    @field_validator("source", "support_source", "original_source", mode="before")
    @classmethod
    def _normalize_source(cls, value: object) -> SourceLabel | None:
        return normalize_source_label(value)

    @model_validator(mode="after")
    def _populate_decision_labels(self) -> "FieldVerdict":
        if self.support_source is None:
            self.support_source = self.source
        self.source = self.support_source
        if self.original_correctness is None:
            self.original_correctness = self.correctness
        if self.original_source is None:
            self.original_source = self.support_source
        if self.original_partial_reason is None:
            self.original_partial_reason = self.partial_reason
        if self.decision_source is None:
            self.decision_source = "initial_evaluation"
        return self

    @computed_field
    @property
    def final_correctness(self) -> CorrectnessLabel:
        return self.correctness

    def set_support_source(self, source: object) -> None:
        """Set current support provenance and keep the legacy alias in sync."""
        normalised = normalize_source_label(source)
        self.support_source = normalised
        self.source = normalised


class AlignmentEntry(BaseModel):
    """Records how one ground-truth instance was matched (or not)."""
    group: str = ""
    gt_index: int
    agent_index: int | None = None
    status: Literal["matched", "missed", "hallucinated"]
    matched_on: str = ""


class EvaluationResult(BaseModel):
    """Complete evaluation output — the single downstream artifact."""
    scenario_id: str = ""
    field_results: dict[str, FieldVerdict] = Field(default_factory=dict)
    alignment_log: list[AlignmentEntry] = Field(default_factory=list)
    unmatched: list[dict[str, Any]] = Field(default_factory=list)
    summary: dict[str, int] = Field(default_factory=dict)
