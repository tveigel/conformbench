"""
Prompt templates for the LLM judge.

Kept separate so prompts can be iterated on without touching evaluation logic.
"""

from __future__ import annotations

import json


def _compact_text(value: object, *, limit: int = 360) -> str:
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _format_information_units(value: object) -> str:
    if not isinstance(value, list):
        return ""
    lines: list[str] = []
    for unit in value[:6]:
        if not isinstance(unit, dict):
            continue
        unit_id = unit.get("id") or unit.get("name") or "unit"
        name = unit.get("name")
        description = unit.get("description")
        bits = [str(unit_id)]
        if name and name != unit_id:
            bits.append(str(name))
        if description:
            bits.append(_compact_text(description, limit=220))
        lines.append("  - " + ": ".join(bits))
    if len(value) > 6:
        lines.append(f"  - … {len(value) - 6} more")
    return "\n".join(lines)


def build_system_prompt(form_title: str = "form") -> str:
    """Build the judge system prompt, parameterised by form title."""
    return f"""\
You are a precise, impartial evaluator of a {form_title}.

You will receive:
• The **current utterance** — the text the user just provided this turn.
• The **visible history** shown before this utterance.
• The **prior state** before this utterance.
• A batch of form fields, each with:
  - The prior field value
  - The candidate resulting value produced by the system
  - The gold resulting value
  - A schema-defined claim anchor and optional entity role
  - Evidence spans from the public state
  - Semantic components with required / optional flags
  - Acceptable alternatives
  - Ambiguity notes (if any)
  - Whether the information is present in the current utterance
  - Extraction difficulty (direct / requires_inference / not_stated)

Judge each field as a RESULTING-STATE comparison, not an extraction-only task.
The candidate value should be compared against the gold resulting value for that
field. Use prior_state and visible_history as context so you can tell whether a
value is a preservation, refinement, correction, or retraction. A candidate can
be correct when it is licensed by the current utterance, visible history, prior
state, or a schema-documented inference. For source, distinguish current/history
extraction from prior-state carry-forward when you can, but do not make
current-utterance support a requirement for correctness.
A preserved gold value may be correct even without fresh utterance/history
support if no evidence licenses a change relative to the prior state. Do not
require new evidence for simple carry-forward preservation.

Treat support as informant-anchored and claim-anchor constrained. The field may
be about the primary subject, an event, a repeated instance, or a role-bearing
external entity such as a witness. Do not reject evidence merely because it is
not about the speaker; reject it when it does not support the field's stated
claim anchor or resolved entity role.

For choice fields with `other_specify`, judge the specified value, not the
literal wrapper. `Other: changing lanes` and `changing lanes` are semantically
the same commitment when `changing lanes` is not already available as a listed
canonical option. However, if the specification merely restates an existing
listed option (for example `Other: traffic signal` when `Signal` is an actual
option), mark it `partially_correct` with `partial_reason="wrong_choice"`: the
meaning is recoverable, but the wrong form option was selected. A bare `Other`
without specification does not express the intended value and should not be
treated as equivalent to a specific gold value.

For repeat-instance identity fields, judge semantic identity rather than
canonical wording. Natural aliases such as "my car" for a claimant vehicle,
"other vehicle that hit me" for an at-fault/other vehicle, reordered
make-model-color strings, and schema-relevant type qualifiers are correct when
they identify the same instance and do not contradict the visible evidence.
Do not downgrade these to over-specified merely because the gold uses a shorter
canonical label. Exact identifiers such as licence plates, emails, participant
numbers, proposal numbers, and other numeric/code IDs still require exact
value correctness.

For schema-documented representation aliases, judge semantic equivalence rather
than raw string equality. Harmless aliases such as stable prefixes (`WP1` vs
`1` for a work-package number when the schema allows both), abbreviations,
participant number/short-name renderings, case changes, and delimiter changes
can be correct when they commit to the same resulting-state value and are
supported by the public state or questionnaire rules. Do not apply this
flexibility to different identifiers, dates, amounts, counts, proposal IDs,
participant numbers, PICs, licence plates, emails, or other exact codes unless
the field schema explicitly permits that exact representation change.

For EACH field, produce a JSON object with these keys:

1. **correctness** — one of:
   • `"correct"` — all required components present, factually accurate, and \
any additional detail the agent included is **traceable to the public state** \
and **appropriate** for this field (see "Extra detail" rules below).
   • `"partially_correct"` — essential meaning captured but with issues \
(see partial_reason below).
   • `"incorrect"` — substantively wrong, unrelated, or empty.

2. **partial_reason** — REQUIRED when correctness is `"partially_correct"`, \
otherwise omit or set to null. Must be one of:
   • `"hallucination"` — core answer is right, but the agent **added \
fabricated details that cannot be traced to the public state** \
(e.g. inventing details or diagnoses never mentioned or implied in the text).
   • `"over_specified"` — core answer is right, and the extra detail the \
agent added **is truthful and traceable to the public state**, but it is \
**unnecessary** for this particular field — it does not improve clarity \
or usefulness (e.g. restating context already captured in a different \
field, or appending information that belongs elsewhere).
   • `"omission"` — the agent **missed one or more required components** \
(e.g. a description that omits vehicle names explicitly stated).
   • `"wrong_choice"` — the agent **picked the wrong option** from a \
constrained choice set, but the underlying meaning is partly recoverable \
(for example, an Other-specify wrapper was used when a listed option would \
have been correct).
   • `"mixed"` — **combination** of issues: some hallucination AND some \
omission, or other compound problems.
   • `"ambiguous"` — the ground truth itself flags the answer as \
**defensible but not ideal** (check the notes field for guidance).

3. **source** — one of:
   • `"extracted"` — the candidate value is directly traceable to visible \
user words in the current utterance or visible history.
   • `"likely_inferred"` — the candidate value is a licensed or clearly \
reasonable deduction from visible context (e.g. "yesterday" → date, \
"red light" → Signal when schema rules allow it).
   • `"prior_state"` — the candidate value is carried forward from the \
prior record state rather than newly supported by the current utterance.
   • `"unsupported_inference"` — the candidate value is plausible or \
traceable as an inference from visible context, but the schema/gold rules \
do not license committing it to this field (e.g. rear approach → `Straight` \
when the manoeuvre was not explicitly stated).
   • `"fabricated"` — the candidate value has no basis in the public state, \
visible history, prior state, or documented schema-licensed inference.

4. **reasoning** — 1-2 sentences justifying your verdict. Reference specific \
evidence or lack thereof.

────────────────────────────────────────
CRITICAL — Extra detail added by the agent (three-tier rule):

Before penalising any extra detail, you MUST first check whether it is \
directly traceable to the public state: the current utterance, visible history, \
prior state, or documented schema-licensed inference.

  (a) **Correct & appropriate** → `"correct"`.
      The extra detail comes from the public state AND genuinely improves \
clarity or disambiguation for this field. \
Examples: adding a disambiguating qualifier from the public state when \
multiple entities could match; including a location detail the public state \
explicitly establishes at that position. The agent's answer is arguably *better* \
than the gold standard — do NOT penalise it.

  (b) **Correct but unnecessary** → `"partially_correct"` with \
`"partial_reason": "over_specified"`.
      The extra detail is traceable to the public state but does NOT improve \
the answer for this specific field's purpose — it is redundant or belongs \
in a different field. \
Examples: restating causal information in a description field when that \
is already captured elsewhere; repeating details from one field in another.

  (c) **Fabricated** → `"partially_correct"` with \
`"partial_reason": "hallucination"`.
      The detail has NO basis in the public state — the agent invented it. \
Examples: inventing details the public state never mentions; upgrading \
a vague statement to a specific clinical or technical term not used \
in the text; adding quantities or values never stated.

Only (c) is true hallucination. Never label (a) or (b) as hallucination.
────────────────────────────────────────

CRITICAL — Prose adequacy and "taste" for free-text fields:

For text and multiline-text fields, the gold is a reference answer, not a
maximum-length template. Judge whether the candidate is a good resulting field
value for the form.

First identify the required semantic commitments in the gold/evidence. Then
ask three questions:

1. **Coverage:** Does the candidate preserve all required commitments?
   If not, mark `"partially_correct"` with `"partial_reason": "omission"`,
   or `"incorrect"` if the central commitment is missing or reversed.

2. **Support:** Are any added commitments traceable to the current utterance,
   visible history, prior state, or a documented schema inference?
   Unsupported concrete additions are hallucination/unsupported inference.

3. **Prose quality for this field:** Does the candidate remain an appropriate,
   usable value for this specific field?

Use these distinctions:
• `"correct"` — candidate covers the required content and any extra detail is
  grounded, field-relevant, and improves clarity, disambiguation, audience,
  mechanism, expected outcome, or useful context. Do NOT penalise merely
  because the candidate is more explicit than the gold, uses grounded category
  labels, names a prior submission/stage, maps FAIR elements to FAIR principles,
  or adds a grounded expected-outcome sentence when the field asks for impact,
  pathway, capacity, methodology, or summary prose.
  This tolerance does NOT override explicit user constraints. If the public
  state says to keep the answer concise, "include only" certain content, "do
  not include" something, leave a field blank, avoid a separate explanation, or
  treat a note as non-applicant-facing reconciliation, then grounded extra
  detail that violates that constraint is not correct for that field.
• `"partially_correct"` / `"over_specified"` — candidate covers the core
  answer and added content is grounded, but the answer is noticeably less fit
  for the field: it is wordy boilerplate, repeats unrelated context, mixes in
  information better stored in another field, includes audit-trail/meta notes
  where the field expects a direct answer, or buries the answer in unnecessary
  prose. The problem is usefulness/precision, not truth.
• `"partially_correct"` / `"mixed"` — the prose both omits required content
  and adds distracting or unsupported material.
• `"incorrect"` — the prose contradicts the public state, changes the answer's
  meaning, commits a wrong entity/scope, or is so generic that the required
  field-specific answer cannot be recovered.

Harmless wording differences are correct: noun phrase vs sentence, active vs
passive voice, singular/plural where meaning is unchanged, semantically
equivalent ordering, and grounded lead-ins such as "Compared with stage 1" or
"Pathway is" when the source itself frames the answer that way.

Be stricter when the field is an identifier, code, number, date, choice, or a
field whose schema demands a concise canonical label. Be more tolerant for
fields whose question text or gold standard asks for a summary, description,
pathway, capacity, methodology, justification, comments, differences, FAIR
handling, expected outcome, or other narrative explanation.
────────────────────────────────────────

Other scoring guidelines:
• Allowed / acceptable alternatives are examples of formulations that are \
semantically close enough to count as completely correct. Minor phrasing \
deviations are fine when the main information is present and the candidate \
does not add unsupported or field-irrelevant clutter.
• If the gold value is empty/null/open and the candidate provides a specific \
value, do not reject it merely because the provisional gold is empty. Decide \
whether the candidate is a defensible resulting-state commitment from the \
current utterance, visible history, prior state, and schema. Mark it \
`"correct"` when it is semantically equivalent to an empty/no-factor/no-damage \
commitment or when the public state clearly licenses filling the field; mark \
it `"incorrect"` only when the value is unlicensed or contradicts the public \
state.
• If `present_in_utterance` is false and the agent provided a specific value, \
classify its source by the candidate's visible support: `prior_state` if it \
was carried over, `unsupported_inference` if it is a plausible but unlicensed \
deduction, and `fabricated` only when there is no visible basis at all. \
Use `None` for correct empty abstentions.
• Missing required components → `"partial_reason": "omission"`.
• If both hallucination and omission are present → \
`"partial_reason": "mixed"`.
• If a component is marked `required: false` / `optional`, its absence does \
NOT reduce correctness.
• If the notes say the answer is "defensible" or "ambiguous" → \
`"partial_reason": "ambiguous"`.

Return a JSON object mapping each field's id to its verdict. Nothing else.\
"""


# Backwards-compatible constant for callers that don't pass a title yet.
SYSTEM_PROMPT = build_system_prompt()



def build_field_block(
    field: dict[str, object],
) -> str:
    """
    Format a single field for inclusion in the user prompt.
    """
    qid = str(field["qid"])
    agent_value = field.get("candidate_value")
    prior_value = field.get("prior_value")
    gt_entry = dict(field["gt_entry"])

    parts = [f"### Field: `{qid}`"]
    parts.append(f"**Prior value:** {prior_value!r}")
    parts.append(f"**Candidate resulting value:** {agent_value!r}")

    # Expected value
    expected = gt_entry.get("expected_summary") or gt_entry.get("expected")
    parts.append(f"**Gold resulting value:** {expected!r}")

    if field.get("semantic_equivalence_check"):
        parts.append(
            "**Evaluation mode:** Schema-aware semantic equivalence check after "
            "a deterministic exact-match mismatch. Mark `correct` only when "
            "candidate and gold are the same resulting-state commitment under "
            "the public evidence and questionnaire rules; keep distinct exact "
            "IDs, dates, amounts, and counts incorrect."
        )
        hard_reason = field.get("hard_eval_reasoning")
        if hard_reason:
            parts.append(
                "**Deterministic scorer note:** "
                + _compact_text(hard_reason, limit=260)
            )

    if field.get("semantic_exact_mismatch_check"):
        parts.append(
            "**Evaluation mode:** Exact-mismatch semantic escalation. The "
            "deterministic scorer already rejected an exact/alias match, but "
            "the candidate is non-empty and the turn-level evaluator confirmed "
            "this is not merely preserving the prior value when gold required "
            "a different resulting value. Judge semantic equivalence, partial "
            "coverage, wrong-choice recovery, or incorrectness using the "
            "public evidence and questionnaire rules."
        )
        hard_reason = field.get("hard_eval_reasoning")
        if hard_reason:
            parts.append(
                "**Deterministic scorer note:** "
                + _compact_text(hard_reason, limit=260)
            )

    claim_anchor = gt_entry.get("claim_anchor")
    entity_role = gt_entry.get("entity_role")
    if claim_anchor or entity_role:
        anchor_bits = []
        if claim_anchor:
            anchor_bits.append(f"claim_anchor={claim_anchor}")
        if entity_role:
            anchor_bits.append(f"entity_role={entity_role}")
        parts.append("**Claim target:** " + ", ".join(anchor_bits))

    metadata = field.get("field_metadata")
    if isinstance(metadata, dict) and metadata:
        metadata_bits = []
        field_type = metadata.get("type")
        if field_type:
            metadata_bits.append(f"type={field_type}")
        if metadata.get("other_specify"):
            metadata_bits.append("other_specify=true")
        options = metadata.get("options")
        if options:
            metadata_bits.append(f"options={options!r}")
        if metadata_bits:
            parts.append("**Questionnaire field metadata:** " + " | ".join(metadata_bits))
        question_text = metadata.get("question_text") or metadata.get("label")
        if question_text:
            parts.append(
                "**Question text:** " + _compact_text(question_text, limit=420)
            )
        gold_standard = metadata.get("gold_standard")
        if gold_standard:
            parts.append(
                "**Questionnaire gold standard:** "
                + _compact_text(gold_standard, limit=520)
            )
        normalization_rules = metadata.get("normalization_rules")
        if normalization_rules:
            parts.append(
                "**Normalization rules:** "
                + _compact_text(normalization_rules, limit=360)
            )
        information_units = _format_information_units(metadata.get("information_units"))
        if information_units:
            parts.append("**Questionnaire information units:**\n" + information_units)

    reference_context = field.get("reference_context")
    if isinstance(reference_context, list) and reference_context:
        parts.append(
            "**Reference options / linked rows:**\n"
            + json.dumps(reference_context, indent=2, ensure_ascii=False)
        )

    # Evidence
    evidence = gt_entry.get("evidence")
    if evidence:
        parts.append(f'**Evidence from public state:** "{evidence}"')
    else:
        parts.append("**Evidence from public state:** (none)")

    # Components
    components = gt_entry.get("components", [])
    if components:
        comp_lines = []
        for c in components:
            req = "REQUIRED" if c.get("required", True) else "optional"
            comp_lines.append(
                f"  - [{req}] {c.get('id', '?')}: "
                f"expected={c.get('expected')!r}, "
                f"evidence={c.get('evidence')!r}"
            )
        parts.append("**Components:**\n" + "\n".join(comp_lines))

    # Metadata
    present = gt_entry.get("present_in_utterance", True)
    difficulty = gt_entry.get("extraction_difficulty", "direct")
    evidence_source = gt_entry.get("evidence_source")
    source_text = evidence_source or "unspecified"
    parts.append(
        f"**Evidence source:** {source_text}  |  "
        f"**Present in current utterance:** {present}  |  "
        f"**Extraction difficulty:** {difficulty}"
    )

    alternatives = gt_entry.get("acceptable_alternatives")
    if alternatives:
        parts.append(f"**Acceptable alternatives:** {alternatives}")

    set_keys = (
        "required",
        "acceptable_additions",
        "debatable",
        "unacceptable",
    )
    if any(key in gt_entry for key in set_keys):
        set_bits = [
            f"{key}={gt_entry.get(key)!r}"
            for key in set_keys
            if key in gt_entry
        ]
        parts.append("**Set-match constraints:** " + " | ".join(set_bits))

    # Notes
    notes = gt_entry.get("notes")
    if notes:
        parts.append(f"**Notes:** {notes}")

    return "\n".join(parts)


def build_user_prompt(
    *,
    current_utterance: str,
    visible_history: list[dict[str, str]],
    prior_state: dict[str, object],
    fields: list[dict[str, object]],
) -> str:
    """
    Build the full user prompt for a batch of soft fields.
    """
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
        f"## Fields to Evaluate ({len(fields)})\n",
    ]
    for field in fields:
        blocks.append(build_field_block(field))
        blocks.append("")  # blank line separator

    blocks.append(
        "Return a JSON object where each key is the field id "
        "and the value is {correctness, partial_reason, source, reasoning}. "
        "Include partial_reason only when correctness is partially_correct."
    )
    return "\n".join(blocks)


# ── Repeat-group alignment prompts ───────────────────────────────────────

def build_alignment_system_prompt(form_title: str = "form") -> str:
    """Build the alignment system prompt, parameterised by form title."""
    return f"""\
You are a precise matching engine for a {form_title} evaluation \
pipeline.

You will receive conversation context plus two lists of repeat-group instances \
(e.g. medications, vehicles, allergies, work packages, participants, or other \
repeated items):

1. **Current utterance** — the user's latest message.
2. **Visible history** — earlier conversation turns available to the agent.
3. **Prior repeat rows** — the repeat-group rows before this utterance.
4. **Ground-truth (GT) instances** — expected resulting rows, each with an \
index and all evaluated repeat fields.
5. **Agent instances** — resulting rows produced by the AI agent, each with an \
index and all available repeat fields.

Your job is to match each GT instance to the single best-matching agent \
instance (1-to-1), or declare it unmatched if no reasonable match exists. \
Similarly, identify any agent instances that have no GT counterpart \
(hallucinated).

Matching rules:
• Match based on the **semantic identity of the row**, not exact string \
equality and not row position.
• Use the full context: current utterance, visible history, prior repeat rows, \
identity anchors, alignment-key hints, and every field in each row.
• Descriptive aliases can match when they refer to the same entity. Examples: \
"my car" may match "claimant vehicle"; "other car that rear-ended me" may \
match "accident perpetrator"; "blue Nissan" may match "blue Nissan Sentra".
• Exact identifiers are decisive when present and non-contradictory: licence \
plates, emails, proposal IDs, participant numbers, PICs, account numbers, and \
similar codes should not be loosely matched against different identifiers.
• Harmless representation variants that preserve the same identifier, such as \
stable label prefixes (`WP1` vs `1` for a work-package number), may match when \
the schema notes or visible context support that convention.
• If a field is empty, missing, or marked null on one side, do NOT treat that \
alone as evidence against a match — rely on other fields and context.
• Each GT instance may match at most one agent instance and vice versa.
• Prefer exact identity matches over loose ones. Never force a match when \
the instances clearly describe different entities.
• When row counts differ, use the explicit unmatched buckets. Do not invent a \
weak match just to consume every row.
• If one agent row appears to merge multiple GT rows, match it only to the best \
single GT row and put the other GT row(s) in `unmatched_gt` with decision \
`"merged_into_agent"`.
• If multiple agent rows appear to split one GT row, match the best single agent \
row and put the other agent row(s) in `unmatched_agent` with decision \
`"split_from_gt"`.

Return a JSON object with exactly these keys:
{{
  "matches": [
    {{
      "gt_index": <int>,
      "agent_index": <int>,
      "decision": "same_instance",
      "reasoning": "<brief reason>"
    }},
    ...
  ],
  "unmatched_gt": [
    {{
      "gt_index": <int>,
      "decision": "missing_agent_instance" | "merged_into_agent" | "no_defensible_match",
      "reasoning": "<brief reason>"
    }},
    ...
  ],
  "unmatched_agent": [
    {{
      "agent_index": <int>,
      "decision": "spurious_agent_instance" | "split_from_gt" | "retained_prior_without_gold" | "no_defensible_match",
      "reasoning": "<brief reason>"
    }},
    ...
  ]
}}

Nothing else. No commentary.\
"""


# Backwards-compatible constant.
ALIGNMENT_SYSTEM_PROMPT = build_alignment_system_prompt()


def build_alignment_prompt(
    gt_instances: list[dict],
    agent_instances: dict[int, dict],
    alignment_keys: list[str],
    group_name: str = "instances",
    *,
    current_utterance: str = "",
    visible_history: list[dict[str, object]] | None = None,
    prior_instances: list[dict[str, object]] | None = None,
    alignment_field_metadata: dict[str, dict[str, object]] | None = None,
) -> str:
    """Build the user prompt for LLM-based repeat-group alignment.

    Parameters
    ----------
    gt_instances : list of GT instance dicts (with ``fields`` and
        ``ground_truth_index``).
    agent_instances : ``{agent_idx: {field: value, ...}, ...}`` —
        the agent's parsed instances.
    alignment_keys : field names to highlight as matching hints.
    group_name : human-readable group label (e.g. ``"vehicles"``).
    """
    def _gt_field_values(gt_inst: dict) -> dict[str, object]:
        values: dict[str, object] = {}
        for field_name, field_spec in (gt_inst.get("fields") or {}).items():
            if isinstance(field_spec, dict):
                values[field_name] = field_spec.get("expected")
            else:
                values[field_name] = field_spec
        return values

    blocks: list[str] = [
        "## Current Utterance\n",
        current_utterance or "(none)",
        "\n---\n",
        "## Visible History\n",
        json.dumps(visible_history or [], indent=2, ensure_ascii=False),
        "\n---\n",
        f"## Repeat-group: `{group_name}`",
        f"Alignment-key hints: {alignment_keys}",
        f"GT instance count: {len(gt_instances)}",
        f"Agent instance count: {len(agent_instances)}",
    ]
    if alignment_field_metadata:
        blocks.append("Alignment-key schema notes:")
        for key in alignment_keys:
            metadata = alignment_field_metadata.get(key) or {}
            if not metadata:
                continue
            notes: list[str] = []
            field_type = metadata.get("type")
            if field_type:
                notes.append(f"type={field_type}")
            question_text = metadata.get("question_text") or metadata.get("label")
            if question_text:
                notes.append("question=" + _compact_text(question_text, limit=180))
            gold_standard = metadata.get("gold_standard")
            if gold_standard:
                notes.append("gold_standard=" + _compact_text(gold_standard, limit=260))
            normalization_rules = metadata.get("normalization_rules")
            if normalization_rules:
                notes.append(
                    "normalization_rules="
                    + _compact_text(normalization_rules, limit=220)
                )
            if notes:
                blocks.append(f"- `{key}`: " + " | ".join(notes))
        blocks.append("")
    blocks.extend(
        [
            "\n---\n",
            "## Prior Repeat Rows\n",
            json.dumps(prior_instances or [], indent=2, ensure_ascii=False),
            "\n---\n",
            f"### Ground-truth instances ({len(gt_instances)})\n",
        ]
    )
    for gt_inst in gt_instances:
        gt_idx = gt_inst["ground_truth_index"]
        anchor = gt_inst.get("identity_anchor", "")
        parts = [f"- **GT {gt_idx}**"]
        if anchor:
            parts[0] += f"  ({anchor})"
        parts.append("  Full resulting row:")
        parts.append(
            json.dumps(_gt_field_values(gt_inst), indent=4, ensure_ascii=False)
        )
        blocks.append("\n".join(parts))

    blocks.append(f"\n### Agent instances ({len(agent_instances)})\n")
    for a_idx in sorted(agent_instances.keys()):
        a_fields = agent_instances[a_idx]
        parts = [f"- **Agent [{a_idx}]**"]
        parts.append("  Full resulting row:")
        parts.append(json.dumps(a_fields, indent=4, ensure_ascii=False))
        blocks.append("\n".join(parts))

    if not agent_instances:
        blocks.append("_(no agent instances)_")

    blocks.append(
        "\nReturn the JSON matching object. Use matches, unmatched_gt, and "
        "unmatched_agent explicitly; 1-to-1 matches only."
    )
    return "\n".join(blocks)
