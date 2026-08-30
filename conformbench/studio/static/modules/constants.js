// -- Constants ----------------------------------------------------------------

// ── Primary delta types (replaces old families + update types) ──────────────
// Each type describes the *relation* between old field value and new gold value.

export const PRIMARY_DELTA_TYPES = [
  { value: "add",     label: "Add",     color: "#0ea5e9", desc: "Old value is empty/unknown → fill it with new supported information." },
  { value: "refine",  label: "Refine",  color: "#8b5cf6", desc: "Old and new values are compatible, but the new one is more specific (e.g. adding a cross-street to an address)." },
  { value: "correct", label: "Correct", color: "#f97316", desc: "Old and new values are incompatible — the old one was wrong and must be replaced." },
  { value: "retract", label: "Retract", color: "#ec4899", desc: "Remove/withdraw a value — move the field back to unknown (e.g. user says 'actually, I'm not sure about that')." },
];

// Keep FAMILIES as an alias for backward compatibility with existing items/imports
export const FAMILIES = PRIMARY_DELTA_TYPES;

export const CONTRAST_ROLES = [
  { value: "anchor", label: "Anchor", desc: "Easy, clean-cut case. Tests the happy path." },
  { value: "failure_revealing", label: "Failure-revealing", desc: "Designed to expose a specific agent failure mode." },
];

export const DIFFICULTY_TIERS = [
  { value: "anchor", label: "Anchor", desc: "Clean, interpretable case with quiet context and low structural pressure." },
  { value: "challenge", label: "Challenge", desc: "Default non-anchor benchmark case. Not a numeric score over the item stressors." },
  { value: "hard", label: "Hard", desc: "Explicitly designated high-density benchmark case. Not derived from a composite stress score." },
];

// ── Prior state condition (unchanged) ────────────────────────────────────────

export const STATE_CONDITIONS = [
  { value: "S1", label: "S1 - Empty", desc: "Nothing has been recorded yet. Blank form." },
  { value: "S2", label: "S2 - Partial correct", desc: "Some fields are filled, all values are correct." },
  { value: "S3", label: "S3 - Partial incorrect", desc: "Some fields filled, at least one value is wrong." },
  { value: "S4", label: "S4 - Inconsistent", desc: "Fields filled with contradictory values." },
];

// ── Legacy history-context shorthand (canonical item metadata lives in evidence.*) ──
// H-codes are kept for reusable context files and older pilot items. The
// benchmark's canonical history semantics are the orthogonal evidence fields:
// history_required, support_distance, and conflict_present.

export const HISTORY_CONDITIONS = [
  { value: "H1", label: "H1 - None", desc: "Cold start. The current utterance is the first turn." },
  { value: "H2", label: "H2 - Recent support", desc: "Prior turns contain nearby evidence consistent with and relevant to the current utterance." },
  { value: "H3", label: "H3 - Distant support", desc: "Relevant evidence exists, but only farther back in the dialogue or buried under intervening turns." },
  { value: "H4", label: "H4 - Conflict", desc: "Prior turns contain claims that conflict with the current utterance or with each other." },
];

export const EVIDENCE_DEFAULTS = {
  history_required: false,
  support_distance: 0,
  conflict_present: false,
};

// Legacy update-family aliases kept for backward compatibility with older pilot
// items. New items should use primary_delta_type + state/evidence metadata.
export const UPDATE_TYPES = [
  { value: "U1", label: "U1 - Initialize", desc: "Legacy alias for add from an empty prior state." },
  { value: "U2", label: "U2 - Add", desc: "Legacy alias for incremental add to a non-empty state." },
  { value: "U3", label: "U3 - Refine", desc: "Legacy alias for specificity-increasing change." },
  { value: "U4", label: "U4 - Correct", desc: "Legacy alias for replacement of a wrong value." },
  { value: "U5", label: "U5 - Retract", desc: "Legacy alias for moving a field back toward unknown." },
  { value: "U6", label: "U6 - History-dependent", desc: "Legacy alias for cases where history retrieval or conflict handling is central." },
];

export const EVIDENCE_SOURCES = ["current_utterance", "recent_history", "distant_history"];
export const EVAL_STRATEGIES = ["exact", "set_match", "semantic"];
export const EXTRACTION_DIFFICULTIES = ["direct", "requires_inference", "ambiguous"];

export const VALUE_SOURCES = [
  { value: "stated",               label: "Stated",               desc: "Value was explicitly stated in the utterance.", color: "#3b82f6" },
  { value: "inferred_obvious",     label: "Inferred (obvious)",   desc: "Value can be trivially deduced from what was said.", color: "#8b5cf6" },
  { value: "inferred_non_obvious", label: "Inferred (non-obvious)", desc: "Value requires non-trivial reasoning or world knowledge.", color: "#ec4899" },
];

export const CLAIM_ANCHORS = [
  { value: "primary_subject", label: "primary subject", shortLabel: "subject", desc: "The field is about the record’s main subject, such as the patient or reporting party." },
  { value: "event", label: "event", shortLabel: "event", desc: "The field is about the incident or event itself rather than any one speaker." },
  { value: "repeat_instance", label: "repeat instance", shortLabel: "instance", desc: "The field is about one resolved item inside a repeat group." },
  { value: "role_entity", label: "role entity", shortLabel: "role", desc: "The field is about an external entity identified by role, such as a witness or other party." },
];

export function getClaimAnchorMeta(value) {
  return CLAIM_ANCHORS.find(anchor => anchor.value === value) || null;
}

export function formatClaimAnchorLabel(claimAnchor, entityRole = null, { short = false } = {}) {
  const anchor = getClaimAnchorMeta(claimAnchor);
  if (!anchor) return entityRole || "";
  const base = short ? anchor.shortLabel : anchor.label;
  return entityRole ? `${base}: ${entityRole}` : base;
}

// ── Per-field delta types ────────────────────────────────────────────────────
// Annotated on each field in gold_annotations to classify the field-level change.

export const FIELD_DELTA_TYPES = [
  { value: "keep",    label: "Keep",    color: "#94a3b8", desc: "No change — value stays the same." },
  { value: "add",     label: "Add",     color: "#0ea5e9", desc: "Field was empty, now filled." },
  { value: "refine",  label: "Refine",  color: "#8b5cf6", desc: "More specific but compatible with old value." },
  { value: "correct", label: "Correct", color: "#f97316", desc: "Old value was wrong, replaced." },
  { value: "retract", label: "Retract", color: "#ec4899", desc: "Value withdrawn — field moved to unknown." },
];

/** Auto-suggest a delta type from the diff kind (SET/CHANGED/CLEAR/null). */
export function suggestDeltaType(diffKind) {
  if (!diffKind) return "keep";
  if (diffKind === "SET") return "add";
  if (diffKind === "CLEAR") return "retract";
  return "correct"; // CHANGED defaults to correct; item authors can override to refine.
}

export const ITEM_STRESSORS = [
  {
    key: "target_binding_pressure",
    label: "Target binding pressure",
    values: ["none", "implicit_unique_target", "prior_mention_target", "competing_candidates"],
    tip: "How much context is required to identify the intended field, entity, event, or prior mention outside repeat-instance routing?",
    warning: "Use this for general target binding only, not for repeated-instance routing.",
    countsAs: "Field/entity/event binding outside repeat-group row selection.",
    notCountsAs: "Choosing the correct row inside a repeat group. That belongs to repeat-instance routing pressure.",
    example: "\"Update the witness statement\" after only one witness has been introduced is implicit unique target binding, not repeat-instance routing.",
    notePrompt: "name the target-binding cue and why repeat-instance routing does not apply",
    rules: {
      none: "The target is directly named or otherwise requires no noteworthy contextual binding beyond ordinary normalization.",
      implicit_unique_target: "The target is not named directly, but one non-repeat schema-compatible target is uniquely recoverable from context or history.",
      prior_mention_target: "The target must be recovered from a previous mention, alias, description, deixis, or other prior contextual reference.",
      competing_candidates: "Two or more plausible non-repeat targets remain and correct routing requires disambiguation.",
    },
  },
  {
    key: "commitment_boundary_pressure",
    label: "Commitment boundary pressure",
    values: ["none", "partial_value", "uncertain_or_qualified", "unresolved_scope_or_target"],
    tip: "Does the item stress the boundary between a licensed structured commitment and conservative abstention?",
    countsAs: "Cases where some content is present but a structured commitment may still be unsafe, incomplete, hedged, or unresolved.",
    notCountsAs: "Target identification problems by themselves. Use target binding pressure or repeat-instance routing pressure instead.",
    example: "\"I think it was sometime after lunch\" can make commitment unsafe even when the target field itself is already clear.",
    notePrompt: "name the commitment boundary and the conservative rule applied",
    rules: {
      none: "The admissible evidence either cleanly licenses the structured commitment or licenses no transition, so the prior value is preserved by default.",
      partial_value: "Some information is present, but it remains incomplete for the slot’s expected commitment.",
      uncertain_or_qualified: "The speaker hedges, expresses uncertainty, attributes the claim to someone else, or asks a question rather than making a direct supported commitment.",
      unresolved_scope_or_target: "Scope, subject, time, or target remains unresolved enough that a structured commitment would be unsafe.",
    },
  },
  {
    key: "distractor_competition",
    label: "Distractor competition",
    values: ["none", "weak_distractor", "same_schema_region", "competing_entity_or_group"],
    tip: "Does explicit schema-compatible non-gold content compete with the intended update?",
    countsAs: "Explicit non-gold details that plausibly map to the same slot, a competing entity, or a broader distractor field.",
    notCountsAs: "World knowledge, prior-state inertia, or pragmatic implication without explicit competing content.",
    example: "If the utterance mentions the patient, spouse, and child symptoms but only the patient is in scope, those explicit extra symptom details are distractor competition.",
    notePrompt: "state the strongest distractor configuration and the observable cue",
    rules: {
      none: "No explicit non-gold content plausibly competes with the intended update.",
      weak_distractor: "Non-gold content is present but remains clearly outside the intended field, entity, or repeated-instance target.",
      same_schema_region: "The strongest distractor sits in a nearby field, section, or semantically related schema region.",
      competing_entity_or_group: "The strongest distractor maps to another plausible entity, repeated instance, or group.",
    },
  },
  {
    key: "unsupported_alternative_affordance",
    label: "Unsupported alternative affordance",
    values: ["none", "single_slot_affordance", "multi_slot_affordance", "cross_entity_or_default_affordance"],
    tip: "Does the item contain salient non-entailed alternatives that are explicitly documented as forbidden commits?",
    warning: "A non-zero value here should be backed by at least one observable forbidden commit with a cue and reason.",
    countsAs: "Documented non-entailed alternatives the system might wrongly commit, regardless of whether the cue comes from prior state, pragmatics, or explicit competitors.",
    notCountsAs: "Generic uncertainty by itself. If the main issue is abstention versus commitment, use commitment boundary pressure.",
    example: "A dramatic description can make a severe injury label look plausible even when the schema field is not licensed; that becomes a stressor only once the forbidden commitment is documented explicitly.",
    notePrompt: "name the forbidden-commit cue and why the alternative is unsupported",
    rules: {
      none: "No salient non-entailed alternative commitment is documented for the item.",
      single_slot_affordance: "One specific unsupported slot commitment is salient enough to document as a forbidden commit.",
      multi_slot_affordance: "Multiple unsupported slot commitments are salient enough to document as forbidden commits.",
      cross_entity_or_default_affordance: "Unsupported commitments become plausible across entities, repeated structures, or schema-default completions.",
    },
  },
  {
    key: "repeat_instance_routing_pressure",
    label: "Repeat-instance routing pressure",
    values: ["none", "unique_instance", "same_group_competition", "cross_group_or_multi_instance"],
    tip: "Repeated-instance routing only: how much pressure is there to distinguish, preserve, create, attach, or avoid edits to repeated instances?",
    warning: "Derived repeat-group involvement is informative but not decisive. This stressor may still be non-none when no repeat-group leaf changes in the gold state.",
    countsAs: "Choosing the correct repeated instance, preserving the right repeated rows, or coordinating repeated-instance operations across one or more groups.",
    notCountsAs: "General non-repeat target binding. That belongs to target binding pressure.",
    example: "A family-member detail should not be attached to the patient's allergy table even though no allergy row should change; that is still repeat-instance routing pressure.",
    notePrompt: "name the repeat group, routing cue, and whether the pressure is about selecting, preserving, creating, attaching, or avoiding an edit",
    rules: {
      none: "Repeated structure is not doing meaningful interpretive work in the item; correct handling does not depend on distinguishing, preserving, creating, attaching, or avoiding edits to repeated instances.",
      unique_instance: "A repeated instance or group is in play, but only one candidate instance is active or uniquely recoverable for the correct interpretation.",
      same_group_competition: "Multiple candidate instances exist within one repeat group, or the system must avoid editing the wrong row inside that group.",
      cross_group_or_multi_instance: "Interpretation spans multiple repeated instances or groups, or the system must coordinate create/attach/preserve/avoid-edit decisions across them.",
    },
  },
];

export const DIMENSIONS = ITEM_STRESSORS;

export const DIMENSION_DEFAULTS = Object.fromEntries(
  DIMENSIONS.map(d => [d.key, d.values[0]])
);

export const CONTEXT_KIND_LABELS = {
  state: "Form state",
  history: "Conversation history",
};

// Delta-type-aware defaults applied when creating items in template status
export const DELTA_DEFAULTS = {
  add:     { state_condition: "S1", history_required: false, support_distance: 0, conflict_present: false },
  refine:  { state_condition: "S2", history_required: false, support_distance: 0, conflict_present: false },
  correct: { state_condition: "S3", history_required: false, support_distance: 0, conflict_present: false },
  retract: { state_condition: "S2", history_required: false, support_distance: 0, conflict_present: false },
};

// Keep FAMILY_DEFAULTS as alias for backward compatibility
export const FAMILY_DEFAULTS = DELTA_DEFAULTS;

// ── Routing rules for repeat groups (with human-readable descriptions) ──────
export const ROUTING_RULES = [
  { value: "stable_key_match",              label: "Stable key match",              desc: "Matched by a unique identifier (e.g. license plate, allergen name, medication)." },
  { value: "unique_alias",                  label: "Unique alias",                  desc: "Same entity referred to by a different name or description." },
  { value: "unique_non_primary_entity",     label: "Unique non-primary entity",     desc: "Only one candidate of this type exists, so routing is unambiguous." },
  { value: "compatible_underspecified_row",  label: "Compatible underspecified row", desc: "Partial match on available fields — row is underspecified but compatible." },
  { value: "new_distinct_entity",           label: "New distinct entity",           desc: "A new instance not seen in prior state — user introduced a new entry." },
  { value: "ambiguous_multiple_candidates", label: "Ambiguous (multiple)",          desc: "Multiple existing rows could match; cannot determine which one." },
];

export const FAILURE_MODE_SUGGESTIONS = [
  "wrong_target_commit",
  "unsupported_commit",
  "input_incoherence_overcommit",
  "missed_supported_update",
  "collateral_edit",
  "failed_correction",
  "failed_retraction",
  "repeat_instance_misalignment",
  "history_evidence_ignored",
  "gate_violation",
  "cross_field_leakage",
];
