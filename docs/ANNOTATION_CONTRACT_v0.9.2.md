# Annotation Contract v0.9.2

Status: active rules for Blind Studio annotation.

Read the quick start first. This contract defines the edge cases.

## 1. Task

For each item, produce the whole-record resulting state after applying the
current user utterance to the prior state.

Start from `prior_state`. Change a field only when the public packet licenses a
transition. Preserve every other field.

Public evidence means:

- `current_utterance`;
- `visible_history`;
- `record_context`;
- field guidance from the schema sheet.

The annotator must not see or use provisional author gold, intended touched
fields, stressor labels, forbidden commits, design metadata, author notes,
private generation notes, item-level IU gold, or another annotator's output.

The student annotation is independent. If it later differs from author gold,
the disagreement is adjudicated. Author gold is provisional until adjudication.

Annotators should not hand-edit full JSON. The studio records edits, evidence,
routing, gate decisions, validity flags, and notes, then materializes the full
resulting state.

## 2. Public Packet

Each item must provide:

- `item_id`;
- `record_context`;
- `prior_state`;
- `visible_history`;
- `current_utterance`;
- applicable schema guidance;
- version stamps for the contract, quick start, schema sheet, studio, and
  materializer when available.

If the subject, speaker role, or time/event scope cannot be resolved from the
public packet, flag the item unless the schema defines one deterministic
outcome.

## 3. Licensed Changes

Commit a value only when one of these is true:

- the value is stated in the current utterance or visible history;
- the value follows by deterministic normalization, such as date formatting,
  unit conversion, or case/whitespace cleanup;
- the schema explicitly licenses the mapping, alias, default, or structural
  consequence.

Do not commit a value because it is plausible. Do not infer causes, roles,
blame, categories, or row identity unless the text or schema licenses them.

Preserve the prior value, or keep an empty field empty, when a change would
require an unlicensed assumption.

## 4. Empty, Unknown, Zero, And Phrasing

These distinctions are part of the annotation target:

- Empty means no committed value. If an empty field is not mentioned, it stays
  empty.
- Unknown is an explicit user commitment when the user says they do not know or
  cannot provide the value. Use the field's unknown representation, such as
  `"Unknown"` when the schema provides one. If the schema has no unknown
  representation, keep the field empty/null and note or flag only when needed.
- Zero is a real value. Speed `0`, count `0`, or duration `0` is not empty.
- `false` is a real value. A supported denial is not the same as an unknown
  field.

Preserve user phrasing as much as the schema allows. For free text, keep
field-relevant wording, uncertainty, and hedges: "around 2 weeks" should not
be flattened to "2 weeks" unless the field requires a normalized duration. For
typed or closed fields, map the value to the schema representation and keep the
original evidence span.

## 5. Prior State Cases

The packet may start from different kinds of prior state:

- **S1 empty**: no prior field commitments.
- **S2 partial-correct**: a faithful partial record. Apply the current update
  and preserve unrelated values.
- **S3 partial-incorrect**: the prior state contains a wrong, stale, or lossy
  value that public evidence explicitly corrects, replaces, or retracts. Fix
  the value when the correction is licensed; preserve unrelated fields.
- **S4 silent mismatch**: the prior state silently disagrees with visible
  history or schema. The assistant does not announce the error. Repair it only
  when comparing prior state, visible history, current utterance, and schema
  determines one defensible resulting state.

Transition labels:

- `set`: empty prior field now has a supported value;
- `change`: prior value is corrected, refined, or replaced;
- `clear`: prior value is retracted;
- `structural_clear`: schema rule clears a dependent field;
- `preserve`: no licensed change.

## 6. Evidence

Every non-preserve transition needs evidence.

Record the field, prior value, transition, resulting value, evidence source,
and evidence span. Select the shortest span that supports the decision. Multiple
spans are allowed only when one span is not enough.

For structural consequences, record or confirm the triggering transition and
schema rule when the UI asks for it. Attach the triggering evidence when
available.

## 7. Silent Prior-State Errors

Some items include a silent error in `prior_state`: stale value, wrong value,
duplicate row, wrong row, or schema-impossible child state. The conversation
should still read like a normal intake.

Repair the prior state when visible history, current utterance, and schema
determine one defensible resulting state. Visible history alone can license the
repair, even when the current utterance is about another field.

If the public packet does not determine the repair, preserve what can be
preserved and flag ambiguity when needed.

## 8. Conflicts, Time Scope, And Uncertainty

Use these rules in order:

1. If a later mention conflicts with a prior value but is not framed as a
   correction, clarification, replacement, or retraction, preserve the prior
   value.
2. If claims belong to different time or event scopes, route each claim to the
   field whose scope it supports.
3. If public evidence contradicts itself in a schema-relevant way, apply the
   schema's conservative rule if one exists; otherwise flag `input_incoherence`.
4. If more than one resulting state remains defensible, flag `ambiguous`.

Hedges are field-local. Some fields commit hedged concrete values; others keep
unknown, use an explicit `"Unknown"` option, or record the claim only in notes.
Follow the field guidance.

Keep `null`, `false`, `0`, `[]`, and `"Unknown"` distinct unless the field
guidance explicitly equates them.

## 9. Repeat Groups

For every repeated-entity mention, choose the route before editing child
fields:

- existing row, when public evidence identifies the row;
- new row, when public evidence introduces a distinct new entity;
- ambiguous, when several rows could fit.

Allowed routing evidence includes stable keys, unique aliases, only-entity
cases, compatible underspecified rows, and schema-specific routing rules.
Hidden author intent is not a routing key.

## 10. Gates, Other Fields, Tables, And Free Text

Resolve controller fields before dependent child fields. If a controller closes
a section and the schema requires child fields to clear, confirm the structural
clear. Do not keep active child values under a closed gate unless the schema
explicitly allows it.

For `Other...` fields, store the concrete value from the evidence. Do not store
bare `Other`.

For table fields, enter rows and columns. Do not replace a table with a prose
summary.

For free-text or semantic fields, the blind annotator writes the resulting
field value and evidence spans. Item-level semantic IU gold is created or
verified later during adjudication, not in the blind pass.

## 11. Validity Flags

Flag instead of guessing:

- `valid`: the item has one defensible resulting state;
- `ambiguous`: multiple resulting states remain defensible;
- `input_incoherence`: public evidence contains a schema-relevant contradiction;
- `schema_gap`: the schema lacks a needed field or rule;
- `contract_gap`: this contract does not settle the case;
- `normalization_gap`: the value cannot be mapped to an allowed representation;
- `hidden_author_intent`: the result depends on non-public author intent.

Items requiring hidden author intent must not be retained in the final
benchmark.

## 12. Current Patent-Form Edge Rules

Apply these when a patent packet contains the relevant fields:

1. Repair silent prior-state mismatches from public history when the repair is
   determinate.
2. If a parent gate is false, dependent count fields are `null` unless the
   schema defines another inactive value. Do not write `0` just because the
   section is closed.
3. Preserve accurate prior values exactly. Correct faulty, stale, or lossy
   prior values only when public evidence supports the correction.
4. A scoped denial such as "not disclosed outside the company" does not settle
   a broader boolean unless the denial covers the full field scope.
5. MTA-copy evidence routes to the agreement-level `copy_attached` field first.
   Use a material-level copy field only when the packet and schema make that
   material-specific link explicit.
6. Keep stated honorifics in names, such as `Dr. Raj Patel`, unless the schema
   gives a normalization rule.

## 13. Adjudication Categories

Disagreements after blind annotation are reviewed as one of:

- annotator error;
- author-gold revision;
- schema gap;
- contract gap;
- normalization gap;
- semantic IU gap;
- tooling error;
- item invalid for blind scoring.

Only public evidence and public rules may justify the final accepted state.
