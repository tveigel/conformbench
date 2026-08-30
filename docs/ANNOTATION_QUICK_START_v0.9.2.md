# Annotation Quick Start v0.9.2

Use this checklist while annotating in Blind Studio.

## Your Job

Start with the prior form state. Apply only the changes licensed by the current
utterance, visible history, or a deterministic schema rule. Leave everything
else unchanged.

You are not checking author gold. You do not see intended touched fields,
stressor labels, forbidden commits, author notes, or item-level IU gold.

## Inputs

Each packet shows:

- `record_context`: who or what the record is about;
- `prior_state`: the form state before the current utterance;
- `visible_history`: earlier public turns;
- `current_utterance`: the newest user message;
- field guidance in the inspector.

The current utterance is usually the main source for changes. Use visible
history when the current utterance refers back to it, when it identifies a
repeat row, or when it exposes a silent error in the prior state.

## Workflow

1. Confirm your annotator ID.
2. Select a packet.
3. For the first packet in a questionnaire, skim **Understand form** once.
4. Read the record context, history, and current utterance.
5. In **Edit form**, walk the form field by field.
6. Change only supported fields.
7. Attach evidence to every changed field.
8. Resolve repeat routing and gate/controller prompts.
9. In **Review & Save**, fix warnings or flag the item, then save.

## Field Loop

For each field, ask:

> Does the public packet license a different resulting value for this field?

If no, preserve the prior value.

If yes:

1. edit the value;
2. attach the shortest evidence span that supports it;
3. add a note only if the decision needs explanation.

If two resulting values remain defensible, do not guess. Mark the item or route
ambiguous.

## Basic Value Rules

- An empty field stays empty when it is not mentioned.
- If the user explicitly says the value is unknown, use the field's unknown
  representation, such as `"Unknown"` when the form offers that value.
- Zero is a value. If speed is `0`, write `0`, not empty.
- `false` is a value. Do not confuse "no" with unknown.
- Preserve the user's phrasing as much as the form allows. For free text, keep
  useful wording and hedges such as "around 2 weeks" or "I think last Friday".
- When a field requires a typed value or closed option, normalize only as much
  as needed to fit the schema, and keep the original evidence span attached.

## Transitions

- `set`: empty prior field now has a supported value.
- `change`: a prior value is corrected, refined, or replaced.
- `clear`: a prior value is retracted.
- `structural_clear`: the schema clears a child field after a gate/controller
  change.
- `preserve`: no supported change.

Unchanged fields need no evidence.

## Evidence

Every non-preserve transition needs evidence.

Use **Select evidence** on the field. Highlight the shortest span that explains
why this field changed to this value. Use multiple spans only when one span is
not enough.

Evidence may come from the current utterance or visible history. Structural
clears also need the triggering evidence when available.

## Repeat Groups

Before editing a repeated row, decide where the mention belongs:

- existing row: the text identifies one prior row;
- new row: the text clearly introduces a new entity;
- ambiguous: more than one row could fit.

Do not move values between rows by intuition.

## Gates And Branches

Resolve controller fields before child fields. If a gate closes and the schema
requires child fields to clear, confirm the structural clear. Do not keep or
edit closed child fields unless the field guidance says so.

## Other, Tables, Free Text

- Use `Other...` only when no listed option fits. Store the concrete value from
  the evidence, not bare `Other`.
- Fill table fields as rows and columns, not as prose.
- For free-text or semantic fields, write the resulting field value and attach
  evidence. Do not create item-level IU gold.

## Prior State Cases

- **S1 empty**: the prior state has no committed values yet.
- **S2 partial-correct**: the prior state is a faithful partial record. Apply
  the current update and preserve the rest.
- **S3 partial-incorrect**: the prior state contains a value that is now
  explicitly corrected or replaced by public evidence. Change the wrong value
  and cite the correction evidence.
- **S4 silent mismatch**: the prior state silently disagrees with visible
  history or schema. The assistant will not point this out. Repair it only when
  visible history plus schema determines one correct result. If not, preserve
  or flag.

## Validity Flags

Use a flag when the packet cannot be solved objectively:

- `ambiguous`: more than one resulting state remains defensible;
- `input_incoherence`: public evidence contradicts itself in a schema-relevant
  way;
- `schema_gap`: the schema lacks a needed field or rule;
- `contract_gap`: the rules do not say what to do;
- `normalization_gap`: the value cannot be mapped to an allowed representation;
- `hidden_author_intent`: the packet requires non-public author intent.

## Final Check

Before saving:

- every changed field has evidence;
- every edited repeat mention has a route decision;
- every gate/controller consequence is confirmed or explained;
- every `Other...` value has concrete text;
- every table edit is structured;
- every unresolved ambiguity is flagged.
