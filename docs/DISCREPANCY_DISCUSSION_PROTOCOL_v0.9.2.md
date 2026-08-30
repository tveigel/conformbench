# Discrepancy Discussion Protocol v0.9.2

Status: protocol for post-annotation adjudication discussions after a blind
annotation pass.

## Purpose

The discussion is not a negotiation over hidden author intent. It is a
controlled method for deciding why two independent annotations differ and what
must change before the benchmark is frozen.

The discussion produces auditable labels and actions. It must not silently
rewrite the gold state, the annotator submission, or the annotation contract.

## When To Discuss

Use this order:

1. The annotator completes the blind pass without discussion.
2. The review script generates row-level comparison tables.
3. A lead reviewer prepares initial adjudication labels.
4. The annotator and reviewer meet to discuss deviations.
5. The reviewer records final labels and actions.
6. Public rules, schema sheets, UI behavior, or item gold are revised only after
   the discussion record is written.

Do not use a discussed item as fresh reliability evidence afterward. If an item
is re-annotated after feedback, that result is a calibration/training check.
Formal inter-annotator agreement must be measured on a fresh blind slice after
the protocol is frozen.

## Inputs

Before the meeting, collect:

- the annotator's saved blind annotation;
- the provisional authored gold item;
- generated review CSV and Markdown report;
- adjudication sidecar JSON, if available;
- current annotation contract, quick-start, schema sheet, and scoring spec;
- the public packet shown to the annotator;
- the UI/tool version used during annotation.

## Meeting Roles

Reviewer:
prepares the row list, asks the protocol questions, assigns final labels, and
records actions.

Annotator:
explains which public evidence and rule justified the annotation. The
annotator should not be asked to infer hidden author intent.

Optional second reviewer:
resolves cases where the reviewer and annotator disagree about whether the
public rule already settled the decision.

## Row Discussion Template

For each non-matching row, record:

| Field | Required note |
| --- | --- |
| `item_id` | Benchmark item identifier. |
| `field_path` | Flattened field path from the review CSV. |
| `auto_verdict` | Mechanical verdict from the review script. |
| `adjudicated_verdict` | Final row-level interpretation. |
| `annotator_rationale` | What evidence/rule the annotator used. |
| `reviewer_rationale` | What evidence/rule the reviewer used. |
| `final_category` | One process category from the list below. |
| `action` | No change, annotator training, schema edit, contract edit, UI fix, scorer/normalizer edit, author-gold revision, or item removal/rewrite. |
| `public_rule_reference` | File/section/line or schema field that settles the decision, if one exists. |

## Final Process Categories

Use exactly one primary category:

- `annotator_error`: the public packet and rules already settled the decision.
- `contract_gap`: the general annotation rule was missing or ambiguous.
- `schema_gap`: the field-local schema instruction was missing or ambiguous.
- `normalization_gap`: both values are acceptable but the scorer/reviewer needs
  a normalization or semantic scoring policy.
- `semantic_rubric_gap`: the field needs explicit IUs or a clearer free-text
  scoring rubric.
- `tooling_error`: the UI made a correct annotation hard or impossible.
- `author_gold_revision`: the provisional author gold is not the best
  recoverable answer under public rules.
- `item_invalid_for_blind_scoring`: more than one resulting state remains
  defensible after applying the public rules.
- `training_only`: the item was discussed or repeated and may no longer be used
  as fresh reliability evidence.

Optional secondary tags may be added for diagnostics, such as
`unsupported_extra_commit`, `missed_supported_commit`, `table_shape_error`,
`other_specify_error`, `fault_overcommit`, `partial_match`, or
`semantic_match`.

## Discussion Questions

Ask these in order:

1. What exact public evidence span supports the annotator's value?
2. What exact public evidence span supports the author value?
3. Are both annotators making the same commit/preserve decision?
4. If both committed, is the disagreement only wording, or does one value omit
   required field information?
5. Does the field have a schema-level instruction or IU rubric that settles the
   difference?
6. Did the annotator infer a cause, action, or fault judgment not directly
   licensed by the conversation?
7. Did the UI allow the intended structure, for example `Other...` free text or
   table rows?
8. Would another trained annotator recover one answer from the public packet
   alone?

## Action Rules

Use these rules after the meeting:

- If the label is `annotator_error`, update training examples or quick-start
  emphasis, but do not change the gold or schema.
- If the label is `contract_gap`, revise the contract and rerun calibration on
  structurally similar but textually different items.
- If the label is `schema_gap`, revise the field-local schema sheet and rerun
  affected item types.
- If the label is `normalization_gap` or `semantic_rubric_gap`, update the
  scoring policy before using the field in primary agreement claims.
- If the label is `tooling_error`, fix the UI and run a training/check pass
  before any scored blind run.
- If the label is `author_gold_revision`, update the provisional gold and
  document the reason.
- If the label is `item_invalid_for_blind_scoring`, remove or rewrite the item
  before the frozen benchmark.

## Scale-Up Rule

After a two-item smoke test, do not jump directly to full benchmark annotation.
Run the next fresh blind slice with 6-8 items covering:

- direct typed fields;
- free-text/IU fields;
- `Other...` values;
- table fields;
- repeat-group routing;
- gates/structural consequences;
- correction or retraction;
- a preserve-by-default case.

Freeze the protocol only after a fresh blind slice introduces no new core
rules, no hidden-author-intent items, and no unresolved tooling blockers.
