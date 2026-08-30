# Calibration Protocol v0.9.2

Status: execution record for the frozen independent-reconstruction audit.

This protocol checks whether benchmark gold states are recoverable from the
public packet and annotation contract. It is not a separate student manual.

## Goal

Blind annotation is used as a recoverability check. Annotators receive the same
public inputs as an evaluated system:

- questionnaire/schema guidance;
- prior record state;
- visible dialogue history;
- current user utterance;
- record context.

They do not see provisional gold, author rationales, stressor labels, forbidden
commits, design metadata, or each other's annotations.

Each annotator produces a post-turn record state and evidence spans for
non-preserving edits. The provisional author gold is only a comparison target
until adjudication.

## Reliability Slice

Use a held-out blind slice, stratified across:

- form domain;
- update family;
- prior-state condition, including S3 and S4;
- evidence location;
- gates or branches;
- repeated-instance involvement;
- item load.

The paper reliability slice contains two independent annotations per item
across 60 items, produced by eight annotators working in four form-specific
pairs. If the item set or contract changes materially, run reliability on a
fresh slice.

## Comparison

Materialize each annotation into a full resulting record. Then compare three
states field by field:

- annotator 1;
- annotator 2;
- provisional gold.

Use the common prior state to compute transition labels. The comparison should
focus on the resulting record, not on superficial differences in how an
annotator described the edit.

Three-way outcomes:

- all three agree: no action;
- both annotators agree and differ from provisional gold: gold-revision
  candidate;
- one annotator agrees with provisional gold: likely annotator error, schema
  ambiguity, or normalization gap;
- all three differ: manual adjudication.

## Adjudication

Adjudication is item-centered and field-specific. The adjudicator reads the
public packet, reviews the full form in schema order, and resolves each
highlighted field difference.

Each decision records:

- category: `annotator_error`, `author_gold_revision`, `normalization_gap`,
  `schema_gap`, `contract_gap`, or `follow_up`;
- short note grounded in public evidence;
- applied value when gold changes;
- evidence spans for substantive gold revisions.

Accepted gold revisions update the benchmark item and must pass validation.
Rejected or schema-illegal annotation paths do not enter the accepted gold.

Freeze the pre-adjudication three-way snapshot as the reliability denominator.
Keep adjudication notes as a sidecar so later reports can separate annotator
error from gold revision and contract/schema gaps.

## Reporting

Report reliability with the same views used for model evaluation:

- whole-record exact match;
- all-field agreement;
- changed-field precision, recall, and F1;
- preservation accuracy;
- transition-label agreement;
- diagnostic disagreement categories.

The paper reports 29/60 whole-record exact matches, 96.8% strict
field-level agreement, 97.1% lenient field-level agreement, transition-label
kappa of 0.86, and 22 annotator-agreement/provisional-gold-difference rows in
the three-way queue.

High agreement supports the claim that the public packet and contract determine
a stable resulting state. The disagreement categories were used to identify
items, schema regions, and rules requiring repair before the benchmark was
frozen.
