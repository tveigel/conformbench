# Inter-Annotator Agreement Protocol for ConFormBench

**Version**: 1.1 (release execution record for the locked 2026-05-16 plan)
**Companion to**: `CALIBRATION_PROTOCOL_v0.9.2.md`, `ANNOTATION_CONTRACT_v0.9.2.md`

---

## 1. Purpose

This protocol records the formal IAA study reported in the paper.
It is separate from calibration: calibration trains annotators and exposes
contract gaps, while IAA estimates whether independent annotators can recover
the same resulting record state from public packets alone.

The locked IAA slice is `reliability_slice_v1`.

The executed study used eight non-author expert annotators, with two annotators
independently reconstructing each 15-item form stratum. The sections below
retain the locked selection and task rules while recording the realized study.

---

## 2. Annotators

- **Executed design**: 8 blind annotators, 2 per 15-item form stratum.
- Annotators must not be item authors.
- Annotators must not have seen or discussed the IAA items before the blind pass.
- Annotators use only the blind annotation workspace and the in-studio reference docs.
- Annotators work independently and do not discuss item decisions during the pass.

Training materials reuse the Blind Studio quick start and the v0.9.2 annotation
contract. Practice or orientation items must not overlap with
`reliability_slice_v1`.

---

## 3. Reliability Slice: `reliability_slice_v1`

### 3.1 Size and Composition

The locked reliability slice contains **60 items**:

| Form | Items |
|------|-------|
| `crash_report` | 15 |
| `grant` | 15 |
| `patient_admin` | 15 |
| `patent_form` | 15 |
| **Total** | **60** |

The released administrative record is:

- `$CONFORMBENCH_DATA_DIR/reports/iaa/reliability_slice_v1/manifest.json`
- `$CONFORMBENCH_DATA_DIR/reports/iaa/reliability_slice_v1/item_ids.txt`
- `$CONFORMBENCH_DATA_DIR/reports/iaa/reliability_slice_v1/items.csv`
- `$CONFORMBENCH_DATA_DIR/reports/iaa/reliability_slice_v1/coverage.md`
- `$CONFORMBENCH_DATA_DIR/reports/iaa/reliability_slice_v1/LOCKED.md`

The standalone administrative files named in the pre-launch plan were not
retained. The released record was reconstructed directly from the sole 60 item
packets in the frozen blind-annotation export and cross-checked against the four
released 15-item audit strata. It contains no raw annotations or personal data.

### 3.2 Selection Criteria

The slice is stratified to cover the task conditions used in the benchmark:

| Dimension | Intended coverage |
|-----------|-------------------|
| Forms | 15 items per form across all four benchmark forms |
| Delta types | Add, refine, correct, and retract represented in each form |
| State conditions | S1 through S4 represented, with explicit S4 coverage |
| Structural stressors | Repeat groups, gates/branches, semantic/free-text, tables, and structural clears where available |
| Reliability risk | Includes ordinary items and stress items likely to expose contract gaps |

Items were excluded from the slice if they were used for calibration, depended
on hidden author intent, had unresolved triage blockers, or had cleaner
alternatives available for the same coverage target.

### 3.3 Labels and Visibility

`reliability_slice_v1` is an administrative label. It is used for export,
coverage accounting, and paper reporting. Annotators must not see selection
labels, stressor labels, intended delta labels, author notes, provisional gold,
forbidden commits, or private design metadata.

---

## 4. Annotation Task

Each annotator receives the identical public packet a model receives:

| Input | Included | Not included |
|-------|----------|--------------|
| Form schema and annotation guidance | Yes | |
| Record context | Yes | |
| Prior state | Yes | |
| Visible history | Yes | |
| Current utterance | Yes | |
| Provisional author gold | | Excluded |
| Other annotator output | | Excluded |
| Forbidden commits/mutations | | Excluded |
| Stressor metadata | | Excluded |
| Item title or design notes | | Excluded |

### 4.1 Annotator Output

For each item, the annotator produces:

1. **Resulting state annotation**: the record after applying the current
   utterance and visible history under the schema.
2. **Non-preserve transitions**: inferred by the UI from field edits as `set`,
   `change`, `clear`, or `structural_clear`.
3. **Evidence spans** for every non-preserve transition.
4. **Repeat-routing decisions** when a repeated entity is edited.
5. **Gate/branch confirmations** when a controller creates structural
   consequences.
6. **Validity flags and notes** when the public packet is ambiguous,
   incoherent, underspecified, or dependent on hidden author intent.

Annotators do **not** create item-level IU gold. For semantic/free-text fields,
they enter the resulting field value and attach evidence. Item-level IU gold,
when needed for scoring, is created or verified later during adjudication.

---

## 5. Agreement Metrics

### 5.1 Primary Metrics

Agreement is reported with the same metric families used for model evaluation:

| Level | Metric | Notes |
|-------|--------|-------|
| Closed/typed fields | Exact agreement | Identical normalized values over scoreable closed/typed fields |
| All scoreable fields | Micro F1 | Symmetric pairwise scoring with the benchmark scorer |
| Changed fields | Changed-field precision, recall, F1 | Restricted to fields changed by at least one annotator |
| Transitions | Cohen's kappa | Over inferred `set`, `change`, `clear`, `structural_clear`, and preserve decisions |
| Repeat instances | Alignment agreement | Uses the benchmark repeat-instance alignment logic |
| Semantic/free-text fields | IU or normalized-text agreement | Uses the final scorer policy for that field |
| Whole record | Exact match | Harsh full-record comparison |

### 5.2 Secondary Metrics

Report or appendix metrics:

- per-form agreement;
- per-delta-type agreement;
- per-state-condition agreement;
- preservation agreement;
- collateral edit rate;
- validity-flag frequency;
- disagreement taxonomy after adjudication.

### 5.3 Paper Interpretation

The same evaluation metrics are applied to annotators and systems. This makes
IAA a direct human agreement reference for model results: if annotators agree at
95 changed-field F1 and the best system reaches 81, the gap is interpretable as
remaining headroom under the public annotation protocol.

---

## 6. Publication Targets

| Metric | Target | If not met |
|--------|--------|------------|
| Closed/typed field agreement | >= 95% | Identify contract/schema gaps and report transparently |
| Changed-field F1 | >= 90% | Inspect disagreements by item, form, and transition type |
| Transition kappa | >= 0.85 | Report per-label breakdown |
| Repeat-instance alignment | >= 90% | Check stable keys and routing guidance |
| Preservation agreement | >= 95% | Investigate unsupported collateral edits |

If targets are not met, do not silently tune the same slice. Classify the
disagreements, revise the public contract if needed, and either report the
initial pass transparently or re-run a fresh reliability slice.

---

## 7. Adjudication Protocol

1. Compute pairwise agreement per item and per field.
2. For each disagreement, classify it using the discrepancy taxonomy:
   `annotator_error`, `contract_gap`, `schema_gap`, `normalization_gap`,
   `semantic_iu_gap`, `hidden_author_intent`, `author_gold_revision`,
   `item_invalid_for_blind_scoring`, or another documented label from
   `DISCREPANCY_DISCUSSION_PROTOCOL_v0.9.2.md`.
3. Resolve disagreements already settled by the public rules.
4. Record genuine ambiguities and decide whether to revise or exclude the item.
5. Set final gold to adjudicated consensus, not automatically to provisional
   author gold.

Paper reporting should include total disagreements, disagreement labels, item
discard/revision counts, and author-gold revision counts.

### 7.1 Escalation Rules

The adjudication pass is disagreement-only: every review-required row receives a
logged decision, but only rows that require substantive judgment are escalated to
secondary discussion or audit. Clear cases may be resolved by the adjudicator
from the public packet, field guidance, and the annotation contract without a
synchronous annotator meeting.

Rows should be escalated when:

- all three sources disagree or no simple two-against-one pattern is available;
- the two annotators agree with each other but their shared answer is not
  supported by explicit public evidence;
- the adjudicator judges an apparent majority answer to conflict with the
  field semantics, schema constraints, or current utterance;
- the disagreement exposes a possible contract, schema, normalization, or
  semantic-IU ambiguity;
- item validity, exclusion, or gold revision is uncertain after inspecting the
  public packet.

Rows need not be escalated when the public packet and field contract settle the
case directly, including clean two-against-one cases, clear annotator omissions,
minor normalization-only differences, and queue false positives. These rows are
still recorded in the adjudication sidecar with a category and a short
evidence-grounded note. Majority agreement is therefore treated as a triage
signal, not as an automatic gold decision.

### 7.2 Specification Cleanup Cases

If adjudication reveals that an item IU, gold note, or field-local policy uses
`unknown / null`, `unknown/null`, or similar wording to denote the empty default
state, treat this as an authoring/specification bug rather than an annotator
error. The intended semantics are:

- `null` means no value is established by the public packet, or the field is
  inapplicable under the current branch;
- `Unknown` is a literal field value only when the user or context explicitly
  establishes that the value is unknown;
- `unknown / null` must not be read as licensing either literal `Unknown` or
  `null` interchangeably.

Affected rows should be classified as a contract/schema or authoring cleanup
case, and corrected before the benchmark release and model evaluation. The raw
IAA snapshot remains unchanged, while the final released benchmark should use
unambiguous wording such as `null if not specified` or `Unknown only if
explicitly stated`.

### 7.3 Repair Accounting and Evaluator Validation

Adjudication decisions and benchmark repairs serve different purposes and must
be accounted for separately.

**Semantic gold revisions** change the intended benchmark answer because the
public packet supports a different resulting state than the provisional author
gold. These count as author-gold revisions in paper reporting.

**Representation repairs** fix stale or inconsistent benchmark encoding without
changing the intended task semantics. Examples include stale enum labels in
prior or gold state, legacy option names after a questionnaire migration, and
`unknown / null` wording that accidentally conflates an empty UI state with a
literal `Unknown` option. These repairs must not be counted as author-gold
revisions, annotator errors, or substantive IAA disagreements. They should be
tracked in a repair ledger with the item id, field path, old value, new value,
repair type, rationale, and whether the row is excluded from annotator-error
attribution.

When a representation repair affects both the prior state and the resulting
gold state, the corrected benchmark should preserve the field if the public task
did not require a semantic change. Do not resolve such cases by selecting an
annotator value in the IAA editor if that would record a misleading
`prior -> final` transition. Instead, record the adjudication row as a
schema/contract cleanup case and apply the representation repair as a separate
benchmark-maintenance action.

Annotator performance summaries should distinguish:

- benchmark-contract-adjusted pairwise agreement used as the paper-facing
  human reliability number when documented UI/IU or representation defects
  created false-positive disagreements;
- raw pre-adjudication pairwise agreement retained as an audit trail;
- adjudicated annotator errors on rows where the public packet and schema were
  valid;
- accepted normalization or semantic-granularity variants;
- rows excluded from annotator-error attribution because of representation
  repairs, contract gaps, schema gaps, or item invalidity.

The evaluator is validated against the adjudication ledger after repairs are
applied. For each adjudicated row, the scorer should reproduce the intended
classification under the final policy: exact/acceptable/partial/incorrect for
semantic fields, legal enum equivalence for closed fields, and no penalty for
rows explicitly excluded as benchmark-specification defects. Any mismatch
between adjudication and evaluator output is treated as an evaluator audit
finding: fix clear scorer-policy or implementation defects before model
evaluation, and document residual limitations rather than changing the
adjudication ledger. The evaluator must not be used to rewrite the raw
pre-adjudication IAA snapshot, but paper-facing IAA may include a documented
benchmark-contract adjustment that removes rows later classified as UI/IU or
representation false positives.

Evaluator-validation reports should separate two directions of reliability:

- **Detection of adjudicated errors**: among annotator-side checks that human
  adjudication says are wrong, how often does the evaluator also mark the side
  non-correct or hard incorrect?
- **False positives against accepted sides**: among adjudicated-correct sides,
  accepted normalization variants, accepted semantic variants, and
  author-revision source sides, how often does the evaluator incorrectly mark
  the side non-correct or hard incorrect?

For declared cascade failures, the evaluator may also report active
root-cause/grouped-error credit. This is a diagnostic view only: field-level
verdicts and standard metrics remain strict, while grouped counts explain how
many field mismatches arise from one declared root cause. Grouped credit should
be used only for items with explicit `evaluation_error_groups` metadata and
reported alongside exact field-level detection rates.

### 7.4 Semantic False-Positive Flags

Some rows are flagged for review because exact or shallow-normalized strings
diverge even though the field-local semantic commitment is the same. These
cases are adjudicated as semantic false positives, not as annotator errors or
author-gold revisions.

A row may receive this treatment when:

- all submitted variants are grounded in the same public evidence;
- the differences are wording, granularity, or conventional phrasing within the
  same field commitment;
- no clinically or task-relevant distinction is lost under the field guidance;
- the canonical gold can remain as the benchmark wording without changing the
  intended answer.

These rows should be logged as `normalization_only` with discrepancy category
`normalization_gap`, and may additionally carry a `semantic_false_positive`
marker in the adjudication ledger. They are excluded from annotator-error and
author-gold-revision counts. They remain useful for evaluator validation: the
final scorer should accept the adjudicated variants as semantically correct.

---

## 8. Paper Presentation

Suggested Section 3.3 framing:

> To establish annotation reliability, eight non-author annotators independently
> produced two resulting states for each item in a 60-item reliability slice,
> with 15 items and two annotators per form stratum. The slice covers delta
> types, state conditions, and structural stressors across four forms.
> We report agreement using the same metrics applied to model evaluation,
> providing a direct human agreement reference for the benchmark.

Key claims to support:

1. The task is recoverable from public packets and public rules.
2. Reliability is measured on a named, reproducible slice.
3. Human agreement is reported in the same units as model performance.
4. Disagreements are not hidden; they feed adjudication and benchmark cleanup.

---

## 9. Execution Record

- [x] `reliability_slice_v1` selected: 60 items, 15 per form.
- [x] Released item IDs, manifest, item table, coverage summary, and lock record verified against the frozen packet set.
- [x] Blind annotation export uses only the 60 locked item packets.
- [x] Blind annotation export contains questionnaires and contexts for all four forms.
- [x] Blind annotation export removes old blind annotations before launch.
- [x] Public packets exclude provisional gold and private author metadata.
- [x] Eight non-author annotators completed the blind pass, two per form stratum.
- [x] Agreement computation and adjudication were completed after returns.
