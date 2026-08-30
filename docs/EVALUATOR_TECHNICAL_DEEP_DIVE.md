# Evaluator Technical Deep Dive

Status: technical implementation note, current as of 2026-05-22.

This document describes the evaluator at the level needed to verify it.
The concept is simple: we score the final record field by field.

## Core Model

For each turn the evaluator compares three record states:

```text
P = prior state before the user utterance
G = gold resulting state after the user utterance
C = candidate resulting state produced by the system
```

The evaluator asks whether `C` makes the same commitments as `G`, while
preserving every field that `G` preserves.

The scorer does not grade the assistant's prose. It grades the materialized
post-turn record.

## Runtime Inputs And Outputs

Each evaluated turn has:

| Artifact | Purpose |
| --- | --- |
| `turn_result.json` | candidate resulting state plus diagnostic metadata |
| `ground_truth.json` | prior state, gold state, field policies, repeat specs, semantic IUs |
| `evaluation.json` | final field verdicts, alignment logs, diagnostics, metric views, prompt traces |

Important code anchors:

| Concern | Source |
| --- | --- |
| turn evaluator | `conformbench/evaluator/run_turn.py::evaluate_turn_result` |
| deterministic scorer | `conformbench/evaluator/hard.py::evaluate_hard_fields` |
| semantic scorer | `conformbench/evaluator/soft.py::evaluate_soft_fields` |
| repeat matcher | `conformbench/evaluator/alignment.py::match_instances` |
| diagnostics | `conformbench/evaluator/diagnostics.py::build_diagnostics` |
| metric views | `conformbench/evaluator/metric_views.py::compute_turn_metric_views` |

## Step 1: Read The Candidate State

The public solver API returns `C` directly. The runner checks that `C` is a JSON
object whose shape matches the questionnaire schema: all top-level schema fields
and repeat groups must be present, repeat groups and table fields must be lists,
repeat rows must contain the repeat group's child fields, and table rows must
contain the table's columns. The only flexible cardinality is the number of
repeat instances or table rows.

The runner does not materialize `C` from operations, repair missing fields, or
preserve omitted values from the prior state. Diagnostic operations may be
computed after the fact for reports, but they are not applied back to `C`.

## Step 2: Align Repeat Instances

Before scoring repeat children, the evaluator aligns gold rows to candidate
rows. This is the only place where row order and row-key paraphrases are
resolved.

The instance matcher receives the prior rows, gold rows, candidate rows,
visible history, current utterance, and alignment-key metadata. It returns a
one-to-one mapping:

- matched rows are scored field by field;
- unmatched gold rows become missing-instance errors;
- unmatched candidate rows become spurious-instance errors.

The alignment result is saved in `evaluation.json.alignment_log`. Alignment
LLM calls are saved in `evaluation.json.prompt_trace`.

## Step 3: Score Every Field

After repeat alignment, every top-level field and every matched repeat child is
scored independently.

The scoring route is:

```text
1. If gold requires a changed resulting value but candidate preserved prior:
       incorrect

2. Else if candidate exactly matches gold after normalization:
       correct

3. Else if the field can be decided deterministically:
       use deterministic scorer

4. Else:
       escalate to an LLM comparator
```

The preserve-miss rule is intentionally first. If `G != P` but `C == P`, the
system missed the requested update. This is incorrect for all field types,
including semantic IU fields.

Implementation anchors:

- preserve-miss guard:
  `conformbench/evaluator/run_turn.py::_candidate_preserved_prior_when_gold_changed`
- preserve-miss verdict:
  `conformbench/evaluator/run_turn.py::_preserved_prior_miss_verdict`
- exact/open mismatch escalation:
  `conformbench/evaluator/hard.py::_should_escalate_exact_mismatch`

## Deterministic Scoring

The deterministic scorer handles cases where exact comparison is reliable:

- booleans;
- numbers;
- dates;
- closed choices;
- empty gold plus empty candidate;
- normalized string equality;
- accepted deterministic alternatives;
- unordered set/list matching for controlled lists.

If deterministic scoring accepts the field, the verdict is `correct` and no LLM
is called.

For `set_match`, deterministic scoring can also return `partially_correct`
when the candidate supplies some useful set content but misses required items
or adds disallowed extras.

If deterministic scoring rejects a closed field, the verdict is `incorrect`.

If the field is an eligible open text, `Other`-style, or semantic-IU field, the
exact check is only a fast path. A non-empty mismatch escalates to the LLM
comparator. Semantic fields without item-level IUs are not automatically
scored by a generic judge; they must be exact/open mismatch checks, schema-alias
checks, or IU-backed fields.

`partially_correct_values` are retired. The evaluator no longer awards partial
credit because a candidate exactly matches a listed partial phrase.

## LLM Comparator

The comparator is used only after deterministic checks fail and the field is
eligible for semantic scoring. It receives:

- current utterance;
- visible conversation history;
- prior field value;
- candidate field value;
- gold field value;
- field policy and scoring notes;
- semantic IUs when the item defines them.

The comparator route produces one field verdict:

```text
correct | partially_correct | incorrect
```

For non-IU fields the model returns that verdict directly. For IU fields the
model returns structured IU facts, and Python deterministically derives the
verdict from those facts.

The evaluator stores the source and reasoning in `evaluation.json.field_results`.

There are two comparator styles.

## Comparator Without IUs

There are only two non-IU comparator routes:

- exact/open mismatches that the hard scorer explicitly marked for semantic
  escalation;
- narrow schema-alias checks.

These are usually easy policy checks: the comparator decides whether the
candidate is semantically equivalent to the gold, partially acceptable, or
wrong under the field policy. This route does not rescue preserve misses; those
were already scored incorrect.

Typical examples:

- same meaning, different phrasing -> `correct`;
- mostly right but over-specified or missing a minor required part -> `partially_correct`;
- unsupported, contradictory, or wrong field value -> `incorrect`.

Decision sources include:

- `semantic_exact_mismatch_judge`;
- `semantic_alias_judge`.

## Comparator With IUs

For semantic IU fields, the comparator checks each required information unit.
The model reports simple facts; Python applies the final rule.

For each IU:

```text
coverage = full | partial | none
updated_to_match = true | false
improved_over_prior = true | false
contradicted = true | false
candidate_span = "..."
```

For the whole field:

```text
improvement_over_prior = true | false
unsupported_extra = true | false
field_irrelevant_extra = true | false
```

Then the evaluator derives the verdict:

| Condition | Verdict |
| --- | --- |
| any required IU is contradicted | `incorrect` |
| all required IUs are fully covered and there is no bad extra | `correct` |
| all required IUs are fully covered but grounded irrelevant/dumped content is present | `partially_correct`, over-specified |
| not all IUs are covered, but at least one IU was newly added or improved this turn | `partially_correct` |
| not all IUs are covered, but the field is closer to gold than preserving `P` would have been | `partially_correct` |
| unsupported extra appears together with otherwise partial progress | `partially_correct`, mixed |
| no IU progress, no field-level improvement, or unsupported-only answer | `incorrect` |

Key definitions:

- `coverage = partial`: the candidate contains a useful part of one IU but
  misses a required subpart, for example a witness name without required
  contact or observation details.
- `improvement_over_prior`: the update made the field more accurate than
  simply preserving the prior value, for example removing a hallucinated car
  description while still missing other required accident details.
- `unsupported_extra`: the candidate adds unlicensed or fabricated facts.
- `field_irrelevant_extra`: the candidate adds grounded information, but it is
  misplaced, dumped, or harmful to this field's usefulness.

Decision source:

```text
semantic_iu_judge
```

Implementation anchors:

- prompt: `conformbench/evaluator/soft.py::_semantic_iu_system_prompt`
- call: `conformbench/evaluator/soft.py::_evaluate_semantic_iu_fields`
- verdict derivation:
  `conformbench/evaluator/soft.py::_semantic_iu_verdict_from_entry`

## Diagnostics And Metrics

After field verdicts are known, diagnostics compare `P -> G` and `P -> C`.
They count things like missed updates, unsupported commits, collateral edits,
failed corrections, repeat alignment errors, and forbidden commits.

Diagnostics do not rewrite field verdicts. They are accounting views over the
verdicts.

Paper-facing metrics are derived from the field verdicts in
`conformbench/evaluator/metric_views.py`.

## Decision Sources To Expect

Current paper-scoring artifacts should use these main decision sources:

| Decision source | Meaning |
| --- | --- |
| `hard_eval` | deterministic scorer decided the field |
| `hard_preserve_miss` | candidate preserved prior while gold changed |
| `semantic_exact_fast_path` | semantic field matched gold exactly after normalization |
| `semantic_empty_gold_fast_path` | semantic field resolved deterministically because gold was empty |
| `semantic_exact_mismatch_judge` | open/exact mismatch escalated to comparator |
| `semantic_iu_judge` | IU comparator was called |
| `semantic_iu_metadata` | accepted semantic variant matched directly |
| `semantic_alias_judge` | schema-supported alias was accepted |
| `instance_alignment` | repeat instance was missing or spurious |



## Verification Checks

Run focused evaluator tests:

```bash
uv run python -m unittest tests.test_turn_evaluate tests.test_summary_report
```

Check removed legacy decision sources:

```bash
rg -n '"decision_source"\s*:\s*"semantic_judge(_error)?"' \
  "$CONFORMBENCH_DATA_DIR/eval_runs" --glob evaluation.json
```

Summarize decision sources in a run:

```bash
python - <<'PY'
import json
import os
from collections import Counter
from pathlib import Path

run = Path(os.environ["CONFORMBENCH_DATA_DIR"]) / "eval_runs" / "<run-id>"
counts = Counter()
for path in run.rglob("evaluation.json"):
    payload = json.loads(path.read_text())
    for verdict in (payload.get("field_results") or {}).values():
        counts[verdict.get("decision_source")] += 1
for key, value in counts.most_common():
    print(f"{key}: {value}")
PY
```

Inspect one evaluated turn:

```bash
jq '{summary, diagnostics: .diagnostics.counts, field_results}' \
  data/eval_runs/<run-id>/runs/<agent>/<questionnaire>/<scenario>/<state>/<utterance>/evaluation.json
```
