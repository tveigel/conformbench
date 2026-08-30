# Public Runner API

This document specifies the developer-facing API for running ConFormBench in an
open source repository. The public API is independent of any particular form
renderer, flow builder, agent framework, edit script, or LLM provider.

The benchmark evaluates one thing: the resulting record state after a public
conversation turn.

## Design Goals

- Make benchmark runs reproducible from a clean checkout.
- Let developers plug in any form-filling logic with minimal glue code.
- Use a state-in/state-out contract: no required edit API.
- Avoid hidden state representations such as UI-only open sentinels.
- Do not repair, fill, or normalize a system's returned state before scoring.
- Produce portable artifacts that can be inspected without the runner.

## Public Entry Point

The package exposes a single high-level runner:

```python
from conformbench import benchmark


def solve(turn):
    # turn.current_utterance: str
    # turn.visible_history: list[dict]
    # turn.prior_state: nested JSON record before the turn
    # turn.schema: questionnaire schema
    resulting_state = run_your_system(
        schema=turn.schema,
        prior_state=turn.prior_state,
        visible_history=turn.visible_history,
        current_utterance=turn.current_utterance,
    )
    return resulting_state


result = benchmark.run(items=items, solver=solve)
```

Items can be loaded from explicit public item packets:

```python
from conformbench import discover_items

items = discover_items("data/items/benchmark/patient_admin")
result = benchmark.run(items=items, solver=solve)
```

## Turn Object

Each solver receives a public turn object with these attributes:

```python
turn.item_id: str
turn.questionnaire_id: str
turn.schema: dict
turn.prior_state: dict
turn.visible_history: list[dict]
turn.current_utterance: str
turn.metadata: dict
```

`turn.form_schema` and `turn.public_state` are legacy aliases for
`turn.schema` and `turn.prior_state`.

The turn packet must contain all information needed to solve the item. If a
required rule is not visible in `schema`, `visible_history`,
`current_utterance`, or `prior_state`, the item is not suitable for a public
benchmark.

## Returned State

The solver returns a JSON object representing the candidate resulting record.
Its shape must match the questionnaire schema exactly, except repeat groups and
table fields may contain any number of row objects, including zero.

The runner performs shape checks only:

- the returned value must be a dict;
- top-level keys must be exactly the schema's field ids and repeat-group ids;
- every top-level schema field and repeat-group id must be present;
- repeat groups must be lists of row objects, with any row count allowed;
- every repeat-group row must contain exactly that repeat group's child field ids;
- table fields must be lists of row objects, with any row count allowed;
- every table row must contain exactly that table's column ids;
- table fields and repeat groups use `[]` when they have no rows.

The runner does not add missing fields, preserve omitted fields, coerce value
types, sort rows, infer deletes, or apply an operation stream. The evaluator
scores the returned state as returned.

The shape check is intentionally not a value-type checker. For example, it
checks that a table is a list of row objects with the right columns, but it does
not decide whether a scalar field's value is semantically correct.

## Optional Metadata

For run provenance, a solver may return a wrapper:

```python
def solve(turn):
    return {
        "resulting_state": {...},
        "provenance": {"generation": {"agent": "my_system"}},
        "agent_response": {"raw_response": "..."},
    }
```

This wrapper is optional. The simple `return state` form is the primary public
contract.

## Diagnostic Operations

After a valid state is returned, the runner computes a diagnostic diff from
`prior_state` to `candidate_state`. This diff is written into `turn_result.json`
for inspection and reports. It is not applied back to construct the candidate
state and is not part of the participant-facing contract.

## Runner Output

Each item directory contains:

```text
<questionnaire>/<scenario>/<state>/<utterance>/turn_result.json
<questionnaire>/<scenario>/<state>/<utterance>/evaluation.json
```

`turn_result.json` contains the public turn packet plus `answers_before`,
`answers_after`, and diagnostic operation metadata. Scoring uses
`answers_after`.

## Minimal Developer Workflow

```bash
uv sync
uv run python -m conformbench run --items public --solver conformbench.systems.YOUR_SYSTEM:solve
```

Data layout:

```text
data/schema/schema.md
data/schema/questionnaires/*.json
data/items/public/<questionnaire>/<item>/ground_truth.json
data/items/benchmark/<questionnaire>/<item>/ground_truth.json
```

The same run should also be possible from Python:

```python
result = benchmark.run(
    items=items,
    solver=solve,
    output_dir="runs/my-run",
)
print(result["item_count"])
```

## Non-Goals

- The public API does not require a conversational UI.
- The public API does not require the benchmark's internal flow builder.
- The public API does not require participants to use tool calls.
- The public API does not require participants to emit operations.
- The public API does not score intermediate questions asked by an agent.
- The public API does not expose private annotation artifacts.
