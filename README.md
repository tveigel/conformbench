# ConFormBench

ConFormBench evaluates whether a conversational form-filling system maintains a
complete structured record whose commitments are supported by the form schema
and dialogue history. Each benchmark turn provides a schema, prior record,
visible dialogue history, and current user utterance. A system returns the full
post-turn record.

This repository contains the public runner, evaluator, baseline systems,
analysis scripts, and ConFormBench Studio. The benchmark data and frozen paper
artifacts are released separately in the versioned
[ConFormBench dataset on Zenodo](https://doi.org/10.5281/zenodo.22166024).

## Installation

ConFormBench requires Python 3.11 or newer.

```bash
git clone https://github.com/tveigel/conformbench.git
cd conformbench
python -m pip install -e .
```

Download and extract `conformbench-data-v1.0.0.zip` from the Zenodo dataset.
Then point the software to the extracted dataset root, which directly contains
`schema/`, `items/`, `reports/`, and `splits/`:

```bash
export CONFORMBENCH_DATA_DIR=/absolute/path/to/conformbench-data-v1.0.0
python -m conformbench list-questionnaires
python -m conformbench list-public-items
```

The environment variable keeps software and data independent. If it is not
set, ConFormBench falls back to a local `./data` directory for development
convenience.

## Run a System

Start from the included skeleton:

```text
conformbench/systems/YOUR_SYSTEM.py
```

Implement `solve(turn)`, then run:

```bash
python -m conformbench run \
  --items public \
  --solver conformbench.systems.YOUR_SYSTEM:solve \
  --output-dir runs/your_system \
  --score
```

`solve(turn)` receives:

```text
turn.item_id
turn.questionnaire_id
turn.schema
turn.prior_state
turn.visible_history
turn.current_utterance
turn.metadata
```

It returns one JSON object containing the complete post-turn record. The runner
validates the returned shape, stores a diagnostic diff, and scores the returned
state as-is.

## Returned State Shape

- Top-level keys exactly match schema field IDs and repeat-group IDs.
- Every top-level schema field and repeat-group ID is present.
- Repeat groups are arrays of row objects, with any row count allowed.
- Every repeat row contains exactly that repeat group's child field IDs.
- Table fields are arrays of row objects, with any row count allowed.
- Every table row contains exactly that table's column IDs.
- Unknown scalar values use `None`; empty repeat and table fields use `[]`.

## Repository Map

```text
conformbench/benchmark.py            public runner
conformbench/systems/YOUR_SYSTEM.py  minimal system skeleton
conformbench/systems/                released baseline systems
conformbench/evaluator/              scoring implementation
conformbench/studio/                 local inspection UI
docs/                                task, annotation, and evaluator docs
scripts/                             report and reproduction utilities
tests/                               offline tests
```

The separate data release contains:

```text
schema/questionnaires/               questionnaire JSON schemas
items/public/                         public development items
items/benchmark/                      complete benchmark item packets
items/splits/                         frozen development/test item trees
reports/                              saved candidates, traces, and analyses
splits/                               schema partitions and partition briefs
```

The released data uses synthetic or pseudonymized form contents. Names, email
addresses, phone numbers, medical-style facts, grant organisations, and
institutional roles inside benchmark items are scenario content, not author or
annotator personal information.

## Paper Reproduction

The paper-facing split is:

```text
$CONFORMBENCH_DATA_DIR/items/splits/conformbench_v1_20_80_seed20260524/
```

Four frozen architecture runs are stored under:

```text
$CONFORMBENCH_DATA_DIR/reports/split_runs/conformbench_v1_20_80_seed20260524/test/
```

The fifth architecture, Diff-then-Verify, is stored under:

```text
$CONFORMBENCH_DATA_DIR/reports/rebuttal_runs/diff_then_verify/test/
```

The historical `rebuttal_runs` directory name is retained so that released
manifests and reproduction scripts continue to resolve their recorded paths.

Each paper run contains `summary_report.json`, `metrics.json`, `results.csv`,
`report.md`, run metadata, evaluator traces, and candidate records. Direct JSON
also includes the manifests for its documented retry-policy and fallback
repairs.

Reproduce the saved five-system analyses without an API key or new model calls:

```bash
mkdir -p reproduced
python scripts/summarize_five_system_results.py \
  --output-dir reproduced/five-system
python scripts/bootstrap_architecture_robustness.py \
  --output-dir reproduced/bootstrap
python scripts/analyze_load_normalized_exact.py \
  --output-dir reproduced/load-normalized
python scripts/analyze_scoring_route_sensitivity.py \
  > reproduced/scoring-route-sensitivity.md
```

The guarded Diff-then-Verify workflow has an offline preflight that validates
the released 36/144 split and archived baseline artifacts without making model
calls:

```bash
python scripts/run_diff_then_verify_experiment.py preflight
```

The completed held-out experiment is already included in the data release and
must not be rerun to reproduce the paper. Its development/freeze/test protocol
is documented in `docs/DIFF_THEN_VERIFY_EXPERIMENT.md`.

Run the offline test suite with:

```bash
python -m pip install pytest
pytest tests
```

## ConFormBench Studio

Start the inspection and extension interface with:

```bash
python -m conformbench studio
```

The Studio reads the dataset from `CONFORMBENCH_DATA_DIR`. Its **How-To** tab
describes how to inspect items and results, add a questionnaire, author item
packets, and run or score a custom system.

## Data and Trace Scope

Generation artifacts include candidate records, raw model responses, model-call
metadata, and source prompt templates or builders. Full generation input-message
transcripts are present only for runs created with
`CONFORMBENCH_TRACE_MESSAGES=1`. Evaluator LLM prompt traces are stored in
`evaluation.json` whenever evaluator model calls occurred.

External source PDFs and forms are not redistributed. Source grounding is
documented through schema `references`, paper citations and provenance text,
and the released derived questionnaire schemas.

## Citation

Please cite the ConFormBench paper and the
[versioned Zenodo dataset](https://doi.org/10.5281/zenodo.22166024) when using
the benchmark.

## License

The runner, evaluator, baseline systems, scripts, and ConFormBench Studio are
released under the MIT License; see `LICENSE`. The companion benchmark data,
schemas, documentation, and audit artifacts are released under CC BY 4.0 in
the Zenodo data record.
