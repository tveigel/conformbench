# Diff-then-Verify Experiment

This protocol documents the frozen Diff-then-Verify experiment without
changing the published Update Tool reference run.

## Fixed design

- Stage 1 is the existing `conformbench.systems.flatagent` Update Tool solver.
- Its sparse updates are materialized into a complete candidate record.
- Stage 2 is one fresh inference call to the same `gpt-5.4-mini` generator.
- Both stages use the same generation settings as the original four
  architecture runs: `max_tokens=32000` and reasoning disabled (`none`).
- The verifier sees the schema, original prior record, complete candidate,
  visible history, and current utterance.
- It emits no patch when the candidate is supported, or one sparse corrective
  patch through the same update tool when correction is needed.
- Evaluation uses the released `gpt-5.4` evaluator path. For compatibility,
  the runner retains the historical `medium` configuration label. The released
  provider traces resolve to `gpt-5.4-2026-03-05` with zero reasoning tokens,
  and the checked-in official-OpenAI helper removes `reasoning_effort` before
  transmission. The frozen Diff-then-Verify run intentionally preserves that effective path
  instead of silently changing evaluator behavior.
- Generation timings use monotonic wall time inside the solver and exclude the
  evaluator. The same-run first stage is the controlled Update Tool latency
  reference.
- The comparison manifest's `cost_tracking_max_tokens=16000` field is only the
  conservative per-call ceiling used by pre-run cost guards. It is not an LLM
  request setting; the released per-call `model_config` traces record the
  actual `max_tokens=32000` generation limit.

## Guarded workflow

From the software root:

```bash
export CONFORMBENCH_DATA_DIR=/absolute/path/to/conformbench-data-v1.0.0
uv sync --frozen
uv run python scripts/run_diff_then_verify_experiment.py preflight
```

Preflight checks solver imports, model routing, the 36/144 frozen split, and the
published 144-item Update Tool report. It makes no LLM calls and does not need
an API key.

After adding `OPENAI_API_KEY` to `.env`, run development only:

```bash
uv run python scripts/run_diff_then_verify_experiment.py dev
```

Inspect the completed 36-item run under
`$CONFORMBENCH_DATA_DIR/reports/rebuttal_runs/diff_then_verify/dev/`. Prompt
changes are permitted only before freezing. Once the chosen development run is
satisfactory, freeze it:

```bash
uv run python scripts/run_diff_then_verify_experiment.py freeze \
  --dev-run arr_diff_then_verify_dev_final_20260710__diff_then_verify__gen-gpt-5-4-mini__judge-gpt-5-4-medium
```

Freezing records SHA-256 digests for the solver, shared Update Tool, prompts,
runner, evaluator, metrics, and summarizer. The held-out command refuses to run
if any frozen source changes afterward.

Run the 144 held-out items once:

```bash
uv run python scripts/run_diff_then_verify_experiment.py test
```

The fixed test path prevents a second completed run. If execution is
interrupted, the same command resumes only missing items. On completion it
writes a one-shot receipt and automatically produces:

- `protocol/author_comment_results.json` (historical filename)
- `protocol/author_comment_results.md` (historical filename)
- a concise results paragraph

The result includes exact records, strict changed-field F1, missed supported
updates, unsupported commitments, the paired item-bootstrap difference and 95%
interval against the published 43/144 Update Tool run, and median total,
first-stage, and verification-stage latency.

Use the status command at any point:

```bash
uv run python scripts/run_diff_then_verify_experiment.py status
```

## Released evidence and offline analysis

The released data archive contains the selected development run, frozen protocol
records, and completed held-out run. The run uses the frozen prompts, source,
benchmark data, evaluator, item order, and common generator configuration. The
receipt preserves the runtime frozen-config hash and normalized archive hash.

Do not rerun the held-out command when inspecting the released results. The
saved run can be re-analysed without an API key using:

```bash
export CONFORMBENCH_DATA_DIR=/absolute/path/to/conformbench-data-v1.0.0
uv run python scripts/summarize_five_system_results.py
uv run python scripts/bootstrap_architecture_robustness.py
uv run python scripts/analyze_load_normalized_exact.py
uv run python scripts/analyze_scoring_route_sensitivity.py
```

The experiment index and generated summaries are under
`$CONFORMBENCH_DATA_DIR/reports/rebuttal_runs/diff_then_verify/` in the data
archive.
