# ConFormBench Package

Start with the repository-level [README](../README.md).

Most developers only need:

```text
systems/YOUR_SYSTEM.py
benchmark.py
```

Data is distributed separately on Zenodo. Set `CONFORMBENCH_DATA_DIR` to the
extracted dataset root, which contains:

```text
$CONFORMBENCH_DATA_DIR/schema/
$CONFORMBENCH_DATA_DIR/items/
$CONFORMBENCH_DATA_DIR/reports/
```

The runner validates returned state shape but does not repair the candidate
state. Scoring uses the state returned by `solve(turn)`.
