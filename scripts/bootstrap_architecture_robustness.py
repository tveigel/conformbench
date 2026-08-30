#!/usr/bin/env python3
"""Paired item-level bootstrap checks for held-out architecture comparisons.

This script reads existing ``summary_report.json`` files. It does not rerun
systems or the evaluator.

Default usage from the software repository root, with
``CONFORMBENCH_DATA_DIR`` pointing to the extracted dataset:

    python scripts/bootstrap_architecture_robustness.py

Custom runs:

    python scripts/bootstrap_architecture_robustness.py \
      --run "Direct JSON=data/reports/split_runs/.../summary_report.json" \
      --run "Update Tool=data/reports/split_runs/.../summary_report.json"
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import random
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get("CONFORMBENCH_DATA_DIR", ROOT / "data")).expanduser().resolve()
RUNS_DIR = (
    DATA_ROOT
    / "reports"
    / "split_runs"
    / "conformbench_v1_20_80_seed20260524"
    / "test"
)
DEFAULT_OUTPUT_DIR = (
    DATA_ROOT
    / "reports"
    / "robustness"
    / "five_system_bootstrap_seed20260710"
)

DEFAULT_RUNS = {
    "Direct JSON": RUNS_DIR
    / "audit_tight_test_20260525__direct__gen-gpt-5-4-mini__judge-gpt-5-4-medium"
    / "summary_report.json",
    "Update Tool": RUNS_DIR
    / "audit_tight_test_20260525__flatagent__gen-gpt-5-4-mini__judge-gpt-5-4-medium"
    / "summary_report.json",
    "Split Agent": RUNS_DIR
    / "audit_tight_test_20260525__split_agent__gen-gpt-5-4-mini__judge-gpt-5-4-medium"
    / "summary_report.json",
    "Triage-Briefed Split Agent": RUNS_DIR
    / "audit_tight_test_20260525__split_agent_briefed__gen-gpt-5-4-mini__judge-gpt-5-4-medium"
    / "summary_report.json",
    "Diff-then-Verify": DATA_ROOT
    / "reports"
    / "rebuttal_runs"
    / "diff_then_verify"
    / "test"
    / "arr_diff_then_verify_test__diff_then_verify__gen-gpt-5-4-mini__judge-gpt-5-4-medium"
    / "summary_report.json",
}

METRICS = (
    "exact_record_rate",
    "changed_field_f1",
    "collateral_edits",
)

METRIC_LABELS = {
    "exact_record_rate": "Exact record rate",
    "changed_field_f1": "Changed-field F1",
    "collateral_edits": "Collateral edits",
}

METRIC_UNITS = {
    "exact_record_rate": "rate",
    "changed_field_f1": "rate",
    "collateral_edits": "edits per test set",
}

ITEM_KEY_FIELDS = ("questionnaire", "scenario_id", "state", "utterance_id")


@dataclass(frozen=True)
class ItemMetrics:
    exact_record: int
    changed_correct: int
    predicted_changed: int
    gold_changed: int
    collateral_edits: int
    preserved_total: int


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def _f1(tp: int, predicted: int, gold: int) -> float:
    precision = _safe_div(tp, predicted)
    recall = _safe_div(tp, gold)
    return _safe_div(2 * precision * recall, precision + recall)


def _item_key(utterance: dict[str, Any]) -> str:
    parts = [str(utterance.get(field) or "") for field in ITEM_KEY_FIELDS]
    return "/".join(parts)


def _load_run_items(summary_path: Path) -> dict[str, ItemMetrics]:
    summary = _read_json(summary_path)
    items: dict[str, ItemMetrics] = {}
    for utterance in summary.get("utterances") or []:
        key = _item_key(utterance)
        views = utterance.get("metric_views") or {}
        exact = views.get("whole_record_exact_match") or {}
        changed = views.get("changed_fields") or {}
        preservation = views.get("preservation") or {}
        if key in items:
            raise ValueError(f"Duplicate item key in {summary_path}: {key}")
        items[key] = ItemMetrics(
            exact_record=1 if exact.get("exact_match") else 0,
            changed_correct=int(changed.get("changed_correct", 0) or 0),
            predicted_changed=int(changed.get("predicted_changed_total", 0) or 0),
            gold_changed=int(changed.get("gold_changed_total", 0) or 0),
            collateral_edits=int(preservation.get("collateral_edit_count", 0) or 0),
            preserved_total=int(preservation.get("preserved_total", 0) or 0),
        )
    if not items:
        raise ValueError(f"No utterance rows found in {summary_path}")
    return items


def _metric(rows: list[ItemMetrics], metric: str) -> float:
    if metric == "exact_record_rate":
        return _safe_div(sum(row.exact_record for row in rows), len(rows))
    if metric == "changed_field_f1":
        return _f1(
            sum(row.changed_correct for row in rows),
            sum(row.predicted_changed for row in rows),
            sum(row.gold_changed for row in rows),
        )
    if metric == "collateral_edits":
        return float(sum(row.collateral_edits for row in rows))
    raise KeyError(f"Unknown metric: {metric}")


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("Cannot compute a percentile of an empty list")
    sorted_values = sorted(values)
    index = (len(sorted_values) - 1) * percentile / 100.0
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = index - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def _ci(values: list[float], alpha: float) -> dict[str, float]:
    return {
        "low": _percentile(values, alpha / 2.0 * 100.0),
        "high": _percentile(values, (1.0 - alpha / 2.0) * 100.0),
    }


def _bootstrap_system(
    rows: list[ItemMetrics],
    *,
    metric: str,
    iterations: int,
    alpha: float,
    rng: random.Random,
) -> dict[str, Any]:
    values: list[float] = []
    n_items = len(rows)
    for _ in range(iterations):
        sample = [rows[rng.randrange(n_items)] for _ in range(n_items)]
        values.append(_metric(sample, metric))
    interval = _ci(values, alpha)
    estimate = _metric(rows, metric)
    return {
        "estimate": estimate,
        "ci_low": interval["low"],
        "ci_high": interval["high"],
    }


def _bootstrap_difference(
    rows_a: list[ItemMetrics],
    rows_b: list[ItemMetrics],
    *,
    metric: str,
    iterations: int,
    alpha: float,
    rng: random.Random,
) -> dict[str, Any]:
    if len(rows_a) != len(rows_b):
        raise ValueError("Paired bootstrap requires equal item counts")
    values: list[float] = []
    n_items = len(rows_a)
    for _ in range(iterations):
        indices = [rng.randrange(n_items) for _ in range(n_items)]
        sample_a = [rows_a[index] for index in indices]
        sample_b = [rows_b[index] for index in indices]
        values.append(_metric(sample_a, metric) - _metric(sample_b, metric))
    interval = _ci(values, alpha)
    estimate_a = _metric(rows_a, metric)
    estimate_b = _metric(rows_b, metric)
    difference = estimate_a - estimate_b
    return {
        "estimate_a": estimate_a,
        "estimate_b": estimate_b,
        "difference": difference,
        "ci_low": interval["low"],
        "ci_high": interval["high"],
        "ci_includes_zero": interval["low"] <= 0.0 <= interval["high"],
    }


def _resolve_summary_path(value: str) -> Path:
    raw_path = Path(value).expanduser()
    candidates = [raw_path]
    if not raw_path.is_absolute():
        candidates.extend([Path.cwd() / raw_path, ROOT / raw_path])

    for candidate in candidates:
        if candidate.is_dir():
            candidate = candidate / "summary_report.json"
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(f"Could not find summary_report.json for: {value}")


def _parse_runs(run_args: list[str]) -> dict[str, Path]:
    if not run_args:
        return {label: path.resolve() for label, path in DEFAULT_RUNS.items()}

    runs: dict[str, Path] = {}
    for arg in run_args:
        if "=" not in arg:
            raise ValueError(
                "--run must have the form 'Label=/path/to/summary_report.json'"
            )
        label, path_text = arg.split("=", 1)
        label = label.strip()
        if not label:
            raise ValueError(f"Missing run label in --run {arg!r}")
        if label in runs:
            raise ValueError(f"Duplicate run label: {label}")
        runs[label] = _resolve_summary_path(path_text.strip())
    return runs


def _parse_pairs(pair_args: list[str], labels: list[str]) -> list[tuple[str, str]]:
    if not pair_args:
        return list(itertools.combinations(labels, 2))

    label_set = set(labels)
    pairs: list[tuple[str, str]] = []
    for arg in pair_args:
        if ":" not in arg:
            raise ValueError("--pair must have the form 'Label A:Label B'")
        left, right = [part.strip() for part in arg.split(":", 1)]
        if left not in label_set or right not in label_set:
            raise ValueError(f"Unknown run label in --pair {arg!r}")
        if left == right:
            raise ValueError(f"Pair compares a run to itself: {arg!r}")
        pairs.append((left, right))
    return pairs


def _aligned_rows(run_items: dict[str, dict[str, ItemMetrics]]) -> tuple[list[str], dict[str, list[ItemMetrics]]]:
    labels = list(run_items)
    key_sets = {label: set(items) for label, items in run_items.items()}
    reference = key_sets[labels[0]]
    mismatches = {
        label: {
            "missing_from_run": sorted(reference - keys),
            "extra_in_run": sorted(keys - reference),
        }
        for label, keys in key_sets.items()
        if keys != reference
    }
    if mismatches:
        preview = json.dumps(mismatches, indent=2)[:2000]
        raise ValueError(f"Runs do not contain the same paired item keys:\n{preview}")

    keys = sorted(reference)
    return keys, {
        label: [items[key] for key in keys]
        for label, items in run_items.items()
    }


def _format_value(metric: str, value: float) -> str:
    if metric in {"exact_record_rate", "changed_field_f1"}:
        return f"{value * 100:.1f}%"
    if metric == "collateral_edits":
        return f"{value:.0f}"
    return f"{value:.4f}"


def _format_diff(metric: str, value: float) -> str:
    if metric in {"exact_record_rate", "changed_field_f1"}:
        return f"{value * 100:+.1f} pp"
    if metric == "collateral_edits":
        return f"{value:+.0f}"
    return f"{value:+.4f}"


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_").lower()


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = report["metadata"]
    confidence_label = f"{metadata['confidence_level'] * 100:.0f}%"
    lines = [
        "# Paired Bootstrap Robustness Check",
        "",
        (
            f"Generated with {metadata['iterations']} paired item-level bootstrap "
            f"samples over {metadata['item_count']} aligned held-out items "
            f"(seed {metadata['seed']})."
        ),
        "",
        "For changed-field F1, each sampled item keeps its changed-field counts "
        "together before recomputing strict micro F1.",
        "",
        "## System Metrics",
        "",
        f"| System | Metric | Estimate | {confidence_label} bootstrap CI |",
        "|---|---:|---:|---:|",
    ]

    for row in report["systems"]:
        metric = row["metric"]
        lines.append(
            "| "
            f"{row['system']} | "
            f"{METRIC_LABELS[metric]} | "
            f"{_format_value(metric, row['estimate'])} | "
            f"[{_format_value(metric, row['ci_low'])}, {_format_value(metric, row['ci_high'])}] |"
        )

    lines.extend([
        "",
        "## Pairwise Differences",
        "",
        "Differences are `System A - System B`. Intervals that include zero do "
        "not support ranking the two systems on that metric.",
        "",
        f"| System A | System B | Metric | Difference | {confidence_label} paired bootstrap CI | Includes 0? |",
        "|---|---|---:|---:|---:|---:|",
    ])

    for row in report["comparisons"]:
        metric = row["metric"]
        lines.append(
            "| "
            f"{row['system_a']} | "
            f"{row['system_b']} | "
            f"{METRIC_LABELS[metric]} | "
            f"{_format_diff(metric, row['difference'])} | "
            f"[{_format_diff(metric, row['ci_low'])}, {_format_diff(metric, row['ci_high'])}] | "
            f"{'yes' if row['ci_includes_zero'] else 'no'} |"
        )

    lines.extend([
        "",
        "## Paper Methods Sentence",
        "",
        (
            "We use paired item-level bootstrap confidence intervals for architecture "
            "differences, resampling held-out items with replacement and recomputing "
            "each metric on the resampled item set. For changed-field \\(F_1\\), "
            "all changed-field counts within a sampled item are kept together before "
            "recomputing strict micro \\(F_1\\)."
        ),
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def build_report(
    runs: dict[str, Path],
    *,
    pairs: list[tuple[str, str]] | None = None,
    iterations: int,
    seed: int,
    alpha: float,
) -> dict[str, Any]:
    if iterations <= 0:
        raise ValueError("--iterations must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("--alpha must be between 0 and 1")

    run_items = {
        label: _load_run_items(path)
        for label, path in runs.items()
    }
    item_keys, rows_by_label = _aligned_rows(run_items)
    labels = list(runs)
    selected_pairs = pairs or list(itertools.combinations(labels, 2))
    rng = random.Random(seed)

    system_rows: list[dict[str, Any]] = []
    for label in labels:
        rows = rows_by_label[label]
        for metric in METRICS:
            result = _bootstrap_system(
                rows,
                metric=metric,
                iterations=iterations,
                alpha=alpha,
                rng=rng,
            )
            system_rows.append({
                "system": label,
                "metric": metric,
                "unit": METRIC_UNITS[metric],
                **result,
            })

    comparison_rows: list[dict[str, Any]] = []
    for left, right in selected_pairs:
        for metric in METRICS:
            result = _bootstrap_difference(
                rows_by_label[left],
                rows_by_label[right],
                metric=metric,
                iterations=iterations,
                alpha=alpha,
                rng=rng,
            )
            comparison_rows.append({
                "comparison": f"{left} - {right}",
                "system_a": left,
                "system_b": right,
                "metric": metric,
                "unit": METRIC_UNITS[metric],
                **result,
            })

    return {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "iterations": iterations,
            "seed": seed,
            "alpha": alpha,
            "confidence_level": 1.0 - alpha,
            "item_count": len(item_keys),
            "item_key_fields": list(ITEM_KEY_FIELDS),
            "run_paths": {label: str(path) for label, path in runs.items()},
            "metric_definitions": {
                "exact_record_rate": "Mean strict exact-record correctness over sampled items.",
                "changed_field_f1": (
                    "Strict micro F1 after summing changed_correct, "
                    "predicted_changed_total, and gold_changed_total within sampled items."
                ),
                "collateral_edits": "Total collateral edits over the sampled item set.",
            },
        },
        "systems": system_rows,
        "comparisons": comparison_rows,
    }


def write_outputs(report: dict[str, Any], output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "architecture_bootstrap.json"
    systems_csv = output_dir / "architecture_bootstrap_systems.csv"
    comparisons_csv = output_dir / "architecture_bootstrap_comparisons.csv"
    markdown_path = output_dir / "architecture_bootstrap.md"

    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_csv(
        systems_csv,
        report["systems"],
        [
            "system",
            "metric",
            "unit",
            "estimate",
            "ci_low",
            "ci_high",
        ],
    )
    _write_csv(
        comparisons_csv,
        report["comparisons"],
        [
            "comparison",
            "system_a",
            "system_b",
            "metric",
            "unit",
            "estimate_a",
            "estimate_b",
            "difference",
            "ci_low",
            "ci_high",
            "ci_includes_zero",
        ],
    )
    _write_markdown(markdown_path, report)
    return {
        "json": str(json_path),
        "systems_csv": str(systems_csv),
        "comparisons_csv": str(comparisons_csv),
        "markdown": str(markdown_path),
    }


def print_summary(report: dict[str, Any]) -> None:
    metadata = report["metadata"]
    confidence_label = f"{metadata['confidence_level'] * 100:.0f}%"
    print(
        f"Paired bootstrap over {metadata['item_count']} items, "
        f"{metadata['iterations']} samples, seed {metadata['seed']}."
    )
    print()
    print("Pairwise differences (System A - System B):")
    for row in report["comparisons"]:
        metric = row["metric"]
        print(
            f"- {row['system_a']} vs {row['system_b']}, "
            f"{METRIC_LABELS[metric]}: "
            f"{_format_diff(metric, row['difference'])}, "
            f"{confidence_label} CI [{_format_diff(metric, row['ci_low'])}, "
            f"{_format_diff(metric, row['ci_high'])}], "
            f"includes 0: {'yes' if row['ci_includes_zero'] else 'no'}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help=(
            "Run label and summary path or run directory. Repeat to compare custom runs. "
            "If omitted, uses the five held-out system runs bundled in the "
            "separate data upload."
        ),
    )
    parser.add_argument(
        "--pair",
        action="append",
        default=[],
        metavar="LABEL_A:LABEL_B",
        help="Specific pair to compare. Repeat as needed. Defaults to all pairwise comparisons.",
    )
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for JSON, CSV, and Markdown outputs.",
    )
    args = parser.parse_args(argv)

    try:
        runs = _parse_runs(args.run)
        pairs = _parse_pairs(args.pair, list(runs)) if args.pair else None
        output_dir = args.output_dir
        if not output_dir.is_absolute():
            output_dir = ROOT / output_dir
        report = build_report(
            runs,
            pairs=pairs,
            iterations=args.iterations,
            seed=args.seed,
            alpha=args.alpha,
        )
        output_paths = write_outputs(report, output_dir.resolve())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print_summary(report)
    print()
    print("Wrote:")
    for name, path in output_paths.items():
        print(f"- {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
