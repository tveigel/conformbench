#!/usr/bin/env python3
"""Produce the frozen author-comment statistics for Diff-then-Verify."""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any


ITEM_KEY_FIELDS = ("questionnaire", "scenario_id", "state", "utterance_id")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def resolve_run(value: str | Path) -> Path:
    path = Path(value).expanduser().resolve()
    if path.name == "summary_report.json":
        path = path.parent
    if not (path / "summary_report.json").is_file():
        raise FileNotFoundError(f"Missing summary_report.json under {path}")
    if not (path / "run_metadata.json").is_file():
        raise FileNotFoundError(f"Missing run_metadata.json under {path}")
    return path


def item_key(utterance: dict[str, Any]) -> str:
    return "/".join(str(utterance.get(field) or "") for field in ITEM_KEY_FIELDS)


def exact_by_item(summary: dict[str, Any]) -> dict[str, int]:
    rows: dict[str, int] = {}
    for utterance in summary.get("utterances") or []:
        key = item_key(utterance)
        if key in rows:
            raise ValueError(f"Duplicate item key: {key}")
        exact = (
            (utterance.get("metric_views") or {})
            .get("whole_record_exact_match", {})
            .get("exact_match")
        )
        rows[key] = 1 if exact else 0
    return rows


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def paired_exact_bootstrap(
    verify: dict[str, int],
    baseline: dict[str, int],
    *,
    iterations: int,
    seed: int,
) -> dict[str, float | int | bool]:
    if set(verify) != set(baseline):
        missing = sorted(set(baseline) - set(verify))
        extra = sorted(set(verify) - set(baseline))
        raise ValueError(
            "Runs are not aligned: "
            f"missing_from_verify={missing[:8]}, extra_in_verify={extra[:8]}"
        )
    keys = sorted(verify)
    rng = random.Random(seed)
    differences: list[float] = []
    for _ in range(iterations):
        sampled = [keys[rng.randrange(len(keys))] for _ in keys]
        differences.append(
            sum(verify[key] - baseline[key] for key in sampled) / len(sampled)
        )
    difference = sum(verify.values()) / len(keys) - sum(baseline.values()) / len(keys)
    low = percentile(differences, 0.025)
    high = percentile(differences, 0.975)
    return {
        "iterations": iterations,
        "seed": seed,
        "difference": difference,
        "ci_low": low,
        "ci_high": high,
        "ci_includes_zero": low <= 0.0 <= high,
    }


def aggregate_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    views = (summary.get("aggregate") or {}).get("metric_views") or {}
    exact = views.get("whole_record_exact_match") or {}
    changed = views.get("changed_fields") or {}
    diagnostic_counts = {
        "missed_supported_update": 0,
        "unsupported_commit": 0,
    }
    for utterance in summary.get("utterances") or []:
        counts = (utterance.get("diagnostics") or {}).get("counts") or {}
        for key in diagnostic_counts:
            diagnostic_counts[key] += int(counts.get(key, 0) or 0)
    return {
        "item_count": int(exact.get("item_count", 0) or 0),
        "exact_record_count": int(exact.get("exact_match_count", 0) or 0),
        "exact_record_rate": float(exact.get("exact_match_rate", 0.0) or 0.0),
        "changed_field_f1": float((changed.get("strict") or {}).get("f1", 0.0) or 0.0),
        "changed_field_precision": float(
            (changed.get("strict") or {}).get("precision", 0.0) or 0.0
        ),
        "changed_field_recall": float(
            (changed.get("strict") or {}).get("recall", 0.0) or 0.0
        ),
        "missed_supported_updates": diagnostic_counts["missed_supported_update"],
        "unsupported_commitments": diagnostic_counts["unsupported_commit"],
    }


def latency_metrics(run_dir: Path) -> dict[str, Any]:
    diff_values: list[float] = []
    verify_values: list[float] = []
    total_values: list[float] = []
    verifier_patch_items = 0
    verifier_tool_calls = 0
    verifier_tool_error_calls = 0

    for path in sorted(run_dir.rglob("turn_result.json")):
        payload = read_json(path)
        response = payload.get("agent_response") or {}
        timing = response.get("timing") or {}
        try:
            diff_values.append(float(timing["diff_stage_seconds"]))
            verify_values.append(float(timing["verification_stage_seconds"]))
            total_values.append(float(timing["total_generation_seconds"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Missing valid timing data in {path}: {exc}") from exc

        verify_stage = (response.get("stages") or {}).get("verify") or {}
        calls = verify_stage.get("tool_updates") or []
        if calls:
            verifier_patch_items += 1
        verifier_tool_calls += len(calls)
        for call in calls:
            result = (call or {}).get("result") or {}
            if result.get("status") == "error" or result.get("error") or result.get("errors"):
                verifier_tool_error_calls += 1

    if not total_values:
        raise ValueError(f"No turn_result.json timing data found under {run_dir}")
    return {
        "timed_item_count": len(total_values),
        "median_diff_stage_seconds": median(diff_values),
        "median_verification_stage_seconds": median(verify_values),
        "median_total_generation_seconds": median(total_values),
        "verifier_patch_item_count": verifier_patch_items,
        "verifier_no_patch_item_count": len(total_values) - verifier_patch_items,
        "verifier_tool_call_count": verifier_tool_calls,
        "verifier_tool_error_call_count": verifier_tool_error_calls,
        "definition": (
            "Per-item monotonic wall time inside the solver; excludes evaluator time. "
            "The diff stage is the same-run Update Tool latency reference."
        ),
    }


def validate_run_metadata(
    verify_metadata: dict[str, Any],
    baseline_metadata: dict[str, Any],
) -> None:
    expected = {
        "verify architecture": (verify_metadata.get("architecture"), "diff_then_verify"),
        "baseline architecture": (baseline_metadata.get("architecture"), "flatagent"),
        "verify generator": (verify_metadata.get("model_id"), "gpt-5.4-mini"),
        "baseline generator": (baseline_metadata.get("model_id"), "gpt-5.4-mini"),
        "verify evaluator": (verify_metadata.get("evaluator_model_id"), "gpt-5.4"),
        "baseline evaluator": (baseline_metadata.get("evaluator_model_id"), "gpt-5.4"),
        "verify evaluator reasoning": (
            verify_metadata.get("evaluator_reasoning_effort"),
            "medium",
        ),
        "baseline evaluator reasoning": (
            baseline_metadata.get("evaluator_reasoning_effort"),
            "medium",
        ),
        "verify status": (verify_metadata.get("status"), "completed"),
        "baseline status": (baseline_metadata.get("status"), "completed"),
    }
    mismatches = {
        label: {"actual": actual, "expected": wanted}
        for label, (actual, wanted) in expected.items()
        if actual != wanted
    }
    if mismatches:
        raise ValueError("Run metadata mismatch: " + json.dumps(mismatches, indent=2))


def format_pp(value: float) -> str:
    return f"{value * 100:+.1f}"


def author_comment_text(report: dict[str, Any]) -> str:
    verify = report["diff_then_verify"]
    baseline = report["update_tool_baseline"]
    paired = report["paired_exact_record_difference"]
    latency = report["latency"]
    return (
        "Diff-then-Verify obtains "
        f"{verify['exact_record_count']}/{verify['item_count']} "
        f"({verify['exact_record_rate'] * 100:.1f}%) exact records versus "
        f"{baseline['exact_record_count']}/{baseline['item_count']} "
        f"({baseline['exact_record_rate'] * 100:.1f}%) for Update Tool, a paired "
        f"difference of {format_pp(float(paired['difference']))} points "
        f"[95% CI: {format_pp(float(paired['ci_low']))}, "
        f"{format_pp(float(paired['ci_high']))}]. Changed-field F1 is "
        f"{verify['changed_field_f1'] * 100:.1f}%, with "
        f"{verify['missed_supported_updates']} missed updates and "
        f"{verify['unsupported_commitments']} unsupported commitments. Median "
        f"generation latency is {latency['median_total_generation_seconds']:.2f} "
        f"seconds versus {latency['median_diff_stage_seconds']:.2f} seconds for "
        "the same-run Update Tool first stage; the added verification stage has "
        f"median latency {latency['median_verification_stage_seconds']:.2f} seconds."
    )


def build_report(
    verify_dir: Path,
    baseline_dir: Path,
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    verify_summary = read_json(verify_dir / "summary_report.json")
    baseline_summary = read_json(baseline_dir / "summary_report.json")
    verify_metadata = read_json(verify_dir / "run_metadata.json")
    baseline_metadata = read_json(baseline_dir / "run_metadata.json")
    validate_run_metadata(verify_metadata, baseline_metadata)

    verify_metrics = aggregate_metrics(verify_summary)
    baseline_metrics = aggregate_metrics(baseline_summary)
    if verify_metrics["item_count"] != 144 or baseline_metrics["item_count"] != 144:
        raise ValueError(
            "Expected two complete 144-item test runs, got "
            f"verify={verify_metrics['item_count']}, baseline={baseline_metrics['item_count']}"
        )

    paired = paired_exact_bootstrap(
        exact_by_item(verify_summary),
        exact_by_item(baseline_summary),
        iterations=iterations,
        seed=seed,
    )
    latency = latency_metrics(verify_dir)
    if latency["timed_item_count"] != 144:
        raise ValueError(
            f"Expected latency for 144 items, got {latency['timed_item_count']}"
        )

    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "verify_run_dir": str(verify_dir),
        "baseline_run_dir": str(baseline_dir),
        "diff_then_verify": verify_metrics,
        "update_tool_baseline": baseline_metrics,
        "paired_exact_record_difference": paired,
        "latency": latency,
        "method": {
            "comparison": "paired by held-out item",
            "bootstrap_unit": "item",
            "confidence_level": 0.95,
            "test_set_policy": "single frozen 144-item run",
            "evaluator_compatibility": (
                "Uses the released gpt-5.4 evaluator path. Run metadata retains "
                "the historical medium label; the checked-in official-OpenAI "
                "helper uses provider-default effective reasoning."
            ),
        },
    }
    report["author_comment_replacement"] = author_comment_text(report)
    return report


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    latency = report["latency"]
    lines = [
        "# Diff-then-Verify Author-Comment Results",
        "",
        report["author_comment_replacement"],
        "",
        "## Audit details",
        "",
        f"- Verify run: `{report['verify_run_dir']}`",
        f"- Baseline run: `{report['baseline_run_dir']}`",
        "- Paired item bootstrap: "
        f"{report['paired_exact_record_difference']['iterations']} samples, "
        f"seed {report['paired_exact_record_difference']['seed']}",
        "- Latency excludes evaluator time and uses monotonic wall time inside the solver.",
        f"- Verifier emitted a patch on {latency['verifier_patch_item_count']}/144 items; "
        f"tool-error calls: {latency['verifier_tool_error_call_count']}.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-run", required=True)
    parser.add_argument("--baseline-run", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260710)
    args = parser.parse_args()

    verify_dir = resolve_run(args.verify_run)
    baseline_dir = resolve_run(args.baseline_run)
    report = build_report(
        verify_dir,
        baseline_dir,
        iterations=args.iterations,
        seed=args.seed,
    )
    output_dir = args.output_dir.expanduser().resolve()
    write_json(output_dir / "author_comment_results.json", report)
    write_markdown(output_dir / "author_comment_results.md", report)
    print(report["author_comment_replacement"])
    print(f"Wrote results to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
