#!/usr/bin/env python3
"""Reproduce the five-system descriptive analyses from saved run summaries.

This script is offline. It reads frozen ``summary_report.json`` files and
benchmark item packets, then writes compact JSON and Markdown summaries. It
does not call a generator or evaluator.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("CONFORMBENCH_DATA_DIR", ROOT / "data")).expanduser().resolve()
SPLIT_ID = "conformbench_v1_20_80_seed20260524"
ORIGINAL_RUNS = DATA_ROOT / "reports" / "split_runs" / SPLIT_ID / "test"
DTV_RUN = (
    DATA_ROOT
    / "reports"
    / "rebuttal_runs"
    / "diff_then_verify"
    / "test"
    / "arr_diff_then_verify_test__diff_then_verify__gen-gpt-5-4-mini__judge-gpt-5-4-medium"
)
RUNS = {
    "Direct JSON": ORIGINAL_RUNS
    / "audit_tight_test_20260525__direct__gen-gpt-5-4-mini__judge-gpt-5-4-medium",
    "Update Tool": ORIGINAL_RUNS
    / "audit_tight_test_20260525__flatagent__gen-gpt-5-4-mini__judge-gpt-5-4-medium",
    "Diff-then-Verify": DTV_RUN,
    "Split Agent": ORIGINAL_RUNS
    / "audit_tight_test_20260525__split_agent__gen-gpt-5-4-mini__judge-gpt-5-4-medium",
    "Triage-Briefed Split Agent": ORIGINAL_RUNS
    / "audit_tight_test_20260525__split_agent_briefed__gen-gpt-5-4-mini__judge-gpt-5-4-medium",
}
ITEM_KEY_FIELDS = ("questionnaire", "scenario_id", "state", "utterance_id")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def item_key(utterance: dict[str, Any]) -> str:
    return "/".join(str(utterance.get(field) or "") for field in ITEM_KEY_FIELDS)


def exact(utterance: dict[str, Any]) -> bool:
    return bool(
        utterance["metric_views"]["whole_record_exact_match"].get("exact_match")
    )


def pct(value: float) -> float:
    return round(100.0 * value, 1)


def load_runs() -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, dict[str, Any]]]]:
    summaries = {
        label: read_json(run_dir / "summary_report.json")
        for label, run_dir in RUNS.items()
    }
    items = {
        label: {item_key(row): row for row in summary["utterances"]}
        for label, summary in summaries.items()
    }
    reference = set(next(iter(items.values())))
    if len(reference) != 144:
        raise ValueError(f"Expected 144 held-out items, found {len(reference)}")
    for label, rows in items.items():
        if len(rows) != len(summaries[label]["utterances"]):
            raise ValueError(f"Duplicate held-out item key in {label}")
        if set(rows) != reference:
            raise ValueError(f"Held-out item keys do not align for {label}")
    return summaries, items


def headline(summary: dict[str, Any]) -> dict[str, Any]:
    aggregate = summary["aggregate"]
    views = aggregate["metric_views"]
    exact_view = views["whole_record_exact_match"]
    all_fields = views["all_fields"]
    changed = views["changed_fields"]
    diagnostics = views["task_diagnostics"]
    missed_updates = sum(
        int(((row.get("diagnostics") or {}).get("counts") or {}).get("missed_supported_update", 0) or 0)
        for row in summary["utterances"]
    )
    return {
        "exact_records": int(exact_view["exact_match_count"]),
        "item_count": int(exact_view["item_count"]),
        "exact_record_rate": float(exact_view["exact_match_rate"]),
        "all_field_accuracy": float(all_fields["accuracy"]),
        "changed_precision": float(changed["strict"]["precision"]),
        "changed_recall": float(changed["strict"]["recall"]),
        "changed_f1": float(changed["strict"]["f1"]),
        "gold_changed_total": int(changed["gold_changed_total"]),
        "missed_updates": missed_updates,
        "unsupported_commitments": int(diagnostics["unsupported_commit_count"]),
        "final_field_checks": int(all_fields["total_evaluated_leaf_fields"]),
    }


def strict_lenient(summary: dict[str, Any]) -> dict[str, Any]:
    aggregate = summary["aggregate"]
    views = aggregate["metric_views"]
    exact_view = views["whole_record_exact_match"]
    changed = views["changed_fields"]
    return {
        "partial_fields_all": int(aggregate["attempted_scores"]["partially_correct"]),
        "partial_fields_changed": int(changed["changed_partial"]),
        "strict_exact_records": int(exact_view["exact_match_count"]),
        "lenient_exact_records": int(exact_view["lenient_exact_match_count"]),
        "strict_changed_f1": float(changed["strict"]["f1"]),
        "lenient_changed_f1": float(changed["lenient"]["f1"]),
    }


def diagnostic(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    views = summary["aggregate"]["metric_views"]
    task = views["task_diagnostics"]
    clear = views["transition_accuracy"]["by_transition"]["clear"]
    return {
        "clear_transitions": {
            "success": int(clear["correct"]),
            "applicable": int(clear["gold_total"]),
            "success_rate": float(clear["accuracy"]),
        },
        "history_recovery": task["history_recovery_success"],
        "gate_execution": task["gate_execution_success"],
        "retraction": task["retraction_success"],
        "repeat_execution": task["repeat_group_execution_success"],
    }


def diagnostic_ranges(
    summaries: dict[str, dict[str, Any]],
    items: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    per_system = {label: diagnostic(summary) for label, summary in summaries.items()}
    ranges: dict[str, Any] = {}
    for name in next(iter(per_system.values())):
        rows = [values[name] for values in per_system.values()]
        rates = [float(row["success_rate"]) for row in rows]
        denominators = [int(row["applicable"]) for row in rows]
        ranges[name] = {
            "minimum_rate": min(rates),
            "maximum_rate": max(rates),
            "minimum_denominator": min(denominators),
            "maximum_denominator": max(denominators),
        }

    s4_rates: list[float] = []
    for rows in items.values():
        s4 = [
            row
            for row in rows.values()
            if row["derived_variables"]["prior_state_condition"] == "S4"
        ]
        if len(s4) != 30:
            raise ValueError(f"Expected 30 S4 items, found {len(s4)}")
        s4_rates.append(sum(exact(row) for row in s4) / len(s4))
    ranges["s4_exact"] = {
        "minimum_rate": min(s4_rates),
        "maximum_rate": max(s4_rates),
        "minimum_denominator": 30,
        "maximum_denominator": 30,
    }
    return {"per_system": per_system, "ranges": ranges}


def shape_solvability(items: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    labels = list(items)
    reference = items[labels[0]]
    solved_count = {
        key: sum(exact(items[label][key]) for label in labels) for key in reference
    }

    def summarize(predicate: Callable[[dict[str, Any]], bool]) -> dict[str, Any]:
        keys = [key for key, row in reference.items() if predicate(row)]
        solved = sum(solved_count[key] > 0 for key in keys)
        return {
            "items": len(keys),
            "solved_by_any": solved,
            "solved_rate": solved / len(keys) if keys else 0.0,
        }

    derived = lambda row: row["derived_variables"]
    shapes = {
        "no_history_required": summarize(lambda row: not derived(row)["history_required"]),
        "history_required": summarize(lambda row: bool(derived(row)["history_required"])),
        "no_repeated_group": summarize(
            lambda row: derived(row)["repeat_group_involvement"] == "none"
        ),
        "single_repeated_group": summarize(
            lambda row: derived(row)["repeat_group_involvement"] == "single_group"
        ),
        "multiple_repeated_groups": summarize(
            lambda row: derived(row)["repeat_group_involvement"] == "multi_group"
        ),
        "low_load": summarize(
            lambda row: derived(row)["form_relative_load_bucket"] == "low"
        ),
        "high_or_scale_load": summarize(
            lambda row: derived(row)["form_relative_load_bucket"] in {"high", "scale"}
        ),
        "mixed_revision": summarize(
            lambda row: derived(row)["revision_operation"] == "mixed"
        ),
        "clear_transition_present": summarize(
            lambda row: row["metric_views"]["transition_accuracy"]["by_transition"]
            .get("clear", {})
            .get("gold_total", 0)
            > 0
        ),
        "s4_silent_mismatch": summarize(
            lambda row: derived(row)["prior_state_condition"] == "S4"
        ),
    }
    distribution = {
        str(count): sum(value == count for value in solved_count.values())
        for count in range(len(labels) + 1)
    }
    return {
        "solved_by_any": sum(value > 0 for value in solved_count.values()),
        "solved_by_none": sum(value == 0 for value in solved_count.values()),
        "solved_by_all": sum(value == len(labels) for value in solved_count.values()),
        "solved_by_system_count_distribution": distribution,
        "shape_slices": shapes,
    }


def dialogue_uniqueness() -> dict[str, int]:
    packets = [read_json(path) for path in sorted((DATA_ROOT / "items/benchmark").rglob("ground_truth.json"))]
    scenario_ids = {str(packet["scenario_id"]) for packet in packets}
    utterances = {str(packet["current_utterance"]) for packet in packets}
    transcripts = {
        json.dumps(
            [*(packet.get("visible_history") or []), {"role": "user", "content": packet["current_utterance"]}],
            ensure_ascii=False,
            sort_keys=True,
        )
        for packet in packets
    }
    return {
        "items": len(packets),
        "unique_scenario_ids": len(scenario_ids),
        "unique_current_utterances": len(utterances),
        "unique_history_plus_current_transcripts": len(transcripts),
    }


def build_report() -> dict[str, Any]:
    summaries, items = load_runs()
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": "Offline analysis of saved summary_report.json files; no generation or evaluation calls.",
        "split_id": SPLIT_ID,
        "input_runs": {
            label: str(path.relative_to(DATA_ROOT)) for label, path in RUNS.items()
        },
        "headline": {label: headline(summary) for label, summary in summaries.items()},
        "transition_accuracy": {
            label: summary["aggregate"]["metric_views"]["transition_accuracy"]["by_transition"]
            for label, summary in summaries.items()
        },
        "strict_vs_lenient": {
            label: strict_lenient(summary) for label, summary in summaries.items()
        },
        "diagnostic_slices": diagnostic_ranges(summaries, items),
        "architecture_overlap": shape_solvability(items),
        "dialogue_uniqueness": dialogue_uniqueness(),
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Five-System Offline Analysis",
        "",
        report["method"],
        "",
        "## Headline results",
        "",
        "| System | Exact records | All fields | Changed P/R/F1 | Missed | Unsupported |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, row in report["headline"].items():
        lines.append(
            f"| {label} | {row['exact_records']}/{row['item_count']} "
            f"({pct(row['exact_record_rate']):.1f}%) | {pct(row['all_field_accuracy']):.1f}% | "
            f"{pct(row['changed_precision']):.1f}/{pct(row['changed_recall']):.1f}/"
            f"{pct(row['changed_f1']):.1f} | {row['missed_updates']} | "
            f"{row['unsupported_commitments']} |"
        )

    overlap = report["architecture_overlap"]
    lines.extend(
        [
            "",
            "## Architecture overlap",
            "",
            f"- Solved by at least one system: {overlap['solved_by_any']}/144",
            f"- Solved by none: {overlap['solved_by_none']}/144",
            f"- Solved by all five: {overlap['solved_by_all']}/144",
            "",
            "## Dialogue uniqueness",
            "",
        ]
    )
    for key, value in report["dialogue_uniqueness"].items():
        lines.append(f"- {key.replace('_', ' ').capitalize()}: {value}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_ROOT / "reports/rebuttal_runs/diff_then_verify/analysis",
    )
    args = parser.parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report = build_report()
    (output_dir / "five_system_analysis.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_markdown(output_dir / "five_system_analysis.md", report)
    print(f"Wrote five-system analysis to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
