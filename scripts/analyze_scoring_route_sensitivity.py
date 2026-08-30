#!/usr/bin/env python3
"""Reproduce Appendix D scoring-route, alignment-exposure, and sensitivity tables."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("CONFORMBENCH_DATA_DIR", ROOT / "data")).expanduser().resolve()
RUNS_ROOT = (
    DATA_ROOT
    / "reports"
    / "split_runs"
    / "conformbench_v1_20_80_seed20260524"
    / "test"
)

RUNS = {
    "Direct JSON": RUNS_ROOT / "audit_tight_test_20260525__direct__gen-gpt-5-4-mini__judge-gpt-5-4-medium",
    "Update Tool": RUNS_ROOT / "audit_tight_test_20260525__flatagent__gen-gpt-5-4-mini__judge-gpt-5-4-medium",
    "Diff-then-Verify": DATA_ROOT
    / "reports"
    / "rebuttal_runs"
    / "diff_then_verify"
    / "test"
    / "arr_diff_then_verify_test__diff_then_verify__gen-gpt-5-4-mini__judge-gpt-5-4-medium",
    "Split Agent": RUNS_ROOT / "audit_tight_test_20260525__split_agent__gen-gpt-5-4-mini__judge-gpt-5-4-medium",
    "Triage-Briefed Split": RUNS_ROOT / "audit_tight_test_20260525__split_agent_briefed__gen-gpt-5-4-mini__judge-gpt-5-4-medium",
}

LLM_SEMANTIC = {
    "semantic_alias_judge",
    "semantic_exact_mismatch_judge",
    "semantic_iu_judge",
}
def read_results(run_dir: Path) -> list[dict[str, str]]:
    with (run_dir / "results.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_alignment_groups(run_dir: Path) -> dict[str, set[str]]:
    """Return repeat groups for which an alignment LLM call actually occurred."""

    groups_by_item: dict[str, set[str]] = {}
    for evaluation_path in run_dir.rglob("evaluation.json"):
        payload = json.loads(evaluation_path.read_text(encoding="utf-8"))
        item_id = evaluation_path.parents[2].name
        groups = {
            str(entry.get("extra", {}).get("group_name"))
            for entry in payload.get("prompt_trace", [])
            if entry.get("phase") == "instance_alignment"
            and entry.get("extra", {}).get("group_name")
        }
        groups_by_item.setdefault(item_id, set()).update(groups)
    return groups_by_item


def is_alignment_dependent(
    row: dict[str, str],
    groups_by_item: dict[str, set[str]],
) -> bool:
    """Whether this child-field check follows an actual LLM row-alignment call."""

    return any(
        row["field_path"].startswith(f"{group_name}[")
        for group_name in groups_by_item.get(row["item_id"], set())
    )


def is_correct(row: dict[str, str]) -> bool:
    return row["correctness"] == "correct"


def exact_count(grouped_rows: dict[str, list[dict[str, str]]], predicate) -> int:
    return sum(all(predicate(row) for row in rows) for rows in grouped_rows.values())


def pct(count: int, total: int) -> str:
    return f"{count}/{total} ({100 * count / total:.1f}%)"


def accuracy(rows: list[dict[str, str]]) -> str:
    correct = sum(is_correct(row) for row in rows)
    return f"{100 * correct / len(rows):.1f}% ({correct}/{len(rows)})"


def main() -> None:
    route_rows: list[list[str]] = []
    sensitivity_rows: list[list[str]] = []
    reduced_accuracy_rows: list[list[str]] = []
    totals = {
        "checks": 0,
        "semantic": 0,
        "alignment_dependent": 0,
        "semantic_and_alignment": 0,
        "any_llm_stage": 0,
        "fully_deterministic": 0,
        "alignment_calls": 0,
    }

    for label, run_dir in RUNS.items():
        rows = read_results(run_dir)
        alignment_groups = read_alignment_groups(run_dir)
        grouped: dict[str, list[dict[str, str]]] = {}
        for row in rows:
            grouped.setdefault(row["item_id"], []).append(row)

        total = len(rows)
        llm_semantic = [row for row in rows if row["decision_source"] in LLM_SEMANTIC]
        alignment_dependent = [
            row for row in rows if is_alignment_dependent(row, alignment_groups)
        ]
        any_llm_stage = [
            row
            for row in rows
            if row["decision_source"] in LLM_SEMANTIC
            or is_alignment_dependent(row, alignment_groups)
        ]
        semantic_and_alignment = [
            row
            for row in rows
            if row["decision_source"] in LLM_SEMANTIC
            and is_alignment_dependent(row, alignment_groups)
        ]
        fully_deterministic = total - len(any_llm_stage)
        alignment_calls = sum(len(groups) for groups in alignment_groups.values())

        assert len(any_llm_stage) == (
            len(llm_semantic) + len(alignment_dependent) - len(semantic_and_alignment)
        )
        assert len(any_llm_stage) + fully_deterministic == total
        assert len(semantic_and_alignment) <= len(llm_semantic)
        assert len(semantic_and_alignment) <= len(alignment_dependent)

        totals["checks"] += total
        totals["semantic"] += len(llm_semantic)
        totals["alignment_dependent"] += len(alignment_dependent)
        totals["semantic_and_alignment"] += len(semantic_and_alignment)
        totals["any_llm_stage"] += len(any_llm_stage)
        totals["fully_deterministic"] += fully_deterministic
        totals["alignment_calls"] += alignment_calls

        route_rows.append(
            [
                label,
                str(total),
                str(len(llm_semantic)),
                str(len(alignment_dependent)),
                str(len(any_llm_stage)),
                str(fully_deterministic),
            ]
        )

        item_total = len(grouped)
        original = exact_count(grouped, is_correct)
        pessimistic = exact_count(
            grouped,
            lambda row: row["decision_source"] not in LLM_SEMANTIC and is_correct(row),
        )
        omit_semantic = exact_count(
            grouped,
            lambda row: row["decision_source"] in LLM_SEMANTIC or is_correct(row),
        )
        alignment_dependent_credited = exact_count(
            grouped,
            lambda row: is_alignment_dependent(row, alignment_groups) or is_correct(row),
        )

        sensitivity_rows.append(
            [
                label,
                pct(original, item_total),
                pct(pessimistic, item_total),
                pct(omit_semantic, item_total),
                pct(alignment_dependent_credited, item_total),
            ]
        )
        alignment_independent_rows = [
            row for row in rows if not is_alignment_dependent(row, alignment_groups)
        ]
        reduced_accuracy_rows.append(
            [
                label,
                accuracy(alignment_independent_rows),
            ]
        )

    assert totals["any_llm_stage"] == (
        totals["semantic"]
        + totals["alignment_dependent"]
        - totals["semantic_and_alignment"]
    )
    assert totals["any_llm_stage"] + totals["fully_deterministic"] == totals["checks"]

    print("# Scoring-Route Accounting")
    print("| Architecture | Final field checks | Direct semantic verdict | Alignment-dependent checks | Any LLM-stage exposure | Fully deterministic |")
    print("|---|---:|---:|---:|---:|---:|")
    for row in route_rows:
        print("| " + " | ".join(row) + " |")

    print()
    print("# Aggregate LLM-Stage Exposure")
    print(f"- Final checks: {totals['checks']}")
    print(f"- Direct semantic final verdicts: {totals['semantic']}")
    print(f"- Checks after an actual LLM row-alignment call: {totals['alignment_dependent']}")
    print(f"- Checks following both semantic and alignment routes: {totals['semantic_and_alignment']}")
    print(f"- Checks exposed to either LLM stage: {totals['any_llm_stage']}")
    print(f"- Fully deterministic checks: {totals['fully_deterministic']}")
    print(f"- Actual row-alignment calls: {totals['alignment_calls']}")

    print()
    print("# LLM-Stage Sensitivity")
    print("| Architecture | Original strict exact records | Direct semantic fields treated incorrect | Direct semantic fields credited correct | Alignment-dependent fields credited correct |")
    print("|---|---:|---:|---:|---:|")
    for row in sensitivity_rows:
        print("| " + " | ".join(row) + " |")

    print()
    print("# Alignment-Independent Field Accuracy")
    print("| Architecture | Alignment-dependent fields omitted |")
    print("|---|---:|")
    for row in reduced_accuracy_rows:
        print("| " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()
