#!/usr/bin/env python3
"""Load-normalized exact-record analysis for the held-out architecture runs.

The analysis compares observed exact-record success with two independent-field
nulls:

* field_count_only: one strict all-field success probability raised to the
  number of evaluated field checks for the item/architecture.
* changed_preserve: separate strict probabilities for gold-changed and
  gold-preserved fields, raised to each item's changed/preserved burden.

Rates are estimated leave-one-item-out within each architecture so that an
item's own outcome does not directly set its expected exact probability.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


ARCHITECTURE_RUNS = {
    "Direct JSON": "data/reports/split_runs/conformbench_v1_20_80_seed20260524/test/"
    "audit_tight_test_20260525__direct__gen-gpt-5-4-mini__judge-gpt-5-4-medium/"
    "summary_report.json",
    "Update Tool": "data/reports/split_runs/conformbench_v1_20_80_seed20260524/test/"
    "audit_tight_test_20260525__flatagent__gen-gpt-5-4-mini__judge-gpt-5-4-medium/"
    "summary_report.json",
    "Split Agent": "data/reports/split_runs/conformbench_v1_20_80_seed20260524/test/"
    "audit_tight_test_20260525__split_agent__gen-gpt-5-4-mini__judge-gpt-5-4-medium/"
    "summary_report.json",
    "Triage-Briefed Split Agent": "data/reports/split_runs/conformbench_v1_20_80_seed20260524/test/"
    "audit_tight_test_20260525__split_agent_briefed__gen-gpt-5-4-mini__judge-gpt-5-4-medium/"
    "summary_report.json",
    "Diff-then-Verify": "data/reports/rebuttal_runs/diff_then_verify/test/"
    "arr_diff_then_verify_test__diff_then_verify__gen-gpt-5-4-mini__judge-gpt-5-4-medium/"
    "summary_report.json",
}

LOAD_BUCKETS = ("low", "medium", "high", "scale", "high+scale")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def data_root() -> Path:
    return Path(
        os.environ.get("CONFORMBENCH_DATA_DIR", repo_root() / "data")
    ).expanduser().resolve()


def item_key(utterance: dict[str, Any]) -> str:
    return f"{utterance['questionnaire']}/{utterance['scenario_id']}"


def load_bucket(utterance: dict[str, Any]) -> str:
    return str(utterance["derived_variables"]["form_relative_load_bucket"])


def is_in_bucket(item_load: str, bucket: str) -> bool:
    if bucket == "high+scale":
        return item_load in {"high", "scale"}
    return item_load == bucket


def exact_match(utterance: dict[str, Any]) -> bool:
    return bool(utterance["metric_views"]["whole_record_exact_match"].get("exact_match"))


def strict_counts(utterance: dict[str, Any]) -> dict[str, int]:
    views = utterance["metric_views"]
    changed = views["changed_fields"]
    preservation = views["preservation"]
    all_fields = views["all_fields"]
    return {
        "changed": int(changed.get("gold_changed_total", 0)),
        "changed_correct": int(changed.get("changed_correct", 0)),
        "preserve": int(preservation.get("preserved_total", 0)),
        "preserve_correct": int(preservation.get("preserved_kept", 0)),
        "total": int(all_fields.get("total_evaluated_leaf_fields", all_fields.get("gt_expected_total", 0))),
        "total_correct": int(all_fields.get("correct", 0)),
    }


def changed_f1(utterances: list[dict[str, Any]]) -> float:
    changed_correct = 0
    predicted_changed = 0
    gold_changed = 0
    for utterance in utterances:
        view = utterance["metric_views"]["changed_fields"]
        changed_correct += int(view.get("changed_correct", 0))
        predicted_changed += int(view.get("predicted_changed_total", 0))
        gold_changed += int(view.get("gold_changed_total", 0))
    precision = changed_correct / predicted_changed if predicted_changed else 0.0
    recall = changed_correct / gold_changed if gold_changed else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def errors_per_100_fields(utterances: list[dict[str, Any]]) -> float:
    correct = 0
    total = 0
    for utterance in utterances:
        view = utterance["metric_views"]["all_fields"]
        correct += int(view.get("correct", 0))
        total += int(view.get("total_evaluated_leaf_fields", view.get("gt_expected_total", 0)))
    return 100.0 * (1.0 - correct / total) if total else 0.0


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def f3(value: float) -> str:
    return f"{value:.3f}"


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    pos = (len(sorted_values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_values[lo]
    return sorted_values[lo] * (hi - pos) + sorted_values[hi] * (pos - lo)


def load_runs(root: Path) -> dict[str, dict[str, dict[str, Any]]]:
    runs: dict[str, dict[str, dict[str, Any]]] = {}
    for architecture, relative_path in ARCHITECTURE_RUNS.items():
        summary_path = root / relative_path.removeprefix("data/")
        with summary_path.open() as f:
            summary = json.load(f)
        runs[architecture] = {item_key(u): u for u in summary["utterances"]}
    keys = sorted(next(iter(runs.values())).keys())
    for architecture, utterances in runs.items():
        if sorted(utterances.keys()) != keys:
            raise SystemExit(f"{architecture} does not contain the same item set")
    return runs


def compute_expected_exact(
    runs: dict[str, dict[str, dict[str, Any]]],
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]], list[dict[str, Any]]]:
    expected_changed_preserve: dict[str, dict[str, float]] = defaultdict(dict)
    expected_field_count_only: dict[str, dict[str, float]] = defaultdict(dict)
    rate_rows: list[dict[str, Any]] = []

    for architecture, utterances in runs.items():
        per_item = {key: strict_counts(u) for key, u in utterances.items()}
        global_counts = {name: sum(c[name] for c in per_item.values()) for name in next(iter(per_item.values()))}

        changed_rate = global_counts["changed_correct"] / global_counts["changed"]
        preserve_rate = global_counts["preserve_correct"] / global_counts["preserve"]
        all_rate = global_counts["total_correct"] / global_counts["total"]
        rate_rows.append(
            {
                "architecture": architecture,
                "changed_correct": global_counts["changed_correct"],
                "changed_total": global_counts["changed"],
                "changed_rate": changed_rate,
                "preserve_correct": global_counts["preserve_correct"],
                "preserve_total": global_counts["preserve"],
                "preserve_rate": preserve_rate,
                "all_correct": global_counts["total_correct"],
                "all_total": global_counts["total"],
                "all_rate": all_rate,
            }
        )

        for key, counts in per_item.items():
            changed_den = global_counts["changed"] - counts["changed"]
            changed_num = global_counts["changed_correct"] - counts["changed_correct"]
            preserve_den = global_counts["preserve"] - counts["preserve"]
            preserve_num = global_counts["preserve_correct"] - counts["preserve_correct"]
            total_den = global_counts["total"] - counts["total"]
            total_num = global_counts["total_correct"] - counts["total_correct"]

            loo_changed_rate = changed_num / changed_den if changed_den else changed_rate
            loo_preserve_rate = preserve_num / preserve_den if preserve_den else preserve_rate
            loo_all_rate = total_num / total_den if total_den else all_rate

            expected_changed_preserve[architecture][key] = (
                (loo_changed_rate ** counts["changed"]) * (loo_preserve_rate ** counts["preserve"])
            )
            expected_field_count_only[architecture][key] = loo_all_rate ** counts["total"]

    return expected_changed_preserve, expected_field_count_only, rate_rows


def summarize_per_architecture(
    runs: dict[str, dict[str, dict[str, Any]]],
    expected_changed_preserve: dict[str, dict[str, float]],
    expected_field_count_only: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    keys = sorted(next(iter(runs.values())).keys())
    for architecture, utterances in runs.items():
        for bucket in LOAD_BUCKETS:
            bucket_keys = [key for key in keys if is_in_bucket(load_bucket(utterances[key]), bucket)]
            bucket_utterances = [utterances[key] for key in bucket_keys]
            if not bucket_keys:
                continue
            exact_count = sum(exact_match(u) for u in bucket_utterances)
            item_count = len(bucket_keys)
            mean_fields = mean(
                [
                    u["metric_views"]["all_fields"].get(
                        "total_evaluated_leaf_fields", u["metric_views"]["all_fields"]["gt_expected_total"]
                    )
                    for u in bucket_utterances
                ]
            )
            mean_changed = mean([u["metric_views"]["changed_fields"]["gold_changed_total"] for u in bucket_utterances])
            expected_cp = mean([expected_changed_preserve[architecture][key] for key in bucket_keys])
            expected_n = mean([expected_field_count_only[architecture][key] for key in bucket_keys])
            observed = exact_count / item_count
            rows.append(
                {
                    "architecture": architecture,
                    "load_bucket": bucket,
                    "item_count": item_count,
                    "mean_fields": mean_fields,
                    "mean_gold_changed_fields": mean_changed,
                    "exact_count": exact_count,
                    "exact_rate": observed,
                    "expected_exact_changed_preserve": expected_cp,
                    "expected_exact_field_count_only": expected_n,
                    "residual_changed_preserve": observed - expected_cp,
                    "residual_field_count_only": observed - expected_n,
                    "errors_per_100_fields": errors_per_100_fields(bucket_utterances),
                    "changed_f1": changed_f1(bucket_utterances),
                }
            )
    return rows


def summarize_family(
    runs: dict[str, dict[str, dict[str, Any]]],
    expected_changed_preserve: dict[str, dict[str, float]],
    expected_field_count_only: dict[str, dict[str, float]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    architectures = list(runs.keys())
    keys = sorted(next(iter(runs.values())).keys())
    reference = runs["Direct JSON"]
    rng = random.Random(seed)

    for bucket in LOAD_BUCKETS:
        bucket_keys = [key for key in keys if is_in_bucket(load_bucket(reference[key]), bucket)]
        if not bucket_keys:
            continue

        def stats(sample_keys: list[str]) -> dict[str, float]:
            solved_count = sum(
                any(exact_match(runs[architecture][key]) for architecture in architectures) for key in sample_keys
            )
            observed = solved_count / len(sample_keys)
            best_cp = mean([max(expected_changed_preserve[a][key] for a in architectures) for key in sample_keys])
            union_cp = mean(
                [
                    1.0 - math.prod(1.0 - expected_changed_preserve[a][key] for a in architectures)
                    for key in sample_keys
                ]
            )
            best_n = mean([max(expected_field_count_only[a][key] for a in architectures) for key in sample_keys])
            union_n = mean(
                [1.0 - math.prod(1.0 - expected_field_count_only[a][key] for a in architectures) for key in sample_keys]
            )
            err_avg = mean(
                [errors_per_100_fields([runs[architecture][key] for key in sample_keys]) for architecture in architectures]
            )
            f1_avg = mean([changed_f1([runs[architecture][key] for key in sample_keys]) for architecture in architectures])
            return {
                "solved_by_any_count": solved_count,
                "solved_by_any_rate": observed,
                "expected_best_changed_preserve": best_cp,
                "expected_union_changed_preserve": union_cp,
                "expected_best_field_count_only": best_n,
                "expected_union_field_count_only": union_n,
                "mean_errors_per_100_fields": err_avg,
                "mean_changed_f1": f1_avg,
            }

        row_stats = stats(bucket_keys)
        boot: dict[str, list[float]] = defaultdict(list)
        for _ in range(bootstrap_samples):
            sample = [bucket_keys[rng.randrange(len(bucket_keys))] for _ in bucket_keys]
            sample_stats = stats(sample)
            for name, value in sample_stats.items():
                if name.endswith("_count"):
                    continue
                boot[name].append(value)

        mean_fields = mean(
            [
                reference[key]["metric_views"]["all_fields"].get(
                    "total_evaluated_leaf_fields", reference[key]["metric_views"]["all_fields"]["gt_expected_total"]
                )
                for key in bucket_keys
            ]
        )
        mean_changed = mean([reference[key]["metric_views"]["changed_fields"]["gold_changed_total"] for key in bucket_keys])
        rows.append(
            {
                "load_bucket": bucket,
                "item_count": len(bucket_keys),
                "mean_fields": mean_fields,
                "mean_gold_changed_fields": mean_changed,
                **row_stats,
                "solved_by_any_rate_ci_low": percentile(boot["solved_by_any_rate"], 0.025),
                "solved_by_any_rate_ci_high": percentile(boot["solved_by_any_rate"], 0.975),
                "expected_union_changed_preserve_ci_low": percentile(
                    boot["expected_union_changed_preserve"], 0.025
                ),
                "expected_union_changed_preserve_ci_high": percentile(
                    boot["expected_union_changed_preserve"], 0.975
                ),
                "expected_union_field_count_only_ci_low": percentile(
                    boot["expected_union_field_count_only"], 0.025
                ),
                "expected_union_field_count_only_ci_high": percentile(
                    boot["expected_union_field_count_only"], 0.975
                ),
                "mean_errors_per_100_fields_ci_low": percentile(boot["mean_errors_per_100_fields"], 0.025),
                "mean_errors_per_100_fields_ci_high": percentile(boot["mean_errors_per_100_fields"], 0.975),
                "residual_vs_best_changed_preserve": row_stats["solved_by_any_rate"]
                - row_stats["expected_best_changed_preserve"],
                "residual_vs_union_changed_preserve": row_stats["solved_by_any_rate"]
                - row_stats["expected_union_changed_preserve"],
                "residual_vs_best_field_count_only": row_stats["solved_by_any_rate"]
                - row_stats["expected_best_field_count_only"],
                "residual_vs_union_field_count_only": row_stats["solved_by_any_rate"]
                - row_stats["expected_union_field_count_only"],
            }
        )
    return rows


def summarize_per_item(
    runs: dict[str, dict[str, dict[str, Any]]],
    expected_changed_preserve: dict[str, dict[str, float]],
    expected_field_count_only: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    architectures = list(runs.keys())
    keys = sorted(next(iter(runs.values())).keys())
    reference = runs["Direct JSON"]
    for key in keys:
        solved_by = [architecture for architecture in architectures if exact_match(runs[architecture][key])]
        rows.append(
            {
                "item_key": key,
                "load_bucket": load_bucket(reference[key]),
                "fields": reference[key]["metric_views"]["all_fields"].get(
                    "total_evaluated_leaf_fields", reference[key]["metric_views"]["all_fields"]["gt_expected_total"]
                ),
                "gold_changed_fields": reference[key]["derived_variables"]["changed_leaf_field_count"],
                "solved_by_any": bool(solved_by),
                "solved_by_count": len(solved_by),
                "solved_by": ";".join(solved_by),
                "expected_best_changed_preserve": max(expected_changed_preserve[a][key] for a in architectures),
                "expected_union_changed_preserve": 1.0
                - math.prod(1.0 - expected_changed_preserve[a][key] for a in architectures),
                "expected_best_field_count_only": max(expected_field_count_only[a][key] for a in architectures),
                "expected_union_field_count_only": 1.0
                - math.prod(1.0 - expected_field_count_only[a][key] for a in architectures),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, family_rows: list[dict[str, Any]], per_arch_rows: list[dict[str, Any]]) -> None:
    def pct_ci(row: dict[str, Any], name: str) -> str:
        return f"{pct(row[name])} [{pct(row[name + '_ci_low'])}, {pct(row[name + '_ci_high'])}]"

    def family_line(row: dict[str, Any]) -> str:
        return (
            f"| {row['load_bucket']} | {row['item_count']} | {row['mean_fields']:.1f} | "
            f"{row['mean_gold_changed_fields']:.1f} | {row['solved_by_any_count']}/{row['item_count']} "
            f"({pct_ci(row, 'solved_by_any_rate')}) | {pct_ci(row, 'expected_union_field_count_only')} | "
            f"{pct_ci(row, 'expected_union_changed_preserve')} | "
            f"{row['mean_errors_per_100_fields']:.2f} "
            f"[{row['mean_errors_per_100_fields_ci_low']:.2f}, "
            f"{row['mean_errors_per_100_fields_ci_high']:.2f}] | {pct(row['mean_changed_f1'])} |"
        )

    lines = [
        "# Load-Normalized Exact-Record Analysis",
        "",
        "The field-count-only null raises each architecture's strict all-field accuracy to each "
        "item/architecture evaluated field count: q_field = p_all(-i)^n_ia. The changed/preserve null uses separate "
        "strict rates for gold-changed and gold-preserved field checks: "
        "q_cp = p_chg(-i)^g_ia * p_pres(-i)^p_ia. Expected values are leave-one-item-out within architecture. "
        "The family rows report architecture-family solvability: an item is solved if at least one "
        "of the five GPT-5.4-mini system rows is exact. Family expected rates use "
        "1 - prod_a(1 - q_ai), so they are descriptive independence approximations rather than "
        "a model of the correlated architecture errors. Brackets are 95% item-bootstrap intervals.",
        "",
        "## Architecture-Family Solvability",
        "",
        "| Load | Items | Mean fields | Mean changed | Observed solved | Field-count null | Changed/preserve null | Mean errors / 100 fields | Mean changed F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    lines.extend(family_line(row) for row in family_rows)
    lines.extend(
        [
            "",
            "## Per-Architecture Exact Rates",
            "",
            "| Architecture | Load | Observed exact | Changed/preserve null | Field-count null | Errors / 100 fields | Changed F1 |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in per_arch_rows:
        if row["load_bucket"] not in {"low", "medium", "high+scale"}:
            continue
        lines.append(
            f"| {row['architecture']} | {row['load_bucket']} | "
            f"{row['exact_count']}/{row['item_count']} ({pct(row['exact_rate'])}) | "
            f"{pct(row['expected_exact_changed_preserve'])} | "
            f"{pct(row['expected_exact_field_count_only'])} | "
            f"{row['errors_per_100_fields']:.2f} | {pct(row['changed_f1'])} |"
        )
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=data_root() / "reports/robustness/five_system_load_normalized_seed20260525",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260525)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = data_root()
    runs = load_runs(root)
    expected_cp, expected_n, rate_rows = compute_expected_exact(runs)
    per_arch_rows = summarize_per_architecture(runs, expected_cp, expected_n)
    family_rows = summarize_family(
        runs,
        expected_cp,
        expected_n,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    item_rows = summarize_per_item(runs, expected_cp, expected_n)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "strict_field_rates.csv", rate_rows)
    write_csv(args.output_dir / "per_arch_load_normalized.csv", per_arch_rows)
    write_csv(args.output_dir / "family_load_normalized.csv", family_rows)
    write_csv(args.output_dir / "per_item_load_normalized.csv", item_rows)
    write_markdown(args.output_dir / "load_normalized_exact.md", family_rows, per_arch_rows)

    summary = {
        "method": {
            "field_count_only": "leave-one-item-out strict all-field accuracy raised to item/architecture evaluated field count",
            "changed_preserve": "leave-one-item-out strict changed-field and preservation rates raised to item burdens",
            "family_solvability": "item is solved if at least one architecture is exact",
            "family_expected": "descriptive independence approximation 1 - prod_a(1 - q_ia)",
        },
        "bootstrap": {"samples": args.bootstrap_samples, "seed": args.seed},
        "input_runs": ARCHITECTURE_RUNS,
        "rate_rows": rate_rows,
        "family_rows": family_rows,
        "per_arch_rows": per_arch_rows,
    }
    (args.output_dir / "load_normalized_exact.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(f"Wrote {args.output_dir}")
    for row in family_rows:
        if row["load_bucket"] in {"low", "medium", "high+scale"}:
            print(
                row["load_bucket"],
                f"observed={row['solved_by_any_count']}/{row['item_count']} "
                f"({pct(row['solved_by_any_rate'])} "
                f"[{pct(row['solved_by_any_rate_ci_low'])}, {pct(row['solved_by_any_rate_ci_high'])}])",
                f"field_count_null={pct(row['expected_union_field_count_only'])}",
                f"changed_preserve_null={pct(row['expected_union_changed_preserve'])}",
                f"errors_per_100={row['mean_errors_per_100_fields']:.2f}",
            )


if __name__ == "__main__":
    main()
