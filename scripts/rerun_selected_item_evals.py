#!/usr/bin/env python3
"""Re-score selected repaired items across the paper's saved run artifacts.

This script keeps model outputs fixed. It reads each saved ``turn_result.json``,
resolves the current ground truth from the split item directory, rewrites only
the selected per-item ``evaluation.json`` files, then optionally refreshes the
run-level reports.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from conformbench import benchmark
from conformbench.evaluator.pipeline import evaluate_solver_turn
from conformbench.items import DATA_ROOT
from scripts.generate_full_run_report import generate_report_for_run_dir


SPLIT_RUNS_DIR = (
    DATA_ROOT
    / "reports"
    / "split_runs"
    / "conformbench_v1_20_80_seed20260524"
    / "test"
)
RUNS_DIR = DATA_ROOT / "reports" / "runs"
DEFAULT_ITEMS_DIR = (
    DATA_ROOT / "items" / "splits" / "conformbench_v1_20_80_seed20260524" / "test"
)

REPORTED_RUNS: dict[str, Path] = {
    "arch_direct_gpt54mini": (
        SPLIT_RUNS_DIR
        / "audit_tight_test_20260525__direct__gen-gpt-5-4-mini__judge-gpt-5-4-medium"
    ),
    "arch_update_tool_gpt54mini": (
        SPLIT_RUNS_DIR
        / "audit_tight_test_20260525__flatagent__gen-gpt-5-4-mini__judge-gpt-5-4-medium"
    ),
    "arch_split_agent_gpt54mini": (
        SPLIT_RUNS_DIR
        / "audit_tight_test_20260525__split_agent__gen-gpt-5-4-mini__judge-gpt-5-4-medium"
    ),
    "arch_triage_briefed_gpt54mini": (
        SPLIT_RUNS_DIR
        / "audit_tight_test_20260525__split_agent_briefed__gen-gpt-5-4-mini__judge-gpt-5-4-medium"
    ),
    "model_direct_gpt54mini": (
        RUNS_DIR / "TEST144.direct-json.gpt-5-4-mini.judge-gpt-5-4.heldout.20260524"
    ),
    "model_direct_claude_opus_4_7": (
        RUNS_DIR
        / "TEST144.direct-json.claude-opus-4-7-medium.judge-gpt-5-4-medium.ablation-promptfix-20260525__direct__gen-anthropic-claude-opus-4-7__judge-openai-gpt-5-4-medium"
    ),
    "model_direct_gpt_oss_120b": (
        RUNS_DIR
        / "TEST144.direct-json.gpt-oss-120b.groq-reasoning-medium.judge-gpt-5-4-medium.ablation-promptfix-20260525__direct__gen-openai-gpt-oss-120b__judge-openai-gpt-5-4-medium"
    ),
    "model_direct_qwen3_32b": (
        RUNS_DIR
        / "TEST144.direct-json.qwen3-32b.groq-reasoning-default.judge-gpt-5-4-medium.ablation-promptfix-20260525__direct__gen-qwen-qwen3-32b__judge-openai-gpt-5-4-medium"
    ),
    "sanity_no_update_split": SPLIT_RUNS_DIR / "NoUpdate_strict_prior_20260524",
    "sanity_no_update_report": (
        RUNS_DIR / "TEST144.no-update-strict-prior.deterministic.heldout.20260524"
    ),
}
COMPARISON_MANIFESTS = [
    SPLIT_RUNS_DIR / "audit_tight_test_20260525__comparison_manifest.json",
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = read_json(path)
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def parse_item_tokens(tokens: list[str]) -> list[str]:
    items: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        for part in token.replace(",", " ").split():
            item_id = part.strip()
            if not item_id or item_id.startswith("#") or item_id in seen:
                continue
            seen.add(item_id)
            items.append(item_id)
    return items


def load_item_ids(args: argparse.Namespace) -> list[str]:
    tokens: list[str] = []
    tokens.extend(args.items or [])
    tokens.extend(args.item or [])
    if args.items_file:
        for line in args.items_file.read_text(encoding="utf-8").splitlines():
            stripped = line.split("#", 1)[0].strip()
            if stripped:
                tokens.append(stripped)
    return parse_item_tokens(tokens)


def reported_run_paths() -> list[tuple[str, Path]]:
    paths: list[tuple[str, Path]] = []
    seen: set[Path] = set()
    for label, path in REPORTED_RUNS.items():
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        paths.append((label, path))
    return paths


def selected_run_paths(args: argparse.Namespace) -> list[tuple[str, Path]]:
    if args.run_dir:
        paths: list[tuple[str, Path]] = []
        seen: set[Path] = set()
        for path in args.run_dir:
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            paths.append((path.name, path))
        return paths
    return reported_run_paths()


def index_turn_results(run_dir: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    duplicates: dict[str, list[Path]] = {}
    for turn_path in sorted(run_dir.rglob("turn_result.json")):
        try:
            turn_result = read_json(turn_path)
        except (OSError, json.JSONDecodeError):
            continue
        item_id = str(turn_result.get("scenario_id") or "")
        if not item_id:
            continue
        if item_id in index:
            duplicates.setdefault(item_id, [index[item_id]]).append(turn_path)
            continue
        index[item_id] = turn_path
    if duplicates:
        duplicate_list = ", ".join(sorted(duplicates))
        raise RuntimeError(f"{run_dir} contains duplicate item turn results: {duplicate_list}")
    return index


def evaluator_config(run_dir: Path) -> tuple[str | None, str | None]:
    metadata = read_optional_json(run_dir / "run_metadata.json")
    model = metadata.get("evaluator_model_id")
    reasoning = metadata.get("evaluator_reasoning_effort")
    if model or reasoning:
        return (
            str(model) if model is not None else None,
            str(reasoning) if reasoning is not None else None,
        )

    for eval_path in sorted(run_dir.rglob("evaluation.json")):
        evaluation = read_optional_json(eval_path)
        provenance = evaluation.get("provenance") or {}
        eval_prov = provenance.get("evaluation") or {}
        if not isinstance(eval_prov, dict):
            continue
        eval_model = eval_prov.get("model") or {}
        if not isinstance(eval_model, dict):
            eval_model = {}
        model = eval_prov.get("requested_model") or eval_model.get("requested_model")
        reasoning = eval_prov.get("reasoning_effort") or eval_model.get("reasoning_effort")
        if model or reasoning:
            return (
                str(model) if model is not None else None,
                str(reasoning) if reasoning is not None else None,
            )
    return None, None


def reevaluate_item(
    *,
    turn_path: Path,
    items_dir: Path,
    evaluator_model_id: str | None,
    evaluator_reasoning_effort: str | None,
) -> Path:
    turn_result = read_json(turn_path)
    ground_truth = benchmark._resolve_ground_truth_for_turn(turn_result, items_dir=items_dir)
    questionnaire_id = (
        turn_result.get("questionnaire")
        or ground_truth.get("questionnaire_id")
        or ground_truth.get("questionnaire")
    )
    evaluation = evaluate_solver_turn(
        turn_result,
        ground_truth,
        model_id=evaluator_model_id,
        reasoning_effort=evaluator_reasoning_effort,
        questionnaire_path=benchmark._questionnaire_path(str(questionnaire_id)),
        runner_version={
            "name": "scripts/rerun_selected_item_evals.py",
            "items_dir": str(items_dir),
        },
    )
    output_path = turn_path.parent / "evaluation.json"
    write_json(output_path, evaluation.model_dump(mode="json"))
    return output_path


def append_rerun_metadata(
    *,
    run_dir: Path,
    item_ids: list[str],
    items_dir: Path,
    refresh_reports: bool,
) -> None:
    metadata_path = run_dir / "run_metadata.json"
    metadata = read_optional_json(metadata_path)
    if not metadata:
        metadata = {}
    log = metadata.setdefault("selected_item_reevaluations", [])
    if isinstance(log, list):
        log.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "script": "scripts/rerun_selected_item_evals.py",
                "items": item_ids,
                "items_dir": str(items_dir),
                "refresh_reports": refresh_reports,
            }
        )
    metadata["latest_items_dir_for_metrics"] = str(items_dir)
    write_json(metadata_path, metadata)


def refresh_comparison_manifest(path: Path) -> Path:
    from scripts.run_architecture_comparison import write_comparison_markdown

    comparison = read_json(path)
    for run in comparison.get("runs") or []:
        run_dir_value = run.get("run_dir")
        if not run_dir_value:
            continue
        run_dir = Path(str(run_dir_value))
        if not run_dir.is_absolute():
            run_dir = ROOT / run_dir
        summary_path = run_dir / "summary_report.json"
        if not summary_path.exists():
            continue
        summary = read_json(summary_path)
        utterances = summary.get("utterances") or []
        views = (summary.get("aggregate") or {}).get("metric_views") or {}
        all_fields = views.get("all_fields") or {}
        changed = (views.get("changed_fields") or {}).get("strict") or {}
        whole_record = views.get("whole_record_exact_match") or {}
        preserve = ((views.get("transition_accuracy") or {}).get("by_transition") or {}).get(
            "preserve",
            {},
        )
        diagnostics = views.get("task_diagnostics") or {}
        run.update(
            {
                "all_field_accuracy": all_fields.get("accuracy"),
                "changed_f1": changed.get("f1"),
                "changed_precision": changed.get("precision"),
                "changed_recall": changed.get("recall"),
                "collateral_edit_count": diagnostics.get("collateral_edit_count"),
                "evaluated_items": len(utterances) or whole_record.get("item_count"),
                "exact_match_count": whole_record.get("exact_match_count"),
                "exact_match_rate": whole_record.get("exact_match_rate"),
                "preservation_success_rate": preserve.get("accuracy"),
            }
        )
    write_json(path, comparison)
    markdown_path = path.with_name(path.name.replace("_manifest.json", ".md"))
    write_comparison_markdown(markdown_path, comparison)
    return markdown_path


def print_reported_runs() -> None:
    for label, path in reported_run_paths():
        status = "ok" if path.exists() else "missing"
        print(f"{label}\t{status}\t{path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("items", nargs="*", help="Item IDs to re-score.")
    parser.add_argument("--item", action="append", help="Additional item ID to re-score.")
    parser.add_argument("--items-file", type=Path, help="Newline/comma separated item IDs.")
    parser.add_argument(
        "--items-dir",
        type=Path,
        default=DEFAULT_ITEMS_DIR,
        help=f"Ground-truth split item directory. Default: {DEFAULT_ITEMS_DIR}",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        action="append",
        help="Run directory to update. May be repeated. Defaults to the paper reported runs.",
    )
    parser.add_argument(
        "--list-reported-runs",
        action="store_true",
        help="List the default reported run directories and exit.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Skip requested items that are absent from a run instead of failing.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve paths and print the work plan without rewriting files.",
    )
    parser.add_argument(
        "--no-refresh-reports",
        action="store_true",
        help="Rewrite selected evaluation.json files but leave run-level reports untouched.",
    )
    parser.add_argument(
        "--refresh-only",
        action="store_true",
        help="Skip item-level evaluation and only rebuild run-level reports.",
    )
    parser.add_argument(
        "--figures",
        action="store_true",
        help="Regenerate figures while refreshing reports. Off by default for repair speed.",
    )
    parser.add_argument(
        "--write-run-metadata-log",
        action="store_true",
        help="Append a selected_item_reevaluations log entry to each run_metadata.json.",
    )
    parser.add_argument(
        "--no-refresh-comparison-manifests",
        action="store_true",
        help="Do not refresh architecture comparison manifest/markdown caches.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.list_reported_runs:
        print_reported_runs()
        return 0

    items_dir = args.items_dir.resolve()
    if not items_dir.exists():
        print(f"Error: items directory not found: {items_dir}", file=sys.stderr)
        return 2

    item_ids = load_item_ids(args)
    if not item_ids and not args.refresh_only:
        print("Error: pass at least one item ID, --items-file, or --refresh-only.", file=sys.stderr)
        return 2

    runs = selected_run_paths(args)
    missing_runs = [(label, path) for label, path in runs if not path.exists()]
    if missing_runs:
        for label, path in missing_runs:
            print(f"Missing run directory for {label}: {path}", file=sys.stderr)
        return 2

    refresh_reports = not args.no_refresh_reports
    print(f"Items dir: {items_dir}")
    if item_ids:
        print(f"Selected items ({len(item_ids)}): {', '.join(item_ids)}")
    else:
        print("Selected items: none (refresh-only)")
    print(f"Run directories ({len(runs)}):")
    for label, path in runs:
        print(f"  - {label}: {path}")

    if args.dry_run:
        for label, run_dir in runs:
            index = index_turn_results(run_dir)
            found = [item_id for item_id in item_ids if item_id in index]
            missing = [item_id for item_id in item_ids if item_id not in index]
            print(f"[dry-run] {label}: {len(found)} item evals would be rewritten")
            for item_id in found:
                print(f"    {item_id}: {index[item_id].parent / 'evaluation.json'}")
            if missing:
                print(f"    missing: {', '.join(missing)}")
        if refresh_reports:
            print("[dry-run] run-level reports would be refreshed")
            if not args.run_dir and not args.no_refresh_comparison_manifests:
                for manifest_path in COMPARISON_MANIFESTS:
                    print(f"[dry-run] comparison manifest would be refreshed: {manifest_path}")
        return 0

    for label, run_dir in runs:
        evaluator_model_id, evaluator_reasoning_effort = evaluator_config(run_dir)
        index = index_turn_results(run_dir)
        missing = [item_id for item_id in item_ids if item_id not in index]
        if missing and not args.allow_missing:
            print(
                f"Error: {label} is missing item(s): {', '.join(missing)}",
                file=sys.stderr,
            )
            return 1
        rewritten: list[Path] = []
        if not args.refresh_only:
            for item_id in item_ids:
                turn_path = index.get(item_id)
                if turn_path is None:
                    continue
                output_path = reevaluate_item(
                    turn_path=turn_path,
                    items_dir=items_dir,
                    evaluator_model_id=evaluator_model_id,
                    evaluator_reasoning_effort=evaluator_reasoning_effort,
                )
                rewritten.append(output_path)
            print(f"{label}: rewrote {len(rewritten)} item evaluation(s)")
            if args.write_run_metadata_log:
                append_rerun_metadata(
                    run_dir=run_dir,
                    item_ids=[item_id for item_id in item_ids if item_id in index],
                    items_dir=items_dir,
                    refresh_reports=refresh_reports,
                )

        if refresh_reports:
            csv_path, md_path, row_count = generate_report_for_run_dir(
                run_dir,
                items_dir=items_dir,
                generate_figures=args.figures,
            )
            print(f"{label}: refreshed {csv_path} ({row_count} rows)")
            print(f"{label}: refreshed {md_path}")

    if refresh_reports and not args.run_dir and not args.no_refresh_comparison_manifests:
        for manifest_path in COMPARISON_MANIFESTS:
            if not manifest_path.exists():
                print(f"comparison manifest missing, skipped: {manifest_path}")
                continue
            markdown_path = refresh_comparison_manifest(manifest_path)
            print(f"refreshed comparison manifest: {manifest_path}")
            print(f"refreshed comparison markdown: {markdown_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
