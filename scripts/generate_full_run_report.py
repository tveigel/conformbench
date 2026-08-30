#!/usr/bin/env python3
"""Generate a CSV + Markdown report for a scored run directory.

Usage:
    uv run python scripts/generate_full_run_report.py --run-dir data/reports/runs/<run_id>
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from conformbench.items import DATA_ROOT

EVAL_RESULTS_DIR = DATA_ROOT / "eval_results"
EVAL_RUNS_DIR = DATA_ROOT / "reports" / "runs"
DEFAULT_OUTPUT_DIR = DATA_ROOT / "reports" / "full_runs"
BENCHMARK_ITEMS_DIR = DATA_ROOT / "items" / "benchmark"
PUBLIC_ITEMS_DIR = DATA_ROOT / "items" / "public"

CSV_FIELDNAMES = [
    "item_id",
    "state",
    "questionnaire",
    "field_path",
    "correctness",
    "decision_source",
    "source",
    "partial_reason",
    "gold_value",
    "candidate_value",
    "reasoning",
]


def _portable_data_path(path: Path, *, shell: bool = False) -> str:
    """Return a portable data path suitable for public reports."""
    try:
        relative = path.resolve().relative_to(DATA_ROOT.resolve())
    except ValueError:
        return "<PATH>"
    prefix = "$CONFORMBENCH_DATA_DIR" if shell else "<CONFORMBENCH_DATA_DIR>"
    return f"{prefix}/{relative.as_posix()}"


# ── Helpers ───────────────────────────────────────────────────────────────


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def slug(text: str) -> str:
    text = text.replace("/", "_").replace(" ", "_")
    text = re.sub(r"[^a-zA-Z0-9._-]", "", text)
    return text


def pct(numerator: float | int, denominator: float | int) -> str:
    if not denominator:
        return "n/a"
    return f"{(numerator / denominator * 100):.1f}%"


def rate_str(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.2f}%"


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


# ── Run resolution ────────────────────────────────────────────────────────


def resolve_run(run_id: str) -> dict[str, Any]:
    path = EVAL_RESULTS_DIR / f"{run_id}.json"
    if not path.exists():
        print(f"Error: no eval result at {path}", file=sys.stderr)
        sys.exit(1)
    metadata = read_json(path)

    display = metadata.get("display") or {}
    provenance = metadata.get("run_provenance") or {}
    agent_info = provenance.get("agent") or provenance.get("generation") or {}

    return {
        "run_id": run_id,
        "metadata": metadata,
        "summary_report_path": Path(metadata["summary_report_path"]),
        "agent_label": display.get("agent_label") or agent_info.get("label", "Agent"),
        "model_label": display.get("model_label") or agent_info.get("model_label", "unknown"),
        "provider": display.get("provider") or agent_info.get("provider", ""),
        "run_label": display.get("run_label", ""),
        "questionnaire_scope": display.get("questionnaire_scope", "all"),
        "scenario_scope": display.get("scenario_scope", "all"),
        "invoked_at": (provenance.get("runner") or {}).get("invoked_at", ""),
        "git_sha": (provenance.get("runner") or {}).get("git_sha", ""),
        "stats": metadata.get("stats") or {},
    }


def output_dir_name(run_info: dict[str, Any]) -> str:
    agent = slug(run_info["agent_label"])
    model = slug(run_info["model_label"])
    ts = run_info.get("invoked_at", "")
    if ts:
        try:
            dt = datetime.fromisoformat(ts)
            ts_slug = dt.strftime("%Y%m%dT%H%M%S")
        except (ValueError, TypeError):
            ts_slug = slug(ts[:19])
    else:
        ts_slug = run_info["run_id"][:12]
    return f"{agent}_{model}_{ts_slug}"


# ── CSV generation ────────────────────────────────────────────────────────


def build_csv_rows(
    runs_dir: Path,
    simulators_dir: Path,
    *,
    items_dir: Path | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for eval_path in sorted(runs_dir.rglob("evaluation.json")):
        evaluation = read_json(eval_path)
        field_results = evaluation.get("field_results") or {}
        candidate_state = evaluation.get("candidate_resulting_state") or {}

        turn_result_path = eval_path.parent / "turn_result.json"
        if turn_result_path.exists():
            turn_result = read_json(turn_result_path)
        else:
            turn_result = {}

        scenario_id = turn_result.get("scenario_id", "")
        questionnaire = turn_result.get("questionnaire", "")
        state = turn_result.get("state", "")

        gt_path = _resolve_ground_truth(
            eval_path,
            simulators_dir,
            questionnaire,
            scenario_id,
            items_dir=items_dir,
        )
        gt_fields: dict[str, Any] = {}
        gold_state: dict[str, Any] = {}
        if gt_path and gt_path.exists():
            gt_data = read_json(gt_path)
            gt_fields = gt_data.get("fields") or {}
            gold_state = gt_data.get("gold_resulting_state") or {}

        for qid, result in field_results.items():
            if not isinstance(result, dict):
                continue
            gold = _extract_gold_value(qid, gt_fields, gold_state)
            candidate = _extract_candidate_value(qid, candidate_state)

            rows.append({
                "item_id": scenario_id,
                "state": state,
                "questionnaire": questionnaire,
                "field_path": qid,
                "correctness": result.get("final_correctness") or result.get("correctness", ""),
                "decision_source": result.get("decision_source", ""),
                "source": result.get("source", ""),
                "partial_reason": result.get("partial_reason") or "",
                "gold_value": _format_value(gold),
                "candidate_value": _format_value(candidate),
                "reasoning": result.get("reasoning", ""),
            })

    return rows


def _resolve_ground_truth(
    eval_path: Path,
    simulators_dir: Path,
    questionnaire: str,
    scenario_id: str,
    *,
    items_dir: Path | None = None,
) -> Path | None:
    if items_dir is not None:
        gt_path = _find_ground_truth_packet(
            scenario_id,
            questionnaire,
            extra_roots=[items_dir],
        )
        if gt_path is not None:
            return gt_path

    sibling = eval_path.parent / "ground_truth.json"
    if sibling.exists():
        return sibling

    parts = eval_path.relative_to(eval_path.parents[4]).parts
    if len(parts) >= 2:
        q_dir = parts[0]
        scenario_dir = parts[1]
        gt_path = simulators_dir / q_dir / scenario_dir / "ground_truth.json"
        if gt_path.exists():
            return gt_path
    return _find_ground_truth_packet(scenario_id, questionnaire)


def _find_ground_truth_packet(
    scenario_id: str,
    questionnaire: str,
    *,
    extra_roots: list[Path] | None = None,
) -> Path | None:
    if not scenario_id:
        return None
    questionnaire = questionnaire.removeprefix("pilot_")
    roots = [*(extra_roots or []), BENCHMARK_ITEMS_DIR, PUBLIC_ITEMS_DIR]
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("ground_truth.json")):
            try:
                packet = read_json(path)
            except (OSError, json.JSONDecodeError):
                continue
            item_id = packet.get("item_id") or packet.get("scenario_id")
            packet_questionnaire = (
                packet.get("questionnaire_id")
                or packet.get("source_questionnaire")
                or packet.get("questionnaire")
                or ""
            )
            packet_questionnaire = str(packet_questionnaire).removeprefix("pilot_")
            if item_id == scenario_id and (
                not questionnaire or packet_questionnaire == questionnaire
            ):
                return path
    return None


def _extract_gold_value(
    qid: str,
    gt_fields: dict[str, Any],
    gold_state: dict[str, Any] | None = None,
) -> Any:
    entry = gt_fields.get(qid)
    if isinstance(entry, dict):
        return entry.get("expected_summary") or entry.get("expected")
    if gold_state:
        value = _extract_candidate_value(qid, gold_state)
        if value is not None:
            return value
    return entry


def _extract_candidate_value(qid: str, candidate_state: dict[str, Any]) -> Any:
    if qid in candidate_state:
        return candidate_state[qid]
    m = re.match(r"^(.+)\[(\d+)\]\.(.+)$", qid)
    if m:
        group, idx, field = m.group(1), int(m.group(2)), m.group(3)
        group_data = candidate_state.get(group)
        if isinstance(group_data, list) and idx < len(group_data):
            row = group_data[idx]
            if isinstance(row, dict):
                return row.get(field)
    return None


def _format_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


# ── Markdown generation ───────────────────────────────────────────────────


def build_markdown(summary: dict[str, Any], run_info: dict[str, Any]) -> str:
    agg = summary.get("aggregate") or {}
    utterances = summary.get("utterances") or []
    provenance = summary.get("provenance") or {}
    split_metadata = run_info.get("split_view_metadata") or {}

    lines: list[str] = []

    # Header
    title_prefix = "Full Run Report"
    if split_metadata.get("split") in {"dev", "train"}:
        title_prefix = "Development Run Report"
    elif split_metadata.get("split") == "test":
        title_prefix = "Held-Out Test Run Report"
    lines.append(f"# {title_prefix}: {run_info['agent_label']} / {run_info['model_label']}")
    lines.append("")

    # Run metadata
    lines.append("## Run Metadata")
    lines.append("")
    lines.append(f"- **Run ID:** `{run_info['run_id']}`")
    lines.append(f"- **Agent:** {run_info['agent_label']}")
    lines.append(f"- **Model:** {run_info['model_label']}")
    if run_info.get("run_reasoning_effort"):
        lines.append(f"- **Run reasoning:** {run_info['run_reasoning_effort']}")
    if run_info.get("judge_model_label"):
        lines.append(f"- **Judge model:** {run_info['judge_model_label']}")
    if run_info.get("judge_reasoning_effort"):
        lines.append(f"- **Judge reasoning config:** {run_info['judge_reasoning_effort']}")
    if run_info.get("provider"):
        lines.append(f"- **Provider:** {run_info['provider']}")
    scenario_scope = run_info["scenario_scope"]
    if split_metadata.get("split"):
        scenario_scope = f"{split_metadata['split']} split"
    lines.append(f"- **Scope:** {run_info['questionnaire_scope']} / {scenario_scope}")
    lines.append(f"- **Items:** {len(utterances)}")
    if split_metadata:
        split_label = split_metadata.get("split") or "unknown"
        split_id = split_metadata.get("split_id") or ""
        lines.append(f"- **Declared split:** {split_label} (`{split_id}`)")
        if split_metadata.get("declared_label"):
            lines.append(f"- **Declared use:** {split_metadata['declared_label']}")
        if split_metadata.get("note"):
            lines.append(f"- **Scope note:** {split_metadata['note']}")
        if split_metadata.get("archived_removed_items_dir"):
            lines.append(f"- **Archived excluded items:** `{split_metadata['archived_removed_items_dir']}`")
    if run_info.get("invoked_at"):
        lines.append(f"- **Run date:** {run_info['invoked_at']}")
    gen = provenance.get("generation") or {}
    model_cfg = gen.get("model") or run_info.get("model_config") or {}
    if isinstance(model_cfg, dict) and model_cfg:
        reasoning = model_cfg.get("reasoning_effort") or run_info.get("run_reasoning_effort") or "n/a"
        lines.append(f"- **Model config:** {model_cfg.get('resolved_model_name', '')} "
                     f"(max_tokens={model_cfg.get('max_tokens', 'n/a')}, "
                     f"temperature={model_cfg.get('temperature', 'n/a')}, "
                     f"reasoning={reasoning})")
    elif model_cfg:
        lines.append(f"- **Model config:** {model_cfg}")
    lines.append("")

    # Reproduce
    lines.append("## Reproduce")
    lines.append("")
    lines.append("```bash")
    if run_info.get("run_dir"):
        command = (
            "uv run python scripts/generate_full_run_report.py "
            f"--run-dir \"{run_info['run_dir']}\""
        )
        if run_info.get("items_dir"):
            command += f" --items-dir \"{run_info['items_dir']}\""
        lines.append(command)
    else:
        lines.append(f"uv run python scripts/generate_full_run_report.py --run-id {run_info['run_id']}")
    lines.append("```")
    lines.append("")

    # Headline stats
    lines.append("## Headline Stats")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")

    total_fields = agg.get("total_fields", 0)
    lines.append(f"| Total fields evaluated | {total_fields} |")
    lines.append(f"| Correct | {agg.get('correct', 0)} |")
    lines.append(f"| Partially correct | {agg.get('partially_correct', 0)} |")
    lines.append(f"| Incorrect | {agg.get('incorrect', 0)} |")
    lines.append(f"| Accuracy (strict) | {rate_str(agg.get('accuracy'))} |")
    lines.append(f"| Accuracy (lenient) | {rate_str(agg.get('lenient_accuracy'))} |")

    views = agg.get("metric_views") or {}
    wrem = views.get("whole_record_exact_match") or {}
    if wrem:
        lines.append(
            f"| Exact-match items | {wrem.get('exact_match_count', 0)}/{wrem.get('item_count', 0)} "
            f"({rate_str(wrem.get('exact_match_rate'))}) |"
        )

    inst = agg.get("instance_alignment") or {}
    if inst:
        lines.append(f"| Instance alignment F1 | {rate_str(inst.get('f1'))} |")

    lines.append("")

    runtime = agg.get("agent_runtime") or {}
    if runtime:
        lines.append("## Agent Runtime Diagnostics")
        lines.append("")
        lines.append("| Diagnostic | Count |")
        lines.append("|---|---:|")
        for key, label in [
            ("operation_count", "Submitted operations"),
            ("tool_call_count", "Tool update calls"),
            ("tool_error_count", "Tool calls with errors/rejections"),
            ("items_with_tool_errors", "Items with tool errors/rejections"),
            ("tool_rejected_update_count", "Rejected update paths"),
            ("empty_tool_update_call_count", "Empty tool update calls"),
            ("zero_operation_item_count", "Zero-operation items"),
            ("zero_operation_with_tool_error_count", "Zero-operation items with tool errors"),
            ("zero_operation_without_tool_call_count", "Zero-operation items without tool calls"),
        ]:
            lines.append(f"| {label} | {runtime.get(key, 0)} |")
        lines.append("")

    # ── Changed-field F1 ────────────────────────────────────────────────
    _append_changed_field_performance(lines, views)

    # Per-questionnaire breakdown
    q_breakdowns = (agg.get("questionnaires") or {}).get("breakdowns") or {}
    if q_breakdowns:
        lines.append("## Per-Questionnaire Breakdown")
        lines.append("")
        lines.append("| Questionnaire | Items | Fields | Correct | Partial | Incorrect | Accuracy | Lenient |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for q_name, q_data in sorted(q_breakdowns.items()):
            mv = (q_data.get("metric_views") or {}).get("whole_form") or {}
            lines.append(
                f"| {q_name} "
                f"| {q_data.get('utterance_count', '')} "
                f"| {mv.get('total_evaluated_leaf_fields', '')} "
                f"| {mv.get('correct', '')} "
                f"| {mv.get('partial', '')} "
                f"| {mv.get('incorrect', '')} "
                f"| {rate_str(mv.get('accuracy'))} "
                f"| {rate_str(mv.get('lenient_accuracy'))} |"
            )
        lines.append("")

    # ── Transition accuracy ──────────────────────────────────────────────
    transition = views.get("transition_accuracy") or {}
    by_transition = transition.get("by_transition") or {}
    if by_transition:
        lines.append("## Transition Accuracy")
        lines.append("")
        lines.append("Field-level accuracy by gold state transition type.")
        lines.append("")
        lines.append("| Transition | Correct | Total | Accuracy |")
        lines.append("|---|---:|---:|---:|")
        for t_name in ["preserve", "set", "change", "clear"]:
            t = by_transition.get(t_name, {})
            correct = t.get("correct", 0)
            total = t.get("gold_total", 0)
            lines.append(f"| {t_name} | {correct} | {total} | {pct(correct, total)} |")
        lines.append("")

    # ── Performance by prior-state condition (S1–S4) ─────────────────────
    if utterances:
        state_buckets: dict[str, dict[str, Any]] = {}
        for utt in utterances:
            dv = utt.get("derived_variables") or {}
            sc = dv.get("prior_state_condition", "")
            if not sc:
                continue
            bucket = state_buckets.setdefault(sc, {
                "items": 0, "fully_correct": 0,
                "correct": 0, "evaluated": 0,
                "gold_changed": 0, "predicted_changed": 0,
                "changed_correct": 0, "changed_incorrect": 0,
            })
            bucket["items"] += 1
            umv = utt.get("metric_views") or {}
            af = umv.get("all_fields") or {}
            bucket["correct"] += af.get("correct", 0)
            bucket["evaluated"] += af.get("gt_expected_total", 0)
            if af.get("accuracy", 0) == 1.0:
                bucket["fully_correct"] += 1
            cf = umv.get("changed_fields") or {}
            bucket["gold_changed"] += cf.get("gold_changed_total", 0)
            bucket["predicted_changed"] += cf.get("predicted_changed_total", 0)
            bucket["changed_correct"] += cf.get("changed_correct", 0)
            bucket["changed_incorrect"] += cf.get("changed_incorrect", 0)

        if state_buckets:
            lines.append("## Performance by Prior-State Condition")
            lines.append("")
            lines.append("S1 = empty prior, S2 = partial correct, S3 = wrong prior (repair licensed), S4 = silent state-history mismatch.")
            lines.append("")
            lines.append("| State | Items | Fully correct | All-field acc | Changed P | Changed R | Changed F1 |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|")
            for sc in sorted(state_buckets):
                b = state_buckets[sc]
                n = b["items"]
                fc = b["fully_correct"]
                af_acc = pct(b["correct"], b["evaluated"])
                tp = b["changed_correct"]
                fp = b["predicted_changed"] - tp
                fn = b["gold_changed"] - tp
                c_p = tp / (tp + fp) if (tp + fp) > 0 else 0
                c_r = tp / (tp + fn) if (tp + fn) > 0 else 0
                c_f1 = 2 * c_p * c_r / (c_p + c_r) if (c_p + c_r) > 0 else 0
                lines.append(
                    f"| {sc} | {n} | {fc}/{n} ({pct(fc, n)}) | {af_acc} "
                    f"| {c_p:.1%} | {c_r:.1%} | {c_f1:.1%} |"
                )
            lines.append("")

    # ── Task diagnostics ─────────────────────────────────────────────────
    diag = views.get("task_diagnostics") or {}
    if diag:
        lines.append("## Task Diagnostics")
        lines.append("")
        lines.append("| Diagnostic | Success | Applicable | Rate |")
        lines.append("|---|---:|---:|---:|")
        for key, label in [
            ("correction_success", "Correction"),
            ("retraction_success", "Retraction"),
            ("history_recovery_success", "History recovery"),
            ("gate_execution_success", "Gate execution"),
            ("repeat_group_execution_success", "Repeat-group execution"),
        ]:
            entry = diag.get(key, {})
            s = entry.get("success", 0)
            a = entry.get("applicable", 0)
            lines.append(f"| {label} | {s} | {a} | {pct(s, a)} |")
        lines.append("")
        lines.append("| Error type | Count |")
        lines.append("|---|---:|")
        for key, label in [
            ("collateral_edit_count", "Collateral edits"),
            ("failed_correction_count", "Failed corrections"),
            ("failed_retraction_count", "Failed retractions"),
            ("repeat_routing_error_count", "Repeat routing errors"),
        ]:
            lines.append(f"| {label} | {diag.get(key, 0)} |")
        lines.append("")

    return "\n".join(lines)


def _append_changed_field_performance(lines: list[str], views: dict[str, Any]) -> None:
    changed = views.get("changed_fields") or {}
    if not changed:
        return

    lines.append("## Changed-Field Performance")
    lines.append("")
    lines.append("Changed fields are those whose gold value differs from the prior state.")
    lines.append("")
    lines.append("| Metric | Strict | Lenient |")
    lines.append("|---|---:|---:|")
    strict = changed.get("strict") or {}
    lenient = changed.get("lenient") or {}
    lines.append(f"| Precision | {rate_str(strict.get('precision'))} | {rate_str(lenient.get('precision'))} |")
    lines.append(f"| Recall | {rate_str(strict.get('recall'))} | {rate_str(lenient.get('recall'))} |")
    lines.append(f"| F1 | {rate_str(strict.get('f1'))} | {rate_str(lenient.get('f1'))} |")
    lines.append(f"| Gold changed fields | {changed.get('gold_changed_total', '')} |  |")
    lines.append(f"| Predicted changed fields | {changed.get('predicted_changed_total', '')} |  |")
    lines.append("")


# ── Simple run-directory report ───────────────────────────────────────────


def generate_report_for_run_dir(
    run_dir: Path,
    *,
    output_dir: Path | None = None,
    items_dir: Path | None = None,
    generate_figures: bool = True,
) -> tuple[Path, Path, int]:
    """Write derived metrics, results.csv, and report.md for a run directory."""

    run_dir = run_dir.resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    out_dir = output_dir.resolve() if output_dir else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    from scripts.compute_run_metrics import compute_run_metrics

    summary = compute_run_metrics(
        run_dir,
        items_dir=items_dir,
        generate_figures=generate_figures,
    )
    simulators_dir = run_dir.parent.parent / "simulators"
    rows = build_csv_rows(run_dir, simulators_dir, items_dir=items_dir)

    csv_path = out_dir / "results.csv"
    write_csv(csv_path, rows, CSV_FIELDNAMES)

    md_path = out_dir / "report.md"
    run_info = _run_info_for_direct_run(run_dir, summary)
    if items_dir is not None:
        run_info["items_dir"] = _portable_data_path(items_dir, shell=True)
    md_path.write_text(build_markdown(summary, run_info), encoding="utf-8")

    return csv_path, md_path, len(rows)


def _run_info_for_direct_run(
    run_dir: Path,
    summary: dict[str, Any],
) -> dict[str, Any]:
    utterances = summary.get("utterances") or []
    questionnaires = sorted({
        str(utterance.get("questionnaire") or "")
        for utterance in utterances
        if utterance.get("questionnaire")
    })
    turn_provenance = _first_turn_result_provenance(run_dir)
    eval_provenance = _first_evaluation_provenance(run_dir)
    run_metadata = _read_optional_json(run_dir / "run_metadata.json")
    split_metadata = _read_optional_json(run_dir / "split_view_metadata.json")
    generation = turn_provenance.get("generation") or {}
    evaluation = eval_provenance.get("evaluation") or {}
    runner = turn_provenance.get("runner") or {}
    model_config = generation.get("model") or {}
    if not isinstance(model_config, dict):
        model_config = {}
    eval_model_config = evaluation.get("model") or {}
    if not isinstance(eval_model_config, dict):
        eval_model_config = {}

    parsed_agent, parsed_model = _labels_from_run_dir_name(run_dir.name)
    agent_label = generation.get("agent") or parsed_agent or "Submission Studio"
    model_label = (
        generation.get("model_label")
        or model_config.get("resolved_model_name")
        or model_config.get("requested_model")
        or parsed_model
        or "unknown"
    )

    return {
        "run_id": run_dir.name,
        "metadata": {},
        "summary_report_path": "summary_report.json",
        "run_dir": _portable_data_path(run_dir, shell=True),
        "agent_label": agent_label,
        "model_label": model_label,
        "model_config": model_config,
        "run_reasoning_effort": (
            run_metadata.get("model_reasoning_effort")
            or model_config.get("reasoning_effort")
            or ""
        ),
        "judge_model_label": (
            run_metadata.get("evaluator_model_id")
            or eval_model_config.get("resolved_model_name")
            or eval_model_config.get("requested_model")
            or ""
        ),
        "judge_reasoning_effort": (
            run_metadata.get("evaluator_reasoning_effort")
            or eval_model_config.get("reasoning_effort")
            or ""
        ),
        "provider": "",
        "run_label": run_dir.name,
        "questionnaire_scope": ", ".join(questionnaires) if questionnaires else "all",
        "scenario_scope": "all",
        "invoked_at": (
            run_metadata.get("started_at")
            or runner.get("invoked_at")
            or ""
        ),
        "git_sha": "",
        "stats": {},
        "split_view_metadata": split_metadata,
    }


def _first_turn_result_provenance(run_dir: Path) -> dict[str, Any]:
    for turn_path in sorted(run_dir.rglob("turn_result.json")):
        try:
            turn_result = read_json(turn_path)
        except (OSError, json.JSONDecodeError):
            continue
        provenance = turn_result.get("provenance")
        if isinstance(provenance, dict):
            return provenance
    return {}


def _first_evaluation_provenance(run_dir: Path) -> dict[str, Any]:
    for eval_path in sorted(run_dir.rglob("evaluation.json")):
        try:
            evaluation = read_json(eval_path)
        except (OSError, json.JSONDecodeError):
            continue
        provenance = evaluation.get("provenance")
        if isinstance(provenance, dict):
            return provenance
    return {}


def _read_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = read_json(path)
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _labels_from_run_dir_name(run_name: str) -> tuple[str, str]:
    parts = run_name.rsplit("_", 2)
    if len(parts) == 3 and re.fullmatch(r"\d{8}T\d{6}", parts[2]):
        return parts[0], parts[1]
    return "", ""


def build_simple_markdown(run_dir: Path, rows: list[dict[str, Any]]) -> str:
    counts = Counter(row.get("correctness") or "unknown" for row in rows)
    total = len(rows)
    correct = counts.get("correct", 0)
    partial = counts.get("partially_correct", 0)
    incorrect = counts.get("incorrect", 0)
    items = sorted({row.get("item_id", "") for row in rows if row.get("item_id")})

    lines = [
        f"# Full Run Report: {run_dir.name}",
        "",
        "## Headline Stats",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Items | {len(items)} |",
        f"| Total fields evaluated | {total} |",
        f"| Correct | {correct} |",
        f"| Partially correct | {partial} |",
        f"| Incorrect | {incorrect} |",
        f"| Accuracy (strict) | {pct(correct, total)} |",
        f"| Accuracy (lenient) | {pct(correct + partial, total)} |",
        "",
    ]

    by_questionnaire: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_questionnaire[row.get("questionnaire") or "unknown"].append(row)

    if by_questionnaire:
        lines.extend([
            "## Per-Questionnaire Breakdown",
            "",
            "| Questionnaire | Items | Fields | Correct | Partial | Incorrect | Accuracy | Lenient |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for questionnaire, q_rows in sorted(by_questionnaire.items()):
            q_counts = Counter(row.get("correctness") or "unknown" for row in q_rows)
            q_total = len(q_rows)
            q_correct = q_counts.get("correct", 0)
            q_partial = q_counts.get("partially_correct", 0)
            q_items = {row.get("item_id", "") for row in q_rows if row.get("item_id")}
            lines.append(
                f"| {questionnaire} | {len(q_items)} | {q_total} | {q_correct} | "
                f"{q_partial} | {q_counts.get('incorrect', 0)} | "
                f"{pct(q_correct, q_total)} | {pct(q_correct + q_partial, q_total)} |"
            )
        lines.append("")

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, help="Run directory containing evaluation.json files.")
    parser.add_argument("--run-id", help="Legacy run ID matching data/eval_results/{id}.json")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory. Defaults to --run-dir for direct runs.",
    )
    parser.add_argument(
        "--items-dir",
        type=Path,
        help="Ground-truth item directory to use when recomputing metrics for --run-dir.",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip regenerating figures while recomputing run metrics.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if bool(args.run_dir) == bool(args.run_id):
        print("Error: pass exactly one of --run-dir or --run-id", file=sys.stderr)
        return 2

    if args.run_dir:
        csv_path, md_path, row_count = generate_report_for_run_dir(
            args.run_dir,
            output_dir=args.output_dir,
            items_dir=args.items_dir,
            generate_figures=not args.no_figures,
        )
        print(f"Wrote {csv_path} ({row_count} rows)")
        print(f"Wrote {md_path}")
        return 0

    run_info = resolve_run(args.run_id)

    summary_path = run_info["summary_report_path"]
    if not summary_path.exists():
        print(f"Error: summary_report.json not found at {summary_path}", file=sys.stderr)
        return 1

    summary = read_json(summary_path)
    runs_dir = summary_path.parent
    simulators_dir = runs_dir.parent.parent / "simulators"

    dir_name = output_dir_name(run_info)
    out_dir = (args.output_dir or DEFAULT_OUTPUT_DIR) / dir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build and write CSV
    print(f"Building CSV from {runs_dir} ...")
    csv_rows = build_csv_rows(runs_dir, simulators_dir)
    csv_path = out_dir / "results.csv"
    write_csv(csv_path, csv_rows, CSV_FIELDNAMES)
    print(f"  Wrote {csv_path} ({len(csv_rows)} rows)")

    # Build and write Markdown
    markdown = build_markdown(summary, run_info)
    md_path = out_dir / "report.md"
    md_path.write_text(markdown, encoding="utf-8")
    print(f"  Wrote {md_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
