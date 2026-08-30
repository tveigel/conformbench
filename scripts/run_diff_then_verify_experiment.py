#!/usr/bin/env python3
"""Guarded ARR experiment workflow for the Diff-then-Verify baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any


SOFTWARE_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(
    os.environ.get("CONFORMBENCH_DATA_DIR", SOFTWARE_ROOT / "data")
).expanduser().resolve()
SPLIT_ID = "conformbench_v1_20_80_seed20260524"
DEV_ITEMS = DATA_ROOT / "items" / "splits" / SPLIT_ID / "dev"
TEST_ITEMS = DATA_ROOT / "items" / "splits" / SPLIT_ID / "test"
EXPERIMENT_ROOT = DATA_ROOT / "reports" / "rebuttal_runs" / "diff_then_verify"
DEV_RUNS = EXPERIMENT_ROOT / "dev"
TEST_RUNS = EXPERIMENT_ROOT / "test"
PROTOCOL_DIR = EXPERIMENT_ROOT / "protocol"
FROZEN_CONFIG = PROTOCOL_DIR / "frozen_config.json"
TEST_RECEIPT = PROTOCOL_DIR / "test_run_receipt.json"
TEST_RUN_PREFIX = "arr_diff_then_verify_test"
TEST_RUN_NAME = (
    TEST_RUN_PREFIX
    + "__diff_then_verify__gen-gpt-5-4-mini__judge-gpt-5-4-medium"
)
BASELINE_RUN = (
    DATA_ROOT
    / "reports"
    / "split_runs"
    / SPLIT_ID
    / "test"
    / "audit_tight_test_20260525__flatagent__gen-gpt-5-4-mini__judge-gpt-5-4-medium"
)

FROZEN_FILES = (
    SOFTWARE_ROOT / "pyproject.toml",
    SOFTWARE_ROOT / "uv.lock",
    SOFTWARE_ROOT / "llm.py",
    SOFTWARE_ROOT / "conformbench" / "benchmark.py",
    SOFTWARE_ROOT / "conformbench" / "llm_accounting.py",
    SOFTWARE_ROOT / "conformbench" / "systems" / "diff_then_verify.py",
    SOFTWARE_ROOT / "conformbench" / "systems" / "flatagent.py",
    SOFTWARE_ROOT / "conformbench" / "systems" / "prompt_context.py",
    SOFTWARE_ROOT / "scripts" / "compute_run_metrics.py",
    SOFTWARE_ROOT / "scripts" / "generate_full_run_report.py",
    SOFTWARE_ROOT / "scripts" / "run_architecture_comparison.py",
    SOFTWARE_ROOT / "scripts" / "run_diff_then_verify_experiment.py",
    SOFTWARE_ROOT / "scripts" / "summarize_diff_then_verify.py",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frozen_file_hashes() -> dict[str, str]:
    paths = [*FROZEN_FILES, *sorted((SOFTWARE_ROOT / "conformbench" / "evaluator").glob("*.py"))]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing frozen source files: " + ", ".join(missing))
    return {
        str(path.relative_to(SOFTWARE_ROOT)): sha256(path)
        for path in paths
    }


def frozen_data_hashes() -> dict[str, str]:
    paths = [
        *sorted(DEV_ITEMS.rglob("ground_truth.json")),
        *sorted(TEST_ITEMS.rglob("ground_truth.json")),
        *sorted((DATA_ROOT / "schema" / "questionnaires").glob("*.json")),
        BASELINE_RUN / "summary_report.json",
        BASELINE_RUN / "run_metadata.json",
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing frozen data files: " + ", ".join(missing))
    return {
        str(path.relative_to(DATA_ROOT)): sha256(path)
        for path in paths
    }


def count_items(path: Path) -> int:
    return sum(1 for _ in path.rglob("ground_truth.json"))


def validate_layout() -> None:
    expected = {
        "dev items": (DEV_ITEMS, 36),
        "test items": (TEST_ITEMS, 144),
    }
    for label, (path, count) in expected.items():
        actual = count_items(path)
        if actual != count:
            raise RuntimeError(f"Expected {count} {label} under {path}, found {actual}")
    if not (BASELINE_RUN / "summary_report.json").is_file():
        raise FileNotFoundError(f"Missing Update Tool baseline: {BASELINE_RUN}")
    baseline = read_json(BASELINE_RUN / "summary_report.json")
    baseline_count = int(
        (((baseline.get("aggregate") or {}).get("metric_views") or {})
         .get("whole_record_exact_match", {})
         .get("item_count", 0))
        or 0
    )
    if baseline_count != 144:
        raise RuntimeError(f"Expected 144 baseline items, found {baseline_count}")


def experiment_env() -> dict[str, str]:
    env = os.environ.copy()
    env["CONFORMBENCH_DATA_DIR"] = str(DATA_ROOT)
    env["LLM_BACKEND"] = "openai"
    env["CONFORMBENCH_TRACE_MESSAGES"] = "1"
    env["CONFORMBENCH_TRACK_LLM_COSTS"] = "true"
    env.pop("OPENAI_BASE_URL", None)
    env.pop("OPENAI_API_BASE", None)
    return env


def architecture_command(
    *,
    items_dir: Path,
    runs_dir: Path,
    run_prefix: str,
    run_purpose: str,
    batch_size: int,
    total_cost_cap: float,
    batch_cost_cap: float,
    plan_only: bool = False,
    resume: bool = False,
) -> list[str]:
    command = [
        sys.executable,
        str(SOFTWARE_ROOT / "scripts" / "run_architecture_comparison.py"),
        "--items-dir",
        str(items_dir),
        "--runs-dir",
        str(runs_dir),
        "--run-prefix",
        run_prefix,
        "--run-purpose",
        run_purpose,
        "--architectures",
        "diff_then_verify",
        "--generator-model",
        "gpt-5.4-mini",
        "--generator-reasoning",
        "none",
        "--evaluator-model",
        "gpt-5.4",
        "--evaluator-reasoning",
        "medium",
        "--batch-size",
        str(batch_size),
        "--workers",
        "1",
        "--changed-f1-floor",
        "0",
        "--max-total-cost-usd",
        str(total_cost_cap),
        "--max-architecture-cost-usd",
        str(total_cost_cap),
        "--max-batch-cost-usd",
        str(batch_cost_cap),
    ]
    if plan_only:
        command.append("--plan-only")
    if resume:
        command.append("--resume")
    return command


def run_checked(command: list[str]) -> None:
    subprocess.run(
        command,
        cwd=SOFTWARE_ROOT,
        env=experiment_env(),
        check=True,
    )


def preflight() -> None:
    validate_layout()
    frozen_file_hashes()
    frozen_data_hashes()
    with tempfile.TemporaryDirectory(prefix="conformbench-dtv-preflight-") as temp:
        root = Path(temp)
        run_checked(
            architecture_command(
                items_dir=DEV_ITEMS,
                runs_dir=root / "dev",
                run_prefix="preflight_dev",
                run_purpose="dev_tuning",
                batch_size=12,
                total_cost_cap=5.0,
                batch_cost_cap=2.0,
                plan_only=True,
            )
        )
        run_checked(
            architecture_command(
                items_dir=TEST_ITEMS,
                runs_dir=root / "test",
                run_prefix="preflight_test",
                run_purpose="selected_system",
                batch_size=20,
                total_cost_cap=12.0,
                batch_cost_cap=3.0,
                plan_only=True,
            )
        )
    print("Preflight passed: solver import, model routing, 36/144 splits, and baseline artifacts are valid.")


def run_dev(run_prefix: str | None) -> None:
    validate_layout()
    if FROZEN_CONFIG.exists():
        raise RuntimeError(
            f"Configuration is already frozen at {FROZEN_CONFIG}; do not tune after freezing."
        )
    prefix = run_prefix or datetime.now(timezone.utc).strftime(
        "arr_diff_then_verify_dev_%Y%m%dT%H%M%SZ"
    )
    run_checked(
        architecture_command(
            items_dir=DEV_ITEMS,
            runs_dir=DEV_RUNS,
            run_prefix=prefix,
            run_purpose="dev_tuning",
            batch_size=12,
            total_cost_cap=5.0,
            batch_cost_cap=2.0,
        )
    )
    print(f"Development run completed under {DEV_RUNS}")


def resolve_dev_run(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        candidates = [Path.cwd() / path, SOFTWARE_ROOT / path, DEV_RUNS / path]
        path = next((candidate for candidate in candidates if candidate.exists()), path)
    path = path.resolve()
    if not (path / "summary_report.json").is_file():
        raise FileNotFoundError(f"Missing development summary_report.json under {path}")
    return path


def validate_dev_run(path: Path) -> None:
    metadata = read_json(path / "run_metadata.json")
    summary = read_json(path / "summary_report.json")
    item_count = int(
        (((summary.get("aggregate") or {}).get("metric_views") or {})
         .get("whole_record_exact_match", {})
         .get("item_count", 0))
        or 0
    )
    expected = {
        "architecture": (metadata.get("architecture"), "diff_then_verify"),
        "model_id": (metadata.get("model_id"), "gpt-5.4-mini"),
        "evaluator_model_id": (metadata.get("evaluator_model_id"), "gpt-5.4"),
        "evaluator_reasoning_effort": (
            metadata.get("evaluator_reasoning_effort"),
            "medium",
        ),
        "status": (metadata.get("status"), "completed"),
        "item_count": (item_count, 36),
    }
    mismatches = {
        key: {"actual": actual, "expected": wanted}
        for key, (actual, wanted) in expected.items()
        if actual != wanted
    }
    if mismatches:
        raise RuntimeError("Invalid development run: " + json.dumps(mismatches, indent=2))


def freeze(dev_run_value: str, *, replace: bool) -> None:
    validate_layout()
    if TEST_RECEIPT.exists() or (TEST_RUNS / TEST_RUN_NAME).exists():
        raise RuntimeError("The held-out test run has started; the frozen configuration cannot change.")
    if FROZEN_CONFIG.exists() and not replace:
        raise RuntimeError(f"Frozen config already exists: {FROZEN_CONFIG}. Use --replace before test only.")
    dev_run = resolve_dev_run(dev_run_value)
    validate_dev_run(dev_run)
    payload = {
        "protocol_version": 1,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "split_id": SPLIT_ID,
        "dev_item_count": 36,
        "test_item_count": 144,
        "dev_run_dir": str(dev_run),
        "baseline_run_dir": str(BASELINE_RUN),
        "test_run_prefix": TEST_RUN_PREFIX,
        "test_run_dir": str(TEST_RUNS / TEST_RUN_NAME),
        "generator_model": "gpt-5.4-mini",
        "generator_reasoning": "none",
        "evaluator_model": "gpt-5.4",
        "evaluator_reasoning_requested": "medium",
        "evaluator_reasoning_effective": "provider_default",
        "evaluator_reasoning_compatibility_note": (
            "Released run metadata labels the evaluator medium, while released "
            "provider traces contain zero reasoning tokens and the checked-in "
            "official-OpenAI helper removes reasoning_effort before transmission."
        ),
        "workers": 1,
        "cost_caps_usd": {
            "development_total": 5.0,
            "held_out_total": 12.0,
            "development_batch": 2.0,
            "held_out_batch": 3.0,
        },
        "generation_trace_messages": True,
        "source_sha256": frozen_file_hashes(),
        "data_sha256": frozen_data_hashes(),
        "policy": (
            "Verifier prompt and implementation were selected on the 36-item dev split. "
            "The 144-item held-out test run may be started once and resumed only if interrupted."
        ),
    }
    write_json(FROZEN_CONFIG, payload)
    print(f"Frozen configuration written to {FROZEN_CONFIG}")


def validate_frozen_config() -> dict[str, Any]:
    if not FROZEN_CONFIG.is_file():
        raise FileNotFoundError(
            f"Missing {FROZEN_CONFIG}. Complete a dev run and freeze it before test."
        )
    frozen = read_json(FROZEN_CONFIG)
    current = frozen_file_hashes()
    expected = frozen.get("source_sha256") or {}
    if current != expected:
        changed = sorted(
            key for key in set(current) | set(expected) if current.get(key) != expected.get(key)
        )
        raise RuntimeError(
            "Frozen source changed after dev selection; refusing held-out test run. "
            "Changed files: " + ", ".join(changed)
        )
    current_data = frozen_data_hashes()
    expected_data = frozen.get("data_sha256") or {}
    if current_data != expected_data:
        changed = sorted(
            key
            for key in set(current_data) | set(expected_data)
            if current_data.get(key) != expected_data.get(key)
        )
        raise RuntimeError(
            "Frozen benchmark data changed after dev selection; refusing held-out "
            "test run. Changed files: " + ", ".join(changed)
        )
    dev_run = Path(str(frozen.get("dev_run_dir") or ""))
    validate_dev_run(dev_run)
    return frozen


def validate_completed_test_run(run_dir: Path) -> None:
    metadata = read_json(run_dir / "run_metadata.json")
    summary = read_json(run_dir / "summary_report.json")
    item_count = int(
        (((summary.get("aggregate") or {}).get("metric_views") or {})
         .get("whole_record_exact_match", {})
         .get("item_count", 0))
        or 0
    )
    if metadata.get("status") != "completed" or item_count != 144:
        raise RuntimeError(
            f"Held-out run did not complete cleanly: status={metadata.get('status')}, "
            f"items={item_count}"
        )


def run_test() -> None:
    validate_layout()
    frozen = validate_frozen_config()
    run_dir = TEST_RUNS / TEST_RUN_NAME
    if TEST_RECEIPT.exists():
        raise RuntimeError(f"Held-out test already completed; receipt: {TEST_RECEIPT}")

    resume = False
    if run_dir.exists():
        metadata_path = run_dir / "run_metadata.json"
        if metadata_path.is_file() and read_json(metadata_path).get("status") == "completed":
            raise RuntimeError(
                f"Held-out test run already completed at {run_dir}; refusing a second run."
            )
        resume = True
        print(f"Resuming interrupted held-out run at {run_dir}")

    run_checked(
        architecture_command(
            items_dir=TEST_ITEMS,
            runs_dir=TEST_RUNS,
            run_prefix=TEST_RUN_PREFIX,
            run_purpose="selected_system",
            batch_size=20,
            total_cost_cap=12.0,
            batch_cost_cap=3.0,
            resume=resume,
        )
    )
    validate_completed_test_run(run_dir)
    receipt = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "frozen_config": str(FROZEN_CONFIG),
        "frozen_config_sha256": sha256(FROZEN_CONFIG),
        "source_sha256": frozen["source_sha256"],
        "data_sha256": frozen["data_sha256"],
        "test_item_count": 144,
        "policy": "First completed run of the frozen held-out configuration.",
    }
    write_json(TEST_RECEIPT, receipt)
    summarize(run_dir)


def summarize(run_dir: Path | None = None) -> None:
    target = run_dir or (TEST_RUNS / TEST_RUN_NAME)
    validate_completed_test_run(target)
    run_checked(
        [
            sys.executable,
            str(SOFTWARE_ROOT / "scripts" / "summarize_diff_then_verify.py"),
            "--verify-run",
            str(target),
            "--baseline-run",
            str(BASELINE_RUN),
            "--output-dir",
            str(PROTOCOL_DIR),
            "--iterations",
            "10000",
            "--seed",
            "20260710",
        ]
    )


def print_status() -> None:
    validate_layout()
    status = {
        "software_root": str(SOFTWARE_ROOT),
        "data_root": str(DATA_ROOT),
        "dev_items": count_items(DEV_ITEMS),
        "test_items": count_items(TEST_ITEMS),
        "baseline_run": str(BASELINE_RUN),
        "frozen_config": str(FROZEN_CONFIG) if FROZEN_CONFIG.exists() else None,
        "test_run": str(TEST_RUNS / TEST_RUN_NAME),
        "test_receipt": str(TEST_RECEIPT) if TEST_RECEIPT.exists() else None,
    }
    print(json.dumps(status, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("preflight", help="Offline validation; makes no LLM calls.")
    dev_parser = subparsers.add_parser("dev", help="Run the 36-item development split.")
    dev_parser.add_argument("--run-prefix")
    freeze_parser = subparsers.add_parser("freeze", help="Freeze a completed dev-selected setup.")
    freeze_parser.add_argument("--dev-run", required=True)
    freeze_parser.add_argument("--replace", action="store_true")
    subparsers.add_parser("test", help="Run or resume the one-shot 144-item held-out test.")
    subparsers.add_parser("summarize", help="Regenerate author-comment statistics.")
    subparsers.add_parser("status", help="Print experiment paths and state.")
    args = parser.parse_args()

    if args.command == "preflight":
        preflight()
    elif args.command == "dev":
        run_dev(args.run_prefix)
    elif args.command == "freeze":
        freeze(args.dev_run, replace=args.replace)
    elif args.command == "test":
        run_test()
    elif args.command == "summarize":
        summarize()
    elif args.command == "status":
        print_status()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
