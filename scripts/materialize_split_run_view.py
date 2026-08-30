#!/usr/bin/env python3
"""Create a dev/test-filtered copy or view of an existing scored run.

The output contains only the per-item scored artifacts whose ``scenario_id``
appears in the requested split. It is intended for offline reporting: run
``scripts/compute_run_metrics.py`` on the output directory to recompute
aggregate metrics without another model pass.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(
    os.environ.get("CONFORMBENCH_DATA_DIR", ROOT / "data")
).expanduser().resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--output-run-dir", type=Path, required=True)
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=DATA_ROOT / "items" / "splits" / "conformbench_v1_20_80_seed20260524",
    )
    parser.add_argument("--split", choices=["dev", "train", "test"], required=True)
    parser.add_argument(
        "--mode",
        choices=["copy", "symlink"],
        default="copy",
        help="Use copy for independent artifacts, or symlink for lightweight views.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_run_dir = args.source_run_dir.resolve()
    output_run_dir = args.output_run_dir.resolve()
    split_dir = args.split_dir.resolve()
    split = "dev" if args.split == "train" else args.split

    if not source_run_dir.exists():
        raise SystemExit(f"Source run directory not found: {source_run_dir}")
    if not split_dir.exists():
        raise SystemExit(f"Split directory not found: {split_dir}")
    if output_run_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"Output already exists: {output_run_dir}. Use --overwrite.")
        shutil.rmtree(output_run_dir)

    item_ids = load_item_ids(split_dir / f"{split}_item_ids.txt")
    output_run_dir.mkdir(parents=True)

    materialized = 0
    seen: set[str] = set()
    for turn_path in sorted(source_run_dir.rglob("turn_result.json")):
        payload = read_json(turn_path)
        scenario_id = str(payload.get("scenario_id") or "")
        if scenario_id not in item_ids:
            continue

        seen.add(scenario_id)
        source_leaf = turn_path.parent
        output_leaf = output_run_dir / source_leaf.relative_to(source_run_dir)
        output_leaf.mkdir(parents=True, exist_ok=True)
        for artifact in sorted(path for path in source_leaf.iterdir() if path.is_file()):
            destination = output_leaf / artifact.name
            if args.mode == "symlink":
                destination.symlink_to(artifact)
            else:
                shutil.copy2(artifact, destination)
        materialized += 1

    for root_artifact in ("run_metadata.json",):
        source = source_run_dir / root_artifact
        if source.exists():
            destination = output_run_dir / root_artifact
            if args.mode == "symlink":
                destination.symlink_to(source)
            else:
                shutil.copy2(source, destination)

    missing = sorted(item_ids - seen)
    metadata = {
        "source_run_dir": str(source_run_dir),
        "split_dir": str(split_dir),
        "split_id": split_dir.name,
        "split": split,
        "mode": args.mode,
        "requested_item_count": len(item_ids),
        "materialized_item_count": materialized,
        "linked_item_count": materialized,
        "missing_item_ids": missing,
    }
    (output_run_dir / "split_view_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"Created {split} {args.mode}: {output_run_dir}")
    print(f"Materialized {materialized}/{len(item_ids)} scored items.")
    if missing:
        print("Missing item IDs: " + ", ".join(missing))
    return 0 if not missing else 2


def load_item_ids(path: Path) -> set[str]:
    if not path.exists():
        raise SystemExit(f"Item ID file not found: {path}")
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
