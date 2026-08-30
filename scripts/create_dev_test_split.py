#!/usr/bin/env python3
"""Create a deterministic 20/80 dev/test split for the frozen benchmark."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_SEED = 20260524
ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(
    os.environ.get("CONFORMBENCH_DATA_DIR", ROOT / "data")
).expanduser().resolve()
FAMILY_TARGET_PER_DOMAIN = {
    "add": 3,
    "refine": 2,
    "correct": 2,
    "retract": 2,
}
STRESSOR_KEYS = [
    "target_binding_pressure",
    "commitment_boundary_pressure",
    "distractor_competition",
    "unsupported_alternative_affordance",
    "repeat_instance_routing_pressure",
    "state_history_mismatch_pressure",
]


@dataclass(frozen=True)
class Item:
    item_id: str
    domain: str
    family: str
    state: str
    history_required: bool
    repeat_involvement: str
    load_bucket: str
    conflict_present: bool
    support_distance: int
    revision_operation: str
    changed_leaf_count: int
    repeat_changed_leaf_count: int
    difficulty_tier: str
    stressors: dict[str, str]
    targeted_failure_modes: tuple[str, ...]
    source_path: Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--items-dir",
        type=Path,
        default=DATA_ROOT / "items" / "benchmark",
        help="Directory containing frozen benchmark ground_truth.json packets.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_ROOT / "items" / "splits" / "conformbench_v1_20_80_seed20260524",
        help="Directory to write split manifests and runnable item trees.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--iterations", type=int, default=250_000)
    args = parser.parse_args()

    items_dir = args.items_dir.resolve()
    output_dir = args.output_dir.resolve()
    items = load_items(items_dir)
    if len(items) != 180:
        raise SystemExit(f"Expected 180 benchmark items, found {len(items)} in {items_dir}")

    dev_items, score = choose_dev_split(items, seed=args.seed, iterations=args.iterations)
    dev_ids = {item.item_id for item in dev_items}
    test_items = [item for item in sorted(items, key=lambda x: (x.domain, x.item_id)) if item.item_id not in dev_ids]

    if len(dev_items) != 36 or len(test_items) != 144:
        raise SystemExit(f"Bad split sizes: dev={len(dev_items)} test={len(test_items)}")

    write_split(
        output_dir=output_dir,
        items_dir=items_dir,
        dev_items=sorted(dev_items, key=lambda x: (x.domain, x.item_id)),
        test_items=test_items,
        seed=args.seed,
        iterations=args.iterations,
        score=score,
    )
    print(f"Wrote split to {output_dir}")
    print(f"Dev items: {len(dev_items)}; test items: {len(test_items)}")
    print(f"Dev family counts: {dict(summary(dev_items)['family'])}")
    print(f"Dev state counts: {dict(summary(dev_items)['state'])}")
    print(f"Dev history-required count: {summary(dev_items)['history_required'][True]}")


def load_items(root: Path) -> list[Item]:
    loaded: list[Item] = []
    for path in sorted(root.rglob("ground_truth.json")):
        packet = json.loads(path.read_text(encoding="utf-8"))
        derived = packet.get("derived_variables") or {}
        profile = packet.get("difficulty_profile") or {}
        stressors = profile.get("stressors") or profile.get("dimensions") or {}
        item_id = packet.get("scenario_id") or packet.get("item_id")
        domain = packet.get("questionnaire") or packet.get("source_questionnaire") or packet.get("questionnaire_id")
        family = derived.get("primary_delta_type") or packet.get("primary_delta_type")
        state = derived.get("prior_state_condition") or packet.get("state_condition")
        targeted = packet.get("targeted_failure_mode") or profile.get("targeted_failure_modes") or []
        if isinstance(targeted, str):
            targeted = [targeted]
        loaded.append(
            Item(
                item_id=str(item_id),
                domain=str(domain),
                family=str(family),
                state=str(state),
                history_required=bool(derived.get("history_required")),
                repeat_involvement=str(
                    derived.get("repeat_group_involvement")
                    or ("single_group" if derived.get("repeat_group_involved") else "none")
                ),
                load_bucket=str(derived.get("form_relative_load_bucket")),
                conflict_present=bool(derived.get("conflict_present")),
                support_distance=int(derived.get("support_distance") or 0),
                revision_operation=str(derived.get("revision_operation")),
                changed_leaf_count=int(derived.get("changed_leaf_field_count") or 0),
                repeat_changed_leaf_count=int(derived.get("repeat_group_changed_leaf_count") or 0),
                difficulty_tier=str(packet.get("difficulty_tier")),
                stressors={key: str(stressors.get(key) or "none") for key in STRESSOR_KEYS},
                targeted_failure_modes=tuple(str(mode) for mode in targeted),
                source_path=path,
            )
        )
    return loaded


def choose_dev_split(items: list[Item], *, seed: int, iterations: int) -> tuple[list[Item], float]:
    rng = random.Random(seed)
    by_domain_family: dict[tuple[str, str], list[Item]] = defaultdict(list)
    domains = sorted({item.domain for item in items})
    for item in items:
        by_domain_family[(item.domain, item.family)].append(item)

    for domain in domains:
        for family, target in FAMILY_TARGET_PER_DOMAIN.items():
            bucket = by_domain_family[(domain, family)]
            if len(bucket) < target:
                raise SystemExit(f"Not enough {domain}/{family} items for target {target}")

    full_summary = summary(items)
    target = split_targets(full_summary, dev_size=36)

    best: tuple[float, list[Item]] | None = None
    for _ in range(iterations):
        candidate: list[Item] = []
        for domain in domains:
            for family, target_count in FAMILY_TARGET_PER_DOMAIN.items():
                candidate.extend(rng.sample(by_domain_family[(domain, family)], target_count))
        score = split_score(candidate, target, domains)
        if best is None or score < best[0]:
            best = (score, candidate)

    assert best is not None
    return best[1], best[0]


def split_targets(full: dict[str, Counter], *, dev_size: int) -> dict[str, dict[Any, int]]:
    return {
        "state": proportional_target(full["state"], dev_size),
        "history_required": proportional_target(full["history_required"], dev_size),
        "repeat_involvement": proportional_target(full["repeat_involvement"], dev_size),
        "load_bucket": proportional_target(full["load_bucket"], dev_size),
        "conflict_present": proportional_target(full["conflict_present"], dev_size),
        "support_distance": proportional_target(full["support_distance"], dev_size),
        "revision_operation": proportional_target(full["revision_operation"], dev_size),
        "difficulty_tier": proportional_target(full["difficulty_tier"], dev_size),
        **{
            f"stressor:{key}": proportional_target(full[f"stressor:{key}"], dev_size)
            for key in STRESSOR_KEYS
        },
    }


def proportional_target(counter: Counter, size: int) -> dict[Any, int]:
    total = sum(counter.values())
    raw = {key: value * size / total for key, value in counter.items()}
    target = {key: int(math.floor(value)) for key, value in raw.items()}
    remainder = size - sum(target.values())
    ranked = sorted(raw, key=lambda key: (raw[key] - target[key], str(key)), reverse=True)
    for key in ranked[:remainder]:
        target[key] += 1
    return target


def split_score(items: list[Item], target: dict[str, dict[Any, int]], domains: list[str]) -> float:
    counts = summary(items)
    score = 0.0
    weights = {
        "state": 5.0,
        "history_required": 3.0,
        "repeat_involvement": 3.0,
        "load_bucket": 2.0,
        "conflict_present": 2.0,
        "support_distance": 2.0,
        "revision_operation": 1.5,
        "difficulty_tier": 1.0,
    }
    for key, weight in weights.items():
        score += weighted_counter_distance(counts[key], target[key], weight)
    for stressor_key in STRESSOR_KEYS:
        score += weighted_counter_distance(
            counts[f"stressor:{stressor_key}"],
            target[f"stressor:{stressor_key}"],
            1.5,
        )

    by_domain = defaultdict(list)
    for item in items:
        by_domain[item.domain].append(item)
    for domain in domains:
        domain_items = by_domain[domain]
        domain_counts = summary(domain_items)
        if domain_counts["state"]["S1"] < 1:
            score += 25.0
        if domain_counts["state"]["S4"] < 1:
            score += 25.0
        if domain_counts["history_required"][True] < 3:
            score += 10.0
        if sum(value for key, value in domain_counts["repeat_involvement"].items() if key != "none") < 3:
            score += 10.0
        if domain_counts["load_bucket"]["scale"] < 1:
            score += 3.0

    changed_target = 1681 * 0.2
    repeat_changed_target = 1032 * 0.2
    score += ((sum(item.changed_leaf_count for item in items) - changed_target) / 70.0) ** 2
    score += ((sum(item.repeat_changed_leaf_count for item in items) - repeat_changed_target) / 55.0) ** 2
    return score


def weighted_counter_distance(actual: Counter, target: dict[Any, int], weight: float) -> float:
    score = 0.0
    for key in set(actual) | set(target):
        expected = target.get(key, 0)
        observed = actual.get(key, 0)
        score += weight * ((observed - expected) ** 2) / max(expected, 1)
    return score


def summary(items: list[Item]) -> dict[str, Counter]:
    out = {
        "domain": Counter(item.domain for item in items),
        "family": Counter(item.family for item in items),
        "state": Counter(item.state for item in items),
        "history_required": Counter(item.history_required for item in items),
        "repeat_involvement": Counter(item.repeat_involvement for item in items),
        "load_bucket": Counter(item.load_bucket for item in items),
        "conflict_present": Counter(item.conflict_present for item in items),
        "support_distance": Counter(item.support_distance for item in items),
        "revision_operation": Counter(item.revision_operation for item in items),
        "difficulty_tier": Counter(item.difficulty_tier for item in items),
    }
    for key in STRESSOR_KEYS:
        out[f"stressor:{key}"] = Counter(item.stressors.get(key, "none") for item in items)
    return out


def write_split(
    *,
    output_dir: Path,
    items_dir: Path,
    dev_items: list[Item],
    test_items: list[Item],
    seed: int,
    iterations: int,
    score: float,
) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    (output_dir / "dev").mkdir(parents=True)
    (output_dir / "test").mkdir(parents=True)

    for split_name, split_items in (("dev", dev_items), ("test", test_items)):
        write_item_tree(output_dir / split_name, items_dir, split_items)
        (output_dir / f"{split_name}_item_ids.txt").write_text(
            "\n".join(item.item_id for item in split_items) + "\n",
            encoding="utf-8",
        )

    manifest = {
        "split_id": output_dir.name,
        "created_by": "scripts/create_dev_test_split.py",
        "seed": seed,
        "search_iterations": iterations,
        "selection_score": round(score, 6),
        "policy": {
            "purpose": "Frozen 20/80 development/test split for ConFormBench architecture and prompt tuning.",
            "dev_size": len(dev_items),
            "test_size": len(test_items),
            "hard_constraints": {
                "domains": "9 dev items per form domain",
                "primary_update_family_per_domain": FAMILY_TARGET_PER_DOMAIN,
            },
            "soft_constraints": [
                "Approximate 20% coverage of prior-state condition, history requirement, repeat involvement, load bucket, conflict, support distance, revision operation, difficulty tier, and human-coded stressor levels.",
                "At least one S1 and one S4 item per form where possible.",
                "At least three history-required and three repeat-involved items per form where possible.",
            ],
            "leakage_rule": "The split is based only on frozen item metadata and does not use model outputs, scores, or evaluator results.",
        },
        "dev_item_ids": [item.item_id for item in dev_items],
        "test_item_ids": [item.item_id for item in test_items],
        "summaries": {
            "dev": jsonable_summary(dev_items),
            "test": jsonable_summary(test_items),
            "full": jsonable_summary(dev_items + test_items),
        },
        "items": {
            "dev": [item_manifest_entry(item, items_dir) for item in dev_items],
            "test": [item_manifest_entry(item, items_dir) for item in test_items],
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(readme_text(output_dir.name), encoding="utf-8")


def write_item_tree(split_dir: Path, items_dir: Path, items: list[Item]) -> None:
    for item in items:
        relative_source = item.source_path.relative_to(items_dir)
        destination = split_dir / relative_source
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item.source_path, destination)


def item_manifest_entry(item: Item, items_dir: Path) -> dict[str, Any]:
    return {
        "item_id": item.item_id,
        "domain": item.domain,
        "primary_update_family": item.family,
        "prior_state_condition": item.state,
        "history_required": item.history_required,
        "repeat_group_involvement": item.repeat_involvement,
        "load_bucket": item.load_bucket,
        "conflict_present": item.conflict_present,
        "support_distance": item.support_distance,
        "revision_operation": item.revision_operation,
        "changed_leaf_count": item.changed_leaf_count,
        "repeat_changed_leaf_count": item.repeat_changed_leaf_count,
        "difficulty_tier": item.difficulty_tier,
        "stressors": item.stressors,
        "targeted_failure_modes": list(item.targeted_failure_modes),
        "source_path": str(item.source_path.relative_to(items_dir)),
    }


def jsonable_summary(items: list[Item]) -> dict[str, Any]:
    counts = summary(items)
    return {
        "item_count": len(items),
        "changed_leaf_total": sum(item.changed_leaf_count for item in items),
        "repeat_changed_leaf_total": sum(item.repeat_changed_leaf_count for item in items),
        **{key: dict(counter) for key, counter in counts.items()},
    }


def readme_text(split_id: str) -> str:
    return f"""# {split_id}

Frozen 20/80 ConFormBench split.

- `dev/` contains 36 copied `ground_truth.json` packets for development and prompt tuning.
- `test/` contains copied `ground_truth.json` packets for the 144 held-out final-reporting items.
- `manifest.json` records the item IDs, source paths, selection policy, and coverage summaries.
- `dev_item_ids.txt` and `test_item_ids.txt` are compact ID lists.

The split was selected from item metadata only. Do not use held-out test results for
prompt or architecture changes.
"""


if __name__ == "__main__":
    main()
