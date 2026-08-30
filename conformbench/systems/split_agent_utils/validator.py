"""Validate a split against the schema: completeness, disjointness, dependencies."""

from __future__ import annotations

from typing import Any

from .schema_analysis import (
    extract_dependency_constraints,
    extract_field_inventory,
    merge_constraints,
)


def validate_split(
    split: dict[str, Any],
    schema: dict[str, Any],
) -> list[str]:
    """Return a list of error strings.  Empty list = valid split."""
    errors: list[str] = []
    inventory = extract_field_inventory(schema)
    all_keys = inventory["all_top_level_keys"]

    partitions = split.get("partitions")
    if not isinstance(partitions, list) or not partitions:
        return ["split must have a non-empty 'partitions' list"]

    assigned: dict[str, str] = {}
    for part in partitions:
        name = part.get("name", "<unnamed>")
        fields = part.get("fields")
        if not isinstance(fields, list) or not fields:
            errors.append(f"Partition '{name}' has no fields")
            continue
        for fid in fields:
            if fid in assigned:
                errors.append(
                    f"Field '{fid}' appears in both '{assigned[fid]}' and '{name}'"
                )
            assigned[fid] = name

    covered = set(assigned)
    missing = all_keys - covered
    extra = covered - all_keys
    if missing:
        errors.append(f"Missing fields (not in any partition): {sorted(missing)}")
    if extra:
        errors.append(f"Unknown fields (not in schema): {sorted(extra)}")

    raw_constraints = extract_dependency_constraints(schema)
    merged = merge_constraints(raw_constraints)
    for constraint in merged:
        partitions_used = {assigned[fid] for fid in constraint if fid in assigned}
        if len(partitions_used) > 1:
            errors.append(
                f"Dependency constraint violated: {{{', '.join(sorted(constraint))}}} "
                f"is split across partitions: {sorted(partitions_used)}"
            )

    return errors
