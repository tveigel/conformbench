"""Extract field inventory and dependency constraints from a questionnaire schema.

The schema has a hierarchical tree of nodes (regular, group, gate, branch,
repeat_group).  The *flattened state* uses two kinds of top-level keys:

  - bare fields   — scalar/table ids (even if nested inside groups/gates/branches)
  - repeat groups — list-valued ids whose children are row-level dicts

A split partitions these top-level keys.  Dependency constraints say which keys
MUST land in the same partition to keep gate/branch/count relationships intact.
"""

from __future__ import annotations

from typing import Any


def extract_field_inventory(schema: dict[str, Any]) -> dict[str, Any]:
    """Return a structured inventory of all top-level state keys.

    Returns
    -------
    {
        "bare_fields": {field_id: {"type": ..., "question_text": ...}},
        "repeat_groups": {group_id: {
            "label": ...,
            "child_fields": {field_id: {"type": ..., "question_text": ...}},
        }},
        "all_top_level_keys": set[str],
    }
    """
    bare_fields: dict[str, dict[str, Any]] = {}
    repeat_groups: dict[str, dict[str, Any]] = {}

    def walk(nodes: list[dict[str, Any]], inside_repeat: str | None = None) -> None:
        for node in nodes:
            if not isinstance(node, dict):
                continue
            st = node.get("structure_type", "regular")

            if st == "regular":
                field_id = node.get("id")
                if not field_id:
                    continue
                info = {
                    "type": node.get("type"),
                    "question_text": node.get("question_text", ""),
                }
                if inside_repeat:
                    repeat_groups.setdefault(inside_repeat, {}).setdefault(
                        "child_fields", {}
                    )[field_id] = info
                else:
                    bare_fields[field_id] = info

            elif st == "repeat_group":
                group_id = node.get("id")
                if not group_id:
                    continue
                repeat_groups.setdefault(group_id, {})
                repeat_groups[group_id]["label"] = node.get("label", "")
                repeat_groups[group_id].setdefault("child_fields", {})
                repeat_cfg = node.get("repeat") or {}
                repeat_groups[group_id]["repeat_mode"] = repeat_cfg.get("mode")
                repeat_groups[group_id]["from_slot"] = repeat_cfg.get("from_slot")
                walk(node.get("fields") or [], inside_repeat=group_id)

            elif st in ("group", "gate"):
                walk(node.get("fields") or [], inside_repeat=inside_repeat)

            elif st == "branch":
                branch = node.get("branch") or {}
                for route in branch.get("routes") or []:
                    walk(route.get("children") or [], inside_repeat=inside_repeat)
                walk(branch.get("default_children") or [], inside_repeat=inside_repeat)

    walk(schema.get("questions") or [])

    all_keys = set(bare_fields) | set(repeat_groups)
    return {
        "bare_fields": bare_fields,
        "repeat_groups": repeat_groups,
        "all_top_level_keys": all_keys,
    }


def extract_dependency_constraints(schema: dict[str, Any]) -> list[set[str]]:
    """Return sets of top-level state keys that MUST be in the same partition.

    Sources of coupling:
      - gate.gate_on  → all bare/repeat keys inside the gate
      - branch.branch_on → all bare/repeat keys inside the branch
      - repeat.from_slot → the count field + the repeat group id
    """
    constraints: list[set[str]] = []

    def walk(
        nodes: list[dict[str, Any]],
        parent_constraint: set[str] | None = None,
        inside_repeat: str | None = None,
    ) -> None:
        for node in nodes:
            if not isinstance(node, dict):
                continue
            st = node.get("structure_type", "regular")

            if st == "regular":
                field_id = node.get("id")
                if not field_id:
                    continue
                key = inside_repeat if inside_repeat else field_id
                if parent_constraint is not None:
                    parent_constraint.add(key)

            elif st == "group":
                walk(node.get("fields") or [], parent_constraint, inside_repeat)

            elif st == "gate":
                gate_on = (node.get("gate") or {}).get("gate_on")
                if not gate_on:
                    walk(node.get("fields") or [], parent_constraint, inside_repeat)
                    continue
                constraint: set[str] = {gate_on}
                walk(node.get("fields") or [], constraint, inside_repeat)
                if len(constraint) > 1:
                    constraints.append(constraint)
                if parent_constraint is not None:
                    parent_constraint.update(constraint)

            elif st == "branch":
                branch = node.get("branch") or {}
                branch_on = branch.get("branch_on")
                if not branch_on:
                    for route in branch.get("routes") or []:
                        walk(route.get("children") or [], parent_constraint, inside_repeat)
                    walk(branch.get("default_children") or [], parent_constraint, inside_repeat)
                    continue
                constraint = {branch_on}
                for route in branch.get("routes") or []:
                    walk(route.get("children") or [], constraint, inside_repeat)
                walk(branch.get("default_children") or [], constraint, inside_repeat)
                if len(constraint) > 1:
                    constraints.append(constraint)
                if parent_constraint is not None:
                    parent_constraint.update(constraint)

            elif st == "repeat_group":
                group_id = node.get("id")
                if not group_id:
                    continue
                if parent_constraint is not None:
                    parent_constraint.add(group_id)
                repeat_cfg = node.get("repeat") or {}
                if repeat_cfg.get("mode") == "from_slot":
                    from_slot = repeat_cfg.get("from_slot")
                    if from_slot:
                        constraints.append({from_slot, group_id})
                walk(node.get("fields") or [], parent_constraint, inside_repeat=group_id)

    walk(schema.get("questions") or [])
    return constraints


def merge_constraints(constraints: list[set[str]]) -> list[set[str]]:
    """Merge overlapping constraint sets (union-find style)."""
    merged: list[set[str]] = []
    for c in constraints:
        overlapping = []
        disjoint = []
        for m in merged:
            if m & c:
                overlapping.append(m)
            else:
                disjoint.append(m)
        combined = set(c)
        for o in overlapping:
            combined |= o
        disjoint.append(combined)
        merged = disjoint
    return merged


def build_splitter_context(schema: dict[str, Any]) -> str:
    """Build a human-readable field summary + constraints for the splitter LLM."""
    inventory = extract_field_inventory(schema)
    raw_constraints = extract_dependency_constraints(schema)
    merged = merge_constraints(raw_constraints)

    lines: list[str] = []
    lines.append("=== FIELD INVENTORY ===\n")
    lines.append("Top-level bare fields (scalar values in the state):")
    for fid, info in sorted(inventory["bare_fields"].items()):
        lines.append(f"  - {fid} ({info['type']}): {info['question_text']}")

    lines.append("\nRepeat groups (list of row objects in the state):")
    for gid, ginfo in sorted(inventory["repeat_groups"].items()):
        from_slot = ginfo.get("from_slot")
        slot_note = f", count from field '{from_slot}'" if from_slot else ""
        lines.append(f"  - {gid} ({ginfo.get('label', '')}{slot_note}):")
        for fid, finfo in sorted(ginfo.get("child_fields", {}).items()):
            lines.append(f"      child: {fid} ({finfo['type']}): {finfo['question_text']}")

    if merged:
        lines.append("\n=== DEPENDENCY CONSTRAINTS ===")
        lines.append("These fields MUST be in the same partition:")
        for i, group in enumerate(merged, 1):
            lines.append(f"  Constraint {i}: {{{', '.join(sorted(group))}}}")

    lines.append(f"\nTotal top-level keys: {len(inventory['all_top_level_keys'])}")
    return "\n".join(lines)
