"""Minimal public benchmark runner."""

from __future__ import annotations

import json
import importlib
import traceback
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable
from uuid import uuid4

from .items import (
    BENCHMARK_ITEMS_DIR,
    PUBLIC_ITEMS_DIR,
    QUESTIONNAIRES_DIR,
    normalize_item,
)
from .questionnaires import load_questionnaire
from .evaluator.pipeline import evaluate_solver_turn


Solver = Callable[[SimpleNamespace], Any]


def run(
    *,
    items: Iterable[dict[str, Any]],
    solver: Solver,
    output_dir: str | Path | None = None,
    run_id: str | None = None,
    workers: int = 1,
    evaluator_model_id: str | None = None,
    evaluator_reasoning_effort: str | None = None,
    score: bool = False,
) -> dict[str, Any]:
    """Run `solver(turn)` over all items.

    The solver receives the public turn packet and returns the full resulting
    state as a dict. The runner validates the returned shape, records a
    diagnostic state diff, and evaluates the returned state as-is.
    """

    run_id = run_id or uuid4().hex
    output = Path(output_dir) if output_dir else None

    raw_items = list(items)
    worker_count = max(1, workers)

    def run_one(raw_item: dict[str, Any]) -> dict[str, Any]:
        item = normalize_item(raw_item)
        schema = load_questionnaire(item["questionnaire_id"])
        prior_state = deepcopy(item["prior_state"])
        visible_history = deepcopy(item["visible_history"])
        metadata = deepcopy(item["metadata"])
        turn = SimpleNamespace(
            item_id=item["item_id"],
            questionnaire_id=item["questionnaire_id"],
            schema=schema,
            form_schema=schema,
            prior_state=prior_state,
            public_state=prior_state,
            visible_history=visible_history,
            current_utterance=item["current_utterance"],
            metadata=metadata,
        )

        try:
            returned = solver(turn)
        except Exception as exc:
            if output:
                item_dir = _item_output_dir(output, item)
                item_dir.mkdir(parents=True, exist_ok=True)
                failure_payload = {
                    "item_id": item["item_id"],
                    "questionnaire_id": item["questionnaire_id"],
                    "stage": "solver_exception",
                    "exception_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                solver_failure = getattr(exc, "conformbench_failure", None)
                if isinstance(solver_failure, dict):
                    failure_payload["solver_failure"] = deepcopy(solver_failure)
                _write_json(item_dir / "solver_failure.json", failure_payload)
            raise
        try:
            candidate_state, extras = _unpack_solver_result(returned)
        except Exception as exc:
            if output:
                item_dir = _item_output_dir(output, item)
                item_dir.mkdir(parents=True, exist_ok=True)
                _write_json(
                    item_dir / "solver_failure.json",
                    {
                        "item_id": item["item_id"],
                        "questionnaire_id": item["questionnaire_id"],
                        "stage": "unpack_solver_result",
                        "exception_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                        "returned_repr": repr(returned)[:20000],
                    },
                )
            raise
        validation_errors = validate_resulting_state(candidate_state, schema)
        if validation_errors:
            message = "; ".join(validation_errors[:8])
            if len(validation_errors) > 8:
                message += f"; ... plus {len(validation_errors) - 8} more"
            if output:
                item_dir = _item_output_dir(output, item)
                item_dir.mkdir(parents=True, exist_ok=True)
                _write_json(
                    item_dir / "solver_failure.json",
                    {
                        "item_id": item["item_id"],
                        "questionnaire_id": item["questionnaire_id"],
                        "stage": "validate_resulting_state",
                        "error": message,
                        "validation_errors": validation_errors,
                        "candidate_state": candidate_state,
                        "agent_response": deepcopy(extras.get("agent_response"))
                        if isinstance(extras.get("agent_response"), dict)
                        else None,
                        "provenance": deepcopy(extras.get("provenance"))
                        if isinstance(extras.get("provenance"), dict)
                        else None,
                    },
                )
            raise ValueError(f"Solver returned an invalid resulting state for {item['item_id']}: {message}")

        operations = diff_state_operations(
            prior_state=prior_state,
            candidate_state=candidate_state,
            schema=schema,
        )

        result = {
            "item_id": item["item_id"],
            "questionnaire_id": item["questionnaire_id"],
            "operations": operations,
            "candidate_state": candidate_state,
            "gold_resulting_state": item["gold_resulting_state"],
        }
        if isinstance(extras.get("provenance"), dict):
            result["provenance"] = deepcopy(extras["provenance"])
        if "agent_response" in extras:
            result["agent_response"] = deepcopy(extras["agent_response"])
        turn_result = _build_turn_result(item, result)
        if score and item["gold_resulting_state"] is not None:
            evaluation = evaluate_solver_turn(
                turn_result,
                raw_item,
                model_id=evaluator_model_id,
                reasoning_effort=evaluator_reasoning_effort,
                questionnaire_path=_questionnaire_path(item["questionnaire_id"]),
            )
            result["evaluation_summary"] = evaluation.summary

        if output:
            item_dir = _item_output_dir(output, item)
            item_dir.mkdir(parents=True, exist_ok=True)
            solver_failure_path = item_dir / "solver_failure.json"
            if solver_failure_path.exists():
                solver_failure_path.unlink()
            _write_json(item_dir / "turn_result.json", turn_result)
            if score and item["gold_resulting_state"] is not None:
                _write_json(item_dir / "evaluation.json", evaluation.model_dump(mode="json"))

        return result

    if worker_count == 1:
        results = [run_one(raw_item) for raw_item in raw_items]
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            results = list(pool.map(run_one, raw_items))

    run_result = {"run_id": run_id, "item_count": len(results), "results": results}
    return run_result


def run_public(
    *,
    solver: Solver,
    output_dir: str | Path | None = None,
    run_id: str | None = None,
    workers: int = 1,
    evaluator_model_id: str | None = None,
    evaluator_reasoning_effort: str | None = None,
    score: bool = False,
) -> dict[str, Any]:
    """Run a solver on the packaged public development items."""

    from .public_items import load_public_items

    return run(
        items=load_public_items(),
        solver=solver,
        output_dir=output_dir,
        run_id=run_id,
        workers=workers,
        evaluator_model_id=evaluator_model_id,
        evaluator_reasoning_effort=evaluator_reasoning_effort,
        score=score,
    )


def load_solver(import_path: str) -> Solver:
    """Load a public benchmark solver from `module:function`."""

    if ":" not in import_path:
        raise ValueError("Solver must be specified as 'module:function'")
    module_name, attr_name = import_path.split(":", 1)
    if not module_name or not attr_name:
        raise ValueError("Solver must be specified as 'module:function'")

    solver = getattr(importlib.import_module(module_name), attr_name)
    if not callable(solver):
        raise TypeError(f"Solver is not callable: {import_path!r}")
    return solver


def exact_state_matches(result: dict[str, Any]) -> tuple[int, int]:
    """Count items whose candidate commitments match packaged gold.

    Strict candidate states may include empty schema fields that sparse
    gold packets omit. This helper ignores empty-only differences so the CLI
    remains a useful smoke-test signal. The evaluator itself still receives
    and scores the candidate state as returned.
    """

    scored = [
        item
        for item in result.get("results", [])
        if item.get("gold_resulting_state") is not None
    ]
    return (
        sum(
            _public_states_match(
                item.get("candidate_state"),
                item.get("gold_resulting_state"),
            )
            for item in scored
        ),
        len(scored),
    )


def validate_resulting_state(state: Any, schema: dict[str, Any]) -> list[str]:
    """Return shape errors for a candidate resulting state.

    This is intentionally a check only: it never repairs, fills, coerces, or
    normalizes the candidate state.
    """

    if not isinstance(state, dict):
        return ["resulting state must be a JSON object/dict"]

    spec = _state_shape(schema)
    known_top_level = spec["bare_fields"] | set(spec["repeat_groups"])
    errors: list[str] = []

    missing = sorted(known_top_level - set(state))
    if missing:
        errors.append(f"missing top-level field(s): {missing}")

    unknown = sorted(set(state) - known_top_level)
    if unknown:
        errors.append(f"unknown top-level field(s): {unknown}")

    for group_id, fields in sorted(spec["repeat_groups"].items()):
        if group_id not in state:
            continue
        rows = state[group_id]
        if rows is None:
            continue
        if not isinstance(rows, list):
            errors.append(f"repeat group {group_id!r} must be a list or null")
            continue
        for row_index, row in enumerate(rows):
            if not isinstance(row, dict):
                errors.append(f"repeat group {group_id!r} row {row_index} must be an object")
                continue
            row_missing = sorted(fields - set(row))
            if row_missing:
                errors.append(
                    f"repeat group {group_id!r} row {row_index} missing field(s): {row_missing}"
                )
            row_unknown = sorted(set(row) - fields)
            if row_unknown:
                errors.append(
                    f"repeat group {group_id!r} row {row_index} has unknown field(s): {row_unknown}"
                )
            for table_id, columns in sorted(spec["repeat_tables"].get(group_id, {}).items()):
                if table_id not in row:
                    continue
                _validate_table_rows(
                    value=row[table_id],
                    columns=columns,
                    label=(
                        f"repeat group {group_id!r} row {row_index} "
                        f"table field {table_id!r}"
                    ),
                    errors=errors,
                )

    for table_id, columns in sorted(spec["top_level_tables"].items()):
        if table_id not in state:
            continue
        _validate_table_rows(
            value=state[table_id],
            columns=columns,
            label=f"table field {table_id!r}",
            errors=errors,
        )

    return errors


def diff_state_operations(
    *,
    prior_state: dict[str, Any],
    candidate_state: dict[str, Any],
    schema: dict[str, Any],
) -> list[dict[str, Any]]:
    """Compute diagnostic public operations from prior state to candidate state.

    The returned operations are for inspection and reports only. They are not
    applied back to construct the candidate state.
    """

    spec = _state_shape(schema)
    ops: list[dict[str, Any]] = []

    for field_id in sorted(spec["bare_fields"]):
        new_val = candidate_state.get(field_id)
        old_val = prior_state.get(field_id)
        if new_val != old_val:
            ops.append({"op": "set", "path": field_id, "value": deepcopy(new_val)})

    for group_id, fields in sorted(spec["repeat_groups"].items()):
        new_rows = candidate_state.get(group_id)
        old_rows = prior_state.get(group_id)
        if not isinstance(new_rows, list):
            new_rows = []
        if not isinstance(old_rows, list):
            old_rows = []

        for row_index in range(len(old_rows) - 1, len(new_rows) - 1, -1):
            ops.append({"op": "delete_instance", "path": group_id, "index": row_index})

        for row_index, row in enumerate(new_rows):
            if not isinstance(row, dict):
                continue
            old_row = old_rows[row_index] if row_index < len(old_rows) else {}
            if not isinstance(old_row, dict):
                old_row = {}
            for field_id in sorted(fields):
                new_val = row.get(field_id)
                old_val = old_row.get(field_id)
                if new_val != old_val:
                    ops.append({
                        "op": "set",
                        "path": f"{group_id}[{row_index}].{field_id}",
                        "value": deepcopy(new_val),
                    })

    return ops


def score_run(
    run_dir: str | Path,
    *,
    evaluator_model_id: str | None = None,
    evaluator_reasoning_effort: str | None = None,
    items_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Score every ``turn_result.json`` in an existing run artifact tree."""

    root = Path(run_dir)
    scored: list[dict[str, Any]] = []
    turn_paths = sorted(root.rglob("turn_result.json"))
    if not turn_paths:
        raise FileNotFoundError(f"Run directory has no turn_result.json files: {root}")

    for turn_path in turn_paths:
        turn_result = json.loads(turn_path.read_text(encoding="utf-8"))
        ground_truth = _resolve_ground_truth_for_turn(
            turn_result,
            items_dir=Path(items_dir) if items_dir else None,
        )
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
            questionnaire_path=_questionnaire_path(str(questionnaire_id)),
        )
        evaluation_path = turn_path.parent / "evaluation.json"
        _write_json(evaluation_path, evaluation.model_dump(mode="json"))
        scored.append(
            {
                "item_id": str(turn_result.get("scenario_id") or turn_path.parent.name),
                "evaluation_path": str(evaluation_path),
                "summary": evaluation.summary,
            }
        )

    return {
        "run_dir": str(root),
        "item_count": len(scored),
        "items": scored,
    }


def _unpack_solver_result(returned: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(returned, dict):
        raise TypeError("Solver must return the full resulting state as a dict")

    for key in ("resulting_state", "candidate_state"):
        state = returned.get(key)
        if isinstance(state, dict):
            extras = {k: v for k, v in returned.items() if k != key}
            return deepcopy(state), extras

    return deepcopy(returned), {}


def _state_shape(schema: dict[str, Any]) -> dict[str, Any]:
    bare_fields: set[str] = set()
    repeat_groups: dict[str, set[str]] = {}
    top_level_tables: dict[str, set[str]] = {}
    repeat_tables: dict[str, dict[str, set[str]]] = {}

    def walk(nodes: list[dict[str, Any]], repeat_stack: tuple[str, ...] = ()) -> None:
        for node in nodes:
            if not isinstance(node, dict):
                continue
            structure_type = node.get("structure_type", "regular")

            if structure_type == "regular":
                field_id = node.get("id")
                if not field_id:
                    continue
                if repeat_stack:
                    group_id = repeat_stack[-1]
                    repeat_groups.setdefault(group_id, set()).add(field_id)
                    if node.get("type") == "table":
                        repeat_tables.setdefault(group_id, {})[field_id] = _table_columns(node)
                else:
                    bare_fields.add(field_id)
                    if node.get("type") == "table":
                        top_level_tables[field_id] = _table_columns(node)
                continue

            if structure_type == "repeat_group":
                group_id = node.get("id")
                if not group_id:
                    continue
                repeat_groups.setdefault(group_id, set())
                walk(node.get("fields") or [], (*repeat_stack, group_id))
                continue

            if structure_type in {"group", "gate"}:
                walk(node.get("fields") or [], repeat_stack)
                continue

            if structure_type == "branch":
                branch = node.get("branch") or {}
                for route in branch.get("routes") or []:
                    walk(route.get("children") or [], repeat_stack)
                walk(branch.get("default_children") or [], repeat_stack)

    walk(schema.get("questions") or [])
    return {
        "bare_fields": bare_fields,
        "repeat_groups": repeat_groups,
        "top_level_tables": top_level_tables,
        "repeat_tables": repeat_tables,
    }


def _table_columns(node: dict[str, Any]) -> set[str]:
    return {
        column["id"]
        for column in node.get("columns") or []
        if isinstance(column, dict) and column.get("id")
    }


def _validate_table_rows(
    *,
    value: Any,
    columns: set[str],
    label: str,
    errors: list[str],
) -> None:
    if value is None:
        return
    if not isinstance(value, list):
        errors.append(f"{label} must be a list or null")
        return
    for row_index, row in enumerate(value):
        if not isinstance(row, dict):
            errors.append(f"{label} row {row_index} must be an object")
            continue
        row_missing = sorted(columns - set(row))
        if row_missing:
            errors.append(f"{label} row {row_index} missing column(s): {row_missing}")
        row_unknown = sorted(set(row) - columns)
        if row_unknown:
            errors.append(f"{label} row {row_index} has unknown column(s): {row_unknown}")


def _build_turn_result(item: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    """Construct a turn_result dict compatible with the evaluator's TurnResultContract."""
    prior_state = item["prior_state"]
    candidate_state = result["candidate_state"]
    agent_response = (
        deepcopy(result.get("agent_response"))
        if isinstance(result.get("agent_response"), dict)
        else {}
    )
    agent_response.setdefault("operations", deepcopy(result.get("operations", [])))
    turn_result = {
        "scenario_id": item["item_id"],
        "questionnaire": item["questionnaire_id"],
        "scenario": item.get("scenario") or item["item_id"],
        "state": item.get("state") or "state_1",
        "utterance_id": item.get("utterance_id") or "u1",
        "turn_index": int(item.get("turn_index") or 0),
        "current_utterance": item["current_utterance"],
        "visible_history": item["visible_history"],
        "answers_before": prior_state,
        "answers_after": candidate_state,
        "answer_updates": _changed_top_level(prior_state, candidate_state),
        "agent_response": agent_response,
        "is_complete": True,
    }
    if isinstance(result.get("provenance"), dict):
        turn_result["provenance"] = result["provenance"]
    return turn_result


def _changed_top_level(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    updates: dict[str, Any] = {}
    for key in sorted(set(before) | set(after)):
        if before.get(key) != after.get(key):
            updates[key] = deepcopy(after.get(key))
    return updates


def _public_states_match(candidate: Any, gold: Any) -> bool:
    candidate_leaves = _flatten_state(candidate)
    gold_leaves = _flatten_state(gold)
    for key in sorted(set(candidate_leaves) | set(gold_leaves)):
        candidate_has = key in candidate_leaves
        gold_has = key in gold_leaves
        candidate_value = candidate_leaves.get(key)
        gold_value = gold_leaves.get(key)
        if not candidate_has and _is_empty_value(gold_value):
            continue
        if not gold_has and _is_empty_value(candidate_value):
            continue
        if candidate_value != gold_value:
            return False
    return True


def _flatten_state(state: Any, prefix: str = "") -> dict[str, Any]:
    if not prefix:
        if not isinstance(state, dict):
            return {}
        flattened: dict[str, Any] = {}
        for key in sorted(state):
            flattened.update(_flatten_state(state[key], str(key)))
        return flattened

    if isinstance(state, dict):
        flattened: dict[str, Any] = {}
        for key in sorted(state):
            flattened.update(_flatten_state(state[key], f"{prefix}.{key}"))
        return flattened

    if isinstance(state, list) and all(isinstance(item, dict) for item in state):
        flattened: dict[str, Any] = {}
        for idx, row in enumerate(state):
            flattened.update(_flatten_state(row, f"{prefix}[{idx}]"))
        return flattened

    return {prefix: state}


def _is_empty_value(value: Any) -> bool:
    if value is None or value == "":
        return True
    if isinstance(value, list):
        return all(_is_empty_value(item) for item in value)
    if isinstance(value, dict):
        return all(_is_empty_value(item) for item in value.values())
    return False


def _item_output_dir(output: Path, item: dict[str, Any]) -> Path:
    return (
        output
        / item["questionnaire_id"]
        / str(item.get("scenario") or item["item_id"])
        / str(item.get("state") or "state_1")
        / str(item.get("utterance_id") or "u1")
    )


def _questionnaire_path(questionnaire_id: str) -> Path | None:
    normalized = questionnaire_id.removeprefix("pilot_")
    path = QUESTIONNAIRES_DIR / f"{normalized}.json"
    return path if path.exists() else None


def _resolve_ground_truth_for_turn(
    turn_result: dict[str, Any],
    *,
    items_dir: Path | None,
) -> dict[str, Any]:
    item_id = str(turn_result.get("scenario_id") or "")
    questionnaire = str(turn_result.get("questionnaire") or "").removeprefix("pilot_")
    if not item_id:
        raise ValueError("turn_result is missing scenario_id")

    for root in _ground_truth_roots(items_dir):
        found = _find_ground_truth_packet(root, item_id, questionnaire)
        if found is not None:
            return found

    try:
        from .public_items import load_public_item_packet

        return load_public_item_packet(item_id)
    except KeyError:
        pass

    raise FileNotFoundError(
        f"Could not resolve ground truth for scenario_id={item_id!r}, "
        f"questionnaire={questionnaire!r}"
    )


def _ground_truth_roots(items_dir: Path | None) -> list[Path]:
    roots: list[Path] = []
    if items_dir is not None:
        roots.append(items_dir)
    roots.append(BENCHMARK_ITEMS_DIR)
    roots.append(PUBLIC_ITEMS_DIR)
    return roots


def _find_ground_truth_packet(
    root: Path,
    item_id: str,
    questionnaire: str,
) -> dict[str, Any] | None:
    if not root.exists():
        return None

    for path in sorted(root.rglob("ground_truth.json")):
        try:
            packet = json.loads(path.read_text(encoding="utf-8"))
            item = normalize_item(packet)
        except (json.JSONDecodeError, ValueError):
            continue
        if item["item_id"] != item_id:
            continue
        if questionnaire and item["questionnaire_id"] != questionnaire:
            continue
        return packet
    return None


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
