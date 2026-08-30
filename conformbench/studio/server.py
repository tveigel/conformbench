"""Small public Studio for item authoring and evaluation runs."""

from __future__ import annotations

import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from copy import deepcopy
from datetime import datetime, timezone
import json
import re
from typing import Any
from uuid import uuid4

try:
    import uvicorn
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse
    from pydantic import BaseModel, Field
except ImportError as exc:  # pragma: no cover - exercised by users without studio deps.
    raise ImportError(
        "ConFormBench Studio requires the optional studio dependencies. "
        "Install with `pip install -e .[studio]`."
    ) from exc

from conformbench import benchmark
from conformbench.items import DATA_ROOT, discover_items, normalize_item
from conformbench.public_items import (
    load_public_item_packet,
    load_public_item_packets,
    load_public_items,
)
from conformbench.questionnaires import list_questionnaires, load_questionnaire


_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,255}$")
_STUDIO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_EVAL_WORKERS = 4
_DEFAULT_DEV_SPLIT_ID = "conformbench_v1_20_80_seed20260524"
_DEFAULT_DEV_ITEMS_DIR = (
    DATA_ROOT / "items" / "splits" / _DEFAULT_DEV_SPLIT_ID / "dev"
)


class StudioConfig(BaseModel):
    items_dir: Path = Field(default_factory=lambda: DATA_ROOT / "items" / "benchmark")
    dev_items_dir: Path = Field(default_factory=lambda: _DEFAULT_DEV_ITEMS_DIR)
    runs_dir: Path = Field(default_factory=lambda: DATA_ROOT / "reports" / "runs")


class EvalRequest(BaseModel):
    solver: str
    item_source: str = "public"
    run_id: str | None = None
    item_ids: list[str] = Field(default_factory=list)
    model_id: str | None = None
    model_reasoning_effort: str | None = None
    evaluator_model_id: str | None = None
    evaluator_reasoning_effort: str | None = None
    workers: int = _DEFAULT_EVAL_WORKERS


_AVAILABLE_MODELS = [
    {"id": "openai:gpt-5.4", "name": "GPT-5.4", "provider": "openai"},
    {"id": "openai:gpt-5.4-mini", "name": "GPT-5.4 Mini", "provider": "openai"},
    {"id": "openai:gpt-5.4-nano", "name": "GPT-5.4 Nano", "provider": "openai"},
    {"id": "anthropic:claude-3-5-haiku-20241022", "name": "Claude Haiku 3.5", "provider": "anthropic"},
    {"id": "anthropic:claude-sonnet-4-20250514", "name": "Claude Sonnet 4", "provider": "anthropic"},
    {"id": "anthropic:claude-opus-4-1-20250805", "name": "Claude Opus 4.1", "provider": "anthropic"},
    {"id": "anthropic:claude-opus-4-7", "name": "Claude Opus 4.7", "provider": "anthropic"},
    {"id": "google:gemini-3.5-flash", "name": "Gemini 3.5 Flash", "provider": "google"},
    {"id": "openai:kimi-k2.6", "name": "Kimi K2.6", "provider": "moonshot"},
]

_AVAILABLE_SYSTEMS = [
    {"id": "conformbench.systems.YOUR_SYSTEM:solve", "name": "YOUR_SYSTEM skeleton"},
    {"id": "conformbench.systems.no_update:solve", "name": "NoUpdate lower bound"},
    {"id": "conformbench.systems.direct:solve", "name": "Direct JSON"},
    {"id": "conformbench.systems.flatagent:solve", "name": "FlatAgent"},
]

_REASONING_LEVELS = {"none", "low", "medium", "high"}

_FIGURE_ORDER = (
    "whole_form_overview.png",
    "task_diagnostics_success.png",
    "transition_accuracy_overview.png",
    "state_changed_f1_comparison.png",
)

_FIGURE_METADATA = {
    "whole_form_overview.png": {
        "title": "Whole Form Overview",
        "note": (
            "Accuracy = correct / evaluated. Strict F1 = 2PR/(P+R), "
            "with P = correct / evaluated and R = correct / gold_expected. "
            "Exact = exact items / scored items. Changed F1 uses "
            "P = changed_correct / predicted_changed and R = changed_correct / gold_changed. "
            "Preservation = preserved fields / preserve fields."
        ),
    },
    "task_diagnostics_success.png": {
        "title": "Task Diagnostic Success",
        "note": "Success rate = successful applicable cases / applicable cases for each diagnostic.",
    },
    "transition_accuracy_overview.png": {
        "title": "Transition Accuracy Overview",
        "note": (
            "Transition accuracy = correct fields for a gold transition / all gold fields "
            "with that transition. Transitions are preserve, set, change, and clear."
        ),
    },
    "state_changed_f1_comparison.png": {
        "title": "State Changed F1 Comparison",
        "note": (
            "For each S1-S4 prior-state bucket: P = changed_correct / predicted_changed, "
            "R = changed_correct / gold_changed, F1 = 2PR/(P+R)."
        ),
    },
}


def create_app(
    *,
    items_dir: str | Path | None = None,
    dev_items_dir: str | Path | None = None,
    train_items_dir: str | Path | None = None,
    runs_dir: str | Path | None = None,
) -> FastAPI:
    dev_dir = dev_items_dir or train_items_dir
    config = StudioConfig(
        items_dir=Path(items_dir) if items_dir else StudioConfig().items_dir,
        dev_items_dir=Path(dev_dir) if dev_dir else StudioConfig().dev_items_dir,
        runs_dir=Path(runs_dir) if runs_dir else StudioConfig().runs_dir,
    )
    config.items_dir.mkdir(parents=True, exist_ok=True)
    config.runs_dir.mkdir(parents=True, exist_ok=True)

    app = FastAPI(title="ConFormBench Studio", docs_url="/api/docs")
    app.state.conformbench_studio = config

    static_dir = Path(__file__).resolve().parent / "static"

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(
            str(static_dir / "index.html"),
            headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
        )

    @app.get("/static/{file_path:path}")
    def static_file(file_path: str) -> FileResponse:
        full = (static_dir / file_path).resolve()
        if not str(full).startswith(str(static_dir.resolve())):
            raise HTTPException(403, "Forbidden")
        if not full.is_file():
            raise HTTPException(404, "Not found")
        return FileResponse(
            str(full),
            headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
        )

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {
            "ok": True,
            "artifact_contract": "per-item turn_result.json plus evaluation.json",
            "server_file": __file__,
            "items_dir": str(config.items_dir),
            "dev_items_dir": str(config.dev_items_dir),
            "runs_dir": str(config.runs_dir),
        }

    @app.get("/api/questionnaires")
    def questionnaires() -> list[str]:
        return list_questionnaires()

    @app.get("/api/questionnaires/{questionnaire_id}")
    def questionnaire(questionnaire_id: str) -> dict[str, Any]:
        try:
            return load_questionnaire(questionnaire_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.get("/api/questionnaire/{questionnaire_id}/fields")
    def questionnaire_fields(questionnaire_id: str) -> dict[str, Any]:
        schema = _load_public_questionnaire(questionnaire_id)
        return {"fields": [field["id"] for field in _questionnaire_leaf_metadata(schema)]}

    @app.get("/api/questionnaire/{questionnaire_id}/field-meta")
    def questionnaire_field_meta(questionnaire_id: str) -> dict[str, Any]:
        schema = _load_public_questionnaire(questionnaire_id)
        return {"fields": _questionnaire_leaf_metadata(schema)}

    @app.get("/api/questionnaire/{questionnaire_id}/tree")
    def questionnaire_tree(questionnaire_id: str) -> dict[str, Any]:
        schema = _load_public_questionnaire(questionnaire_id)
        return {"questions": schema.get("questions") or []}

    @app.get("/api/items")
    def items() -> list[dict[str, Any]]:
        return _authoring_item_summaries(config.items_dir)

    @app.get("/api/items/{item_id}")
    def item(item_id: str) -> dict[str, Any]:
        packet_info = _get_item_packet(config.items_dir, item_id)
        return _authoring_item(
            packet_info["packet"],
            source=packet_info["source"],
            read_only=packet_info["read_only"],
        )

    @app.post("/api/items")
    def create_item(packet: dict[str, Any]) -> dict[str, Any]:
        if "prior_state" not in packet:
            packet = _new_public_packet(packet)
        return _save_item(config.items_dir, packet, allow_public_overwrite=False)

    @app.post("/api/items/{item_id}/copy")
    def copy_item(item_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        _validate_id(item_id, label="item_id")
        new_item_id = str(payload.get("new_item_id") or "").strip()
        _validate_id(new_item_id, label="new_item_id")
        if _item_exists(config.items_dir, new_item_id):
            raise HTTPException(409, f"Item already exists: {new_item_id}")
        source = _get_item_packet(config.items_dir, item_id)
        packet = {**source["packet"], "item_id": new_item_id}
        packet["title"] = packet.get("title") or _title_from_item_id(new_item_id)
        return _save_item(config.items_dir, packet, allow_public_overwrite=False)

    @app.put("/api/items/{item_id}")
    def update_item(item_id: str, packet: dict[str, Any]) -> dict[str, Any]:
        pending_id = packet.pop("_pendingItemId", None)
        if pending_id and pending_id != item_id:
            raise HTTPException(400, "Use the rename endpoint before saving with a new item_id.")
        packet = _public_packet_from_authoring(packet, item_id=item_id)
        return _save_item(config.items_dir, packet, allow_public_overwrite=False)

    @app.delete("/api/items/{item_id}")
    def delete_item(item_id: str) -> dict[str, Any]:
        _validate_id(item_id, label="item_id")
        packet_info = _get_item_packet(config.items_dir, item_id)
        if packet_info["read_only"]:
            raise HTTPException(409, "Packaged public items cannot be deleted.")
        path = Path(packet_info.get("path") or "")
        if not path.exists():
            raise HTTPException(404, f"Unknown item: {item_id}")
        path.unlink()
        _remove_empty_parents(path.parent, stop=config.items_dir)
        return {"deleted": True, "item_id": item_id}

    @app.post("/api/items/{item_id}/rename")
    def rename_item(item_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        _validate_id(item_id, label="item_id")
        new_item_id = str(payload.get("new_item_id") or "").strip()
        _validate_id(new_item_id, label="new_item_id")
        if new_item_id in {public["item_id"] for public in load_public_items()}:
            raise HTTPException(409, "Packaged public item ids are read-only.")
        old = _get_item_packet(config.items_dir, item_id)
        if old["read_only"]:
            raise HTTPException(409, "Packaged public items cannot be renamed.")
        if _item_exists(config.items_dir, new_item_id):
            raise HTTPException(409, f"Item already exists: {new_item_id}")
        packet = {**old["packet"], "item_id": new_item_id}
        saved = _save_item(config.items_dir, packet, allow_public_overwrite=False)
        old_path = Path(old.get("path") or "")
        if old_path.exists():
            old_path.unlink()
        return {"ok": True, "item_id": new_item_id, **saved}

    @app.get("/api/items/{item_id}/resolve-prior-state")
    def resolve_prior_state(item_id: str) -> dict[str, Any]:
        packet_info = _get_item_packet(config.items_dir, item_id)
        item = normalize_item(packet_info["packet"])
        return {
            "questionnaire": item["questionnaire_id"],
            "questionnaire_answers": item["prior_state"] or None,
        }

    @app.get("/api/contexts/{kind}")
    def list_contexts(kind: str) -> list[dict[str, Any]]:
        if kind not in {"state", "history"}:
            raise HTTPException(404, f"Unknown context kind: {kind}")
        contexts = []
        for packet, source, read_only in _all_packets(config.items_dir):
            authoring = _authoring_item(packet, source=source, read_only=read_only)
            ref = authoring["state_ref"] if kind == "state" else authoring["history_ref"]
            contexts.append(_context_summary(kind, ref, authoring))
        return sorted(contexts, key=lambda context: context["ref"])

    @app.get("/api/contexts/{kind}/{ref:path}/usage")
    def context_usage(kind: str, ref: str) -> dict[str, Any]:
        if kind not in {"state", "history"}:
            raise HTTPException(404, f"Unknown context kind: {kind}")
        items = []
        for packet, source, read_only in _all_packets(config.items_dir):
            authoring = _authoring_item(packet, source=source, read_only=read_only)
            expected_ref = authoring["state_ref"] if kind == "state" else authoring["history_ref"]
            if expected_ref == ref:
                items.append({"item_id": authoring["item_id"], "title": authoring.get("title", "")})
        return {"items": items}

    @app.put("/api/contexts/{kind}/{ref:path}")
    def update_context(kind: str, ref: str, payload: dict[str, Any]) -> dict[str, Any]:
        if kind not in {"state", "history"}:
            raise HTTPException(404, f"Unknown context kind: {kind}")
        packet_info = _packet_for_context_ref(config.items_dir, kind, ref)
        if packet_info["read_only"]:
            raise HTTPException(409, "Packaged public items are read-only. Duplicate the item before editing its context.")
        packet = dict(packet_info["packet"])
        if kind == "state":
            answers = payload.get("questionnaire_answers")
            packet["prior_state"] = answers if isinstance(answers, dict) else {}
            packet["state_condition"] = payload.get("condition_code") or packet.get("state_condition") or "S1"
            packet["state_ref"] = ref
        else:
            packet["visible_history"] = _normalise_history_turns(payload.get("turns") or [])
            packet["history_condition"] = payload.get("condition_code") or packet.get("history_condition") or "H1"
            packet["history_ref"] = ref
        _save_item(config.items_dir, packet, allow_public_overwrite=False)
        return {"saved": True, "ref": ref, "kind": kind}

    @app.get("/api/contexts/state/{ref:path}")
    def state_context(ref: str) -> dict[str, Any]:
        return _context_detail(config.items_dir, "state", ref)

    @app.get("/api/contexts/history/{ref:path}")
    def history_context(ref: str) -> dict[str, Any]:
        return _context_detail(config.items_dir, "history", ref)

    @app.get("/api/scenarios")
    def scenarios() -> list[dict[str, Any]]:
        return []

    @app.get("/api/evals")
    def evals() -> list[dict[str, Any]]:
        summaries: list[dict[str, Any]] = []
        for run_dir in sorted((path for path in config.runs_dir.iterdir() if path.is_dir()), reverse=True):
            summaries.append(_run_summary_from_artifacts(run_dir))
        return summaries

    @app.get("/api/evals/{run_id}")
    def eval_result(run_id: str) -> dict[str, Any]:
        _validate_id(run_id, label="run_id")
        run_dir = config.runs_dir / run_id
        if not run_dir.exists():
            raise HTTPException(404, f"Unknown run: {run_id}")
        return _run_summary_from_artifacts(run_dir)

    @app.get("/api/evals/{run_id}/summary")
    def eval_summary_report(run_id: str) -> dict[str, Any]:
        _validate_id(run_id, label="run_id")
        run_dir = config.runs_dir / run_id
        summary_path = run_dir / "summary_report.json"
        if not summary_path.exists():
            raise HTTPException(404, f"No summary_report.json for run: {run_id}")
        return json.loads(summary_path.read_text(encoding="utf-8"))

    @app.get("/api/evals/{run_id}/report")
    def eval_markdown_report(run_id: str) -> dict[str, Any]:
        _validate_id(run_id, label="run_id")
        run_dir = config.runs_dir / run_id
        report_path = run_dir / "report.md"
        if not report_path.exists():
            raise HTTPException(404, f"No report.md for run: {run_id}")
        return {
            "run_id": run_id,
            "filename": report_path.name,
            "markdown": report_path.read_text(encoding="utf-8"),
        }

    @app.get("/api/evals/{run_id}/figures")
    def eval_figures(run_id: str) -> list[dict[str, Any]]:
        _validate_id(run_id, label="run_id")
        run_dir = config.runs_dir / run_id
        figures_dir = run_dir / "figures"
        if not figures_dir.exists():
            return []
        figures: list[dict[str, Any]] = []
        for filename in _FIGURE_ORDER:
            path = figures_dir / filename
            if not path.exists():
                continue
            metadata = _FIGURE_METADATA.get(filename, {})
            figures.append({
                "filename": path.name,
                "title": metadata.get("title") or path.stem.replace("_", " ").title(),
                "note": metadata.get("note", ""),
                "url": f"/api/evals/{run_id}/figures/{path.name}",
            })
        return figures

    @app.get("/api/evals/{run_id}/figures/{filename}")
    def eval_figure_file(run_id: str, filename: str) -> FileResponse:
        _validate_id(run_id, label="run_id")
        if Path(filename).name != filename or not filename.endswith(".png"):
            raise HTTPException(400, "Invalid figure filename.")
        path = config.runs_dir / run_id / "figures" / filename
        if not path.exists():
            raise HTTPException(404, f"Unknown figure: {filename}")
        return FileResponse(str(path), media_type="image/png")

    @app.get("/api/evals/{run_id}/items/{item_id}")
    def eval_item_detail(run_id: str, item_id: str) -> dict[str, Any]:
        _validate_id(run_id, label="run_id")
        _validate_id(item_id, label="item_id")
        run_dir = config.runs_dir / run_id
        item_dir = _find_run_item_dir(run_dir, item_id)
        if item_dir is None:
            raise HTTPException(404, f"Unknown run item: {run_id}/{item_id}")

        def read_json(name: str, default: Any = None) -> Any:
            path = item_dir / name
            if not path.exists():
                return default
            return json.loads(path.read_text(encoding="utf-8"))

        turn_result = read_json("turn_result.json", {})
        ground_truth: dict[str, Any] = {}
        if turn_result:
            try:
                ground_truth = benchmark._resolve_ground_truth_for_turn(
                    turn_result,
                    items_dir=config.items_dir,
                )
            except Exception:
                ground_truth = {}

        return {
            "run_id": run_id,
            "item_id": item_id,
            "turn_result": turn_result,
            "evaluation": read_json("evaluation.json", {}),
            "ground_truth": ground_truth,
        }

    @app.get("/api/eval/item-tree")
    def eval_item_tree() -> dict[str, Any]:
        return _build_item_tree(config.items_dir)

    @app.get("/api/models")
    def list_models() -> list[dict[str, Any]]:
        return _AVAILABLE_MODELS

    @app.get("/api/systems")
    def list_systems() -> list[dict[str, Any]]:
        return _AVAILABLE_SYSTEMS

    @app.post("/api/evals")
    def run_eval(request: EvalRequest) -> dict[str, Any]:
        return _run_eval(config, request)

    @app.post("/api/items/{item_id}/continue-next-turn")
    def continue_next_turn(item_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        _validate_id(item_id, label="item_id")
        new_item_id = str(payload.get("new_item_id") or "").strip()
        _validate_id(new_item_id, label="new_item_id")
        if _item_exists(config.items_dir, new_item_id):
            raise HTTPException(409, f"Item already exists: {new_item_id}")

        source_info = _get_item_packet(config.items_dir, item_id)
        source = normalize_item(source_info["packet"])
        current_utterance = payload.get("current_utterance") or source["current_utterance"]
        if isinstance(current_utterance, dict):
            turn_text = str(current_utterance.get("text") or "")
            speaker = str(current_utterance.get("speaker") or "user")
        else:
            turn_text = str(current_utterance or "")
            speaker = "user"
        next_history = _normalise_history_turns(source["visible_history"])
        if turn_text:
            next_history.append({"speaker": speaker, "text": turn_text})

        packet = {
            **source_info["packet"],
            "item_id": new_item_id,
            "title": _title_from_item_id(new_item_id),
            "status": "draft",
            "primary_delta_type": payload.get("primary_delta_type") or payload.get("family") or "add",
            "family": payload.get("family") or payload.get("primary_delta_type") or "add",
            "contrast_role": payload.get("contrast_role") or source["metadata"].get("contrast_role") or "benchmark",
            "prior_state": payload.get("gold_resulting_state") or source["gold_resulting_state"] or {},
            "visible_history": next_history,
            "current_utterance": "",
            "gold_resulting_state": {},
            "state_ref": f"{source['questionnaire_id']}/{new_item_id}_prior_state",
            "history_ref": f"{source['questionnaire_id']}/{new_item_id}_visible_history",
            "state_condition": "S2",
            "history_condition": "H2" if next_history else "H1",
        }
        return _save_item(config.items_dir, packet, allow_public_overwrite=False)

    return app


def _item_summaries(items_dir: Path) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for packet in load_public_item_packets():
        summaries.append(_summary_for_packet(packet, source="public", read_only=True))
    for packet, path in _custom_packets(items_dir):
        summary = _summary_for_packet(packet, source="studio", read_only=False)
        summary["path"] = str(path)
        summaries.append(summary)
    return sorted(summaries, key=lambda item: (item["source"], item["questionnaire_id"], item["item_id"]))


def _authoring_item_summaries(items_dir: Path) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for packet, source, read_only in _all_packets(items_dir):
        authoring = _authoring_item(packet, source=source, read_only=read_only)
        summaries.append({
            "item_id": authoring["item_id"],
            "title": authoring.get("title", ""),
            "primary_delta_type": authoring.get("primary_delta_type", ""),
            "family": authoring.get("family", ""),
            "contrast_role": authoring.get("contrast_role", ""),
            "status": authoring.get("status", "draft"),
            "state_condition": authoring.get("state_condition", ""),
            "state_ref": authoring.get("state_ref"),
            "history_condition": authoring.get("history_condition", ""),
            "history_ref": authoring.get("history_ref"),
            "update_type": authoring.get("update_type", []),
            "evidence": authoring.get("evidence", {}),
            "derived_variables": authoring.get("derived_variables", {}),
            "questionnaire": authoring.get("questionnaire", {}).get("source", ""),
            "scenario": authoring.get("scenario", ""),
            "read_only": read_only,
            "source": source,
        })
    return sorted(summaries, key=lambda item: (item["questionnaire"], item["item_id"]))


def _all_packets(items_dir: Path) -> list[tuple[dict[str, Any], str, bool]]:
    packets: list[tuple[dict[str, Any], str, bool]] = [
        (packet, "public", True)
        for packet in load_public_item_packets()
    ]
    packets.extend((packet, "studio", False) for packet, _path in _custom_packets(items_dir))
    return packets


def _get_item_packet(items_dir: Path, item_id: str) -> dict[str, Any]:
    _validate_id(item_id, label="item_id")
    for packet, path in _custom_packets(items_dir):
        if normalize_item(packet)["item_id"] == item_id:
            return {
                "source": "studio",
                "read_only": False,
                "path": str(path),
                "packet": packet,
            }
    try:
        return {
            "source": "public",
            "read_only": True,
            "packet": load_public_item_packet(item_id),
        }
    except KeyError as exc:
        raise HTTPException(404, f"Unknown item: {item_id}") from exc


def _authoring_item(packet: dict[str, Any], *, source: str, read_only: bool) -> dict[str, Any]:
    item = normalize_item(packet)
    metadata = item["metadata"]
    item_id = item["item_id"]
    questionnaire_id = item["questionnaire_id"]
    state_ref = str(metadata.get("state_ref") or f"{questionnaire_id}/{item_id}_prior_state")
    history_ref = str(metadata.get("history_ref") or f"{questionnaire_id}/{item_id}_visible_history")
    visible_history = _normalise_history_turns(item["visible_history"])
    primary_delta_type = str(
        metadata.get("primary_delta_type")
        or metadata.get("family")
        or metadata.get("delta_type")
        or "add"
    )
    state_condition = str(metadata.get("state_condition") or ("S1" if not item["prior_state"] else "S2"))
    history_condition = str(metadata.get("history_condition") or ("H2" if visible_history else "H1"))
    title = str(metadata.get("title") or _title_from_item_id(item_id))
    difficulty_tier = str(metadata.get("difficulty_tier") or "challenge")
    status = str(metadata.get("status") or ("ready" if read_only else "draft"))
    evidence = {
        "history_required": bool(metadata.get("history_required", bool(visible_history))),
        "support_distance": int(metadata.get("support_distance", 1 if visible_history else 0) or 0),
        "conflict_present": bool(metadata.get("conflict_present", False)),
    }
    derived_variables = {
        "changed_leaf_field_count": 0,
        "primary_delta_type": primary_delta_type,
        "prior_state_condition": state_condition,
        "history_required": evidence["history_required"],
        "support_distance": evidence["support_distance"],
        "conflict_present": evidence["conflict_present"],
        "repeat_group_involvement": "none",
        "repeat_group_involved": False,
        "repeat_group_names": [],
        "repeat_group_changed_leaf_count": 0,
        "revision_operation": "no_change",
    }
    expected_outcome = metadata.get("expected_outcome")
    if not isinstance(expected_outcome, dict):
        expected_outcome = {
            "fields": metadata.get("fields") or {},
            "repeat_groups": metadata.get("repeat_groups") or {},
        }
    return {
        "item_id": item_id,
        "title": title,
        "status": status,
        "primary_delta_type": primary_delta_type,
        "family": primary_delta_type,
        "contrast_role": metadata.get("contrast_role") or "benchmark",
        "schema_subset": metadata.get("schema_subset") or {},
        "questionnaire": {
            "source": questionnaire_id,
            "notes": metadata.get("questionnaire_notes") or "",
        },
        "state_condition": state_condition,
        "history_condition": history_condition,
        "state_ref": state_ref,
        "history_ref": history_ref,
        "evidence": evidence,
        "prior_state": item["prior_state"],
        "current_state": {"questionnaire_answers": item["prior_state"] or None},
        "visible_history": visible_history,
        "current_utterance": {"speaker": "user", "text": item["current_utterance"]},
        "expected_outcome": expected_outcome,
        "gold_resulting_state": item["gold_resulting_state"] or {},
        "forbidden_commits": metadata.get("forbidden_commits") or [],
        "forbidden_mutations": metadata.get("forbidden_mutations") or [],
        "targeted_failure_mode": metadata.get("targeted_failure_mode") or [],
        "gold_state_diff": metadata.get("gold_state_diff") or [],
        "gold_annotations": metadata.get("gold_annotations") or {},
        "semantic_ius": metadata.get("semantic_ius") or item.get("semantic_ius") or [],
        "non_preserve_transitions": metadata.get("non_preserve_transitions") or [],
        "evidence_spans": metadata.get("evidence_spans") or [],
        "derived_variables": metadata.get("derived_variables") or derived_variables,
        "difficulty_profile": metadata.get("difficulty_profile") or {
            "dimensions": {},
            "stressors": {},
            "dimension_notes": {},
            "stressor_notes": {},
            "targeted_failure_modes": [],
            "failure_explanation": "",
        },
        "difficulty_tier": difficulty_tier,
        "author_notes": metadata.get("author_notes") or "",
        "scenario": metadata.get("scenario") or "",
        "materialization": metadata.get("materialization") or {
            "scenario_name": f"public_{item_id}",
            "difficulty": difficulty_tier,
            "iu_description": "",
            "state_id": "state_1",
            "utterance_id": "public_turn_1",
        },
        "_read_only": read_only,
        "_source": source,
    }


def _new_public_packet(payload: dict[str, Any]) -> dict[str, Any]:
    item_id = str(payload.get("item_id") or "").strip()
    _validate_id(item_id, label="item_id")
    questionnaire_id = str(payload.get("questionnaire_id") or payload.get("questionnaire") or list_questionnaires()[0])
    if questionnaire_id not in list_questionnaires():
        raise HTTPException(400, f"Unknown questionnaire_id: {questionnaire_id!r}")
    return {
        "item_id": item_id,
        "questionnaire_id": questionnaire_id,
        "title": payload.get("title") or _title_from_item_id(item_id),
        "status": "draft",
        "primary_delta_type": payload.get("primary_delta_type") or payload.get("family") or "add",
        "family": payload.get("family") or payload.get("primary_delta_type") or "add",
        "contrast_role": payload.get("contrast_role") or "benchmark",
        "state_condition": "S1",
        "history_condition": "H1",
        "prior_state": {},
        "visible_history": [],
        "current_utterance": "",
        "gold_resulting_state": {},
        "semantic_ius": [],
    }


def _public_packet_from_authoring(authoring: dict[str, Any], *, item_id: str) -> dict[str, Any]:
    questionnaire = authoring.get("questionnaire") or {}
    questionnaire_id = str(
        questionnaire.get("source")
        or authoring.get("questionnaire_id")
        or authoring.get("source_questionnaire")
        or ""
    )
    current_utterance = authoring.get("current_utterance")
    if isinstance(current_utterance, dict):
        utterance_text = str(current_utterance.get("text") or "")
    else:
        utterance_text = str(current_utterance or "")

    expected_outcome = authoring.get("expected_outcome") or {"fields": {}, "repeat_groups": {}}
    fields_contract = authoring.get("fields")
    repeat_groups_contract = authoring.get("repeat_groups")
    if not isinstance(fields_contract, dict):
        fields_contract = expected_outcome.get("fields") if isinstance(expected_outcome, dict) else {}
    if not isinstance(repeat_groups_contract, dict):
        repeat_groups_contract = expected_outcome.get("repeat_groups") if isinstance(expected_outcome, dict) else {}

    packet = {
        "item_id": item_id,
        "questionnaire_id": questionnaire_id,
        "title": authoring.get("title") or _title_from_item_id(item_id),
        "status": authoring.get("status") or "draft",
        "primary_delta_type": authoring.get("primary_delta_type") or authoring.get("family") or "add",
        "family": authoring.get("family") or authoring.get("primary_delta_type") or "add",
        "contrast_role": authoring.get("contrast_role") or "benchmark",
        "state_condition": authoring.get("state_condition") or "S1",
        "history_condition": authoring.get("history_condition") or "H1",
        "state_ref": authoring.get("state_ref"),
        "history_ref": authoring.get("history_ref"),
        "evidence": authoring.get("evidence") or {},
        "prior_state": authoring.get("prior_state")
        if isinstance(authoring.get("prior_state"), dict)
        else (authoring.get("current_state") or {}).get("questionnaire_answers") or {},
        "visible_history": _normalise_history_turns(authoring.get("visible_history") or []),
        "current_utterance": utterance_text,
        "gold_resulting_state": authoring.get("gold_resulting_state") or {},
        "expected_outcome": expected_outcome,
        "fields": fields_contract or {},
        "repeat_groups": repeat_groups_contract or {},
        "gold_annotations": authoring.get("gold_annotations") or {},
        "semantic_ius": authoring.get("semantic_ius") or [],
        "non_preserve_transitions": authoring.get("non_preserve_transitions") or [],
        "evidence_spans": authoring.get("evidence_spans") or [],
        "derived_variables": authoring.get("derived_variables") or {},
        "difficulty_profile": authoring.get("difficulty_profile") or {},
        "difficulty_tier": authoring.get("difficulty_tier") or "challenge",
        "author_notes": authoring.get("author_notes") or "",
        "scenario": authoring.get("scenario") or "",
        "materialization": authoring.get("materialization") or {},
    }
    return packet


def _normalise_history_turns(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    turns: list[dict[str, str]] = []
    for entry in value:
        if isinstance(entry, dict):
            text = str(entry.get("text") or entry.get("content") or "")
            speaker = str(entry.get("speaker") or entry.get("role") or "user")
        else:
            text = str(entry)
            speaker = "user"
        if text:
            turns.append({"speaker": speaker, "text": text})
    return turns


def _context_summary(kind: str, ref: str, authoring: dict[str, Any]) -> dict[str, Any]:
    if kind == "state":
        is_empty = not bool(authoring.get("prior_state"))
        condition_code = authoring.get("state_condition") or "S1"
    else:
        is_empty = not bool(authoring.get("visible_history"))
        condition_code = authoring.get("history_condition") or "H1"
    return {
        "kind": kind,
        "ref": ref,
        "questionnaire": authoring.get("questionnaire", {}).get("source", ""),
        "condition_code": condition_code,
        "description": authoring.get("title", ""),
        "is_empty": is_empty,
    }


def _context_detail(items_dir: Path, kind: str, ref: str) -> dict[str, Any]:
    for packet, source, read_only in _all_packets(items_dir):
        authoring = _authoring_item(packet, source=source, read_only=read_only)
        expected_ref = authoring["state_ref"] if kind == "state" else authoring["history_ref"]
        if expected_ref != ref:
            continue
        if kind == "state":
            return {
                "condition_code": authoring.get("state_condition") or "S1",
                "questionnaire": authoring.get("questionnaire", {}).get("source", ""),
                "description": authoring.get("title", ""),
                "questionnaire_answers": authoring.get("prior_state") or None,
            }
        return {
            "condition_code": authoring.get("history_condition") or "H1",
            "questionnaire": authoring.get("questionnaire", {}).get("source", ""),
            "description": authoring.get("title", ""),
            "turns": authoring.get("visible_history") or [],
        }
    raise HTTPException(404, f"Unknown {kind} context: {ref}")


def _packet_for_context_ref(items_dir: Path, kind: str, ref: str) -> dict[str, Any]:
    for packet, source, read_only in _all_packets(items_dir):
        authoring = _authoring_item(packet, source=source, read_only=read_only)
        expected_ref = authoring["state_ref"] if kind == "state" else authoring["history_ref"]
        if expected_ref == ref:
            result = {
                "source": source,
                "read_only": read_only,
                "packet": packet,
            }
            if not read_only:
                for custom_packet, path in _custom_packets(items_dir):
                    if normalize_item(custom_packet)["item_id"] == authoring["item_id"]:
                        result["path"] = str(path)
                        break
            return result
    raise HTTPException(404, f"Unknown {kind} context: {ref}")


def _load_public_questionnaire(questionnaire_id: str) -> dict[str, Any]:
    try:
        return load_questionnaire(questionnaire_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc


def _questionnaire_leaf_metadata(schema: dict[str, Any]) -> list[dict[str, Any]]:
    fields: list[dict[str, Any]] = []

    def walk(nodes: Any, *, repeat_group_id: str | None = None) -> None:
        if not isinstance(nodes, list):
            return
        for node in nodes:
            if not isinstance(node, dict):
                continue
            structure_type = node.get("structure_type", "regular")
            if structure_type == "regular":
                field_id = node.get("id")
                if not field_id:
                    continue
                field = dict(node)
                field.setdefault("label", node.get("question_text") or field_id)
                if repeat_group_id:
                    field["repeat_group_id"] = repeat_group_id
                fields.append(field)
                continue
            if structure_type == "repeat_group":
                group_id = node.get("id") or repeat_group_id
                walk(node.get("fields") or node.get("questions") or [], repeat_group_id=group_id)
                continue
            if structure_type == "branch":
                branch = node.get("branch") or {}
                for route in branch.get("routes") or []:
                    walk(route.get("children") or [], repeat_group_id=repeat_group_id)
                walk(branch.get("default_children") or [], repeat_group_id=repeat_group_id)
                continue
            walk(node.get("fields") or node.get("questions") or [], repeat_group_id=repeat_group_id)

    walk(schema.get("questions") or [])
    return fields


def _title_from_item_id(item_id: str) -> str:
    return item_id.replace("_", " ").replace("-", " ").strip().title() or item_id


def _item_exists(items_dir: Path, item_id: str) -> bool:
    if item_id in {item["item_id"] for item in load_public_items()}:
        return True
    return any(normalize_item(packet)["item_id"] == item_id for packet, _ in _custom_packets(items_dir))


def _remove_empty_parents(path: Path, *, stop: Path) -> None:
    stop = stop.resolve()
    current = path.resolve()
    while current != stop and stop in current.parents:
        try:
            current.rmdir()
        except OSError:
            break
        current = current.parent


def _save_item(
    items_dir: Path,
    packet: dict[str, Any],
    *,
    allow_public_overwrite: bool,
) -> dict[str, Any]:
    item = _validate_packet(packet)
    if not allow_public_overwrite and item["item_id"] in {public["item_id"] for public in load_public_items()}:
        raise HTTPException(
            409,
            "Packaged public items are read-only. Use a different item_id for an editable copy.",
        )

    destination = items_dir / item["questionnaire_id"] / item["item_id"] / "ground_truth.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    normalized_packet = _public_packet(packet, item)
    destination.write_text(
        json.dumps(normalized_packet, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "saved": True,
        "path": str(destination),
        "item": _summary_for_packet(normalized_packet, source="studio", read_only=False),
    }


def _run_eval(config: StudioConfig, request: EvalRequest) -> dict[str, Any]:
    source = request.item_source.strip().lower()
    if source == "train":
        source = "dev"
    if request.item_ids:
        source = "all"
    if source not in {"public", "studio", "dev", "all"}:
        raise HTTPException(400, "item_source must be one of: public, studio, dev, all")

    items = _items_for_source(config, source)
    if request.item_ids:
        wanted = set(request.item_ids)
        items = [item for item in items if item["item_id"] in wanted]
    if not items:
        raise HTTPException(400, "No items matched this evaluation request.")

    model_reasoning_effort = _normalise_reasoning_effort(
        request.model_reasoning_effort,
        label="model_reasoning_effort",
    )
    evaluator_reasoning_effort = _normalise_reasoning_effort(
        request.evaluator_reasoning_effort,
        label="evaluator_reasoning_effort",
    )

    if request.model_id:
        for item in items:
            item["model_id"] = request.model_id
    if model_reasoning_effort:
        for item in items:
            item["model_reasoning_effort"] = model_reasoning_effort

    try:
        solver = benchmark.load_solver(request.solver)
    except Exception as exc:
        raise HTTPException(400, f"Could not import solver: {exc}") from exc

    run_id = request.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    _validate_id(run_id, label="run_id")
    output_dir = config.runs_dir / run_id
    if output_dir.exists():
        raise HTTPException(409, f"Run already exists: {run_id}")

    try:
        result = benchmark.run(
            items=deepcopy(items),
            solver=solver,
            output_dir=output_dir,
            run_id=run_id,
            workers=request.workers,
            evaluator_model_id=request.evaluator_model_id,
            evaluator_reasoning_effort=evaluator_reasoning_effort,
            score=True,
        )
    except Exception as exc:
        raise HTTPException(500, f"Evaluation failed: {exc}") from exc

    run_metadata = {
        "run_id": run_id,
        "solver": request.solver,
        "model_id": request.model_id,
        "model_reasoning_effort": model_reasoning_effort,
        "evaluator_model_id": request.evaluator_model_id,
        "evaluator_reasoning_effort": evaluator_reasoning_effort,
        "item_source": source,
        "item_ids": request.item_ids,
        "workers": request.workers,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(output_dir / "run_metadata.json", run_metadata)

    try:
        from scripts.generate_full_run_report import generate_report_for_run_dir

        csv_path, report_path, report_rows = generate_report_for_run_dir(output_dir)
    except Exception as exc:
        raise HTTPException(500, f"Report generation failed: {exc}") from exc

    exact, total = benchmark.exact_state_matches(result)
    rows = [
        {
            "item_id": item["item_id"],
            "questionnaire_id": item["questionnaire_id"],
            "operation_count": len(item.get("operations") or []),
            "exact_match": (
                item.get("candidate_state") == item.get("gold_resulting_state")
                if item.get("gold_resulting_state") is not None
                else None
            ),
            "evaluation_summary": item.get("evaluation_summary"),
        }
        for item in result.get("results", [])
    ]
    summary = {
        "run_id": run_id,
        "solver": request.solver,
        "model_id": request.model_id,
        "model_reasoning_effort": model_reasoning_effort,
        "evaluator_model_id": request.evaluator_model_id,
        "evaluator_reasoning_effort": evaluator_reasoning_effort,
        "item_source": source,
        "item_count": len(rows),
        "exact_matches": exact,
        "scored_items": total,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "csv_path": str(csv_path),
        "report_path": str(report_path),
        "report_rows": report_rows,
        "items": rows,
    }
    return summary


def _run_summary_from_artifacts(run_dir: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for turn_path in sorted(run_dir.rglob("turn_result.json")):
        item_dir = turn_path.parent
        try:
            turn_result = json.loads(turn_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        evaluation_path = item_dir / "evaluation.json"
        evaluation: dict[str, Any] = {}
        if evaluation_path.exists():
            try:
                evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                evaluation = {}
        summary = evaluation.get("summary") or {}
        exact_match = None
        if summary:
            exact_match = (
                summary.get("incorrect", 0) == 0
                and summary.get("partially_correct", 0) == 0
            )
        rows.append(
            {
                "item_id": turn_result.get("scenario_id") or item_dir.name,
                "questionnaire_id": turn_result.get("questionnaire", ""),
                "operation_count": len(
                    ((turn_result.get("agent_response") or {}).get("operations") or [])
                    if isinstance(turn_result.get("agent_response"), dict)
                    else []
                ),
                "exact_match": exact_match,
                "evaluation_summary": summary or None,
            }
        )

    exact_matches = sum(1 for row in rows if row.get("exact_match") is True)
    result = {
        "run_id": run_dir.name,
        "item_count": len(rows),
        "exact_matches": exact_matches,
        "scored_items": sum(1 for row in rows if row.get("evaluation_summary")),
        "output_dir": str(run_dir),
        "items": rows,
    }
    split_metadata = _read_optional_json(run_dir / "split_view_metadata.json")
    split_name = split_metadata.get("split")
    if not split_name:
        if run_dir.name.startswith("TEST144."):
            split_name = "test"
        elif run_dir.name.startswith(("DEV36.", "TRAIN36.")):
            split_name = "dev"
    if split_metadata.get("split_id"):
        result["split_id"] = split_metadata.get("split_id")
    if split_metadata.get("mode"):
        result["split_materialization"] = split_metadata.get("mode")
    if split_metadata.get("run_purpose"):
        result["run_purpose"] = split_metadata.get("run_purpose")
    if split_metadata.get("display_name"):
        result["display_name"] = split_metadata.get("display_name")
    run_metadata = _read_optional_json(run_dir / "run_metadata.json")
    if run_metadata:
        result.update({
            "solver": run_metadata.get("solver"),
            "model_id": run_metadata.get("model_id"),
            "model_reasoning_effort": run_metadata.get("model_reasoning_effort"),
            "evaluator_model_id": run_metadata.get("evaluator_model_id"),
            "evaluator_reasoning_effort": run_metadata.get("evaluator_reasoning_effort"),
            "run_purpose": run_metadata.get("run_purpose") or result.get("run_purpose"),
            "display_name": run_metadata.get("display_name") or result.get("display_name"),
        })
        if not split_name:
            split_name = run_metadata.get("split") or _split_from_items_dir(run_metadata.get("items_dir"))
        if not result.get("split_id") and run_metadata.get("split_id"):
            result["split_id"] = run_metadata.get("split_id")
    if split_name == "train":
        split_name = "dev"
    if split_name:
        result["split"] = split_name
    if not result.get("display_name") and result.get("run_purpose"):
        architecture = run_metadata.get("architecture") if run_metadata else None
        result["display_name"] = (
            f"{result['run_purpose']} · {architecture}"
            if architecture else result["run_purpose"]
        )
    summary_path = run_dir / "summary_report.json"
    report_path = run_dir / "report.md"
    metrics_path = run_dir / "metrics.json"
    figures_dir = run_dir / "figures"
    if report_path.exists():
        result["report_path"] = str(report_path)
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            views = ((summary.get("aggregate") or {}).get("metric_views") or {})
            all_fields = views.get("all_fields") or {}
            changed = views.get("changed_fields") or {}
            exact = views.get("whole_record_exact_match") or {}
            preservation = views.get("preservation") or {}
            result.update({
                "summary_report_path": str(summary_path),
                "metrics_path": str(metrics_path) if metrics_path.exists() else None,
                "figures_count": sum(
                    1 for filename in _FIGURE_ORDER if (figures_dir / filename).exists()
                ) if figures_dir.exists() else 0,
                "metrics": {
                    "all_field_accuracy": all_fields.get("accuracy"),
                    "all_field_lenient_accuracy": all_fields.get("lenient_accuracy"),
                    "changed_field_f1": (changed.get("strict") or {}).get("f1"),
                    "exact_match_rate": exact.get("exact_match_rate"),
                    "preservation_error_rate": preservation.get("preservation_error_rate"),
                },
            })
        except json.JSONDecodeError:
            result["summary_report_path"] = str(summary_path)
    return result


def _split_from_items_dir(items_dir: str | None) -> str | None:
    if not items_dir:
        return None
    parts = Path(items_dir).parts
    if not parts:
        return None
    split = parts[-1]
    if split == "train":
        return "dev"
    if split in {"dev", "test"}:
        return split
    return None


def _normalise_reasoning_effort(value: str | None, *, label: str) -> str | None:
    if value is None:
        return None
    cleaned = value.strip().lower()
    if not cleaned:
        return None
    if cleaned not in _REASONING_LEVELS:
        raise HTTPException(
            400,
            f"{label} must be one of: {', '.join(sorted(_REASONING_LEVELS))}",
        )
    if cleaned == "none":
        return None
    return cleaned


def _read_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _find_run_item_dir(run_dir: Path, item_id: str) -> Path | None:
    if not run_dir.exists():
        return None
    for turn_path in sorted(run_dir.rglob("turn_result.json")):
        try:
            turn_result = json.loads(turn_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if turn_result.get("scenario_id") == item_id:
            return turn_path.parent
    return None


def _items_for_source(config: StudioConfig, source: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if source in {"public", "all"}:
        items.extend(load_public_items())
    if source in {"dev", "train"}:
        items.extend(discover_items(config.dev_items_dir))
    if source in {"studio", "all"}:
        items.extend(discover_items(config.items_dir))
    return items


def _build_item_tree(items_dir: Path) -> dict[str, Any]:
    public_items = load_public_items()
    benchmark_items = discover_items(items_dir)

    pub_ids = {it["item_id"] for it in public_items}

    public_by_q: dict[str, list[str]] = {}
    for it in sorted(public_items, key=lambda x: x["item_id"]):
        public_by_q.setdefault(it["questionnaire_id"], []).append(it["item_id"])

    bench_by_q: dict[str, list[str]] = {}
    for it in sorted(benchmark_items, key=lambda x: x["item_id"]):
        if it["item_id"] not in pub_ids:
            bench_by_q.setdefault(it["questionnaire_id"], []).append(it["item_id"])

    return {
        "public": {q: ids for q, ids in sorted(public_by_q.items())},
        "benchmark": {q: ids for q, ids in sorted(bench_by_q.items())},
    }


def _custom_packets(items_dir: Path) -> list[tuple[dict[str, Any], Path]]:
    packets: list[tuple[dict[str, Any], Path]] = []
    if not items_dir.exists():
        return packets
    for path in sorted(items_dir.rglob("ground_truth.json")):
        try:
            packets.append((json.loads(path.read_text(encoding="utf-8")), path))
        except json.JSONDecodeError:
            continue
    return packets


def _summary_for_packet(packet: dict[str, Any], *, source: str, read_only: bool) -> dict[str, Any]:
    item = normalize_item(packet)
    return {
        "item_id": item["item_id"],
        "questionnaire_id": item["questionnaire_id"],
        "source": source,
        "read_only": read_only,
        "has_gold": item["gold_resulting_state"] is not None,
        "history_turns": len(item["visible_history"]),
        "utterance_preview": item["current_utterance"][:160],
    }


def _validate_packet(packet: dict[str, Any]) -> dict[str, Any]:
    try:
        item = normalize_item(packet)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    _validate_id(item["item_id"], label="item_id")
    if item["questionnaire_id"] not in list_questionnaires():
        raise HTTPException(400, f"Unknown questionnaire_id: {item['questionnaire_id']!r}")
    return item


def _public_packet(original: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
    packet = {
        "item_id": item["item_id"],
        "questionnaire_id": item["questionnaire_id"],
        "prior_state": item["prior_state"],
        "visible_history": item["visible_history"],
        "current_utterance": item["current_utterance"],
        "gold_resulting_state": item["gold_resulting_state"],
    }
    packet.update(item["metadata"])
    for key, value in original.items():
        if key not in packet and key != "metadata":
            packet[key] = value
    return packet


def _validate_id(value: str, *, label: str) -> None:
    if not _SAFE_ID_RE.match(str(value or "")):
        raise HTTPException(
            400,
            f"Invalid {label}: use letters, numbers, dots, underscores, or hyphens.",
        )


def run(
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    items_dir: str | Path | None = None,
    dev_items_dir: str | Path | None = None,
    train_items_dir: str | Path | None = None,
    runs_dir: str | Path | None = None,
) -> None:
    uvicorn.run(
        create_app(
            items_dir=items_dir,
            dev_items_dir=dev_items_dir,
            train_items_dir=train_items_dir,
            runs_dir=runs_dir,
        ),
        host=host,
        port=port,
    )


if __name__ == "__main__":
    run()
