"""Command line entrypoint for the public benchmark API."""

from __future__ import annotations

import argparse
from pathlib import Path

from . import benchmark
from .items import discover_items
from .public_items import list_public_item_ids
from .questionnaires import list_questionnaires


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="conformbench")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_q_parser = subparsers.add_parser(
        "list-questionnaires",
        help="List packaged public questionnaire ids.",
    )
    list_q_parser.set_defaults(handler=_cmd_list_questionnaires)

    list_items_parser = subparsers.add_parser(
        "list-public-items",
        help="List packaged public development item ids.",
    )
    list_items_parser.set_defaults(handler=_cmd_list_public_items)

    run_parser = subparsers.add_parser("run", help="Run a solver on public item packets.")
    run_parser.add_argument("--solver", required=True, help="Import path, e.g. my_agent:solve")
    run_parser.add_argument(
        "--items",
        default="public",
        help="Use 'public' or a directory containing ground_truth.json item packets",
    )
    run_parser.add_argument("--output-dir", type=Path)
    run_parser.add_argument("--run-id")
    run_parser.add_argument(
        "--evaluator-model",
        help="Optional judge model id for rich semantic fields.",
    )
    run_parser.add_argument(
        "--evaluator-reasoning",
        choices=["low", "medium", "high"],
        help="Optional judge reasoning effort for rich semantic fields.",
    )
    run_parser.add_argument(
        "--score",
        action="store_true",
        help="Also write per-item evaluation.json files during the run.",
    )
    run_parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel worker threads for solver/evaluator work.",
    )
    run_parser.set_defaults(handler=_cmd_run)

    score_parser = subparsers.add_parser(
        "score",
        help="Write evaluation.json files for an existing run directory.",
    )
    score_parser.add_argument("run_dir", type=Path)
    score_parser.add_argument(
        "--items-dir",
        type=Path,
        help="Directory containing benchmark ground_truth.json item packets.",
    )
    score_parser.add_argument(
        "--evaluator-model",
        help="Optional judge model id for rich semantic fields.",
    )
    score_parser.add_argument(
        "--evaluator-reasoning",
        choices=["low", "medium", "high"],
        help="Optional judge reasoning effort for rich semantic fields.",
    )
    score_parser.set_defaults(handler=_cmd_score)

    studio_parser = subparsers.add_parser("studio", help="Start the public item/eval Studio.")
    studio_parser.add_argument("--host", default="127.0.0.1")
    studio_parser.add_argument("--port", type=int, default=8000)
    studio_parser.add_argument("--items-dir", type=Path)
    studio_parser.add_argument("--runs-dir", type=Path)
    studio_parser.set_defaults(handler=_cmd_studio)

    args = parser.parse_args(argv)
    return args.handler(args)


def _cmd_list_questionnaires(_args: argparse.Namespace) -> int:
    for questionnaire_id in list_questionnaires():
        print(questionnaire_id)
    return 0


def _cmd_list_public_items(_args: argparse.Namespace) -> int:
    for item_id in list_public_item_ids():
        print(item_id)
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    solver = benchmark.load_solver(args.solver)
    if args.items == "public":
        result = benchmark.run_public(
            solver=solver,
            output_dir=args.output_dir,
            run_id=args.run_id,
            workers=args.workers,
            evaluator_model_id=args.evaluator_model,
            evaluator_reasoning_effort=args.evaluator_reasoning,
            score=args.score,
        )
    else:
        result = benchmark.run(
            items=discover_items(args.items),
            solver=solver,
            output_dir=args.output_dir,
            run_id=args.run_id,
            workers=args.workers,
            evaluator_model_id=args.evaluator_model,
            evaluator_reasoning_effort=args.evaluator_reasoning,
            score=args.score,
        )

    print(f"Ran {result['item_count']} item(s).")
    exact, total = benchmark.exact_state_matches(result)
    if total:
        print(f"Exact commitments: {exact}/{total}.")
        if args.score:
            scored = sum(1 for item in result.get("results", []) if item.get("evaluation_summary"))
            print(f"Field-level evaluations: {scored}/{total}.")
    return 0


def _cmd_score(args: argparse.Namespace) -> int:
    result = benchmark.score_run(
        args.run_dir,
        evaluator_model_id=args.evaluator_model,
        evaluator_reasoning_effort=args.evaluator_reasoning,
        items_dir=args.items_dir,
    )
    print(f"Scored {result['item_count']} item(s).")
    return 0


def _cmd_studio(args: argparse.Namespace) -> int:
    try:
        from .studio.server import run
    except ImportError as exc:
        raise SystemExit(str(exc)) from exc

    run(
        host=args.host,
        port=args.port,
        items_dir=args.items_dir,
        runs_dir=args.runs_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
