"""Entry point for AMD Hackathon Track 1 agent."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app.agent import Agent
from app.config import get_settings
from app.utils.io import read_tasks, write_results
from app.utils.logger import get_logger, log_event, setup_logging

logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AMD Hackathon Track 1 — General Purpose AI Agent",
    )
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Process tasks and write results")
    run_parser.add_argument(
        "--input",
        default=None,
        help="Path to tasks.json (default: from config)",
    )
    run_parser.add_argument(
        "--output",
        default=None,
        help="Path to results.json (default: from config)",
    )

    bench_parser = subparsers.add_parser("benchmark", help="Run tasks with detailed metrics")
    bench_parser.add_argument(
        "--input",
        default=None,
        help="Path to tasks.json (default: from config)",
    )
    bench_parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write benchmark report JSON",
    )

    return parser


def cmd_run(input_path: str | None, output_path: str | None) -> int:
    settings = get_settings()
    input_file = input_path or settings.input_path
    output_file = output_path or settings.output_path

    log_event(logger, "run_start", input=input_file, output=output_file)

    tasks = read_tasks(input_file)
    agent = Agent()
    results = agent.process_tasks(tasks)
    write_results(output_file, results)

    log_event(logger, "run_complete", tasks=len(tasks), results_written=len(results))

    return 0


def cmd_benchmark(input_path: str | None, output_path: str | None) -> int:
    settings = get_settings()
    input_file = input_path or settings.input_path

    log_event(logger, "benchmark_start", input=input_file)

    tasks = read_tasks(input_file)
    agent = Agent()
    report = agent.benchmark(tasks)

    payload = report.model_dump()
    print(json.dumps(payload, indent=2, default=str))

    if output_path:
        Path(output_path).write_text(
            json.dumps(payload, indent=2, default=str),
            encoding="utf-8",
        )

    log_event(
        logger,
        "benchmark_complete",
        total_tasks=report.total_tasks,
        total_tokens=report.total_tokens,
        estimated_cost_usd=report.estimated_cost_usd,
    )

    return 0


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "benchmark":
        return cmd_benchmark(args.input, args.output)
    if args.command == "run" or args.command is None:
        return cmd_run(
            getattr(args, "input", None),
            getattr(args, "output", None),
        )

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
