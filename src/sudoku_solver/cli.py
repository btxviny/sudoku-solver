"""Command-line entry point for the sudoku solver."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import PipelineConfig
from .pipeline import PIPELINE_PATHS, SudokuPipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sudoku-solver",
        description="End-to-end sudoku solver: detect grid, read digits, solve puzzle.",
    )
    parser.add_argument(
        "image", nargs="?", help="Path to sudoku image (omit with --list-paths)"
    )
    parser.add_argument(
        "--path",
        choices=[p.key for p in PIPELINE_PATHS],
        default=None,
        help="Model combination to use (default: best available)",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Inference device (default: auto)",
    )
    parser.add_argument(
        "--list-paths",
        action="store_true",
        help="List model combinations and whether their weights are present",
    )
    args = parser.parse_args(argv)
    if args.image is None and not args.list_paths:
        parser.error("the following arguments are required: image")

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    cfg = PipelineConfig(device=args.device)

    try:
        pipe = SudokuPipeline(cfg)

        if args.list_paths:
            for p in PIPELINE_PATHS:
                mark = "available" if pipe.path_available(p) else "missing weights"
                print(f"{p.key:26} [{mark}]  {p.label}")
            return 0

        available = pipe.available_paths()
        if not available:
            print("No pipeline path is available — check model weights.", file=sys.stderr)
            return 2

        if args.path:
            path = next(p for p in PIPELINE_PATHS if p.key == args.path)
            if not pipe.path_available(path):
                print(
                    f"Path '{path.key}' is missing weights. {path.hint}",
                    file=sys.stderr,
                )
                return 2
        else:
            path = available[0]

        print(f"Path: {path.label}")
        result = pipe.run_path(args.image, path)
        if result.errors:
            for stage, msg in result.errors.items():
                print(f"Error ({stage}): {msg}", file=sys.stderr)
            return 2
        SudokuPipeline.print_result(result)
        return 0
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except RuntimeError as e:
        print(f"Pipeline error: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 3


def cli():
    sys.exit(main())


if __name__ == "__main__":
    cli()
