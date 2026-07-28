import argparse
import sys
from pathlib import Path

from .pipeline import SudokuPipeline
from .config import PipelineConfig, GridDetectorConfig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sudoku-solver",
        description="End-to-end sudoku solver: detect grid, read digits, solve puzzle.",
    )
    parser.add_argument("image", help="Path to sudoku image")
    parser.add_argument(
        "--maskrcnn",
        default=None,
        help="Path to Mask R-CNN weights (overrides config default)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Grid detection confidence threshold",
    )
    args = parser.parse_args(argv)

    cfg = PipelineConfig()
    if args.maskrcnn:
        cfg.grid_detector = GridDetectorConfig(
            model_path=Path(args.maskrcnn),
            detection_threshold=args.threshold,
        )

    try:
        pipe = SudokuPipeline(cfg)
        result = pipe.run(args.image)
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
