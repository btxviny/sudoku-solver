import argparse
import sys

from .pipeline import SudokuPipeline, PipelineResult


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sudoku-solver",
        description="End-to-end sudoku solver: detect grid, classify digits, solve puzzle.",
    )
    parser.add_argument("image", help="Path to sudoku image")
    parser.add_argument(
        "--maskrcnn",
        default="models/weights/maskrcnn_sudoku_20250912_115605.pth",
        help="Path to Mask R-CNN weights",
    )
    parser.add_argument(
        "--xgboost",
        default="models/weights/xgboost_digit_classifier.model",
        help="Path to XGBoost digit classifier",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Grid detection confidence threshold",
    )
    args = parser.parse_args(argv)

    from pathlib import Path
    from .config import PipelineConfig, GridDetectorConfig, DigitClassifierConfig

    cfg = PipelineConfig(
        grid_detector=GridDetectorConfig(
            model_path=Path(args.maskrcnn),
            detection_threshold=args.threshold,
        ),
        digit_classifier=DigitClassifierConfig(model_path=Path(args.xgboost)),
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