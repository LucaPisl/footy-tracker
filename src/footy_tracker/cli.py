"""CLI entrypoints for Footy Tracker."""

from __future__ import annotations

import argparse

from footy_tracker.pipeline.process_video import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Footy Tracker demo pipeline (frame extraction + stitching)."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the pipeline YAML config (e.g. configs/pipeline.yaml).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_pipeline(args.config)


__all__ = ["build_parser", "main"]
