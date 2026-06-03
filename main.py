"""CLI entry point for video anomaly analysis."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.config import load_config
from src.pipeline.processor import PipelineProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect temporal and spatial anomalies in long-term camera test videos."
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to YAML configuration file (default: config.yaml)",
    )
    parser.add_argument(
        "--video",
        default=None,
        help="Override video path from config",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config file not found: {config_path}", file=sys.stderr)
        return 1

    config = load_config(config_path)
    if args.video:
        config.input.video_path = args.video

    if not config.input.video_path:
        print(
            "No video path configured. Set input.video_path in config.yaml or use --video.",
            file=sys.stderr,
        )
        return 1

    processor = PipelineProcessor(config)
    try:
        run_id = processor.run()
    except Exception as exc:
        logging.exception("Pipeline failed")
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Analysis complete. Test run ID: {run_id}")
    print(f"Events stored in: {config.persistence.database_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
