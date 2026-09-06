from __future__ import annotations

import argparse
from pathlib import Path

from waveclust.pipeline import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one WaveClust experiment.")
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "config.yaml")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--no-gpu", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_experiment(
        config_path=args.config,
        output_dir=args.output_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        use_gpu=not args.no_gpu,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
