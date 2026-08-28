from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine strict pure-price WaveClust result directories.")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--prefer", choices=["metrics", "summary"], default="metrics")
    parser.add_argument("--keep-duplicates", action="store_true")
    return parser.parse_args()


def read_manifest(result_dir: Path) -> dict[str, Any]:
    manifest_path = result_dir / "run_manifest.json"
    if not manifest_path.exists():
        return {}
    with manifest_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_result_frame(result_dir: Path, *, prefer: str) -> pd.DataFrame | None:
    names = ["metrics.csv", "summary.csv"] if prefer == "metrics" else ["summary.csv", "metrics.csv"]
    for name in names:
        path = result_dir / name
        if path.exists():
            frame = pd.read_csv(path)
            frame["_source_file"] = str(path)
            return frame
    return None


def combine_result_dirs(input_root: Path, *, prefer: str = "metrics", dedupe: bool = True) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for result_dir in sorted(path for path in Path(input_root).iterdir() if path.is_dir()):
        frame = read_result_frame(result_dir, prefer=prefer)
        if frame is None or frame.empty:
            continue
        manifest = read_manifest(result_dir)
        window_id = str(manifest.get("window_id") or result_dir.name)
        frame["_result_dir"] = str(result_dir)
        for column, default in (
            ("window_id", window_id),
            ("window_start", str(manifest.get("start_date") or "")),
            ("window_end", str(manifest.get("end_date") or "")),
        ):
            if column in frame.columns:
                frame[column] = frame[column].fillna(default).replace("", default)
            else:
                frame[column] = default
        if "trial_id" in frame.columns:
            frame["global_trial_id"] = frame["window_id"].astype(str) + "_" + frame["trial_id"].astype(str)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True, sort=False)
    if "SW1_ari" in combined.columns:
        combined["_SW1_ari_sort"] = pd.to_numeric(combined["SW1_ari"], errors="coerce")
    else:
        combined["_SW1_ari_sort"] = float("nan")
    if "status" in combined.columns:
        combined["_status_ok_sort"] = (combined["status"].astype(str) == "ok").astype(int)
    else:
        combined["_status_ok_sort"] = 0
    combined["_row_completeness_sort"] = combined.notna().sum(axis=1)
    combined = combined.sort_values(
        ["_status_ok_sort", "_SW1_ari_sort", "_row_completeness_sort"],
        ascending=[False, False, False],
    )
    if dedupe and "global_trial_id" in combined.columns:
        combined = combined.drop_duplicates(subset=["global_trial_id"], keep="first")
    combined = combined.drop(columns=["_SW1_ari_sort", "_status_ok_sort", "_row_completeness_sort"])
    if "SW1_ari" in combined.columns:
        combined["_SW1_ari_sort"] = pd.to_numeric(combined["SW1_ari"], errors="coerce")
        combined = combined.sort_values("_SW1_ari_sort", ascending=False).drop(columns=["_SW1_ari_sort"])
    return combined


def main() -> int:
    args = parse_args()
    combined = combine_result_dirs(args.input_root, prefer=args.prefer, dedupe=not args.keep_duplicates)
    output_csv = args.output_csv or args.input_root / "combined_all_metrics.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"saved {output_csv}")
    print(f"rows {len(combined)}")
    if not combined.empty and "SW1_ari" in combined.columns:
        print(combined.head(20).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
