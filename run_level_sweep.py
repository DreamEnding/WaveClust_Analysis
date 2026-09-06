from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
from datetime import datetime
from pathlib import Path
import platform
import sys
import time

import pandas as pd
import yaml

from waveclust.config import CODE_DIR, WORKSPACE_DIR, load_config, resolve_workspace_path
from waveclust.data import load_stock_basic
from waveclust.model import StockWaveClust, WaveClustParams
from waveclust.pipeline import build_params, load_prices, write_json


DEFAULT_INFLATIONS = [round(1.2 + 0.2 * idx, 1) for idx in range(9)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full MODWT WaveClust experiments for level/inflation sweeps.")
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "config.yaml")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--levels", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    parser.add_argument(
        "--inflations",
        type=float,
        nargs="+",
        default=None,
        help="MCL inflation values. Defaults to sweep.inflations in config.yaml.",
    )
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--force", action="store_true", help="rerun experiments even when metrics.csv already exists")
    return parser.parse_args()


def format_float_for_path(value: float) -> str:
    return f"{value:.1f}".replace(".", "p")


def resolve_inflations(base_config: dict, requested: list[float] | None) -> list[float]:
    if requested is not None:
        return [float(value) for value in requested]
    configured = base_config.get("sweep", {}).get("inflations", DEFAULT_INFLATIONS)
    if not isinstance(configured, list) or not configured:
        raise ValueError("sweep.inflations must be a non-empty list of numeric values")
    return [float(value) for value in configured]


def write_sweep_config(base_config: dict, output_root: Path, level: int, inflation: float) -> Path:
    config = deepcopy(base_config)
    config.setdefault("wavelet", {})["levels"] = int(level)
    config.setdefault("wavelet", {})["transform"] = "modwt"
    config.setdefault("clustering", {})["inflation"] = float(inflation)
    inflation_name = format_float_for_path(float(inflation))
    config_path = output_root / f"config_level_{level}_inflation_{inflation_name}.yaml"
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return config_path


def build_result_row(
    *,
    output_dir: Path,
    params: WaveClustParams,
    model: StockWaveClust,
    prices: pd.DataFrame,
    setup_seconds: float,
    cluster_seconds: float,
) -> dict:
    metrics = model.metrics()
    return {
        "run_id": output_dir.name,
        "wavelet_transform": params.wavelet_transform,
        "wavelet": params.wavelet,
        "levels": params.levels,
        "k": params.k,
        "q_threshold": params.q_threshold,
        "inflation": params.inflation,
        "pruning_threshold": params.pruning_threshold,
        "winsor_limits": params.winsor_limits,
        "do_vol_zscore": params.do_vol_zscore,
        "rolling_window": params.rolling_window,
        "min_stock_threshold_ratio": params.min_stock_threshold_ratio,
        "dtype": params.dtype,
        "gpu_ids": params.gpu_ids,
        "multi_gpu_similarity": params.multi_gpu_similarity,
        "similarity_workers": params.similarity_workers,
        "wavelet_workers": params.wavelet_workers,
        "mcl_backend": params.mcl_backend,
        **metrics,
        "price_rows": int(prices.shape[0]),
        "price_cols": int(prices.shape[1]),
        "return_rows": int(model.returns.shape[0]) if model.returns is not None else 0,
        "return_cols": int(model.returns.shape[1]) if model.returns is not None else 0,
        "return_start": str(model.returns.index.min().date()) if model.returns is not None and len(model.returns.index) else "",
        "return_end": str(model.returns.index.max().date()) if model.returns is not None and len(model.returns.index) else "",
        "setup_seconds": float(setup_seconds),
        "cluster_seconds": float(cluster_seconds),
        "seconds": float(setup_seconds + cluster_seconds),
    }


def write_run_outputs(
    *,
    output_dir: Path,
    row: dict,
    model: StockWaveClust,
    config_path: Path,
    started: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(output_dir / "metrics.csv", index=False, encoding="utf-8-sig")
    write_json(output_dir / "metrics.json", row)
    model.cluster_assignments().to_csv(output_dir / "cluster_assignments.csv", index=False, encoding="utf-8-sig")
    write_json(
        output_dir / "run_manifest.json",
        {
            "run_id": output_dir.name,
            "started_at": datetime.fromtimestamp(started).astimezone().isoformat(),
            "finished_at": datetime.now().astimezone().isoformat(),
            "command": " ".join(sys.argv),
            "workspace_dir": WORKSPACE_DIR,
            "code_dir": CODE_DIR,
            "config_path": config_path,
            "python": sys.version,
            "platform": platform.platform(),
            "contract": "optimized pure-price MODWT level/inflation sweep; ARI/NMI limited to currently clustered stocks",
        },
    )


def existing_row(output_dir: Path) -> dict | None:
    metrics_path = output_dir / "metrics.csv"
    if not metrics_path.exists():
        return None
    return pd.read_csv(metrics_path).iloc[0].to_dict()


def main() -> int:
    args = parse_args()
    base_config = load_config(args.config)
    inflations = resolve_inflations(base_config, args.inflations)

    invalid = [level for level in args.levels if level < 2 or level > 6]
    if invalid:
        raise ValueError(f"levels must be in [2, 6], got {invalid}")
    invalid_inflations = [inflation for inflation in inflations if inflation < 1.2 or inflation > 2.8]
    if invalid_inflations:
        raise ValueError(f"inflations must be in [1.2, 2.8], got {invalid_inflations}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = []

    print("Loading price panel and stock metadata once for the whole sweep...")
    prices = load_prices(base_config, start_date=args.start_date, end_date=args.end_date)
    stock_info = load_stock_basic(resolve_workspace_path(base_config.get("data", {}).get("stock_basic_path", "DATA/stock_basic.csv")))

    for level in args.levels:
        pending: list[tuple[float, str, Path, Path]] = []
        for inflation in inflations:
            inflation_name = format_float_for_path(float(inflation))
            output_dir = args.output_root / f"level_{level}_inflation_{inflation_name}"
            level_config = write_sweep_config(base_config, args.output_root, int(level), float(inflation))
            if not args.force:
                row = existing_row(output_dir)
                if row is not None:
                    print(f"Skipping existing result: {output_dir / 'metrics.csv'}")
                    rows.append(row)
                    continue
            pending.append((float(inflation), inflation_name, output_dir, level_config))

        if not pending:
            continue

        print(f"\n========== MODWT level {level}: building shared price network once ==========")
        level_config_dict = deepcopy(base_config)
        level_config_dict.setdefault("wavelet", {})["levels"] = int(level)
        level_config_dict.setdefault("wavelet", {})["transform"] = "modwt"
        params = build_params(level_config_dict, use_gpu=not args.no_gpu)
        setup_started = time.time()
        model = StockWaveClust(prices=prices, stock_info=stock_info, params=params)
        model.preprocess()
        model.decompose_wavelets()
        model.build_interaction_network()
        setup_seconds = time.time() - setup_started
        print(f"    Shared setup seconds for level {level}: {setup_seconds:.3f}")

        for inflation, inflation_name, output_dir, level_config in pending:
            print(f"\n========== MODWT level {level}, inflation {inflation:.1f} ==========")
            started = time.time()
            current_params = replace(params, inflation=float(inflation))
            model.params = current_params
            cluster_started = time.time()
            model.cluster()
            cluster_seconds = time.time() - cluster_started
            row = build_result_row(
                output_dir=output_dir,
                params=current_params,
                model=model,
                prices=prices,
                setup_seconds=setup_seconds,
                cluster_seconds=cluster_seconds,
            )
            write_run_outputs(output_dir=output_dir, row=row, model=model, config_path=level_config, started=started)
            print(pd.DataFrame([row]).to_string(index=False))
            print(f"saved metrics: {output_dir / 'metrics.csv'}")
            rows.append(row)

    summary = pd.DataFrame(rows)
    if not summary.empty and "ari" in summary.columns:
        summary = summary.sort_values(["ari", "nmi"], ascending=[False, False])
    summary_path = args.output_root / "summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print("\n========== Sweep summary ==========")
    print(summary.to_string(index=False))
    print(f"saved summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
