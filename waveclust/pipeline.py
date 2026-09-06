from __future__ import annotations

import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from waveclust.config import CODE_DIR, WORKSPACE_DIR, load_config, resolve_workspace_path
from waveclust.data import load_price_panel, load_prices_from_directory, load_stock_basic, load_trade_calendar
from waveclust.model import StockWaveClust, WaveClustParams


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (Path,)):
        return str(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def build_params(cfg: dict[str, Any], *, use_gpu: bool) -> WaveClustParams:
    prep = cfg.get("preprocessing", {})
    wavelet = cfg.get("wavelet", {})
    clustering = cfg.get("clustering", {})
    performance = cfg.get("performance", {})
    gpu_ids = performance.get("gpu_ids")
    return WaveClustParams(
        wavelet_transform=str(wavelet.get("transform", "modwt")),
        wavelet=str(wavelet.get("wavelet", "db4")),
        levels=int(wavelet.get("levels", 6)),
        k=float(wavelet.get("k", 0.0)),
        q_threshold=float(wavelet.get("q_threshold", 0.82)),
        inflation=float(clustering.get("inflation", 1.4)),
        pruning_threshold=float(clustering.get("pruning_threshold", 0.05)),
        winsor_limits=tuple(prep.get("winsor_limits", [0.01, 0.01])) if prep.get("winsor_limits") else None,
        do_vol_zscore=bool(prep.get("do_vol_zscore", True)),
        rolling_window=prep.get("rolling_window"),
        min_stock_threshold_ratio=float(prep.get("min_stock_threshold_ratio", 0.05)),
        use_gpu=bool(use_gpu and clustering.get("use_gpu", True)),
        dtype=str(performance.get("dtype", "float32")),
        gpu_ids=tuple(int(item) for item in gpu_ids) if gpu_ids else None,
        multi_gpu_similarity=bool(performance.get("multi_gpu_similarity", False)),
        similarity_workers=int(performance.get("similarity_workers", 0) or 0),
        wavelet_workers=int(performance.get("wavelet_workers", 0) or 0),
        mcl_backend=str(performance.get("mcl_backend", "cpu")),
    )


def load_prices(cfg: dict[str, Any], *, start_date: str | None, end_date: str | None) -> pd.DataFrame:
    data = cfg.get("data", {})
    performance = cfg.get("performance", {})
    panel_path = resolve_workspace_path(data.get("panel_path", "DATA/stock_price_panel.csv"))
    prefer_panel = bool(data.get("prefer_panel", True))
    if prefer_panel and panel_path.exists():
        return load_price_panel(
            panel_path,
            start_date=start_date,
            end_date=end_date,
            dtype=str(performance.get("dtype", "float32")),
            prefer_parquet=bool(data.get("prefer_parquet", True)),
        )

    trade_dates = load_trade_calendar(
        resolve_workspace_path(data.get("trade_cal_path", "DATA/trade_cal.csv")),
        exchange=str(data.get("exchange", "SSE")),
    )
    return load_prices_from_directory(
        resolve_workspace_path(data.get("source_dir", "DATA/K")),
        trade_dates=trade_dates,
        start_date=start_date,
        end_date=end_date,
    )


def run_experiment(
    *,
    config_path: Path,
    output_dir: Path,
    start_date: str | None = None,
    end_date: str | None = None,
    use_gpu: bool = True,
) -> dict[str, Any]:
    started = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(config_path)
    params = build_params(cfg, use_gpu=use_gpu)
    prices = load_prices(cfg, start_date=start_date, end_date=end_date)
    stock_info = load_stock_basic(resolve_workspace_path(cfg.get("data", {}).get("stock_basic_path", "DATA/stock_basic.csv")))

    model = StockWaveClust(prices=prices, stock_info=stock_info, params=params).fit()
    metrics = model.metrics()
    row = {
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
        "seconds": float(time.time() - started),
    }

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
            "contract": "MODWT/SWT levels configurable in [1, 6]; ARI/NMI limited to current clustered stocks",
        },
    )
    print(pd.DataFrame([row]).to_string(index=False))
    print(f"saved metrics: {output_dir / 'metrics.csv'}")
    return row
