from __future__ import annotations

import numpy as np
import pandas as pd


def winsorize_df(df: pd.DataFrame, limits: tuple[float, float] = (0.01, 0.01)) -> pd.DataFrame:
    arr = df.to_numpy(dtype=np.float32, copy=True)
    arr = np.where(np.isfinite(arr), arr, np.nan)
    lower_limit, upper_limit = float(limits[0]), float(limits[1])
    if lower_limit > 0:
        lower = np.nanquantile(arr, lower_limit, axis=0).astype(np.float32, copy=False)
        arr = np.maximum(arr, lower)
    if upper_limit > 0:
        upper = np.nanquantile(arr, 1.0 - upper_limit, axis=0).astype(np.float32, copy=False)
        arr = np.minimum(arr, upper)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return pd.DataFrame(arr, index=df.index, columns=df.columns)


def zscore_df(df: pd.DataFrame, eps: float = 1e-8) -> pd.DataFrame:
    arr = df.to_numpy(dtype=np.float32, copy=False)
    mean = np.nanmean(arr, axis=0, dtype=np.float64).astype(np.float32, copy=False)
    std = np.nanstd(arr, axis=0, ddof=1, dtype=np.float64).astype(np.float32, copy=False)
    std = np.where((std == 0) | ~np.isfinite(std), np.float32(eps), std)
    out = (arr - mean) / std
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    return pd.DataFrame(out, index=df.index, columns=df.columns)
