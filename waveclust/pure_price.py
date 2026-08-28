from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from waveclust.model import StockWaveClust, WaveClustParams
from waveclust.preprocessing import winsorize_df, zscore_df


def filter_prices_by_window_eligibility(
    prices: pd.DataFrame,
    *,
    min_price_coverage: float = 0.0,
    max_flat_return_rate: float | None = None,
    min_stocks: int = 2,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if prices.empty:
        raise RuntimeError("price panel is empty")
    n_rows = int(len(prices.index))
    if n_rows < 2:
        raise RuntimeError("at least two price rows are required")

    coverage = prices.notna().sum(axis=0).astype(np.float64) / float(n_rows)
    keep = coverage >= float(min_price_coverage)

    flat_rate = pd.Series(np.nan, index=prices.columns, dtype=np.float64)
    if max_flat_return_rate is not None:
        observed = prices.notna()
        valid_pairs = observed & observed.shift(1).fillna(False)
        moves = prices.astype(np.float64).diff().abs()
        flat_pairs = valid_pairs & (moves <= 1e-12)
        pair_counts = valid_pairs.sum(axis=0).replace(0, np.nan)
        flat_rate = (flat_pairs.sum(axis=0).astype(np.float64) / pair_counts).fillna(1.0)
        keep &= flat_rate <= float(max_flat_return_rate)

    selected = [column for column in prices.columns if bool(keep.loc[column])]
    if len(selected) < int(min_stocks):
        raise RuntimeError(
            "not enough eligible stocks after price coverage/suspension filtering: "
            f"{len(selected)} < {int(min_stocks)}"
        )

    stats = {
        "price_filter_enabled": bool(float(min_price_coverage) > 0 or max_flat_return_rate is not None),
        "price_filter_min_coverage": float(min_price_coverage),
        "price_filter_max_flat_return_rate": (
            np.nan if max_flat_return_rate is None else float(max_flat_return_rate)
        ),
        "price_filter_min_stocks": int(min_stocks),
        "price_filter_input_stock_count": int(prices.shape[1]),
        "eligible_stock_count": int(len(selected)),
        "coverage_median": float(coverage.median()) if len(coverage) else np.nan,
        "coverage_min_selected": float(coverage.loc[selected].min()) if selected else np.nan,
        "flat_return_rate_median": float(flat_rate.median()) if max_flat_return_rate is not None else np.nan,
        "flat_return_rate_max_selected": (
            float(flat_rate.loc[selected].max()) if max_flat_return_rate is not None and selected else np.nan
        ),
    }
    return prices.loc[:, selected].copy(), stats


def preprocess_returns_no_metadata(
    prices: pd.DataFrame,
    *,
    winsor_limit: float,
    do_vol_zscore: bool,
    return_mode: str = "raw",
    dtype: str = "float32",
) -> pd.DataFrame:
    prices_filled = prices.ffill()
    returns = np.log(prices_filled / prices_filled.shift(1))
    returns = returns.dropna(how="all", axis=0).dropna(how="all", axis=1).fillna(0)
    returns = returns.replace([np.inf, -np.inf], np.nan).fillna(0)
    returns = apply_return_mode(returns, mode=return_mode)
    returns = winsorize_df(returns, limits=(float(winsor_limit), float(winsor_limit)))
    if do_vol_zscore:
        returns = zscore_df(returns)
    returns = returns.replace([np.inf, -np.inf], np.nan).fillna(0)
    returns = returns.loc[:, returns.std(axis=0) > 1e-6]
    if returns.empty or returns.shape[1] < 2:
        raise RuntimeError("not enough effective stocks or observations for pure-price WaveClust")
    return returns.astype(dtype, copy=False)


def apply_return_mode(returns: pd.DataFrame, *, mode: str) -> pd.DataFrame:
    mode = str(mode).strip().lower()
    if mode in {"raw", "none", ""}:
        return returns
    market = returns.mean(axis=1).astype(np.float64)
    if mode in {"market_demean", "demean", "cross_section_demean"}:
        return returns.sub(market, axis=0)
    if mode in {"market_residual", "demarket", "ols_residual"}:
        market_values = market.to_numpy(dtype=np.float64)
        centered_market = market_values - market_values.mean()
        denom = float(centered_market @ centered_market)
        if not np.isfinite(denom) or denom <= 1e-12:
            return returns
        values = returns.to_numpy(dtype=np.float64, copy=False)
        centered_values = values - values.mean(axis=0, keepdims=True)
        beta = (centered_market[:, None] * centered_values).sum(axis=0) / denom
        alpha = values.mean(axis=0) - beta * market_values.mean()
        residual = values - alpha[None, :] - market_values[:, None] * beta[None, :]
        return pd.DataFrame(residual, index=returns.index, columns=returns.columns)
    raise ValueError(f"unknown return_mode: {mode}")


def build_similarity_mats_no_metadata(
    returns: pd.DataFrame,
    *,
    params: WaveClustParams,
) -> tuple[list[np.ndarray], np.ndarray, list[str]]:
    model = StockWaveClust(prices=returns, stock_info=None, params=params)
    model.returns = returns.astype(params.dtype, copy=False)
    model.decompose_wavelets()
    level_mats, stock_names = model.prepare_level_matrices()
    sim_mats = [matrix.astype(np.float32, copy=False) for matrix in model.compute_similarity_matrices(level_mats)]
    all_sims = np.concatenate([sim.ravel() for sim in sim_mats]).astype(np.float32, copy=False)
    return sim_mats, all_sims, stock_names


def build_adjacency_no_metadata(
    sim_mats: list[np.ndarray],
    all_sims: np.ndarray,
    *,
    q_threshold: float,
    k: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    n = sim_mats[0].shape[0]
    threshold = max(float(np.quantile(all_sims, float(q_threshold))), 0.0)
    best_score = np.zeros((n, n), dtype=np.float32)
    sim_low = sim_mats[0]
    for high_level in range(1, len(sim_mats)):
        sim_high = sim_mats[high_level]
        mask = (sim_low > threshold) & (sim_high > threshold)
        if not mask.any():
            continue
        rows, cols = np.nonzero(mask)
        raw = np.sqrt(sim_low[rows, cols] * sim_high[rows, cols]).astype(np.float32, copy=False)
        score = raw * np.float32(float(k) * high_level + 1.0)
        best_score[rows, cols] = np.maximum(best_score[rows, cols], score)
    np.fill_diagonal(best_score, 0.0)
    best_score = np.maximum(best_score, best_score.T)
    n_edges = int(np.count_nonzero(best_score) // 2)
    possible_edges = n * (n - 1) / 2
    return best_score, {
        "n_edges": n_edges,
        "edge_density": float(n_edges / possible_edges) if possible_edges else 0.0,
        "n_stocks": int(n),
    }
