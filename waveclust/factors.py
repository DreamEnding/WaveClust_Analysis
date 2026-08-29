from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from waveclust.data import normalize_stock_code
from waveclust.preprocessing import winsorize_df, zscore_df


@dataclass(frozen=True)
class FactorBranchResult:
    returns: dict[str, pd.DataFrame]
    per_stock_diagnostics: dict[str, pd.DataFrame]
    summary: dict[str, Any]


@dataclass(frozen=True)
class FactorUniverse:
    prices: pd.DataFrame
    industries: pd.Series
    exclusions: pd.DataFrame


def select_factor_common_universe(
    prices: pd.DataFrame,
    stock_info: pd.DataFrame,
    *,
    min_industry_size: int = 2,
) -> FactorUniverse:
    if "industry" not in stock_info.columns:
        raise ValueError("stock_info must contain an industry column")
    if int(min_industry_size) < 2:
        raise ValueError("min_industry_size must be at least two")

    industry_by_code: dict[str, str] = {}
    for row in stock_info.itertuples(index=False):
        raw_industry = getattr(row, "industry", None)
        industry = "" if pd.isna(raw_industry) else str(raw_industry).strip()
        if industry.upper() in {"", "NAN", "NONE", "UNK", "SKIP", "UNKNOWN"}:
            continue
        for column in ("ts_code", "symbol"):
            code = normalize_stock_code(getattr(row, column, ""))
            if not code:
                continue
            previous = industry_by_code.get(code)
            if previous is not None and previous != industry:
                raise ValueError(f"conflicting industry labels for stock {code}: {previous!r} vs {industry!r}")
            industry_by_code[code] = industry

    price_codes = [normalize_stock_code(column) for column in prices.columns]
    labels = pd.Series([industry_by_code.get(code, "") for code in price_codes], index=prices.columns, dtype="string")
    valid_labels = labels[labels != ""]
    group_sizes = valid_labels.value_counts()

    selected_columns = [
        column
        for column in prices.columns
        if labels.loc[column] != "" and int(group_sizes.loc[labels.loc[column]]) >= int(min_industry_size)
    ]
    if len(selected_columns) < 2:
        raise ValueError("factor_common_universe contains fewer than two stocks")

    exclusion_rows = []
    for column, code in zip(prices.columns, price_codes, strict=True):
        industry = str(labels.loc[column])
        if not industry:
            exclusion_rows.append({"stock": code, "industry": "", "reason": "missing_industry"})
        elif int(group_sizes.loc[industry]) < int(min_industry_size):
            exclusion_rows.append({"stock": code, "industry": industry, "reason": "industry_size_lt_2"})

    selected_industries = labels.loc[selected_columns].astype(str)
    selected_industries.name = "industry"
    return FactorUniverse(
        prices=prices.loc[:, selected_columns].copy(),
        industries=selected_industries,
        exclusions=pd.DataFrame(exclusion_rows, columns=["stock", "industry", "reason"]),
    )


def _validate_industries(columns: pd.Index, industries: pd.Series) -> pd.Series:
    normalized = industries.copy()
    normalized.index = normalized.index.astype(str)
    expected = pd.Index([str(column) for column in columns])
    missing = expected.difference(normalized.index)
    if not missing.empty:
        raise ValueError(f"industry labels do not cover all stocks: {missing.tolist()[:5]}")
    labels = normalized.reindex(expected).astype("string").str.strip()
    invalid = labels.isna() | labels.str.upper().isin({"", "NAN", "NONE", "UNK", "SKIP", "UNKNOWN"})
    if invalid.any():
        raise ValueError(f"invalid industry labels for stocks: {expected[invalid.to_numpy()].tolist()[:5]}")
    counts = labels.value_counts()
    too_small = counts[counts < 2]
    if not too_small.empty:
        raise ValueError(f"industry groups must contain at least two stocks: {too_small.to_dict()}")
    labels.index = columns
    return labels.astype(str)


def _leave_one_out_market_regression(
    returns: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    values = returns.to_numpy(dtype=np.float64, copy=False)
    stock_count = int(values.shape[1])
    if stock_count < 2:
        raise ValueError("leave-one-out market regression requires at least two stocks")

    factors = (values.sum(axis=1, keepdims=True) - values) / float(stock_count - 1)
    factor_mean = factors.mean(axis=0)
    return_mean = values.mean(axis=0)
    centered_factors = factors - factor_mean
    centered_returns = values - return_mean
    denominator = np.einsum("ti,ti->i", centered_factors, centered_factors)
    if np.any(~np.isfinite(denominator)) or np.any(denominator <= 1e-20):
        raise ValueError("leave-one-out market factor has zero or non-finite variance")

    beta = np.einsum("ti,ti->i", centered_factors, centered_returns) / denominator
    alpha = return_mean - beta * factor_mean
    residual = values - alpha - factors * beta
    residual_inner = np.einsum("ti,ti->i", residual, factors)
    residual_mean = residual.mean(axis=0)
    residual_ss = np.einsum("ti,ti->i", residual, residual)
    total_ss = np.einsum("ti,ti->i", centered_returns, centered_returns)
    r_squared = np.where(total_ss > 1e-20, 1.0 - residual_ss / total_ss, 0.0)
    return_variance = np.var(values, axis=0)
    residual_variance = np.var(residual, axis=0)
    explained_variance = np.where(
        return_variance > 1e-20,
        1.0 - residual_variance / return_variance,
        0.0,
    )

    diagnostics = pd.DataFrame(
        {
            "alpha": alpha,
            "beta": beta,
            "r_squared": r_squared,
            "explained_variance": explained_variance,
            "residual_mean": residual_mean,
            "residual_factor_inner_product": residual_inner,
        },
        index=returns.columns,
    )
    if not np.isfinite(diagnostics.to_numpy(dtype=np.float64)).all():
        raise ValueError("market regression produced non-finite diagnostics")
    return pd.DataFrame(residual, index=returns.index, columns=returns.columns), diagnostics


def neutralize_by_industry_strict(
    returns: pd.DataFrame,
    industries: pd.Series,
) -> tuple[pd.DataFrame, dict[str, float]]:
    labels = _validate_industries(returns.columns, industries)
    values = returns.to_numpy(dtype=np.float64, copy=True)
    for industry in labels.unique():
        positions = np.flatnonzero(labels.to_numpy() == industry)
        values[:, positions] -= values[:, positions].mean(axis=1, keepdims=True)

    residual = pd.DataFrame(values.astype(np.float32), index=returns.index, columns=returns.columns)
    group_means = residual.T.groupby(labels).mean().T
    max_abs_group_mean = float(np.max(np.abs(group_means.to_numpy(dtype=np.float64))))
    return residual, {
        "industry_count": float(labels.nunique()),
        "industry_mean_max_abs": max_abs_group_mean,
    }


def prepare_factor_branches(
    prices: pd.DataFrame,
    industries: pd.Series,
    *,
    winsor_limit: float,
    dtype: str = "float32",
) -> FactorBranchResult:
    labels = _validate_industries(prices.columns, industries)
    prices_filled = prices.ffill()
    log_returns = np.log(prices_filled / prices_filled.shift(1))
    log_returns = log_returns.replace([np.inf, -np.inf], np.nan)
    log_returns = log_returns.dropna(how="all", axis=0).dropna(how="all", axis=1).fillna(0.0)
    if log_returns.columns.tolist() != prices.columns.tolist():
        raise ValueError("factor branches must preserve the frozen stock universe and column order")

    winsorized = winsorize_df(
        log_returns,
        limits=(float(winsor_limit), float(winsor_limit)),
    )
    raw = zscore_df(winsorized).astype(dtype)

    market_residual, market_diagnostics = _leave_one_out_market_regression(winsorized)
    market_standardized = zscore_df(market_residual).astype(dtype)

    sequential, industry_summary = neutralize_by_industry_strict(market_standardized, labels)
    sequential = sequential.astype(dtype)
    _sequential_residual, sequential_market_diagnostics = _leave_one_out_market_regression(sequential)

    return FactorBranchResult(
        returns={
            "raw": raw,
            "market_residual_internal": market_standardized,
            "sequential_market_industry_adjusted": sequential,
        },
        per_stock_diagnostics={
            "market_residual_internal": market_diagnostics,
            "sequential_market_industry_adjusted": sequential_market_diagnostics,
        },
        summary={
            "n_observations": int(raw.shape[0]),
            "n_stocks": int(raw.shape[1]),
            **industry_summary,
        },
    )
