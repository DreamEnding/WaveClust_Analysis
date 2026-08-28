from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

DATE_ALIASES = {"date", "time", "datetime", "trade_date", "日期", "时间"}
CODE_ALIASES = {"code", "symbol", "ts_code", "股票代码", "证券代码"}
CLOSE_ALIASES = {"close", "price", "收盘", "收盘价"}


def normalize_stock_code(value: object) -> str:
    code = str(value).strip().upper()
    if not code or code == "NAN":
        return ""
    code = code.split(".", 1)[0]
    return code.zfill(6) if code.isdigit() else code


def find_column(columns: pd.Index, aliases: set[str], default: object | None = None) -> object | None:
    for column in columns:
        if str(column).strip().lower() in aliases:
            return column
    return default


def load_trade_calendar(cal_path: Path, exchange: str = "SSE") -> pd.DatetimeIndex:
    if not cal_path.exists():
        raise FileNotFoundError(f"trade calendar not found: {cal_path}")
    cal = pd.read_csv(cal_path, dtype={"exchange": str, "cal_date": str, "is_open": int})
    cal = cal[(cal["exchange"] == exchange) & (cal["is_open"] == 1)]
    dates = pd.to_datetime(cal["cal_date"], format="%Y%m%d", errors="coerce")
    return pd.DatetimeIndex(dates.dropna().sort_values().unique())


def align_series_to_calendar(series: pd.Series, trade_dates: pd.DatetimeIndex) -> pd.Series:
    series = series.sort_index()
    if series.empty:
        return series
    dates = trade_dates[(trade_dates >= series.index.min()) & (trade_dates <= series.index.max())]
    return series.reindex(dates).ffill()


def read_stock_series(
    path: Path,
    *,
    trade_dates: pd.DatetimeIndex | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.Series | None:
    encodings = ("utf-8", "utf-8-sig", "gbk", "gb18030")
    last_error: Exception | None = None
    for encoding in encodings:
        try:
            header = pd.read_csv(path, nrows=0, encoding=encoding)
            date_col = find_column(header.columns, DATE_ALIASES, default=header.columns[0])
            close_col = find_column(header.columns, CLOSE_ALIASES)
            code_col = find_column(header.columns, CODE_ALIASES)
            if close_col is None:
                return None
            usecols = [date_col, close_col]
            if code_col is not None and code_col not in usecols:
                usecols.append(code_col)
            dtype = {code_col: str} if code_col is not None else None
            data = pd.read_csv(path, usecols=usecols, encoding=encoding, dtype=dtype)
            break
        except Exception as exc:
            last_error = exc
    else:
        raise RuntimeError(f"failed to read {path}: {last_error}")

    dates = pd.to_datetime(data[date_col].astype(str), errors="coerce")
    values = pd.to_numeric(data[close_col], errors="coerce")
    valid = dates.notna() & values.notna() & np.isfinite(values) & (values > 0)
    if not valid.any():
        return None

    code = path.stem.split("_")[0]
    if code_col is not None:
        codes = data.loc[valid, code_col].astype(str).str.strip()
        codes = codes[codes != ""]
        if not codes.empty:
            code = codes.iloc[0]

    series = pd.Series(values.loc[valid].to_numpy(dtype=np.float64), index=dates.loc[valid])
    series = series[~series.index.duplicated(keep="last")].sort_index()
    if start_date:
        series = series.loc[series.index >= pd.Timestamp(start_date)]
    if end_date:
        series = series.loc[series.index <= pd.Timestamp(end_date)]
    if trade_dates is not None:
        series = align_series_to_calendar(series, trade_dates)
    series.name = normalize_stock_code(code)
    return series if series.size > 1 else None


def load_prices_from_directory(
    source_dir: Path,
    *,
    trade_dates: pd.DatetimeIndex | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    if not source_dir.exists():
        raise FileNotFoundError(f"source directory not found: {source_dir}")
    files = sorted(path for path in source_dir.iterdir() if path.is_file() and path.suffix.lower() == ".csv")
    if limit is not None and limit > 0:
        files = files[:limit]
    if not files:
        raise FileNotFoundError(f"no stock CSV files found in {source_dir}")

    def read_one(path: Path) -> pd.Series | None:
        return read_stock_series(path, trade_dates=trade_dates, start_date=start_date, end_date=end_date)

    with ThreadPoolExecutor() as executor:
        series_list = [series for series in executor.map(read_one, files) if series is not None]
    if not series_list:
        raise RuntimeError(f"no valid stock series found in {source_dir}")
    prices = pd.concat(series_list, axis=1).sort_index()
    prices = prices.loc[:, ~prices.columns.duplicated(keep="last")]
    return prices


def load_price_panel(
    panel_path: Path,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    dtype: str = "float32",
    prefer_parquet: bool = True,
) -> pd.DataFrame:
    if not panel_path.exists():
        raise FileNotFoundError(f"price panel not found: {panel_path}")
    read_path = panel_path
    if prefer_parquet and panel_path.suffix.lower() == ".csv":
        parquet_path = panel_path.with_suffix(".parquet")
        if parquet_path.exists():
            read_path = parquet_path

    if read_path.suffix.lower() == ".parquet":
        try:
            prices = pd.read_parquet(read_path)
        except ImportError as exc:
            raise ImportError(
                "reading parquet price panels requires pyarrow or fastparquet; "
                "install pyarrow or set data.prefer_parquet=false"
            ) from exc
        if not isinstance(prices.index, pd.DatetimeIndex):
            prices.index = pd.to_datetime(prices.index, errors="coerce")
    else:
        prices = pd.read_csv(read_path, index_col=0, parse_dates=True, dtype=dtype)
    prices.index = pd.DatetimeIndex(prices.index)
    prices = prices.sort_index()
    if start_date:
        prices = prices.loc[prices.index >= pd.Timestamp(start_date)]
    if end_date:
        prices = prices.loc[prices.index <= pd.Timestamp(end_date)]
    prices.columns = [normalize_stock_code(column) for column in prices.columns]
    prices = prices.dropna(how="all", axis=0).dropna(how="all", axis=1)
    return prices.astype(dtype, copy=False)


def load_stock_basic(stock_basic_path: Path) -> pd.DataFrame:
    if not stock_basic_path.exists():
        raise FileNotFoundError(f"stock_basic not found: {stock_basic_path}")
    return pd.read_csv(stock_basic_path, dtype={"symbol": str, "ts_code": str})
