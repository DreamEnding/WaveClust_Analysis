from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import LabelEncoder

from waveclust.data import normalize_stock_code


def normalize_universe(values: Any) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        code = normalize_stock_code(value)
        if code and code not in seen:
            out.append(code)
            seen.add(code)
    return out


def current_model_stocks(model: Any, current_stocks: Any | None = None) -> list[str]:
    if current_stocks is not None:
        return normalize_universe(current_stocks)
    if getattr(model, "adjacency", None) is not None:
        return normalize_universe(model.adjacency.index)
    if getattr(model, "returns", None) is not None:
        return normalize_universe(model.returns.columns)
    if getattr(model, "prices", None) is not None:
        return normalize_universe(model.prices.columns)
    return []


def cluster_label_map(model: Any) -> dict[str, int]:
    labels: dict[str, int] = {}
    if getattr(model, "clusters", None) is None:
        return labels
    for cluster_id, members in model.clusters.items():
        for stock in members:
            code = normalize_stock_code(stock)
            if code:
                labels[code] = int(cluster_id)
    return labels


def industry_label_map(stock_info: pd.DataFrame, industry_col: str = "industry") -> dict[str, str]:
    if stock_info is None or industry_col not in stock_info.columns:
        return {}
    labels: dict[str, str] = {}
    for _, row in stock_info.iterrows():
        industry = str(row.get(industry_col, "")).strip()
        if not industry or industry.lower() == "nan":
            continue
        for key in (row.get("symbol", ""), row.get("ts_code", "")):
            code = normalize_stock_code(key)
            if code:
                labels[code] = industry
    return labels


def build_evaluation_frame(
    model: Any,
    *,
    current_stocks: Any | None = None,
    industry_col: str = "industry",
) -> pd.DataFrame:
    cluster_labels = cluster_label_map(model)
    industry_labels = industry_label_map(getattr(model, "stock_info", None), industry_col)
    rows = []
    for stock in current_model_stocks(model, current_stocks):
        rows.append(
            {
                "stock": stock,
                "cluster_id": cluster_labels.get(stock),
                "industry": industry_labels.get(stock),
            }
        )
    frame = pd.DataFrame(rows, columns=["stock", "cluster_id", "industry"])
    if frame.empty:
        return frame
    frame["is_used"] = frame["cluster_id"].notna() & frame["industry"].notna()
    return frame


def compute_industry_metrics(
    model: Any,
    *,
    current_stocks: Any | None = None,
    industry_col: str = "industry",
) -> dict[str, Any]:
    frame = build_evaluation_frame(model, current_stocks=current_stocks, industry_col=industry_col)
    n_current = int(len(frame))
    if n_current == 0:
        return {
            "ari": np.nan,
            "nmi": np.nan,
            "n_current_stocks": 0,
            "n_used": 0,
            "n_unlabeled": 0,
            "n_clusters_used": 0,
            "n_labels": 0,
            "coverage": np.nan,
        }
    used = frame[frame["is_used"]].copy()
    n_used = int(len(used))
    metrics: dict[str, Any] = {
        "ari": np.nan,
        "nmi": np.nan,
        "n_current_stocks": n_current,
        "n_used": n_used,
        "n_unlabeled": int(n_current - n_used),
        "n_clusters_used": int(used["cluster_id"].nunique()) if n_used else 0,
        "n_labels": int(used["industry"].nunique()) if n_used else 0,
        "coverage": float(n_used / n_current),
    }
    if n_used == 0:
        return metrics
    y_true = LabelEncoder().fit_transform(used["industry"].astype(str))
    y_pred = used["cluster_id"].astype(str).to_numpy()
    metrics["ari"] = float(adjusted_rand_score(y_true, y_pred))
    metrics["nmi"] = float(normalized_mutual_info_score(y_true, y_pred, average_method="arithmetic"))
    return metrics


def compute_ari(model: Any, current_stocks: Any | None = None) -> float:
    return float(compute_industry_metrics(model, current_stocks=current_stocks)["ari"])


def compute_nmi(model: Any, current_stocks: Any | None = None) -> float:
    return float(compute_industry_metrics(model, current_stocks=current_stocks)["nmi"])

