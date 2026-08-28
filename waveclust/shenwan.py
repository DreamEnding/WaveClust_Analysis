from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import LabelEncoder

from waveclust.data import normalize_stock_code


SW_LEVELS = ("SW1", "SW2", "SW3")
VALID_LEVELS = set(SW_LEVELS)


def _load_universe_rows(universe_json: Path) -> list[dict[str, Any]]:
    with Path(universe_json).open("r", encoding="utf-8-sig") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"expected a list in {universe_json}, got {type(data).__name__}")
    return data


def parse_shenwan_label(row: dict[str, Any], level: str) -> str:
    desc = str(row.get("description", "")).strip()
    if ":" in desc:
        return desc.split(":", 1)[1].strip()
    if "：" in desc:
        return desc.split("：", 1)[1].strip()
    name = str(row.get("name", "")).strip()
    if name.startswith(level):
        return name[len(level) :].strip()
    return name


def infer_shenwan_level(row: dict[str, Any]) -> str:
    explicit = str(row.get("class", "")).upper()
    if explicit in VALID_LEVELS:
        return explicit
    universe_id = str(row.get("id", "")).upper()
    for level in SW_LEVELS:
        if f"_{level}_" in universe_id:
            return level
    name = str(row.get("name", "")).upper()
    for level in SW_LEVELS:
        if name.startswith(level):
            return level
    return ""


def load_shenwan_label_table(universe_json: str | Path, level: str = "SW1") -> pd.DataFrame:
    level = level.upper()
    if level not in VALID_LEVELS:
        raise ValueError(f"unsupported Shenwan level: {level}. Expected one of {sorted(VALID_LEVELS)}")

    rows = _load_universe_rows(Path(universe_json))
    by_stock: dict[str, dict[str, Any]] = {}
    for row in rows:
        if infer_shenwan_level(row) != level:
            continue
        if str(row.get("category", "")).lower() != "equity":
            continue
        label = parse_shenwan_label(row, level)
        if not label:
            continue
        universe_id = str(row.get("id", "")).strip()
        for full_symbol in row.get("symbols", []) or []:
            stock = normalize_stock_code(full_symbol)
            if not stock:
                continue
            item = by_stock.setdefault(
                stock,
                {
                    "stock": stock,
                    "sw_level": level,
                    "full_symbols": set(),
                    "labels": set(),
                    "source_universe_ids": set(),
                },
            )
            item["full_symbols"].add(str(full_symbol).strip().upper())
            item["labels"].add(label)
            if universe_id:
                item["source_universe_ids"].add(universe_id)

    out_rows = []
    for stock, item in sorted(by_stock.items()):
        labels = sorted(item["labels"])
        is_conflict = len(labels) > 1
        out_rows.append(
            {
                "stock": stock,
                "sw_level": level,
                "sw_label": np.nan if is_conflict else labels[0],
                "is_conflict": bool(is_conflict),
                "conflict_labels": "|".join(labels) if is_conflict else "",
                "full_symbols": "|".join(sorted(item["full_symbols"])),
                "source_universe_ids": "|".join(sorted(item["source_universe_ids"])),
                "n_source_universes": int(len(item["source_universe_ids"])),
            }
        )
    return pd.DataFrame(
        out_rows,
        columns=[
            "stock",
            "sw_level",
            "sw_label",
            "is_conflict",
            "conflict_labels",
            "full_symbols",
            "source_universe_ids",
            "n_source_universes",
        ],
    )


def align_clusters_to_shenwan(assignments: pd.DataFrame, label_table: pd.DataFrame) -> pd.DataFrame:
    required = {"stock", "cluster_id"}
    missing = required.difference(assignments.columns)
    if missing:
        raise ValueError(f"cluster assignments missing columns: {sorted(missing)}")
    normalized = assignments.copy()
    normalized["stock"] = normalized["stock"].map(normalize_stock_code)
    normalized = normalized[normalized["stock"] != ""].copy()

    labels = label_table.copy()
    labels["stock"] = labels["stock"].map(normalize_stock_code)
    aligned = normalized.merge(
        labels[
            [
                "stock",
                "sw_level",
                "sw_label",
                "is_conflict",
                "conflict_labels",
                "full_symbols",
                "source_universe_ids",
                "n_source_universes",
            ]
        ],
        how="left",
        on="stock",
    )
    aligned["is_conflict"] = aligned["is_conflict"].fillna(False).astype(bool)
    aligned["is_labeled"] = aligned["sw_label"].notna() & (~aligned["is_conflict"])
    return aligned


def compute_shenwan_metrics(
    assignments: pd.DataFrame,
    label_table: pd.DataFrame,
    level: str = "SW1",
) -> tuple[dict[str, Any], pd.DataFrame]:
    level = level.upper()
    aligned = align_clusters_to_shenwan(assignments, label_table)
    used = aligned[aligned["is_labeled"]].copy()
    n_assigned = int(len(aligned))
    n_conflict = int(aligned["is_conflict"].sum())
    n_unlabeled = int((~aligned["is_labeled"] & ~aligned["is_conflict"]).sum())
    n_used = int(len(used))
    metrics: dict[str, Any] = {
        "sw_level": level,
        "ari": np.nan,
        "nmi": np.nan,
        "n_assigned": n_assigned,
        "n_used": n_used,
        "n_unlabeled": n_unlabeled,
        "n_conflict": n_conflict,
        "coverage": float(n_used / n_assigned) if n_assigned else np.nan,
        "n_labels": int(used["sw_label"].nunique()) if n_used else 0,
        "n_clusters_used": int(used["cluster_id"].nunique()) if n_used else 0,
    }
    if n_used == 0:
        return metrics, aligned
    y_true = LabelEncoder().fit_transform(used["sw_label"].astype(str))
    y_pred = used["cluster_id"].astype(str).to_numpy()
    metrics["ari"] = float(adjusted_rand_score(y_true, y_pred))
    metrics["nmi"] = float(normalized_mutual_info_score(y_true, y_pred, average_method="arithmetic"))
    return metrics, aligned


def finite_mean(values: list[Any]) -> float:
    nums = [float(value) for value in values if value is not None and np.isfinite(float(value))]
    return float(np.mean(nums)) if nums else float("nan")


def evaluate_assignments(assignments: pd.DataFrame, label_tables: dict[str, pd.DataFrame]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    ari_values = []
    nmi_values = []
    for level in SW_LEVELS:
        metrics, _aligned = compute_shenwan_metrics(assignments, label_tables[level], level=level)
        out[f"{level}_ari"] = metrics["ari"]
        out[f"{level}_nmi"] = metrics["nmi"]
        out[f"{level}_n_used"] = metrics["n_used"]
        out[f"{level}_n_unlabeled"] = metrics["n_unlabeled"]
        out[f"{level}_n_conflict"] = metrics["n_conflict"]
        out[f"{level}_n_labels"] = metrics["n_labels"]
        out[f"{level}_n_clusters_used"] = metrics["n_clusters_used"]
        ari_values.append(metrics["ari"])
        nmi_values.append(metrics["nmi"])
    out["mean_ari"] = finite_mean(ari_values)
    out["mean_nmi"] = finite_mean(nmi_values)
    return out

