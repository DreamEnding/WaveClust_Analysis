from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import mutual_info_score


@dataclass(frozen=True)
class CrossBandDependence:
    spearman: pd.DataFrame
    mi: pd.DataFrame
    nmi: pd.DataFrame
    vi: pd.DataFrame


def rank_normalize_rows_for_spearman(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values)
    if matrix.ndim != 2:
        raise ValueError("Spearman coefficient input must be a two-dimensional matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("Spearman coefficient input must be finite")
    ranks = rankdata(matrix, axis=1, method="average").astype(np.float64, copy=False)
    ranks -= ranks.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(ranks, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (ranks / norms).astype(np.float32, copy=False)


def _equal_frequency_bins(values: np.ndarray, n_bins: int) -> np.ndarray:
    flat = np.asarray(values).reshape(-1)
    if int(n_bins) < 2 or int(n_bins) > flat.size:
        raise ValueError(f"n_bins must be in [2, {flat.size}], got {n_bins}")
    if not np.isfinite(flat).all():
        raise ValueError("dependence vectors must be finite")
    order = np.argsort(flat, kind="mergesort")
    labels = np.empty(flat.size, dtype=np.int32)
    labels[order] = np.minimum(
        np.arange(flat.size, dtype=np.int64) * int(n_bins) // flat.size,
        int(n_bins) - 1,
    ).astype(np.int32)
    return labels


def _pearson_of_ranks(left: np.ndarray, right: np.ndarray) -> float:
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    if denominator <= 0 or not np.isfinite(denominator):
        return float("nan")
    value = float(np.dot(left_centered, right_centered) / denominator)
    if abs(value - 1.0) <= 1e-15:
        return 1.0
    if abs(value + 1.0) <= 1e-15:
        return -1.0
    return value


def summarize_cross_band_dependence(
    similarity_matrices: list[np.ndarray],
    *,
    band_names: list[str],
    n_bins: int = 64,
) -> CrossBandDependence:
    if len(similarity_matrices) < 2:
        raise ValueError("at least two band similarity matrices are required")
    if len(similarity_matrices) != len(band_names):
        raise ValueError("band_names length must match similarity_matrices")
    matrices = [np.asarray(matrix, dtype=np.float32) for matrix in similarity_matrices]
    shape = matrices[0].shape
    if len(shape) != 2 or shape[0] != shape[1] or any(matrix.shape != shape for matrix in matrices[1:]):
        raise ValueError("band similarity matrices must have the same square shape")

    upper = np.triu_indices(shape[0], k=1)
    vectors = [matrix[upper].astype(np.float64, copy=False) for matrix in matrices]
    ranks = [rankdata(vector, method="average") for vector in vectors]
    bins = [_equal_frequency_bins(vector, int(n_bins)) for vector in vectors]
    band_count = len(matrices)
    spearman = np.eye(band_count, dtype=np.float64)
    mi = np.zeros((band_count, band_count), dtype=np.float64)
    nmi = np.eye(band_count, dtype=np.float64)
    vi = np.zeros((band_count, band_count), dtype=np.float64)
    entropies = np.array([mutual_info_score(labels, labels) for labels in bins], dtype=np.float64)
    np.fill_diagonal(mi, entropies)

    for left in range(band_count):
        for right in range(left + 1, band_count):
            rho = _pearson_of_ranks(ranks[left], ranks[right])
            mutual_information = float(mutual_info_score(bins[left], bins[right]))
            entropy_sum = float(entropies[left] + entropies[right])
            normalized = 0.0 if entropy_sum <= 0 else float(2.0 * mutual_information / entropy_sum)
            variation = float(entropy_sum - 2.0 * mutual_information)
            spearman[left, right] = spearman[right, left] = rho
            mi[left, right] = mi[right, left] = mutual_information
            nmi[left, right] = nmi[right, left] = normalized
            vi[left, right] = vi[right, left] = variation

    def frame(values: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(values, index=band_names, columns=band_names)

    return CrossBandDependence(
        spearman=frame(spearman),
        mi=frame(mi),
        nmi=frame(nmi),
        vi=frame(vi),
    )
