from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse

try:
    import cupy as cp
except ImportError:
    cp = None


def normalize_columns_sparse(matrix: sparse.spmatrix) -> sparse.csr_matrix:
    matrix = matrix.tocsr(copy=False)
    col_sums = np.asarray(matrix.sum(axis=0)).ravel().astype(np.float32, copy=False)
    inv = np.zeros_like(col_sums, dtype=np.float32)
    mask = col_sums > 0
    inv[mask] = 1.0 / col_sums[mask]
    return (matrix @ sparse.diags(inv, dtype=np.float32, format="csr")).tocsr()


def prune_sparse(matrix: sparse.csr_matrix, threshold: float) -> sparse.csr_matrix:
    if threshold <= 0:
        return matrix
    csc = matrix.tocsc(copy=True)
    data_parts: list[np.ndarray] = []
    index_parts: list[np.ndarray] = []
    indptr = [0]
    for col in range(csc.shape[1]):
        start, end = csc.indptr[col], csc.indptr[col + 1]
        data = csc.data[start:end]
        rows = csc.indices[start:end]
        if data.size:
            keep = data >= threshold
            keep[int(np.argmax(data))] = True
            kept_data = data[keep].astype(np.float32, copy=False)
            kept_rows = rows[keep].astype(np.int32, copy=False)
            data_parts.append(kept_data)
            index_parts.append(kept_rows)
            indptr.append(indptr[-1] + int(kept_data.size))
        else:
            indptr.append(indptr[-1])
    data_out = np.concatenate(data_parts) if data_parts else np.array([], dtype=np.float32)
    idx_out = np.concatenate(index_parts) if index_parts else np.array([], dtype=np.int32)
    pruned = sparse.csc_matrix((data_out, idx_out, np.asarray(indptr, dtype=np.int32)), shape=csc.shape)
    pruned.eliminate_zeros()
    return pruned.tocsr()


def run_mcl_sparse(
    matrix: Any,
    *,
    expansion: int = 2,
    inflation: float = 2.0,
    loop_value: float = 1.0,
    iterations: int = 100,
    pruning_threshold: float = 0.001,
) -> sparse.csr_matrix:
    result = sparse.csr_matrix(np.asarray(matrix, dtype=np.float32))
    if result.shape[0] != result.shape[1]:
        raise ValueError("MCL matrix must be square")
    if loop_value > 0:
        result.setdiag(float(loop_value))
        result.eliminate_zeros()
    result = normalize_columns_sparse(result)
    for _ in range(int(iterations)):
        previous = result.copy()
        for _power in range(max(1, int(expansion)) - 1):
            result = (result @ result).tocsr()
            result.eliminate_zeros()
        result.data = np.power(result.data, float(inflation)).astype(np.float32, copy=False)
        result = normalize_columns_sparse(result)
        result = prune_sparse(result, float(pruning_threshold))
        if (result - previous).nnz == 0:
            break
    return result


def run_mcl_cuda(
    matrix: Any,
    *,
    expansion: int = 2,
    inflation: float = 2.0,
    loop_value: float = 1.0,
    iterations: int = 100,
    pruning_threshold: float = 0.001,
) -> np.ndarray:
    if cp is None:
        raise RuntimeError("CuPy is not available")
    gpu_matrix = cp.asarray(matrix, dtype=cp.float32)
    if gpu_matrix.shape[0] != gpu_matrix.shape[1]:
        raise ValueError("MCL matrix must be square")
    if loop_value > 0:
        cp.fill_diagonal(gpu_matrix, float(loop_value))
    col_sums = cp.sum(gpu_matrix, axis=0, keepdims=True)
    gpu_matrix = gpu_matrix / cp.where(col_sums == 0, 1.0, col_sums)
    for _ in range(int(iterations)):
        previous = gpu_matrix.copy()
        for _power in range(max(1, int(expansion)) - 1):
            gpu_matrix = gpu_matrix @ gpu_matrix
        gpu_matrix = cp.power(gpu_matrix, float(inflation))
        col_sums = cp.sum(gpu_matrix, axis=0, keepdims=True)
        gpu_matrix = gpu_matrix / cp.where(col_sums == 0, 1.0, col_sums)
        if pruning_threshold > 0:
            max_rows = cp.argmax(gpu_matrix, axis=0)
            cols = cp.arange(gpu_matrix.shape[1])
            max_values = gpu_matrix[max_rows, cols]
            gpu_matrix = cp.where(gpu_matrix >= float(pruning_threshold), gpu_matrix, 0)
            gpu_matrix[max_rows, cols] = max_values
        if bool(cp.allclose(gpu_matrix, previous)):
            break
    out = cp.asnumpy(gpu_matrix).astype(np.float32, copy=False)
    del gpu_matrix
    cp.get_default_memory_pool().free_all_blocks()
    return out


def run_mcl_auto(matrix: Any, *, use_gpu: bool = True, fallback_on_error: bool = True, **kwargs: Any) -> Any:
    if use_gpu and cp is not None:
        try:
            return run_mcl_cuda(matrix, **kwargs)
        except Exception:
            if not fallback_on_error:
                raise
            cp.get_default_memory_pool().free_all_blocks()
    return run_mcl_sparse(matrix, **kwargs)


def compact_labels(labels: np.ndarray) -> np.ndarray:
    mapping: dict[int, int] = {}
    out = np.empty(labels.shape[0], dtype=np.int32)
    for idx, label in enumerate(labels.tolist()):
        key = int(label)
        if key not in mapping:
            mapping[key] = len(mapping)
        out[idx] = mapping[key]
    return out


def labels_from_mcl_result(result: Any, *, eps: float = 1e-7) -> np.ndarray:
    if sparse.issparse(result):
        matrix = result.tocsr()
        n = matrix.shape[0]
        attractors = np.flatnonzero(matrix.diagonal() > eps)
        if attractors.size == 0:
            attractors = np.arange(n, dtype=np.int32)
        sub = matrix[attractors, :].tocsc()
        labels = np.full(n, -1, dtype=np.int32)
        for col in range(n):
            start, end = sub.indptr[col], sub.indptr[col + 1]
            if start == end:
                continue
            data = sub.data[start:end]
            best = int(np.argmax(data))
            if float(data[best]) > eps:
                labels[col] = int(sub.indices[start + best])
        missing = np.flatnonzero(labels < 0)
        for offset, col in enumerate(missing):
            labels[col] = int(attractors.size + offset)
        return compact_labels(labels)

    matrix = np.asarray(result)
    n = matrix.shape[0]
    attractors = np.flatnonzero(np.diag(matrix) > eps)
    if attractors.size == 0:
        attractors = np.arange(n, dtype=np.int32)
    sub = matrix[attractors, :]
    choices = np.argmax(sub, axis=0).astype(np.int32, copy=False)
    weights = sub[choices, np.arange(n)]
    labels = choices.copy()
    missing = np.flatnonzero(weights <= eps)
    for offset, col in enumerate(missing):
        labels[col] = int(attractors.size + offset)
    return compact_labels(labels)


def get_clusters(result: Any) -> tuple[tuple[int, ...], ...]:
    labels = labels_from_mcl_result(result)
    return tuple(tuple(np.flatnonzero(labels == label).astype(int).tolist()) for label in np.unique(labels))

