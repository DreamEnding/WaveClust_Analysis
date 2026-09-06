from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import pywt

if "CUDA_PATH" not in os.environ and os.path.isdir("/usr/local/cuda"):
    os.environ["CUDA_PATH"] = "/usr/local/cuda"
if "CUDA_HOME" not in os.environ and os.path.isdir("/usr/local/cuda"):
    os.environ["CUDA_HOME"] = "/usr/local/cuda"

from waveclust.data import normalize_stock_code
from waveclust.evaluation import compute_industry_metrics
from waveclust.mcl import get_clusters, run_mcl_auto
from waveclust.preprocessing import winsorize_df, zscore_df

try:
    import cupy as cp
except ImportError:
    cp = None


@dataclass(frozen=True)
class WaveClustParams:
    wavelet_transform: str = "modwt"
    wavelet: str = "db4"
    levels: int = 6
    k: float = 0.0
    q_threshold: float = 0.82
    inflation: float = 1.4
    pruning_threshold: float = 0.05
    winsor_limits: tuple[float, float] | None = (0.01, 0.01)
    do_vol_zscore: bool = True
    rolling_window: int | None = None
    min_stock_threshold_ratio: float = 0.05
    use_gpu: bool = True
    dtype: str = "float32"
    gpu_ids: tuple[int, ...] | None = None
    multi_gpu_similarity: bool = False
    similarity_workers: int = 0
    wavelet_workers: int = 0
    mcl_backend: str = "cpu"


class StockWaveClust:
    def __init__(
        self,
        prices: pd.DataFrame,
        stock_info: pd.DataFrame | None,
        params: WaveClustParams,
    ) -> None:
        if int(params.levels) < 1 or int(params.levels) > 6:
            raise ValueError(f"wavelet levels must be in [1, 6], got {params.levels}")
        if params.wavelet_transform.strip().lower() not in {"modwt", "swt", "dwt", "wavedec"}:
            raise ValueError(f"unsupported wavelet transform: {params.wavelet_transform}; use modwt or dwt")
        self.prices = prices
        self.stock_info = stock_info
        self.params = params
        self.gpu_ids = self._resolve_gpu_ids(params.gpu_ids)
        self.use_gpu = bool(params.use_gpu and cp is not None and self.gpu_ids)

        self.returns: pd.DataFrame | None = None
        self.coefficients: dict[str, dict[int, np.ndarray]] = {}
        self.adjacency: pd.DataFrame | None = None
        self.clusters: dict[int, list[str]] | None = None

    @staticmethod
    def _resolve_gpu_ids(requested: tuple[int, ...] | None) -> tuple[int, ...]:
        if cp is None:
            return ()
        try:
            count = int(cp.cuda.runtime.getDeviceCount())
        except Exception:
            return ()
        if requested is None:
            return tuple(range(count))
        return tuple(int(device_id) for device_id in requested if 0 <= int(device_id) < count)

    def preprocess(self) -> None:
        print(">>> [Step 1] Computing log returns and robust preprocessing...")
        prices_filled = self.prices.ffill()
        returns = np.log(prices_filled / prices_filled.shift(1))
        returns = returns.replace([np.inf, -np.inf], np.nan)

        valid_counts = returns.notna().sum(axis=1)
        if not valid_counts.empty:
            max_concurrent = int(valid_counts.max())
            threshold = max(2, int(max_concurrent * self.params.min_stock_threshold_ratio))
            keep = valid_counts >= threshold
            if keep.any():
                start = valid_counts[keep].index[0]
                print(f"    Sparse leading rows removed; analysis starts at {start}")
                returns = returns.loc[start:]

        returns = returns.dropna(how="all", axis=0).dropna(how="all", axis=1).fillna(0)
        if self.params.winsor_limits is not None:
            returns = winsorize_df(returns, limits=self.params.winsor_limits)
        if self.params.do_vol_zscore:
            returns = zscore_df(returns)
        returns = returns.replace([np.inf, -np.inf], np.nan).fillna(0)
        returns = returns.loc[:, returns.std(axis=0) > 1e-6]
        if self.params.rolling_window is not None and int(self.params.rolling_window) > 10:
            returns = returns.tail(int(self.params.rolling_window))
        if returns.empty or returns.shape[1] < 2:
            raise RuntimeError("not enough effective stocks or observations for WaveClust")
        self.returns = returns.astype(self.params.dtype, copy=False)
        print(f"    Effective return matrix: {self.returns.shape}")

    def decompose_wavelets(self) -> None:
        if self.returns is None:
            raise RuntimeError("preprocess must run before wavelet decomposition")
        transform = self.params.wavelet_transform.strip().lower()
        print(
            f">>> [Step 2] {transform.upper()} decomposition "
            f"(wavelet={self.params.wavelet}, levels={self.params.levels})..."
        )
        stock_names = [str(stock) for stock in self.returns.columns]
        values = self.returns.to_numpy(dtype=np.float32, copy=False)

        def decompose_one(item: tuple[int, str]) -> tuple[str, dict[int, np.ndarray]]:
            col_idx, stock = item
            signal = np.asarray(values[:, col_idx], dtype=np.float64)
            coeffs = self._wavelet_coefficients(signal)
            return stock, {idx: np.asarray(coef, dtype=np.float32) for idx, coef in enumerate(coeffs)}

        workers = self.params.wavelet_workers or min(16, max(1, (os.cpu_count() or 1) // 2))
        if workers > 1 and len(stock_names) > 64:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                self.coefficients = {stock: coeffs for stock, coeffs in executor.map(decompose_one, enumerate(stock_names))}
        else:
            self.coefficients = dict(decompose_one(item) for item in enumerate(stock_names))

    def _modwt_coefficients(self, signal: np.ndarray) -> list[np.ndarray]:
        block = 2 ** int(self.params.levels)
        original_len = int(signal.shape[0])
        remainder = original_len % block
        pad_width = 0 if remainder == 0 else block - remainder
        if pad_width:
            padded = np.pad(signal, (0, pad_width), mode="symmetric")
        else:
            padded = signal
        coeffs = pywt.swt(
            padded,
            self.params.wavelet,
            level=int(self.params.levels),
            trim_approx=True,
            norm=True,
        )
        return [np.asarray(coef[:original_len], dtype=np.float32) for coef in coeffs]

    def _dwt_coefficients(self, signal: np.ndarray) -> list[np.ndarray]:
        coeffs = pywt.wavedec(signal, self.params.wavelet, level=int(self.params.levels))
        return [np.asarray(coef, dtype=np.float32) for coef in coeffs]

    def _wavelet_coefficients(self, signal: np.ndarray) -> list[np.ndarray]:
        transform = self.params.wavelet_transform.strip().lower()
        if transform in {"modwt", "swt"}:
            return self._modwt_coefficients(signal)
        return self._dwt_coefficients(signal)

    def prepare_level_matrices(self) -> tuple[list[np.ndarray], list[str]]:
        if self.returns is None:
            raise RuntimeError("returns are not available")
        stock_names = [str(stock) for stock in self.returns.columns]
        band_count = int(self.params.levels) + 1
        min_lengths = [min(len(self.coefficients[stock][level]) for stock in stock_names) for level in range(band_count)]
        matrices = []
        for level in range(band_count):
            matrix = np.empty((len(stock_names), min_lengths[level]), dtype=np.float32)
            for row_idx, stock in enumerate(stock_names):
                matrix[row_idx, :] = self.coefficients[stock][level][: min_lengths[level]]
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            matrices.append((matrix / norms).astype(np.float32, copy=False))
        return matrices, stock_names

    def release_wavelet_coefficients(self) -> None:
        """Release per-stock coefficient storage after level matrices are materialized.

        This keeps ``prepare_level_matrices`` backward compatible for callers that
        inspect coefficients, while allowing bounded batch runners to lower their
        peak host-memory use before dense similarity construction.
        """
        self.coefficients.clear()

    def compute_similarity_matrix(self, matrix: np.ndarray, *, device_id: int | None = None) -> np.ndarray:
        if self.use_gpu and cp is not None:
            try:
                target_device = self.gpu_ids[0] if device_id is None else int(device_id)
                with cp.cuda.Device(target_device):
                    gpu_matrix = cp.asarray(matrix, dtype=cp.float32)
                    similarity = gpu_matrix @ gpu_matrix.T
                    cp.fill_diagonal(similarity, 0.0)
                    out = cp.asnumpy(similarity).astype(np.float32, copy=False)
                    del gpu_matrix, similarity
                    cp.get_default_memory_pool().free_all_blocks()
                    return out
            except cp.cuda.memory.OutOfMemoryError:
                if device_id is not None:
                    with cp.cuda.Device(int(device_id)):
                        cp.get_default_memory_pool().free_all_blocks()
                else:
                    cp.get_default_memory_pool().free_all_blocks()
                print("    GPU memory exhausted; falling back to CPU for this similarity matrix")
        similarity = matrix @ matrix.T
        np.fill_diagonal(similarity, 0.0)
        return similarity.astype(np.float32, copy=False)

    def compute_similarity_matrices(self, level_matrices: list[np.ndarray]) -> list[np.ndarray]:
        if not self.use_gpu:
            return [self.compute_similarity_matrix(matrix) for matrix in level_matrices]
        if not self.params.multi_gpu_similarity or len(self.gpu_ids) <= 1 or len(level_matrices) <= 1:
            return [self.compute_similarity_matrix(matrix, device_id=self.gpu_ids[0]) for matrix in level_matrices]

        workers = self.params.similarity_workers or len(self.gpu_ids)
        workers = min(max(1, int(workers)), len(self.gpu_ids), len(level_matrices))
        print(f"    Using GPUs {list(self.gpu_ids[:workers])} for similarity matrices")
        results: list[np.ndarray | None] = [None] * len(level_matrices)
        try:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {}
                for level, matrix in enumerate(level_matrices):
                    device_id = self.gpu_ids[level % workers]
                    futures[executor.submit(self.compute_similarity_matrix, matrix, device_id=device_id)] = level
                for future in as_completed(futures):
                    results[futures[future]] = future.result()
            return [matrix for matrix in results if matrix is not None]
        except Exception as exc:
            print(f"    Multi-GPU similarity failed ({exc}); falling back to single-GPU path")
            return [self.compute_similarity_matrix(matrix, device_id=self.gpu_ids[0]) for matrix in level_matrices]

    def build_interaction_network(self) -> None:
        print(">>> [Step 3] Building multiscale interaction network...")
        level_matrices, stock_names = self.prepare_level_matrices()
        sim_mats = self.compute_similarity_matrices(level_matrices)
        all_sims = np.concatenate([sim.ravel() for sim in sim_mats])
        non_zero = all_sims[all_sims != 0]
        if non_zero.size == 0:
            self.adjacency = pd.DataFrame(np.zeros((len(stock_names), len(stock_names))), index=stock_names, columns=stock_names)
            return
        threshold = max(float(np.quantile(all_sims, self.params.q_threshold)), 0.0)
        print(
            f"    Similarity non-zero min={float(non_zero.min()):.4f}, "
            f"max={float(non_zero.max()):.4f}, mean={float(non_zero.mean()):.4f}"
        )
        print(f"    Threshold q={self.params.q_threshold}: {threshold:.4f}")

        score = np.zeros_like(sim_mats[0], dtype=np.float32)
        low = sim_mats[0]
        for high_level in range(1, len(sim_mats)):
            high = sim_mats[high_level]
            mask = (low > threshold) & (high > threshold)
            if not mask.any():
                continue
            rows, cols = np.nonzero(mask)
            candidate = np.sqrt(low[rows, cols] * high[rows, cols]).astype(np.float32, copy=False)
            candidate *= np.float32(self.params.k * high_level + 1.0)
            score[rows, cols] = np.maximum(score[rows, cols], candidate)
        np.fill_diagonal(score, 0.0)
        score = np.maximum(score, score.T)
        self.adjacency = pd.DataFrame(score, index=stock_names, columns=stock_names)
        print(f"    Effective undirected edges: {int(np.count_nonzero(score) // 2)}")

    def cluster(self) -> None:
        if self.adjacency is None:
            raise RuntimeError("build_interaction_network must run before cluster")
        print(
            ">>> [Step 4] MCL clustering "
            f"(inflation={self.params.inflation}, pruning={self.params.pruning_threshold})..."
        )
        mcl_backend = self.params.mcl_backend.strip().lower()
        use_gpu_mcl = self.use_gpu if mcl_backend == "auto" else mcl_backend == "gpu"
        matrix = self.adjacency.values.astype(np.float32, copy=False)
        if use_gpu_mcl and cp is not None and self.gpu_ids:
            with cp.cuda.Device(self.gpu_ids[0]):
                result = run_mcl_auto(
                    matrix,
                    inflation=self.params.inflation,
                    pruning_threshold=self.params.pruning_threshold,
                    use_gpu=True,
                )
        else:
            result = run_mcl_auto(
                matrix,
                inflation=self.params.inflation,
                pruning_threshold=self.params.pruning_threshold,
                use_gpu=False,
            )
        clusters = get_clusters(result)
        stock_names = list(self.adjacency.index)
        self.clusters = {}
        for cluster_id, indices in enumerate(clusters):
            members = [stock_names[idx] for idx in indices]
            if len(members) > 1:
                strength = self.adjacency.loc[members, members].sum(axis=1)
                members = strength.sort_values(ascending=False).index.tolist()
            self.clusters[int(cluster_id)] = members
        singletons = sum(1 for members in self.clusters.values() if len(members) == 1)
        print(f"    Found {len(self.clusters)} communities; singletons={singletons}")

    def fit(self) -> "StockWaveClust":
        self.preprocess()
        self.decompose_wavelets()
        self.build_interaction_network()
        self.cluster()
        return self

    @property
    def current_stocks(self) -> list[str]:
        if self.adjacency is not None:
            return [normalize_stock_code(stock) for stock in self.adjacency.index]
        if self.returns is not None:
            return [normalize_stock_code(stock) for stock in self.returns.columns]
        return [normalize_stock_code(stock) for stock in self.prices.columns]

    def cluster_assignments(self) -> pd.DataFrame:
        rows = []
        if self.clusters:
            for cluster_id, members in self.clusters.items():
                for stock in members:
                    rows.append({"stock": normalize_stock_code(stock), "cluster_id": int(cluster_id)})
        return pd.DataFrame(rows, columns=["stock", "cluster_id"])

    def metrics(self) -> dict[str, Any]:
        out = compute_industry_metrics(self, current_stocks=self.current_stocks)
        out.update(
            {
                "n_clusters": int(len(self.clusters) if self.clusters else 0),
                "n_singletons": int(sum(1 for members in self.clusters.values() if len(members) == 1)) if self.clusters else 0,
                "n_edges": int(np.count_nonzero(self.adjacency.values) // 2) if self.adjacency is not None else 0,
                "n_stocks": int(len(self.current_stocks)),
            }
        )
        return out

    @property
    def adj_matrix(self) -> pd.DataFrame | None:
        return self.adjacency

    @property
    def log_returns(self) -> pd.DataFrame | None:
        return self.returns
