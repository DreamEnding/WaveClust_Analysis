from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, triu
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans, SpectralClustering
from threadpoolctl import threadpool_limits

from waveclust.data import normalize_stock_code
from waveclust.mcl import get_clusters, run_mcl_auto


def labels_to_assignments(labels: np.ndarray, stock_names: list[str]) -> pd.DataFrame:
    labels = np.asarray(labels)
    if labels.shape[0] != len(stock_names):
        raise ValueError("labels length must match stock_names")
    return pd.DataFrame(
        {
            "stock": [normalize_stock_code(stock) for stock in stock_names],
            "cluster_id": labels.astype(str),
        }
    )


def labels_to_clusters(labels: np.ndarray, stock_names: list[str]) -> dict[int, list[str]]:
    labels = np.asarray(labels)
    if labels.shape[0] != len(stock_names):
        raise ValueError("labels length must match stock_names")
    clusters: dict[int, list[str]] = {}
    for raw_label, stock in zip(labels, stock_names, strict=True):
        cluster_id = int(raw_label)
        clusters.setdefault(cluster_id, []).append(str(stock))
    return {cluster_id: members for cluster_id, members in sorted(clusters.items())}


def cluster_stats(labels: np.ndarray, *, n_edges: int, edge_density: float, n_stocks: int) -> dict[str, Any]:
    _, counts = np.unique(np.asarray(labels).astype(str), return_counts=True)
    return {
        "n_stocks": int(n_stocks),
        "n_edges": int(n_edges),
        "edge_density": float(edge_density),
        "n_communities": int(len(counts)),
        "n_singletons": int((counts == 1).sum()),
        "largest_community": int(counts.max()) if len(counts) else 0,
        "median_community_size": float(np.median(counts)) if len(counts) else 0.0,
    }


def build_dense_waveclust_score(sim_mats: list[np.ndarray], *, k: float) -> np.ndarray:
    if len(sim_mats) < 2:
        raise ValueError("at least two WaveClust similarity matrices are required")
    sim_low = np.maximum(np.asarray(sim_mats[0], dtype=np.float32), 0.0)
    score = np.zeros_like(sim_low, dtype=np.float32)
    for high_level in range(1, len(sim_mats)):
        sim_high = np.maximum(np.asarray(sim_mats[high_level], dtype=np.float32), 0.0)
        raw = np.sqrt(sim_low * sim_high)
        score = np.maximum(score, raw * (float(k) * high_level + 1.0))
    np.fill_diagonal(score, 0.0)
    return np.maximum(score, score.T)


def build_signed_dual_waveclust_score(sim_mats: list[np.ndarray], *, k: float, neg_weight: float) -> np.ndarray:
    if len(sim_mats) < 2:
        raise ValueError("at least two WaveClust similarity matrices are required")
    low = np.asarray(sim_mats[0], dtype=np.float32)
    pos_low = np.maximum(low, 0.0)
    neg_low = np.maximum(-low, 0.0)
    score = np.zeros_like(pos_low, dtype=np.float32)
    for high_level in range(1, len(sim_mats)):
        high = np.asarray(sim_mats[high_level], dtype=np.float32)
        pos_high = np.maximum(high, 0.0)
        neg_high = np.maximum(-high, 0.0)
        pos_raw = np.sqrt(pos_low * pos_high)
        neg_raw = np.sqrt(neg_low * neg_high) * float(neg_weight)
        score = np.maximum(score, (pos_raw + neg_raw) * (float(k) * high_level + 1.0))
    np.fill_diagonal(score, 0.0)
    return np.maximum(score, score.T)


def normalized_affinity(similarity: np.ndarray) -> np.ndarray:
    affinity = np.maximum(np.asarray(similarity, dtype=np.float32), 0.0)
    max_weight = float(np.max(affinity))
    if np.isfinite(max_weight) and max_weight > 0:
        affinity = affinity / max_weight
    np.fill_diagonal(affinity, 1.0)
    return affinity


def power_affinity(similarity: np.ndarray, gamma: float) -> np.ndarray:
    affinity = normalized_affinity(similarity)
    if float(gamma) != 1.0:
        affinity = np.power(affinity, float(gamma)).astype(np.float32, copy=False)
        np.fill_diagonal(affinity, 1.0)
    return affinity


def topk_affinity(similarity: np.ndarray, *, top_k: int, gamma: float) -> np.ndarray:
    affinity = power_affinity(similarity, gamma)
    n = affinity.shape[0]
    k = min(max(int(top_k), 1), max(n - 1, 1))
    out = np.zeros_like(affinity, dtype=np.float32)
    for row_idx in range(n):
        row = affinity[row_idx].copy()
        row[row_idx] = -np.inf
        idx = np.argpartition(row, -k)[-k:]
        out[row_idx, idx] = affinity[row_idx, idx]
    out = np.maximum(out, out.T)
    np.fill_diagonal(out, 1.0)
    return out


def normalized_distance_from_similarity(similarity: np.ndarray) -> np.ndarray:
    max_weight = float(np.max(similarity))
    if not np.isfinite(max_weight) or max_weight <= 0:
        dist = np.ones_like(similarity, dtype=np.float32)
        np.fill_diagonal(dist, 0.0)
        return dist
    dist = 1.0 - (np.asarray(similarity, dtype=np.float32) / max_weight)
    np.fill_diagonal(dist, 0.0)
    return dist.astype(np.float32, copy=False)


def cluster_spectral_assign(
    affinity: np.ndarray,
    *,
    n_clusters: int,
    seed: int,
    assign_labels: str,
    n_jobs: int | None = None,
) -> np.ndarray:
    prepared = normalized_affinity(affinity)
    cluster_count = min(int(n_clusters), prepared.shape[0])
    model = SpectralClustering(
        n_clusters=cluster_count,
        affinity="precomputed",
        assign_labels=str(assign_labels),
        random_state=int(seed),
        n_init=5,
        eigen_solver="arpack",
        n_jobs=n_jobs,
    )
    return model.fit_predict(prepared).astype(np.int32)


def cluster_dense_kmeans(score: np.ndarray, *, n_clusters: int, seed: int) -> np.ndarray:
    cluster_count = min(int(n_clusters), score.shape[0])
    model = MiniBatchKMeans(
        n_clusters=cluster_count,
        random_state=int(seed),
        batch_size=2048,
        n_init=3,
        max_iter=150,
        reassignment_ratio=0.01,
    )
    return model.fit_predict(score).astype(np.int32)


def cluster_dense_agglomerative(score: np.ndarray, *, n_clusters: int, linkage: str) -> np.ndarray:
    dist = normalized_distance_from_similarity(score)
    cluster_count = min(int(n_clusters), score.shape[0])
    model = AgglomerativeClustering(n_clusters=cluster_count, metric="precomputed", linkage=str(linkage))
    return model.fit_predict(dist).astype(np.int32)


def _nystrom_embedding_cpu(
    similarity: np.ndarray,
    *,
    n_clusters: int,
    gamma: float,
    n_landmarks: int,
    seed: int,
) -> np.ndarray:
    affinity = power_affinity(similarity, gamma)
    n = affinity.shape[0]
    k = min(int(n_clusters), n)
    m = min(max(int(n_landmarks), k + 1), n)
    rng = np.random.default_rng(int(seed))
    landmark_idx = np.arange(n) if m == n else np.sort(rng.choice(n, size=m, replace=False))

    w = affinity[np.ix_(landmark_idx, landmark_idx)].astype(np.float32, copy=False)
    w = (w + w.T) * 0.5
    w.flat[:: m + 1] += 1e-5
    c = affinity[:, landmark_idx].astype(np.float32, copy=False)

    eigvals, eigvecs = np.linalg.eigh(w)
    valid = eigvals > 1e-6
    take = min(k, int(valid.sum()))
    if take < 1:
        return c
    basis = eigvecs[:, -take:] / np.sqrt(eigvals[-take:])[None, :]
    embedding = c @ basis
    norms = np.linalg.norm(embedding, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (embedding / norms).astype(np.float32, copy=False)


def cluster_nystrom_spectral_power(
    similarity: np.ndarray,
    *,
    n_clusters: int,
    gamma: float,
    n_landmarks: int,
    seed: int,
    kmeans_n_init: int = 1,
    kmeans_max_iter: int = 40,
    kmeans_threads: int | None = 4,
) -> np.ndarray:
    n = similarity.shape[0]
    cluster_count = min(int(n_clusters), n)
    if cluster_count <= 1:
        return np.zeros(n, dtype=np.int32)
    embedding = _nystrom_embedding_cpu(
        similarity,
        n_clusters=cluster_count,
        gamma=float(gamma),
        n_landmarks=int(n_landmarks),
        seed=int(seed),
    )
    model = MiniBatchKMeans(
        n_clusters=cluster_count,
        random_state=int(seed),
        batch_size=min(max(2048, cluster_count * 8), max(n, 1)),
        n_init=max(1, int(kmeans_n_init)),
        max_iter=max(1, int(kmeans_max_iter)),
        reassignment_ratio=0.01,
    )
    if kmeans_threads is None or int(kmeans_threads) <= 0:
        return model.fit_predict(embedding).astype(np.int32)
    with threadpool_limits(limits=int(kmeans_threads)):
        return model.fit_predict(embedding).astype(np.int32)


def cluster_connected_components(adj: np.ndarray) -> np.ndarray:
    n_components, labels = connected_components(csr_matrix(adj > 0), directed=False, return_labels=True)
    if n_components < 1:
        return np.arange(adj.shape[0], dtype=np.int32)
    return labels.astype(np.int32)


def graph_from_adjacency(adj: np.ndarray) -> nx.Graph:
    sparse_upper = triu(csr_matrix(adj), k=1, format="csr")
    graph = nx.from_scipy_sparse_array(sparse_upper, edge_attribute="weight")
    graph.add_nodes_from(range(adj.shape[0]))
    return graph


def cluster_louvain(adj: np.ndarray, *, resolution: float, seed: int) -> np.ndarray:
    graph = graph_from_adjacency(adj)
    communities = nx.algorithms.community.louvain_communities(
        graph,
        weight="weight",
        resolution=float(resolution),
        seed=int(seed),
    )
    labels = np.full(adj.shape[0], -1, dtype=np.int32)
    for cluster_id, members in enumerate(communities):
        labels[list(members)] = cluster_id
    if np.any(labels < 0):
        labels[labels < 0] = np.arange(np.sum(labels < 0), dtype=np.int32) + len(communities)
    return labels


def cluster_mcl(adj: np.ndarray, *, inflation: float, pruning_threshold: float, use_gpu: bool) -> np.ndarray:
    result = run_mcl_auto(
        np.asarray(adj, dtype=np.float32),
        inflation=float(inflation),
        pruning_threshold=float(pruning_threshold),
        use_gpu=bool(use_gpu),
    )
    clusters = get_clusters(result)
    labels = np.full(adj.shape[0], -1, dtype=np.int32)
    for cluster_id, indices in enumerate(clusters):
        labels[list(indices)] = cluster_id
    if np.any(labels < 0):
        labels[labels < 0] = np.arange(np.sum(labels < 0), dtype=np.int32) + len(clusters)
    return labels


def run_price_clusterer(
    *,
    cluster_params: dict[str, Any],
    dense_score: np.ndarray,
    adjacency: np.ndarray,
    spectral_jobs: int | None,
    use_gpu_mcl: bool,
) -> np.ndarray:
    clusterer = str(cluster_params["clusterer"])
    seed = int(cluster_params.get("seed", 42))
    if clusterer == "dense_spectral_power":
        affinity = power_affinity(dense_score, float(cluster_params.get("gamma", 1.0)))
        return cluster_spectral_assign(
            affinity,
            n_clusters=int(cluster_params["n_clusters"]),
            seed=seed,
            assign_labels=str(cluster_params.get("assign_labels", "kmeans")),
            n_jobs=spectral_jobs,
        )
    if clusterer == "dense_spectral_assign":
        return cluster_spectral_assign(
            dense_score,
            n_clusters=int(cluster_params["n_clusters"]),
            seed=seed,
            assign_labels=str(cluster_params.get("assign_labels", "kmeans")),
            n_jobs=spectral_jobs,
        )
    if clusterer == "knn_spectral":
        affinity = topk_affinity(
            dense_score,
            top_k=int(cluster_params.get("top_k", 80)),
            gamma=float(cluster_params.get("gamma", 1.0)),
        )
        return cluster_spectral_assign(
            affinity,
            n_clusters=int(cluster_params["n_clusters"]),
            seed=seed,
            assign_labels=str(cluster_params.get("assign_labels", "kmeans")),
            n_jobs=spectral_jobs,
        )
    if clusterer == "dense_kmeans":
        return cluster_dense_kmeans(dense_score, n_clusters=int(cluster_params["n_clusters"]), seed=seed)
    if clusterer == "dense_agglomerative":
        return cluster_dense_agglomerative(
            dense_score,
            n_clusters=int(cluster_params["n_clusters"]),
            linkage=str(cluster_params.get("linkage", "average")),
        )
    if clusterer == "dense_spectral_signed_dual_power":
        signed_score = cluster_params.get("_signed_dual_score")
        if signed_score is None:
            raise ValueError("_signed_dual_score is required for dense_spectral_signed_dual_power")
        affinity = power_affinity(np.asarray(signed_score), float(cluster_params.get("gamma", 1.0)))
        return cluster_spectral_assign(
            affinity,
            n_clusters=int(cluster_params["n_clusters"]),
            seed=seed,
            assign_labels=str(cluster_params.get("assign_labels", "kmeans")),
            n_jobs=spectral_jobs,
        )
    if clusterer == "nystrom_spectral_power":
        return cluster_nystrom_spectral_power(
            dense_score,
            n_clusters=int(cluster_params["n_clusters"]),
            gamma=float(cluster_params.get("gamma", 1.0)),
            n_landmarks=int(cluster_params.get("n_landmarks", 1024)),
            seed=seed,
            kmeans_n_init=int(cluster_params.get("kmeans_n_init", 1)),
            kmeans_max_iter=int(cluster_params.get("kmeans_max_iter", 40)),
            kmeans_threads=(
                None
                if cluster_params.get("kmeans_threads") is None
                else int(cluster_params.get("kmeans_threads", 4))
            ),
        )
    if clusterer == "nystrom_spectral_signed_dual_power":
        signed_score = cluster_params.get("_signed_dual_score")
        if signed_score is None:
            raise ValueError("_signed_dual_score is required for nystrom_spectral_signed_dual_power")
        return cluster_nystrom_spectral_power(
            np.asarray(signed_score),
            n_clusters=int(cluster_params["n_clusters"]),
            gamma=float(cluster_params.get("gamma", 1.0)),
            n_landmarks=int(cluster_params.get("n_landmarks", 1024)),
            seed=seed,
            kmeans_n_init=int(cluster_params.get("kmeans_n_init", 1)),
            kmeans_max_iter=int(cluster_params.get("kmeans_max_iter", 40)),
            kmeans_threads=(
                None
                if cluster_params.get("kmeans_threads") is None
                else int(cluster_params.get("kmeans_threads", 4))
            ),
        )
    if clusterer == "louvain":
        return cluster_louvain(adjacency, resolution=float(cluster_params["resolution"]), seed=seed)
    if clusterer == "mcl":
        return cluster_mcl(
            adjacency,
            inflation=float(cluster_params.get("inflation", 1.4)),
            pruning_threshold=float(cluster_params.get("pruning_threshold", 0.05)),
            use_gpu=use_gpu_mcl,
        )
    if clusterer == "connected_components":
        return cluster_connected_components(adjacency)
    raise ValueError(f"unknown clusterer: {clusterer}")
