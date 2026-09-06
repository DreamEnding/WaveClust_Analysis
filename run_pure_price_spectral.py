from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from copy import deepcopy
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from waveclust.config import CODE_DIR, WORKSPACE_DIR, load_config, resolve_workspace_path
from waveclust.pipeline import build_params, load_prices, write_json
from waveclust.pure_price import (
    build_adjacency_no_metadata,
    build_similarity_mats_no_metadata,
    filter_prices_by_window_eligibility,
    preprocess_returns_no_metadata,
)
from waveclust.shenwan import SW_LEVELS, evaluate_assignments, load_shenwan_label_table
from waveclust.spectral import (
    build_dense_waveclust_score,
    build_signed_dual_waveclust_score,
    cluster_stats,
    labels_to_assignments,
    run_price_clusterer,
)


DEFAULT_INFLATIONS = [round(1.2 + 0.2 * idx, 1) for idx in range(9)]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run strict pure-price WaveClust dense spectral-power experiments.")
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "config.yaml")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--transforms", nargs="+", default=["modwt"])
    parser.add_argument("--wavelets", nargs="+", default=["db4", "sym2", "sym4", "coif1"])
    parser.add_argument("--levels", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    parser.add_argument("--q-thresholds", type=float, nargs="+", default=[0.996])
    parser.add_argument("--winsor-limits", type=float, nargs="+", default=[0.03])
    parser.add_argument("--return-modes", nargs="+", default=["raw"], choices=["raw", "market_demean", "market_residual"])
    parser.add_argument("--k-values", type=float, nargs="+", default=[0.0])
    parser.add_argument("--gammas", type=float, nargs="+", default=[0.3])
    parser.add_argument("--n-clusters", type=int, nargs="+", default=[380])
    parser.add_argument(
        "--clusterers",
        nargs="+",
        default=["dense_spectral_power"],
        choices=[
            "dense_spectral_power",
            "dense_spectral_signed_dual_power",
            "dense_spectral_assign",
            "knn_spectral",
            "nystrom_spectral_power",
            "nystrom_spectral_signed_dual_power",
            "dense_kmeans",
            "dense_agglomerative",
            "louvain",
            "mcl",
            "connected_components",
        ],
    )
    parser.add_argument("--top-k-values", type=int, nargs="+", default=[80])
    parser.add_argument("--neg-weights", type=float, nargs="+", default=[0.5])
    parser.add_argument("--n-landmarks", type=int, nargs="+", default=[512])
    parser.add_argument("--kmeans-n-init", type=int, default=1)
    parser.add_argument("--kmeans-max-iter", type=int, default=40)
    parser.add_argument("--kmeans-threads", type=int, default=4)
    parser.add_argument("--linkages", nargs="+", default=["average"])
    parser.add_argument("--resolutions", type=float, nargs="+", default=[0.1])
    parser.add_argument("--inflations", type=float, nargs="+", default=DEFAULT_INFLATIONS)
    parser.add_argument("--pruning-thresholds", type=float, nargs="+", default=[0.05])
    parser.add_argument("--assign-labels", default="kmeans")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--spectral-jobs", type=int, default=-1)
    parser.add_argument("--use-gpu-mcl", action="store_true")
    parser.add_argument("--target-min", type=float, default=0.2)
    parser.add_argument("--window-id", type=str, default="")
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--min-price-coverage", type=float, default=0.0)
    parser.add_argument("--max-flat-return-rate", type=float, default=None)
    parser.add_argument("--min-eligible-stocks", type=int, default=2)
    parser.add_argument("--max-trials", type=int, default=None)
    parser.add_argument("--include-trial-ids-file", type=Path, default=None)
    parser.add_argument("--stop-on-target", action="store_true")
    parser.add_argument("--write-best-assignments", action="store_true")
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def make_trial_id(feature: dict[str, Any], cluster: dict[str, Any]) -> str:
    payload = {"feature": feature, "cluster": cluster, "strict_no_metadata": True}
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha1(encoded.encode("utf-8")).hexdigest()[:16]


def make_similarity_key(feature: dict[str, Any]) -> tuple[str, str, int, float, str]:
    return (
        str(feature["wavelet_transform"]),
        str(feature["wavelet"]),
        int(feature["levels"]),
        float(feature["winsor_limit"]),
        str(feature["return_mode"]),
    )


def make_score_key(feature: dict[str, Any]) -> tuple[str, str, int, float, str, float]:
    return (*make_similarity_key(feature), float(feature["k"]))


def load_trial_id_filter(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    trial_ids = {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    return trial_ids


def expand_grid(args: argparse.Namespace) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    features = [
        {
            "wavelet_transform": transform,
            "wavelet": wavelet,
            "levels": int(level),
            "q_threshold": float(q_threshold),
            "winsor_limit": float(winsor_limit),
            "return_mode": str(return_mode),
            "k": float(k_value),
        }
        for transform, wavelet, level, q_threshold, winsor_limit, return_mode, k_value in product(
            args.transforms,
            args.wavelets,
            args.levels,
            args.q_thresholds,
            args.winsor_limits,
            args.return_modes,
            args.k_values,
        )
    ]
    clusters: list[dict[str, Any]] = []
    for clusterer in args.clusterers:
        if clusterer == "dense_spectral_power":
            clusters.extend(
                {
                    "clusterer": clusterer,
                    "gamma": float(gamma),
                    "n_clusters": int(n_clusters),
                    "assign_labels": str(args.assign_labels),
                    "seed": int(args.seed),
                }
                for gamma, n_clusters in product(args.gammas, args.n_clusters)
            )
        elif clusterer == "dense_spectral_signed_dual_power":
            clusters.extend(
                {
                    "clusterer": clusterer,
                    "gamma": float(gamma),
                    "neg_weight": float(neg_weight),
                    "n_clusters": int(n_clusters),
                    "assign_labels": str(args.assign_labels),
                    "seed": int(args.seed),
                }
                for gamma, neg_weight, n_clusters in product(args.gammas, args.neg_weights, args.n_clusters)
            )
        elif clusterer == "dense_spectral_assign":
            clusters.extend(
                {
                    "clusterer": clusterer,
                    "n_clusters": int(n_clusters),
                    "assign_labels": str(args.assign_labels),
                    "seed": int(args.seed),
                }
                for n_clusters in args.n_clusters
            )
        elif clusterer == "knn_spectral":
            clusters.extend(
                {
                    "clusterer": clusterer,
                    "top_k": int(top_k),
                    "gamma": float(gamma),
                    "n_clusters": int(n_clusters),
                    "assign_labels": str(args.assign_labels),
                    "seed": int(args.seed),
                }
                for top_k, gamma, n_clusters in product(args.top_k_values, args.gammas, args.n_clusters)
            )
        elif clusterer == "nystrom_spectral_power":
            clusters.extend(
                {
                    "clusterer": clusterer,
                    "gamma": float(gamma),
                    "n_clusters": int(n_clusters),
                    "n_landmarks": int(n_landmarks),
                    "kmeans_n_init": int(args.kmeans_n_init),
                    "kmeans_max_iter": int(args.kmeans_max_iter),
                    "kmeans_threads": int(args.kmeans_threads),
                    "seed": int(args.seed),
                }
                for gamma, n_clusters, n_landmarks in product(args.gammas, args.n_clusters, args.n_landmarks)
            )
        elif clusterer == "nystrom_spectral_signed_dual_power":
            clusters.extend(
                {
                    "clusterer": clusterer,
                    "gamma": float(gamma),
                    "neg_weight": float(neg_weight),
                    "n_clusters": int(n_clusters),
                    "n_landmarks": int(n_landmarks),
                    "kmeans_n_init": int(args.kmeans_n_init),
                    "kmeans_max_iter": int(args.kmeans_max_iter),
                    "kmeans_threads": int(args.kmeans_threads),
                    "seed": int(args.seed),
                }
                for gamma, neg_weight, n_clusters, n_landmarks in product(
                    args.gammas,
                    args.neg_weights,
                    args.n_clusters,
                    args.n_landmarks,
                )
            )
        elif clusterer == "dense_kmeans":
            clusters.extend(
                {"clusterer": clusterer, "n_clusters": int(n_clusters), "seed": int(args.seed)}
                for n_clusters in args.n_clusters
            )
        elif clusterer == "dense_agglomerative":
            clusters.extend(
                {"clusterer": clusterer, "n_clusters": int(n_clusters), "linkage": str(linkage)}
                for n_clusters, linkage in product(args.n_clusters, args.linkages)
            )
        elif clusterer == "louvain":
            clusters.extend(
                {"clusterer": clusterer, "resolution": float(resolution), "seed": int(args.seed)}
                for resolution in args.resolutions
            )
        elif clusterer == "mcl":
            clusters.extend(
                {
                    "clusterer": clusterer,
                    "inflation": float(inflation),
                    "pruning_threshold": float(pruning_threshold),
                    "seed": int(args.seed),
                }
                for inflation, pruning_threshold in product(args.inflations, args.pruning_thresholds)
            )
        elif clusterer == "connected_components":
            clusters.append({"clusterer": clusterer, "seed": int(args.seed)})
    pairs = [(feature, cluster) for feature in features for cluster in clusters]
    include_trial_ids = load_trial_id_filter(args.include_trial_ids_file)
    if include_trial_ids is not None:
        pairs = [
            (feature, cluster)
            for feature, cluster in pairs
            if make_trial_id(feature, cluster) in include_trial_ids
        ]
    if int(args.num_shards) < 1:
        raise ValueError("--num-shards must be >= 1")
    if int(args.shard_index) < 0 or int(args.shard_index) >= int(args.num_shards):
        raise ValueError("--shard-index must satisfy 0 <= shard-index < num-shards")
    pairs = [
        pair
        for pair_index, pair in enumerate(pairs)
        if pair_index % int(args.num_shards) == int(args.shard_index)
    ]
    if args.max_trials is not None:
        pairs = pairs[: int(args.max_trials)]
    return pairs


def load_completed(csv_path: Path, *, force: bool) -> set[str]:
    if force or not csv_path.exists():
        return set()
    frame = pd.read_csv(csv_path)
    if "trial_id" not in frame.columns:
        return set()
    return set(frame["trial_id"].astype(str))


def append_row(csv_path: Path, jsonl_path: Path, row: dict[str, Any]) -> None:
    safe_row = json_safe(row)
    if csv_path.exists():
        columns = pd.read_csv(csv_path, nrows=0).columns.tolist()
        frame = pd.DataFrame([safe_row]).reindex(columns=columns)
    else:
        frame = pd.DataFrame([safe_row])
    frame.to_csv(csv_path, mode="a", index=False, header=not csv_path.exists(), encoding="utf-8-sig")
    with jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(safe_row, ensure_ascii=False) + "\n")


def build_feature_config(base_config: dict[str, Any], feature: dict[str, Any]) -> dict[str, Any]:
    cfg = deepcopy(base_config)
    cfg.setdefault("wavelet", {})["transform"] = str(feature["wavelet_transform"])
    cfg.setdefault("wavelet", {})["wavelet"] = str(feature["wavelet"])
    cfg.setdefault("wavelet", {})["levels"] = int(feature["levels"])
    cfg.setdefault("wavelet", {})["k"] = float(feature["k"])
    cfg.setdefault("wavelet", {})["q_threshold"] = float(feature["q_threshold"])
    return cfg


def make_base_result_row(
    *,
    args: argparse.Namespace,
    feature: dict[str, Any],
    cluster: dict[str, Any],
    price_shape: tuple[int, int],
    price_filter_stats: dict[str, Any],
) -> dict[str, Any]:
    n_clusters = int(cluster["n_clusters"]) if "n_clusters" in cluster else np.nan
    return {
        "trial_id": make_trial_id(feature, cluster),
        "status": "running",
        "started_at": datetime.now().astimezone().isoformat(),
        "window_id": str(args.window_id or ""),
        "window_start": str(args.start_date or ""),
        "window_end": str(args.end_date or ""),
        **feature,
        "clusterer": cluster["clusterer"],
        "clusterer_params": json.dumps(json_safe(cluster), ensure_ascii=False, sort_keys=True),
        "gamma": float(cluster["gamma"]) if "gamma" in cluster else np.nan,
        "neg_weight": float(cluster["neg_weight"]) if "neg_weight" in cluster else np.nan,
        "top_k": int(cluster["top_k"]) if "top_k" in cluster else np.nan,
        "n_landmarks": int(cluster["n_landmarks"]) if "n_landmarks" in cluster else np.nan,
        "n_clusters_requested": n_clusters,
        "n_cluster": n_clusters,
        "assign_labels": str(cluster.get("assign_labels", "")),
        "resolution": float(cluster["resolution"]) if "resolution" in cluster else np.nan,
        "inflation": float(cluster["inflation"]) if "inflation" in cluster else np.nan,
        "pruning_threshold": float(cluster["pruning_threshold"]) if "pruning_threshold" in cluster else np.nan,
        "seed": int(cluster["seed"]),
        "strict_no_metadata": True,
        "metadata_sources_for_clustering": [],
        "uses_waveclust_network": True,
        "uses_metadata_seed": False,
        "uses_shenwan_for_clustering": False,
        "n_price_rows": int(price_shape[0]),
        "n_price_cols": int(price_shape[1]),
        **price_filter_stats,
        "assignments_path": "",
        "error_type": "",
        "error_message": "",
    }


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    assignments_dir = args.output_dir / "assignments"
    assignments_dir.mkdir(exist_ok=True)
    metrics_path = args.output_dir / "metrics.csv"
    jsonl_path = args.output_dir / "metrics.jsonl"

    base_config = load_config(args.config)
    trials = expand_grid(args)
    completed = load_completed(metrics_path, force=args.force)
    if args.force:
        metrics_path.unlink(missing_ok=True)
        jsonl_path.unlink(missing_ok=True)

    prices = load_prices(base_config, start_date=args.start_date, end_date=args.end_date)
    prices, price_filter_stats = filter_prices_by_window_eligibility(
        prices,
        min_price_coverage=float(args.min_price_coverage),
        max_flat_return_rate=args.max_flat_return_rate,
        min_stocks=int(args.min_eligible_stocks),
    )
    universe_path = resolve_workspace_path("DATA/tickflow_universes/universe_list.json")
    label_tables = {level: load_shenwan_label_table(universe_path, level=level) for level in SW_LEVELS}

    write_json(
        args.output_dir / "run_manifest.json",
        {
            "run_id": args.output_dir.name,
            "started_at": datetime.now().astimezone().isoformat(),
            "command": " ".join(sys.argv),
            "workspace_dir": WORKSPACE_DIR,
            "code_dir": CODE_DIR,
            "config_path": args.config,
            "python": sys.version,
            "platform": platform.platform(),
            "trial_count": len(trials),
            "include_trial_ids_file": args.include_trial_ids_file,
            "include_trial_id_count": len(load_trial_id_filter(args.include_trial_ids_file) or []),
            "num_shards": int(args.num_shards),
            "shard_index": int(args.shard_index),
            "window_id": str(args.window_id or ""),
            "start_date": args.start_date,
            "end_date": args.end_date,
            "price_filter": price_filter_stats,
            "strict_no_metadata": True,
            "metadata_sources_for_clustering": [],
            "validity_boundary": "Only prices are used for clustering; Shenwan labels are loaded after clustering for evaluation only.",
        },
    )

    rows: list[dict[str, Any]] = []
    best_row: dict[str, Any] | None = None
    best_assignments: pd.DataFrame | None = None
    returns_cache: dict[tuple[float, str], pd.DataFrame] = {}
    sim_cache: dict[tuple[str, str, int, float, str], tuple[list[np.ndarray], np.ndarray, list[str]]] = {}
    adj_cache: dict[tuple[tuple[str, str, int, float, str, float], float], tuple[np.ndarray, dict[str, Any]]] = {}
    dense_cache: dict[tuple[str, str, int, float, str, float], np.ndarray] = {}
    signed_cache: dict[tuple[tuple[str, str, int, float, str, float], float], np.ndarray] = {}

    started_all = time.time()
    for index, (feature, cluster) in enumerate(trials, start=1):
        trial_id = make_trial_id(feature, cluster)
        if trial_id in completed:
            print(f"skip completed {index}/{len(trials)} id={trial_id}")
            continue

        row: dict[str, Any] = make_base_result_row(
            args=args,
            feature=feature,
            cluster=cluster,
            price_shape=prices.shape,
            price_filter_stats=price_filter_stats,
        )
        trial_started = time.time()
        try:
            winsor = float(feature["winsor_limit"])
            return_mode = str(feature["return_mode"])
            returns_key = (winsor, return_mode)
            if returns_key not in returns_cache:
                returns_cache[returns_key] = preprocess_returns_no_metadata(
                    prices,
                    winsor_limit=winsor,
                    do_vol_zscore=bool(base_config.get("preprocessing", {}).get("do_vol_zscore", True)),
                    return_mode=return_mode,
                    dtype=str(base_config.get("performance", {}).get("dtype", "float32")),
                )
            returns = returns_cache[returns_key]
            sim_key = make_similarity_key(feature)
            score_key = make_score_key(feature)
            if sim_key not in sim_cache:
                cfg = build_feature_config(base_config, feature)
                params = build_params(cfg, use_gpu=not args.no_gpu)
                sim_cache.clear()
                adj_cache.clear()
                dense_cache.clear()
                signed_cache.clear()
                sim_cache[sim_key] = build_similarity_mats_no_metadata(returns, params=params)
            sim_mats, all_sims, stock_names = sim_cache[sim_key]

            adj_key = (score_key, float(feature["q_threshold"]))
            if adj_key not in adj_cache:
                adj_cache[adj_key] = build_adjacency_no_metadata(
                    sim_mats,
                    all_sims,
                    q_threshold=float(feature["q_threshold"]),
                    k=float(feature["k"]),
                )
            _adj, graph_stats = adj_cache[adj_key]

            if score_key not in dense_cache:
                dense_cache[score_key] = build_dense_waveclust_score(sim_mats, k=float(feature["k"]))
            dense_score = dense_cache[score_key]
            cluster_for_run = dict(cluster)
            if str(cluster["clusterer"]) in {"dense_spectral_signed_dual_power", "nystrom_spectral_signed_dual_power"}:
                signed_key = (score_key, float(cluster.get("neg_weight", 1.0)))
                if signed_key not in signed_cache:
                    signed_cache[signed_key] = build_signed_dual_waveclust_score(
                        sim_mats,
                        k=float(feature["k"]),
                        neg_weight=float(cluster.get("neg_weight", 1.0)),
                    )
                cluster_for_run["_signed_dual_score"] = signed_cache[signed_key]
            labels = run_price_clusterer(
                cluster_params=cluster_for_run,
                dense_score=dense_score,
                adjacency=_adj,
                spectral_jobs=None if int(args.spectral_jobs) == 0 else int(args.spectral_jobs),
                use_gpu_mcl=bool(args.use_gpu_mcl and not args.no_gpu),
            )
            assignments = labels_to_assignments(labels, stock_names)
            sw_metrics = evaluate_assignments(assignments, label_tables)

            row.update(cluster_stats(labels, **graph_stats))
            row.update(sw_metrics)
            row["ari"] = row["SW1_ari"]
            row["nmi"] = row["SW1_nmi"]
            row["n_return_rows"] = int(returns.shape[0])
            row["n_return_cols"] = int(returns.shape[1])
            row["return_start"] = str(returns.index.min().date()) if len(returns.index) else ""
            row["return_end"] = str(returns.index.max().date()) if len(returns.index) else ""
            row["status"] = "ok"
            row["target_reached"] = bool(float(row["SW1_ari"]) >= float(args.target_min))
            if best_row is None or float(row["SW1_ari"]) > float(best_row["SW1_ari"]):
                best_row = dict(row)
                best_assignments = assignments
                if args.write_best_assignments:
                    path = assignments_dir / f"{trial_id}_best.csv"
                    assignments.to_csv(path, index=False, encoding="utf-8-sig")
                    row["assignments_path"] = str(path)
                    best_row["assignments_path"] = str(path)
            print(
                f"ok {index}/{len(trials)} id={trial_id} transform={feature['wavelet_transform']} "
                f"wavelet={feature['wavelet']} level={feature['levels']} q={feature['q_threshold']} "
                f"winsor={winsor} clusterer={cluster['clusterer']} params={json.dumps(json_safe(cluster), sort_keys=True)} "
                f"SW1_ari={float(row['SW1_ari']):.12f} SW1_nmi={float(row['SW1_nmi']):.12f}"
            )
        except Exception as exc:
            row["status"] = "failed"
            row["error_type"] = type(exc).__name__
            row["error_message"] = str(exc)
            print(f"failed {index}/{len(trials)} id={trial_id}: {type(exc).__name__}: {exc}")
        finally:
            row["finished_at"] = datetime.now().astimezone().isoformat()
            row["seconds"] = float(time.time() - trial_started)
            append_row(metrics_path, jsonl_path, row)
            rows.append(row)

        if row.get("status") == "ok" and bool(row.get("target_reached")) and args.stop_on_target:
            break

    if metrics_path.exists():
        summary = pd.read_csv(metrics_path)
    else:
        summary = pd.DataFrame(rows)
    if not summary.empty and "SW1_ari" in summary.columns:
        summary = summary.sort_values(["SW1_ari", "SW1_nmi"], ascending=[False, False])
    summary.to_csv(args.output_dir / "summary.csv", index=False, encoding="utf-8-sig")

    if best_row is not None:
        if args.write_best_assignments and best_assignments is not None and not best_row.get("assignments_path"):
            path = assignments_dir / f"{best_row['trial_id']}_best.csv"
            best_assignments.to_csv(path, index=False, encoding="utf-8-sig")
            best_row["assignments_path"] = str(path)
        write_json(args.output_dir / "best_observed.json", best_row)
    write_json(
        args.output_dir / "final_summary.json",
        {
            "finished_at": datetime.now().astimezone().isoformat(),
            "seconds": float(time.time() - started_all),
            "n_rows": int(len(summary)),
            "best_observed": best_row,
        },
    )
    print("\n========== Pure-price spectral summary ==========")
    if not summary.empty:
        display_cols = [
            "trial_id",
            "wavelet_transform",
            "wavelet",
            "levels",
            "q_threshold",
            "winsor_limit",
            "gamma",
            "neg_weight",
            "top_k",
            "n_landmarks",
            "resolution",
            "inflation",
            "n_clusters_requested",
            "SW1_ari",
            "SW1_nmi",
            "n_communities",
            "n_edges",
            "seconds",
        ]
        print(summary[[col for col in display_cols if col in summary.columns]].head(20).to_string(index=False))
    print(f"saved summary: {args.output_dir / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
