from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pywt
import scipy
import sklearn
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score

from waveclust.data import load_price_panel, normalize_stock_code
from waveclust.model import StockWaveClust, WaveClustParams
from waveclust.pure_price import filter_prices_by_window_eligibility, preprocess_returns_no_metadata
from waveclust.shenwan import SW_LEVELS, evaluate_assignments, load_shenwan_label_table
from waveclust.spectral import (
    SignedDualScoreCache,
    build_signed_dual_score_cache,
    build_signed_dual_waveclust_score,
    cluster_stats,
    labels_to_assignments,
    run_price_clusterer,
)


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
OPERATOR_CONFIGS: dict[str, tuple[str, bool]] = {
    "max_weighted": ("max", True),
    "mean_weighted": ("mean", True),
    "median_weighted": ("median", True),
    "geometric_weighted": ("geometric", True),
    "max_unweighted": ("max", False),
    "mean_unweighted": ("mean", False),
}
ARCHIVED_MAX_SW1_ARI = {
    2: 0.38093810251628313,
    3: 0.30232940615394444,
    4: 0.22026961030186798,
    5: 0.18628199806632129,
    6: 0.08356198572369729,
}
ARCHIVED_J2_STATS = {
    "n_communities": 25,
    "n_singletons": 0,
    "largest_community": 220,
    "median_community_size": 115.0,
}


@dataclass(frozen=True)
class ResolvedPaths:
    workspace_root: Path
    data_panel: Path
    shenwan_universe: Path
    archived_assignment: Path
    output_dir: Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the supplementary signed-dual operator ablation.")
    parser.add_argument("--workspace-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--data-panel", type=Path, default=None)
    parser.add_argument("--shenwan-universe", type=Path, default=None)
    parser.add_argument("--archived-assignment", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--levels", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    parser.add_argument("--operators", nargs="+", choices=sorted(OPERATOR_CONFIGS), default=list(OPERATOR_CONFIGS))
    parser.add_argument("--start-date", default="2019-01-01")
    parser.add_argument("--end-date", default="2025-12-31")
    parser.add_argument("--min-price-coverage", type=float, default=0.95)
    parser.add_argument("--max-flat-return-rate", type=float, default=0.05)
    parser.add_argument("--min-eligible-stocks", type=int, default=1000)
    parser.add_argument("--smoke-stocks", type=int, default=0)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--gpu-ids", type=int, nargs="+", default=None)
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--wavelet-workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _resolve(root: Path, value: Path | None, default: str) -> Path:
    path = Path(default) if value is None else value
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def resolve_paths(args: argparse.Namespace) -> ResolvedPaths:
    root = args.workspace_root.resolve()
    return ResolvedPaths(
        workspace_root=root,
        data_panel=_resolve(root, args.data_panel, "DATA/stock_price_panel.csv"),
        shenwan_universe=_resolve(root, args.shenwan_universe, "DATA/tickflow_universes/universe_list.json"),
        archived_assignment=_resolve(
            root,
            args.archived_assignment,
            "output/waveclust_dense_signed_full_grid_20260525/w2019_2025/assignments/"
            "a85036d07d14690a_best.csv",
        ),
        output_dir=_resolve(root, args.output_dir, str(args.output_dir)),
    )


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


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def environment_snapshot() -> dict[str, Any]:
    try:
        import cupy as cp

        cupy_version = cp.__version__
        try:
            gpu_count = int(cp.cuda.runtime.getDeviceCount())
        except Exception as exc:
            gpu_count = 0
            gpu_error = f"{type(exc).__name__}: {exc}"
        else:
            gpu_error = ""
    except Exception as exc:
        cupy_version = None
        gpu_count = 0
        gpu_error = f"{type(exc).__name__}: {exc}"
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "pandas": pd.__version__,
        "pywavelets": pywt.__version__,
        "cupy": cupy_version,
        "cupy_gpu_count": gpu_count,
        "cupy_gpu_error": gpu_error,
    }


def _make_params(args: argparse.Namespace, level: int) -> WaveClustParams:
    gpu_ids = tuple(int(device) for device in (args.gpu_ids or [args.gpu_id]))
    return WaveClustParams(
        wavelet_transform="modwt",
        wavelet="sym2",
        levels=int(level),
        k=1.5,
        q_threshold=0.996,
        winsor_limits=(0.03, 0.03),
        do_vol_zscore=True,
        use_gpu=not args.no_gpu,
        dtype="float32",
        gpu_ids=gpu_ids,
        multi_gpu_similarity=len(gpu_ids) > 1,
        similarity_workers=len(gpu_ids),
        wavelet_workers=int(args.wavelet_workers),
        mcl_backend="cpu",
    )


def build_level_similarities(
    returns: pd.DataFrame,
    args: argparse.Namespace,
    level: int,
) -> tuple[list[np.ndarray], list[str], dict[str, float]]:
    model = StockWaveClust(prices=returns, stock_info=None, params=_make_params(args, level))
    if not args.no_gpu and not model.use_gpu:
        raise RuntimeError(
            "GPU execution requested but CuPy cannot see a CUDA device. Run the trade interpreter with user site "
            "disabled and restore /dev/nvidia* before a formal run."
        )
    model.returns = returns
    wavelet_started = time.perf_counter()
    model.decompose_wavelets()
    level_matrices, stock_names = model.prepare_level_matrices()
    model.release_wavelet_coefficients()
    wavelet_seconds = time.perf_counter() - wavelet_started
    similarity_started = time.perf_counter()
    sim_mats = model.compute_similarity_matrices(level_matrices)
    similarity_seconds = time.perf_counter() - similarity_started
    return sim_mats, stock_names, {
        "wavelet_seconds": float(wavelet_seconds),
        "similarity_seconds": float(similarity_seconds),
    }


def run_operator_partition(
    *,
    sim_mats: list[np.ndarray],
    stock_names: list[str],
    label_tables: dict[str, pd.DataFrame],
    level: int,
    operator: str,
    seed: int,
    score_cache: SignedDualScoreCache | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    reducer, layer_weighting = OPERATOR_CONFIGS[operator]
    score_started = time.perf_counter()
    if score_cache is None:
        score = build_signed_dual_waveclust_score(
            sim_mats,
            k=1.5,
            neg_weight=1.25,
            reducer=reducer,
            layer_weighting=layer_weighting,
            interaction="sqrt",
        )
    else:
        score = score_cache.build(
            k=1.5,
            reducer=reducer,
            layer_weighting=layer_weighting,
        )
    score_seconds = time.perf_counter() - score_started
    cluster_started = time.perf_counter()
    labels = run_price_clusterer(
        cluster_params={
            "clusterer": "dense_spectral_signed_dual_power",
            "gamma": 0.35,
            "neg_weight": 1.25,
            "n_clusters": 25,
            "assign_labels": "discretize",
            "seed": int(seed),
            "_signed_dual_score": score,
        },
        dense_score=score,
        adjacency=score,
        spectral_jobs=1,
        use_gpu_mcl=False,
    )
    cluster_seconds = time.perf_counter() - cluster_started
    assignments = labels_to_assignments(labels, stock_names)
    possible_edges = len(labels) * (len(labels) - 1) / 2
    edge_count = int(np.count_nonzero(score) // 2)
    row = {
        "experiment": "operator_ablation",
        "level": int(level),
        "operator": operator,
        "reducer": reducer,
        "layer_weighting": bool(layer_weighting),
        "interaction": "sqrt",
        "k": 1.5,
        "neg_weight": 1.25,
        "gamma": 0.35,
        "n_clusters_requested": 25,
        "assign_labels": "discretize",
        "seed": int(seed),
        "q_threshold_manifest_only": 0.996,
        **cluster_stats(
            labels,
            n_edges=edge_count,
            edge_density=float(edge_count / possible_edges) if possible_edges else 0.0,
            n_stocks=len(labels),
        ),
        **evaluate_assignments(assignments, label_tables),
        "score_seconds": float(score_seconds),
        "cluster_seconds": float(cluster_seconds),
    }
    row["ari"] = row["SW1_ari"]
    row["nmi"] = row["SW1_nmi"]
    if operator == "max_weighted":
        row["archived_SW1_ari"] = ARCHIVED_MAX_SW1_ARI[int(level)]
        row["archived_SW1_ari_delta"] = float(row["SW1_ari"] - ARCHIVED_MAX_SW1_ARI[int(level)])
    return row, assignments


def assignment_ari_against_archive(current: pd.DataFrame, archived_path: Path) -> float:
    archived = pd.read_csv(archived_path, dtype={"stock": str, "cluster_id": str})
    current = current.astype({"stock": str, "cluster_id": str}).copy()
    archived["stock"] = archived["stock"].map(normalize_stock_code)
    current["stock"] = current["stock"].map(normalize_stock_code)
    if archived["stock"].duplicated().any() or current["stock"].duplicated().any():
        raise ValueError("baseline assignments contain duplicate stocks")
    merged = archived.merge(current, on="stock", suffixes=("_archived", "_current"), validate="one_to_one")
    if len(merged) != len(archived) or len(merged) != len(current):
        raise ValueError(
            f"baseline stock sets differ: archived={len(archived)} current={len(current)} aligned={len(merged)}"
        )
    return float(adjusted_rand_score(merged["cluster_id_archived"], merged["cluster_id_current"]))


def validate_reference_gate(
    row: dict[str, Any],
    assignments: pd.DataFrame,
    archived_assignment: Path,
) -> dict[str, Any]:
    partition_ari = assignment_ari_against_archive(assignments, archived_assignment)
    checks = {
        "partition_ari_is_one": bool(partition_ari == 1.0),
        "SW1_ari_exact": bool(float(row["SW1_ari"]) == ARCHIVED_MAX_SW1_ARI[2]),
        **{
            f"{key}_exact": bool(float(row[key]) == float(expected))
            for key, expected in ARCHIVED_J2_STATS.items()
        },
    }
    return {
        "passed": bool(all(checks.values())),
        "partition_ari": partition_ari,
        "checks": checks,
        "observed": {key: row[key] for key in ["SW1_ari", *ARCHIVED_J2_STATS]},
        "expected": {"SW1_ari": ARCHIVED_MAX_SW1_ARI[2], **ARCHIVED_J2_STATS},
    }


def monotonicity_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for operator, frame in metrics.groupby("operator", sort=False):
        frame = frame.sort_values("level")
        levels = frame["level"].to_numpy(dtype=np.int64)
        ari = frame["SW1_ari"].to_numpy(dtype=np.float64)
        if len(levels) != 5 or levels.tolist() != [2, 3, 4, 5, 6]:
            classification = "incomplete"
            adjacent = np.full(4, np.nan)
            endpoint = np.nan
            rho = np.nan
        else:
            adjacent = np.diff(ari)
            endpoint = float(ari[-1] - ari[0])
            rho = float(spearmanr(levels, ari).statistic)
            if bool(np.all(adjacent < 0)):
                classification = "strictly_preserved"
            elif bool(ari[-1] < ari[0] and rho <= -0.9):
                classification = "direction_preserved"
            else:
                classification = "not_preserved"
        rows.append(
            {
                "operator": operator,
                **{f"delta_J{level}_to_J{level + 1}": float(adjacent[index]) for index, level in enumerate(range(2, 6))},
                "endpoint_J2_to_J6": endpoint,
                "spearman_rho_level_SW1_ari": rho,
                "classification": classification,
            }
        )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    requested_levels = sorted(set(int(level) for level in args.levels))
    cache_scores = not args.baseline_only and len(args.operators) > 1
    invalid_levels = sorted(set(requested_levels).difference(range(2, 7)))
    if invalid_levels:
        raise ValueError(f"levels must be in [2, 6], got {invalid_levels}")
    if int(args.seed) != 42:
        raise ValueError("operator-ablation runs use the preregistered seed 42")
    paths = resolve_paths(args)
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = paths.output_dir / "metrics.csv"
    if metrics_path.exists() and not args.force:
        raise FileExistsError(f"metrics already exist: {metrics_path}; use a new output directory or --force")
    assignments_dir = paths.output_dir / "assignments"
    assignments_dir.mkdir(exist_ok=True)
    started = time.time()
    env = environment_snapshot()
    formal = int(args.smoke_stocks) <= 0
    manifest = {
        "run_id": paths.output_dir.name,
        "experiment": "operator_ablation",
        "started_at": datetime.fromtimestamp(started).astimezone().isoformat(),
        "command": " ".join(sys.argv if argv is None else [sys.argv[0], *argv]),
        "paths": asdict(paths),
        "formal": formal,
        "environment": env,
        "contract": {
            "levels": requested_levels,
            "operators": args.operators,
            "start_date": args.start_date,
            "end_date": args.end_date,
            "min_price_coverage": args.min_price_coverage,
            "max_flat_return_rate": args.max_flat_return_rate,
            "min_eligible_stocks": args.min_eligible_stocks,
            "smoke_stocks": args.smoke_stocks,
            "gpu_ids": list(args.gpu_ids or [args.gpu_id]),
            "use_gpu": not args.no_gpu,
            "spectral_jobs": 1,
            "wavelet_workers": args.wavelet_workers,
            "seed": args.seed,
            "signed_interaction_cache": {
                "enabled": cache_scores,
                "scope": "one level at a time",
                "reason": "reuse identical signed interactions across requested reducers",
            },
        },
        "failures": [],
    }
    write_json(paths.output_dir / "run_manifest.json", manifest)

    prices = load_price_panel(
        paths.data_panel,
        start_date=args.start_date,
        end_date=args.end_date,
        dtype="float32",
        prefer_parquet=True,
    )
    prices, filter_stats = filter_prices_by_window_eligibility(
        prices,
        min_price_coverage=float(args.min_price_coverage),
        max_flat_return_rate=float(args.max_flat_return_rate),
        min_stocks=int(args.min_eligible_stocks if formal else 2),
    )
    if not formal:
        prices = prices.iloc[:, : int(args.smoke_stocks)].copy()
    if formal and int(prices.shape[1]) != 2773:
        raise RuntimeError(f"the formal operator ablation requires exactly 2773 stocks after filtering, got {prices.shape[1]}")
    returns = preprocess_returns_no_metadata(
        prices,
        winsor_limit=0.03,
        do_vol_zscore=True,
        return_mode="raw",
        dtype="float32",
    )
    label_tables = {level: load_shenwan_label_table(paths.shenwan_universe, level=level) for level in SW_LEVELS}
    manifest["price_filter"] = filter_stats
    manifest["price_shape"] = list(prices.shape)
    manifest["return_shape"] = list(returns.shape)
    manifest["return_start"] = str(returns.index.min().date())
    manifest["return_end"] = str(returns.index.max().date())

    sim_cache: dict[int, tuple[list[np.ndarray], list[str], dict[str, float]]] = {}
    sim_cache[2] = build_level_similarities(returns, args, 2)
    baseline_score_cache = None
    if cache_scores:
        baseline_score_cache = build_signed_dual_score_cache(
            sim_cache[2][0],
            neg_weight=1.25,
            interaction="sqrt",
        )
    baseline_row, baseline_assignments = run_operator_partition(
        sim_mats=sim_cache[2][0],
        stock_names=sim_cache[2][1],
        label_tables=label_tables,
        level=2,
        operator="max_weighted",
        seed=args.seed,
        score_cache=baseline_score_cache,
    )
    baseline_row.update(sim_cache[2][2])
    baseline_path = assignments_dir / "level_2_max_weighted.csv"
    baseline_assignments.to_csv(baseline_path, index=False, encoding="utf-8-sig")
    baseline_row["assignments_path"] = str(baseline_path)
    if formal:
        gate = validate_reference_gate(baseline_row, baseline_assignments, paths.archived_assignment)
    else:
        gate = {"passed": None, "status": "smoke_not_formal", "reason": "stock universe was truncated"}
    write_json(paths.output_dir / "baseline_gate.json", gate)
    manifest["baseline_gate"] = gate
    write_json(paths.output_dir / "run_manifest.json", manifest)
    if formal and not gate["passed"]:
        pd.DataFrame([baseline_row]).to_csv(metrics_path, index=False, encoding="utf-8-sig")
        write_json(
            paths.output_dir / "evidence_index.json",
            {
                "supplement": "operator_ablation",
                "experiment": "maximum_operator_ablation",
                "baseline_gate": str(paths.output_dir / "baseline_gate.json"),
                "metrics": str(metrics_path),
                "assignments": str(assignments_dir),
                "candidate_placement": ["appendix", "supplementary_materials"],
                "status": "blocked_reference_gate",
                "blocker": "The current data lineage does not reproduce the archived J=2 reference.",
            },
        )
        manifest["failures"].append(
            {
                "stage": "reference_gate",
                "reason": "the operator reference baseline did not reproduce the archived partition and metrics",
            }
        )
        manifest["status"] = "blocked_reference_gate"
        manifest["finished_at"] = datetime.now().astimezone().isoformat()
        manifest["seconds"] = float(time.time() - started)
        manifest["artifacts"] = ["metrics.csv", "baseline_gate.json", "evidence_index.json", "assignments/"]
        write_json(paths.output_dir / "run_manifest.json", manifest)
        raise RuntimeError(f"operator reference gate failed: {gate}")
    if args.baseline_only:
        pd.DataFrame([baseline_row]).to_csv(metrics_path, index=False, encoding="utf-8-sig")
        manifest["finished_at"] = datetime.now().astimezone().isoformat()
        manifest["seconds"] = float(time.time() - started)
        manifest["status"] = "baseline_only_complete"
        manifest["artifacts"] = ["metrics.csv", "baseline_gate.json", "assignments/"]
        write_json(paths.output_dir / "run_manifest.json", manifest)
        return 0

    if 2 not in requested_levels:
        sim_cache.clear()
        baseline_score_cache = None

    rows = []
    for level in requested_levels:
        if level == 2:
            sim_mats, stock_names, level_timing = sim_cache.pop(2)
            score_cache = baseline_score_cache
        else:
            sim_mats, stock_names, level_timing = build_level_similarities(returns, args, level)
            score_cache = (
                build_signed_dual_score_cache(
                    sim_mats,
                    neg_weight=1.25,
                    interaction="sqrt",
                )
                if cache_scores
                else None
            )
        for operator in args.operators:
            if level == 2 and operator == "max_weighted":
                row = dict(baseline_row)
                assignments = baseline_assignments
            else:
                row, assignments = run_operator_partition(
                    sim_mats=sim_mats,
                    stock_names=stock_names,
                    label_tables=label_tables,
                    level=level,
                    operator=operator,
                    seed=args.seed,
                    score_cache=score_cache,
                )
                row.update(level_timing)
                assignment_path = assignments_dir / f"level_{level}_{operator}.csv"
                assignments.to_csv(assignment_path, index=False, encoding="utf-8-sig")
                row["assignments_path"] = str(assignment_path)
            rows.append(row)
            print(
                f"__DS_PROGRESS__ {json.dumps({'experiment': 'operator_ablation', 'level': level, 'operator': operator, 'SW1_ari': row['SW1_ari']})}",
                flush=True,
            )
        del score_cache, sim_mats
        if level == 2:
            baseline_score_cache = None

    metrics = pd.DataFrame(rows).sort_values(["operator", "level"])
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    for metric in ("SW1_ari", "SW1_nmi", "SW2_ari", "SW2_nmi", "SW3_ari", "SW3_nmi"):
        metrics.pivot(index="level", columns="operator", values=metric).to_csv(
            paths.output_dir / f"{metric}_pivot.csv",
            encoding="utf-8-sig",
        )
    monotonicity = monotonicity_summary(metrics)
    monotonicity.to_csv(paths.output_dir / "monotonicity.csv", index=False, encoding="utf-8-sig")
    evidence_index = {
        "supplement": "operator_ablation",
        "experiment": "maximum_operator_ablation",
        "baseline_gate": str(paths.output_dir / "baseline_gate.json"),
        "metrics": str(metrics_path),
        "monotonicity": str(paths.output_dir / "monotonicity.csv"),
        "assignments": str(assignments_dir),
        "manifest": str(paths.output_dir / "run_manifest.json"),
        "configuration": {
            "levels": requested_levels,
            "operators": {
                operator: {
                    "reducer": OPERATOR_CONFIGS[operator][0],
                    "layer_weighting": OPERATOR_CONFIGS[operator][1],
                }
                for operator in args.operators
            },
            "reference_parameters": {
                "wavelet": "sym2",
                "k": 1.5,
                "neg_weight": 1.25,
                "gamma": 0.35,
                "n_clusters": 25,
                "seed": int(args.seed),
            },
        },
        "key_conclusion": {
            "SW1_ari_level_decay_classification": monotonicity.set_index("operator")["classification"].to_dict(),
        },
        "candidate_placement": ["appendix", "supplementary_materials"],
        "status": "complete" if len(metrics) == len(requested_levels) * len(args.operators) else "partial",
    }
    write_json(paths.output_dir / "evidence_index.json", evidence_index)
    manifest["finished_at"] = datetime.now().astimezone().isoformat()
    manifest["seconds"] = float(time.time() - started)
    manifest["n_metrics"] = int(len(metrics))
    manifest["status"] = evidence_index["status"]
    manifest["conclusion"] = evidence_index["key_conclusion"]
    manifest["artifacts"] = [
        "metrics.csv",
        "baseline_gate.json",
        "monotonicity.csv",
        "evidence_index.json",
        "assignments/",
    ]
    write_json(paths.output_dir / "run_manifest.json", manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
