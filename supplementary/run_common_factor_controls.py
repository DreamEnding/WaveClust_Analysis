from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from supplementary.run_operator_ablation import environment_snapshot, write_json
from waveclust.data import load_price_panel, load_stock_basic, normalize_stock_code
from waveclust.dependence import rank_normalize_rows_for_spearman, summarize_cross_band_dependence
from waveclust.factors import (
    FactorBranchResult,
    neutralize_by_industry_strict,
    prepare_factor_branches,
    select_factor_common_universe,
)
from waveclust.model import StockWaveClust, WaveClustParams
from waveclust.preprocessing import winsorize_df, zscore_df
from waveclust.pure_price import filter_prices_by_window_eligibility, preprocess_returns_no_metadata
from waveclust.shenwan import SW_LEVELS, evaluate_assignments, load_shenwan_label_table
from waveclust.spectral import (
    build_signed_dual_waveclust_score,
    cluster_stats,
    labels_to_assignments,
    run_price_clusterer,
)


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
MAIN_BRANCHES = (
    "raw",
    "market_residual_internal",
    "sequential_market_industry_adjusted",
)
FIGURE3_LEVEL = 4
FIGURE3_WAVELET = "coif1"
FIGURE3_WINSOR_LIMIT = 0.035
FIGURE3_ARCHIVED_MEAN_SPEARMAN = 0.7689447104930878
FIGURE3_MATRIX_TOLERANCE = 5e-6


@dataclass(frozen=True)
class ResolvedPaths:
    workspace_root: Path
    data_panel: Path
    stock_basic: Path
    shenwan_universe: Path
    archived_figure3_dir: Path
    operator_baseline_gate: Path
    output_dir: Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the supplementary common-factor controls.")
    parser.add_argument("--workspace-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--data-panel", type=Path, default=None)
    parser.add_argument("--stock-basic", type=Path, default=None)
    parser.add_argument("--shenwan-universe", type=Path, default=None)
    parser.add_argument("--archived-figure3-dir", type=Path, default=None)
    parser.add_argument("--operator-baseline-gate", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--levels", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    parser.add_argument("--start-date", default="2019-01-01")
    parser.add_argument("--end-date", default="2025-12-31")
    parser.add_argument("--min-price-coverage", type=float, default=0.95)
    parser.add_argument("--max-flat-return-rate", type=float, default=0.05)
    parser.add_argument("--min-eligible-stocks", type=int, default=1000)
    parser.add_argument("--smoke-stocks", type=int, default=0)
    parser.add_argument("--gpu-id", type=int, default=2)
    parser.add_argument("--gpu-ids", type=int, nargs="+", default=None)
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--wavelet-workers", type=int, default=1)
    parser.add_argument("--dependence-bins", type=int, default=64)
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
        stock_basic=_resolve(root, args.stock_basic, "DATA/stock_basic.csv"),
        shenwan_universe=_resolve(root, args.shenwan_universe, "DATA/tickflow_universes/universe_list.json"),
        archived_figure3_dir=_resolve(
            root,
            args.archived_figure3_dir,
            "analysis/waveclust_dense_signed_full_grid_20260525/"
            "ca4_cd1234_spearman_matrices_20260527",
        ),
        operator_baseline_gate=_resolve(
            root,
            args.operator_baseline_gate,
            "output/supplementary/operator_ablation/formal/baseline_gate.json",
        ),
        output_dir=_resolve(root, args.output_dir, str(args.output_dir)),
    )


def band_names(level: int) -> list[str]:
    return [f"CA{int(level)}", *[f"CD{detail}" for detail in range(int(level), 0, -1)]]


def _make_params(
    args: argparse.Namespace,
    *,
    level: int,
    wavelet: str,
    use_gpu: bool | None = None,
) -> WaveClustParams:
    gpu_ids = tuple(int(device) for device in (args.gpu_ids or [args.gpu_id]))
    gpu_enabled = not args.no_gpu if use_gpu is None else bool(use_gpu)
    return WaveClustParams(
        wavelet_transform="modwt",
        wavelet=str(wavelet),
        levels=int(level),
        k=1.5,
        q_threshold=0.996,
        winsor_limits=(0.03, 0.03),
        do_vol_zscore=True,
        use_gpu=gpu_enabled,
        dtype="float32",
        gpu_ids=gpu_ids,
        multi_gpu_similarity=gpu_enabled and len(gpu_ids) > 1,
        similarity_workers=len(gpu_ids),
        wavelet_workers=int(args.wavelet_workers),
        mcl_backend="cpu",
    )


def build_band_similarities(
    returns: pd.DataFrame,
    args: argparse.Namespace,
    *,
    level: int,
    wavelet: str,
    spearman: bool,
    use_gpu: bool | None = None,
) -> tuple[list[np.ndarray], list[str], dict[str, float]]:
    model = StockWaveClust(
        prices=returns,
        stock_info=None,
        params=_make_params(args, level=level, wavelet=wavelet, use_gpu=use_gpu),
    )
    gpu_requested = not args.no_gpu if use_gpu is None else bool(use_gpu)
    if gpu_requested and not model.use_gpu:
        raise RuntimeError(
            "GPU execution requested but CuPy cannot see a CUDA device. Run the trade interpreter with user site "
            "disabled and restore /dev/nvidia* before a formal run."
        )
    model.returns = returns.astype("float32")
    wavelet_started = time.perf_counter()
    model.decompose_wavelets()
    level_matrices, stock_names = model.prepare_level_matrices()
    model.release_wavelet_coefficients()
    if spearman:
        level_matrices = [rank_normalize_rows_for_spearman(matrix) for matrix in level_matrices]
    wavelet_seconds = time.perf_counter() - wavelet_started
    similarity_started = time.perf_counter()
    similarities = model.compute_similarity_matrices(level_matrices)
    for matrix in similarities:
        if spearman:
            np.fill_diagonal(matrix, 1.0)
        else:
            np.fill_diagonal(matrix, 0.0)
    similarity_seconds = time.perf_counter() - similarity_started
    return similarities, stock_names, {
        "wavelet_seconds": float(wavelet_seconds),
        "similarity_seconds": float(similarity_seconds),
    }


def prepare_legacy_industry_only(
    prices: pd.DataFrame,
    industries: pd.Series,
    *,
    winsor_limit: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    filled = prices.ffill()
    log_returns = np.log(filled / filled.shift(1))
    log_returns = log_returns.replace([np.inf, -np.inf], np.nan)
    log_returns = log_returns.dropna(how="all", axis=0).dropna(how="all", axis=1).fillna(0.0)
    if log_returns.columns.tolist() != prices.columns.tolist():
        raise ValueError("legacy branch must preserve the frozen stock universe")
    winsorized = winsorize_df(
        log_returns,
        limits=(float(winsor_limit), float(winsor_limit)),
    )
    neutralized, projection = neutralize_by_industry_strict(winsorized, industries)
    standardized = zscore_df(neutralized).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return standardized.astype("float32"), projection


def mean_off_diagonal(frame: pd.DataFrame) -> float:
    values = frame.to_numpy(dtype=np.float64)
    upper = values[np.triu_indices(len(values), k=1)]
    return float(np.mean(upper))


def _matrix_summary(branch: str, band: str, matrix: np.ndarray, matrix_path: Path) -> dict[str, Any]:
    upper = matrix[np.triu_indices(matrix.shape[0], k=1)]
    quantiles = np.quantile(upper, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    return {
        "branch": branch,
        "band": band,
        "n_stocks": int(matrix.shape[0]),
        "matrix_path": str(matrix_path),
        "dtype": str(matrix.dtype),
        "diagonal_min": float(np.diag(matrix).min()),
        "diagonal_max": float(np.diag(matrix).max()),
        "upper_mean": float(np.mean(upper)),
        "upper_std": float(np.std(upper)),
        "upper_min": float(np.min(upper)),
        "upper_max": float(np.max(upper)),
        **{
            f"upper_q{int(q * 100):02d}": float(value)
            for q, value in zip(
                [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99],
                quantiles,
                strict=True,
            )
        },
    }


def save_figure3_branch(
    *,
    branch: str,
    similarities: list[np.ndarray],
    names: list[str],
    output_dir: Path,
    n_bins: int,
    provenance: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    branch_dir = output_dir / branch
    matrix_dir = branch_dir / "matrices_npy"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    matrix_rows = []
    for band, matrix in zip(names, similarities, strict=True):
        matrix_path = matrix_dir / f"{band}_stock_spearman.npy"
        np.save(matrix_path, np.asarray(matrix, dtype=np.float32), allow_pickle=False)
        matrix_rows.append(_matrix_summary(branch, band, matrix, matrix_path))
    dependence = summarize_cross_band_dependence(similarities, band_names=names, n_bins=int(n_bins))
    dependence.spearman.to_csv(branch_dir / "cross_band_spearman.csv", encoding="utf-8-sig")
    dependence.mi.to_csv(branch_dir / "cross_band_mi.csv", encoding="utf-8-sig")
    dependence.nmi.to_csv(branch_dir / "cross_band_nmi.csv", encoding="utf-8-sig")
    dependence.vi.to_csv(branch_dir / "cross_band_vi.csv", encoding="utf-8-sig")
    summary = {
        "branch": branch,
        "provenance": provenance,
        "n_stocks": int(similarities[0].shape[0]),
        "bands": names,
        "n_bins": int(n_bins),
        "mean_off_diagonal_spearman": mean_off_diagonal(dependence.spearman),
        "mean_off_diagonal_mi": mean_off_diagonal(dependence.mi),
        "mean_off_diagonal_nmi": mean_off_diagonal(dependence.nmi),
        "mean_off_diagonal_vi": mean_off_diagonal(dependence.vi),
    }
    write_json(branch_dir / "summary.json", summary)
    return summary, matrix_rows


def validate_archived_figure3(
    *,
    current: list[np.ndarray],
    current_stocks: list[str],
    archive_dir: Path,
) -> dict[str, Any]:
    names = band_names(FIGURE3_LEVEL)
    archived_order = pd.read_csv(archive_dir / "stock_order.csv", dtype={"stock_code": str})
    archived_stocks = archived_order["stock_code"].map(normalize_stock_code).tolist()
    stock_order_exact = current_stocks == archived_stocks
    matrix_checks: dict[str, Any] = {}
    for band, observed in zip(names, current, strict=True):
        expected = np.load(archive_dir / "matrices_npy" / f"{band}_stock_spearman.npy", mmap_mode="r")
        shape_exact = observed.shape == expected.shape
        max_abs = float(np.max(np.abs(observed - expected))) if shape_exact else float("inf")
        matrix_checks[band] = {
            "shape_exact": bool(shape_exact),
            "max_abs_difference": max_abs,
            "within_tolerance": bool(shape_exact and max_abs <= FIGURE3_MATRIX_TOLERANCE),
        }
    current_dependence = summarize_cross_band_dependence(current, band_names=names, n_bins=64).spearman
    archived_dependence = pd.read_csv(archive_dir / "cross_band_matrix_spearman.csv", index_col=0)
    archived_dependence = archived_dependence.loc[names, names]
    cross_band_max_abs = float(
        np.max(
            np.abs(
                current_dependence.to_numpy(dtype=np.float64)
                - archived_dependence.to_numpy(dtype=np.float64)
            )
        )
    )
    mean_spearman = mean_off_diagonal(current_dependence)
    checks = {
        "stock_order_exact": bool(stock_order_exact),
        "all_band_matrices_within_tolerance": bool(
            all(item["within_tolerance"] for item in matrix_checks.values())
        ),
        "cross_band_matrix_within_tolerance": bool(cross_band_max_abs <= FIGURE3_MATRIX_TOLERANCE),
        "mean_off_diagonal_within_tolerance": bool(
            abs(mean_spearman - FIGURE3_ARCHIVED_MEAN_SPEARMAN) <= FIGURE3_MATRIX_TOLERANCE
        ),
    }
    return {
        "passed": bool(all(checks.values())),
        "tolerance": FIGURE3_MATRIX_TOLERANCE,
        "checks": checks,
        "matrix_checks": matrix_checks,
        "observed_mean_off_diagonal_spearman": mean_spearman,
        "expected_mean_off_diagonal_spearman": FIGURE3_ARCHIVED_MEAN_SPEARMAN,
        "cross_band_max_abs_difference": cross_band_max_abs,
        "current_stock_count": len(current_stocks),
        "archived_stock_count": len(archived_stocks),
    }


def validate_operator_gate(path: Path, *, formal: bool) -> dict[str, Any]:
    if not formal:
        return {"passed": None, "status": "smoke_not_formal"}
    if not path.exists():
        raise FileNotFoundError(f"formal common-factor controls require the accepted operator baseline gate: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("passed") is not True:
        raise RuntimeError(f"operator reference gate is not accepted: {payload}")
    return payload


def run_cluster_partition(
    *,
    similarities: list[np.ndarray],
    stock_names: list[str],
    label_tables: dict[str, pd.DataFrame],
    branch: str,
    level: int,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    score_started = time.perf_counter()
    score = build_signed_dual_waveclust_score(
        similarities,
        k=1.5,
        neg_weight=1.25,
        reducer="max",
        layer_weighting=True,
        interaction="sqrt",
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
        "experiment": "common_factor_controls",
        "branch": branch,
        "level": int(level),
        "wavelet": "sym2",
        "winsor_limit": 0.03,
        "operator": "max_weighted",
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
    return row, assignments


def _save_factor_diagnostics(
    result: FactorBranchResult,
    *,
    output_dir: Path,
    suffix: str,
) -> None:
    diagnostic_dir = output_dir / "factor_diagnostics"
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    for branch, frame in result.per_stock_diagnostics.items():
        out = frame.copy()
        out.index.name = "stock"
        out.to_csv(diagnostic_dir / f"{branch}_{suffix}.csv", encoding="utf-8-sig")
    write_json(diagnostic_dir / f"summary_{suffix}.json", result.summary)


def _relative_figure3_summary(summaries: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(summaries)
    raw_row = frame.loc[frame["branch"] == "raw"].iloc[0]
    for metric in ("spearman", "mi", "nmi", "vi"):
        column = f"mean_off_diagonal_{metric}"
        raw_value = float(raw_row[column])
        frame[f"delta_vs_raw_{metric}"] = frame[column].astype(float) - raw_value
        frame[f"retention_vs_raw_{metric}"] = (
            frame[column].astype(float) / raw_value if raw_value != 0 else np.nan
        )
    return frame


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    invalid_levels = sorted(set(args.levels).difference(range(2, 7)))
    if invalid_levels:
        raise ValueError(f"levels must be in [2, 6], got {invalid_levels}")
    if int(args.seed) != 42:
        raise ValueError("common-factor control runs use the preregistered seed 42")
    if int(args.dependence_bins) < 2:
        raise ValueError("dependence-bins must be at least two")

    paths = resolve_paths(args)
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = paths.output_dir / "level_metrics.csv"
    if metrics_path.exists() and not args.force:
        raise FileExistsError(f"metrics already exist: {metrics_path}; use a new output directory or --force")
    started = time.time()
    formal = int(args.smoke_stocks) <= 0
    external_gate: dict[str, Any] = {"passed": None, "status": "not_checked"}
    manifest: dict[str, Any] = {
        "run_id": paths.output_dir.name,
        "experiment": "common_factor_controls",
        "started_at": datetime.fromtimestamp(started).astimezone().isoformat(),
        "command": " ".join(sys.argv if argv is None else [sys.argv[0], *argv]),
        "paths": asdict(paths),
        "formal": formal,
        "environment": environment_snapshot(),
        "operator_reference_gate": external_gate,
        "contract": {
            "levels": sorted(set(int(level) for level in args.levels)),
            "main_branches": list(MAIN_BRANCHES),
            "figure3_legacy_branch": "legacy_industry_only",
            "start_date": args.start_date,
            "end_date": args.end_date,
            "min_price_coverage": args.min_price_coverage,
            "max_flat_return_rate": args.max_flat_return_rate,
            "min_eligible_stocks": args.min_eligible_stocks,
            "smoke_stocks": args.smoke_stocks,
            "gpu_ids": list(args.gpu_ids or [args.gpu_id]),
            "use_gpu": not args.no_gpu,
            "figure3_similarity_device": "cpu",
            "spectral_jobs": 1,
            "wavelet_workers": args.wavelet_workers,
            "dependence_bins": args.dependence_bins,
            "seed": args.seed,
        },
        "failures": [],
    }
    write_json(paths.output_dir / "run_manifest.json", manifest)
    try:
        external_gate = validate_operator_gate(paths.operator_baseline_gate, formal=formal)
    except (FileNotFoundError, RuntimeError) as exc:
        write_json(
            paths.output_dir / "evidence_index.json",
            {
                "supplement": "common_factor_controls",
                "experiment": "hierarchical_common_factor_controls",
                "operator_reference_gate": str(paths.operator_baseline_gate),
                "candidate_placement": ["main_text", "appendix", "supplementary_materials"],
                "status": "blocked_operator_reference_gate",
                "blocker": str(exc),
            },
        )
        manifest["operator_reference_gate"] = {"passed": False, "error": str(exc)}
        manifest["failures"].append(
            {
                "stage": "operator_reference_gate",
                "reason": str(exc),
            }
        )
        manifest["status"] = "blocked_operator_reference_gate"
        manifest["finished_at"] = datetime.now().astimezone().isoformat()
        manifest["seconds"] = float(time.time() - started)
        manifest["artifacts"] = ["evidence_index.json"]
        write_json(paths.output_dir / "run_manifest.json", manifest)
        raise
    manifest["operator_reference_gate"] = external_gate
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
        raise RuntimeError(f"formal common-factor controls require exactly 2773 stocks after raw filtering, got {prices.shape[1]}")

    stock_info = load_stock_basic(paths.stock_basic)
    universe = select_factor_common_universe(prices, stock_info, min_industry_size=2)
    universe.exclusions.to_csv(
        paths.output_dir / "factor_common_universe_exclusions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    common_order = [normalize_stock_code(stock) for stock in universe.prices.columns]
    pd.DataFrame(
        {
            "stock_index": np.arange(len(common_order)),
            "stock": common_order,
            "industry": universe.industries.to_numpy(),
        }
    ).to_csv(paths.output_dir / "factor_common_universe.csv", index=False, encoding="utf-8-sig")
    manifest["price_filter"] = filter_stats
    manifest["raw_price_shape"] = list(prices.shape)
    manifest["factor_common_universe_stock_count"] = int(universe.prices.shape[1])
    manifest["factor_common_universe_exclusion_count"] = int(len(universe.exclusions))
    write_json(paths.output_dir / "run_manifest.json", manifest)

    raw_returns_figure3 = preprocess_returns_no_metadata(
        prices,
        winsor_limit=FIGURE3_WINSOR_LIMIT,
        do_vol_zscore=True,
        return_mode="raw",
        dtype="float32",
    )
    raw_figure3_matrices, raw_stock_names, raw_figure3_timing = build_band_similarities(
        raw_returns_figure3,
        args,
        level=FIGURE3_LEVEL,
        wavelet=FIGURE3_WAVELET,
        spearman=True,
        use_gpu=False,
    )
    raw_original_summary, raw_original_matrix_rows = save_figure3_branch(
        branch="raw_original_universe",
        similarities=raw_figure3_matrices,
        names=band_names(FIGURE3_LEVEL),
        output_dir=paths.output_dir / "figure3",
        n_bins=int(args.dependence_bins),
        provenance="archived Figure 3 raw-universe reproduction",
    )
    if formal:
        figure3_gate = validate_archived_figure3(
            current=raw_figure3_matrices,
            current_stocks=[normalize_stock_code(stock) for stock in raw_stock_names],
            archive_dir=paths.archived_figure3_dir,
        )
    else:
        figure3_gate = {
            "passed": None,
            "status": "smoke_not_formal",
            "reason": "stock universe was truncated",
            "observed_mean_off_diagonal_spearman": raw_original_summary["mean_off_diagonal_spearman"],
        }
    figure3_gate["timing"] = raw_figure3_timing
    write_json(paths.output_dir / "figure3_baseline_gate.json", figure3_gate)
    manifest["figure3_baseline_gate"] = figure3_gate
    write_json(paths.output_dir / "run_manifest.json", manifest)
    if formal and not figure3_gate["passed"]:
        write_json(
            paths.output_dir / "evidence_index.json",
            {
                "supplement": "common_factor_controls",
                "experiment": "hierarchical_common_factor_controls",
                "operator_reference_gate": str(paths.operator_baseline_gate),
                "figure3_reference_gate": str(paths.output_dir / "figure3_baseline_gate.json"),
                "candidate_placement": ["main_text", "appendix", "supplementary_materials"],
                "status": "blocked_figure3_reference_gate",
                "blocker": "The current data lineage does not reproduce the archived raw Figure 3 reference.",
            },
        )
        manifest["failures"].append(
            {
                "stage": "figure3_reference_gate",
                "reason": "the raw Figure 3 baseline did not reproduce the archived matrices and diagnostics",
            }
        )
        manifest["status"] = "blocked_figure3_reference_gate"
        manifest["finished_at"] = datetime.now().astimezone().isoformat()
        manifest["seconds"] = float(time.time() - started)
        manifest["artifacts"] = ["figure3_baseline_gate.json", "evidence_index.json", "figure3/"]
        write_json(paths.output_dir / "run_manifest.json", manifest)
        raise RuntimeError(f"Figure 3 reference gate failed: {figure3_gate}")
    if args.baseline_only:
        pd.DataFrame(raw_original_matrix_rows).to_csv(
            paths.output_dir / "figure3" / "band_matrix_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )
        manifest["finished_at"] = datetime.now().astimezone().isoformat()
        manifest["seconds"] = float(time.time() - started)
        manifest["status"] = "baseline_only_complete"
        write_json(paths.output_dir / "run_manifest.json", manifest)
        return 0

    factor_figure3 = prepare_factor_branches(
        universe.prices,
        universe.industries,
        winsor_limit=FIGURE3_WINSOR_LIMIT,
        dtype="float32",
    )
    expected_raw_common = raw_returns_figure3.loc[:, universe.prices.columns]
    observed_raw_common = factor_figure3.returns["raw"]
    raw_common_returns_max_abs = float(
        np.max(
            np.abs(
                expected_raw_common.to_numpy(dtype=np.float32)
                - observed_raw_common.to_numpy(dtype=np.float32)
            )
        )
    )
    if not expected_raw_common.index.equals(observed_raw_common.index) or raw_common_returns_max_abs > 1e-7:
        raise RuntimeError(
            "raw common-universe returns are not equivalent to the accepted raw-universe subset: "
            f"max_abs={raw_common_returns_max_abs}"
        )
    manifest["raw_common_subset_invariant"] = {
        "index_exact": True,
        "max_abs_difference": raw_common_returns_max_abs,
        "passed": True,
    }
    write_json(paths.output_dir / "run_manifest.json", manifest)

    raw_position = {normalize_stock_code(stock): index for index, stock in enumerate(raw_stock_names)}
    missing_from_raw = [stock for stock in common_order if stock not in raw_position]
    if missing_from_raw:
        raise RuntimeError(f"common universe is not a subset of raw Figure 3 stocks: {missing_from_raw[:5]}")
    common_positions = np.array([raw_position[stock] for stock in common_order], dtype=np.int64)
    raw_common_matrices = [
        matrix[np.ix_(common_positions, common_positions)].astype(np.float32, copy=False)
        for matrix in raw_figure3_matrices
    ]
    raw_common_summary, raw_common_matrix_rows = save_figure3_branch(
        branch="raw",
        similarities=raw_common_matrices,
        names=band_names(FIGURE3_LEVEL),
        output_dir=paths.output_dir / "figure3",
        n_bins=int(args.dependence_bins),
        provenance="factor_common_universe comparator sliced from accepted raw-universe matrices",
    )
    del raw_figure3_matrices, raw_common_matrices

    _save_factor_diagnostics(
        factor_figure3,
        output_dir=paths.output_dir,
        suffix="figure3_winsor_0035",
    )
    figure3_summaries = [raw_common_summary]
    matrix_rows = [*raw_original_matrix_rows, *raw_common_matrix_rows]
    for branch in MAIN_BRANCHES[1:]:
        matrices, branch_stocks, timing = build_band_similarities(
            factor_figure3.returns[branch],
            args,
            level=FIGURE3_LEVEL,
            wavelet=FIGURE3_WAVELET,
            spearman=True,
            use_gpu=False,
        )
        if [normalize_stock_code(stock) for stock in branch_stocks] != common_order:
            raise RuntimeError(f"Figure 3 stock order drifted in branch {branch}")
        summary, rows = save_figure3_branch(
            branch=branch,
            similarities=matrices,
            names=band_names(FIGURE3_LEVEL),
            output_dir=paths.output_dir / "figure3",
            n_bins=int(args.dependence_bins),
            provenance="preregistered main factor branch",
        )
        summary.update(timing)
        figure3_summaries.append(summary)
        matrix_rows.extend(rows)
        del matrices

    legacy_returns, legacy_projection = prepare_legacy_industry_only(
        universe.prices,
        universe.industries,
        winsor_limit=FIGURE3_WINSOR_LIMIT,
    )
    legacy_matrices, legacy_stocks, legacy_timing = build_band_similarities(
        legacy_returns,
        args,
        level=FIGURE3_LEVEL,
        wavelet=FIGURE3_WAVELET,
        spearman=True,
        use_gpu=False,
    )
    if [normalize_stock_code(stock) for stock in legacy_stocks] != common_order:
        raise RuntimeError("Figure 3 stock order drifted in legacy_industry_only")
    legacy_summary, legacy_rows = save_figure3_branch(
        branch="legacy_industry_only",
        similarities=legacy_matrices,
        names=band_names(FIGURE3_LEVEL),
        output_dir=paths.output_dir / "figure3",
        n_bins=int(args.dependence_bins),
        provenance=(
            "legacy winsorize -> industry projection -> stock z-score order; strict common universe makes the "
            "historical projection algebraically identical while excluding historical silent fallbacks"
        ),
    )
    legacy_summary.update(legacy_timing)
    legacy_summary.update(legacy_projection)
    write_json(
        paths.output_dir / "figure3" / "legacy_industry_only" / "summary.json",
        legacy_summary,
    )
    figure3_summaries.append(legacy_summary)
    matrix_rows.extend(legacy_rows)
    del legacy_matrices, legacy_returns

    figure3_summary = _relative_figure3_summary(figure3_summaries)
    figure3_summary.to_csv(
        paths.output_dir / "figure3" / "summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(matrix_rows).to_csv(
        paths.output_dir / "figure3" / "band_matrix_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    factor_cluster = prepare_factor_branches(
        universe.prices,
        universe.industries,
        winsor_limit=0.03,
        dtype="float32",
    )
    _save_factor_diagnostics(
        factor_cluster,
        output_dir=paths.output_dir,
        suffix="clustering_winsor_003",
    )
    label_tables = {
        level: load_shenwan_label_table(paths.shenwan_universe, level=level)
        for level in SW_LEVELS
    }
    assignments_dir = paths.output_dir / "assignments"
    assignments_dir.mkdir(exist_ok=True)
    rows: list[dict[str, Any]] = []
    for branch in MAIN_BRANCHES:
        returns = factor_cluster.returns[branch]
        for level in sorted(set(int(level) for level in args.levels)):
            similarities, stock_names, timing = build_band_similarities(
                returns,
                args,
                level=level,
                wavelet="sym2",
                spearman=False,
            )
            if [normalize_stock_code(stock) for stock in stock_names] != common_order:
                raise RuntimeError(f"clustering stock order drifted in branch={branch}, level={level}")
            row, assignments = run_cluster_partition(
                similarities=similarities,
                stock_names=stock_names,
                label_tables=label_tables,
                branch=branch,
                level=level,
                seed=int(args.seed),
            )
            row.update(timing)
            assignment_path = assignments_dir / f"{branch}_level_{level}.csv"
            assignments.to_csv(assignment_path, index=False, encoding="utf-8-sig")
            row["assignments_path"] = str(assignment_path)
            rows.append(row)
            print(
                f"__DS_PROGRESS__ {json.dumps({'experiment': 'common_factor_controls', 'branch': branch, 'level': level, 'SW1_ari': row['SW1_ari']})}",
                flush=True,
            )
            del similarities

    metrics = pd.DataFrame(rows).sort_values(["branch", "level"])
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    for metric in ("SW1_ari", "SW1_nmi", "SW2_ari", "SW2_nmi", "SW3_ari", "SW3_nmi"):
        metrics.pivot(index="level", columns="branch", values=metric).to_csv(
            paths.output_dir / f"{metric}_pivot.csv",
            encoding="utf-8-sig",
        )

    evidence_index = {
        "supplement": "common_factor_controls",
        "experiment": "hierarchical_common_factor_controls",
        "operator_reference_gate": str(paths.operator_baseline_gate),
        "figure3_reference_gate": str(paths.output_dir / "figure3_baseline_gate.json"),
        "figure3_summary": str(paths.output_dir / "figure3" / "summary.csv"),
        "level_metrics": str(metrics_path),
        "assignments": str(assignments_dir),
        "factor_diagnostics": str(paths.output_dir / "factor_diagnostics"),
        "exclusions": str(paths.output_dir / "factor_common_universe_exclusions.csv"),
        "manifest": str(paths.output_dir / "run_manifest.json"),
        "configuration": {
            "levels": sorted(set(int(level) for level in args.levels)),
            "main_branches": list(MAIN_BRANCHES),
            "figure3": {
                "wavelet": FIGURE3_WAVELET,
                "level": FIGURE3_LEVEL,
                "winsor_limit": FIGURE3_WINSOR_LIMIT,
                "dependence_bins": int(args.dependence_bins),
            },
            "clustering": {
                "wavelet": "sym2",
                "operator": "max_weighted",
                "k": 1.5,
                "neg_weight": 1.25,
                "gamma": 0.35,
                "n_clusters": 25,
                "seed": int(args.seed),
            },
        },
        "key_conclusion": {
            "figure3_mean_off_diagonal_spearman": {
                summary["branch"]: float(summary["mean_off_diagonal_spearman"])
                for summary in figure3_summaries
            },
            "supplementary_claim": (
                "Cross-frequency dependence overlap only; title-level common-factor conclusions require the "
                "the additional preregistered robustness evidence."
            ),
        },
        "candidate_placement": ["main_text", "appendix", "supplementary_materials"],
        "status": "complete" if len(metrics) == len(args.levels) * len(MAIN_BRANCHES) else "partial",
        "claim_status": (
            "The common-factor controls report cross-frequency dependence overlap only; the title-level claim also "
            "requires bootstrap, bin-robustness, and single-band partition evidence."
        ),
    }
    write_json(paths.output_dir / "evidence_index.json", evidence_index)
    manifest["finished_at"] = datetime.now().astimezone().isoformat()
    manifest["seconds"] = float(time.time() - started)
    manifest["n_level_metrics"] = int(len(metrics))
    manifest["status"] = evidence_index["status"]
    manifest["conclusion"] = evidence_index["key_conclusion"]
    manifest["artifacts"] = [
        "figure3_baseline_gate.json",
        "figure3/",
        "level_metrics.csv",
        "assignments/",
        "factor_diagnostics/",
        "factor_common_universe.csv",
        "factor_common_universe_exclusions.csv",
        "evidence_index.json",
    ]
    write_json(paths.output_dir / "run_manifest.json", manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
