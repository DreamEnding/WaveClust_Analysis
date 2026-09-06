from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from supplementary.run_common_factor_controls import MAIN_BRANCHES, build_band_similarities, run_cluster_partition
from supplementary.run_operator_ablation import environment_snapshot, write_json
from waveclust.data import load_price_panel, load_stock_basic, normalize_stock_code
from waveclust.factors import prepare_factor_branches, select_factor_common_universe
from waveclust.pure_price import filter_prices_by_window_eligibility
from waveclust.shenwan import SW_LEVELS, load_shenwan_label_table


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
J1_DEFINITION = "one-level MODWT using CA1 and CD1"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the supplementary one-level MODWT analysis.")
    parser.add_argument("--workspace-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--data-panel", type=Path, default=None)
    parser.add_argument("--stock-basic", type=Path, default=None)
    parser.add_argument("--shenwan-universe", type=Path, default=None)
    parser.add_argument("--reference-run", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--wavelet-workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def resolve_path(root: Path, value: Path | None, default: str) -> Path:
    path = Path(default) if value is None else value
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_reference_run(reference_run: Path) -> dict[str, Any]:
    manifest_path = reference_run / "run_manifest.json"
    metrics_path = reference_run / "level_metrics.csv"
    universe_path = reference_run / "factor_common_universe.csv"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metrics = pd.read_csv(metrics_path)
    checks = {
        "formal_complete": manifest.get("formal") is True and manifest.get("status") == "complete",
        "operator_gate_passed": manifest.get("operator_reference_gate", {}).get("passed") is True,
        "figure3_gate_passed": manifest.get("figure3_baseline_gate", {}).get("passed") is True,
        "levels_2_to_6": sorted(metrics["level"].astype(int).unique().tolist()) == [2, 3, 4, 5, 6],
        "three_main_branches": set(metrics["branch"]) == set(MAIN_BRANCHES),
        "fifteen_rows": len(metrics) == 15,
        "n_stocks_2770": set(metrics["n_stocks"].astype(int)) == {2770},
        "fixed_configuration": (
            set(metrics["wavelet"]) == {"sym2"}
            and set(metrics["operator"]) == {"max_weighted"}
            and set(metrics["k"].astype(float)) == {1.5}
            and set(metrics["neg_weight"].astype(float)) == {1.25}
            and set(metrics["gamma"].astype(float)) == {0.35}
            and set(metrics["n_clusters_requested"].astype(int)) == {25}
            and set(metrics["seed"].astype(int)) == {42}
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"reference common-factor run failed validation: {checks}")
    return {
        "checks": checks,
        "hashes": {
            "run_manifest.json": sha256(manifest_path),
            "level_metrics.csv": sha256(metrics_path),
            "factor_common_universe.csv": sha256(universe_path),
        },
        "contract": manifest["contract"],
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if int(args.seed) != 42:
        raise ValueError("the single-level analysis uses the preregistered seed 42")
    root = args.workspace_root.resolve()
    data_panel = resolve_path(root, args.data_panel, "DATA/stock_price_panel.csv")
    stock_basic = resolve_path(root, args.stock_basic, "DATA/stock_basic.csv")
    shenwan_universe = resolve_path(root, args.shenwan_universe, "DATA/tickflow_universes/universe_list.json")
    reference_run = resolve_path(
        root,
        args.reference_run,
        "output/supplementary/common_factor_controls/formal",
    )
    output_dir = resolve_path(root, args.output_dir, str(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=False)

    started = time.time()
    reference = validate_reference_run(reference_run)
    contract = reference["contract"]
    manifest: dict[str, Any] = {
        "run_id": output_dir.name,
        "experiment": "single_level_common_factor_controls",
        "j1_definition": J1_DEFINITION,
        "started_at": datetime.fromtimestamp(started).astimezone().isoformat(),
        "command": " ".join(sys.argv if argv is None else [sys.argv[0], *argv]),
        "environment": environment_snapshot(),
        "reference_run": str(reference_run),
        "reference_validation": reference,
        "contract": {
            "level": 1,
            "bands": ["CA1", "CD1"],
            "branches": list(MAIN_BRANCHES),
            "start_date": contract["start_date"],
            "end_date": contract["end_date"],
            "min_price_coverage": contract["min_price_coverage"],
            "max_flat_return_rate": contract["max_flat_return_rate"],
            "min_eligible_stocks": contract["min_eligible_stocks"],
            "wavelet": "sym2",
            "winsor_limit": 0.03,
            "operator": "max_weighted",
            "k": 1.5,
            "neg_weight": 1.25,
            "gamma": 0.35,
            "n_clusters": 25,
            "assign_labels": "discretize",
            "seed": int(args.seed),
            "device": "cpu",
        },
    }
    write_json(output_dir / "run_manifest.json", manifest)

    prices = load_price_panel(
        data_panel,
        start_date=contract["start_date"],
        end_date=contract["end_date"],
        dtype="float32",
        prefer_parquet=False,
    )
    prices, filter_stats = filter_prices_by_window_eligibility(
        prices,
        min_price_coverage=float(contract["min_price_coverage"]),
        max_flat_return_rate=float(contract["max_flat_return_rate"]),
        min_stocks=int(contract["min_eligible_stocks"]),
    )
    if prices.shape[1] != 2773:
        raise RuntimeError(f"the formal single-level analysis requires 2773 filtered stocks, got {prices.shape[1]}")

    stock_info = load_stock_basic(stock_basic)
    universe = select_factor_common_universe(prices, stock_info, min_industry_size=2)
    current_universe = pd.DataFrame(
        {
            "stock": [normalize_stock_code(stock) for stock in universe.prices.columns],
            "industry": universe.industries.astype(str).to_numpy(),
        }
    )
    reference_universe = pd.read_csv(
        reference_run / "factor_common_universe.csv",
        dtype={"stock": str, "industry": str},
    )[["stock", "industry"]]
    reference_universe["stock"] = reference_universe["stock"].map(normalize_stock_code)
    if not current_universe.equals(reference_universe):
        raise RuntimeError("J=1 factor-common universe differs from the accepted common-factor run")
    current_universe.insert(0, "stock_index", np.arange(len(current_universe)))
    current_universe.to_csv(output_dir / "factor_common_universe.csv", index=False, encoding="utf-8-sig")

    factor_result = prepare_factor_branches(
        universe.prices,
        universe.industries,
        winsor_limit=0.03,
        dtype="float32",
    )
    label_tables = {
        level: load_shenwan_label_table(shenwan_universe, level=level)
        for level in SW_LEVELS
    }
    runtime_args = SimpleNamespace(
        gpu_ids=None,
        gpu_id=0,
        no_gpu=True,
        wavelet_workers=int(args.wavelet_workers),
    )
    assignments_dir = output_dir / "assignments"
    assignments_dir.mkdir()
    rows: list[dict[str, Any]] = []
    assignment_counts: dict[str, int] = {}
    common_order = current_universe["stock"].tolist()
    for branch in MAIN_BRANCHES:
        similarities, stock_names, timing = build_band_similarities(
            factor_result.returns[branch],
            runtime_args,
            level=1,
            wavelet="sym2",
            spearman=False,
            use_gpu=False,
        )
        observed_order = [normalize_stock_code(stock) for stock in stock_names]
        if observed_order != common_order:
            raise RuntimeError(f"J=1 stock order drifted in branch {branch}")
        row, assignments = run_cluster_partition(
            similarities=similarities,
            stock_names=stock_names,
            label_tables=label_tables,
            branch=branch,
            level=1,
            seed=int(args.seed),
        )
        row.update(timing)
        row["j1_definition"] = J1_DEFINITION
        assignment_path = assignments_dir / f"{branch}_level_1.csv"
        assignments.to_csv(assignment_path, index=False, encoding="utf-8-sig")
        row["assignments_path"] = str(assignment_path)
        rows.append(row)
        assignment_counts[branch] = len(assignments)
        print(
            f"__DS_PROGRESS__ {json.dumps({'experiment': 'single_level_analysis', 'branch': branch, 'SW1_ari': row['SW1_ari']})}",
            flush=True,
        )
        del similarities

    metrics = pd.DataFrame(rows).sort_values("branch")
    metrics.to_csv(output_dir / "level_metrics.csv", index=False, encoding="utf-8-sig")
    for metric in ("SW1_ari", "SW1_nmi", "SW2_ari", "SW2_nmi", "SW3_ari", "SW3_nmi"):
        metrics.pivot(index="level", columns="branch", values=metric).to_csv(
            output_dir / f"{metric}_pivot.csv",
            encoding="utf-8-sig",
        )

    checks = {
        "three_rows": len(metrics) == 3,
        "three_main_branches": set(metrics["branch"]) == set(MAIN_BRANCHES),
        "level_one": set(metrics["level"].astype(int)) == {1},
        "n_stocks_2770": set(metrics["n_stocks"].astype(int)) == {2770},
        "assignments_complete": set(assignment_counts.values()) == {2770},
        "finite_sw1_metrics": bool(np.isfinite(metrics[["SW1_ari", "SW1_nmi"]].to_numpy()).all()),
    }
    if not all(checks.values()):
        raise RuntimeError(f"J=1 supplement validation failed: {checks}")
    evidence = {
        "status": "complete",
        "j1_definition": J1_DEFINITION,
        "level_metrics": str(output_dir / "level_metrics.csv"),
        "assignments": str(assignments_dir),
        "validation": checks,
    }
    write_json(output_dir / "evidence_index.json", evidence)
    manifest.update(
        {
            "finished_at": datetime.now().astimezone().isoformat(),
            "seconds": float(time.time() - started),
            "price_filter": filter_stats,
            "raw_price_shape": list(prices.shape),
            "factor_common_universe_stock_count": len(current_universe),
            "validation": checks,
            "status": "complete",
            "artifacts": [
                "level_metrics.csv",
                "assignments/",
                "factor_common_universe.csv",
                "evidence_index.json",
            ],
        }
    )
    write_json(output_dir / "run_manifest.json", manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
