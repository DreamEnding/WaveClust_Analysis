from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import run_rebuttal_part3 as part3_runner
from run_rebuttal_part8 import parse_args as parse_part8_args
from run_rebuttal_part3 import band_names, parse_args as parse_part3_args
from waveclust.dependence import rank_normalize_rows_for_spearman, summarize_cross_band_dependence
from waveclust.factors import prepare_factor_branches, select_factor_common_universe
from waveclust.model import StockWaveClust, WaveClustParams
from waveclust.spectral import build_signed_dual_score_cache, build_signed_dual_waveclust_score


def _two_stock_similarity(value: float) -> np.ndarray:
    return np.array([[0.0, value], [value, 0.0]], dtype=np.float32)


def test_signed_dual_score_supports_weighted_mean_reducer() -> None:
    sim_mats = [
        _two_stock_similarity(1.0),
        _two_stock_similarity(1.0),
        _two_stock_similarity(0.25),
        _two_stock_similarity(0.0625),
    ]

    score = build_signed_dual_waveclust_score(
        sim_mats,
        k=1.0,
        neg_weight=1.25,
        reducer="mean",
        layer_weighting=True,
    )

    # B=(1, 1/2, 1/4), d=(2, 3, 4), hence mean(d*B)=3/2.
    assert score.dtype == np.float32
    assert np.array_equal(score, _two_stock_similarity(1.5))


def test_signed_dual_score_matches_preregistered_operator_formulas() -> None:
    sim_mats = [
        _two_stock_similarity(1.0),
        _two_stock_similarity(1.0),
        _two_stock_similarity(0.25),
        _two_stock_similarity(0.0625),
    ]

    observed = [
        build_signed_dual_waveclust_score(
            sim_mats,
            k=1.0,
            neg_weight=1.25,
            reducer=reducer,
            layer_weighting=weighted,
        )[0, 1]
        for reducer, weighted in [
            ("max", True),
            ("mean", True),
            ("median", True),
            ("geometric", True),
            ("max", False),
            ("mean", False),
        ]
    ]

    # B=(1, 1/2, 1/4); weighted X=(2, 3/2, 1).
    expected = np.array([2.0, 1.5, 1.5, np.cbrt(3.0), 1.0, 7.0 / 12.0])
    assert np.allclose(observed, expected, rtol=1e-6, atol=1e-7)


def test_default_signed_dual_max_is_bitwise_legacy_equivalent() -> None:
    rng = np.random.default_rng(42)
    sim_mats = []
    for _ in range(4):
        matrix = rng.uniform(-1.0, 1.0, size=(7, 7)).astype(np.float32)
        matrix = ((matrix + matrix.T) * np.float32(0.5)).astype(np.float32)
        np.fill_diagonal(matrix, 0.0)
        sim_mats.append(matrix)

    low = sim_mats[0]
    pos_low = np.maximum(low, 0.0)
    neg_low = np.maximum(-low, 0.0)
    expected = np.zeros_like(pos_low, dtype=np.float32)
    for detail_index, high in enumerate(sim_mats[1:], start=1):
        pos_raw = np.sqrt(pos_low * np.maximum(high, 0.0))
        neg_raw = np.sqrt(neg_low * np.maximum(-high, 0.0)) * 1.25
        expected = np.maximum(expected, (pos_raw + neg_raw) * (1.5 * detail_index + 1.0))
    np.fill_diagonal(expected, 0.0)
    expected = np.maximum(expected, expected.T)

    observed = build_signed_dual_waveclust_score(
        sim_mats,
        k=1.5,
        neg_weight=1.25,
        interaction="sqrt",
    )

    assert np.array_equal(observed, expected)


def test_signed_dual_score_cache_matches_individual_operator_scores() -> None:
    rng = np.random.default_rng(17)
    sim_mats = []
    for _ in range(5):
        matrix = rng.uniform(-1.0, 1.0, size=(9, 9)).astype(np.float32)
        matrix = ((matrix + matrix.T) * np.float32(0.5)).astype(np.float32)
        np.fill_diagonal(matrix, 0.0)
        sim_mats.append(matrix)

    cache = build_signed_dual_score_cache(sim_mats, neg_weight=1.25)
    for reducer, layer_weighting in [
        ("max", True),
        ("mean", True),
        ("median", True),
        ("geometric", True),
        ("max", False),
        ("mean", False),
    ]:
        expected = build_signed_dual_waveclust_score(
            sim_mats,
            k=1.5,
            neg_weight=1.25,
            reducer=reducer,
            layer_weighting=layer_weighting,
        )
        observed = cache.build(
            k=1.5,
            reducer=reducer,
            layer_weighting=layer_weighting,
        )
        assert np.array_equal(observed, expected)


def test_signed_dual_score_rejects_mismatched_matrix_shapes() -> None:
    with pytest.raises(ValueError, match="same square shape"):
        build_signed_dual_waveclust_score(
            [np.zeros((3, 3), dtype=np.float32), np.zeros((2, 2), dtype=np.float32)],
            k=1.5,
            neg_weight=1.25,
        )


def test_signed_dual_score_rejects_non_boolean_layer_weighting() -> None:
    with pytest.raises(ValueError, match="layer_weighting must be boolean"):
        build_signed_dual_waveclust_score(
            [_two_stock_similarity(1.0), _two_stock_similarity(0.5)],
            k=1.5,
            neg_weight=1.25,
            layer_weighting="weighted",  # type: ignore[arg-type]
        )


def test_signed_dual_score_is_nonnegative_symmetric_zero_diagonal_and_geometric_zero_preserving() -> None:
    sim_mats = [
        np.array([[0.0, 1.0, -1.0], [0.5, 0.0, 0.25], [-0.5, 0.5, 0.0]], dtype=np.float32),
        np.array([[0.0, 0.0, -0.25], [0.0, 0.0, 0.5], [-0.75, 0.25, 0.0]], dtype=np.float32),
        np.array([[0.0, 0.5, -1.0], [0.25, 0.0, 1.0], [-1.0, 0.5, 0.0]], dtype=np.float32),
    ]

    score = build_signed_dual_waveclust_score(
        sim_mats,
        k=1.5,
        neg_weight=1.25,
        reducer="geometric",
    )

    assert np.all(score >= 0)
    assert np.array_equal(score, score.T)
    assert np.array_equal(np.diag(score), np.zeros(3, dtype=np.float32))
    assert score[0, 1] == 0.0


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        ("reducer", "sum", "unsupported signed-dual reducer"),
        ("interaction", "product", "unsupported signed-dual interaction"),
        ("k", -1.0, "k must be finite and non-negative"),
        ("neg_weight", -1.0, "neg_weight must be finite and non-negative"),
    ],
)
def test_signed_dual_score_rejects_invalid_configuration(keyword, value, message) -> None:
    kwargs = {
        "k": 1.5,
        "neg_weight": 1.25,
        "reducer": "max",
        "interaction": "sqrt",
    }
    kwargs[keyword] = value
    with pytest.raises(ValueError, match=message):
        build_signed_dual_waveclust_score(
            [_two_stock_similarity(1.0), _two_stock_similarity(0.5)],
            **kwargs,
        )


def test_factor_branches_preserve_sample_and_satisfy_projection_invariants() -> None:
    dates = pd.date_range("2024-01-01", periods=25, freq="D")
    stocks = ["000001", "000002", "000003", "000004", "000005", "000006"]
    industries = pd.Series(["A", "A", "B", "B", "C", "C"], index=stocks, name="industry")
    time = np.linspace(-1.0, 1.0, len(dates) - 1)
    market = 0.012 * np.sin(np.pi * time)
    industry = np.column_stack(
        [
            0.004 * time,
            0.004 * time,
            -0.003 * time,
            -0.003 * time,
            np.full_like(time, 0.002),
            np.full_like(time, 0.002),
        ]
    )
    idiosyncratic = np.column_stack([np.sin((idx + 2) * time) for idx in range(len(stocks))]) * 0.001
    betas = np.array([0.6, 0.9, 1.1, 1.3, 1.6, 1.9])
    log_returns = market[:, None] * betas[None, :] + industry + idiosyncratic
    log_prices = np.vstack([np.zeros(len(stocks)), np.cumsum(log_returns, axis=0)])
    prices = pd.DataFrame(np.exp(log_prices), index=dates, columns=stocks)

    result = prepare_factor_branches(prices, industries, winsor_limit=0.0)

    expected_keys = {"raw", "market_residual_internal", "sequential_market_industry_adjusted"}
    assert set(result.returns) == expected_keys
    for branch in result.returns.values():
        assert branch.columns.tolist() == stocks
        assert branch.index.equals(dates[1:])
        assert np.isfinite(branch.to_numpy()).all()

    market_diag = result.per_stock_diagnostics["market_residual_internal"]
    assert np.isfinite(market_diag[["alpha", "beta", "r_squared", "explained_variance"]]).all().all()
    assert float(market_diag["residual_factor_inner_product"].abs().max()) < 1e-12

    sequential = result.returns["sequential_market_industry_adjusted"]
    group_means = sequential.T.groupby(industries).mean().T
    assert float(group_means.abs().to_numpy().max()) < 1e-6


def test_factor_common_universe_excludes_missing_and_singleton_industries() -> None:
    prices = pd.DataFrame(
        np.arange(18, dtype=np.float32).reshape(3, 6) + 1.0,
        index=pd.date_range("2024-01-01", periods=3),
        columns=["000001", "000002", "000003", "000004", "000005", "000006"],
    )
    stock_info = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ", "000005.SZ", "000006.SZ"],
            "symbol": ["000001", "000002", "000003", "000004", "000005", "000006"],
            "industry": ["A", "A", "B", np.nan, "C", "C"],
        }
    )

    universe = select_factor_common_universe(prices, stock_info, min_industry_size=2)

    assert universe.prices.columns.tolist() == ["000001", "000002", "000005", "000006"]
    assert universe.industries.tolist() == ["A", "A", "C", "C"]
    assert universe.exclusions.set_index("stock")["reason"].to_dict() == {
        "000003": "industry_size_lt_2",
        "000004": "missing_industry",
    }


def test_part8_runner_accepts_explicit_workspace_data_and_output_paths(tmp_path) -> None:
    args = parse_part8_args(
        [
            "--workspace-root",
            str(tmp_path),
            "--data-panel",
            str(tmp_path / "prices.csv"),
            "--shenwan-universe",
            str(tmp_path / "labels.json"),
            "--archived-assignment",
            str(tmp_path / "assignment.csv"),
            "--output-dir",
            str(tmp_path / "outputs"),
            "--baseline-only",
            "--gpu-ids",
            "0",
            "1",
        ]
    )

    assert args.workspace_root == tmp_path
    assert args.data_panel == tmp_path / "prices.csv"
    assert args.shenwan_universe == tmp_path / "labels.json"
    assert args.archived_assignment == tmp_path / "assignment.csv"
    assert args.output_dir == tmp_path / "outputs"
    assert args.gpu_ids == [0, 1]


def test_cross_band_dependence_handles_identical_and_reversed_rankings() -> None:
    upper = np.triu_indices(4, k=1)
    first = np.zeros((4, 4), dtype=np.float32)
    second = np.zeros((4, 4), dtype=np.float32)
    first[upper] = np.arange(1, 7, dtype=np.float32)
    second[upper] = np.arange(6, 0, -1, dtype=np.float32)
    first = first + first.T
    second = second + second.T

    summary = summarize_cross_band_dependence(
        [first, second],
        band_names=["A", "D"],
        n_bins=2,
    )

    assert np.array_equal(summary.spearman.to_numpy(), np.array([[1.0, -1.0], [-1.0, 1.0]]))
    assert np.allclose(summary.nmi.to_numpy(), np.ones((2, 2)))
    assert np.allclose(summary.vi.to_numpy(), np.zeros((2, 2)), atol=1e-15)


def test_rank_normalization_turns_row_dot_products_into_spearman_correlation() -> None:
    values = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [4.0, 3.0, 2.0, 1.0],
        ],
        dtype=np.float32,
    )

    normalized = rank_normalize_rows_for_spearman(values)
    similarity = normalized @ normalized.T

    assert normalized.dtype == np.float32
    assert np.allclose(similarity, np.array([[1.0, -1.0], [-1.0, 1.0]]), atol=1e-7)


def test_part3_runner_accepts_explicit_workspace_data_and_output_paths(tmp_path) -> None:
    args = parse_part3_args(
        [
            "--workspace-root",
            str(tmp_path),
            "--data-panel",
            str(tmp_path / "prices.csv"),
            "--stock-basic",
            str(tmp_path / "stock_basic.csv"),
            "--shenwan-universe",
            str(tmp_path / "labels.json"),
            "--archived-figure3-dir",
            str(tmp_path / "figure3"),
            "--part8-baseline-gate",
            str(tmp_path / "baseline_gate.json"),
            "--output-dir",
            str(tmp_path / "outputs"),
            "--baseline-only",
            "--gpu-ids",
            "2",
            "3",
        ]
    )

    assert args.workspace_root == tmp_path
    assert args.data_panel == tmp_path / "prices.csv"
    assert args.stock_basic == tmp_path / "stock_basic.csv"
    assert args.archived_figure3_dir == tmp_path / "figure3"
    assert args.output_dir == tmp_path / "outputs"
    assert args.gpu_ids == [2, 3]


def test_part3_runner_records_a_part8_gate_blocker(tmp_path, monkeypatch) -> None:
    output_dir = tmp_path / "part3"

    def reject_part8_gate(*_args, **_kwargs):
        raise RuntimeError("Part 8 reference gate rejected for test")

    monkeypatch.setattr(part3_runner, "validate_part8_gate", reject_part8_gate)
    with pytest.raises(RuntimeError, match="rejected for test"):
        part3_runner.main(
            [
                "--workspace-root",
                str(tmp_path),
                "--output-dir",
                str(output_dir),
            ]
        )

    manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    evidence_index = json.loads((output_dir / "evidence_index.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "blocked_part8_reference_gate"
    assert manifest["part8_reference_gate"]["passed"] is False
    assert manifest["failures"] == [
        {
            "stage": "part8_reference_gate",
            "reason": "Part 8 reference gate rejected for test",
        }
    ]
    assert evidence_index["status"] == "blocked_part8_reference_gate"
    assert evidence_index["blocker"] == "Part 8 reference gate rejected for test"


def test_part3_band_order_matches_pywavelets_trimmed_swt_contract() -> None:
    assert band_names(4) == ["CA4", "CD4", "CD3", "CD2", "CD1"]


def test_prepare_level_matrices_preserves_legacy_row_order_and_can_release_coefficients() -> None:
    stocks = ["000001", "000002", "000003"]
    returns = pd.DataFrame(
        np.ones((6, len(stocks)), dtype=np.float32),
        columns=stocks,
    )
    model = StockWaveClust(
        prices=returns,
        stock_info=None,
        params=WaveClustParams(levels=2, use_gpu=False),
    )
    model.returns = returns
    model.coefficients = {
        stock: {
            0: np.array([index + 1, index + 2, index + 3, index + 4], dtype=np.float32),
            1: np.array([index + 2, index + 3, index + 4], dtype=np.float32),
            2: np.array([index + 3, index + 4], dtype=np.float32),
        }
        for index, stock in enumerate(stocks)
    }

    expected = []
    for level, length in enumerate([4, 3, 2]):
        rows = np.empty((len(stocks), length), dtype=np.float32)
        for row_index, stock in enumerate(stocks):
            rows[row_index] = model.coefficients[stock][level][:length]
        norms = np.linalg.norm(rows, axis=1, keepdims=True)
        expected.append(rows / norms)

    observed, order = model.prepare_level_matrices()
    assert order == stocks
    for actual, legacy in zip(observed, expected, strict=True):
        assert np.array_equal(actual, legacy)

    model.release_wavelet_coefficients()
    assert model.coefficients == {}
    assert all(np.isfinite(matrix).all() for matrix in observed)
