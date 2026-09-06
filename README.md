# WaveClust Analysis

This repository contains the core algorithms, experiment entry points, and supplementary analyses for the paper *Cross-Frequency Redundancy in the Chinese A-Share Dependence Network*.

## Repository layout

```text
.
├── waveclust/                         Core algorithm package
├── supplementary/                    Supplementary analyses
│   ├── run_operator_ablation.py       Aggregation-operator ablation
│   ├── run_common_factor_controls.py  Common-factor controls
│   └── run_single_level_analysis.py   Single-level MODWT analysis
├── tests/                             Tests for supplementary analyses
├── config.yaml                        Default experiment configuration
├── run_experiment.py                  Single WaveClust experiment
├── run_level_sweep.py                 Level and MCL-parameter sweep
├── run_pure_price_spectral.py         Pure-price spectral clustering
└── combine_pure_price_results.py      Merge sharded experiment results
```

The `waveclust` package implements data loading, preprocessing, MODWT/SWT decomposition, similarity construction, MCL and spectral clustering, and evaluation against Shenwan industry labels.

## Installation

Python 3.10 or later is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Install the development dependencies to run the test suite:

```powershell
python -m pip install -r requirements-dev.txt
```

GPU acceleration is optional. Install the CuPy package that matches your local CUDA version to enable it. All experiment entry points that support GPU execution also provide `--no-gpu` for CPU-only runs.

## Data preparation

The default configuration reads data from `DATA/` at the repository root:

```text
DATA/
├── stock_price_panel.csv              Date-by-stock closing-price panel
├── stock_basic.csv                    Stock metadata
├── trade_cal.csv                      Trading calendar
├── K/                                 One daily-price CSV per stock
└── tickflow_universes/
    └── universe_list.json             Shenwan industry labels
```

The first column of `stock_price_panel.csv` must contain the date index; each remaining column represents one stock. Data paths can be changed in `config.yaml` or overridden with the command-line options exposed by each entry point.

If `stock_price_panel.parquet` exists beside the CSV file, the loader prefers the Parquet file. Reading Parquet requires either `pyarrow` or `fastparquet`.

Raw market data is not distributed with this repository because it is subject to the data provider's licensing terms.

## Core experiments

Run one WaveClust experiment:

```powershell
python run_experiment.py --output-dir output/example --no-gpu
```

Sweep wavelet levels and MCL inflation values:

```powershell
python run_level_sweep.py --output-root output/level_sweep --levels 2 3 4 5 6 --no-gpu
```

Run an example of the pure-price spectral-clustering configuration used in the paper:

```powershell
python run_pure_price_spectral.py `
  --output-dir output/pure_price `
  --wavelets sym2 `
  --levels 2 `
  --k-values 1.5 `
  --q-thresholds 0.996 `
  --gammas 0.35 `
  --neg-weights 1.25 `
  --n-clusters 25 `
  --clusterers dense_spectral_signed_dual_power `
  --assign-labels discretize `
  --winsor-limits 0.03 `
  --write-best-assignments `
  --no-gpu
```

Merge results from multiple run directories:

```powershell
python combine_pure_price_results.py --input-root output/pure_price
```

## Supplementary analyses

Run supplementary entry points as modules from the repository root:

```powershell
python -m supplementary.run_operator_ablation --output-dir output/supplementary/operator_ablation --no-gpu
python -m supplementary.run_common_factor_controls --output-dir output/supplementary/common_factor_controls --no-gpu
python -m supplementary.run_single_level_analysis --output-dir output/supplementary/single_level
```

The operator ablation produces a baseline-validation file that the common-factor controls read by default. The single-level analysis reads the completed common-factor-control run. Use `--help` on any entry point to view and override its input paths.

## Verification

```powershell
python -m pytest -q
python -m compileall -q waveclust supplementary
```

Please cite the corresponding paper when using this code. The code is provided for academic research; contact the authors regarding other uses.
