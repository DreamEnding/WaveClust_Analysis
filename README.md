# WaveClust Analysis

本仓库提供论文 *Cross-Frequency Redundancy in the Chinese A-Share Dependence Network* 的核心算法、实验入口与补充分析代码。

## 目录结构

```text
.
├── waveclust/                         核心算法包
├── supplementary/                    补充分析
│   ├── run_operator_ablation.py       聚合算子消融
│   ├── run_common_factor_controls.py  共同因子控制
│   └── run_single_level_analysis.py   单层 MODWT 分析
├── tests/                             补充分析单元测试
├── config.yaml                        默认实验配置
├── run_experiment.py                  单次 WaveClust 实验
├── run_level_sweep.py                 分解层级与 MCL 参数扫描
├── run_pure_price_spectral.py         纯价格谱聚类实验
└── combine_pure_price_results.py      合并分片实验结果
```

`waveclust` 包含数据加载、预处理、MODWT/SWT 分解、相似度构建、MCL 与谱聚类、申万行业标签评估等实现。补充目录使用描述实验内容的名称，不依赖内部开发阶段编号。

## 环境

建议使用 Python 3.10 或更高版本：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

运行测试还需安装：

```powershell
python -m pip install -r requirements-dev.txt
```

GPU 加速为可选功能。根据本机 CUDA 版本另行安装对应的 CuPy 包；不安装时可通过 `--no-gpu` 使用 CPU。

## 数据准备

默认配置从仓库根目录下的 `DATA/` 读取：

```text
DATA/
├── stock_price_panel.csv              日期 × 股票的收盘价面板
├── stock_basic.csv                    股票基础信息
├── trade_cal.csv                      交易日历
├── K/                                 按股票拆分的日行情 CSV
└── tickflow_universes/
    └── universe_list.json             申万行业标签
```

价格面板首列为日期索引，其余列为股票代码。也可在 `config.yaml` 或各命令行参数中指定其他路径。若同目录存在 `stock_price_panel.parquet`，默认优先读取 Parquet；此时需要安装 `pyarrow` 或 `fastparquet`。

原始行情数据受数据提供方许可约束，不随代码仓库分发。

## 核心实验

运行单次实验：

```powershell
python run_experiment.py --output-dir output/example --no-gpu
```

扫描小波层级和 MCL inflation：

```powershell
python run_level_sweep.py --output-root output/level_sweep --levels 2 3 4 5 6 --no-gpu
```

运行论文使用的纯价格谱聚类配置示例：

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

合并多个结果目录：

```powershell
python combine_pure_price_results.py --input-root output/pure_price
```

## 补充分析

补充入口均从仓库根目录以模块方式运行：

```powershell
python -m supplementary.run_operator_ablation --output-dir output/supplementary/operator_ablation --no-gpu
python -m supplementary.run_common_factor_controls --output-dir output/supplementary/common_factor_controls --no-gpu
python -m supplementary.run_single_level_analysis --output-dir output/supplementary/single_level
```

算子消融会生成基线校验文件；共同因子控制默认读取该文件。单层分析默认读取共同因子控制的正式结果。所有输入路径都可通过对应命令的 `--help` 查看和覆盖。

## 验证

```powershell
python -m pytest -q
python -m compileall -q waveclust supplementary
```

## 版本来源

核心实现来自 2026-05-25 论文网格实验所使用的代码快照。当前版本在该快照基础上移除了本地绝对路径，统一了公开目录和命名，并加入算子消融、共同因子控制及单层分解所需的最小扩展。

如使用本代码，请引用对应论文。代码供学术研究使用；其他用途请联系作者。
