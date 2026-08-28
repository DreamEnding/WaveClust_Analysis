# 论文官方实验代码
## Official Experiment Code for Paper

**论文标题**: Cross-Frequency Redundancy in the Chinese A-Share Dependence Network  
**代码版本**: Grid Search 20260525  
**验证日期**: 2026-08-28

---

## 📁 目录结构

```
PAPER_CODE_OFFICIAL/
├── README.md                           本文件
├── config.yaml                         实验配置文件
├── run_level_sweep.py                  层级扫描主脚本
├── run_experiment.py                   单次实验运行器
├── run_pure_price_spectral.py          纯价格谱聚类
├── combine_pure_price_results.py       结果合并脚本
│
└── waveclust_legacy/                   核心算法包
    ├── __init__.py                     包初始化
    ├── model.py                        WaveClust核心模型 ⭐
    ├── data.py                         数据加载处理
    ├── evaluation.py                   评估指标计算
    ├── mcl.py                          MCL聚类实现
    ├── preprocessing.py                预处理函数
    ├── spectral.py                     谱聚类实现
    ├── shenwan.py                      申万行业标签
    ├── pure_price.py                   纯价格处理
    ├── pipeline.py                     Pipeline管理
    └── config.py                       配置管理
```

---

## ✅ 代码验证信息

### MD5哈希验证
```
model.py:          6cf47266479828d61d8284ec0a12951f
data.py:           6199155dc969418509f46e39f2a7175b
evaluation.py:     c5832037f61e91761d3100c044d13671
mcl.py:            711a70c2c6bee8a0e76a3e87d4e2b47a
preprocessing.py:  39aa86d6a6361ff4c007c8d57d9703c2
spectral.py:       2ddc6ecc2b55f971e0607b9377542f4a
```

### 实验信息
- **运行时间**: 2026-05-25
- **配置数量**: 15,360 (4 wavelets × 5 levels × 4 k × 192 其他组合)
- **数据产出**: 16MB combined_all_metrics.csv
- **参考配置**: trial_id a85036d07d14690a (Sym2, Level 2, k=1.5)

---

## 🔧 核心功能

### 1. waveclust_legacy.model.StockWaveClust

**主类**: 股票小波聚类模型

**关键参数**:
```python
WaveClustParams(
    wavelet_transform="modwt",      # MODWT变换
    wavelet="db4",                  # 小波基
    levels=6,                       # 分解层级
    k=0.0,                          # 层距权重
    q_threshold=0.82,               # 相似度阈值
    inflation=1.4,                  # MCL膨胀参数
    pruning_threshold=0.05,         # MCL剪枝阈值
    use_gpu=True,                   # GPU加速
)
```

**主要方法**:
- `fit()`: 执行完整的小波分解和聚类
- `compute_similarity_matrix()`: 计算频带间相似度
- `signed_dual_power_transform()`: 正负样本分离的幂变换
- `cluster_spectral()`: 谱聚类

### 2. waveclust_legacy.mcl

**MCL聚类**: Markov Cluster Algorithm实现
- 支持CPU和CUDA加速
- 自动选择最优后端

### 3. waveclust_legacy.evaluation

**评估指标**:
- `compute_industry_metrics()`: 计算ARI, NMI等指标
- 支持多层级申万行业标签对比

---

## 🚀 使用方法

### 基本使用示例

```python
from waveclust_legacy.model import StockWaveClust, WaveClustParams
from waveclust_legacy.data import load_stock_prices
from waveclust_legacy.evaluation import compute_industry_metrics

# 1. 加载数据
prices_df = load_stock_prices(data_dir='DATA/K/')

# 2. 配置参数
params = WaveClustParams(
    wavelet="sym2",
    levels=2,
    k=1.5,
    q_threshold=0.996,
    inflation=1.4,
)

# 3. 运行模型
model = StockWaveClust(prices_df, params)
model.fit()

# 4. 获取结果
assignments = model.get_assignments()
metrics = compute_industry_metrics(assignments, shenwan_labels)

print(f"ARI: {metrics['SW1_ari']:.4f}")
print(f"Communities: {metrics['n_communities']}")
```

### 运行Grid Search

```bash
# 运行层级扫描
python run_level_sweep.py --config config.yaml

# 运行单个实验
python run_experiment.py \
    --wavelet sym2 \
    --levels 2 \
    --k 1.5 \
    --q_threshold 0.996
```

---

## 📊 参考配置重现

### 论文中的参考配置 (ARI = 0.3809)

```python
params = WaveClustParams(
    wavelet_transform="modwt",
    wavelet="sym2",
    levels=2,
    k=1.5,
    q_threshold=0.996,
    inflation=1.4,
    pruning_threshold=0.05,
    # 聚类参数
    clusterer="dense_spectral_signed_dual_power",
    n_clusters=25,
    gamma=0.35,
    neg_weight=1.25,
    assign_labels="discretize",
    seed=42,
    # 预处理
    winsor_limit=0.03,
    do_vol_zscore=True,
)

# 数据时间窗口
window = {
    'start': '2019-01-01',
    'end': '2025-12-31',
    'eligible_stocks': 2773,
}

# 预期结果
expected_results = {
    'SW1_ari': 0.3809381025162831,
    'SW1_nmi': 0.5081638078138633,
    'n_communities': 25,
    'largest_community': 220,
    'median_size': 115,
    'n_edges': 9704,
    'n_singletons': 0,
}
```

---

## 📦 依赖环境

### Python版本
- Python >= 3.8

### 必需包
```
numpy >= 1.20
pandas >= 1.3
pywt >= 1.1.1              # 小波变换
scikit-learn >= 0.24       # 谱聚类
markov_clustering >= 0.0.6 # MCL
scipy >= 1.7
```

### 可选包 (GPU加速)
```
cupy >= 9.0                # CUDA加速
```

### 安装命令
```bash
pip install numpy pandas PyWavelets scikit-learn markov_clustering scipy

# GPU加速 (可选)
pip install cupy-cuda11x  # 根据CUDA版本选择
```

---

## 📝 代码特性

### 1. 模块化设计
- 清晰的包结构
- 独立的数据、模型、评估模块
- 易于扩展和修改

### 2. GPU加速支持
- 自动检测CUDA可用性
- CPU/GPU后端自动选择
- 大规模数据高效处理

### 3. 完整的配置管理
- YAML配置文件支持
- 参数验证
- 默认值管理

### 4. 实验可重复性
- 固定随机种子 (seed=42)
- 完整的参数记录
- 中间结果可保存

---

## 🔬 实验验证

### 已验证的配置空间

**小波基**: db4, sym2, sym4, coif1  
**分解层级**: 2, 3, 4, 5, 6  
**层距权重k**: 0.0, 0.5, 1.0, 1.5  
**时间窗口**: 2019-2025, 2021-2025, 2023-2025, 2024-2025  

**总配置数**: 15,360  
**验证状态**: ✅ 全部通过

### 关键结果验证

| 验证项 | 预期值 | 实际值 | 状态 |
|--------|--------|--------|------|
| 参考配置ARI | 0.3809 | 0.3809381 | ✅ |
| Level 2最佳 | 0.3809 | 0.3809381 | ✅ |
| Level 6最佳 | 0.1385 | 0.1385454 | ✅ |
| Spearman均值 | 0.769 | 0.769 | ✅ |

---

## 📖 引用信息

如果您使用此代码，请引用：

```bibtex
@article{mo2026cross,
  title={Cross-Frequency Redundancy in the Chinese A-Share Dependence Network},
  author={Mo, Jimao and Zhang, Jiaqi and Chen, Shijia and Zhang, Hua},
  journal={Entropy},
  year={2026},
  publisher={MDPI}
}
```

---

## 📧 联系方式

**技术问题**:
- Jimao Mo (第一作者)

**数据获取**:
- Shijia Chen: chenshijia@sztu.edu.cn
- Hua Zhang: zhanghua@sztu.edu.cn

---

## 📜 许可证

本代码仅供学术研究使用。如需商业使用，请联系作者。

---

## 🔄 版本历史

- **v1.0 (2026-05-25)**: 初始版本，Grid Search实验
- **v1.0-archived (2026-06-05)**: 归档版本，论文官方代码
- **v1.0-verified (2026-08-28)**: MD5验证通过，论文提交版本

---

**最后更新**: 2026-08-28  
**验证状态**: ✅ 100%通过  
**代码来源**: analysis/waveclust_dense_signed_full_grid_20260525/code/wavelet/
