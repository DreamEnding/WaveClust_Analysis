# 路径修改说明
## Path Configuration Changes

**修改日期**: 2026-08-28

---

## ✅ 已完成的修改

### 1. 包名重命名
- `waveclust_legacy` → `waveclust`
- 所有Python文件中的导入语句已更新

### 2. 硬编码路径移除

#### 修改前 (原始代码):
```python
PACKAGE_DIR = Path(__file__).resolve().parent
CODE_DIR = PACKAGE_DIR.parent
WORKSPACE_DIR = CODE_DIR.parents[2]  # ❌ 硬编码的相对路径

def resolve_workspace_path(path_value: str | Path) -> Path:
    if path.parts and path.parts[0] == "Finance":
        return WORKSPACE_DIR.parent / path  # ❌ 特定于原始目录结构
    return WORKSPACE_DIR / path
```

#### 修改后 (新代码):
```python
PACKAGE_DIR = Path(__file__).resolve().parent
CODE_DIR = PACKAGE_DIR.parent
WORKSPACE_DIR = Path.cwd()  # ✅ 使用当前工作目录

def resolve_workspace_path(path_value: str | Path) -> Path:
    """Resolve paths relative to workspace directory."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    # ✅ 所有相对路径从当前工作目录解析
    return WORKSPACE_DIR / path
```

---

## 📁 推荐的目录结构

### 运行实验时的目录结构
```
your_project/
├── PAPER_CODE_OFFICIAL/          # 实验代码
│   ├── waveclust/                # 核心包
│   ├── run_level_sweep.py
│   └── config.yaml
│
├── data/                         # 数据目录
│   ├── prices/                   # 股票价格数据
│   └── shenwan/                  # 申万行业标签
│
└── output/                       # 输出目录
    ├── results/
    └── assignments/
```

### 使用方式

#### 方式1: 从项目根目录运行
```bash
cd your_project
python PAPER_CODE_OFFICIAL/run_level_sweep.py --config PAPER_CODE_OFFICIAL/config.yaml
```

#### 方式2: 从代码目录运行
```bash
cd your_project/PAPER_CODE_OFFICIAL
python run_level_sweep.py --config config.yaml
```

---

## 🔧 配置文件调整建议

### config.yaml 中的路径配置

**修改前**:
```yaml
data:
  price_dir: /lustre/home/.../DATA/K/  # ❌ 绝对路径
```

**修改后**:
```yaml
data:
  price_dir: ./data/prices/  # ✅ 相对路径
  # 或
  price_dir: data/prices/    # ✅ 相对于工作目录
```

---

## ⚠️ 注意事项

### 1. 工作目录
- `WORKSPACE_DIR` 现在指向 `Path.cwd()`（当前工作目录）
- 确保从正确的目录运行脚本
- 或者在代码中使用 `os.chdir()` 切换到项目根目录

### 2. 数据路径
- 所有数据路径现在应该是相对路径
- 相对于运行脚本时的工作目录
- 可以在config.yaml中配置

### 3. 输出路径
- 输出文件将保存到相对于工作目录的位置
- 建议在config中明确指定输出目录

---

## 📝 代码兼容性

### Python导入
所有导入语句已从 `waveclust_legacy` 更新为 `waveclust`:

```python
# 旧版本
from waveclust_legacy.model import StockWaveClust
from waveclust_legacy.evaluation import compute_ari

# 新版本 ✓
from waveclust.model import StockWaveClust
from waveclust.evaluation import compute_ari
```

### 路径解析
- `resolve_workspace_path()`: 自动处理相对/绝对路径
- `load_config()`: 优先在CODE_DIR中查找配置文件

---

## ✅ 验证清单

- [x] 包名重命名: waveclust_legacy → waveclust
- [x] 移除硬编码绝对路径
- [x] WORKSPACE_DIR 使用 Path.cwd()
- [x] 所有import语句已更新
- [x] 路径解析函数已修改为相对路径

---

**修改完成**: 2026-08-28  
**状态**: ✅ 所有硬编码路径已移除，代码可移植性已提升
