
# 房价预测（Gradient Boosting Regressor）

该项目基于 `data/melb_data.csv` 的墨尔本房产交易数据，通过梯度提升回归模型（Gradient Boosting Regressor）对房价进行建模与预测。

## 功能概览

- `src/property/melb_price_model.py`：封装数据加载、特征工程、模型训练、评估与预测的可复用逻辑。
- `src/property/cli.py`：实现 `property` CLI 的全部子命令。
- 训练完成后会自动将模型保存至 `models/melb_gbr_pipeline.joblib`，可直接加载进行推理。

## 环境准备

1. 创建或激活 Python 3.11+ 虚拟环境。
2. 安装项目依赖（若已通过 `uv` 或其他工具同步，可跳过）：

```bash
uv add pandas scikit-learn scipy matplotlib ipykernel joblib
```

脚本将：

1. 读取 `data/melb_data.csv`，对缺失值进行填补并对类别特征进行独热编码。
2. 使用 80/20 训练-验证划分训练梯度提升回归模型。
3. 打印 R²、MAE、RMSE 等指标，并展示对样例房源的预测结果。
4. 将训练好的 Pipeline 序列化保存到 `models/melb_gbr_pipeline.joblib`，方便后续加载推理。

## CLI：`property train` & `property calc`
1. 在虚拟环境中直接通过 `property` 命令调用，可执行可编辑安装：

```bash
pip install -e .
```

安装完成后，可在任意位置使用 `property train` / `property calc`（使用相对路径或参数指定数据/模型位置）。

2. 训练：

```bash
./property train --data data/melb_data.csv --model models/melb_gbr_pipeline.joblib
```

	- 默认启用 `HistGradientBoostingRegressor`，可自动利用多核心。
	- 如需限制 CPU 线程，可添加 `--n-threads 4` 等参数。
	- 若因兼容问题希望退回单核版本，可追加 `--disable-hist`。

3. 预测：

```bash
./property calc --rooms 3 --bathroom 2 --car 1 --distance 6.5
```

	- 所有未显式指定的特征会自动使用数据集中位数/众数作为默认值。
	- 可以使用 `--feature COLUMN=VALUE` 形式覆盖任何额外列（可重复传入）。
	- 若模型文件不存在，可先运行 `property train` 生成。


## Flask API 服务

在保留 CLI 能力的同时，项目新增了一个轻量级 Flask API，可通过 GET 请求对房价进行预测。

1. 确保已经训练并保存好模型（默认 `models/melb_gbr_pipeline.joblib`）。
2. 启动服务：

```bash
export FLASK_APP=property.api
export PYTHONPATH=src  # 若未安装为包，可临时添加
flask run --host 0.0.0.0 --port 5000
```

也可以直接运行 `python -m property.api`，通过 `HOST`、`PORT` 环境变量调整监听地址。

### 路由说明

- `GET /health`：返回模型路径、数据路径、目标列等基础信息。
- `GET /predict`：根据查询参数进行推理，返回 JSON，例如：

```
GET /predict?Rooms=3&Bathroom=2&Car=1&Distance=6.5
```

> 所有特征（包括 CLI 中的别名）都可以作为查询参数；未提供的特征将继续使用数据集中位数/众数作为默认值。若需要覆盖其他列，可多次传入 `feature=Column=Value`。

响应示例：

```json
{
	"prediction": 880123.42,
	"currency": "AUD",
	"features": {
		"Rooms": 3,
		"Bathroom": 2,
		"Car": 1,
		"Distance": 6.5,
		"...": "..."
	},
	"overrides": {
		"Rooms": 3,
		"Bathroom": 2,
		"Car": 1,
		"Distance": 6.5
	}
}
```

若需要使用自定义模型/数据位置，可在启动前设置 `PROPERTY_MODEL_PATH`、`PROPERTY_DATA_PATH`、`PROPERTY_TARGET` 环境变量。



## 多核心训练说明

- 默认训练器为 `HistGradientBoostingRegressor`，内部使用 OpenMP 并行，可自动利用全部可用 CPU 核心。
- 通过 `--n-threads` 可显式限定使用的线程数（同时影响 MKL/OPENBLAS/NumExpr 等数值库）。
- 若遇到特定平台不支持的情况，可用 `--disable-hist` 回退到经典 `GradientBoostingRegressor`（单核心实现）。

## 自定义与复用

- 在其他脚本中引入 `property.melb_price_model`，通过 `train_gradient_boosting` 获取 Pipeline，并调用 `predict_price` 进行自定义输入的房价预测。

## 数据来源

`data/melb_data.csv` 为公开的墨尔本房价数据集，存放在仓库 `data/` 目录下，可根据需要拓展特征或替换为其他房产数据。
