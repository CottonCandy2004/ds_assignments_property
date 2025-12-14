
# 房价预测（Gradient Boosting Regressor）

该项目基于 `data/melb_data.csv` 的墨尔本房产交易数据，通过梯度提升回归模型（Gradient Boosting Regressor）对房价进行建模与预测。

## 功能概览

- `src/database_ml/melb_price_model.py`：封装数据加载、特征工程、模型训练、评估与预测的可复用逻辑。
- `src/database_ml/cli.py`：实现 `property` CLI 的全部子命令。
- `main.py`：命令行入口，执行训练、输出关键评估指标，并保存可复用的 `scikit-learn` Pipeline（含预处理与模型）。
- 训练完成后会自动将模型保存至 `models/melb_gbr_pipeline.joblib`，可直接加载进行推理。

## 环境准备

1. 创建或激活 Python 3.11+ 虚拟环境。
2. 安装项目依赖（若已通过 `uv` 或其他工具同步，可跳过）：

```bash
/run/media/CandySanjo/E85E61615E612990/Code_Workspace/database/.venv/bin/python -m pip install pandas scikit-learn scipy matplotlib ipykernel joblib
```

## 运行训练

```bash
/run/media/CandySanjo/E85E61615E612990/Code_Workspace/database/.venv/bin/python main.py
```

脚本将：

1. 读取 `data/melb_data.csv`，对缺失值进行填补并对类别特征进行独热编码。
2. 使用 80/20 训练-验证划分训练梯度提升回归模型。
3. 打印 R²、MAE、RMSE 等指标，并展示对样例房源的预测结果。
4. 将训练好的 Pipeline 序列化保存到 `models/melb_gbr_pipeline.joblib`，方便后续加载推理。

## CLI：`property train` & `property calc`

在项目根目录提供了可执行脚本 `property`，封装了常用的训练与预测操作。

1. 首次使用可为脚本添加执行权限（或直接使用 `python property ...`）：

```bash
chmod +x property
```

2. 训练：

```bash
./property train --data data/melb_data.csv --model models/melb_gbr_pipeline.joblib
```

3. 预测：

```bash
./property calc --rooms 3 --bathroom 2 --car 1 --distance 6.5
```

	- 所有未显式指定的特征会自动使用数据集中位数/众数作为默认值。
	- 可以使用 `--feature COLUMN=VALUE` 形式覆盖任何额外列（可重复传入）。
	- 若模型文件不存在，可先运行 `property train` 生成。

4. 如希望在虚拟环境中直接通过 `property` 命令调用，可执行可编辑安装：

```bash
pip install -e .
```

安装完成后，可在任意位置使用 `property train` / `property calc`（使用相对路径或参数指定数据/模型位置）。

## 自定义与复用

- 若需调整数据路径、随机种子或模型保存路径，可在 `main.py` 中修改 `TrainingConfig` 参数。
- 也可以直接在其他脚本中引入 `src.melb_price_model`，通过 `train_gradient_boosting` 获取 Pipeline，并调用 `predict_price` 进行自定义输入的房价预测。

## 数据来源

`data/melb_data.csv` 为公开的墨尔本房价数据集，存放在仓库 `data/` 目录下，可根据需要拓展特征或替换为其他房产数据。
