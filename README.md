
# 房价预测（Gradient Boosting Regressor）

该项目基于 `data/melb_data.csv` 的墨尔本房产交易数据，通过梯度提升回归模型（Gradient Boosting Regressor）对房价进行建模与预测。

## 功能概览

- `src/property/melb_price_model.py`：封装数据加载、特征工程、模型训练、评估与预测的可复用逻辑。
- `src/property/cli.py`：实现 `property` CLI 的全部子命令。
- 训练完成后会自动将模型保存至 `models/melb_gbr_pipeline.joblib`，可直接加载进行推理。

## 环境准备

1. 创建或激活 Python 3.11+ 虚拟环境。
2. 安装项目依赖（若已通过 `uv` 或其他工具同步，可跳过）：

### Linux
```bash
uv venv
uv sync
source ./.venv/bin/activate
```

### Windows
```bash
uv venv
uv sync
.venv/bin/activate
```

脚本将：

1. 读取 `data/melb_data.csv`，对缺失值进行填补并对类别特征进行独热编码。
2. 使用 80/20 训练-验证划分训练梯度提升回归模型。
3. 打印 R²、MAE、RMSE 等指标，并展示对样例房源的预测结果。
4. 将训练好的 Pipeline 序列化保存到 `models/melb_gbr_pipeline.joblib`，方便后续加载推理。

## 配置文件（TOML）

Flask API 会在启动时通过 `property.config.load_settings` 读取 `config/settings.toml`（可通过环境变量 `PROPERTY_CONFIG_PATH` 指定自定义路径），其中存放关键配置：

```toml
[app]
secret_key = "change-me"

[database]
url = "mysql+pymysql://property_user:property_password@localhost:3306/property"
pool_size = 5
pool_recycle = 1800
pool_timeout = 30

[security]
token_exp_minutes = 60
token_algorithm = "HS256"
```

- `app.secret_key`：用于 Flask/加密用途，请替换为随机字符串。
- `database`：配置 MySQL 连接字符串以及可选的连接池参数。启动时若库不存在表，将自动创建 `users` 表。
- `security`：控制基于 PyJWT 的无状态访问令牌（默认 60 分钟、HS256 签名）。
- 若需限制跨域来源，可设置 `PROPERTY_CORS_ORIGINS` 环境变量（逗号分隔），默认允许所有来源，便于本地前端调试。
- 生产环境建议将此文件放在独立的安全位置，并通过 `PROPERTY_CONFIG_PATH` 指向该文件。

## CLI：`property train` & `property calc`
1. 在虚拟环境中直接通过 `property` 命令调用 `property train` / `property calc`。

2. 训练：

```bash
property train [--data data/melb_data.csv] [--model models/melb_gbr_pipeline.joblib]
```

	- 默认启用 `HistGradientBoostingRegressor`，可自动利用多核心。
	- 如需限制 CPU 线程，可添加 `--n-threads 4` 等参数。
	- 若因兼容问题希望退回单核版本，可追加 `--disable-hist`。
	- 详细参数可使用 `property train -h`查看
3. 预测：

```bash
property calc [--rooms 3] [--bathroom 2] [--car 1] [--distance 6.5]
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

- `POST /auth/register`：接收形如 `{"username": "alice", "password": "hunter42"}` 的 JSON。密码至少 8 位，用户名唯一。服务器使用 bcrypt 存储密码，并返回用户信息与 `Bearer` 访问令牌。
- `POST /auth/login`：同样接受用户名与密码 JSON。校验通过后返回用户信息与新的无状态访问令牌，响应中包含 `token_type` 与 `expires_in`（秒）。

认证路由依赖于配置文件中的 MySQL 数据库。只需预先创建数据库实例（无需手动建表），服务启动后会自动迁移 `users` 表。客户端可将令牌存储在本地并在后续请求中通过 `Authorization: Bearer <token>` 头发送，无需服务器端会话状态。

## Web 用户端

- `web/` 目录提供了一个单页前端，按 “登录 → 控制台 → 计算房价” 的流程组织，默认通过 `web/config.js` 中的 `apiBase` 访问 API。
- 部署到 Docker 时，`docker-compose.yml` 已包含 `web` 服务，运行 `docker compose up --build` 后即可在 `http://localhost:9000` 打开用户端，`PROPERTY_CORS_ORIGINS` 自动与之匹配。
- 若在其他环境运行，只需修改 `web/config.js` 的 `apiBase`（例如指向云端域名），再重新构建 `Dockerfile.web` 或通过任意静态服务器托管该目录。


## Docker 部署

项目提供生产可用的 `Dockerfile`，默认使用 `gunicorn` 运行 Flask API。

1. 训练模型（或将已有的 `models/melb_gbr_pipeline.joblib` 放入仓库）。
2. 构建镜像：

```bash
docker build -t property-api:latest .
```

3. 运行容器：

```bash
docker run --rm -p 8000:8000 \
	-e PROPERTY_MODEL_PATH=/app/models/melb_gbr_pipeline.joblib \
	-e PROPERTY_DATA_PATH=/app/data/melb_data.csv \
	property-api:latest
```

容器启动后，可通过 `http://localhost:8000/health` 或 `/predict` 访问 API。若希望挂载自定义模型或数据集，可使用 `-v /host/path:/app/models` 等方式覆盖镜像内默认文件。

### docker-compose 快速启动

项目根目录提供 `docker-compose.yml`，可自动构建镜像并挂载本地 `models/` 与 `data/` 目录：

> 不要忘了训练模型

```bash
docker compose up --build
```

启动完成后同样监听 `http://localhost:8000`。若需要修改端口或环境变量，可直接编辑 compose 文件中的 `ports` 或 `environment` 字段。


## 多核心训练说明

- 默认训练器为 `HistGradientBoostingRegressor`，内部使用 OpenMP 并行，可自动利用全部可用 CPU 核心。
- 通过 `--n-threads` 可显式限定使用的线程数（同时影响 MKL/OPENBLAS/NumExpr 等数值库）。
- 若遇到特定平台不支持的情况，可用 `--disable-hist` 回退到经典 `GradientBoostingRegressor`（单核心实现）。

## 自定义与复用

- 在其他脚本中引入 `property.melb_price_model`，通过 `train_gradient_boosting` 获取 Pipeline，并调用 `predict_price` 进行自定义输入的房价预测。

## 数据来源

`data/melb_data.csv` 为公开的墨尔本房价数据集，存放在仓库 `data/` 目录下，可根据需要拓展特征或替换为其他房产数据。
