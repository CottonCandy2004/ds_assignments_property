# Production image for the property Flask API
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 自动检测 Debian 版本并设置对应的源
RUN DEB_VERSION=$(grep VERSION_CODENAME /etc/os-release | cut -d= -f2) \
    && echo "检测到 Debian 版本: $DEB_VERSION" \
    && rm -f /etc/apt/sources.list /etc/apt/sources.list.d/* \
    && echo "deb https://mirrors.tuna.tsinghua.edu.cn/debian/ $DEB_VERSION main contrib non-free" > /etc/apt/sources.list \
    && echo "deb https://mirrors.tuna.tsinghua.edu.cn/debian/ $DEB_VERSION-updates main contrib non-free" >> /etc/apt/sources.list \
    && echo "deb https://mirrors.tuna.tsinghua.edu.cn/debian/ $DEB_VERSION-backports main contrib non-free" >> /etc/apt/sources.list \
    && echo "deb https://mirrors.tuna.tsinghua.edu.cn/debian-security/ $DEB_VERSION-security main contrib non-free" >> /etc/apt/sources.list


RUN apt-get update && apt-get install -y --no-install-recommends \
    curl

WORKDIR /app

# Copy project metadata first (better cache behaviour when code changes)
COPY pyproject.toml README.md ./

# Copy application source and supporting assets
COPY src ./src
COPY data ./data
COPY models ./models
COPY config ./config

RUN pip install --upgrade -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple pip setuptools wheel \
    && pip install --no-cache-dir -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple .

ENV PROPERTY_MODEL_PATH="/app/models/melb_gbr_pipeline.joblib" \
    PROPERTY_DATA_PATH="/app/data/melb_data.csv" \
    PROPERTY_TARGET="Price" \
    HOST="0.0.0.0" \
    PORT="8000"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

CMD ["gunicorn", "--workers", "2", "--timeout", "120", "--bind", "0.0.0.0:8000", "property.api:app"]
