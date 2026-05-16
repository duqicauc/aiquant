# =============================================================================
# AIQuant v5.0 - Docker Production Image
# 多阶段构建：Node.js 构建前端 → Python + Nginx 运行全栈
# =============================================================================

# ─── Stage 1: 构建前端 ───
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --quiet

COPY frontend/ ./
RUN VITE_API_BASE='' npm run build

# ─── Stage 2: Python 运行时 ───
FROM python:3.12-slim

# 替换为国内 apt 源加速构建
RUN sed -i 's|http://deb.debian.org/debian|https://mirrors.cloud.tencent.com/debian|g' /etc/apt/sources.list.d/debian.sources && \
    sed -i 's|http://deb.debian.org/debian-security|https://mirrors.cloud.tencent.com/debian-security|g' /etc/apt/sources.list.d/debian.sources

# 安装系统依赖（含 Nginx、编译工具）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    nginx \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 配置 pip 国内镜像加速 Python 依赖下载
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

WORKDIR /app

# 安装 Python 依赖
COPY requirements.txt .
# 修复 requirements.txt 中 PyPI 上已不可用/版本号错误的依赖
RUN sed -i 's/^pandas-ta/# pandas-ta/' requirements.txt && \
    sed -i 's/vectorbt>=1.9.0/vectorbt>=0.26.0/' requirements.txt && \
    sed -i 's/mplfinance>=0.12.10/mplfinance>=0.12.10b0/' requirements.txt && \
    pip install --no-cache-dir -r requirements.txt \
    fastapi uvicorn bcrypt \
    && rm -rf /root/.cache/pip

# 复制后端代码
COPY src/ ./src/
COPY config/ ./config/
COPY scripts/ ./scripts/
COPY docs/ ./docs/
COPY env_template.txt ./

# 创建数据/日志目录
RUN mkdir -p data/database data/cache data/prediction data/results logs

# 复制前端构建产物
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Nginx 配置
COPY scripts/deploy/nginx-docker.conf /etc/nginx/conf.d/default.conf
RUN rm -f /etc/nginx/sites-enabled/default

# 容器启动脚本
COPY scripts/deploy/docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# 环境变量
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV TZ=Asia/Shanghai

# 暴露端口（Nginx 80）
EXPOSE 80

# 健康检查
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost/api/health || exit 1

ENTRYPOINT ["docker-entrypoint.sh"]
