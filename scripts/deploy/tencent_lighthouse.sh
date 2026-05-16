#!/bin/bash
################################################################################
# AIQuant 腾讯云轻量服务器一键部署脚本
# 支持: Ubuntu 22.04 LTS / Debian 12
# 架构: Nginx + FastAPI(uvicorn) + React 静态文件 + SQLite
################################################################################

set -e

# ─── 颜色定义 ───
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# ─── 配置 ───
INSTALL_DIR="/opt/aiquant"
APP_USER="aiquant"
APP_GROUP="aiquant"
PYTHON_BIN="python3"
NODE_VERSION="20"
NGINX_CONF="/etc/nginx/sites-available/aiquant"
SYSTEMD_SERVICE="/etc/systemd/system/aiquant-api.service"

# ─── 辅助函数 ───
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_ok()   { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_err()  { echo -e "${RED}[ERROR]${NC} $1"; }

check_root() {
    if [[ $EUID -ne 0 ]]; then
        log_err "请使用 root 用户执行此脚本"
        exit 1
    fi
}

check_os() {
    if [[ -f /etc/os-release ]]; then
        . /etc/os-release
        OS=$ID
        VER=$VERSION_ID
        log_info "检测到系统: $NAME $VER"
    else
        log_err "无法识别操作系统"
        exit 1
    fi

    if [[ "$OS" != "ubuntu" && "$OS" != "debian" ]]; then
        log_warn "当前仅测试过 Ubuntu 22.04 / Debian 12，其他系统可能不兼容"
        read -p "是否继续? [y/N] " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
}

get_public_ip() {
    # 优先从腾讯云元数据获取，否则用公网 API
    local ip
    ip=$(curl -s --connect-timeout 3 http://metadata.tencentyun.com/latest/meta-data/public-ipv4 2>/dev/null || true)
    if [[ -z "$ip" ]]; then
        ip=$(curl -s --connect-timeout 3 https://api.ipify.org 2>/dev/null || true)
    fi
    if [[ -z "$ip" ]]; then
        ip=$(hostname -I | awk '{print $1}')
    fi
    echo "$ip"
}

# ─── 第 0 步: 前置检查 ───
echo -e "${CYAN}"
echo "=============================================="
echo "  AIQuant v5.0 - 腾讯云轻量服务器部署"
echo "=============================================="
echo -e "${NC}"

check_root
check_os

PUBLIC_IP=$(get_public_ip)
log_info "公网 IP: $PUBLIC_IP"

# ─── 第 1 步: 更新系统 ───
log_info "更新系统软件包..."
apt-get update -qq
apt-get upgrade -y -qq
log_ok "系统更新完成"

# ─── 第 2 步: 安装系统依赖 ───
log_info "安装系统依赖..."
apt-get install -y -qq \
    git curl wget unzip \
    build-essential pkg-config \
    nginx \
    libta-lib0-dev \
    sqlite3 \
    $PYTHON_BIN $PYTHON_BIN-venv $PYTHON_BIN-pip \
    2>/dev/null || apt-get install -y -qq \
    git curl wget unzip \
    build-essential pkg-config \
    nginx \
    libta-lib0-dev \
    sqlite3 \
    $PYTHON_BIN $PYTHON_BIN-venv $PYTHON_BIN-pip

# 安装 Node.js (NodeSource)
if ! command -v node &>/dev/null; then
    log_info "安装 Node.js ${NODE_VERSION}..."
    curl -fsSL https://deb.nodesource.com/setup_${NODE_VERSION}.x | bash - >/dev/null 2>&1
    apt-get install -y -qq nodejs
fi

NODE_VER=$(node --version 2>/dev/null || echo "unknown")
NPM_VER=$(npm --version 2>/dev/null || echo "unknown")
log_ok "Node.js ${NODE_VER} / npm ${NPM_VER} 安装完成"

PYTHON_VER=$($PYTHON_BIN --version 2>&1)
log_ok "${PYTHON_VER} 就绪"

# ─── 第 3 步: 创建应用用户 ───
if ! id "$APP_USER" &>/dev/null; then
    log_info "创建用户: $APP_USER"
    useradd -r -s /bin/false -d "$INSTALL_DIR" -m "$APP_USER"
fi
log_ok "用户 $APP_USER 就绪"

# ─── 第 4 步: 获取代码 ───
if [[ -d "$INSTALL_DIR/.git" ]]; then
    log_info "检测到已有代码，执行 git pull..."
    cd "$INSTALL_DIR"
    git pull origin $(git rev-parse --abbrev-ref HEAD) --quiet
else
    log_info "克隆代码仓库..."
    rm -rf "$INSTALL_DIR"
    git clone https://github.com/javaadu/aiquant.git "$INSTALL_DIR" --quiet 2>/dev/null || {
        log_warn "GitHub 克隆失败，尝试 Gitee..."
        git clone https://gitee.com/javaadu/aiquant.git "$INSTALL_DIR" --quiet 2>/dev/null || {
            log_err "代码仓库克隆失败，请检查网络连接"
            exit 1
        }
    }
fi
chown -R "$APP_USER:$APP_GROUP" "$INSTALL_DIR"
log_ok "代码已就绪: $INSTALL_DIR"

# ─── 第 5 步: 配置环境变量 ───
log_info "配置环境变量..."
if [[ ! -f "$INSTALL_DIR/.env" ]]; then
    if [[ -f "$INSTALL_DIR/env_template.txt" ]]; then
        cp "$INSTALL_DIR/env_template.txt" "$INSTALL_DIR/.env"
    else
        touch "$INSTALL_DIR/.env"
    fi
fi

# 交互式输入 TUSHARE_TOKEN
CURRENT_TOKEN=$(grep "^TUSHARE_TOKEN=" "$INSTALL_DIR/.env" | cut -d= -f2 || true)
if [[ -z "$CURRENT_TOKEN" || "$CURRENT_TOKEN" == "YOUR_TUSHARE_TOKEN" ]]; then
    echo ""
    echo -e "${YELLOW}────────────────────────────────────────${NC}"
    echo -e "${YELLOW}  请配置 Tushare API Token${NC}"
    echo -e "${YELLOW}  注册地址: https://tushare.pro/register${NC}"
    echo -e "${YELLOW}────────────────────────────────────────${NC}"
    read -p "请输入你的 TUSHARE_TOKEN: " TUSHARE_TOKEN
    if [[ -n "$TUSHARE_TOKEN" ]]; then
        # 替换或添加 TUSHARE_TOKEN
        if grep -q "^TUSHARE_TOKEN=" "$INSTALL_DIR/.env"; then
            sed -i "s/^TUSHARE_TOKEN=.*/TUSHARE_TOKEN=${TUSHARE_TOKEN}/" "$INSTALL_DIR/.env"
        else
            echo "TUSHARE_TOKEN=${TUSHARE_TOKEN}" >> "$INSTALL_DIR/.env"
        fi
        log_ok "TUSHARE_TOKEN 已配置"
    else
        log_warn "未输入 Token，预测功能将无法使用"
    fi
else
    log_ok "TUSHARE_TOKEN 已存在，跳过配置"
fi

# ─── 第 6 步: Python 虚拟环境 & 依赖 ───
log_info "创建 Python 虚拟环境并安装依赖..."
cd "$INSTALL_DIR"

if [[ ! -d "$INSTALL_DIR/venv" ]]; then
    $PYTHON_BIN -m venv venv
fi

source venv/bin/activate

# 升级 pip
pip install --upgrade pip -q

# 安装核心依赖（排除开发/测试依赖以节省时间和空间）
log_info "安装 Python 依赖（这可能需要 5-15 分钟）..."
pip install -q \
    fastapi uvicorn \
    pandas numpy scipy \
    tushare akshare yfinance \
    sqlalchemy joblib tqdm loguru python-dotenv pyyaml click requests \
    scikit-learn xgboost \
    ta-lib pandas-ta \
    backtrader lightgbm catboost \
    h5py pyarrow pymysql psycopg2-binary redis \
    streamlit plotly matplotlib seaborn mplfinance pyecharts \
    apscheduler

# 可选: 安装所有依赖（如果内存足够）
# pip install -r requirements.txt -q

log_ok "Python 依赖安装完成"

# ─── 第 7 步: 构建前端 ───
log_info "构建前端..."
cd "$INSTALL_DIR/frontend"

if [[ ! -d "node_modules" ]]; then
    npm install --quiet
fi

# 生产构建: API 使用相对路径，由 Nginx 代理
VITE_API_BASE='' npm run build --quiet 2>/dev/null || VITE_API_BASE='' npm run build

if [[ ! -d "$INSTALL_DIR/frontend/dist" ]]; then
    log_err "前端构建失败，请检查 logs"
    exit 1
fi

log_ok "前端构建完成"

# ─── 第 8 步: 初始化数据库 ───
log_info "初始化数据库..."
cd "$INSTALL_DIR"

# 创建数据目录
mkdir -p data/database data/cache data/prediction data/results logs
chown -R "$APP_USER:$APP_GROUP" data logs

# 运行数据库初始化（如果存在 init_db 脚本）
if $PYTHON_BIN -c "from src.scheduler.models import init_db; init_db()" 2>/dev/null; then
    log_ok "数据库初始化完成"
else
    log_warn "自动初始化失败，将在首次启动时自动创建"
fi

# ─── 第 9 步: Nginx 配置 ───
log_info "配置 Nginx..."

cat > "$NGINX_CONF" << 'EOF'
server {
    listen 80 default_server;
    listen [::]:80 default_server;
    server_name _;

    client_max_body_size 50M;

    # 前端静态文件
    location / {
        root /opt/aiquant/frontend/dist;
        index index.html;
        try_files $uri $uri/ /index.html;
        expires 1d;
        add_header Cache-Control "public, immutable";
    }

    # API 代理
    location /api/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_connect_timeout 60s;
        proxy_send_timeout 120s;
        proxy_read_timeout 120s;
    }

    # API 文档
    location /docs {
        proxy_pass http://127.0.0.1:8000/docs;
        proxy_set_header Host $host;
    }

    location /redoc {
        proxy_pass http://127.0.0.1:8000/redoc;
        proxy_set_header Host $host;
    }

    # 健康检查
    location /api/health {
        proxy_pass http://127.0.0.1:8000/api/health;
        proxy_set_header Host $host;
        access_log off;
    }
}
EOF

# 启用配置
rm -f /etc/nginx/sites-enabled/aiquant
ln -sf "$NGINX_CONF" /etc/nginx/sites-enabled/aiquant
rm -f /etc/nginx/sites-enabled/default 2>/dev/null || true

nginx -t 2>/dev/null || {
    log_err "Nginx 配置测试失败"
    exit 1
}

systemctl reload nginx
log_ok "Nginx 配置完成"

# ─── 第 10 步: Systemd 服务 ───
log_info "配置 Systemd 服务..."

cat > "$SYSTEMD_SERVICE" << EOF
[Unit]
Description=AIQuant FastAPI API
After=network.target nginx.service
Wants=nginx.service

[Service]
Type=simple
User=$APP_USER
Group=$APP_GROUP
WorkingDirectory=$INSTALL_DIR
Environment=PYTHONPATH=$INSTALL_DIR
Environment=PYTHONUNBUFFERED=1
EnvironmentFile=$INSTALL_DIR/.env
ExecStart=$INSTALL_DIR/venv/bin/uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --workers 1
ExecReload=/bin/kill -HUP \$MAINPID
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal
SyslogIdentifier=aiquant-api

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable aiquant-api
log_ok "Systemd 服务配置完成"

# ─── 第 11 步: 防火墙配置 ───
log_info "配置防火墙..."
if command -v ufw &>/dev/null; then
    ufw allow 80/tcp >/dev/null 2>&1 || true
    ufw allow 443/tcp >/dev/null 2>&1 || true
    ufw allow 22/tcp >/dev/null 2>&1 || true
    log_ok "UFW 防火墙规则已添加"
fi

# 腾讯云轻量服务器默认使用轻量应用服务器防火墙，需在控制台开放端口
log_warn "请确保在腾讯云控制台 > 防火墙中放通 TCP 80 端口"

# ─── 第 12 步: 启动服务 ───
log_info "启动 AIQuant 服务..."
systemctl start aiquant-api
sleep 3

# 检查服务状态
if systemctl is-active --quiet aiquant-api; then
    log_ok "AIQuant API 服务已启动"
else
    log_err "AIQuant API 启动失败，查看日志: journalctl -u aiquant-api -n 50"
    exit 1
fi

# 检查 API 是否响应
if curl -s http://127.0.0.1:8000/api/health | grep -q "ok"; then
    log_ok "API 健康检查通过"
else
    log_warn "API 健康检查未通过，可能还在启动中..."
fi

# ─── 完成 ───
echo ""
echo -e "${GREEN}==============================================${NC}"
echo -e "${GREEN}  🎉 AIQuant 部署成功！${NC}"
echo -e "${GREEN}==============================================${NC}"
echo ""
echo -e "  ${CYAN}🌐 Web 界面:${NC}   http://${PUBLIC_IP}"
echo -e "  ${CYAN}📡 API 文档:${NC}   http://${PUBLIC_IP}/docs"
echo -e "  ${CYAN}📋 API 状态:${NC}   http://${PUBLIC_IP}/api/health"
echo ""
echo -e "  ${YELLOW}常用命令:${NC}"
echo -e "    查看服务状态: ${BLUE}systemctl status aiquant-api${NC}"
echo -e "    查看服务日志: ${BLUE}journalctl -u aiquant-api -f${NC}"
echo -e "    重启服务:     ${BLUE}systemctl restart aiquant-api${NC}"
echo -e "    停止服务:     ${BLUE}systemctl stop aiquant-api${NC}"
echo -e "    重启 Nginx:   ${BLUE}systemctl reload nginx${NC}"
echo -e "    更新代码:     ${BLUE}cd $INSTALL_DIR && git pull && ./scripts/deploy/update.sh${NC}"
echo ""
echo -e "  ${YELLOW}文件位置:${NC}"
echo -e "    代码目录:     ${BLUE}$INSTALL_DIR${NC}"
echo -e "    前端构建:     ${BLUE}$INSTALL_DIR/frontend/dist${NC}"
echo -e "    数据库:       ${BLUE}$INSTALL_DIR/data/database/aiquant.db${NC}"
echo -e "    日志:         ${BLUE}$INSTALL_DIR/logs/${NC}"
echo -e "    Nginx 配置:   ${BLUE}$NGINX_CONF${NC}"
echo -e "    环境变量:     ${BLUE}$INSTALL_DIR/.env${NC}"
echo ""
echo -e "  ${RED}⚠️  重要提示:${NC}"
echo -e "    1. 请在腾讯云控制台 > 防火墙中放通 TCP 80 端口"
echo -e "    2. 如需 HTTPS，请配置域名并申请 SSL 证书"
echo -e "    3. 本系统仅供学习研究，不构成投资建议"
echo -e "    4. 首次使用前请在 ${BLUE}.env${NC} 中配置 TUSHARE_TOKEN"
echo ""
echo -e "${GREEN}==============================================${NC}"
