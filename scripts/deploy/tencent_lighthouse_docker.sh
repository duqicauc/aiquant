#!/bin/bash
################################################################################
# AIQuant 腾讯云轻量服务器 - Docker 一键部署脚本
# 架构: docker-compose (Nginx + FastAPI + React 静态文件 + SQLite)
################################################################################

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_ok()   { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_err()  { echo -e "${RED}[ERROR]${NC} $1"; }

check_root() {
    if [[ $EUID -ne 0 ]]; then
        log_err "请使用 root 用户执行"
        exit 1
    fi
}

get_public_ip() {
    local ip
    ip=$(curl -s --connect-timeout 3 http://metadata.tencentyun.com/latest/meta-data/public-ipv4 2>/dev/null || true)
    [[ -z "$ip" ]] && ip=$(curl -s --connect-timeout 3 https://api.ipify.org 2>/dev/null || true)
    [[ -z "$ip" ]] && ip=$(hostname -I | awk '{print $1}')
    echo "$ip"
}

# ─── 前置检查 ───
echo -e "${CYAN}"
echo "=============================================="
echo "  AIQuant v5.0 - Docker 一键部署"
echo "=============================================="
echo -e "${NC}"

check_root

PUBLIC_IP=$(get_public_ip)
log_info "公网 IP: $PUBLIC_IP"

# ─── 第 1 步: 安装 Docker ───
if ! command -v docker &>/dev/null; then
    log_info "安装 Docker..."
    curl -fsSL https://get.docker.com | sh
    systemctl enable docker
    systemctl start docker
    log_ok "Docker 安装完成"
else
    log_ok "Docker 已安装: $(docker --version)"
fi

if ! command -v docker-compose &>/dev/null && ! docker compose version &>/dev/null; then
    log_info "安装 Docker Compose..."
    DOCKER_CONFIG=${DOCKER_CONFIG:-$HOME/.docker}
    mkdir -p $DOCKER_CONFIG/cli-plugins
    curl -SL https://github.com/docker/compose/releases/download/v2.27.0/docker-compose-linux-x86_64 -o $DOCKER_CONFIG/cli-plugins/docker-compose
    chmod +x $DOCKER_CONFIG/cli-plugins/docker-compose
    ln -sf $DOCKER_CONFIG/cli-plugins/docker-compose /usr/local/bin/docker-compose
    log_ok "Docker Compose 安装完成"
else
    log_ok "Docker Compose 已就绪"
fi

# ─── 第 2 步: 获取代码 ───
INSTALL_DIR="/opt/aiquant"
if [[ -d "$INSTALL_DIR/.git" ]]; then
    log_info "更新代码..."
    cd "$INSTALL_DIR" && git pull origin $(git rev-parse --abbrev-ref HEAD) --quiet
else
    log_info "克隆代码..."
    rm -rf "$INSTALL_DIR"
    git clone https://github.com/javaadu/aiquant.git "$INSTALL_DIR" --quiet 2>/dev/null || {
        log_warn "GitHub 失败，尝试 Gitee..."
        git clone https://gitee.com/javaadu/aiquant.git "$INSTALL_DIR" --quiet
    }
fi
log_ok "代码已就绪: $INSTALL_DIR"

cd "$INSTALL_DIR"

# ─── 第 3 步: 配置环境变量 ───
if [[ ! -f ".env" ]]; then
    cp env_template.txt .env
fi

CURRENT_TOKEN=$(grep "^TUSHARE_TOKEN=" .env | cut -d= -f2 || true)
if [[ -z "$CURRENT_TOKEN" || "$CURRENT_TOKEN" == "YOUR_TUSHARE_TOKEN" ]]; then
    echo ""
    echo -e "${YELLOW}────────────────────────────────────────${NC}"
    echo -e "${YELLOW}  请配置 Tushare API Token${NC}"
    echo -e "${YELLOW}  注册: https://tushare.pro/register${NC}"
    echo -e "${YELLOW}────────────────────────────────────────${NC}"
    read -p "请输入 TUSHARE_TOKEN: " TUSHARE_TOKEN
    if [[ -n "$TUSHARE_TOKEN" ]]; then
        if grep -q "^TUSHARE_TOKEN=" .env; then
            sed -i "s/^TUSHARE_TOKEN=.*/TUSHARE_TOKEN=${TUSHARE_TOKEN}/" .env
        else
            echo "TUSHARE_TOKEN=${TUSHARE_TOKEN}" >> .env
        fi
        log_ok "Token 已配置"
    else
        log_warn "未输入 Token"
    fi
fi

# ─── 第 4 步: 构建并启动 ───
log_info "构建 Docker 镜像（首次约 10-20 分钟，请耐心等待）..."
docker compose build --no-cache

log_info "启动服务..."
docker compose up -d

# ─── 第 5 步: 等待服务就绪 ───
log_info "等待服务启动..."
for i in {1..30}; do
    if curl -s http://localhost/api/health | grep -q "ok"; then
        log_ok "服务健康检查通过"
        break
    fi
    sleep 2
    if [[ $i -eq 30 ]]; then
        log_warn "健康检查超时，请查看日志: docker compose logs -f"
    fi
done

# ─── 第 6 步: 防火墙 ───
if command -v ufw &>/dev/null; then
    ufw allow 80/tcp >/dev/null 2>&1 || true
fi
log_warn "请确保腾讯云控制台防火墙已放通 TCP 80 端口"

# ─── 完成 ───
echo ""
echo -e "${GREEN}==============================================${NC}"
echo -e "${GREEN}  🎉 AIQuant Docker 部署成功！${NC}"
echo -e "${GREEN}==============================================${NC}"
echo ""
echo -e "  ${CYAN}🌐 Web 界面:${NC}   http://${PUBLIC_IP}"
echo -e "  ${CYAN}📡 API 文档:${NC}   http://${PUBLIC_IP}/docs"
echo -e "  ${CYAN}📋 健康检查:${NC}   http://${PUBLIC_IP}/api/health"
echo ""
echo -e "  ${YELLOW}常用命令:${NC}"
echo -e "    查看日志:     ${BLUE}cd $INSTALL_DIR && docker compose logs -f${NC}"
echo -e "    重启服务:     ${BLUE}cd $INSTALL_DIR && docker compose restart${NC}"
echo -e "    停止服务:     ${BLUE}cd $INSTALL_DIR && docker compose down${NC}"
echo -e "    更新代码:     ${BLUE}cd $INSTALL_DIR && git pull && docker compose up -d --build${NC}"
echo -e "    进入容器:     ${BLUE}docker exec -it aiquant bash${NC}"
echo ""
echo -e "  ${YELLOW}数据持久化:${NC}"
echo -e "    数据库:       ${BLUE}$INSTALL_DIR/data/database/aiquant.db${NC}"
echo -e "    日志:         ${BLUE}$INSTALL_DIR/logs/${NC}"
echo -e "    配置:         ${BLUE}$INSTALL_DIR/.env${NC}"
echo ""
echo -e "  ${RED}⚠️  提示:${NC}"
echo -e "    1. 首次构建包含 npm install + pip install，耗时较长"
echo -e "    2. 后续更新只需 ${BLUE}git pull && docker compose up -d --build${NC}"
echo -e "    3. 本系统仅供学习研究，不构成投资建议"
echo ""
echo -e "${GREEN}==============================================${NC}"
