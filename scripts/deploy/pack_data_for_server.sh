#!/bin/bash
################################################################################
# AIQuant 服务器数据打包脚本
# 将本地必需数据打包，上传到服务器后解压即可运行
################################################################################

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PACK_NAME="aiquant_data_$(date +%Y%m%d).tar.gz"
PACK_DIR="$PROJECT_ROOT/$PACK_NAME"

echo "======================================"
echo "  AIQuant 服务器数据打包"
echo "======================================"
echo ""

# 临时目录
TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

mkdir -p "$TMPDIR/data"

# ─── P0: 模型文件（必需，约 1MB）───
echo "[P0] 打包模型文件..."

# v3.1.0 Breakout模型
if [[ -d "$PROJECT_ROOT/data/models/v310/breakout" ]]; then
    mkdir -p "$TMPDIR/data/models/v310/breakout"
    # 复制最新版本
    LATEST_VER=$(ls -td "$PROJECT_ROOT"/data/models/v310/breakout/*/ 2>/dev/null | head -1)
    if [[ -n "$LATEST_VER" ]]; then
        cp -r "$LATEST_VER" "$TMPDIR/data/models/v310/breakout/"
        echo "  ✓ v3.1.0 Breakout: $(basename "$LATEST_VER")"
    fi
fi

# v3.1.0 Bounce模型（备用）
if [[ -d "$PROJECT_ROOT/data/models/v310/bounce" ]]; then
    mkdir -p "$TMPDIR/data/models/v310/bounce"
    LATEST_BU=$(ls -td "$PROJECT_ROOT"/data/models/v310/bounce/*/ 2>/dev/null | head -1)
    if [[ -n "$LATEST_BU" ]]; then
        cp -r "$LATEST_BU" "$TMPDIR/data/models/v310/bounce/"
        echo "  ✓ v3.1.0 Bounce: $(basename "$LATEST_BU")"
    fi
fi

# 保留旧版模型配置兼容
cp "$PROJECT_ROOT/data/models/breakout_launch_scorer/current.json" "$TMPDIR/data/models/breakout_launch_scorer/" 2>/dev/null || true

# 全局 current.json
cp "$PROJECT_ROOT/data/models/current.json" "$TMPDIR/data/models/" 2>/dev/null || true

# ─── P1: 预测结果（推荐，前端展示需要）───
echo "[P1] 打包预测结果..."
if [[ -d "$PROJECT_ROOT/data/prediction" ]]; then
    # 只打包最近的预测目录（按修改时间排序，取最新的2个版本）
    mkdir -p "$TMPDIR/data/prediction"
    for d in $(ls -td "$PROJECT_ROOT"/data/prediction/v* 2>/dev/null | head -2); do
        cp -r "$d" "$TMPDIR/data/prediction/"
        echo "  ✓ $(basename "$d")"
    done
    # 打包其他小文件
    cp "$PROJECT_ROOT"/data/prediction/*.csv "$TMPDIR/data/prediction/" 2>/dev/null || true
fi

# ─── P2: 回测结果（可选，前端展示需要）───
echo "[P2] 打包回测结果..."
if [[ -d "$PROJECT_ROOT/data/results" ]]; then
    mkdir -p "$TMPDIR/data/results"
    # 只打包 p22_ 开头的策略回测结果（最新的3个）
    for d in $(ls -td "$PROJECT_ROOT"/data/results/p2* 2>/dev/null | head -3); do
        cp -r "$d" "$TMPDIR/data/results/"
        echo "  ✓ $(basename "$d")"
    done
fi

# ─── P3: 配置 ───
echo "[P3] 打包配置..."
cp "$PROJECT_ROOT/.env" "$TMPDIR/" 2>/dev/null && echo "  ✓ .env" || echo "  ⚠ .env 不存在"
cp "$PROJECT_ROOT/env_template.txt" "$TMPDIR/" 2>/dev/null || true

# ─── 打包 ───
echo ""
echo "正在打包: $PACK_NAME"
tar czf "$PACK_DIR" -C "$TMPDIR" .

SIZE=$(du -sh "$PACK_DIR" | cut -f1)
echo ""
echo "======================================"
echo "  ✅ 打包完成"
echo "======================================"
echo ""
echo "文件: $PACK_DIR"
echo "大小: $SIZE"
echo ""
echo "上传到服务器:"
echo "  scp $PACK_DIR root@<服务器IP>:/opt/aiquant/"
echo ""
echo "服务器上解压:"
echo "  cd /opt/aiquant && tar xzf $PACK_NAME"
echo ""
