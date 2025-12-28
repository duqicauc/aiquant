#!/bin/bash
# 左侧潜力牛股模型样本准备启动脚本
# 修复SSL权限问题并确保使用缓存

cd "$(dirname "$0")/.."

# 设置SSL证书路径（在Python导入前）
# 使用Python获取certifi路径，但避免导入requests
CERT_PATH=$(python3 << 'PYEOF'
import sys
import os
# 在导入certifi之前设置环境变量
try:
    import certifi
    print(certifi.where())
except:
    # 尝试使用conda的证书
    conda_cert = os.path.expanduser("~/miniconda3/ssl/cacert.pem")
    if os.path.exists(conda_cert):
        print(conda_cert)
    else:
        print("")
PYEOF
)

if [ -n "$CERT_PATH" ] && [ -f "$CERT_PATH" ]; then
    export REQUESTS_CA_BUNDLE="$CERT_PATH"
    export SSL_CERT_FILE="$CERT_PATH"
    export CURL_CA_BUNDLE="$CERT_PATH"
    echo "✓ 设置SSL证书路径: $CERT_PATH"
else
    echo "⚠️  未找到SSL证书，将尝试使用系统默认证书"
fi

# 运行脚本
echo "🚀 启动左侧潜力牛股模型样本准备..."
echo "📊 将优先使用缓存数据，减少API调用"
echo ""

# 使用env命令确保环境变量被传递
env python3 scripts/prepare_left_breakout_samples.py "$@"

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "✅ 样本准备完成！"
else
    echo ""
    echo "❌ 样本准备失败，退出码: $exit_code"
fi

exit $exit_code

