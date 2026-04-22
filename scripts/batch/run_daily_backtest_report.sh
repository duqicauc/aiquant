#!/usr/bin/env bash
# 收盘后一键：互补预测（若需）+ 回测 + 报告输出到 data/prediction/results
# 用法：在 crontab 中配置，例如每个交易日 16:05：
#   5 16 * * 1-5 /path/to/aiquant/scripts/run_daily_backtest_report.sh >> /path/to/aiquant/logs/daily_backtest.log 2>&1

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

END_DATE="${END_DATE:-$(date +%Y%m%d)}"
# 默认回测约 3 个月窗口（可按需改 START_DATE）
START_DATE="${START_DATE:-$(python3 - <<'PY'
from datetime import datetime, timedelta
print((datetime.now() - timedelta(days=90)).strftime("%Y%m%d"))
PY
)}"

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

echo "[$(date -Iseconds)] daily backtest ${START_DATE} -> ${END_DATE}"

python3 scripts/backtest_v232_v270_complementary.py \
  --start-date "$START_DATE" \
  --end-date "$END_DATE" \
  --stop-loss-mode close \
  --output-dir "${ROOT}/data/prediction/results"

echo "[$(date -Iseconds)] done"
