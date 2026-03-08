#!/usr/bin/env bash
# 使用 conda 虚拟环境：v232+v270 评分、集成、回测
# 用法:
#   bash scripts/run_score_and_backtest_conda.sh                    # 使用默认日期
#   bash scripts/run_score_and_backtest_conda.sh 20260225            # 指定评分日期
#   bash scripts/run_score_and_backtest_conda.sh 20260225 20260106 20260225  # 评分日 + 回测起止
#   SCORE_DATE=20260225 START=20260106 END=20260225 bash scripts/run_score_and_backtest_conda.sh  # 环境变量
# 或: conda activate base && bash scripts/run_score_and_backtest_conda.sh

set -e
cd "$(dirname "$0")/.."

# 默认日期（可通过参数或环境变量覆盖）
DEFAULT_SCORE="20260304"
DEFAULT_START="20260105"
DEFAULT_END="20260304"

# 优先：命令行参数（评分日 [回测起始 回测结束]）
if [ -n "$1" ]; then
  DATE_224="$1"
  [ -n "$2" ] && START="$2" || START="${START:-$DEFAULT_START}"
  [ -n "$3" ] && END="$3" || END="${END:-$DEFAULT_END}"
else
  # 其次：环境变量
  DATE_224="${SCORE_DATE:-$DEFAULT_SCORE}"
  START="${START:-$DEFAULT_START}"
  END="${END:-$DEFAULT_END}"
fi

# 若在子 shell 中未激活 conda，尝试激活 base
if ! command -v python &>/dev/null || ! python -c "import loguru" 2>/dev/null; then
  if [ -n "$CONDA_PREFIX" ]; then
    echo "使用当前 conda 环境: $CONDA_PREFIX"
  else
    echo "请先激活 conda 环境，例如: conda activate base"
    echo "然后重新运行: bash scripts/run_score_and_backtest_conda.sh"
    exit 1
  fi
fi

echo "=============================================="
echo "1. v232 模型评分 ($DATE_224 收盘后)"
echo "=============================================="
python scripts/predict_v232_top10.py --date "$DATE_224"

echo ""
echo "=============================================="
echo "2. v270 集成模型评分 ($DATE_224 收盘后)"
echo "=============================================="
python scripts/predict_v270_ensemble_top50.py "$DATE_224"

echo ""
echo "=============================================="
echo "3. 集成模型（互补策略）Top10（生成100行候选池供回测）"
echo "=============================================="
python scripts/combine_v232_v270.py --date "$DATE_224" --strategy complementary --top 10 --base-top-n 100 --v232-top-n 100

echo ""
echo "=============================================="
echo "4. 组合策略回测 $START ~ $END"
echo "=============================================="
python scripts/backtest_v232_v270_complementary.py --start-date "$START" --end-date "$END" --stop-loss-mode close

echo ""
echo "=============================================="
echo "完成。报告与 CSV 在: data/prediction/results/"
echo "=============================================="
