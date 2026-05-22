#!/usr/bin/env python3
"""
v3.1.0 双模型预测 + 回测流水线

1. BreakoutPredictor 生成突破信号
2. BouncePredictor 生成反弹信号
3. 信号融合（等权 / 动态权重）
4. qlib / vectorbt 回测
5. 结果对比与报告

Usage:
    python scripts/predict_and_backtest_v310.py \
        --start 20240101 --end 20241231 \
        --fusion equal --top_k 20 --backtest
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.bounce_predictor import BouncePredictor
from src.models.breakout_predictor import BreakoutPredictor
from src.utils.logger import log

PREDICTION_DIR = PROJECT_ROOT / "data" / "prediction" / "v3.1.0"


# ============================================================================
# 信号融合策略
# ============================================================================
def fuse_signals(
    df_breakout: pd.DataFrame,
    df_bounce: pd.DataFrame,
    strategy: str = "equal",
    top_k: int = 20,
) -> pd.DataFrame:
    """
    融合 Breakout + Bounce 信号

    Args:
        df_breakout: Breakout 预测结果 (ts_code, prob_cal, rank)
        df_bounce: Bounce 预测结果 (ts_code, prob_cal, rank)
        strategy: 融合策略
            - "equal": 等权组合（各取 top_k/2，去重后补全）
            - "intersection": 取两个模型共同推荐的股票
            - "breakout_only": 只用 Breakout
            - "bounce_only": 只用 Bounce
        top_k: 最终选股数量

    Returns:
        DataFrame: ts_code, prob_fused, source
    """
    if df_breakout.empty and df_bounce.empty:
        return pd.DataFrame()

    if strategy == "breakout_only":
        return df_breakout.head(top_k).assign(source="breakout")

    if strategy == "bounce_only":
        return df_bounce.head(top_k).assign(source="bounce")

    if strategy == "intersection":
        # 取交集，按概率和排序
        breakout_set = set(df_breakout.head(top_k)["ts_code"])
        bounce_set = set(df_bounce.head(top_k)["ts_code"])
        common = breakout_set & bounce_set
        if not common:
            log.warning("交集为空，回退到等权策略")
            strategy = "equal"
        else:
            merged = pd.merge(
                df_breakout[["ts_code", "prob_cal"]].rename(columns={"prob_cal": "prob_bo"}),
                df_bounce[["ts_code", "prob_cal"]].rename(columns={"prob_cal": "prob_bu"}),
                on="ts_code",
                how="inner",
            )
            merged["prob_fused"] = merged["prob_bo"] + merged["prob_bu"]
            merged = merged.sort_values("prob_fused", ascending=False).head(top_k)
            return merged[["ts_code", "prob_fused"]].assign(source="intersection")

    if strategy == "equal":
        # 等权：各取一半，去重
        half_k = top_k // 2
        bo_top = df_breakout.head(half_k)[["ts_code", "prob_cal"]].copy()
        bu_top = df_bounce.head(half_k)[["ts_code", "prob_cal"]].copy()

        # 去重：如果某股票同时被两个模型推荐，保留概率高的
        combined = pd.concat([bo_top, bu_top], ignore_index=True)
        combined = combined.sort_values("prob_cal", ascending=False).drop_duplicates("ts_code")

        # 如果去重后不足 top_k，从剩余股票补全
        if len(combined) < top_k:
            remaining_bo = df_breakout[~df_breakout["ts_code"].isin(combined["ts_code"])]
            remaining_bu = df_bounce[~df_bounce["ts_code"].isin(combined["ts_code"])]
            remaining = pd.concat(
                [remaining_bo[["ts_code", "prob_cal"]], remaining_bu[["ts_code", "prob_cal"]]]
            ).sort_values("prob_cal", ascending=False)
            n_fill = top_k - len(combined)
            combined = pd.concat([combined, remaining.head(n_fill)], ignore_index=True)

        combined = combined.sort_values("prob_cal", ascending=False).head(top_k).reset_index(drop=True)
        return combined.rename(columns={"prob_cal": "prob_fused"}).assign(source="equal")

    raise ValueError(f"未知融合策略: {strategy}")


# ============================================================================
# 预测
# ============================================================================
def run_predictions(start_date: str, end_date: str, top_k: int, fusion_strategy: str = "equal") -> dict:
    """运行预测，返回每日信号（支持单模型跳过）"""
    log.info(f"{'='*60}")
    log.info("v3.1.0 预测")
    log.info(f"{'='*60}")

    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

    bo_results = {}
    bu_results = {}

    # Breakout预测（仅在非 bounce_only 时运行）
    if fusion_strategy != "bounce_only":
        bo_predictor = BreakoutPredictor()
        bo_results = bo_predictor.predict_range(start_date, end_date, top_k=None)
        log.success(f"Breakout预测完成: {len(bo_results)} 个交易日")

    # Bounce预测（仅在非 breakout_only 时运行）
    if fusion_strategy != "breakout_only":
        bu_predictor = BouncePredictor()
        bu_results = bu_predictor.predict_range(start_date, end_date, top_k=None)
        log.success(f"Bounce预测完成: {len(bu_results)} 个交易日")

    # 获取所有交易日
    all_dates = sorted(set(bo_results.keys()) | set(bu_results.keys()))
    log.info(f"预测覆盖 {len(all_dates)} 个交易日")

    fused_results = {}
    for date in all_dates:
        df_bo = bo_results.get(date, pd.DataFrame())
        df_bu = bu_results.get(date, pd.DataFrame())

        df_fused = fuse_signals(df_bo, df_bu, strategy=fusion_strategy, top_k=top_k)

        if not df_fused.empty:
            # 保存各模型原始预测
            if not df_bo.empty:
                df_bo.head(top_k).to_csv(PREDICTION_DIR / f"breakout_{date}_top{top_k}.csv", index=False)
            if not df_bu.empty:
                df_bu.head(top_k).to_csv(PREDICTION_DIR / f"bounce_{date}_top{top_k}.csv", index=False)
            # 保存融合结果
            if fusion_strategy == "breakout_only":
                prefix = "breakout_"
            elif fusion_strategy == "bounce_only":
                prefix = "bounce_"
            else:
                prefix = "fused_"
            df_fused.to_csv(PREDICTION_DIR / f"{prefix}{date}_top{top_k}.csv", index=False)
            fused_results[date] = df_fused

    log.success(f"预测完成: {len(fused_results)} 个交易日")
    return fused_results


# ============================================================================
# 回测
# ============================================================================
def run_qlib_backtest(start_date: str, end_date: str, top_k: int, fusion_strategy: str):
    """qlib 风格回测"""
    from src.backtest.qlib_backtest import QlibStyleBacktest

    log.info(f"{'='*60}")
    log.info("qlib 风格回测")
    log.info(f"{'='*60}")

    # qlib回测需要读取预测文件
    bt = QlibStyleBacktest(prediction_dir=str(PREDICTION_DIR))
    # 根据策略选择预测文件前缀
    if fusion_strategy == "breakout_only":
        bt.prediction_prefix = "breakout_"
    elif fusion_strategy == "bounce_only":
        bt.prediction_prefix = "bounce_"
    elif fusion_strategy == "equal":
        bt.prediction_prefix = "fused_equal_"
    elif fusion_strategy == "intersection":
        bt.prediction_prefix = "fused_intersection_"
    else:
        bt.prediction_prefix = "fused_"

    result = bt.run(
        start_date=start_date,
        end_date=end_date,
        top_k=top_k,
        drop_n=5,
        hold_days=3,  # v3.1.0 保持3日持有期
    )

    if result:
        out = PROJECT_ROOT / "data" / "backtest" / "v310" / "qlib"
        out.mkdir(parents=True, exist_ok=True)
        bt.save_report(result, out / "report.json")
        result["daily_returns"].to_csv(out / "daily_returns.csv", index=False)
        result["portfolio"].to_csv(out / "portfolio.csv", index=False)
        log.success(f"qlib 回测报告: {out / 'report.json'}")
    return result


def run_vbt_backtest(start_date: str, end_date: str, top_k: int, fusion_strategy: str):
    """vectorbt 回测"""
    from src.backtest.vbt_backtest import VBTBacktest

    log.info(f"{'='*60}")
    log.info("vectorbt 回测")
    log.info(f"{'='*60}")

    bt = VBTBacktest(prediction_dir=str(PREDICTION_DIR))
    if fusion_strategy == "breakout_only":
        bt.prediction_prefix = "breakout_"
    elif fusion_strategy == "bounce_only":
        bt.prediction_prefix = "bounce_"
    elif fusion_strategy == "equal":
        bt.prediction_prefix = "fused_equal_"
    elif fusion_strategy == "intersection":
        bt.prediction_prefix = "fused_intersection_"
    else:
        bt.prediction_prefix = "fused_"

    result = bt.run(
        start_date=start_date,
        end_date=end_date,
        top_k=top_k,
        hold_days=3,
        stop_loss=0.10,  # 10% 止损
    )

    if result:
        out = PROJECT_ROOT / "data" / "backtest" / "v310" / "vbt"
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "report.json", "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        log.success(f"vbt 回测报告: {out / 'report.json'}")
    return result


# ============================================================================
# 主流程
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="v3.1.0 双模型预测+回测")
    parser.add_argument("--start", required=True, help="开始日期 YYYYMMDD")
    parser.add_argument("--end", required=True, help="结束日期 YYYYMMDD")
    parser.add_argument(
        "--fusion",
        default="equal",
        choices=["equal", "intersection", "breakout_only", "bounce_only"],
        help="信号融合策略",
    )
    parser.add_argument("--top_k", type=int, default=20, help="每日选股数量")
    parser.add_argument("--backtest", action="store_true", help="是否运行回测")
    parser.add_argument("--skip_prediction", action="store_true", help="跳过预测（已有预测文件）")
    args = parser.parse_args()

    log.info("=" * 80)
    log.info("v3.1.0 双模型预测 + 回测流水线")
    log.info(f"  日期范围: {args.start} ~ {args.end}")
    log.info(f"  融合策略: {args.fusion}")
    log.info(f"  每日选股: {args.top_k}")
    log.info(f"  回测: {'是' if args.backtest else '否'}")
    log.info("=" * 80)

    # 1. 预测
    if not args.skip_prediction:
        run_predictions(args.start, args.end, args.top_k, fusion_strategy=args.fusion)
    else:
        log.info("跳过预测，使用已有预测文件")

    # 2. 回测
    if args.backtest:
        run_qlib_backtest(args.start, args.end, args.top_k, args.fusion)
        run_vbt_backtest(args.start, args.end, args.top_k, args.fusion)

    log.info("\n" + "=" * 80)
    log.info("流水线完成")
    log.info("=" * 80)


if __name__ == "__main__":
    main()
