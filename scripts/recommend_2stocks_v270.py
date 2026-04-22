#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v2.7.0集成模型 - 自动推荐2支股票

根据模型预测结果，自动筛选出最适合实盘操作的2支股票
"""
import sys
import pandas as pd
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import log


def calculate_comprehensive_score(row):
    """计算综合得分"""
    # 模型概率得分 (0-40分)
    prob_score = row["probability"] * 40

    # 收益潜力得分 (0-30分)
    # 基于历史最高收益，假设未来可能达到类似水平
    max_return = row.get("max_return", 0)
    if max_return > 0:
        return_potential = min(max_return / 20 * 30, 30)  # 最高收益20%得30分
    else:
        return_potential = 15  # 默认15分

    # 风险控制得分 (0-30分)
    # 回撤越小得分越高
    max_drawdown = abs(row.get("max_drawdown", -2))
    if max_drawdown < 1:
        risk_score = 30
    elif max_drawdown < 1.5:
        risk_score = 25
    elif max_drawdown < 2:
        risk_score = 20
    else:
        risk_score = 10

    # 技术指标加分
    bonus = 0
    if row.get("rsi_6", 50) < 50:  # RSI未超买
        bonus += 5
    if row.get("pct_chg", 0) < 0:  # 预测日下跌，可能是买入机会
        bonus += 3
    if row.get("close", 10) < 5:  # 低价股
        bonus += 2

    total_score = prob_score + return_potential + risk_score + bonus

    return total_score


def recommend_2stocks(predict_file, eval_file=None):
    """推荐2支股票"""
    log.info("=" * 80)
    log.info("v2.7.0集成模型 - 2支股票推荐")
    log.info("=" * 80)

    # 读取预测结果
    df_pred = pd.read_csv(predict_file)
    log.info(f"加载预测结果: {len(df_pred)} 只股票")

    # 如果有评估结果，合并
    if eval_file and Path(eval_file).exists():
        df_eval = pd.read_csv(eval_file)
        df = pd.merge(df_pred, df_eval[["ts_code", "returns", "max_return", "max_drawdown"]], on="ts_code", how="left")
        log.info("已合并评估结果")
    else:
        df = df_pred.copy()
        # 如果没有评估结果，使用默认值
        df["returns"] = 0
        df["max_return"] = df["probability"] * 15  # 估算
        df["max_drawdown"] = -2  # 默认

    # 筛选条件
    log.info("\n应用筛选条件...")

    # 必选条件
    mask = (df["probability"] >= 0.63) & (df["rsi_6"] < 70) & (df["pct_chg"] < 5)  # 模型概率  # 未超买  # 未追高

    if "max_drawdown" in df.columns:
        mask = mask & (df["max_drawdown"] > -2.5)  # 风险可控

    df_filtered = df[mask].copy()
    log.info(f"筛选后剩余: {len(df_filtered)} 只股票")

    if len(df_filtered) < 2:
        log.warning("符合条件的股票不足2只，放宽条件...")
        mask = (df["probability"] >= 0.63) & (df["rsi_6"] < 80)
        df_filtered = df[mask].copy()

    # 计算综合得分
    df_filtered["comprehensive_score"] = df_filtered.apply(calculate_comprehensive_score, axis=1)

    # 排序并选择Top 2
    df_filtered = df_filtered.sort_values("comprehensive_score", ascending=False)
    top2 = df_filtered.head(2)

    # 输出推荐结果
    log.info("\n" + "=" * 80)
    log.info("推荐2支股票")
    log.info("=" * 80)

    for idx, (_, row) in enumerate(top2.iterrows(), 1):
        log.info(f"\n【推荐{idx}】{row['ts_code']} {row['name']}")
        log.info(f"  集成概率: {row['probability']:.4f}")
        log.info(f"  预测价: {row['close']:.2f}元")
        log.info(f"  预测日涨幅: {row['pct_chg']:+.2f}%")
        log.info(f"  RSI_6: {row['rsi_6']:.1f}")
        if "returns" in row and not pd.isna(row["returns"]):
            log.info(f"  历史收益: {row['returns']:+.2f}%")
        if "max_return" in row and not pd.isna(row["max_return"]):
            log.info(f"  最高收益: {row['max_return']:+.2f}%")
        if "max_drawdown" in row and not pd.isna(row["max_drawdown"]):
            log.info(f"  最大回撤: {row['max_drawdown']:+.2f}%")
        log.info(f"  综合得分: {row['comprehensive_score']:.1f}")

    # 组合分析
    log.info("\n" + "=" * 80)
    log.info("组合分析")
    log.info("=" * 80)

    if "returns" in top2.columns and top2["returns"].notna().all():
        avg_return = top2["returns"].mean()
        win_rate = (top2["returns"] > 0).sum() / len(top2) * 100
        avg_max_return = top2["max_return"].mean() if "max_return" in top2.columns else 0
        avg_drawdown = top2["max_drawdown"].mean() if "max_drawdown" in top2.columns else 0

        log.info(f"  平均收益: {avg_return:+.2f}%")
        log.info(f"  胜率: {win_rate:.0f}%")
        if avg_max_return > 0:
            log.info(f"  平均最高收益: {avg_max_return:+.2f}%")
        if avg_drawdown < 0:
            log.info(f"  平均最大回撤: {avg_drawdown:+.2f}%")

    # 仓位建议
    log.info("\n仓位建议:")
    log.info(f"  {top2.iloc[0]['name']}: 50%")
    log.info(f"  {top2.iloc[1]['name']}: 50%")

    # 保存推荐结果
    output_file = (
        PROJECT_ROOT
        / "data"
        / "prediction"
        / "trading_plan"
        / f'v270_recommended_2stocks_{datetime.now().strftime("%Y%m%d")}.csv'
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)

    top2_output = top2[
        [
            "ts_code",
            "name",
            "probability",
            "close",
            "pct_chg",
            "rsi_6",
            "returns",
            "max_return",
            "max_drawdown",
            "comprehensive_score",
        ]
    ].copy()
    top2_output.to_csv(output_file, index=False)
    log.info(f"\n推荐结果已保存: {output_file}")

    return top2


def main():
    import sys

    # 从命令行参数获取预测文件，或使用默认
    if len(sys.argv) > 1:
        predict_file = Path(sys.argv[1])
        if not predict_file.is_absolute():
            predict_file = PROJECT_ROOT / predict_file
    else:
        # 默认使用最新的预测结果（按日期排序）
        results_dir = PROJECT_ROOT / "data" / "prediction" / "results"
        predict_files = sorted(results_dir.glob("v270_ensemble_top10_*.csv"), reverse=True)
        if predict_files:
            predict_file = predict_files[0]
            log.info(f"使用最新预测文件: {predict_file.name}")
        else:
            log.error("未找到预测文件")
            return

    if not predict_file.exists():
        log.error(f"预测文件不存在: {predict_file}")
        return

    # 尝试找到对应的评估文件
    predict_date = predict_file.stem.split("_")[-1]  # 从文件名提取日期
    eval_dir = PROJECT_ROOT / "data" / "prediction" / "evaluation"
    eval_files = list(eval_dir.glob(f"v270_ensemble_eval_{predict_date}_to_*.csv"))
    eval_file = eval_files[0] if eval_files else None

    recommend_2stocks(predict_file, eval_file)


if __name__ == "__main__":
    main()
