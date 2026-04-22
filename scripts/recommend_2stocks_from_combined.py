#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从互补策略结果中选择股票（支持2只、3只或更多）

基于互补策略的结果，智能选择最适合实盘操作的股票
支持偏好设置：热门板块、高收益等
"""
import sys
import argparse
import warnings
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings("ignore")

from src.utils.logger import log
from src.data.data_manager import DataManager


def get_technical_indicators(dm, ts_code, predict_date):
    """
    获取股票的技术指标（MACD、KDJ）

    Returns:
        dict: {
            'macd_signal': 'golden_cross'/'death_cross'/'bullish'/'bearish'/'neutral',
            'kdj_signal': 'golden_cross'/'death_cross'/'bullish'/'bearish'/'neutral',
            'macd_dif': float,
            'macd_dea': float,
            'kdj_k': float,
            'kdj_d': float,
            'kdj_j': float
        }
    """
    try:
        end_date = predict_date
        start_date = (datetime.strptime(predict_date, "%Y%m%d") - timedelta(days=100)).strftime("%Y%m%d")

        df = dm.get_daily_data(ts_code, start_date, end_date)
        if df is None or len(df) < 30:
            return None

        df = df.sort_values("trade_date").reset_index(drop=True)
        close = df["close"]
        high = df["high"]
        low = df["low"]

        # 计算MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_dif = ema12 - ema26
        macd_dea = macd_dif.ewm(span=9, adjust=False).mean()

        # 判断MACD金叉死叉
        if len(macd_dif) >= 2:
            prev_dif = macd_dif.iloc[-2]
            prev_dea = macd_dea.iloc[-2]
            curr_dif = macd_dif.iloc[-1]
            curr_dea = macd_dea.iloc[-1]

            if prev_dif <= prev_dea and curr_dif > curr_dea:
                macd_signal = "golden_cross"  # 金叉
            elif prev_dif >= prev_dea and curr_dif < curr_dea:
                macd_signal = "death_cross"  # 死叉
            elif curr_dif > curr_dea:
                macd_signal = "bullish"  # 多头
            else:
                macd_signal = "bearish"  # 空头
        else:
            macd_signal = "neutral"

        # 计算KDJ
        low_9 = low.rolling(9).min()
        high_9 = high.rolling(9).max()
        rsv = (close - low_9) / (high_9 - low_9 + 1e-8) * 100
        kdj_k = rsv.ewm(com=2, adjust=False).mean()
        kdj_d = kdj_k.ewm(com=2, adjust=False).mean()
        kdj_j = 3 * kdj_k - 2 * kdj_d

        # 判断KDJ金叉死叉
        if len(kdj_k) >= 2:
            prev_k = kdj_k.iloc[-2]
            prev_d = kdj_d.iloc[-2]
            curr_k = kdj_k.iloc[-1]
            curr_d = kdj_d.iloc[-1]

            if prev_k <= prev_d and curr_k > curr_d:
                kdj_signal = "golden_cross"  # 金叉
            elif prev_k >= prev_d and curr_k < curr_d:
                kdj_signal = "death_cross"  # 死叉
            elif curr_k > curr_d:
                kdj_signal = "bullish"  # 多头
            else:
                kdj_signal = "bearish"  # 空头
        else:
            kdj_signal = "neutral"

        return {
            "macd_signal": macd_signal,
            "kdj_signal": kdj_signal,
            "macd_dif": float(macd_dif.iloc[-1]) if len(macd_dif) > 0 else 0,
            "macd_dea": float(macd_dea.iloc[-1]) if len(macd_dea) > 0 else 0,
            "kdj_k": float(kdj_k.iloc[-1]) if len(kdj_k) > 0 else 50,
            "kdj_d": float(kdj_d.iloc[-1]) if len(kdj_d) > 0 else 50,
            "kdj_j": float(kdj_j.iloc[-1]) if len(kdj_j) > 0 else 50,
        }
    except Exception as e:
        log.debug(f"获取{ts_code}技术指标失败: {e}")
        return None


def calculate_selection_score(row, prefer_hot_sector=True, prefer_high_return=True, tech_indicators=None):
    """
    计算选择得分

    综合考虑：
    1. 综合得分（dual_score）- 基础
    2. 风险等级（低风险优先）
    3. 热门板块（有加成，可调整权重）
    4. 收益潜力（基于涨幅和动量，可调整权重）
    5. 价格合理性（适中价格）
    6. RSI状态（未超买）

    Args:
        row: 股票数据行
        prefer_hot_sector: 是否偏好热门板块（默认True）
        prefer_high_return: 是否偏好高收益（默认True）
    """
    score = 0

    # 1. 综合得分（基础分，0-50分）
    dual_score = row.get("dual_score", 0)
    score += dual_score * 50

    # 2. 风险等级（0-15分）
    risk_level = row.get("risk_level", "low")
    if risk_level == "low":
        score += 15
    elif risk_level == "medium":
        score += 8
    else:
        score += 3  # 高风险也有基础分，因为可能带来高收益

    # 3. 热门板块（0-25分，可调整权重）
    is_hot = row.get("is_hot_sector", False)
    if pd.notna(is_hot) and is_hot:
        if prefer_hot_sector:
            score += 25  # 热门板块大幅加分
        else:
            score += 10  # 普通权重

    # 4. 收益潜力（0-20分，可调整权重）
    if prefer_high_return:
        # 基于当日涨幅
        pct_chg = row.get("pct_chg", 0)
        if pd.notna(pct_chg):
            if pct_chg > 5:
                score += 20  # 涨幅>5%，高收益潜力
            elif pct_chg > 2:
                score += 15
            elif pct_chg > 0:
                score += 10
            elif pct_chg > -2:
                score += 5  # 小幅下跌可能是买入机会
            else:
                score += 0

        # 基于v2.3.2的预期收益（如果有）
        if pd.notna(row.get("final_score")):
            final_score = row.get("final_score", 0)
            if final_score > 0.6:
                score += 5  # 高预期收益加分

        # 基于动量（如果有）
        momentum = row.get("momentum_10d", 0)
        if pd.notna(momentum) and momentum > 5:
            score += 5  # 强动量加分
    else:
        # 不偏好高收益时，小幅下跌反而更好（买入机会）
        pct_chg = row.get("pct_chg", 0)
        if pd.notna(pct_chg):
            if -2 <= pct_chg <= 2:
                score += 10  # 平稳
            elif pct_chg < -2:
                score += 5  # 下跌可能是机会

    # 5. RSI状态（0-5分，未超买但也不要太低）
    rsi_6 = row.get("rsi_6", 50)
    if pd.notna(rsi_6):
        if 40 <= rsi_6 < 60:
            score += 5  # RSI适中最佳
        elif 30 <= rsi_6 < 70:
            score += 3
        elif rsi_6 < 30:
            score += 1  # 超卖，可能反弹
        elif rsi_6 < 80:
            score += 1  # 轻度超买
        else:
            score += 0  # 严重超买

    # 6. MACD和KDJ技术信号（0-10分）
    if tech_indicators:
        macd_signal = tech_indicators.get("macd_signal", "neutral")
        kdj_signal = tech_indicators.get("kdj_signal", "neutral")

        # MACD信号评分
        if macd_signal == "golden_cross":
            score += 5  # MACD金叉，强烈买入信号
        elif macd_signal == "bullish":
            score += 3  # MACD多头
        elif macd_signal == "death_cross":
            score -= 3  # MACD死叉，卖出信号
        elif macd_signal == "bearish":
            score -= 1  # MACD空头

        # KDJ信号评分
        if kdj_signal == "golden_cross":
            score += 5  # KDJ金叉，买入信号
        elif kdj_signal == "bullish":
            score += 2  # KDJ多头
        elif kdj_signal == "death_cross":
            score -= 2  # KDJ死叉，卖出信号
        elif kdj_signal == "bearish":
            score -= 1  # KDJ空头

        # 双重金叉加分（MACD和KDJ同时金叉）
        if macd_signal == "golden_cross" and kdj_signal == "golden_cross":
            score += 3  # 双重金叉，强烈买入信号

        # 双重死叉减分
        if macd_signal == "death_cross" and kdj_signal == "death_cross":
            score -= 5  # 双重死叉，强烈卖出信号

    return score


def recommend_stocks_from_combined(combined_file, date=None, prefer_hot_sector=True, prefer_high_return=True, top_n=2):
    """
    从互补策略结果中选择股票

    Args:
        combined_file: 互补策略结果文件路径
        date: 日期（用于输出文件名）
        prefer_hot_sector: 是否偏好热门板块（默认True）
        prefer_high_return: 是否偏好高收益（默认True）
        top_n: 推荐股票数量（默认2）
    """
    log.info("=" * 80)
    log.info(f"从互补策略结果选择{top_n}只股票")
    log.info("=" * 80)
    if prefer_hot_sector:
        log.info("【偏好热门板块】")
    if prefer_high_return:
        log.info("【偏好高收益】")
    log.info("")

    # 读取结果
    if not Path(combined_file).exists():
        log.error(f"文件不存在: {combined_file}")
        return None

    df = pd.read_csv(combined_file)
    log.info(f"加载结果: {len(df)} 只股票")

    if len(df) < top_n:
        log.error(f"股票数量不足{top_n}只，无法推荐")
        return None

    # 获取技术指标（仅对Top20进行技术分析，提高效率）
    log.info("\n获取技术指标（MACD、KDJ）...")
    dm = DataManager()
    tech_indicators_dict = {}

    # 先按综合得分排序，只对Top20进行技术分析
    df_temp = df.sort_values("dual_score", ascending=False).head(20)

    for _, row in df_temp.iterrows():
        ts_code = row["ts_code"]
        tech_indicators = get_technical_indicators(dm, ts_code, date or datetime.now().strftime("%Y%m%d"))
        if tech_indicators:
            tech_indicators_dict[ts_code] = tech_indicators

    log.info(f"✓ 获取到 {len(tech_indicators_dict)} 只股票的技术指标")

    # 计算选择得分（包含技术指标）
    def calc_score(row):
        tech_indicators = tech_indicators_dict.get(row["ts_code"], None)
        return calculate_selection_score(row, prefer_hot_sector, prefer_high_return, tech_indicators)

    df["selection_score"] = df.apply(calc_score, axis=1)

    # 添加技术指标信息到DataFrame（用于输出和保存）
    df["macd_signal"] = df["ts_code"].apply(lambda x: tech_indicators_dict.get(x, {}).get("macd_signal", "neutral"))
    df["kdj_signal"] = df["ts_code"].apply(lambda x: tech_indicators_dict.get(x, {}).get("kdj_signal", "neutral"))
    df["macd_dif"] = df["ts_code"].apply(lambda x: tech_indicators_dict.get(x, {}).get("macd_dif", 0))
    df["macd_dea"] = df["ts_code"].apply(lambda x: tech_indicators_dict.get(x, {}).get("macd_dea", 0))
    df["kdj_k"] = df["ts_code"].apply(lambda x: tech_indicators_dict.get(x, {}).get("kdj_k", 50))
    df["kdj_d"] = df["ts_code"].apply(lambda x: tech_indicators_dict.get(x, {}).get("kdj_d", 50))

    # 排序并选择Top N（保持互补性：v2.7.0和v2.3.2交替选择）
    df_v270 = df[df["source"] == "v2.7.0"].sort_values("selection_score", ascending=False)
    df_v232 = df[df["source"] == "v2.3.2"].sort_values("selection_score", ascending=False)

    # 确定每个来源的配额（各占一半，至少1只）
    v270_quota = max(1, top_n // 2)
    v232_quota = max(1, top_n - v270_quota)

    # 交替选择
    selected = []
    v270_idx, v232_idx = 0, 0
    while len(selected) < top_n:
        # 优先选择v2.7.0（稳定）
        if v270_idx < len(df_v270) and v270_idx < v270_quota:
            selected.append(df_v270.iloc[v270_idx].to_dict())
            v270_idx += 1
        # 然后选择v2.3.2（热门）
        if len(selected) < top_n and v232_idx < len(df_v232) and v232_idx < v232_quota:
            selected.append(df_v232.iloc[v232_idx].to_dict())
            v232_idx += 1
        # 如果某一方不够，从另一方补充
        if v270_idx >= min(len(df_v270), v270_quota) and v232_idx >= min(len(df_v232), v232_quota):
            break

    # 如果数量不足，从剩余的股票中补充
    remaining_v270 = df_v270.iloc[v270_idx:] if v270_idx < len(df_v270) else pd.DataFrame()
    remaining_v232 = df_v232.iloc[v232_idx:] if v232_idx < len(df_v232) else pd.DataFrame()
    remaining = pd.concat([remaining_v270, remaining_v232]).sort_values("selection_score", ascending=False)

    while len(selected) < top_n and len(remaining) > 0:
        selected.append(remaining.iloc[0].to_dict())
        remaining = remaining.iloc[1:]

    top_stocks = pd.DataFrame(selected)

    # 统计来源配比
    v270_count = sum(1 for s in selected if s.get("source") == "v2.7.0")
    v232_count = sum(1 for s in selected if s.get("source") == "v2.3.2")
    log.info(f"\n来源配比: v2.7.0={v270_count}只, v2.3.2={v232_count}只")

    # 输出推荐结果
    log.info("\n" + "=" * 80)
    log.info(f"🏆 推荐{top_n}只股票")
    log.info("=" * 80)

    for idx, (_, row) in enumerate(top_stocks.iterrows(), 1):
        log.info(f"\n【推荐{idx}】{row['ts_code']} {row['name']}")
        log.info(f"  综合得分: {row['dual_score']:.4f}")
        log.info(f"  选择得分: {row['selection_score']:.2f}")
        log.info(f"  来源模型: {row['source']}")
        log.info(f"  风险等级: {row['risk_level']}")
        log.info(f"  当前价格: {row['close']:.2f}元")

        # v2.7.0概率
        if pd.notna(row.get("v270_prob")):
            log.info(f"  v2.7.0概率: {row['v270_prob']:.4f}")

        # v2.3.2评分
        if pd.notna(row.get("v232_score_norm")):
            log.info(f"  v2.3.2评分: {row['v232_score_norm']:.4f}")
        elif pd.notna(row.get("final_score")):
            log.info(f"  v2.3.2评分: {row['final_score']:.4f}")

        # 热门板块
        hot_sectors = row.get("hot_sectors", "")
        if pd.notna(hot_sectors) and hot_sectors:
            log.info(f"  热门板块: {hot_sectors}")

        # 技术指标
        if pd.notna(row.get("rsi_6")):
            log.info(f"  RSI_6: {row['rsi_6']:.1f}")
        if pd.notna(row.get("pct_chg")):
            log.info(f"  当日涨幅: {row['pct_chg']:+.2f}%")

        # MACD信号
        macd_signal = row.get("macd_signal", "neutral")
        macd_signal_desc = {
            "golden_cross": "🟢 MACD金叉（买入信号）",
            "bullish": "🟡 MACD多头",
            "death_cross": "🔴 MACD死叉（卖出信号）",
            "bearish": "🔴 MACD空头",
            "neutral": "⚪ MACD中性",
        }
        log.info(f"  {macd_signal_desc.get(macd_signal, 'MACD: ' + macd_signal)}")
        if pd.notna(row.get("macd_dif")):
            log.info(f"    MACD_DIF: {row['macd_dif']:.4f}, MACD_DEA: {row['macd_dea']:.4f}")

        # KDJ信号
        kdj_signal = row.get("kdj_signal", "neutral")
        kdj_signal_desc = {
            "golden_cross": "🟢 KDJ金叉（买入信号）",
            "bullish": "🟡 KDJ多头",
            "death_cross": "🔴 KDJ死叉（卖出信号）",
            "bearish": "🔴 KDJ空头",
            "neutral": "⚪ KDJ中性",
        }
        log.info(f"  {kdj_signal_desc.get(kdj_signal, 'KDJ: ' + kdj_signal)}")
        if pd.notna(row.get("kdj_k")):
            log.info(f"    KDJ_K: {row['kdj_k']:.2f}, KDJ_D: {row['kdj_d']:.2f}")

        # 双重信号提示
        if macd_signal == "golden_cross" and kdj_signal == "golden_cross":
            log.info("  🎯 双重金叉！MACD和KDJ同时金叉，强烈买入信号")
        elif macd_signal == "death_cross" and kdj_signal == "death_cross":
            log.info("  ⚠️ 双重死叉！MACD和KDJ同时死叉，强烈卖出信号")

    # 组合分析
    log.info("\n" + "=" * 80)
    log.info("📊 组合分析")
    log.info("=" * 80)

    # 来源分布
    sources = top_stocks["source"].value_counts()
    log.info("来源分布:")
    for source, count in sources.items():
        log.info(f"  - {source}: {count} 只")

    # 风险分布
    risk_levels = top_stocks["risk_level"].value_counts()
    log.info("风险分布:")
    for risk, count in risk_levels.items():
        log.info(f"  - {risk}: {count} 只")

    # 热门板块
    hot_count = top_stocks.get("is_hot_sector", pd.Series([False] * len(top_stocks))).sum()
    log.info(f"热门板块股票: {hot_count} 只")

    # 平均价格
    avg_price = top_stocks["close"].mean()
    log.info(f"平均价格: {avg_price:.2f}元")

    # 平均综合得分
    avg_dual_score = top_stocks["dual_score"].mean()
    log.info(f"平均综合得分: {avg_dual_score:.4f}")

    # 仓位建议
    log.info("\n" + "=" * 80)
    log.info("💰 仓位建议")
    log.info("=" * 80)

    # 根据选择得分分配仓位（得分高的仓位稍多）
    total_score = top_stocks["selection_score"].sum()
    weights = []

    for idx, (_, row) in enumerate(top_stocks.iterrows()):
        if total_score > 0:
            weight = row["selection_score"] / total_score
        else:
            weight = 1.0 / len(top_stocks)
        weights.append(weight * 100)
        log.info(f"{row['name']} ({row['ts_code']}): {weight*100:.1f}%")

    # 保存推荐结果
    if date is None:
        date = datetime.now().strftime("%Y%m%d")

    output_dir = PROJECT_ROOT / "data" / "prediction" / "results"
    output_file = output_dir / f"v232_v270_recommended_{top_n}stocks_{date}.csv"

    # 选择要保存的列
    output_cols = [
        "ts_code",
        "name",
        "close",
        "source",
        "risk_level",
        "dual_score",
        "selection_score",
        "v270_prob",
        "hot_sectors",
        "rsi_6",
        "pct_chg",
        "macd_signal",
        "kdj_signal",
        "macd_dif",
        "macd_dea",
        "kdj_k",
        "kdj_d",
    ]
    output_cols = [col for col in output_cols if col in top_stocks.columns]

    top_stocks_output = top_stocks[output_cols].copy()
    top_stocks_output["weight"] = weights
    top_stocks_output.to_csv(output_file, index=False, encoding="utf-8-sig")

    log.success(f"\n✓ 推荐结果已保存: {output_file}")

    return top_stocks


def main():
    parser = argparse.ArgumentParser(description="从互补策略结果选择股票")
    parser.add_argument("--file", type=str, help="互补策略结果文件路径（如果不提供，会自动查找最新文件）")
    parser.add_argument("--date", type=str, help="日期(YYYYMMDD)，用于查找文件或输出文件名")
    parser.add_argument("--top-n", type=int, default=2, help="推荐股票数量（默认2）")
    parser.add_argument("--prefer-hot", action="store_true", default=True, help="偏好热门板块（默认开启）")
    parser.add_argument("--no-prefer-hot", dest="prefer_hot", action="store_false", help="不偏好热门板块")
    parser.add_argument("--prefer-return", action="store_true", default=True, help="偏好高收益（默认开启）")
    parser.add_argument(
        "--no-prefer-return", dest="prefer_return", action="store_false", help="不偏好高收益（偏好稳健）"
    )

    args = parser.parse_args()

    # 确定文件路径
    if args.file:
        combined_file = Path(args.file)
    else:
        # 自动查找最新文件
        results_dir = PROJECT_ROOT / "data" / "prediction" / "results"
        if args.date:
            combined_file = results_dir / f"v232_v270_complementary_{args.date}.csv"
        else:
            # 查找最新的互补策略结果
            pattern = "v232_v270_complementary_*.csv"
            files = list(results_dir.glob(pattern))
            if not files:
                log.error("未找到互补策略结果文件")
                log.error("请先运行: python scripts/combine_v232_v270.py --date YYYYMMDD --strategy complementary")
                return
            combined_file = max(files, key=lambda p: p.stat().st_mtime)
            log.info(f"自动找到最新文件: {combined_file.name}")

    # 确定日期
    date = args.date
    if not date and args.file:
        # 从文件名提取日期
        filename = Path(args.file).stem
        parts = filename.split("_")
        for part in reversed(parts):
            if part.isdigit() and len(part) == 8:
                date = part
                break

    log.info(f"偏好设置: 热门板块={args.prefer_hot}, 高收益={args.prefer_return}")
    log.info("")

    recommend_stocks_from_combined(
        combined_file, date, prefer_hot_sector=args.prefer_hot, prefer_high_return=args.prefer_return, top_n=args.top_n
    )


if __name__ == "__main__":
    main()
