#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分析指定股票在指定时间段内的模型评分趋势
对比模型评分排名与实际股价走势，判断模型评分的前置性
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def get_trading_dates(start_date, end_date):
    """获取交易日列表（从预测结果文件中推断）"""
    results_dir = PROJECT_ROOT / "data" / "prediction" / "results"
    trading_dates = []

    # 遍历日期范围，查找存在的预测文件
    start_dt = datetime.strptime(start_date, "%Y%m%d")
    end_dt = datetime.strptime(end_date, "%Y%m%d")
    current_dt = start_dt

    while current_dt <= end_dt:
        date_str = current_dt.strftime("%Y%m%d")
        file_path = results_dir / f"v2.3.2_full_{date_str}.csv"
        if file_path.exists():
            trading_dates.append(date_str)
        current_dt += timedelta(days=1)

    return sorted(trading_dates)


def load_v232_predictions(date):
    """加载v2.3.2模型预测结果"""
    results_dir = PROJECT_ROOT / "data" / "prediction" / "results"
    file_path = results_dir / f"v2.3.2_full_{date}.csv"

    if not file_path.exists():
        return None

    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"读取{date}预测结果失败: {e}")
        return None


def get_stock_price_from_predictions(ts_code, trading_dates):
    """从预测结果文件中提取价格数据"""
    results_dir = PROJECT_ROOT / "data" / "prediction" / "results"
    price_data = []

    for date in trading_dates:
        file_path = results_dir / f"v2.3.2_full_{date}.csv"
        if not file_path.exists():
            continue

        try:
            df = pd.read_csv(file_path)
            stock_data = df[df["ts_code"] == ts_code]
            if not stock_data.empty:
                row = stock_data.iloc[0]
                price_data.append(
                    {
                        "trade_date": date,
                        "close": row.get("close", 0),
                        "pct_chg": row.get("pct_chg", 0),
                    }
                )
        except Exception:
            continue

    if not price_data:
        return None

    df = pd.DataFrame(price_data)
    df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
    return df


def analyze_stock_score_trend(ts_code, stock_name, start_date, end_date):
    """分析股票模型评分趋势"""

    print("=" * 100)
    print(f"分析股票: {stock_name} ({ts_code})")
    print(f"时间段: {start_date} 至 {end_date}")
    print("=" * 100)

    # 获取交易日列表
    trading_dates = get_trading_dates(start_date, end_date)
    print(f"\n交易日数量: {len(trading_dates)}")

    # 从预测结果中提取股价数据
    print("\n[1] 从预测结果中提取股价数据...")
    price_df = get_stock_price_from_predictions(ts_code, trading_dates)
    if price_df is None or price_df.empty:
        print("无法获取股价数据")
        return

    # 收集预测数据
    print("\n[2] 收集模型预测数据...")
    predictions = []

    for date in trading_dates:
        pred_df = load_v232_predictions(date)
        if pred_df is None:
            continue

        # 查找该股票
        stock_data = pred_df[pred_df["ts_code"] == ts_code]
        if stock_data.empty:
            continue

        row = stock_data.iloc[0]

        # 计算排名
        pred_df_sorted = pred_df.sort_values("final_score", ascending=False).reset_index(drop=True)
        rank = pred_df_sorted[pred_df_sorted["ts_code"] == ts_code].index[0] + 1
        total_stocks = len(pred_df)

        predictions.append(
            {
                "date": date,
                "close": row.get("close", 0),
                "pct_chg": row.get("pct_chg", 0),
                "raw_probability": row.get("raw_probability", 0),
                "calibrated_probability": row.get("calibrated_probability", 0),
                "expected_return_score": row.get("expected_return_score", 0),
                "final_score": row.get("final_score", 0),
                "rank": rank,
                "total_stocks": total_stocks,
                "rank_percentile": (total_stocks - rank + 1) / total_stocks * 100,
                "rsi_6": row.get("rsi_6", 0),
                "momentum_strength": row.get("momentum_strength", 0),
            }
        )

    if not predictions:
        print("未找到任何预测数据")
        return

    pred_df = pd.DataFrame(predictions)
    pred_df["date"] = pd.to_datetime(pred_df["date"])

    # 合并股价数据
    price_df["trade_date_str"] = price_df["trade_date"].dt.strftime("%Y%m%d")
    pred_df["date_str"] = pred_df["date"].dt.strftime("%Y%m%d")

    # 重命名price_df的列以避免冲突
    price_df_renamed = price_df[["trade_date_str", "close", "pct_chg"]].rename(
        columns={"close": "close_actual", "pct_chg": "pct_chg_actual"}
    )

    merged_df = pred_df.merge(price_df_renamed, left_on="date_str", right_on="trade_date_str", how="left")

    # 计算未来收益率（用于判断前置性）
    print("\n[3] 计算未来收益率...")
    merged_df = merged_df.sort_values("date")
    merged_df["future_1d_return"] = merged_df["close_actual"].pct_change(1).shift(-1) * 100
    merged_df["future_3d_return"] = merged_df["close_actual"].pct_change(3).shift(-3) * 100
    merged_df["future_5d_return"] = merged_df["close_actual"].pct_change(5).shift(-5) * 100
    merged_df["future_10d_return"] = merged_df["close_actual"].pct_change(10).shift(-10) * 100

    # 计算累计收益率（以第一个交易日为基准）
    first_price = merged_df["close_actual"].iloc[0]
    merged_df["cumulative_return"] = (merged_df["close_actual"] - first_price) / first_price * 100

    # 输出分析结果
    print("\n" + "=" * 100)
    print("详细数据表")
    print("=" * 100)
    print(
        f"{'日期':<12} {'收盘价':<10} {'当日涨跌':<10} {'模型评分':<12} {'排名':<8} {'排名分位':<10} {'未来1日':<10} {'未来3日':<10} {'未来5日':<10} {'累计收益':<10}"
    )
    print("-" * 100)

    for _, row in merged_df.iterrows():
        close_actual = row["close_actual"] if pd.notna(row["close_actual"]) else row.get("close", 0)
        pct_chg = (
            row.get("pct_chg_actual", row.get("pct_chg", 0))
            if pd.notna(row.get("pct_chg_actual", row.get("pct_chg", 0)))
            else 0
        )
        future_1d = row["future_1d_return"] if pd.notna(row["future_1d_return"]) else 0
        future_3d = row["future_3d_return"] if pd.notna(row["future_3d_return"]) else 0
        future_5d = row["future_5d_return"] if pd.notna(row["future_5d_return"]) else 0
        cumulative = row["cumulative_return"] if pd.notna(row["cumulative_return"]) else 0

        print(
            f"{row['date'].strftime('%Y-%m-%d'):<12} "
            f"{close_actual:<10.2f} "
            f"{pct_chg:>+7.2f}% "
            f"{row['final_score']:<12.4f} "
            f"{int(row['rank']):<8} "
            f"{row['rank_percentile']:>6.1f}% "
            f"{future_1d:>+7.2f}% "
            f"{future_3d:>+7.2f}% "
            f"{future_5d:>+7.2f}% "
            f"{cumulative:>+7.2f}%"
        )

    # 统计分析
    print("\n" + "=" * 100)
    print("统计分析")
    print("=" * 100)

    print("\n模型评分统计:")
    print(f"  平均评分: {merged_df['final_score'].mean():.4f}")
    print(
        f"  最高评分: {merged_df['final_score'].max():.4f} (日期: {merged_df.loc[merged_df['final_score'].idxmax(), 'date'].strftime('%Y-%m-%d')})"
    )
    print(
        f"  最低评分: {merged_df['final_score'].min():.4f} (日期: {merged_df.loc[merged_df['final_score'].idxmin(), 'date'].strftime('%Y-%m-%d')})"
    )
    print(f"  评分标准差: {merged_df['final_score'].std():.4f}")

    print("\n排名统计:")
    print(f"  平均排名: {merged_df['rank'].mean():.0f}")
    print(
        f"  最佳排名: {merged_df['rank'].min()} (日期: {merged_df.loc[merged_df['rank'].idxmin(), 'date'].strftime('%Y-%m-%d')})"
    )
    print(
        f"  最差排名: {merged_df['rank'].max()} (日期: {merged_df.loc[merged_df['rank'].idxmax(), 'date'].strftime('%Y-%m-%d')})"
    )
    print(f"  平均排名分位数: {merged_df['rank_percentile'].mean():.1f}%")

    print("\n股价统计:")
    print(f"  起始价格: {merged_df['close_actual'].iloc[0]:.2f}")
    print(f"  结束价格: {merged_df['close_actual'].iloc[-1]:.2f}")
    print(
        f"  期间涨跌: {(merged_df['close_actual'].iloc[-1] - merged_df['close_actual'].iloc[0]) / merged_df['close_actual'].iloc[0] * 100:+.2f}%"
    )
    print(
        f"  最高价格: {merged_df['close_actual'].max():.2f} (日期: {merged_df.loc[merged_df['close_actual'].idxmax(), 'date'].strftime('%Y-%m-%d')})"
    )
    print(
        f"  最低价格: {merged_df['close_actual'].min():.2f} (日期: {merged_df.loc[merged_df['close_actual'].idxmin(), 'date'].strftime('%Y-%m-%d')})"
    )

    # 前置性分析
    print("\n" + "=" * 100)
    print("前置性分析")
    print("=" * 100)

    # 计算评分与未来收益的相关性
    valid_data = merged_df.dropna(subset=["final_score", "future_1d_return", "future_3d_return", "future_5d_return"])

    if len(valid_data) > 1:
        corr_1d = valid_data["final_score"].corr(valid_data["future_1d_return"])
        corr_3d = valid_data["final_score"].corr(valid_data["future_3d_return"])
        corr_5d = valid_data["final_score"].corr(valid_data["future_5d_return"])

        print("\n模型评分与未来收益率的相关性:")
        print(f"  与未来1日收益率相关性: {corr_1d:.4f}")
        print(f"  与未来3日收益率相关性: {corr_3d:.4f}")
        print(f"  与未来5日收益率相关性: {corr_5d:.4f}")

        # 分析评分变化与股价变化的关系
        merged_df["score_change"] = merged_df["final_score"].diff()
        merged_df["price_change"] = merged_df["close_actual"].pct_change() * 100

        # 计算评分变化与未来价格变化的相关性
        valid_change = merged_df.dropna(subset=["score_change", "future_1d_return"])
        if len(valid_change) > 1:
            corr_score_change = valid_change["score_change"].corr(valid_change["future_1d_return"])
            print(f"\n评分变化与未来1日收益率相关性: {corr_score_change:.4f}")

        # 分析高评分后的表现
        high_score_threshold = merged_df["final_score"].quantile(0.75)
        high_score_days = merged_df[merged_df["final_score"] >= high_score_threshold]

        if len(high_score_days) > 0:
            print(f"\n高评分日(评分>={high_score_threshold:.4f})的后续表现:")
            print(f"  高评分日数量: {len(high_score_days)}")
            print(f"  高评分日后1日平均收益: {high_score_days['future_1d_return'].mean():+.2f}%")
            print(f"  高评分日后3日平均收益: {high_score_days['future_3d_return'].mean():+.2f}%")
            print(f"  高评分日后5日平均收益: {high_score_days['future_5d_return'].mean():+.2f}%")

        # 分析低评分后的表现
        low_score_threshold = merged_df["final_score"].quantile(0.25)
        low_score_days = merged_df[merged_df["final_score"] <= low_score_threshold]

        if len(low_score_days) > 0:
            print(f"\n低评分日(评分<={low_score_threshold:.4f})的后续表现:")
            print(f"  低评分日数量: {len(low_score_days)}")
            print(f"  低评分日后1日平均收益: {low_score_days['future_1d_return'].mean():+.2f}%")
            print(f"  低评分日后3日平均收益: {low_score_days['future_3d_return'].mean():+.2f}%")
            print(f"  低评分日后5日平均收益: {low_score_days['future_5d_return'].mean():+.2f}%")

    # 保存结果
    output_dir = PROJECT_ROOT / "data" / "prediction" / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{ts_code}_{stock_name}_score_analysis_{start_date}_{end_date}.csv"
    merged_df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"\n结果已保存到: {output_file}")

    print("\n" + "=" * 100)
    print("分析完成")
    print("=" * 100)


if __name__ == "__main__":
    # 分析航天工程
    ts_code = "603698.SH"
    stock_name = "航天工程"
    start_date = "20251231"
    end_date = "20260123"

    analyze_stock_score_trend(ts_code, stock_name, start_date, end_date)
