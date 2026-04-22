#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
预测效果回顾脚本
跟踪历史预测的实际表现，计算胜率和收益
"""

import sys
import json
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import log
from src.data.data_manager import DataManager


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="预测效果回顾")
    parser.add_argument("--period", type=str, default="1w", help="回顾周期: 1w(1周), 2w(2周), 4w(4周), 6w(6周)")
    parser.add_argument("--prediction_date", type=str, default=None, help="预测日期 (格式: YYYYMMDD)，默认最新预测")
    parser.add_argument("--top_n", type=int, default=50, help="回顾Top N推荐")
    return parser.parse_args()


def get_latest_prediction():
    """获取最新的预测记录"""
    index_file = project_root / "data" / "predictions" / "index.json"

    if not index_file.exists():
        log.error("预测索引文件不存在！")
        return None

    with open(index_file, "r", encoding="utf-8") as f:
        index = json.load(f)

    if not index["predictions"]:
        log.error("没有找到预测记录！")
        return None

    return index["predictions"][0]  # 最新的预测


def load_prediction_data(prediction_date):
    """加载预测数据"""
    pred_dir = project_root / "data" / "predictions" / prediction_date

    if not pred_dir.exists():
        log.error(f"预测目录不存在: {pred_dir}")
        return None

    # 查找 top stocks 文件
    import glob

    top_files = glob.glob(str(pred_dir / "top_*.csv"))

    if not top_files:
        log.error(f"找不到推荐股票文件: {pred_dir}")
        return None

    df = pd.read_csv(top_files[0])
    log.info(f"加载预测数据: {len(df)} 只股票")

    return df


def calculate_returns(df_predictions, period_weeks):
    """计算实际收益"""
    log.info(f"\n计算 {period_weeks} 周收益...")

    dm = DataManager()

    results = []

    for idx, row in df_predictions.iterrows():
        stock_code = row["股票代码"]
        stock_name = row["股票名称"]
        pred_price = row["最新价格"]
        pred_date = row["数据日期"]
        probability = row["牛股概率"]

        # 计算结束日期
        pred_dt = pd.to_datetime(pred_date)
        end_dt = pred_dt + timedelta(weeks=period_weeks)
        end_date = end_dt.strftime("%Y%m%d")

        try:
            # 获取期间数据
            data = dm.get_daily_data(stock_code=stock_code, start_date=pred_date.replace("-", ""), end_date=end_date)

            if data is None or len(data) == 0:
                log.warning(f"  {stock_name} 无数据")
                continue

            # 计算收益
            start_price = data.iloc[0]["close"]
            end_price = data.iloc[-1]["close"]
            actual_return = (end_price - start_price) / start_price * 100

            # 计算期间最高和最低
            max_price = data["high"].max()
            min_price = data["low"].min()
            max_return = (max_price - start_price) / start_price * 100
            max_drawdown = (min_price - start_price) / start_price * 100

            results.append(
                {
                    "股票代码": stock_code,
                    "股票名称": stock_name,
                    "预测概率": probability,
                    "预测价格": pred_price,
                    "期初价格": start_price,
                    "期末价格": end_price,
                    "实际收益%": actual_return,
                    "期间最高收益%": max_return,
                    "期间最大回撤%": max_drawdown,
                    "是否盈利": actual_return > 0,
                    "数据日期": pred_date,
                    "交易天数": len(data),
                }
            )

            if (idx + 1) % 10 == 0:
                log.info(f"  进度: {idx+1}/{len(df_predictions)}")

        except Exception as e:
            log.warning(f"  {stock_name} 处理失败: {e}")
            continue

    df_results = pd.DataFrame(results)
    log.success(f"✓ 成功计算 {len(df_results)} 只股票的收益")

    return df_results


def analyze_performance(df_results, period_weeks):
    """分析预测表现"""
    log.info("=" * 80)
    log.info("📊 预测效果分析")
    log.info("=" * 80)

    if len(df_results) == 0:
        log.error("没有可分析的数据！")
        return None

    analysis = {}

    # 1. 整体收益统计
    analysis["总体表现"] = {
        "评估股票数": len(df_results),
        "平均收益%": df_results["实际收益%"].mean(),
        "中位数收益%": df_results["实际收益%"].median(),
        "最大收益%": df_results["实际收益%"].max(),
        "最大亏损%": df_results["实际收益%"].min(),
        "收益标准差%": df_results["实际收益%"].std(),
    }

    # 2. 胜率统计
    win_count = (df_results["实际收益%"] > 0).sum()
    total_count = len(df_results)
    win_rate = win_count / total_count * 100

    analysis["胜率统计"] = {
        "盈利股票数": int(win_count),
        "亏损股票数": int(total_count - win_count),
        "整体胜率%": win_rate,
    }

    # 3. 分概率区间统计
    df_results["概率区间"] = pd.cut(
        df_results["预测概率"], bins=[0, 0.7, 0.8, 0.9, 1.0], labels=["<70%", "70-80%", "80-90%", ">90%"]
    )

    group_stats = []
    for prob_range, group in df_results.groupby("概率区间"):
        group_stats.append(
            {
                "概率区间": prob_range,
                "数量": len(group),
                "胜率%": (group["实际收益%"] > 0).sum() / len(group) * 100,
                "平均收益%": group["实际收益%"].mean(),
            }
        )

    analysis["分层表现"] = group_stats

    # 4. 风险指标
    returns = df_results["实际收益%"].values / 100
    sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(52 / period_weeks) if np.std(returns) > 0 else 0

    analysis["风险指标"] = {
        "夏普比率": sharpe_ratio,
        "平均最大回撤%": df_results["期间最大回撤%"].mean(),
    }

    return analysis


def generate_review_report(prediction_date, period_weeks, df_results, analysis):
    """生成回顾报告"""
    report = []
    report.append("=" * 80)
    report.append("📊 预测效果回顾报告")
    report.append("=" * 80)

    # 计算回顾日期范围
    pred_dt = pd.to_datetime(prediction_date)
    end_dt = pred_dt + timedelta(weeks=period_weeks)

    report.append(f"\n📅 预测日期: {pred_dt.strftime('%Y年%m月%d日')}")
    report.append(f"⏱️  回顾周期: {period_weeks}周 ({pred_dt.strftime('%Y-%m-%d')} 至 {end_dt.strftime('%Y-%m-%d')})")
    report.append(f"📈 评估股票: {len(df_results)} 只")

    # 一、整体表现
    report.append("\n" + "=" * 80)
    report.append("一、整体表现")
    report.append("=" * 80)

    overall = analysis["总体表现"]
    report.append("\n1. 收益统计")
    report.append(f"   - 平均收益率: {overall['平均收益%']:+.2f}%")
    report.append(f"   - 中位数收益率: {overall['中位数收益%']:+.2f}%")
    report.append(f"   - 最大收益: {overall['最大收益%']:+.2f}%")
    report.append(f"   - 最大亏损: {overall['最大亏损%']:+.2f}%")
    report.append(f"   - 收益波动率: {overall['收益标准差%']:.2f}%")

    winrate = analysis["胜率统计"]
    report.append("\n2. 胜率统计")
    report.append(
        f"   - 整体胜率: {winrate['整体胜率%']:.1f}% ({winrate['盈利股票数']}/{winrate['盈利股票数'] + winrate['亏损股票数']})"
    )
    report.append(f"   - 盈利股票: {winrate['盈利股票数']} 只")
    report.append(f"   - 亏损股票: {winrate['亏损股票数']} 只")

    risk = analysis["风险指标"]
    report.append("\n3. 风险指标")
    report.append(f"   - 夏普比率: {risk['夏普比率']:.2f}")
    report.append(f"   - 平均最大回撤: {risk['平均最大回撤%']:.2f}%")

    # 二、分层表现
    report.append("\n" + "=" * 80)
    report.append("二、分层表现分析")
    report.append("=" * 80)

    report.append(f"\n{'概率区间':<12} {'数量':<8} {'胜率':<12} {'平均收益':<12}")
    report.append("-" * 50)

    for stat in analysis["分层表现"]:
        report.append(
            f"{stat['概率区间']:<12} {stat['数量']:<8} " f"{stat['胜率%']:<11.1f}% {stat['平均收益%']:<11.2f}%"
        )

    # 三、Top 10 表现回顾
    report.append("\n" + "=" * 80)
    report.append("三、Top 10 表现回顾")
    report.append("=" * 80)

    df_top10 = df_results.head(10).copy()

    for i, row in df_top10.iterrows():
        status = "✅" if row["实际收益%"] > 0 else "❌"
        report.append(f"\n【第 {i+1} 名】{row['股票名称']}（{row['股票代码']}）")
        report.append(f"  预测概率: {row['预测概率']*100:.2f}%")
        report.append(f"  预测价格: {row['预测价格']:.2f}")
        report.append(f"  期末价格: {row['期末价格']:.2f}")
        report.append(f"  实际收益: {row['实际收益%']:+.2f}% {status}")

        if row["实际收益%"] > 10:
            comment = "表现优秀，超预期"
        elif row["实际收益%"] > 5:
            comment = "表现良好，符合预期"
        elif row["实际收益%"] > 0:
            comment = "小幅盈利"
        elif row["实际收益%"] > -10:
            comment = "小幅亏损"
        else:
            comment = "表现不佳"

        report.append(f"  评价: {comment}")

    # 四、模型评估
    report.append("\n" + "=" * 80)
    report.append("四、模型评估")
    report.append("=" * 80)

    # 检查模型校准度
    if overall["平均收益%"] > 5 and winrate["整体胜率%"] > 60:
        report.append("\n✅ 模型表现优秀")
        report.append("   - 平均收益和胜率都达到预期目标")
        report.append("   - 建议继续使用当前模型")
    elif overall["平均收益%"] > 3 and winrate["整体胜率%"] > 55:
        report.append("\n✅ 模型表现良好")
        report.append("   - 表现基本符合预期")
        report.append("   - 可继续观察后续表现")
    else:
        report.append("\n⚠️  模型表现需要改进")
        report.append("   - 平均收益或胜率低于预期")
        report.append("   - 建议检查模型并考虑重新训练")

    # 检查概率校准
    layered = analysis["分层表现"]
    high_prob_group = [g for g in layered if g["概率区间"] == ">90%"]
    if high_prob_group and high_prob_group[0]["胜率%"] > 70:
        report.append("\n✅ 模型校准度良好")
        report.append("   - 高概率股票确实表现更好")
    else:
        report.append("\n⚠️  模型校准度有待提升")
        report.append("   - 概率与实际表现相关性不强")

    # 五、改进建议
    report.append("\n" + "=" * 80)
    report.append("五、改进建议")
    report.append("=" * 80)

    report.append("\n1. 选股策略")
    if high_prob_group and high_prob_group[0]["数量"] > 5:
        report.append(f"   💡 重点关注概率>{high_prob_group[0]['概率区间']}的股票")
    report.append("   💡 结合基本面和技术面进行二次筛选")

    report.append("\n2. 风控建议")
    if abs(overall["最大亏损%"]) > 15:
        report.append("   ⚠️  存在较大单股亏损，建议严格执行止损")
    report.append("   💡 建议止损点: -15%")
    report.append(f"   💡 建议止盈点: +{max(20, overall['平均收益%']*2):.0f}%")

    report.append("\n3. 仓位管理")
    report.append("   💡 单股仓位不超过5-10%")
    report.append("   💡 优先配置高概率股票")

    # 结束
    report.append("\n" + "=" * 80)
    report.append("报告结束")
    report.append("=" * 80)

    return "\n".join(report)


def save_review_results(prediction_date, period_weeks, df_results, report_content):
    """保存回顾结果"""
    # 创建回顾目录
    review_dir = project_root / "data" / "reviews"
    review_dir.mkdir(parents=True, exist_ok=True)

    # 保存详细结果CSV
    csv_file = review_dir / f"review_{prediction_date}_{period_weeks}w_detail.csv"
    df_results.to_csv(csv_file, index=False, encoding="utf-8-sig")
    log.success(f"✓ 详细结果已保存: {csv_file}")

    # 保存报告
    report_file = review_dir / f"review_{prediction_date}_{period_weeks}w.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_content)
    log.success(f"✓ 回顾报告已保存: {report_file}")

    # 打印报告
    log.info("\n" + report_content)

    return csv_file, report_file


def main():
    """主函数"""
    args = parse_args()

    log.info("=" * 80)
    log.info("📊 开始预测效果回顾")
    log.info("=" * 80)

    # 解析周期
    period_map = {"1w": 1, "2w": 2, "4w": 4, "6w": 6}
    period_weeks = period_map.get(args.period, 1)

    log.info(f"回顾周期: {period_weeks} 周")

    # 获取预测日期
    if args.prediction_date:
        prediction_date = args.prediction_date
    else:
        latest = get_latest_prediction()
        if not latest:
            log.error("无法获取预测记录！")
            return
        prediction_date = latest["date"]

    log.info(f"预测日期: {prediction_date}")

    # 检查是否已经过去足够的时间
    pred_dt = datetime.strptime(prediction_date, "%Y%m%d")
    days_passed = (datetime.now() - pred_dt).days

    if days_passed < period_weeks * 7:
        log.warning(f"⚠️  距离预测日期仅过去 {days_passed} 天，可能数据不完整")
        log.warning(f"   建议等待 {period_weeks * 7} 天后再进行回顾")
        response = input("是否继续? (y/n): ")
        if response.lower() != "y":
            log.info("已取消")
            return

    # 加载预测数据
    df_predictions = load_prediction_data(prediction_date)
    if df_predictions is None:
        return

    # 限制Top N
    df_predictions = df_predictions.head(args.top_n)

    # 计算实际收益
    df_results = calculate_returns(df_predictions, period_weeks)
    if len(df_results) == 0:
        log.error("无法计算收益！")
        return

    # 分析表现
    analysis = analyze_performance(df_results, period_weeks)
    if not analysis:
        return

    # 生成报告
    report_content = generate_review_report(prediction_date, period_weeks, df_results, analysis)

    # 保存结果
    save_review_results(prediction_date, period_weeks, df_results, report_content)

    log.success("\n✅ 预测效果回顾完成！")


if __name__ == "__main__":
    main()
