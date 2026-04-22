"""
市场状态分析 - 命令行工具
判断当前市场是牛市、熊市还是震荡市
"""

import sys
from pathlib import Path
import json
from datetime import datetime
import argparse

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.analysis.market_analyzer import MarketAnalyzer


def print_market_report(report: dict):
    """打印市场分析报告"""

    print("\n" + "=" * 80)
    print("市场状态分析报告")
    print("=" * 80)

    print(f"\n📅 分析日期: {report['analysis_date']}")

    # 市场状态
    market_state = report.get("market_state", "未知")
    market_score = report.get("market_score", 50)

    print(f"\n{'=' * 80}")
    if market_score >= 70:
        print(f"🟢 【市场状态】: {market_state}")
    elif market_score >= 55:
        print(f"🔵 【市场状态】: {market_state}")
    elif market_score >= 45:
        print(f"🟡 【市场状态】: {market_state}")
    else:
        print(f"🔴 【市场状态】: {market_state}")

    # 综合评分
    stars = "★" * int(market_score / 20) + "☆" * (5 - int(market_score / 20))
    print(f"【综合评分】: {market_score:.2f}/100 {stars}")
    print("=" * 80)

    # 主要指数分析
    print("\n📊 【主要指数分析】")
    print("-" * 80)

    indices = report.get("indices_analysis", {})
    if indices:
        for name, analysis in indices.items():
            if name != "average_score" and isinstance(analysis, dict):
                state = analysis.get("state", "N/A")
                score = analysis.get("score", 0)
                trend = analysis.get("trend", {})

                print(f"\n  {name}:")
                print(f"    状态: {state} (评分: {score:.1f})")
                print(f"    均线排列: {trend.get('alignment', 'N/A')}")
                print(f"    5日涨幅: {trend.get('returns_5d', 0):.2f}%")
                print(f"    20日涨幅: {trend.get('returns_20d', 0):.2f}%")
                print(f"    60日涨幅: {trend.get('returns_60d', 0):.2f}%")

    # 市场广度
    print("\n📈 【市场广度分析】")
    print("-" * 80)

    breadth = report.get("market_breadth", {})
    if breadth:
        print(f"  状态: {breadth.get('state', 'N/A')}")
        print(f"  上涨家数: {breadth.get('up_count', 0)}")
        print(f"  下跌家数: {breadth.get('down_count', 0)}")
        print(f"  平盘家数: {breadth.get('flat_count', 0)}")
        print(f"  上涨比例: {breadth.get('up_ratio', 0):.2f}%")

        up_ratio = breadth.get("up_ratio", 0)
        if up_ratio > 70:
            print("  💡 市场普涨，赚钱效应好")
        elif up_ratio > 60:
            print("  💡 市场强势，多数股票上涨")
        elif up_ratio > 40:
            print("  💡 市场分化，结构性机会")
        elif up_ratio > 30:
            print("  💡 市场弱势，少数股票上涨")
        else:
            print("  💡 市场普跌，亏钱效应明显")

    # 市场情绪
    print("\n😱 【市场情绪分析】")
    print("-" * 80)

    sentiment = report.get("market_sentiment", {})
    if sentiment:
        fear_greed = sentiment.get("fear_greed_index", 50)
        sentiment_trend = sentiment.get("trend", "中性")

        print(f"  恐慌贪婪指数: {fear_greed:.2f}/100")
        print(f"  市场情绪: {sentiment_trend}")

        if fear_greed >= 75:
            print("  💡 市场情绪过热，注意回调风险")
        elif fear_greed >= 60:
            print("  💡 市场情绪积极，但需警惕过度乐观")
        elif fear_greed >= 45:
            print("  💡 市场情绪中性偏多，可适度参与")
        elif fear_greed >= 35:
            print("  💡 市场情绪中性，观望为主")
        elif fear_greed >= 25:
            print("  💡 市场情绪恐慌，谨慎操作")
        else:
            print("  💡 市场极度恐慌，可能是抄底机会")

    # 投资建议
    print("\n💡 【投资策略建议】")
    print("-" * 80)

    recommendations = report.get("recommendations", [])
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")

    print("\n" + "=" * 80)
    print("⚠️  风险提示: 市场判断仅供参考，投资需谨慎")
    print("=" * 80 + "\n")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="市场状态分析工具")
    parser.add_argument("--days", type=int, default=120, help="分析天数，默认120天")
    parser.add_argument("--save", action="store_true", help="保存分析报告")
    parser.add_argument("--json", action="store_true", help="输出JSON格式")

    args = parser.parse_args()

    try:
        print("\n🔍 开始分析市场状态...")

        analyzer = MarketAnalyzer()
        report = analyzer.analyze_market(days=args.days)

        if "error" in report:
            print(f"\n❌ 分析失败: {report['error']}")
            sys.exit(1)

        # JSON输出
        if args.json:
            print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
            sys.exit(0)

        # 打印报告
        print_market_report(report)

        # 保存报告
        if args.save:
            output_dir = Path("data/market_analysis")
            output_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # 保存JSON
            json_file = output_dir / f"market_report_{timestamp}.json"
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2, default=str)

            # 保存文本
            txt_file = output_dir / f"market_report_{timestamp}.txt"
            with open(txt_file, "w", encoding="utf-8") as f:
                import io
                import contextlib

                # 重定向print到文件
                f_buffer = io.StringIO()
                with contextlib.redirect_stdout(f_buffer):
                    print_market_report(report)
                f.write(f_buffer.getvalue())

            print("✅ 报告已保存:")
            print(f"   - JSON: {json_file}")
            print(f"   - TXT: {txt_file}\n")

    except KeyboardInterrupt:
        print("\n\n⚠️  分析被用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 分析失败: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
