"""
Tushare Pro 高级功能演示

展示如何使用Tushare Pro的高级API获取数据
"""

import sys
from pathlib import Path
import warnings

# 过滤 pandas FutureWarning（来自 tushare 库内部）
warnings.filterwarnings("ignore", category=FutureWarning, module="tushare")

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_manager import DataManager
from src.utils.logger import log


def demo_weekly_data():
    """演示：直接获取周线数据"""
    log.info("\n" + "=" * 80)
    log.info("演示1: 获取周线数据")
    log.info("=" * 80)

    dm = DataManager(source="tushare")

    # 使用Tushare Pro的weekly API直接获取周线
    df_weekly = dm.get_weekly_data(stock_code="600519.SH", start_date="20220101", end_date="20241231", adjust="qfq")

    log.success(f"✓ 获取周线数据: {len(df_weekly)} 周")
    print("\n周线数据预览：")
    print(df_weekly.head(10))

    log.info("\n优势：")
    log.info("  ✓ 无需本地转换日线数据")
    log.info("  ✓ 数据质量更高")
    log.info("  ✓ 支持复权")


def demo_daily_basic():
    """演示：获取每日指标"""
    log.info("\n" + "=" * 80)
    log.info("演示2: 获取每日指标（市值、量比等）")
    log.info("=" * 80)

    dm = DataManager(source="tushare")

    # 获取每日指标
    df_basic = dm.get_daily_basic(stock_code="600519.SH", start_date="20220101", end_date="20221231")

    log.success(f"✓ 获取每日指标: {len(df_basic)} 天")
    print("\n每日指标预览：")
    print(df_basic.head(10))

    log.info("\n包含字段：")
    for col in df_basic.columns:
        log.info(f"  - {col}")


def demo_stk_factor():
    """演示：获取技术因子（需要5000积分）"""
    log.info("\n" + "=" * 80)
    log.info("演示3: 获取技术因子（MA、MACD、KDJ、RSI等）")
    log.info("=" * 80)
    log.warning("⚠️  需要5000积分才能访问此接口")

    dm = DataManager(source="tushare")

    try:
        # 获取技术因子
        df_factor = dm.get_stk_factor(stock_code="600519.SH", start_date="20220101", end_date="20221231")

        if not df_factor.empty:
            log.success(f"✓ 获取技术因子: {len(df_factor)} 天")
            print("\n技术因子预览：")
            print(df_factor.head(10))

            log.info("\n包含的技术指标：")
            indicators = [col for col in df_factor.columns if col not in ["ts_code", "trade_date"]]
            for indicator in indicators[:20]:  # 只显示前20个
                log.info(f"  - {indicator}")

            if len(indicators) > 20:
                log.info(f"  ... 还有 {len(indicators)-20} 个指标")

            log.info("\n💡 优势：")
            log.info("  ✓ 无需本地计算任何技术指标")
            log.info("  ✓ 包含100+专业技术指标")
            log.info("  ✓ 数据质量高，专业团队维护")
            log.info("  ✓ 节省大量开发时间")
        else:
            log.warning("未获取到技术因子数据")

    except Exception as e:
        log.error(f"技术因子获取失败: {e}")
        log.info("\n可能原因：")
        log.info("  1. 积分不足（需要5000积分）")
        log.info("  2. 网络问题")
        log.info("  3. Token权限不足")
        log.info("\n💡 建议：")
        log.info("  - 访问 https://tushare.pro/community 捐赠获取积分")
        log.info("  - 技术因子API非常值得投资！")


def demo_complete_data():
    """演示：获取完整数据（行情+指标）"""
    log.info("\n" + "=" * 80)
    log.info("演示4: 获取完整数据（自动合并行情和指标）")
    log.info("=" * 80)

    dm = DataManager(source="tushare")

    # 一次性获取行情+指标
    df = dm.get_complete_data(stock_code="600519.SH", start_date="20220101", end_date="20221231", adjust="qfq")

    log.success(f"✓ 获取完整数据: {len(df)} 天")
    print("\n完整数据预览：")
    print(df.head(10))

    log.info("\n数据字段：")
    for col in df.columns:
        log.info(f"  - {col}")


def demo_trade_calendar():
    """演示：获取交易日历"""
    log.info("\n" + "=" * 80)
    log.info("演示5: 获取交易日历")
    log.info("=" * 80)

    dm = DataManager(source="tushare")

    # 获取交易日历
    df_cal = dm.get_trade_calendar(start_date="20240101", end_date="20241231", exchange="SSE")

    log.success(f"✓ 获取交易日历: {len(df_cal)} 天")

    # 筛选交易日
    trading_days = df_cal[df_cal["is_open"] == 1]
    log.info(f"  交易日: {len(trading_days)} 天")
    log.info(f"  非交易日: {len(df_cal) - len(trading_days)} 天")

    print("\n交易日历预览：")
    print(df_cal.head(10))

    log.info("\n用途：")
    log.info("  ✓ 准确计算交易日天数")
    log.info("  ✓ 回看N个交易日")
    log.info("  ✓ 回测系统必备")


def demo_comparison():
    """演示：对比本地计算 vs Tushare Pro"""
    log.info("\n" + "=" * 80)
    log.info("演示6: 性能对比")
    log.info("=" * 80)

    dm = DataManager(source="tushare")
    stock_code = "600519.SH"
    start_date = "20220101"
    end_date = "20221231"

    import time

    # 方法1: 本地转换（旧方法）
    log.info("\n方法1: 本地转换日线到周线")
    t1 = time.time()
    df_daily = dm.get_daily_data(stock_code, start_date, end_date, adjust="qfq")
    df_weekly_local = (
        df_daily.set_index("trade_date")
        .resample("W-FRI")
        .agg({"open": "first", "close": "last", "high": "max", "low": "min"})
    )
    t2 = time.time()
    log.info(f"  耗时: {t2-t1:.2f}秒")
    log.info(f"  数据量: {len(df_weekly_local)} 周")

    # 方法2: 直接获取周线（新方法）
    log.info("\n方法2: 直接使用Tushare Pro周线API")
    t1 = time.time()
    df_weekly_api = dm.get_weekly_data(stock_code, start_date, end_date, adjust="qfq")
    t2 = time.time()
    log.info(f"  耗时: {t2-t1:.2f}秒")
    log.info(f"  数据量: {len(df_weekly_api)} 周")

    log.info("\n✅ 结论：")
    log.info("  - Tushare Pro API更快、更准确")
    log.info("  - 无需本地复杂计算")
    log.info("  - 代码更简洁")


def main():
    """主函数"""
    log.info("=" * 80)
    log.info("Tushare Pro 高级功能演示")
    log.info("=" * 80)
    log.info("\n本脚本展示如何使用Tushare Pro的高级API")
    log.info("详细文档: docs/TUSHARE_PRO_FEATURES.md\n")

    try:
        # 演示1: 周线数据
        demo_weekly_data()

        # 演示2: 每日指标
        demo_daily_basic()

        # 演示3: 技术因子（可能需要5000积分）
        demo_stk_factor()

        # 演示4: 完整数据
        demo_complete_data()

        # 演示5: 交易日历
        demo_trade_calendar()

        # 演示6: 性能对比
        demo_comparison()

        # 总结
        log.info("\n" + "=" * 80)
        log.success("✅ 演示完成！")
        log.info("=" * 80)
        log.info("\n💡 重要提示：")
        log.info("  1. 基础功能（周线、每日指标）：免费或120积分")
        log.info("  2. 交易日历：2000积分")
        log.info("  3. 技术因子：5000积分（强烈推荐！）")
        log.info("\n💰 如何获取积分：")
        log.info("  - 注册: 120积分")
        log.info("  - 完善资料: 300积分")
        log.info("  - 每日签到: 1积分/天")
        log.info("  - 捐赠: 快速获得5000+积分（推荐）")
        log.info("\n📚 更多信息：")
        log.info("  - Tushare Pro文档: https://tushare.pro/document/2")
        log.info("  - 社区捐助: https://tushare.pro/community")
        log.info("  - 项目文档: docs/TUSHARE_PRO_FEATURES.md")

    except Exception as e:
        log.error(f"演示过程中出错: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
