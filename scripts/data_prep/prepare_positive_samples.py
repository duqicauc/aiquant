"""
准备正样本数据的主脚本

运行步骤：
1. 筛选符合条件的正样本
2. 提取每个样本T1前34天的特征数据
3. 保存结果
"""

import sys
from pathlib import Path
import warnings

# 过滤 pandas FutureWarning（来自 tushare 库内部，不影响功能）
warnings.filterwarnings("ignore", category=FutureWarning, module="tushare")

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_manager import DataManager
from src.models.screening.positive_sample_screener import PositiveSampleScreener
from src.utils.logger import log
from src.utils.human_intervention import HumanInterventionChecker, require_human_confirmation
from config.settings import settings
import pandas as pd
from datetime import datetime


def main():
    """主函数"""

    log.info("=" * 80)
    log.info("正样本数据准备流程")
    log.info("=" * 80)

    # 定义文件路径
    samples_file = PROJECT_ROOT / "data" / "training" / "samples" / "positive_samples.csv"
    features_file = PROJECT_ROOT / "data" / "training" / "processed" / "feature_data_34d.csv"

    # 从配置文件读取日期范围
    START_DATE = settings.get("data.sample_preparation.start_date", "20000101")
    END_DATE = settings.get("data.sample_preparation.end_date", None)

    # 检查是否已有正样本数据
    if samples_file.exists():
        log.info(f"\n📂 检测到已有正样本文件: {samples_file}")
        df_samples = pd.read_csv(samples_file)
        log.success(f"✓ 已加载本地正样本数据，共 {len(df_samples)} 条记录")

        # 打印统计信息
        log.info("\n" + "=" * 80)
        log.info("正样本统计（本地数据）")
        log.info("=" * 80)
        log.info(f"样本总数: {len(df_samples)}")
        log.info(f"股票数量: {df_samples['ts_code'].nunique()}")
        if "total_return" in df_samples.columns:
            log.info(f"平均总涨幅: {df_samples['total_return'].mean():.2f}%")
        if "max_return" in df_samples.columns:
            log.info(f"平均最高涨幅: {df_samples['max_return'].mean():.2f}%")

        # 询问是否重新筛选
        use_existing = require_human_confirmation("是否使用已有的正样本数据？（选择N将重新筛选）", default=True)

        if not use_existing:
            df_samples = None  # 标记需要重新筛选
            log.info("将重新筛选正样本...")
    else:
        df_samples = None
        log.info("\n📂 未检测到正样本文件，将进行筛选...")

    # 1. 初始化数据管理器
    log.info("\n[步骤1] 初始化数据管理器...")
    dm = DataManager(source="tushare")

    # 2. 初始化筛选器（从配置文件读取正样本筛选参数）
    log.info("\n[步骤2] 初始化正样本筛选器...")

    # v2.4.0: 从配置文件读取正样本筛选参数
    positive_criteria = settings.get("data.sample_preparation.positive_criteria", {})
    screener_config = {
        "consecutive_weeks": positive_criteria.get("consecutive_weeks", 3),
        "total_return_threshold": positive_criteria.get("total_return_threshold", 50),
        "max_return_threshold": positive_criteria.get("max_return_threshold", 70),
        "min_listing_days": positive_criteria.get("min_listing_days", 180),
        # v2.4.0新增：反追龙头约束
        "pre_t1_return_max": positive_criteria.get("pre_t1_return_max", 25),
        "pre_t1_volatility_max": positive_criteria.get("pre_t1_volatility_max", 4),
        "enable_anti_chasing": positive_criteria.get("enable_anti_chasing", True),
    }

    log.info(f"筛选配置: {screener_config}")
    screener = PositiveSampleScreener(dm, config=screener_config)

    # 3. 筛选正样本（仅当没有本地数据时）
    if df_samples is None:
        log.info("\n[步骤3] 开始筛选正样本（这可能需要较长时间）...")
        log.info("筛选条件：")
        log.info("  - 周K连续三周收阳线")
        log.info("  - 总涨幅超50%")
        log.info("  - 最高涨幅超70%")
        log.info("  - 剔除ST股票")
        log.info("  - 上市超过半年")
        log.info("")

        log.info(f"📅 数据范围配置：{START_DATE} - {END_DATE or '今天'}")
        log.info("💡 如需修改，请编辑 config/settings.yaml")

        # 👤 人工介入检查：正样本筛选条件
        checker = HumanInterventionChecker()
        criteria_check = checker.check_positive_sample_criteria()
        needs_intervention = checker.print_intervention_reminder("正样本筛选条件", criteria_check)

        if needs_intervention:
            confirmed = require_human_confirmation(
                "⚠️  检测到正样本筛选条件可能需要调整。\n" "请检查上述警告和建议，确认是否继续使用当前配置。",
                default=True,  # 默认继续，在自动模式下会自动确认
            )
            if not confirmed:
                log.warning("用户取消操作。请修改 config/settings.yaml 后重新运行。")
                return

        try:
            df_samples = screener.screen_all_stocks(start_date=START_DATE, end_date=END_DATE)

            if df_samples.empty:
                log.error("未找到符合条件的正样本！请检查筛选条件或数据质量")
                return

            # 保存正样本列表
            samples_file.parent.mkdir(parents=True, exist_ok=True)
            df_samples.to_csv(samples_file, index=False, encoding="utf-8-sig")
            log.success(f"✓ 正样本列表已保存: {samples_file}")

            # 打印统计信息
            log.info("\n" + "=" * 80)
            log.info("正样本统计")
            log.info("=" * 80)
            log.info(f"样本总数: {len(df_samples)}")
            log.info(f"股票数量: {df_samples['ts_code'].nunique()}")
            log.info(f"平均总涨幅: {df_samples['total_return'].mean():.2f}%")
            log.info(f"平均最高涨幅: {df_samples['max_return'].mean():.2f}%")
            log.info("\n前5个样本:")
            print(df_samples.head())

            # 👤 人工介入提醒：检查正样本质量
            log.warning("\n" + "=" * 80)
            log.warning("👤 人工介入提醒：请检查正样本质量")
            log.warning("=" * 80)
            log.warning("请确认：")
            log.warning("  1. 样本数量是否合理（建议：1000-5000个）")
            log.warning("  2. 平均涨幅是否符合预期")
            log.warning("  3. 样本分布是否合理")
            log.warning("  4. 是否需要调整筛选条件")
            log.warning("=" * 80)

        except Exception as e:
            log.error(f"正样本筛选失败: {e}")
            import traceback

            traceback.print_exc()
            return
    else:
        log.info("\n[步骤3] 跳过正样本筛选（使用本地数据）")

    try:
        # 4. 提取特征数据
        log.info("\n[步骤4] 提取特征数据（T1前34天）...")

        df_features = screener.extract_features(df_samples, lookback_days=34)

        if df_features.empty:
            log.error("特征提取失败！")
            return

        # 4.1 数据质量处理
        log.info("\n[步骤4.1] 数据质量处理...")

        # 统计原始缺失值
        missing_before = df_features.isnull().sum()
        total_missing_before = missing_before.sum()
        log.info(f"原始缺失值总数: {total_missing_before}")
        if total_missing_before > 0:
            for col, count in missing_before.items():
                if count > 0:
                    log.info(f"  - {col}: {count} ({count/len(df_features)*100:.2f}%)")

        # 定义需要填充的数值列
        numeric_cols = [
            "close",
            "pct_chg",
            "total_mv",
            "circ_mv",
            "ma5",
            "ma10",
            "volume_ratio",
            "macd_dif",
            "macd_dea",
            "macd",
            "rsi_6",
            "rsi_12",
            "rsi_24",
        ]
        numeric_cols = [col for col in numeric_cols if col in df_features.columns]

        # 按样本分组进行前向填充+后向填充
        log.info("执行缺失值填充（按样本分组：前向填充 + 后向填充）...")
        df_features[numeric_cols] = df_features.groupby("sample_id")[numeric_cols].transform(
            lambda x: x.ffill().bfill()
        )

        # 检查填充后的缺失值
        missing_after = df_features.isnull().sum()
        total_missing_after = missing_after.sum()
        log.info(f"填充后缺失值总数: {total_missing_after}")

        # 4.2 过滤数据不足的样本
        log.info("\n[步骤4.2] 过滤数据不足的样本...")
        min_days = 30  # 最少需要30天数据

        days_per_sample = df_features.groupby("sample_id").size()
        valid_samples = days_per_sample[days_per_sample >= min_days].index
        invalid_samples = days_per_sample[days_per_sample < min_days]

        if len(invalid_samples) > 0:
            log.warning(f"发现 {len(invalid_samples)} 个样本数据不足{min_days}天，将被过滤:")
            for sample_id, days in invalid_samples.items():
                sample_info = df_features[df_features["sample_id"] == sample_id].iloc[0]
                log.warning(f"  - 样本{sample_id}: {sample_info['ts_code']} {sample_info['name']} - 仅{days}天")

            df_features = df_features[df_features["sample_id"].isin(valid_samples)]
            log.info(f"过滤后剩余样本数: {df_features['sample_id'].nunique()}")
            log.info(f"过滤后剩余记录数: {len(df_features)}")
        else:
            log.success(f"✓ 所有样本数据完整（均≥{min_days}天）")

        # 4.3 最终数据质量检查
        log.info("\n[步骤4.3] 最终数据质量检查...")
        final_missing = df_features.isnull().sum().sum()
        if final_missing > 0:
            log.warning(f"仍有 {final_missing} 个缺失值，将使用列均值填充...")
            df_features[numeric_cols] = df_features[numeric_cols].fillna(df_features[numeric_cols].mean())
        log.success(f"✓ 数据质量处理完成，最终缺失值: {df_features.isnull().sum().sum()}")

        # 保存特征数据
        features_file.parent.mkdir(parents=True, exist_ok=True)
        df_features.to_csv(features_file, index=False, encoding="utf-8-sig")
        log.success(f"✓ 特征数据已保存: {features_file}")

        # 特征统计
        log.info("\n" + "=" * 80)
        log.info("特征数据统计")
        log.info("=" * 80)
        log.info(f"总记录数: {len(df_features)}")
        log.info(f"样本数: {df_features['sample_id'].nunique()}")
        log.info(f"每样本天数: {len(df_features) / df_features['sample_id'].nunique():.1f}")
        log.info("\n数据字段:")
        for col in df_features.columns:
            log.info(f"  - {col}")
        log.info("\n数据预览:")
        print(df_features.head(10))

        # 5. 生成统计报告
        log.info("\n[步骤5] 生成统计报告...")

        stats = {
            "generation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "date_range": f"{START_DATE} - {END_DATE or 'today'}",
            "total_samples": int(len(df_samples)),
            "unique_stocks": int(df_samples["ts_code"].nunique()),
            "avg_total_return": float(df_samples["total_return"].mean()),
            "avg_max_return": float(df_samples["max_return"].mean()),
            "feature_records": int(len(df_features)),
            "feature_samples": int(df_features["sample_id"].nunique()),
            "lookback_days": 34,
            "min_days_required": min_days,
            "data_quality": {
                "missing_values_before": int(total_missing_before),
                "missing_values_after": int(df_features.isnull().sum().sum()),
                "filtered_samples": int(len(invalid_samples)) if len(invalid_samples) > 0 else 0,
                "avg_days_per_sample": float(df_features.groupby("sample_id").size().mean()),
            },
            "sample_files": {"positive_samples": str(samples_file), "feature_data": str(features_file)},
        }

        import json

        stats_file = PROJECT_ROOT / "data" / "training" / "processed" / "sample_statistics.json"
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        log.success(f"✓ 统计报告已保存: {stats_file}")

        # 完成
        log.info("\n" + "=" * 80)
        log.success("✅ 正样本数据准备完成！")
        log.info("=" * 80)
        log.info("\n生成的文件:")
        log.info(f"  1. 正样本列表: {samples_file}")
        log.info(f"  2. 特征数据: {features_file}")
        log.info(f"  3. 统计报告: {stats_file}")
        log.info("\n下一步:")
        log.info("  - 查看数据质量")
        log.info("  - 准备负样本")
        log.info("  - 开始模型训练")

    except Exception as e:
        log.error(f"执行失败: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
