#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
样本准备监控脚本

功能：
1. 监控正样本准备状态
2. 监控负样本准备状态
3. 当正负样本都准备好后，自动触发模型训练流程
"""
import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime
import pandas as pd
import json

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import log


class SamplePreparationMonitor:
    """样本准备监控器"""

    def __init__(self):
        """初始化监控器"""
        self.project_root = PROJECT_ROOT
        self.data_dir = self.project_root / "data" / "processed"

        # 正样本文件
        self.positive_samples_file = self.data_dir / "positive_samples.csv"
        self.positive_features_file = self.data_dir / "feature_data_34d.csv"
        self.positive_stats_file = self.data_dir / "sample_statistics.json"

        # 负样本文件（V2版本）
        self.negative_samples_file = self.data_dir / "negative_samples_v2.csv"
        self.negative_features_file = self.data_dir / "negative_feature_data_v2_34d.csv"
        self.negative_stats_file = self.data_dir / "negative_sample_statistics_v2.json"

        # 训练流程脚本
        self.quality_check_script = self.project_root / "scripts" / "check_sample_quality.py"
        self.train_script = self.project_root / "scripts" / "train_xgboost_timeseries.py"
        self.walk_forward_script = self.project_root / "scripts" / "walk_forward_validation.py"

    def check_positive_samples(self):
        """
        检查正样本是否准备好

        Returns:
            tuple: (是否准备好, 详细信息)
        """
        log.info("=" * 80)
        log.info("检查正样本准备状态")
        log.info("=" * 80)

        # 检查必需文件是否存在
        required_files = [self.positive_samples_file, self.positive_features_file]

        missing_files = []
        for file in required_files:
            if not file.exists():
                missing_files.append(str(file))

        if missing_files:
            log.warning("✗ 正样本文件缺失:")
            for f in missing_files:
                log.warning(f"  - {f}")
            return False, {"status": "missing_files", "files": missing_files}

        # 检查文件是否有效（非空）
        try:
            df_samples = pd.read_csv(self.positive_samples_file)
            df_features = pd.read_csv(self.positive_features_file)

            if df_samples.empty:
                log.warning("✗ 正样本列表为空")
                return False, {"status": "empty_samples"}

            if df_features.empty:
                log.warning("✗ 正样本特征数据为空")
                return False, {"status": "empty_features"}

            # 检查基本字段
            required_cols = ["ts_code", "t1_date", "total_return", "max_return"]
            missing_cols = [col for col in required_cols if col not in df_samples.columns]

            if missing_cols:
                log.warning(f"✗ 正样本缺少必需字段: {missing_cols}")
                return False, {"status": "missing_columns", "columns": missing_cols}

            # 统计信息
            stats = {
                "status": "ready",
                "sample_count": len(df_samples),
                "feature_count": len(df_features),
                "unique_stocks": df_samples["ts_code"].nunique(),
                "avg_total_return": float(df_samples["total_return"].mean()),
                "avg_max_return": float(df_samples["max_return"].mean()),
            }

            log.success("✓ 正样本已准备好")
            log.info(f"  样本数量: {stats['sample_count']}")
            log.info(f"  特征记录: {stats['feature_count']}")
            log.info(f"  股票数量: {stats['unique_stocks']}")
            log.info(f"  平均总涨幅: {stats['avg_total_return']:.2f}%")
            log.info(f"  平均最高涨幅: {stats['avg_max_return']:.2f}%")

            return True, stats

        except Exception as e:
            log.error(f"✗ 检查正样本时出错: {e}")
            return False, {"status": "error", "error": str(e)}

    def check_negative_samples(self):
        """
        检查负样本是否准备好

        Returns:
            tuple: (是否准备好, 详细信息)
        """
        log.info("\n" + "=" * 80)
        log.info("检查负样本准备状态")
        log.info("=" * 80)

        # 检查必需文件是否存在
        required_files = [self.negative_samples_file, self.negative_features_file]

        missing_files = []
        for file in required_files:
            if not file.exists():
                missing_files.append(str(file))

        if missing_files:
            log.warning("✗ 负样本文件缺失:")
            for f in missing_files:
                log.warning(f"  - {f}")
            return False, {"status": "missing_files", "files": missing_files}

        # 检查文件是否有效（非空）
        try:
            df_samples = pd.read_csv(self.negative_samples_file)
            df_features = pd.read_csv(self.negative_features_file)

            if df_samples.empty:
                log.warning("✗ 负样本列表为空")
                return False, {"status": "empty_samples"}

            if df_features.empty:
                log.warning("✗ 负样本特征数据为空")
                return False, {"status": "empty_features"}

            # 检查基本字段
            required_cols = ["ts_code", "t1_date"]
            missing_cols = [col for col in required_cols if col not in df_samples.columns]

            if missing_cols:
                log.warning(f"✗ 负样本缺少必需字段: {missing_cols}")
                return False, {"status": "missing_columns", "columns": missing_cols}

            # 检查label字段（应该都是0）
            if "label" in df_features.columns:
                label_counts = df_features["label"].value_counts()
                if 0 not in label_counts.index or label_counts[0] < len(df_features) * 0.9:
                    log.warning("⚠️  负样本特征数据中label=0的比例异常")

            # 统计信息
            stats = {
                "status": "ready",
                "sample_count": len(df_samples),
                "feature_count": len(df_features),
                "unique_stocks": df_samples["ts_code"].nunique(),
            }

            log.success("✓ 负样本已准备好")
            log.info(f"  样本数量: {stats['sample_count']}")
            log.info(f"  特征记录: {stats['feature_count']}")
            log.info(f"  股票数量: {stats['unique_stocks']}")

            return True, stats

        except Exception as e:
            log.error(f"✗ 检查负样本时出错: {e}")
            return False, {"status": "error", "error": str(e)}

    def check_all_samples(self):
        """
        检查所有样本是否准备好

        Returns:
            dict: 检查结果
        """
        log.info("\n" + "=" * 80)
        log.info("样本准备状态检查")
        log.info("=" * 80)
        log.info(f"检查时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log.info("")

        positive_ready, positive_info = self.check_positive_samples()
        negative_ready, negative_info = self.check_negative_samples()

        result = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "positive_samples": {"ready": positive_ready, "info": positive_info},
            "negative_samples": {"ready": negative_ready, "info": negative_info},
            "all_ready": positive_ready and negative_ready,
        }

        log.info("\n" + "=" * 80)
        log.info("检查结果总结")
        log.info("=" * 80)
        log.info(f"正样本: {'✓ 已准备好' if positive_ready else '✗ 未准备好'}")
        log.info(f"负样本: {'✓ 已准备好' if negative_ready else '✗ 未准备好'}")
        log.info(f"总体状态: {'✅ 所有样本已准备好，可以开始训练' if result['all_ready'] else '⏳ 等待样本准备完成'}")
        log.info("")

        return result

    def run_training_pipeline(self):
        """
        运行模型训练流程

        Returns:
            bool: 是否成功
        """
        log.info("=" * 80)
        log.info("🚀 开始自动执行模型训练流程")
        log.info("=" * 80)
        log.info("")

        steps = [
            ("数据质量检查", self.quality_check_script),
            ("模型训练", self.train_script),
            ("Walk-forward验证", self.walk_forward_script),
        ]

        for step_name, script in steps:
            if not script.exists():
                log.warning(f"⚠️  脚本不存在，跳过: {script}")
                continue

            log.info("=" * 80)
            log.info(f"执行步骤: {step_name}")
            log.info(f"脚本: {script}")
            log.info("")

            try:
                # 执行脚本
                result = subprocess.run(
                    [sys.executable, str(script)], cwd=str(self.project_root), capture_output=False, text=True
                )

                if result.returncode != 0:
                    log.error(f"❌ {step_name} 执行失败 (退出码: {result.returncode})")
                    return False
                else:
                    log.success(f"✅ {step_name} 执行成功")
                    log.info("")

            except Exception as e:
                log.error(f"❌ {step_name} 执行异常: {e}")
                import traceback

                traceback.print_exc()
                return False

        log.info("=" * 80)
        log.success("✅ 模型训练流程全部完成！")
        log.info("=" * 80)
        return True

    def monitor_once(self, auto_run=False):
        """
        执行一次检查（不循环）

        Args:
            auto_run: 如果样本都准备好，是否自动运行训练流程

        Returns:
            dict: 检查结果
        """
        result = self.check_all_samples()

        if result["all_ready"] and auto_run:
            log.info("\n" + "=" * 80)
            log.info("检测到所有样本已准备好，自动启动训练流程...")
            log.info("=" * 80)
            log.info("")

            success = self.run_training_pipeline()
            result["training_pipeline"] = {"executed": True, "success": success}
        else:
            result["training_pipeline"] = {
                "executed": False,
                "reason": "samples_not_ready" if not result["all_ready"] else "auto_run_disabled",
            }

        return result

    def monitor_loop(self, interval=300, auto_run=True):
        """
        循环监控（每interval秒检查一次）

        Args:
            interval: 检查间隔（秒），默认5分钟
            auto_run: 如果样本都准备好，是否自动运行训练流程
        """
        log.info("=" * 80)
        log.info("🔄 样本准备监控器已启动（循环模式）")
        log.info("=" * 80)
        log.info(f"检查间隔: {interval} 秒 ({interval/60:.1f} 分钟)")
        log.info(f"自动运行: {'是' if auto_run else '否'}")
        log.info("")
        log.info("💡 提示: 按 Ctrl+C 停止监控")
        log.info("")

        try:
            while True:
                result = self.monitor_once(auto_run=auto_run)

                # 如果已经执行了训练流程，停止监控
                if result.get("training_pipeline", {}).get("executed"):
                    if result["training_pipeline"].get("success"):
                        log.info("\n" + "=" * 80)
                        log.success("✅ 训练流程已完成，监控器将退出")
                        log.info("=" * 80)
                        break
                    else:
                        log.warning("\n⚠️  训练流程执行失败，继续监控...")

                # 等待下次检查
                log.info(f"\n⏳ 等待 {interval} 秒后进行下次检查...")
                log.info(f"下次检查时间: {(datetime.now().timestamp() + interval):.0f}")
                log.info("")
                time.sleep(interval)

        except KeyboardInterrupt:
            log.info("\n" + "=" * 80)
            log.info("⏹️  监控器已停止（用户中断）")
            log.info("=" * 80)
        except Exception as e:
            log.error(f"\n❌ 监控器异常: {e}")
            import traceback

            traceback.print_exc()


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="样本准备监控脚本")
    parser.add_argument(
        "--mode", choices=["once", "loop"], default="once", help="运行模式: once=检查一次, loop=循环监控"
    )
    parser.add_argument("--interval", type=int, default=300, help="循环模式下的检查间隔（秒），默认300秒（5分钟）")
    parser.add_argument("--auto-run", action="store_true", default=False, help="如果样本都准备好，自动运行训练流程")
    parser.add_argument("--no-auto-run", dest="auto_run", action="store_false", help="不自动运行训练流程（仅检查）")

    args = parser.parse_args()

    monitor = SamplePreparationMonitor()

    if args.mode == "once":
        # 单次检查模式
        result = monitor.monitor_once(auto_run=args.auto_run)

        # 保存检查结果
        result_file = PROJECT_ROOT / "data" / "processed" / "sample_preparation_status.json"
        result_file.parent.mkdir(parents=True, exist_ok=True)
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        log.info(f"\n检查结果已保存: {result_file}")

        # 返回适当的退出码
        if result["all_ready"]:
            sys.exit(0)
        else:
            sys.exit(1)
    else:
        # 循环监控模式
        monitor.monitor_loop(interval=args.interval, auto_run=args.auto_run)


if __name__ == "__main__":
    main()
