#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
重新准备数据（从2000-01-01开始），然后训练v1.4.0模型

v1.4.0 更新：
- 完全重新构建正样本数据（从2000年1月1日开始）
- 完全重新构建负样本数据
- 使用新的去重逻辑：重叠时间段合并，不重叠时间段分别保留
"""

import sys
import os
from pathlib import Path
import subprocess
import time
from datetime import datetime

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import log

VERSION = "v1.4.0"


def main():
    """主函数"""
    start_time = time.time()

    log.info("=" * 80)
    log.info(f"重新准备数据并训练{VERSION}模型")
    log.info("=" * 80)
    log.info("")
    log.info("步骤1: 重新准备正样本数据（从2000-01-01开始）")
    log.info("步骤2: 重新准备负样本数据")
    log.info(f"步骤3: 训练{VERSION}模型")
    log.info("")
    log.info(f"📌 {VERSION} 更新内容:")
    log.info("  - 使用新的去重逻辑：重叠时间段合并，不重叠时间段分别保留")
    log.info("  - 同一股票可能有多个不重叠的正样本")
    log.info("")

    # 1. 检查配置
    from config.settings import settings

    start_date = settings.get("data.sample_preparation.start_date", "20000101")
    log.info(f"配置的起始日期: {start_date}")

    if start_date != "20000101":
        log.warning(f"⚠️  配置的起始日期不是20000101，而是{start_date}")
        log.warning("请确认是否要修改配置文件 config/settings.yaml")

    log.info("")

    # 2. 备份旧的样本数据文件（按版本保存）
    log.info("=" * 80)
    log.info("第〇步：备份旧的样本数据（强制重新筛选）")
    log.info("=" * 80)

    # 创建备份目录（按时间戳）
    backup_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = f"data/backup/training_data_{backup_timestamp}"
    os.makedirs(backup_dir, exist_ok=True)
    log.info(f"备份目录: {backup_dir}")

    old_files = [
        "data/training/samples/positive_samples.csv",
        "data/training/samples/negative_samples_v2.csv",
        "data/training/features/feature_data_34d.csv",
        "data/training/features/negative_feature_data_v2_34d.csv",
    ]

    import shutil

    for old_file in old_files:
        if os.path.exists(old_file):
            # 备份文件
            backup_path = os.path.join(backup_dir, os.path.basename(old_file))
            shutil.copy2(old_file, backup_path)
            log.info(f"备份文件: {old_file} -> {backup_path}")
            # 删除原文件
            os.remove(old_file)
            log.success(f"✓ 已备份并删除: {old_file}")
        else:
            log.info(f"文件不存在，跳过: {old_file}")

    log.info("")

    # 3. 重新准备正样本数据
    log.info("=" * 80)
    log.info("第一步：重新准备正样本数据")
    log.info("=" * 80)
    log.info("注意：这可能需要较长时间（数小时）")
    log.info("")

    # 设置环境变量以跳过人工确认
    env = os.environ.copy()
    env["AUTO_CONFIRM"] = "1"  # 自动确认（使用默认值True）

    cmd_pos = ["python", "scripts/prepare_positive_samples.py"]
    log.info(f"执行命令: {' '.join(cmd_pos)}")
    log.info("(已设置自动确认模式: AUTO_CONFIRM=1)")
    log.info("")

    try:
        result = subprocess.run(cmd_pos, check=True, env=env)
        log.success("✓ 正样本数据准备完成")
    except subprocess.CalledProcessError as e:
        log.error(f"✗ 正样本数据准备失败: {e}")
        return
    except KeyboardInterrupt:
        log.warning("正样本数据准备被用户中断")
        return

    log.info("")

    # 3. 重新准备负样本数据
    log.info("=" * 80)
    log.info("第二步：重新准备负样本数据")
    log.info("=" * 80)
    log.info("注意：这可能需要较长时间")
    log.info("")

    cmd_neg = ["python", "scripts/prepare_negative_samples_v2.py"]
    log.info(f"执行命令: {' '.join(cmd_neg)}")
    log.info("")

    try:
        result = subprocess.run(cmd_neg, check=True, capture_output=False)
        log.success("✓ 负样本数据准备完成")
    except subprocess.CalledProcessError as e:
        log.error(f"✗ 负样本数据准备失败: {e}")
        return
    except KeyboardInterrupt:
        log.warning("负样本数据准备被用户中断")
        return

    log.info("")

    # 4. 检查数据文件
    pos_file = "data/training/features/feature_data_34d.csv"
    neg_file = "data/training/features/negative_feature_data_v2_34d.csv"

    if not os.path.exists(pos_file):
        log.error(f"正样本数据文件不存在: {pos_file}")
        return

    if not os.path.exists(neg_file):
        log.error(f"负样本数据文件不存在: {neg_file}")
        return

    # 统计样本数量
    import pandas as pd

    df_pos = pd.read_csv(pos_file)
    df_neg = pd.read_csv(neg_file)

    n_pos_samples = df_pos["sample_id"].nunique()
    n_neg_samples = df_neg["sample_id"].nunique()

    log.success("✓ 数据文件已准备完成")
    log.info(f"  正样本数量: {n_pos_samples}")
    log.info(f"  负样本数量: {n_neg_samples}")
    log.info("")

    # 5. 删除旧的模型（如果存在）
    old_model_dir = f"data/models/breakout_launch_scorer/versions/{VERSION}"
    if os.path.exists(old_model_dir):
        log.info(f"删除旧的{VERSION}模型: {old_model_dir}")
        import shutil

        shutil.rmtree(old_model_dir)
        log.success("✓ 旧模型已删除")
        log.info("")

    # 6. 训练新模型
    log.info("=" * 80)
    log.info(f"第三步：训练{VERSION}模型")
    log.info("=" * 80)
    log.info("")

    cmd_train = ["python", "scripts/train_breakout_launch_scorer.py", "--version", VERSION, "--neg-version", "v2"]

    log.info(f"执行命令: {' '.join(cmd_train)}")
    log.info("")

    try:
        result = subprocess.run(cmd_train, check=True, capture_output=False)
        log.success("✓ 模型训练完成")
    except subprocess.CalledProcessError as e:
        log.error(f"✗ 模型训练失败: {e}")
        return
    except KeyboardInterrupt:
        log.warning("模型训练被用户中断")
        return

    # 计算总耗时
    elapsed_time = time.time() - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)

    log.info("")
    log.info("=" * 80)
    log.success(f"✅ {VERSION} 模型训练全部完成！")
    log.info("=" * 80)
    log.info("")
    log.info(f"⏱️  总耗时: {hours}小时{minutes}分钟{seconds}秒")
    log.info("")
    log.info("数据准备和模型训练已完成")
    log.info("可以使用以下命令进行预测:")
    log.info(f"  python scripts/score_current_stocks.py --date {datetime.now().strftime('%Y%m%d')} --version {VERSION}")
    log.info("")


if __name__ == "__main__":
    main()
