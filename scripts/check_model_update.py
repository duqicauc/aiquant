#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型更新检查脚本
检查模型是否需要更新，并给出建议
"""

import os
import sys
import json
from datetime import datetime, timedelta
from pathlib import Path
import joblib

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import log


def check_model_age():
    """检查模型年龄"""
    model_dir = project_root / 'data' / 'models' / 'stock_selection'
    model_file = model_dir / 'xgboost_timeseries_v2.joblib'
    
    if not model_file.exists():
        log.warning("⚠️  当前模型文件不存在")
        return None, True
    
    # 获取模型文件的修改时间
    model_time = datetime.fromtimestamp(model_file.stat().st_mtime)
    days_old = (datetime.now() - model_time).days
    
    log.info(f"当前模型训练日期: {model_time.strftime('%Y-%m-%d')}")
    log.info(f"模型已使用: {days_old} 天")
    
    # 建议更新周期：30-90天
    needs_update = days_old > 90
    
    if days_old > 90:
        log.warning(f"⚠️  模型已使用超过90天，强烈建议更新")
        return days_old, True
    elif days_old > 60:
        log.warning(f"⚠️  模型已使用超过60天，建议考虑更新")
        return days_old, False
    elif days_old > 30:
        log.info(f"✓ 模型使用时间适中")
        return days_old, False
    else:
        log.info(f"✓ 模型较新，暂不需要更新")
        return days_old, False


def check_prediction_performance():
    """检查近期预测表现"""
    review_dir = project_root / 'data' / 'reviews'
    
    if not review_dir.exists():
        log.warning("⚠️  没有找到历史回顾数据")
        return None, False
    
    # 查找最近的回顾文件
    import glob
    review_files = glob.glob(str(review_dir / 'review_*_4w_detail.csv'))
    
    if not review_files:
        log.warning("⚠️  没有找到4周回顾数据")
        return None, False
    
    # 读取最近的回顾
    latest_review = max(review_files, key=os.path.getctime)
    
    import pandas as pd
    df = pd.read_csv(latest_review)
    
    # 计算表现指标
    win_rate = (df['实际收益%'] > 0).sum() / len(df) * 100
    avg_return = df['实际收益%'].mean()
    
    log.info(f"\n近期预测表现:")
    log.info(f"  - 胜率: {win_rate:.1f}%")
    log.info(f"  - 平均收益: {avg_return:+.2f}%")
    
    # 判断是否需要更新
    needs_update = win_rate < 55 or avg_return < 2
    
    if needs_update:
        log.warning(f"⚠️  预测表现低于预期，建议更新模型")
    else:
        log.info(f"✓ 预测表现良好")
    
    return {'win_rate': win_rate, 'avg_return': avg_return}, needs_update


def generate_update_recommendation():
    """生成更新建议"""
    log.info("="*80)
    log.info("🔍 模型更新检查")
    log.info("="*80)
    
    # 检查1: 模型年龄
    log.info("\n1. 检查模型年龄")
    model_age, age_needs_update = check_model_age()
    
    # 检查2: 预测表现
    log.info("\n2. 检查预测表现")
    performance, perf_needs_update = check_prediction_performance()
    
    # 综合判断
    log.info("\n" + "="*80)
    log.info("📋 更新建议")
    log.info("="*80)
    
    needs_update = age_needs_update or perf_needs_update
    
    if needs_update:
        log.warning("\n⚠️  建议更新模型")
        log.info("\n原因:")
        if age_needs_update:
            log.info("  - 模型使用时间过长")
        if perf_needs_update:
            log.info("  - 预测表现不理想")
        
        log.info("\n更新步骤:")
        log.info("  1. 运行数据更新: python scripts/update_data.py")
        log.info("  2. 准备正样本: python scripts/prepare_positive_samples.py")
        log.info("  3. 准备负样本: python scripts/prepare_negative_samples_v2.py")
        log.info("  4. 质量检查: python scripts/check_sample_quality.py")
        log.info("  5. 训练模型: python scripts/train_xgboost_timeseries.py")
        log.info("  6. 验证模型: python scripts/walk_forward_validation.py")
        
        log.info("\n或使用一键更新脚本:")
        log.info("  bash scripts/update_model_pipeline.sh")
    else:
        log.success("\n✅ 当前模型状态良好，暂不需要更新")
    
    # 保存检查结果
    save_check_result(model_age, performance, needs_update)
    
    return needs_update


def save_check_result(model_age, performance, needs_update):
    """保存检查结果"""
    check_dir = project_root / 'data' / 'models' / 'stock_selection'
    check_dir.mkdir(parents=True, exist_ok=True)
    
    result = {
        'check_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_age_days': model_age,
        'performance': performance,
        'needs_update': needs_update
    }
    
    # 读取历史检查记录
    history_file = check_dir / 'update_check_history.json'
    
    if history_file.exists():
        with open(history_file, 'r', encoding='utf-8') as f:
            history = json.load(f)
    else:
        history = {'checks': []}
    
    # 添加本次检查
    history['checks'].append(result)
    
    # 保留最近20次记录
    history['checks'] = history['checks'][-20:]
    
    # 保存
    with open(history_file, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    
    log.info(f"\n✓ 检查结果已保存: {history_file}")


def main():
    """主函数"""
    try:
        needs_update = generate_update_recommendation()
        sys.exit(1 if needs_update else 0)  # 需要更新时返回1
    except Exception as e:
        log.error(f"❌ 检查失败: {e}", exc_info=True)
        sys.exit(2)


if __name__ == '__main__':
    main()

