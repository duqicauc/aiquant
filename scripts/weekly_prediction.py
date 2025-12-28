#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
每周股票预测主脚本
自动执行股票评分和推荐报告生成
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import log


def run_weekly_prediction():
    """执行每周预测流程"""
    log.info("="*80)
    log.info("🚀 开始每周股票预测")
    log.info("="*80)
    
    prediction_date = datetime.now().strftime('%Y%m%d')
    log.info(f"预测日期: {prediction_date}")
    
    # Step 1: 运行评分脚本
    log.info("\n" + "="*80)
    log.info("Step 1: 执行股票评分")
    log.info("="*80)
    
    score_script = project_root / 'scripts' / 'score_current_stocks.py'
    ret = os.system(f"python {score_script}")
    
    if ret != 0:
        log.error("❌ 股票评分失败！")
        send_alert("每周预测失败", "股票评分脚本执行失败", level="ERROR")
        return False
    
    log.success("✅ 股票评分完成")
    
    # Step 2: 整理预测结果到专门目录
    log.info("\n" + "="*80)
    log.info("Step 2: 整理预测结果")
    log.info("="*80)
    
    organize_prediction_results(prediction_date)
    
    # Step 3: 更新预测索引
    log.info("\n" + "="*80)
    log.info("Step 3: 更新预测索引")
    log.info("="*80)
    
    update_prediction_index(prediction_date)
    
    # Step 4: 发送完成通知
    log.info("\n" + "="*80)
    log.success("✅ 每周预测完成！")
    log.info("="*80)
    
    send_notification(prediction_date)
    
    return True


def organize_prediction_results(prediction_date):
    """整理预测结果到专门目录"""
    # 创建预测日期目录
    pred_dir = project_root / 'data' / 'predictions' / prediction_date
    pred_dir.mkdir(parents=True, exist_ok=True)
    
    # 查找最新的结果文件
    results_dir = project_root / 'data' / 'results'
    
    # 移动或复制文件
    import shutil
    import glob
    
    # 查找今天的文件
    today = datetime.now().strftime('%Y%m%d')
    
    for pattern in [f'stock_scores_{today}*.csv', 
                    f'top_*_stocks_{today}*.csv',
                    f'prediction_report_{today}*.txt']:
        files = glob.glob(str(results_dir / pattern))
        if files:
            latest_file = max(files, key=os.path.getctime)
            filename = os.path.basename(latest_file)
            
            # 复制到预测目录
            dest = pred_dir / filename
            shutil.copy2(latest_file, dest)
            log.info(f"  ✓ 已复制: {filename}")
    
    log.success(f"✓ 结果已整理到: {pred_dir}")


def update_prediction_index(prediction_date):
    """更新预测索引"""
    index_file = project_root / 'data' / 'predictions' / 'index.json'
    
    # 读取现有索引
    if index_file.exists():
        with open(index_file, 'r', encoding='utf-8') as f:
            index = json.load(f)
    else:
        index = {'predictions': []}
    
    # 读取本次预测的摘要信息
    pred_dir = project_root / 'data' / 'predictions' / prediction_date
    
    # 查找 top stocks 文件
    import glob
    top_files = glob.glob(str(pred_dir / 'top_*.csv'))
    
    if top_files:
        import pandas as pd
        top_file = top_files[0]
        df_top = pd.read_csv(top_file)
        
        # 提取关键信息
        prediction_record = {
            'date': prediction_date,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_recommended': len(df_top),
            'top_3': [
                {
                    'rank': i+1,
                    'code': row['股票代码'],
                    'name': row['股票名称'],
                    'probability': float(row['牛股概率']),
                    'price': float(row['最新价格'])
                }
                for i, row in df_top.head(3).iterrows()
            ],
            'directory': str(pred_dir.relative_to(project_root))
        }
        
        # 添加到索引（避免重复）
        index['predictions'] = [p for p in index['predictions'] if p['date'] != prediction_date]
        index['predictions'].append(prediction_record)
        
        # 按日期倒序排序
        index['predictions'].sort(key=lambda x: x['date'], reverse=True)
        
        # 保存索引
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(index, f, indent=2, ensure_ascii=False)
        
        log.success(f"✓ 预测索引已更新: {index_file}")


def send_notification(prediction_date):
    """发送完成通知"""
    pred_dir = project_root / 'data' / 'predictions' / prediction_date
    
    message = f"""
📊 每周股票预测已完成

📅 预测日期: {prediction_date}
📁 结果目录: {pred_dir}

请查看预测报告了解详细推荐。
    """
    
    log.info(message)
    
    # 这里可以添加邮件、微信等通知方式
    # send_alert("每周预测完成", message, level="INFO")


def send_alert(title, message, level="INFO"):
    """发送告警通知（占位函数，后续可扩展）"""
    log.info(f"[{level}] {title}: {message}")
    
    # TODO: 可以在这里添加邮件、微信、钉钉等通知方式
    # 例如：
    # - 邮件: smtplib
    # - 微信: Server酱
    # - 钉钉: webhook
    pass


def main():
    """主函数"""
    try:
        success = run_weekly_prediction()
        sys.exit(0 if success else 1)
    except Exception as e:
        log.error(f"❌ 执行失败: {e}", exc_info=True)
        send_alert("每周预测异常", str(e), level="ERROR")
        sys.exit(1)


if __name__ == '__main__':
    main()

