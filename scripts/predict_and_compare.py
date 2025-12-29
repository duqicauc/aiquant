"""
使用最新模型预测指定日期，并与之前的预测结果比对

用法:
    python scripts/predict_and_compare.py --dates 20251225 20250919
"""
import sys
import os
import subprocess
import json
from pathlib import Path
from datetime import datetime
import argparse

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.logger import log


def run_prediction(date_str):
    """运行预测脚本"""
    log.info("="*80)
    log.info(f"开始预测日期: {date_str}")
    log.info("="*80)
    
    cmd = ['python', 'scripts/score_current_stocks.py', '--date', date_str]
    
    try:
        result = subprocess.run(
            cmd,
            cwd=os.path.dirname(os.path.dirname(__file__)),
            capture_output=True,
            text=True,
            timeout=3600  # 1小时超时
        )
        
        if result.returncode == 0:
            log.success(f"✓ 预测完成: {date_str}")
            return True
        else:
            log.error(f"✗ 预测失败: {date_str}")
            log.error(f"错误输出: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        log.error(f"✗ 预测超时: {date_str}")
        return False
    except Exception as e:
        log.error(f"✗ 预测异常: {date_str}, {e}")
        return False


def find_prediction_metadata(date_str):
    """查找指定日期的预测元数据"""
    metadata_dir = Path('data/prediction/metadata')
    
    # 查找该日期的元数据文件
    pattern = f'prediction_metadata_{date_str}*.json'
    metadata_files = list(metadata_dir.glob(pattern))
    
    if not metadata_files:
        return None
    
    # 如果有多个，选择最新的
    if len(metadata_files) > 1:
        metadata_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        log.info(f"找到多个元数据文件，使用最新的: {metadata_files[0].name}")
    
    with open(metadata_files[0], 'r', encoding='utf-8') as f:
        return json.load(f)


def compare_predictions_for_date(date_str):
    """比对指定日期的新旧预测结果"""
    log.info("="*80)
    log.info(f"比对日期: {date_str} 的新旧预测结果")
    log.info("="*80)
    
    # 查找所有该日期的元数据文件
    metadata_dir = Path('data/prediction/metadata')
    pattern = f'prediction_metadata_{date_str}*.json'
    metadata_files = list(metadata_dir.glob(pattern))
    
    # 如果元数据文件少于2个，尝试从结果文件创建元数据
    if len(metadata_files) < 2:
        log.info(f"日期 {date_str} 只有 {len(metadata_files)} 个元数据文件，尝试从结果文件查找...")
        
        # 查找所有该日期的结果文件
        results_dir = Path('data/prediction/results')
        top_pattern = f'top_50_stocks_{date_str}*.csv'
        top_files = list(results_dir.glob(top_pattern))
        
        if len(top_files) < 2:
            log.warning(f"日期 {date_str} 只有 {len(top_files)} 个预测结果文件，无法比对")
            return None
        
        # 按修改时间排序，最新的在前
        top_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # 创建临时元数据用于比对
        import pandas as pd
        
        # 对于20251225，232545.csv是旧模型，需要找到最新模型的结果
        # 如果最新文件是232545，说明还没有用新模型预测，需要先预测
        old_file = None
        new_file = None
        
        for f in top_files:
            if '232545' in f.name:
                old_file = f
                break
        
        # 如果找到了232545（旧模型），其他文件可能是新模型
        if old_file:
            # 找不是232545的文件作为新模型
            for f in top_files:
                if '232545' not in f.name:
                    new_file = f
                    break
            
            # 如果没找到新模型文件，说明需要先预测
            if not new_file:
                log.warning(f"未找到新模型预测结果，请先使用最新模型预测 {date_str}")
                return None
        else:
            # 如果没有232545，按时间排序，最新的作为新模型
            new_file = top_files[0]
            old_file = top_files[1] if len(top_files) > 1 else None
        
        if not old_file or not new_file:
            log.warning(f"无法确定新旧模型文件")
            return None
        
        # 读取旧的结果（旧模型）
        df_old = pd.read_csv(old_file, encoding='utf-8-sig')
        old_metadata = {
            'prediction_date': date_str,
            'model_path': '旧模型(232545)',
            'top_file': str(old_file),
            'scores_file': str(old_file).replace('top_50_stocks', 'stock_scores')
        }
        
        # 读取最新的结果（新模型）
        df_new = pd.read_csv(new_file, encoding='utf-8-sig')
        new_metadata = {
            'prediction_date': date_str,
            'model_path': '最新模型(v1.0.0)',
            'top_file': str(new_file),
            'scores_file': str(new_file).replace('top_50_stocks', 'stock_scores')
        }
        
        log.info(f"旧模型文件: {old_file.name}")
        log.info(f"新模型文件: {new_file.name}")
        log.info("")
    else:
        # 按修改时间排序，最新的在前
        metadata_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # 最新的（新模型）
        with open(metadata_files[0], 'r', encoding='utf-8') as f:
            new_metadata = json.load(f)
        
        # 之前的（旧模型）
        with open(metadata_files[1], 'r', encoding='utf-8') as f:
            old_metadata = json.load(f)
        
        log.info(f"新预测模型: {new_metadata.get('model_path', 'N/A')}")
        log.info(f"旧预测模型: {old_metadata.get('model_path', 'N/A')}")
        log.info("")
    
    # 运行比对脚本
    from scripts.compare_predictions import compare_predictions
    
    result = compare_predictions(
        old_metadata,
        new_metadata,
        date1_label=f'旧模型({old_metadata.get("model_path", "N/A")})',
        date2_label=f'新模型({new_metadata.get("model_path", "N/A")})'
    )
    
    return result


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='使用最新模型预测并比对结果')
    parser.add_argument('--dates', type=str, nargs='+', required=True,
                       help='要预测的日期列表（格式：YYYYMMDD），例如：--dates 20251225 20250919')
    parser.add_argument('--skip-prediction', action='store_true',
                       help='跳过预测步骤，直接比对已有结果')
    parser.add_argument('--skip-comparison', action='store_true',
                       help='跳过比对步骤，只进行预测')
    
    args = parser.parse_args()
    
    log.info("="*80)
    log.info("使用最新模型预测并比对结果")
    log.info("="*80)
    log.info("")
    
    # 步骤1: 预测
    if not args.skip_prediction:
        log.info("步骤1: 使用最新模型进行预测")
        log.info("")
        
        for date_str in args.dates:
            # 检查是否已有最新模型的预测结果
            metadata = find_prediction_metadata(date_str)
            if metadata:
                model_path = metadata.get('model_path', '')
                # 检查是否是v1.0.0模型
                if 'v1.0.0' in model_path or 'breakout_launch_scorer' in str(metadata.get('model_path', '')):
                    log.info(f"日期 {date_str} 已有最新模型预测结果，跳过")
                    continue
            
            success = run_prediction(date_str)
            if not success:
                log.error(f"预测 {date_str} 失败，跳过比对")
                continue
            
            log.info("")
    else:
        log.info("跳过预测步骤")
        log.info("")
    
    # 步骤2: 比对
    if not args.skip_comparison:
        log.info("步骤2: 比对新旧预测结果")
        log.info("")
        
        for date_str in args.dates:
            result = compare_predictions_for_date(date_str)
            if result:
                log.success(f"✓ 比对完成: {date_str}")
            else:
                log.warning(f"⚠ 比对失败或跳过: {date_str}")
            log.info("")
    else:
        log.info("跳过比对步骤")
    
    log.info("="*80)
    log.success("✅ 所有任务完成！")
    log.info("="*80)


if __name__ == '__main__':
    main()

