"""
预测准确率分析脚本

功能：
1. 读取历史预测结果
2. 获取实际股价表现
3. 计算准确率、收益率等指标
4. 生成分析报告

使用方法：
  python scripts/analyze_prediction_accuracy.py --date 20250919 --weeks 4
  python scripts/analyze_prediction_accuracy.py --all  # 分析所有历史预测
"""
import sys
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import argparse

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.data_manager import DataManager
from src.utils.logger import log


def load_prediction_metadata(prediction_date):
    """加载指定日期的预测元数据"""
    metadata_dir = Path('data/prediction/metadata')
    
    # 查找该日期的元数据文件
    pattern = f'prediction_metadata_{prediction_date}*.json'
    metadata_files = list(metadata_dir.glob(pattern))
    
    if not metadata_files:
        log.warning(f"未找到 {prediction_date} 的预测元数据")
        return None
    
    # 取最新的
    metadata_file = max(metadata_files, key=lambda x: x.stat().st_mtime)
    
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    log.info(f"✓ 加载预测元数据: {metadata_file}")
    return metadata


def get_actual_performance(dm, stock_code, prediction_date, weeks=4):
    """
    获取股票实际表现
    
    Args:
        dm: DataManager实例
        stock_code: 股票代码
        prediction_date: 预测日期（YYYYMMDD）
        weeks: 观察周期（周数）
    
    Returns:
        dict: 包含收益率、是否达到50%涨幅等信息
    """
    try:
        # 计算结束日期（预测日期 + weeks周）
        pred_date = datetime.strptime(prediction_date, '%Y%m%d')
        end_date = pred_date + timedelta(weeks=weeks)
        end_date_str = end_date.strftime('%Y%m%d')
        
        # 获取预测日期当天的价格
        df_pred = dm.get_daily_data(stock_code, prediction_date, prediction_date)
        if df_pred.empty:
            return None
        
        pred_price = df_pred['close'].iloc[0]
        
        # 获取观察期内的数据
        df_period = dm.get_daily_data(stock_code, prediction_date, end_date_str)
        if df_period.empty:
            return None
        
        # 计算最大涨幅、最大跌幅、最终涨幅
        max_price = df_period['close'].max()
        min_price = df_period['close'].min()
        final_price = df_period['close'].iloc[-1]
        
        max_return = (max_price / pred_price - 1) * 100
        min_return = (min_price / pred_price - 1) * 100
        final_return = (final_price / pred_price - 1) * 100
        
        # 判断是否达到50%涨幅（牛股标准）
        is_bull_stock = max_return >= 50
        
        return {
            'pred_price': float(pred_price),
            'max_price': float(max_price),
            'min_price': float(min_price),
            'final_price': float(final_price),
            'max_return': float(max_return),
            'min_return': float(min_return),
            'final_return': float(final_return),
            'is_bull_stock': is_bull_stock,
            'observation_days': len(df_period)
        }
    except Exception as e:
        log.warning(f"获取 {stock_code} 实际表现失败: {e}")
        return None


def analyze_prediction(prediction_date, weeks=4):
    """分析单次预测的准确率"""
    log.info("="*80)
    log.info(f"分析预测准确率: {prediction_date} ({weeks}周)")
    log.info("="*80)
    
    # 加载预测元数据
    metadata = load_prediction_metadata(prediction_date)
    if not metadata:
        return None
    
    # 初始化数据管理器
    dm = DataManager()
    
    # 分析每只推荐股票的实际表现
    results = []
    top_stocks = metadata.get('top_stocks', [])
    
    log.info(f"\n分析 {len(top_stocks)} 只推荐股票的实际表现...")
    
    for i, stock in enumerate(top_stocks):
        if (i + 1) % 10 == 0:
            log.info(f"进度: {i+1}/{len(top_stocks)}")
        
        stock_code = stock['code']
        performance = get_actual_performance(dm, stock_code, prediction_date, weeks)
        
        if performance:
            result = {
                'rank': stock['rank'],
                'code': stock_code,
                'name': stock['name'],
                'predicted_prob': stock['probability'],
                'predicted_price': stock['price'],
                'actual_max_return': performance['max_return'],
                'actual_final_return': performance['final_return'],
                'actual_min_return': performance['min_return'],
                'is_bull_stock': performance['is_bull_stock'],
                'max_price': performance['max_price'],
                'final_price': performance['final_price']
            }
            results.append(result)
    
    if not results:
        log.warning("没有获取到任何股票的实际表现数据")
        return None
    
    df_results = pd.DataFrame(results)
    
    # 计算统计指标
    total = len(df_results)
    bull_stocks = df_results['is_bull_stock'].sum()
    accuracy = bull_stocks / total * 100 if total > 0 else 0
    
    avg_max_return = df_results['actual_max_return'].mean()
    avg_final_return = df_results['actual_final_return'].mean()
    
    positive_count = (df_results['actual_final_return'] > 0).sum()
    positive_rate = positive_count / total * 100 if total > 0 else 0
    
    analysis = {
        'prediction_date': prediction_date,
        'weeks': weeks,
        'total_stocks': total,
        'bull_stocks': int(bull_stocks),
        'accuracy': float(accuracy),
        'avg_max_return': float(avg_max_return),
        'avg_final_return': float(avg_final_return),
        'positive_count': int(positive_count),
        'positive_rate': float(positive_rate),
        'results': df_results.to_dict('records')
    }
    
    return analysis


def generate_accuracy_report(analysis):
    """生成准确率分析报告"""
    report = []
    report.append("=" * 80)
    report.append("📊 预测准确率分析报告")
    report.append("=" * 80)
    
    report.append(f"\n📅 预测日期: {analysis['prediction_date']}")
    report.append(f"⏱️  观察周期: {analysis['weeks']} 周")
    report.append(f"📈 分析股票数: {analysis['total_stocks']} 只")
    
    report.append("\n" + "=" * 80)
    report.append("一、整体表现")
    report.append("=" * 80)
    
    report.append(f"\n1. 准确率（达到50%涨幅）")
    report.append(f"   - 牛股数量: {analysis['bull_stocks']} 只")
    report.append(f"   - 准确率: {analysis['accuracy']:.2f}%")
    
    report.append(f"\n2. 收益率统计")
    report.append(f"   - 平均最大涨幅: {analysis['avg_max_return']:.2f}%")
    report.append(f"   - 平均最终涨幅: {analysis['avg_final_return']:.2f}%")
    
    report.append(f"\n3. 盈利情况")
    report.append(f"   - 盈利股票数: {analysis['positive_count']} 只")
    report.append(f"   - 盈利比例: {analysis['positive_rate']:.2f}%")
    
    # Top 10 表现
    report.append("\n" + "=" * 80)
    report.append("二、Top 10 表现详情")
    report.append("=" * 80)
    
    df_results = pd.DataFrame(analysis['results'])
    df_sorted = df_results.sort_values('actual_max_return', ascending=False).head(10)
    
    for i, row in df_sorted.iterrows():
        report.append(f"\n【第 {row['rank']} 名】{row['name']}（{row['code']}）")
        report.append(f"  预测概率: {row['predicted_prob']*100:.2f}%")
        report.append(f"  最大涨幅: {row['actual_max_return']:.2f}%")
        report.append(f"  最终涨幅: {row['actual_final_return']:.2f}%")
        report.append(f"  是否牛股: {'✅ 是' if row['is_bull_stock'] else '❌ 否'}")
    
    report.append("\n" + "=" * 80)
    report.append("报告结束")
    report.append("=" * 80)
    
    return "\n".join(report)


def save_analysis_results(analysis, report_content):
    """保存分析结果"""
    output_dir = Path('data/prediction/analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    prediction_date = analysis['prediction_date']
    weeks = analysis['weeks']
    
    # 保存详细结果CSV
    df_results = pd.DataFrame(analysis['results'])
    csv_file = output_dir / f'accuracy_{prediction_date}_{weeks}w.csv'
    df_results.to_csv(csv_file, index=False, encoding='utf-8-sig')
    log.success(f"✓ 详细结果已保存: {csv_file}")
    
    # 保存分析报告
    report_file = output_dir / f'accuracy_report_{prediction_date}_{weeks}w.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_content)
    log.success(f"✓ 分析报告已保存: {report_file}")
    
    # 保存JSON元数据
    json_file = output_dir / f'accuracy_{prediction_date}_{weeks}w.json'
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    log.success(f"✓ 分析元数据已保存: {json_file}")
    
    return csv_file, report_file, json_file


def analyze_all_predictions(weeks=4):
    """分析所有历史预测"""
    metadata_dir = Path('data/prediction/metadata')
    
    # 查找所有元数据文件
    metadata_files = list(metadata_dir.glob('prediction_metadata_*.json'))
    
    if not metadata_files:
        log.warning("未找到任何预测元数据")
        return
    
    log.info(f"找到 {len(metadata_files)} 个历史预测")
    
    all_analyses = []
    
    for metadata_file in metadata_files:
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        prediction_date = metadata['prediction_date']
        
        # 只分析非回测的预测（实际预测）
        if metadata.get('is_backtest', False):
            continue
        
        log.info(f"\n分析预测: {prediction_date}")
        analysis = analyze_prediction(prediction_date, weeks)
        
        if analysis:
            all_analyses.append(analysis)
    
    if not all_analyses:
        log.warning("没有可分析的预测结果")
        return
    
    # 生成汇总报告
    log.info("\n" + "="*80)
    log.info("汇总分析结果")
    log.info("="*80)
    
    total_predictions = len(all_analyses)
    avg_accuracy = np.mean([a['accuracy'] for a in all_analyses])
    avg_max_return = np.mean([a['avg_max_return'] for a in all_analyses])
    avg_final_return = np.mean([a['avg_final_return'] for a in all_analyses])
    
    log.info(f"\n总预测次数: {total_predictions}")
    log.info(f"平均准确率: {avg_accuracy:.2f}%")
    log.info(f"平均最大涨幅: {avg_max_return:.2f}%")
    log.info(f"平均最终涨幅: {avg_final_return:.2f}%")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='预测准确率分析')
    parser.add_argument('--date', type=str, default=None,
                        help='预测日期（格式：YYYYMMDD），例如：--date 20250919')
    parser.add_argument('--weeks', type=int, default=4,
                        help='观察周期（周数），默认4周')
    parser.add_argument('--all', action='store_true',
                        help='分析所有历史预测')
    
    args = parser.parse_args()
    
    if args.all:
        analyze_all_predictions(weeks=args.weeks)
    elif args.date:
        analysis = analyze_prediction(args.date, weeks=args.weeks)
        
        if analysis:
            report_content = generate_accuracy_report(analysis)
            save_analysis_results(analysis, report_content)
            log.info("\n" + report_content)
        else:
            log.error("分析失败")
    else:
        parser.print_help()
        log.error("请指定 --date 或 --all 参数")


if __name__ == '__main__':
    main()

