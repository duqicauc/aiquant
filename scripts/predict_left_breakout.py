#!/usr/bin/env python3
"""
左侧潜力牛股模型 - 股票预测脚本

使用训练好的左侧模型对当前市场进行预测

使用方法:
python scripts/predict_left_breakout.py

可选参数:
--date           指定预测日期（默认今天）
--top-n          返回前N个结果（默认50）
--min-prob       最小概率阈值（默认0.1）
--max-stocks     最大处理股票数（默认全部）
--config-file    指定配置文件路径
--no-report      不生成报告
"""

import sys
import os
import argparse
import warnings
from datetime import datetime

# 忽略 FutureWarning（fillna method 已废弃的警告）
warnings.filterwarnings('ignore', category=FutureWarning)

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_manager import DataManager
from src.models.stock_selection.left_breakout import LeftBreakoutModel
from src.models.stock_selection.left_breakout.left_predictor import LeftBreakoutPredictor
from config.settings import settings
from src.utils.logger import log


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='左侧潜力牛股模型预测')
    parser.add_argument('--date', type=str, default=None,
                       help='预测日期（YYYYMMDD格式，默认今天）')
    parser.add_argument('--top-n', type=int, default=50,
                       help='返回前N个结果')
    parser.add_argument('--min-prob', type=float, default=0.1,
                       help='最小概率阈值')
    parser.add_argument('--max-stocks', type=int, default=None,
                       help='最大处理股票数（默认全部，用于快速测试）')
    parser.add_argument('--config-file', type=str, default='config/settings.yaml',
                       help='配置文件路径')
    parser.add_argument('--no-report', action='store_true',
                       help='不生成报告')

    args = parser.parse_args()

    try:
        log.info("="*60)
        log.info("🚀 左侧潜力牛股模型 - 股票预测")
        log.info("="*60)

        # 1. 加载配置和初始化
        config = settings._config
        if not config.get('left_breakout', {}).get('model', {}).get('enabled', True):
            log.warning("⚠️  左侧模型未启用")
            return

        dm = DataManager(config.get('data', {}).get('source', 'tushare'))
        left_model = LeftBreakoutModel(dm, config.get('left_breakout', {}))

        if not left_model.load_model():
            log.error("❌ 无法加载模型，请先运行训练脚本")
            return

        predictor = LeftBreakoutPredictor(left_model)

        # 2. 执行预测
        start_time = datetime.now()
        prediction_date_str = args.date or datetime.now().strftime('%Y%m%d')
        log.info(f"📅 预测日期: {prediction_date_str}")
        if args.max_stocks:
            log.info(f"📊 处理范围: 前 {args.max_stocks} 只股票")
        log.info("⏳ 开始特征提取和预测...")

        predictions = predictor.predict_current_market(
            prediction_date=args.date,
            top_n=args.top_n,
            min_probability=args.min_prob,
            max_stocks=args.max_stocks
        )

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # 3. 输出预测结果
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        log.info("\n" + "="*60)
        log.info("📈 预测结果")
        log.info("="*60)

        if predictions.empty:
            log.warning("⚠️  没有找到符合条件的股票")
            return

        prediction_date = predictions['prediction_date'].iloc[0]
        log.info(f"📊 推荐股票: {len(predictions)} 只 | ⏱️  耗时: {duration:.1f} 秒")
        
        # 显示Top 10推荐（简化格式）
        log.info("\n🏆 Top 10 推荐:")
        for i, (_, stock) in enumerate(predictions.head(10).iterrows(), 1):
            prob_pct = stock['probability'] * 100
            rank_icon = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i:2d}"
            log.info(f"  {rank_icon} {stock.get('ts_code', 'N/A'):<12} {stock.get('name', 'N/A'):<12} {prob_pct:>6.2f}%")

        # 4. 生成报告和保存结果
        # 最新结果存放在 data/result/{model_name}/
        output_dir = "data/result/left_breakout"
        os.makedirs(output_dir, exist_ok=True)

        if not args.no_report:
            try:
                report_file = predictor.generate_prediction_report(predictions, output_dir=output_dir)
                if report_file:
                    log.info(f"📝 报告: {report_file}")
            except Exception as e:
                log.warning(f"生成报告失败: {e}")

        # CSV文件也保存在同一目录
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_file = os.path.join(output_dir, f"left_breakout_predictions_{timestamp}.csv")
        predictions.to_csv(csv_file, index=False, encoding='utf-8')
        log.info(f"💾 CSV: {csv_file}")

        log.info("\n" + "="*60)
        log.info("✅ 预测完成")
        log.info("="*60)

    except Exception as e:
        log.error(f"❌ 预测失败: {e}")
        import traceback
        log.error(traceback.format_exc())
        sys.exit(1)


def print_prediction_table(predictions):
    """打印预测结果表格"""
    if predictions.empty:
        return

    # 打印表头
    print(f"{'排名':<6} {'股票代码':<12} {'股票名称':<10} {'概率':<10}")
    print("-" * 80)

    # 打印前10个结果
    for i, (_, row) in enumerate(predictions.head(10).iterrows(), 1):
        rank = f"{i:2d}"
        name = f"{row.get('name', 'N/A'):8}"
        prob = f"{row.get('probability', 0) * 100:6.2f}%"
        print(f"{rank:<6} {row.get('ts_code', 'N/A'):<12} {name:<10} {prob:<10}")


if __name__ == "__main__":
    main()
