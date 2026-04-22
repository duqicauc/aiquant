"""
生成静态可视化图表
无需Web界面，直接生成PNG/HTML图表文件
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path
import sys

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import log

class ChartGenerator:
    """图表生成器"""
    
    def __init__(self, output_dir: str = "data/training/charts"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"图表输出目录: {self.output_dir}")
    
    def generate_sample_distribution(self):
        """生成样本分布图"""
        log.info("生成样本分布图...")
        
        # 加载正样本（使用新路径）
        positive_file = Path("data/training/samples/positive_samples.csv")
        if not positive_file.exists():
            log.warning("正样本文件不存在")
            return
        
        df = pd.read_csv(positive_file)
        
        # 1. 涨幅分布
        if 'rise_pct' in df.columns:
            fig = px.histogram(df, x='rise_pct', nbins=50,
                             title='正样本涨幅分布',
                             labels={'rise_pct': '涨幅 (%)', 'count': '样本数'})
            fig.update_traces(marker_color='#1f77b4')
            fig.write_html(str(self.output_dir / "sample_rise_distribution.html"))
            log.info("✓ 生成: sample_rise_distribution.html")
        
        # 2. 时间分布
        if 'start_date' in df.columns:
            df['year'] = pd.to_datetime(df['start_date']).dt.year
            year_counts = df['year'].value_counts().sort_index()
            
            fig = px.bar(x=year_counts.index, y=year_counts.values,
                        title='正样本时间分布',
                        labels={'x': '年份', 'y': '样本数'})
            fig.update_traces(marker_color='#2ca02c')
            fig.write_html(str(self.output_dir / "sample_time_distribution.html"))
            log.info("✓ 生成: sample_time_distribution.html")
        
        # 3. 股票分布 (Top 20)
        if 'ts_code' in df.columns:
            stock_counts = df['ts_code'].value_counts().head(20)
            
            fig = px.bar(x=stock_counts.index, y=stock_counts.values,
                        title='正样本股票分布 (Top 20)',
                        labels={'x': '股票代码', 'y': '样本数'})
            fig.update_xaxes(tickangle=45)
            fig.write_html(str(self.output_dir / "sample_stock_distribution.html"))
            log.info("✓ 生成: sample_stock_distribution.html")
    
    def generate_feature_importance(self):
        """生成特征重要性图"""
        log.info("生成特征重要性图...")
        
        # 尝试从模型中提取特征重要性（使用新路径）
        # 查找最新的模型文件
        model_dir = Path("data/training/models")
        model_files = list(model_dir.glob("xgboost_timeseries_*.json"))
        if not model_files:
            log.warning("模型文件不存在")
            return
        
        model_file = max(model_files, key=lambda x: x.stat().st_mtime)
        if not model_file.exists():
            log.warning("模型文件不存在")
            return
        
        try:
            import joblib
            model = joblib.load(model_file)
            
            if hasattr(model, 'feature_importances_'):
                feature_names = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else [f"feature_{i}" for i in range(len(model.feature_importances_))]
                importance = model.feature_importances_
                
                # 排序并取Top 20
                df_importance = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importance
                }).sort_values('importance', ascending=True).tail(20)
                
                fig = px.bar(df_importance, x='importance', y='feature',
                            orientation='h',
                            title='特征重要性 (Top 20)',
                            labels={'importance': '重要性', 'feature': '特征'})
                fig.write_html(str(self.output_dir / "feature_importance.html"))
                log.info("✓ 生成: feature_importance.html")
        
        except Exception as e:
            log.error(f"生成特征重要性图失败: {e}")
    
    def generate_prediction_analysis(self):
        """生成预测结果分析图"""
        log.info("生成预测结果分析图...")
        
        # 查找最新预测结果（使用新路径）
        pred_results_dir = Path("data/prediction/results")
        if not pred_results_dir.exists():
            log.warning("预测结果目录不存在")
            return
        
        result_files = sorted(pred_results_dir.glob("stock_scores_*.csv"), reverse=True)
        if not result_files:
            log.warning("没有找到评分文件")
            return
        
        df = pd.read_csv(result_files[0])
        
        # 1. 概率分布
        if '牛股概率' in df.columns:
            fig = px.histogram(df, x='牛股概率', nbins=50,
                             title='预测概率分布',
                             labels={'牛股概率': '牛股概率', 'count': '股票数量'})
            fig.update_traces(marker_color='#ff7f0e')
            fig.write_html(str(self.output_dir / "prediction_probability_distribution.html"))
            log.info("✓ 生成: prediction_probability_distribution.html")
        
        # 2. Top 20 概率条形图
        if '牛股概率' in df.columns and '股票代码' in df.columns:
            top_20 = df.head(20)
            
            fig = px.bar(top_20, x='股票代码', y='牛股概率',
                        title='Top 20 股票预测概率',
                        labels={'股票代码': '股票代码', '牛股概率': '牛股概率'},
                        text='股票名称' if '股票名称' in top_20.columns else None)
            fig.update_xaxes(tickangle=45)
            fig.write_html(str(self.output_dir / "prediction_top20.html"))
            log.info("✓ 生成: prediction_top20.html")
        
        # 3. 概率 vs 涨幅散点图
        if '牛股概率' in df.columns and '34日涨幅%' in df.columns:
            fig = px.scatter(df.head(100), x='34日涨幅%', y='牛股概率',
                           title='预测概率 vs 历史涨幅 (Top 100)',
                           labels={'34日涨幅%': '34日涨幅 (%)', '牛股概率': '牛股概率'},
                           hover_data=['股票代码', '股票名称'] if '股票代码' in df.columns else None)
            fig.write_html(str(self.output_dir / "prediction_scatter.html"))
            log.info("✓ 生成: prediction_scatter.html")
    
    def generate_walk_forward_analysis(self):
        """生成Walk-Forward验证分析图"""
        log.info("生成Walk-Forward验证分析图...")
        
        # 使用新路径查找Walk-Forward验证结果
        result_file = Path("data/training/metrics/walk_forward_validation_results.json")
        if not result_file.exists():
            log.warning("Walk-Forward验证结果不存在")
            return
        
        with open(result_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        windows = results.get('windows', [])
        if not windows:
            log.warning("没有验证窗口数据")
            return
        
        df = pd.DataFrame(windows)
        
        # 创建2x2子图
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('准确率', 'AUC-ROC', '精确率', '召回率')
        )
        
        # 准确率
        fig.add_trace(
            go.Scatter(x=df['window_id'], y=df['accuracy'],
                      mode='lines+markers', name='准确率',
                      line=dict(color='#1f77b4', width=3)),
            row=1, col=1
        )
        
        # AUC
        fig.add_trace(
            go.Scatter(x=df['window_id'], y=df['auc'],
                      mode='lines+markers', name='AUC',
                      line=dict(color='#ff7f0e', width=3)),
            row=1, col=2
        )
        
        # 精确率
        fig.add_trace(
            go.Scatter(x=df['window_id'], y=df['precision'],
                      mode='lines+markers', name='精确率',
                      line=dict(color='#2ca02c', width=3)),
            row=2, col=1
        )
        
        # 召回率
        fig.add_trace(
            go.Scatter(x=df['window_id'], y=df['recall'],
                      mode='lines+markers', name='召回率',
                      line=dict(color='#d62728', width=3)),
            row=2, col=2
        )
        
        fig.update_layout(height=800, showlegend=False,
                         title_text="Walk-Forward 验证结果")
        fig.write_html(str(self.output_dir / "walk_forward_results.html"))
        log.info("✓ 生成: walk_forward_results.html")
    
    def generate_all(self):
        """生成所有图表"""
        log.info("=" * 60)
        log.info("开始生成所有图表...")
        log.info("=" * 60)
        
        self.generate_sample_distribution()
        self.generate_feature_importance()
        self.generate_prediction_analysis()
        self.generate_walk_forward_analysis()
        
        log.info("=" * 60)
        log.info(f"✓ 所有图表已生成到: {self.output_dir}")
        log.info("=" * 60)
        
        # 生成索引页面
        self.generate_index()
    
    def generate_index(self):
        """生成图表索引页面"""
        html_content = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AIQuant 可视化图表</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            color: #1f77b4;
            text-align: center;
            margin-bottom: 30px;
        }
        .chart-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 30px;
        }
        .chart-card {
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }
        .chart-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        .chart-card h3 {
            color: #333;
            margin-top: 0;
        }
        .chart-card p {
            color: #666;
            margin: 10px 0;
        }
        .chart-card a {
            display: inline-block;
            background-color: #1f77b4;
            color: white;
            padding: 10px 20px;
            border-radius: 4px;
            text-decoration: none;
            margin-top: 10px;
        }
        .chart-card a:hover {
            background-color: #1557a0;
        }
        .category {
            margin-top: 40px;
        }
        .category h2 {
            color: #2ca02c;
            border-bottom: 2px solid #2ca02c;
            padding-bottom: 10px;
        }
    </style>
</head>
<body>
    <h1>📊 AIQuant 可视化图表</h1>
    
    <div class="category">
        <h2>📈 样本分析</h2>
        <div class="chart-grid">
            <div class="chart-card">
                <h3>涨幅分布</h3>
                <p>正样本的涨幅分布情况</p>
                <a href="sample_rise_distribution.html" target="_blank">查看图表 →</a>
            </div>
            <div class="chart-card">
                <h3>时间分布</h3>
                <p>正样本在各年份的分布</p>
                <a href="sample_time_distribution.html" target="_blank">查看图表 →</a>
            </div>
            <div class="chart-card">
                <h3>股票分布</h3>
                <p>产生最多正样本的股票 Top 20</p>
                <a href="sample_stock_distribution.html" target="_blank">查看图表 →</a>
            </div>
        </div>
    </div>
    
    <div class="category">
        <h2>🎯 模型分析</h2>
        <div class="chart-grid">
            <div class="chart-card">
                <h3>特征重要性</h3>
                <p>模型中最重要的 Top 20 特征</p>
                <a href="feature_importance.html" target="_blank">查看图表 →</a>
            </div>
            <div class="chart-card">
                <h3>Walk-Forward验证</h3>
                <p>模型在不同时间窗口的性能表现</p>
                <a href="walk_forward_results.html" target="_blank">查看图表 →</a>
            </div>
        </div>
    </div>
    
    <div class="category">
        <h2>💎 预测分析</h2>
        <div class="chart-grid">
            <div class="chart-card">
                <h3>概率分布</h3>
                <p>预测概率的整体分布情况</p>
                <a href="prediction_probability_distribution.html" target="_blank">查看图表 →</a>
            </div>
            <div class="chart-card">
                <h3>Top 20 股票</h3>
                <p>预测概率最高的 20 只股票</p>
                <a href="prediction_top20.html" target="_blank">查看图表 →</a>
            </div>
            <div class="chart-card">
                <h3>概率-涨幅关系</h3>
                <p>预测概率与历史涨幅的关系</p>
                <a href="prediction_scatter.html" target="_blank">查看图表 →</a>
            </div>
        </div>
    </div>
    
    <div class="category">
    <footer style="text-align: center; margin-top: 50px; color: #666;">
        <p>AIQuant v3.0 | 生成时间: """ + str(pd.Timestamp.now()) + """</p>
    </footer>
</body>
</html>
        """
        
        with open(self.output_dir / "index.html", 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        log.info(f"✓ 生成索引页面: {self.output_dir / 'index.html'}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='生成可视化图表')
    parser.add_argument('--output', type=str, default='data/charts',
                       help='输出目录，默认: data/charts')
    parser.add_argument('--type', type=str, default='all',
                       choices=['all', 'sample', 'feature', 'prediction', 'walk_forward'],
                       help='图表类型，默认: all')
    
    args = parser.parse_args()
    
    generator = ChartGenerator(output_dir=args.output)
    
    if args.type == 'all':
        generator.generate_all()
    elif args.type == 'sample':
        generator.generate_sample_distribution()
    elif args.type == 'feature':
        generator.generate_feature_importance()
    elif args.type == 'prediction':
        generator.generate_prediction_analysis()
    elif args.type == 'walk_forward':
        generator.generate_walk_forward_analysis()
    
    log.info(f"\n📊 查看图表: open {args.output}/index.html")


if __name__ == '__main__':
    main()

