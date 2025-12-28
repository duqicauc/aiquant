#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
正负样本质量对比可视化工具

功能：
1. 对比正负样本的基础统计
2. 可视化特征分布对比
3. 时间分布对比
4. 生成HTML报告
"""
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
try:
    import seaborn as sns
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

from src.utils.logger import log

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class SampleComparisonVisualizer:
    """正负样本对比可视化器"""
    
    def __init__(self):
        """初始化"""
        self.project_root = PROJECT_ROOT
        # 更新路径：使用新的目录结构
        self.training_dir = self.project_root / 'data' / 'training'
        self.samples_dir = self.training_dir / 'samples'
        self.features_dir = self.training_dir / 'features'
        self.output_dir = self.training_dir / 'charts'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 文件路径（使用新路径）
        self.positive_samples_file = self.samples_dir / 'positive_samples.csv'
        self.positive_features_file = self.features_dir / 'feature_data_34d.csv'
        self.negative_samples_file = self.samples_dir / 'negative_samples_v2.csv'
        self.negative_features_file = self.features_dir / 'negative_feature_data_v2_34d.csv'
        
        # 数据
        self.df_pos_samples = None
        self.df_pos_features = None
        self.df_neg_samples = None
        self.df_neg_features = None
        
        self._load_data()
    
    def _load_data(self):
        """加载数据"""
        log.info("="*80)
        log.info("加载正负样本数据")
        log.info("="*80)
        
        # 加载正样本
        if self.positive_samples_file.exists():
            self.df_pos_samples = pd.read_csv(self.positive_samples_file)
            log.success(f"✓ 正样本列表: {len(self.df_pos_samples)} 条")
        else:
            log.warning(f"✗ 正样本列表不存在: {self.positive_samples_file}")
        
        if self.positive_features_file.exists():
            self.df_pos_features = pd.read_csv(self.positive_features_file)
            log.success(f"✓ 正样本特征: {len(self.df_pos_features)} 条")
        else:
            log.warning(f"✗ 正样本特征不存在: {self.positive_features_file}")
        
        # 加载负样本
        if self.negative_samples_file.exists():
            self.df_neg_samples = pd.read_csv(self.negative_samples_file)
            log.success(f"✓ 负样本列表: {len(self.df_neg_samples)} 条")
        else:
            log.warning(f"✗ 负样本列表不存在: {self.negative_samples_file}")
        
        if self.negative_features_file.exists():
            self.df_neg_features = pd.read_csv(self.negative_features_file)
            log.success(f"✓ 负样本特征: {len(self.df_neg_features)} 条")
        else:
            log.warning(f"✗ 负样本特征不存在: {self.negative_features_file}")
        
        log.info("")
    
    def compare_basic_stats(self):
        """对比基础统计"""
        log.info("="*80)
        log.info("基础统计对比")
        log.info("="*80)
        
        stats = []
        
        if self.df_pos_samples is not None:
            stats.append({
                '类型': '正样本',
                '样本数': len(self.df_pos_samples),
                '股票数': self.df_pos_samples['ts_code'].nunique() if 'ts_code' in self.df_pos_samples.columns else 0,
                '特征记录数': len(self.df_pos_features) if self.df_pos_features is not None else 0
            })
            
            if 'total_return' in self.df_pos_samples.columns:
                stats[-1]['平均涨幅'] = self.df_pos_samples['total_return'].mean()
                stats[-1]['涨幅中位数'] = self.df_pos_samples['total_return'].median()
                stats[-1]['涨幅范围'] = f"{self.df_pos_samples['total_return'].min():.1f}% - {self.df_pos_samples['total_return'].max():.1f}%"
        
        if self.df_neg_samples is not None:
            stats.append({
                '类型': '负样本',
                '样本数': len(self.df_neg_samples),
                '股票数': self.df_neg_samples['ts_code'].nunique() if 'ts_code' in self.df_neg_samples.columns else 0,
                '特征记录数': len(self.df_neg_features) if self.df_neg_features is not None else 0
            })
        
        df_stats = pd.DataFrame(stats)
        print(df_stats.to_string(index=False))
        log.info("")
        
        return df_stats
    
    def visualize_sample_count_comparison(self):
        """可视化样本数量对比"""
        log.info("生成样本数量对比图...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 样本数量对比
        categories = ['样本数', '股票数', '特征记录数']
        pos_values = [
            len(self.df_pos_samples) if self.df_pos_samples is not None else 0,
            self.df_pos_samples['ts_code'].nunique() if self.df_pos_samples is not None and 'ts_code' in self.df_pos_samples.columns else 0,
            len(self.df_pos_features) if self.df_pos_features is not None else 0
        ]
        neg_values = [
            len(self.df_neg_samples) if self.df_neg_samples is not None else 0,
            self.df_neg_samples['ts_code'].nunique() if self.df_neg_samples is not None and 'ts_code' in self.df_neg_samples.columns else 0,
            len(self.df_neg_features) if self.df_neg_features is not None else 0
        ]
        
        x = np.arange(len(categories))
        width = 0.35
        
        axes[0].bar(x - width/2, pos_values, width, label='正样本', color='#2ecc71', alpha=0.8)
        axes[0].bar(x + width/2, neg_values, width, label='负样本', color='#e74c3c', alpha=0.8)
        axes[0].set_xlabel('指标')
        axes[0].set_ylabel('数量')
        axes[0].set_title('样本数量对比', fontsize=14, fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(categories)
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)
        
        # 添加数值标签
        for i, (pos, neg) in enumerate(zip(pos_values, neg_values)):
            axes[0].text(i - width/2, pos, f'{pos:,}', ha='center', va='bottom', fontsize=9)
            axes[0].text(i + width/2, neg, f'{neg:,}', ha='center', va='bottom', fontsize=9)
        
        # 样本比例饼图
        if self.df_pos_samples is not None and self.df_neg_samples is not None:
            sizes = [len(self.df_pos_samples), len(self.df_neg_samples)]
            labels = ['正样本', '负样本']
            colors = ['#2ecc71', '#e74c3c']
            
            axes[1].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', 
                       startangle=90, textprops={'fontsize': 12})
            axes[1].set_title('样本比例分布', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        output_file = self.output_dir / 'sample_count_comparison.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        log.success(f"✓ 已保存: {output_file}")
    
    def visualize_time_distribution(self):
        """可视化时间分布对比"""
        if self.df_pos_samples is None or self.df_neg_samples is None:
            log.warning("缺少样本数据，跳过时间分布可视化")
            return
        
        if 't1_date' not in self.df_pos_samples.columns or 't1_date' not in self.df_neg_samples.columns:
            log.warning("缺少日期字段，跳过时间分布可视化")
            return
        
        log.info("生成时间分布对比图...")
        
        df_pos = self.df_pos_samples.copy()
        df_neg = self.df_neg_samples.copy()
        
        df_pos['t1_date'] = pd.to_datetime(df_pos['t1_date'])
        df_neg['t1_date'] = pd.to_datetime(df_neg['t1_date'])
        
        df_pos['year'] = df_pos['t1_date'].dt.year
        df_neg['year'] = df_neg['t1_date'].dt.year
        
        # 统计每年数量
        pos_year_counts = df_pos['year'].value_counts().sort_index()
        neg_year_counts = df_neg['year'].value_counts().sort_index()
        
        # 合并年份
        all_years = sorted(set(pos_year_counts.index) | set(neg_year_counts.index))
        pos_counts = [pos_year_counts.get(year, 0) for year in all_years]
        neg_counts = [neg_year_counts.get(year, 0) for year in all_years]
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        x = np.arange(len(all_years))
        width = 0.35
        
        ax.bar(x - width/2, pos_counts, width, label='正样本', color='#2ecc71', alpha=0.8)
        ax.bar(x + width/2, neg_counts, width, label='负样本', color='#e74c3c', alpha=0.8)
        
        ax.set_xlabel('年份', fontsize=12)
        ax.set_ylabel('样本数量', fontsize=12)
        ax.set_title('正负样本时间分布对比', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(all_years, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_file = self.output_dir / 'time_distribution_comparison.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        log.success(f"✓ 已保存: {output_file}")
    
    def visualize_feature_distribution(self):
        """可视化特征分布对比"""
        if self.df_pos_features is None or self.df_neg_features is None:
            log.warning("缺少特征数据，跳过特征分布可视化")
            return
        
        log.info("生成特征分布对比图...")
        
        # 选择数值型特征进行对比
        numeric_cols = ['close', 'pct_chg', 'volume_ratio', 'macd', 'rsi_6', 'rsi_12', 
                       'ma5', 'ma10', 'total_mv', 'circ_mv']
        
        available_cols = [col for col in numeric_cols if col in self.df_pos_features.columns 
                         and col in self.df_neg_features.columns]
        
        if len(available_cols) == 0:
            log.warning("没有可用的特征列进行对比")
            return
        
        # 选择前6个特征
        selected_cols = available_cols[:6]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, col in enumerate(selected_cols):
            ax = axes[idx]
            
            pos_data = self.df_pos_features[col].dropna()
            neg_data = self.df_neg_features[col].dropna()
            
            if len(pos_data) > 0 and len(neg_data) > 0:
                # 计算合理的bins
                all_data = pd.concat([pos_data, neg_data])
                bins = np.linspace(all_data.min(), all_data.max(), 30)
                
                ax.hist(pos_data, bins=bins, alpha=0.6, label='正样本', color='#2ecc71', density=True)
                ax.hist(neg_data, bins=bins, alpha=0.6, label='负样本', color='#e74c3c', density=True)
                
                ax.set_xlabel(col, fontsize=10)
                ax.set_ylabel('密度', fontsize=10)
                ax.set_title(f'{col} 分布对比', fontsize=11, fontweight='bold')
                ax.legend()
                ax.grid(alpha=0.3)
        
        plt.tight_layout()
        output_file = self.output_dir / 'feature_distribution_comparison.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        log.success(f"✓ 已保存: {output_file}")
    
    def visualize_return_distribution(self):
        """可视化涨幅分布（仅正样本）"""
        if self.df_pos_samples is None or 'total_return' not in self.df_pos_samples.columns:
            log.warning("缺少涨幅数据，跳过涨幅分布可视化")
            return
        
        log.info("生成涨幅分布图...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 总涨幅分布
        returns = self.df_pos_samples['total_return']
        axes[0].hist(returns, bins=50, color='#2ecc71', alpha=0.7, edgecolor='black')
        axes[0].axvline(returns.mean(), color='red', linestyle='--', linewidth=2, label=f'均值: {returns.mean():.1f}%')
        axes[0].axvline(returns.median(), color='blue', linestyle='--', linewidth=2, label=f'中位数: {returns.median():.1f}%')
        axes[0].set_xlabel('总涨幅 (%)', fontsize=12)
        axes[0].set_ylabel('样本数量', fontsize=12)
        axes[0].set_title('正样本总涨幅分布', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # 最高涨幅分布
        if 'max_return' in self.df_pos_samples.columns:
            max_returns = self.df_pos_samples['max_return']
            axes[1].hist(max_returns, bins=50, color='#3498db', alpha=0.7, edgecolor='black')
            axes[1].axvline(max_returns.mean(), color='red', linestyle='--', linewidth=2, label=f'均值: {max_returns.mean():.1f}%')
            axes[1].axvline(max_returns.median(), color='blue', linestyle='--', linewidth=2, label=f'中位数: {max_returns.median():.1f}%')
            axes[1].set_xlabel('最高涨幅 (%)', fontsize=12)
            axes[1].set_ylabel('样本数量', fontsize=12)
            axes[1].set_title('正样本最高涨幅分布', fontsize=14, fontweight='bold')
            axes[1].legend()
            axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        output_file = self.output_dir / 'return_distribution.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        log.success(f"✓ 已保存: {output_file}")
    
    def generate_html_report(self):
        """生成HTML报告"""
        log.info("生成HTML报告...")
        
        html_file = self.output_dir / 'sample_quality_comparison.html'
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>正负样本质量对比报告</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
        }}
        .stats-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        .stats-table th, .stats-table td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        .stats-table th {{
            background-color: #3498db;
            color: white;
        }}
        .stats-table tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        .image-container {{
            text-align: center;
            margin: 30px 0;
        }}
        .image-container img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 5px;
        }}
        .timestamp {{
            color: #7f8c8d;
            font-size: 14px;
            margin-top: 30px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 正负样本质量对比报告</h1>
        <p class="timestamp">生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>📈 基础统计对比</h2>
        <table class="stats-table">
            <tr>
                <th>指标</th>
                <th>正样本</th>
                <th>负样本</th>
            </tr>
"""
        
        # 添加统计信息
        if self.df_pos_samples is not None and self.df_neg_samples is not None:
            html_content += f"""
            <tr>
                <td><strong>样本数量</strong></td>
                <td>{len(self.df_pos_samples):,}</td>
                <td>{len(self.df_neg_samples):,}</td>
            </tr>
            <tr>
                <td><strong>股票数量</strong></td>
                <td>{self.df_pos_samples['ts_code'].nunique() if 'ts_code' in self.df_pos_samples.columns else 'N/A':,}</td>
                <td>{self.df_neg_samples['ts_code'].nunique() if 'ts_code' in self.df_neg_samples.columns else 'N/A':,}</td>
            </tr>
            <tr>
                <td><strong>特征记录数</strong></td>
                <td>{len(self.df_pos_features) if self.df_pos_features is not None else 0:,}</td>
                <td>{len(self.df_neg_features) if self.df_neg_features is not None else 0:,}</td>
            </tr>
"""
            
            if 'total_return' in self.df_pos_samples.columns:
                html_content += f"""
            <tr>
                <td><strong>平均涨幅</strong></td>
                <td>{self.df_pos_samples['total_return'].mean():.2f}%</td>
                <td>N/A</td>
            </tr>
            <tr>
                <td><strong>涨幅中位数</strong></td>
                <td>{self.df_pos_samples['total_return'].median():.2f}%</td>
                <td>N/A</td>
            </tr>
"""
        
        html_content += """
        </table>
        
        <h2>📊 可视化图表</h2>
        
        <div class="image-container">
            <h3>样本数量对比</h3>
            <img src="sample_count_comparison.png" alt="样本数量对比">
        </div>
"""
        
        # 检查图表文件是否存在
        if (self.output_dir / 'time_distribution_comparison.png').exists():
            html_content += """
        <div class="image-container">
            <h3>时间分布对比</h3>
            <img src="time_distribution_comparison.png" alt="时间分布对比">
        </div>
"""
        
        if (self.output_dir / 'feature_distribution_comparison.png').exists():
            html_content += """
        <div class="image-container">
            <h3>特征分布对比</h3>
            <img src="feature_distribution_comparison.png" alt="特征分布对比">
        </div>
"""
        
        if (self.output_dir / 'return_distribution.png').exists():
            html_content += """
        <div class="image-container">
            <h3>正样本涨幅分布</h3>
            <img src="return_distribution.png" alt="涨幅分布">
        </div>
"""
        
        html_content += """
    </div>
</body>
</html>
"""
        
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        log.success(f"✓ HTML报告已保存: {html_file}")
        log.info(f"💡 在浏览器中打开查看: {html_file}")
    
    def generate_all(self):
        """生成所有可视化"""
        log.info("\n" + "="*80)
        log.info("开始生成正负样本质量对比可视化")
        log.info("="*80)
        log.info("")
        
        # 1. 基础统计对比
        self.compare_basic_stats()
        
        # 2. 生成图表
        self.visualize_sample_count_comparison()
        self.visualize_time_distribution()
        self.visualize_feature_distribution()
        self.visualize_return_distribution()
        
        # 3. 生成HTML报告
        self.generate_html_report()
        
        log.info("\n" + "="*80)
        log.success("✅ 所有可视化图表已生成完成！")
        log.info("="*80)
        log.info(f"📁 输出目录: {self.output_dir}")
        log.info(f"📄 HTML报告: {self.output_dir / 'sample_quality_comparison.html'}")
        log.info("")


def main():
    """主函数"""
    visualizer = SampleComparisonVisualizer()
    visualizer.generate_all()


if __name__ == '__main__':
    main()

