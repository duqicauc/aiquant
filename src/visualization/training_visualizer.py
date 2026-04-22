"""
训练过程可视化工具
用于新训练框架下的样本质量检查和因子重要性分析
"""

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

from src.utils.logger import log


class TrainingVisualizer:
    """训练过程可视化器"""

    def __init__(self, output_dir: str = "data/training/charts"):
        """
        初始化可视化器

        Args:
            output_dir: 图表输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"可视化图表输出目录: {self.output_dir}")

    def visualize_sample_quality(self, df_samples: pd.DataFrame, save_prefix: str = "sample_quality"):
        """
        可视化样本质量

        Args:
            df_samples: 样本DataFrame（包含正样本或负样本）
            save_prefix: 保存文件前缀
        """
        log.info("=" * 80)
        log.info("生成样本质量可视化图表")
        log.info("=" * 80)

        if df_samples is None or len(df_samples) == 0:
            log.warning("样本数据为空，跳过可视化")
            return

        # 1. 涨幅分布直方图
        if "total_return" in df_samples.columns:
            self._plot_return_distribution(df_samples, save_prefix)

        # 2. 时间分布
        if "t1_date" in df_samples.columns:
            self._plot_time_distribution(df_samples, save_prefix)

        # 3. 涨幅统计箱线图
        if "total_return" in df_samples.columns and "max_return" in df_samples.columns:
            self._plot_return_boxplot(df_samples, save_prefix)

        # 4. 异常值检测可视化
        if "total_return" in df_samples.columns:
            self._plot_anomaly_detection(df_samples, save_prefix)

        # 5. 样本质量综合报告
        self._generate_quality_report(df_samples, save_prefix)

        log.success(f"✓ 样本质量可视化图表已生成到: {self.output_dir}")

    def _plot_return_distribution(self, df: pd.DataFrame, prefix: str):
        """绘制涨幅分布图"""
        fig = make_subplots(rows=2, cols=1, subplot_titles=("总涨幅分布", "最高涨幅分布"), vertical_spacing=0.15)

        # 总涨幅分布
        fig.add_trace(
            go.Histogram(x=df["total_return"], nbinsx=50, name="总涨幅", marker_color="#1f77b4", opacity=0.7),
            row=1,
            col=1,
        )

        # 最高涨幅分布
        if "max_return" in df.columns:
            fig.add_trace(
                go.Histogram(x=df["max_return"], nbinsx=50, name="最高涨幅", marker_color="#ff7f0e", opacity=0.7),
                row=2,
                col=1,
            )

        fig.update_layout(height=800, title_text="样本涨幅分布分析", showlegend=False)

        fig.update_xaxes(title_text="涨幅 (%)", row=1, col=1)
        fig.update_xaxes(title_text="涨幅 (%)", row=2, col=1)
        fig.update_yaxes(title_text="样本数", row=1, col=1)
        fig.update_yaxes(title_text="样本数", row=2, col=1)

        output_file = self.output_dir / f"{prefix}_return_distribution.html"
        fig.write_html(str(output_file))
        log.info(f"✓ 生成: {output_file.name}")

    def _plot_time_distribution(self, df: pd.DataFrame, prefix: str):
        """绘制时间分布图"""
        df["t1_date"] = pd.to_datetime(df["t1_date"])
        df["year"] = df["t1_date"].dt.year
        df["month"] = df["t1_date"].dt.month

        # 按年份统计
        year_counts = df["year"].value_counts().sort_index()

        fig = make_subplots(
            rows=2, cols=1, subplot_titles=("按年份分布", "按月份分布（所有年份）"), vertical_spacing=0.15
        )

        # 年份分布
        fig.add_trace(
            go.Bar(
                x=year_counts.index,
                y=year_counts.values,
                name="样本数",
                marker_color="#2ca02c",
                text=year_counts.values,
                textposition="outside",
            ),
            row=1,
            col=1,
        )

        # 月份分布
        month_counts = df["month"].value_counts().sort_index()
        fig.add_trace(
            go.Bar(
                x=month_counts.index,
                y=month_counts.values,
                name="样本数",
                marker_color="#d62728",
                text=month_counts.values,
                textposition="outside",
            ),
            row=2,
            col=1,
        )

        fig.update_layout(height=800, title_text="样本时间分布分析", showlegend=False)

        fig.update_xaxes(title_text="年份", row=1, col=1)
        fig.update_xaxes(title_text="月份", row=2, col=1)
        fig.update_yaxes(title_text="样本数", row=1, col=1)
        fig.update_yaxes(title_text="样本数", row=2, col=1)

        output_file = self.output_dir / f"{prefix}_time_distribution.html"
        fig.write_html(str(output_file))
        log.info(f"✓ 生成: {output_file.name}")

    def _plot_return_boxplot(self, df: pd.DataFrame, prefix: str):
        """绘制涨幅箱线图"""
        fig = go.Figure()

        fig.add_trace(go.Box(y=df["total_return"], name="总涨幅", marker_color="#1f77b4"))

        if "max_return" in df.columns:
            fig.add_trace(go.Box(y=df["max_return"], name="最高涨幅", marker_color="#ff7f0e"))

        fig.update_layout(title="涨幅统计箱线图", yaxis_title="涨幅 (%)", height=500, boxmode="group")

        output_file = self.output_dir / f"{prefix}_return_boxplot.html"
        fig.write_html(str(output_file))
        log.info(f"✓ 生成: {output_file.name}")

    def _plot_anomaly_detection(self, df: pd.DataFrame, prefix: str):
        """绘制异常值检测图"""
        # 使用IQR方法检测异常值
        Q1 = df["total_return"].quantile(0.25)
        Q3 = df["total_return"].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        df["is_outlier"] = (df["total_return"] < lower_bound) | (df["total_return"] > upper_bound)
        df["is_extreme"] = df["total_return"] > 200

        fig = go.Figure()

        # 正常值
        normal = df[~df["is_outlier"]]
        if len(normal) > 0:
            fig.add_trace(
                go.Scatter(
                    x=normal.index,
                    y=normal["total_return"],
                    mode="markers",
                    name="正常值",
                    marker=dict(color="#2ca02c", size=5, opacity=0.6),
                )
            )

        # 异常值
        outliers = df[df["is_outlier"] & ~df["is_extreme"]]
        if len(outliers) > 0:
            fig.add_trace(
                go.Scatter(
                    x=outliers.index,
                    y=outliers["total_return"],
                    mode="markers",
                    name="异常值 (IQR)",
                    marker=dict(color="#ff7f0e", size=8, opacity=0.8),
                )
            )

        # 极端值
        extreme = df[df["is_extreme"]]
        if len(extreme) > 0:
            fig.add_trace(
                go.Scatter(
                    x=extreme.index,
                    y=extreme["total_return"],
                    mode="markers",
                    name="极端值 (>200%)",
                    marker=dict(color="#d62728", size=10, opacity=0.9),
                )
            )

        # 添加阈值线
        fig.add_hline(
            y=50, line_dash="dash", line_color="blue", annotation_text="最低阈值 (50%)", annotation_position="right"
        )
        fig.add_hline(
            y=upper_bound,
            line_dash="dash",
            line_color="orange",
            annotation_text=f"异常值上界 ({upper_bound:.1f}%)",
            annotation_position="right",
        )

        fig.update_layout(
            title="异常值检测分析", xaxis_title="样本索引", yaxis_title="涨幅 (%)", height=600, hovermode="closest"
        )

        output_file = self.output_dir / f"{prefix}_anomaly_detection.html"
        fig.write_html(str(output_file))
        log.info(f"✓ 生成: {output_file.name}")

    def _generate_quality_report(self, df: pd.DataFrame, prefix: str):
        """生成样本质量综合报告"""
        report = {
            "生成时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "样本总数": len(df),
            "股票数量": df["ts_code"].nunique() if "ts_code" in df.columns else 0,
        }

        if "total_return" in df.columns:
            report["总涨幅统计"] = {
                "平均值": f"{df['total_return'].mean():.2f}%",
                "中位数": f"{df['total_return'].median():.2f}%",
                "最小值": f"{df['total_return'].min():.2f}%",
                "最大值": f"{df['total_return'].max():.2f}%",
                "标准差": f"{df['total_return'].std():.2f}%",
            }

        if "max_return" in df.columns:
            report["最高涨幅统计"] = {
                "平均值": f"{df['max_return'].mean():.2f}%",
                "中位数": f"{df['max_return'].median():.2f}%",
                "最小值": f"{df['max_return'].min():.2f}%",
                "最大值": f"{df['max_return'].max():.2f}%",
            }

        if "t1_date" in df.columns:
            df["t1_date"] = pd.to_datetime(df["t1_date"])
            report["时间范围"] = {
                "最早日期": df["t1_date"].min().strftime("%Y-%m-%d"),
                "最晚日期": df["t1_date"].max().strftime("%Y-%m-%d"),
            }

        # 数据质量检查
        issues = []
        if df.isnull().sum().sum() > 0:
            issues.append(f"存在空值: {df.isnull().sum().sum()} 个")

        if "total_return" in df.columns and "max_return" in df.columns:
            invalid = len(df[df["total_return"] > df["max_return"]])
            if invalid > 0:
                issues.append(f"逻辑错误: {invalid} 个样本总涨幅 > 最高涨幅")

        report["数据质量"] = {"问题数量": len(issues), "问题列表": issues if issues else ["无问题"]}

        # 保存为JSON

        report_file = self.output_dir / f"{prefix}_quality_report.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        log.info(f"✓ 生成: {report_file.name}")

    def visualize_feature_importance(
        self, feature_importance: pd.DataFrame, model_name: str = "model", top_n: int = 20
    ):
        """
        可视化因子重要性

        Args:
            feature_importance: 特征重要性DataFrame，包含'feature'和'importance'列
            model_name: 模型名称
            top_n: 显示前N个重要特征
        """
        log.info("=" * 80)
        log.info("生成因子重要性可视化图表")
        log.info("=" * 80)

        if feature_importance is None or len(feature_importance) == 0:
            log.warning("特征重要性数据为空，跳过可视化")
            return

        # 排序并取Top N
        df_sorted = feature_importance.sort_values("importance", ascending=False).head(top_n)

        # 1. 水平条形图（Top N）
        self._plot_importance_bar(df_sorted, model_name, top_n)

        # 2. 特征重要性分布
        self._plot_importance_distribution(feature_importance, model_name)

        # 3. 特征重要性热力图（按类别分组）
        self._plot_importance_heatmap(df_sorted, model_name)

        # 4. 累积重要性
        self._plot_cumulative_importance(feature_importance, model_name)

        log.success(f"✓ 因子重要性可视化图表已生成到: {self.output_dir}")

    def _plot_importance_bar(self, df: pd.DataFrame, model_name: str, top_n: int):
        """绘制特征重要性条形图"""
        fig = go.Figure()

        # 按重要性排序（从大到小）
        df_sorted = df.sort_values("importance", ascending=True)

        # 使用渐变色
        colors = px.colors.sequential.Viridis_r[: len(df_sorted)]

        fig.add_trace(
            go.Bar(
                x=df_sorted["importance"],
                y=df_sorted["feature"],
                orientation="h",
                marker=dict(color=colors, line=dict(color="rgba(0,0,0,0.3)", width=1)),
                text=[f"{v:.4f}" for v in df_sorted["importance"]],
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>重要性: %{x:.4f}<extra></extra>",
            )
        )

        fig.update_layout(
            title=f"特征重要性 Top {top_n}",
            xaxis_title="重要性",
            yaxis_title="特征",
            height=max(600, len(df_sorted) * 30),
            hovermode="closest",
            margin=dict(l=200, r=50, t=50, b=50),
        )

        output_file = self.output_dir / f"{model_name}_feature_importance_top{top_n}.html"
        fig.write_html(str(output_file))
        log.info(f"✓ 生成: {output_file.name}")

    def _plot_importance_distribution(self, df: pd.DataFrame, model_name: str):
        """绘制特征重要性分布直方图"""
        fig = go.Figure()

        fig.add_trace(
            go.Histogram(x=df["importance"], nbinsx=50, marker_color="#1f77b4", opacity=0.7, name="重要性分布")
        )

        # 添加统计线
        mean_importance = df["importance"].mean()
        median_importance = df["importance"].median()

        fig.add_vline(
            x=mean_importance,
            line_dash="dash",
            line_color="red",
            annotation_text=f"平均值: {mean_importance:.4f}",
            annotation_position="top",
        )

        fig.add_vline(
            x=median_importance,
            line_dash="dash",
            line_color="orange",
            annotation_text=f"中位数: {median_importance:.4f}",
            annotation_position="top",
        )

        fig.update_layout(title="特征重要性分布", xaxis_title="重要性", yaxis_title="特征数量", height=500)

        output_file = self.output_dir / f"{model_name}_feature_importance_distribution.html"
        fig.write_html(str(output_file))
        log.info(f"✓ 生成: {output_file.name}")

    def _plot_importance_heatmap(self, df: pd.DataFrame, model_name: str):
        """绘制特征重要性热力图（按特征类别分组）"""
        # 根据特征名称推断类别
        df["category"] = df["feature"].apply(self._categorize_feature)

        # 按类别分组统计
        category_importance = df.groupby("category")["importance"].sum().sort_values(ascending=False)

        fig = go.Figure(
            data=go.Heatmap(
                z=[category_importance.values],
                x=category_importance.index,
                y=["重要性总和"],
                colorscale="Viridis",
                text=[[f"{v:.4f}" for v in category_importance.values]],
                texttemplate="%{text}",
                textfont={"size": 12},
                showscale=True,
            )
        )

        fig.update_layout(
            title="特征重要性按类别汇总",
            xaxis_title="特征类别",
            yaxis_title="",
            height=300,
            margin=dict(l=100, r=50, t=50, b=100),
        )

        output_file = self.output_dir / f"{model_name}_feature_importance_heatmap.html"
        fig.write_html(str(output_file))
        log.info(f"✓ 生成: {output_file.name}")

    def _plot_cumulative_importance(self, df: pd.DataFrame, model_name: str):
        """绘制累积重要性图"""
        df_sorted = df.sort_values("importance", ascending=False)
        df_sorted["cumulative"] = df_sorted["importance"].cumsum()
        df_sorted["cumulative_pct"] = df_sorted["cumulative"] / df_sorted["importance"].sum() * 100

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=list(range(1, len(df_sorted) + 1)),
                y=df_sorted["cumulative_pct"],
                mode="lines+markers",
                name="累积重要性",
                line=dict(color="#1f77b4", width=3),
                marker=dict(size=5),
            )
        )

        # 添加80%和90%线
        fig.add_hline(y=80, line_dash="dash", line_color="orange", annotation_text="80%", annotation_position="right")
        fig.add_hline(y=90, line_dash="dash", line_color="red", annotation_text="90%", annotation_position="right")

        # 找出达到80%和90%需要的特征数
        n_80 = (
            (df_sorted["cumulative_pct"] >= 80).idxmax() + 1
            if (df_sorted["cumulative_pct"] >= 80).any()
            else len(df_sorted)
        )
        (
            (df_sorted["cumulative_pct"] >= 90).idxmax() + 1
            if (df_sorted["cumulative_pct"] >= 90).any()
            else len(df_sorted)
        )

        fig.add_annotation(x=n_80, y=80, text=f"前{n_80}个特征占80%", showarrow=True, arrowhead=2, ax=0, ay=-40)

        fig.update_layout(
            title="特征累积重要性分析",
            xaxis_title="特征数量（按重要性排序）",
            yaxis_title="累积重要性 (%)",
            height=600,
            hovermode="x unified",
        )

        output_file = self.output_dir / f"{model_name}_feature_cumulative_importance.html"
        fig.write_html(str(output_file))
        log.info(f"✓ 生成: {output_file.name}")

    def _categorize_feature(self, feature_name: str) -> str:
        """根据特征名称推断类别"""
        name_lower = feature_name.lower()

        if "close" in name_lower or "price" in name_lower:
            return "价格特征"
        elif "pct_chg" in name_lower or "return" in name_lower or "gain" in name_lower or "loss" in name_lower:
            return "涨跌幅特征"
        elif "volume" in name_lower:
            return "成交量特征"
        elif "macd" in name_lower:
            return "MACD特征"
        elif "ma" in name_lower:
            return "均线特征"
        elif "mv" in name_lower or "市值" in name_lower:
            return "市值特征"
        elif "momentum" in name_lower or "trend" in name_lower:
            return "动量特征"
        else:
            return "其他特征"

    def generate_index_page(self, model_name: str = "training"):
        """生成可视化图表索引页面"""
        html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>训练可视化图表 - {model_name}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #1f77b4;
            text-align: center;
            margin-bottom: 30px;
        }}
        .chart-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 20px;
            margin-top: 30px;
        }}
        .chart-card {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }}
        .chart-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }}
        .chart-card h3 {{
            color: #333;
            margin-top: 0;
        }}
        .chart-card p {{
            color: #666;
            margin: 10px 0;
        }}
        .chart-card a {{
            display: inline-block;
            background-color: #1f77b4;
            color: white;
            padding: 10px 20px;
            border-radius: 4px;
            text-decoration: none;
            margin-top: 10px;
        }}
        .chart-card a:hover {{
            background-color: #1557a0;
        }}
        .category {{
            margin-top: 40px;
        }}
        .category h2 {{
            color: #2ca02c;
            border-bottom: 2px solid #2ca02c;
            padding-bottom: 10px;
        }}
    </style>
</head>
<body>
    <h1>📊 训练可视化图表 - {model_name}</h1>
    <p style="text-align: center; color: #666;">生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

    <div class="category">
        <h2>📈 样本质量分析</h2>
        <div class="chart-grid">
            <div class="chart-card">
                <h3>涨幅分布</h3>
                <p>样本涨幅的分布情况，包括总涨幅和最高涨幅</p>
                <a href="sample_quality_return_distribution.html" target="_blank">查看图表 →</a>
            </div>
            <div class="chart-card">
                <h3>时间分布</h3>
                <p>样本在不同年份和月份的分布情况</p>
                <a href="sample_quality_time_distribution.html" target="_blank">查看图表 →</a>
            </div>
            <div class="chart-card">
                <h3>涨幅箱线图</h3>
                <p>涨幅的统计分布，包括中位数、四分位数等</p>
                <a href="sample_quality_return_boxplot.html" target="_blank">查看图表 →</a>
            </div>
            <div class="chart-card">
                <h3>异常值检测</h3>
                <p>使用IQR方法检测的异常值和极端值</p>
                <a href="sample_quality_anomaly_detection.html" target="_blank">查看图表 →</a>
            </div>
        </div>
    </div>

    <div class="category">
        <h2>🔍 特征质量评估</h2>
        <div class="chart-grid">
            <div class="chart-card">
                <h3>正负样本特征分布对比</h3>
                <p>Top 10特征在正负样本中的分布对比</p>
                <a href="{model_name}_feature_distribution_comparison.html" target="_blank">查看图表 →</a>
            </div>
            <div class="chart-card">
                <h3>特征相关性热力图</h3>
                <p>Top 20特征之间的相关性分析</p>
                <a href="{model_name}_feature_correlation.html" target="_blank">查看图表 →</a>
            </div>
            <div class="chart-card">
                <h3>特征缺失值分析</h3>
                <p>各特征的缺失值情况</p>
                <a href="{model_name}_feature_missing_values.html" target="_blank">查看图表 →</a>
            </div>
            <div class="chart-card">
                <h3>特征统计信息</h3>
                <p>Top 20特征的统计信息（均值、标准差等）</p>
                <a href="{model_name}_feature_statistics.html" target="_blank">查看图表 →</a>
            </div>
        </div>
    </div>

    <div class="category">
        <h2>🎯 因子重要性分析</h2>
        <div class="chart-grid">
            <div class="chart-card">
                <h3>Top 20 特征重要性</h3>
                <p>最重要的20个特征及其重要性得分</p>
                <a href="{model_name}_feature_importance_top20.html" target="_blank">查看图表 →</a>
            </div>
            <div class="chart-card">
                <h3>重要性分布</h3>
                <p>所有特征重要性的分布情况</p>
                <a href="{model_name}_feature_importance_distribution.html" target="_blank">查看图表 →</a>
            </div>
            <div class="chart-card">
                <h3>类别汇总热力图</h3>
                <p>按特征类别汇总的重要性</p>
                <a href="{model_name}_feature_importance_heatmap.html" target="_blank">查看图表 →</a>
            </div>
            <div class="chart-card">
                <h3>累积重要性</h3>
                <p>累积重要性分析，显示需要多少特征达到80%/90%</p>
                <a href="{model_name}_feature_cumulative_importance.html" target="_blank">查看图表 →</a>
            </div>
        </div>
    </div>

    <div class="category">
        <h2>📈 模型训练过程</h2>
        <div class="chart-grid">
            <div class="chart-card">
                <h3>训练曲线</h3>
                <p>模型训练过程中的损失和指标变化</p>
                <a href="{model_name}_training_curves.html" target="_blank">查看图表 →</a>
            </div>
            <div class="chart-card">
                <h3>学习曲线</h3>
                <p>不同训练集大小下的模型性能</p>
                <a href="{model_name}_learning_curves.html" target="_blank">查看图表 →</a>
            </div>
        </div>
    </div>

    <div class="category">
        <h2>📊 模型结果评测</h2>
        <div class="chart-grid">
            <div class="chart-card">
                <h3>ROC曲线</h3>
                <p>接收者操作特征曲线，评估分类性能</p>
                <a href="{model_name}_roc_curve.html" target="_blank">查看图表 →</a>
            </div>
            <div class="chart-card">
                <h3>PR曲线</h3>
                <p>精确率-召回率曲线</p>
                <a href="{model_name}_pr_curve.html" target="_blank">查看图表 →</a>
            </div>
            <div class="chart-card">
                <h3>混淆矩阵</h3>
                <p>分类结果的混淆矩阵可视化</p>
                <a href="{model_name}_confusion_matrix.html" target="_blank">查看图表 →</a>
            </div>
            <div class="chart-card">
                <h3>预测概率分布</h3>
                <p>正负样本的预测概率分布对比</p>
                <a href="{model_name}_prediction_distribution.html" target="_blank">查看图表 →</a>
            </div>
            <div class="chart-card">
                <h3>预测结果分析</h3>
                <p>不同阈值下的精确率、召回率和F1分数</p>
                <a href="{model_name}_prediction_analysis.html" target="_blank">查看图表 →</a>
            </div>
        </div>
    </div>

    <footer style="text-align: center; margin-top: 50px; color: #666;">
        <p>AIQuant 训练可视化 | 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </footer>
</body>
</html>
        """

        index_file = self.output_dir / "index.html"
        with open(index_file, "w", encoding="utf-8") as f:
            f.write(html_content)

        log.info(f"✓ 生成索引页面: {index_file}")

    def visualize_feature_quality(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame = None,
        y_test: pd.Series = None,
        model_name: str = "model",
    ):
        """
        可视化特征质量评估

        Args:
            X_train: 训练集特征
            y_train: 训练集标签
            X_test: 测试集特征（可选）
            y_test: 测试集标签（可选）
            model_name: 模型名称
        """
        log.info("=" * 80)
        log.info("生成特征质量可视化图表")
        log.info("=" * 80)

        # 1. 特征分布对比（正负样本）
        self._plot_feature_distribution_comparison(X_train, y_train, model_name)

        # 2. 特征相关性热力图
        self._plot_feature_correlation(X_train, model_name)

        # 3. 特征缺失值分析
        self._plot_missing_values(X_train, model_name)

        # 4. 特征统计信息
        self._plot_feature_statistics(X_train, y_train, model_name)

        log.success(f"✓ 特征质量可视化图表已生成到: {self.output_dir}")

    def _plot_feature_distribution_comparison(self, X: pd.DataFrame, y: pd.Series, model_name: str):
        """绘制正负样本特征分布对比（Top 10特征）"""
        # 选择前10个特征
        top_features = X.columns[:10].tolist()

        fig = make_subplots(rows=2, cols=5, subplot_titles=top_features, vertical_spacing=0.15, horizontal_spacing=0.1)

        for idx, feature in enumerate(top_features):
            row = (idx // 5) + 1
            col = (idx % 5) + 1

            # 正样本分布
            pos_data = X[y == 1][feature].dropna()
            # 负样本分布
            neg_data = X[y == 0][feature].dropna()

            if len(pos_data) > 0:
                fig.add_trace(
                    go.Histogram(
                        x=pos_data, name="正样本", marker_color="#2ca02c", opacity=0.6, nbinsx=30, showlegend=(idx == 0)
                    ),
                    row=row,
                    col=col,
                )

            if len(neg_data) > 0:
                fig.add_trace(
                    go.Histogram(
                        x=neg_data, name="负样本", marker_color="#d62728", opacity=0.6, nbinsx=30, showlegend=(idx == 0)
                    ),
                    row=row,
                    col=col,
                )

        fig.update_layout(height=800, title_text="正负样本特征分布对比（Top 10特征）", showlegend=True)

        output_file = self.output_dir / f"{model_name}_feature_distribution_comparison.html"
        fig.write_html(str(output_file))
        log.info(f"✓ 生成: {output_file.name}")

    def _plot_feature_correlation(self, X: pd.DataFrame, model_name: str):
        """绘制特征相关性热力图"""
        # 计算相关性矩阵（只计算前20个特征，避免图表过大）
        top_features = X.columns[:20].tolist()
        corr_matrix = X[top_features].corr()

        fig = go.Figure(
            data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                colorscale="RdBu",
                zmid=0,
                text=corr_matrix.values.round(2),
                texttemplate="%{text}",
                textfont={"size": 8},
                colorbar=dict(title="相关系数"),
            )
        )

        fig.update_layout(
            title="特征相关性热力图（Top 20特征）", height=800, width=1000, xaxis_title="特征", yaxis_title="特征"
        )

        output_file = self.output_dir / f"{model_name}_feature_correlation.html"
        fig.write_html(str(output_file))
        log.info(f"✓ 生成: {output_file.name}")

    def _plot_missing_values(self, X: pd.DataFrame, model_name: str):
        """绘制特征缺失值分析"""
        missing_counts = X.isnull().sum()
        missing_pct = (missing_counts / len(X)) * 100

        # 只显示有缺失值的特征
        missing_data = pd.DataFrame(
            {"feature": missing_counts.index, "missing_count": missing_counts.values, "missing_pct": missing_pct.values}
        )
        missing_data = missing_data[missing_data["missing_count"] > 0].sort_values("missing_count", ascending=False)

        if len(missing_data) == 0:
            log.info("✓ 没有缺失值，跳过缺失值分析图")
            return

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=missing_data["feature"],
                y=missing_data["missing_pct"],
                marker_color="#ff7f0e",
                text=[f"{v:.1f}%" for v in missing_data["missing_pct"]],
                textposition="outside",
            )
        )

        fig.update_layout(
            title="特征缺失值分析", xaxis_title="特征", yaxis_title="缺失值百分比 (%)", height=600, xaxis_tickangle=-45
        )

        output_file = self.output_dir / f"{model_name}_feature_missing_values.html"
        fig.write_html(str(output_file))
        log.info(f"✓ 生成: {output_file.name}")

    def _plot_feature_statistics(self, X: pd.DataFrame, y: pd.Series, model_name: str):
        """绘制特征统计信息"""
        stats = []
        for col in X.columns[:20]:  # 只显示前20个特征
            stats.append(
                {
                    "feature": col,
                    "mean": X[col].mean(),
                    "std": X[col].std(),
                    "min": X[col].min(),
                    "max": X[col].max(),
                    "median": X[col].median(),
                }
            )

        df_stats = pd.DataFrame(stats)

        fig = make_subplots(
            rows=2, cols=2, subplot_titles=("均值", "标准差", "最小值", "最大值"), vertical_spacing=0.15
        )

        # 均值
        fig.add_trace(
            go.Bar(x=df_stats["feature"], y=df_stats["mean"], name="均值", marker_color="#1f77b4"), row=1, col=1
        )

        # 标准差
        fig.add_trace(
            go.Bar(x=df_stats["feature"], y=df_stats["std"], name="标准差", marker_color="#ff7f0e"), row=1, col=2
        )

        # 最小值
        fig.add_trace(
            go.Bar(x=df_stats["feature"], y=df_stats["min"], name="最小值", marker_color="#2ca02c"), row=2, col=1
        )

        # 最大值
        fig.add_trace(
            go.Bar(x=df_stats["feature"], y=df_stats["max"], name="最大值", marker_color="#d62728"), row=2, col=2
        )

        fig.update_layout(height=800, title_text="特征统计信息（Top 20特征）", showlegend=False)

        fig.update_xaxes(tickangle=-45, row=1, col=1)
        fig.update_xaxes(tickangle=-45, row=1, col=2)
        fig.update_xaxes(tickangle=-45, row=2, col=1)
        fig.update_xaxes(tickangle=-45, row=2, col=2)

        output_file = self.output_dir / f"{model_name}_feature_statistics.html"
        fig.write_html(str(output_file))
        log.info(f"✓ 生成: {output_file.name}")

    def visualize_training_process(
        self,
        model,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str = "model",
    ):
        """
        可视化模型训练过程

        Args:
            model: 训练好的模型
            X_train, y_train: 训练集
            X_test, y_test: 测试集
            model_name: 模型名称
        """
        log.info("=" * 80)
        log.info("生成模型训练过程可视化图表")
        log.info("=" * 80)

        # 1. 训练曲线（如果模型支持）
        if hasattr(model, "evals_result_") and model.evals_result_:
            self._plot_training_curves(model, model_name)

        # 2. 学习曲线（训练集和测试集性能对比）
        self._plot_learning_curves(model, X_train, y_train, X_test, y_test, model_name)

        log.success(f"✓ 训练过程可视化图表已生成到: {self.output_dir}")

    def _plot_training_curves(self, model, model_name: str):
        """绘制训练曲线"""
        evals_result = model.evals_result_

        if not evals_result:
            return

        fig = go.Figure()

        # XGBoost的evals_result_结构
        for eval_set_name, metrics_dict in evals_result.items():
            for metric_name, values in metrics_dict.items():
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(values))),
                        y=values,
                        mode="lines+markers",
                        name=f"{eval_set_name} - {metric_name}",
                        line=dict(width=2),
                    )
                )

        fig.update_layout(
            title="模型训练曲线", xaxis_title="迭代次数", yaxis_title="指标值", height=600, hovermode="x unified"
        )

        output_file = self.output_dir / f"{model_name}_training_curves.html"
        fig.write_html(str(output_file))
        log.info(f"✓ 生成: {output_file.name}")

    def _plot_learning_curves(self, model, X_train, y_train, X_test, y_test, model_name: str):
        """绘制学习曲线（不同训练集大小下的性能）"""
        from sklearn.metrics import accuracy_score, roc_auc_score

        train_sizes = np.linspace(0.1, 1.0, 10)
        train_scores = []
        test_scores = []
        train_aucs = []
        test_aucs = []

        for size in train_sizes:
            n_samples = int(len(X_train) * size)
            X_train_subset = X_train.iloc[:n_samples]
            y_train_subset = y_train.iloc[:n_samples]

            # 训练模型
            model_copy = type(model)(**model.get_params())
            model_copy.fit(X_train_subset, y_train_subset, verbose=False)

            # 评估
            train_pred = model_copy.predict(X_train_subset)
            train_prob = model_copy.predict_proba(X_train_subset)[:, 1]
            test_pred = model_copy.predict(X_test)
            test_prob = model_copy.predict_proba(X_test)[:, 1]

            train_scores.append(accuracy_score(y_train_subset, train_pred))
            test_scores.append(accuracy_score(y_test, test_pred))

            try:
                train_aucs.append(roc_auc_score(y_train_subset, train_prob))
                test_aucs.append(roc_auc_score(y_test, test_prob))
            except Exception:
                train_aucs.append(0)
                test_aucs.append(0)

        fig = make_subplots(rows=1, cols=2, subplot_titles=("准确率学习曲线", "AUC学习曲线"))

        # 准确率
        fig.add_trace(
            go.Scatter(
                x=train_sizes,
                y=train_scores,
                name="训练集准确率",
                mode="lines+markers",
                line=dict(color="#1f77b4", width=2),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=train_sizes,
                y=test_scores,
                name="测试集准确率",
                mode="lines+markers",
                line=dict(color="#ff7f0e", width=2),
            ),
            row=1,
            col=1,
        )

        # AUC
        fig.add_trace(
            go.Scatter(
                x=train_sizes, y=train_aucs, name="训练集AUC", mode="lines+markers", line=dict(color="#2ca02c", width=2)
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=train_sizes, y=test_aucs, name="测试集AUC", mode="lines+markers", line=dict(color="#d62728", width=2)
            ),
            row=1,
            col=2,
        )

        fig.update_layout(height=500, title_text="模型学习曲线", showlegend=True)

        fig.update_xaxes(title_text="训练集比例", row=1, col=1)
        fig.update_xaxes(title_text="训练集比例", row=1, col=2)
        fig.update_yaxes(title_text="准确率", row=1, col=1)
        fig.update_yaxes(title_text="AUC", row=1, col=2)

        output_file = self.output_dir / f"{model_name}_learning_curves.html"
        fig.write_html(str(output_file))
        log.info(f"✓ 生成: {output_file.name}")

    def visualize_model_results(
        self, y_true: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray, model_name: str = "model"
    ):
        """
        可视化模型结果评测

        Args:
            y_true: 真实标签
            y_pred: 预测标签
            y_prob: 预测概率
            model_name: 模型名称
        """
        log.info("=" * 80)
        log.info("生成模型结果评测可视化图表")
        log.info("=" * 80)

        # 1. ROC曲线
        self._plot_roc_curve(y_true, y_prob, model_name)

        # 2. PR曲线
        self._plot_pr_curve(y_true, y_prob, model_name)

        # 3. 混淆矩阵
        self._plot_confusion_matrix(y_true, y_pred, model_name)

        # 4. 预测概率分布
        self._plot_prediction_distribution(y_true, y_prob, model_name)

        # 5. 预测结果分析
        self._plot_prediction_analysis(y_true, y_pred, y_prob, model_name)

        log.success(f"✓ 模型结果评测可视化图表已生成到: {self.output_dir}")

    def _plot_roc_curve(self, y_true, y_prob, model_name: str):
        """绘制ROC曲线"""
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        from sklearn.metrics import roc_auc_score

        auc = roc_auc_score(y_true, y_prob)

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=fpr, y=tpr, mode="lines", name=f"ROC曲线 (AUC = {auc:.4f})", line=dict(color="#1f77b4", width=3)
            )
        )

        # 对角线（随机分类器）
        fig.add_trace(
            go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines", name="随机分类器", line=dict(color="red", width=2, dash="dash")
            )
        )

        fig.update_layout(
            title=f"ROC曲线 (AUC = {auc:.4f})",
            xaxis_title="假正率 (FPR)",
            yaxis_title="真正率 (TPR)",
            height=600,
            hovermode="x unified",
        )

        output_file = self.output_dir / f"{model_name}_roc_curve.html"
        fig.write_html(str(output_file))
        log.info(f"✓ 生成: {output_file.name}")

    def _plot_pr_curve(self, y_true, y_prob, model_name: str):
        """绘制PR曲线"""
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        from sklearn.metrics import average_precision_score

        ap = average_precision_score(y_true, y_prob)

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=recall,
                y=precision,
                mode="lines",
                name=f"PR曲线 (AP = {ap:.4f})",
                line=dict(color="#2ca02c", width=3),
                fill="tozeroy",
            )
        )

        fig.update_layout(
            title=f"精确率-召回率曲线 (AP = {ap:.4f})",
            xaxis_title="召回率 (Recall)",
            yaxis_title="精确率 (Precision)",
            height=600,
            hovermode="x unified",
        )

        output_file = self.output_dir / f"{model_name}_pr_curve.html"
        fig.write_html(str(output_file))
        log.info(f"✓ 生成: {output_file.name}")

    def _plot_confusion_matrix(self, y_true, y_pred, model_name: str):
        """绘制混淆矩阵"""
        cm = confusion_matrix(y_true, y_pred)

        fig = go.Figure(
            data=go.Heatmap(
                z=cm,
                x=["预测负样本", "预测正样本"],
                y=["实际负样本", "实际正样本"],
                colorscale="Blues",
                text=cm,
                texttemplate="%{text}",
                textfont={"size": 16},
                colorbar=dict(title="样本数"),
            )
        )

        fig.update_layout(title="混淆矩阵", height=400, width=600)

        output_file = self.output_dir / f"{model_name}_confusion_matrix.html"
        fig.write_html(str(output_file))
        log.info(f"✓ 生成: {output_file.name}")

    def _plot_prediction_distribution(self, y_true, y_prob, model_name: str):
        """绘制预测概率分布"""
        fig = make_subplots(rows=1, cols=2, subplot_titles=("正样本预测概率分布", "负样本预测概率分布"))

        # 正样本
        pos_probs = y_prob[y_true == 1]
        fig.add_trace(
            go.Histogram(x=pos_probs, nbinsx=50, name="正样本", marker_color="#2ca02c", opacity=0.7), row=1, col=1
        )

        # 负样本
        neg_probs = y_prob[y_true == 0]
        fig.add_trace(
            go.Histogram(x=neg_probs, nbinsx=50, name="负样本", marker_color="#d62728", opacity=0.7), row=1, col=2
        )

        fig.update_layout(height=500, title_text="预测概率分布", showlegend=False)

        fig.update_xaxes(title_text="预测概率", row=1, col=1)
        fig.update_xaxes(title_text="预测概率", row=1, col=2)
        fig.update_yaxes(title_text="样本数", row=1, col=1)
        fig.update_yaxes(title_text="样本数", row=1, col=2)

        output_file = self.output_dir / f"{model_name}_prediction_distribution.html"
        fig.write_html(str(output_file))
        log.info(f"✓ 生成: {output_file.name}")

    def _plot_prediction_analysis(self, y_true, y_pred, y_prob, model_name: str):
        """绘制预测结果分析"""
        # 按概率阈值分析
        thresholds = np.linspace(0, 1, 100)
        precisions = []
        recalls = []
        f1_scores = []

        for threshold in thresholds:
            y_pred_thresh = (y_prob >= threshold).astype(int)
            cm = confusion_matrix(y_true, y_pred_thresh)

            if cm.sum() > 0:
                tp = cm[1, 1] if cm.shape == (2, 2) else 0
                fp = cm[0, 1] if cm.shape == (2, 2) else 0
                fn = cm[1, 0] if cm.shape == (2, 2) else 0

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

                precisions.append(precision)
                recalls.append(recall)
                f1_scores.append(f1)
            else:
                precisions.append(0)
                recalls.append(0)
                f1_scores.append(0)

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(x=thresholds, y=precisions, mode="lines", name="精确率", line=dict(color="#1f77b4", width=2))
        )

        fig.add_trace(
            go.Scatter(x=thresholds, y=recalls, mode="lines", name="召回率", line=dict(color="#ff7f0e", width=2))
        )

        fig.add_trace(
            go.Scatter(x=thresholds, y=f1_scores, mode="lines", name="F1分数", line=dict(color="#2ca02c", width=2))
        )

        fig.update_layout(
            title="不同阈值下的性能指标",
            xaxis_title="概率阈值",
            yaxis_title="指标值",
            height=600,
            hovermode="x unified",
        )

        output_file = self.output_dir / f"{model_name}_prediction_analysis.html"
        fig.write_html(str(output_file))
        log.info(f"✓ 生成: {output_file.name}")
