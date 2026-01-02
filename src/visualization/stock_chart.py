"""
股票图表可视化 - 使用 PyEcharts 专业金融图表
K线图、技术指标、买卖点标注、资金流向、行业对比、交易计划

基于百度 ECharts 的专业金融可视化方案
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.data_manager import DataManager
from src.utils.logger import log

# 尝试导入 PyEcharts
try:
    from pyecharts import options as opts
    from pyecharts.charts import (
        Kline, Line, Bar, Grid, Page, Tab, 
        Gauge, Liquid, Radar, Pie, Scatter
    )
    from pyecharts.commons.utils import JsCode
    from pyecharts.globals import ThemeType
    HAS_PYECHARTS = True
except ImportError:
    HAS_PYECHARTS = False
    log.warning("PyEcharts 未安装，将使用 Plotly 作为备选")

# 备选方案：Plotly
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


class StockChartVisualizer:
    """股票图表可视化器 - PyEcharts 版"""
    
    # 专业金融配色
    COLORS = {
        'up': '#ec0000',           # 上涨红色
        'down': '#00da3c',         # 下跌绿色
        'ma5': '#FF6B6B',
        'ma10': '#4ECDC4', 
        'ma20': '#45B7D1',
        'ma60': '#FFA07A',
        'ma120': '#9B59B6',
        'ma233': '#E91E63',        # 233日均线 - 粉红色
        'volume_up': 'rgba(236, 0, 0, 0.6)',
        'volume_down': 'rgba(0, 218, 60, 0.6)',
        'macd_up': '#ec0000',
        'macd_down': '#00da3c',
        'dif': '#2196F3',
        'dea': '#FF9800',
    }
    
    def __init__(self):
        self.dm = DataManager()
        self.use_pyecharts = HAS_PYECHARTS
    
    def create_comprehensive_chart(self, stock_code: str, report: dict, days: int = 120):
        """创建综合分析图表"""
        if self.use_pyecharts:
            return self._create_pyecharts_kline(stock_code, report, days)
        elif HAS_PLOTLY:
            return self._create_plotly_kline(stock_code, report, days)
        else:
            raise RuntimeError("没有可用的可视化库")
    
    def _create_pyecharts_kline(self, stock_code: str, report: dict, days: int = 120):
        """使用 PyEcharts 创建K线图"""
        # 获取历史数据 - 需要额外获取233天数据用于计算233日均线
        end_date = datetime.now().strftime('%Y%m%d')
        # 确保获取足够数据：显示天数 + 233日均线所需 + 缓冲
        # 注意：timedelta 是日历天数，交易日约为日历天的 68%，需要乘以 1.5 转换
        fetch_trading_days = max(days * 2, days + 250)  # 需要的交易日数
        fetch_calendar_days = int(fetch_trading_days * 1.5)  # 转换为日历天数
        start_date = (datetime.now() - timedelta(days=fetch_calendar_days)).strftime('%Y%m%d')
        df = self.dm.get_daily_data(stock_code, start_date, end_date)
        
        if df is None or df.empty:
            log.warning(f"无数据: {stock_code}")
            return None
        
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        
        # 先在完整数据上计算均线（确保233日均线有足够数据）
        df['ma5'] = df['close'].rolling(5).mean().round(2)
        df['ma10'] = df['close'].rolling(10).mean().round(2)
        df['ma20'] = df['close'].rolling(20).mean().round(2)
        df['ma60'] = df['close'].rolling(60).mean().round(2)
        df['ma233'] = df['close'].rolling(233).mean().round(2)
        
        # 计算完均线后，再截取需要显示的天数
        df = df.tail(days).reset_index(drop=True)
        
        # 准备数据
        dates = df['trade_date'].dt.strftime('%Y-%m-%d').tolist()
        kline_data = df[['open', 'close', 'low', 'high']].values.tolist()
        volumes = df['vol'].tolist()
        
        # 获取均线数据
        ma5 = df['ma5'].tolist()
        ma10 = df['ma10'].tolist()
        ma20 = df['ma20'].tolist()
        ma60 = df['ma60'].tolist()
        ma233 = df['ma233'].tolist()
        
        # 计算 MACD
        macd_data = self._calculate_macd(df['close'])
        
        # 计算成交量颜色
        vol_colors = []
        for i in range(len(df)):
            if df.iloc[i]['close'] >= df.iloc[i]['open']:
                vol_colors.append(self.COLORS['volume_up'])
            else:
                vol_colors.append(self.COLORS['volume_down'])
        
        # 基本信息
        basic = report.get('basic_info', {})
        stock_name = basic.get('name', stock_code)
        score = report.get('overall_score', 0)
        
        # 准备成交量数据（带颜色信息）- 根据涨跌标记
        vol_data_with_color = []
        for i in range(len(df)):
            is_up = df.iloc[i]['close'] >= df.iloc[i]['open']
            vol_data_with_color.append({
                'value': volumes[i],
                'itemStyle': {'color': self.COLORS['up'] if is_up else self.COLORS['down']}
            })
        
        # 准备MACD柱状图数据（带颜色信息）- 根据正负标记
        macd_bar_data = []
        for val in macd_data['macd']:
            if val is None:
                macd_bar_data.append({'value': 0, 'itemStyle': {'color': '#888'}})
            else:
                macd_bar_data.append({
                    'value': val,
                    'itemStyle': {'color': self.COLORS['up'] if val >= 0 else self.COLORS['down']}
                })
        
        # 创建K线图
        kline = (
            Kline()
            .add_xaxis(dates)
            .add_yaxis(
                series_name="K线",
                y_axis=kline_data,
                itemstyle_opts=opts.ItemStyleOpts(
                    color=self.COLORS['up'],
                    color0=self.COLORS['down'],
                    border_color=self.COLORS['up'],
                    border_color0=self.COLORS['down'],
                ),
            )
            .set_global_opts(
                title_opts=opts.TitleOpts(
                    title=f"{stock_name} ({stock_code})",
                    subtitle=f"综合评分: {score:.1f}",
                    pos_left="2%",  # 标题靠左
                    title_textstyle_opts=opts.TextStyleOpts(font_size=16),
                    subtitle_textstyle_opts=opts.TextStyleOpts(font_size=12),
                ),
                legend_opts=opts.LegendOpts(
                    is_show=True,
                    pos_top="0%",
                    pos_left="center",  # 图例居中
                    orient="horizontal",
                    item_width=12,  # 图例标记宽度
                    item_height=8,  # 图例标记高度
                    item_gap=20,  # 增加图例间距
                    textstyle_opts=opts.TextStyleOpts(font_size=10),
                    selected_mode="multiple",
                ),
                xaxis_opts=opts.AxisOpts(
                    type_="category",
                    is_scale=True,
                    boundary_gap=False,
                    axisline_opts=opts.AxisLineOpts(is_on_zero=False),
                    splitline_opts=opts.SplitLineOpts(is_show=False),
                    split_number=20,
                    min_="dataMin",
                    max_="dataMax",
                ),
                yaxis_opts=opts.AxisOpts(
                    name="价格",  # Y轴标题
                    name_location="middle",
                    name_gap=40,
                    name_textstyle_opts=opts.TextStyleOpts(font_size=12, color="#aaa"),
                    is_scale=True,
                    splitarea_opts=opts.SplitAreaOpts(
                        is_show=True,
                        areastyle_opts=opts.AreaStyleOpts(opacity=1)
                    ),
                ),
                tooltip_opts=opts.TooltipOpts(
                    trigger="axis",
                    axis_pointer_type="cross",
                ),
                datazoom_opts=[
                    opts.DataZoomOpts(
                        is_show=True,
                        type_="inside",
                        xaxis_index=[0, 1, 2],
                        range_start=70,
                        range_end=100,
                    ),
                    opts.DataZoomOpts(
                        is_show=True,
                        xaxis_index=[0, 1, 2],
                        type_="slider",
                        pos_top="92%",
                        range_start=70,
                        range_end=100,
                    ),
                ],
            )
        )
        
        # 添加均线 - 简化图例名称
        ma_line = (
            Line()
            .add_xaxis(dates)
            .add_yaxis("5日", ma5, is_smooth=True, 
                      linestyle_opts=opts.LineStyleOpts(width=1.5, color=self.COLORS['ma5']),
                      label_opts=opts.LabelOpts(is_show=False))
            .add_yaxis("10日", ma10, is_smooth=True,
                      linestyle_opts=opts.LineStyleOpts(width=1.5, color=self.COLORS['ma10']),
                      label_opts=opts.LabelOpts(is_show=False))
            .add_yaxis("20日", ma20, is_smooth=True,
                      linestyle_opts=opts.LineStyleOpts(width=1.5, color=self.COLORS['ma20']),
                      label_opts=opts.LabelOpts(is_show=False))
            .add_yaxis("60日", ma60, is_smooth=True,
                      linestyle_opts=opts.LineStyleOpts(width=1.5, color=self.COLORS['ma60']),
                      label_opts=opts.LabelOpts(is_show=False))
            .add_yaxis("233日", ma233, is_smooth=True,
                      linestyle_opts=opts.LineStyleOpts(width=2, color=self.COLORS['ma233']),  # 233日均线加粗
                      label_opts=opts.LabelOpts(is_show=False))
            .set_global_opts(
                xaxis_opts=opts.AxisOpts(type_="category"),
            )
        )
        
        # 叠加均线到K线
        kline.overlap(ma_line)
        
        # 成交量柱状图 - 使用预处理好的带颜色数据（不显示图例）
        bar = (
            Bar()
            .add_xaxis(dates)
            .add_yaxis(
                "",  # 空名称，不显示图例
                vol_data_with_color,  # 使用带颜色信息的数据
                xaxis_index=1,
                yaxis_index=1,
                label_opts=opts.LabelOpts(is_show=False),
            )
            .set_global_opts(
                xaxis_opts=opts.AxisOpts(
                    type_="category",
                    is_scale=True,
                    grid_index=1,
                    boundary_gap=False,
                    axisline_opts=opts.AxisLineOpts(is_on_zero=False),
                    axistick_opts=opts.AxisTickOpts(is_show=False),
                    splitline_opts=opts.SplitLineOpts(is_show=False),
                    axislabel_opts=opts.LabelOpts(is_show=False),
                    split_number=20,
                    min_="dataMin",
                    max_="dataMax",
                ),
                yaxis_opts=opts.AxisOpts(
                    name="成交量",  # Y轴标题
                    name_location="middle",
                    name_gap=50,
                    name_textstyle_opts=opts.TextStyleOpts(font_size=11, color="#aaa"),
                    grid_index=1,
                    is_scale=True,
                    split_number=2,
                    axislabel_opts=opts.LabelOpts(is_show=True),
                    axisline_opts=opts.AxisLineOpts(is_show=True),
                    axistick_opts=opts.AxisTickOpts(is_show=True),
                    splitline_opts=opts.SplitLineOpts(is_show=True),
                ),
            )
        )
        
        # MACD 图 - 使用预处理好的带颜色数据（不显示图例）
        macd_bar = (
            Bar()
            .add_xaxis(dates)
            .add_yaxis(
                "",  # 空名称，不显示图例
                macd_bar_data,  # 使用带颜色信息的数据
                xaxis_index=2,
                yaxis_index=2,
                label_opts=opts.LabelOpts(is_show=False),
            )
            .set_global_opts(
                xaxis_opts=opts.AxisOpts(
                    type_="category",
                    grid_index=2,
                    axislabel_opts=opts.LabelOpts(is_show=False),
                ),
                yaxis_opts=opts.AxisOpts(
                    name="MACD",  # Y轴标题
                    name_location="middle",
                    name_gap=50,
                    name_textstyle_opts=opts.TextStyleOpts(font_size=11, color="#aaa"),
                    grid_index=2,
                    split_number=4,
                    axisline_opts=opts.AxisLineOpts(is_on_zero=False),
                    axistick_opts=opts.AxisTickOpts(is_show=False),
                    splitline_opts=opts.SplitLineOpts(is_show=False),
                    axislabel_opts=opts.LabelOpts(is_show=True),
                ),
            )
        )
        
        macd_line = (
            Line()
            .add_xaxis(dates)
            .add_yaxis("", macd_data['dif'], is_smooth=True, xaxis_index=2, yaxis_index=2,  # 不显示DIF图例
                      linestyle_opts=opts.LineStyleOpts(width=1.5, color=self.COLORS['dif']),
                      label_opts=opts.LabelOpts(is_show=False))
            .add_yaxis("", macd_data['dea'], is_smooth=True, xaxis_index=2, yaxis_index=2,  # 不显示DEA图例
                      linestyle_opts=opts.LineStyleOpts(width=1.5, color=self.COLORS['dea']),
                      label_opts=opts.LabelOpts(is_show=False))
        )
        
        macd_bar.overlap(macd_line)
        
        # 使用 Grid 组合图表
        grid = (
            Grid(init_opts=opts.InitOpts(width="100%", height="800px", theme=ThemeType.DARK))
            .add(
                kline,
                grid_opts=opts.GridOpts(pos_left="10%", pos_right="8%", pos_top="10%", height="46%"),  # 给图例更多空间
            )
            .add(
                bar,
                grid_opts=opts.GridOpts(pos_left="10%", pos_right="8%", pos_top="60%", height="12%"),
            )
            .add(
                macd_bar,
                grid_opts=opts.GridOpts(pos_left="10%", pos_right="8%", pos_top="75%", height="12%"),
            )
        )
        
        return grid
    
    def _calculate_macd(self, close: pd.Series):
        """计算MACD"""
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        dif = ema12 - ema26
        dea = dif.ewm(span=9, adjust=False).mean()
        macd = (dif - dea) * 2
        
        return {
            'dif': dif.round(4).tolist(),
            'dea': dea.round(4).tolist(),
            'macd': macd.round(4).tolist()
        }
    
    def create_indicators_heatmap(self, report: dict):
        """创建指标健康度仪表盘"""
        if not self.use_pyecharts:
            return self._create_plotly_heatmap(report)
        
        # 收集指标数据
        indicators = []
        
        tech = report.get('technical_analysis', {})
        model = report.get('model_prediction', {})
        risk = report.get('risk_assessment', {})
        market = report.get('market_context', {})
        
        if tech:
            trend_score = tech.get('trend', {}).get('alignment_score', 5)
            indicators.append(('趋势', trend_score))
            pv_score = tech.get('volume_analysis', {}).get('pv_score', 5)
            indicators.append(('量价', pv_score))
        
        if model and 'score' in model:
            indicators.append(('AI预测', model['score']))
        
        if risk:
            indicators.append(('风险', risk.get('risk_score', 5)))
        
        if market:
            market_score = market.get('market_score', 50) / 10
            indicators.append(('市场', market_score))
        
        # 创建多个仪表盘
        from pyecharts.charts import Page
        
        gauges = []
        for name, score in indicators:
            gauge = (
                Gauge(init_opts=opts.InitOpts(width="300px", height="250px", theme=ThemeType.DARK))
                .add(
                    series_name=name,
                    data_pair=[(name, round(score * 10, 1))],
                    radius="80%",
                    axisline_opts=opts.AxisLineOpts(
                        linestyle_opts=opts.LineStyleOpts(
                            color=[
                                (0.3, "#fd666d"),
                                (0.7, "#37a2da"),
                                (1, "#67e0e3"),
                            ],
                            width=20,
                        )
                    ),
                    detail_label_opts=opts.LabelOpts(
                        formatter="{value}",
                        font_size=20,
                    ),
                )
                .set_global_opts(
                    title_opts=opts.TitleOpts(title=name, pos_left="center"),
                )
            )
            gauges.append(gauge)
        
        # 组合仪表盘
        page = Page(layout=Page.SimplePageLayout)
        for g in gauges:
            page.add(g)
        
        return page
    
    def create_sector_comparison_chart(self, report: dict):
        """创建行业对比图表"""
        comparison = report.get('sector_comparison', {})
        
        if not comparison or comparison.get('rank') == '未知':
            return None
        
        if not self.use_pyecharts:
            return self._create_plotly_sector(report)
        
        # 数据
        stock_return = comparison.get('20d_returns', 0)
        industry_avg = comparison.get('industry_avg', 0)
        industry_max = comparison.get('industry_max', 0)
        industry_min = comparison.get('industry_min', 0)
        
        categories = ['个股', '行业均值', '行业最高', '行业最低']
        values = [stock_return, industry_avg, industry_max, industry_min]
        
        bar = (
            Bar(init_opts=opts.InitOpts(width="600px", height="400px", theme=ThemeType.DARK))
            .add_xaxis(categories)
            .add_yaxis(
                "20日涨跌幅 (%)",
                values,
                itemstyle_opts=opts.ItemStyleOpts(
                    color=JsCode("""
                    function(params) {
                        var colors = ['#5470c6', '#91cc75', '#ee6666', '#73c0de'];
                        return colors[params.dataIndex];
                    }
                    """)
                ),
                label_opts=opts.LabelOpts(
                    is_show=True,
                    position="top",
                    formatter="{c}%"
                ),
            )
            .set_global_opts(
                title_opts=opts.TitleOpts(
                    title=f"行业对比 - {comparison.get('industry', '')}",
                    subtitle=f"排名: {comparison.get('rank', '')} | {comparison.get('relative_strength', '')}",
                ),
                yaxis_opts=opts.AxisOpts(
                    name="涨跌幅 (%)",
                    axislabel_opts=opts.LabelOpts(formatter="{value}%"),
                ),
            )
        )
        
        return bar
    
    def create_money_flow_chart(self, report: dict):
        """创建资金流向图表"""
        money_flow = report.get('money_flow', {})
        
        if not money_flow or money_flow.get('inflow', 0) == 0:
            return None
        
        if not self.use_pyecharts:
            return self._create_plotly_money_flow(report)
        
        inflow = money_flow.get('inflow', 0) / 1e8  # 转换为亿
        outflow = money_flow.get('outflow', 0) / 1e8
        net_ratio = money_flow.get('net_flow_ratio', 0)
        
        # 饼图
        pie = (
            Pie(init_opts=opts.InitOpts(width="500px", height="350px", theme=ThemeType.DARK))
            .add(
                series_name="资金流向",
                data_pair=[
                    ("流入", round(inflow, 2)),
                    ("流出", round(outflow, 2)),
                ],
                radius=["40%", "70%"],
                center=["50%", "55%"],
                label_opts=opts.LabelOpts(
                    formatter="{b}: {c}亿\n({d}%)",
                ),
            )
            .set_colors(["#67e0e3", "#fd666d"])
            .set_global_opts(
                title_opts=opts.TitleOpts(
                    title="资金流向分析",
                    subtitle=f"净流入比: {net_ratio:.1f}% | {money_flow.get('trend', '')}",
                ),
            )
        )
        
        return pie
    
    def create_trading_plan_chart(self, report: dict):
        """创建交易计划可视化"""
        plan = report.get('trading_plan', {})
        basic = report.get('basic_info', {})
        
        current_price = basic.get('latest_price', 0)
        if current_price <= 0:
            return None
        
        if not self.use_pyecharts:
            return self._create_plotly_trading_plan(report)
        
        entry = plan.get('entry', {})
        exit_plan = plan.get('exit', {})
        
        stop_loss = exit_plan.get('stop_loss', current_price * 0.95)
        ideal_price = entry.get('ideal_price', current_price * 0.98)
        tp1 = exit_plan.get('take_profit_1', current_price * 1.05)
        tp2 = exit_plan.get('take_profit_2', current_price * 1.10)
        
        # 使用水平柱状图展示价位
        categories = ['止损位', '建议买入', '当前价', '目标1', '目标2']
        values = [stop_loss, ideal_price, current_price, tp1, tp2]
        
        bar = (
            Bar(init_opts=opts.InitOpts(width="600px", height="350px", theme=ThemeType.DARK))
            .add_xaxis(categories)
            .add_yaxis(
                "价格",
                values,
                itemstyle_opts=opts.ItemStyleOpts(
                    color=JsCode("""
                    function(params) {
                        var colors = ['#fd666d', '#5470c6', '#fac858', '#67e0e3', '#73c0de'];
                        return colors[params.dataIndex];
                    }
                    """)
                ),
                label_opts=opts.LabelOpts(
                    is_show=True,
                    position="top",
                    formatter="¥{c}"
                ),
            )
            .set_global_opts(
                title_opts=opts.TitleOpts(
                    title="交易计划",
                    subtitle=f"建议仓位: {plan.get('position', {}).get('suggested', 'N/A')}",
                ),
            )
            .reversal_axis()
        )
        
        return bar
    
    def create_pattern_analysis_chart(self, report: dict):
        """创建K线形态分析图表"""
        patterns = report.get('pattern_analysis', {})
        
        all_patterns = []
        for p in patterns.get('single_patterns', []) + patterns.get('compound_patterns', []) + patterns.get('trend_patterns', []):
            if isinstance(p, dict):
                all_patterns.append(p)
        
        if not all_patterns:
            return None
        
        # 简单的文本列表
        return all_patterns  # 在HTML模板中渲染
    
    def create_integrated_html_report(self, stock_code: str, report: dict, days: int = 120) -> str:
        """创建集成的单页HTML报告"""
        
        # 获取基本信息
        basic = report.get('basic_info', {})
        stock_name = basic.get('name', stock_code)
        score = report.get('overall_score', 0)
        recommendation = report.get('recommendation', '')
        signals = report.get('trading_signals', {})
        action = signals.get('action', '观望')
        
        # 生成图表HTML
        chart_htmls = {}
        
        # 1. K线图
        try:
            kline_chart = self.create_comprehensive_chart(stock_code, report, days)
            if kline_chart and self.use_pyecharts:
                chart_id = f"chart_{stock_code.replace('.', '_')}"
                # PyEcharts: 使用 dump_options_with_quotes 获取配置
                js_code = kline_chart.dump_options_with_quotes()
                chart_htmls['kline'] = f'''
                <div id="{chart_id}" style="width: 100%; height: 800px;"></div>
                <script type="text/javascript">
                    var chart_{chart_id} = echarts.init(document.getElementById('{chart_id}'), 'dark');
                    var option_{chart_id} = {js_code};
                    chart_{chart_id}.setOption(option_{chart_id});
                    window.addEventListener('resize', function() {{ chart_{chart_id}.resize(); }});
                </script>
                '''
            elif kline_chart:
                from plotly.io import to_html
                chart_htmls['kline'] = to_html(kline_chart, full_html=False, include_plotlyjs=False)
        except Exception as e:
            import traceback
            chart_htmls['kline'] = f'<div class="error">K线图生成失败: {e}<br>{traceback.format_exc()}</div>'
        
        # 2. 行业对比
        try:
            sector_chart = self.create_sector_comparison_chart(report)
            if sector_chart and self.use_pyecharts:
                chart_htmls['sector'] = sector_chart.render_embed()
        except Exception as e:
            chart_htmls['sector'] = f'<div class="error">行业对比图生成失败: {e}</div>'
        
        # 3. 资金流向
        try:
            money_chart = self.create_money_flow_chart(report)
            if money_chart and self.use_pyecharts:
                chart_htmls['money_flow'] = money_chart.render_embed()
        except Exception as e:
            chart_htmls['money_flow'] = f'<div class="error">资金流向图生成失败: {e}</div>'
        
        # 4. 交易计划
        try:
            plan_chart = self.create_trading_plan_chart(report)
            if plan_chart and self.use_pyecharts:
                chart_htmls['trading_plan'] = plan_chart.render_embed()
        except Exception as e:
            chart_htmls['trading_plan'] = f'<div class="error">交易计划图生成失败: {e}</div>'
        
        # 信号摘要HTML
        buy_signals = signals.get('buy_signals', [])
        sell_signals = signals.get('sell_signals', [])
        warning_signals = signals.get('warning_signals', [])
        
        # 交易计划数据
        plan = report.get('trading_plan', {})
        entry = plan.get('entry', {})
        exit_plan = plan.get('exit', {})
        position = plan.get('position', {})
        
        # K线形态
        patterns = report.get('pattern_analysis', {})
        all_patterns = []
        for p in patterns.get('single_patterns', []) + patterns.get('compound_patterns', []) + patterns.get('trend_patterns', []):
            if isinstance(p, dict):
                all_patterns.append(p)
        
        # 模型预测
        model = report.get('model_prediction', {})
        
        # 技术分析
        tech = report.get('technical_analysis', {})
        trend = tech.get('trend', {})
        indicators = tech.get('indicators', {})
        
        # 风险评估
        risk = report.get('risk_assessment', {})
        
        # 行业对比
        sector = report.get('sector_comparison', {})
        
        # 资金流向
        money_flow = report.get('money_flow', {})
        
        # 构建HTML
        html_content = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{stock_name} ({stock_code}) - 股票全方位体检报告</title>
    <script src="https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"></script>
    <style>
        :root {{
            --bg-primary: #0d1117;
            --bg-secondary: #161b22;
            --bg-tertiary: #21262d;
            --border-color: #30363d;
            --text-primary: #c9d1d9;
            --text-secondary: #8b949e;
            --accent-blue: #58a6ff;
            --accent-green: #3fb950;
            --accent-red: #f85149;
            --accent-yellow: #d29922;
            --accent-purple: #a371f7;
        }}
        
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans', Helvetica, Arial, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            padding: 20px;
        }}
        
        .container {{ max-width: 1400px; margin: 0 auto; }}
        
        /* 头部卡片 */
        .header-card {{
            background: linear-gradient(135deg, var(--bg-secondary) 0%, var(--bg-tertiary) 100%);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 20px;
        }}
        
        .header-top {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            flex-wrap: wrap;
            gap: 20px;
        }}
        
        .stock-info h1 {{
            font-size: 28px;
            font-weight: 600;
            margin-bottom: 8px;
        }}
        
        .stock-info .meta {{
            color: var(--text-secondary);
            font-size: 14px;
        }}
        
        .score-container {{
            display: flex;
            align-items: center;
            gap: 20px;
        }}
        
        .score-badge {{
            width: 100px;
            height: 100px;
            border-radius: 50%;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            font-weight: bold;
        }}
        
        .score-badge.high {{ background: linear-gradient(135deg, #238636 0%, #2ea043 100%); }}
        .score-badge.medium {{ background: linear-gradient(135deg, #9e6a03 0%, #d29922 100%); }}
        .score-badge.low {{ background: linear-gradient(135deg, #da3633 0%, #f85149 100%); }}
        
        .score-badge .number {{ font-size: 28px; }}
        .score-badge .label {{ font-size: 12px; opacity: 0.9; }}
        
        .action-tag {{
            padding: 8px 20px;
            border-radius: 20px;
            font-size: 16px;
            font-weight: 600;
        }}
        
        .action-tag.buy {{ background: var(--accent-green); color: #fff; }}
        .action-tag.sell {{ background: var(--accent-red); color: #fff; }}
        .action-tag.hold {{ background: var(--bg-tertiary); border: 1px solid var(--border-color); }}
        
        .recommendation {{
            margin-top: 16px;
            padding: 16px;
            background: var(--bg-primary);
            border-radius: 8px;
            border-left: 4px solid var(--accent-blue);
            white-space: pre-wrap;
            font-size: 14px;
        }}
        
        /* 网格布局 */
        .grid {{
            display: grid;
            gap: 20px;
        }}
        
        .grid-2 {{ grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); }}
        .grid-3 {{ grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); }}
        
        /* 卡片 */
        .card {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 20px;
        }}
        
        .card-title {{
            font-size: 16px;
            font-weight: 600;
            margin-bottom: 16px;
            padding-bottom: 12px;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        
        /* 信号列表 */
        .signal-list {{ list-style: none; }}
        .signal-list li {{
            padding: 8px 12px;
            margin: 6px 0;
            border-radius: 6px;
            font-size: 13px;
        }}
        
        .signal-list.buy li {{ background: rgba(63, 185, 80, 0.15); border-left: 3px solid var(--accent-green); }}
        .signal-list.sell li {{ background: rgba(248, 81, 73, 0.15); border-left: 3px solid var(--accent-red); }}
        .signal-list.warning li {{ background: rgba(210, 153, 34, 0.15); border-left: 3px solid var(--accent-yellow); }}
        
        /* 数据项 */
        .data-row {{
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid var(--border-color);
        }}
        
        .data-row:last-child {{ border-bottom: none; }}
        .data-row .label {{ color: var(--text-secondary); }}
        .data-row .value {{ font-weight: 500; }}
        .data-row .value.up {{ color: var(--accent-red); }}
        .data-row .value.down {{ color: var(--accent-green); }}
        .data-row .value.profit {{ color: var(--accent-green); }}
        .data-row .value.loss {{ color: var(--accent-red); }}
        
        /* 图表容器 */
        .chart-container {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
        }}
        
        .chart-container h3 {{
            margin-bottom: 16px;
            font-size: 16px;
        }}
        
        /* 形态标签 */
        .pattern-tag {{
            display: inline-block;
            padding: 4px 12px;
            margin: 4px;
            border-radius: 16px;
            font-size: 12px;
        }}
        
        .pattern-tag.bullish {{ background: rgba(63, 185, 80, 0.2); color: var(--accent-green); }}
        .pattern-tag.bearish {{ background: rgba(248, 81, 73, 0.2); color: var(--accent-red); }}
        .pattern-tag.neutral {{ background: rgba(139, 148, 158, 0.2); color: var(--text-secondary); }}
        
        /* 指标条 */
        .indicator-bar {{
            margin: 12px 0;
        }}
        
        .indicator-bar .header {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 6px;
            font-size: 13px;
        }}
        
        .indicator-bar .bar {{
            height: 8px;
            background: var(--bg-tertiary);
            border-radius: 4px;
            overflow: hidden;
        }}
        
        .indicator-bar .fill {{
            height: 100%;
            border-radius: 4px;
            transition: width 0.3s ease;
        }}
        
        /* 页脚 */
        .footer {{
            text-align: center;
            padding: 20px;
            color: var(--text-secondary);
            font-size: 12px;
            margin-top: 20px;
        }}
        
        /* 响应式 */
        @media (max-width: 768px) {{
            .header-top {{ flex-direction: column; }}
            .grid-2, .grid-3 {{ grid-template-columns: 1fr; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- 头部 -->
        <div class="header-card">
            <div class="header-top">
                <div class="stock-info">
                    <h1>{stock_name} ({stock_code})</h1>
                    <div class="meta">
                        行业: {basic.get('industry', 'N/A')} | 
                        最新价: ¥{basic.get('latest_price', 0):.2f} |
                        涨跌: {'+' if basic.get('pct_chg', 0) > 0 else ''}{basic.get('pct_chg', 0):.2f}% |
                        体检时间: {report.get('check_time', '')}
                    </div>
                </div>
                <div class="score-container">
                    <div class="score-badge {'high' if score >= 70 else 'medium' if score >= 50 else 'low'}">
                        <span class="number">{score:.0f}</span>
                        <span class="label">综合评分</span>
                    </div>
                    <span class="action-tag {'buy' if action == '买入' else 'sell' if action == '卖出' else 'hold'}">{action}</span>
                </div>
            </div>
            <div class="recommendation">{recommendation}</div>
        </div>
        
        <!-- 核心指标 -->
        <div class="grid grid-3" style="margin-bottom: 20px;">
            <!-- 交易信号 -->
            <div class="card">
                <div class="card-title">🎯 交易信号</div>
                {'<div><h4 style="color: var(--accent-green); margin-bottom: 8px;">买入信号</h4><ul class="signal-list buy">' + ''.join(f'<li>{s}</li>' for s in buy_signals[:5]) + '</ul></div>' if buy_signals else ''}
                {'<div style="margin-top: 12px;"><h4 style="color: var(--accent-red); margin-bottom: 8px;">卖出信号</h4><ul class="signal-list sell">' + ''.join(f'<li>{s}</li>' for s in sell_signals[:5]) + '</ul></div>' if sell_signals else ''}
                {'<div style="margin-top: 12px;"><h4 style="color: var(--accent-yellow); margin-bottom: 8px;">警告信号</h4><ul class="signal-list warning">' + ''.join(f'<li>{s}</li>' for s in warning_signals) + '</ul></div>' if warning_signals else ''}
            </div>
            
            <!-- 交易计划 -->
            <div class="card">
                <div class="card-title">📝 交易计划</div>
                <div class="data-row">
                    <span class="label">建议买入价</span>
                    <span class="value">¥{entry.get('ideal_price', 0):.2f}</span>
                </div>
                <div class="data-row">
                    <span class="label">止损位</span>
                    <span class="value loss">¥{exit_plan.get('stop_loss', 0):.2f} ({exit_plan.get('stop_loss_pct', 0):.1f}%)</span>
                </div>
                <div class="data-row">
                    <span class="label">止盈目标1</span>
                    <span class="value profit">¥{exit_plan.get('take_profit_1', 0):.2f}</span>
                </div>
                <div class="data-row">
                    <span class="label">止盈目标2</span>
                    <span class="value profit">¥{exit_plan.get('take_profit_2', 0):.2f}</span>
                </div>
                <div class="data-row">
                    <span class="label">建议仓位</span>
                    <span class="value">{position.get('suggested', 'N/A')}</span>
                </div>
                <div class="data-row">
                    <span class="label">风险收益比</span>
                    <span class="value">{position.get('risk_ratio', 'N/A')}</span>
                </div>
            </div>
            
            <!-- AI预测 -->
            <div class="card">
                <div class="card-title">🤖 AI模型预测</div>
                <div style="text-align: center; padding: 20px 0;">
                    <div style="font-size: 48px; font-weight: bold; color: {'var(--accent-green)' if model.get('probability', 0) > 0.6 else 'var(--accent-red)' if model.get('probability', 0) < 0.4 else 'var(--accent-yellow)'};">
                        {model.get('probability', 0) * 100:.1f}%
                    </div>
                    <div style="font-size: 18px; margin-top: 8px;">{model.get('signal', 'N/A')}</div>
                    <div style="color: var(--text-secondary); font-size: 12px; margin-top: 8px;">
                        置信度: {model.get('confidence', 'N/A')} | 
                        版本: {model.get('model_version', 'N/A')}
                    </div>
                </div>
            </div>
        </div>
        
        <!-- K线图 -->
        <div class="chart-container">
            <h3>📈 技术分析</h3>
            {chart_htmls.get('kline', '<div>图表加载中...</div>')}
        </div>
        
        <!-- 详细分析 -->
        <div class="grid grid-2">
            <!-- 技术指标 -->
            <div class="card">
                <div class="card-title">📊 技术指标</div>
                <div class="indicator-bar">
                    <div class="header">
                        <span>RSI(14)</span>
                        <span style="color: {'var(--accent-red)' if indicators.get('rsi', 0) > 70 else 'var(--accent-green)' if indicators.get('rsi', 0) < 30 else 'var(--accent-blue)'};">{indicators.get('rsi', 0):.1f} {'⚠️超买' if indicators.get('rsi', 0) > 70 else '✅超卖' if indicators.get('rsi', 0) < 30 else ''}</span>
                    </div>
                    <div class="bar">
                        <div class="fill" style="width: {min(indicators.get('rsi', 0), 100)}%; background: {'var(--accent-red)' if indicators.get('rsi', 0) > 70 else 'var(--accent-green)' if indicators.get('rsi', 0) < 30 else 'var(--accent-blue)'};"></div>
                    </div>
                </div>
                <div class="data-row">
                    <span class="label">均线排列</span>
                    <span class="value" style="color: {'var(--accent-green)' if '多头' in trend.get('alignment', '') else 'var(--accent-red)' if '空头' in trend.get('alignment', '') else 'var(--accent-yellow)'};">
                        {'✅' if '多头' in trend.get('alignment', '') else '❌' if '空头' in trend.get('alignment', '') else '⚡'} {trend.get('alignment', 'N/A')}
                    </span>
                </div>
                <div class="data-row">
                    <span class="label">MACD</span>
                    <span class="value" style="color: {'var(--accent-green)' if '多头' in str(indicators.get('macd', {}).get('signal', '')) or '金叉' in str(indicators.get('macd', {}).get('signal', '')) else 'var(--accent-red)' if '空头' in str(indicators.get('macd', {}).get('signal', '')) or '死叉' in str(indicators.get('macd', {}).get('signal', '')) else 'inherit'};">
                        {'✅' if '多头' in str(indicators.get('macd', {}).get('signal', '')) or '金叉' in str(indicators.get('macd', {}).get('signal', '')) else '❌' if '空头' in str(indicators.get('macd', {}).get('signal', '')) or '死叉' in str(indicators.get('macd', {}).get('signal', '')) else ''} {indicators.get('macd', {}).get('signal', 'N/A')}
                    </span>
                </div>
                <div class="data-row">
                    <span class="label">KDJ</span>
                    <span class="value" style="color: {'var(--accent-green)' if '多头' in str(indicators.get('kdj', {}).get('signal', '')) or '金叉' in str(indicators.get('kdj', {}).get('signal', '')) else 'var(--accent-red)' if '空头' in str(indicators.get('kdj', {}).get('signal', '')) or '死叉' in str(indicators.get('kdj', {}).get('signal', '')) else 'inherit'};">
                        {'✅' if '多头' in str(indicators.get('kdj', {}).get('signal', '')) or '金叉' in str(indicators.get('kdj', {}).get('signal', '')) else '❌' if '空头' in str(indicators.get('kdj', {}).get('signal', '')) or '死叉' in str(indicators.get('kdj', {}).get('signal', '')) else ''} {indicators.get('kdj', {}).get('signal', 'N/A')}
                    </span>
                </div>
                <div class="data-row">
                    <span class="label">布林带</span>
                    <span class="value" style="color: {'var(--accent-red)' if '超买' in str(indicators.get('bollinger', {}).get('signal', '')) or '上轨' in str(indicators.get('bollinger', {}).get('signal', '')) else 'var(--accent-green)' if '超卖' in str(indicators.get('bollinger', {}).get('signal', '')) or '下轨' in str(indicators.get('bollinger', {}).get('signal', '')) else 'inherit'};">
                        {'⚠️' if '超买' in str(indicators.get('bollinger', {}).get('signal', '')) or '上轨' in str(indicators.get('bollinger', {}).get('signal', '')) else '✅' if '超卖' in str(indicators.get('bollinger', {}).get('signal', '')) or '下轨' in str(indicators.get('bollinger', {}).get('signal', '')) else ''} {indicators.get('bollinger', {}).get('signal', 'N/A')}
                    </span>
                </div>
                <div class="data-row">
                    <span class="label">量价配合</span>
                    <span class="value" style="color: {'var(--accent-green)' if '量增价涨' in str(tech.get('volume_analysis', {}).get('price_volume', '')) or '健康' in str(tech.get('volume_analysis', {}).get('price_volume', '')) else 'var(--accent-red)' if '量增价跌' in str(tech.get('volume_analysis', {}).get('price_volume', '')) or '背离' in str(tech.get('volume_analysis', {}).get('price_volume', '')) else 'var(--accent-yellow)'};">
                        {'✅' if '量增价涨' in str(tech.get('volume_analysis', {}).get('price_volume', '')) or '健康' in str(tech.get('volume_analysis', {}).get('price_volume', '')) else '❌' if '量增价跌' in str(tech.get('volume_analysis', {}).get('price_volume', '')) or '背离' in str(tech.get('volume_analysis', {}).get('price_volume', '')) else '⚡'} {tech.get('volume_analysis', {}).get('price_volume', 'N/A')}
                    </span>
                </div>
            </div>
            
            <!-- 风险评估 -->
            <div class="card">
                <div class="card-title">⚠️ 风险评估</div>
                <div class="data-row">
                    <span class="label">年化波动率</span>
                    <span class="value" style="color: {'var(--accent-green)' if risk.get('volatility', 0) < 30 else 'var(--accent-red)' if risk.get('volatility', 0) > 50 else 'var(--accent-yellow)'};">
                        {'✅' if risk.get('volatility', 0) < 30 else '⚠️' if risk.get('volatility', 0) > 50 else '⚡'} {risk.get('volatility', 0):.1f}% ({risk.get('volatility_level', 'N/A')})
                    </span>
                </div>
                <div class="data-row">
                    <span class="label">最大回撤</span>
                    <span class="value" style="color: {'var(--accent-green)' if abs(risk.get('max_drawdown', 0)) < 15 else 'var(--accent-red)' if abs(risk.get('max_drawdown', 0)) > 30 else 'var(--accent-yellow)'};">
                        {'✅' if abs(risk.get('max_drawdown', 0)) < 15 else '❌' if abs(risk.get('max_drawdown', 0)) > 30 else '⚠️'} {risk.get('max_drawdown', 0):.1f}%
                    </span>
                </div>
                <div class="data-row">
                    <span class="label">夏普比率</span>
                    <span class="value" style="color: {'var(--accent-green)' if risk.get('sharpe_ratio', 0) > 1 else 'var(--accent-red)' if risk.get('sharpe_ratio', 0) < 0 else 'var(--accent-yellow)'};">
                        {'✅' if risk.get('sharpe_ratio', 0) > 1 else '❌' if risk.get('sharpe_ratio', 0) < 0 else '⚡'} {risk.get('sharpe_ratio', 0):.2f} ({risk.get('sharpe_level', 'N/A')})
                    </span>
                </div>
                <div class="data-row">
                    <span class="label">VaR(95%)</span>
                    <span class="value" style="color: {'var(--accent-green)' if abs(risk.get('var_95', 0)) < 3 else 'var(--accent-red)' if abs(risk.get('var_95', 0)) > 5 else 'var(--accent-yellow)'};">
                        {'✅' if abs(risk.get('var_95', 0)) < 3 else '❌' if abs(risk.get('var_95', 0)) > 5 else '⚠️'} {risk.get('var_95', 0):.2f}%
                    </span>
                </div>
                <div class="data-row">
                    <span class="label">综合风险</span>
                    <span class="value" style="color: {'var(--accent-green)' if '低' in risk.get('overall_risk', '') else 'var(--accent-red)' if '高' in risk.get('overall_risk', '') else 'var(--accent-yellow)'};">
                        {'✅' if '低' in risk.get('overall_risk', '') else '❌' if '高' in risk.get('overall_risk', '') else '⚡'} {risk.get('overall_risk', 'N/A')}
                    </span>
                </div>
            </div>
        </div>
        
        <div class="grid grid-2" style="margin-top: 20px;">
            <!-- 行业对比 -->
            <div class="card">
                <div class="card-title">🏭 行业对比</div>
                {f'''
                <div class="data-row">
                    <span class="label">所属行业</span>
                    <span class="value">{sector.get('industry', 'N/A')}</span>
                </div>
                <div class="data-row">
                    <span class="label">行业排名</span>
                    <span class="value">{sector.get('rank', 'N/A')}</span>
                </div>
                <div class="data-row">
                    <span class="label">相对强度</span>
                    <span class="value">{sector.get('relative_strength', 'N/A')}</span>
                </div>
                <div class="data-row">
                    <span class="label">个股20日涨幅</span>
                    <span class="value {'up' if sector.get('20d_returns', 0) > 0 else 'down'}">{sector.get('20d_returns', 0):.2f}%</span>
                </div>
                <div class="data-row">
                    <span class="label">行业平均涨幅</span>
                    <span class="value">{sector.get('industry_avg', 0):.2f}%</span>
                </div>
                ''' if sector.get('rank') != '未知' else '<div style="color: var(--text-secondary);">行业数据暂不可用</div>'}
            </div>
            
            <!-- 资金流向 -->
            <div class="card">
                <div class="card-title">💰 资金流向</div>
                <div class="data-row">
                    <span class="label">资金趋势</span>
                    <span class="value" style="color: {'var(--accent-green)' if '流入' in money_flow.get('trend', '') else 'var(--accent-red)' if '流出' in money_flow.get('trend', '') else 'inherit'};">
                        {money_flow.get('trend', 'N/A')}
                    </span>
                </div>
                <div class="data-row">
                    <span class="label">净流入比</span>
                    <span class="value {'up' if money_flow.get('net_flow_ratio', 0) > 0 else 'down'}">{money_flow.get('net_flow_ratio', 0):.1f}%</span>
                </div>
                <div class="data-row">
                    <span class="label">近5日趋势</span>
                    <span class="value">{money_flow.get('recent_5d_trend', 'N/A')}</span>
                </div>
            </div>
        </div>
        
        <!-- K线形态 -->
        {'<div class="card" style="margin-top: 20px;"><div class="card-title">🕯️ K线形态</div><div>' + ''.join(f'<span class="pattern-tag {"bullish" if "涨" in p.get("signal", "") or "底" in p.get("signal", "") else "bearish" if "跌" in p.get("signal", "") or "顶" in p.get("signal", "") else "neutral"}">{p.get("name", "")} - {p.get("signal", "")}</span>' for p in all_patterns) + '</div></div>' if all_patterns else ''}
        
        <!-- 页脚 -->
        <div class="footer">
            <p>⚠️ 风险提示: 本报告仅供参考，不构成投资建议。投资有风险，入市需谨慎。</p>
            <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    </div>
</body>
</html>'''
        
        return html_content
    
    # ========== Plotly 备选实现 ==========
    
    def _create_plotly_kline(self, stock_code: str, report: dict, days: int = 120):
        """使用 Plotly 创建K线图（备选方案）"""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # 获取数据
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=days*2)).strftime('%Y%m%d')
        df = self.dm.get_daily_data(stock_code, start_date, end_date)
        
        if df is None or df.empty:
            return go.Figure()
        
        df = df.tail(days).reset_index(drop=True)
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        
        # 创建子图
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.6, 0.2, 0.2]
        )
        
        # K线
        fig.add_trace(
            go.Candlestick(
                x=df['trade_date'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                increasing_line_color='#ec0000',
                decreasing_line_color='#00da3c',
            ),
            row=1, col=1
        )
        
        # 成交量
        colors = ['#ec0000' if row['close'] >= row['open'] else '#00da3c' for _, row in df.iterrows()]
        fig.add_trace(
            go.Bar(x=df['trade_date'], y=df['vol'], marker_color=colors),
            row=2, col=1
        )
        
        # MACD
        macd = self._calculate_macd(df['close'])
        fig.add_trace(go.Bar(x=df['trade_date'], y=macd['macd'], name='MACD'), row=3, col=1)
        fig.add_trace(go.Scatter(x=df['trade_date'], y=macd['dif'], name='DIF', line=dict(width=1)), row=3, col=1)
        fig.add_trace(go.Scatter(x=df['trade_date'], y=macd['dea'], name='DEA', line=dict(width=1)), row=3, col=1)
        
        fig.update_layout(
            height=800,
            xaxis_rangeslider_visible=False,
            template='plotly_dark',
        )
        
        return fig
    
    def _create_plotly_heatmap(self, report: dict):
        """Plotly 仪表盘"""
        return None
    
    def _create_plotly_sector(self, report: dict):
        """Plotly 行业对比"""
        return None
    
    def _create_plotly_money_flow(self, report: dict):
        """Plotly 资金流向"""
        return None
    
    def _create_plotly_trading_plan(self, report: dict):
        """Plotly 交易计划"""
        return None


if __name__ == '__main__':
    print(f"PyEcharts 可用: {HAS_PYECHARTS}")
    print(f"Plotly 可用: {HAS_PLOTLY}")
    visualizer = StockChartVisualizer()
    print("Stock chart visualizer ready")
