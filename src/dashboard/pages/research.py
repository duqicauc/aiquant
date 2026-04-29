"""
股票研究页 (Stock Research Page)
交互式 K 线图 + 技术指标 + 全方位诊断
数据直连 DataManager，充分利用本地缓存
"""
import sys
from datetime import datetime, timedelta
from pathlib import Path

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html
from dash.dependencies import Input, Output, State
from plotly.subplots import make_subplots

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.data_manager import DataManager

# New analysis modules
from src.analysis.technical_indicators import (
    calculate_adx_dmi,
    calculate_ad_line,
    calculate_atr_channel,
    calculate_cmf,
    calculate_ichimoku,
    calculate_mfi,
    calculate_pvo,
    calculate_sar,
    calculate_supertrend,
    calculate_volume_profile,
    calculate_vwap,
    detect_fractals,
    detect_harmonic_patterns,
)
from src.analysis.mtfa import analyze_resonance
from src.analysis.moneyflow_analysis import analyze_full_moneyflow

from src.dashboard.app import app

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _calc_ma(df):
    """Calculate moving averages."""
    df = df.copy()
    for period in [5, 10, 20, 60, 120, 233]:
        if len(df) >= period:
            df[f"ma{period}"] = df["close"].rolling(period).mean()
    return df


def _calc_rsi(prices, period=14):
    """Calculate RSI."""
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _calc_macd(prices):
    """Calculate MACD."""
    s = pd.Series(prices)
    ema12 = s.ewm(span=12, adjust=False).mean()
    ema26 = s.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    macd = (dif - dea) * 2
    return dif.iloc[-1], dea.iloc[-1], macd.iloc[-1]


def _calc_kdj(high, low, close):
    """Calculate KDJ."""
    n = 9
    lowest = pd.Series(low).rolling(n).min().iloc[-1]
    highest = pd.Series(high).rolling(n).max().iloc[-1]
    rsv = (close[-1] - lowest) / (highest - lowest) * 100 if (highest - lowest) != 0 else 50
    k = rsv * 1/3 + 50 * 2/3
    d = k * 1/3 + 50 * 2/3
    j = 3 * k - 2 * d
    return k, d, j


def _calc_bollinger(prices, period=20):
    """Calculate Bollinger Bands."""
    arr = np.array(prices[-period:])
    ma = np.mean(arr)
    std = np.std(arr)
    return ma + 2*std, ma, ma - 2*std


def _resample_to_monthly(df):
    """Resample daily data to monthly OHLC."""
    df = df.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.set_index("trade_date").sort_index()
    monthly = df.resample("ME").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "vol": "sum",
    }).reset_index()
    monthly["trade_date"] = monthly["trade_date"].dt.strftime("%Y%m%d")
    return monthly


def _build_kline_figure(df, ts_code, period="daily"):
    """Build Plotly candlestick chart with MAs and volume."""
    df = _calc_ma(df)
    # Show last 120 bars regardless of period
    df = df.tail(120)

    period_label = {"daily": "日线", "weekly": "周线", "monthly": "月线"}.get(period, "K线")
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=(f"{ts_code} {period_label}", "成交量", "MACD"),
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df["trade_date"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="K线",
            increasing_line_color="#f85149",
            decreasing_line_color="#3fb950",
        ),
        row=1, col=1,
    )

    # Moving averages
    colors = {"ma5": "#FF6B6B", "ma10": "#4ECDC4", "ma20": "#45B7D1",
              "ma60": "#FFA07A", "ma120": "#9B59B6", "ma233": "#E91E63"}
    for ma_name, color in colors.items():
        if ma_name in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["trade_date"],
                    y=df[ma_name],
                    mode="lines",
                    name=ma_name.upper(),
                    line=dict(color=color, width=1),
                    hovertemplate="%{y:.2f}",
                ),
                row=1, col=1,
            )

    # Volume
    colors_vol = ["#f85149" if c >= o else "#3fb950"
                  for c, o in zip(df["close"], df["open"])]
    fig.add_trace(
        go.Bar(
            x=df["trade_date"],
            y=df["vol"],
            name="成交量",
            marker_color=colors_vol,
            showlegend=False,
        ),
        row=2, col=1,
    )

    # MACD
    close_vals = df["close"].values
    if len(close_vals) >= 26:
        s = pd.Series(close_vals)
        ema12 = s.ewm(span=12, adjust=False).mean()
        ema26 = s.ewm(span=26, adjust=False).mean()
        dif = ema12 - ema26
        dea = dif.ewm(span=9, adjust=False).mean()
        macd = (dif - dea) * 2

        fig.add_trace(
            go.Scatter(x=df["trade_date"], y=dif, mode="lines", name="DIF",
                       line=dict(color="#2196F3", width=1), showlegend=False),
            row=3, col=1,
        )
        fig.add_trace(
            go.Scatter(x=df["trade_date"], y=dea, mode="lines", name="DEA",
                       line=dict(color="#FF9800", width=1), showlegend=False),
            row=3, col=1,
        )
        colors_macd = ["#f85149" if v >= 0 else "#3fb950" for v in macd]
        fig.add_trace(
            go.Bar(x=df["trade_date"], y=macd, name="MACD",
                   marker_color=colors_macd, showlegend=False),
            row=3, col=1,
        )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#c9d1d9",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10)),
        margin=dict(l=40, r=40, t=60, b=40),
        hovermode="x unified",
        height=600,
    )
    fig.update_xaxes(showgrid=False, tickfont=dict(size=10))
    fig.update_yaxes(showgrid=True, gridcolor="#30363d", tickfont=dict(size=10))
    fig.update_xaxes(rangeslider_visible=False)

    return fig


def _indicator_card(title, value, signal, signal_color_map=None):
    """Technical indicator card."""
    if signal_color_map is None:
        signal_color_map = {
            "超买": "#f85149", "严重超买": "#f85149",
            "超卖": "#3fb950", "严重超卖": "#3fb950",
            "金叉": "#3fb950", "死叉": "#f85149",
            "多头": "#3fb950", "空头": "#f85149",
        }
    color = signal_color_map.get(signal, "#8b949e")
    return dbc.Card(
        dbc.CardBody(
            [
                html.H6(title, className="text-muted", style={"fontSize": "0.8rem"}),
                html.H4(value, style={"color": "#c9d1d9", "fontWeight": "bold"}),
                html.Span(signal, style={"color": color, "fontSize": "0.85rem"}),
            ]
        ),
        style={"backgroundColor": "#161b22", "border": "1px solid #30363d"},
    )


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

layout = html.Div(
    [
        dcc.Store(id="selected-period", data="daily"),
        html.H1("🔍 股票研究", className="mb-3", style={"color": "#c9d1d9", "fontWeight": "bold"}),

        # Input row
        dbc.Row(
            [
                dbc.Col(
                    dbc.Input(
                        id="stock-code-input",
                        placeholder="输入股票代码，如 000001.SZ",
                        value="000001.SZ",
                        style={"backgroundColor": "#21262d", "color": "#c9d1d9", "borderColor": "#30363d"},
                    ),
                    width=3,
                ),
                dbc.Col(
                    dbc.Input(
                        id="stock-days-input",
                        type="number",
                        value=120,
                        min=30,
                        max=500,
                        style={"backgroundColor": "#21262d", "color": "#c9d1d9", "borderColor": "#30363d"},
                    ),
                    width=1,
                ),
                dbc.Col(
                    dbc.ButtonGroup(
                        [
                            dbc.Button("日线", id="period-daily", color="primary", outline=False, size="sm"),
                            dbc.Button("周线", id="period-weekly", color="secondary", outline=True, size="sm"),
                            dbc.Button("月线", id="period-monthly", color="secondary", outline=True, size="sm"),
                        ],
                        id="period-selector",
                        className="me-2",
                    ),
                    width="auto",
                ),
                dbc.Col(
                    dbc.Button("🔍 分析", id="analyze-btn", color="primary", className="me-2"),
                    width="auto",
                ),
                dbc.Col(
                    html.Div(
                        [
                            dbc.Button("贵州茅台", id="ex-600519", size="sm", color="secondary", className="me-1"),
                            dbc.Button("宁德时代", id="ex-300750", size="sm", color="secondary", className="me-1"),
                            dbc.Button("比亚迪", id="ex-002594", size="sm", color="secondary", className="me-1"),
                        ]
                    ),
                    width="auto",
                ),
            ],
            className="mb-3 g-2 align-items-center",
        ),

        # Stock info header
        html.Div(id="stock-info-header", className="mb-3"),

        # Indicator cards
        dbc.Row(
            [
                dbc.Col(html.Div(id="indicator-rsi"), width=3),
                dbc.Col(html.Div(id="indicator-macd"), width=3),
                dbc.Col(html.Div(id="indicator-kdj"), width=3),
                dbc.Col(html.Div(id="indicator-boll"), width=3),
            ],
            className="mb-3 g-3",
        ),

        # Volume-Price Analysis cards
        html.H5("📊 量价分析", className="mb-2", style={"color": "#8b949e", "fontWeight": "bold"}),
        dbc.Row(
            [
                dbc.Col(html.Div(id="indicator-vwap"), width=2),
                dbc.Col(html.Div(id="indicator-cmf"), width=2),
                dbc.Col(html.Div(id="indicator-mfi"), width=2),
                dbc.Col(html.Div(id="indicator-pvo"), width=2),
                dbc.Col(html.Div(id="indicator-ad"), width=2),
                dbc.Col(html.Div(id="indicator-vol-profile"), width=2),
            ],
            className="mb-3 g-3",
        ),

        # Advanced Trend cards
        html.H5("🎯 高级趋势", className="mb-2", style={"color": "#8b949e", "fontWeight": "bold"}),
        dbc.Row(
            [
                dbc.Col(html.Div(id="indicator-adx"), width=3),
                dbc.Col(html.Div(id="indicator-supertrend"), width=3),
                dbc.Col(html.Div(id="indicator-ichimoku"), width=3),
                dbc.Col(html.Div(id="indicator-sar"), width=3),
            ],
            className="mb-3 g-3",
        ),

        # K-line chart
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            dcc.Graph(
                                id="kline-chart",
                                config={"displayModeBar": True, "scrollZoom": True},
                                style={"height": "600px"},
                            ),
                            style={"backgroundColor": "#161b22", "padding": "0"},
                        ),
                        style={"border": "1px solid #30363d"},
                    ),
                    width=12,
                ),
            ],
            className="mb-3 g-3",
        ),

        # Multi-timeframe resonance panel
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("🔔 多周期共振", style={"backgroundColor": "#21262d", "color": "#c9d1d9", "fontWeight": "bold"}),
                            dbc.CardBody(html.Div(id="mtfa-panel"), style={"backgroundColor": "#161b22", "color": "#c9d1d9"}),
                        ],
                        style={"border": "1px solid #30363d"},
                    ),
                    width=6,
                ),
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("💰 主力资金", style={"backgroundColor": "#21262d", "color": "#c9d1d9", "fontWeight": "bold"}),
                            dbc.CardBody(html.Div(id="moneyflow-panel"), style={"backgroundColor": "#161b22", "color": "#c9d1d9"}),
                        ],
                        style={"border": "1px solid #30363d"},
                    ),
                    width=6,
                ),
            ],
            className="mb-3 g-3",
        ),

        # Volume profile heatmap
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("🌡️ 成交量分布（筹码热力图）", style={"backgroundColor": "#21262d", "color": "#c9d1d9", "fontWeight": "bold"}),
                            dbc.CardBody(
                                dcc.Graph(id="volume-profile-chart", config={"displayModeBar": False}, style={"height": "300px"}),
                                style={"backgroundColor": "#161b22", "padding": "0"},
                            ),
                        ],
                        style={"border": "1px solid #30363d"},
                    ),
                    width=6,
                ),
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("🎨 形态识别", style={"backgroundColor": "#21262d", "color": "#c9d1d9", "fontWeight": "bold"}),
                            dbc.CardBody(html.Div(id="pattern-panel"), style={"backgroundColor": "#161b22", "color": "#c9d1d9"}),
                        ],
                        style={"border": "1px solid #30363d"},
                    ),
                    width=6,
                ),
            ],
            className="mb-3 g-3",
        ),

        # Diagnosis report
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("📋 股票诊断报告", style={"backgroundColor": "#21262d", "color": "#c9d1d9", "fontWeight": "bold"}),
                            dbc.CardBody(html.Div(id="diagnosis-report"), style={"backgroundColor": "#161b22", "color": "#c9d1d9"}),
                        ],
                        style={"border": "1px solid #30363d"},
                    ),
                    width=12,
                ),
            ],
            className="g-3",
        ),
    ]
)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

@app.callback(
    [
        Output("stock-code-input", "value"),
        Output("analyze-btn", "n_clicks"),
    ],
    [
        Input("ex-600519", "n_clicks"),
        Input("ex-300750", "n_clicks"),
        Input("ex-002594", "n_clicks"),
    ],
    prevent_initial_call=True,
)
def on_example_click(n1, n2, n3):
    """Handle example stock buttons."""
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update
    btn_id = ctx.triggered[0]["prop_id"].split(".")[0]
    code_map = {
        "ex-600519": "600519.SH",
        "ex-300750": "300750.SZ",
        "ex-002594": "002594.SZ",
    }
    return code_map.get(btn_id, "000001.SZ"), 1


@app.callback(
    [
        Output("stock-info-header", "children"),
        Output("indicator-rsi", "children"),
        Output("indicator-macd", "children"),
        Output("indicator-kdj", "children"),
        Output("indicator-boll", "children"),
        Output("kline-chart", "figure"),
        Output("diagnosis-report", "children"),
        Output("period-daily", "color"),
        Output("period-daily", "outline"),
        Output("period-weekly", "color"),
        Output("period-weekly", "outline"),
        Output("period-monthly", "color"),
        Output("period-monthly", "outline"),
        # Volume-Price
        Output("indicator-vwap", "children"),
        Output("indicator-cmf", "children"),
        Output("indicator-mfi", "children"),
        Output("indicator-pvo", "children"),
        Output("indicator-ad", "children"),
        Output("indicator-vol-profile", "children"),
        # Advanced Trend
        Output("indicator-adx", "children"),
        Output("indicator-supertrend", "children"),
        Output("indicator-ichimoku", "children"),
        Output("indicator-sar", "children"),
        # Panels
        Output("mtfa-panel", "children"),
        Output("moneyflow-panel", "children"),
        Output("volume-profile-chart", "figure"),
        Output("pattern-panel", "children"),
    ],
    Input("analyze-btn", "n_clicks"),
    [
        State("stock-code-input", "value"),
        State("stock-days-input", "value"),
        State("period-daily", "color"),
        State("period-weekly", "color"),
        State("period-monthly", "color"),
    ],
    prevent_initial_call=True,
)
def on_analyze(n_clicks, ts_code, days, daily_color, weekly_color, monthly_color):
    """Main analysis callback — enhanced with volume-price, advanced trend, MTFA, moneyflow."""
    period = "daily"
    if weekly_color == "primary":
        period = "weekly"
    elif monthly_color == "primary":
        period = "monthly"

    if not ts_code:
        return [html.P("请输入股票代码", className="text-warning")] + [dash.no_update] * 26

    def period_btn_states(p):
        return ("primary", False) if p == period else ("secondary", True)

    # Default no-update placeholders for new outputs (indices 13-26)
    nu = dash.no_update

    try:
        dm = DataManager()
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=days * 2)).strftime("%Y%m%d")

        if period == "weekly":
            df = dm.get_weekly_data(ts_code, start_date, end_date)
        else:
            df = dm.get_daily_data(ts_code, start_date, end_date)
            if period == "monthly":
                df = _resample_to_monthly(df)
        if df is None or df.empty:
            return [html.P(f"❌ 无法获取 {ts_code} 的数据", className="text-danger")] + [nu] * 6 + [dash.no_update] * 20

        df = df.sort_values("trade_date").reset_index(drop=True)
        close_vals = df["close"].values
        high_vals = df["high"].values
        low_vals = df["low"].values

        # Stock info header
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        pct_chg = (latest["close"] - prev["close"]) / prev["close"] * 100
        color = "#f85149" if pct_chg >= 0 else "#3fb950"

        info_header = html.Div(
            [
                html.H3(ts_code, style={"color": "#c9d1d9", "display": "inline-block", "marginRight": "1rem"}),
                html.Span(f"¥{latest['close']:.2f}", style={"color": color, "fontSize": "1.5rem", "fontWeight": "bold"}),
                html.Span(f" ({pct_chg:+.2f}%)", style={"color": color, "fontSize": "1.2rem", "marginLeft": "0.5rem"}),
                html.Span(f" 成交量: {latest.get('vol', 0)/1e4:.0f}万手", style={"color": "#8b949e", "marginLeft": "1rem"}),
            ]
        )

        # Basic indicators
        rsi_val = _calc_rsi(close_vals)
        rsi_card = _indicator_card("RSI(14)", f"{rsi_val:.1f}", "超买" if rsi_val > 70 else "超卖" if rsi_val < 30 else "正常")

        dif, dea, macd_val = _calc_macd(close_vals)
        macd_card = _indicator_card("MACD", f"{macd_val:.3f}", "金叉" if macd_val > 0 else "死叉")

        k, d, j = _calc_kdj(high_vals, low_vals, close_vals)
        kdj_card = _indicator_card("KDJ", f"K:{k:.1f} D:{d:.1f}", "超买" if k > 80 else "超卖" if k < 20 else "正常")

        upper, middle, lower = _calc_bollinger(close_vals)
        boll_card = _indicator_card("布林带", f"¥{close_vals[-1]:.2f}", "突破上轨" if close_vals[-1] > upper else "跌破下轨" if close_vals[-1] < lower else "中轨附近")

        # K-line chart
        fig = _build_kline_figure(df, ts_code, period)

        # Diagnosis report
        try:
            from src.analysis.stock_health_checker import StockHealthChecker
            checker = StockHealthChecker()
            report = checker.check_stock(ts_code, days)

            if "error" in report:
                diagnosis = html.P(f"诊断失败: {report['error']}", className="text-danger")
            else:
                score = report.get("overall_score", 0)
                score_color = "#3fb950" if score >= 70 else "#d29922" if score >= 50 else "#f85149"
                rec = report.get("recommendation", "")
                signals = report.get("trading_signals", {})
                action = signals.get("action", "观望")
                action_color = "#3fb950" if action == "买入" else "#f85149" if action == "卖出" else "#d29922"

                diagnosis = html.Div(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    html.Div([html.H1(f"{score:.0f}", style={"color": score_color, "fontWeight": "bold"}), html.P("综合评分", className="text-muted")], style={"textAlign": "center"}),
                                    width=2,
                                ),
                                dbc.Col(
                                    html.Div([html.H5(rec, style={"color": "#c9d1d9"}), html.H4(f"操作建议: {action}", style={"color": action_color}), html.P(f"牛股概率: {report.get('model_prediction', {}).get('probability', 0)*100:.1f}%", className="text-muted")]),
                                    width=10,
                                ),
                            ],
                            className="mb-3",
                        ),
                        html.Hr(style={"borderColor": "#30363d"}),
                        dbc.Row(
                            [
                                dbc.Col([html.H6("买入信号", className="text-success"), html.P("\n".join(signals.get("buy_signals", [])[:5]), style={"whiteSpace": "pre-line"})], width=4),
                                dbc.Col([html.H6("卖出/警告", className="text-danger"), html.P("\n".join(signals.get("sell_signals", [])[:3] + signals.get("warning_signals", [])[:3]), style={"whiteSpace": "pre-line"})], width=4),
                                dbc.Col([html.H6("风险评估", className="text-warning"), html.P(f"波动率: {report.get('risk_assessment', {}).get('volatility', 0):.1f}%\n最大回撤: {report.get('risk_assessment', {}).get('max_drawdown', 0):.1f}%\n夏普比率: {report.get('risk_assessment', {}).get('sharpe_ratio', 0):.2f}", style={"whiteSpace": "pre-line"})], width=4),
                            ]
                        ),
                    ]
                )
        except Exception as e:
            diagnosis = html.P(f"诊断模块加载失败: {str(e)}", className="text-warning")

        # --- Volume-Price Indicators ---
        try:
            vwap = calculate_vwap(df)
            vwap_card = _indicator_card("VWAP", f"¥{vwap['value']:.2f}", vwap['signal'])
        except Exception:
            vwap_card = _indicator_card("VWAP", "—", "计算失败")

        try:
            cmf = calculate_cmf(df)
            cmf_card = _indicator_card("CMF", f"{cmf['value']:.3f}", cmf['signal'])
        except Exception:
            cmf_card = _indicator_card("CMF", "—", "计算失败")

        try:
            mfi = calculate_mfi(df)
            mfi_card = _indicator_card("MFI", f"{mfi['value']:.1f}", mfi['signal'])
        except Exception:
            mfi_card = _indicator_card("MFI", "—", "计算失败")

        try:
            pvo = calculate_pvo(df)
            pvo_card = _indicator_card("PVO", f"{pvo['value']:.1f}", pvo['signal'])
        except Exception:
            pvo_card = _indicator_card("PVO", "—", "计算失败")

        try:
            ad = calculate_ad_line(df)
            ad_card = _indicator_card("A/D Line", f"{ad['value']:.0f}", ad['signal'])
        except Exception:
            ad_card = _indicator_card("A/D Line", "—", "计算失败")

        try:
            vp = calculate_volume_profile(df)
            vp_card = _indicator_card("筹码分布", f"POC ¥{vp['detail'].get('poc', 0):.2f}", vp['signal'])
        except Exception:
            vp_card = _indicator_card("筹码分布", "—", "计算失败")

        # --- Advanced Trend Indicators ---
        try:
            adx = calculate_adx_dmi(df)
            adx_card = _indicator_card("ADX/DMI", f"{adx['detail']['adx']:.1f}", adx['signal'])
        except Exception:
            adx_card = _indicator_card("ADX/DMI", "—", "计算失败")

        try:
            st = calculate_supertrend(df)
            st_card = _indicator_card("SuperTrend", f"¥{st['value']:.2f}", st['signal'])
        except Exception:
            st_card = _indicator_card("SuperTrend", "—", "计算失败")

        try:
            ichi = calculate_ichimoku(df)
            ichi_card = _indicator_card("一目均衡", ichi['detail']['tk_cross'], ichi['signal'])
        except Exception:
            ichi_card = _indicator_card("一目均衡", "—", "计算失败")

        try:
            sar = calculate_sar(df)
            sar_card = _indicator_card("SAR", f"¥{sar['value']:.2f}", sar['signal'])
        except Exception:
            sar_card = _indicator_card("SAR", "—", "计算失败")

        # --- MTFA Panel ---
        try:
            # Fetch weekly and monthly data for resonance
            df_weekly = dm.get_weekly_data(ts_code, start_date, end_date) if period == "daily" else None
            df_monthly = None
            if period == "daily":
                df_monthly_raw = dm.get_daily_data(ts_code, (datetime.now() - timedelta(days=days * 10)).strftime("%Y%m%d"), end_date)
                if df_monthly_raw is not None and not df_monthly_raw.empty:
                    df_monthly = _resample_to_monthly(df_monthly_raw)
            mtfa = analyze_resonance(df, df_weekly, df_monthly)
            mtfa_panel = _build_mtfa_panel(mtfa)
        except Exception as e:
            mtfa_panel = html.P(f"共振分析失败: {str(e)}", className="text-warning")

        # --- Moneyflow Panel ---
        try:
            mf = analyze_full_moneyflow(ts_code, days=10)
            mf_panel = _build_moneyflow_panel(mf)
        except Exception:
            mf_panel = _build_moneyflow_panel(None)

        # --- Volume Profile Chart ---
        try:
            vp_fig = _build_volume_profile_figure(df)
        except Exception:
            vp_fig = go.Figure().update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")

        # --- Pattern Panel ---
        try:
            harmonic = detect_harmonic_patterns(df)
            fractals = detect_fractals(df)
            pattern_panel = _build_pattern_panel(harmonic, fractals)
        except Exception:
            pattern_panel = html.P("形态识别失败", className="text-warning")

        dc, do = period_btn_states("daily")
        wc, wo = period_btn_states("weekly")
        mc, mo = period_btn_states("monthly")

        return (
            info_header, rsi_card, macd_card, kdj_card, boll_card, fig, diagnosis,
            dc, do, wc, wo, mc, mo,
            vwap_card, cmf_card, mfi_card, pvo_card, ad_card, vp_card,
            adx_card, st_card, ichi_card, sar_card,
            mtfa_panel, mf_panel, vp_fig, pattern_panel,
        )

    except Exception as e:
        dc, do = period_btn_states("daily")
        wc, wo = period_btn_states("weekly")
        mc, mo = period_btn_states("monthly")
        return (
            html.P(f"❌ 分析出错: {str(e)}", className="text-danger"),
            nu, nu, nu, nu, go.Figure(), html.P("分析出错"),
            dc, do, wc, wo, mc, mo,
            nu, nu, nu, nu, nu, nu, nu, nu, nu, nu,
            nu, nu, go.Figure(), html.P("分析出错"),
        )


# ---------------------------------------------------------------------------
# New helper: Volume Profile chart
# ---------------------------------------------------------------------------

def _build_volume_profile_figure(df):
    """Build horizontal volume profile heatmap."""
    from src.analysis.technical_indicators import calculate_volume_profile
    vp = calculate_volume_profile(df)
    detail = vp.get("detail", {})
    if not detail or not detail.get("bin_centers"):
        return go.Figure().update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            annotations=[{"text": "数据不足", "showarrow": False, "font": {"color": "#8b949e"}}],
        )

    centers = detail["bin_centers"]
    volumes = detail["volumes"]
    poc = detail.get("poc", 0)
    va_low = detail.get("value_area_low", 0)
    va_high = detail.get("value_area_high", 0)

    fig = go.Figure()
    colors = ["#3fb950" if va_low <= c <= va_high else "#21262d" for c in centers]
    bar_colors = [f"rgba(63,185,80,{min(1,v/max(volumes)*3+0.2)})" if va_low <= c <= va_high
                  else f"rgba(139,148,158,{min(1,v/max(volumes)*2+0.1)})"
                  for c, v in zip(centers, volumes)]

    fig.add_trace(go.Bar(
        y=[f"{c:.2f}" for c in centers],
        x=volumes,
        orientation="h",
        marker_color=bar_colors,
        hovertemplate="价格: %{y}<br>成交量: %{x:.0f}<extra></extra>",
        showlegend=False,
    ))

    # Add POC line
    poc_idx = min(range(len(centers)), key=lambda i: abs(centers[i] - poc))
    fig.add_hline(y=poc_idx, line_dash="dash", line_color="#d29922",
                  annotation_text=f"POC {poc:.2f}", annotation_position="right")

    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#c9d1d9", margin=dict(l=60, r=80, t=20, b=20),
        xaxis_title="成交量", yaxis_title="价格区间",
        yaxis=dict(tickfont=dict(size=8)),
    )
    return fig


def _build_mtfa_panel(mtfa_result):
    """Build multi-timeframe resonance panel content."""
    if not mtfa_result or "error" in mtfa_result:
        return html.P("数据不足", className="text-muted")

    score = mtfa_result.get("overall_score", 50)
    score_color = "#3fb950" if score >= 65 else "#d29922" if score >= 45 else "#f85149"
    resonance = mtfa_result.get("resonance", "")
    rec = mtfa_result.get("recommendation", "")
    action = mtfa_result.get("action", "观望")
    action_color = "#3fb950" if action == "买入" else "#f85149" if action == "卖出" else "#d29922"

    # Matrix rows
    matrix = mtfa_result.get("matrix", {})
    periods = ["日线"]
    if "周线" in (mtfa_result.get("weekly") or {}):
        periods.append("周线")
    if "月线" in (mtfa_result.get("monthly") or {}):
        periods.append("月线")

    rows = []
    ind_labels = {"rsi": "RSI", "macd": "MACD", "ma_alignment": "均线", "bollinger": "布林带", "price_vs_ma20": "偏离MA20"}
    for ind_key, label in ind_labels.items():
        cells = [html.Td(label, style={"color": "#8b949e", "fontSize": "0.8rem"})]
        for p in periods:
            s = matrix.get(ind_key, {}).get(p, 5)
            color = "#3fb950" if s >= 7 else "#f85149" if s <= 3 else "#d29922" if s >= 5 else "#8b949e"
            cells.append(html.Td("●", style={"color": color, "textAlign": "center", "fontSize": "1.2rem"}))
        rows.append(html.Tr(cells))

    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            [
                                html.H1(f"{score:.0f}", style={"color": score_color, "fontWeight": "bold"}),
                                html.P("共振评分", className="text-muted"),
                            ],
                            style={"textAlign": "center"},
                        ),
                        width=3,
                    ),
                    dbc.Col(
                        html.Div(
                            [
                                html.H5(resonance, style={"color": "#c9d1d9"}),
                                html.H4(f"建议: {rec}", style={"color": action_color}),
                            ]
                        ),
                        width=9,
                    ),
                ],
                className="mb-3",
            ),
            html.Table(
                [html.Thead(html.Tr([html.Th("")]+[html.Th(p, style={"color": "#8b949e", "fontSize": "0.8rem", "textAlign": "center"}) for p in periods]))]
                + [html.Tbody(rows)],
                className="table table-dark table-sm",
                style={"fontSize": "0.85rem"},
            ),
        ]
    )


def _build_moneyflow_panel(mf_result):
    """Build capital flow panel content."""
    if not mf_result or "error" in mf_result:
        return html.P("资金流向数据暂不可用（需Tushare积分权限）", className="text-muted")

    score = mf_result.get("composite_score", 5)
    score_color = "#3fb950" if score >= 7 else "#d29922" if score >= 5 else "#f85149"
    overall = mf_result.get("overall", "")
    action = mf_result.get("action", "观望")
    action_color = "#3fb950" if action == "买入" else "#f85149" if action == "卖出" else "#d29922"

    main = mf_result.get("main_force", {})
    retail = mf_result.get("retail_contrarian", {})
    trend = mf_result.get("capital_trend", {})

    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            [
                                html.H1(f"{score:.1f}", style={"color": score_color, "fontWeight": "bold"}),
                                html.P("资金评分", className="text-muted"),
                            ],
                            style={"textAlign": "center"},
                        ),
                        width=3,
                    ),
                    dbc.Col(
                        html.Div(
                            [
                                html.H5(overall, style={"color": "#c9d1d9"}),
                                html.H4(f"操作建议: {action}", style={"color": action_color}),
                            ]
                        ),
                        width=9,
                    ),
                ],
                className="mb-3",
            ),
            dbc.Row(
                [
                    dbc.Col([html.H6("主力动向", className="text-info"), html.P(main.get("signal", "—"), style={"fontSize": "0.85rem"})], width=4),
                    dbc.Col([html.H6("散户信号", className="text-warning"), html.P(retail.get("signal", "—"), style={"fontSize": "0.85rem"})], width=4),
                    dbc.Col([html.H6("资金趋势", className="text-success"), html.P(trend.get("signal", "—"), style={"fontSize": "0.85rem"})], width=4),
                ]
            ),
        ]
    )


def _build_pattern_panel(harmonic, fractals):
    """Build pattern recognition panel content."""
    elements = []

    # Harmonic patterns
    if harmonic and len(harmonic) > 0:
        elements.append(html.H6("谐波形态", className="text-info"))
        for p in harmonic[:3]:
            dir_color = "#3fb950" if p["direction"] == "看涨" else "#f85149"
            elements.append(
                html.Div(
                    [
                        html.Span(f"{p['name']} · ", style={"fontWeight": "bold"}),
                        html.Span(p["direction"], style={"color": dir_color}),
                        html.Span(f" | 目标 {p['target']:.2f} | 止损 {p['stop']:.2f} | R:R {p.get('risk_reward', 0):.1f}", style={"color": "#8b949e", "fontSize": "0.8rem"}),
                    ],
                    style={"marginBottom": "0.3rem", "fontSize": "0.85rem"},
                )
            )
    else:
        elements.append(html.H6("谐波形态", className="text-muted"))
        elements.append(html.P("未检测到谐波形态", style={"fontSize": "0.8rem", "color": "#8b949e"}))

    elements.append(html.Hr(style={"borderColor": "#30363d", "margin": "0.5rem 0"}))

    # Fractals
    if fractals:
        detail = fractals.get("detail", {})
        bull = detail.get("bullish_fractals", [])
        bear = detail.get("bearish_fractals", [])
        elements.append(html.H6("分形高低点", className="text-info"))
        if bull:
            elements.append(html.P(f"最近低点分形: {bull[-1]['price']:.2f}", style={"color": "#3fb950", "fontSize": "0.8rem"}))
        if bear:
            elements.append(html.P(f"最近高点分形: {bear[-1]['price']:.2f}", style={"color": "#f85149", "fontSize": "0.8rem"}))
        elements.append(html.P(fractals.get("signal", ""), style={"fontSize": "0.8rem", "color": "#8b949e"}))
    else:
        elements.append(html.H6("分形高低点", className="text-muted"))
        elements.append(html.P("未检测到分形", style={"fontSize": "0.8rem", "color": "#8b949e"}))

    return html.Div(elements)
