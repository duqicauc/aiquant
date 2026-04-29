"""
市场分析页 (Market Analysis Page)
行业热力图 | 指数对比 | 市场情绪 | 涨跌分布
"""
import sys
from datetime import datetime, timedelta
from pathlib import Path

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import dcc, html
from dash.dependencies import Input, Output

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.data_manager import DataManager

from src.dashboard.app import app

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _fetch_sector_data():
    """Fetch sector performance data via Tushare."""
    try:
        import tushare as ts
        pro = ts.pro_api()

        # Try to get index_daily for sector indices
        today = datetime.now().strftime("%Y%m%d")
        # Common sector indices (use mock for now, real sector data requires specific codes)
        sectors = [
            {"name": "人工智能", "pct_chg": np.random.uniform(-2, 4)},
            {"name": "半导体", "pct_chg": np.random.uniform(-2, 4)},
            {"name": "新能源", "pct_chg": np.random.uniform(-2, 4)},
            {"name": "医药生物", "pct_chg": np.random.uniform(-2, 4)},
            {"name": "银行", "pct_chg": np.random.uniform(-2, 4)},
            {"name": "房地产", "pct_chg": np.random.uniform(-2, 4)},
            {"name": "煤炭", "pct_chg": np.random.uniform(-2, 4)},
            {"name": "钢铁", "pct_chg": np.random.uniform(-2, 4)},
            {"name": "食品饮料", "pct_chg": np.random.uniform(-2, 4)},
            {"name": "电子", "pct_chg": np.random.uniform(-2, 4)},
            {"name": "计算机", "pct_chg": np.random.uniform(-2, 4)},
            {"name": "通信", "pct_chg": np.random.uniform(-2, 4)},
            {"name": "传媒", "pct_chg": np.random.uniform(-2, 4)},
            {"name": "汽车", "pct_chg": np.random.uniform(-2, 4)},
            {"name": "机械设备", "pct_chg": np.random.uniform(-2, 4)},
            {"name": "化工", "pct_chg": np.random.uniform(-2, 4)},
            {"name": "有色金属", "pct_chg": np.random.uniform(-2, 4)},
            {"name": "建筑材料", "pct_chg": np.random.uniform(-2, 4)},
            {"name": "国防军工", "pct_chg": np.random.uniform(-2, 4)},
            {"name": "公用事业", "pct_chg": np.random.uniform(-2, 4)},
        ]
        # Sort by pct_chg descending
        sectors.sort(key=lambda x: x["pct_chg"], reverse=True)
        return sectors
    except Exception:
        return []


def _fetch_index_comparison():
    """Fetch multiple indices for comparison chart."""
    dm = DataManager()
    indices = {
        "上证指数": "000001.SH",
        "深证成指": "399001.SZ",
        "创业板指": "399006.SZ",
        "沪深300": "000300.SH",
    }
    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=60)).strftime("%Y%m%d")

    result = {}
    for name, code in indices.items():
        try:
            df = dm.get_index_daily(code, start_date, end_date)
            if df is not None and not df.empty:
                df = df.sort_values("trade_date")
                # Normalize to percentage change from start
                base = df["close"].iloc[0]
                df["normalized"] = (df["close"] / base - 1) * 100
                result[name] = df
        except Exception:
            continue
    return result


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

layout = html.Div(
    [
        html.H1("📈 市场分析", className="mb-3", style={"color": "#c9d1d9", "fontWeight": "bold"}),

        # Top stats
        dbc.Row(
            [
                dbc.Col(dbc.Card(id="market-stat-up", style={"backgroundColor": "#161b22", "border": "1px solid #30363d"}), width=3),
                dbc.Col(dbc.Card(id="market-stat-down", style={"backgroundColor": "#161b22", "border": "1px solid #30363d"}), width=3),
                dbc.Col(dbc.Card(id="market-stat-limit", style={"backgroundColor": "#161b22", "border": "1px solid #30363d"}), width=3),
                dbc.Col(dbc.Card(id="market-stat-amount", style={"backgroundColor": "#161b22", "border": "1px solid #30363d"}), width=3),
            ],
            className="mb-3 g-3",
        ),

        # Charts row
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("🔥 行业热力图", style={"backgroundColor": "#21262d", "color": "#c9d1d9", "fontWeight": "bold"}),
                            dbc.CardBody(
                                dcc.Graph(id="sector-heatmap", config={"displayModeBar": False}, style={"height": "400px"}),
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
                            dbc.CardHeader("📊 指数对比 (近60日涨跌幅)", style={"backgroundColor": "#21262d", "color": "#c9d1d9", "fontWeight": "bold"}),
                            dbc.CardBody(
                                dcc.Graph(id="index-comparison", config={"displayModeBar": False}, style={"height": "400px"}),
                                style={"backgroundColor": "#161b22", "padding": "0"},
                            ),
                        ],
                        style={"border": "1px solid #30363d"},
                    ),
                    width=6,
                ),
            ],
            className="mb-3 g-3",
        ),

        # Sector ranking bars
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("🏆 板块涨幅排行", style={"backgroundColor": "#21262d", "color": "#c9d1d9", "fontWeight": "bold"}),
                            dbc.CardBody(
                                dcc.Graph(id="sector-ranking", config={"displayModeBar": False}, style={"height": "350px"}),
                                style={"backgroundColor": "#161b22", "padding": "0"},
                            ),
                        ],
                        style={"border": "1px solid #30363d"},
                    ),
                    width=12,
                ),
            ],
            className="g-3",
        ),

        dcc.Interval(id="market-interval", interval=60 * 1000, n_intervals=0),
    ]
)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

@app.callback(
    [
        Output("market-stat-up", "children"),
        Output("market-stat-down", "children"),
        Output("market-stat-limit", "children"),
        Output("market-stat-amount", "children"),
        Output("sector-heatmap", "figure"),
        Output("index-comparison", "figure"),
        Output("sector-ranking", "figure"),
    ],
    Input("market-interval", "n_intervals"),
)
def update_market(n):
    """Update market analysis data."""
    sectors = _fetch_sector_data()

    # Market stats (placeholder for real data)
    stat_up = dbc.CardBody([html.H5("上涨家数", className="text-muted"), html.H3("--", style={"color": "#f85149"})])
    stat_down = dbc.CardBody([html.H5("下跌家数", className="text-muted"), html.H3("--", style={"color": "#3fb950"})])
    stat_limit = dbc.CardBody([html.H5("涨停/跌停", className="text-muted"), html.H3("-- / --", style={"color": "#d29922"})])
    stat_amount = dbc.CardBody([html.H5("两市成交额", className="text-muted"), html.H3("-- 亿", style={"color": "#58a6ff"})])

    # Sector heatmap
    if sectors:
        df_sectors = pd.DataFrame(sectors)
        n_sectors = len(df_sectors)
        # Reshape to roughly square grid
        cols = 5
        rows = (n_sectors + cols - 1) // cols

        heatmap_data = []
        labels = []
        for i in range(rows):
            row_vals = []
            row_labels = []
            for j in range(cols):
                idx = i * cols + j
                if idx < n_sectors:
                    row_vals.append(df_sectors.iloc[idx]["pct_chg"])
                    row_labels.append(df_sectors.iloc[idx]["name"])
                else:
                    row_vals.append(np.nan)
                    row_labels.append("")
            heatmap_data.append(row_vals)
            labels.append(row_labels)

        fig_heatmap = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            text=labels,
            texttemplate="%{text}<br>%{z:.1f}%",
            textfont={"size": 10},
            colorscale=["#3fb950", "#21262d", "#f85149"],
            zmid=0,
            hoverongaps=False,
        ))
    else:
        fig_heatmap = go.Figure()

    fig_heatmap.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#c9d1d9",
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_visible=False,
        yaxis_visible=False,
    )

    # Index comparison
    index_data = _fetch_index_comparison()
    fig_index = go.Figure()
    colors = ["#58a6ff", "#3fb950", "#d29922", "#f85149"]
    for i, (name, df) in enumerate(index_data.items()):
        fig_index.add_trace(go.Scatter(
            x=df["trade_date"].astype(str),
            y=df["normalized"],
            mode="lines",
            name=name,
            line=dict(color=colors[i % len(colors)], width=1.5),
        ))

    fig_index.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#c9d1d9",
        margin=dict(l=40, r=40, t=40, b=40),
        xaxis_title="日期",
        yaxis_title="涨跌幅 %",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )

    # Sector ranking bar chart
    if sectors:
        df_rank = pd.DataFrame(sectors).head(15)
        colors_bar = ["#f85149" if v >= 0 else "#3fb950" for v in df_rank["pct_chg"]]
        fig_rank = go.Figure(data=go.Bar(
            x=df_rank["pct_chg"],
            y=df_rank["name"],
            orientation="h",
            marker_color=colors_bar,
        ))
        fig_rank.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#c9d1d9",
            margin=dict(l=100, r=40, t=40, b=40),
            xaxis_title="涨跌幅 %",
            yaxis=dict(autorange="reversed"),
        )
    else:
        fig_rank = go.Figure()

    return stat_up, stat_down, stat_limit, stat_amount, fig_heatmap, fig_index, fig_rank
