
from src.dashboard.app import app
"""
回测中心页 (Backtest Center Page)
回测列表 | 净值曲线 | 回撤分析 | 交易明细 | 月度收益矩阵
"""
import sys
from pathlib import Path

import dash_ag_grid as dag
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html
from dash.dependencies import Input, Output

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _list_backtests(pattern="p22_*"):
    """List all available backtest result directories."""
    results_dir = project_root / "data" / "results"
    if not results_dir.exists():
        return []
    dirs = sorted([d for d in results_dir.glob(pattern) if d.is_dir()], reverse=True)
    return [{"label": d.name, "value": str(d)} for d in dirs]


def _list_all_backtests():
    """List all backtest directories for comparison."""
    results_dir = project_root / "data" / "results"
    if not results_dir.exists():
        return []
    # Include p21, p22, p23, sector, baseline, etc.
    patterns = ["p21_*", "p22_*", "p23_*", "v291_baseline_*", "sector_*"]
    all_dirs = []
    for pattern in patterns:
        all_dirs.extend(results_dir.glob(pattern))
    dirs = sorted([d for d in all_dirs if d.is_dir()], reverse=True)
    return [{"label": d.name, "value": str(d)} for d in dirs]


def _load_backtest_data(backtest_path):
    """Load daily data and transactions for a backtest."""
    path = Path(backtest_path)
    daily_csv = path / "backtest_daily.csv"
    txn_csv = path / "backtest_transactions.csv"

    df_daily = None
    df_txn = None

    if daily_csv.exists():
        df_daily = pd.read_csv(daily_csv, encoding="utf-8-sig")
        df_daily["date"] = pd.to_datetime(df_daily["date"], format="%Y%m%d")
        if "total_value" in df_daily.columns:
            df_daily["peak"] = df_daily["total_value"].cummax()
            df_daily["drawdown"] = (df_daily["total_value"] - df_daily["peak"]) / df_daily["peak"] * 100

    if txn_csv.exists():
        df_txn = pd.read_csv(txn_csv, encoding="utf-8-sig")

    return df_daily, df_txn


def _calc_metrics(df_daily, df_txn):
    """Calculate key performance metrics."""
    if df_daily is None or df_daily.empty or "total_value" not in df_daily.columns:
        return {}

    total_return = (df_daily["total_value"].iloc[-1] / df_daily["total_value"].iloc[0] - 1) * 100
    max_dd = df_daily["drawdown"].min() if "drawdown" in df_daily.columns else 0

    # Win rate & profit ratio from transactions
    win_rate = 0
    profit_ratio = 0
    if df_txn is not None and not df_txn.empty and "action" in df_txn.columns:
        sells = df_txn[df_txn["action"] == "SELL"].copy()
        if "profit" in sells.columns and not sells.empty:
            sells["profit"] = pd.to_numeric(sells["profit"], errors="coerce")
            wins = (sells["profit"] > 0).sum()
            total = len(sells)
            win_rate = wins / total * 100 if total > 0 else 0
            avg_profit = sells[sells["profit"] > 0]["profit"].mean() if wins > 0 else 0
            avg_loss = abs(sells[sells["profit"] <= 0]["profit"].mean()) if (total - wins) > 0 else 1
            profit_ratio = avg_profit / avg_loss if avg_loss > 0 else 0

    return {
        "total_return": total_return,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "profit_ratio": profit_ratio,
        "days": len(df_daily),
    }


def _monthly_returns(df_daily):
    """Calculate monthly return matrix."""
    if df_daily is None or df_daily.empty or "total_value" not in df_daily.columns:
        return pd.DataFrame()

    df = df_daily.copy()
    df["year_month"] = df["date"].dt.strftime("%Y-%m")
    monthly = df.groupby("year_month").agg({"total_value": ["first", "last"]})
    monthly.columns = ["first", "last"]
    monthly["return"] = (monthly["last"] / monthly["first"] - 1) * 100
    monthly = monthly.reset_index()
    monthly["year"] = monthly["year_month"].str[:4]
    monthly["month"] = monthly["year_month"].str[5:7]
    pivot = monthly.pivot(index="year", columns="month", values="return")
    return pivot


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

layout = html.Div(
    [
        html.H1("📉 回测中心", className="mb-3", style={"color": "#c9d1d9", "fontWeight": "bold"}),

        # Backtest selector
        dbc.Row(
            [
                dbc.Col(
                    dcc.Dropdown(
                        id="backtest-selector",
                        options=_list_backtests(),
                        placeholder="选择回测结果查看详情...",
                        style={"color": "#21262d"},
                    ),
                    width=4,
                ),
            ],
            className="mb-3",
        ),

        # Multi-strategy comparison
        dbc.Card(
            [
                dbc.CardHeader("🔀 多策略对比", style={"backgroundColor": "#21262d", "color": "#c9d1d9", "fontWeight": "bold"}),
                dbc.CardBody(
                    [
                        dcc.Dropdown(
                            id="backtest-compare-selector",
                            options=_list_all_backtests(),
                            placeholder="选择多个回测结果进行对比...",
                            multi=True,
                            style={"color": "#21262d", "marginBottom": "1rem"},
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    dcc.Graph(id="bt-compare-chart", config={"displayModeBar": False}, style={"height": "350px"}),
                                    width=8,
                                ),
                                dbc.Col(
                                    dag.AgGrid(
                                        id="bt-compare-table",
                                        columnDefs=[
                                            {"field": "strategy", "headerName": "策略", "width": 180, "sortable": True},
                                            {"field": "total_return", "headerName": "总收益%", "width": 100, "sortable": True, "valueFormatter": {"function": "params.value ? params.value.toFixed(2) + '%' : ''"}},
                                            {"field": "max_drawdown", "headerName": "最大回撤%", "width": 110, "sortable": True, "valueFormatter": {"function": "params.value ? params.value.toFixed(2) + '%' : ''"}},
                                            {"field": "win_rate", "headerName": "胜率%", "width": 90, "sortable": True, "valueFormatter": {"function": "params.value ? params.value.toFixed(1) + '%' : ''"}},
                                            {"field": "profit_ratio", "headerName": "盈亏比", "width": 90, "sortable": True, "valueFormatter": {"function": "params.value ? params.value.toFixed(2) : ''"}},
                                            {"field": "days", "headerName": "交易日", "width": 90, "sortable": True},
                                        ],
                                        defaultColDef={"resizable": True, "sortable": True},
                                        dashGridOptions={"domLayout": "autoHeight"},
                                        style={"height": "350px"},
                                    ),
                                    width=4,
                                ),
                            ],
                            className="g-3",
                        ),
                    ],
                    style={"backgroundColor": "#161b22", "color": "#c9d1d9"},
                ),
            ],
            style={"border": "1px solid #30363d", "marginBottom": "1rem"},
        ),

        # Metrics cards
        dbc.Row(
            [
                dbc.Col(dbc.Card(id="bt-stat-return", style={"backgroundColor": "#161b22", "border": "1px solid #30363d"}), width=3),
                dbc.Col(dbc.Card(id="bt-stat-drawdown", style={"backgroundColor": "#161b22", "border": "1px solid #30363d"}), width=3),
                dbc.Col(dbc.Card(id="bt-stat-winrate", style={"backgroundColor": "#161b22", "border": "1px solid #30363d"}), width=3),
                dbc.Col(dbc.Card(id="bt-stat-ratio", style={"backgroundColor": "#161b22", "border": "1px solid #30363d"}), width=3),
            ],
            className="mb-3 g-3",
        ),

        # Charts row
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("📈 净值曲线", style={"backgroundColor": "#21262d", "color": "#c9d1d9", "fontWeight": "bold"}),
                            dbc.CardBody(
                                dcc.Graph(id="bt-equity-chart", config={"displayModeBar": False}, style={"height": "350px"}),
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
                            dbc.CardHeader("📉 回撤曲线", style={"backgroundColor": "#21262d", "color": "#c9d1d9", "fontWeight": "bold"}),
                            dbc.CardBody(
                                dcc.Graph(id="bt-drawdown-chart", config={"displayModeBar": False}, style={"height": "350px"}),
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

        # Monthly returns + Transactions
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("🗓️ 月度收益矩阵", style={"backgroundColor": "#21262d", "color": "#c9d1d9", "fontWeight": "bold"}),
                            dbc.CardBody(
                                dcc.Graph(id="bt-monthly-chart", config={"displayModeBar": False}, style={"height": "250px"}),
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
                            dbc.CardHeader("📋 交易明细", style={"backgroundColor": "#21262d", "color": "#c9d1d9", "fontWeight": "bold"}),
                            dbc.CardBody(
                                dag.AgGrid(
                                    id="bt-transaction-table",
                                    columnDefs=[
                                        {"field": "date", "headerName": "日期", "width": 100},
                                        {"field": "action", "headerName": "操作", "width": 80},
                                        {"field": "ts_code", "headerName": "代码", "width": 120},
                                        {"field": "price", "headerName": "价格", "width": 90, "valueFormatter": {"function": "params.value ? params.value.toFixed(2) : ''"}},
                                        {"field": "quantity", "headerName": "数量", "width": 80},
                                        {"field": "profit", "headerName": "盈亏", "width": 90, "valueFormatter": {"function": "params.value ? params.value.toFixed(2) : ''"}},
                                    ],
                                    defaultColDef={"resizable": True, "sortable": True},
                                    dashGridOptions={"pagination": True, "paginationPageSize": 10, "domLayout": "autoHeight"},
                                    style={"height": "250px"},
                                ),
                                style={"backgroundColor": "#161b22", "padding": "0"},
                            ),
                        ],
                        style={"border": "1px solid #30363d"},
                    ),
                    width=6,
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
        Output("bt-stat-return", "children"),
        Output("bt-stat-drawdown", "children"),
        Output("bt-stat-winrate", "children"),
        Output("bt-stat-ratio", "children"),
        Output("bt-equity-chart", "figure"),
        Output("bt-drawdown-chart", "figure"),
        Output("bt-monthly-chart", "figure"),
        Output("bt-transaction-table", "rowData"),
    ],
    Input("backtest-selector", "value"),
    prevent_initial_call=True,
)
def update_backtest(backtest_path):
    """Load and display backtest data."""
    if not backtest_path:
        empty = go.Figure().update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            annotations=[{"text": "请选择回测结果", "showarrow": False, "font": {"size": 16, "color": "#8b949e"}}],
        )
        return [dbc.CardBody([html.H5("--", className="text-muted")])] * 4 + [empty, empty, empty, []]

    df_daily, df_txn = _load_backtest_data(backtest_path)
    metrics = _calc_metrics(df_daily, df_txn)

    # Stats cards
    ret_val = metrics.get("total_return", 0)
    ret_color = "#3fb950" if ret_val >= 0 else "#f85149"
    stat_return = dbc.CardBody([
        html.H5("总收益率", className="text-muted"),
        html.H3(f"{ret_val:+.2f}%", style={"color": ret_color}),
    ])

    dd_val = metrics.get("max_drawdown", 0)
    stat_dd = dbc.CardBody([
        html.H5("最大回撤", className="text-muted"),
        html.H3(f"{dd_val:.2f}%", style={"color": "#f85149"}),
    ])

    wr_val = metrics.get("win_rate", 0)
    wr_color = "#3fb950" if wr_val >= 50 else "#d29922" if wr_val >= 40 else "#f85149"
    stat_wr = dbc.CardBody([
        html.H5("胜率", className="text-muted"),
        html.H3(f"{wr_val:.1f}%", style={"color": wr_color}),
    ])

    pr_val = metrics.get("profit_ratio", 0)
    pr_color = "#3fb950" if pr_val >= 1.0 else "#d29922" if pr_val >= 0.8 else "#f85149"
    stat_pr = dbc.CardBody([
        html.H5("盈亏比", className="text-muted"),
        html.H3(f"{pr_val:.2f}", style={"color": pr_color}),
    ])

    # Equity chart
    if df_daily is not None and not df_daily.empty:
        fig_equity = go.Figure()
        fig_equity.add_trace(go.Scatter(
            x=df_daily["date"], y=df_daily["total_value"],
            mode="lines", name="净值",
            line=dict(color="#3fb950", width=2),
            fill="tozeroy", fillcolor="rgba(63,185,80,0.1)",
        ))
        init = df_daily["total_value"].iloc[0]
        fig_equity.add_hline(y=init, line_dash="dash", line_color="#8b949e", annotation_text=f"初始 {init:,.0f}")
    else:
        fig_equity = go.Figure()

    fig_equity.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#c9d1d9", margin=dict(l=40, r=40, t=40, b=40),
        xaxis_title="日期", yaxis_title="净值", showlegend=False,
    )

    # Drawdown chart
    if df_daily is not None and not df_daily.empty and "drawdown" in df_daily.columns:
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=df_daily["date"], y=df_daily["drawdown"],
            mode="lines", name="回撤",
            fill="tozeroy", line=dict(color="#f85149", width=1),
            fillcolor="rgba(248,81,73,0.2)",
        ))
    else:
        fig_dd = go.Figure()

    fig_dd.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#c9d1d9", margin=dict(l=40, r=40, t=40, b=40),
        xaxis_title="日期", yaxis_title="回撤 %", showlegend=False,
    )

    # Monthly returns heatmap
    pivot = _monthly_returns(df_daily)
    if not pivot.empty:
        fig_monthly = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale=["#3fb950", "#21262d", "#f85149"],
            zmid=0,
            text=[[f"{v:.1f}%" if not np.isnan(v) else "" for v in row] for row in pivot.values],
            texttemplate="%{text}",
            textfont={"size": 10},
        ))
    else:
        fig_monthly = go.Figure()

    fig_monthly.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#c9d1d9", margin=dict(l=40, r=40, t=40, b=40),
        xaxis_title="月份", yaxis_title="年份",
    )

    # Transaction table
    txn_rows = []
    if df_txn is not None and not df_txn.empty:
        for _, row in df_txn.iterrows():
            txn_rows.append({
                "date": str(row.get("date", "")),
                "action": str(row.get("action", "")),
                "ts_code": str(row.get("ts_code", row.get("code", ""))),
                "price": float(row.get("price", 0)) if pd.notna(row.get("price")) else 0,
                "quantity": int(row.get("quantity", row.get("qty", 0))) if pd.notna(row.get("quantity", row.get("qty"))) else 0,
                "profit": float(row.get("profit", 0)) if pd.notna(row.get("profit")) else 0,
            })

    return stat_return, stat_dd, stat_wr, stat_pr, fig_equity, fig_dd, fig_monthly, txn_rows



@app.callback(
    [
        Output("bt-compare-chart", "figure"),
        Output("bt-compare-table", "rowData"),
    ],
    Input("backtest-compare-selector", "value"),
    prevent_initial_call=True,
)
def update_compare(selected_paths):
    """Compare multiple backtest strategies."""
    if not selected_paths:
        empty_fig = go.Figure().update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            annotations=[{"text": "请选择多个回测结果进行对比", "showarrow": False, "font": {"size": 16, "color": "#8b949e"}}],
        )
        return empty_fig, []

    fig = go.Figure()
    colors = ["#58a6ff", "#3fb950", "#d29922", "#f85149", "#a371f7", "#f778ba", "#79c0ff", "#56d364"]
    compare_rows = []

    for i, path in enumerate(selected_paths):
        df_daily, df_txn = _load_backtest_data(path)
        if df_daily is None or df_daily.empty or "total_value" not in df_daily.columns:
            continue

        name = Path(path).name
        color = colors[i % len(colors)]

        # Normalize to percentage change from start
        base = df_daily["total_value"].iloc[0]
        normalized = (df_daily["total_value"] / base - 1) * 100

        fig.add_trace(go.Scatter(
            x=df_daily["date"],
            y=normalized,
            mode="lines",
            name=name,
            line=dict(color=color, width=1.5),
            hovertemplate="%{x}<br>" + name + ": %{y:+.2f}%<extra></extra>",
        ))

        # Metrics for comparison table
        metrics = _calc_metrics(df_daily, df_txn)
        compare_rows.append({
            "strategy": name,
            "total_return": metrics.get("total_return", 0),
            "max_drawdown": metrics.get("max_drawdown", 0),
            "win_rate": metrics.get("win_rate", 0),
            "profit_ratio": metrics.get("profit_ratio", 0),
            "days": metrics.get("days", 0),
        })

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#c9d1d9",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10)),
        margin=dict(l=50, r=40, t=60, b=40),
        xaxis_title="日期",
        yaxis_title="收益率 %",
        hovermode="x unified",
    )

    # Sort by total return descending
    compare_rows.sort(key=lambda x: x["total_return"], reverse=True)

    return fig, compare_rows
