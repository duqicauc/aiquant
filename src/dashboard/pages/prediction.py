
from src.dashboard.app import app
"""
模型预测页 (Model Prediction Page)
今日选股 Top N | 概率分布 | 模型版本切换 | 行业分布
"""
import sys
from pathlib import Path

import dash_ag_grid as dag
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import dcc, html
from dash.dependencies import Input, Output, State

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load_latest_predictions(top_n=50):
    """Load latest prediction CSV from all prediction directories."""
    pred_root = project_root / "data" / "prediction"
    if not pred_root.exists():
        return None, "No prediction directory"

    # Search all subdirectories for prediction CSVs
    all_files = []
    for subdir in pred_root.iterdir():
        if subdir.is_dir():
            # Look for integrated predictions first (most recent)
            integrated = list(subdir.glob("predictions_*_integrated_*.csv"))
            top50 = list(subdir.glob("predictions_*_top50.csv"))
            top100 = list(subdir.glob("predictions_*_top100.csv"))
            all_files.extend(integrated + top50 + top100)

    if not all_files:
        return None, "No prediction files found"

    # Sort by modification time (newest first)
    all_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    latest_file = all_files[0]

    try:
        df = pd.read_csv(latest_file)
        filename = f"{latest_file.parent.name}/{latest_file.name}"
    except Exception as e:
        return None, f"Failed to read {latest_file}: {e}"

    if df is None or df.empty:
        return None, "Empty prediction file"

    # Normalize probability/score column
    prob_col = None
    for col in ["prob", "probability", "score", "牛股概率", "pred", "prediction"]:
        if col in df.columns:
            prob_col = col
            break

    if prob_col:
        # Ensure probability is in [0, 1] range
        max_val = df[prob_col].max()
        if max_val > 1:
            df[prob_col] = df[prob_col] / 100
        df = df.sort_values(prob_col, ascending=False).head(top_n)
    else:
        # If no probability column, just take first N rows
        df = df.head(top_n)

    return df, filename


def _get_sector_distribution(df):
    """Get industry distribution from predictions."""
    if df is None or df.empty:
        return pd.DataFrame()
    if "industry" in df.columns:
        return df["industry"].value_counts().reset_index()
    if "行业" in df.columns:
        return df["行业"].value_counts().reset_index()
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

layout = html.Div(
    [
        html.H1("🤖 模型预测", className="mb-3", style={"color": "#c9d1d9", "fontWeight": "bold"}),

        # Top controls
        dbc.Row(
            [
                dbc.Col(
                    dbc.InputGroup(
                        [
                            dbc.InputGroupText("显示数量", style={"backgroundColor": "#21262d", "color": "#8b949e", "borderColor": "#30363d"}),
                            dbc.Input(
                                id="pred-top-n",
                                type="number",
                                value=30,
                                min=5,
                                max=100,
                                style={"backgroundColor": "#21262d", "color": "#c9d1d9", "borderColor": "#30363d"},
                            ),
                        ],
                        className="mb-2",
                    ),
                    width=2,
                ),
                dbc.Col(
                    dbc.Button("🔄 刷新数据", id="pred-refresh", color="primary", className="mb-2"),
                    width="auto",
                ),
                dbc.Col(html.Div(id="pred-filename", className="text-muted mb-2"), width="auto"),
            ],
            className="g-2 align-items-center mb-3",
        ),

        # Stats row
        dbc.Row(
            [
                dbc.Col(dbc.Card(id="pred-stat-count", style={"backgroundColor": "#161b22", "border": "1px solid #30363d"}), width=3),
                dbc.Col(dbc.Card(id="pred-stat-avg", style={"backgroundColor": "#161b22", "border": "1px solid #30363d"}), width=3),
                dbc.Col(dbc.Card(id="pred-stat-high", style={"backgroundColor": "#161b22", "border": "1px solid #30363d"}), width=3),
                dbc.Col(dbc.Card(id="pred-stat-top10", style={"backgroundColor": "#161b22", "border": "1px solid #30363d"}), width=3),
            ],
            className="mb-3 g-3",
        ),

        # Main content: Table + Charts
        dbc.Row(
            [
                # Left: Prediction table
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("📋 选股列表", style={"backgroundColor": "#21262d", "color": "#c9d1d9", "fontWeight": "bold"}),
                            dbc.CardBody(
                                dag.AgGrid(
                                    id="pred-table",
                                    columnDefs=[
                                        {"field": "rank", "headerName": "排名", "width": 70, "sortable": True},
                                        {"field": "ts_code", "headerName": "代码", "width": 120, "sortable": True, "filter": True},
                                        {"field": "name", "headerName": "名称", "width": 120, "sortable": True, "filter": True},
                                        {"field": "industry", "headerName": "行业", "width": 120, "sortable": True, "filter": True},
                                        {"field": "prob", "headerName": "牛股概率", "width": 110, "sortable": True, "valueFormatter": {"function": "params.value ? (params.value * 100).toFixed(1) + '%' : ''"}},
                                        {"field": "adjusted_score", "headerName": "综合得分", "width": 110, "sortable": True, "valueFormatter": {"function": "params.value ? params.value.toFixed(3) : ''"}},
                                        {"field": "close", "headerName": "最新价", "width": 90, "sortable": True, "valueFormatter": {"function": "params.value ? params.value.toFixed(2) : ''"}},
                                        {"field": "pct_chg", "headerName": "涨跌幅%", "width": 100, "sortable": True, "valueFormatter": {"function": "params.value ? params.value.toFixed(2) + '%' : ''"}},
                                    ],
                                    defaultColDef={"resizable": True, "sortable": True},
                                    dashGridOptions={
                                        "pagination": True,
                                        "paginationPageSize": 20,
                                        "rowSelection": "single",
                                        "domLayout": "autoHeight",
                                    },
                                    style={"height": "500px"},
                                ),
                                style={"backgroundColor": "#161b22", "padding": "0"},
                            ),
                        ],
                        style={"border": "1px solid #30363d"},
                    ),
                    width=7,
                ),

                # Right: Charts
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader("📊 概率分布", style={"backgroundColor": "#21262d", "color": "#c9d1d9", "fontWeight": "bold"}),
                                dbc.CardBody(
                                    dcc.Graph(id="pred-histogram", config={"displayModeBar": False}, style={"height": "230px"}),
                                    style={"backgroundColor": "#161b22", "padding": "0"},
                                ),
                            ],
                            style={"border": "1px solid #30363d", "marginBottom": "1rem"},
                        ),
                        dbc.Card(
                            [
                                dbc.CardHeader("🏭 行业分布", style={"backgroundColor": "#21262d", "color": "#c9d1d9", "fontWeight": "bold"}),
                                dbc.CardBody(
                                    dcc.Graph(id="pred-sector-pie", config={"displayModeBar": False}, style={"height": "230px"}),
                                    style={"backgroundColor": "#161b22", "padding": "0"},
                                ),
                            ],
                            style={"border": "1px solid #30363d"},
                        ),
                    ],
                    width=5,
                ),
            ],
            className="g-3",
        ),

        # Interval for auto-refresh
        dcc.Interval(id="pred-interval", interval=60 * 1000, n_intervals=0),
    ]
)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

@app.callback(
    [
        Output("pred-table", "rowData"),
        Output("pred-filename", "children"),
        Output("pred-stat-count", "children"),
        Output("pred-stat-avg", "children"),
        Output("pred-stat-high", "children"),
        Output("pred-stat-top10", "children"),
        Output("pred-histogram", "figure"),
        Output("pred-sector-pie", "figure"),
    ],
    [
        Input("pred-interval", "n_intervals"),
        Input("pred-refresh", "n_clicks"),
    ],
    State("pred-top-n", "value"),
    prevent_initial_call=False,
)
def update_predictions(n_interval, n_clicks, top_n):
    """Load and display prediction data."""
    df, filename = _load_latest_predictions(top_n or 30)

    if df is None:
        empty_fig = go.Figure().update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            annotations=[{"text": "暂无预测数据", "showarrow": False, "font": {"size": 16, "color": "#8b949e"}}],
        )
        return [], f"文件: {filename}", "--", "--", "--", "--", empty_fig, empty_fig

    # Prepare table data
    prob_col = None
    for col in ["牛股概率", "probability", "prob", "score"]:
        if col in df.columns:
            prob_col = col
            break

    rows = []
    for i, (_, row) in enumerate(df.iterrows()):
        rec = {
            "rank": i + 1,
            "ts_code": str(row.get("ts_code", row.get("code", ""))),
            "name": str(row.get("name", "")),
            "industry": str(row.get("industry", row.get("行业", ""))),
            "probability": float(row.get(prob_col, 0)) if prob_col else 0,
            "score": float(row.get("score", row.get(prob_col, 0))),
            "model_version": str(row.get("model_version", "v2.9.5")),
        }
        rows.append(rec)

    # Stats cards
    probs = [r["probability"] for r in rows]
    avg_prob = sum(probs) / len(probs) if probs else 0
    high_prob = max(probs) if probs else 0
    top10_avg = sum(probs[:10]) / 10 if len(probs) >= 10 else avg_prob

    stat_count = dbc.CardBody([html.H5("股票数量", className="text-muted"), html.H3(f"{len(rows)}")])
    stat_avg = dbc.CardBody([html.H5("平均概率", className="text-muted"), html.H3(f"{avg_prob * 100:.1f}%", style={"color": "#58a6ff"})])
    stat_high = dbc.CardBody([html.H5("最高概率", className="text-muted"), html.H3(f"{high_prob * 100:.1f}%", style={"color": "#3fb950"})])
    stat_top10 = dbc.CardBody([html.H5("Top10平均", className="text-muted"), html.H3(f"{top10_avg * 100:.1f}%", style={"color": "#d29922"})])

    # Histogram
    fig_hist = go.Figure(
        data=[go.Histogram(x=[p * 100 for p in probs], nbinsx=20, marker_color="#58a6ff")]
    )
    fig_hist.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#c9d1d9",
        margin=dict(l=30, r=30, t=30, b=30),
        xaxis_title="牛股概率 (%)",
        yaxis_title="股票数量",
    )

    # Sector pie
    sector_df = _get_sector_distribution(df)
    if not sector_df.empty:
        sector_df.columns = ["industry", "count"]
        fig_pie = px.pie(
            sector_df.head(10),
            values="count",
            names="industry",
            hole=0.4,
            color_discrete_sequence=px.colors.sequential.Blues_r,
        )
    else:
        fig_pie = go.Figure()
    fig_pie.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#c9d1d9",
        margin=dict(l=20, r=20, t=30, b=20),
        showlegend=True,
        legend=dict(font=dict(size=10)),
    )

    return rows, f"文件: {filename}", stat_count, stat_avg, stat_high, stat_top10, fig_hist, fig_pie
