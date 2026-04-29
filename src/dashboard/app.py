"""
[DEPRECATED] AIQuant Professional Dashboard v5.0 (Plotly Dash)
Built with Plotly Dash + Dash Bootstrap Components

⚠️ 废弃说明：可视化已迁移至 React + FastAPI 前端 (frontend/ + src/api/)。
   本文件及 src/dashboard/ 目录下所有页面不再维护，仅作历史参考。
   请使用 http://localhost:5173 访问新版界面。
"""
import sys
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Initialize Dash app with Bootstrap theme FIRST
# (page modules use dash.get_app() which requires app to exist)
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    suppress_callback_exceptions=True,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)

# Import page modules AFTER app is created
from src.dashboard.pages import overview, research, prediction, backtest, trading, system

app.title = "AIQuant v5.0 - 专业量化交易仪表盘"

# Navigation sidebar items
nav_items = [
    dbc.NavLink([html.I(className="fas fa-chart-line me-2"), "总览驾驶舱"], href="/", active="exact"),
    dbc.NavLink([html.I(className="fas fa-globe me-2"), "市场分析"], href="/market", active="exact"),
    dbc.NavLink([html.I(className="fas fa-search me-2"), "股票研究"], href="/research", active="exact"),
    dbc.NavLink([html.I(className="fas fa-robot me-2"), "模型预测"], href="/prediction", active="exact"),
    dbc.NavLink([html.I(className="fas fa-chart-bar me-2"), "回测中心"], href="/backtest", active="exact"),
    dbc.NavLink([html.I(className="fas fa-briefcase me-2"), "实盘交易"], href="/trading", active="exact"),
    dbc.NavLink([html.I(className="fas fa-cog me-2"), "系统管理"], href="/system", active="exact"),
]

# Sidebar
sidebar = html.Div(
    [
        html.H4("📈 AIQuant", className="display-6", style={"color": "#58a6ff", "padding": "1rem"}),
        html.Hr(style={"borderColor": "#30363d"}),
        dbc.Nav(nav_items, vertical=True, pills=True, className="gap-2"),
        html.Hr(style={"borderColor": "#30363d"}),
        html.Div(
            [
                html.Small("v5.0.0", className="text-muted"),
                html.Br(),
                html.Small("⚠️ 投资有风险，入市需谨慎", className="text-muted"),
            ],
            style={"padding": "1rem", "position": "absolute", "bottom": "0", "width": "100%"},
        ),
    ],
    style={
        "position": "fixed",
        "top": 0,
        "left": 0,
        "bottom": 0,
        "width": "16rem",
        "padding": "0",
        "backgroundColor": "#161b22",
        "borderRight": "1px solid #30363d",
    },
)

# Main content area
content = html.Div(
    id="page-content",
    style={
        "marginLeft": "16rem",
        "padding": "2rem",
        "backgroundColor": "#0d1117",
        "minHeight": "100vh",
    },
)

# App layout
app.layout = html.Div([dcc.Location(id="url"), sidebar, content])


# Page routing callback
@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname"),
)
def display_page(pathname):
    if pathname == "/" or pathname == "/overview":
        return overview.layout
    elif pathname == "/market":
        return html.Div([
            html.H2("📈 市场分析", className="mb-4", style={"color": "#c9d1d9"}),
            html.P("市场分析页面正在开发中...", style={"color": "#8b949e"}),
        ])
    elif pathname == "/research":
        return research.layout
    elif pathname == "/prediction":
        return prediction.layout
    elif pathname == "/backtest":
        return backtest.layout
    elif pathname == "/trading":
        return trading.layout
    elif pathname == "/system":
        return system.layout
    else:
        return html.Div([
            html.H2("404", className="mb-4", style={"color": "#c9d1d9"}),
            html.P("页面未找到", style={"color": "#8b949e"}),
        ])




if __name__ == "__main__":
    app.run(debug=True, port=8050, host="0.0.0.0")
