
from src.dashboard.app import app
"""
实盘交易页 (Live Trading Page)
持仓监控 | 盈亏分析 | 交易日志 | 下单面板（模拟）
"""
import sys
from datetime import datetime
from pathlib import Path

import dash_ag_grid as dag
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

layout = html.Div(
    [
        html.H1("💼 实盘交易", className="mb-3", style={"color": "#c9d1d9", "fontWeight": "bold"}),

        # Warning banner
        dbc.Alert(
            [
                html.I(className="fas fa-info-circle me-2"),
                "实盘交易功能需接入 BrokerAdapter（MiniQMT）后方可启用。当前为模拟展示。",
            ],
            color="warning",
            className="mb-3",
        ),

        # Portfolio stats
        dbc.Row(
            [
                dbc.Col(dbc.Card(id="trd-total-asset", style={"backgroundColor": "#161b22", "border": "1px solid #30363d"}), width=3),
                dbc.Col(dbc.Card(id="trd-market-value", style={"backgroundColor": "#161b22", "border": "1px solid #30363d"}), width=3),
                dbc.Col(dbc.Card(id="trd-today-pnl", style={"backgroundColor": "#161b22", "border": "1px solid #30363d"}), width=3),
                dbc.Col(dbc.Card(id="trd-available", style={"backgroundColor": "#161b22", "border": "1px solid #30363d"}), width=3),
            ],
            className="mb-3 g-3",
        ),

        # Main content: positions + order panel
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("📋 持仓列表", style={"backgroundColor": "#21262d", "color": "#c9d1d9", "fontWeight": "bold"}),
                            dbc.CardBody(
                                dag.AgGrid(
                                    id="position-table",
                                    columnDefs=[
                                        {"field": "ts_code", "headerName": "代码", "width": 120},
                                        {"field": "name", "headerName": "名称", "width": 100},
                                        {"field": "quantity", "headerName": "数量", "width": 90},
                                        {"field": "avg_cost", "headerName": "成本价", "width": 100, "valueFormatter": {"function": "params.value ? params.value.toFixed(2) : ''"}},
                                        {"field": "current_price", "headerName": "现价", "width": 100, "valueFormatter": {"function": "params.value ? params.value.toFixed(2) : ''"}},
                                        {"field": "market_value", "headerName": "市值", "width": 110, "valueFormatter": {"function": "params.value ? params.value.toFixed(0) : ''"}},
                                        {"field": "pnl", "headerName": "盈亏额", "width": 110, "valueFormatter": {"function": "params.value ? params.value.toFixed(0) : ''"}},
                                        {"field": "pnl_pct", "headerName": "盈亏%", "width": 90, "valueFormatter": {"function": "params.value ? params.value.toFixed(1) + '%' : ''"}},
                                        {"field": "weight", "headerName": "仓位", "width": 80, "valueFormatter": {"function": "params.value ? params.value.toFixed(1) + '%' : ''"}},
                                    ],
                                    defaultColDef={"resizable": True, "sortable": True},
                                    dashGridOptions={"pagination": True, "paginationPageSize": 10, "domLayout": "autoHeight"},
                                    style={"height": "350px"},
                                    rowData=[],
                                ),
                                style={"backgroundColor": "#161b22", "padding": "0"},
                            ),
                        ],
                        style={"border": "1px solid #30363d"},
                    ),
                    width=8,
                ),
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("📝 模拟下单", style={"backgroundColor": "#21262d", "color": "#c9d1d9", "fontWeight": "bold"}),
                            dbc.CardBody(
                                [
                                    dbc.InputGroup(
                                        [
                                            dbc.InputGroupText("代码", style={"backgroundColor": "#30363d", "color": "#8b949e", "borderColor": "#30363d"}),
                                            dbc.Input(placeholder="000001.SZ", style={"backgroundColor": "#21262d", "color": "#c9d1d9", "borderColor": "#30363d"}),
                                        ],
                                        className="mb-2",
                                    ),
                                    dbc.InputGroup(
                                        [
                                            dbc.InputGroupText("数量", style={"backgroundColor": "#30363d", "color": "#8b949e", "borderColor": "#30363d"}),
                                            dbc.Input(type="number", placeholder="100", style={"backgroundColor": "#21262d", "color": "#c9d1d9", "borderColor": "#30363d"}),
                                        ],
                                        className="mb-2",
                                    ),
                                    dbc.InputGroup(
                                        [
                                            dbc.InputGroupText("价格", style={"backgroundColor": "#30363d", "color": "#8b949e", "borderColor": "#30363d"}),
                                            dbc.Input(type="number", placeholder="限价", style={"backgroundColor": "#21262d", "color": "#c9d1d9", "borderColor": "#30363d"}),
                                        ],
                                        className="mb-2",
                                    ),
                                    dbc.RadioItems(
                                        options=[{"label": "买入", "value": "buy"}, {"label": "卖出", "value": "sell"}],
                                        value="buy",
                                        inline=True,
                                        className="mb-3",
                                        label_style={"color": "#c9d1d9"},
                                    ),
                                    dbc.Button("提交订单", color="primary", className="w-100", disabled=True),
                                    html.Small("⚠️ 模拟模式，订单不会真实执行", className="text-muted d-block mt-2"),
                                ],
                                style={"backgroundColor": "#161b22", "color": "#c9d1d9"},
                            ),
                        ],
                        style={"border": "1px solid #30363d"},
                    ),
                    width=4,
                ),
            ],
            className="mb-3 g-3",
        ),

        # PnL calendar placeholder
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("📅 盈亏日历", style={"backgroundColor": "#21262d", "color": "#c9d1d9", "fontWeight": "bold"}),
                            dbc.CardBody(
                                html.P("实盘接入后展示每日盈亏热力图", className="text-muted text-center"),
                                style={"backgroundColor": "#161b22", "color": "#c9d1d9"},
                            ),
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
