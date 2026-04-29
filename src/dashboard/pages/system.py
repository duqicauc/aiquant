
from src.dashboard.app import app
"""
系统管理页 (System Management Page)
监控状态 | 日志查看 | 参数配置 | 模型管理
"""
import sys
from pathlib import Path

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import dcc, html
from dash.dependencies import Input, Output

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def _monitor_tab():
    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(dbc.Card(id="sys-psi-card", style={"backgroundColor": "#161b22", "border": "1px solid #30363d"}), width=4),
                    dbc.Col(dbc.Card(id="sys-winrate-card", style={"backgroundColor": "#161b22", "border": "1px solid #30363d"}), width=4),
                    dbc.Col(dbc.Card(id="sys-coverage-card", style={"backgroundColor": "#161b22", "border": "1px solid #30363d"}), width=4),
                ],
                className="mb-3 g-3",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("PSI 趋势", style={"backgroundColor": "#21262d", "color": "#c9d1d9"}),
                                dbc.CardBody(
                                    dcc.Graph(id="sys-psi-chart", config={"displayModeBar": False}, style={"height": "250px"}),
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
                                dbc.CardHeader("胜率趋势", style={"backgroundColor": "#21262d", "color": "#c9d1d9"}),
                                dbc.CardBody(
                                    dcc.Graph(id="sys-winrate-chart", config={"displayModeBar": False}, style={"height": "250px"}),
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
def _logs_tab():
    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Textarea(
                            id="sys-log-text",
                            value="系统日志加载中...",
                            style={
                                "width": "100%",
                                "height": "500px",
                                "backgroundColor": "#0d1117",
                                "color": "#c9d1d9",
                                "border": "1px solid #30363d",
                                "fontFamily": "monospace",
                                "fontSize": "0.85rem",
                            },
                            readOnly=True,
                        ),
                        width=12,
                    ),
                ]
            ),
        ]
    )
def _config_tab():
    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("策略参数", style={"backgroundColor": "#21262d", "color": "#c9d1d9", "fontWeight": "bold"}),
                                dbc.CardBody(
                                    [
                                        _config_item("初始资金", "500,000"),
                                        _config_item("单票基础金额", "50,000"),
                                        _config_item("TopN 买入", "10"),
                                        _config_item("止损比例", "4%"),
                                        _config_item("移动止盈回撤", "5%"),
                                        _config_item("强牛单票上限", "10%"),
                                        _config_item("震荡单票上限", "6%"),
                                        _config_item("市场环境阈值", "1.3"),
                                    ],
                                    style={"backgroundColor": "#161b22", "color": "#c9d1d9"},
                                ),
                            ],
                            style={"border": "1px solid #30363d"},
                        ),
                        width=6,
                    ),
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("费用参数", style={"backgroundColor": "#21262d", "color": "#c9d1d9", "fontWeight": "bold"}),
                                dbc.CardBody(
                                    [
                                        _config_item("买入滑点", "15bp"),
                                        _config_item("卖出滑点", "20bp"),
                                        _config_item("佣金费率", "0.03%"),
                                        _config_item("印花税", "0.1% (卖出)"),
                                        _config_item("过户费", "0.001%"),
                                    ],
                                    style={"backgroundColor": "#161b22", "color": "#c9d1d9"},
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
def _models_tab():
    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("模型版本", style={"backgroundColor": "#21262d", "color": "#c9d1d9", "fontWeight": "bold"}),
                                dbc.CardBody(
                                    html.Div(
                                        [
                                            dbc.ListGroup(
                                                [
                                                    dbc.ListGroupItem(
                                                        [
                                                            html.Strong("v2.9.1-ensemble", className="text-success"),
                                                            html.Span(" — 当前生产模型", className="text-muted ms-2"),
                                                            dbc.Badge("生产中", color="success", className="ms-2"),
                                                        ],
                                                        style={"backgroundColor": "#21262d", "color": "#c9d1d9", "borderColor": "#30363d"},
                                                    ),
                                                    dbc.ListGroupItem(
                                                        [
                                                            html.Strong("v2.7.0-ensemble", className="text-info"),
                                                            html.Span(" — Fallback 模型", className="text-muted ms-2"),
                                                            dbc.Badge("备用", color="info", className="ms-2"),
                                                        ],
                                                        style={"backgroundColor": "#21262d", "color": "#c9d1d9", "borderColor": "#30363d"},
                                                    ),
                                                    dbc.ListGroupItem(
                                                        [
                                                            html.Strong("v2.9.2-catboost", className="text-warning"),
                                                            html.Span(" — 测试候选", className="text-muted ms-2"),
                                                            dbc.Badge("测试", color="warning", className="ms-2"),
                                                        ],
                                                        style={"backgroundColor": "#21262d", "color": "#c9d1d9", "borderColor": "#30363d"},
                                                    ),
                                                ],
                                                flush=True,
                                            ),
                                        ]
                                    ),
                                    style={"backgroundColor": "#161b22"},
                                ),
                            ],
                            style={"border": "1px solid #30363d"},
                        ),
                        width=12,
                    ),
                ]
            ),
        ]
    )
def _config_item(label, value):
    return html.Div(
        [
            html.Span(label, style={"display": "inline-block", "width": "150px", "color": "#8b949e"}),
            html.Strong(value, style={"color": "#c9d1d9"}),
        ],
        style={"marginBottom": "8px"},
    )

layout = html.Div(
    [
        html.H1("⚙️ 系统管理", className="mb-3", style={"color": "#c9d1d9", "fontWeight": "bold"}),

        # Tabs
        dbc.Tabs(
            [
                dbc.Tab(label="📡 监控状态", tab_id="tab-monitor", children=_monitor_tab()),
                dbc.Tab(label="📜 系统日志", tab_id="tab-logs", children=_logs_tab()),
                dbc.Tab(label="🔧 参数配置", tab_id="tab-config", children=_config_tab()),
                dbc.Tab(label="🤖 模型管理", tab_id="tab-models", children=_models_tab()),
            ],
            id="system-tabs",
            active_tab="tab-monitor",
            className="mb-3",
        ),

        dcc.Interval(id="system-interval", interval=30 * 1000, n_intervals=0),
    ]
)












# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

@app.callback(
    [
        Output("sys-psi-card", "children"),
        Output("sys-winrate-card", "children"),
        Output("sys-coverage-card", "children"),
        Output("sys-psi-chart", "figure"),
        Output("sys-winrate-chart", "figure"),
    ],
    Input("system-interval", "n_intervals"),
)
def update_system_monitor(n):
    """Update system monitoring cards and charts."""
    from src.monitoring.model_monitor import ModelMonitor

    try:
        monitor_dir = project_root / "data" / "prediction"
        results_dir = project_root / "data" / "results"
        monitor = ModelMonitor(prediction_dir=monitor_dir, results_dir=results_dir)
        tq = monitor.check_trade_quality(lookback_days=7)
        wr = tq.get("avg_win_rate", 0)
        pr = tq.get("avg_profit_ratio", 0)
    except Exception:
        wr, pr = 0, 0

    psi_card = dbc.CardBody([
        html.H5("PSI (预测漂移)", className="text-muted"),
        html.H3("N/A", className="text-success"),
        html.Small("数据不足", className="text-muted"),
    ])

    wr_color = "text-success" if wr >= 0.45 else "text-warning" if wr >= 0.30 else "text-danger"
    wr_card = dbc.CardBody([
        html.H5("近7日胜率", className="text-muted"),
        html.H3(f"{wr * 100:.1f}%", className=wr_color),
        html.Small("目标: ≥45%", className="text-muted"),
    ])

    coverage_card = dbc.CardBody([
        html.H5("预测覆盖率", className="text-muted"),
        html.H3("--", className="text-info"),
        html.Small("目标: 全市场", className="text-muted"),
    ])

    # Placeholder charts
    empty_fig = go.Figure().update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        annotations=[{"text": "数据积累中...", "showarrow": False, "font": {"size": 14, "color": "#8b949e"}}],
        margin=dict(l=20, r=20, t=20, b=20),
    )

    return psi_card, wr_card, coverage_card, empty_fig, empty_fig


@app.callback(
    Output("sys-log-text", "value"),
    Input("system-interval", "n_intervals"),
)
def update_logs(n):
    """Load recent system logs."""
    try:
        log_file = project_root / "logs" / "aiquant.log"
        if not log_file.exists():
            return "日志文件不存在"
        with open(log_file, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        return "".join(lines[-100:])
    except Exception as e:
        return f"读取日志失败: {str(e)}"
