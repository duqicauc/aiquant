"""
个股分析工具 — AgentScope Tool 封装 (v1.0.19)
"""

import json

from agentscope.message import TextBlock
from agentscope.tool import ToolResponse

from src.analysis.technical_indicators import calculate_cmf, calculate_mfi, calculate_vwap
from src.data.arctic_provider import ArcticDataProvider
from src.data.fetcher.tushare_fetcher import TushareFetcher


def _json_to_text(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def query_stock_kline(ts_code: str, days: int = 60) -> ToolResponse:
    """
    查询个股近N日K线数据（OHLCV）。

    Args:
        ts_code: 股票代码，如 000001.SZ
        days: 返回天数，默认60
    """
    try:
        from datetime import datetime, timedelta

        end = datetime.now().strftime("%Y%m%d")
        start = (datetime.now() - timedelta(days=days + 30)).strftime("%Y%m%d")

        provider = ArcticDataProvider()
        df = provider.read_daily_ohlcv(start, end)
        if df.empty:
            return ToolResponse(content=[TextBlock(text="无数据")])

        df = df[df["ts_code"] == ts_code].sort_values("trade_date").tail(days)
        records = []
        for _, row in df.iterrows():
            records.append(
                {
                    "date": str(row.get("trade_date", "")),
                    "open": float(row.get("open", 0)),
                    "high": float(row.get("high", 0)),
                    "low": float(row.get("low", 0)),
                    "close": float(row.get("close", 0)),
                    "vol": float(row.get("vol", 0)),
                    "pct_chg": float(row.get("pct_chg", 0)),
                }
            )
        return ToolResponse(
            content=[TextBlock(text=_json_to_text({"ts_code": ts_code, "count": len(records), "data": records}))]
        )
    except Exception as e:
        return ToolResponse(content=[TextBlock(text=f"查询失败: {e}")])


def query_stock_technical(ts_code: str, days: int = 60) -> ToolResponse:
    """
    查询个股技术指标（VWAP、CMF、MFI）。

    Args:
        ts_code: 股票代码
        days: 分析天数
    """
    try:
        resp = query_stock_kline(ts_code, days)
        # ToolResponse content 是 TextBlock 列表，提取 text
        text = resp.content[0].text if resp.content else "{}"
        data_obj = json.loads(text)
        data = data_obj.get("data", [])
        if not data:
            return ToolResponse(content=[TextBlock(text="数据不足")])
        if len(data) < 20:
            return ToolResponse(content=[TextBlock(text="数据不足20天")])

        import pandas as pd

        df = pd.DataFrame(data)

        signals = {}
        try:
            vwap = calculate_vwap(df)
            signals["VWAP"] = vwap
        except Exception:
            pass
        try:
            cmf = calculate_cmf(df, period=20)
            signals["CMF(20)"] = cmf
        except Exception:
            pass
        try:
            mfi = calculate_mfi(df, period=14)
            signals["MFI(14)"] = mfi
        except Exception:
            pass

        latest = data[-1]
        prev = data[-2] if len(data) >= 2 else latest
        trend = "上涨" if latest["close"] > prev["close"] else "下跌"
        ma20 = df["close"].rolling(20).mean().iloc[-1]
        above_ma20 = latest["close"] > ma20

        payload = {
            "ts_code": ts_code,
            "latest_price": latest["close"],
            "latest_pct": latest["pct_chg"],
            "trend": trend,
            "above_ma20": above_ma20,
            "ma20": round(float(ma20), 2) if pd.notna(ma20) else None,
            "technical_signals": signals,
        }
        return ToolResponse(content=[TextBlock(text=_json_to_text(payload))])
    except Exception as e:
        return ToolResponse(content=[TextBlock(text=f"查询失败: {e}")])


def query_stock_moneyflow(ts_code: str, days: int = 5) -> ToolResponse:
    """
    查询个股近N日资金流向（主力净流入）。

    Args:
        ts_code: 股票代码
        days: 天数，默认5
    """
    try:
        fetcher = TushareFetcher()
        from datetime import datetime, timedelta

        end = datetime.now()
        start = end - timedelta(days=days + 10)

        df = fetcher.get_moneyflow(
            ts_code=ts_code,
            start_date=start.strftime("%Y%m%d"),
            end_date=end.strftime("%Y%m%d"),
        )
        if df is None or df.empty:
            return ToolResponse(content=[TextBlock(text="无资金流向数据")])

        df = df.sort_values("trade_date").tail(days)
        records = []
        for _, row in df.iterrows():
            records.append(
                {
                    "date": str(row.get("trade_date", "")),
                    "net_mf_amount": float(row.get("net_mf_amount", 0)),
                    "buy_elg_amount": float(row.get("buy_elg_amount", 0)),
                    "sell_elg_amount": float(row.get("sell_elg_amount", 0)),
                }
            )

        total_net = sum(r["net_mf_amount"] for r in records)
        payload = {
            "ts_code": ts_code,
            "days": len(records),
            "total_net_mf": round(total_net, 2),
            "avg_daily_net": round(total_net / len(records), 2) if records else 0,
            "detail": records,
        }
        return ToolResponse(content=[TextBlock(text=_json_to_text(payload))])
    except Exception as e:
        return ToolResponse(content=[TextBlock(text=f"查询失败: {e}")])
