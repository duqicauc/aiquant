"""
Stock data API endpoints.
Provides stock diagnosis, K-line data, and technical analysis.
"""
import math
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.api.dependencies import get_data_manager
from src.api.schemas.stock import StockDiagnosisResponse

router = APIRouter()


import math

def _sanitize_for_json(obj):
    """Recursively convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    elif isinstance(obj, (np.bool_, np.bool)):
        return bool(obj)
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        val = float(obj)
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    elif isinstance(obj, np.ndarray):
        return _sanitize_for_json(obj.tolist())
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    return obj


@router.get("/{ts_code}/kline")
async def get_stock_kline(
    ts_code: str,
    days: int = Query(120, ge=30, le=500),
    include_ma: bool = Query(True, description="Include moving averages"),
):
    """Get stock K-line data with optional moving averages."""
    try:
        import tushare as ts
        dm = get_data_manager()
        pro = ts.pro_api()
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=days * 2)).strftime("%Y%m%d")

        # Fetch directly from Tushare API to avoid stale cache
        df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        if df is None or df.empty:
            df = dm.get_daily_data(ts_code, start_date, end_date)
        if df is None or df.empty:
            raise HTTPException(status_code=404, detail=f"No data for {ts_code}")

        df = df.sort_values("trade_date").tail(days).reset_index(drop=True)

        if include_ma:
            df["ma5"] = df["close"].rolling(5).mean().round(2)
            df["ma10"] = df["close"].rolling(10).mean().round(2)
            df["ma20"] = df["close"].rolling(20).mean().round(2)
            df["ma60"] = df["close"].rolling(60).mean().round(2)
            df["ma120"] = df["close"].rolling(120).mean().round(2)
            df["ma233"] = df["close"].rolling(233).mean().round(2)

        records = []
        for _, row in df.iterrows():
            rec = {
                "date": str(row["trade_date"]),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row.get("vol", 0)),
                "amount": float(row.get("amount", 0)),
            }
            if include_ma:
                rec.update({
                    "ma5": float(row["ma5"]) if pd.notna(row.get("ma5")) else None,
                    "ma10": float(row["ma10"]) if pd.notna(row.get("ma10")) else None,
                    "ma20": float(row["ma20"]) if pd.notna(row.get("ma20")) else None,
                    "ma60": float(row["ma60"]) if pd.notna(row.get("ma60")) else None,
                    "ma120": float(row["ma120"]) if pd.notna(row.get("ma120")) else None,
                    "ma233": float(row["ma233"]) if pd.notna(row.get("ma233")) else None,
                })
            records.append(rec)

        return {"ts_code": ts_code, "data": records, "count": len(records)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Kline failed: {str(e)}")


@router.get("/{ts_code}/diagnosis")
async def get_stock_diagnosis(
    ts_code: str,
    days: int = Query(120, ge=30, le=500),
):
    """Get full stock diagnosis report."""
    try:
        from src.analysis.stock_health_checker import StockHealthChecker

        checker = StockHealthChecker()
        report = checker.check_stock(ts_code, days)

        if "error" in report:
            raise HTTPException(status_code=500, detail=report["error"])

        basic = report.get("basic_info", {})
        tech = report.get("technical_analysis", {})
        model = report.get("model_prediction", {})
        risk = report.get("risk_assessment", {})
        signals = report.get("trading_signals", {})
        swing = report.get("swing_plan", {})

        return StockDiagnosisResponse(
            ts_code=ts_code,
            name=basic.get("name"),
            overall_score=report.get("overall_score", 0),
            recommendation=report.get("recommendation", ""),
            basic_info=basic,
            technical=tech,
            model_prediction=model,
            risk_assessment=risk,
            trading_signals=signals,
            swing_plan=swing if swing else None,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Diagnosis failed: {str(e)}")


@router.get("/{ts_code}/basic")
async def get_stock_basic(ts_code: str):
    """Get basic stock info."""
    try:
        dm = get_data_manager()
        stock_list = dm.get_stock_list()

        info = stock_list[stock_list["ts_code"] == ts_code]
        if info.empty:
            raise HTTPException(status_code=404, detail=f"Stock {ts_code} not found")

        row = info.iloc[0]
        return {
            "ts_code": ts_code,
            "name": row.get("name", ""),
            "industry": row.get("industry", ""),
            "market": row.get("market", ""),
            "area": row.get("area", ""),
            "list_date": row.get("list_date", ""),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Basic info failed: {str(e)}")


@router.get("/{ts_code}/advanced-indicators")
async def get_advanced_indicators(
    ts_code: str,
    days: int = Query(120, ge=30, le=500),
    period: str = Query("daily", description="daily, weekly, or monthly"),
    indicators: Optional[str] = Query(None, description="Comma-separated filter: vwap,cmf,mfi,pvo,ad_line,volume_profile,adx_dmi,supertrend,ichimoku,sar,atr_channel,harmonic,fractals"),
):
    """
    Get advanced technical indicators for a stock.
    Includes volume-price analysis, advanced trend, pattern recognition, MTFA, and moneyflow.
    """
    try:
        import tushare as ts

        from src.analysis.technical_indicators import calculate_all_indicators
        from src.analysis.mtfa import analyze_resonance
        from src.analysis.moneyflow_analysis import analyze_full_moneyflow
        from src.dashboard.pages.research import _resample_to_monthly

        dm = get_data_manager()
        pro = ts.pro_api()
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=days * 2)).strftime("%Y%m%d")

        # Fetch primary data directly from Tushare API to avoid stale cache
        if period == "weekly":
            df = dm.get_weekly_data(ts_code, start_date, end_date)
        else:
            df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
            if df is None or df.empty:
                df = dm.get_daily_data(ts_code, start_date, end_date)
            if period == "monthly":
                df = _resample_to_monthly(df)

        if df is None or df.empty:
            raise HTTPException(status_code=404, detail=f"No data for {ts_code}")

        df = df.sort_values("trade_date").reset_index(drop=True)

        # Calculate all indicators
        all_results = calculate_all_indicators(df)

        # Filter if requested
        if indicators:
            keep = [i.strip() for i in indicators.split(",")]
            filtered = {k: v for k, v in all_results.items() if k in keep}
        else:
            filtered = all_results

        # MTFA (only for daily)
        mtfa_result = None
        if period == "daily":
            try:
                df_weekly = dm.get_weekly_data(ts_code, start_date, end_date)
                df_monthly_raw = dm.get_daily_data(ts_code, (datetime.now() - timedelta(days=days * 10)).strftime("%Y%m%d"), end_date)
                df_monthly = None
                if df_monthly_raw is not None and not df_monthly_raw.empty:
                    df_monthly = _resample_to_monthly(df_monthly_raw)
                mtfa_result = analyze_resonance(df, df_weekly, df_monthly)
            except Exception:
                pass

        # Moneyflow (reuse existing DataManager to avoid re-initialization overhead)
        moneyflow_result = None
        try:
            moneyflow_result = analyze_full_moneyflow(ts_code, days=10, dm=dm)
        except Exception:
            pass

        return _sanitize_for_json({
            "ts_code": ts_code,
            "period": period,
            "indicators": filtered,
            "mtfa": mtfa_result,
            "moneyflow": moneyflow_result,
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Advanced indicators failed: {str(e)}")


@router.get("/{ts_code}/moneyflow-detail")
async def get_stock_moneyflow_detail(
    ts_code: str,
    days: int = Query(30, ge=5, le=120),
):
    """Get detailed moneyflow for a stock (super-large/large/medium/small orders). Tushare primary."""
    try:
        import tushare as ts

        pro = ts.pro_api()
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=days * 2)).strftime("%Y%m%d")

        df = pro.moneyflow(ts_code=ts_code, start_date=start_date, end_date=end_date)
        if df is None or df.empty:
            raise HTTPException(status_code=404, detail=f"No moneyflow data for {ts_code}")

        df = df.sort_values("trade_date").reset_index(drop=True)

        records = []
        for _, row in df.iterrows():
            records.append({
                "date": str(row["trade_date"]),
                "buy_elg": _sanitize_for_json(float(row.get("buy_elg_amount", 0))),   # 特大单买入
                "sell_elg": _sanitize_for_json(float(row.get("sell_elg_amount", 0))), # 特大单卖出
                "buy_lg": _sanitize_for_json(float(row.get("buy_lg_amount", 0))),     # 大单买入
                "sell_lg": _sanitize_for_json(float(row.get("sell_lg_amount", 0))),   # 大单卖出
                "buy_md": _sanitize_for_json(float(row.get("buy_md_amount", 0))),     # 中单买入
                "sell_md": _sanitize_for_json(float(row.get("sell_md_amount", 0))),   # 中单卖出
                "buy_sm": _sanitize_for_json(float(row.get("buy_sm_amount", 0))),     # 小单买入
                "sell_sm": _sanitize_for_json(float(row.get("sell_sm_amount", 0))),   # 小单卖出
                "net_mf": _sanitize_for_json(float(row.get("net_mf_amount", 0))),     # 主力净流入
            })

        # Calculate trends
        net_series = [r["net_mf"] for r in records if r["net_mf"] is not None]
        consecutive_inflow = 0
        for v in reversed(net_series):
            if v > 0:
                consecutive_inflow += 1
            else:
                break

        total_elg = sum(r["buy_elg"] - r["sell_elg"] for r in records if r["buy_elg"] is not None)
        total_lg = sum(r["buy_lg"] - r["sell_lg"] for r in records if r["buy_lg"] is not None)
        total_md = sum(r["buy_md"] - r["sell_md"] for r in records if r["buy_md"] is not None)
        total_sm = sum(r["buy_sm"] - r["sell_sm"] for r in records if r["buy_sm"] is not None)

        return {
            "ts_code": ts_code,
            "days": len(records),
            "data": records,
            "summary": {
                "consecutive_inflow_days": consecutive_inflow,
                "total_net_mf": round(sum(net_series), 2) if net_series else 0,
                "super_large_net": round(total_elg, 2),
                "large_net": round(total_lg, 2),
                "medium_net": round(total_md, 2),
                "small_net": round(total_sm, 2),
            },
            "update_time": datetime.now().isoformat(),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Moneyflow detail failed: {str(e)}")


@router.get("/{ts_code}/technical")
async def get_stock_technical(
    ts_code: str,
    days: int = Query(120, ge=30, le=500),
):
    """Get Tushare official technical indicators (MACD/KDJ/RSI/BOLL/CCI)."""
    try:
        import tushare as ts

        pro = ts.pro_api()
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=days * 2)).strftime("%Y%m%d")

        df = pro.stk_factor(
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
            fields="ts_code,trade_date,close,macd_dif,macd_dea,macd,kdj_k,kdj_d,kdj_j,rsi_6,rsi_12,rsi_24,boll_upper,boll_mid,boll_lower,cci",
        )
        if df is None or df.empty:
            raise HTTPException(status_code=404, detail=f"No technical data for {ts_code}")

        df = df.sort_values("trade_date").reset_index(drop=True)

        records = []
        for _, row in df.iterrows():
            records.append({
                "date": str(row["trade_date"]),
                "close": _sanitize_for_json(float(row.get("close", 0))),
                "macd": {
                    "dif": _sanitize_for_json(float(row.get("macd_dif", 0))),
                    "dea": _sanitize_for_json(float(row.get("macd_dea", 0))),
                    "macd": _sanitize_for_json(float(row.get("macd", 0))),
                },
                "kdj": {
                    "k": _sanitize_for_json(float(row.get("kdj_k", 0))),
                    "d": _sanitize_for_json(float(row.get("kdj_d", 0))),
                    "j": _sanitize_for_json(float(row.get("kdj_j", 0))),
                },
                "rsi": {
                    "rsi_6": _sanitize_for_json(float(row.get("rsi_6", 0))),
                    "rsi_12": _sanitize_for_json(float(row.get("rsi_12", 0))),
                    "rsi_24": _sanitize_for_json(float(row.get("rsi_24", 0))),
                },
                "boll": {
                    "upper": _sanitize_for_json(float(row.get("boll_upper", 0))),
                    "mid": _sanitize_for_json(float(row.get("boll_mid", 0))),
                    "lower": _sanitize_for_json(float(row.get("boll_lower", 0))),
                },
                "cci": _sanitize_for_json(float(row.get("cci", 0))),
            })

        # Latest signal detection
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest

        macd_signal = "中性"
        if len(df) >= 2:
            if float(prev.get("macd_dif", 0)) <= float(prev.get("macd_dea", 0)) and float(latest.get("macd_dif", 0)) > float(latest.get("macd_dea", 0)):
                macd_signal = "金叉"
            elif float(prev.get("macd_dif", 0)) >= float(prev.get("macd_dea", 0)) and float(latest.get("macd_dif", 0)) < float(latest.get("macd_dea", 0)):
                macd_signal = "死叉"

        k = float(latest.get("kdj_k", 50))
        d = float(latest.get("kdj_d", 50))
        kdj_signal = "超买区" if k > 80 and d > 80 else "超卖区" if k < 20 and d < 20 else "金叉" if k > d else "死叉"

        rsi6 = float(latest.get("rsi_6", 50))
        rsi_signal = "超买" if rsi6 > 70 else "超卖" if rsi6 < 30 else "中性"

        return {
            "ts_code": ts_code,
            "days": len(records),
            "data": records,
            "latest_signals": {
                "macd": macd_signal,
                "kdj": kdj_signal,
                "rsi": rsi_signal,
                "close": _sanitize_for_json(float(latest.get("close", 0))),
            },
            "update_time": datetime.now().isoformat(),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Technical indicators failed: {str(e)}")


@router.get("/{ts_code}/lhb-detail")
async def get_stock_lhb_detail(
    ts_code: str,
    days: int = Query(30, ge=5, le=90),
):
    """Get Dragon-Tiger List detail with institution analysis for a specific stock."""
    try:
        import tushare as ts

        pro = ts.pro_api()
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")

        # Get trade calendar to find valid trade dates
        cal = pro.trade_cal(exchange="SSE", start_date=start_date, end_date=end_date, is_open="1")
        trade_dates = cal["cal_date"].tolist() if cal is not None and not cal.empty else []

        all_list = []
        all_inst = []
        for td in trade_dates:
            try:
                df_list = pro.top_list(ts_code=ts_code, trade_date=td)
                if df_list is not None and not df_list.empty:
                    all_list.append(df_list)
            except Exception:
                pass
            try:
                df_inst = pro.top_inst(ts_code=ts_code, trade_date=td)
                if df_inst is not None and not df_inst.empty:
                    all_inst.append(df_inst)
            except Exception:
                pass

        df_list = pd.concat(all_list, ignore_index=True) if all_list else pd.DataFrame()
        df_inst = pd.concat(all_inst, ignore_index=True) if all_inst else pd.DataFrame()

        if df_list.empty and df_inst.empty:
            raise HTTPException(status_code=404, detail=f"No LHB data for {ts_code}")

        # Institution analysis
        inst_buy = 0.0
        inst_sell = 0.0
        inst_records = []
        if not df_inst.empty:
            for _, row in df_inst.iterrows():
                buy = float(row.get("buy", 0) or 0)
                sell = float(row.get("sell", 0) or 0)
                inst_buy += buy
                inst_sell += sell
                inst_records.append({
                    "date": str(row.get("trade_date", "")),
                    "exalter": str(row.get("exalter", "")),
                    "buy": round(buy / 1e4, 2),    # 万元
                    "sell": round(sell / 1e4, 2),
                    "net_buy": round((buy - sell) / 1e4, 2),
                    "side": "买入" if str(row.get("side", "")) == "0" else "卖出",
                    "reason": str(row.get("reason", "")),
                })

        # Famous dealers identification
        famous_dealers = {
            "东方财富拉萨": "散户集中营（拉萨帮）",
            "中金公司": "量化/机构",
            "华泰证券": "游资",
            "国泰君安": "游资",
            "中信证券": "机构/游资",
        }
        dealer_tags = []
        if not df_inst.empty:
            for ex in df_inst["exalter"].astype(str).unique():
                for key, tag in famous_dealers.items():
                    if key in ex:
                        dealer_tags.append({"exalter": ex, "tag": tag})
                        break

        return {
            "ts_code": ts_code,
            "days": days,
            "institution_summary": {
                "inst_buy": round(inst_buy / 1e4, 2),
                "inst_sell": round(inst_sell / 1e4, 2),
                "inst_net": round((inst_buy - inst_sell) / 1e4, 2),
                "inst_count": len(inst_records),
            },
            "institution_details": inst_records,
            "dealer_tags": dealer_tags,
            "update_time": datetime.now().isoformat(),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LHB detail failed: {str(e)}")
