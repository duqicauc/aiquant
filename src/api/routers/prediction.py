"""
Prediction API endpoints.
Provides latest predictions, historical tracking, and model comparison.
"""
import sys
from pathlib import Path
from typing import Any, List, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

router = APIRouter()

# 简单内存缓存：pipeline-status
_pipeline_status_cache = {"data": None, "timestamp": 0}
_PIPELINE_STATUS_CACHE_TTL = 30  # 秒


def _compute_distribution(df: Any, prob_col: str = "prob") -> dict:
    """计算概率分布（7分箱）"""
    if df is None or df.empty or prob_col not in df.columns:
        return {}
    probs = df[prob_col].dropna()
    if probs.max() > 1:
        probs = probs / 100
    bins = [
        {"label": "0-2%", "min": 0.0, "max": 0.02},
        {"label": "2-5%", "min": 0.02, "max": 0.05},
        {"label": "5-10%", "min": 0.05, "max": 0.10},
        {"label": "10-20%", "min": 0.10, "max": 0.20},
        {"label": "20-50%", "min": 0.20, "max": 0.50},
        {"label": "50-80%", "min": 0.50, "max": 0.80},
        {"label": "80-100%", "min": 0.80, "max": 1.00},
    ]
    total = len(probs)
    result = []
    for b in bins:
        count = int(((probs >= b["min"]) & (probs < b["max"])).sum())
        result.append({
            "label": b["label"],
            "count": count,
            "pct": round(count / total * 100, 2) if total > 0 else 0,
        })
    # 80-100% 包含等于1.0的情况
    last_count = int((probs >= 0.80).sum())
    result[-1]["count"] = last_count
    result[-1]["pct"] = round(last_count / total * 100, 2) if total > 0 else 0
    return {"total": total, "bins": result}


def _find_all_csv(pred_dirs: list, date_str: str) -> Optional[Path]:
    """根据日期查找对应的 _all.csv（优先 enriched）"""
    for pred_dir in pred_dirs:
        if not pred_dir.exists():
            continue
        f_enriched = pred_dir / f"predictions_{date_str}_all_enriched.csv"
        if f_enriched.exists():
            return f_enriched
        f = pred_dir / f"predictions_{date_str}_all.csv"
        if f.exists():
            return f
    return None


@router.get("/latest")
async def get_latest_predictions(
    top_n: int = Query(50, ge=1, le=200),
    min_prob: Optional[float] = Query(None, ge=0, le=1),
    min_mv: Optional[float] = Query(None, ge=0, description="最小总市值(亿元)"),
    max_mv: Optional[float] = Query(None, ge=0, description="最大总市值(亿元)"),
    min_turnover: Optional[float] = Query(None, ge=0, description="最小换手率(%)"),
):
    """Get latest prediction results."""
    # Unwrap Query parameters when called directly (not via FastAPI router)
    top_n_val = top_n.default if hasattr(top_n, "default") else top_n
    min_prob_val = min_prob.default if hasattr(min_prob, "default") else min_prob
    min_mv_val = min_mv.default if hasattr(min_mv, "default") else min_mv
    max_mv_val = max_mv.default if hasattr(max_mv, "default") else max_mv
    min_turnover_val = min_turnover.default if hasattr(min_turnover, "default") else min_turnover

    try:
        import pandas as pd

        # Try multiple directories for prediction data (v3.1.0优先)
        pred_dirs = [
            project_root / "data" / "prediction" / "v3.1.0_daily",
            project_root / "data" / "prediction" / "v3.1.0",
            project_root / "data" / "prediction" / "v3.0.0",
            project_root / "data" / "prediction" / "v3.0.0_daily",
            project_root / "data" / "prediction" / "v295_stk_factor_2026q1q2",
            project_root / "data" / "prediction" / "v295_stk_factor",
            project_root / "data" / "prediction" / "v294_stk_factor",
            project_root / "data" / "prediction" / "v294_daily",
            project_root / "data" / "prediction" / "v291_integrated",
            project_root / "data" / "prediction" / "v291_stk_factor",
            project_root / "data" / "prediction",
        ]

        # Gather candidate files across all directories, then pick the latest by date in filename
        # 优先 enriched 文件，其次普通文件
        candidates = []
        for pred_dir in pred_dirs:
            if not pred_dir.exists():
                continue
            # 优先 enriched 格式
            candidates.extend(pred_dir.glob("predictions_*_top50_enriched.csv"))
            candidates.extend(pred_dir.glob("predictions_*_top100_enriched.csv"))
            candidates.extend(pred_dir.glob("*integrated_top50_enriched.csv"))
            # v291_integrated format
            candidates.extend(pred_dir.glob("*integrated_top50.csv"))
            # v291_stk_factor format
            candidates.extend(pred_dir.glob("predictions_*_top50.csv"))
            candidates.extend(pred_dir.glob("predictions_*_top100.csv"))
            # legacy formats
            candidates.extend(pred_dir.glob("top_*_advanced_*.csv"))
            candidates.extend(pred_dir.glob("stock_scores_*.csv"))

        # Extract date from filename and sort descending
        def _extract_date(path: Path):
            name = path.name
            # predictions_YYYYMMDD_... or top_YYYYMMDD_advanced_...
            parts = name.split("_")
            if len(parts) >= 2 and parts[1].isdigit() and len(parts[1]) == 8:
                return parts[1]
            return "00000000"

        candidates = sorted(candidates, key=lambda p: _extract_date(p), reverse=True)

        df = None
        filename = ""
        for cand in candidates:
            try:
                df = pd.read_csv(cand)
                filename = cand.name
                break
            except Exception:
                continue

        if df is None or df.empty:
            raise HTTPException(status_code=404, detail="No prediction results available")

        # Normalize probability/score column
        prob_candidates = ["prob", "probability", "adjusted_score", "牛股概率"]
        prob_col = None
        for c in prob_candidates:
            if c in df.columns:
                prob_col = c
                break

        if prob_col is None:
            raise HTTPException(status_code=404, detail="Prediction file has no recognizable probability column")

        if prob_col in df.columns:
            # Ensure probability is in [0, 1] range
            if df[prob_col].max() > 1:
                df[prob_col] = df[prob_col] / 100
            # Unwrap Query parameter if called directly (not via FastAPI router)
            min_prob_val = min_prob
            if hasattr(min_prob, "default"):
                min_prob_val = min_prob.default
            if min_prob_val is not None:
                df = df[df[prob_col] >= min_prob_val]

            # ---------- 市值与换手率筛选 ----------
            # total_mv in CSV is 万元; params are 亿元
            if min_mv_val is not None and "total_mv" in df.columns:
                df = df[df["total_mv"] >= min_mv_val * 10000]
            if max_mv_val is not None and "total_mv" in df.columns:
                df = df[df["total_mv"] <= max_mv_val * 10000]
            if min_turnover_val is not None and "turnover_rate" in df.columns:
                df = df[df["turnover_rate"] >= min_turnover_val]

            df = df.sort_values(prob_col, ascending=False).head(top_n_val)

        # Enrich with stock name and industry from tushare
        try:
            import tushare as ts
            pro = ts.pro_api()
            stock_basic = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,industry')
            if stock_basic is not None and not stock_basic.empty:
                # Drop existing name/industry columns to avoid _x/_y suffixes
                for col in ['name', 'industry']:
                    if col in df.columns:
                        df = df.drop(columns=[col])
                df = df.merge(stock_basic, on='ts_code', how='left')
        except Exception:
            pass

        # Format period date
        raw_period = filename.split("_")[1] if "_" in filename else "unknown"
        if raw_period != "unknown" and len(raw_period) == 8 and raw_period.isdigit():
            formatted_period = f"{raw_period[:4]}-{raw_period[4:6]}-{raw_period[6:8]}"
            display_period = f"{int(raw_period[4:6])}月{int(raw_period[6:8])}日"
        else:
            formatted_period = raw_period
            display_period = raw_period

        # ---------- 全市场概率分布 + 概率分位 ----------
        full_dist = {}
        prob_percentile_map = {}
        all_csv = _find_all_csv(pred_dirs, raw_period)
        if all_csv and all_csv.exists():
            try:
                df_all = pd.read_csv(all_csv)
                full_dist = _compute_distribution(df_all, prob_col)
                # 基于全市场计算概率分位
                if prob_col in df_all.columns and not df_all.empty:
                    probs_all = df_all[prob_col].dropna()
                    if len(probs_all) > 0:
                        if probs_all.max() > 1:
                            probs_all = probs_all / 100
                        sorted_probs = probs_all.sort_values(ascending=False).reset_index(drop=True)
                        prob_to_rank = {float(p): i + 1 for i, p in enumerate(sorted_probs)}
                        total_all = len(sorted_probs)
                        for _, row in df_all.iterrows():
                            p = row.get(prob_col)
                            if pd.notna(p):
                                p_norm = float(p) / 100 if float(p) > 1 else float(p)
                                prob_percentile_map[row['ts_code']] = round((prob_to_rank.get(p_norm, total_all) - 1) / total_all * 100, 1)
            except Exception:
                pass

        # 从 ArcticDB 补充 close 和 pct_chg
        try:
            from src.data.arctic_provider import ArcticDataProvider
            arctic = ArcticDataProvider()
            df_ohlcv = arctic.read_daily_ohlcv(raw_period, raw_period)
            if not df_ohlcv.empty and 'ts_code' in df_ohlcv.columns:
                price_map = df_ohlcv.set_index('ts_code')[['close', 'pct_chg']].to_dict('index')
                for col in ['close', 'pct_chg']:
                    if col not in df.columns:
                        df[col] = None
                for idx, row in df.iterrows():
                    ts = row.get('ts_code')
                    if ts in price_map:
                        df.at[idx, 'close'] = price_map[ts].get('close')
                        df.at[idx, 'pct_chg'] = price_map[ts].get('pct_chg')
        except Exception:
            pass

        # 将全市场概率分位 merge 到 df
        if prob_percentile_map:
            df['prob_percentile'] = df['ts_code'].map(prob_percentile_map)

        # Handle NaN values for JSON serialization
        df_clean = df.astype(object).where(pd.notna(df), None)
        records = df_clean.to_dict("records")
        avg_prob = None
        if prob_col in df.columns and not df.empty:
            m = df[prob_col].mean()
            avg_prob = float(m) if pd.notna(m) else None
        return {
            "filename": filename,
            "count": len(records),
            "avg_probability": avg_prob,
            "period": formatted_period,
            "display_period": display_period,
            "full_distribution": full_dist,
            "data": records,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction fetch failed: {str(e)}")


@router.get("/history")
async def get_prediction_history(
    ts_code: str = Query(..., description="Stock code"),
    days: int = Query(30, ge=1, le=90),
):
    """Get historical predictions for a specific stock to track accuracy."""
    try:
        pred_dir = project_root / "data" / "prediction"
        if not pred_dir.exists():
            raise HTTPException(status_code=404, detail="No prediction history")

        # This would require storing daily predictions per stock
        # For now, return placeholder
        return {
            "ts_code": ts_code,
            "message": "Historical prediction tracking requires daily archived predictions",
            "days_requested": days,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"History fetch failed: {str(e)}")


@router.get("/models")
async def get_available_models():
    """Get list of available model versions."""
    try:
        models_dir = project_root / "data" / "models"
        models = []

        # Scan for model versions
        for subdir in models_dir.rglob("metadata.json"):
            try:
                import json
                with open(subdir, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                models.append({
                    "name": meta.get("name", subdir.parent.name),
                    "version": meta.get("version", "unknown"),
                    "path": str(subdir.parent),
                    "description": meta.get("description", ""),
                })
            except Exception:
                continue

        return {"models": models, "count": len(models)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model list failed: {str(e)}")


@router.get("/distribution")
async def get_prediction_distribution(
    date: Optional[str] = Query(None, description="预测日期(YYYYMMDD)，默认最新"),
    exclude_bj: bool = Query(True, description="排除北交所股票(8/9开头)"),
    exclude_st: bool = Query(True, description="排除ST/*ST股票"),
    exclude_suspended: bool = Query(True, description="排除停牌股票"),
    min_mv: Optional[float] = Query(None, ge=0, description="最小流通市值(亿元)"),
):
    """
    获取全市场概率分布，支持智能过滤。

    过滤规则:
    - exclude_bj=True: 排除 ts_code 以 8/9 开头（北交所）
    - exclude_st=True: 排除 name 含 ST/*ST（需股票基本信息）
    - exclude_suspended=True: 排除停牌股票（需当日行情数据判断 volume=0）
    - min_mv: 最小流通市值（需市值数据）
    """
    try:
        import pandas as pd
        import sqlite3

        pred_dirs = [
            project_root / "data" / "prediction" / "v3.1.0_daily",
            project_root / "data" / "prediction" / "v3.1.0",
            project_root / "data" / "prediction" / "v3.0.0",
            project_root / "data" / "prediction" / "v3.0.0_daily",
            project_root / "data" / "prediction" / "v295_stk_factor_2026q1q2",
            project_root / "data" / "prediction" / "v295_stk_factor",
            project_root / "data" / "prediction" / "v294_stk_factor",
            project_root / "data" / "prediction" / "v294_daily",
            project_root / "data" / "prediction" / "v291_integrated",
            project_root / "data" / "prediction" / "v291_stk_factor",
            project_root / "data" / "prediction",
        ]

        # 确定日期
        date_str = date
        if not date_str:
            candidates = []
            for pred_dir in pred_dirs:
                if pred_dir.exists():
                    candidates.extend(pred_dir.glob("predictions_*_all.csv"))
            # 按日期排序取最新
            def _extract_date(path: Path):
                parts = path.name.split("_")
                return parts[1] if len(parts) >= 2 and parts[1].isdigit() and len(parts[1]) == 8 else "00000000"
            candidates = sorted(candidates, key=lambda p: _extract_date(p), reverse=True)
            date_str = _extract_date(candidates[0]) if candidates else None

        if not date_str:
            raise HTTPException(status_code=404, detail="No prediction data available")

        # 读取 _all.csv
        all_csv = _find_all_csv(pred_dirs, date_str)
        if not all_csv or not all_csv.exists():
            raise HTTPException(status_code=404, detail=f"No all-market prediction for date {date_str}")

        df = pd.read_csv(all_csv)
        prob_col = "prob" if "prob" in df.columns else ("probability" if "probability" in df.columns else None)
        if not prob_col:
            raise HTTPException(status_code=404, detail="No probability column found")

        if df[prob_col].max() > 1:
            df[prob_col] = df[prob_col] / 100

        # ---------- 过滤 ----------
        filters_applied = []
        total_before = len(df)

        if exclude_bj:
            df = df[~df["ts_code"].str.match(r"^[89]", na=False)]
            filters_applied.append("exclude_bj")

        # 加载股票基本信息用于 ST 过滤（优先 ArcticDB）
        stock_info = {}
        try:
            from src.data.arctic_provider import ArcticDataProvider
            arctic = ArcticDataProvider()
            stock_info = arctic.get_stock_basic_dict()
        except Exception:
            # 回退 SQLite
            try:
                db_path = project_root / "data" / "cache" / "quant_data.db"
                if db_path.exists():
                    import sqlite3
                    conn = sqlite3.connect(str(db_path))
                    cursor = conn.cursor()
                    cursor.execute("SELECT ts_code, name FROM stock_basic LIMIT 5000")
                    for row in cursor.fetchall():
                        stock_info[row[0]] = {"name": row[1]}
                    conn.close()
            except Exception:
                pass

        if exclude_st and stock_info:
            st_mask = df["ts_code"].apply(
                lambda x: "ST" in str(stock_info.get(x, ""))
                or "*ST" in str(stock_info.get(x, ""))
            )
            df = df[~st_mask]
            filters_applied.append("exclude_st")

        # 停牌过滤：查当日 volume=0（优先 ArcticDB）
        if exclude_suspended:
            try:
                from src.data.arctic_provider import ArcticDataProvider
                arctic = ArcticDataProvider()
                suspended_codes = arctic.get_suspended_stocks(date_str)
                df = df[~df["ts_code"].isin(suspended_codes)]
                filters_applied.append("exclude_suspended")
            except Exception:
                # 回退 SQLite
                try:
                    db_path = project_root / "data" / "cache" / "quant_data.db"
                    if db_path.exists():
                        import sqlite3
                        conn = sqlite3.connect(str(db_path))
                        cursor = conn.cursor()
                        cursor.execute(
                            "SELECT ts_code FROM daily_data WHERE trade_date=? AND (volume=0 OR volume IS NULL)",
                            (date_str,),
                        )
                        suspended_codes = {r[0] for r in cursor.fetchall()}
                        conn.close()
                        df = df[~df["ts_code"].isin(suspended_codes)]
                        filters_applied.append("exclude_suspended")
                except Exception:
                    pass

        total_after = len(df)
        filtered_out = total_before - total_after

        # 计算分布
        dist = _compute_distribution(df, prob_col)
        dist["filters_applied"] = filters_applied
        dist["filtered_out"] = filtered_out
        dist["date"] = date_str

        return dist
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Distribution fetch failed: {str(e)}")


@router.get("/pipeline-status")
async def get_pipeline_status():
    """Get daily pipeline execution status: data freshness, prediction status, and monitor."""
    import time
    now = time.time()
    if _pipeline_status_cache["data"] and (now - _pipeline_status_cache["timestamp"]) < _PIPELINE_STATUS_CACHE_TTL:
        return _pipeline_status_cache["data"]

    try:
        import json
        import sqlite3
        from datetime import datetime, timedelta

        today_str = datetime.now().strftime("%Y%m%d")
        today_iso = datetime.now().strftime("%Y-%m-%d")

        # ---------- DB freshness ----------
        db_latest_date = None
        try:
            from src.data.arctic_provider import ArcticDataProvider
            arctic = ArcticDataProvider()
            db_latest_date = arctic.get_latest_trade_date()
        except Exception:
            # 回退 SQLite
            try:
                db_path = project_root / "data" / "cache" / "quant_data.db"
                import sqlite3
                conn = sqlite3.connect(str(db_path))
                cursor = conn.cursor()
                cursor.execute("SELECT MAX(trade_date) FROM daily_data")
                row = cursor.fetchone()
                db_latest_date = row[0] if row else None
                conn.close()
            except Exception:
                pass

        # 数据新鲜度判断（考虑周末休市）
        weekday = datetime.now().weekday()
        is_weekend = weekday >= 5
        if db_latest_date == today_str:
            is_data_fresh = True
        elif is_weekend and db_latest_date:
            # 周末时，数据只需要更新到最近一个交易日即可
            last_friday = (datetime.now() - timedelta(days=(weekday - 4))).strftime("%Y%m%d")
            is_data_fresh = db_latest_date >= last_friday
        else:
            is_data_fresh = False

        # ---------- Latest prediction ----------
        # 优先检查 v3.0.0，回退到旧目录
        pred_dirs_check = [
            project_root / "data" / "prediction" / "v3.0.0",
            project_root / "data" / "prediction" / "v295_stk_factor_2026q1q2",
            project_root / "data" / "prediction" / "v294_stk_factor",
        ]
        latest_prediction_date = None
        latest_prediction_count = 0
        prediction_file_exists = False
        try:
            for pred_dir in pred_dirs_check:
                if not pred_dir.exists():
                    continue
                all_files = sorted(pred_dir.glob("predictions_*_all.csv"), reverse=True)
                if all_files:
                    latest_file = all_files[0]
                    latest_prediction_date = latest_file.stem.split("_")[1]
                    import pandas as pd
                    df_pred = pd.read_csv(latest_file)
                    latest_prediction_count = len(df_pred)
                    prediction_file_exists = True
                    break
        except Exception:
            pass

        # ---------- Today's pipeline report ----------
        monitor_dir = project_root / "logs" / "auto_pipeline_v300"
        today_report = None
        has_run_today = False
        monitor = {}
        try:
            if monitor_dir.exists():
                report_files = sorted(monitor_dir.glob("report_*.json"), reverse=True)
                if report_files:
                    latest_report = json.loads(report_files[0].read_text(encoding="utf-8"))
                    report_date = latest_report.get("run_id", "")[:8]
                    if report_date == today_str:
                        has_run_today = True
                        today_report = {
                            "run_id": latest_report.get("run_id"),
                            "start_time": latest_report.get("start_time"),
                            "end_time": latest_report.get("end_time"),
                            "steps": latest_report.get("steps", {}),
                            "prediction_file_exists": latest_report.get("prediction_file_exists", False),
                            "prediction_count": latest_report.get("prediction_count", 0),
                        }
                    monitor = latest_report.get("monitor", {})
        except Exception:
            pass

        # ---------- Scheduler tasks status ----------
        scheduler_tasks = {}
        try:
            from src.scheduler.models import get_session_factory, JobHistory
            from sqlalchemy import func
            session_factory = get_session_factory()
            with session_factory() as session:
                for job_id in ["daily_fill_data", "daily_arctic_sync", "daily_validate"]:
                    hist = (
                        session.query(JobHistory)
                        .filter(
                            JobHistory.job_id == job_id,
                            func.date(JobHistory.run_time) == func.date("now"),
                        )
                        .order_by(JobHistory.run_time.desc())
                        .first()
                    )
                    scheduler_tasks[job_id] = {
                        "status": hist.status if hist else "pending",
                        "run_time": hist.run_time.isoformat() if hist and hist.run_time else None,
                        "duration_ms": hist.duration_ms if hist else None,
                    }
        except Exception:
            pass

        # ---------- Pipeline alert ----------
        pipeline_alert = None
        try:
            fill_status = scheduler_tasks.get("daily_fill_data", {}).get("status", "pending")
            if fill_status == "failed":
                # 周末/节假日时，如果数据本身已最新，不显示错误（休市日本无新数据）
                if is_weekend and is_data_fresh:
                    pipeline_alert = None
                else:
                    pipeline_alert = {
                        "level": "error",
                        "message": "每日补数据任务失败，请前往任务调度页面查看原因并手动重试",
                        "action": "goto_scheduler",
                    }
            elif not is_data_fresh and not has_run_today:
                pipeline_alert = {
                    "level": "warning",
                    "message": "今日数据尚未更新，建议执行 Pipeline",
                    "action": "run_pipeline",
                }
        except Exception:
            pass

        result = {
            "today": today_iso,
            "db_latest_date": db_latest_date,
            "is_data_fresh": is_data_fresh,
            "latest_prediction_date": latest_prediction_date,
            "latest_prediction_count": latest_prediction_count,
            "prediction_file_exists": prediction_file_exists,
            "has_run_today": has_run_today,
            "today_report": today_report,
            "monitor": monitor,
            "scheduler_tasks": scheduler_tasks,
            "pipeline_alert": pipeline_alert,
        }
        _pipeline_status_cache["data"] = result
        _pipeline_status_cache["timestamp"] = time.time()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline status fetch failed: {str(e)}")



@router.post("/run-pipeline")
async def run_pipeline():
    """一键执行今日 Pipeline：按顺序触发 daily_fill_data → daily_arctic_sync → daily_validate"""
    try:
        from src.scheduler.service import SchedulerService

        service = SchedulerService()
        triggered = []

        pipeline_steps = [
            ("daily_fill_data", "每日补全数据"),
            ("daily_arctic_sync", "每日 ArcticDB 同步"),
            ("daily_validate", "每日数据验证与预测"),
        ]

        for job_id, job_name in pipeline_steps:
            try:
                run_id = service.run_job_now(job_id)
                triggered.append({"job_id": job_id, "job_name": job_name, "run_id": run_id, "status": "triggered"})
            except Exception as e:
                triggered.append({"job_id": job_id, "job_name": job_name, "status": "failed", "error": str(e)})
                # 如果某个步骤触发失败，继续尝试后续步骤（用户可以在任务调度页面单独重试失败的步骤）

        return {
            "message": "Pipeline 已触发",
            "triggered": triggered,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"触发 Pipeline 失败: {str(e)}")


@router.get("/strategy-pool")
async def get_strategy_pool(
    min_prob: Optional[float] = Query(None, ge=0, le=1, description="最低中期概率阈值"),
    allowed_stages: Optional[str] = Query(None, description="允许的市场阶段，逗号分隔，如'拉升初期,拉升中期'"),
    top_n: int = Query(100, ge=1, le=500),
):
    """
    战略股票池 — 基于 enrich 后的预测结果，按中期概率 + 市场阶段筛选。
    """
    try:
        import pandas as pd

        pred_dirs = [
            project_root / "data" / "prediction" / "v3.1.0_daily",
            project_root / "data" / "prediction" / "v3.1.0",
            project_root / "data" / "prediction" / "v3.0.0",
            project_root / "data" / "prediction" / "v3.0.0_daily",
            project_root / "data" / "prediction" / "v295_stk_factor_2026q1q2",
            project_root / "data" / "prediction" / "v295_stk_factor",
            project_root / "data" / "prediction" / "v294_stk_factor",
            project_root / "data" / "prediction" / "v294_daily",
            project_root / "data" / "prediction" / "v291_integrated",
            project_root / "data" / "prediction" / "v291_stk_factor",
            project_root / "data" / "prediction",
        ]

        # 查找最新的 enriched all 文件
        df = None
        for pred_dir in pred_dirs:
            if not pred_dir.exists():
                continue
            enriched_files = sorted(pred_dir.glob("predictions_*_all_enriched.csv"), reverse=True)
            for f in enriched_files:
                try:
                    df = pd.read_csv(f)
                    break
                except Exception:
                    continue
            if df is not None and not df.empty:
                break

        if df is None or df.empty:
            raise HTTPException(status_code=404, detail="暂无战略股票池数据，请先运行预测 enrich")

        # 确保 enrich 字段存在
        for col in ["market_stage", "left_side_signal"]:
            if col not in df.columns:
                df[col] = "" if col == "left_side_signal" else "未知"

        # 概率列归一化
        prob_col = None
        for c in ["prob", "probability", "adjusted_score"]:
            if c in df.columns:
                prob_col = c
                break
        if prob_col:
            if df[prob_col].max() > 1:
                df[prob_col] = df[prob_col] / 100

        # 应用概率过滤器
        if min_prob is not None and prob_col:
            df = df[df[prob_col] >= min_prob]

        # 应用市场阶段过滤器
        if allowed_stages:
            stages = [s.strip() for s in allowed_stages.split(",") if s.strip()]
            df = df[df["market_stage"].isin(stages)]

        # 排序: 中期概率降序
        if prob_col:
            df = df.sort_values(prob_col, ascending=False)

        df = df.head(top_n)

        # 补充 name / industry
        try:
            import tushare as ts
            pro = ts.pro_api()
            stock_basic = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,industry')
            if stock_basic is not None and not stock_basic.empty:
                for col in ['name', 'industry']:
                    if col in df.columns:
                        df = df.drop(columns=[col])
                df = df.merge(stock_basic, on='ts_code', how='left')
        except Exception:
            pass

        # 构造返回字段
        records = []
        for _, row in df.iterrows():
            left_sig = str(row.get("left_side_signal", "")).strip()
            left_signals = [s.strip() for s in left_sig.split("、") if s.strip()] if left_sig and left_sig not in ("nan", "None", "") else []

            prob_val = row.get("prob", row.get("probability", row.get("adjusted_score", 0)))
            prob_mid = float(prob_val) if pd.notna(prob_val) else 0.0
            if prob_mid > 1:
                prob_mid = prob_mid / 100

            records.append({
                "ts_code": row.get("ts_code", ""),
                "name": row.get("name", ""),
                "industry": row.get("industry", ""),
                "prob": prob_mid,
                "market_stage": str(row.get("market_stage", "未知")),
                "left_side_signals": left_signals,
                "update_date": str(row.get("update_date", "")) if "update_date" in row else "",
            })

        return {
            "count": len(records),
            "filters": {"min_prob": min_prob, "allowed_stages": allowed_stages},
            "data": records,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"战略股票池查询失败: {str(e)}")
