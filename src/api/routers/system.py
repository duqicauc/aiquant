"""
System monitoring API endpoints.
Provides model drift detection, system status, and logs.
"""
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Depends

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.api.dependencies import get_model_monitor
from src.api.routers.auth import get_current_user_optional, get_current_user
from src.scheduler.models import User

router = APIRouter()


@router.get("/monitor")
async def get_system_monitor():
    """Get system monitoring status (PSI, trade quality, etc.)."""
    try:
        monitor_dir = project_root / "logs" / "auto_pipeline_v294"
        report_files = sorted(monitor_dir.glob("report_*.json"), reverse=True)

        if not report_files:
            return {
                "status": "no_data",
                "message": "No monitoring data available. Run auto pipeline first.",
            }

        import json

        latest_report = json.loads(report_files[0].read_text(encoding="utf-8"))
        monitor_data = latest_report.get("monitor", {})

        psi_info = monitor_data.get("psi", {})
        tq = monitor_data.get("trade_quality", {})
        coverage = monitor_data.get("prediction_coverage", 0)

        return {
            "status": "ok",
            "report_date": latest_report.get("date", "unknown"),
            "psi": {
                "value": psi_info.get("psi", "N/A"),
                "status": psi_info.get("status", "unknown"),
            },
            "trade_quality": {
                "avg_win_rate": tq.get("avg_win_rate", 0),
                "avg_profit_ratio": tq.get("avg_profit_ratio", 0),
                "alerts": tq.get("alerts", []),
            },
            "prediction_coverage": coverage,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Monitor failed: {str(e)}")


@router.get("/logs")
async def get_system_logs(
    lines: int = 100,
    level: Optional[str] = None,
):
    """Get recent system logs."""
    try:
        log_file = project_root / "logs" / "aiquant.log"
        if not log_file.exists():
            return {"logs": [], "message": "No log file found"}

        with open(log_file, "r", encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()

        # Filter by level if specified
        if level:
            all_lines = [l for l in all_lines if level.upper() in l.upper()]

        recent = all_lines[-lines:]
        return {"logs": recent, "total_lines": len(all_lines)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Log fetch failed: {str(e)}")


@router.get("/status")
async def get_system_status():
    """Get overall system status."""
    try:
        from src.data.data_manager import DataManager

        dm = DataManager()

        # Check data freshness
        stock_list = dm.get_stock_list()
        data_status = {
            "stock_list_count": len(stock_list) if stock_list is not None else 0,
            "db_path": str(project_root / "data" / "cache" / "quant_data.db"),
            "db_exists": (project_root / "data" / "cache" / "quant_data.db").exists(),
        }

        # Check model files
        models_dir = project_root / "data" / "models"
        model_status = {
            "models_dir_exists": models_dir.exists(),
            "model_count": len(list(models_dir.rglob("*.json"))) if models_dir.exists() else 0,
        }

        return {
            "status": "ok",
            "timestamp": datetime.now().isoformat(),
            "data": data_status,
            "models": model_status,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")


# ---------- Alert Config ----------

ALERT_CONFIG_DEFAULT = {
    "wechat_webhook": "",
    "dingtalk_webhook": "",
    "smtp_config": "",
    "alert_strike_zone": True,
    "alert_stop_loss": True,
    "alert_model_drift": True,
    "alert_watchlist": False,
    "quiet_start": "22:00",
    "quiet_end": "08:00",
}


def _get_alert_config(user_id: int = 1) -> dict:
    """从 SQLite user_settings 读取 alert_config"""
    import json
    import sqlite3

    db_path = project_root / "data" / "database" / "aiquant.db"
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute(
            "SELECT setting_value FROM user_settings WHERE user_id = ? AND setting_key = 'alert_config' ORDER BY id DESC LIMIT 1",
            (user_id,),
        )
        row = cursor.fetchone()
        conn.close()
        if row and row[0]:
            return {**ALERT_CONFIG_DEFAULT, **json.loads(row[0])}
    except Exception:
        pass
    return ALERT_CONFIG_DEFAULT.copy()


def _save_alert_config(user_id: int, config: dict):
    """保存 alert_config 到 SQLite user_settings"""
    import json
    import sqlite3
    from datetime import datetime

    db_path = project_root / "data" / "database" / "aiquant.db"
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    # 先删除旧配置
    cursor.execute("DELETE FROM user_settings WHERE user_id = ? AND setting_key = 'alert_config'", (user_id,))
    # 插入新配置
    cursor.execute(
        "INSERT INTO user_settings (user_id, setting_key, setting_value, created_at) VALUES (?, ?, ?, ?)",
        (user_id, "alert_config", json.dumps(config), datetime.now()),
    )
    conn.commit()
    conn.close()


@router.get("/alert-config")
async def get_alert_config(user: User = Depends(get_current_user_optional)):
    """获取当前用户的预警配置（支持未登录使用默认配置）"""
    user_id = user.id if user else 1
    return _get_alert_config(user_id)


@router.post("/alert-config")
async def save_alert_config(config: dict, user: User = Depends(get_current_user)):
    """保存预警配置（需登录）"""
    try:
        # 只保存已知字段
        allowed = set(ALERT_CONFIG_DEFAULT.keys())
        filtered = {k: v for k, v in config.items() if k in allowed}
        _save_alert_config(user.id, filtered)
        return {"success": True, "message": "配置已保存"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"保存配置失败: {str(e)}")
