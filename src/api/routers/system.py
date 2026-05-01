"""
System monitoring API endpoints.
Provides model drift detection, system status, and logs.
"""
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.api.dependencies import get_model_monitor

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
