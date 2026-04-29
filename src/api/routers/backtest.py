"""
Backtest API endpoints.
Provides backtest results, reports, and daily data.
"""
import sys
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

router = APIRouter()


def _find_backtest_dirs() -> List[Path]:
    """Find all backtest result directories."""
    results_dir = project_root / "data" / "results"
    if not results_dir.exists():
        return []
    # Look for p22_* directories (v291 realistic backtest)
    dirs = sorted([d for d in results_dir.glob("p22_*") if d.is_dir()], reverse=True)
    return dirs


@router.get("/list")
async def list_backtests():
    """List all available backtest results."""
    try:
        dirs = _find_backtest_dirs()
        results = []
        for d in dirs:
            # Try to extract metadata
            meta_file = d / "backtest_report.md"
            name = d.name
            results.append({
                "id": name,
                "name": name,
                "path": str(d),
                "has_report": meta_file.exists(),
            })
        return {"backtests": results, "count": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"List failed: {str(e)}")


@router.get("/{backtest_id}/report")
async def get_backtest_report(backtest_id: str):
    """Get backtest report markdown."""
    try:
        results_dir = project_root / "data" / "results"
        backtest_dir = results_dir / backtest_id
        if not backtest_dir.exists():
            raise HTTPException(status_code=404, detail=f"Backtest {backtest_id} not found")

        report_md = backtest_dir / "backtest_report.md"
        if report_md.exists():
            content = report_md.read_text(encoding="utf-8", errors="replace")
            return {"backtest_id": backtest_id, "report": content}
        else:
            raise HTTPException(status_code=404, detail="No report file found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report fetch failed: {str(e)}")


@router.get("/{backtest_id}/daily")
async def get_backtest_daily(backtest_id: str):
    """Get daily net value and drawdown data."""
    try:
        results_dir = project_root / "data" / "results"
        backtest_dir = results_dir / backtest_id
        daily_csv = backtest_dir / "backtest_daily.csv"

        if not daily_csv.exists():
            raise HTTPException(status_code=404, detail="No daily data found")

        import pandas as pd

        df = pd.read_csv(daily_csv, encoding="utf-8-sig")
        df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")

        # Calculate drawdown
        if "total_value" in df.columns:
            df["peak"] = df["total_value"].cummax()
            df["drawdown"] = (df["total_value"] - df["peak"]) / df["peak"] * 100

        records = df.to_dict("records")
        return {"backtest_id": backtest_id, "data": records}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Daily data failed: {str(e)}")


@router.get("/{backtest_id}/transactions")
async def get_backtest_transactions(backtest_id: str):
    """Get transaction details."""
    try:
        results_dir = project_root / "data" / "results"
        backtest_dir = results_dir / backtest_id
        txn_csv = backtest_dir / "backtest_transactions.csv"

        if not txn_csv.exists():
            raise HTTPException(status_code=404, detail="No transaction data found")

        import pandas as pd

        df = pd.read_csv(txn_csv, encoding="utf-8-sig")
        records = df.to_dict("records")
        return {"backtest_id": backtest_id, "data": records, "count": len(records)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transaction data failed: {str(e)}")
