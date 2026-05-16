"""
Backtest API endpoints.
Provides backtest results, reports, and daily data.
"""
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, HTTPException

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

router = APIRouter()


def _find_backtest_dirs() -> List[Path]:
    """Find all backtest result directories."""
    results_dir = project_root / "data" / "results"
    if not results_dir.exists():
        return []
    # Look for p2*_* directories (p21 legacy + p22 new)
    dirs = sorted([d for d in results_dir.glob("p2*_*") if d.is_dir()], reverse=True)
    return dirs


def _parse_report_for_metrics(report_text: str) -> Dict[str, Any]:
    """Parse metrics from report markdown."""
    metrics = {}
    for line in report_text.split('\n'):
        if '| 总收益率 |' in line:
            try:
                val = line.split('|')[2].strip().replace('%', '').replace('+', '')
                metrics['total_return'] = float(val)
            except:
                pass
        elif '| 最大回撤 |' in line:
            try:
                val = line.split('|')[2].strip().replace('%', '')
                metrics['max_drawdown'] = float(val)
            except:
                pass
        elif '| 胜率 |' in line:
            try:
                val = line.split('|')[2].strip().replace('%', '')
                metrics['win_rate'] = float(val)
            except:
                pass
        elif '| 总卖出次数 |' in line:
            try:
                val = line.split('|')[2].strip()
                metrics['trade_count'] = int(val)
            except:
                pass
        elif '| 盈亏比 |' in line:
            try:
                val = line.split('|')[2].strip()
                metrics['profit_factor'] = float(val)
            except:
                pass
        elif '| 平均盈利 |' in line:
            try:
                val = line.split('|')[2].strip().replace(',', '')
                metrics['avg_win'] = float(val)
            except:
                pass
        elif '| 平均亏损 |' in line:
            try:
                val = line.split('|')[2].strip().replace(',', '')
                metrics['avg_loss'] = float(val)
            except:
                pass
    return metrics


def _get_friendly_name(backtest_dir: Path) -> str:
    """Generate a friendly display name for a backtest directory."""
    raw_name = backtest_dir.name
    # Try to extract from report.md
    report_md = backtest_dir / "backtest_report.md"
    if report_md.exists():
        try:
            text = report_md.read_text(encoding="utf-8", errors="replace")
            # Look for date range line: **回测期**: 20241008 ~ 20241231
            date_range = ""
            for line in text.split('\n'):
                if '**回测期**' in line or '回测期' in line:
                    parts = line.split('~')
                    if len(parts) == 2:
                        start = parts[0].strip().split()[-1]
                        end = parts[1].strip()
                        date_range = f"{start}~{end}"
                    break
            # For p21 directories (legacy), use quarter label
            if raw_name.startswith("p21_"):
                quarter = raw_name.replace("p21_", "")
                return f"📊 {quarter}" + (f" ({date_range})" if date_range else "")
            # For p22_strategy directories, extract strategy info
            if raw_name.startswith("p22_strategy_"):
                parts = raw_name.split('_')
                if len(parts) >= 4:
                    strategy_id = parts[2][:8]
                    dates = '_'.join(parts[3:5]) if len(parts) >= 5 else ''
                    return f"🎯 策略回测 {strategy_id} {dates}"
            # For p22_scan directories
            if raw_name.startswith("p22_scan_"):
                return f"🔬 参数扫描 {raw_name.replace('p22_scan_', '')[:8]}"
            if date_range:
                return f"📈 {date_range}"
        except Exception:
            pass
    # Fallback
    if raw_name.startswith("p21_"):
        return f"📊 {raw_name.replace('p21_', '')}"
    return raw_name


def _get_backtest_metrics(backtest_dir: Path) -> Dict[str, Any]:
    """Extract metrics from backtest result files."""
    metrics = {
        "total_return": None,
        "max_drawdown": None,
        "win_rate": None,
        "trade_count": None,
        "profit_factor": None,
        "avg_win": None,
        "avg_loss": None,
        "start_date": None,
        "end_date": None,
    }

    # Try report.md first
    report_md = backtest_dir / "backtest_report.md"
    if report_md.exists():
        try:
            text = report_md.read_text(encoding="utf-8", errors="replace")
            parsed = _parse_report_for_metrics(text)
            metrics.update(parsed)
            # Parse date range
            for line in text.split('\n'):
                if '**回测期**' in line or '回测期' in line:
                    parts = line.split('~')
                    if len(parts) == 2:
                        metrics['start_date'] = parts[0].strip().split()[-1]
                        metrics['end_date'] = parts[1].strip()
                    break
        except Exception:
            pass

    # Fallback to CSV calculations
    daily_csv = backtest_dir / "backtest_daily.csv"
    if daily_csv.exists() and metrics.get("total_return") is None:
        try:
            import pandas as pd
            df = pd.read_csv(daily_csv, encoding="utf-8-sig")
            if not df.empty and "total_value" in df.columns:
                initial = df["total_value"].iloc[0]
                final = df["total_value"].iloc[-1]
                if initial and initial > 0:
                    metrics["total_return"] = round((final - initial) / initial * 100, 2)
                df["peak"] = df["total_value"].cummax()
                df["dd"] = (df["peak"] - df["total_value"]) / df["peak"] * 100
                metrics["max_drawdown"] = round(df["dd"].max(), 2)
                metrics["start_date"] = metrics["start_date"] or str(df["date"].iloc[0])
                metrics["end_date"] = metrics["end_date"] or str(df["date"].iloc[-1])
        except Exception:
            pass

    txn_csv = backtest_dir / "backtest_transactions.csv"
    if txn_csv.exists() and metrics.get("trade_count") is None:
        try:
            import pandas as pd
            df = pd.read_csv(txn_csv, encoding="utf-8-sig")
            if not df.empty:
                sells = df[df["action"] == "SELL"] if "action" in df.columns else df
                metrics["trade_count"] = len(sells)
                if "profit" in sells.columns and not sells.empty:
                    wins = sells[sells["profit"] > 0]
                    losses = sells[sells["profit"] <= 0]
                    metrics["win_rate"] = round(len(wins) / len(sells) * 100, 1) if len(sells) > 0 else 0
                    metrics["avg_win"] = round(wins["profit"].mean(), 0) if not wins.empty else 0
                    metrics["avg_loss"] = round(losses["profit"].mean(), 0) if not losses.empty else 0
                    if not losses.empty and losses["profit"].sum() != 0:
                        metrics["profit_factor"] = round(abs(wins["profit"].sum() / losses["profit"].sum()), 2)
        except Exception:
            pass

    return metrics


@router.get("/list")
async def list_backtests():
    """List all available backtest results with metrics."""
    try:
        dirs = _find_backtest_dirs()
        results = []
        for d in dirs:
            meta_file = d / "backtest_report.md"
            name = _get_friendly_name(d)
            metrics = _get_backtest_metrics(d)
            results.append({
                "id": d.name,
                "name": name,
                "path": str(d),
                "has_report": meta_file.exists(),
                **metrics,
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

        # Format date as YYYY-MM-DD string
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], format="%Y%m%d").dt.strftime("%Y-%m-%d")

        # Calculate drawdown as positive percentage (e.g. 2.5 means 2.5% drawdown)
        if "total_value" in df.columns:
            df["peak"] = df["total_value"].cummax()
            df["drawdown"] = ((df["peak"] - df["total_value"]) / df["peak"] * 100).round(2)

        # Keep only needed columns for frontend
        cols = ["date", "capital", "holding_value", "total_value", "holdings_count", "return_pct", "drawdown"]
        cols = [c for c in cols if c in df.columns]
        df = df[cols]

        records = df.to_dict("records")
        return {"backtest_id": backtest_id, "data": records}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Daily data failed: {str(e)}")


def _load_stock_names(ts_codes: List[str]) -> Dict[str, str]:
    """Load stock name mapping from SQLite cache or prediction CSVs."""
    name_map = {}
    if not ts_codes:
        return name_map

    # Try SQLite cache first
    try:
        import sqlite3
        db_path = project_root / "data" / "cache" / "quant_data.db"
        if db_path.exists():
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            placeholders = ','.join('?' * len(ts_codes))
            cursor.execute(f"SELECT ts_code, name FROM stock_basic WHERE ts_code IN ({placeholders})", ts_codes)
            for row in cursor.fetchall():
                if row[1]:
                    name_map[row[0]] = row[1]
            conn.close()
    except Exception:
        pass

    # Fallback: try prediction CSVs
    if len(name_map) < len(ts_codes):
        try:
            import pandas as pd
            pred_root = project_root / "data" / "prediction"
            for pred_dir in pred_root.iterdir():
                if not pred_dir.is_dir():
                    continue
                for pf in pred_dir.glob("predictions_*.csv"):
                    try:
                        df = pd.read_csv(pf, encoding="utf-8-sig", nrows=1000)
                        if "ts_code" in df.columns and "name" in df.columns:
                            for _, row in df.iterrows():
                                code = str(row["ts_code"]) if pd.notna(row.get("ts_code")) else None
                                name = str(row["name"]) if pd.notna(row.get("name")) else None
                                if code and name and code not in name_map:
                                    name_map[code] = name
                    except Exception:
                        pass
                    if len(name_map) >= len(ts_codes):
                        break
                if len(name_map) >= len(ts_codes):
                    break
        except Exception:
            pass

    return name_map


@router.get("/{backtest_id}/transactions")
async def get_backtest_transactions(backtest_id: str):
    """Get transaction details with stock names."""
    try:
        results_dir = project_root / "data" / "results"
        backtest_dir = results_dir / backtest_id
        txn_csv = backtest_dir / "backtest_transactions.csv"

        if not txn_csv.exists():
            raise HTTPException(status_code=404, detail="No transaction data found")

        import pandas as pd

        df = pd.read_csv(txn_csv, encoding="utf-8-sig")

        # Load stock names for all ts_codes
        ts_codes = df["ts_code"].dropna().unique().tolist() if "ts_code" in df.columns else []
        name_map = _load_stock_names(ts_codes)

        records = []
        for _, row in df.iterrows():
            rec = row.to_dict()
            ts_code = rec.get("ts_code", "")
            rec["name"] = name_map.get(ts_code, ts_code)
            # Ensure commission field exists
            if "commission" not in rec:
                rec["commission"] = 0.0
            # Ensure reason field exists
            if "reason" not in rec:
                rec["reason"] = ""
            records.append(rec)

        return {"backtest_id": backtest_id, "data": records, "count": len(records)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transaction data failed: {str(e)}")
