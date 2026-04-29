"""
宏观数据 API endpoints.
Provides China macro, US macro, global indices/commodities, and event calendar.
"""
import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.macro_provider import macro_service

router = APIRouter()


@router.get("/overview")
async def get_macro_overview():
    """Get comprehensive macro overview (China + US + Global)."""
    try:
        data = macro_service.get_overview()
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Macro overview failed: {str(e)}")


@router.get("/events")
async def get_macro_events():
    """Get current macro event calendar (FOMC, policy windows, etc.)."""
    try:
        events = macro_service.get_events()
        return {"events": events, "count": len(events)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Macro events failed: {str(e)}")
