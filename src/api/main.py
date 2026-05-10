"""
AIQuant FastAPI Application
Provides REST API and WebSocket endpoints for the Dash dashboard.
"""
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.api.routers import auth, admin, backtest, macro, market, prediction, scheduler, stock, system, trading, watchlist


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    print("🚀 AIQuant API starting up...")
    from src.scheduler.service import SchedulerService
    scheduler_service = SchedulerService()
    scheduler_service.start()
    print("🚀 AIQuant Scheduler started")
    yield
    scheduler_service.shutdown(wait=True)
    print("🛑 AIQuant API shutting down...")


app = FastAPI(
    title="AIQuant API",
    description="Professional quantitative trading platform API",
    version="5.0.0",
    lifespan=lifespan,
)

# CORS for Dash frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8050", "http://127.0.0.1:8050", "http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:5174", "http://127.0.0.1:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(auth.router, prefix="/api/auth", tags=["Auth"])
app.include_router(admin.router, prefix="/api/admin", tags=["Admin"])
app.include_router(macro.router, prefix="/api/macro", tags=["Macro"])
app.include_router(market.router, prefix="/api/market", tags=["Market"])
app.include_router(stock.router, prefix="/api/stock", tags=["Stock"])
app.include_router(prediction.router, prefix="/api/prediction", tags=["Prediction"])
app.include_router(backtest.router, prefix="/api/backtest", tags=["Backtest"])
app.include_router(system.router, prefix="/api/system", tags=["System"])
app.include_router(watchlist.router, prefix="/api/watchlist", tags=["Watchlist"])
app.include_router(scheduler.router, prefix="/api/scheduler", tags=["Scheduler"])
app.include_router(trading.router, prefix="/api/trading", tags=["Trading"])


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "version": "5.0.0"}


@app.get("/")
async def root():
    """Root redirect to docs."""
    return {
        "message": "AIQuant API v5.0",
        "docs": "/docs",
        "redoc": "/redoc",
    }
