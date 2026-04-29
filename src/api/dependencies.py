"""
FastAPI dependencies for dependency injection.
Provides singleton instances of DataManager, ModelMonitor, etc.
"""
import sys
from functools import lru_cache
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.data_manager import DataManager
from src.monitoring.model_monitor import ModelMonitor


@lru_cache()
def get_data_manager() -> DataManager:
    """Singleton DataManager instance."""
    return DataManager()


@lru_cache()
def get_model_monitor() -> ModelMonitor:
    """Singleton ModelMonitor instance."""
    prediction_dir = project_root / "data" / "prediction"
    results_dir = project_root / "data" / "results"
    return ModelMonitor(prediction_dir=prediction_dir, results_dir=results_dir)
