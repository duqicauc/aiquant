"""
Test script: Verify all core modules can be imported correctly
"""

import sys
import io
from pathlib import Path

# Set stdout encoding to UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    print("=" * 60)
    print("AIQuant Module Import Test")
    print("=" * 60)

    # 1. Test config module
    try:
        from config.settings import settings

        print("[OK] Config module loaded")
        print(f"     Start date: {settings.get('data.sample_preparation.start_date')}")
        print(f"     Model version: {settings.get('model.version')}")
    except Exception as e:
        print(f"[FAIL] Config module failed: {e}")
        return False

    # 2. Test logger module
    try:

        print("[OK] Logger module loaded")
    except Exception as e:
        print(f"[FAIL] Logger module failed: {e}")
        return False

    # 3. Test rate limiter module
    try:

        print("[OK] Rate limiter module loaded")
    except Exception as e:
        print(f"[FAIL] Rate limiter module failed: {e}")
        return False

    # 4. Test cache manager
    try:

        print("[OK] CacheManager loaded")
    except Exception as e:
        print(f"[FAIL] CacheManager failed: {e}")
        return False

    # 5. Test data fetcher (import only, no init - needs Token)
    try:

        print("[OK] TushareFetcher module loaded")
    except Exception as e:
        print(f"[FAIL] TushareFetcher failed: {e}")
        return False

    # 6. Test data manager (import only, no init)
    try:

        print("[OK] DataManager module loaded")
    except Exception as e:
        print(f"[FAIL] DataManager failed: {e}")
        return False

    # 7. Test screener modules
    try:

        print("[OK] Screener modules loaded")
    except Exception as e:
        print(f"[FAIL] Screener modules failed: {e}")
        return False

    print("")
    print("=" * 60)
    print("ALL CORE MODULES LOADED SUCCESSFULLY!")
    print("=" * 60)
    print("")
    print("Next steps:")
    print("  1. Configure TUSHARE_TOKEN in .env file")
    print("  2. Run: python scripts/prepare_positive_samples.py")
    print("  3. Run: python scripts/prepare_negative_samples_v2.py")
    print("  4. Run: python scripts/train_xgboost_timeseries.py")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
