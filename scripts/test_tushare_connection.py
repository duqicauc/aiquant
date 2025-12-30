"""
Test Tushare connection and configuration
"""
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

def main():
    print("="*50)
    print("Tushare Configuration Check")
    print("="*50)
    
    token = os.getenv('TUSHARE_TOKEN', '')
    points = os.getenv('TUSHARE_POINTS', '120')
    
    token_status = "Configured" if token and token != "YOUR_TUSHARE_TOKEN" else "NOT SET"
    print(f"Token: {token_status}")
    print(f"Points: {points}")
    
    if not token or token == "YOUR_TUSHARE_TOKEN":
        print("")
        print("[WARN] Please configure TUSHARE_TOKEN in .env file first")
        return False
    
    print("")
    print("Testing Tushare connection...")
    
    try:
        import tushare as ts
        ts.set_token(token)
        pro = ts.pro_api()
        
        # Simple test - get trade calendar
        df = pro.trade_cal(exchange='SSE', start_date='20241201', end_date='20241205')
        print(f"[OK] Tushare connection successful! Got {len(df)} records")
        
        # Test stock list
        df_stocks = pro.stock_basic(list_status='L', fields='ts_code,name')
        print(f"[OK] Stock list: {len(df_stocks)} stocks")
        
        print("")
        print("="*50)
        print("ALL CHECKS PASSED! Ready to train model.")
        print("="*50)
        return True
        
    except Exception as e:
        print(f"[FAIL] Tushare connection failed: {e}")
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
