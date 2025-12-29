"""
等待预测完成并自动比对结果
"""
import sys
import os
import time
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.logger import log


def wait_for_prediction(max_wait_minutes=60):
    """等待预测结果文件生成"""
    log.info("等待预测完成...")
    log.info(f"最多等待 {max_wait_minutes} 分钟")
    
    results_dir = Path('data/prediction/results')
    pattern = 'top_50_stocks_20251225*breakout_launch_scorer*v1.1.0-test*.csv'
    
    start_time = time.time()
    max_wait_seconds = max_wait_minutes * 60
    
    while True:
        new_files = list(results_dir.glob(pattern))
        
        if new_files:
            # 选择最新的文件
            new_file = max(new_files, key=lambda x: x.stat().st_mtime)
            # 检查文件是否完整（至少50行，包含表头）
            try:
                import pandas as pd
                df = pd.read_csv(new_file, encoding='utf-8-sig')
                if len(df) >= 50:
                    log.success(f"✓ 找到预测结果: {new_file}")
                    return str(new_file)
            except:
                pass
        
        elapsed = time.time() - start_time
        if elapsed > max_wait_seconds:
            log.error("等待超时，未找到预测结果")
            return None
        
        # 每30秒检查一次
        time.sleep(30)
        log.info(f"等待中... ({int(elapsed/60)} 分钟)")


def main():
    """主函数"""
    log.info("="*80)
    log.info("等待预测完成并比对结果")
    log.info("="*80)
    log.info("")
    
    # 等待预测完成
    new_file = wait_for_prediction(max_wait_minutes=60)
    
    if new_file is None:
        log.error("未找到预测结果，请检查预测是否完成")
        return
    
    log.info("")
    
    # 运行比对脚本
    log.info("开始比对...")
    import subprocess
    
    old_file = '/Users/javaadu/Documents/GitHub/aiquant/data/prediction/results/top_50_stocks_20251225_232545.csv'
    
    cmd = [
        'python', 'scripts/compare_new_old_predictions.py',
        '--new-file', new_file,
        '--old-file', old_file
    ]
    
    result = subprocess.run(
        cmd,
        cwd=os.path.dirname(os.path.dirname(__file__))
    )
    
    if result.returncode == 0:
        log.success("✅ 比对完成！")
    else:
        log.error("❌ 比对失败")


if __name__ == '__main__':
    main()

