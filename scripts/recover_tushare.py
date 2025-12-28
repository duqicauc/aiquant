#!/usr/bin/env python3
"""
Tushare API恢复脚本

用于检查和恢复Tushare API连接问题
"""
import os
import sys
import time
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import log
from scripts.utils.network_monitor import NetworkMonitor
from config.data_source import data_source_config


def check_tushare_connection():
    """检查Tushare连接"""
    print("=" * 60)
    print("检查Tushare API连接状态...")
    print("=" * 60)
    
    # 1. 检查Token
    print("\n1. 检查Tushare Token...")
    try:
        data_source_config.validate_tushare()
        print("✓ Token配置正确")
    except Exception as e:
        print(f"✗ Token配置错误: {e}")
        print("  请检查 .env 文件中的 TUSHARE_TOKEN 配置")
        return False
    
    # 2. 检查网络连接
    print("\n2. 检查网络连接...")
    monitor = NetworkMonitor()
    
    # 检查HTTP连接
    if monitor.check_network_http():
        print("✓ HTTP连接正常")
    else:
        print("✗ HTTP连接失败")
    
    # 检查Tushare API
    if monitor.check_tushare_api():
        print("✓ Tushare API连接正常")
        return True
    else:
        print("✗ Tushare API连接失败")
        return False


def test_simple_api_call():
    """测试简单的API调用"""
    print("\n3. 测试API调用...")
    try:
        import tushare as ts
        ts.set_token(data_source_config.TUSHARE_TOKEN)
        pro = ts.pro_api(data_source_config.TUSHARE_TOKEN)
        
        # 测试获取交易日历（最简单的接口）
        df = pro.trade_cal(exchange='SSE', start_date='20250101', end_date='20250101')
        if df is not None and not df.empty:
            print("✓ API调用成功")
            return True
        else:
            print("✗ API返回空数据")
            return False
    except Exception as e:
        print(f"✗ API调用失败: {e}")
        return False


def suggest_solutions():
    """提供解决方案建议"""
    print("\n" + "=" * 60)
    print("恢复建议:")
    print("=" * 60)
    print("""
1. 等待一段时间后重试
   - Tushare API可能暂时不可用
   - 建议等待5-10分钟后重试

2. 检查网络连接
   - 确保网络连接正常
   - 如果使用代理，检查代理设置

3. 检查API配额
   - 登录 https://tushare.pro 查看积分和配额
   - 确认没有超过调用限制

4. 使用缓存数据
   - 如果之前有缓存数据，可以暂时使用缓存
   - 缓存位置: data/cache/quant_data.db

5. 调整重试参数
   - 已自动增强重试机制（5次重试，延迟2秒）
   - 如果仍然失败，可以等待更长时间

6. 分批处理
   - 如果正在批量处理数据，建议分批进行
   - 每批之间间隔一段时间
    """)


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("Tushare API恢复工具")
    print("=" * 60)
    
    # 检查连接
    connection_ok = check_tushare_connection()
    
    if connection_ok:
        # 测试API调用
        api_ok = test_simple_api_call()
        
        if api_ok:
            print("\n" + "=" * 60)
            print("✓ Tushare API工作正常，可以继续使用")
            print("=" * 60)
            return 0
        else:
            suggest_solutions()
            return 1
    else:
        suggest_solutions()
        return 1


if __name__ == '__main__':
    sys.exit(main())

