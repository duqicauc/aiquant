"""
网络监控和自动恢复脚本

功能:
1. 定期检查网络连接状态
2. 检测到网络问题时自动刷新 Clash 配置
3. 支持多种检测方式: ping、HTTP、API测试
4. 自动重试和日志记录
"""

import time
import subprocess
import requests
import logging
from pathlib import Path
import sys
import os

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import log

class NetworkMonitor:
    """网络监控器"""
    
    def __init__(self, 
                 check_interval: int = 60,  # 检查间隔（秒）
                 max_retry: int = 3,         # 最大重试次数
                 clash_api: str = "http://127.0.0.1:9090",  # Clash API地址
                 clash_secret: str = None):  # Clash API密钥
        """
        初始化网络监控器
        
        Args:
            check_interval: 检查间隔（秒），默认60秒
            max_retry: 网络失败后的最大重试次数
            clash_api: Clash API地址
            clash_secret: Clash API密钥（如果需要）
        """
        self.check_interval = check_interval
        self.max_retry = max_retry
        self.clash_api = clash_api.rstrip('/')
        self.clash_secret = clash_secret
        self.consecutive_failures = 0
        
    def check_network_ping(self) -> bool:
        """
        通过ping检查网络连接
        
        Returns:
            bool: 网络是否正常
        """
        try:
            # Ping百度和谷歌DNS
            targets = ['8.8.8.8', 'www.baidu.com']
            for target in targets:
                result = subprocess.run(
                    ['ping', '-c', '1', '-W', '3', target],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=5
                )
                if result.returncode == 0:
                    return True
            return False
        except Exception as e:
            log.warning(f"Ping检查失败: {e}")
            return False
    
    def check_network_http(self) -> bool:
        """
        通过HTTP请求检查网络连接
        
        Returns:
            bool: 网络是否正常
        """
        try:
            # 尝试访问多个网站
            urls = [
                'https://www.baidu.com',
                'https://www.google.com',
                'http://www.163.com'
            ]
            for url in urls:
                try:
                    response = requests.get(url, timeout=5)
                    if response.status_code == 200:
                        return True
                except:
                    continue
            return False
        except Exception as e:
            log.warning(f"HTTP检查失败: {e}")
            return False
    
    def check_tushare_api(self) -> bool:
        """
        检查Tushare API连接
        
        Returns:
            bool: API是否可访问
        """
        try:
            import tushare as ts
            import os
            from dotenv import load_dotenv
            
            load_dotenv()
            token = os.getenv('TUSHARE_TOKEN')
            if not token:
                return False
                
            pro = ts.pro_api(token)
            # 简单的API调用测试
            df = pro.trade_cal(exchange='SSE', start_date='20250101', end_date='20250101')
            return df is not None and not df.empty
        except Exception as e:
            log.warning(f"Tushare API检查失败: {e}")
            return False
    
    def check_network(self) -> bool:
        """
        综合检查网络连接
        
        Returns:
            bool: 网络是否正常
        """
        # 优先级: Tushare API > HTTP > Ping
        checks = [
            ("Tushare API", self.check_tushare_api),
            ("HTTP", self.check_network_http),
            ("Ping", self.check_network_ping)
        ]
        
        for name, check_func in checks:
            try:
                if check_func():
                    log.debug(f"✓ 网络检查通过 ({name})")
                    return True
            except Exception as e:
                log.warning(f"✗ {name}检查异常: {e}")
                continue
        
        return False
    
    def reload_clash_config(self) -> bool:
        """
        通过API重新加载Clash配置
        
        Returns:
            bool: 是否成功
        """
        try:
            headers = {}
            if self.clash_secret:
                headers['Authorization'] = f'Bearer {self.clash_secret}'
            
            # 方法1: 重新加载配置
            url = f"{self.clash_api}/configs"
            response = requests.put(
                url,
                json={"path": ""},
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 204:
                log.info("✓ Clash配置已通过API重新加载")
                return True
            else:
                log.warning(f"Clash API返回状态码: {response.status_code}")
                
        except Exception as e:
            log.warning(f"通过API重载Clash配置失败: {e}")
        
        return False
    
    def restart_clash_service(self) -> bool:
        """
        重启Clash服务（备用方法）
        
        Returns:
            bool: 是否成功
        """
        try:
            # 方法1: 通过brew services重启
            result = subprocess.run(
                ['brew', 'services', 'restart', 'clash'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30
            )
            if result.returncode == 0:
                log.info("✓ Clash服务已通过brew重启")
                time.sleep(5)  # 等待服务启动
                return True
                
        except Exception as e:
            log.warning(f"通过brew重启Clash失败: {e}")
        
        try:
            # 方法2: 查找并重启Clash进程
            # 先查找Clash进程
            result = subprocess.run(
                ['pgrep', '-f', 'clash'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=10
            )
            
            if result.returncode == 0:
                pids = result.stdout.decode().strip().split('\n')
                for pid in pids:
                    if pid:
                        # 发送SIGHUP信号重载配置
                        subprocess.run(['kill', '-HUP', pid], timeout=5)
                        log.info(f"✓ 已向Clash进程({pid})发送重载信号")
                
                time.sleep(3)
                return True
                
        except Exception as e:
            log.warning(f"通过信号重启Clash失败: {e}")
        
        return False
    
    def recover_network(self) -> bool:
        """
        尝试恢复网络连接
        
        Returns:
            bool: 是否成功恢复
        """
        log.warning("⚠️ 检测到网络问题，尝试恢复...")
        
        # 方法1: 重新加载Clash配置
        if self.reload_clash_config():
            time.sleep(5)
            if self.check_network():
                log.info("✓ 网络已通过重载Clash配置恢复")
                return True
        
        # 方法2: 重启Clash服务
        if self.restart_clash_service():
            time.sleep(5)
            if self.check_network():
                log.info("✓ 网络已通过重启Clash服务恢复")
                return True
        
        log.error("✗ 无法自动恢复网络连接")
        return False
    
    def monitor(self, duration: int = None):
        """
        持续监控网络状态
        
        Args:
            duration: 监控时长（秒），None表示永久监控
        """
        log.info(f"🔍 网络监控已启动")
        log.info(f"   检查间隔: {self.check_interval}秒")
        log.info(f"   Clash API: {self.clash_api}")
        
        start_time = time.time()
        check_count = 0
        failure_count = 0
        recovery_count = 0
        
        try:
            while True:
                check_count += 1
                
                # 检查网络状态
                if self.check_network():
                    self.consecutive_failures = 0
                    if check_count % 10 == 0:  # 每10次检查输出一次
                        log.info(f"✓ 网络正常 (已检查{check_count}次)")
                else:
                    self.consecutive_failures += 1
                    failure_count += 1
                    log.error(f"✗ 网络检查失败 (连续失败{self.consecutive_failures}次)")
                    
                    # 达到重试阈值，尝试恢复
                    if self.consecutive_failures >= self.max_retry:
                        log.warning(f"⚠️ 连续失败{self.consecutive_failures}次，开始恢复...")
                        if self.recover_network():
                            recovery_count += 1
                            self.consecutive_failures = 0
                        else:
                            log.error("✗ 网络恢复失败，将在下次检查时继续尝试")
                
                # 检查是否达到监控时长
                if duration and (time.time() - start_time) >= duration:
                    break
                
                # 等待下次检查
                time.sleep(self.check_interval)
                
        except KeyboardInterrupt:
            log.info("\n⚠️ 监控被用户中断")
        finally:
            elapsed = time.time() - start_time
            log.info(f"📊 监控统计:")
            log.info(f"   运行时长: {elapsed/3600:.2f}小时")
            log.info(f"   检查次数: {check_count}")
            log.info(f"   失败次数: {failure_count}")
            log.info(f"   恢复次数: {recovery_count}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='网络监控和自动恢复')
    parser.add_argument('--interval', type=int, default=60, 
                       help='检查间隔（秒），默认60')
    parser.add_argument('--retry', type=int, default=3,
                       help='最大重试次数，默认3')
    parser.add_argument('--duration', type=int, default=None,
                       help='监控时长（秒），默认永久')
    parser.add_argument('--clash-api', type=str, default='http://127.0.0.1:9090',
                       help='Clash API地址')
    parser.add_argument('--clash-secret', type=str, default=None,
                       help='Clash API密钥')
    parser.add_argument('--test', action='store_true',
                       help='测试模式：只检查一次网络状态')
    
    args = parser.parse_args()
    
    monitor = NetworkMonitor(
        check_interval=args.interval,
        max_retry=args.retry,
        clash_api=args.clash_api,
        clash_secret=args.clash_secret
    )
    
    if args.test:
        log.info("🧪 测试模式")
        if monitor.check_network():
            log.info("✓ 网络连接正常")
        else:
            log.error("✗ 网络连接异常")
            log.info("尝试恢复网络...")
            if monitor.recover_network():
                log.info("✓ 网络恢复成功")
            else:
                log.error("✗ 网络恢复失败")
    else:
        monitor.monitor(duration=args.duration)


if __name__ == '__main__':
    main()

