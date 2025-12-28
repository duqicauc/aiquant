"""
数据源配置
"""
import os
from dotenv import load_dotenv

load_dotenv()


class DataSourceConfig:
    """数据源配置"""
    
    # Tushare配置
    TUSHARE_TOKEN = os.getenv('TUSHARE_TOKEN', '')
    TUSHARE_TIMEOUT = int(os.getenv('TUSHARE_TIMEOUT', '30'))
    
    # RQData配置
    RQDATA_USERNAME = os.getenv('RQDATA_USERNAME', '')
    RQDATA_PASSWORD = os.getenv('RQDATA_PASSWORD', '')
    
    # JQData配置
    JQDATA_USERNAME = os.getenv('JQDATA_USERNAME', '')
    JQDATA_PASSWORD = os.getenv('JQDATA_PASSWORD', '')
    
    # 默认数据源
    DEFAULT_SOURCE = os.getenv('DEFAULT_DATA_SOURCE', 'tushare')
    
    @classmethod
    def validate_tushare(cls):
        """验证Tushare配置"""
        if not cls.TUSHARE_TOKEN:
            raise ValueError(
                "请设置TUSHARE_TOKEN！\n"
                "1. 访问 https://tushare.pro/register 注册账号\n"
                "2. 获取Token\n"
                "3. 在.env文件中设置 TUSHARE_TOKEN=你的token"
            )
        return True


# 数据源配置实例
data_source_config = DataSourceConfig()

