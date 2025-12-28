"""
SSL证书权限修复模块

在macOS上修复requests库的SSL证书权限问题
必须在导入任何使用requests的模块之前导入此模块
"""
import os
import sys

def fix_ssl_permissions():
    """修复SSL证书权限问题"""
    try:
        import certifi
        cert_path = certifi.where()
        
        # 设置环境变量
        os.environ['REQUESTS_CA_BUNDLE'] = cert_path
        os.environ['SSL_CERT_FILE'] = cert_path
        os.environ['CURL_CA_BUNDLE'] = cert_path
        
        # 尝试monkey patch requests的SSL上下文
        # 这需要在requests导入之前执行
        return True
    except ImportError:
        # certifi未安装，尝试其他方法
        pass
    
    return False

# 自动执行修复
fix_ssl_permissions()

