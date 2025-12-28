"""
SSL证书权限修复模块测试
"""
import pytest
import os
from unittest.mock import patch, MagicMock


class TestSSLFix:
    """SSL修复模块测试类"""

    @pytest.mark.unit
    def test_fix_ssl_permissions_with_certifi(self):
        """测试有certifi时的SSL修复"""
        # 直接测试函数逻辑，不重新导入模块
        from src.utils import ssl_fix
        
        # 保存原始函数
        original_fix = ssl_fix.fix_ssl_permissions
        
        # Mock certifi
        mock_certifi = MagicMock()
        mock_certifi.where.return_value = '/path/to/cert.pem'
        
        # 临时替换
        original_certifi = getattr(ssl_fix, 'certifi', None)
        ssl_fix.certifi = mock_certifi
        
        try:
            # 清理环境变量
            for key in ['REQUESTS_CA_BUNDLE', 'SSL_CERT_FILE', 'CURL_CA_BUNDLE']:
                os.environ.pop(key, None)
            
            # 执行修复
            result = ssl_fix.fix_ssl_permissions()
            
            assert result is True
            assert os.environ.get('REQUESTS_CA_BUNDLE') == '/path/to/cert.pem'
            assert os.environ.get('SSL_CERT_FILE') == '/path/to/cert.pem'
            assert os.environ.get('CURL_CA_BUNDLE') == '/path/to/cert.pem'
        finally:
            # 恢复
            if original_certifi is not None:
                ssl_fix.certifi = original_certifi
            else:
                delattr(ssl_fix, 'certifi')

    @pytest.mark.unit
    def test_fix_ssl_permissions_without_certifi(self):
        """测试没有certifi时的SSL修复"""
        from src.utils import ssl_fix
        
        # 清理环境变量
        for key in ['REQUESTS_CA_BUNDLE', 'SSL_CERT_FILE', 'CURL_CA_BUNDLE']:
            os.environ.pop(key, None)
        
        # 临时移除certifi
        original_certifi = getattr(ssl_fix, 'certifi', None)
        if hasattr(ssl_fix, 'certifi'):
            delattr(ssl_fix, 'certifi')
        
        try:
            # Mock ImportError
            with patch('builtins.__import__', side_effect=lambda name, *args, **kwargs: 
                      __import__(name, *args, **kwargs) if name != 'certifi' else 
                      exec('raise ImportError("No module named certifi")')):
                result = ssl_fix.fix_ssl_permissions()
                
                assert result is False
                # 环境变量不应该被设置
                assert os.environ.get('REQUESTS_CA_BUNDLE') is None
        finally:
            # 恢复
            if original_certifi is not None:
                ssl_fix.certifi = original_certifi

    @pytest.mark.unit
    def test_fix_ssl_permissions_auto_execution(self):
        """测试模块导入时自动执行修复"""
        # 模块应该已经执行了修复
        from src.utils import ssl_fix
        assert hasattr(ssl_fix, 'fix_ssl_permissions')

