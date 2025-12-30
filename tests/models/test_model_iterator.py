"""
测试模型版本迭代器（ModelIterator）

测试内容：
- 版本创建和管理
- 当前版本指针（current.json）
- 版本比较
- 版本清理和归档
"""
import pytest
import json
import shutil
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, Mock

from src.models.lifecycle.iterator import ModelIterator, VersionMetadata


@pytest.fixture
def test_model_name():
    """测试模型名称"""
    return "test_model"


@pytest.fixture
def clean_test_model_dir(test_model_name, temp_dir):
    """清理测试模型目录"""
    model_dir = temp_dir / "models" / test_model_name
    if model_dir.exists():
        shutil.rmtree(model_dir)
    yield model_dir
    # 测试后清理
    if model_dir.exists():
        shutil.rmtree(model_dir)


@pytest.fixture
def iterator(test_model_name, clean_test_model_dir, temp_dir):
    """创建测试用的ModelIterator"""
    # 使用临时目录作为模型存储位置
    model_base = temp_dir / "models" / test_model_name
    model_base.mkdir(parents=True, exist_ok=True)
    
    # 创建iterator并临时修改路径
    iterator = ModelIterator(test_model_name)
    iterator.base_path = model_base
    iterator.versions_path = model_base / "versions"
    iterator.current_file = model_base / "current.json"
    iterator.versions_path.mkdir(parents=True, exist_ok=True)
    iterator._ensure_current_file()
    
    return iterator


class TestModelIterator:
    """测试ModelIterator类"""
    
    def test_create_version(self, iterator):
        """测试创建版本"""
        version = iterator.create_version("v1.0.0")
        assert version == "v1.0.0"
        
        # 检查版本目录是否存在
        version_path = iterator.versions_path / "v1.0.0"
        assert version_path.exists()
        assert (version_path / "model").exists()
        assert (version_path / "training").exists()
        assert (version_path / "evaluation").exists()
        
        # 检查元数据文件
        metadata_file = version_path / "metadata.json"
        assert metadata_file.exists()
        
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            assert metadata['version'] == "v1.0.0"
            assert metadata['model_name'] == iterator.model_name
            assert metadata['status'] == 'development'
    
    def test_get_version_info(self, iterator):
        """测试获取版本信息"""
        iterator.create_version("v1.0.0")
        
        info = iterator.get_version_info("v1.0.0")
        assert info['version'] == "v1.0.0"
        assert info['model_name'] == iterator.model_name
    
    def test_list_versions(self, iterator):
        """测试列出所有版本"""
        iterator.create_version("v1.0.0")
        iterator.create_version("v1.1.0")
        iterator.create_version("v1.2.0")
        
        versions = iterator.list_versions()
        assert len(versions) == 3
        assert "v1.0.0" in versions
        assert "v1.1.0" in versions
        assert "v1.2.0" in versions
    
    def test_get_latest_version(self, iterator):
        """测试获取最新版本"""
        iterator.create_version("v1.0.0")
        iterator.create_version("v1.1.0")
        iterator.create_version("v1.2.0")
        
        latest = iterator.get_latest_version()
        assert latest == "v1.2.0"
    
    def test_current_version_pointer(self, iterator):
        """测试当前版本指针"""
        iterator.create_version("v1.0.0")
        iterator.create_version("v1.1.0")
        
        # 设置当前版本
        iterator.set_current_version("v1.1.0", "production")
        
        # 获取当前版本
        current = iterator.get_current_version("production")
        assert current == "v1.1.0"
        
        # 获取所有环境的当前版本
        all_current = iterator.get_current_versions()
        assert all_current['production'] == "v1.1.0"
    
    def test_promote_version(self, iterator):
        """测试版本提升"""
        iterator.create_version("v1.0.0")
        
        # 从development提升到testing
        iterator.promote_version("v1.0.0", "testing")
        
        current = iterator.get_current_version("testing")
        assert current == "v1.0.0"
        
        # 检查状态已更新
        info = iterator.get_version_info("v1.0.0")
        assert info['status'] == 'testing'
    
    def test_compare_versions(self, iterator):
        """测试版本比较"""
        # 创建两个版本并设置指标
        iterator.create_version("v1.0.0")
        iterator.create_version("v1.1.0")
        
        # 设置v1.0.0的指标
        iterator.update_version_metadata("v1.0.0", metrics={
            'test': {
                'accuracy': 0.75,
                'precision': 0.70,
                'recall': 0.80,
                'f1': 0.75,
                'auc': 0.80
            }
        })
        
        # 设置v1.1.0的指标（更好的性能）
        iterator.update_version_metadata("v1.1.0", metrics={
            'test': {
                'accuracy': 0.85,
                'precision': 0.80,
                'recall': 0.85,
                'f1': 0.82,
                'auc': 0.90
            }
        })
        
        # 比较版本
        comparison = iterator.compare_versions("v1.0.0", "v1.1.0")
        
        assert comparison.version_a == "v1.0.0"
        assert comparison.version_b == "v1.1.0"
        assert 'accuracy' in comparison.metrics_diff
        assert comparison.metrics_diff['accuracy']['improved'] is True
    
    def test_find_stale_versions(self, iterator):
        """测试查找过时版本"""
        # 创建多个版本
        for i in range(5):
            iterator.create_version(f"v1.{i}.0")
        
        # 设置一些版本的状态
        iterator.update_version_metadata("v1.0.0", status="production")
        iterator.update_version_metadata("v1.1.0", status="development")
        iterator.update_version_metadata("v1.2.0", status="development")
        iterator.update_version_metadata("v1.3.0", status="development")
        iterator.update_version_metadata("v1.4.0", status="development")
        
        # 查找过时版本（保留最新3个development版本）
        stale = iterator.find_stale_versions(keep_latest_n=3)
        
        # v1.0.0是production，不应该被标记为过时
        assert "v1.0.0" not in stale
        # v1.1.0应该被标记为过时（保留v1.2.0, v1.3.0, v1.4.0）
        assert "v1.1.0" in stale
    
    def test_archive_version(self, iterator):
        """测试归档版本"""
        iterator.create_version("v1.0.0")
        
        # 归档版本
        archived_path = iterator.archive_version("v1.0.0")
        
        # 检查版本目录已移动
        version_path = iterator.versions_path / "v1.0.0"
        assert not version_path.exists()
        
        # 检查归档目录存在
        assert Path(archived_path).exists()
    
    def test_delete_version(self, iterator):
        """测试删除版本"""
        iterator.create_version("v1.0.0")
        iterator.create_version("v1.1.0")
        
        # 删除development版本
        iterator.delete_version("v1.0.0", force=False)
        
        # 检查版本已删除
        versions = iterator.list_versions()
        assert "v1.0.0" not in versions
        assert "v1.1.0" in versions
    
    def test_delete_production_version_fails(self, iterator):
        """测试删除production版本应该失败"""
        iterator.create_version("v1.0.0")
        iterator.update_version_metadata("v1.0.0", status="production")
        iterator.set_current_version("v1.0.0", "production")
        
        # 尝试删除production版本（应该失败）
        with pytest.raises(ValueError, match="无法删除.*production"):
            iterator.delete_version("v1.0.0", force=False)
    
    def test_cleanup(self, iterator):
        """测试清理功能"""
        # 创建多个版本
        for i in range(5):
            iterator.create_version(f"v1.{i}.0")
        
        # 预览清理（dry_run=True）
        stale = iterator.cleanup(keep_latest_n=3, dry_run=True)
        
        # 应该找到过时版本
        assert len(stale) > 0
        
        # 实际清理（dry_run=False）
        archived = iterator.cleanup(keep_latest_n=3, dry_run=False)
        
        # 应该归档了一些版本
        assert len(archived) > 0
    
    def test_get_version_path(self, iterator):
        """测试获取版本路径"""
        iterator.create_version("v1.0.0")
        
        # 获取版本路径
        version_path = iterator.get_version_path("v1.0.0")
        assert version_path.exists()
        
        # 获取模型文件路径
        model_path = iterator.get_model_path("v1.0.0")
        assert model_path.parent.exists()
    
    def test_list_versions_by_status(self, iterator):
        """测试按状态列出版本"""
        iterator.create_version("v1.0.0")
        iterator.create_version("v1.1.0")
        iterator.create_version("v1.2.0")
        
        iterator.update_version_metadata("v1.0.0", status="production")
        iterator.update_version_metadata("v1.1.0", status="development")
        iterator.update_version_metadata("v1.2.0", status="development")
        
        by_status = iterator.list_versions_by_status()
        
        assert len(by_status['production']) == 1
        assert len(by_status['development']) == 2
        assert "v1.0.0" in by_status['production']
        assert "v1.1.0" in by_status['development']
        assert "v1.2.0" in by_status['development']
    
    def test_version_key_sorting(self, iterator):
        """测试版本号排序"""
        iterator.create_version("v1.0.0")
        iterator.create_version("v1.10.0")
        iterator.create_version("v1.2.0")
        iterator.create_version("v2.0.0")
        
        versions = iterator.list_versions()
        
        # 检查排序正确
        assert versions[0] == "v1.0.0"
        assert versions[1] == "v1.2.0"
        assert versions[2] == "v1.10.0"
        assert versions[3] == "v2.0.0"


class TestVersionMetadata:
    """测试VersionMetadata数据类"""
    
    def test_version_metadata_creation(self):
        """测试创建版本元数据"""
        metadata = VersionMetadata(
            version="v1.0.0",
            model_name="test_model",
            status="development"
        )
        
        assert metadata.version == "v1.0.0"
        assert metadata.model_name == "test_model"
        assert metadata.status == "development"
        assert metadata.created_at is not None
        assert metadata.metrics == {}
        assert metadata.changes == []

