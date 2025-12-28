"""
模型迭代器 - 管理模型版本
"""
import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass, asdict


@dataclass
class VersionMetadata:
    """版本元数据"""
    version: str
    model_name: str
    status: str = 'development'  # development, testing, staging, production
    created_at: str = None
    created_by: str = 'system'
    parent_version: Optional[str] = None
    metrics: Dict = None
    changes: List[Dict] = None
    notes: str = ''
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        if self.metrics is None:
            self.metrics = {}
        if self.changes is None:
            self.changes = []


class ModelIterator:
    """模型迭代器 - 管理模型版本"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.base_path = Path(f"data/models/{model_name}")
        self.versions_path = self.base_path / "versions"
        self.versions_path.mkdir(parents=True, exist_ok=True)
    
    def create_version(
        self,
        version: str,
        base_version: Optional[str] = None,
        changes: Dict = None,
        created_by: str = None
    ) -> str:
        """创建新版本"""
        version_path = self.versions_path / version
        version_path.mkdir(parents=True, exist_ok=True)
        
        # 创建版本目录结构
        (version_path / "model").mkdir(exist_ok=True)
        (version_path / "training").mkdir(exist_ok=True)
        (version_path / "evaluation").mkdir(exist_ok=True)
        (version_path / "experiments").mkdir(exist_ok=True)
        
        # 创建元数据
        metadata = VersionMetadata(
            version=version,
            model_name=self.model_name,
            status='development',
            created_by=created_by or 'system',
            parent_version=base_version,
            changes=changes or {}
        )
        
        # 保存元数据
        self._save_metadata(version, metadata)
        
        return version
    
    def get_version_info(self, version: str) -> Dict:
        """获取版本信息"""
        metadata_path = self.versions_path / version / "metadata.json"
        if not metadata_path.exists():
            raise ValueError(f"版本 {version} 不存在")
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def update_version_metadata(self, version: str, **kwargs):
        """更新版本元数据"""
        info = self.get_version_info(version)
        info.update(kwargs)
        self._save_metadata(version, info)
    
    def list_versions(self, status: str = None) -> List[str]:
        """列出所有版本"""
        versions = []
        for version_dir in self.versions_path.iterdir():
            if version_dir.is_dir():
                try:
                    info = self.get_version_info(version_dir.name)
                    if status and info.get('status') != status:
                        continue
                    versions.append(version_dir.name)
                except:
                    continue
        
        # 按版本号排序
        versions.sort(key=lambda v: self._version_key(v))
        return versions
    
    def get_latest_version(self) -> Optional[str]:
        """获取最新版本"""
        versions = self.list_versions()
        return versions[-1] if versions else None
    
    def _version_key(self, version: str) -> tuple:
        """将版本号转换为可排序的元组"""
        # 移除 'v' 前缀和标识符
        version = version.lstrip('v')
        if '-' in version:
            version = version.split('-')[0]
        
        parts = version.split('.')
        return tuple(int(p) if p.isdigit() else 0 for p in parts)
    
    def _save_metadata(self, version: str, metadata):
        """保存元数据"""
        metadata_path = self.versions_path / version / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            if isinstance(metadata, VersionMetadata):
                json.dump(asdict(metadata), f, indent=2, ensure_ascii=False)
            else:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

