"""
模型迭代器 - 管理模型版本

功能：
- 版本创建、查询、比较
- 当前版本指针管理（current.json）
- 版本清理和归档
- 版本状态流转：development → testing → staging → production
"""
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
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


@dataclass
class VersionComparison:
    """版本比较结果"""
    version_a: str
    version_b: str
    metrics_diff: Dict
    config_diff: Dict
    recommendation: str


class ModelIterator:
    """
    模型迭代器 - 管理模型版本
    
    功能：
    - 版本CRUD操作
    - 当前版本指针（current.json）
    - 版本比较
    - 版本清理
    """
    
    # 版本状态定义
    STATUS_DEVELOPMENT = 'development'
    STATUS_TESTING = 'testing'
    STATUS_STAGING = 'staging'
    STATUS_PRODUCTION = 'production'
    
    VALID_STATUSES = [STATUS_DEVELOPMENT, STATUS_TESTING, STATUS_STAGING, STATUS_PRODUCTION]
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.base_path = Path(f"data/models/{model_name}")
        self.versions_path = self.base_path / "versions"
        self.current_file = self.base_path / "current.json"
        self.versions_path.mkdir(parents=True, exist_ok=True)
        
        # 确保 current.json 存在
        self._ensure_current_file()
    
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
    
    # =========================================================================
    # 当前版本指针管理（current.json）
    # =========================================================================
    
    def _ensure_current_file(self):
        """确保 current.json 文件存在"""
        if not self.current_file.exists():
            default_current = {
                "production": None,
                "staging": None,
                "testing": None,
                "development": None,
                "updated_at": datetime.now().isoformat()
            }
            with open(self.current_file, 'w', encoding='utf-8') as f:
                json.dump(default_current, f, indent=2, ensure_ascii=False)
    
    def get_current_versions(self) -> Dict[str, Optional[str]]:
        """获取各环境的当前版本"""
        with open(self.current_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_current_version(self, env: str = 'production') -> Optional[str]:
        """获取指定环境的当前版本"""
        current = self.get_current_versions()
        return current.get(env)
    
    def set_current_version(self, version: str, env: str = 'production') -> bool:
        """
        设置指定环境的当前版本
        
        Args:
            version: 版本号
            env: 环境（production/staging/testing/development）
        
        Returns:
            是否设置成功
        """
        if env not in self.VALID_STATUSES:
            raise ValueError(f"无效环境: {env}，有效值: {self.VALID_STATUSES}")
        
        # 验证版本是否存在
        if not (self.versions_path / version).exists():
            raise ValueError(f"版本 {version} 不存在")
        
        current = self.get_current_versions()
        current[env] = version
        current['updated_at'] = datetime.now().isoformat()
        
        with open(self.current_file, 'w', encoding='utf-8') as f:
            json.dump(current, f, indent=2, ensure_ascii=False)
        
        # 同步更新版本的status
        self.update_version_metadata(version, status=env)
        
        return True
    
    def promote_version(self, version: str, to_env: str) -> bool:
        """
        提升版本到指定环境
        
        状态流转: development → testing → staging → production
        
        Args:
            version: 版本号
            to_env: 目标环境
        """
        env_order = {
            self.STATUS_DEVELOPMENT: 0,
            self.STATUS_TESTING: 1,
            self.STATUS_STAGING: 2,
            self.STATUS_PRODUCTION: 3
        }
        
        info = self.get_version_info(version)
        current_env = info.get('status', self.STATUS_DEVELOPMENT)
        
        if env_order.get(to_env, -1) <= env_order.get(current_env, -1):
            raise ValueError(f"无法从 {current_env} 提升到 {to_env}，只能向更高环境提升")
        
        return self.set_current_version(version, to_env)
    
    # =========================================================================
    # 版本比较功能
    # =========================================================================
    
    def compare_versions(self, version_a: str, version_b: str) -> VersionComparison:
        """
        比较两个版本的差异
        
        Args:
            version_a: 版本A（通常是旧版本）
            version_b: 版本B（通常是新版本）
        
        Returns:
            VersionComparison: 比较结果
        """
        info_a = self.get_version_info(version_a)
        info_b = self.get_version_info(version_b)
        
        # 比较指标
        metrics_diff = self._compare_metrics(
            info_a.get('metrics', {}),
            info_b.get('metrics', {})
        )
        
        # 比较配置
        config_diff = self._compare_config(
            info_a.get('config', {}),
            info_b.get('config', {})
        )
        
        # 生成建议
        recommendation = self._generate_recommendation(metrics_diff, version_a, version_b)
        
        return VersionComparison(
            version_a=version_a,
            version_b=version_b,
            metrics_diff=metrics_diff,
            config_diff=config_diff,
            recommendation=recommendation
        )
    
    def _compare_metrics(self, metrics_a: Dict, metrics_b: Dict) -> Dict:
        """比较指标差异"""
        diff = {}
        
        # 获取所有指标键
        all_keys = set()
        for section in ['training', 'validation', 'test']:
            if section in metrics_a:
                all_keys.update(metrics_a[section].keys())
            if section in metrics_b:
                all_keys.update(metrics_b[section].keys())
        
        # 主要比较 test 指标
        test_a = metrics_a.get('test', {})
        test_b = metrics_b.get('test', {})
        
        for key in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
            val_a = test_a.get(key)
            val_b = test_b.get(key)
            
            if val_a is not None and val_b is not None:
                change = val_b - val_a
                change_pct = (change / val_a * 100) if val_a != 0 else 0
                diff[key] = {
                    'version_a': round(val_a, 4),
                    'version_b': round(val_b, 4),
                    'change': round(change, 4),
                    'change_pct': round(change_pct, 2),
                    'improved': change > 0
                }
        
        return diff
    
    def _compare_config(self, config_a: Dict, config_b: Dict) -> Dict:
        """比较配置差异"""
        diff = {}
        
        # 比较模型参数
        params_a = config_a.get('model_params', {})
        params_b = config_b.get('model_params', {})
        
        all_params = set(params_a.keys()) | set(params_b.keys())
        param_changes = {}
        
        for param in all_params:
            val_a = params_a.get(param)
            val_b = params_b.get(param)
            if val_a != val_b:
                param_changes[param] = {
                    'version_a': val_a,
                    'version_b': val_b
                }
        
        if param_changes:
            diff['model_params'] = param_changes
        
        return diff
    
    def _generate_recommendation(self, metrics_diff: Dict, version_a: str, version_b: str) -> str:
        """根据指标差异生成建议"""
        improvements = sum(1 for m in metrics_diff.values() if m.get('improved', False))
        total = len(metrics_diff)
        
        if total == 0:
            return f"无法比较：两个版本缺少可比较的指标"
        
        improvement_rate = improvements / total
        
        # 检查关键指标
        auc_improved = metrics_diff.get('auc', {}).get('improved', False)
        f1_improved = metrics_diff.get('f1', {}).get('improved', False)
        
        if improvement_rate >= 0.8 and auc_improved:
            return f"✅ 强烈推荐: {version_b} 在大多数指标上优于 {version_a}，建议升级"
        elif improvement_rate >= 0.5:
            return f"⚠️ 谨慎推荐: {version_b} 部分指标有提升，建议进一步测试后升级"
        elif improvement_rate > 0:
            return f"❌ 不推荐: {version_b} 只有少量指标提升，建议保持 {version_a}"
        else:
            return f"❌ 不推荐: {version_b} 没有明显改进，建议保持 {version_a}"
    
    def print_comparison(self, comparison: VersionComparison):
        """打印版本比较结果"""
        print("=" * 70)
        print(f"📊 版本比较: {comparison.version_a} vs {comparison.version_b}")
        print("=" * 70)
        
        print("\n📈 指标对比:")
        print("-" * 70)
        print(f"{'指标':<15} {comparison.version_a:<15} {comparison.version_b:<15} {'变化':<15} {'状态':<10}")
        print("-" * 70)
        
        for metric, data in comparison.metrics_diff.items():
            status = "✅ 提升" if data['improved'] else "❌ 下降"
            change_str = f"{data['change']:+.4f} ({data['change_pct']:+.2f}%)"
            print(f"{metric:<15} {data['version_a']:<15.4f} {data['version_b']:<15.4f} {change_str:<15} {status:<10}")
        
        if comparison.config_diff:
            print("\n⚙️ 配置变化:")
            print("-" * 70)
            for section, changes in comparison.config_diff.items():
                print(f"  [{section}]")
                for param, vals in changes.items():
                    print(f"    {param}: {vals['version_a']} → {vals['version_b']}")
        
        print("\n" + "=" * 70)
        print(f"💡 建议: {comparison.recommendation}")
        print("=" * 70)
    
    # =========================================================================
    # 版本清理功能
    # =========================================================================
    
    def list_versions_by_status(self) -> Dict[str, List[str]]:
        """按状态分组列出所有版本"""
        result = {status: [] for status in self.VALID_STATUSES}
        result['unknown'] = []
        
        for version in self.list_versions():
            try:
                info = self.get_version_info(version)
                status = info.get('status', 'unknown')
                if status in result:
                    result[status].append(version)
                else:
                    result['unknown'].append(version)
            except:
                result['unknown'].append(version)
        
        return result
    
    def find_stale_versions(self, keep_latest_n: int = 3) -> List[str]:
        """
        查找过时的版本（可以清理的版本）
        
        规则：
        - 保留所有 production/staging 版本
        - 保留最近 N 个 development/testing 版本
        - 返回可以清理的版本列表
        
        Args:
            keep_latest_n: 保留的最新开发/测试版本数量
        
        Returns:
            可以清理的版本列表
        """
        by_status = self.list_versions_by_status()
        
        stale = []
        
        # development 版本：保留最新 N 个
        dev_versions = by_status.get(self.STATUS_DEVELOPMENT, [])
        if len(dev_versions) > keep_latest_n:
            stale.extend(dev_versions[:-keep_latest_n])
        
        # testing 版本：保留最新 N 个
        test_versions = by_status.get(self.STATUS_TESTING, [])
        if len(test_versions) > keep_latest_n:
            stale.extend(test_versions[:-keep_latest_n])
        
        # unknown 版本：全部标记为可清理
        stale.extend(by_status.get('unknown', []))
        
        return stale
    
    def archive_version(self, version: str) -> str:
        """
        归档版本（移动到 archive 目录）
        
        Args:
            version: 版本号
        
        Returns:
            归档后的路径
        """
        version_path = self.versions_path / version
        if not version_path.exists():
            raise ValueError(f"版本 {version} 不存在")
        
        archive_path = self.base_path / "archive"
        archive_path.mkdir(exist_ok=True)
        
        # 添加归档时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archived_name = f"{version}_archived_{timestamp}"
        target_path = archive_path / archived_name
        
        shutil.move(str(version_path), str(target_path))
        
        return str(target_path)
    
    def delete_version(self, version: str, force: bool = False) -> bool:
        """
        删除版本
        
        Args:
            version: 版本号
            force: 是否强制删除（即使是 production/staging）
        
        Returns:
            是否删除成功
        """
        version_path = self.versions_path / version
        if not version_path.exists():
            raise ValueError(f"版本 {version} 不存在")
        
        # 安全检查：不能删除 production/staging 版本
        info = self.get_version_info(version)
        status = info.get('status', '')
        
        if status in [self.STATUS_PRODUCTION, self.STATUS_STAGING] and not force:
            raise ValueError(f"无法删除 {status} 环境的版本，请使用 force=True 强制删除")
        
        # 检查是否是当前版本
        current = self.get_current_versions()
        for env, ver in current.items():
            if ver == version and env != 'updated_at':
                raise ValueError(f"版本 {version} 是 {env} 环境的当前版本，请先切换再删除")
        
        shutil.rmtree(str(version_path))
        return True
    
    def cleanup(self, keep_latest_n: int = 3, dry_run: bool = True) -> List[str]:
        """
        清理过时版本
        
        Args:
            keep_latest_n: 保留的最新开发/测试版本数量
            dry_run: 是否只是预览，不实际删除
        
        Returns:
            被清理（或将被清理）的版本列表
        """
        stale = self.find_stale_versions(keep_latest_n)
        
        if dry_run:
            print(f"🔍 预览模式：以下 {len(stale)} 个版本将被归档")
            for v in stale:
                print(f"  - {v}")
            return stale
        
        archived = []
        for version in stale:
            try:
                self.archive_version(version)
                archived.append(version)
                print(f"✅ 已归档: {version}")
            except Exception as e:
                print(f"❌ 归档失败 {version}: {e}")
        
        return archived
    
    # =========================================================================
    # 便捷方法
    # =========================================================================
    
    def get_version_path(self, version: str = None) -> Path:
        """
        获取版本目录路径
        
        Args:
            version: 版本号，None 表示获取当前生产版本
        """
        if version is None:
            version = self.get_current_version('production')
            if version is None:
                version = self.get_latest_version()
        
        return self.versions_path / version
    
    def get_model_path(self, version: str = None) -> Path:
        """获取模型文件路径"""
        version_path = self.get_version_path(version)
        return version_path / "model" / "model.json"
    
    def print_status(self):
        """打印当前版本状态"""
        print("=" * 60)
        print(f"📦 模型: {self.model_name}")
        print("=" * 60)
        
        # 当前版本
        current = self.get_current_versions()
        print("\n🎯 当前版本:")
        for env in self.VALID_STATUSES:
            version = current.get(env)
            status = f"  {env:<15}: {version or '(未设置)'}"
            if env == self.STATUS_PRODUCTION and version:
                status += " ⭐"
            print(status)
        
        # 版本统计
        by_status = self.list_versions_by_status()
        print("\n📊 版本统计:")
        for status, versions in by_status.items():
            if versions:
                print(f"  {status:<15}: {len(versions)} 个版本")
        
        # 最新版本
        latest = self.get_latest_version()
        print(f"\n🔄 最新版本: {latest}")
        print("=" * 60)

