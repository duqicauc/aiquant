#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型版本训练完整流程脚本

完整流程：
1. 准备正样本数据
2. 准备负样本数据
3. 添加高级技术因子
4. 训练模型（带版本管理）
5. Walk-forward 验证
6. 备份训练数据到版本目录
7. 生成版本报告

使用示例：
    # 训练新版本（完整流程）
    python scripts/train_model_version.py --version v1.5.0

    # 跳过数据准备（使用现有数据）
    python scripts/train_model_version.py --version v1.5.0 --skip-data-prep

    # 只备份数据（用于已有版本）
    python scripts/train_model_version.py --version v1.4.0 --backup-only
"""

import sys
import os
import argparse
import subprocess
import shutil
import json
import yaml
from pathlib import Path
from datetime import datetime
import time

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import log


class ModelVersionTrainer:
    """模型版本训练器"""
    
    def __init__(self, version: str, model_name: str = 'breakout_launch_scorer'):
        self.version = version
        self.model_name = model_name
        self.start_time = time.time()
        
        # 路径配置
        self.version_dir = Path(f'data/models/{model_name}/versions/{version}')
        self.training_data_dir = self.version_dir / 'training_data'
        
        # 数据源路径
        self.samples_dir = Path('data/training/samples')
        self.processed_dir = Path('data/training/processed')
        self.features_dir = Path('data/training/features')
        self.charts_dir = Path('data/training/charts')
        self.metrics_dir = Path('data/training/metrics')
    
    def run_full_pipeline(
        self,
        skip_data_prep: bool = False,
        skip_training: bool = False,
        skip_validation: bool = False,
        backup_only: bool = False,
        use_advanced_factors: bool = True,
        neg_version: str = 'v2'
    ):
        """
        运行完整训练流程
        
        Args:
            skip_data_prep: 跳过数据准备步骤
            skip_training: 跳过训练步骤
            skip_validation: 跳过验证步骤
            backup_only: 只执行备份（用于已有版本）
            use_advanced_factors: 使用高级技术因子
            neg_version: 负样本版本
        """
        log.info("="*80)
        log.info(f"🚀 模型版本训练流程 - {self.model_name} {self.version}")
        log.info("="*80)
        log.info("")
        log.info(f"配置:")
        log.info(f"  版本号: {self.version}")
        log.info(f"  模型名称: {self.model_name}")
        log.info(f"  负样本版本: {neg_version}")
        log.info(f"  使用高级因子: {use_advanced_factors}")
        log.info(f"  跳过数据准备: {skip_data_prep}")
        log.info(f"  跳过训练: {skip_training}")
        log.info(f"  跳过验证: {skip_validation}")
        log.info(f"  仅备份: {backup_only}")
        log.info("")
        
        if backup_only:
            # 只执行备份
            self.backup_training_data(use_advanced_factors, neg_version)
            self._print_summary()
            return
        
        try:
            # Step 1: 数据准备
            if not skip_data_prep:
                self._step_prepare_positive_samples()
                self._step_prepare_negative_samples(neg_version)
                if use_advanced_factors:
                    self._step_add_advanced_factors()
            else:
                log.info("⏭️  跳过数据准备步骤")
            
            # Step 2: 训练模型
            if not skip_training:
                self._step_train_model(use_advanced_factors, neg_version)
            else:
                log.info("⏭️  跳过训练步骤")
            
            # Step 3: Walk-forward 验证
            if not skip_validation:
                self._step_walk_forward_validation(use_advanced_factors, neg_version)
            else:
                log.info("⏭️  跳过验证步骤")
            
            # Step 4: 备份训练数据
            self.backup_training_data(use_advanced_factors, neg_version)
            
            # Step 5: 生成版本报告
            self._generate_version_report(use_advanced_factors, neg_version)
            
            # 完成
            self._print_summary()
            
        except Exception as e:
            log.error(f"❌ 流程执行失败: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _step_prepare_positive_samples(self):
        """Step 1a: 准备正样本"""
        log.info("")
        log.info("="*80)
        log.info("Step 1a: 准备正样本数据")
        log.info("="*80)
        
        env = os.environ.copy()
        env['AUTO_CONFIRM'] = '1'  # 自动确认
        
        cmd = ['python', 'scripts/prepare_positive_samples.py']
        log.info(f"执行命令: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, env=env)
        if result.returncode != 0:
            raise RuntimeError("正样本准备失败")
        
        log.success("✓ 正样本准备完成")
    
    def _step_prepare_negative_samples(self, neg_version: str):
        """Step 1b: 准备负样本"""
        log.info("")
        log.info("="*80)
        log.info("Step 1b: 准备负样本数据")
        log.info("="*80)
        
        if neg_version == 'v2':
            cmd = ['python', 'scripts/prepare_negative_samples_v2.py']
        else:
            cmd = ['python', 'scripts/prepare_negative_samples.py']
        
        log.info(f"执行命令: {' '.join(cmd)}")
        
        result = subprocess.run(cmd)
        if result.returncode != 0:
            raise RuntimeError("负样本准备失败")
        
        log.success("✓ 负样本准备完成")
    
    def _step_add_advanced_factors(self):
        """Step 1c: 添加高级技术因子"""
        log.info("")
        log.info("="*80)
        log.info("Step 1c: 添加高级技术因子")
        log.info("="*80)
        
        # 正样本添加因子
        cmd_pos = ['python', 'scripts/add_advanced_factors.py', '--sample-type', 'positive']
        log.info(f"执行命令: {' '.join(cmd_pos)}")
        result = subprocess.run(cmd_pos)
        if result.returncode != 0:
            raise RuntimeError("正样本添加高级因子失败")
        
        # 负样本添加因子
        cmd_neg = ['python', 'scripts/add_advanced_factors.py', '--sample-type', 'negative']
        log.info(f"执行命令: {' '.join(cmd_neg)}")
        result = subprocess.run(cmd_neg)
        if result.returncode != 0:
            raise RuntimeError("负样本添加高级因子失败")
        
        log.success("✓ 高级技术因子添加完成")
    
    def _step_train_model(self, use_advanced_factors: bool, neg_version: str):
        """Step 2: 训练模型"""
        log.info("")
        log.info("="*80)
        log.info("Step 2: 训练模型")
        log.info("="*80)
        
        cmd = [
            'python', 'scripts/train_xgboost_timeseries.py',
            '--neg-version', neg_version,
            '--version', self.version,
            '--model-name', self.model_name
        ]
        
        if use_advanced_factors:
            cmd.append('--use-advanced-factors')
        
        log.info(f"执行命令: {' '.join(cmd)}")
        
        result = subprocess.run(cmd)
        if result.returncode != 0:
            raise RuntimeError("模型训练失败")
        
        log.success("✓ 模型训练完成")
    
    def _step_walk_forward_validation(self, use_advanced_factors: bool, neg_version: str):
        """Step 3: Walk-forward 验证"""
        log.info("")
        log.info("="*80)
        log.info("Step 3: Walk-forward 验证")
        log.info("="*80)
        
        cmd = [
            'python', 'scripts/walk_forward_validation.py',
            '--neg-version', neg_version,
            '--version', self.version,
            '--model-name', self.model_name
        ]
        
        if use_advanced_factors:
            cmd.append('--use-advanced-factors')
        
        log.info(f"执行命令: {' '.join(cmd)}")
        
        result = subprocess.run(cmd)
        if result.returncode != 0:
            log.warning("⚠️ Walk-forward 验证失败，但继续执行后续步骤")
        else:
            log.success("✓ Walk-forward 验证完成")
    
    def backup_training_data(self, use_advanced_factors: bool = True, neg_version: str = 'v2'):
        """
        Step 4: 备份训练数据到版本目录
        
        备份内容：
        - 正样本数据
        - 负样本数据
        - 正样本特征数据
        - 负样本特征数据
        - 训练图表
        - 训练指标
        """
        log.info("")
        log.info("="*80)
        log.info("Step 4: 备份训练数据")
        log.info("="*80)
        
        # 创建目录结构
        backup_dirs = {
            'samples': self.training_data_dir / 'samples',
            'positive_features': self.training_data_dir / 'positive_features',
            'negative_features': self.training_data_dir / 'negative_features',
        }
        
        for name, dir_path in backup_dirs.items():
            dir_path.mkdir(parents=True, exist_ok=True)
            log.info(f"  创建目录: {dir_path}")
        
        # 1. 备份样本数据
        log.info("")
        log.info("📁 备份样本数据...")
        sample_files = [
            ('positive_samples.csv', '正样本'),
            (f'negative_samples_{neg_version}.csv', '负样本'),
            (f'negative_sample_statistics_{neg_version}.json', '负样本统计'),
            ('quality_report.txt', '质量报告'),
        ]
        
        for filename, desc in sample_files:
            src = self.samples_dir / filename
            if src.exists():
                dst = backup_dirs['samples'] / filename
                shutil.copy2(src, dst)
                log.success(f"  ✓ {desc}: {filename}")
            else:
                log.warning(f"  ⚠️ 文件不存在: {src}")
        
        # 2. 备份正样本特征数据
        log.info("")
        log.info("📁 备份正样本特征数据...")
        
        feature_type = 'advanced' if use_advanced_factors else 'with_market'
        pos_feature_files = [
            f'feature_data_34d.csv',
            f'feature_data_34d_{feature_type}.csv',
            f'feature_data_34d_full.csv',
            'sample_statistics.json',
        ]
        
        for filename in pos_feature_files:
            src = self.processed_dir / filename
            if src.exists():
                dst = backup_dirs['positive_features'] / filename
                shutil.copy2(src, dst)
                log.success(f"  ✓ {filename}")
        
        # 3. 备份负样本特征数据
        log.info("")
        log.info("📁 备份负样本特征数据...")
        
        neg_feature_files = [
            f'negative_feature_data_{neg_version}_34d.csv',
            f'negative_feature_data_{neg_version}_34d_{feature_type}.csv',
            f'negative_feature_data_{neg_version}_34d_full.csv',
        ]
        
        for filename in neg_feature_files:
            src = self.features_dir / filename
            if src.exists():
                dst = backup_dirs['negative_features'] / filename
                shutil.copy2(src, dst)
                log.success(f"  ✓ {filename}")
        
        # 4. 备份训练图表
        log.info("")
        log.info("📁 备份训练图表...")
        
        if self.charts_dir.exists():
            for f in self.charts_dir.iterdir():
                if f.is_file():
                    dst = self.training_data_dir / f.name
                    shutil.copy2(f, dst)
            log.success(f"  ✓ 图表文件已复制")
        
        # 5. 备份训练指标
        log.info("")
        log.info("📁 备份训练指标...")
        
        if self.metrics_dir.exists():
            for f in self.metrics_dir.iterdir():
                if f.is_file():
                    dst = self.training_data_dir / f.name
                    shutil.copy2(f, dst)
            log.success(f"  ✓ 指标文件已复制")
        
        # 6. 生成备份说明文件
        self._generate_backup_readme(use_advanced_factors, neg_version)
        
        log.success(f"\n✅ 训练数据备份完成: {self.training_data_dir}")
    
    def _generate_backup_readme(self, use_advanced_factors: bool, neg_version: str):
        """生成备份说明文件"""
        
        # 统计文件数量和大小
        total_size = 0
        file_count = 0
        
        for f in self.training_data_dir.rglob('*'):
            if f.is_file():
                file_count += 1
                total_size += f.stat().st_size
        
        size_mb = total_size / (1024 * 1024)
        
        feature_type = 'advanced' if use_advanced_factors else 'with_market'
        
        readme_content = f"""# 模型版本 {self.version} 训练数据备份

## 版本信息

| 属性 | 值 |
|------|-----|
| **版本号** | {self.version} |
| **模型名称** | {self.model_name} |
| **备份时间** | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} |
| **文件数量** | {file_count} 个 |
| **总大小** | {size_mb:.1f} MB |

## 训练配置

| 配置项 | 值 |
|--------|-----|
| **特征类型** | {feature_type} |
| **负样本版本** | {neg_version} |
| **训练脚本** | `scripts/train_xgboost_timeseries.py` |
| **验证脚本** | `scripts/walk_forward_validation.py` |

## 训练命令

```bash
# 训练模型
python scripts/train_xgboost_timeseries.py \\
  --neg-version {neg_version} \\
  --use-advanced-factors \\
  --version {self.version} \\
  --model-name {self.model_name}

# Walk-forward 验证
python scripts/walk_forward_validation.py \\
  --neg-version {neg_version} \\
  --use-advanced-factors
```

## 备份文件清单

### 1. 样本数据 (`samples/`)
- `positive_samples.csv` - 正样本数据
- `negative_samples_{neg_version}.csv` - 负样本数据
- `negative_sample_statistics_{neg_version}.json` - 负样本统计
- `quality_report.txt` - 样本质量报告

### 2. 正样本特征数据 (`positive_features/`)
- `feature_data_34d.csv` - 基础特征
- `feature_data_34d_{feature_type}.csv` - 高级特征
- `sample_statistics.json` - 样本统计

### 3. 负样本特征数据 (`negative_features/`)
- `negative_feature_data_{neg_version}_34d.csv` - 基础特征
- `negative_feature_data_{neg_version}_34d_{feature_type}.csv` - 高级特征

## 如何复现训练

```bash
# 方式1：使用完整 pipeline（从数据准备开始）
python scripts/train_model_version.py --version {self.version}

# 方式2：使用备份数据重新训练
# 1. 将备份数据复制回训练目录
cp -r training_data/samples/* data/training/samples/
cp -r training_data/positive_features/* data/training/processed/
cp -r training_data/negative_features/* data/training/features/

# 2. 运行训练
python scripts/train_xgboost_timeseries.py \\
  --neg-version {neg_version} \\
  --use-advanced-factors \\
  --version {self.version}
```

---
*由 train_model_version.py 自动生成*
"""
        
        readme_path = self.training_data_dir / 'BACKUP_README.md'
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        log.success(f"  ✓ 备份说明: BACKUP_README.md")
    
    def _generate_version_report(self, use_advanced_factors: bool, neg_version: str):
        """Step 5: 生成版本报告"""
        log.info("")
        log.info("="*80)
        log.info("Step 5: 生成版本报告")
        log.info("="*80)
        
        # 读取训练指标
        metrics = {}
        metrics_file = self.version_dir / 'training' / 'metrics.json'
        if metrics_file.exists():
            with open(metrics_file, 'r', encoding='utf-8') as f:
                metrics = json.load(f)
        
        # 读取验证结果
        validation_results = {}
        validation_file = Path('data/results/walk_forward_validation_results.json')
        if validation_file.exists():
            with open(validation_file, 'r', encoding='utf-8') as f:
                validation_results = json.load(f)
        
        # 生成报告
        report = {
            'version': self.version,
            'model_name': self.model_name,
            'generated_at': datetime.now().isoformat(),
            'training': {
                'script': 'scripts/train_xgboost_timeseries.py',
                'config': {
                    'feature_type': 'advanced' if use_advanced_factors else 'with_market',
                    'neg_version': neg_version
                },
                'metrics': metrics
            },
            'validation': {
                'script': 'scripts/walk_forward_validation.py',
                'results': validation_results.get('summary', {})
            },
            'backup': {
                'location': str(self.training_data_dir),
                'timestamp': datetime.now().isoformat()
            }
        }
        
        report_file = self.version_dir / 'version_report.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        log.success(f"✓ 版本报告: {report_file}")
    
    def _print_summary(self):
        """打印执行总结"""
        elapsed_time = time.time() - self.start_time
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)
        
        log.info("")
        log.info("="*80)
        log.success(f"✅ 模型版本 {self.version} 训练流程完成！")
        log.info("="*80)
        log.info("")
        log.info(f"⏱️  总耗时: {hours}小时{minutes}分钟{seconds}秒")
        log.info("")
        log.info("📁 文件位置:")
        log.info(f"   模型目录: {self.version_dir}")
        log.info(f"   训练数据备份: {self.training_data_dir}")
        log.info("")
        log.info("💡 下一步:")
        log.info(f"   1. 查看版本报告: cat {self.version_dir}/version_report.json")
        log.info(f"   2. 查看训练指标: cat {self.version_dir}/training/metrics.json")
        log.info(f"   3. 提升版本状态:")
        log.info(f"      python -c \"from src.models.lifecycle import ModelIterator; mi = ModelIterator('{self.model_name}'); mi.set_current_version('{self.version}', 'production')\"")
        log.info(f"   4. 运行预测: python scripts/score_current_stocks.py --version {self.version}")
        log.info("")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='模型版本训练完整流程',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 完整训练流程（从数据准备到备份）
  python scripts/train_model_version.py --version v1.5.0

  # 跳过数据准备（使用现有数据）
  python scripts/train_model_version.py --version v1.5.0 --skip-data-prep

  # 只训练不验证
  python scripts/train_model_version.py --version v1.5.0 --skip-data-prep --skip-validation

  # 只备份数据（用于已有版本）
  python scripts/train_model_version.py --version v1.4.0 --backup-only

  # 使用基础特征（不使用高级因子）
  python scripts/train_model_version.py --version v1.5.0 --no-advanced-factors
        """
    )
    
    parser.add_argument('--version', type=str, required=True,
                       help='模型版本号（如 v1.5.0）')
    parser.add_argument('--model-name', type=str, default='breakout_launch_scorer',
                       help='模型名称（默认: breakout_launch_scorer）')
    parser.add_argument('--neg-version', type=str, default='v2', choices=['v1', 'v2'],
                       help='负样本版本（默认: v2）')
    parser.add_argument('--skip-data-prep', action='store_true',
                       help='跳过数据准备步骤（使用现有数据）')
    parser.add_argument('--skip-training', action='store_true',
                       help='跳过训练步骤')
    parser.add_argument('--skip-validation', action='store_true',
                       help='跳过 walk-forward 验证步骤')
    parser.add_argument('--backup-only', action='store_true',
                       help='只执行备份（用于已有版本）')
    parser.add_argument('--no-advanced-factors', action='store_true',
                       help='不使用高级技术因子')
    
    args = parser.parse_args()
    
    # 创建训练器并执行
    trainer = ModelVersionTrainer(
        version=args.version,
        model_name=args.model_name
    )
    
    trainer.run_full_pipeline(
        skip_data_prep=args.skip_data_prep,
        skip_training=args.skip_training,
        skip_validation=args.skip_validation,
        backup_only=args.backup_only,
        use_advanced_factors=not args.no_advanced_factors,
        neg_version=args.neg_version
    )


if __name__ == '__main__':
    main()

