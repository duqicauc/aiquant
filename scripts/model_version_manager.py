#!/usr/bin/env python3
"""
模型版本管理命令行工具

功能：
- 查看版本状态
- 比较版本差异
- 设置当前版本
- 清理过时版本
- 版本提升（promotion）

使用方法：
    # 查看状态
    python scripts/model_version_manager.py status
    python scripts/model_version_manager.py status --model breakout_launch_scorer
    
    # 列出所有版本
    python scripts/model_version_manager.py list
    python scripts/model_version_manager.py list --status development
    
    # 比较版本
    python scripts/model_version_manager.py compare v1.3.0 v1.4.0
    
    # 设置当前版本
    python scripts/model_version_manager.py set-current v1.4.0 --env production
    
    # 提升版本
    python scripts/model_version_manager.py promote v1.4.0 --to staging
    
    # 清理过时版本
    python scripts/model_version_manager.py cleanup --dry-run
    python scripts/model_version_manager.py cleanup --keep 3
    
    # 归档版本
    python scripts/model_version_manager.py archive v1.0.0-legacy
"""
import sys
import os
import argparse

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.lifecycle.iterator import ModelIterator
from src.utils.logger import log


def get_default_model():
    """获取默认模型名称"""
    return "breakout_launch_scorer"


def cmd_status(args):
    """显示版本状态"""
    iterator = ModelIterator(args.model)
    iterator.print_status()


def cmd_list(args):
    """列出所有版本"""
    iterator = ModelIterator(args.model)
    
    if args.status:
        versions = iterator.list_versions(status=args.status)
        print(f"📋 {args.status} 状态的版本:")
    else:
        versions = iterator.list_versions()
        print(f"📋 所有版本 ({len(versions)} 个):")
    
    if not versions:
        print("  (无)")
        return
    
    for v in versions:
        try:
            info = iterator.get_version_info(v)
            status = info.get('status', 'unknown')
            created = info.get('created_at', '')[:10]
            
            # 获取测试集指标
            metrics = info.get('metrics', {}).get('test', {})
            auc = metrics.get('auc', 0)
            f1 = metrics.get('f1', 0)
            
            status_icon = {
                'production': '🟢',
                'staging': '🟡',
                'testing': '🟠',
                'development': '⚪',
            }.get(status, '❓')
            
            print(f"  {status_icon} {v:<20} [{status:<12}] AUC={auc:.4f} F1={f1:.4f} ({created})")
        except Exception as e:
            print(f"  ❓ {v:<20} [读取失败: {e}]")


def cmd_compare(args):
    """比较两个版本"""
    iterator = ModelIterator(args.model)
    
    try:
        comparison = iterator.compare_versions(args.version_a, args.version_b)
        iterator.print_comparison(comparison)
    except Exception as e:
        log.error(f"比较失败: {e}")
        sys.exit(1)


def cmd_set_current(args):
    """设置当前版本"""
    iterator = ModelIterator(args.model)
    
    try:
        iterator.set_current_version(args.version, args.env)
        log.success(f"✅ 已设置 {args.env} 环境的当前版本为 {args.version}")
    except Exception as e:
        log.error(f"设置失败: {e}")
        sys.exit(1)


def cmd_promote(args):
    """提升版本到指定环境"""
    iterator = ModelIterator(args.model)
    
    try:
        iterator.promote_version(args.version, args.to)
        log.success(f"✅ 已将 {args.version} 提升到 {args.to} 环境")
    except Exception as e:
        log.error(f"提升失败: {e}")
        sys.exit(1)


def cmd_cleanup(args):
    """清理过时版本"""
    iterator = ModelIterator(args.model)
    
    if args.dry_run:
        log.info("🔍 预览模式（不会实际删除）")
    
    cleaned = iterator.cleanup(keep_latest_n=args.keep, dry_run=args.dry_run)
    
    if not cleaned:
        log.info("✅ 没有需要清理的版本")
    elif args.dry_run:
        log.warning(f"⚠️  发现 {len(cleaned)} 个可清理版本，使用 --no-dry-run 执行清理")


def cmd_archive(args):
    """归档指定版本"""
    iterator = ModelIterator(args.model)
    
    try:
        archived_path = iterator.archive_version(args.version)
        log.success(f"✅ 已归档: {args.version} → {archived_path}")
    except Exception as e:
        log.error(f"归档失败: {e}")
        sys.exit(1)


def cmd_info(args):
    """显示版本详细信息"""
    iterator = ModelIterator(args.model)
    
    try:
        info = iterator.get_version_info(args.version)
        
        print("=" * 70)
        print(f"📦 版本详情: {args.version}")
        print("=" * 70)
        
        print(f"\n📋 基本信息:")
        print(f"  模型名称: {info.get('model_name')}")
        print(f"  显示名称: {info.get('display_name', '-')}")
        print(f"  状态: {info.get('status')}")
        print(f"  创建时间: {info.get('created_at')}")
        print(f"  创建者: {info.get('created_by')}")
        print(f"  父版本: {info.get('parent_version', '-')}")
        
        # 指标
        metrics = info.get('metrics', {}).get('test', {})
        if metrics:
            print(f"\n📊 测试集指标:")
            print(f"  准确率: {metrics.get('accuracy', 0):.4f}")
            print(f"  精确率: {metrics.get('precision', 0):.4f}")
            print(f"  召回率: {metrics.get('recall', 0):.4f}")
            print(f"  F1分数: {metrics.get('f1', 0):.4f}")
            print(f"  AUC: {metrics.get('auc', 0):.4f}")
        
        # 训练信息
        training = info.get('training', {})
        if training:
            print(f"\n🏋️ 训练信息:")
            print(f"  训练样本: {training.get('samples', {}).get('train', '-')}")
            print(f"  测试样本: {training.get('samples', {}).get('test', '-')}")
            print(f"  训练时间: {training.get('duration_seconds', '-')} 秒")
            print(f"  训练数据范围: {training.get('train_date_range', '-')}")
            print(f"  测试数据范围: {training.get('test_date_range', '-')}")
        
        # 备注
        notes = info.get('notes')
        if notes:
            print(f"\n📝 备注: {notes}")
        
        print("=" * 70)
        
    except Exception as e:
        log.error(f"获取版本信息失败: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='模型版本管理工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--model', '-m',
        default=get_default_model(),
        help=f'模型名称 (默认: {get_default_model()})'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # status 命令
    parser_status = subparsers.add_parser('status', help='显示版本状态')
    parser_status.set_defaults(func=cmd_status)
    
    # list 命令
    parser_list = subparsers.add_parser('list', help='列出所有版本')
    parser_list.add_argument('--status', '-s', help='按状态过滤')
    parser_list.set_defaults(func=cmd_list)
    
    # info 命令
    parser_info = subparsers.add_parser('info', help='显示版本详细信息')
    parser_info.add_argument('version', help='版本号')
    parser_info.set_defaults(func=cmd_info)
    
    # compare 命令
    parser_compare = subparsers.add_parser('compare', help='比较两个版本')
    parser_compare.add_argument('version_a', help='版本A（通常是旧版本）')
    parser_compare.add_argument('version_b', help='版本B（通常是新版本）')
    parser_compare.set_defaults(func=cmd_compare)
    
    # set-current 命令
    parser_set = subparsers.add_parser('set-current', help='设置当前版本')
    parser_set.add_argument('version', help='版本号')
    parser_set.add_argument('--env', '-e', default='production',
                           choices=['production', 'staging', 'testing', 'development'],
                           help='环境 (默认: production)')
    parser_set.set_defaults(func=cmd_set_current)
    
    # promote 命令
    parser_promote = subparsers.add_parser('promote', help='提升版本到指定环境')
    parser_promote.add_argument('version', help='版本号')
    parser_promote.add_argument('--to', '-t', required=True,
                               choices=['testing', 'staging', 'production'],
                               help='目标环境')
    parser_promote.set_defaults(func=cmd_promote)
    
    # cleanup 命令
    parser_cleanup = subparsers.add_parser('cleanup', help='清理过时版本')
    parser_cleanup.add_argument('--keep', '-k', type=int, default=3,
                               help='保留的最新开发/测试版本数量 (默认: 3)')
    parser_cleanup.add_argument('--dry-run', action='store_true', default=True,
                               help='预览模式，不实际删除 (默认)')
    parser_cleanup.add_argument('--no-dry-run', dest='dry_run', action='store_false',
                               help='执行实际清理')
    parser_cleanup.set_defaults(func=cmd_cleanup)
    
    # archive 命令
    parser_archive = subparsers.add_parser('archive', help='归档指定版本')
    parser_archive.add_argument('version', help='版本号')
    parser_archive.set_defaults(func=cmd_archive)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    args.func(args)


if __name__ == '__main__':
    main()

