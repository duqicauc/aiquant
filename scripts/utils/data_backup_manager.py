#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据备份管理工具

功能：
1. 导出SQLite数据到CSV
2. 从CSV导入数据到SQLite
3. 查看备份统计
4. 清理备份数据
"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.storage.backup_cache_manager import BackupCacheManager
from src.utils.logger import log


def export_to_csv(args):
    """导出所有数据到CSV"""
    log.info("="*80)
    log.info("📤 导出SQLite数据到CSV")
    log.info("="*80)
    
    cache = BackupCacheManager(enable_backup=True)
    
    # 指定数据类型
    data_types = None
    if args.data_type:
        data_types = [args.data_type]
    
    cache.export_all_to_csv(data_types=data_types)
    
    log.success("\n✅ 导出完成！")
    log.info("\n💡 提示：")
    log.info("  - CSV文件位置: data/backup/")
    log.info("  - 可以直接用Excel打开查看")
    log.info("  - 可以打包整个backup目录进行迁移")


def import_from_csv(args):
    """从CSV导入数据"""
    log.info("="*80)
    log.info("📥 从CSV导入数据到SQLite")
    log.info("="*80)
    
    cache = BackupCacheManager(enable_backup=True)
    
    # 指定数据类型
    data_types = None
    if args.data_type:
        data_types = [args.data_type]
    
    cache.import_from_csv(data_types=data_types)
    
    log.success("\n✅ 导入完成！")


def show_stats(args):
    """显示备份统计"""
    cache = BackupCacheManager(enable_backup=True)
    
    log.info("="*80)
    log.info("📊 数据备份统计")
    log.info("="*80)
    
    stats = cache.get_backup_stats()
    
    # SQLite统计
    log.info("\n📁 SQLite缓存:")
    if 'sqlite' in stats:
        for key, value in stats['sqlite'].items():
            log.info(f"  {key}: {value:,}")
    
    # CSV备份统计
    log.info("\n📄 CSV备份:")
    if 'csv' in stats and stats['csv']:
        total_files = 0
        for data_type, count in stats['csv'].items():
            log.info(f"  {data_type}: {count:,} 个文件")
            total_files += count
        log.info(f"  总计: {total_files:,} 个文件")
    else:
        log.info("  (无CSV备份)")
    
    # 备份索引
    index_file = cache.backup_dir / 'metadata' / 'backup_index.json'
    if index_file.exists():
        import json
        with open(index_file, 'r', encoding='utf-8') as f:
            index = json.load(f)
        
        log.info("\n📑 备份索引:")
        log.info(f"  备份时间: {index.get('backup_time', 'N/A')}")
        log.info(f"  股票总数: {index.get('total_stocks', 0):,}")
        log.info(f"  文件总数: {index.get('total_files', 0):,}")
    
    log.info("="*80)


def clear_backup(args):
    """清理备份数据"""
    cache = BackupCacheManager(enable_backup=True)
    
    if args.confirm != 'yes':
        log.error("❌ 需要确认才能清理！请使用 --confirm yes")
        return
    
    log.warning("⚠️  警告：即将清理备份数据！")
    
    if args.ts_code:
        cache.clear_csv_backup(ts_code=args.ts_code, data_type=args.data_type)
        log.success(f"✓ 已清理 {args.ts_code} 的备份")
    else:
        cache.clear_csv_backup()
        log.success("✓ 已清理所有CSV备份")


def sync_data(args):
    """同步SQLite和CSV数据"""
    log.info("="*80)
    log.info("🔄 同步数据")
    log.info("="*80)
    
    cache = BackupCacheManager(enable_backup=True)
    
    if args.direction == 'to_csv':
        log.info("从SQLite同步到CSV...")
        cache.export_all_to_csv()
    elif args.direction == 'to_sqlite':
        log.info("从CSV同步到SQLite...")
        cache.import_from_csv()
    else:
        log.error(f"未知的同步方向: {args.direction}")
        return
    
    log.success("✓ 同步完成")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='数据备份管理工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 导出所有数据到CSV
  python scripts/utils/data_backup_manager.py export

  # 导出指定类型数据
  python scripts/utils/data_backup_manager.py export --data-type daily_data

  # 从CSV导入数据
  python scripts/utils/data_backup_manager.py import

  # 查看统计信息
  python scripts/utils/data_backup_manager.py stats

  # 同步数据
  python scripts/utils/data_backup_manager.py sync --direction to_csv

  # 清理备份（需要确认）
  python scripts/utils/data_backup_manager.py clear --confirm yes
        """
    )
    
    subparsers = parser.add_subparsers(dest='action', help='操作类型')
    
    # 导出命令
    parser_export = subparsers.add_parser('export', help='导出数据到CSV')
    parser_export.add_argument('--data-type', choices=['daily_data', 'weekly_data', 'daily_basic', 'stk_factor'],
                              help='指定数据类型（不指定则导出全部）')
    
    # 导入命令
    parser_import = subparsers.add_parser('import', help='从CSV导入数据')
    parser_import.add_argument('--data-type', choices=['daily_data', 'weekly_data', 'daily_basic', 'stk_factor'],
                              help='指定数据类型（不指定则导入全部）')
    
    # 统计命令
    parser_stats = subparsers.add_parser('stats', help='显示统计信息')
    
    # 清理命令
    parser_clear = subparsers.add_parser('clear', help='清理备份数据')
    parser_clear.add_argument('--ts-code', help='股票代码（不指定则清理全部）')
    parser_clear.add_argument('--data-type', choices=['daily_data', 'weekly_data', 'daily_basic', 'stk_factor'],
                             help='数据类型')
    parser_clear.add_argument('--confirm', help='确认清理（必须输入yes）')
    
    # 同步命令
    parser_sync = subparsers.add_parser('sync', help='同步数据')
    parser_sync.add_argument('--direction', choices=['to_csv', 'to_sqlite'], required=True,
                            help='同步方向')
    
    args = parser.parse_args()
    
    if not args.action:
        parser.print_help()
        return
    
    try:
        if args.action == 'export':
            export_to_csv(args)
        elif args.action == 'import':
            import_from_csv(args)
        elif args.action == 'stats':
            show_stats(args)
        elif args.action == 'clear':
            clear_backup(args)
        elif args.action == 'sync':
            sync_data(args)
    
    except KeyboardInterrupt:
        log.warning("\n⚠️  操作已取消")
        sys.exit(1)
    
    except Exception as e:
        log.error(f"\n❌ 操作失败: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()

