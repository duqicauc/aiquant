#!/usr/bin/env python3
"""
AIQuant 项目清理脚本

安全清理项目中的缓存、旧日志、临时文件和生成文件。
大文件（数据库备份、模型备份）仅做报告，不自动删除。

用法:
    python scripts/cleanup_project.py [--dry-run] [--logs-days 30]
"""

import argparse
import os
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.resolve()


def get_size_readable(size_bytes: int) -> str:
    """将字节大小转换为可读格式."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def get_directory_size(path: Path) -> int:
    """递归计算目录大小."""
    total = 0
    try:
        for entry in os.scandir(path):
            if entry.is_file(follow_symlinks=False):
                total += entry.stat(follow_symlinks=False).st_size
            elif entry.is_dir(follow_symlinks=False):
                total += get_directory_size(Path(entry.path))
    except (OSError, PermissionError):
        pass
    return total


def remove_path(path: Path, dry_run: bool, reason: str) -> int:
    """删除文件或目录，返回释放的字节数."""
    if not path.exists():
        return 0

    size = get_directory_size(path) if path.is_dir() else path.stat().st_size
    action = "[DRY-RUN 将删除]" if dry_run else "[已删除]"
    print(f"  {action} {reason}: {path.relative_to(PROJECT_ROOT)} ({get_size_readable(size)})")

    if not dry_run:
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        except (OSError, PermissionError) as e:
            print(f"    ⚠️ 删除失败: {e}")
            return 0
    return size


def clean_pycache(dry_run: bool) -> int:
    """清理所有 __pycache__ 目录和 .pyc/.pyo 文件."""
    print("\n🧹 清理 Python 缓存...")
    total = 0
    count = 0

    for root, dirs, files in os.walk(PROJECT_ROOT):
        root_path = Path(root)
        # 跳过 .git 目录
        if ".git" in root_path.parts:
            continue

        for d in dirs:
            if d == "__pycache__":
                p = root_path / d
                total += remove_path(p, dry_run, "Python缓存")
                count += 1

        for f in files:
            if f.endswith((".pyc", ".pyo")):
                p = root_path / f
                total += remove_path(p, dry_run, "编译文件")
                count += 1

    print(f"  共处理 {count} 项，释放 {get_size_readable(total)}")
    return total


def clean_tool_caches(dry_run: bool) -> int:
    """清理工具生成的缓存目录."""
    print("\n🧹 清理工具缓存...")
    total = 0

    cache_dirs = [
        PROJECT_ROOT / ".pytest_cache",
        PROJECT_ROOT / ".ruff_cache",
        PROJECT_ROOT / "htmlcov",
        PROJECT_ROOT / "catboost_info",
        PROJECT_ROOT / ".mypy_cache",
    ]

    for d in cache_dirs:
        if d.exists():
            total += remove_path(d, dry_run, "工具缓存")

    return total


def clean_old_logs(dry_run: bool, days: int) -> int:
    """清理超过 N 天的日志文件."""
    print(f"\n🧹 清理 {days} 天前的日志文件...")
    total = 0
    count = 0
    cutoff = datetime.now() - timedelta(days=days)
    logs_dir = PROJECT_ROOT / "logs"

    if not logs_dir.exists():
        print("  logs/ 目录不存在，跳过")
        return 0

    # 保留的日志模式（即使很旧也不删）
    protected_patterns = ["aiquant.log"]

    for f in logs_dir.iterdir():
        if not f.is_file():
            continue
        if f.name in protected_patterns:
            continue

        try:
            mtime = datetime.fromtimestamp(f.stat().st_mtime)
            if mtime < cutoff:
                total += remove_path(f, dry_run, f"旧日志 ({mtime.strftime('%Y-%m-%d')})")
                count += 1
        except (OSError, PermissionError):
            continue

    print(f"  共处理 {count} 个文件，释放 {get_size_readable(total)}")
    return total


def clean_root_temp_files(dry_run: bool) -> int:
    """清理根目录临时文件."""
    print("\n🧹 清理根目录临时文件...")
    total = 0

    temp_files = [
        PROJECT_ROOT / "align_output.log",
    ]

    # 空的 quant_data.db
    empty_db = PROJECT_ROOT / "data" / "quant_data.db"
    if empty_db.exists() and empty_db.stat().st_size == 0:
        temp_files.append(empty_db)

    for f in temp_files:
        if f.exists():
            total += remove_path(f, dry_run, "临时文件")

    return total


def report_large_files() -> None:
    """报告大文件和备份（仅报告，不删除）."""
    print("\n📊 大文件/备份报告（仅报告，请手动处理）:")
    print("-" * 60)

    items = []

    # 数据库备份
    cache_dir = PROJECT_ROOT / "data" / "cache"
    if cache_dir.exists():
        for f in cache_dir.iterdir():
            if f.is_file() and ("backup" in f.name.lower() or f.name.startswith("quant_data_backup")):
                size = f.stat().st_size
                items.append((f, size, "数据库备份"))

    # 模型备份
    models_backup = PROJECT_ROOT / "data" / "models_backup_v270_20260422"
    if models_backup.exists():
        size = get_directory_size(models_backup)
        items.append((models_backup, size, "模型备份"))

    # logs 中的大 zip 文件
    logs_dir = PROJECT_ROOT / "logs"
    if logs_dir.exists():
        for f in logs_dir.iterdir():
            if f.is_file() and f.suffix == ".zip":
                size = f.stat().st_size
                items.append((f, size, "日志归档"))

    # scripts/archive
    archive_dir = PROJECT_ROOT / "scripts" / "archive"
    if archive_dir.exists():
        size = get_directory_size(archive_dir)
        items.append((archive_dir, size, "历史脚本归档"))

    total_size = sum(s for _, s, _ in items)

    if not items:
        print("  未发现需要报告的大文件/备份")
        return

    for f, size, reason in sorted(items, key=lambda x: x[1], reverse=True):
        print(f"  {get_size_readable(size):>10} | {reason:<12} | {f.relative_to(PROJECT_ROOT)}")

    print("-" * 60)
    print(f"  合计: {get_size_readable(total_size)}")
    print(f"\n  💡 建议手动删除不需要的备份，或使用:")
    print(f"     rm -rf data/cache/quant_data.db.backup_*")
    print(f"     rm -rf data/models_backup_v270_20260422")


def report_version_bloat() -> None:
    """报告版本膨胀的脚本（仅报告）."""
    print("\n📊 版本膨胀脚本报告（请考虑合并为通用脚本+配置）:")
    print("-" * 60)

    scripts_dir = PROJECT_ROOT / "scripts"
    patterns = {
        "predict_v": "预测脚本",
        "train_v": "训练脚本",
        "evaluate_v": "评估脚本",
        "backtest_v": "回测脚本",
    }

    for prefix, label in patterns.items():
        files = sorted([f for f in scripts_dir.iterdir() if f.is_file() and f.name.startswith(prefix) and f.suffix == ".py"])
        if files:
            print(f"\n  {label} ({len(files)} 个):")
            for f in files:
                print(f"    - {f.name}")

    # data_prep 中的多版本
    data_prep_dir = scripts_dir / "data_prep"
    if data_prep_dir.exists():
        multi_versions = []
        for f in data_prep_dir.iterdir():
            if f.is_file() and f.suffix == ".py":
                base = f.name
                # 检测带 _v2, _v3, _optimized 等后缀的
                if any(suffix in base for suffix in ["_v2", "_v3", "_optimized", "_fast", "_advanced", "_checkpoint"]):
                    multi_versions.append(f.name)
        if multi_versions:
            print(f"\n  data_prep 多版本脚本 ({len(multi_versions)} 个):")
            for name in sorted(multi_versions):
                print(f"    - {name}")


def main():
    parser = argparse.ArgumentParser(description="AIQuant 项目清理工具")
    parser.add_argument("--dry-run", action="store_true", help="模拟运行，不实际删除文件")
    parser.add_argument("--logs-days", type=int, default=30, help="清理超过 N 天的日志（默认30）")
    parser.add_argument("--skip-logs", action="store_true", help="跳过日志清理")
    parser.add_argument("--report-only", action="store_true", help="仅报告，不清理（等同于 --dry-run + 显示所有报告）")
    args = parser.parse_args()

    dry_run = args.dry_run or args.report_only

    print("=" * 60)
    print("🚀 AIQuant 项目清理工具")
    print("=" * 60)
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"模式: {'模拟运行 (不删除)' if dry_run else '实际删除'}")

    total_saved = 0

    # 执行清理
    total_saved += clean_pycache(dry_run)
    total_saved += clean_tool_caches(dry_run)
    total_saved += clean_root_temp_files(dry_run)

    if not args.skip_logs:
        total_saved += clean_old_logs(dry_run, args.logs_days)

    # 报告部分
    report_large_files()
    report_version_bloat()

    print("\n" + "=" * 60)
    print(f"✅ 清理完成！预计释放空间: {get_size_readable(total_saved)}")
    print("=" * 60)

    if dry_run:
        print("\n💡 这是模拟运行，实际清理请执行:")
        print(f"   python {Path(__file__).relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
