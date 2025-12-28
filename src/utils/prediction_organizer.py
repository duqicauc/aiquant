"""
预测结果整理工具

用于将最新预测结果移动到历史目录，并管理预测结果的归档
"""
import os
import shutil
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, List
from src.utils.logger import log


def archive_prediction_to_history(
    model_name: str,
    prediction_date: str,
    result_dir: str = None,
    history_dir: str = None
) -> bool:
    """
    将最新预测结果归档到历史目录

    Args:
        model_name: 模型名称
        prediction_date: 预测日期（YYYYMMDD格式）
        result_dir: 最新结果目录（默认: data/result/{model_name}）
        history_dir: 历史目录（默认: data/prediction/history/{model_name}/{prediction_date}）

    Returns:
        是否成功归档
    """
    if result_dir is None:
        result_dir = f"data/result/{model_name}"
    if history_dir is None:
        history_dir = f"data/prediction/history/{model_name}/{prediction_date}"

    result_path = Path(result_dir)
    history_path = Path(history_dir)

    if not result_path.exists():
        log.warning(f"结果目录不存在: {result_dir}")
        return False

    # 创建历史目录
    history_path.mkdir(parents=True, exist_ok=True)

    # 查找该日期的所有预测文件
    files_moved = []
    patterns = [
        f"*{prediction_date}*.csv",
        f"*{prediction_date}*.txt",
        f"*{prediction_date}*.json"
    ]

    for pattern in patterns:
        for file_path in result_path.glob(pattern):
            if file_path.is_file():
                dest_path = history_path / file_path.name
                # 如果目标文件已存在，添加时间戳后缀
                if dest_path.exists():
                    timestamp = datetime.now().strftime('%H%M%S')
                    name_parts = dest_path.stem.split('_')
                    name_parts.append(timestamp)
                    dest_path = history_path / f"{'_'.join(name_parts)}{dest_path.suffix}"

                shutil.copy2(file_path, dest_path)
                files_moved.append(file_path.name)
                log.info(f"  ✓ 已归档: {file_path.name}")

    if files_moved:
        # 更新历史索引
        update_history_index(model_name, prediction_date, history_path, files_moved)
        log.success(f"✓ 预测结果已归档到: {history_dir} ({len(files_moved)} 个文件)")
        return True
    else:
        log.warning(f"未找到预测日期 {prediction_date} 的文件")
        return False


def update_history_index(
    model_name: str,
    prediction_date: str,
    history_path: Path,
    files: List[str]
):
    """
    更新历史预测索引文件

    Args:
        model_name: 模型名称
        prediction_date: 预测日期
        history_path: 历史目录路径
        files: 归档的文件列表
    """
    index_file = history_path / "index.json"

    if index_file.exists():
        with open(index_file, 'r', encoding='utf-8') as f:
            index_data = json.load(f)
    else:
        index_data = {
            "model_name": model_name,
            "prediction_date": prediction_date,
            "archived_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "files": []
        }

    # 更新文件列表
    for file in files:
        if file not in index_data["files"]:
            index_data["files"].append(file)

    index_data["last_updated"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    with open(index_file, 'w', encoding='utf-8') as f:
        json.dump(index_data, f, indent=2, ensure_ascii=False)


def clean_old_results(
    model_name: str,
    keep_days: int = 7,
    result_dir: str = None
) -> int:
    """
    清理旧的结果文件（保留最近N天）

    Args:
        model_name: 模型名称
        keep_days: 保留天数（默认7天）
        result_dir: 结果目录（默认: data/result/{model_name}）

    Returns:
        清理的文件数量
    """
    if result_dir is None:
        result_dir = f"data/result/{model_name}"

    result_path = Path(result_dir)
    if not result_path.exists():
        return 0

    cutoff_time = datetime.now().timestamp() - (keep_days * 24 * 60 * 60)
    files_removed = 0

    for file_path in result_path.iterdir():
        if file_path.is_file():
            if file_path.stat().st_mtime < cutoff_time:
                file_path.unlink()
                files_removed += 1
                log.debug(f"已删除旧文件: {file_path.name}")

    if files_removed > 0:
        log.info(f"✓ 已清理 {files_removed} 个超过 {keep_days} 天的旧文件")

    return files_removed

