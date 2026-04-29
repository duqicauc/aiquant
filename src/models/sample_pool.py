#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
样本池管理器 (Sample Pool Manager)

管理训练样本的存储、增量更新、采样和质量检查。
样本池是模型数据飞轮的核心：每天新产生的 label + feature 自动汇入池子，
定期抽样生成训练集。

Usage:
    from src.models.sample_pool import SamplePool
    pool = SamplePool()
    pool.append(new_samples_df)          # 增量添加
    train, val, test = pool.split()      # 时间切分
    pool.export_training_set("v293")     # 导出为训练格式
"""

import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import log


class SamplePool:
    """样本池管理器

    目录结构:
        data/training/pool/
            pool.csv          # 主样本池
            meta.json         # 元数据（最新日期、样本统计）
            snapshots/        # 定期快照
    """

    def __init__(self, pool_dir: Optional[Path] = None):
        self.pool_dir = pool_dir or PROJECT_ROOT / "data" / "training" / "pool"
        self.pool_dir.mkdir(parents=True, exist_ok=True)
        self.pool_file = self.pool_dir / "pool.csv"
        self.meta_file = self.pool_dir / "meta.json"
        self.snapshot_dir = self.pool_dir / "snapshots"
        self.snapshot_dir.mkdir(exist_ok=True)

        self._df: Optional[pd.DataFrame] = None
        self.meta = self._load_meta()

    # ==================== 元数据管理 ====================

    def _load_meta(self) -> dict:
        if self.meta_file.exists():
            try:
                return json.loads(self.meta_file.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {
            "version": "1.0",
            "created_at": pd.Timestamp.now().isoformat(),
            "latest_date": None,
            "total_samples": 0,
            "positive_samples": 0,
            "feature_count": 0,
        }

    def _save_meta(self):
        self.meta["updated_at"] = pd.Timestamp.now().isoformat()
        self.meta_file.write_text(json.dumps(self.meta, indent=2, ensure_ascii=False), encoding="utf-8")

    # ==================== 数据加载 ====================

    def load(self) -> pd.DataFrame:
        """加载样本池"""
        if self._df is not None:
            return self._df

        if self.pool_file.exists():
            self._df = pd.read_csv(self.pool_file)
            log.info(f"样本池加载完成: {len(self._df)} 条")
        else:
            self._df = pd.DataFrame()
            log.info("样本池为空")

        return self._df

    def save(self):
        """保存样本池到磁盘"""
        if self._df is None or self._df.empty:
            log.warning("空样本池，跳过保存")
            return

        self.pool_file.parent.mkdir(parents=True, exist_ok=True)
        self._df.to_csv(self.pool_file, index=False)

        # 更新元数据
        pos = int(self._df["label"].sum()) if "label" in self._df.columns else 0
        latest_date = None
        if "trade_date" in self._df.columns:
            try:
                td = pd.to_datetime(self._df["trade_date"], format="mixed", errors="coerce")
                latest_date = td.max()
                if hasattr(latest_date, "isoformat"):
                    latest_date = latest_date.isoformat()
            except Exception:
                latest_date = str(self._df["trade_date"].max())
        self.meta.update({
            "total_samples": len(self._df),
            "positive_samples": pos,
            "latest_date": latest_date,
            "feature_count": len([c for c in self._df.columns if c not in self._exclude_cols()]),
        })
        self._save_meta()
        log.info(f"样本池已保存: {len(self._df)} 条, 正样本 {pos}")

    # ==================== 样本操作 ====================

    def _exclude_cols(self) -> set:
        return {
            "label", "sample_id", "ts_code", "name",
            "t1_date", "t2_date", "trade_date", "list_date",
            "pattern_type", "days_to_t1",
            "future_high", "future_close_ret", "future_low",
            "future_max_drawdown", "max_return", "label_close",
            "open", "high", "low", "close", "pre_close",
            "change", "pct_chg", "vol", "amount",
        }

    def append(self, df_new: pd.DataFrame, dedup_key: Optional[List[str]] = None):
        """增量添加新样本

        参数:
            df_new: 新样本 DataFrame
            dedup_key: 去重键，默认 ["ts_code", "trade_date"]
        """
        if df_new is None or df_new.empty:
            log.warning("新增样本为空，跳过")
            return

        dedup_key = dedup_key or ["ts_code", "trade_date"]

        df_existing = self.load()

        if not df_existing.empty:
            # 确保列一致
            common_cols = list(set(df_existing.columns) & set(df_new.columns))
            df_existing = df_existing[common_cols]
            df_new = df_new[common_cols]

            # 合并并去重
            df_merged = pd.concat([df_existing, df_new], ignore_index=True)
            df_merged = df_merged.drop_duplicates(subset=dedup_key, keep="last")
            added = len(df_merged) - len(df_existing)
        else:
            df_merged = df_new.copy()
            added = len(df_merged)

        self._df = df_merged
        self.save()
        log.info(f"增量添加完成: 新增 {added} 条, 池子总计 {len(self._df)} 条")

    def snapshot(self, tag: Optional[str] = None) -> Path:
        """创建快照"""
        df = self.load()
        if df.empty:
            log.warning("空样本池，跳过快照")
            return Path()

        tag = tag or pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        snap_path = self.snapshot_dir / f"pool_snapshot_{tag}.csv"
        df.to_csv(snap_path, index=False)
        log.info(f"快照已创建: {snap_path}")
        return snap_path

    # ==================== 数据集切分 ====================

    def split(
        self,
        val_days: int = 60,
        test_days: int = 30,
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """按时间切分训练/验证/测试集

        参数:
            val_days: 验证集天数（从最新日期往前数）
            test_days: 测试集天数
            random_state: 随机种子
        """
        df = self.load()
        if df.empty or "trade_date" not in df.columns:
            log.error("样本池为空或缺少 trade_date")
            return df, pd.DataFrame(), pd.DataFrame()

        df["trade_date"] = pd.to_datetime(df["trade_date"], format="mixed", errors="coerce")
        df = df.sort_values("trade_date")

        max_date = df["trade_date"].max()
        test_cutoff = max_date - pd.Timedelta(days=test_days)
        val_cutoff = test_cutoff - pd.Timedelta(days=val_days)

        test = df[df["trade_date"] > test_cutoff].copy()
        val = df[(df["trade_date"] > val_cutoff) & (df["trade_date"] <= test_cutoff)].copy()
        train = df[df["trade_date"] <= val_cutoff].copy()

        log.info(
            f"数据集切分: 训练 {len(train)} ({train['label'].sum()}正), "
            f"验证 {len(val)} ({val['label'].sum()}正), "
            f"测试 {len(test)} ({test['label'].sum()}正)"
        )
        return train, val, test

    def stratified_sample(
        self,
        n_total: int = 5000,
        pos_ratio: Optional[float] = None,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """分层采样

        参数:
            n_total: 总样本数
            pos_ratio: 正样本比例，None 则保持原比例
            random_state: 随机种子
        """
        df = self.load()
        if df.empty or "label" not in df.columns:
            return df

        pos = df[df["label"] == 1]
        neg = df[df["label"] == 0]

        if pos_ratio is None:
            pos_ratio = len(pos) / len(df)

        n_pos = min(int(n_total * pos_ratio), len(pos))
        n_neg = min(n_total - n_pos, len(neg))

        pos_sample = pos.sample(n=n_pos, random_state=random_state) if n_pos > 0 else pd.DataFrame()
        neg_sample = neg.sample(n=n_neg, random_state=random_state) if n_neg > 0 else pd.DataFrame()

        sampled = pd.concat([pos_sample, neg_sample], ignore_index=True)
        log.info(f"分层采样: {len(sampled)} 条, 正样本率 {sampled['label'].mean()*100:.2f}%")
        return sampled

    # ==================== 训练集导出 ====================

    def export_training_set(
        self,
        version: str = "v293",
        output_dir: Optional[Path] = None,
        feature_cols: Optional[List[str]] = None,
    ) -> Dict[str, Path]:
        """导出为模型训练格式

        返回训练/验证/测试集的 CSV 路径。
        """
        train, val, test = self.split()
        if train.empty:
            log.error("训练集为空")
            return {}

        output_dir = output_dir or PROJECT_ROOT / "data" / "training" / f"auto_{version}"
        output_dir.mkdir(parents=True, exist_ok=True)

        # 自动推断特征列
        if feature_cols is None:
            exclude = self._exclude_cols()
            feature_cols = [c for c in train.columns if c not in exclude]

        paths = {}
        for name, df in [("train", train), ("val", val), ("test", test)]:
            if df.empty:
                continue
            path = output_dir / f"{name}.csv"
            # 保留特征 + 标签 + 标识列
            keep_cols = ["ts_code", "trade_date", "label"] + feature_cols
            keep_cols = [c for c in keep_cols if c in df.columns]
            df[keep_cols].to_csv(path, index=False)
            paths[name] = path
            log.info(f"导出 {name}: {path} ({len(df)} 条)")

        # 保存特征列表
        meta = {
            "version": version,
            "feature_count": len(feature_cols),
            "features": feature_cols,
            "train_samples": len(train),
            "val_samples": len(val),
            "test_samples": len(test),
            "positive_rate": float(train["label"].mean()) if "label" in train.columns else 0,
        }
        meta_path = output_dir / "meta.json"
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

        return paths

    # ==================== 质量检查 ====================

    def quality_report(self) -> dict:
        """样本池质量报告"""
        df = self.load()
        if df.empty:
            return {"status": "empty", "message": "样本池为空"}

        td_min = td_max = None
        if "trade_date" in df.columns:
            try:
                td = pd.to_datetime(df["trade_date"], format="mixed", errors="coerce")
                td_min = str(td.min())
                td_max = str(td.max())
            except Exception:
                td_min = str(df["trade_date"].min())
                td_max = str(df["trade_date"].max())
        report = {
            "status": "ok",
            "total_samples": len(df),
            "date_range": {
                "min": td_min,
                "max": td_max,
            },
            "stock_count": int(df["ts_code"].nunique()) if "ts_code" in df.columns else 0,
        }

        if "label" in df.columns:
            pos = int(df["label"].sum())
            report["positive_samples"] = pos
            report["negative_samples"] = len(df) - pos
            report["positive_rate"] = round(pos / len(df) * 100, 2)

        # 特征缺失率检查
        exclude = self._exclude_cols()
        feature_cols = [c for c in df.columns if c not in exclude]
        missing_rates = {c: round(df[c].isna().mean() * 100, 2) for c in feature_cols if df[c].isna().any()}
        report["high_missing_features"] = {k: v for k, v in missing_rates.items() if v > 10}

        # 标签泄漏检查（future 列不应出现在特征中）
        leak_cols = [c for c in df.columns if c in ["future_high", "future_close_ret", "max_return"]]
        report["potential_leakage"] = leak_cols

        return report


# ==================== CLI ====================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="样本池管理")
    parser.add_argument("action", choices=["info", "snapshot", "export", "quality"])
    parser.add_argument("--version", default="v293", help="训练集版本")
    parser.add_argument("--output", help="输出目录")
    args = parser.parse_args()

    pool = SamplePool()

    if args.action == "info":
        df = pool.load()
        print(f"样本池: {len(df)} 条")
        print(pool.meta)
    elif args.action == "snapshot":
        pool.snapshot()
    elif args.action == "export":
        out = Path(args.output) if args.output else None
        pool.export_training_set(args.version, out)
    elif args.action == "quality":
        import json
        print(json.dumps(pool.quality_report(), indent=2, ensure_ascii=False))
