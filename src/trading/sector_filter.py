#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
热点板块过滤器 - 结合Tushare板块数据增强选股

功能：
1. 获取当日热门行业/概念板块
2. 判断股票是否属于热点板块
3. 计算板块热度加成系数（用于调整买入金额）
4. 支持十五五政策主题映射
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd

from src.data.fetcher.tushare_fetcher import TushareFetcher
from src.utils.logger import log


# 政策主题 → 关键词映射（可扩展）
POLICY_THEME_MAP = {
    "商业航天": ["航天", "卫星", "北斗", "火箭", "商业航天", "低空经济", "通航"],
    "创新药": ["创新药", "生物医药", "CXO", "CRO", "生物制品", "仿制药", "疫苗"],
    "CPO": ["CPO", "光模块", "光通信", "硅光", "光芯片", "光器件"],
    "PCB": ["PCB", "印制电路", "覆铜板", "电子元件", "电路板", "HDI"],
    "电力": ["电力", "电网", "特高压", "储能", "新能源", "核电", "水电", "火电", "风电", "光伏"],
    "人工智能": ["人工智能", "AI", "大模型", "算力", "智算", "AIGC", "ChatGPT"],
    "半导体": ["半导体", "芯片", "集成电路", "光刻", "刻蚀", "EDA", "先进封装"],
    "高端制造": ["高端装备", "工业机器人", "数控机床", "智能制造", "工业母机"],
    "新质生产力": ["新质生产力", "量子", "脑机", "6G", "人形机器人", "固态电池"],
    "自主可控": ["信创", "国产替代", "操作系统", "数据库", "中间件", "办公软件"],
}


class SectorFilter:
    """热点板块过滤器"""

    def __init__(
        self,
        tushare_fetcher: Optional[TushareFetcher] = None,
        cache_dir: str = "data/cache/sector",
        hot_industry_top_n: int = 10,
        hot_concept_top_n: int = 20,
        hot_moneyflow_top_n: int = 15,
        industry_boost_max: float = 0.30,
        concept_boost_max: float = 0.40,
        policy_boost_max: float = 0.25,
        enable_policy: bool = True,
    ):
        self.fetcher = tushare_fetcher or TushareFetcher()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.hot_industry_top_n = hot_industry_top_n
        self.hot_concept_top_n = hot_concept_top_n
        self.hot_moneyflow_top_n = hot_moneyflow_top_n
        self.industry_boost_max = industry_boost_max
        self.concept_boost_max = concept_boost_max
        self.policy_boost_max = policy_boost_max
        self.enable_policy = enable_policy

        # 内存缓存
        self._hot_sectors_cache: Dict[str, dict] = {}
        self._stock_industry_cache: Dict[str, str] = {}
        self._stock_concepts_cache: Dict[str, List[str]] = {}
        self._concept_members_cache: Dict[str, Set[str]] = {}

    # ------------------------------------------------------------------
    # 缓存读写
    # ------------------------------------------------------------------
    def _cache_path(self, trade_date: str, suffix: str) -> Path:
        return self.cache_dir / f"{trade_date}_{suffix}.json"

    def _load_cache(self, trade_date: str, suffix: str) -> Optional[dict]:
        path = self._cache_path(trade_date, suffix)
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    def _save_cache(self, trade_date: str, suffix: str, data: dict):
        path = self._cache_path(trade_date, suffix)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            log.debug(f"缓存写入失败: {e}")

    # ------------------------------------------------------------------
    # 热点板块获取
    # ------------------------------------------------------------------
    def get_hot_sectors(self, trade_date: str, force_refresh: bool = False) -> dict:
        """
        获取当日热门板块信息（三层数据源）

        Returns:
            {
                "industries": ["行业名1", "行业名2", ...],   # 申万热门行业TopN
                "concepts": ["概念名1", "概念名2", ...],     # 同花顺热门概念TopN
                "moneyflow": ["板块名1", ...],               # 资金流向TopN
                "policy_themes": ["政策主题1", ...],         # 当日活跃的政策主题
            }
        """
        if not force_refresh and trade_date in self._hot_sectors_cache:
            return self._hot_sectors_cache[trade_date]

        # 尝试从本地缓存读取
        cached = self._load_cache(trade_date, "hot_sectors")
        if cached and not force_refresh:
            self._hot_sectors_cache[trade_date] = cached
            return cached

        result = {
            "industries": [],
            "concepts": [],
            "moneyflow": [],
            "policy_themes": [],
        }

        # 1. 热门申万行业
        try:
            df_ind = self.fetcher.get_hot_industries(
                trade_date, top_n=self.hot_industry_top_n, min_pct_chg=0.5
            )
            if not df_ind.empty and "name" in df_ind.columns:
                result["industries"] = df_ind["name"].astype(str).tolist()
        except Exception as e:
            log.debug(f"获取热门行业失败 {trade_date}: {e}")

        # 2. 热门概念（同花顺热榜）
        try:
            df_con = self.fetcher.get_ths_hot(
                trade_date, market="概念板块", top_n=self.hot_concept_top_n
            )
            if not df_con.empty and "ts_name" in df_con.columns:
                result["concepts"] = df_con["ts_name"].astype(str).tolist()
        except Exception as e:
            log.debug(f"获取热门概念失败 {trade_date}: {e}")

        # 3. 板块资金流向
        try:
            df_mf = self.fetcher.get_sector_moneyflow(
                trade_date, top_n=self.hot_moneyflow_top_n
            )
            if not df_mf.empty and "ts_name" in df_mf.columns:
                result["moneyflow"] = df_mf["ts_name"].astype(str).tolist()
        except Exception as e:
            log.debug(f"获取板块资金流向失败 {trade_date}: {e}")

        # 4. 政策主题匹配：从热门概念/行业中识别活跃政策主题
        if self.enable_policy:
            all_hot_names = result["industries"] + result["concepts"] + result["moneyflow"]
            matched_themes = set()
            for theme, keywords in POLICY_THEME_MAP.items():
                for hot_name in all_hot_names:
                    for kw in keywords:
                        if kw in hot_name or hot_name in kw:
                            matched_themes.add(theme)
                            break
            result["policy_themes"] = list(matched_themes)

        self._hot_sectors_cache[trade_date] = result
        self._save_cache(trade_date, "hot_sectors", result)
        log.info(f"  热点板块 [{trade_date}]: 行业{len(result['industries'])}个, 概念{len(result['concepts'])}个, 政策主题{len(result['policy_themes'])}个")
        return result

    # ------------------------------------------------------------------
    # 股票板块信息获取
    # ------------------------------------------------------------------
    def get_stock_industry(self, ts_code: str) -> str:
        """获取股票所属申万行业（带缓存）"""
        if ts_code in self._stock_industry_cache:
            return self._stock_industry_cache[ts_code]

        try:
            mapping = self.fetcher.get_stock_industry_map([ts_code])
            industry = mapping.get(ts_code, "")
        except Exception:
            industry = ""

        self._stock_industry_cache[ts_code] = industry
        return industry

    def get_stock_concepts(self, ts_code: str) -> List[str]:
        """获取股票所属概念列表（带缓存）"""
        if ts_code in self._stock_concepts_cache:
            return self._stock_concepts_cache[ts_code]

        try:
            df = self.fetcher.concept_detail(ts_code=ts_code)
            if df is not None and not df.empty and "concept_name" in df.columns:
                concepts = df["concept_name"].astype(str).tolist()
            else:
                concepts = []
        except Exception:
            concepts = []

        self._stock_concepts_cache[ts_code] = concepts
        return concepts

    # ------------------------------------------------------------------
    # 热度加成计算
    # ------------------------------------------------------------------
    def get_sector_boost(
        self,
        ts_code: str,
        trade_date: str,
        hot_sectors: Optional[dict] = None,
    ) -> float:
        """
        计算股票的板块热度加成系数

        Args:
            ts_code: 股票代码
            trade_date: 交易日期 YYYYMMDD
            hot_sectors: 预计算的热点板块（避免重复查询）

        Returns:
            boost: 加成系数，默认1.0，热点股票>1.0，最大约2.0
        """
        if hot_sectors is None:
            hot_sectors = self.get_hot_sectors(trade_date)

        boost = 1.0

        # 1. 行业热点匹配
        if hot_sectors["industries"]:
            stock_ind = self.get_stock_industry(ts_code)
            if stock_ind:
                for rank, hot_ind in enumerate(hot_sectors["industries"]):
                    if stock_ind == hot_ind or hot_ind in stock_ind or stock_ind in hot_ind:
                        weight = 1.0 - rank / len(hot_sectors["industries"])
                        boost *= (1.0 + self.industry_boost_max * weight)
                        break

        # 2. 概念热点匹配
        if hot_sectors["concepts"]:
            stock_concepts = self.get_stock_concepts(ts_code)
            for rank, hot_con in enumerate(hot_sectors["concepts"]):
                matched = False
                for sc in stock_concepts:
                    if hot_con in sc or sc in hot_con:
                        matched = True
                        break
                if matched:
                    weight = 1.0 - rank / len(hot_sectors["concepts"])
                    boost *= (1.0 + self.concept_boost_max * weight)
                    break

        # 3. 政策主题匹配（额外加成）
        if self.enable_policy and hot_sectors.get("policy_themes"):
            stock_concepts = self.get_stock_concepts(ts_code)
            stock_ind = self.get_stock_industry(ts_code)
            all_labels = stock_concepts + [stock_ind]

            for theme in hot_sectors["policy_themes"]:
                keywords = POLICY_THEME_MAP.get(theme, [])
                for label in all_labels:
                    for kw in keywords:
                        if kw in label or label in kw:
                            boost *= (1.0 + self.policy_boost_max)
                            break

        return min(boost, 2.5)

    def filter_hot_stocks(
        self,
        df_preds: pd.DataFrame,
        trade_date: str,
        min_boost: float = 1.0,
        hot_sectors: Optional[dict] = None,
    ) -> pd.DataFrame:
        """
        对预测结果叠加板块热度筛选/排序

        Args:
            df_preds: 预测结果DataFrame，含 ts_code, score/prob 等
            trade_date: 交易日期
            min_boost: 最小加成阈值（<1.0表示允许非热点但降低权重）
            hot_sectors: 预计算的热点板块

        Returns:
            增加 'sector_boost' 列的DataFrame，按 (score * boost) 降序排列
        """
        if hot_sectors is None:
            hot_sectors = self.get_hot_sectors(trade_date)

        df = df_preds.copy()
        df["sector_boost"] = df["ts_code"].apply(
            lambda x: self.get_sector_boost(x, trade_date, hot_sectors)
        )

        # 如果模型有 score/prob 列，叠加板块加成重新排序
        score_col = None
        for col in ["score", "prob", "prediction", "predicted_score"]:
            if col in df.columns:
                score_col = col
                break

        if score_col:
            df["adjusted_score"] = df[score_col] * df["sector_boost"]
            df = df.sort_values("adjusted_score", ascending=False)
        else:
            df = df.sort_values("sector_boost", ascending=False)

        return df
