#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
热点板块过滤器 v2 - 结合Tushare板块数据增强选股

优化点：
1. 市场环境感知：bear/oscillation市降低加成、非热点反向过滤
2. 热点持续性：连续多日上榜的热点才给高加成
3. 板块轮动检测：热点切换过快时整体降权
4. 政策主题映射：十五五政策主题关键词匹配
"""

import json
import time
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

# 市场环境 → 加成参数映射
MARKET_BOOST_CONFIG = {
    "strong_bull": {
        "industry_max": 0.28,
        "concept_max": 0.35,      # 恢复激进，强牛追概念有效
        "policy_max": 0.22,
        "non_hot_multiplier": 1.0,
        "overall_cap": 2.3,       # 接近原值2.5
    },
    "weak_bull": {
        "industry_max": 0.20,
        "concept_max": 0.15,      # ↓ 从0.30降半，弱牛追概念风险高
        "policy_max": 0.10,       # ↓ 从0.15降低
        "non_hot_multiplier": 0.85,
        "overall_cap": 1.8,       # ↓ 从2.0降低
    },
    "oscillating": {
        "industry_max": 0.08,
        "concept_max": 0.05,      # ↓ 从0.08降低
        "policy_max": 0.03,       # ↓ 从0.05降低
        "non_hot_multiplier": 0.60,
        "overall_cap": 1.2,       # ↓ 从1.3降低
    },
    "bear": {
        "industry_max": 0.05,
        "concept_max": 0.05,
        "policy_max": 0.0,
        "non_hot_multiplier": 0.50,
        "overall_cap": 1.15,
    },
}


class SectorFilter:
    """热点板块过滤器 v2"""

    def __init__(
        self,
        tushare_fetcher: Optional[TushareFetcher] = None,
        cache_dir: str = "data/cache/sector",
        hot_industry_top_n: int = 10,
        hot_concept_top_n: int = 15,  # 适度扩大覆盖（原20→现15）
        hot_moneyflow_top_n: int = 15,
        enable_policy: bool = True,
        lookback_days: int = 2,
        rotation_threshold: float = 0.30,
        rotation_penalty: float = 0.70,
        strong_bull_only: bool = False,  # 仅strong_bull启用（默认False，全市场禁用）
    ):
        self.fetcher = tushare_fetcher or TushareFetcher()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.hot_industry_top_n = hot_industry_top_n
        self.hot_concept_top_n = hot_concept_top_n
        self.hot_moneyflow_top_n = hot_moneyflow_top_n
        self.enable_policy = enable_policy
        self.lookback_days = lookback_days
        self.rotation_threshold = rotation_threshold
        self.rotation_penalty = rotation_penalty
        self.strong_bull_only = strong_bull_only

        # 内存缓存
        self._hot_sectors_cache: Dict[str, dict] = {}
        self._stock_industry_cache: Dict[str, str] = {}
        self._stock_concepts_cache: Dict[str, List[str]] = {}
        self._top_list_cache: Dict[str, dict] = {}

        # 加载行业/概念预缓存（消除API调用瓶颈）
        self._load_precache()

    # ------------------------------------------------------------------
    # 预缓存（行业/概念映射）
    # ------------------------------------------------------------------
    def _load_precache(self):
        """加载行业/概念预缓存到内存"""
        precache_dir = Path("data/cache/sector")
        # 行业预缓存
        ind_path = precache_dir / "stock_industry_cache.json"
        if ind_path.exists():
            try:
                with open(ind_path, "r", encoding="utf-8") as f:
                    self._stock_industry_cache = json.load(f)
                log.info(f"行业预缓存加载: {len(self._stock_industry_cache)} 只股票")
            except Exception as e:
                log.debug(f"行业预缓存加载失败: {e}")
        # 概念预缓存
        con_path = precache_dir / "stock_concepts_cache.json"
        if con_path.exists():
            try:
                with open(con_path, "r", encoding="utf-8") as f:
                    self._stock_concepts_cache = json.load(f)
                log.info(f"概念预缓存加载: {len(self._stock_concepts_cache)} 只股票")
            except Exception as e:
                log.debug(f"概念预缓存加载失败: {e}")

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
        """获取当日热门板块信息"""
        if not force_refresh and trade_date in self._hot_sectors_cache:
            return self._hot_sectors_cache[trade_date]

        cached = self._load_cache(trade_date, "hot_sectors")
        if cached and not force_refresh:
            self._hot_sectors_cache[trade_date] = cached
            return cached

        result = {
            "industries": [],
            "concepts": [],
            "moneyflow": [],
            "policy_themes": [],
            "all_hot_names": [],
            "top_list": {},  # {ts_code: net_amount}
            "top_inst": {},  # {ts_code: inst_net_buy}
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

        # 4. 龙虎榜数据
        try:
            df_top = self.fetcher.get_top_list(trade_date)
            if not df_top.empty and "ts_code" in df_top.columns:
                for _, row in df_top.iterrows():
                    code = row["ts_code"]
                    net = row.get("net_amount", 0)
                    try:
                        net_val = float(net) if net else 0
                    except Exception:
                        net_val = 0
                    result["top_list"][code] = net_val
        except Exception as e:
            log.debug(f"获取龙虎榜失败 {trade_date}: {e}")

        try:
            df_inst = self.fetcher.get_top_inst(trade_date)
            if not df_inst.empty and "ts_code" in df_inst.columns:
                inst_map = {}
                for _, row in df_inst.iterrows():
                    code = row["ts_code"]
                    net_buy = row.get("net_buy", 0)
                    try:
                        nb = float(net_buy) if net_buy else 0
                    except Exception:
                        nb = 0
                    inst_map[code] = inst_map.get(code, 0) + nb
                result["top_inst"] = inst_map
        except Exception as e:
            log.debug(f"获取龙虎榜机构明细失败 {trade_date}: {e}")

        # 5. 政策主题匹配
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
            result["all_hot_names"] = all_hot_names

        self._hot_sectors_cache[trade_date] = result
        self._save_cache(trade_date, "hot_sectors", result)
        log.info(
            f"  热点板块 [{trade_date}]: 行业{len(result['industries'])}个, "
            f"概念{len(result['concepts'])}个, 政策主题{len(result['policy_themes'])}个"
        )
        return result

    # ------------------------------------------------------------------
    # 热点持续性 & 板块轮动检测
    # ------------------------------------------------------------------
    def _get_sector_history(self, trade_date: str) -> List[dict]:
        """获取过去N天的热点板块历史"""
        history = []
        # trade_date格式为YYYYMMDD，转为int便于减天数
        try:
            from datetime import datetime, timedelta
            base = datetime.strptime(trade_date, "%Y%m%d")
            for i in range(1, self.lookback_days + 1):
                prev_date = (base - timedelta(days=i)).strftime("%Y%m%d")
                # 检查缓存（可能非交易日，尝试向前找）
                for j in range(5):
                    check_date = (base - timedelta(days=i + j)).strftime("%Y%m%d")
                    cached = self._load_cache(check_date, "hot_sectors")
                    if cached:
                        history.append(cached)
                        break
        except Exception:
            pass
        return history

    def _is_persistent_hot(
        self, sector_name: str, history: List[dict], sector_type: str = "concept"
    ) -> bool:
        """检查板块是否持续上榜（过去N天至少出现1次）"""
        if not history:
            return False
        key = "concepts" if sector_type == "concept" else "industries"
        count = 0
        for h in history:
            for name in h.get(key, []):
                if sector_name in name or name in sector_name:
                    count += 1
                    break
        return count >= 1  # 过去N天至少出现1次即视为持续

    def _detect_rotation(self, today: dict, history: List[dict]) -> float:
        """检测板块轮动速度，返回惩罚系数（1.0=无惩罚）"""
        if not history:
            return 1.0

        yesterday = history[0]
        today_set = set(today.get("concepts", []) + today.get("industries", []))
        yest_set = set(yesterday.get("concepts", []) + yesterday.get("industries", []))

        if not today_set or not yest_set:
            return 1.0

        intersection = len(today_set & yest_set)
        union = len(today_set | yest_set)
        overlap_ratio = intersection / union if union > 0 else 1.0

        if overlap_ratio < self.rotation_threshold:
            log.info(f"  板块轮动剧烈(重叠{overlap_ratio:.0%})，整体加成×{self.rotation_penalty:.0%}")
            return self.rotation_penalty
        return 1.0

    # ------------------------------------------------------------------
    # 股票板块信息获取
    # ------------------------------------------------------------------
    def get_stock_industry(self, ts_code: str) -> str:
        """获取股票所属申万行业（优先预缓存，否则API兜底）"""
        if ts_code in self._stock_industry_cache:
            return self._stock_industry_cache[ts_code]
        # 预缓存未命中时，批量获取并更新（回测场景不应发生）
        log.warning(f"行业预缓存未命中: {ts_code}，尝试API获取")
        try:
            mapping = self.fetcher.get_stock_industry_map([ts_code])
            industry = mapping.get(ts_code, "")
        except Exception:
            industry = ""
        self._stock_industry_cache[ts_code] = industry
        return industry

    def get_stock_concepts(self, ts_code: str) -> List[str]:
        """获取股票所属概念列表（优先预缓存，否则API兜底）"""
        if ts_code in self._stock_concepts_cache:
            return self._stock_concepts_cache[ts_code]
        log.warning(f"概念预缓存未命中: {ts_code}，尝试API获取")
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
    # 热度加成计算 v2
    # ------------------------------------------------------------------
    def get_sector_boost(
        self,
        ts_code: str,
        trade_date: str,
        market_state: str = "weak_bull",
        hot_sectors: Optional[dict] = None,
    ) -> float:
        """
        计算股票的板块热度加成系数 v2

        核心逻辑：
        1. 根据市场环境设定加成上限（bear市几乎无加成）
        2. 热点持续性：首日上榜只给50%加成，持续上榜给100%
        3. 板块轮动：热点切换过快时整体降权
        4. 反向过滤：非热点股票乘以市场对应的折扣系数
        """
        cfg = MARKET_BOOST_CONFIG.get(market_state, MARKET_BOOST_CONFIG["weak_bull"])

        if hot_sectors is None:
            hot_sectors = self.get_hot_sectors(trade_date)

        # 获取历史热点（用于持续性和轮动检测）
        history = self._get_sector_history(trade_date)

        # 轮动检测惩罚
        rotation_penalty = self._detect_rotation(hot_sectors, history)

        boost = 1.0
        is_hot = False

        # 1. 行业热点匹配
        if hot_sectors["industries"]:
            stock_ind = self.get_stock_industry(ts_code)
            if stock_ind:
                for rank, hot_ind in enumerate(hot_sectors["industries"]):
                    if stock_ind == hot_ind or hot_ind in stock_ind or stock_ind in hot_ind:
                        is_hot = True
                        weight = 1.0 - rank / len(hot_sectors["industries"])
                        # 持续性检查
                        persistent = self._is_persistent_hot(hot_ind, history, "industry")
                        persist_factor = 1.0 if persistent else 0.5
                        boost *= (1.0 + cfg["industry_max"] * weight * persist_factor * rotation_penalty)
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
                    is_hot = True
                    weight = 1.0 - rank / len(hot_sectors["concepts"])
                    persistent = self._is_persistent_hot(hot_con, history, "concept")
                    persist_factor = 1.0 if persistent else 0.5
                    boost *= (1.0 + cfg["concept_max"] * weight * persist_factor * rotation_penalty)
                    break

        # 3. 政策主题匹配（额外加成）
        if cfg["policy_max"] > 0 and self.enable_policy and hot_sectors.get("policy_themes"):
            stock_concepts = self.get_stock_concepts(ts_code)
            stock_ind = self.get_stock_industry(ts_code)
            all_labels = stock_concepts + [stock_ind]

            for theme in hot_sectors["policy_themes"]:
                keywords = POLICY_THEME_MAP.get(theme, [])
                for label in all_labels:
                    for kw in keywords:
                        if kw in label or label in kw:
                            is_hot = True
                            boost *= (1.0 + cfg["policy_max"] * rotation_penalty)
                            break

        # 4. 龙虎榜加成（仅保留机构净买入>1亿，弱市过滤由板块轮动惩罚覆盖）
        top_inst = hot_sectors.get("top_inst", {})
        if ts_code in top_inst:
            inst_net = top_inst[ts_code]
            if inst_net > 100_000_000:  # 机构净买入>1亿
                boost *= 1.12
                log.info(f"  龙虎榜加成 {ts_code}: 机构净买入{inst_net/1e8:.2f}亿 +12%")

        # 5. 反向过滤：非热点股票打折
        if not is_hot:
            boost = cfg["non_hot_multiplier"]

        return min(boost, cfg["overall_cap"])

    def get_stock_sector_labels(self, ts_code: str) -> dict:
        """获取股票的所有板块标签（用于同板块去重）"""
        return {
            "industry": self.get_stock_industry(ts_code),
            "concepts": self.get_stock_concepts(ts_code),
        }

    def filter_hot_stocks(
        self,
        df_preds: pd.DataFrame,
        trade_date: str,
        market_state: str = "weak_bull",
        hot_sectors: Optional[dict] = None,
    ) -> pd.DataFrame:
        """对预测结果叠加板块热度筛选/排序"""
        # strong_bull_only 模式下，非强牛市场直接返回原预测（不做板块干预）
        if self.strong_bull_only and market_state != "strong_bull":
            return df_preds.copy()

        if hot_sectors is None:
            hot_sectors = self.get_hot_sectors(trade_date)

        df = df_preds.copy()
        df["sector_boost"] = df["ts_code"].apply(
            lambda x: self.get_sector_boost(x, trade_date, market_state, hot_sectors)
        )

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
