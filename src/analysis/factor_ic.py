#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
因子有效性分析模块（IC 计算）

计算各因子维度（价值/动量/质量/波动率/左侧/资金流）的 Rank IC、IR、
IC 时间序列、因子相关性矩阵、分组 IC 等。

数据源：Tushare stk_factor_pro（已包含 daily + daily_basic + 技术指标）
"""

import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tushare as ts
from dotenv import load_dotenv

from src.utils.logger import log
from src.data.tushare_data_provider import STK_FACTOR_RENAME

load_dotenv()

# ─── 因子维度定义 ───
# 每个维度映射到 stk_factor_pro 中的具体字段
# direction: 1 表示因子值越大越好，-1 表示因子值越小越好
FACTOR_DIMENSIONS = {
    "value": {
        "name": "价值因子",
        "field": "pb",
        "direction": -1,
        "desc": "市净率倒数（BP），低PB代表价值属性",
        "color": "#58a6ff",
    },
    "momentum": {
        "name": "动量因子",
        "field": "roc",
        "direction": 1,
        "desc": "ROC变动率，趋势延续性",
        "color": "#3fb950",
    },
    "quality": {
        "name": "质量因子",
        "field": "dv_ratio",
        "direction": 1,
        "desc": "股息率，高股息代表高质量",
        "color": "#a371f7",
    },
    "volatility": {
        "name": "波动率因子",
        "field": "atr",
        "direction": -1,
        "desc": "负向ATR，低波动防御属性",
        "color": "#d29922",
    },
    "contrarian": {
        "name": "左侧因子",
        "field": "rsi_6",
        "direction": -1,
        "desc": "负向RSI，超卖反转信号",
        "color": "#f85149",
    },
    "moneyflow": {
        "name": "资金流因子",
        "field": "mfi",
        "direction": 1,
        "desc": "资金流量指标MFI",
        "color": "#58a6ff",
    },
}

# 用于相关性矩阵的扩展因子列表（在同一数据源内）
CORR_FACTORS = {
    "pb": "PB",
    "pe_ttm": "PE_TTM",
    "dv_ratio": "股息率",
    "turnover_rate": "换手率",
    "roc": "ROC",
    "macd": "MACD",
    "rsi_6": "RSI6",
    "atr": "ATR",
    "cci": "CCI",
    "mfi": "MFI",
    "wr": "WR",
}


def _get_pro_api():
    """获取 Tushare pro_api（复用环境变量中的 token）"""
    return ts.pro_api()


def _trade_date_str(dt: Optional[datetime] = None) -> str:
    if dt is None:
        dt = datetime.now()
    return dt.strftime("%Y%m%d")


def _prev_trade_date(pro, trade_date: str) -> str:
    """返回前一个交易日"""
    try:
        cal = pro.trade_cal(
            exchange="SSE", start_date=trade_date, end_date=trade_date,
            fields="exchange,cal_date,is_open,pretrade_date"
        )
        if cal is not None and not cal.empty:
            pre = cal.iloc[0].get("pretrade_date")
            if pre and str(pre) != "nan":
                return str(pre)
    except Exception:
        pass
    y = datetime.strptime(trade_date, "%Y%m%d") - timedelta(days=1)
    return y.strftime("%Y%m%d")


def _get_trade_dates(pro, end_date: str, n: int) -> List[str]:
    """获取 end_date 往前 n 个交易日（包含 end_date）"""
    dates = [end_date]
    current = end_date
    for _ in range(n - 1):
        prev = _prev_trade_date(pro, current)
        if prev == current:
            break
        dates.append(prev)
        current = prev
    return list(reversed(dates))


def _fetch_stk_factor_pro(trade_date: str) -> pd.DataFrame:
    """获取单日 stk_factor_pro 全市场数据"""
    pro = _get_pro_api()
    try:
        df = pro.stk_factor_pro(trade_date=trade_date)
        if df is None or df.empty:
            return pd.DataFrame()
        # 统一列名（将 *_qfq 映射为简化名）
        rename_map = {k: v for k, v in STK_FACTOR_RENAME.items() if k in df.columns}
        df = df.rename(columns=rename_map)
        # 需要的字段
        needed = ["ts_code", "trade_date", "close", "pct_chg", "total_mv"]
        for dim in FACTOR_DIMENSIONS.values():
            needed.append(dim["field"])
        for f in CORR_FACTORS.keys():
            if f not in needed:
                needed.append(f)
        cols = [c for c in needed if c in df.columns]
        df = df[cols].copy()
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        # 过滤 ST / *ST / 退市 / 北交所（可选）
        df = df[~df["ts_code"].str.contains(r"\*ST|ST\b|退", regex=True, na=False)]
        df = df[~df["ts_code"].str.endswith(".BJ")]
        return df
    except Exception as e:
        log.warning(f"stk_factor_pro({trade_date}) 获取失败: {e}")
        return pd.DataFrame()


def _calc_rank_ic(factor_series: pd.Series, return_series: pd.Series) -> float:
    """计算 Rank IC（Spearman 秩相关系数）"""
    df = pd.DataFrame({"factor": factor_series, "ret": return_series}).dropna()
    if len(df) < 10:
        return np.nan
    return float(df["factor"].corr(df["ret"], method="spearman"))


def _fetch_stk_factor_pro_batch(dates: List[str], max_workers: int = 5) -> Dict[str, pd.DataFrame]:
    """并发获取多日的 stk_factor_pro 数据"""
    factor_frames = {}
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _fetch_one(d: str):
        return d, _fetch_stk_factor_pro(d)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_date = {executor.submit(_fetch_one, d): d for d in dates}
        for future in as_completed(future_to_date):
            d, df = future.result()
            if not df.empty:
                factor_frames[d] = df
    return factor_frames


def calculate_ic_series(
    end_date: Optional[str] = None,
    lookback_days: int = 20,
    horizon_days: int = 1,
    preloaded_frames: Optional[Dict[str, pd.DataFrame]] = None,
) -> Dict:
    """
    计算近 lookback_days 个交易日各因子维度的每日 Rank IC。

    Args:
        end_date: 结束日期（YYYYMMDD），默认今天
        lookback_days: 回看多长交易日
        horizon_days: 预测未来几天收益（默认次日）
        preloaded_frames: 预加载的因子数据（避免重复获取）

    Returns:
        {
            "dates": ["20260501", ...],
            "ic_series": {
                "value": [0.01, 0.02, ...],
                "momentum": [...],
                ...
            },
            "ic_mean": {"value": 0.015, ...},
            "ic_std": {"value": 0.08, ...},
            "ir": {"value": 0.19, ...},
        }
    """
    pro = _get_pro_api()
    end_date = end_date or _trade_date_str()

    # 需要 lookback_days + horizon_days 个交易日数据
    dates = _get_trade_dates(pro, end_date, lookback_days + horizon_days)
    if len(dates) < lookback_days + horizon_days:
        log.warning(f"交易日不足: {len(dates)} < {lookback_days + horizon_days}")

    # 获取因子数据（使用预加载数据或重新获取）
    if preloaded_frames is not None:
        factor_frames = {d: preloaded_frames[d] for d in dates if d in preloaded_frames}
    else:
        factor_frames = _fetch_stk_factor_pro_batch(dates)

    if len(factor_frames) < 2:
        log.warning("因子数据不足，无法计算IC")
        return _empty_ic_result(lookback_days)

    # 计算每日 IC
    result_dates = []
    ic_series = {k: [] for k in FACTOR_DIMENSIONS}

    # 可用的计算日（需要有当日因子 + 未来 horizon 日收益率）
    valid_dates = sorted(factor_frames.keys())
    for i, d in enumerate(valid_dates):
        # 找未来 horizon_days 的日期
        future_idx = i + horizon_days
        if future_idx >= len(valid_dates):
            continue
        future_d = valid_dates[future_idx]
        df_today = factor_frames[d]
        df_future = factor_frames[future_d]

        # 合并获取未来收益率
        merged = df_today[["ts_code", "close"]].merge(
            df_future[["ts_code", "close"]], on="ts_code", suffixes=("", "_f")
        )
        if merged.empty:
            continue
        merged["future_ret"] = (merged["close_f"] - merged["close"]) / merged["close"]

        # 合并因子值
        for key, dim in FACTOR_DIMENSIONS.items():
            field = dim["field"]
            if field not in df_today.columns:
                ic_series[key].append(np.nan)
                continue
            fac = df_today[["ts_code", field]].merge(merged[["ts_code", "future_ret"]], on="ts_code")
            fac = fac.dropna()
            if len(fac) < 10:
                ic_series[key].append(np.nan)
                continue
            # 根据 direction 调整因子方向
            factor_vals = fac[field] * dim["direction"]
            ic = _calc_rank_ic(factor_vals, fac["future_ret"])
            ic_series[key].append(ic)

        result_dates.append(d)

    # 汇总统计
    ic_mean = {}
    ic_std = {}
    ir = {}
    for key in FACTOR_DIMENSIONS:
        arr = pd.Series(ic_series[key]).dropna()
        if len(arr) == 0:
            ic_mean[key] = 0.0
            ic_std[key] = 0.0
            ir[key] = 0.0
        else:
            m = float(arr.mean())
            s = float(arr.std())
            ic_mean[key] = round(m, 4)
            ic_std[key] = round(s, 4)
            ir[key] = round(m / s, 4) if s > 1e-6 else 0.0

    return {
        "dates": result_dates,
        "ic_series": ic_series,
        "ic_mean": ic_mean,
        "ic_std": ic_std,
        "ir": ir,
        "lookback_days": lookback_days,
        "horizon_days": horizon_days,
    }


def calculate_factor_correlation(
    end_date: Optional[str] = None,
    lookback_days: int = 5,
    preloaded_frames: Optional[Dict[str, pd.DataFrame]] = None,
) -> Dict:
    """
    计算扩展因子列表的相关性矩阵。
    使用最近 lookback_days 的因子截面数据取平均后计算相关性。
    """
    pro = _get_pro_api()
    end_date = end_date or _trade_date_str()
    dates = _get_trade_dates(pro, end_date, lookback_days)

    if preloaded_frames is not None:
        all_frames = [preloaded_frames[d] for d in dates if d in preloaded_frames]
    else:
        all_frames = []
        for d in dates:
            df = _fetch_stk_factor_pro(d)
            if not df.empty:
                all_frames.append(df)

    if not all_frames:
        return {"labels": list(CORR_FACTORS.values()), "matrix": [[1.0]]}

    # 合并多日的因子数据（取中位数代表该股票近期因子水平）
    combined = pd.concat(all_frames, ignore_index=True)
    factor_cols = [f for f in CORR_FACTORS.keys() if f in combined.columns]
    if not factor_cols:
        return {"labels": list(CORR_FACTORS.values()), "matrix": [[1.0]]}

    median_df = combined.groupby("ts_code")[factor_cols].median().reset_index()
    corr = median_df[factor_cols].corr(method="spearman")

    labels = [CORR_FACTORS.get(c, c) for c in factor_cols]
    matrix = []
    for c in factor_cols:
        row = []
        for c2 in factor_cols:
            v = corr.loc[c, c2]
            row.append(round(v, 3) if not math.isnan(v) else 0.0)
        matrix.append(row)

    return {"labels": labels, "matrix": matrix}


def calculate_group_ic(
    end_date: Optional[str] = None,
    lookback_days: int = 10,
    horizon_days: int = 1,
    preloaded_frames: Optional[Dict[str, pd.DataFrame]] = None,
) -> Dict:
    """
    按市值分组计算各因子维度的 IC。
    分组：大盘（top 30% total_mv）、中盘（30%~70%）、小盘（bottom 30%）
    """
    pro = _get_pro_api()
    end_date = end_date or _trade_date_str()
    dates = _get_trade_dates(pro, end_date, lookback_days + horizon_days)

    if preloaded_frames is not None:
        factor_frames = {d: preloaded_frames[d] for d in dates if d in preloaded_frames and "total_mv" in preloaded_frames[d].columns}
    else:
        factor_frames = {}
        for d in dates:
            df = _fetch_stk_factor_pro(d)
            if not df.empty and "total_mv" in df.columns:
                factor_frames[d] = df

    valid_dates = sorted(factor_frames.keys())
    group_ic = {"large_cap": {k: [] for k in FACTOR_DIMENSIONS},
                "mid_cap": {k: [] for k in FACTOR_DIMENSIONS},
                "small_cap": {k: [] for k in FACTOR_DIMENSIONS}}

    for i, d in enumerate(valid_dates):
        future_idx = i + horizon_days
        if future_idx >= len(valid_dates):
            continue
        future_d = valid_dates[future_idx]
        df_today = factor_frames[d]
        df_future = factor_frames[future_d]

        merged = df_today[["ts_code", "close", "total_mv"]].merge(
            df_future[["ts_code", "close"]], on="ts_code", suffixes=("", "_f")
        )
        if merged.empty:
            continue
        merged["future_ret"] = (merged["close_f"] - merged["close"]) / merged["close"]

        # 按市值分三组
        merged = merged.dropna(subset=["total_mv"])
        if len(merged) < 30:
            continue
        q70 = merged["total_mv"].quantile(0.7)
        q30 = merged["total_mv"].quantile(0.3)

        groups = {
            "large_cap": merged[merged["total_mv"] >= q70],
            "mid_cap": merged[(merged["total_mv"] >= q30) & (merged["total_mv"] < q70)],
            "small_cap": merged[merged["total_mv"] < q30],
        }

        for gname, gdf in groups.items():
            if len(gdf) < 10:
                continue
            for key, dim in FACTOR_DIMENSIONS.items():
                field = dim["field"]
                if field not in df_today.columns:
                    continue
                fac = df_today[["ts_code", field]].merge(gdf[["ts_code", "future_ret"]], on="ts_code")
                fac = fac.dropna()
                if len(fac) < 5:
                    continue
                factor_vals = fac[field] * dim["direction"]
                ic = _calc_rank_ic(factor_vals, fac["future_ret"])
                if not math.isnan(ic):
                    group_ic[gname][key].append(ic)

    # 汇总
    result = {}
    for gname in group_ic:
        result[gname] = {}
        for key in FACTOR_DIMENSIONS:
            arr = pd.Series(group_ic[gname][key]).dropna()
            result[gname][key] = round(float(arr.mean()), 4) if len(arr) > 0 else 0.0

    return result


def _empty_ic_result(lookback_days: int) -> Dict:
    return {
        "dates": [],
        "ic_series": {k: [0.0] * lookback_days for k in FACTOR_DIMENSIONS},
        "ic_mean": {k: 0.0 for k in FACTOR_DIMENSIONS},
        "ic_std": {k: 0.0 for k in FACTOR_DIMENSIONS},
        "ir": {k: 0.0 for k in FACTOR_DIMENSIONS},
        "lookback_days": lookback_days,
        "horizon_days": 1,
    }


def _get_factor_advice(key: str, ic_long: float, ir: float) -> Dict:
    """
    根据因子IC值生具体结论与A股做多操作建议。
    原则：A股只能做多，看空即空仓/避开。
    """
    dim = FACTOR_DIMENSIONS[key]
    direction = dim["direction"]

    # 根据 direction 和 IC 判断因子方向含义
    # direction=+1: 因子值越大越好；direction=-1: 因子值越小越好
    # IC>0 表示因子值与未来收益正相关（因子有效）
    # IC<0 表示因子值与未来收益负相关（因子反向有效）

    # 判断因子有效性强度
    if abs(ic_long) >= 0.05:
        strength = "strong"
    elif abs(ic_long) >= 0.03:
        strength = "moderate"
    elif abs(ic_long) >= 0.01:
        strength = "weak"
    else:
        strength = "none"

    if key == "value":
        # field=pb, direction=-1 → 因子值=-pb（pb越低越好）
        if ic_long >= 0.03:
            conclusion = "低市净率价值股跑赢，价值风格主导"
            action = "做多低PB价值股，关注银行、地产等低估值板块"
        elif ic_long <= -0.03:
            conclusion = "高市净率成长股跑赢，成长风格主导"
            action = "避开低PB价值股，关注科技、新能源等高估值成长股；看空价值即空仓价值"
        else:
            conclusion = "价值/成长风格不明显，估值区分度低"
            action = "不刻意区分价值/成长，精选个股为主"

    elif key == "momentum":
        # field=roc, direction=+1 → 因子值=roc（roc越高越好）
        if ic_long >= 0.03:
            conclusion = "强势股持续跑赢，趋势延续性强"
            action = "做多近期涨幅大的强势股，顺势追涨"
        elif ic_long <= -0.03:
            conclusion = "强势股回调，趋势反转或震荡"
            action = "不追涨强势股，关注超跌反弹机会或空仓观望"
        else:
            conclusion = "动量效应不明显，趋势不延续"
            action = "不追逐热点，避免追涨杀跌"

    elif key == "quality":
        # field=dv_ratio, direction=+1 → 因子值=dv_ratio（股息率越高越好）
        if ic_long >= 0.03:
            conclusion = "高股息蓝筹股跑赢，质量风格有效"
            action = "做多高股息、低负债、现金流稳定的蓝筹股"
        elif ic_long <= -0.03:
            conclusion = "高股息蓝筹跑输，成长股主导市场"
            action = "避开高股息蓝筹，转向成长型中小盘；看空蓝筹即空仓"
        else:
            conclusion = "质量因子区分度低，股息率无明显预测力"
            action = "不特别看重股息率，以成长性为优先"

    elif key == "volatility":
        # field=atr, direction=-1 → 因子值=-atr（atr越低越好，即低波动）
        if ic_long >= 0.03:
            conclusion = "低波动股票跑赢，防御属性凸显"
            action = "做多低波动稳健股，控制回撤优先"
        elif ic_long <= -0.03:
            conclusion = "高波动强势股跑赢，市场风险偏好高"
            action = "做多高波动强势股，积极参与题材炒作；不看空，但低波动股应空仓"
        else:
            conclusion = "波动率因子无明显区分度"
            action = "不特别看重波动率，以趋势和基本面为优先"

    elif key == "contrarian":
        # field=rsi_6, direction=-1 → 因子值=-rsi_6（rsi越低越好，超卖）
        if ic_long >= 0.03:
            conclusion = "超卖股反弹明显，左侧布局有效"
            action = "做多RSI超卖、跌幅大的超跌股，抄底反弹"
        elif ic_long <= -0.03:
            conclusion = "强势股强者恒强，逆势抄底胜率低"
            action = "不抄底，做多趋势向上的强势股；看空逆势即空仓观望"
        else:
            conclusion = "反转信号不明显，市场方向不明"
            action = "不逆势操作，等待明确信号"

    elif key == "moneyflow":
        # field=mfi, direction=+1 → 因子值=mfi（mfi越高越好）
        if ic_long >= 0.03:
            conclusion = "资金流入股持续跑赢，跟随资金有效"
            action = "做多资金持续流入、成交量放大的个股"
        elif ic_long <= -0.03:
            conclusion = "资金热点可能见顶，高MFI股回调"
            action = "不追资金热点，关注被错杀的冷门股或空仓观望"
        else:
            conclusion = "资金流向无明显预测力"
            action = "不跟随资金流向，以基本面为优先"

    else:
        conclusion = "因子状态不明"
        action = "观望"

    # 根据IR补充稳定性判断
    if abs(ir) >= 0.3:
        stability = "因子信号稳定，可信度高"
    elif abs(ir) >= 0.15:
        stability = "因子信号一般，注意波动"
    else:
        stability = "因子信号不稳定，谨慎参考"

    return {
        "conclusion": conclusion,
        "action": action,
        "stability": stability,
        "strength": strength,
    }


def _get_overall_strategy(factors: List[Dict], group_ic: Dict) -> Dict:
    """
    基于全部因子IC与分组IC，生成A股综合策略建议。
    原则：只能做多，看空即空仓/避开。
    """
    # 提取各因子IC
    ic_map = {f["key"]: f["ic_long"] for f in factors}
    status_map = {f["key"]: f["status"] for f in factors}

    value_ic = ic_map.get("value", 0)
    momentum_ic = ic_map.get("momentum", 0)
    quality_ic = ic_map.get("quality", 0)
    vol_ic = ic_map.get("volatility", 0)
    contra_ic = ic_map.get("contrarian", 0)
    mf_ic = ic_map.get("moneyflow", 0)

    # ── 风格判断 ──
    style_parts = []
    if value_ic < -0.02 and quality_ic < -0.02:
        style_parts.append("成长风格主导")
    elif value_ic > 0.02 and quality_ic > 0.02:
        style_parts.append("价值风格主导")
    else:
        style_parts.append("风格不明朗")

    if abs(vol_ic) >= 0.03 and vol_ic < 0:
        style_parts.append("高波动股活跃")
    elif abs(vol_ic) >= 0.03 and vol_ic > 0:
        style_parts.append("低波动防御占优")

    # 分组IC判断大小盘
    small_better = False
    if group_ic and "small_cap" in group_ic and "large_cap" in group_ic:
        # 取各因子在大小盘的IC均值
        small_vals = group_ic["small_cap"].get("values", [])
        large_vals = group_ic["large_cap"].get("values", [])
        if small_vals and large_vals:
            small_avg = sum(abs(v) for v in small_vals) / len(small_vals)
            large_avg = sum(abs(v) for v in large_vals) / len(large_vals)
            if small_avg > large_avg * 1.3:
                style_parts.append("小盘股强势")
                small_better = True
            elif large_avg > small_avg * 1.3:
                style_parts.append("大盘股稳健")

    style = "，".join(style_parts)

    # ── 总体策略 ──
    # 统计有效因子数量
    valid_forward = sum(1 for s in status_map.values() if s in ("强势", "有效"))
    valid_reverse = sum(1 for s in status_map.values() if s in ("反向强势", "反向有效"))

    if valid_forward >= 2 and valid_reverse <= 1:
        strategy = "积极做多"
        tone = "进攻"
    elif valid_reverse >= 2 and valid_forward <= 1:
        strategy = "灵活做多"
        tone = "结构性机会"
    elif valid_forward == 0 and valid_reverse == 0:
        strategy = "空仓观望"
        tone = "防守"
    else:
        strategy = "谨慎做多"
        tone = "控制仓位"

    # ── 选股方向 ──
    # A股只能做多：正向有效 → 做多因子值高的一端；反向有效 → 做多因子值低的一端
    long_directions = []
    short_directions = []

    for f in factors:
        key = f["key"]
        ic = f["ic_long"]
        action_text = f["action"]
        if ic >= 0.03 or ic <= -0.03:
            # 将操作建议提炼为方向
            long_directions.append(action_text)

    if not long_directions:
        long_directions.append("暂无明确做多方向，等待有效因子出现")
    if not short_directions:
        short_directions.append("暂无明确回避方向")

    # ── 风险提示 ──
    risk_parts = []
    if abs(momentum_ic) < 0.01:
        risk_parts.append("趋势不延续，忌追涨杀跌")
    if abs(contra_ic) < 0.01:
        risk_parts.append("反转信号弱，忌抄底")
    if mf_ic < -0.02:
        risk_parts.append("资金热点可能见顶，忌追热点")

    if not risk_parts:
        risk_parts.append("因子信号整体较弱，控制仓位")

    return {
        "style": style,
        "strategy": strategy,
        "tone": tone,
        "long_directions": long_directions,
        "short_directions": short_directions,
        "risks": risk_parts,
        "summary": f"当前A股处于{style}阶段。{strategy}，{tone}为主。",
    }


def build_radar_response(
    end_date: Optional[str] = None,
    lookback_short: int = 5,
    lookback_long: int = 20,
) -> Dict:
    """
    构建因子雷达完整响应数据（供 API 使用）。

    Returns:
        {
            "radar": {
                "indicators": [{"name": "价值", "max": 0.1}, ...],
                "data": [
                    {"name": "近1周IC", "value": [...]},
                    {"name": "近1月IC", "value": [...]},
                ],
            },
            "factors": [
                {
                    "key": "value",
                    "name": "价值因子",
                    "ic_short": 0.02,
                    "ic_long": 0.035,
                    "ir": 0.22,
                    "status": "有效",
                    "desc": "...",
                    "color": "#58a6ff",
                    "ic_series": [{"date": "...", "ic": 0.01}, ...],
                },
                ...
            ],
            "correlation": {"labels": [...], "matrix": [[...]]},
            "group_ic": {"large_cap": {...}, "mid_cap": {...}, "small_cap": {...}},
        }
    """
    end_date = end_date or _trade_date_str()
    pro = _get_pro_api()

    # 统一获取所有需要的交易日数据（避免各子函数重复获取）
    # 长期IC需要最多数据: lookback_long + 1 个交易日
    max_lookback = max(lookback_short, lookback_long, 5, max(lookback_short, 10)) + 1
    all_dates = _get_trade_dates(pro, end_date, max_lookback)
    preloaded_frames = _fetch_stk_factor_pro_batch(all_dates)

    if not preloaded_frames:
        log.warning("无法获取任何因子数据，返回空结果")
        empty_radar = {
            "radar": {
                "indicators": [{"name": n.replace("因子", ""), "max": 0.1} for n in [FACTOR_DIMENSIONS[k]["name"] for k in FACTOR_DIMENSIONS]],
                "data": [],
            },
            "factors": [],
            "correlation": {"labels": list(CORR_FACTORS.values()), "matrix": [[1.0]]},
            "group_ic": {},
            "update_time": datetime.now().isoformat(),
        }
        return empty_radar

    # 1. 短期 + 长期 IC
    short_result = calculate_ic_series(end_date, lookback_days=lookback_short, preloaded_frames=preloaded_frames)
    long_result = calculate_ic_series(end_date, lookback_days=lookback_long, preloaded_frames=preloaded_frames)

    # 2. 相关性矩阵
    corr_result = calculate_factor_correlation(end_date, lookback_days=5, preloaded_frames=preloaded_frames)

    # 3. 分组 IC（用中期窗口）
    group_result = calculate_group_ic(end_date, lookback_days=max(lookback_short, 10), preloaded_frames=preloaded_frames)

    # 4. 组装 radar 数据
    dim_names = [FACTOR_DIMENSIONS[k]["name"].replace("因子", "") for k in FACTOR_DIMENSIONS]
    radar_data = []
    if short_result["dates"]:
        radar_data.append({
            "name": f"近{lookback_short}日IC",
            "value": [round(short_result["ic_mean"].get(k, 0.0), 3) for k in FACTOR_DIMENSIONS],
        })
    if long_result["dates"]:
        radar_data.append({
            "name": f"近{lookback_long}日IC",
            "value": [round(long_result["ic_mean"].get(k, 0.0), 3) for k in FACTOR_DIMENSIONS],
        })

    # 5. 组装 factors 详情
    factors = []
    for key in FACTOR_DIMENSIONS:
        dim = FACTOR_DIMENSIONS[key]
        ic_short = short_result["ic_mean"].get(key, 0.0)
        ic_long = long_result["ic_mean"].get(key, 0.0)
        ir_val = long_result["ir"].get(key, 0.0)

        # 状态判定（区分正向/反向有效）
        if ic_long >= 0.05:
            status = "强势"
        elif ic_long <= -0.05:
            status = "反向强势"
        elif ic_long >= 0.03:
            status = "有效"
        elif ic_long <= -0.03:
            status = "反向有效"
        elif abs(ic_long) >= 0.01:
            status = "偏弱"
        else:
            status = "失效"

        # 生成具体结论与操作建议
        advice = _get_factor_advice(key, ic_long, ir_val)

        # IC 时间序列（使用长期数据）
        ic_series = []
        for d, ic in zip(long_result["dates"], long_result["ic_series"][key]):
            if not math.isnan(ic):
                ic_series.append({"date": str(d), "ic": round(ic, 4)})

        factors.append({
            "key": key,
            "name": dim["name"],
            "ic_short": round(ic_short, 4),
            "ic_long": round(ic_long, 4),
            "ir": round(ir_val, 4),
            "status": status,
            "desc": dim["desc"],
            "color": dim["color"],
            "conclusion": advice["conclusion"],
            "action": advice["action"],
            "stability": advice["stability"],
            "strength": advice["strength"],
            "ic_series": ic_series,
        })

    # 6. 组装 group_ic（按因子维度）
    group_ic_output = {}
    for gname in ["large_cap", "mid_cap", "small_cap"]:
        group_ic_output[gname] = {
            "label": {"large_cap": "大盘", "mid_cap": "中盘", "small_cap": "小盘"}[gname],
            "values": [round(group_result.get(gname, {}).get(k, 0.0), 4) for k in FACTOR_DIMENSIONS],
        }

    # 7. 综合策略建议
    strategy = _get_overall_strategy(factors, group_ic_output)

    return {
        "radar": {
            "indicators": [{"name": n, "max": 0.1} for n in dim_names],
            "data": radar_data,
        },
        "factors": factors,
        "correlation": corr_result,
        "group_ic": group_ic_output,
        "strategy": strategy,
        "update_time": datetime.now().isoformat(),
    }
