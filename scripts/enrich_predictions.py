#!/usr/bin/env python3
"""
预测结果 enrich 脚本（v2：支持模型打分 + 规则打分回退）

在预测 CSV 生成后运行，为每只标的添加：
- prob_short: 短期动量评分（优先加载 ShortTermScorer 模型，不存在时回退到规则打分）
- prob_long: 长期质量评分（优先加载 LongTermScorer 模型，不存在时回退到规则打分）
- market_stage: 四阶段（基于 Tushare ohlcv + 自算 ADX/MA）
- left_side_signal: 左侧信号文本
- resonance_score: 三灯共振评分（新增）

用法:
    python scripts/enrich_predictions.py --date 20260430
    # 或在 auto_daily_pipeline.py 中自动调用
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.market_stage import classify_market_stage, get_stage_detail
from src.data.arctic_provider import ArcticDataProvider
from src.models.short_term_scorer import ShortTermScorer
from src.models.long_term_scorer import LongTermScorer
from src.utils.logger import log

PREDICTION_DIR = PROJECT_ROOT / "data" / "prediction" / "v294_stk_factor"


# ============================================================================
# 规则打分（回退方案）
# ============================================================================

def calc_prob_short_rule(row: pd.Series) -> float:
    """
    短期动量评分（规则版）— 当 ShortTermScorer 模型不可用时作为回退。
    基于 Tushare factors 数据。
    """
    score = 0.5
    # RSI: 从 <50 突破至 50-65 加分（趋势确认）
    rsi = row.get("rsi_12", 50)
    if pd.notna(rsi):
        if 50 <= rsi <= 65:
            score += 0.2
        elif rsi > 65:
            score += 0.1  # 强势但接近超买，加分减少
        elif rsi > 50:
            score += 0.05
    # MACD: >0 加分
    macd = row.get("macd", 0)
    if pd.notna(macd) and macd > 0:
        score += 0.15
    # 近5日涨幅: 5%-20% 加分
    ret5 = row.get("return_5d", 0)
    if pd.notna(ret5):
        if 0.05 <= ret5 <= 0.20:
            score += 0.15
        elif ret5 > 0.20:
            score -= 0.1
    # 量比: 1.2-3.0 加分
    vr = row.get("volume_ratio", 1)
    if pd.notna(vr):
        if 1.2 <= vr <= 3.0:
            score += 0.1
    return max(0.0, min(1.0, score))


def calc_prob_long_rule(row: pd.Series) -> float:
    """
    长期质量评分（规则版）— 当 LongTermScorer 模型不可用时作为回退。
    基于 Tushare basic 数据。
    """
    score = 0.5
    # PE: 10-30 倍加分
    pe = row.get("pe", None)
    if pd.notna(pe) and pe > 0:
        if 10 <= pe <= 30:
            score += 0.2
        elif pe < 10:
            score += 0.1
        elif pe > 50:
            score -= 0.15
    # PB: 1-3 倍加分
    pb = row.get("pb", None)
    if pd.notna(pb) and pb > 0:
        if 1 <= pb <= 3:
            score += 0.15
        elif pb > 5:
            score -= 0.1
    # 市值: 50-500亿加分
    mv = row.get("total_mv", None)
    if pd.notna(mv) and mv > 0:
        mv_yi = mv / 10000
        if 50 <= mv_yi <= 500:
            score += 0.1
    return max(0.0, min(1.0, score))


def calc_left_side_signal(row: pd.Series) -> str:
    """左侧信号判断，返回文本标签或空字符串。"""
    signals = []
    rsi = row.get("rsi_12", None)
    if pd.notna(rsi) and rsi < 35:
        signals.append("RSI超卖")
    vr = row.get("volume_ratio", None)
    if pd.notna(vr) and vr < 0.7:
        signals.append("缩量")
    ret20 = row.get("return_20d", None)
    if pd.notna(ret20) and ret20 < -0.15:
        signals.append("深度回调")
    ret5 = row.get("return_5d", None)
    ret1 = row.get("return_1d", None)
    if pd.notna(ret5) and pd.notna(ret1) and ret5 < -0.05 and ret1 > -0.02:
        signals.append("止跌迹象")
    return "、".join(signals) if signals else ""


# ============================================================================
# 模型打分
# ============================================================================

def load_scorers() -> tuple:
    """加载短期/长期模型，不存在时返回 None"""
    short_scorer = ShortTermScorer()
    long_scorer = LongTermScorer()

    short_ok = short_scorer.model_exists()
    long_ok = long_scorer.model_exists()

    if short_ok:
        try:
            short_scorer.load_model()
            log.info("ShortTermScorer 模型已加载")
        except Exception as e:
            log.warning(f"ShortTermScorer 加载失败: {e}")
            short_ok = False

    if long_ok:
        try:
            long_scorer.load_model()
            log.info("LongTermScorer 模型已加载")
        except Exception as e:
            log.warning(f"LongTermScorer 加载失败: {e}")
            long_ok = False

    return (short_scorer if short_ok else None), (long_scorer if long_ok else None)


def calc_prob_short_model(scorer, merged: pd.DataFrame) -> pd.Series:
    """使用 ShortTermScorer 模型预测"""
    if scorer is None or merged.empty:
        return pd.Series(np.nan, index=merged.index)

    # 准备特征（复用 ShortTermScorer 的特征计算逻辑）
    # 由于 merged 已经是合并后的数据，我们直接提取模型需要的特征列
    feature_cols = scorer.feature_names if scorer.feature_names else scorer.FEATURE_COLS

    # 检查缺失特征
    missing = [c for c in feature_cols if c not in merged.columns]
    if missing:
        log.debug(f"短期模型缺失特征: {missing}")
        # 尝试从已有列计算
        for col in missing:
            if col == "return_1d" and "pct_chg" in merged.columns:
                merged[col] = merged["pct_chg"] / 100
            elif col == "return_3d" and "return_5d" in merged.columns:
                merged[col] = merged["return_5d"] * 0.6  # 近似
            elif col == "return_10d" and "return_20d" in merged.columns:
                merged[col] = merged["return_20d"] * 0.5  # 近似
            elif col == "volatility_5d" and "volatility_10d" in merged.columns:
                merged[col] = merged["volatility_10d"] * 0.7
            elif col == "excess_return_5d":
                merged[col] = 0.0
            elif col in ["close_ma20_ratio", "close_ma60_ratio"]:
                merged[col] = 0.0
            else:
                merged[col] = 0.0

    X = merged[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0).astype(float)

    try:
        probs = scorer.predict(X)
        return pd.Series(probs, index=merged.index)
    except Exception as e:
        log.warning(f"短期模型预测失败: {e}")
        return pd.Series(np.nan, index=merged.index)


def calc_prob_long_model(scorer, merged: pd.DataFrame) -> pd.Series:
    """使用 LongTermScorer 模型预测"""
    if scorer is None or merged.empty:
        return pd.Series(np.nan, index=merged.index)

    feature_cols = scorer.feature_names if scorer.feature_names else scorer.FEATURE_COLS

    missing = [c for c in feature_cols if c not in merged.columns]
    if missing:
        log.debug(f"长期模型缺失特征: {missing}")
        for col in missing:
            if col in ["pe_industry_zscore", "pb_industry_zscore"]:
                merged[col] = 0.0
            elif col in ["total_mv_log", "circ_mv_log"] and "total_mv" in merged.columns:
                mv = merged["total_mv"]
                merged[col] = np.log(mv.clip(lower=1))
            elif col == "max_drawdown_60d":
                merged[col] = 0.0
            elif col == "trend_strength_60d":
                merged[col] = 0.0
            elif col in ["return_20d", "return_60d", "return_120d"]:
                merged[col] = merged.get("return_5d", 0) * 5 if "return_5d" in merged.columns else 0
            elif col in ["volatility_60d", "volatility_120d"]:
                merged[col] = 0.0
            elif col in ["close_ma60_ratio", "close_ma120_ratio"]:
                merged[col] = 0.0
            else:
                merged[col] = 0.0

    X = merged[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0).astype(float)

    try:
        probs = scorer.predict(X)
        return pd.Series(probs, index=merged.index)
    except Exception as e:
        log.warning(f"长期模型预测失败: {e}")
        return pd.Series(np.nan, index=merged.index)


# ============================================================================
# 共振评分
# ============================================================================

def load_resonance_config() -> dict:
    """加载共振评分配置"""
    config_path = PROJECT_ROOT / "config" / "3l_scoring.yaml"
    if not config_path.exists():
        return {
            "weights": {"short": 0.30, "mid": 0.40, "long": 0.20},
            "stage_bonus_weight": 0.10,
            "stage_bonus": {
                "all_green_early_rally": 0.15,
                "all_green_mid_rally": 0.10,
                "two_green_early_rally": 0.05,
                "any_red": 0.0,
            },
        }
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f).get("resonance", {})


def calc_resonance(row: pd.Series, config: dict) -> float:
    """计算共振评分"""
    w = config.get("weights", {})
    stage_w = config.get("stage_bonus_weight", 0.1)
    stage_bonus_cfg = config.get("stage_bonus", {})

    prob_short = pd.to_numeric(row.get("prob_short", 0), errors="coerce") or 0
    prob_mid = pd.to_numeric(row.get("prob", row.get("probability", row.get("adjusted_score", 0))), errors="coerce") or 0
    prob_long = pd.to_numeric(row.get("prob_long", 0), errors="coerce") or 0

    # 中期概率可能是 0-100，转换为 0-1
    if prob_mid > 1:
        prob_mid = prob_mid / 100

    base = (
        w.get("short", 0.30) * prob_short
        + w.get("mid", 0.40) * prob_mid
        + w.get("long", 0.20) * prob_long
    )

    # 阶段加成
    stage = str(row.get("market_stage", ""))
    n_green = sum([prob_short >= 0.7, prob_mid >= 0.7, prob_long >= 0.7])
    n_yellow = sum([0.5 <= p < 0.7 for p in [prob_short, prob_mid, prob_long]])

    bonus = 0.0
    if n_green == 3:
        if stage in ("拉升初期",):
            bonus = stage_bonus_cfg.get("all_green_early_rally", 0.15)
        elif stage in ("拉升中期",):
            bonus = stage_bonus_cfg.get("all_green_mid_rally", 0.10)
    elif n_green == 2 and n_yellow == 1 and stage in ("拉升初期",):
        bonus = stage_bonus_cfg.get("two_green_early_rally", 0.05)

    return min(1.0, base + stage_w * bonus)


# ============================================================================
# 主流程
# ============================================================================

def enrich_predictions(date_str: str):
    """对指定日期的预测结果进行 enrich"""
    pred_file = PREDICTION_DIR / f"predictions_{date_str}_all.csv"
    if not pred_file.exists():
        log.warning(f"预测文件不存在: {pred_file}")
        for suffix in ["top100.csv", "top50.csv"]:
            alt = PREDICTION_DIR / f"predictions_{date_str}_{suffix}"
            if alt.exists():
                pred_file = alt
                break
        if not pred_file.exists():
            return

    log.info(f"Enriching predictions: {pred_file.name}")
    df_pred = pd.read_csv(pred_file)
    if df_pred.empty:
        log.warning("预测结果为空")
        return

    ts_codes = df_pred["ts_code"].unique().tolist()
    log.info(f"需要 enrich 的标的数: {len(ts_codes)}")

    provider = ArcticDataProvider()

    # 读取数据
    start_dt = pd.to_datetime(date_str) - pd.Timedelta(days=150)
    start_str = start_dt.strftime("%Y%m%d")

    df_ohlcv = provider.read_daily_ohlcv(start_str, date_str)
    df_factors = provider.read_daily_factors(start_str, date_str)
    df_basic = provider.read_daily_basic(start_str, date_str)

    log.info(f"ArcticDB 数据: ohlcv={len(df_ohlcv)}, factors={len(df_factors)}, basic={len(df_basic)}")

    # 加载模型
    short_scorer, long_scorer = load_scorers()
    use_model_short = short_scorer is not None
    use_model_long = long_scorer is not None

    if use_model_short:
        log.info("短期评分使用模型预测")
    else:
        log.info("短期评分使用规则打分（模型未找到或加载失败）")

    if use_model_long:
        log.info("长期评分使用模型预测")
    else:
        log.info("长期评分使用规则打分（模型未找到或加载失败）")

    # 加载共振配置
    resonance_cfg = load_resonance_config()

    # 读取完整配置（包含中期校准参数）
    config_path = PROJECT_ROOT / "config" / "3l_scoring.yaml"
    full_cfg = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            full_cfg = yaml.safe_load(f) or {}

    # 预计算行业 z-score（用于长期模型）
    try:
        df_basic_ref = provider.read_stock_basic()
        industry_map = df_basic_ref.set_index("ts_code")["industry"].to_dict()
        name_map = df_basic_ref.set_index("ts_code")["name"].to_dict()
    except Exception:
        name_map = {}

    # 补充股票名称（上游预测文件可能缺失）
    if "name" not in df_pred.columns or df_pred["name"].isna().all():
        if name_map:
            df_pred["name"] = df_pred["ts_code"].map(name_map)
            filled = df_pred["name"].notna().sum()
            log.info(f"已从 stock_basic 补充名称: {filled}/{len(df_pred)}")

    try:
        df_basic_zscore = df_basic.copy()
        df_basic_zscore["industry"] = df_basic_zscore["ts_code"].map(industry_map)
        def _zscore(group):
            group = group.copy()
            for col in ["pe", "pb"]:
                if col in group.columns:
                    mean = group[col].mean()
                    std = group[col].std()
                    if std and std > 0:
                        group[f"{col}_industry_zscore"] = (group[col] - mean) / std
                    else:
                        group[f"{col}_industry_zscore"] = 0
            return group
        df_basic_zscore = df_basic_zscore.groupby(["trade_date", "industry"], group_keys=False).apply(_zscore)
        zscore_map = df_basic_zscore.set_index(["ts_code", "trade_date"])[["pe_industry_zscore", "pb_industry_zscore"]].to_dict("index")
    except Exception:
        zscore_map = {}

    # 初始化 enrich 列
    df_pred["prob_short"] = np.nan
    df_pred["prob_long"] = np.nan
    df_pred["market_stage"] = "未知"
    df_pred["left_side_signal"] = ""
    df_pred["resonance_score"] = np.nan

    # 中期模型概率校准（解决 ensemble 后分布极度右偏问题）
    mid_cal_cfg = full_cfg.get("mid_term_calibration", {})
    if mid_cal_cfg.get("enabled", False) and "prob" in df_pred.columns:
        alpha = mid_cal_cfg.get("alpha", 0.35)
        min_p = mid_cal_cfg.get("min_prob", 0.03)
        max_p = mid_cal_cfg.get("max_prob", 0.99)
        # 保留原始 prob 用于对比
        df_pred["prob_raw"] = df_pred["prob"].copy()
        # 幂变换校准
        df_pred["prob"] = np.clip(np.power(df_pred["prob"], alpha), min_p, max_p)
        log.info(f"中期概率已校准: alpha={alpha}, 校准前均值={df_pred['prob_raw'].mean():.3f}, 校准后均值={df_pred['prob'].mean():.3f}")

    enriched_count = 0
    short_model_probs = []
    long_model_probs = []

    for idx, row in df_pred.iterrows():
        ts_code = row["ts_code"]

        # 取该股票的数据
        f = df_factors[df_factors["ts_code"] == ts_code]
        factor_row = f.iloc[-1] if not f.empty else pd.Series()

        b = df_basic[df_basic["ts_code"] == ts_code]
        basic_row = b.iloc[-1] if not b.empty else pd.Series()

        o = df_ohlcv[df_ohlcv["ts_code"] == ts_code].sort_values("trade_date")

        # 合并单行数据
        merged = pd.concat([factor_row, basic_row])

        # 计算近N日涨幅（从ohlcv）
        if len(o) >= 4:
            merged["return_3d"] = o["close"].iloc[-1] / o["close"].iloc[-4] - 1
        else:
            merged["return_3d"] = np.nan
        if len(o) >= 5:
            merged["return_5d"] = o["close"].iloc[-1] / o["close"].iloc[-5] - 1
        else:
            merged["return_5d"] = np.nan
        if len(o) >= 11:
            merged["return_10d"] = o["close"].iloc[-1] / o["close"].iloc[-11] - 1
        else:
            merged["return_10d"] = np.nan
        if len(o) >= 20:
            merged["return_20d"] = o["close"].iloc[-1] / o["close"].iloc[-20] - 1
        else:
            merged["return_20d"] = np.nan
        merged["return_1d"] = o["pct_chg"].iloc[-1] / 100 if len(o) > 0 else np.nan

        # 计算其他衍生特征
        close_vals = o["close"].values if len(o) > 0 else []
        if len(close_vals) >= 20:
            merged["close_ma20_ratio"] = close_vals[-1] / np.mean(close_vals[-20:]) - 1
        if len(close_vals) >= 60:
            merged["close_ma60_ratio"] = close_vals[-1] / np.mean(close_vals[-60:]) - 1
        if len(close_vals) >= 120:
            merged["close_ma120_ratio"] = close_vals[-1] / np.mean(close_vals[-120:]) - 1
        if len(close_vals) >= 10:
            pct = pd.Series(close_vals).pct_change().dropna()
            merged["volatility_10d"] = pct.tail(10).std() if len(pct) >= 10 else 0
            merged["volatility_60d"] = pct.tail(60).std() if len(pct) >= 60 else 0
            merged["volatility_120d"] = pct.tail(min(120, len(pct))).std() if len(pct) >= 20 else 0
        if len(close_vals) >= 5:
            pct = pd.Series(close_vals).pct_change().dropna()
            merged["volatility_5d"] = pct.tail(5).std() if len(pct) >= 5 else 0

        # 成交量变化（近5日 vs 前5日）与成交额比率
        vol_vals = o["vol"].values if len(o) > 0 and "vol" in o.columns else []
        amt_vals = o["amount"].values if len(o) > 0 and "amount" in o.columns else []
        if len(vol_vals) >= 10:
            recent_vol = np.mean(vol_vals[-5:])
            prev_vol = np.mean(vol_vals[-10:-5])
            merged["vol_change_5d"] = recent_vol / prev_vol - 1 if prev_vol > 0 else 0
        if len(amt_vals) >= 10:
            merged["amount_ratio"] = amt_vals[-1] / np.mean(amt_vals[-20:]) if len(amt_vals) >= 20 else amt_vals[-1] / np.mean(amt_vals[-10:])

        # 长期动量
        if len(close_vals) >= 21:
            merged["return_20d"] = close_vals[-1] / close_vals[-21] - 1
        if len(close_vals) >= 61:
            merged["return_60d"] = close_vals[-1] / close_vals[-61] - 1
        if len(close_vals) >= 121:
            merged["return_120d"] = close_vals[-1] / close_vals[-121] - 1

        # 60 日最大回撤
        if len(close_vals) >= 60:
            rc = close_vals[-60:]
            rolling_max = pd.Series(rc).cummax()
            dd = (rc - rolling_max) / rolling_max
            merged["max_drawdown_60d"] = dd.min()

        # 趋势强度
        if len(close_vals) >= 60:
            rc = close_vals[-60:]
            xi = np.arange(len(rc))
            slope = np.polyfit(xi, rc, 1)[0]
            resid = rc - np.polyval(np.polyfit(xi, rc, 1), xi)
            se = np.std(resid) / np.sqrt(len(rc)) if len(rc) > 1 else 1e-6
            merged["trend_strength_60d"] = slope / max(se, 1e-6)

        # 行业 z-score
        t1 = pd.to_datetime(row.get("trade_date", date_str))
        zscore_key = (ts_code, t1)
        if zscore_key in zscore_map:
            merged["pe_industry_zscore"] = zscore_map[zscore_key].get("pe_industry_zscore", 0)
            merged["pb_industry_zscore"] = zscore_map[zscore_key].get("pb_industry_zscore", 0)
        else:
            merged["pe_industry_zscore"] = 0
            merged["pb_industry_zscore"] = 0

        # 市值对数
        mv = merged.get("total_mv", np.nan)
        if pd.notna(mv) and mv > 0:
            merged["total_mv_log"] = np.log(mv)
        cmv = merged.get("circ_mv", np.nan)
        if pd.notna(cmv) and cmv > 0:
            merged["circ_mv_log"] = np.log(cmv)

        # 4. 计算 enrich 字段
        # prob_short
        try:
            if use_model_short:
                # 收集到批量预测列表中
                short_model_probs.append((idx, merged.copy()))
            else:
                df_pred.at[idx, "prob_short"] = round(calc_prob_short_rule(merged), 3)
        except Exception:
            pass

        # prob_long
        try:
            if use_model_long:
                long_model_probs.append((idx, merged.copy()))
            else:
                df_pred.at[idx, "prob_long"] = round(calc_prob_long_rule(merged), 3)
        except Exception:
            pass

        # market_stage
        try:
            if len(o) >= 60:
                stage = classify_market_stage(o)
                df_pred.at[idx, "market_stage"] = stage
            else:
                df_pred.at[idx, "market_stage"] = "数据不足"
        except Exception as e:
            log.debug(f"{ts_code} 四阶段识别失败: {e}")

        # left_side_signal
        try:
            df_pred.at[idx, "left_side_signal"] = calc_left_side_signal(merged)
        except Exception:
            pass

        enriched_count += 1

    # 批量模型预测（短期）
    if use_model_short and short_model_probs:
        try:
            indices, rows = zip(*short_model_probs)
            # 用 dict 构建 DataFrame，避免 Series 索引冲突
            df_batch = pd.DataFrame([dict(r) for r in rows])
            # 去重列名（factor 和 basic 可能有重复列如 ts_code/trade_date）
            df_batch = df_batch.loc[:, ~df_batch.columns.duplicated()]
            probs = calc_prob_short_model(short_scorer, df_batch)
            for i, idx in enumerate(indices):
                df_pred.at[idx, "prob_short"] = round(probs[i], 3)
        except Exception as e:
            log.warning(f"短期模型批量预测失败: {e}，回退到规则打分")
            for idx, merged in short_model_probs:
                try:
                    df_pred.at[idx, "prob_short"] = round(calc_prob_short_rule(merged), 3)
                except Exception:
                    pass

    # 批量模型预测（长期）
    if use_model_long and long_model_probs:
        try:
            indices, rows = zip(*long_model_probs)
            df_batch = pd.DataFrame([dict(r) for r in rows])
            df_batch = df_batch.loc[:, ~df_batch.columns.duplicated()]
            probs = calc_prob_long_model(long_scorer, df_batch)
            for i, idx in enumerate(indices):
                df_pred.at[idx, "prob_long"] = round(probs[i], 3)
        except Exception as e:
            log.warning(f"长期模型批量预测失败: {e}，回退到规则打分")
            for idx, merged in long_model_probs:
                try:
                    df_pred.at[idx, "prob_long"] = round(calc_prob_long_rule(merged), 3)
                except Exception:
                    pass

    # 计算共振评分
    for idx in df_pred.index:
        try:
            df_pred.at[idx, "resonance_score"] = round(calc_resonance(df_pred.loc[idx], resonance_cfg), 3)
        except Exception:
            pass

    log.info(f"Enrich 完成: {enriched_count}/{len(df_pred)}")

    # 保存 enriched 文件
    out_file = PREDICTION_DIR / f"predictions_{date_str}_all_enriched.csv"
    df_pred.to_csv(out_file, index=False)
    log.info(f"已保存 enriched 文件: {out_file}")

    # 同时保存 top50 / top100 版本
    for top_n in [50, 100]:
        prob_col = None
        for c in ["prob", "probability", "adjusted_score"]:
            if c in df_pred.columns:
                prob_col = c
                break
        if prob_col:
            df_top = df_pred.sort_values(prob_col, ascending=False).head(top_n)
            out_top = PREDICTION_DIR / f"predictions_{date_str}_top{top_n}_enriched.csv"
            df_top.to_csv(out_top, index=False)
            log.info(f"已保存 top{top_n} enriched: {out_top}")


def main():
    parser = argparse.ArgumentParser(description="Enrich prediction results with multi-cycle features")
    parser.add_argument("--date", required=True, help="预测日期 YYYYMMDD")
    args = parser.parse_args()
    enrich_predictions(args.date)


if __name__ == "__main__":
    main()
