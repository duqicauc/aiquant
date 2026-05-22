"""
ETF 统一评分引擎单元测试（Tushare 成熟数据优先版）
"""

import numpy as np
import pandas as pd

from src.analysis.etf_scorer import (
    calc_etf_opportunity_score,
    calc_theme_opportunity_score,
    recommendation_label,
    _merge_tushare_data,
)


def _make_daily_df(n=60, trend="up"):
    """生成模拟 fund_daily 数据"""
    np.random.seed(42)
    dates = pd.date_range(end="2026-05-18", periods=n, freq="B")
    base = 100.0
    prices = [base]
    for i in range(1, n):
        noise = np.random.normal(0, 0.015)
        if trend == "up":
            drift = 0.003
        elif trend == "down":
            drift = -0.003
        else:
            drift = 0.0
        prices.append(prices[-1] * (1 + drift + noise))
    prices = np.array(prices)
    df = pd.DataFrame({
        "trade_date": dates,
        "open": prices * 0.99,
        "high": prices * 1.02,
        "low": prices * 0.98,
        "close": prices,
        "pre_close": np.roll(prices, 1),
        "vol": np.random.randint(100000, 500000, n),
        "amount": np.random.randint(1000000, 5000000, n),
        "pct_chg": np.diff(prices, prepend=prices[0]) / np.roll(prices, 1) * 100,
    })
    return df


def _make_factor_df(df_daily):
    """生成模拟 stk_factor 数据（Tushare 技术因子）"""
    np.random.seed(43)
    n = len(df_daily)
    close = df_daily["close"].values
    high = df_daily["high"].values
    low = df_daily["low"].values

    # MACD
    ema12 = pd.Series(close).ewm(span=12, adjust=False).mean()
    ema26 = pd.Series(close).ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    macd = (dif - dea) * 2

    # KDJ
    lowest_low = pd.Series(low).rolling(9).min()
    highest_high = pd.Series(high).rolling(9).max()
    rsv = (close - lowest_low) / (highest_high - lowest_low) * 100
    k = rsv.ewm(com=2, adjust=False).mean()
    d = k.ewm(com=2, adjust=False).mean()
    j = 3 * k - 2 * d

    # RSI
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    # BOLL
    ma20 = pd.Series(close).rolling(20).mean()
    std20 = pd.Series(close).rolling(20).std()

    # MA
    ma5 = pd.Series(close).rolling(5).mean()
    ma10 = pd.Series(close).rolling(10).mean()
    ma60 = pd.Series(close).rolling(60).mean()

    # ATR
    tr = pd.concat([
        pd.Series(high) - pd.Series(low),
        abs(pd.Series(high) - pd.Series(close).shift(1)),
        abs(pd.Series(low) - pd.Series(close).shift(1)),
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()

    # DMI
    pdi = pd.Series(np.random.normal(25, 5, n)).clip(10, 50)
    mdi = pd.Series(np.random.normal(20, 5, n)).clip(10, 50)
    adx = pd.Series(np.random.normal(25, 8, n)).clip(10, 50)

    # TAQ
    taq_up = pd.Series(close).rolling(20).max()
    taq_down = pd.Series(close).rolling(20).min()
    taq_mid = (taq_up + taq_down) / 2

    df = pd.DataFrame({
        "trade_date": df_daily["trade_date"],
        "macd_dif": dif.values,
        "macd_dea": dea.values,
        "macd": macd.values,
        "kdj_k": k.values,
        "kdj_d": d.values,
        "kdj_j": j.values,
        "rsi_6": rsi.values,
        "rsi_12": rsi.values,
        "boll_upper": (ma20 + 2 * std20).values,
        "boll_mid": ma20.values,
        "boll_lower": (ma20 - 2 * std20).values,
        "ma5": ma5.values,
        "ma10": ma10.values,
        "ma_20d": ma20.values,
        "ma60": ma60.values,
        "ema_5": pd.Series(close).ewm(span=5, adjust=False).mean().values,
        "ema_10": pd.Series(close).ewm(span=10, adjust=False).mean().values,
        "ema_20": pd.Series(close).ewm(span=20, adjust=False).mean().values,
        "ema_60": pd.Series(close).ewm(span=60, adjust=False).mean().values,
        "cci": np.random.normal(0, 50, n),
        "atr": atr.values,
        "dmi_pdi": pdi.values,
        "dmi_mdi": mdi.values,
        "dmi_adx": adx.values,
        "dmi_adxr": adx.values,
        "taq_up": taq_up.values,
        "taq_mid": taq_mid.values,
        "taq_down": taq_down.values,
        "trix": np.random.normal(0.001, 0.01, n),
        "trma": np.random.normal(0.001, 0.01, n),
        "obv": np.cumsum(np.sign(np.diff(close, prepend=close[0])) * df_daily["vol"].values),
        "mfi": np.random.normal(50, 15, n).clip(10, 90),
        "emv": np.random.normal(0, 1, n),
        "maemv": np.random.normal(0, 1, n),
        "vr": np.random.normal(150, 50, n).clip(50, 300),
        "wr": np.random.normal(-50, 15, n).clip(-100, -10),
        "wr1": np.random.normal(-50, 15, n).clip(-100, -10),
        "psy": np.random.normal(50, 10, n).clip(20, 80),
        "psyma": np.random.normal(50, 5, n).clip(30, 70),
        "bias_short": np.random.normal(0, 2, n).clip(-8, 8),
        "bias_mid": np.random.normal(0, 3, n).clip(-10, 10),
        "bias_long": np.random.normal(0, 4, n).clip(-12, 12),
        "mass": np.random.normal(24, 2, n).clip(18, 30),
        "ma_mass": np.random.normal(24, 1, n).clip(20, 28),
        "ktn_upper": (ma20 + std20).values,
        "ktn_mid": ma20.values,
        "ktn_down": (ma20 - std20).values,
        "roc": np.random.normal(0, 3, n),
        "maroc": np.random.normal(0, 2, n),
        "cr": np.random.normal(100, 20, n),
        "brar_br": np.random.normal(100, 20, n),
        "brar_ar": np.random.normal(100, 20, n),
        "bbi": ma20.values,
        "dpo": np.random.normal(0, 2, n),
        "madpo": np.random.normal(0, 1, n),
        "asi": np.random.normal(0, 100, n),
        "asit": np.random.normal(0, 50, n),
        "expma_12": pd.Series(close).ewm(span=12, adjust=False).mean().values,
        "expma_50": pd.Series(close).ewm(span=50, adjust=False).mean().values,
        "mtm": np.random.normal(0, 3, n),
        "mtmma": np.random.normal(0, 2, n),
        "xsii_td1": np.random.normal(0, 5, n),
        "xsii_td2": np.random.normal(0, 5, n),
        "xsii_td3": np.random.normal(0, 5, n),
        "xsii_td4": np.random.normal(0, 5, n),
    })
    return df


def _make_moneyflow_df(df_daily):
    """生成模拟 moneyflow 数据"""
    np.random.seed(44)
    n = len(df_daily)
    net_mf = np.random.normal(1000, 5000, n)
    buy_elg = np.random.normal(5000, 3000, n)
    sell_elg = np.random.normal(4000, 3000, n)
    df = pd.DataFrame({
        "trade_date": df_daily["trade_date"],
        "net_mf_amount": net_mf,
        "buy_elg_amount": buy_elg,
        "sell_elg_amount": sell_elg,
        "buy_lg_amount": buy_elg * 1.2,
        "sell_lg_amount": sell_elg * 1.1,
        "buy_md_amount": buy_elg * 0.8,
        "sell_md_amount": sell_elg * 0.9,
        "buy_sm_amount": buy_elg * 0.5,
        "sell_sm_amount": sell_elg * 0.6,
    })
    return df


def _make_share_df(df_daily):
    """生成模拟 fund_share 数据"""
    np.random.seed(45)
    n = len(df_daily)
    df = pd.DataFrame({
        "trade_date": df_daily["trade_date"],
        "fd_share": [100000000] * n,
        "fd_share_change": [500000] * n,
    })
    return df


def _make_daily_basic_df(df_daily):
    """生成模拟 daily_basic 数据"""
    np.random.seed(46)
    n = len(df_daily)
    df = pd.DataFrame({
        "trade_date": df_daily["trade_date"],
        "turnover_rate": np.random.normal(2.5, 1.5, n).clip(0.1, 10.0),
        "turnover_rate_f": np.random.normal(2.5, 1.5, n).clip(0.1, 10.0),
        "volume_ratio": np.random.normal(1.0, 0.5, n).clip(0.3, 3.0),
        "total_share": [1000000000] * n,
        "float_share": [800000000] * n,
        "total_mv": [100000000] * n,
        "circ_mv": [80000000] * n,
    })
    return df


def test_merge_tushare_data():
    df_daily = _make_daily_df(60, "up")
    df_factor = _make_factor_df(df_daily)
    df_moneyflow = _make_moneyflow_df(df_daily)
    df_share = _make_share_df(df_daily)
    df_db = _make_daily_basic_df(df_daily)

    merged = _merge_tushare_data(df_daily, df_factor, df_moneyflow, df_share, df_db)
    assert "macd_dif" in merged.columns
    assert "net_mf_amount" in merged.columns
    assert "fd_share_change" in merged.columns
    assert "turnover_rate" in merged.columns
    assert "volume_ratio" in merged.columns
    assert len(merged) == len(df_daily)
    print("✅ merge_tushare_data passed")


def test_calc_etf_opportunity_score_full():
    df_daily = _make_daily_df(60, "up")
    df_factor = _make_factor_df(df_daily)
    df_moneyflow = _make_moneyflow_df(df_daily)
    df_share = _make_share_df(df_daily)
    df_db = _make_daily_basic_df(df_daily)

    # 完整 Tushare 数据
    result_full = calc_etf_opportunity_score(
        df_daily,
        df_factor=df_factor,
        df_moneyflow=df_moneyflow,
        df_share=df_share,
        df_daily_basic=df_db,
    )
    assert 0 <= result_full["opportunity_score"] <= 100
    assert result_full["recommendation"] in ["强烈买入", "买入", "关注", "观望", "回避"]
    assert result_full["confidence"] > 0.85  # 多源Tushare数据时置信度高
    assert len(result_full["dimensions"]) == 6
    # 验证新增 bonus 字段
    assert "base_score" in result_full
    assert "trend_strength_score" in result_full
    assert "risk_discount" in result_full
    assert "trend_strength_bonus" in result_full
    assert "synergy_bonus" in result_full
    assert "bonuses" in result_full
    assert "trend_weights" in result_full
    assert result_full["trend_strength_bonus"] >= 0
    assert result_full["synergy_bonus"] >= 0
    # 双轨制验证：趋势强度分应 >= 基础分（因为风险折扣 <= 1）
    assert result_full["trend_strength_score"] >= result_full["base_score"]
    print(f"Full Tushare: score={result_full['opportunity_score']}, rec={result_full['recommendation']}, conf={result_full['confidence']}")

    # 验证各维度使用了 Tushare 数据
    tm = result_full["dimensions"]["trend_momentum"]["breakdown"]
    assert tm.get("data_source") == "tushare_stk_factor"
    assert "dmi_source" in tm
    assert "taq_source" in tm
    assert "trix_source" in tm
    print(f"  TM: dmi={tm['dmi_source']}, taq={tm['taq_source']}, trix={tm['trix_source']}")

    vp = result_full["dimensions"]["volume_price"]["breakdown"]
    assert vp.get("turnover_source") == "tushare_daily_basic"
    assert vp.get("volume_ratio_source") == "tushare_daily_basic"
    assert vp.get("obv_source") == "tushare"
    assert vp.get("mfi_source") == "tushare"
    assert vp.get("vr_source") == "tushare"
    print(f"  VP: turnover={vp['turnover_source']}, obv={vp['obv_source']}, mfi={vp['mfi_source']}, vr={vp['vr_source']}")

    tp = result_full["dimensions"]["technical_pattern"]["breakdown"]
    assert tp.get("kdj_source") == "tushare"
    assert tp.get("boll_source") == "tushare"
    assert tp.get("cci_source") == "tushare"
    assert tp.get("wr_source") == "tushare"
    assert tp.get("psy_source") == "tushare"
    print(f"  TP: kdj={tp['kdj_source']}, boll={tp['boll_source']}, wr={tp['wr_source']}, psy={tp['psy_source']}")

    cf = result_full["dimensions"]["capital_flow"]["breakdown"]
    assert cf.get("moneyflow_available") is True
    assert cf.get("share_available") is True
    assert cf.get("emv_source") == "tushare"
    print(f"  CF: moneyflow={cf['moneyflow_available']}, share={cf['share_available']}, emv={cf['emv_source']}")

    vr = result_full["dimensions"]["volatility_risk"]["breakdown"]
    assert vr.get("atr_source") == "tushare"
    assert vr.get("mass_source") == "tushare"
    print(f"  VR: atr={vr['atr_source']}, mass={vr['mass_source']}")

    mr = result_full["dimensions"]["mean_reversion"]["breakdown"]
    assert mr.get("bias_source") == "tushare"
    assert mr.get("boll_source") == "tushare"
    assert mr.get("ktn_source") == "tushare"
    print(f"  MR: bias={mr['bias_source']}, boll={mr['boll_source']}, ktn={mr['ktn_source']}")

    # 仅 fund_daily（回退模式）
    result_basic = calc_etf_opportunity_score(df_daily)
    assert result_basic["confidence"] <= result_full["confidence"]
    assert "base_score" in result_basic
    print(f"Basic only: score={result_basic['opportunity_score']}, rec={result_basic['recommendation']}, conf={result_basic['confidence']}")

    print("✅ calc_etf_opportunity_score_full passed")


def test_calc_etf_opportunity_score_down_trend():
    df_daily = _make_daily_df(60, "down")
    df_factor = _make_factor_df(df_daily)

    result = calc_etf_opportunity_score(df_daily, df_factor=df_factor)
    assert result["opportunity_score"] < 55
    print(f"Down trend: score={result['opportunity_score']}, rec={result['recommendation']}")
    print("✅ calc_etf_opportunity_score_down_trend passed")


def test_calc_theme_opportunity_score():
    scores = []
    for trend in ["up", "up", "flat", "down"]:
        df = _make_daily_df(60, trend)
        factor = _make_factor_df(df)
        scores.append(calc_etf_opportunity_score(df, df_factor=factor))
    theme_result = calc_theme_opportunity_score(scores)
    assert 0 <= theme_result["opportunity_score"] <= 100
    assert theme_result["etf_count"] == 4
    print(f"Theme: score={theme_result['opportunity_score']}, rec={theme_result['recommendation']}, dispersion={theme_result.get('dispersion')}")
    print("✅ calc_theme_opportunity_score passed")


def test_recommendation_label():
    assert recommendation_label(80) == "强烈买入"
    assert recommendation_label(73) == "买入"
    assert recommendation_label(65) == "关注"
    assert recommendation_label(50) == "观望"
    assert recommendation_label(40) == "回避"
    print("✅ recommendation_label passed")


if __name__ == "__main__":
    test_merge_tushare_data()
    test_calc_etf_opportunity_score_full()
    test_calc_etf_opportunity_score_down_trend()
    test_calc_theme_opportunity_score()
    test_recommendation_label()
    print("\n✅ All tests passed!")
