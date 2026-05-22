#!/usr/bin/env python3
"""BreakoutScorer 快速回测 — 使用测试集样本直接验证"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
import joblib

from src.features.breakout_feature_extractor import BreakoutFeatureExtractor
from src.features.multits_flattener import flatten_multits
from src.utils.logger import log

# 加载测试集样本
sample_dir = PROJECT_ROOT / "data" / "training" / "samples" / "v310" / "breakout" / "test"
samples = []
for fname, label in [("positive.csv", 1), ("negative.csv", 0), ("hard_negative.csv", 0)]:
    df = pd.read_csv(sample_dir / fname)
    df["label"] = label
    df["sample_type"] = fname.replace(".csv", "")
    samples.append(df)
samples_df = pd.concat(samples, ignore_index=True)
log.info(f"测试集样本: {len(samples_df)} 个")

# 加载模型
model_dir = sorted([d for d in (PROJECT_ROOT / "data" / "models" / "v310" / "breakout").iterdir() if d.is_dir()], reverse=True)[0]
model = xgb.Booster()
model.load_model(str(model_dir / "model.json"))
calibrator = joblib.load(model_dir / "calibrator.pkl")
with open(model_dir / "feature_names.json") as f:
    feature_names = json.load(f)
log.info(f"模型加载: {model_dir}")

# 批量提取特征
extractor = BreakoutFeatureExtractor(use_cache=True)
df_features = extractor.extract_for_samples(samples_df, lookback_days=34, batch_size=100)
log.info(f"特征提取: {len(df_features)} 行")

# 展平
meta_cols = ["sample_id", "ts_code", "trade_date", "days_to_t1", "label", "sample_type"]
feature_cols = [c for c in df_features.columns if c not in meta_cols and c not in ["name", "t1_date", "t1_close", "t1_vol"]]
df_flat = flatten_multits(df_features, feature_cols)
log.info(f"展平: {len(df_flat)} 样本")

# 合并 confirm_return（从原始样本）
samples_df["t1_date_str"] = pd.to_datetime(samples_df["t1_date"].astype(str)).dt.strftime("%Y%m%d")
confirm_map = samples_df.set_index(["ts_code", "t1_date_str"])["confirm_return"].to_dict()
df_flat["t1_date"] = pd.to_datetime(df_flat["trade_date"]).dt.strftime("%Y%m%d")
df_flat["confirm_return"] = df_flat.apply(lambda r: confirm_map.get((r["ts_code"], r["t1_date"]), np.nan), axis=1)

# 对齐特征
X = df_flat[feature_names].values
y = df_flat["label"].values

# 预测
pred_raw = model.predict(xgb.DMatrix(X, feature_names=feature_names))
pred_cal = calibrator.transform(pred_raw) if hasattr(calibrator, "transform") else pred_raw

df_flat["prob_raw"] = pred_raw
df_flat["prob_cal"] = pred_cal

# 按日期分组，模拟Top K选股
results = []
for date, group in df_flat.groupby("t1_date"):
    group = group.sort_values("prob_cal", ascending=False)
    top_k = group.head(20)
    if len(top_k) == 0 or top_k["confirm_return"].isna().all():
        continue
    avg_return = top_k["confirm_return"].mean()
    win_rate = (top_k["confirm_return"] > 0).mean()
    results.append({
        "date": date,
        "n_selected": len(top_k),
        "avg_return": avg_return,
        "win_rate": win_rate,
        "avg_prob": top_k["prob_cal"].mean(),
    })

results_df = pd.DataFrame(results).sort_values("date")
results_df["cum_return"] = (1 + results_df["avg_return"] / 100).cumprod() - 1

# 计算回测指标
returns = results_df["avg_return"] / 100
sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
max_dd = (results_df["cum_return"].cummax() - results_df["cum_return"]).max()
overall_win = (returns > 0).mean()

print(f"\n{'='*60}")
print(f"BreakoutScorer 快速回测结果 (测试集: {len(results_df)} 交易日)")
print(f"{'='*60}")
print(f"日均收益:     {returns.mean()*100:.3f}%")
print(f"日收益标准差: {returns.std()*100:.3f}%")
print(f"年化夏普:     {sharpe:.3f}")
print(f"累计收益:     {results_df['cum_return'].iloc[-1]*100:.2f}%")
print(f"最大回撤:     {max_dd*100:.2f}%")
print(f"日胜率:       {overall_win*100:.1f}%")
print(f"{'='*60}")

# 保存结果
out_dir = PROJECT_ROOT / "data" / "backtest" / "v310" / "quick"
out_dir.mkdir(parents=True, exist_ok=True)
results_df.to_csv(out_dir / "breakout_only_daily.csv", index=False)
print(f"结果已保存: {out_dir / 'breakout_only_daily.csv'}")
