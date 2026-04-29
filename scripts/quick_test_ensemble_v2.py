#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""快速测试 ensemble v2"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import json
import joblib
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.isotonic import IsotonicRegression

# 加载模型
model_dir = PROJECT_ROOT / "data" / "models" / "breakout_launch_scorer" / "versions" / "v2.9.2-ensemble-v2" / "model"

with open(model_dir / "feature_names.json") as f:
    feature_names = json.load(f)

with open(model_dir / "weights.json") as f:
    weights = json.load(f)

xgb_model = xgb.Booster()
xgb_model.load_model(str(model_dir / "xgboost.ubj"))

lgb_model = lgb.Booster(model_file=str(model_dir / "lightgbm.txt"))

cat_model = CatBoostClassifier()
cat_model.load_model(str(model_dir / "catboost.cbm"))

cal_xgb = joblib.load(str(model_dir / "calibrator_xgboost.pkl"))
cal_lgb = joblib.load(str(model_dir / "calibrator_lightgbm.pkl"))
cal_cat = joblib.load(str(model_dir / "calibrator_catboost.pkl"))

# 加载 2024-10-08 数据测试
print("=== Ensemble v2 概率分布测试 ===")
df = pd.read_csv('data/prediction/v292_conservative_2024q4/predictions_20241008_all.csv')

# 获取特征列
exclude = ['ts_code', 'name', 'prob_raw', 'prob', 'rank', 'industry', 'trade_date']
feature_cols = [c for c in df.columns if c not in exclude and c in feature_names]

# 对齐特征
X = pd.DataFrame(index=df.index)
for col in feature_names:
    if col in df.columns:
        X[col] = pd.to_numeric(df[col], errors='coerce')
    else:
        X[col] = 0.0
X = X.astype(float).fillna(0)

# 预测
pred_xgb_raw = xgb_model.predict(xgb.DMatrix(X, feature_names=feature_names))
pred_lgb_raw = lgb_model.predict(X, num_iteration=lgb_model.best_iteration)
pred_cat_raw = cat_model.predict_proba(X)[:, 1]

pred_xgb = cal_xgb.predict(pred_xgb_raw)
pred_lgb = cal_lgb.predict(pred_lgb_raw)
pred_cat = cal_cat.predict(pred_cat_raw)

ens_pred = pred_xgb * weights['xgboost'] + pred_lgb * weights['lightgbm'] + pred_cat * weights['catboost']

df['ens_prob'] = ens_pred

top10 = df.sort_values('ens_prob', ascending=False).head(10)
print(f'Top10 ens_prob: mean={top10["ens_prob"].mean():.4f}, std={top10["ens_prob"].std():.4f}, range={top10["ens_prob"].min():.4f}~{top10["ens_prob"].max():.4f}')
print(f'Top10 total_mv: mean={top10["total_mv"].mean():.0f}万')

# 对比 v291
v291 = pd.read_csv('data/prediction/v291_stk_factor/predictions_20241008_top50.csv')
v291_top10 = v291.head(10)
print(f'\nv291 Top10 prob: mean={v291_top10["prob"].mean():.4f}, std={v291_top10["prob"].std():.4f}')
print(f'v291 Top10 total_mv: mean={v291_top10["total_mv"].mean():.0f}万')

print('\n=== Ensemble v2 Top10 股票 ===')
for i, row in top10.iterrows():
    print(f'{row["ts_code"]}: ens={row["ens_prob"]:.4f}, mv={row["total_mv"]:.0f}万')
