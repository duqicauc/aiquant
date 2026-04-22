"""
XGBoost模型训练脚本

推荐作为第一个baseline模型！
"""

import sys
import os
import warnings
import pandas as pd
from datetime import datetime
import json

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 忽略警告
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb
from src.utils.logger import log


def load_and_prepare_data(neg_version="v2"):
    """
    加载并准备训练数据

    Args:
        neg_version: 负样本版本 ('v1' 或 'v2')

    Returns:
        df_features: 特征DataFrame
    """
    log.info("=" * 80)
    log.info("第一步：加载数据")
    log.info("=" * 80)

    # 加载正样本
    df_pos = pd.read_csv("data/processed/feature_data_34d.csv")
    df_pos["label"] = 1
    log.success(f"✓ 正样本加载完成: {len(df_pos)} 条")

    # 加载负样本
    if neg_version == "v2":
        neg_file = "data/processed/negative_feature_data_v2_34d.csv"
    else:
        neg_file = "data/processed/negative_feature_data_34d.csv"

    df_neg = pd.read_csv(neg_file)
    log.success(f"✓ 负样本加载完成: {len(df_neg)} 条 (版本: {neg_version})")

    # 合并
    df = pd.concat([df_pos, df_neg])
    log.info(f"✓ 数据合并完成: {len(df)} 条")
    log.info(f"  - 正样本: {len(df_pos)} 条")
    log.info(f"  - 负样本: {len(df_neg)} 条")
    log.info("")

    return df


def extract_features(df):
    """
    从34天的时序数据中提取统计特征

    Args:
        df: 原始DataFrame（每行是一天的数据）

    Returns:
        df_features: 特征DataFrame（每行是一个样本）
    """
    log.info("=" * 80)
    log.info("第二步：特征工程")
    log.info("=" * 80)
    log.info("将34天时序数据转换为统计特征...")

    features = []
    sample_ids = df["sample_id"].unique()

    for i, sample_id in enumerate(sample_ids):
        if (i + 1) % 500 == 0:
            log.info(f"进度: {i+1}/{len(sample_ids)}")

        sample_data = df[df["sample_id"] == sample_id].sort_values("days_to_t1")

        if len(sample_data) < 20:  # 至少20天数据
            continue

        feature_dict = {
            "sample_id": sample_id,
            "label": int(sample_data["label"].iloc[0]),
        }

        # 价格特征
        feature_dict["close_mean"] = sample_data["close"].mean()
        feature_dict["close_std"] = sample_data["close"].std()
        feature_dict["close_max"] = sample_data["close"].max()
        feature_dict["close_min"] = sample_data["close"].min()
        feature_dict["close_trend"] = (
            (sample_data["close"].iloc[-1] - sample_data["close"].iloc[0]) / sample_data["close"].iloc[0] * 100
        )

        # 涨跌幅特征
        feature_dict["pct_chg_mean"] = sample_data["pct_chg"].mean()
        feature_dict["pct_chg_std"] = sample_data["pct_chg"].std()
        feature_dict["pct_chg_sum"] = sample_data["pct_chg"].sum()
        feature_dict["positive_days"] = (sample_data["pct_chg"] > 0).sum()
        feature_dict["negative_days"] = (sample_data["pct_chg"] < 0).sum()
        feature_dict["max_gain"] = sample_data["pct_chg"].max()
        feature_dict["max_loss"] = sample_data["pct_chg"].min()

        # 量比特征
        if "volume_ratio" in sample_data.columns:
            feature_dict["volume_ratio_mean"] = sample_data["volume_ratio"].mean()
            feature_dict["volume_ratio_max"] = sample_data["volume_ratio"].max()
            feature_dict["volume_ratio_gt_2"] = (sample_data["volume_ratio"] > 2).sum()
            feature_dict["volume_ratio_gt_4"] = (sample_data["volume_ratio"] > 4).sum()

        # MACD特征
        if "macd" in sample_data.columns:
            macd_data = sample_data["macd"].dropna()
            if len(macd_data) > 0:
                feature_dict["macd_mean"] = macd_data.mean()
                feature_dict["macd_positive_days"] = (macd_data > 0).sum()
                feature_dict["macd_max"] = macd_data.max()

        # MA特征
        if "ma5" in sample_data.columns:
            feature_dict["ma5_mean"] = sample_data["ma5"].mean()
            feature_dict["price_above_ma5"] = (sample_data["close"] > sample_data["ma5"]).sum()

        if "ma10" in sample_data.columns:
            feature_dict["ma10_mean"] = sample_data["ma10"].mean()
            feature_dict["price_above_ma10"] = (sample_data["close"] > sample_data["ma10"]).sum()

        # 市值特征
        if "total_mv" in sample_data.columns:
            mv_data = sample_data["total_mv"].dropna()
            if len(mv_data) > 0:
                feature_dict["total_mv_mean"] = mv_data.mean()

        if "circ_mv" in sample_data.columns:
            circ_mv_data = sample_data["circ_mv"].dropna()
            if len(circ_mv_data) > 0:
                feature_dict["circ_mv_mean"] = circ_mv_data.mean()

        # 动量特征（分段收益率）
        days = len(sample_data)
        if days >= 7:
            feature_dict["return_1w"] = (
                (sample_data["close"].iloc[-1] - sample_data["close"].iloc[-7]) / sample_data["close"].iloc[-7] * 100
            )
        if days >= 14:
            feature_dict["return_2w"] = (
                (sample_data["close"].iloc[-1] - sample_data["close"].iloc[-14]) / sample_data["close"].iloc[-14] * 100
            )

        features.append(feature_dict)

    df_features = pd.DataFrame(features)

    log.success(f"✓ 特征提取完成: {len(df_features)} 个样本")
    log.info(f"✓ 特征维度: {len(df_features.columns) - 2} 个特征")
    log.info(f"✓ 特征列表: {list(df_features.columns[2:])}")
    log.info("")

    return df_features


def train_model(df_features, test_size=0.2):
    """
    训练XGBoost模型

    Args:
        df_features: 特征DataFrame
        test_size: 测试集比例

    Returns:
        model, metrics
    """
    log.info("=" * 80)
    log.info("第三步：训练XGBoost模型")
    log.info("=" * 80)

    # 准备数据
    X = df_features.drop(["sample_id", "label"], axis=1)
    y = df_features["label"]

    # 处理缺失值
    X = X.fillna(0)

    log.info(f"特征矩阵: {X.shape}")
    log.info(f"标签分布: {y.value_counts().to_dict()}")
    log.info("")

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

    log.info(f"训练集: {len(X_train)} 个样本")
    log.info(f"测试集: {len(X_test)} 个样本")
    log.info("")

    # 训练模型
    log.info("开始训练...")
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        eval_metric="logloss",
    )

    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    log.success("✓ 模型训练完成！")
    log.info("")

    # 预测
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # 评估
    log.info("=" * 80)
    log.info("第四步：模型评估")
    log.info("=" * 80)

    # 分类报告
    log.info("\n分类报告:")
    report = classification_report(y_test, y_pred, target_names=["负样本", "正样本"], output_dict=True)
    print(classification_report(y_test, y_pred, target_names=["负样本", "正样本"]))

    # AUC
    auc = roc_auc_score(y_test, y_prob)
    log.info(f"\nAUC-ROC: {auc:.4f}")

    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    log.info("\n混淆矩阵:")
    log.info(f"  真负例(TN): {cm[0,0]:4d}  |  假正例(FP): {cm[0,1]:4d}")
    log.info(f"  假负例(FN): {cm[1,0]:4d}  |  真正例(TP): {cm[1,1]:4d}")

    # 特征重要性
    feature_importance = pd.DataFrame({"feature": X.columns, "importance": model.feature_importances_}).sort_values(
        "importance", ascending=False
    )

    log.info("\n" + "=" * 80)
    log.info("特征重要性 Top 10:")
    log.info("=" * 80)
    for idx, row in feature_importance.head(10).iterrows():
        log.info(f"  {row['feature']:25s}: {row['importance']:.4f}")

    # 汇总指标
    metrics = {
        "accuracy": report["accuracy"],
        "precision": report["正样本"]["precision"],
        "recall": report["正样本"]["recall"],
        "f1_score": report["正样本"]["f1-score"],
        "auc": auc,
        "confusion_matrix": cm.tolist(),
        "feature_importance": feature_importance.to_dict("records"),
    }

    return model, metrics, X_test, y_test, y_prob


def save_model(model, metrics, neg_version):
    """保存模型和结果"""
    log.info("\n" + "=" * 80)
    log.info("第五步：保存模型")
    log.info("=" * 80)

    # 创建目录
    os.makedirs("models", exist_ok=True)
    os.makedirs("data/results", exist_ok=True)

    # 保存模型
    model_file = f'models/xgboost_{neg_version}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    model.save_model(model_file)
    log.success(f"✓ 模型已保存: {model_file}")

    # 保存指标
    metrics_file = f"data/results/xgboost_{neg_version}_metrics.json"
    metrics["model_file"] = model_file
    metrics["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metrics["neg_version"] = neg_version

    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    log.success(f"✓ 评估报告已保存: {metrics_file}")
    log.info("")


def main():
    """主函数"""
    log.info("=" * 80)
    log.info("XGBoost 股票选股模型训练")
    log.info("=" * 80)
    log.info("")

    # 选择负样本版本
    NEG_VERSION = "v2"  # 'v1' 或 'v2'

    log.info("配置:")
    log.info(f"  负样本版本: {NEG_VERSION}")
    log.info("  测试集比例: 0.2")
    log.info("  模型: XGBoost")
    log.info("")

    try:
        # 1. 加载数据
        df = load_and_prepare_data(neg_version=NEG_VERSION)

        # 2. 特征工程
        df_features = extract_features(df)

        # 3. 训练模型
        model, metrics, X_test, y_test, y_prob = train_model(df_features)

        # 4. 保存模型
        save_model(model, metrics, NEG_VERSION)

        # 5. 最终总结
        log.info("=" * 80)
        log.success("✅ 模型训练完成！")
        log.info("=" * 80)
        log.info("")
        log.info("📊 模型性能总结:")
        log.info(f"  准确率 (Accuracy):  {metrics['accuracy']:.2%}")
        log.info(f"  精确率 (Precision): {metrics['precision']:.2%}")
        log.info(f"  召回率 (Recall):    {metrics['recall']:.2%}")
        log.info(f"  F1分数 (F1-Score):  {metrics['f1_score']:.2%}")
        log.info(f"  AUC-ROC:            {metrics['auc']:.4f}")
        log.info("")
        log.info("🎯 下一步:")
        log.info("  1. 查看特征重要性，优化特征工程")
        log.info("  2. 尝试不同的负样本版本（v1 vs v2）")
        log.info("  3. 调整超参数提升性能")
        log.info("  4. 使用模型进行回测验证")
        log.info("")

    except FileNotFoundError as e:
        log.error(f"✗ 文件未找到: {e}")
        log.error("请先运行以下命令准备数据:")
        log.error("  1. python scripts/prepare_positive_samples.py")
        log.error("  2. python scripts/prepare_negative_samples_v2.py")
    except Exception as e:
        log.error(f"✗ 训练过程出错: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
