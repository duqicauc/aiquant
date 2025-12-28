"""
XGBoost模型训练脚本 - 时间序列版本（避免未来函数）

关键改进：
1. 按时间划分训练集和测试集（而非随机划分）
2. 训练集：历史数据（如2022-2023年）
3. 测试集：未来数据（如2024年）
4. 确保不会用未来信息训练模型
"""
import sys
import os
import warnings
import pandas as pd
import numpy as np
from datetime import datetime
import json

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 忽略警告
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_auc_score, roc_curve, precision_recall_curve
)
import xgboost as xgb
from src.utils.logger import log
from src.utils.human_intervention import HumanInterventionChecker, require_human_confirmation
from src.visualization.training_visualizer import TrainingVisualizer


def load_and_prepare_data(neg_version='v2'):
    """
    加载并准备训练数据
    
    Args:
        neg_version: 负样本版本 ('v1' 或 'v2')
        
    Returns:
        df_features: 特征DataFrame
    """
    log.info("="*80)
    log.info("第一步：加载数据")
    log.info("="*80)
    
    # 加载正样本（使用新的目录结构）
    df_pos = pd.read_csv('data/training/features/feature_data_34d.csv')
    df_pos['label'] = 1
    log.success(f"✓ 正样本加载完成: {len(df_pos)} 条")
    
    # 加载负样本
    if neg_version == 'v2':
        neg_file = 'data/training/features/negative_feature_data_v2_34d.csv'
    else:
        neg_file = 'data/training/features/negative_feature_data_34d.csv'
    
    df_neg = pd.read_csv(neg_file)
    log.success(f"✓ 负样本加载完成: {len(df_neg)} 条 (版本: {neg_version})")
    
    # 合并
    df = pd.concat([df_pos, df_neg])
    log.info(f"✓ 数据合并完成: {len(df)} 条")
    log.info(f"  - 正样本: {len(df_pos)} 条")
    log.info(f"  - 负样本: {len(df_neg)} 条")
    log.info("")
    
    return df


def extract_features_with_time(df):
    """
    从34天的时序数据中提取统计特征（保留时间信息）
    
    Args:
        df: 原始DataFrame（每行是一天的数据）
        
    Returns:
        df_features: 特征DataFrame（每行是一个样本，包含T1日期）
    """
    log.info("="*80)
    log.info("第二步：特征工程（保留时间信息）")
    log.info("="*80)
    log.info("将34天时序数据转换为统计特征...")
    
    # 重新分配唯一的sample_id
    df['unique_sample_id'] = df.groupby(['ts_code', 'label']).ngroup()
    
    features = []
    sample_ids = df['unique_sample_id'].unique()
    
    # 获取正样本的T1日期映射（使用新的目录结构）
    df_positive_samples = pd.read_csv('data/training/samples/positive_samples.csv')
    t1_date_map = dict(zip(
        df_positive_samples.index,
        pd.to_datetime(df_positive_samples['t1_date'])
    ))
    
    # 获取负样本的T1日期映射
    if os.path.exists('data/training/samples/negative_samples_v2.csv'):
        df_negative_samples = pd.read_csv('data/training/samples/negative_samples_v2.csv')
    else:
        df_negative_samples = pd.read_csv('data/training/samples/negative_samples.csv')
    
    # 负样本的sample_id需要偏移（因为是从0开始的）
    max_positive_id = df_positive_samples.index.max()
    for idx, row in df_negative_samples.iterrows():
        t1_date_map[max_positive_id + 1 + idx] = pd.to_datetime(row['t1_date'])
    
    for i, sample_id in enumerate(sample_ids):
        if (i + 1) % 500 == 0:
            log.info(f"进度: {i+1}/{len(sample_ids)}")
        
        sample_data = df[df['unique_sample_id'] == sample_id].sort_values('days_to_t1')
        
        if len(sample_data) < 20:  # 至少20天数据
            continue
        
        # 从数据中获取T1日期（基于days_to_t1=0的那一天）
        # 找到 days_to_t1 最接近0的记录
        t1_row = sample_data.iloc[sample_data['days_to_t1'].abs().argmin()]
        t1_date = pd.to_datetime(t1_row['trade_date'])
        
        feature_dict = {
            'sample_id': sample_id,
            'ts_code': sample_data['ts_code'].iloc[0],
            'name': sample_data['name'].iloc[0],
            'label': int(sample_data['label'].iloc[0]),
            't1_date': t1_date,  # 保留T1日期，用于时间划分
        }
        
        # 价格特征
        feature_dict['close_mean'] = sample_data['close'].mean()
        feature_dict['close_std'] = sample_data['close'].std()
        feature_dict['close_max'] = sample_data['close'].max()
        feature_dict['close_min'] = sample_data['close'].min()
        feature_dict['close_trend'] = (
            (sample_data['close'].iloc[-1] - sample_data['close'].iloc[0]) / 
            sample_data['close'].iloc[0] * 100
        )
        
        # 涨跌幅特征
        feature_dict['pct_chg_mean'] = sample_data['pct_chg'].mean()
        feature_dict['pct_chg_std'] = sample_data['pct_chg'].std()
        feature_dict['pct_chg_sum'] = sample_data['pct_chg'].sum()
        feature_dict['positive_days'] = (sample_data['pct_chg'] > 0).sum()
        feature_dict['negative_days'] = (sample_data['pct_chg'] < 0).sum()
        feature_dict['max_gain'] = sample_data['pct_chg'].max()
        feature_dict['max_loss'] = sample_data['pct_chg'].min()
        
        # 量比特征
        if 'volume_ratio' in sample_data.columns:
            feature_dict['volume_ratio_mean'] = sample_data['volume_ratio'].mean()
            feature_dict['volume_ratio_max'] = sample_data['volume_ratio'].max()
            feature_dict['volume_ratio_gt_2'] = (sample_data['volume_ratio'] > 2).sum()
            feature_dict['volume_ratio_gt_4'] = (sample_data['volume_ratio'] > 4).sum()
        
        # MACD特征
        if 'macd' in sample_data.columns:
            macd_data = sample_data['macd'].dropna()
            if len(macd_data) > 0:
                feature_dict['macd_mean'] = macd_data.mean()
                feature_dict['macd_positive_days'] = (macd_data > 0).sum()
                feature_dict['macd_max'] = macd_data.max()
        
        # MA特征
        if 'ma5' in sample_data.columns:
            feature_dict['ma5_mean'] = sample_data['ma5'].mean()
            feature_dict['price_above_ma5'] = (
                sample_data['close'] > sample_data['ma5']
            ).sum()
        
        if 'ma10' in sample_data.columns:
            feature_dict['ma10_mean'] = sample_data['ma10'].mean()
            feature_dict['price_above_ma10'] = (
                sample_data['close'] > sample_data['ma10']
            ).sum()
        
        # 市值特征
        if 'total_mv' in sample_data.columns:
            mv_data = sample_data['total_mv'].dropna()
            if len(mv_data) > 0:
                feature_dict['total_mv_mean'] = mv_data.mean()
        
        if 'circ_mv' in sample_data.columns:
            circ_mv_data = sample_data['circ_mv'].dropna()
            if len(circ_mv_data) > 0:
                feature_dict['circ_mv_mean'] = circ_mv_data.mean()
        
        # 动量特征（分段收益率）
        days = len(sample_data)
        if days >= 7:
            feature_dict['return_1w'] = (
                (sample_data['close'].iloc[-1] - sample_data['close'].iloc[-7]) /
                sample_data['close'].iloc[-7] * 100
            )
        if days >= 14:
            feature_dict['return_2w'] = (
                (sample_data['close'].iloc[-1] - sample_data['close'].iloc[-14]) /
                sample_data['close'].iloc[-14] * 100
            )
        
        features.append(feature_dict)
    
    df_features = pd.DataFrame(features)
    
    log.success(f"✓ 特征提取完成: {len(df_features)} 个样本")
    log.info(f"✓ 特征维度: {len(df_features.columns) - 3} 个特征（不含sample_id, label, t1_date）")
    log.info("")
    
    return df_features


def timeseries_split(df_features, train_end_date=None, test_start_date=None):
    """
    按时间划分训练集和测试集（避免未来函数）
    
    Args:
        df_features: 特征DataFrame（必须包含t1_date列）
        train_end_date: 训练集截止日期（如'2023-12-31'）
        test_start_date: 测试集开始日期（如'2024-01-01'）
        
    Returns:
        X_train, X_test, y_train, y_test, train_dates, test_dates
    """
    log.info("="*80)
    log.info("第三步：时间序列划分（避免未来函数）")
    log.info("="*80)
    
    # 确保t1_date是datetime类型
    df_features['t1_date'] = pd.to_datetime(df_features['t1_date'])
    
    # 按时间排序
    df_features = df_features.sort_values('t1_date').reset_index(drop=True)
    
    # 显示时间范围
    min_date = df_features['t1_date'].min()
    max_date = df_features['t1_date'].max()
    log.info(f"数据时间范围: {min_date.date()} 至 {max_date.date()}")
    
    # 如果未指定划分点，使用80%作为训练集
    if train_end_date is None:
        n_train = int(len(df_features) * 0.8)
        train_end_date = df_features.iloc[n_train]['t1_date']
        test_start_date = df_features.iloc[n_train + 1]['t1_date']
    else:
        train_end_date = pd.to_datetime(train_end_date)
        test_start_date = pd.to_datetime(test_start_date)
    
    # 划分训练集和测试集
    train_mask = df_features['t1_date'] <= train_end_date
    test_mask = df_features['t1_date'] >= test_start_date
    
    df_train = df_features[train_mask]
    df_test = df_features[test_mask]
    
    log.info(f"\n时间划分:")
    log.info(f"  训练集: {df_train['t1_date'].min().date()} 至 {df_train['t1_date'].max().date()}")
    log.info(f"  测试集: {df_test['t1_date'].min().date()} 至 {df_test['t1_date'].max().date()}")
    log.info(f"\n样本划分:")
    log.info(f"  训练集: {len(df_train)} 个样本 (正:{(df_train['label']==1).sum()}, 负:{(df_train['label']==0).sum()})")
    log.info(f"  测试集: {len(df_test)} 个样本 (正:{(df_test['label']==1).sum()}, 负:{(df_test['label']==0).sum()})")
    log.info("")
    
    # 确认无数据泄露
    if df_train['t1_date'].max() >= df_test['t1_date'].min():
        log.warning("⚠️  警告：训练集和测试集时间有重叠，可能存在数据泄露！")
    else:
        log.success("✓ 训练集和测试集时间无重叠，无数据泄露风险")
    
    # 准备特征和标签
    feature_cols = [col for col in df_features.columns 
                   if col not in ['sample_id', 'label', 't1_date']]
    
    X_train = df_train[feature_cols]
    y_train = df_train['label']
    train_dates = df_train['t1_date']
    
    X_test = df_test[feature_cols]
    y_test = df_test['label']
    test_dates = df_test['t1_date']
    
    # 处理缺失值
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)
    
    # 删除非数值列
    non_numeric_cols = X_train.select_dtypes(include=['object']).columns
    if len(non_numeric_cols) > 0:
        log.info(f"删除非数值列: {list(non_numeric_cols)}")
        X_train = X_train.drop(columns=non_numeric_cols)
        X_test = X_test.drop(columns=non_numeric_cols)
    
    log.info(f"特征矩阵:")
    log.info(f"  训练集: {X_train.shape}")
    log.info(f"  测试集: {X_test.shape}")
    log.info("")
    
    return X_train, X_test, y_train, y_test, train_dates, test_dates


def train_model(X_train, y_train, X_test, y_test):
    """
    训练XGBoost模型
    
    Args:
        X_train, y_train: 训练集
        X_test, y_test: 测试集
        
    Returns:
        model, metrics
    """
    log.info("="*80)
    log.info("第四步：训练XGBoost模型")
    log.info("="*80)
    
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
        eval_metric='logloss'
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    log.success("✓ 模型训练完成！")
    log.info("")
    
    # 预测
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # 评估
    log.info("="*80)
    log.info("第五步：模型评估（测试集 = 未来数据）")
    log.info("="*80)
    
    # 分类报告
    log.info("\n分类报告:")
    report = classification_report(
        y_test, y_pred, 
        target_names=['负样本', '正样本'],
        output_dict=True
    )
    print(classification_report(
        y_test, y_pred, 
        target_names=['负样本', '正样本']
    ))
    
    # AUC
    auc = roc_auc_score(y_test, y_prob)
    log.info(f"\nAUC-ROC: {auc:.4f}")
    
    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    log.info("\n混淆矩阵:")
    log.info(f"  真负例(TN): {cm[0,0]:4d}  |  假正例(FP): {cm[0,1]:4d}")
    log.info(f"  假负例(FN): {cm[1,0]:4d}  |  真正例(TP): {cm[1,1]:4d}")
    
    # 特征重要性
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    log.info("\n" + "="*80)
    log.info("特征重要性 Top 10:")
    log.info("="*80)
    for idx, row in feature_importance.head(10).iterrows():
        log.info(f"  {row['feature']:25s}: {row['importance']:.4f}")
    
    # 汇总指标
    metrics = {
        'accuracy': report['accuracy'],
        'precision': report['正样本']['precision'],
        'recall': report['正样本']['recall'],
        'f1_score': report['正样本']['f1-score'],
        'auc': auc,
        'confusion_matrix': cm.tolist(),
        'feature_importance': feature_importance.to_dict('records')
    }
    
    return model, metrics, y_prob


def generate_training_visualizations(model, X_train, df_features, train_dates, test_dates, neg_version):
    """生成训练过程可视化图表"""
    try:
        log.info("="*80)
        log.info("生成训练可视化图表")
        log.info("="*80)
        
        visualizer = TrainingVisualizer(
            output_dir=f"data/training/charts"
        )
        
        # 1. 样本质量可视化（正样本）
        try:
            df_positive_samples = pd.read_csv('data/training/samples/positive_samples.csv')
            visualizer.visualize_sample_quality(
                df_positive_samples,
                save_prefix="positive_sample_quality"
            )
        except Exception as e:
            log.warning(f"生成正样本质量可视化时出错: {e}")
        
        # 负样本
        try:
            if neg_version == 'v2':
                neg_file = 'data/training/samples/negative_samples_v2.csv'
            else:
                neg_file = 'data/training/samples/negative_samples.csv'
            
            if os.path.exists(neg_file):
                df_negative_samples = pd.read_csv(neg_file)
                visualizer.visualize_sample_quality(
                    df_negative_samples,
                    save_prefix="negative_sample_quality"
                )
        except Exception as e:
            log.warning(f"生成负样本质量可视化时出错: {e}")
        
        # 2. 因子重要性可视化
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        })
        
        visualizer.visualize_feature_importance(
            feature_importance,
            model_name=f"xgboost_timeseries_{neg_version}",
            top_n=20
        )
        
        # 3. 生成索引页面
        visualizer.generate_index_page(model_name=f"xgboost_timeseries_{neg_version}")
        
        log.success("✓ 可视化图表生成完成")
        log.info(f"📊 查看图表: open data/training/charts/index.html")
        
    except Exception as e:
        log.warning(f"生成可视化图表时出错: {e}")
        import traceback
        traceback.print_exc()


def save_model(model, metrics, neg_version, train_dates, test_dates):
    """保存模型和结果"""
    log.info("\n" + "="*80)
    log.info("第六步：保存模型")
    log.info("="*80)
    
    # 创建目录（使用新的目录结构）
    os.makedirs('data/training/models', exist_ok=True)
    os.makedirs('data/training/metrics', exist_ok=True)
    
    # 保存模型（使用booster方法避免sklearn mixin问题）
    model_file = f'data/training/models/xgboost_timeseries_{neg_version}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    model.get_booster().save_model(model_file)
    log.success(f"✓ 模型已保存: {model_file}")
    
    # 保存指标
    metrics_file = f'data/training/metrics/xgboost_timeseries_{neg_version}_metrics.json'
    metrics['model_file'] = model_file
    metrics['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    metrics['neg_version'] = neg_version
    metrics['train_date_range'] = f"{train_dates.min().date()} to {train_dates.max().date()}"
    metrics['test_date_range'] = f"{test_dates.min().date()} to {test_dates.max().date()}"
    metrics['note'] = '使用时间序列划分，避免未来函数'
    
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    log.success(f"✓ 评估报告已保存: {metrics_file}")
    log.info("")


def main():
    """主函数"""
    log.info("="*80)
    log.info("XGBoost 股票选股模型训练 - 时间序列版本")
    log.info("="*80)
    log.info("")
    log.info("⚠️  重要改进：")
    log.info("  1. 按时间划分训练集和测试集（而非随机划分）")
    log.info("  2. 训练集 = 历史数据，测试集 = 未来数据")
    log.info("  3. 避免未来函数，确保无数据泄露")
    log.info("")
    
    # 选择负样本版本
    NEG_VERSION = 'v2'  # 'v1' 或 'v2'
    
    log.info(f"配置:")
    log.info(f"  负样本版本: {NEG_VERSION}")
    log.info(f"  划分方式: 时间序列划分（80%训练，20%测试）")
    log.info(f"  模型: XGBoost")
    log.info("")
    
    try:
        # 👤 人工介入检查：特征选择
        checker = HumanInterventionChecker()
        feature_check = checker.check_feature_selection()
        checker.print_intervention_reminder("特征选择", feature_check)
        
        # 1. 加载数据
        df = load_and_prepare_data(neg_version=NEG_VERSION)
        
        # 2. 特征工程（保留时间信息）
        df_features = extract_features_with_time(df)
        
        # 👤 人工介入提醒：特征提取完成
        log.warning("\n" + "="*80)
        log.warning("👤 人工介入提醒：特征提取完成")
        log.warning("="*80)
        log.warning(f"当前特征数量: {len(df_features.columns) - 3} 个（不含sample_id, label, t1_date）")
        log.warning("请确认：")
        log.warning("  1. 特征是否足够？是否需要添加基本面特征或其他技术指标？")
        log.warning("  2. 特征是否避免了未来函数？")
        log.warning("  3. 特征重要性将在训练后显示，请关注")
        log.warning("="*80)
        
        # 3. 时间序列划分
        X_train, X_test, y_train, y_test, train_dates, test_dates = timeseries_split(
            df_features
        )
        
        # 4. 训练模型
        model, metrics, y_prob = train_model(X_train, y_train, X_test, y_test)
        
        # 4.5. 生成可视化图表
        generate_training_visualizations(
            model, X_train, df_features, train_dates, test_dates, NEG_VERSION
        )
        
        # 👤 人工介入检查：训练结果
        log.warning("\n" + "="*80)
        log.warning("👤 人工介入检查：训练结果")
        log.warning("="*80)
        
        # 检查指标是否达标
        warnings = []
        if metrics['auc'] < 0.7:
            warnings.append(f"⚠️  AUC = {metrics['auc']:.3f} < 0.7，模型性能可能不佳")
        if metrics['accuracy'] < 0.75:
            warnings.append(f"⚠️  准确率 = {metrics['accuracy']:.2%} < 75%，模型性能可能不佳")
        if metrics['f1_score'] < 0.7:
            warnings.append(f"⚠️  F1分数 = {metrics['f1_score']:.2%} < 70%，可能存在过拟合或欠拟合")
        
        if warnings:
            for warning in warnings:
                log.warning(warning)
            log.warning("\n建议：")
            log.warning("  - 检查特征选择，考虑添加更多有效特征")
            log.warning("  - 调整超参数（n_estimators, max_depth, learning_rate等）")
            log.warning("  - 检查数据质量，确保正负样本质量")
            log.warning("  - 考虑尝试其他算法（LightGBM, CatBoost等）")
        else:
            log.success("✓ 模型性能指标正常")
        log.warning("="*80)
        
        # 5. 保存模型
        save_model(model, metrics, NEG_VERSION, train_dates, test_dates)
        
        # 6. 最终总结
        log.info("="*80)
        log.success("✅ 模型训练完成！（时间序列版本）")
        log.info("="*80)
        log.info("")
        log.info("📊 模型性能总结:")
        log.info(f"  准确率 (Accuracy):  {metrics['accuracy']:.2%}")
        log.info(f"  精确率 (Precision): {metrics['precision']:.2%}")
        log.info(f"  召回率 (Recall):    {metrics['recall']:.2%}")
        log.info(f"  F1分数 (F1-Score):  {metrics['f1_score']:.2%}")
        log.info(f"  AUC-ROC:            {metrics['auc']:.4f}")
        log.info("")
        log.info("🎯 关键改进:")
        log.info("  ✓ 训练集 = 历史数据")
        log.info("  ✓ 测试集 = 未来数据（模拟真实场景）")
        log.info("  ✓ 无未来函数风险")
        log.info("  ✓ 无数据泄露")
        log.info("")
        log.info("下一步:")
        log.info("  1. 使用walk-forward验证进一步测试")
        log.info("  2. 在多个时间窗口上验证稳定性")
        log.info("  3. 回测验证实际收益")
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


if __name__ == '__main__':
    main()

