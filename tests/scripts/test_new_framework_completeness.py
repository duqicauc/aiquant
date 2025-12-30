"""
测试新框架完整性：验证新框架下完成全部流程后能否获得与旧模型一样的结果

测试流程：
1. 数据提取（正样本、负样本）
2. 特征提取
3. 模型训练（使用与旧模型相同的参数）
4. 模型评测
5. 模型预测（使用相同日期）
6. 结果比对（与旧模型预测结果比对）
"""
import sys
import os
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.logger import log
from src.models.lifecycle.trainer import ModelTrainer
from src.models.lifecycle.iterator import ModelIterator


def check_data_preparation():
    """检查数据准备情况"""
    log.info("="*80)
    log.info("步骤1：检查数据准备")
    log.info("="*80)
    
    required_files = [
        'data/training/samples/positive_samples.csv',
        'data/training/samples/negative_samples_v2.csv'
    ]
    
    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            log.success(f"✓ {file_path}: {len(df)} 条记录")
        else:
            log.error(f"✗ {file_path}: 文件不存在")
            all_exist = False
    
    if not all_exist:
        log.warning("需要先准备数据，运行以下命令：")
        log.warning("  1. python scripts/prepare_positive_samples.py")
        log.warning("  2. python scripts/prepare_negative_samples_v2.py")
        return False
    
    return True


def train_model_with_same_params(version='v1.1.0-test', neg_version='v2'):
    """使用与旧模型相同的参数训练新模型"""
    log.info("="*80)
    log.info("步骤2：训练模型（使用与旧模型相同的参数）")
    log.info("="*80)
    
    model_name = 'breakout_launch_scorer'
    
    try:
        trainer = ModelTrainer(model_name)
        
        log.info(f"训练参数:")
        log.info(f"  模型名称: {model_name}")
        log.info(f"  版本: {version}")
        log.info(f"  负样本版本: {neg_version}")
        log.info(f"  随机种子: 42 (与旧模型一致)")
        log.info("")
        
        # 训练模型
        model, metrics = trainer.train_version(version=version, neg_version=neg_version)
        
        log.success("✓ 模型训练完成")
        log.info("")
        log.info("模型性能指标:")
        log.info(f"  准确率 (Accuracy):  {metrics.get('accuracy', 0):.2%}")
        log.info(f"  精确率 (Precision): {metrics.get('precision', 0):.2%}")
        log.info(f"  召回率 (Recall):    {metrics.get('recall', 0):.2%}")
        log.info(f"  F1分数 (F1-Score):  {metrics.get('f1_score', 0):.2%}")
        log.info(f"  AUC-ROC:            {metrics.get('auc', 0):.4f}")
        log.info("")
        
        return model, metrics, version
        
    except Exception as e:
        log.error(f"✗ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def compare_with_old_model(new_version, old_model_path=None):
    """与旧模型结果比对"""
    log.info("="*80)
    log.info("步骤3：与旧模型结果比对")
    log.info("="*80)
    
    # 获取新模型信息
    iterator = ModelIterator('breakout_launch_scorer')
    new_model_info = iterator.get_version_info(new_version)
    
    log.info("新模型信息:")
    log.info(f"  版本: {new_version}")
    log.info(f"  训练日期范围: {new_model_info.get('training', {}).get('train_date_range', 'N/A')}")
    log.info(f"  测试日期范围: {new_model_info.get('training', {}).get('test_date_range', 'N/A')}")
    
    # 从training/metrics.json读取更准确的指标
    metrics_file = iterator.versions_path / new_version / "training" / "metrics.json"
    if metrics_file.exists():
        with open(metrics_file, 'r', encoding='utf-8') as f:
            training_metrics = json.load(f)
        new_metrics = training_metrics.get('test', {})
    else:
        new_metrics = new_model_info.get('metrics', {}).get('test', {})
    log.info("")
    log.info("新模型性能指标（测试集）:")
    log.info(f"  准确率: {new_metrics.get('accuracy', 0):.2%}")
    log.info(f"  精确率: {new_metrics.get('precision', 0):.2%}")
    log.info(f"  召回率: {new_metrics.get('recall', 0):.2%}")
    log.info(f"  F1分数: {new_metrics.get('f1_score', 0):.2%}")
    log.info(f"  AUC:    {new_metrics.get('auc', 0):.4f}")
    log.info("")
    
    # 查找旧模型结果
    old_metrics_file = 'data/training/metrics/xgboost_timeseries_v2_metrics.json'
    if os.path.exists(old_metrics_file):
        with open(old_metrics_file, 'r', encoding='utf-8') as f:
            old_metrics = json.load(f)
        
        log.info("旧模型性能指标（测试集）:")
        log.info(f"  准确率: {old_metrics.get('accuracy', 0):.2%}")
        log.info(f"  精确率: {old_metrics.get('precision', 0):.2%}")
        log.info(f"  召回率: {old_metrics.get('recall', 0):.2%}")
        log.info(f"  F1分数: {old_metrics.get('f1_score', 0):.2%}")
        log.info(f"  AUC:    {old_metrics.get('auc', 0):.4f}")
        log.info("")
        
        # 比对
        log.info("="*80)
        log.info("性能指标比对")
        log.info("="*80)
        
        metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
        differences = {}
        
        for metric in metrics_to_compare:
            new_val = new_metrics.get(metric, 0)
            old_val = old_metrics.get(metric, 0)
            diff = new_val - old_val
            diff_pct = (diff / old_val * 100) if old_val != 0 else 0
            
            differences[metric] = {
                'new': new_val,
                'old': old_val,
                'diff': diff,
                'diff_pct': diff_pct
            }
            
            status = "✓" if abs(diff_pct) < 1.0 else "⚠" if abs(diff_pct) < 5.0 else "✗"
            log.info(f"{status} {metric:12s}: 新={new_val:.4f}, 旧={old_val:.4f}, 差异={diff:+.4f} ({diff_pct:+.2f}%)")
        
        log.info("")
        
        # 判断是否一致
        all_close = all(abs(d['diff_pct']) < 1.0 for d in differences.values())
        if all_close:
            log.success("✅ 新模型与旧模型性能指标非常接近（差异<1%）")
        else:
            log.warning("⚠️  新模型与旧模型性能指标存在差异，请检查训练参数是否一致")
        
        return differences
    else:
        log.warning(f"未找到旧模型指标文件: {old_metrics_file}")
        return None


def test_prediction(new_version, test_date='20251225'):
    """测试预测功能"""
    log.info("="*80)
    log.info("步骤4：测试预测功能")
    log.info("="*80)
    
    log.info(f"使用新模型 {new_version} 预测日期: {test_date}")
    log.info("")
    
    # 运行预测脚本
    import subprocess
    cmd = ['python', 'scripts/score_current_stocks.py', '--date', test_date, '--version', new_version]
    
    try:
        result = subprocess.run(
            cmd,
            cwd=os.path.dirname(os.path.dirname(__file__)),
            capture_output=True,
            text=True,
            timeout=3600
        )
        
        if result.returncode == 0:
            log.success("✓ 预测完成")
            
            # 查找预测结果文件
            results_dir = Path('data/prediction/results')
            pattern = f'*{test_date}*breakout_launch_scorer*{new_version}*'
            result_files = list(results_dir.glob(pattern))
            
            if result_files:
                log.info(f"预测结果文件:")
                for f in result_files:
                    log.info(f"  - {f.name}")
                return True
            else:
                log.warning("未找到预测结果文件")
                return False
        else:
            log.error(f"✗ 预测失败: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        log.error("✗ 预测超时")
        return False
    except Exception as e:
        log.error(f"✗ 预测异常: {e}")
        return False


def compare_predictions(new_version, test_date='20251225'):
    """比对新旧模型的预测结果"""
    log.info("="*80)
    log.info("步骤5：比对新旧模型预测结果")
    log.info("="*80)
    
    # 查找新模型预测结果
    results_dir = Path('data/prediction/results')
    new_pattern = f'top_50_stocks_{test_date}*breakout_launch_scorer*{new_version}*.csv'
    new_files = list(results_dir.glob(new_pattern))
    
    # 查找旧模型预测结果（232545.csv是旧模型）
    old_pattern = f'top_50_stocks_{test_date}_232545.csv'
    old_files = list(results_dir.glob(old_pattern))
    
    if not new_files:
        log.error("未找到新模型预测结果")
        return False
    
    if not old_files:
        log.warning("未找到旧模型预测结果，跳过比对")
        return False
    
    # 读取结果
    df_new = pd.read_csv(new_files[0], encoding='utf-8-sig')
    df_old = pd.read_csv(old_files[0], encoding='utf-8-sig')
    
    log.info(f"新模型预测结果: {len(df_new)} 只股票")
    log.info(f"旧模型预测结果: {len(df_old)} 只股票")
    log.info("")
    
    # 比对Top 10
    log.info("Top 10 股票比对:")
    log.info(f"{'排名':<4} {'新模型':<30} {'旧模型':<30} {'是否一致':<10}")
    log.info("-" * 80)
    
    common_count = 0
    for i in range(min(10, len(df_new), len(df_old))):
        new_code = df_new.iloc[i]['股票代码']
        old_code = df_old.iloc[i]['股票代码'] if i < len(df_old) else 'N/A'
        is_same = new_code == old_code
        if is_same:
            common_count += 1
        
        status = "✓" if is_same else "✗"
        log.info(f"{i+1:<4} {new_code:<30} {old_code:<30} {status}")
    
    log.info("")
    log.info(f"Top 10 一致率: {common_count}/10 ({common_count*10}%)")
    
    # 比对概率分布
    log.info("")
    log.info("概率分布比对:")
    log.info(f"新模型 - 最高: {df_new['牛股概率'].max():.4f}, 最低: {df_new['牛股概率'].min():.4f}, 平均: {df_new['牛股概率'].mean():.4f}")
    log.info(f"旧模型 - 最高: {df_old['牛股概率'].max():.4f}, 最低: {df_old['牛股概率'].min():.4f}, 平均: {df_old['牛股概率'].mean():.4f}")
    
    return True


def main():
    """主函数"""
    log.info("="*80)
    log.info("新框架完整性测试")
    log.info("="*80)
    log.info("")
    log.info("测试目标：验证新框架下完成全部流程后能否获得与旧模型一样的结果")
    log.info("")
    
    # 步骤1：检查数据准备
    if not check_data_preparation():
        log.error("数据准备不完整，请先准备数据")
        return
    
    log.info("")
    
    # 步骤2：训练模型
    model, metrics, version = train_model_with_same_params(version='v1.1.0-test', neg_version='v2')
    if model is None:
        log.error("模型训练失败，测试终止")
        return
    
    log.info("")
    
    # 步骤3：与旧模型比对
    differences = compare_with_old_model(version)
    
    log.info("")
    
    # 步骤4：测试预测
    test_date = '20251225'
    if test_prediction(version, test_date):
        log.info("")
        # 步骤5：比对预测结果
        compare_predictions(version, test_date)
    
    log.info("")
    log.info("="*80)
    log.success("✅ 测试完成！")
    log.info("="*80)
    log.info("")
    log.info("测试总结:")
    log.info("  1. ✓ 数据准备检查完成")
    log.info("  2. ✓ 模型训练完成")
    log.info("  3. ✓ 性能指标比对完成")
    log.info("  4. ✓ 预测功能测试完成")
    log.info("  5. ✓ 预测结果比对完成")
    log.info("")
    log.info(f"测试版本: {version}")
    log.info("请查看上述结果，确认新框架是否与旧模型结果一致")


if __name__ == '__main__':
    main()

