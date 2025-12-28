"""
准备正样本数据的主脚本

运行步骤：
1. 筛选符合条件的正样本
2. 提取每个样本T1前34天的特征数据
3. 保存结果
"""
import sys
from pathlib import Path
import warnings

# 过滤 pandas FutureWarning（来自 tushare 库内部，不影响功能）
warnings.filterwarnings('ignore', category=FutureWarning, module='tushare')

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_manager import DataManager
from src.strategy.screening.positive_sample_screener import PositiveSampleScreener
from src.utils.logger import log
from config.settings import settings
import pandas as pd
from datetime import datetime


def main():
    """主函数"""
    
    log.info("="*80)
    log.info("正样本数据准备流程")
    log.info("="*80)
    
    # 1. 初始化数据管理器
    log.info("\n[步骤1] 初始化数据管理器...")
    dm = DataManager(source='tushare')
    
    # 2. 初始化筛选器
    log.info("\n[步骤2] 初始化正样本筛选器...")
    screener = PositiveSampleScreener(dm)
    
    # 3. 筛选正样本
    log.info("\n[步骤3] 开始筛选正样本（这可能需要较长时间）...")
    log.info("筛选条件：")
    log.info("  - 周K连续三周收阳线")
    log.info("  - 总涨幅超50%")
    log.info("  - 最高涨幅超70%")
    log.info("  - 剔除ST股票")
    log.info("  - 上市超过半年")
    log.info("")
    
    # 从配置文件读取日期范围
    START_DATE = settings.get('data.sample_preparation.start_date', '20000101')
    END_DATE = settings.get('data.sample_preparation.end_date', None)
    
    log.info(f"📅 数据范围配置：{START_DATE} - {END_DATE or '今天'}")
    log.info(f"💡 如需修改，请编辑 config/settings.yaml")
    
    try:
        df_samples = screener.screen_all_stocks(
            start_date=START_DATE,
            end_date=END_DATE
        )
        
        if df_samples.empty:
            log.error("未找到符合条件的正样本！请检查筛选条件或数据质量")
            return
        
        # 保存正样本列表
        samples_file = PROJECT_ROOT / 'data' / 'training' / 'samples' / 'positive_samples.csv'
        samples_file.parent.mkdir(parents=True, exist_ok=True)
        df_samples.to_csv(samples_file, index=False, encoding='utf-8-sig')
        log.success(f"✓ 正样本列表已保存: {samples_file}")
        
        # 打印统计信息
        log.info("\n" + "="*80)
        log.info("正样本统计")
        log.info("="*80)
        log.info(f"样本总数: {len(df_samples)}")
        log.info(f"股票数量: {df_samples['ts_code'].nunique()}")
        log.info(f"平均总涨幅: {df_samples['total_return'].mean():.2f}%")
        log.info(f"平均最高涨幅: {df_samples['max_return'].mean():.2f}%")
        log.info(f"\n前5个样本:")
        print(df_samples.head())
        
        # 4. 提取特征数据
        log.info("\n[步骤4] 提取特征数据（T1前34天）...")
        
        df_features = screener.extract_features(
            df_samples,
            lookback_days=34
        )
        
        if df_features.empty:
            log.error("特征提取失败！")
            return
        
        # 保存特征数据
        features_file = PROJECT_ROOT / 'data' / 'processed' / 'feature_data_34d.csv'
        df_features.to_csv(features_file, index=False, encoding='utf-8-sig')
        log.success(f"✓ 特征数据已保存: {features_file}")
        
        # 特征统计
        log.info("\n" + "="*80)
        log.info("特征数据统计")
        log.info("="*80)
        log.info(f"总记录数: {len(df_features)}")
        log.info(f"样本数: {df_features['sample_id'].nunique()}")
        log.info(f"每样本天数: {len(df_features) / df_features['sample_id'].nunique():.1f}")
        log.info(f"\n数据字段:")
        for col in df_features.columns:
            log.info(f"  - {col}")
        log.info(f"\n数据预览:")
        print(df_features.head(10))
        
        # 5. 生成统计报告
        log.info("\n[步骤5] 生成统计报告...")
        
        stats = {
            'generation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'date_range': f"{START_DATE} - {END_DATE or 'today'}",
            'total_samples': int(len(df_samples)),
            'unique_stocks': int(df_samples['ts_code'].nunique()),
            'avg_total_return': float(df_samples['total_return'].mean()),
            'avg_max_return': float(df_samples['max_return'].mean()),
            'feature_records': int(len(df_features)),
            'lookback_days': 34,
            'sample_files': {
                'positive_samples': str(samples_file),
                'feature_data': str(features_file)
            }
        }
        
        import json
        stats_file = PROJECT_ROOT / 'data' / 'processed' / 'sample_statistics.json'
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        log.success(f"✓ 统计报告已保存: {stats_file}")
        
        # 完成
        log.info("\n" + "="*80)
        log.success("✅ 正样本数据准备完成！")
        log.info("="*80)
        log.info("\n生成的文件:")
        log.info(f"  1. 正样本列表: {samples_file}")
        log.info(f"  2. 特征数据: {features_file}")
        log.info(f"  3. 统计报告: {stats_file}")
        log.info("\n下一步:")
        log.info("  - 查看数据质量")
        log.info("  - 准备负样本")
        log.info("  - 开始模型训练")
        
    except Exception as e:
        log.error(f"执行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

