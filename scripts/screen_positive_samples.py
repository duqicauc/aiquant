"""
正样本筛选脚本 - 只筛选正样本，不提取特征

筛选条件：
1. 周K三连阳：连续3周，每周的收盘价 > 每周的开盘价
2. 总涨幅 > 50%：(第3周收盘价 - 第1周开盘价) / 第1周开盘价 > 50%
3. 最高涨幅 > 70%：(3周内最高价 - 第1周开盘价) / 第1周开盘价 > 70%
4. 上市时间 ≥ 180天

过滤规则：
- ST: 剔除ST股票
- HALT: 剔除T1日期停牌的股票
- DELISTING: 剔除退市股票
- DELISTING_SORTING: 剔除退市整理期股票
- 北交所: 剔除北交所股票

输出：
- data/training/samples/positive_samples.csv
"""
import sys
from pathlib import Path
import warnings

warnings.filterwarnings('ignore', category=FutureWarning, module='tushare')

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_manager import DataManager
from src.models.screening.positive_sample_screener import PositiveSampleScreener
from src.utils.logger import log
from config.settings import settings
import pandas as pd
from datetime import datetime


def main():
    """主函数"""
    
    log.info("="*80)
    log.info("正样本筛选")
    log.info("="*80)
    
    # 定义文件路径
    samples_file = PROJECT_ROOT / 'data' / 'training' / 'samples' / 'positive_samples.csv'
    
    # 从配置文件读取日期范围
    START_DATE = settings.get('data.sample_preparation.start_date', '20000101')
    END_DATE = settings.get('data.sample_preparation.end_date', None)
    
    log.info(f"\n当前设置：")
    log.info(f"  筛选日期范围: {START_DATE} - {END_DATE or '今天'}")
    log.info(f"  输出文件: {samples_file}")
    log.info("")
    
    # 1. 初始化数据管理器
    log.info("\n[步骤1] 初始化数据管理器...")
    dm = DataManager(source='tushare')
    
    # 2. 初始化筛选器
    log.info("\n[步骤2] 初始化正样本筛选器...")
    screener = PositiveSampleScreener(dm)
    
    # 3. 执行筛选
    log.info("\n[步骤3] 执行正样本筛选...")
    log.info("="*80)
    
    try:
        df_samples = screener.screen_all_stocks(
            start_date=START_DATE,
            end_date=END_DATE
        )
        
        if df_samples.empty:
            log.error("未找到符合条件的正样本！请检查筛选条件或数据质量")
            return
        
        # 4. 保存正样本列表
        log.info("\n[步骤4] 保存正样本列表...")
        samples_file.parent.mkdir(parents=True, exist_ok=True)
        df_samples.to_csv(samples_file, index=False, encoding='utf-8-sig')
        log.success(f"✓ 正样本列表已保存: {samples_file}")
        
        # 5. 打印统计信息
        log.info("\n" + "="*80)
        log.info("正样本筛选完成 - 统计信息")
        log.info("="*80)
        log.info(f"样本总数: {len(df_samples)}")
        log.info(f"股票数量: {df_samples['ts_code'].nunique()}")
        log.info(f"平均总涨幅: {df_samples['total_return'].mean():.2f}%")
        log.info(f"平均最高涨幅: {df_samples['max_return'].mean():.2f}%")
        log.info(f"days_since_list 范围: {df_samples['days_since_list'].min()} - {df_samples['days_since_list'].max()}")
        
        # T1日期分布
        df_samples['year'] = df_samples['t1_date'].astype(str).str[:4].astype(int)
        log.info(f"\nT1日期年份分布:")
        year_counts = df_samples['year'].value_counts().sort_index()
        for year, count in year_counts.items():
            log.info(f"  {year}: {count}")
        
        # 显示样本预览
        log.info("\n前10个样本:")
        print(df_samples[['ts_code', 'name', 't1_date', 'total_return', 'max_return', 'days_since_list']].head(10))
        
        # 验证数据质量
        log.info("\n" + "="*80)
        log.info("数据质量验证")
        log.info("="*80)
        
        # 检查days_since_list是否都>=180
        invalid_count = (df_samples['days_since_list'] < 180).sum()
        if invalid_count > 0:
            log.error(f"⚠️ 发现 {invalid_count} 条样本不符合上市≥180天条件！")
        else:
            log.success(f"✓ 所有样本均符合上市≥180天条件")
        
        # 检查涨幅条件
        invalid_return = (df_samples['total_return'] < 50).sum()
        if invalid_return > 0:
            log.error(f"⚠️ 发现 {invalid_return} 条样本总涨幅<50%！")
        else:
            log.success(f"✓ 所有样本总涨幅≥50%")
        
        invalid_max = (df_samples['max_return'] < 70).sum()
        if invalid_max > 0:
            log.error(f"⚠️ 发现 {invalid_max} 条样本最高涨幅<70%！")
        else:
            log.success(f"✓ 所有样本最高涨幅≥70%")
        
        log.info("\n" + "="*80)
        log.success("✅ 正样本筛选完成！")
        log.info("="*80)
        log.info("")
        log.info("下一步：运行 reextract_positive_features.py 提取特征")
        log.info("")
        
    except Exception as e:
        log.error(f"正样本筛选失败: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == '__main__':
    main()


