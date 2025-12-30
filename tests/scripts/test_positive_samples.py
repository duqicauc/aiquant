"""
快速测试脚本 - 只测试几只股票

用于验证逻辑是否正确，避免长时间等待
"""
import sys
from pathlib import Path
import warnings

# 过滤 pandas FutureWarning（来自 tushare 库内部）
warnings.filterwarnings('ignore', category=FutureWarning, module='tushare')

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_manager import DataManager
from src.strategy.screening.positive_sample_screener import PositiveSampleScreener
from src.utils.logger import log


def test_single_stock():
    """测试单只股票"""
    
    log.info("="*80)
    log.info("快速测试：贵州茅台（600519.SH）")
    log.info("="*80)
    
    # 初始化
    dm = DataManager(source='tushare')
    screener = PositiveSampleScreener(dm)
    
    # 测试数据
    ts_code = '600519.SH'
    name = '贵州茅台'
    start_date = '20220101'
    end_date = '20241231'
    
    log.info(f"\n获取 {name} 的数据: {start_date} - {end_date}")
    
    try:
        # 获取日线数据
        df_daily = dm.get_daily_data(ts_code, start_date, end_date, adjust='qfq')
        log.info(f"✓ 获取日线数据: {len(df_daily)} 条")
        
        # 转换为周线
        df_weekly = screener._convert_to_weekly(df_daily)
        log.info(f"✓ 转换为周线: {len(df_weekly)} 周")
        print("\n周线数据预览：")
        print(df_weekly.head(10))
        
        # 筛选正样本（使用内部方法测试）
        import pandas as pd
        list_date = pd.to_datetime('20010801')  # 茅台上市日期
        
        samples = screener._screen_single_stock(
            ts_code, name, list_date, start_date, end_date
        )
        
        if samples:
            log.success(f"\n✓ 找到 {len(samples)} 个正样本:")
            for sample in samples:
                log.info(f"  T1日期: {sample['t1_date']}")
                log.info(f"  总涨幅: {sample['total_return']:.2f}%")
                log.info(f"  最高涨幅: {sample['max_return']:.2f}%")
            
            # 测试特征提取
            log.info("\n测试特征提取...")
            import pandas as pd
            df_sample = pd.DataFrame(samples)
            
            df_features = screener.extract_features(df_sample, lookback_days=34)
            
            if not df_features.empty:
                log.success(f"✓ 提取特征数据: {len(df_features)} 条")
                print("\n特征数据预览：")
                print(df_features.head(10))
                print("\n字段列表：")
                print(df_features.columns.tolist())
            else:
                log.warning("未提取到特征数据")
        else:
            log.info(f"\n{name} 在此期间未找到符合条件的样本")
            log.info("这是正常的，不是所有股票都会有三连阳涨50%+的情况")
        
        log.success("\n✅ 测试完成！逻辑运行正常")
        
    except Exception as e:
        log.error(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()


def test_few_stocks():
    """测试几只股票"""
    
    log.info("="*80)
    log.info("批量测试：5只股票")
    log.info("="*80)
    
    # 初始化
    dm = DataManager(source='tushare')
    screener = PositiveSampleScreener(dm)
    
    # 测试这些股票
    test_stocks = [
        ('600519.SH', '贵州茅台'),
        ('000858.SZ', '五粮液'),
        ('600036.SH', '招商银行'),
        ('000001.SZ', '平安银行'),
        ('300750.SZ', '宁德时代')
    ]
    
    start_date = '20220101'
    end_date = '20241231'
    
    all_samples = []
    
    for ts_code, name in test_stocks:
        log.info(f"\n处理: {ts_code} {name}")
        
        try:
            # 获取数据
            df_daily = dm.get_daily_data(ts_code, start_date, end_date, adjust='qfq')
            
            # 筛选
            import pandas as pd
            list_date = pd.to_datetime('20000101')  # 简化处理
            
            samples = screener._screen_single_stock(
                ts_code, name, list_date, start_date, end_date
            )
            
            if samples:
                log.success(f"✓ {name}: 找到 {len(samples)} 个样本")
                all_samples.extend(samples)
            else:
                log.info(f"  {name}: 无符合条件的样本")
                
        except Exception as e:
            log.error(f"✗ {name}: {e}")
            continue
    
    # 汇总
    log.info("\n" + "="*80)
    log.info(f"测试完成！共找到 {len(all_samples)} 个正样本")
    
    if all_samples:
        import pandas as pd
        df_samples = pd.DataFrame(all_samples)
        print("\n样本汇总：")
        print(df_samples)
        
        log.info("\n可以运行完整脚本了:")
        log.info("  python scripts/prepare_positive_samples.py")


if __name__ == '__main__':
    print("\n选择测试模式:")
    print("1. 测试单只股票（贵州茅台）")
    print("2. 测试5只股票")
    
    choice = input("\n请输入选项 (1/2): ").strip()
    
    if choice == '1':
        test_single_stock()
    elif choice == '2':
        test_few_stocks()
    else:
        print("无效选项")

