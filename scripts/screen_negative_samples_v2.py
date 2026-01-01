"""
负样本筛选脚本 V2 - 只筛选负样本，不提取特征

输出：
- data/training/samples/negative_samples_v2.csv
"""
import sys
import os
import warnings
import pandas as pd
import json
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
warnings.filterwarnings('ignore', category=FutureWarning)

from src.data.data_manager import DataManager
from src.models.screening.negative_sample_screener_v2 import NegativeSampleScreenerV2
from src.utils.logger import log


def main():
    """主函数"""
    log.info("="*80)
    log.info("负样本筛选 V2 - 同周期其他股票法")
    log.info("="*80)
    
    # 配置参数
    SAMPLES_PER_POSITIVE = 2  # 每个正样本对应的负样本数量
    RANDOM_SEED = 42
    
    POSITIVE_SAMPLES_FILE = 'data/training/samples/positive_samples.csv'
    OUTPUT_NEGATIVE_SAMPLES = 'data/training/samples/negative_samples_v2.csv'
    
    log.info(f"\n当前设置：")
    log.info(f"  方法: 同周期其他股票法")
    log.info(f"  正样本文件: {POSITIVE_SAMPLES_FILE}")
    log.info(f"  每正样本对应负样本数: {SAMPLES_PER_POSITIVE}")
    log.info(f"  随机种子: {RANDOM_SEED}")
    log.info("")
    
    # 1. 加载正样本数据
    log.info("="*80)
    log.info("第一步：加载正样本数据")
    log.info("="*80)
    
    try:
        df_positive_samples = pd.read_csv(POSITIVE_SAMPLES_FILE)
        log.success(f"✓ 正样本加载成功: {len(df_positive_samples)} 个")
        
        # 验证正样本数据
        invalid_count = (df_positive_samples['days_since_list'] < 180).sum()
        if invalid_count > 0:
            log.error(f"正样本中有 {invalid_count} 条不符合条件（days_since_list < 180）")
            log.error("请先修正正样本数据")
            return
        log.success(f"✓ 正样本数据验证通过（均满足上市≥180天）")
        
    except Exception as e:
        log.error(f"✗ 加载正样本数据失败: {e}")
        log.error("请先运行正样本筛选脚本生成正样本数据")
        return
    
    # 2. 初始化
    log.info("\n" + "="*80)
    log.info("第二步：初始化筛选器 V2")
    log.info("="*80)
    
    dm = DataManager()
    screener = NegativeSampleScreenerV2(dm)
    log.success("✓ 筛选器 V2 初始化完成")
    
    # 3. 筛选负样本
    log.info("\n" + "="*80)
    log.info("第三步：筛选负样本（同周期其他股票法）")
    log.info("="*80)
    
    df_negative_samples = screener.screen_negative_samples(
        positive_samples_df=df_positive_samples,
        samples_per_positive=SAMPLES_PER_POSITIVE,
        random_seed=RANDOM_SEED
    )
    
    if df_negative_samples.empty:
        log.error("✗ 未找到负样本")
        return
    
    # 4. 保存结果
    log.info("\n" + "="*80)
    log.info("第四步：保存负样本列表")
    log.info("="*80)
    
    # 确保目录存在
    os.makedirs(os.path.dirname(OUTPUT_NEGATIVE_SAMPLES), exist_ok=True)
    
    df_negative_samples.to_csv(OUTPUT_NEGATIVE_SAMPLES, index=False)
    log.success(f"✓ 负样本列表已保存: {OUTPUT_NEGATIVE_SAMPLES}")
    
    # 5. 显示统计
    log.info("\n" + "="*80)
    log.info("负样本筛选完成 - 统计信息")
    log.info("="*80)
    log.info(f"  正样本数: {len(df_positive_samples)}")
    log.info(f"  负样本数: {len(df_negative_samples)}")
    log.info(f"  正负比例: 1:{len(df_negative_samples)/len(df_positive_samples):.1f}")
    log.info("")
    
    # 显示样本预览
    log.info("负样本预览（前10条）:")
    print(df_negative_samples.head(10))
    
    log.info("\n" + "="*80)
    log.success("✅ 负样本筛选完成！")
    log.info("="*80)
    log.info("")
    log.info("下一步：运行 reextract_negative_features.py 提取特征")
    log.info("")


if __name__ == '__main__':
    main()


