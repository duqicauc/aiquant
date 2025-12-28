"""
正样本数据质量核查工具

功能：
1. 数据完整性检查
2. 数据一致性验证
3. 涨幅计算验证
4. 可视化分析
5. 异常样本检测
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from datetime import datetime
from src.utils.logger import log
from src.visualization.training_visualizer import TrainingVisualizer


class SampleQualityChecker:
    """数据质量核查器"""
    
    def __init__(self, samples_file: str, features_file: str = None):
        """
        初始化
        
        Args:
            samples_file: 正样本文件路径
            features_file: 特征数据文件路径（可选）
        """
        self.samples_file = Path(samples_file)
        self.features_file = Path(features_file) if features_file else None
        
        self.df_samples = None
        self.df_features = None
        
        self._load_data()
    
    def _load_data(self):
        """加载数据"""
        log.info("="*80)
        log.info("加载数据")
        log.info("="*80)
        
        # 加载正样本
        if not self.samples_file.exists():
            log.error(f"正样本文件不存在: {self.samples_file}")
            return
        
        self.df_samples = pd.read_csv(self.samples_file)
        log.success(f"✓ 正样本加载成功: {len(self.df_samples)} 条")
        
        # 加载特征数据
        if self.features_file and self.features_file.exists():
            self.df_features = pd.read_csv(self.features_file)
            log.success(f"✓ 特征数据加载成功: {len(self.df_features)} 条")
        elif self.features_file:
            log.warning(f"特征数据文件不存在: {self.features_file}")
    
    def check_all(self):
        """执行所有检查"""
        if self.df_samples is None:
            log.error("数据未加载，无法检查")
            return
        
        log.info("\n" + "="*80)
        log.info("开始数据质量核查")
        log.info("="*80)
        
        # 1. 基础统计
        self.check_basic_stats()
        
        # 2. 数据完整性
        self.check_completeness()
        
        # 3. 数据一致性
        self.check_consistency()
        
        # 4. 涨幅验证
        self.check_returns()
        
        # 5. 异常检测
        self.check_anomalies()
        
        # 6. 去重检查
        self.check_duplicates()
        
        # 7. 日期检查
        self.check_dates()
        
        # 8. 生成可视化图表
        self.generate_visualizations()
        
        # 生成总结
        self.generate_summary()
    
    def check_basic_stats(self):
        """基础统计"""
        log.info("\n" + "="*80)
        log.info("【1】基础统计")
        log.info("="*80)
        
        df = self.df_samples
        
        log.info(f"样本总数: {len(df)}")
        log.info(f"股票数量: {df['ts_code'].nunique()}")
        log.info(f"股票名称数: {df['name'].nunique()}")
        
        # 涨幅统计
        log.info(f"\n涨幅统计:")
        log.info(f"  总涨幅 - 平均: {df['total_return'].mean():.2f}%")
        log.info(f"  总涨幅 - 中位数: {df['total_return'].median():.2f}%")
        log.info(f"  总涨幅 - 最小: {df['total_return'].min():.2f}%")
        log.info(f"  总涨幅 - 最大: {df['total_return'].max():.2f}%")
        
        log.info(f"\n  最高涨幅 - 平均: {df['max_return'].mean():.2f}%")
        log.info(f"  最高涨幅 - 中位数: {df['max_return'].median():.2f}%")
        log.info(f"  最高涨幅 - 最小: {df['max_return'].min():.2f}%")
        log.info(f"  最高涨幅 - 最大: {df['max_return'].max():.2f}%")
        
        # 时间统计
        if 't1_date' in df.columns:
            df['t1_date'] = pd.to_datetime(df['t1_date'])
            log.info(f"\nT1日期范围:")
            log.info(f"  最早: {df['t1_date'].min()}")
            log.info(f"  最晚: {df['t1_date'].max()}")
    
    def check_completeness(self):
        """数据完整性检查"""
        log.info("\n" + "="*80)
        log.info("【2】数据完整性检查")
        log.info("="*80)
        
        df = self.df_samples
        
        # 检查必需字段
        required_fields = ['ts_code', 'name', 't1_date', 'total_return', 'max_return']
        missing_fields = [f for f in required_fields if f not in df.columns]
        
        if missing_fields:
            log.error(f"✗ 缺少必需字段: {missing_fields}")
        else:
            log.success("✓ 所有必需字段都存在")
        
        # 检查空值
        null_counts = df.isnull().sum()
        null_fields = null_counts[null_counts > 0]
        
        if len(null_fields) > 0:
            log.warning(f"发现空值:")
            for field, count in null_fields.items():
                log.warning(f"  {field}: {count} 个空值 ({count/len(df)*100:.2f}%)")
        else:
            log.success("✓ 没有空值")
    
    def check_consistency(self):
        """数据一致性检查"""
        log.info("\n" + "="*80)
        log.info("【3】数据一致性检查")
        log.info("="*80)
        
        df = self.df_samples
        
        # 检查总涨幅和最高涨幅的关系
        invalid = df[df['total_return'] > df['max_return']]
        if len(invalid) > 0:
            log.error(f"✗ 发现 {len(invalid)} 个样本的总涨幅 > 最高涨幅（不合理）")
            print(invalid[['ts_code', 'name', 'total_return', 'max_return']])
        else:
            log.success("✓ 总涨幅和最高涨幅关系正确")
        
        # 检查涨幅范围
        min_total = df['total_return'].min()
        min_max = df['max_return'].min()
        
        if min_total < 50:
            log.warning(f"⚠️  发现总涨幅 < 50%的样本 (最小: {min_total:.2f}%)")
        else:
            log.success("✓ 所有样本总涨幅 >= 50%")
        
        if min_max < 70:
            log.warning(f"⚠️  发现最高涨幅 < 70%的样本 (最小: {min_max:.2f}%)")
        else:
            log.success("✓ 所有样本最高涨幅 >= 70%")
    
    def check_returns(self):
        """涨幅计算验证"""
        log.info("\n" + "="*80)
        log.info("【4】涨幅计算验证")
        log.info("="*80)
        
        df = self.df_samples
        
        # 验证涨幅计算（如果有开盘价和收盘价）
        if 'week1_open' in df.columns and 'week3_close' in df.columns:
            # 重新计算总涨幅
            df['calculated_return'] = (df['week3_close'] - df['week1_open']) / df['week1_open'] * 100
            
            # 比较
            diff = abs(df['total_return'] - df['calculated_return'])
            max_diff = diff.max()
            
            if max_diff > 0.1:  # 允许0.1%的误差
                problematic = df[diff > 0.1]
                log.warning(f"⚠️  发现 {len(problematic)} 个样本涨幅计算可能有误差")
                print(problematic[['ts_code', 'name', 'total_return', 'calculated_return']].head())
            else:
                log.success("✓ 涨幅计算正确")
        else:
            log.info("没有足够的价格数据进行涨幅验证")
    
    def check_anomalies(self):
        """异常检测"""
        log.info("\n" + "="*80)
        log.info("【5】异常值检测")
        log.info("="*80)
        
        df = self.df_samples
        
        # 使用IQR方法检测异常涨幅
        Q1 = df['total_return'].quantile(0.25)
        Q3 = df['total_return'].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df['total_return'] < lower_bound) | (df['total_return'] > upper_bound)]
        
        if len(outliers) > 0:
            log.info(f"发现 {len(outliers)} 个总涨幅异常值（IQR方法）:")
            for _, row in outliers.iterrows():
                log.info(f"  {row['ts_code']} {row['name']}: {row['total_return']:.2f}%")
        else:
            log.success("✓ 没有检测到明显的涨幅异常值")
        
        # 检测极端值（超过200%）
        extreme = df[df['total_return'] > 200]
        if len(extreme) > 0:
            log.warning(f"⚠️  发现 {len(extreme)} 个极端涨幅样本（>200%）:")
            print(extreme[['ts_code', 'name', 'total_return', 'max_return']])
        else:
            log.info("没有极端涨幅样本")
    
    def check_duplicates(self):
        """去重检查"""
        log.info("\n" + "="*80)
        log.info("【6】重复数据检查")
        log.info("="*80)
        
        df = self.df_samples
        
        # 检查完全重复的记录
        duplicates = df[df.duplicated(keep=False)]
        if len(duplicates) > 0:
            log.warning(f"⚠️  发现 {len(duplicates)} 条完全重复的记录")
            print(duplicates)
        else:
            log.success("✓ 没有完全重复的记录")
        
        # 检查同一股票多个T1日期
        stock_counts = df['ts_code'].value_counts()
        multi_samples = stock_counts[stock_counts > 1]
        
        if len(multi_samples) > 0:
            log.info(f"发现 {len(multi_samples)} 只股票有多个样本:")
            for ts_code, count in multi_samples.items():
                name = df[df['ts_code'] == ts_code]['name'].iloc[0]
                log.info(f"  {ts_code} {name}: {count} 个样本")
            
            log.info("\n⚠️  注意: 按规则应该每只股票只保留最早的样本")
        else:
            log.success("✓ 每只股票只有一个样本（符合去重规则）")
    
    def check_dates(self):
        """日期检查"""
        log.info("\n" + "="*80)
        log.info("【7】日期合理性检查")
        log.info("="*80)
        
        df = self.df_samples
        
        if 't1_date' not in df.columns:
            log.warning("没有T1日期数据")
            return
        
        df['t1_date'] = pd.to_datetime(df['t1_date'])
        
        # 检查未来日期
        today = pd.Timestamp.now()
        future_dates = df[df['t1_date'] > today]
        
        if len(future_dates) > 0:
            log.error(f"✗ 发现 {len(future_dates)} 个未来日期（错误）")
            print(future_dates[['ts_code', 'name', 't1_date']])
        else:
            log.success("✓ 没有未来日期")
        
        # 检查过于久远的日期
        very_old = df[df['t1_date'] < pd.Timestamp('2000-01-01')]
        if len(very_old) > 0:
            log.warning(f"⚠️  发现 {len(very_old)} 个2000年之前的样本")
        
        # 检查日期分布
        df['year'] = df['t1_date'].dt.year
        year_counts = df['year'].value_counts().sort_index()
        
        log.info(f"\nT1日期年份分布:")
        for year, count in year_counts.items():
            log.info(f"  {year}年: {count} 个样本")
    
    def generate_summary(self):
        """生成检查总结"""
        log.info("\n" + "="*80)
        log.info("数据质量核查总结")
        log.info("="*80)
        
        df = self.df_samples
        
        # 计算质量得分
        issues = []
        
        # 1. 完整性检查
        if df.isnull().sum().sum() > 0:
            issues.append("存在空值")
        
        # 2. 一致性检查
        if len(df[df['total_return'] > df['max_return']]) > 0:
            issues.append("涨幅逻辑错误")
        
        if df['total_return'].min() < 50 or df['max_return'].min() < 70:
            issues.append("不满足涨幅条件")
        
        # 3. 重复检查
        if len(df[df.duplicated()]) > 0:
            issues.append("存在重复记录")
        
        # 4. 日期检查
        if 't1_date' in df.columns:
            df['t1_date'] = pd.to_datetime(df['t1_date'])
            if len(df[df['t1_date'] > pd.Timestamp.now()]) > 0:
                issues.append("存在未来日期")
        
        # 总结
        if len(issues) == 0:
            log.success("\n✅ 数据质量良好！未发现重大问题")
            quality_score = 100
        else:
            log.warning(f"\n⚠️  发现 {len(issues)} 个问题:")
            for issue in issues:
                log.warning(f"  - {issue}")
            quality_score = max(0, 100 - len(issues) * 15)
        
        log.info(f"\n数据质量评分: {quality_score}/100")
        
        if quality_score >= 85:
            log.success("评级: 优秀 ⭐⭐⭐⭐⭐")
        elif quality_score >= 70:
            log.info("评级: 良好 ⭐⭐⭐⭐")
        elif quality_score >= 60:
            log.warning("评级: 中等 ⭐⭐⭐")
        else:
            log.error("评级: 需要改进 ⭐⭐")
    
    def generate_visualizations(self):
        """生成样本质量可视化图表"""
        if self.df_samples is None:
            return
        
        try:
            log.info("\n" + "="*80)
            log.info("生成样本质量可视化图表")
            log.info("="*80)
            
            # 确定输出目录
            output_dir = PROJECT_ROOT / 'data' / 'training' / 'charts'
            visualizer = TrainingVisualizer(output_dir=str(output_dir))
            
            # 生成可视化
            visualizer.visualize_sample_quality(
                self.df_samples,
                save_prefix="sample_quality_check"
            )
            
            # 生成索引页面
            visualizer.generate_index_page(model_name="sample_quality_check")
            
            log.success(f"✓ 可视化图表已生成到: {output_dir}")
            log.info(f"📊 查看图表: open {output_dir}/index.html")
            
        except Exception as e:
            log.warning(f"生成可视化图表时出错: {e}")
            import traceback
            traceback.print_exc()


def main():
    """主函数"""
    log.info("="*80)
    log.info("正样本数据质量核查工具")
    log.info("="*80)
    
    # 文件路径（使用新的目录结构）
    samples_file = PROJECT_ROOT / 'data' / 'training' / 'samples' / 'positive_samples.csv'
    features_file = PROJECT_ROOT / 'data' / 'training' / 'features' / 'feature_data_34d.csv'
    
    # 检查文件是否存在
    if not samples_file.exists():
        log.error(f"正样本文件不存在: {samples_file}")
        log.info("请先运行 scripts/prepare_positive_samples.py 生成数据")
        return
    
    # 创建检查器
    checker = SampleQualityChecker(samples_file, features_file)
    
    # 执行检查
    checker.check_all()
    
    log.info("\n" + "="*80)
    log.info("核查完成！")
    log.info("="*80)


if __name__ == '__main__':
    main()
