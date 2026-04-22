# v2.5.0优化版断点续传功能说明

## 概述

所有涉及数据处理或获取Tushare数据的脚本都已实现断点续传功能，确保在中断后可以继续执行，避免重复处理。

## 已实现断点续传的脚本

### 1. 特征工程脚本

#### `scripts/add_advanced_factors_optimized.py`
- **功能**: 为样本数据添加优化后的高级技术因子
- **断点文件**: 
  - 正样本: `data/training/processed/.checkpoint_pos_optimized.csv`
  - 负样本: `data/training/features/.checkpoint_neg_optimized.csv`
- **断点机制**: 
  - 按样本ID记录已处理样本
  - 每批处理100个样本后保存checkpoint
  - 支持从任意中断点继续
- **使用方法**: 
  ```bash
  python scripts/add_advanced_factors_optimized.py
  # 如果中断，重新运行会自动从checkpoint继续
  ```

### 2. 样本准备脚本

#### `scripts/prepare_positive_samples_checkpoint.py`
- **功能**: 扫描和准备正样本数据
- **断点文件**: `data/training/samples/.checkpoint_positive.csv`
- **断点机制**: 
  - 按股票批次记录已处理股票
  - 每处理100只股票保存一次checkpoint
  - 支持断点续传
- **使用方法**: 
  ```bash
  python scripts/prepare_positive_samples_checkpoint.py
  ```

#### `scripts/prepare_negative_samples_checkpoint.py`
- **功能**: 准备负样本数据（支持市值分层抽样）
- **断点文件**: 
  - 样本: `data/training/samples/.checkpoint_negative_samples.csv`
  - 特征: `data/training/features/.checkpoint_negative_features.csv`
- **断点机制**: 
  - 按T1日期批次记录已处理日期
  - 每10个T1日期保存一次checkpoint
  - 特征提取也支持断点续传
- **使用方法**: 
  ```bash
  python scripts/prepare_negative_samples_checkpoint.py
  ```

#### `scripts/prepare_hard_negative_samples.py` ✅ 新增
- **功能**: 生成硬负样本（伪突破样本）
- **断点文件**: `data/training/samples/.checkpoint_hard_negative.csv`
- **断点机制**: 
  - 按股票批次记录已处理股票
  - 每批处理50只股票后保存checkpoint
  - 支持从任意中断点继续
- **使用方法**: 
  ```bash
  python scripts/prepare_hard_negative_samples.py
  # 如果中断，重新运行会自动从checkpoint继续
  ```

### 3. 训练脚本

#### `scripts/train_v250_model.py`
- **功能**: 训练v2.5.0模型
- **断点机制**: 
  - 训练脚本本身不涉及大量数据获取，主要使用已准备好的特征数据
  - 如果特征数据准备中断，可以重新运行特征工程脚本（会自动续传）

## 断点续传工作原理

### 通用机制

1. **检查断点文件**: 脚本启动时检查是否存在checkpoint文件
2. **加载已处理数据**: 如果存在，加载已处理的样本/股票列表
3. **筛选待处理项**: 从全部待处理项中排除已处理项
4. **批量处理**: 按批次处理剩余项
5. **保存checkpoint**: 每批处理后立即保存checkpoint
6. **完成清理**: 处理完成后自动清理checkpoint文件

### 示例流程

```python
# 1. 检查checkpoint
if checkpoint_file.exists():
    df_checkpoint = pd.read_csv(checkpoint_file)
    processed_ids = set(df_checkpoint['id'].unique())
    # 加载已处理数据
    existing_results.append(df_checkpoint)

# 2. 筛选待处理项
pending_items = all_items[~all_items['id'].isin(processed_ids)]

# 3. 批量处理
for batch in batches:
    # 处理批次
    batch_results = process_batch(batch)
    # 保存checkpoint
    save_checkpoint(batch_results)

# 4. 完成清理
if all_done:
    cleanup_checkpoint()
```

## 注意事项

### 1. Checkpoint文件位置
- 所有checkpoint文件都保存在对应的数据目录下
- 文件名以 `.checkpoint_` 开头，便于识别
- 处理完成后会自动清理

### 2. 中断恢复
- 如果脚本中断，直接重新运行即可
- 脚本会自动检测checkpoint并从断点继续
- 无需手动干预

### 3. 强制重新开始
如果需要强制重新开始（忽略checkpoint）：
```bash
# 删除checkpoint文件
rm data/training/**/.checkpoint_*.csv

# 然后重新运行脚本
```

### 4. 网络中断处理
- Tushare API调用失败时会记录警告但继续处理
- 已处理的样本会保存在checkpoint中
- 重新运行时会跳过已处理的样本

### 5. 数据一致性
- Checkpoint文件与最终输出文件格式一致
- 可以随时查看checkpoint文件了解进度
- 最终输出完成后checkpoint会被清理

## 完整执行流程

使用完整优化脚本执行时，所有步骤都支持断点续传：

```bash
bash scripts/run_v250_optimization_full.sh
```

执行流程：
1. ✅ 检查正样本（支持断点续传）
2. ✅ 准备负样本（支持断点续传）
3. ✅ 生成硬负样本（支持断点续传）✅ 新增
4. ✅ 添加优化特征（支持断点续传）
5. ✅ 训练模型

## 验证断点续传功能

### 测试方法

1. **手动中断测试**:
   ```bash
   # 运行脚本
   python scripts/add_advanced_factors_optimized.py
   # 在运行过程中按 Ctrl+C 中断
   # 重新运行，应该从断点继续
   python scripts/add_advanced_factors_optimized.py
   ```

2. **检查checkpoint文件**:
   ```bash
   # 查看checkpoint文件
   ls -lh data/training/**/.checkpoint_*.csv
   # 检查内容
   head data/training/processed/.checkpoint_pos_optimized.csv
   ```

3. **验证进度**:
   - 脚本会输出已处理/待处理的样本数量
   - 可以通过checkpoint文件行数估算进度

## 总结

所有涉及Tushare数据获取和大量数据处理的脚本都已实现断点续传功能，确保：

- ✅ 网络中断后可继续
- ✅ 程序异常退出后可恢复
- ✅ 避免重复处理已完成的样本
- ✅ 节省API调用次数
- ✅ 提高执行效率

用户可以在任何时候安全地中断和恢复数据处理流程。
