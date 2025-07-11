# Starship数据集文件结构分析综合报告

生成时间: 2025-07-11 09:17:48

## 执行摘要

- **发现的starship_data目录数**: 3
- **总文件数**: 165
- **数据匹配成功率**: 2.08%
- **完整样本数**: 1
- **问题严重程度**: 严重

## 目录结构分析

### 1-2000/starship_data

**发现的子目录**: airfoils, extracted_params, results, param_visualizations, pointclouds
**文件数量统计**:
  - airfoils: 16
  - extracted_params: 13
  - results: 13
  - param_visualizations: 6
  - pointclouds: 21
**文件扩展名统计**:
  - .ply: 21
  - .json: 19
  - .dat: 16
  - .csv: 7
  - .png: 6

### 1-10000/starship_data

**发现的子目录**: airfoils, extracted_params, results, param_visualizations, pointclouds
**文件数量统计**:
  - airfoils: 5
  - extracted_params: 6
  - results: 4
  - param_visualizations: 5
  - pointclouds: 9
**文件扩展名统计**:
  - .ply: 9
  - .json: 8
  - .dat: 5
  - .png: 5
  - .csv: 2

### 1-100/starship_data

**发现的子目录**: airfoils, extracted_params, results, param_visualizations, pointclouds
**文件数量统计**:
  - airfoils: 13
  - extracted_params: 10
  - results: 13
  - param_visualizations: 11
  - pointclouds: 20
**文件扩展名统计**:
  - .ply: 20
  - .json: 17
  - .dat: 13
  - .png: 11
  - .csv: 6

## 文件类型分析

**总体文件扩展名分布**:
  - .ply: 50 (30.3%)
  - .json: 44 (26.7%)
  - .dat: 34 (20.6%)
  - .png: 22 (13.3%)
  - .csv: 15 (9.1%)

**按文件类别分布**:
  - 数据文件: 93 (56.4%)
  - 图像文件: 22 (13.3%)
  - 点云文件: 50 (30.3%)

## 发现的问题和建议

### 问题 1 🔴 [HIGH]
**类型**: FILE_STRUCTURE
**描述**: 文件匹配成功率极低，需要检查文件命名规则和目录结构
**建议行动**: 1. 检查实际文件命名模式
2. 更新文件路径匹配逻辑
3. 验证目录结构完整性

## 技术细节

### 文件匹配逻辑分析
- 发现的唯一ID数量: 48
- 完整匹配样本: 1
- 部分匹配样本: 27
- 仅点云文件样本: 1

### 数据处理结果分析
- 点云文件数: 50
- 完整样本数: 0
- 不完整样本数: 50
- 处理成功率: 0.00%
- 处理时间: 0.01秒

## 下一步行动计划

### 紧急行动（成功率极低）
1. **立即检查文件命名规则** - 验证实际文件命名模式
2. **更新路径匹配逻辑** - 修正文件查找算法
3. **验证目录结构** - 确认所有必要的子目录存在
4. **数据完整性检查** - 验证数据复制是否完整

## 文件命名模式分析

### 1-2000/starship_data
**airfoils目录的命名模式**:
  - airfoil_N
**extracted_params目录的命名模式**:
  - params_N
**results目录的命名模式**:
  - merged_N
  - aero_N
**param_visualizations目录的命名模式**:
  - viz_N
**pointclouds目录的命名模式**:
  - sample_N

### 1-10000/starship_data
**airfoils目录的命名模式**:
  - airfoil_N
**extracted_params目录的命名模式**:
  - params_N
**results目录的命名模式**:
  - merged_N
  - aero_N
**param_visualizations目录的命名模式**:
  - viz_N
**pointclouds目录的命名模式**:
  - sample_N

### 1-100/starship_data
**airfoils目录的命名模式**:
  - airfoil_N
**extracted_params目录的命名模式**:
  - params_N
**results目录的命名模式**:
  - merged_N
  - aero_N
**param_visualizations目录的命名模式**:
  - viz_N
**pointclouds目录的命名模式**:
  - sample_N
