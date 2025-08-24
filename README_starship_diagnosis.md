# Starship数据集文件结构诊断工具

该工具集用于诊断和解决Starship数据集文件结构问题，特别是解决"扫描到10,000个点云文件但所有样本都报告缺失其他类型文件"的问题。

## 问题背景

- 数据处理脚本能发现10,000个点云文件
- 但所有样本都报告缺失airfoil、params、visualization、aero_data文件
- 成功率为0%，表明文件路径或命名规则不匹配

## 工具组件

### 1. 主诊断程序 (`main_diagnosis.py`)
一键式完整诊断工具，整合了所有功能：
```bash
# 使用现有数据集
python main_diagnosis.py /path/to/dataset

# 使用演示数据集
python main_diagnosis.py --demo

# 指定输出目录
python main_diagnosis.py /path/to/dataset --output-dir my_diagnosis
```

### 2. 文件结构诊断器 (`diagnose_file_structure.py`)
核心诊断工具，扫描实际文件结构：
```bash
python diagnose_file_structure.py /path/to/dataset
python diagnose_file_structure.py /path/to/dataset --output report.json
```

**功能：**
- 扫描所有starship_data目录
- 分析文件命名模式
- 统计文件类型和数量
- 识别缺失文件
- 生成详细诊断报告

### 3. 数据处理模拟器 (`data_processing_simulation.py`)
模拟当前数据处理过程，展示问题：
```bash
python data_processing_simulation.py /path/to/dataset
python data_processing_simulation.py /path/to/dataset --output results.json
```

**功能：**
- 模拟现有数据处理逻辑
- 展示文件匹配失败情况
- 统计成功率和缺失文件类型
- 生成处理结果报告

### 4. 文件结构分析器 (`file_structure_analyzer.py`)
生成可视化分析报告：
```bash
python file_structure_analyzer.py diagnostic_report.json
python file_structure_analyzer.py diagnostic_report.json --output-dir analysis
```

**功能：**
- 生成文件分布可视化图表
- 分析文件扩展名分布
- 创建匹配成功率分析
- 生成综合文字报告

### 5. 测试数据集创建器 (`create_test_dataset.py`)
创建模拟数据集用于测试：
```bash
python create_test_dataset.py --output test_dataset --samples 100
```

## 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 运行演示
```bash
# 创建演示数据集并运行完整诊断
python main_diagnosis.py --demo
```

### 3. 诊断实际数据集
```bash
# 替换为您的实际数据集路径
python main_diagnosis.py F:/essay_gt/daset
```

### 4. 查看结果
诊断完成后，在输出目录中查看：
- `diagnostic_report.json` - 详细诊断报告
- `processing_results.json` - 数据处理结果
- `analysis/` - 可视化分析图表和综合报告
- `solution_script.py` - 自动生成的解决方案脚本

## 期望的数据结构

```
F:/essay_gt/daset/
├── 1-100/starship_data/
│   ├── airfoils/
│   ├── extracted_params/
│   ├── param_visualizations/
│   ├── pointclouds/
│   └── results/
│       ├── aero_analysis/
│       └── merged_data/
├── 1-2000/starship_data/
│   └── ...
└── 1-10000/starship_data/
    └── ...
```

## 支持的文件类型

- **点云文件**: `.ply`, `.pcd`, `.xyz`, `.pts`
- **翼型文件**: `.dat`, `.txt`, `.csv`
- **参数文件**: `.json`, `.csv`, `.txt`
- **可视化文件**: `.png`, `.jpg`, `.svg`, `.pdf`
- **气动数据**: `.json`, `.csv`, `.parquet`

## 常见问题解决

### 问题1: 文件命名不匹配
**症状**: 成功率很低，大量文件报告缺失
**解决方案**: 
1. 运行诊断工具查看实际命名模式
2. 使用生成的解决方案脚本
3. 更新文件匹配逻辑

### 问题2: 目录结构不完整
**症状**: 某些子目录缺失
**解决方案**:
1. 检查数据复制是否完整
2. 创建缺失的目录结构
3. 验证数据完整性

### 问题3: 文件路径错误
**症状**: 路径构建错误
**解决方案**:
1. 检查基础路径配置
2. 验证目录层级结构
3. 更新路径构建逻辑

## 输出文件说明

### 诊断报告 (diagnostic_report.json)
```json
{
  "scan_summary": {
    "total_starship_directories": 3,
    "total_files_scanned": 10000,
    "scan_time": "2024-01-01 12:00:00"
  },
  "directory_structure": {...},
  "matching_analysis": {
    "total_unique_ids": 10000,
    "complete_samples": 0,
    "partial_samples": 2000,
    "pointcloud_only": 8000,
    "success_rate": 0.0
  },
  "recommendations": [...]
}
```

### 处理结果 (processing_results.json)
```json
{
  "statistics": {
    "total_pointclouds_found": 10000,
    "complete_samples": 0,
    "incomplete_samples": 10000,
    "success_rate": 0.0
  },
  "samples": [...]
}
```

## 性能优化

对于大型数据集（10,000+样本）：
1. 使用多进程扫描
2. 分批处理样本
3. 缓存文件路径信息
4. 使用高效的文件匹配算法

## 贡献指南

1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 创建 Pull Request

## 许可证

本项目采用 MIT 许可证。

## 联系信息

如有问题或建议，请创建 Issue 或联系项目维护者。

---

**注意**: 该工具集专门设计用于解决Starship数据集的文件结构问题，可根据具体需求进行定制和扩展。