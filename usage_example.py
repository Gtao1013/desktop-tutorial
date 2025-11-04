#!/usr/bin/env python3
"""
Starship数据集诊断工具使用示例

该脚本演示了如何使用诊断工具来解决数据集文件结构问题。
"""

import os
import sys
from pathlib import Path


def example_usage():
    """演示诊断工具的使用方法"""
    
    print("="*60)
    print("Starship数据集诊断工具使用示例")
    print("="*60)
    
    print("\n1. 快速开始 - 使用演示数据集:")
    print("   python main_diagnosis.py --demo")
    
    print("\n2. 诊断实际数据集:")
    print("   python main_diagnosis.py F:/essay_gt/daset")
    
    print("\n3. 仅运行文件结构诊断:")
    print("   python diagnose_file_structure.py F:/essay_gt/daset")
    
    print("\n4. 仅运行数据处理模拟:")
    print("   python data_processing_simulation.py F:/essay_gt/daset")
    
    print("\n5. 使用修正后的数据处理器:")
    print("   python fixed_data_processor.py F:/essay_gt/daset")
    
    print("\n6. 生成分析报告:")
    print("   python simple_analyzer.py diagnostic_report.json")
    
    print("\n7. 创建测试数据集:")
    print("   python create_test_dataset.py --output test_data --samples 100")
    
    print("\n" + "="*60)
    print("工具输出说明:")
    print("- diagnostic_report.json: 详细诊断报告")
    print("- processing_results.json: 数据处理结果")
    print("- analysis/comprehensive_report.md: 综合分析报告")
    print("- solution_script.py: 自动生成的解决方案")
    print("="*60)
    
    print("\n问题解决流程:")
    print("1. 运行诊断工具识别问题")
    print("2. 查看诊断报告了解具体问题")
    print("3. 使用修正后的处理器验证改进")
    print("4. 根据建议调整数据处理流程")
    print("5. 重新验证数据处理成功率")


def demonstrate_improvement():
    """演示诊断工具的改进效果"""
    
    print("\n" + "="*60)
    print("诊断工具效果演示")
    print("="*60)
    
    # 创建临时演示数据集
    from create_test_dataset import create_test_dataset
    demo_path = "temp_demo_dataset"
    
    print(f"\n正在创建演示数据集: {demo_path}")
    create_test_dataset(demo_path, num_samples=20)
    
    try:
        # 运行原始数据处理模拟
        print("\n步骤 1: 运行原始数据处理模拟...")
        from data_processing_simulation import StarshipDataProcessor
        
        processor = StarshipDataProcessor(demo_path)
        processor.process_samples()
        original_success_rate = processor.processing_stats['success_rate']
        
        # 运行修正后的处理器
        print("\n步骤 2: 运行修正后的数据处理器...")
        from fixed_data_processor import FixedStarshipDataProcessor
        
        fixed_processor = FixedStarshipDataProcessor(demo_path)
        improved_success_rate = fixed_processor.process_with_fixes()
        
        # 比较结果
        print(f"\n" + "="*40)
        print("结果对比:")
        print(f"原始成功率: {original_success_rate:.2%}")
        print(f"修正后成功率: {improved_success_rate:.2%}")
        improvement = improved_success_rate - original_success_rate
        print(f"改进幅度: +{improvement:.2%}")
        print("="*40)
        
        if improvement > 0:
            print("✓ 诊断工具成功改进了数据处理效果！")
        else:
            print("! 需要进一步优化数据处理逻辑")
            
    finally:
        # 清理临时文件
        import shutil
        if os.path.exists(demo_path):
            shutil.rmtree(demo_path)
            print(f"\n已清理临时文件: {demo_path}")


def check_requirements():
    """检查运行环境"""
    
    print("\n检查运行环境:")
    
    # 检查Python版本
    if sys.version_info >= (3, 7):
        print("✓ Python版本满足要求")
    else:
        print("✗ 需要Python 3.7或更高版本")
    
    # 检查必要模块
    required_modules = ['pathlib', 'json', 'logging', 'argparse', 're']
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"✓ {module} 模块可用")
        except ImportError:
            missing_modules.append(module)
            print(f"✗ {module} 模块缺失")
    
    # 检查可选模块
    optional_modules = ['matplotlib', 'seaborn', 'pandas', 'numpy']
    for module in optional_modules:
        try:
            __import__(module)
            print(f"✓ {module} 模块可用（可选）")
        except ImportError:
            print(f"⚠ {module} 模块缺失（可选，用于可视化）")
    
    if missing_modules:
        print(f"\n请安装缺失的模块: pip install {' '.join(missing_modules)}")
        return False
    else:
        print("\n✓ 运行环境检查通过")
        return True


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Starship数据集诊断工具使用示例')
    parser.add_argument('--demo', action='store_true', help='运行效果演示')
    parser.add_argument('--check', action='store_true', help='检查运行环境')
    
    args = parser.parse_args()
    
    if args.check:
        check_requirements()
    elif args.demo:
        if check_requirements():
            demonstrate_improvement()
    else:
        example_usage()


if __name__ == "__main__":
    main()