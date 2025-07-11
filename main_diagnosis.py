#!/usr/bin/env python3
"""
Starship数据集问题诊断和解决方案主程序

该脚本整合了所有诊断工具，提供一键式问题诊断和解决方案。
"""

import os
import sys
import argparse
import logging
from pathlib import Path


def setup_logging(log_level: str = "INFO"):
    """设置日志"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def run_complete_diagnosis(dataset_path: str, output_dir: str = "diagnosis_output"):
    """运行完整诊断流程"""
    print("="*60)
    print("Starship数据集完整诊断流程")
    print("="*60)
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    try:
        # 1. 运行文件结构诊断
        print("\n步骤 1: 运行文件结构诊断...")
        from diagnose_file_structure import FileStructureDiagnostic
        
        diagnostic = FileStructureDiagnostic(dataset_path)
        diagnostic_report = diagnostic.run_diagnostic()
        
        # 移动报告到输出目录
        import shutil
        diagnostic_output = output_path / "diagnostic_report.json"
        shutil.move(diagnostic_report, diagnostic_output)
        print(f"诊断报告保存到: {diagnostic_output}")
        
        # 2. 运行数据处理模拟
        print("\n步骤 2: 运行数据处理模拟...")
        from data_processing_simulation import StarshipDataProcessor
        
        processor = StarshipDataProcessor(dataset_path)
        processor.process_samples()
        processor.print_statistics()
        
        processing_output = output_path / "processing_results.json"
        processor.save_results(str(processing_output))
        print(f"处理结果保存到: {processing_output}")
        
        # 3. 生成综合分析报告
        print("\n步骤 3: 生成综合分析报告...")
        try:
            from file_structure_analyzer import FileStructureAnalyzer
            analyzer = FileStructureAnalyzer()
            analysis_output = str(output_path / "analysis")
            analyzer.generate_comprehensive_report(
                str(diagnostic_output),
                str(processing_output),
                analysis_output
            )
        except ImportError as e:
            print(f"可视化库不可用 ({e})，使用简化版分析器...")
            from simple_analyzer import SimpleFileStructureAnalyzer
            analyzer = SimpleFileStructureAnalyzer()
            analysis_output = str(output_path / "analysis")
            analyzer.generate_comprehensive_report(
                str(diagnostic_output),
                str(processing_output),
                analysis_output
            )
        
        # 4. 生成解决方案建议
        print("\n步骤 4: 生成解决方案建议...")
        generate_solution_script(diagnostic_output, output_path)
        
        print(f"\n{'='*60}")
        print("诊断完成！")
        print(f"所有结果保存在: {output_path}")
        print("包含文件:")
        print(f"  - 诊断报告: diagnostic_report.json")
        print(f"  - 处理结果: processing_results.json") 
        print(f"  - 综合分析: analysis/")
        print(f"  - 解决方案: solution_script.py")
        print(f"{'='*60}")
        
        return str(output_path)
        
    except Exception as e:
        logging.error(f"诊断过程中发生错误: {e}")
        raise


def generate_solution_script(diagnostic_report_path: Path, output_dir: Path):
    """生成解决方案脚本"""
    import json
    
    # 读取诊断报告
    with open(diagnostic_report_path, 'r', encoding='utf-8') as f:
        report = json.load(f)
    
    matching_analysis = report.get('matching_analysis', {})
    success_rate = matching_analysis.get('success_rate', 0)
    
    script_content = f'''#!/usr/bin/env python3
"""
自动生成的数据处理修正脚本

基于诊断结果，该脚本提供了修正数据处理问题的具体实现。
诊断成功率: {success_rate:.2%}
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Optional


class FixedStarshipDataProcessor:
    """修正后的Starship数据处理器"""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.samples = []
    
    def find_improved_file_patterns(self, sample_id: str, pointcloud_path: str) -> Dict[str, Optional[str]]:
        """改进的文件查找模式"""
        pointcloud_dir = Path(pointcloud_path).parent
        starship_data_dir = pointcloud_dir.parent
        
        # 基于诊断结果的改进匹配模式
        # 转换sample_id为整数以便格式化
        try:
            sample_id_int = int(sample_id)
            sample_id_padded = f"{{sample_id_int:04d}}"
        except ValueError:
            sample_id_padded = sample_id
        
        file_patterns = {{
            'airfoil': [
                # 添加更多可能的命名模式
                starship_data_dir / 'airfoils' / f'{{sample_id}}.dat',
                starship_data_dir / 'airfoils' / f'airfoil_{{sample_id}}.dat',
                starship_data_dir / 'airfoils' / f'sample_{{sample_id}}.dat',
                starship_data_dir / 'airfoils' / f'{{sample_id_padded}}.dat',
                starship_data_dir / 'airfoils' / f'airfoil_{{sample_id_padded}}.dat',
            ],
            'params': [
                starship_data_dir / 'extracted_params' / f'{{sample_id}}.json',
                starship_data_dir / 'extracted_params' / f'params_{{sample_id}}.json',
                starship_data_dir / 'extracted_params' / f'{{sample_id}}.csv',
                starship_data_dir / 'extracted_params' / f'{{sample_id_padded}}.json',
                starship_data_dir / 'extracted_params' / f'params_{{sample_id_padded}}.json',
            ],
            'visualization': [
                starship_data_dir / 'param_visualizations' / f'{{sample_id}}.png',
                starship_data_dir / 'param_visualizations' / f'viz_{{sample_id}}.png',
                starship_data_dir / 'param_visualizations' / f'{{sample_id}}.jpg',
                starship_data_dir / 'param_visualizations' / f'{{sample_id_padded}}.png',
                starship_data_dir / 'param_visualizations' / f'viz_{{sample_id_padded}}.png',
            ],
            'aero_data': [
                starship_data_dir / 'results' / 'aero_analysis' / f'{{sample_id}}.json',
                starship_data_dir / 'results' / 'aero_analysis' / f'aero_{{sample_id}}.json',
                starship_data_dir / 'results' / 'merged_data' / f'merged_{{sample_id}}.json',
                starship_data_dir / 'results' / 'aero_analysis' / f'{{sample_id_padded}}.json',
                starship_data_dir / 'results' / 'aero_analysis' / f'aero_{{sample_id_padded}}.json',
            ]
        }}
        
        found_files = {{}}
        
        for file_type, patterns in file_patterns.items():
            for pattern in patterns:
                # 格式化路径
                try:
                    formatted_path = str(pattern).format(sample_id=sample_id)
                    if Path(formatted_path).exists():
                        found_files[file_type] = formatted_path
                        break
                except (ValueError, KeyError):
                    continue
        
        return found_files
    
    def extract_improved_sample_id(self, pointcloud_path: str) -> str:
        """改进的样本ID提取"""
        filename = os.path.basename(pointcloud_path)
        
        # 基于诊断结果的改进ID提取模式
        patterns = [
            r'sample_(\\d{{4,}})',      # sample_0001格式
            r'sample_(\\d+)',           # sample_1格式
            r'starship_(\\d+)',         # starship_1格式
            r'pc_(\\d+)',               # pc_1格式
            r'(\\d{{4,}})',             # 4位或更多数字
            r'(\\d+)',                  # 任意数字
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                return match.group(1)
        
        # 如果都没匹配，返回文件名（不含扩展名）
        return os.path.splitext(filename)[0]
    
    def process_with_fixes(self):
        """使用修正后的逻辑处理数据"""
        print("使用修正后的逻辑处理数据...")
        
        # 扫描点云文件
        pointcloud_files = []
        for root, dirs, files in os.walk(self.dataset_path):
            if 'pointclouds' in os.path.basename(root):
                for file in files:
                    if file.endswith(('.ply', '.pcd', '.xyz', '.pts')):
                        pointcloud_files.append(os.path.join(root, file))
        
        print(f"发现 {{len(pointcloud_files)}} 个点云文件")
        
        # 处理每个样本
        complete_samples = 0
        for pointcloud_path in pointcloud_files:
            sample_id = self.extract_improved_sample_id(pointcloud_path)
            found_files = self.find_improved_file_patterns(sample_id, pointcloud_path)
            
            # 检查样本完整性
            required_types = ['airfoil', 'params', 'visualization', 'aero_data']
            if all(file_type in found_files for file_type in required_types):
                complete_samples += 1
        
        success_rate = complete_samples / len(pointcloud_files) if pointcloud_files else 0
        
        print(f"修正后的结果:")
        print(f"  总样本数: {{len(pointcloud_files)}}")
        print(f"  完整样本数: {{complete_samples}}")
        print(f"  成功率: {{success_rate:.2%}}")
        
        return success_rate


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='修正后的数据处理器')
    parser.add_argument('dataset_path', help='数据集路径')
    
    args = parser.parse_args()
    
    processor = FixedStarshipDataProcessor(args.dataset_path)
    processor.process_with_fixes()


if __name__ == "__main__":
    main()
'''
    
    solution_file = output_dir / "solution_script.py"
    with open(solution_file, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    print(f"解决方案脚本已生成: {solution_file}")


def create_demo():
    """创建演示数据集"""
    print("创建演示数据集...")
    from create_test_dataset import create_test_dataset
    
    demo_path = "demo_dataset"
    create_test_dataset(demo_path, num_samples=50)
    return demo_path


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Starship数据集问题诊断和解决方案')
    parser.add_argument('dataset_path', nargs='?', help='数据集路径')
    parser.add_argument('--output-dir', '-o', default='diagnosis_output', help='输出目录')
    parser.add_argument('--log-level', default='INFO', help='日志级别')
    parser.add_argument('--demo', action='store_true', help='创建并使用演示数据集')
    
    args = parser.parse_args()
    
    setup_logging(args.log_level)
    
    # 处理演示模式
    if args.demo:
        dataset_path = create_demo()
        print(f"使用演示数据集: {dataset_path}")
    else:
        if not args.dataset_path:
            print("错误: 需要提供数据集路径或使用 --demo 选项")
            sys.exit(1)
        dataset_path = args.dataset_path
    
    # 检查数据集路径
    if not os.path.exists(dataset_path):
        print(f"错误: 数据集路径不存在: {dataset_path}")
        sys.exit(1)
    
    # 运行完整诊断
    try:
        output_dir = run_complete_diagnosis(dataset_path, args.output_dir)
        print(f"\\n诊断完成！查看结果: {output_dir}")
        
    except Exception as e:
        print(f"诊断失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()