#!/usr/bin/env python3
"""
自动生成的数据处理修正脚本

基于诊断结果，该脚本提供了修正数据处理问题的具体实现。
诊断成功率: 2.08%
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
        file_patterns = {
            'airfoil': [
                # 添加更多可能的命名模式
                starship_data_dir / 'airfoils' / f'{sample_id}.dat',
                starship_data_dir / 'airfoils' / f'airfoil_{sample_id}.dat',
                starship_data_dir / 'airfoils' / f'sample_{sample_id}.dat',
                starship_data_dir / 'airfoils' / f'{sample_id:04d}.dat',
                starship_data_dir / 'airfoils' / f'airfoil_{sample_id:04d}.dat',
            ],
            'params': [
                starship_data_dir / 'extracted_params' / f'{sample_id}.json',
                starship_data_dir / 'extracted_params' / f'params_{sample_id}.json',
                starship_data_dir / 'extracted_params' / f'{sample_id}.csv',
                starship_data_dir / 'extracted_params' / f'{sample_id:04d}.json',
                starship_data_dir / 'extracted_params' / f'params_{sample_id:04d}.json',
            ],
            'visualization': [
                starship_data_dir / 'param_visualizations' / f'{sample_id}.png',
                starship_data_dir / 'param_visualizations' / f'viz_{sample_id}.png',
                starship_data_dir / 'param_visualizations' / f'{sample_id}.jpg',
                starship_data_dir / 'param_visualizations' / f'{sample_id:04d}.png',
                starship_data_dir / 'param_visualizations' / f'viz_{sample_id:04d}.png',
            ],
            'aero_data': [
                starship_data_dir / 'results' / 'aero_analysis' / f'{sample_id}.json',
                starship_data_dir / 'results' / 'aero_analysis' / f'aero_{sample_id}.json',
                starship_data_dir / 'results' / 'merged_data' / f'merged_{sample_id}.json',
                starship_data_dir / 'results' / 'aero_analysis' / f'{sample_id:04d}.json',
                starship_data_dir / 'results' / 'aero_analysis' / f'aero_{sample_id:04d}.json',
            ]
        }
        
        found_files = {}
        
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
            r'sample_(\d{4,})',      # sample_0001格式
            r'sample_(\d+)',           # sample_1格式
            r'starship_(\d+)',         # starship_1格式
            r'pc_(\d+)',               # pc_1格式
            r'(\d{4,})',             # 4位或更多数字
            r'(\d+)',                  # 任意数字
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
        
        print(f"发现 {len(pointcloud_files)} 个点云文件")
        
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
        print(f"  总样本数: {len(pointcloud_files)}")
        print(f"  完整样本数: {complete_samples}")
        print(f"  成功率: {success_rate:.2%}")
        
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
