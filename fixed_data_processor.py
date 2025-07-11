#!/usr/bin/env python3
"""
修正后的数据处理脚本

该脚本提供了修正数据处理问题的具体实现，改进了文件匹配逻辑。
"""

import os
import re
from pathlib import Path
from typing import Dict, Optional


class FixedStarshipDataProcessor:
    """修正后的Starship数据处理器"""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.samples = []
    
    def find_improved_file_patterns(self, sample_id: str, pointcloud_path: str) -> Dict[str, Optional[str]]:
        """改进的文件查找模式"""
        pointcloud_dir = Path(pointcloud_path).parent
        starship_data_dir = pointcloud_dir.parent
        
        # 准备不同格式的sample_id
        try:
            sample_id_int = int(sample_id)
            sample_id_padded = f"{sample_id_int:04d}"
        except ValueError:
            sample_id_padded = sample_id
        
        # 基于诊断结果的改进匹配模式
        file_patterns = {
            'airfoil': [
                starship_data_dir / 'airfoils' / f'{sample_id}.dat',
                starship_data_dir / 'airfoils' / f'airfoil_{sample_id}.dat',
                starship_data_dir / 'airfoils' / f'sample_{sample_id}.dat',
                starship_data_dir / 'airfoils' / f'{sample_id_padded}.dat',
                starship_data_dir / 'airfoils' / f'airfoil_{sample_id_padded}.dat',
            ],
            'params': [
                starship_data_dir / 'extracted_params' / f'{sample_id}.json',
                starship_data_dir / 'extracted_params' / f'params_{sample_id}.json',
                starship_data_dir / 'extracted_params' / f'{sample_id}.csv',
                starship_data_dir / 'extracted_params' / f'{sample_id_padded}.json',
                starship_data_dir / 'extracted_params' / f'params_{sample_id_padded}.json',
            ],
            'visualization': [
                starship_data_dir / 'param_visualizations' / f'{sample_id}.png',
                starship_data_dir / 'param_visualizations' / f'viz_{sample_id}.png',
                starship_data_dir / 'param_visualizations' / f'{sample_id}.jpg',
                starship_data_dir / 'param_visualizations' / f'{sample_id_padded}.png',
                starship_data_dir / 'param_visualizations' / f'viz_{sample_id_padded}.png',
            ],
            'aero_data': [
                starship_data_dir / 'results' / 'aero_analysis' / f'{sample_id}.json',
                starship_data_dir / 'results' / 'aero_analysis' / f'aero_{sample_id}.json',
                starship_data_dir / 'results' / 'merged_data' / f'merged_{sample_id}.json',
                starship_data_dir / 'results' / 'aero_analysis' / f'{sample_id_padded}.json',
                starship_data_dir / 'results' / 'aero_analysis' / f'aero_{sample_id_padded}.json',
                starship_data_dir / 'results' / 'merged_data' / f'merged_{sample_id_padded}.csv',
            ]
        }
        
        found_files = {}
        
        for file_type, patterns in file_patterns.items():
            for pattern in patterns:
                if pattern.exists():
                    found_files[file_type] = str(pattern)
                    break
        
        return found_files
    
    def extract_improved_sample_id(self, pointcloud_path: str) -> str:
        """改进的样本ID提取"""
        filename = os.path.basename(pointcloud_path)
        
        # 基于诊断结果的改进ID提取模式
        patterns = [
            r'sample_(\d{4,})',      # sample_0001格式
            r'sample_(\d+)',         # sample_1格式
            r'starship_(\d+)',       # starship_1格式
            r'pc_(\d+)',             # pc_1格式
            r'(\d{4,})',             # 4位或更多数字
            r'(\d+)',                # 任意数字
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
        partial_samples = 0
        samples_detail = []
        
        for pointcloud_path in pointcloud_files:
            sample_id = self.extract_improved_sample_id(pointcloud_path)
            found_files = self.find_improved_file_patterns(sample_id, pointcloud_path)
            
            # 检查样本完整性
            required_types = ['airfoil', 'params', 'visualization', 'aero_data']
            found_types = list(found_files.keys())
            
            sample_info = {
                'sample_id': sample_id,
                'pointcloud_path': pointcloud_path,
                'found_files': found_files,
                'missing_types': [t for t in required_types if t not in found_types],
                'is_complete': all(t in found_types for t in required_types)
            }
            
            samples_detail.append(sample_info)
            
            if sample_info['is_complete']:
                complete_samples += 1
            elif found_files:
                partial_samples += 1
        
        success_rate = complete_samples / len(pointcloud_files) if pointcloud_files else 0
        
        print(f"\n修正后的结果:")
        print(f"  总样本数: {len(pointcloud_files)}")
        print(f"  完整样本数: {complete_samples}")
        print(f"  部分样本数: {partial_samples}")
        print(f"  成功率: {success_rate:.2%}")
        
        # 显示一些样本的详细信息
        print(f"\n样本详细分析（前5个）:")
        for i, sample in enumerate(samples_detail[:5]):
            print(f"\n样本 {i+1} (ID: {sample['sample_id']}):")
            print(f"  完整性: {'✓' if sample['is_complete'] else '✗'}")
            print(f"  发现的文件类型: {list(sample['found_files'].keys())}")
            if sample['missing_types']:
                print(f"  缺失的文件类型: {sample['missing_types']}")
        
        # 统计缺失文件类型
        missing_stats = {}
        for sample in samples_detail:
            for missing_type in sample['missing_types']:
                missing_stats[missing_type] = missing_stats.get(missing_type, 0) + 1
        
        if missing_stats:
            print(f"\n缺失文件统计:")
            for file_type, count in missing_stats.items():
                percentage = count / len(pointcloud_files) * 100
                print(f"  {file_type}: {count} ({percentage:.1f}%)")
        
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