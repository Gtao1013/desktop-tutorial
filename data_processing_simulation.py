#!/usr/bin/env python3
"""
Starship数据处理模拟脚本

该脚本模拟当前数据处理过程中遇到的问题：
- 能扫描到10,000个点云文件
- 但所有样本都报告缺失其他类型文件
- 成功率为0%

用于演示问题并测试诊断工具的效果。
"""

import os
import time
import logging
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class DataSample:
    """数据样本类"""
    sample_id: str
    pointcloud_path: str = None
    airfoil_path: str = None
    params_path: str = None
    visualization_path: str = None
    aero_data_path: str = None
    
    @property
    def is_complete(self) -> bool:
        """检查样本是否完整"""
        return all([
            self.pointcloud_path,
            self.airfoil_path,
            self.params_path,
            self.visualization_path,
            self.aero_data_path
        ])
    
    @property
    def missing_files(self) -> List[str]:
        """获取缺失的文件类型"""
        missing = []
        if not self.pointcloud_path:
            missing.append('pointcloud')
        if not self.airfoil_path:
            missing.append('airfoil')
        if not self.params_path:
            missing.append('params')
        if not self.visualization_path:
            missing.append('visualization')
        if not self.aero_data_path:
            missing.append('aero_data')
        return missing


class StarshipDataProcessor:
    """Starship数据处理器"""
    
    def __init__(self, dataset_path: str):
        """
        初始化数据处理器
        
        Args:
            dataset_path: 数据集基础路径
        """
        self.dataset_path = Path(dataset_path)
        self.samples = []
        self.processing_stats = {
            'total_pointclouds_found': 0,
            'complete_samples': 0,
            'incomplete_samples': 0,
            'success_rate': 0.0,
            'processing_time': 0.0
        }
        
        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def scan_pointclouds(self) -> List[str]:
        """扫描点云文件"""
        self.logger.info("开始扫描点云文件...")
        
        pointcloud_files = []
        
        # 查找所有starship_data/pointclouds目录
        for root, dirs, files in os.walk(self.dataset_path):
            if 'pointclouds' in os.path.basename(root):
                for file in files:
                    if file.endswith(('.ply', '.pcd', '.xyz', '.pts')):
                        pointcloud_files.append(os.path.join(root, file))
        
        self.processing_stats['total_pointclouds_found'] = len(pointcloud_files)
        self.logger.info(f"发现 {len(pointcloud_files)} 个点云文件")
        
        return pointcloud_files
    
    def extract_sample_id(self, pointcloud_path: str) -> str:
        """从点云文件路径中提取样本ID"""
        filename = os.path.basename(pointcloud_path)
        # 尝试多种ID提取模式
        import re
        
        patterns = [
            r'(\d+)',  # 纯数字
            r'sample_(\d+)',  # sample_数字
            r'starship_(\d+)',  # starship_数字
            r'pc_(\d+)',  # pc_数字
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                return match.group(1)
        
        # 如果没有找到模式，使用文件名（不含扩展名）
        return os.path.splitext(filename)[0]
    
    def find_matching_files(self, sample_id: str, pointcloud_path: str) -> DataSample:
        """为给定样本ID查找匹配的文件"""
        sample = DataSample(sample_id=sample_id, pointcloud_path=pointcloud_path)
        
        # 获取点云文件的基础目录（starship_data目录）
        pointcloud_dir = Path(pointcloud_path).parent
        starship_data_dir = pointcloud_dir.parent
        
        # 构建期望的文件路径（这些路径模式可能不正确，导致文件缺失）
        expected_paths = {
            'airfoil': [
                starship_data_dir / 'airfoils' / f'{sample_id}.dat',
                starship_data_dir / 'airfoils' / f'airfoil_{sample_id}.dat',
                starship_data_dir / 'airfoils' / f'sample_{sample_id}.dat',
            ],
            'params': [
                starship_data_dir / 'extracted_params' / f'{sample_id}.json',
                starship_data_dir / 'extracted_params' / f'params_{sample_id}.json',
                starship_data_dir / 'extracted_params' / f'sample_{sample_id}.csv',
            ],
            'visualization': [
                starship_data_dir / 'param_visualizations' / f'{sample_id}.png',
                starship_data_dir / 'param_visualizations' / f'viz_{sample_id}.png',
                starship_data_dir / 'param_visualizations' / f'sample_{sample_id}.jpg',
            ],
            'aero_data': [
                starship_data_dir / 'results' / 'aero_analysis' / f'{sample_id}.json',
                starship_data_dir / 'results' / 'aero_analysis' / f'aero_{sample_id}.csv',
                starship_data_dir / 'results' / 'merged_data' / f'merged_{sample_id}.json',
            ]
        }
        
        # 尝试找到匹配的文件
        for file_type, possible_paths in expected_paths.items():
            found = False
            for path in possible_paths:
                if path.exists():
                    setattr(sample, f'{file_type}_path', str(path))
                    found = True
                    break
            
            if not found:
                # 记录未找到的文件
                self.logger.debug(f"样本 {sample_id} 缺失 {file_type} 文件")
        
        return sample
    
    def process_samples(self):
        """处理所有样本"""
        start_time = time.time()
        self.logger.info("开始处理样本...")
        
        # 1. 扫描点云文件
        pointcloud_files = self.scan_pointclouds()
        
        if not pointcloud_files:
            self.logger.warning("未找到任何点云文件！")
            return
        
        # 2. 为每个点云文件查找匹配的其他文件
        for i, pointcloud_path in enumerate(pointcloud_files):
            if i % 1000 == 0:  # 每1000个样本输出一次进度
                self.logger.info(f"处理进度: {i}/{len(pointcloud_files)} ({i/len(pointcloud_files)*100:.1f}%)")
            
            sample_id = self.extract_sample_id(pointcloud_path)
            sample = self.find_matching_files(sample_id, pointcloud_path)
            self.samples.append(sample)
        
        # 3. 统计结果
        complete_samples = sum(1 for sample in self.samples if sample.is_complete)
        incomplete_samples = len(self.samples) - complete_samples
        
        self.processing_stats.update({
            'complete_samples': complete_samples,
            'incomplete_samples': incomplete_samples,
            'success_rate': complete_samples / len(self.samples) if self.samples else 0.0,
            'processing_time': time.time() - start_time
        })
        
        self.logger.info(f"处理完成！总用时: {self.processing_stats['processing_time']:.2f}秒")
    
    def print_statistics(self):
        """打印处理统计信息"""
        print("\n" + "="*50)
        print("Starship数据处理统计")
        print("="*50)
        
        stats = self.processing_stats
        print(f"发现的点云文件数: {stats['total_pointclouds_found']}")
        print(f"完整样本数: {stats['complete_samples']}")
        print(f"不完整样本数: {stats['incomplete_samples']}")
        print(f"成功率: {stats['success_rate']:.2%}")
        print(f"处理时间: {stats['processing_time']:.2f}秒")
        
        # 统计缺失文件类型
        if self.samples:
            missing_stats = defaultdict(int)
            for sample in self.samples:
                for missing_type in sample.missing_files:
                    missing_stats[missing_type] += 1
            
            print("\n缺失文件统计:")
            for file_type, count in missing_stats.items():
                percentage = count / len(self.samples) * 100
                print(f"  {file_type}: {count} ({percentage:.1f}%)")
        
        # 显示一些样本示例
        print("\n样本示例:")
        for i, sample in enumerate(self.samples[:5]):  # 只显示前5个
            print(f"\n样本 {i+1} (ID: {sample.sample_id}):")
            print(f"  完整性: {'✓' if sample.is_complete else '✗'}")
            print(f"  点云文件: {'✓' if sample.pointcloud_path else '✗'}")
            print(f"  翼型文件: {'✓' if sample.airfoil_path else '✗'}")
            print(f"  参数文件: {'✓' if sample.params_path else '✗'}")
            print(f"  可视化文件: {'✓' if sample.visualization_path else '✗'}")
            print(f"  气动数据: {'✓' if sample.aero_data_path else '✗'}")
            if sample.missing_files:
                print(f"  缺失类型: {', '.join(sample.missing_files)}")
        
        print("\n" + "="*50)
    
    def save_results(self, output_file: str = None):
        """保存处理结果"""
        if output_file is None:
            output_file = f'processing_results_{int(time.time())}.json'
        
        import json
        
        results = {
            'statistics': self.processing_stats,
            'samples': []
        }
        
        for sample in self.samples:
            results['samples'].append({
                'sample_id': sample.sample_id,
                'is_complete': sample.is_complete,
                'missing_files': sample.missing_files,
                'files': {
                    'pointcloud': sample.pointcloud_path,
                    'airfoil': sample.airfoil_path,
                    'params': sample.params_path,
                    'visualization': sample.visualization_path,
                    'aero_data': sample.aero_data_path
                }
            })
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"处理结果已保存到: {output_file}")
        return output_file


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Starship数据处理模拟脚本')
    parser.add_argument('dataset_path', help='数据集基础路径')
    parser.add_argument('--output', '-o', help='输出结果文件名')
    
    args = parser.parse_args()
    
    # 创建处理器并运行
    processor = StarshipDataProcessor(args.dataset_path)
    processor.process_samples()
    processor.print_statistics()
    
    if args.output:
        processor.save_results(args.output)
    else:
        processor.save_results()


if __name__ == "__main__":
    main()