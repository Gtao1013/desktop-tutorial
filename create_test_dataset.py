#!/usr/bin/env python3
"""
创建测试数据结构

该脚本创建一个模拟的Starship数据集结构，用于测试诊断工具。
"""

import os
import json
import random
from pathlib import Path


def create_test_dataset(base_path: str, num_samples: int = 100):
    """
    创建测试数据集结构
    
    Args:
        base_path: 基础路径
        num_samples: 样本数量
    """
    base_path = Path(base_path)
    
    # 创建基础目录结构
    starship_dirs = [
        base_path / "1-100" / "starship_data",
        base_path / "1-2000" / "starship_data", 
        base_path / "1-10000" / "starship_data"
    ]
    
    for starship_dir in starship_dirs:
        # 创建子目录
        subdirs = [
            "airfoils",
            "extracted_params",
            "param_visualizations", 
            "pointclouds",
            "results/aero_analysis",
            "results/merged_data"
        ]
        
        for subdir in subdirs:
            (starship_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    # 创建样本文件
    for i in range(1, num_samples + 1):
        # 随机选择一个starship目录
        starship_dir = random.choice(starship_dirs)
        
        # 点云文件（总是存在）
        pointcloud_file = starship_dir / "pointclouds" / f"sample_{i:04d}.ply"
        pointcloud_file.write_text(f"# Sample pointcloud {i}\n")
        
        # 其他文件（随机缺失一些，模拟问题）
        files_to_create = []
        
        # 70%概率有翼型文件
        if random.random() < 0.7:
            files_to_create.append(("airfoils", f"airfoil_{i:04d}.dat"))
        
        # 60%概率有参数文件
        if random.random() < 0.6:
            files_to_create.append(("extracted_params", f"params_{i:04d}.json"))
        
        # 50%概率有可视化文件
        if random.random() < 0.5:
            files_to_create.append(("param_visualizations", f"viz_{i:04d}.png"))
        
        # 40%概率有气动分析文件
        if random.random() < 0.4:
            files_to_create.append(("results/aero_analysis", f"aero_{i:04d}.json"))
        
        # 30%概率有合并数据文件
        if random.random() < 0.3:
            files_to_create.append(("results/merged_data", f"merged_{i:04d}.csv"))
        
        # 创建文件
        for subdir, filename in files_to_create:
            file_path = starship_dir / subdir / filename
            file_path.write_text(f"# Sample data for {i}\n")
    
    print(f"测试数据集已创建在: {base_path}")
    print(f"样本数量: {num_samples}")
    print(f"Starship目录数: {len(starship_dirs)}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='创建测试数据集')
    parser.add_argument('--output', '-o', default='test_dataset', help='输出目录')
    parser.add_argument('--samples', '-n', type=int, default=100, help='样本数量')
    
    args = parser.parse_args()
    
    create_test_dataset(args.output, args.samples)


if __name__ == "__main__":
    main()