#!/usr/bin/env python3
"""
Starship数据集文件结构诊断工具

该脚本用于诊断Starship数据集的实际文件结构，识别文件命名模式，
并生成详细的诊断报告来解决数据处理中的文件缺失问题。

Author: AI Assistant
Date: 2024
"""

import os
import json
import logging
import argparse
import time
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set
import re


class FileStructureDiagnostic:
    """文件结构诊断器"""
    
    def __init__(self, base_path: str, log_level: str = "INFO"):
        """
        初始化诊断器
        
        Args:
            base_path: 数据集基础路径
            log_level: 日志级别
        """
        self.base_path = Path(base_path)
        self.setup_logging(log_level)
        self.logger = logging.getLogger(__name__)
        
        # 期望的文件结构
        self.expected_structure = {
            'airfoils': ['.dat', '.txt', '.csv'],
            'extracted_params': ['.json', '.csv', '.txt'],
            'param_visualizations': ['.png', '.jpg', '.svg', '.pdf'],
            'pointclouds': ['.ply', '.pcd', '.xyz', '.pts'],
            'results/aero_analysis': ['.json', '.csv', '.txt'],
            'results/merged_data': ['.json', '.csv', '.parquet']
        }
        
        # 诊断结果
        self.diagnostic_results = {
            'scan_summary': {},
            'file_counts': defaultdict(dict),
            'naming_patterns': defaultdict(list),
            'missing_files': defaultdict(list),
            'unexpected_files': defaultdict(list),
            'directory_structure': {},
            'recommendations': []
        }
    
    def setup_logging(self, log_level: str):
        """设置日志配置"""
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'diagnostic_{int(time.time())}.log'),
                logging.StreamHandler()
            ]
        )
    
    def scan_directory_structure(self) -> Dict:
        """扫描目录结构"""
        self.logger.info(f"开始扫描目录结构: {self.base_path}")
        
        if not self.base_path.exists():
            self.logger.error(f"基础路径不存在: {self.base_path}")
            return {}
        
        structure = {}
        total_files = 0
        
        # 查找所有starship_data目录
        starship_dirs = []
        for root, dirs, files in os.walk(self.base_path):
            if 'starship_data' in dirs:
                starship_path = Path(root) / 'starship_data'
                starship_dirs.append(starship_path)
        
        self.logger.info(f"找到 {len(starship_dirs)} 个starship_data目录")
        
        for starship_dir in starship_dirs:
            rel_path = starship_dir.relative_to(self.base_path)
            structure[str(rel_path)] = self._scan_starship_directory(starship_dir)
            total_files += sum(structure[str(rel_path)]['file_counts'].values())
        
        self.diagnostic_results['scan_summary'] = {
            'total_starship_directories': len(starship_dirs),
            'total_files_scanned': total_files,
            'scan_time': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        self.diagnostic_results['directory_structure'] = structure
        return structure
    
    def _scan_starship_directory(self, starship_dir: Path) -> Dict:
        """扫描单个starship_data目录"""
        self.logger.info(f"扫描starship目录: {starship_dir}")
        
        result = {
            'subdirectories': [],
            'file_counts': defaultdict(int),
            'file_extensions': defaultdict(int),
            'naming_patterns': defaultdict(set),
            'sample_files': defaultdict(list)
        }
        
        if not starship_dir.exists():
            self.logger.warning(f"目录不存在: {starship_dir}")
            return result
        
        # 扫描子目录
        for item in starship_dir.iterdir():
            if item.is_dir():
                result['subdirectories'].append(item.name)
                self._scan_subdirectory(item, result)
        
        return result
    
    def _scan_subdirectory(self, subdir: Path, result: Dict):
        """扫描子目录"""
        subdir_name = subdir.name
        
        for file_path in subdir.rglob('*'):
            if file_path.is_file():
                # 统计文件数量
                result['file_counts'][subdir_name] += 1
                
                # 统计文件扩展名
                ext = file_path.suffix.lower()
                result['file_extensions'][ext] += 1
                
                # 分析文件命名模式
                filename = file_path.stem
                pattern = self._extract_naming_pattern(filename)
                result['naming_patterns'][subdir_name].add(pattern)
                
                # 收集样本文件名（最多10个）
                if len(result['sample_files'][subdir_name]) < 10:
                    result['sample_files'][subdir_name].append({
                        'filename': file_path.name,
                        'path': str(file_path.relative_to(subdir)),
                        'size': file_path.stat().st_size
                    })
    
    def _extract_naming_pattern(self, filename: str) -> str:
        """提取文件命名模式"""
        # 替换数字为占位符来识别模式
        pattern = re.sub(r'\d+', 'N', filename)
        pattern = re.sub(r'N+', 'N', pattern)  # 合并连续的N
        return pattern
    
    def analyze_file_matching(self):
        """分析文件匹配情况"""
        self.logger.info("开始分析文件匹配情况")
        
        # 收集所有文件的ID模式
        file_ids = defaultdict(set)
        
        for starship_path, structure in self.diagnostic_results['directory_structure'].items():
            for subdir in structure['subdirectories']:
                if subdir in ['pointclouds', 'airfoils', 'extracted_params', 
                             'param_visualizations', 'results']:
                    # 从样本文件中提取ID
                    samples = structure.get('sample_files', {}).get(subdir, [])
                    for sample in samples:
                        file_id = self._extract_file_id(sample['filename'])
                        if file_id:
                            file_ids[file_id].add(subdir)
        
        # 分析匹配情况
        complete_samples = 0
        partial_samples = 0
        pointcloud_only = 0
        
        expected_dirs = set(['pointclouds', 'airfoils', 'extracted_params', 
                           'param_visualizations', 'results'])
        
        for file_id, found_dirs in file_ids.items():
            if expected_dirs.issubset(found_dirs):
                complete_samples += 1
            elif 'pointclouds' in found_dirs:
                if len(found_dirs) == 1:
                    pointcloud_only += 1
                else:
                    partial_samples += 1
        
        self.diagnostic_results['matching_analysis'] = {
            'total_unique_ids': len(file_ids),
            'complete_samples': complete_samples,
            'partial_samples': partial_samples,
            'pointcloud_only': pointcloud_only,
            'success_rate': complete_samples / len(file_ids) if file_ids else 0
        }
    
    def _extract_file_id(self, filename: str) -> str:
        """从文件名中提取ID"""
        # 尝试不同的ID提取模式
        patterns = [
            r'(\d+)',  # 纯数字
            r'sample_(\d+)',  # sample_数字
            r'starship_(\d+)',  # starship_数字
            r'(\d+)_',  # 数字_
            r'_(\d+)',  # _数字
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                return match.group(1)
        
        return None
    
    def generate_recommendations(self):
        """生成修正建议"""
        self.logger.info("生成修正建议")
        
        recommendations = []
        
        # 基于扫描结果生成建议
        matching = self.diagnostic_results.get('matching_analysis', {})
        success_rate = matching.get('success_rate', 0)
        
        if success_rate < 0.1:
            recommendations.append({
                'priority': 'HIGH',
                'type': 'FILE_STRUCTURE',
                'description': '文件匹配成功率极低，需要检查文件命名规则和目录结构',
                'action': '1. 检查实际文件命名模式\n2. 更新文件路径匹配逻辑\n3. 验证目录结构完整性'
            })
        
        # 检查缺失的子目录
        expected_subdirs = set(['airfoils', 'extracted_params', 'param_visualizations', 
                               'pointclouds', 'results'])
        
        for starship_path, structure in self.diagnostic_results['directory_structure'].items():
            found_subdirs = set(structure['subdirectories'])
            missing_subdirs = expected_subdirs - found_subdirs
            
            if missing_subdirs:
                recommendations.append({
                    'priority': 'MEDIUM',
                    'type': 'MISSING_DIRECTORIES',
                    'description': f'目录 {starship_path} 缺少子目录: {", ".join(missing_subdirs)}',
                    'action': f'检查并创建缺失的子目录: {", ".join(missing_subdirs)}'
                })
        
        # 检查文件扩展名
        for starship_path, structure in self.diagnostic_results['directory_structure'].items():
            file_exts = structure.get('file_extensions', {})
            if not file_exts:
                recommendations.append({
                    'priority': 'HIGH',
                    'type': 'NO_FILES',
                    'description': f'目录 {starship_path} 中没有找到任何文件',
                    'action': '检查数据是否正确复制到该目录'
                })
        
        self.diagnostic_results['recommendations'] = recommendations
    
    def generate_report(self, output_file: str = None):
        """生成诊断报告"""
        if output_file is None:
            output_file = f'diagnostic_report_{int(time.time())}.json'
        
        # 转换set为list以便JSON序列化
        report = {}
        for key, value in self.diagnostic_results.items():
            if key == 'directory_structure':
                report[key] = {}
                for path, struct in value.items():
                    report[key][path] = {}
                    for k, v in struct.items():
                        if k == 'naming_patterns':
                            report[key][path][k] = {subdir: list(patterns) 
                                                   for subdir, patterns in v.items()}
                        else:
                            report[key][path][k] = dict(v) if isinstance(v, defaultdict) else v
            else:
                report[key] = dict(value) if isinstance(value, defaultdict) else value
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"诊断报告已保存到: {output_file}")
        return output_file
    
    def print_summary(self):
        """打印诊断摘要"""
        print("\n" + "="*60)
        print("Starship数据集文件结构诊断摘要")
        print("="*60)
        
        summary = self.diagnostic_results['scan_summary']
        print(f"扫描时间: {summary.get('scan_time', 'N/A')}")
        print(f"发现的starship_data目录数: {summary.get('total_starship_directories', 0)}")
        print(f"总文件数: {summary.get('total_files_scanned', 0)}")
        
        if 'matching_analysis' in self.diagnostic_results:
            matching = self.diagnostic_results['matching_analysis']
            print(f"\n文件匹配分析:")
            print(f"  唯一ID数量: {matching.get('total_unique_ids', 0)}")
            print(f"  完整样本数: {matching.get('complete_samples', 0)}")
            print(f"  部分样本数: {matching.get('partial_samples', 0)}")
            print(f"  仅点云文件: {matching.get('pointcloud_only', 0)}")
            print(f"  成功率: {matching.get('success_rate', 0):.2%}")
        
        print(f"\n发现的问题数: {len(self.diagnostic_results.get('recommendations', []))}")
        
        for i, rec in enumerate(self.diagnostic_results.get('recommendations', []), 1):
            print(f"\n问题 {i} [{rec['priority']}]:")
            print(f"  类型: {rec['type']}")
            print(f"  描述: {rec['description']}")
            print(f"  建议: {rec['action']}")
        
        print("\n" + "="*60)
    
    def run_diagnostic(self):
        """运行完整诊断"""
        self.logger.info("开始完整诊断流程")
        
        try:
            # 1. 扫描目录结构
            self.scan_directory_structure()
            
            # 2. 分析文件匹配
            self.analyze_file_matching()
            
            # 3. 生成建议
            self.generate_recommendations()
            
            # 4. 打印摘要
            self.print_summary()
            
            # 5. 生成报告
            report_file = self.generate_report()
            
            self.logger.info("诊断完成")
            return report_file
            
        except Exception as e:
            self.logger.error(f"诊断过程中发生错误: {e}")
            raise


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Starship数据集文件结构诊断工具')
    parser.add_argument('dataset_path', help='数据集基础路径')
    parser.add_argument('--log-level', default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='日志级别')
    parser.add_argument('--output', '-o', help='输出报告文件名')
    
    args = parser.parse_args()
    
    # 创建诊断器并运行
    diagnostic = FileStructureDiagnostic(args.dataset_path, args.log_level)
    report_file = diagnostic.run_diagnostic()
    
    if args.output:
        # 重命名报告文件
        import shutil
        shutil.move(report_file, args.output)
        print(f"\n报告已保存到: {args.output}")
    else:
        print(f"\n报告已保存到: {report_file}")


if __name__ == "__main__":
    main()