#!/usr/bin/env python3
"""
简化版文件结构分析器（无需matplotlib）

用于在没有可视化依赖的环境中生成分析报告。
"""

import os
import json
from pathlib import Path
from typing import Dict
from collections import defaultdict, Counter


class SimpleFileStructureAnalyzer:
    """简化版文件结构分析器"""
    
    def __init__(self):
        """初始化分析器"""
        pass
    
    def load_diagnostic_report(self, report_file: str) -> Dict:
        """加载诊断报告"""
        with open(report_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def generate_comprehensive_report(self, diagnostic_report_file: str, 
                                    processing_results_file: str = None,
                                    output_dir: str = "analysis_output"):
        """生成综合分析报告"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载诊断报告
        diagnostic_report = self.load_diagnostic_report(diagnostic_report_file)
        
        # 生成文字报告
        self.generate_text_report(diagnostic_report, processing_results_file, output_dir)
        
        print(f"\n综合分析报告已生成到目录: {output_dir}")
        print(f"包含内容:")
        print(f"  - 详细报告: {output_dir}/comprehensive_report.md")
    
    def generate_text_report(self, diagnostic_report: Dict, 
                           processing_results_file: str = None,
                           output_dir: str = "analysis_output"):
        """生成文字报告"""
        report_content = []
        
        # 报告标题
        report_content.append("# Starship数据集文件结构分析综合报告\n")
        report_content.append(f"生成时间: {diagnostic_report.get('scan_summary', {}).get('scan_time', 'Unknown')}\n")
        
        # 执行摘要
        report_content.append("## 执行摘要\n")
        scan_summary = diagnostic_report.get('scan_summary', {})
        matching_analysis = diagnostic_report.get('matching_analysis', {})
        
        report_content.append(f"- **发现的starship_data目录数**: {scan_summary.get('total_starship_directories', 0)}")
        report_content.append(f"- **总文件数**: {scan_summary.get('total_files_scanned', 0):,}")
        report_content.append(f"- **数据匹配成功率**: {matching_analysis.get('success_rate', 0):.2%}")
        report_content.append(f"- **完整样本数**: {matching_analysis.get('complete_samples', 0):,}")
        report_content.append(f"- **问题严重程度**: {'严重' if matching_analysis.get('success_rate', 0) < 0.1 else '中等' if matching_analysis.get('success_rate', 0) < 0.5 else '轻微'}\n")
        
        # 目录结构分析
        report_content.append("## 目录结构分析\n")
        directory_structure = diagnostic_report.get('directory_structure', {})
        
        for starship_path, structure in directory_structure.items():
            report_content.append(f"### {starship_path}\n")
            
            subdirs = structure.get('subdirectories', [])
            report_content.append(f"**发现的子目录**: {', '.join(subdirs) if subdirs else '无'}")
            
            file_counts = structure.get('file_counts', {})
            if file_counts:
                report_content.append("**文件数量统计**:")
                for subdir, count in file_counts.items():
                    report_content.append(f"  - {subdir}: {count:,}")
            
            file_extensions = structure.get('file_extensions', {})
            if file_extensions:
                report_content.append("**文件扩展名统计**:")
                for ext, count in sorted(file_extensions.items(), key=lambda x: x[1], reverse=True)[:5]:
                    report_content.append(f"  - {ext}: {count:,}")
            
            report_content.append("")  # 空行
        
        # 文件扩展名分析
        report_content.append("## 文件类型分析\n")
        all_extensions = Counter()
        
        for starship_path, structure in directory_structure.items():
            file_exts = structure.get('file_extensions', {})
            for ext, count in file_exts.items():
                all_extensions[ext] += count
        
        if all_extensions:
            report_content.append("**总体文件扩展名分布**:")
            for ext, count in all_extensions.most_common(10):
                percentage = count / sum(all_extensions.values()) * 100
                report_content.append(f"  - {ext}: {count:,} ({percentage:.1f}%)")
            
            # 按类别分组
            extension_categories = {
                '点云文件': ['.ply', '.pcd', '.xyz', '.pts'],
                '图像文件': ['.png', '.jpg', '.jpeg', '.svg', '.pdf'],
                '数据文件': ['.json', '.csv', '.txt', '.dat'],
            }
            
            category_counts = defaultdict(int)
            for ext, count in all_extensions.items():
                categorized = False
                for category, exts in extension_categories.items():
                    if ext.lower() in exts:
                        category_counts[category] += count
                        categorized = True
                        break
                if not categorized:
                    category_counts['其他'] += count
            
            report_content.append("\n**按文件类别分布**:")
            for category, count in category_counts.items():
                percentage = count / sum(all_extensions.values()) * 100
                report_content.append(f"  - {category}: {count:,} ({percentage:.1f}%)")
        
        # 问题和建议
        report_content.append("\n## 发现的问题和建议\n")
        recommendations = diagnostic_report.get('recommendations', [])
        
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                priority_emoji = "🔴" if rec['priority'] == 'HIGH' else "🟡" if rec['priority'] == 'MEDIUM' else "🟢"
                report_content.append(f"### 问题 {i} {priority_emoji} [{rec['priority']}]")
                report_content.append(f"**类型**: {rec['type']}")
                report_content.append(f"**描述**: {rec['description']}")
                report_content.append(f"**建议行动**: {rec['action']}")
                report_content.append("")
        else:
            report_content.append("未发现特定问题。")
        
        # 技术细节
        report_content.append("## 技术细节\n")
        report_content.append("### 文件匹配逻辑分析")
        
        if matching_analysis:
            report_content.append(f"- 发现的唯一ID数量: {matching_analysis.get('total_unique_ids', 0):,}")
            report_content.append(f"- 完整匹配样本: {matching_analysis.get('complete_samples', 0):,}")
            report_content.append(f"- 部分匹配样本: {matching_analysis.get('partial_samples', 0):,}")
            report_content.append(f"- 仅点云文件样本: {matching_analysis.get('pointcloud_only', 0):,}")
        
        # 加载处理结果
        if processing_results_file and os.path.exists(processing_results_file):
            report_content.append("\n### 数据处理结果分析")
            with open(processing_results_file, 'r', encoding='utf-8') as f:
                processing_results = json.load(f)
            
            stats = processing_results.get('statistics', {})
            report_content.append(f"- 点云文件数: {stats.get('total_pointclouds_found', 0):,}")
            report_content.append(f"- 完整样本数: {stats.get('complete_samples', 0):,}")
            report_content.append(f"- 不完整样本数: {stats.get('incomplete_samples', 0):,}")
            report_content.append(f"- 处理成功率: {stats.get('success_rate', 0):.2%}")
            report_content.append(f"- 处理时间: {stats.get('processing_time', 0):.2f}秒")
        
        # 下一步行动计划
        report_content.append("\n## 下一步行动计划\n")
        success_rate = matching_analysis.get('success_rate', 0)
        
        if success_rate < 0.1:
            report_content.append("### 紧急行动（成功率极低）")
            report_content.append("1. **立即检查文件命名规则** - 验证实际文件命名模式")
            report_content.append("2. **更新路径匹配逻辑** - 修正文件查找算法")
            report_content.append("3. **验证目录结构** - 确认所有必要的子目录存在")
            report_content.append("4. **数据完整性检查** - 验证数据复制是否完整")
        elif success_rate < 0.5:
            report_content.append("### 优化行动（成功率中等）")
            report_content.append("1. **细化匹配规则** - 改进文件ID提取逻辑")
            report_content.append("2. **处理边缘情况** - 处理特殊命名格式")
            report_content.append("3. **验证数据质量** - 检查文件完整性")
        else:
            report_content.append("### 维护行动（成功率良好）")
            report_content.append("1. **监控数据质量** - 定期检查数据完整性")
            report_content.append("2. **优化处理性能** - 提升处理速度")
        
        # 文件命名模式分析
        report_content.append("\n## 文件命名模式分析\n")
        for starship_path, structure in directory_structure.items():
            naming_patterns = structure.get('naming_patterns', {})
            if naming_patterns:
                report_content.append(f"### {starship_path}")
                for subdir, patterns in naming_patterns.items():
                    if patterns:
                        report_content.append(f"**{subdir}目录的命名模式**:")
                        for pattern in patterns[:5]:  # 只显示前5个模式
                            report_content.append(f"  - {pattern}")
                report_content.append("")
        
        # 保存报告
        report_file = f"{output_dir}/comprehensive_report.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_content))
        
        print(f"详细报告已保存到: {report_file}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='简化版文件结构分析工具')
    parser.add_argument('diagnostic_report', help='诊断报告JSON文件路径')
    parser.add_argument('--processing-results', help='数据处理结果JSON文件路径')
    parser.add_argument('--output-dir', default='analysis_output', help='输出目录')
    
    args = parser.parse_args()
    
    analyzer = SimpleFileStructureAnalyzer()
    analyzer.generate_comprehensive_report(
        args.diagnostic_report,
        args.processing_results,
        args.output_dir
    )


if __name__ == "__main__":
    main()