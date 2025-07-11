#!/usr/bin/env python3
"""
文件结构分析和可视化工具

提供用于分析Starship数据集文件结构的实用函数和可视化功能。
"""

import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict, Counter
import numpy as np


class FileStructureAnalyzer:
    """文件结构分析器"""
    
    def __init__(self):
        """初始化分析器"""
        # 设置matplotlib支持中文
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 设置seaborn样式
        sns.set_style("whitegrid")
        sns.set_palette("husl")
    
    def load_diagnostic_report(self, report_file: str) -> Dict:
        """加载诊断报告"""
        with open(report_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def visualize_file_distribution(self, report: Dict, output_dir: str = "visualizations"):
        """可视化文件分布"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 准备数据
        data = []
        for starship_path, structure in report['directory_structure'].items():
            file_counts = structure.get('file_counts', {})
            for subdir, count in file_counts.items():
                data.append({
                    'starship_directory': starship_path,
                    'subdirectory': subdir,
                    'file_count': count
                })
        
        if not data:
            print("没有文件分布数据可视化")
            return
        
        df = pd.DataFrame(data)
        
        # 1. 按子目录的文件数量分布
        plt.figure(figsize=(12, 8))
        if not df.empty:
            subdir_counts = df.groupby('subdirectory')['file_count'].sum().sort_values(ascending=False)
            
            plt.subplot(2, 2, 1)
            subdir_counts.plot(kind='bar')
            plt.title('各子目录文件数量分布')
            plt.xlabel('子目录')
            plt.ylabel('文件数量')
            plt.xticks(rotation=45)
            
            # 2. 按starship目录的文件数量分布
            plt.subplot(2, 2, 2)
            starship_counts = df.groupby('starship_directory')['file_count'].sum().sort_values(ascending=False)
            starship_counts.plot(kind='bar')
            plt.title('各starship目录文件数量分布')
            plt.xlabel('starship目录')
            plt.ylabel('文件数量')
            plt.xticks(rotation=45)
            
            # 3. 热力图
            plt.subplot(2, 2, 3)
            pivot_table = df.pivot_table(values='file_count', 
                                       index='starship_directory', 
                                       columns='subdirectory', 
                                       fill_value=0)
            if not pivot_table.empty:
                sns.heatmap(pivot_table, annot=True, fmt='d', cmap='YlOrRd')
                plt.title('文件分布热力图')
                plt.xlabel('子目录')
                plt.ylabel('starship目录')
            
            # 4. 总体统计
            plt.subplot(2, 2, 4)
            total_files = df['file_count'].sum()
            total_dirs = len(df['starship_directory'].unique())
            avg_files_per_dir = total_files / total_dirs if total_dirs > 0 else 0
            
            stats_text = f"""
总文件数: {total_files:,}
starship目录数: {total_dirs}
平均每目录文件数: {avg_files_per_dir:.1f}
最多文件的子目录: {subdir_counts.index[0] if not subdir_counts.empty else 'N/A'}
最多文件数: {subdir_counts.iloc[0] if not subdir_counts.empty else 0}
            """
            plt.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center')
            plt.axis('off')
            plt.title('统计摘要')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/file_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def visualize_file_extensions(self, report: Dict, output_dir: str = "visualizations"):
        """可视化文件扩展名分布"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 收集所有文件扩展名
        all_extensions = Counter()
        
        for starship_path, structure in report['directory_structure'].items():
            file_exts = structure.get('file_extensions', {})
            for ext, count in file_exts.items():
                all_extensions[ext] += count
        
        if not all_extensions:
            print("没有文件扩展名数据可视化")
            return
        
        # 创建可视化
        plt.figure(figsize=(15, 10))
        
        # 1. 扩展名分布饼图
        plt.subplot(2, 2, 1)
        top_extensions = dict(all_extensions.most_common(10))
        plt.pie(top_extensions.values(), labels=top_extensions.keys(), autopct='%1.1f%%')
        plt.title('前10个文件扩展名分布')
        
        # 2. 扩展名数量条形图
        plt.subplot(2, 2, 2)
        ext_df = pd.DataFrame(list(all_extensions.items()), columns=['extension', 'count'])
        ext_df = ext_df.sort_values('count', ascending=False).head(15)
        
        plt.bar(ext_df['extension'], ext_df['count'])
        plt.title('文件扩展名数量分布（前15个）')
        plt.xlabel('文件扩展名')
        plt.ylabel('文件数量')
        plt.xticks(rotation=45)
        
        # 3. 按类别分组的扩展名
        plt.subplot(2, 2, 3)
        extension_categories = {
            '点云文件': ['.ply', '.pcd', '.xyz', '.pts'],
            '图像文件': ['.png', '.jpg', '.jpeg', '.svg', '.pdf'],
            '数据文件': ['.json', '.csv', '.txt', '.dat'],
            '其他': []
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
        
        plt.pie(category_counts.values(), labels=category_counts.keys(), autopct='%1.1f%%')
        plt.title('按文件类别分布')
        
        # 4. 扩展名统计表
        plt.subplot(2, 2, 4)
        stats_text = "文件扩展名统计:\n\n"
        for ext, count in all_extensions.most_common(10):
            percentage = count / sum(all_extensions.values()) * 100
            stats_text += f"{ext:>8}: {count:>6,} ({percentage:>5.1f}%)\n"
        
        plt.text(0.1, 0.5, stats_text, fontsize=10, family='monospace', 
                verticalalignment='center')
        plt.axis('off')
        plt.title('扩展名详细统计')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/file_extensions.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def visualize_matching_analysis(self, report: Dict, output_dir: str = "visualizations"):
        """可视化文件匹配分析"""
        os.makedirs(output_dir, exist_ok=True)
        
        matching = report.get('matching_analysis', {})
        if not matching:
            print("没有匹配分析数据可视化")
            return
        
        plt.figure(figsize=(12, 8))
        
        # 1. 样本完整性分布
        plt.subplot(2, 2, 1)
        categories = ['完整样本', '部分样本', '仅点云']
        values = [
            matching.get('complete_samples', 0),
            matching.get('partial_samples', 0),
            matching.get('pointcloud_only', 0)
        ]
        colors = ['green', 'orange', 'red']
        
        plt.pie(values, labels=categories, colors=colors, autopct='%1.1f%%')
        plt.title('样本完整性分布')
        
        # 2. 成功率指标
        plt.subplot(2, 2, 2)
        success_rate = matching.get('success_rate', 0)
        
        # 创建成功率仪表盘风格图
        angles = np.linspace(0, np.pi, 100)
        values_gauge = np.ones_like(angles)
        
        ax = plt.gca()
        ax.fill_between(angles, 0, values_gauge, alpha=0.3, color='lightgray')
        
        # 成功率指针
        success_angle = success_rate * np.pi
        plt.arrow(0, 0, np.cos(success_angle), np.sin(success_angle), 
                 head_width=0.1, head_length=0.1, fc='red', ec='red')
        
        plt.xlim(-1.2, 1.2)
        plt.ylim(-0.2, 1.2)
        plt.title(f'成功率: {success_rate:.2%}')
        plt.axis('off')
        
        # 3. 数量对比
        plt.subplot(2, 2, 3)
        metrics = ['总ID数', '完整样本', '部分样本', '仅点云']
        values = [
            matching.get('total_unique_ids', 0),
            matching.get('complete_samples', 0),
            matching.get('partial_samples', 0),
            matching.get('pointcloud_only', 0)
        ]
        
        plt.bar(metrics, values, color=['blue', 'green', 'orange', 'red'])
        plt.title('样本数量对比')
        plt.ylabel('数量')
        plt.xticks(rotation=45)
        
        # 4. 问题严重程度
        plt.subplot(2, 2, 4)
        total_samples = matching.get('total_unique_ids', 1)
        complete_rate = matching.get('complete_samples', 0) / total_samples
        partial_rate = matching.get('partial_samples', 0) / total_samples
        missing_rate = matching.get('pointcloud_only', 0) / total_samples
        
        severity_text = f"""
数据质量评估:

总样本数: {total_samples:,}
完整率: {complete_rate:.2%}
部分完整率: {partial_rate:.2%}
严重缺失率: {missing_rate:.2%}

质量评级: {'优秀' if complete_rate > 0.9 else '良好' if complete_rate > 0.7 else '一般' if complete_rate > 0.5 else '较差' if complete_rate > 0.2 else '严重问题'}
        """
        
        plt.text(0.1, 0.5, severity_text, fontsize=11, verticalalignment='center')
        plt.axis('off')
        plt.title('数据质量评估')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/matching_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_comprehensive_report(self, diagnostic_report_file: str, 
                                    processing_results_file: str = None,
                                    output_dir: str = "analysis_output"):
        """生成综合分析报告"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载诊断报告
        diagnostic_report = self.load_diagnostic_report(diagnostic_report_file)
        
        # 生成可视化
        viz_dir = f"{output_dir}/visualizations"
        self.visualize_file_distribution(diagnostic_report, viz_dir)
        self.visualize_file_extensions(diagnostic_report, viz_dir)
        self.visualize_matching_analysis(diagnostic_report, viz_dir)
        
        # 生成文字报告
        self.generate_text_report(diagnostic_report, processing_results_file, output_dir)
        
        print(f"\n综合分析报告已生成到目录: {output_dir}")
        print(f"包含内容:")
        print(f"  - 可视化图表: {viz_dir}/")
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
        
        # 问题和建议
        report_content.append("## 发现的问题和建议\n")
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
        
        # 保存报告
        report_file = f"{output_dir}/comprehensive_report.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_content))
        
        print(f"详细报告已保存到: {report_file}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='文件结构分析和可视化工具')
    parser.add_argument('diagnostic_report', help='诊断报告JSON文件路径')
    parser.add_argument('--processing-results', help='数据处理结果JSON文件路径')
    parser.add_argument('--output-dir', default='analysis_output', help='输出目录')
    
    args = parser.parse_args()
    
    analyzer = FileStructureAnalyzer()
    analyzer.generate_comprehensive_report(
        args.diagnostic_report,
        args.processing_results,
        args.output_dir
    )


if __name__ == "__main__":
    main()