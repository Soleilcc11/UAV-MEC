#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
论文结果生成器 - 分析实验数据并生成论文中描述的结果表格和图表
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import argparse
import glob
from collections import defaultdict
import seaborn as sns
from matplotlib.ticker import MaxNLocator

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

def load_algorithm_data(base_dir, algorithm_pattern):
    """
    加载特定算法的训练数据
    
    参数:
        base_dir: 基础目录
        algorithm_pattern: 算法文件模式
    
    返回:
        算法数据字典
    """
    data_files = []
    
    # 首先尝试在results目录查找
    results_pattern = os.path.join(base_dir, "results", f"{algorithm_pattern}*.json")
    data_files.extend(glob.glob(results_pattern))
    
    # 然后在sim_results目录的每个子目录中查找
    sim_results_dir = os.path.join(base_dir, "sim_results")
    if os.path.exists(sim_results_dir):
        for subdir in os.listdir(sim_results_dir):
            subdir_path = os.path.join(sim_results_dir, subdir)
            if os.path.isdir(subdir_path):
                subdir_pattern = os.path.join(subdir_path, f"{algorithm_pattern}*.json")
                data_files.extend(glob.glob(subdir_pattern))
    
    if not data_files:
        print(f"警告: 没有找到{algorithm_pattern}算法的数据文件")
        return None
    
    print(f"找到{algorithm_pattern}算法的{len(data_files)}个数据文件: {data_files}")
    
    # 合并所有数据
    all_rewards = []
    all_delays = []
    all_energies = []
    all_success_rates = []
    
    for data_file in data_files:
        try:
            with open(data_file, 'r') as f:
                data = json.load(f)
                
                if isinstance(data, dict):
                    # 提取数据
                    if 'rewards' in data:
                        all_rewards.append(data['rewards'])
                    if 'delays' in data:
                        all_delays.append(data['delays'])
                    if 'energies' in data:
                        all_energies.append(data['energies'])
                    if 'success_rates' in data:
                        all_success_rates.append(data['success_rates'])
        except Exception as e:
            print(f"读取文件{data_file}时出错: {str(e)}")
    
    # 计算平均值
    avg_rewards = np.mean(all_rewards, axis=0) if all_rewards else []
    avg_delays = np.mean(all_delays, axis=0) if all_delays else []
    avg_energies = np.mean(all_energies, axis=0) if all_energies else []
    avg_success_rates = np.mean(all_success_rates, axis=0) if all_success_rates else []
    
    # 计算收敛回合数（假设后100回合的变化小于2%）
    convergence_episode = calculate_convergence_episode(avg_rewards, threshold=0.02, window=100)
    
    # 计算学习稳定性（后200回合奖励的方差）
    if len(avg_rewards) > 200:
        stability = np.var(avg_rewards[-200:])
    else:
        stability = np.var(avg_rewards) if len(avg_rewards) > 0 else 0
    
    # 计算最终指标值（后100回合的平均值）
    final_rewards = np.mean(avg_rewards[-100:]) if len(avg_rewards) >= 100 else np.mean(avg_rewards)
    final_delays = np.mean(avg_delays[-100:]) if len(avg_delays) >= 100 else np.mean(avg_delays)
    final_energies = np.mean(avg_energies[-100:]) if len(avg_energies) >= 100 else np.mean(avg_energies)
    final_success_rates = np.mean(avg_success_rates[-100:]) if len(avg_success_rates) >= 100 else np.mean(avg_success_rates)
    
    return {
        'rewards': avg_rewards,
        'delays': avg_delays,
        'energies': avg_energies,
        'success_rates': avg_success_rates,
        'final_rewards': final_rewards,
        'final_delays': final_delays,
        'final_energies': final_energies,
        'final_success_rates': final_success_rates * 100,  # 转换为百分比
        'convergence_episode': convergence_episode,
        'stability': stability
    }

def calculate_convergence_episode(rewards, threshold=0.02, window=100):
    """
    计算收敛回合数
    
    参数:
        rewards: 奖励列表
        threshold: 变化阈值
        window: 窗口大小
    
    返回:
        收敛回合数
    """
    if len(rewards) < window:
        return len(rewards)
    
    for i in range(window, len(rewards)):
        window_avg = np.mean(rewards[i-window:i])
        recent_avg = np.mean(rewards[i-window//2:i])
        
        # 如果近期平均值与窗口平均值的相对变化小于阈值，认为收敛
        if abs((recent_avg - window_avg) / (abs(window_avg) + 1e-10)) < threshold:
            return i
    
    return len(rewards)

def generate_comparison_table(algorithms_data):
    """
    生成算法对比表格
    
    参数:
        algorithms_data: 算法数据字典
    
    返回:
        对比表格DataFrame
    """
    table_data = {
        '指标名称': ['平均奖励', '平均时延 (ms)', '平均能耗 (J)', 
                 '任务成功率 (%)', '平均收敛时间 (回合)', '学习稳定性 (方差)']
    }
    
    # 添加每个算法的数据
    for algo_name, data in algorithms_data.items():
        if data:
            table_data[algo_name] = [
                f"{data['final_rewards']:.4f}",
                f"{data['final_delays']:.4f}",
                f"{data['final_energies']:.4f}",
                f"{data['final_success_rates']:.2f}",
                f"{data['convergence_episode']}",
                f"{data['stability']:.3f}"
            ]
        else:
            table_data[algo_name] = ['N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A']
    
    return pd.DataFrame(table_data)

def plot_learning_curves(algorithms_data, metric, title, xlabel, ylabel, output_path, smooth=True):
    """
    绘制学习曲线
    
    参数:
        algorithms_data: 算法数据字典
        metric: 指标名称
        title: 图表标题
        xlabel: x轴标签
        ylabel: y轴标签
        output_path: 输出路径
        smooth: 是否平滑曲线
    """
    plt.figure(figsize=(12, 7))
    
    # 设置颜色
    colors = {
        'PPO': 'blue',
        'DQN': 'green',
        '优化版DDPG': 'red',
        '标准DDPG': 'orange'
    }
    
    # 绘制每个算法的曲线
    for algo_name, data in algorithms_data.items():
        if data and len(data[metric]) > 0:
            episodes = range(len(data[metric]))
            values = data[metric]
            
            if smooth and len(values) > 50:
                # 使用移动平均平滑曲线
                window_size = min(50, len(values) // 10)
                smoothed = np.convolve(values, np.ones(window_size)/window_size, mode='valid')
                episodes = range(window_size-1, window_size-1+len(smoothed))
                plt.plot(episodes, smoothed, label=algo_name, color=colors.get(algo_name, None), linewidth=2)
            else:
                plt.plot(episodes, values, label=algo_name, color=colors.get(algo_name, None), linewidth=2)
    
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    # 设置x轴为整数
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"已保存图表到: {output_path}")
    plt.close()

def generate_thesis_results(base_dir, output_dir):
    """
    生成论文结果
    
    参数:
        base_dir: 基础目录
        output_dir: 输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载各算法数据
    algorithms_data = {
        'PPO': load_algorithm_data(base_dir, "ppo"),
        'DQN': load_algorithm_data(base_dir, "dqn"),
        '优化版DDPG': load_algorithm_data(base_dir, "priddpg"),
        '标准DDPG': load_algorithm_data(base_dir, "ddpg")
    }
    
    # 生成对比表格
    comparison_table = generate_comparison_table(algorithms_data)
    table_path = os.path.join(output_dir, "表6-1_各算法性能指标比较.csv")
    comparison_table.to_csv(table_path, index=False, encoding='utf-8-sig')
    print(f"已保存表格到: {table_path}")
    
    # 绘制学习曲线
    plot_learning_curves(
        algorithms_data, 
        'delays', 
        '图6-1 各算法平均时延学习曲线比较', 
        '回合', 
        '平均时延 (ms)', 
        os.path.join(output_dir, "图6-1_各算法平均时延学习曲线比较.png")
    )
    
    plot_learning_curves(
        algorithms_data, 
        'energies', 
        '图6-2 各算法平均能耗学习曲线比较', 
        '回合', 
        '平均能耗 (J)', 
        os.path.join(output_dir, "图6-2_各算法平均能耗学习曲线比较.png")
    )
    
    plot_learning_curves(
        algorithms_data, 
        'success_rates', 
        '图6-3 各算法任务成功率学习曲线比较', 
        '回合', 
        '任务成功率', 
        os.path.join(output_dir, "图6-3_各算法任务成功率学习曲线比较.png")
    )
    
    plot_learning_curves(
        algorithms_data, 
        'rewards', 
        '图6-4 各算法平均奖励学习曲线比较', 
        '回合', 
        '平均奖励', 
        os.path.join(output_dir, "图6-4_各算法平均奖励学习曲线比较.png")
    )
    
    # 生成性能分析报告
    generate_performance_analysis(algorithms_data, output_dir)

def generate_performance_analysis(algorithms_data, output_dir):
    """
    生成性能分析报告
    
    参数:
        algorithms_data: 算法数据字典
        output_dir: 输出目录
    """
    # 提取最终性能指标
    final_metrics = {}
    for algo_name, data in algorithms_data.items():
        if data:
            final_metrics[algo_name] = {
                'rewards': data['final_rewards'],
                'delays': data['final_delays'],
                'energies': data['final_energies'],
                'success_rates': data['final_success_rates'],
                'convergence': data['convergence_episode'],
                'stability': data['stability']
            }
    
    # 性能排名
    metrics_ranking = {
        'rewards': sorted(final_metrics.keys(), key=lambda x: final_metrics[x]['rewards'], reverse=True),
        'delays': sorted(final_metrics.keys(), key=lambda x: final_metrics[x]['delays']),
        'energies': sorted(final_metrics.keys(), key=lambda x: final_metrics[x]['energies']),
        'success_rates': sorted(final_metrics.keys(), key=lambda x: final_metrics[x]['success_rates'], reverse=True)
    }
    
    # 生成报告文件
    report_path = os.path.join(output_dir, "算法性能分析报告.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# UAV-MEC任务卸载算法性能分析报告\n\n")
        
        # 写入性能排名
        f.write("## 1. 各算法性能排名\n\n")
        
        f.write("### 1.1 平均奖励排名\n")
        f.write("| 排名 | 算法 | 平均奖励 |\n")
        f.write("|------|------|----------|\n")
        for i, algo in enumerate(metrics_ranking['rewards']):
            f.write(f"| {i+1} | {algo} | {final_metrics[algo]['rewards']:.4f} |\n")
        
        f.write("\n### 1.2 平均时延排名\n")
        f.write("| 排名 | 算法 | 平均时延 (ms) |\n")
        f.write("|------|------|---------------|\n")
        for i, algo in enumerate(metrics_ranking['delays']):
            f.write(f"| {i+1} | {algo} | {final_metrics[algo]['delays']:.4f} |\n")
        
        f.write("\n### 1.3 平均能耗排名\n")
        f.write("| 排名 | 算法 | 平均能耗 (J) |\n")
        f.write("|------|------|-------------|\n")
        for i, algo in enumerate(metrics_ranking['energies']):
            f.write(f"| {i+1} | {algo} | {final_metrics[algo]['energies']:.4f} |\n")
        
        f.write("\n### 1.4 任务成功率排名\n")
        f.write("| 排名 | 算法 | 任务成功率 (%) |\n")
        f.write("|------|------|----------------|\n")
        for i, algo in enumerate(metrics_ranking['success_rates']):
            f.write(f"| {i+1} | {algo} | {final_metrics[algo]['success_rates']:.2f} |\n")
        
        # 写入算法特性分析
        f.write("\n## 2. 算法特性分析\n\n")
        
        if '优化版DDPG' in final_metrics and '标准DDPG' in final_metrics:
            std_ddpg = final_metrics['标准DDPG']
            opt_ddpg = final_metrics['优化版DDPG']
            
            success_rate_improvement = (opt_ddpg['success_rates'] - std_ddpg['success_rates']) / std_ddpg['success_rates'] * 100
            delay_reduction = (std_ddpg['delays'] - opt_ddpg['delays']) / std_ddpg['delays'] * 100
            
            f.write("### 2.1 优化版DDPG对标准DDPG的改进\n\n")
            f.write(f"- 任务成功率提升: {success_rate_improvement:.1f}%\n")
            f.write(f"- 平均任务完成时间减少: {delay_reduction:.1f}%\n")
            f.write(f"- 收敛速度提升: {(std_ddpg['convergence'] - opt_ddpg['convergence']) / std_ddpg['convergence'] * 100:.1f}%\n")
            f.write(f"- 学习稳定性提升: {(std_ddpg['stability'] - opt_ddpg['stability']) / std_ddpg['stability'] * 100:.1f}%\n")
        
        if 'PPO' in final_metrics and 'DQN' in final_metrics:
            ppo = final_metrics['PPO']
            dqn = final_metrics['DQN']
            
            f.write("\n### 2.2 PPO与DQN特性对比\n\n")
            f.write(f"- PPO比DQN的平均奖励高: {(ppo['rewards'] - dqn['rewards']) / abs(dqn['rewards']) * 100:.1f}%\n")
            f.write(f"- PPO比DQN的平均时延低: {(dqn['delays'] - ppo['delays']) / dqn['delays'] * 100:.1f}%\n")
            f.write(f"- DQN比PPO的平均能耗低: {(ppo['energies'] - dqn['energies']) / (ppo['energies'] + 1e-10) * 100:.1f}%\n")
            f.write(f"- PPO比DQN的任务成功率高: {(ppo['success_rates'] - dqn['success_rates']) / dqn['success_rates'] * 100:.1f}%\n")
        
        # 写入多目标权衡分析
        f.write("\n## 3. 多目标权衡分析\n\n")
        f.write("### 3.1 时延-能耗权衡\n\n")
        f.write("| 算法 | 平均时延 (ms) | 平均能耗 (J) | 时延-能耗比 |\n")
        f.write("|------|---------------|--------------|------------|\n")
        for algo in final_metrics:
            delay = final_metrics[algo]['delays']
            energy = final_metrics[algo]['energies']
            balance = delay / (energy + 1e-10)
            f.write(f"| {algo} | {delay:.4f} | {energy:.4f} | {balance:.2f} |\n")
        
        f.write("\n### 3.2 成功率-能耗权衡\n\n")
        f.write("| 算法 | 任务成功率 (%) | 平均能耗 (J) | 成功率-能耗比 |\n")
        f.write("|------|----------------|--------------|---------------|\n")
        for algo in final_metrics:
            success = final_metrics[algo]['success_rates']
            energy = final_metrics[algo]['energies']
            efficiency = success / (energy + 1e-10)
            f.write(f"| {algo} | {success:.2f} | {energy:.4f} | {efficiency:.2f} |\n")
    
    print(f"已保存性能分析报告到: {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成论文结果")
    parser.add_argument('--base_dir', type=str, default='.', help='基础目录')
    parser.add_argument('--output_dir', type=str, default='thesis_results', help='输出目录')
    
    args = parser.parse_args()
    generate_thesis_results(args.base_dir, args.output_dir)
    print(f"论文结果生成完成，已保存到: {args.output_dir}") 