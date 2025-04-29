"""
实用工具函数
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import json

def calculate_path_loss(distance, height, frequency=2.4):
    """
    计算空地信道的路径损耗
    
    参数:
        distance: 地面距离 (m)
        height: 高度 (m)
        frequency: 频率 (GHz)
    
    返回:
        path_loss: 路径损耗 (dB)
    """
    # 计算三维距离
    distance_3d = np.sqrt(distance**2 + height**2)
    
    # 自由空间路径损耗模型
    path_loss = 20 * np.log10(distance_3d) + 20 * np.log10(frequency) + 32.44
    
    # 考虑LoS概率
    path_loss_parameter = 9.61
    path_loss_exponent = 0.16
    
    theta = np.arctan(height / distance) * 180 / np.pi
    los_prob = 1 / (1 + path_loss_parameter * np.exp(-path_loss_exponent * (theta - path_loss_parameter)))
    
    # 添加NLOS附加损耗
    additional_path_loss = 20
    final_path_loss = los_prob * path_loss + (1 - los_prob) * (path_loss + additional_path_loss)
    
    return final_path_loss

def calculate_data_rate(distance, height, bandwidth, tx_power, noise_power=1e-13):
    """
    计算传输速率
    
    参数:
        distance: 地面距离 (m)
        height: 高度 (m)
        bandwidth: 带宽 (MHz)
        tx_power: 发射功率 (W)
        noise_power: 噪声功率 (W)
    
    返回:
        data_rate: 传输速率 (Mbps)
    """
    path_loss = calculate_path_loss(distance, height)
    channel_gain = 10 ** (-path_loss / 10)
    sinr = tx_power * channel_gain / noise_power
    data_rate = bandwidth * np.log2(1 + sinr)
    
    return data_rate

def visualize_environment(users, uavs, area_size=1000):
    """
    可视化环境
    
    参数:
        users: 用户位置数组 shape=(n, 2)
        uavs: 无人机位置数组 shape=(m, 3)
        area_size: 区域大小
    """
    plt.figure(figsize=(10, 10))
    
    # 绘制用户
    plt.scatter(users[:, 0], users[:, 1], c='b', marker='o', label='Users')
    
    # 绘制无人机投影
    plt.scatter(uavs[:, 0], uavs[:, 1], c='r', marker='^', s=100, label='UAVs')
    
    # 绘制覆盖范围
    for i in range(len(uavs)):
        height = uavs[i, 2]
        # 简单估计覆盖半径 (高度的2倍)
        coverage_radius = height * 2
        circle = plt.Circle((uavs[i, 0], uavs[i, 1]), coverage_radius, 
                           color='r', fill=False, alpha=0.3)
        plt.gca().add_patch(circle)
    
    plt.xlim(0, area_size)
    plt.ylim(0, area_size)
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('UAV-MEC Environment')
    plt.legend()
    plt.grid(True)
    
    return plt

def compare_methods(scenario_results):
    """
    比较不同方法的性能
    
    参数:
        scenario_results: 字典，键为场景名，值为每个方法的结果
    """
    scenarios = list(scenario_results.keys())
    methods = list(scenario_results[scenarios[0]].keys())
    metrics = ['delays', 'energies', 'success_rates']
    metric_names = ['Average Delay (s)', 'Average Energy (J)', 'Success Rate (%)']
    
    fig, axes = plt.subplots(len(scenarios), 3, figsize=(15, 5 * len(scenarios)))
    
    for i, scenario in enumerate(scenarios):
        for j, metric in enumerate(metrics):
            data = []
            labels = []
            
            for method in methods:
                if metric == 'success_rates':
                    # 转换为百分比
                    value = np.mean(scenario_results[scenario][method][metric][-10:]) * 100
                else:
                    value = np.mean(scenario_results[scenario][method][metric][-10:])
                data.append(value)
                labels.append(method)
            
            axes[i, j].bar(labels, data)
            axes[i, j].set_title(f'{scenario} - {metric_names[j]}')
            axes[i, j].set_ylabel(metric_names[j])
            
            # 添加数值标签
            for k, v in enumerate(data):
                axes[i, j].text(k, v, f'{v:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    return fig

def save_results(results, filename):
    """
    保存结果到JSON文件
    
    参数:
        results: 结果字典
        filename: 文件名
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # 将numpy数组转换为列表
    serializable_results = {}
    for scenario, scenario_data in results.items():
        serializable_results[scenario] = {}
        for method, method_data in scenario_data.items():
            serializable_results[scenario][method] = {}
            for metric, values in method_data.items():
                serializable_results[scenario][method][metric] = [float(v) for v in values]
    
    # 保存到文件
    with open(filename, 'w') as f:
        json.dump(serializable_results, f, indent=2)

def load_results(filename):
    """
    从JSON文件加载结果
    
    参数:
        filename: 文件名
    
    返回:
        results: 结果字典
    """
    with open(filename, 'r') as f:
        return json.load(f)
    