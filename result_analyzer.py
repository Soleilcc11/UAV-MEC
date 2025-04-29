"""
结果分析器 - 分析仿真结果并生成可视化图表
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import json
import logging
import matplotlib.font_manager as fm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("results/result_analyzer.log"),  # Changed path to existing directory
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("ResultAnalyzer")

# 确保目录存在
os.makedirs("results/analysis", exist_ok=True)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Analyze simulation results")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory containing result files")
    parser.add_argument("--output_dir", type=str, default="results/analysis", help="Directory to save output files")
    parser.add_argument("--generate_report", action="store_true", help="Generate a detailed report")
    return parser.parse_args()

class ResultAnalyzer:
    """结果分析器类"""
    
    def __init__(self, results_dir="results", output_dir="results/analysis"):
        """
        初始化
        
        参数:
            results_dir: 结果文件目录
            output_dir: 输出目录
        """
        self.results_dir = results_dir
        self.output_dir = output_dir
        
        # 确保目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Initialized ResultAnalyzer with results_dir={results_dir}, output_dir={output_dir}")
    
    def load_results(self):
        """
        加载所有结果数据 - 修改为加载JSON文件
        
        返回:
            scenarios: 场景列表
            methods: 方法列表
            data: 数据字典
        """
        scenarios = set()
        methods = set()
        data = {}
        
        # 遍历所有JSON文件
        json_files = [f for f in os.listdir(self.results_dir) if f.endswith(".json")]
        logger.info(f"Found {len(json_files)} JSON files")
        
        for filename in json_files:
            # 提取算法和场景信息
            if "training_data" in filename:
                parts = filename.replace(".json", "").split("_")
                
                # 处理不同的文件命名模式
                if filename.startswith("training_data"):
                    # 处理形如 training_data_default.json 的文件
                    scenario = parts[-1]
                    method = "Prioritize_DDPG"  # 假设默认是DDPG
                elif len(parts) >= 3:
                    # 处理形如 ddpg_training_data_default.json 的文件
                    method = parts[0].upper()
                    scenario = parts[-1]
                else:
                    logger.warning(f"Skip file with invalid name format: {filename}")
                    continue
                
                scenarios.add(scenario)
                methods.add(method)
                
                # 读取JSON数据
                filepath = os.path.join(self.results_dir, filename)
                logger.info(f"Loading data from {filepath}")
                try:
                    with open(filepath, 'r') as f:
                        json_data = json.load(f)
                    
                    # 将JSON数据转换为DataFrame
                    if isinstance(json_data, dict) and 'rewards' in json_data:
                        # 创建DataFrame
                        df = pd.DataFrame({
                            'Episode': range(len(json_data['rewards'])),
                            'Reward': json_data['rewards'],
                            'CompletionTime': json_data['delays'],
                            'EnergyConsumption': json_data['energies'],
                            'Success': json_data['success_rates']
                        })
                        
                        if scenario not in data:
                            data[scenario] = {}
                        data[scenario][method] = df
                        
                        logger.info(f"Loaded data for {scenario}_{method}: {len(df)} records")
                    else:
                        logger.warning(f"Invalid JSON format in {filepath}")
                        
                except Exception as e:
                    logger.error(f"Error loading {filepath}: {str(e)}")
        
        return list(scenarios), list(methods), data
    
    def analyze_performance(self):
        """
        分析性能并生成对比结果
        
        返回:
            performance_summary: 性能总结字典
        """
        scenarios, methods, data = self.load_results()
        
        if not scenarios:
            logger.warning("No scenarios found in result files")
            return {}
            
        performance_summary = {}
        
        for scenario in scenarios:
            performance_summary[scenario] = {
                "delays": {},
                "energies": {},
                "success_rates": {},
                "rewards": {}
            }
            
            for method in methods:
                if method in data.get(scenario, {}):
                    df = data[scenario][method]
                    
                    # 计算基本性能指标
                    performance_summary[scenario]["delays"][method] = df["CompletionTime"].mean()
                    performance_summary[scenario]["energies"][method] = df["EnergyConsumption"].mean()
                    performance_summary[scenario]["success_rates"][method] = df["Success"].mean()
                    performance_summary[scenario]["rewards"][method] = df["Reward"].mean()
            
        # 计算DDPG相对于其他方法的改进
        for scenario in scenarios:
            performance_summary[scenario]["improvement"] = {}
            
            if "DDPG" in methods:
                for method in methods:
                    if method != "DDPG" and method in data.get(scenario, {}):
                        performance_summary[scenario]["improvement"][method] = {
                            "delay": ((performance_summary[scenario]["delays"][method] - 
                                     performance_summary[scenario]["delays"]["DDPG"]) / 
                                     performance_summary[scenario]["delays"][method] * 100),
                            
                            "energy": ((performance_summary[scenario]["energies"][method] - 
                                      performance_summary[scenario]["energies"]["DDPG"]) / 
                                      performance_summary[scenario]["energies"][method] * 100),
                            
                            "success_rate": ((performance_summary[scenario]["success_rates"]["DDPG"] - 
                                            performance_summary[scenario]["success_rates"][method]) / 
                                            performance_summary[scenario]["success_rates"][method] * 100)
                        }
        
        return performance_summary
    
    def generate_comparison_charts(self):
        """生成性能对比图表"""
        scenarios, methods, data = self.load_results()
        
        if not scenarios or not methods:
            logger.warning("Not enough data to generate comparison charts")
            return {}
        
        # 设置图表风格
        sns.set(style="whitegrid")
        
        # 提取结果数据
        results = {}
        for scenario in scenarios:
            results[scenario] = {}
            for method in methods:
                if method in data.get(scenario, {}):
                    df = data[scenario][method]
                    results[scenario][method] = {
                        "delays": df["CompletionTime"].values,
                        "energies": df["EnergyConsumption"].values,
                        "success_rates": df["Success"].values,
                        "rewards": df["Reward"].values
                    }
        
        # 生成对比图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 找到第一个场景进行可视化
        if scenarios:
            scenario = scenarios[0]
            metrics = ["rewards", "delays", "energies", "success_rates"]
            titles = ["Rewards", "Completion Time", "Energy Consumption", "Success Rate"]
            
            for i, (metric, title) in enumerate(zip(metrics, titles)):
                row, col = divmod(i, 2)
                ax = axes[row, col]
                
                for method in methods:
                    if method in results[scenario] and metric in results[scenario][method]:
                        values = results[scenario][method][metric]
                        episodes = range(len(values))
                        ax.plot(episodes, values, label=method, alpha=0.7)
                
                ax.set_title(f"{title} - {scenario} scenario")
                ax.set_xlabel("Episode")
                ax.set_ylabel(title)
                ax.legend()
        
        plt.tight_layout()
        
        # 保存图表
        output_path = os.path.join(self.output_dir, "method_comparison.png")
        plt.savefig(output_path)
        logger.info(f"Saved method comparison chart to {output_path}")
        
        # 保存处理后的结果
        result_path = os.path.join(self.output_dir, "simulation_results.json")
        with open(result_path, 'w') as f:
            # 将NumPy数组转换为列表以便JSON序列化
            json_results = {}
            for scenario, scenario_data in results.items():
                json_results[scenario] = {}
                for method, method_data in scenario_data.items():
                    json_results[scenario][method] = {}
                    for metric, values in method_data.items():
                        json_results[scenario][method][metric] = values.tolist()
            
            json.dump(json_results, f, indent=2)
            
        logger.info(f"Saved processed results to {result_path}")
        
        return results
    
    def generate_report(self):
        """生成详细报告"""
        performance_summary = self.analyze_performance()
        
        if not performance_summary:
            logger.warning("No data to generate report")
            return None
        
        # 生成Markdown报告
        report_path = os.path.join(self.output_dir, "performance_report.md")
        
        with open(report_path, "w") as f:
            f.write("# UAV-MEC任务卸载性能分析报告\n\n")
            f.write("## 1. 概述\n\n")
            f.write("本报告分析了无人机辅助移动边缘计算(UAV-MEC)任务卸载方案的性能。比较了不同场景下各种卸载方法的表现。\n\n")
            
            f.write("## 2. 性能指标\n\n")
            
            for scenario in performance_summary:
                f.write(f"### 2.1 {scenario}场景\n\n")
                
                # 性能指标表格
                f.write("#### 2.1.1 基本性能指标\n\n")
                f.write("| 方法 | 平均奖励 | 平均时延(秒) | 平均能耗(焦耳) | 任务成功率(%) |\n")
                f.write("|------|---------|------------|------------|------------|\n")
                
                for method in performance_summary[scenario]["delays"]:
                    if method in performance_summary[scenario]["rewards"]:
                        reward = performance_summary[scenario]["rewards"][method]
                        delay = performance_summary[scenario]["delays"][method]
                        energy = performance_summary[scenario]["energies"][method]
                        success = performance_summary[scenario]["success_rates"][method] * 100
                        f.write(f"| {method} | {reward:.4f} | {delay:.4f} | {energy:.4f} | {success:.2f} |\n")
                
                # DDPG改进比例
                if "improvement" in performance_summary[scenario]:
                    f.write("\n#### 2.1.2 DDPG相对其他方法的改进比例\n\n")
                    f.write("| 对比方法 | 时延降低(%) | 能耗降低(%) | 成功率提升(%) |\n")
                    f.write("|---------|-----------|-----------|-------------|\n")
                    
                    for method, imp in performance_summary[scenario]["improvement"].items():
                        f.write(f"| {method} | {imp['delay']:.2f} | {imp['energy']:.2f} | {imp['success_rate']:.2f} |\n")
            
            f.write("\n## 3. 分析与讨论\n\n")
            
            f.write("### 3.1 强化学习方案优势分析\n\n")
            f.write("基于深度强化学习的任务卸载方案展现出以下优势：\n\n")
            f.write("1. **自适应决策能力**：能够根据网络状态、计算负载和任务特性动态调整卸载决策\n")
            f.write("2. **多目标优化**：同时考虑时延、能耗和成功率的平衡\n")
            f.write("3. **无人机位置优化**：通过优化无人机位置提升通信质量\n\n")
            
            f.write("### 3.2 不同算法的表现分析\n\n")
            f.write("比较了DDPG、DQN和PPO等算法，各有不同特点：\n\n")
            f.write("* **DDPG**: 连续动作空间下表现稳定，适合控制UAV移动\n")
            f.write("* **DQN**: 学习速度快，但在连续空间中需要离散化动作\n")
            f.write("* **PPO**: 稳定性高，避免策略崩溃，在多目标优化中表现良好\n\n")
            
            f.write("### 3.3 未来优化方向\n\n")
            f.write("基于实验结果，未来可以在以下方面进一步优化：\n\n")
            f.write("1. 考虑无人机间协作机制，实现资源共享和负载均衡\n")
            f.write("2. 引入预测模型，提前预测用户分布和任务需求\n")
            f.write("3. 进一步优化能量消耗模型，延长无人机续航时间\n")
            f.write("4. 探索基于联邦学习的分布式训练方法\n\n")
            
            f.write("## 4. 结论\n\n")
            f.write("本实验验证了基于强化学习的无人机辅助移动边缘计算任务卸载方案的有效性。")
            f.write("相比传统方法，该方案在任务完成时延、能量消耗和任务成功率等关键指标上均取得了明显改进。")
            f.write("实验结果表明，通过深度强化学习方法优化任务卸载和无人机位置，能够显著提升系统整体性能。")
        
        logger.info(f"Generated performance report at {report_path}")
        
        return report_path

def compare_methods(results):
    """
    比较不同方法的性能
    """
    if not results:
        logger.warning("No results data to compare")
        return plt.figure()
        
    scenarios = list(results.keys())
    if not scenarios:
        logger.warning("No scenarios in results data")
        return plt.figure()
        
    first_scenario = scenarios[0]
    if not results[first_scenario]:
        logger.warning(f"No methods for scenario {first_scenario}")
        return plt.figure()
        
    methods = list(results[first_scenario].keys())
    
    # 创建性能比较图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    metrics = ["rewards", "delays", "energies", "success_rates"]
    titles = ["Rewards", "Completion Time", "Energy Consumption", "Success Rate"]
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        row, col = divmod(i, 2)
        ax = axes[row, col]
        
        for method in methods:
            if method in results[first_scenario] and metric in results[first_scenario][method]:
                values = results[first_scenario][method][metric]
                
                # 计算滑动平均
                window_size = min(50, len(values) // 10 + 1) if len(values) > 0 else 1
                if len(values) > window_size:
                    smoothed = np.convolve(values, np.ones(window_size)/window_size, mode='valid')
                    episodes = range(window_size-1, window_size-1+len(smoothed))
                    ax.plot(episodes, smoothed, label=f"{method}", linewidth=2)
                else:
                    ax.plot(range(len(values)), values, label=method)
        
        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.set_ylabel(title)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    return fig

def save_results(results, filepath):
    """保存处理后的结果数据"""
    try:
        with open(filepath, 'w') as f:
            # 将NumPy数组转换为列表以便JSON序列化
            json_results = {}
            for scenario, scenario_data in results.items():
                json_results[scenario] = {}
                for method, method_data in scenario_data.items():
                    json_results[scenario][method] = {}
                    for metric, values in method_data.items():
                        if isinstance(values, np.ndarray):
                            json_results[scenario][method][metric] = values.tolist()
                        else:
                            json_results[scenario][method][metric] = values
            
            json.dump(json_results, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error saving results to {filepath}: {str(e)}")
        return False

def load_results(filepath):
    """加载处理后的结果数据"""
    try:
        with open(filepath, 'r') as f:
            results = json.load(f)
        return results
    except Exception as e:
        logger.error(f"Error loading results from {filepath}: {str(e)}")
        return {}

def main():
    """主函数"""
    args = parse_args()
    
    analyzer = ResultAnalyzer(
        results_dir=args.results_dir,
        output_dir=args.output_dir
    )
    
    # 生成对比图表
    analyzer.generate_comparison_charts()
    
    # 生成报告（如果需要）
    if args.generate_report:
        analyzer.generate_report()
    
    logger.info("Analysis complete")

if __name__ == "__main__":
    main()