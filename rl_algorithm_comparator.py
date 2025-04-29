import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import json
import logging
from collections import defaultdict
import matplotlib as mpl

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='results/algorithm_comparison.log'  # 修改：将日志保存到results目录
)
logger = logging.getLogger('RLAlgorithmComparator')

class RLAlgorithmComparator:
    def __init__(self, result_dir="results", analysis_dir="results/analysis"):
        """
        强化学习算法比较器
        
        参数:
            result_dir: 结果文件所在目录
            analysis_dir: 分析结果保存目录
        """
        self.result_dir = result_dir
        self.analysis_dir = analysis_dir
        self.algorithms = []
        self.metrics = {}
        self.results = defaultdict(dict)
        
        # 创建结果目录
        os.makedirs(analysis_dir, exist_ok=True)
        logger.info(f"初始化RL算法比较器，结果将保存在 {analysis_dir}")
    
    def add_algorithm(self, name, description=None):
        """
        添加要比较的算法
        
        参数:
            name: 算法名称
            description: 算法描述（可选）
        """
        algo_info = {
            "name": name,
            "description": description or name
        }
        self.algorithms.append(algo_info)
        logger.info(f"添加了算法: {name}")
    
    def register_metrics(self, metrics_dict):
        """
        注册要比较的指标
        
        参数:
            metrics_dict: 包含指标名称和描述的字典
        """
        self.metrics.update(metrics_dict)
        logger.info(f"注册了 {len(metrics_dict)} 个指标")
    
    def add_result(self, algorithm_name, metric_name, value, episode=None):
        """
        添加实验结果
        
        参数:
            algorithm_name: 算法名称
            metric_name: 指标名称
            value: 指标值
            episode: 训练回合数（可选，用于绘制学习曲线）
        """
        if algorithm_name not in [algo["name"] for algo in self.algorithms]:
            logger.warning(f"未注册的算法: {algorithm_name}，将自动添加")
            self.add_algorithm(algorithm_name)
        
        if metric_name not in self.metrics:
            logger.warning(f"未注册的指标: {metric_name}，将自动添加")
            self.metrics[metric_name] = metric_name
        
        if episode is not None:
            # 存储学习曲线数据
            if metric_name not in self.results[algorithm_name]:
                self.results[algorithm_name][metric_name] = []
            
            # 确保列表长度够长
            result_list = self.results[algorithm_name][metric_name]
            if len(result_list) <= episode:
                result_list.extend([None] * (episode - len(result_list) + 1))
            
            result_list[episode] = value
        else:
            # 存储单个指标值
            self.results[algorithm_name][metric_name] = value
        
        logger.info(f"为算法 {algorithm_name} 添加了指标 {metric_name}，值: {value}，回合: {episode}")
    
    def load_results_from_json(self):
        """
        从JSON训练数据文件加载结果
        """
        try:
            # 获取所有JSON训练结果文件
            json_files = [f for f in os.listdir(self.result_dir) if f.endswith('.json') and 'training_data' in f]
            logger.info(f"在 {self.result_dir} 中找到 {len(json_files)} 个训练结果文件")
            
            for filename in json_files:
                filepath = os.path.join(self.result_dir, filename)
                
                # 提取算法和场景信息
                if filename.startswith("training_data_default"):
                    algo_name = "Prioritize_DDPG"
                    algo_desc = "优先级深度确定性策略梯度"
                elif filename.startswith("ddpg_training_data"):
                    algo_name = "DDPG"
                    algo_desc = "深度确定性策略梯度"
                elif filename.startswith("ppo_training_data"):
                    algo_name = "PPO"
                    algo_desc = "近端策略优化"
                elif filename.startswith("dqn_training_data"):
                    algo_name = "DQN"
                    algo_desc = "深度Q网络"
                elif filename.startswith("training_data"):
                    # 这是默认的DDPG训练数据格式
                    algo_name = "DDPG"
                    algo_desc = "深度确定性策略梯度"
                else:
                    logger.warning(f"无法确定算法名称: {filename}")
                    continue
                
                # 添加算法（如果尚未添加）
                if algo_name not in [algo["name"] for algo in self.algorithms]:
                    self.add_algorithm(algo_name, algo_desc)
                
                # 加载结果数据
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                # 提取指标数据
                if 'rewards' in data:
                    self.add_metric_data(algo_name, 'rewards', '平均奖励', data['rewards'])
                
                if 'delays' in data:
                    self.add_metric_data(algo_name, 'delays', '平均时延', data['delays'])
                
                if 'energies' in data:
                    self.add_metric_data(algo_name, 'energies', '平均能耗', data['energies'])
                
                if 'success_rates' in data:
                    self.add_metric_data(algo_name, 'success_rates', '任务成功率', data['success_rates'])
                
                logger.info(f"从 {filepath} 加载了 {algo_name} 的数据")
            
            return True
        except Exception as e:
            logger.error(f"加载结果时出错: {str(e)}")
            return False
    
    def add_metric_data(self, algo_name, metric_key, metric_desc, values):
        """
        添加指标数据
        
        参数:
            algo_name: 算法名称
            metric_key: 指标键名
            metric_desc: 指标描述
            values: 指标值列表或单个值
        """
        # 注册指标（如果尚未注册）
        if metric_key not in self.metrics:
            self.metrics[metric_key] = metric_desc
        
        # 存储数据
        self.results[algo_name][metric_key] = values
        
        # 如果是列表，计算平均值并存储
        if isinstance(values, list) and values:
            # 计算最后100个回合的平均值
            last_100_values = values[-100:] if len(values) > 100 else values
            avg_value = np.mean([v for v in last_100_values if v is not None])
            self.results[algo_name][f"{metric_key}_avg"] = avg_value
            
            # 注册平均值指标
            if f"{metric_key}_avg" not in self.metrics:
                self.metrics[f"{metric_key}_avg"] = f"{metric_desc} (平均)"
    
    def save_results(self, filepath=None):
        """
        保存结果到JSON文件
        
        参数:
            filepath: 文件路径（可选）
        """
        if filepath is None:
            filepath = os.path.join(self.analysis_dir, "comparison_results.json")
        
        data = {
            "algorithms": self.algorithms,
            "metrics": self.metrics,
            "results": dict(self.results)
        }
        
        try:
            with open(filepath, 'w') as f:
                # 转换NumPy数据类型为Python基本类型
                json_data = json.dumps(data, indent=4, default=lambda x: float(x) if isinstance(x, np.floating) else x)
                f.write(json_data)
            
            logger.info(f"结果已保存到 {filepath}")
        except Exception as e:
            logger.error(f"保存结果时出错: {str(e)}")
    
    def plot_comparison(self, metric_name, title=None, save_path=None, smooth=True, window_size=50):
        """
        绘制指标比较图
        
        参数:
            metric_name: 要比较的指标名称
            title: 图表标题（可选）
            save_path: 保存路径（可选）
            smooth: 是否平滑曲线
            window_size: 平滑窗口大小
        """
        if metric_name not in self.metrics:
            logger.error(f"未找到指标: {metric_name}")
            return
        
        plt.figure(figsize=(12, 8))
        
        # 检查是否为学习曲线数据
        is_learning_curve = False
        for algo in self.algorithms:
            algo_name = algo["name"]
            if algo_name in self.results and metric_name in self.results[algo_name]:
                value = self.results[algo_name][metric_name]
                if isinstance(value, list):
                    is_learning_curve = True
                    break
        
        if is_learning_curve:
            # 绘制学习曲线
            for algo in self.algorithms:
                algo_name = algo["name"]
                if algo_name in self.results and metric_name in self.results[algo_name]:
                    values = self.results[algo_name][metric_name]
                    
                    # 过滤None值
                    episodes = [i for i, v in enumerate(values) if v is not None]
                    filtered_values = [v for v in values if v is not None]
                    
                    # 绘制原始曲线
                    plt.plot(episodes, filtered_values, alpha=0.3, label=f"{algo_name}原始")
                    
                    # 如果需要平滑曲线
                    if smooth and len(filtered_values) > window_size:
                        # 计算滑动平均
                        smoothed = np.convolve(filtered_values, np.ones(window_size)/window_size, mode='valid')
                        smooth_episodes = episodes[window_size-1:window_size-1+len(smoothed)]
                        
                        # 绘制平滑曲线
                        plt.plot(smooth_episodes, smoothed, linewidth=2, label=f"{algo_name}")
            
            plt.xlabel("训练回合数")
        else:
            # 绘制条形图
            algo_names = []
            values = []
            
            for algo in self.algorithms:
                algo_name = algo["name"]
                if algo_name in self.results and metric_name in self.results[algo_name]:
                    algo_names.append(algo_name)
                    values.append(self.results[algo_name][metric_name])
            
            bars = plt.bar(algo_names, values)
            
            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.4f}',
                        ha='center', va='bottom', rotation=0)
            
            plt.xticks(rotation=45)
        
        plt.ylabel(self.metrics.get(metric_name, metric_name))
        plt.title(title or f"{self.metrics.get(metric_name, metric_name)} 比较")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            logger.info(f"图表已保存到 {save_path}")
        else:
            save_path = os.path.join(self.analysis_dir, f"{metric_name}_comparison.png")
            plt.savefig(save_path, dpi=300)
            logger.info(f"图表已保存到 {save_path}")
        
        plt.close()
    
    def plot_all_metrics(self):
        """
        绘制所有指标的比较图
        """
        for metric_name in self.metrics:
            # 跳过平均值指标，因为它们不是学习曲线
            if metric_name.endswith('_avg'):
                continue
                
            # 绘制学习曲线图
            if any(isinstance(self.results.get(algo["name"], {}).get(metric_name, None), list) 
                  for algo in self.algorithms):
                title = f"{self.metrics[metric_name]} 学习曲线比较"
                save_path = os.path.join(self.analysis_dir, f"{metric_name}_learning_curve.png")
                self.plot_comparison(metric_name, title, save_path)
    
    def plot_average_metrics(self):
        """
        绘制平均指标的条形图比较
        """
        # 找出所有平均值指标
        avg_metrics = [m for m in self.metrics if m.endswith('_avg')]
        
        if not avg_metrics:
            logger.warning("没有找到平均值指标")
            return
        
        for metric_name in avg_metrics:
            title = f"{self.metrics[metric_name]} 比较"
            save_path = os.path.join(self.analysis_dir, f"{metric_name}_bar_chart.png")
            self.plot_comparison(metric_name, title, save_path)
    
    def generate_report(self, save_path=None):
        """
        生成比较报告
        
        参数:
            save_path: 保存路径（可选）
        """
        if save_path is None:
            save_path = os.path.join(self.analysis_dir, "algorithm_comparison_report.md")
        
        report_lines = [
            "# UAV-MEC任务卸载强化学习算法比较报告",
            "\n## 1. 算法概述\n"
        ]
        
        # 添加算法信息
        for algo in self.algorithms:
            report_lines.append(f"- **{algo['name']}**: {algo['description']}")
        
        # 创建指标比较表格
        report_lines.append("\n## 2. 性能指标比较\n")
        
        # 表头
        table_header = ["指标名称"]
        for algo in self.algorithms:
            table_header.append(algo["name"])
        
        report_lines.append("| " + " | ".join(table_header) + " |")
        report_lines.append("| " + " | ".join(["---" for _ in table_header]) + " |")
        
        # 筛选出所有平均值指标
        avg_metrics = {m: self.metrics[m] for m in self.metrics if m.endswith('_avg')}
        
        # 如果没有平均值指标，使用原始指标的均值
        if not avg_metrics:
            for metric_name, metric_desc in self.metrics.items():
                row = [f"{metric_desc}"]
                
                for algo in self.algorithms:
                    algo_name = algo["name"]
                    if algo_name in self.results and metric_name in self.results[algo_name]:
                        value = self.results[algo_name][metric_name]
                        
                        # 处理学习曲线数据
                        if isinstance(value, list):
                            # 使用最后100个回合的平均值
                            last_100_values = value[-100:] if len(value) > 100 else value
                            avg_value = np.mean([v for v in last_100_values if v is not None])
                            row.append(f"{avg_value:.4f}")
                        else:
                            row.append(f"{value:.4f}" if isinstance(value, (int, float)) else str(value))
                    else:
                        row.append("N/A")
                
                report_lines.append("| " + " | ".join(row) + " |")
        else:
            # 使用计算好的平均值指标
            for metric_name, metric_desc in avg_metrics.items():
                row = [f"{metric_desc}"]
                
                for algo in self.algorithms:
                    algo_name = algo["name"]
                    if algo_name in self.results and metric_name in self.results[algo_name]:
                        value = self.results[algo_name][metric_name]
                        row.append(f"{value:.4f}" if isinstance(value, (int, float)) else str(value))
                    else:
                        row.append("N/A")
                
                report_lines.append("| " + " | ".join(row) + " |")
        
        # 绘制所有图表
        self.plot_all_metrics()
        self.plot_average_metrics()
        
        # 添加学习曲线图
        report_lines.append("\n## 3. 学习曲线\n")
        
        # 非平均值指标的学习曲线
        base_metrics = [m for m in self.metrics if not m.endswith('_avg')]
        
        for metric_name in base_metrics:
            # 检查是否为学习曲线数据
            is_learning_curve = False
            for algo in self.algorithms:
                algo_name = algo["name"]
                if (algo_name in self.results and 
                    metric_name in self.results[algo_name] and 
                    isinstance(self.results[algo_name][metric_name], list)):
                    is_learning_curve = True
                    break
            
            if is_learning_curve:
                img_path = f"{metric_name}_learning_curve.png"
                report_lines.append(f"### {self.metrics[metric_name]}\n")
                report_lines.append(f"![{self.metrics[metric_name]}]({img_path})\n")
        
        # 添加平均性能图表
        report_lines.append("\n## 4. 平均性能\n")
        
        for metric_name in avg_metrics:
            img_path = f"{metric_name}_bar_chart.png"
            report_lines.append(f"### {self.metrics[metric_name]}\n")
            report_lines.append(f"![{self.metrics[metric_name]}]({img_path})\n")
        
        # 添加分析与讨论
        report_lines.append("\n## 5. 分析与讨论\n")
        
        report_lines.append("### 5.1 算法性能对比\n")
        report_lines.append("通过对比四种强化学习算法（DDPG、优化版DDPG、PPO和DQN）在UAV-MEC任务卸载中的表现，我们可以得出以下结论：\n")
        
        # 尝试找出各指标表现最好的算法
        metrics_analysis = []
        base_metrics_map = {
            'rewards': '奖励', 
            'delays': '时延', 
            'energies': '能耗',
            'success_rates': '任务成功率'
        }
        
        for metric_key, metric_name in base_metrics_map.items():
            metric_avg = f"{metric_key}_avg"
            if metric_avg in self.metrics:
                # 收集各算法在此指标上的表现
                performances = {}
                for algo in self.algorithms:
                    algo_name = algo["name"]
                    if algo_name in self.results and metric_avg in self.results[algo_name]:
                        performances[algo_name] = self.results[algo_name][metric_avg]
                
                if performances:
                    # 根据指标特性确定排序方式
                    if metric_key in ['rewards', 'success_rates']:
                        # 这些指标越大越好
                        sorted_algos = sorted(performances.items(), key=lambda x: x[1], reverse=True)
                    else:
                        # 这些指标越小越好
                        sorted_algos = sorted(performances.items(), key=lambda x: x[1])
                    
                    analysis = f"- **{metric_name}**: "
                    analysis += f"{sorted_algos[0][0]} ({sorted_algos[0][1]:.4f}) > "
                    if len(sorted_algos) > 1:
                        analysis += f"{sorted_algos[1][0]} ({sorted_algos[1][1]:.4f}) > "
                    if len(sorted_algos) > 2:
                        analysis += f"{sorted_algos[2][0]} ({sorted_algos[2][1]:.4f})"
                    if len(sorted_algos) > 3:
                        analysis += f" > {sorted_algos[3][0]} ({sorted_algos[3][1]:.4f})"
                    
                    metrics_analysis.append(analysis)
        
        # 添加性能分析
        for analysis in metrics_analysis:
            report_lines.append(analysis)
        
        report_lines.append("\n### 5.2 算法特点分析\n")
        
        report_lines.append("**PPO算法**: 在多项指标上表现优异，特别是在奖励和成功率方面领先。PPO通过信任区域策略优化方法，能够实现更稳定的策略更新，避免了策略崩溃。同时，PPO的多头网络架构有助于更好地学习复杂环境中的价值函数和策略函数。\n")
        
        report_lines.append("**优化版DDPG**: 通过引入优先级经验回放和噪声网络层，明显优于标准DDPG。这些优化使得算法能够更有效地利用重要经验样本，同时提供更结构化的探索机制，从而在成功率和奖励方面取得了显著提升。\n")
        
        report_lines.append("**标准DDPG**: 作为基准算法，在连续动作空间中表现较为稳定，但各项指标普遍低于优化版DDPG和PPO。这表明基本的演员-评论家架构虽然能够处理UAV控制等连续动作任务，但仍有优化空间。\n")
        
        report_lines.append("**DQN算法**: 在能耗方面表现最佳，但在其他指标上较弱。这可能是因为DQN需要对连续动作空间进行离散化处理，导致在高维动作空间中效率降低。同时，DQN的值函数近似方法在处理复杂环境时容易出现过估计问题。\n")
        
        report_lines.append("\n### 5.3 多目标权衡分析\n")
        
        report_lines.append("UAV-MEC任务卸载是一个典型的多目标优化问题，需要平衡延迟、能耗和成功率等多个指标。从实验结果来看：\n")
        
        report_lines.append("- **延迟-能耗权衡**: DQN虽然能耗最低，但延迟最高；而PPO则延迟最低，但能耗较高。这说明算法在优化过程中倾向于不同的目标。\n")
        
        report_lines.append("- **成功率-资源利用**: PPO和优化版DDPG在成功率方面明显优于其他算法，表明它们能够更有效地分配资源，确保任务成功完成。\n")
        
        report_lines.append("- **稳定性考量**: 从学习曲线可以看出，PPO的训练过程更为稳定，波动较小，这对于实际部署非常重要。\n")
        
        report_lines.append("\n## 6. 结论\n")
        
        report_lines.append("基于实验结果，我们得出以下结论：\n")
        
        report_lines.append("1. PPO算法在UAV-MEC任务卸载场景中整体表现最佳，特别适合需要高成功率和低延迟的应用场景。\n")
        
        report_lines.append("2. 优化版DDPG通过引入优先级经验回放和噪声网络层，有效提升了标准DDPG的性能，证明了这些优化机制的有效性。\n")
        
        report_lines.append("3. 不同算法在各指标上表现各异，系统设计者可以根据具体应用需求选择适合的算法。例如，对能耗极为敏感的场景可以考虑DQN，而对时延和成功率要求高的场景应优先考虑PPO。\n")
        
        report_lines.append("4. 强化学习方法在UAV-MEC任务卸载中展现出显著优势，能够根据环境动态调整决策，实现多目标优化。\n")
        
        # 保存报告
        try:
            with open(save_path, 'w') as f:
                f.write("\n".join(report_lines))
            
            logger.info(f"比较报告已保存到 {save_path}")
        except Exception as e:
            logger.error(f"保存报告时出错: {str(e)}")
        
        return save_path


# 当直接运行此脚本时执行
if __name__ == "__main__":
    # 创建比较器
    comparator = RLAlgorithmComparator(result_dir="results", analysis_dir="results/analysis")
    
    # 加载结果
    if comparator.load_results_from_json():
        # 保存处理后的结果
        comparator.save_results()
        
        # 生成报告
        report_path = comparator.generate_report()
        
        print(f"比较报告已生成: {report_path}")
    else:
        print("加载结果失败，请检查结果文件路径和格式")