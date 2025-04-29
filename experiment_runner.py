import os
import time
import json
import logging
import argparse
import subprocess
import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='experiment_runner.log'
)
logger = logging.getLogger('ExperimentRunner')

class ExperimentRunner:
    def __init__(self, config_path=None, output_dir="experiment_results"):
        """
        实验运行器，用于自动化运行多种算法的模拟实验
        
        参数:
            config_path: 配置文件路径
            output_dir: 实验结果输出目录
        """
        self.output_dir = output_dir
        self.config = {}
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results = defaultdict(dict)
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 如果提供了配置文件，则加载配置
        if config_path:
            self.load_config(config_path)
        
        logger.info(f"初始化实验运行器，实验ID: {self.experiment_id}")
    
    def load_config(self, config_path):
        """
        加载实验配置
        
        参数:
            config_path: 配置文件路径
        """
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            
            logger.info(f"从 {config_path} 加载配置成功")
        except Exception as e:
            logger.error(f"加载配置时出错: {str(e)}")
            raise
    
    def set_config(self, config):
        """
        设置实验配置
        
        参数:
            config: 配置字典
        """
        self.config = config
        logger.info("手动设置配置成功")
    
    def add_algorithm(self, name, params=None):
        """
        添加要测试的算法
        
        参数:
            name: 算法名称
            params: 算法参数（可选）
        """
        if "algorithms" not in self.config:
            self.config["algorithms"] = []
        
        algo_config = {
            "name": name,
            "params": params or {}
        }
        
        self.config["algorithms"].append(algo_config)
        logger.info(f"添加算法: {name}，参数: {params}")
    
    def add_scenario(self, name, params=None):
        """
        添加要测试的场景
        
        参数:
            name: 场景名称
            params: 场景参数（可选）
        """
        if "scenarios" not in self.config:
            self.config["scenarios"] = []
        
        scenario_config = {
            "name": name,
            "params": params or {}
        }
        
        self.config["scenarios"].append(scenario_config)
        logger.info(f"添加场景: {name}，参数: {params}")
    
    def prepare_experiment(self, max_tasks=10000, sim_time=300):
        """
        准备实验配置
        
        参数:
            max_tasks: 最大任务数
            sim_time: 模拟时间（秒）
        """
        if "experiment" not in self.config:
            self.config["experiment"] = {}
        
        self.config["experiment"]["max_tasks"] = max_tasks
        self.config["experiment"]["sim_time"] = sim_time
        self.config["experiment"]["id"] = self.experiment_id
        
        logger.info(f"准备实验: 最大任务数={max_tasks}, 模拟时间={sim_time}秒")
    
    def run_all_experiments(self, repeats=1):
        """
        运行所有配置的实验
        
        参数:
            repeats: 每个配置重复运行的次数
        """
        if not self.config:
            logger.error("没有实验配置，请先加载或设置配置")
            return
        
        if "algorithms" not in self.config or not self.config["algorithms"]:
            logger.error("没有配置算法")
            return
        
        if "scenarios" not in self.config or not self.config["scenarios"]:
            logger.error("没有配置场景")
            return
        
        logger.info(f"开始运行所有实验，共 {len(self.config['algorithms'])} 个算法，"
                   f"{len(self.config['scenarios'])} 个场景，每个重复 {repeats} 次")
        
        # 保存实验配置
        config_path = os.path.join(self.output_dir, f"config_{self.experiment_id}.json")
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=4)
        
        # 运行所有配置组合
        for algo_config in self.config["algorithms"]:
            algo_name = algo_config["name"]
            algo_params = algo_config["params"]
            
            for scenario_config in self.config["scenarios"]:
                scenario_name = scenario_config["name"]
                scenario_params = scenario_config["params"]
                
                for repeat in range(1, repeats + 1):
                    logger.info(f"运行实验: 算法={algo_name}, 场景={scenario_name}, 重复={repeat}/{repeats}")
                    
                    # 创建当前实验的输出目录
                    exp_output_dir = os.path.join(
                        self.output_dir, 
                        f"{algo_name}_{scenario_name}_{repeat}"
                    )
                    os.makedirs(exp_output_dir, exist_ok=True)
                    
                    # 运行单个实验
                    result = self.run_single_experiment(
                        algo_name, algo_params,
                        scenario_name, scenario_params,
                        exp_output_dir
                    )
                    
                    # 存储结果
                    key = f"{algo_name}_{scenario_name}_{repeat}"
                    self.results[key] = result
        
        # 分析并保存所有实验结果
        self.analyze_results()
        
        logger.info("所有实验运行完成")
    
    def run_single_experiment(self, algo_name, algo_params,
                             scenario_name, scenario_params,
                             output_dir):
        """
        运行单个实验配置
        
        参数:
            algo_name: 算法名称
            algo_params: 算法参数
            scenario_name: 场景名称
            scenario_params: 场景参数
            output_dir: 输出目录
        
        返回:
            result: 实验结果字典
        """
        # 记录开始时间
        start_time = time.time()
        
        try:
            # 创建实验配置文件
            exp_config = {
                "algorithm": {
                    "name": algo_name,
                    "params": algo_params
                },
                "scenario": {
                    "name": scenario_name,
                    "params": scenario_params
                },
                "experiment": self.config.get("experiment", {})
            }
            
            config_file = os.path.join(output_dir, "config.json")
            with open(config_file, 'w') as f:
                json.dump(exp_config, f, indent=4)
            
            # 准备启动Java模拟器的命令
            main_class = "UAVMECMainApp"
            classpath = ".:./bin:./lib/*"  # 假设二进制文件和依赖库的位置
            
            # 构建Java命令行参数
            java_args = [
                f"-Dalgorithm={algo_name}",
                f"-Dscenario={scenario_name}",
                f"-Dconfig={config_file}",
                f"-Doutput={output_dir}"
            ]
            
            # 将算法参数转换为Java命令行参数
            for param_name, param_value in algo_params.items():
                java_args.append(f"-D{param_name}={param_value}")
            
            # 将场景参数转换为Java命令行参数
            for param_name, param_value in scenario_params.items():
                java_args.append(f"-D{param_name}={param_value}")
            
            # 构建完整命令
            command = ["java", "-cp", classpath] + java_args + [main_class]
            
            # 执行命令
            logger.info(f"执行命令: {' '.join(command)}")
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # 捕获输出
            stdout, stderr = process.communicate()
            
            # 检查执行状态
            if process.returncode != 0:
                logger.error(f"实验执行失败，错误码: {process.returncode}")
                logger.error(f"错误输出: {stderr}")
                return {"status": "failed", "error": stderr}
            
            # 记录结束时间
            end_time = time.time()
            execution_time = end_time - start_time
            
            # 尝试解析输出结果（假设以JSON格式输出）
            result = {"status": "success", "execution_time": execution_time}
            
            # 解析性能指标结果文件
            metrics_file = os.path.join(output_dir, "metrics.json")
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                result["metrics"] = metrics
            
            logger.info(f"实验运行成功，耗时: {execution_time:.2f}秒")
            return result
            
        except Exception as e:
            logger.error(f"运行实验时出错: {str(e)}")
            return {"status": "failed", "error": str(e)}
    
    def analyze_results(self):
        """分析所有实验结果并生成报告"""
        if not self.results:
            logger.warning("没有要分析的结果")
            return
        
        try:
            # 准备数据
            analysis_data = []
            
            for exp_key, result in self.results.items():
                if result.get("status") != "success":
                    continue
                
                # 分解实验键
                parts = exp_key.split('_')
                algo_name = parts[0]
                scenario_name = parts[1]
                repeat = parts[2]
                
                # 提取指标
                metrics = result.get("metrics", {})
                
                # 创建数据行
                row = {
                    "algorithm": algo_name,
                    "scenario": scenario_name,
                    "repeat": repeat,
                    "execution_time": result.get("execution_time", 0)
                }
                
                # 添加所有指标
                for metric_name, metric_value in metrics.items():
                    row[metric_name] = metric_value
                
                analysis_data.append(row)
            
            # 创建数据框
            if not analysis_data:
                logger.warning("没有成功的实验结果可供分析")
                return
            
            df = pd.DataFrame(analysis_data)
            
            # 按算法和场景分组，计算每个指标的平均值和标准差
            grouped = df.groupby(["algorithm", "scenario"])
            summary = grouped.agg([np.mean, np.std])
            
            # 保存结果
            results_file = os.path.join(self.output_dir, f"results_{self.experiment_id}.csv")
            df.to_csv(results_file, index=False)
            
            summary_file = os.path.join(self.output_dir, f"summary_{self.experiment_id}.csv")
            summary.to_csv(summary_file)
            
            # 创建简单的Markdown报告
            report_lines = [
                "# 实验结果报告",
                f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"\n实验ID: {self.experiment_id}",
                "\n## 实验概述",
                f"\n总实验数: {len(self.results)}",
                f"成功实验数: {len(analysis_data)}",
                f"失败实验数: {len(self.results) - len(analysis_data)}",
                "\n## 算法性能比较\n"
            ]
            
            # 添加性能指标表格
            metric_columns = [col for col in df.columns if col not in ["algorithm", "scenario", "repeat", "execution_time"]]
            
            if metric_columns:
                # 计算每个算法的平均性能
                algo_performance = df.groupby("algorithm")[metric_columns].mean()
                
                # 添加到报告
                report_lines.append("### 各算法平均性能\n")
                report_lines.append(algo_performance.to_markdown())
                
                # 在不同场景下的性能比较
                report_lines.append("\n### 各算法在不同场景下的性能\n")
                
                for metric in metric_columns:
                    pivot = df.pivot_table(
                        values=metric, 
                        index="scenario", 
                        columns="algorithm", 
                        aggfunc=np.mean
                    )
                    
                    report_lines.append(f"\n#### {metric}\n")
                    report_lines.append(pivot.to_markdown())
            
            # 保存报告
            report_file = os.path.join(self.output_dir, f"report_{self.experiment_id}.md")
            with open(report_file, 'w') as f:
                f.write("\n".join(report_lines))
            
            logger.info(f"结果分析完成，报告保存到 {report_file}")
            
        except Exception as e:
            logger.error(f"分析结果时出错: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="边缘云模拟实验运行器")
    parser.add_argument("--config", help="配置文件路径")
    parser.add_argument("--output", default="experiment_results", help="输出目录")
    parser.add_argument("--repeats", type=int, default=3, help="每个配置重复次数")
    args = parser.parse_args()
    
    runner = ExperimentRunner(config_path=args.config, output_dir=args.output)
    
    # 如果没有提供配置文件，设置默认配置
    if not args.config:
        # 添加算法
        runner.add_algorithm("ddpg", {"lr": 0.001, "batch_size": 64})
        runner.add_algorithm("prioritized_ddpg", {"lr": 0.001, "batch_size": 64})
        runner.add_algorithm("ppo", {"lr": 0.0003, "batch_size": 64})
        runner.add_algorithm("dqn", {"lr": 0.001, "batch_size": 64})
        
        # 添加场景
        runner.add_scenario("urban", {"uav_count": 5, "user_density": "high"})
        runner.add_scenario("rural", {"uav_count": 3, "user_density": "low"})
        
        # 准备实验
        runner.prepare_experiment(max_tasks=5000, sim_time=300)
    
    # 运行所有实验
    runner.run_all_experiments(repeats=args.repeats)