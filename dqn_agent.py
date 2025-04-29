import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque
import logging
import time
import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


logger = logging.getLogger('DQNAgent')

class ReplayBuffer:
    def __init__(self, capacity=10000):
        """经验回放缓冲区"""
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        """添加经验到缓冲区"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """随机采样一批经验"""
        batch = random.sample(self.buffer, min(len(self.buffer), batch_size))
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)
    
    def size(self):
        """返回缓冲区当前大小"""
        return len(self.buffer)

class DQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        """DQN网络结构"""
        super(DQNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_dim)
    
    def forward(self, x):
        """前向传播"""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_dim=12, action_dim=3, lr=0.001, gamma=0.99, 
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, 
                 update_target_freq=10, batch_size=64):
        """
        DQN代理初始化
        
        参数:
            state_dim: 状态空间维度
            action_dim: 动作空间维度
            lr: 学习率
            gamma: 折扣因子
            epsilon_start: 起始探索率
            epsilon_end: 最小探索率
            epsilon_decay: 探索率衰减因子
            update_target_freq: 目标网络更新频率
            batch_size: 批量大小
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.update_target_freq = update_target_freq
        self.batch_size = batch_size
        self.train_step = 0
        
        # 创建在线网络和目标网络
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = DQNetwork(state_dim, action_dim).to(self.device)
        self.target_network = DQNetwork(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # 设置优化器
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # 创建经验回放缓冲区
        self.replay_buffer = ReplayBuffer()
        
        # 添加训练摘要收集
        self.train_summaries = []
        
        logger.info(f"DQN代理在 {self.device} 上初始化，状态维度: {state_dim}，动作维度: {action_dim}")
    
    def get_action(self, state, deterministic=False):
        """
        选择动作，使用ε-贪婪策略（探索与利用的平衡）
        
        参数:
            state: 当前状态
            deterministic: 是否确定性选择动作（用于评估）
        
        返回:
            selected_action: 选择的动作
        """
        # 转换状态为张量
        if isinstance(state, list):
            state = np.array(state)
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        
        # 确定性动作选择（用于评估）或探索率小于阈值时
        if deterministic or random.random() > self.epsilon:
            with torch.no_grad():
                q_values = self.q_network(state)
                action_idx = torch.argmax(q_values, dim=1).item()
                
                # 将离散动作转换为连续值，以与其他算法接口兼容
                # 假设动作空间为[-1, 1]的连续空间
                action = np.zeros(self.action_dim)
                action[action_idx] = 1.0
                return action
        else:
            # 随机探索
            action_idx = random.randint(0, self.action_dim - 1)
            action = np.zeros(self.action_dim)
            action[action_idx] = 1.0
            return action
    
    def update(self, state, action, reward, next_state, done):
        """
        更新DQN代理
        
        参数:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否结束
        """
        # 转换数据类型
        if isinstance(state, list):
            state = np.array(state)
        if isinstance(action, list):
            action = np.array(action)
        if isinstance(next_state, list):
            next_state = np.array(next_state)
        
        # 将连续动作转换为离散动作索引（兼容接口）
        action_idx = np.argmax(action)
        
        # 向经验回放缓冲区添加经验
        self.replay_buffer.add(
            state.reshape(-1),
            action_idx,
            reward,
            next_state.reshape(-1),
            done
        )
        
        # 如果缓冲区大小足够，则执行经验回放学习
        if self.replay_buffer.size() > self.batch_size:
            train_info = self._learn()
            
            # 衰减探索率
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            return train_info
        return None

    def select_action(self, state, deterministic=False):
        return self.get_action(state, deterministic)
    
    
    def _learn(self):
        """从经验回放缓冲区中学习"""
        # 采样一批经验
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # 转换为张量
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # 计算当前Q值
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # 计算目标Q值（使用目标网络）
        with torch.no_grad():
            max_next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values
        
        # 计算损失
        loss = F.smooth_l1_loss(current_q_values.squeeze(), target_q_values)
        
        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # 定期更新目标网络
        self.train_step += 1
        if self.train_step % self.update_target_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            logger.info(f"目标网络已更新，训练步数: {self.train_step}")
        
        # 收集训练摘要
        train_info = {
            'loss': loss.item(),
            'q_value_mean': current_q_values.mean().item(),
            'target_q_mean': target_q_values.mean().item(),
            'epsilon': self.epsilon
        }
        
        self.train_summaries.append(train_info)
        
        return train_info
    
    def save(self, path):
        """保存模型到指定路径"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'train_step': self.train_step
        }, path)
        logger.info(f"模型已保存到 {path}")
    
    def load(self, path):
        """从指定路径加载模型"""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.q_network.load_state_dict(checkpoint['q_network'])
            self.target_network.load_state_dict(checkpoint['target_network'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epsilon = checkpoint['epsilon']
            self.train_step = checkpoint['train_step']
            logger.info(f"成功从 {path} 加载模型")
            return True
        except Exception as e:
            logger.error(f"加载模型失败: {str(e)}")
            return False
            
    def get_training_summaries(self):
        """获取训练过程的摘要统计信息"""
        if not self.train_summaries:
            return None
            
        # 计算摘要统计数据
        summaries = {}
        for key in self.train_summaries[0].keys():
            values = [summary[key] for summary in self.train_summaries]
            summaries[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'last': values[-1]
            }
        
        return summaries

# 训练函数
def train_dqn(env, args):
    """
    训练DQN模型
    
    参数:
        env: 环境
        args: 训练参数
    """
    # 获取状态和动作维度
    state = env.reset()
    state_dim = len(state)
    action_dim = env.num_users + env.num_uavs * 3  # 任务卸载决策 + 无人机移动控制
    
    logger.info(f"环境初始化，状态维度={state_dim}，动作维度={action_dim}")
    logger.info(f"训练参数：{args.num_users}个用户，{args.num_uavs}个UAV，场景：{args.scenario}")
    
    # 初始化DQN代理
    policy = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=args.lr,
        gamma=args.discount,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995,
        update_target_freq=10,
        batch_size=args.batch_size
    )
    
    # 加载已有模型（如果指定）
    if args.load_model:
        model_path = os.path.join(args.model_dir, f"dqn_{args.scenario}.pt")
        if os.path.exists(model_path):
            logger.info(f"尝试从 {model_path} 加载模型")
            if policy.load(model_path):
                logger.info("模型加载成功")
            else:
                logger.warning("模型加载失败，使用随机初始化模型")
        else:
            logger.warning("未找到预训练模型，使用随机初始化模型")
    
    # 记录训练数据
    episode_rewards = []
    avg_delays = []
    avg_energies = []
    success_rates = []
    train_infos = []
    
    # 为保存检查点创建目录
    os.makedirs(args.model_dir, exist_ok=True)
    
    # 开始训练
    logger.info(f"开始训练 {args.episodes} 个回合...")
    best_reward = -float('inf')
    episodes_without_improvement = 0
    
    for episode in range(args.episodes):
        start_time = time.time()
        
        # 重置环境
        state = env.reset()
        
        episode_reward = 0
        episode_delay = 0
        episode_energy = 0
        episode_success = 0
        episode_steps = 0
        episode_train_info = []
        
        # 一个回合的交互
        for step in range(args.steps):
            # 选择动作
            action = policy.select_action(state)
            
            # 执行动作
            next_state, reward, done, info = env.step(action)
            
            # 更新代理
            train_info = policy.update(state, action, reward, next_state, done)
            if train_info:
                episode_train_info.append(train_info)
            
            # 更新状态
            state = next_state
            
            # 累积奖励和指标
            episode_reward += reward
            episode_delay += info["delay"]
            episode_energy += info["energy"]
            episode_success = info["success_rate"]  # 最后一个时间步的成功率
            episode_steps += 1
            
            if done:
                break
        
        # 计算平均指标
        avg_delay = episode_delay / episode_steps if episode_steps > 0 else 0
        avg_energy = episode_energy / episode_steps if episode_steps > 0 else 0
        
        # 记录数据
        episode_rewards.append(episode_reward)
        avg_delays.append(avg_delay)
        avg_energies.append(avg_energy)
        success_rates.append(episode_success)
        
        # 如果有训练信息，计算平均值
        if episode_train_info:
            avg_train_info = {}
            for key in episode_train_info[0].keys():
                avg_train_info[key] = np.mean([info[key] for info in episode_train_info])
            train_infos.append(avg_train_info)
        
        # 打印训练进度
        elapsed_time = time.time() - start_time
        if (episode + 1) % args.log_interval == 0 or episode == 0:
            logger.info(f"回合 {episode+1}/{args.episodes} - " +
                      f"奖励: {episode_reward:.2f}, 延迟: {avg_delay:.4f}s, " +
                      f"能耗: {avg_energy:.4f}J, 成功率: {episode_success*100:.1f}%, " +
                      f"步数: {episode_steps}, 时间: {elapsed_time:.1f}s")
        
        # 检查是否有改进
        if episode_reward > best_reward:
            best_reward = episode_reward
            episodes_without_improvement = 0
            
            # 保存最佳模型
            best_model_path = os.path.join(args.model_dir, f"dqn_{args.scenario}_best.pt")
            policy.save(best_model_path)
            logger.info(f"发现更好的模型 (奖励: {best_reward:.2f})，已保存到 {best_model_path}")
        else:
            episodes_without_improvement += 1
        
        # 学习率衰减
        if args.lr_decay and episodes_without_improvement > args.patience:
            for param_group in policy.optimizer.param_groups:
                param_group['lr'] *= 0.9
            episodes_without_improvement = 0
            logger.info(f"学习率衰减: {policy.optimizer.param_groups[0]['lr']:.6f}")
        
        # 定期保存检查点
        if (episode + 1) % args.save_interval == 0:
            checkpoint_path = os.path.join(args.model_dir, f"dqn_{args.scenario}_ep{episode+1}.pt")
            policy.save(checkpoint_path)
            logger.info(f"检查点已保存到 {checkpoint_path}")
    
    # 保存最终模型
    final_model_path = os.path.join(args.model_dir, f"dqn_{args.scenario}_final.pt")
    policy.save(final_model_path)
    logger.info(f"最终模型已保存到 {final_model_path}")
    
    # 保存训练结果数据
    result_data = {
        "rewards": episode_rewards,
        "delays": avg_delays,
        "energies": avg_energies,
        "success_rates": success_rates,
        "train_info": train_infos
    }
    
    # 创建结果目录
    os.makedirs(args.results_dir, exist_ok=True)
    
    # 保存训练数据
    import json
    result_path = os.path.join(args.results_dir, f"dqn_training_data_{args.scenario}.json")
    with open(result_path, 'w') as f:
        # 将numpy数组转换为列表
        for key in result_data:
            if isinstance(result_data[key], np.ndarray):
                result_data[key] = result_data[key].tolist()
            elif isinstance(result_data[key], list) and result_data[key] and isinstance(result_data[key][0], np.ndarray):
                result_data[key] = [arr.tolist() for arr in result_data[key]]
        json.dump(result_data, f, indent=2)
    
    # 绘制训练结果图表
    plot_training_results(
        episode_rewards, 
        avg_delays, 
        avg_energies, 
        success_rates,
        train_infos if train_infos else None,
        os.path.join(args.results_dir, f"dqn_training_results_{args.scenario}.png"),
        args.scenario
    )
    
    logger.info(f"训练结果已保存到 {args.results_dir}")
    
    return policy, result_data

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS'] 
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def plot_training_results(rewards, delays, energies, success_rates, train_info=None, 
                          save_path="training_results.png", scenario="default"):
    """绘制训练结果图表"""
  
    # 生成带时间戳的文件名
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename, ext = os.path.splitext(save_path)
    timestamped_path = f"{filename}_{timestamp}{ext}"

    plt.figure(figsize=(20, 15), dpi=100)
    
    # 奖励图
    plt.subplot(3, 2, 1)
    plt.plot(rewards)
    plt.title('回合奖励')
    plt.xlabel('回合')
    plt.ylabel('奖励')
    
    # 应用平滑窗口
    window_size = min(50, len(rewards) // 10 + 1)
    smoothed_rewards = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
    plt.plot(range(window_size-1, window_size-1+len(smoothed_rewards)), smoothed_rewards, 'r-', alpha=0.7)
    plt.legend(['原始', f'平滑 (窗口={window_size})'])
    
    # 延迟图
    plt.subplot(3, 2, 2)
    plt.plot(delays)
    smoothed_delays = np.convolve(delays, np.ones(window_size)/window_size, mode='valid')
    plt.plot(range(window_size-1, window_size-1+len(smoothed_delays)), smoothed_delays, 'r-', alpha=0.7)
    plt.title('平均延迟')
    plt.xlabel('回合')
    plt.ylabel('延迟 (秒)')
    plt.legend(['原始', f'平滑 (窗口={window_size})'])
    
    # 能耗图
    plt.subplot(3, 2, 3)
    plt.plot(energies)
    smoothed_energies = np.convolve(energies, np.ones(window_size)/window_size, mode='valid')
    plt.plot(range(window_size-1, window_size-1+len(smoothed_energies)), smoothed_energies, 'r-', alpha=0.7)
    plt.title('平均能耗')
    plt.xlabel('回合')
    plt.ylabel('能耗 (焦耳)')
    plt.legend(['原始', f'平滑 (窗口={window_size})'])
    
    # 成功率图
    plt.subplot(3, 2, 4)
    plt.plot(success_rates)
    smoothed_success = np.convolve(success_rates, np.ones(window_size)/window_size, mode='valid')
    plt.plot(range(window_size-1, window_size-1+len(smoothed_success)), smoothed_success, 'r-', alpha=0.7)
    plt.title('任务成功率')
    plt.xlabel('回合')
    plt.ylabel('成功率')
    plt.legend(['原始', f'平滑 (窗口={window_size})'])
    
    # 如果有训练信息，绘制损失和Q值
    if train_info:
        # 提取信息
        losses = [info.get('loss', 0) for info in train_info]
        q_values = [info.get('q_value_mean', 0) for info in train_info]
        
        # 损失图
        plt.subplot(3, 2, 5)
        plt.plot(losses)
        # 平滑损失
        if len(losses) > window_size:
            smoothed_loss = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
            plt.plot(range(window_size-1, window_size-1+len(smoothed_loss)), smoothed_loss, 'r-', alpha=0.7)
        plt.title('损失值')
        plt.xlabel('回合')
        plt.ylabel('损失值')
        plt.legend(['原始', f'平滑 (窗口={window_size})'])
        
        # Q值图
        plt.subplot(3, 2, 6)
        plt.plot(q_values)
        # 平滑Q值
        if len(q_values) > window_size:
            smoothed_q = np.convolve(q_values, np.ones(window_size)/window_size, mode='valid')
            plt.plot(range(window_size-1, window_size-1+len(smoothed_q)), smoothed_q, 'r-', alpha=0.7)
        plt.title('平均Q值')
        plt.xlabel('回合')
        plt.ylabel('Q值')
        plt.legend(['原始', f'平滑 (窗口={window_size})'])
    
    plt.tight_layout()
    plt.suptitle(f'DQN训练结果 - {scenario}场景', fontsize=16)
    plt.subplots_adjust(top=0.95)
    
    # 保存带时间戳的图表
    plt.savefig(timestamped_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path)
    
    logger.info(f"训练结果图表已保存到 {timestamped_path}")
    logger.info(f"训练结果图表已保存到 {save_path}")
    
    plt.close()
    return save_path

# 评估函数
def evaluate_policy(policy, env, episodes=10, render=False):
    """评估策略性能"""
    rewards = []
    delays = []
    energies = []
    success_rates = []
    
    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        total_delay = 0
        total_energy = 0
        step_count = 0
        success_rate = 0
        done = False
        
        while not done:
            # 不带噪声地选择动作
            action = policy.select_action(state, deterministic=True)
            next_state, reward, done, info = env.step(action)
            
            # 累积信息
            total_reward += reward
            total_delay += info["delay"]
            total_energy += info["energy"]
            success_rate = info["success_rate"]  # 最后一步的成功率
            step_count += 1
            
            # 更新状态
            state = next_state
            
            # 如果需要渲染
            if render:
                env.render()
        
        # 计算平均值
        avg_delay = total_delay / step_count if step_count > 0 else 0
        avg_energy = total_energy / step_count if step_count > 0 else 0
        
        # 记录结果
        rewards.append(total_reward)
        delays.append(avg_delay)
        energies.append(avg_energy)
        success_rates.append(success_rate)
        
        print(f"评估回合 {ep+1}/{episodes} - 奖励: {total_reward:.2f}, 延迟: {avg_delay:.4f}s, "
              f"能耗: {avg_energy:.4f}J, 成功率: {success_rate*100:.1f}%")
    
    # 计算平均评估性能
    avg_reward = np.mean(rewards)
    avg_delay = np.mean(delays)
    avg_energy = np.mean(energies)
    avg_success = np.mean(success_rates)
    
    print("\n====== 评估结果 ======")
    print(f"平均奖励: {avg_reward:.4f} ± {np.std(rewards):.4f}")
    print(f"平均延迟: {avg_delay:.4f}s ± {np.std(delays):.4f}")
    print(f"平均能耗: {avg_energy:.4f}J ± {np.std(energies):.4f}")
    print(f"平均成功率: {avg_success*100:.2f}% ± {np.std(success_rates)*100:.2f}")
    
    return {
        "rewards": rewards,
        "delays": delays,
        "energies": energies,
        "success_rates": success_rates,
        "mean_reward": avg_reward,
        "mean_delay": avg_delay,
        "mean_energy": avg_energy,
        "mean_success": avg_success
    }

if __name__ == "__main__":
    import argparse
    from uav_mec_env import UAVMECEnvironment
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("dqn_agent.log"),
            logging.StreamHandler()
        ]
    )
    
    parser = argparse.ArgumentParser(description="训练DQN用于UAV-MEC任务卸载")
    # 环境参数
    parser.add_argument("--num_users", type=int, default=50, help="用户数量")
    parser.add_argument("--num_uavs", type=int, default=3, help="UAV数量")
    parser.add_argument("--scenario", type=str, default="default", help="场景名称 (default, urban, rural)")
    
    # 训练参数
    parser.add_argument("--episodes", type=int, default=1000, help="训练回合数")
    parser.add_argument("--steps", type=int, default=200, help="每回合最大步数")
    parser.add_argument("--batch_size", type=int, default=64, help="训练批次大小")
    parser.add_argument("--lr", type=float, default=0.001, help="学习率")
    parser.add_argument("--discount", type=float, default=0.99, help="折扣因子")
    parser.add_argument("--buffer_size", type=int, default=int(1e5), help="经验回放缓冲区容量")
    
    # 其他选项
    parser.add_argument("--load_model", action="store_true", help="加载现有模型")
    parser.add_argument("--model_dir", type=str, default="saved_models", help="模型保存目录")
    parser.add_argument("--results_dir", type=str, default="results/plots", help="结果保存目录")
    parser.add_argument("--log_interval", type=int, default=1, help="日志记录间隔")
    parser.add_argument("--save_interval", type=int, default=100, help="模型保存间隔")
    parser.add_argument("--evaluate", action="store_true", help="评估模式")
    parser.add_argument("--lr_decay", action="store_true", help="启用学习率衰减")
    parser.add_argument("--patience", type=int, default=50, help="学习率衰减的耐心参数")
    args = parser.parse_args()
    
    # 创建环境
    if args.scenario == "urban":
        env = UAVMECEnvironment(
            num_users=args.num_users,
            num_uavs=args.num_uavs,
            area_size=800,
            max_task_size=3,
            max_task_cycles=1.5,
            max_energy=5000
        )
    elif args.scenario == "rural":
        env = UAVMECEnvironment(
            num_users=args.num_users // 2,  # 减少用户数量
            num_uavs=args.num_uavs,
            area_size=1500,  # 扩大区域
            max_task_size=10,  # 增大任务
            max_task_cycles=5,
            max_energy=8000  # 增加能量
        )
    else:  # default
        env = UAVMECEnvironment(
            num_users=args.num_users,
            num_uavs=args.num_uavs,
            area_size=1000,
            max_task_size=5,
            max_task_cycles=2,
            max_energy=5000
        )
    
    if args.evaluate:
        # 评估模式
        state_dim = len(env.reset())
        action_dim = env.num_users + env.num_uavs * 3
        
        # 加载模型
        policy = DQNAgent(state_dim, action_dim)
        model_path = os.path.join(args.model_dir, f"dqn_{args.scenario}_best.pt")
        
        if policy.load(model_path):
            print(f"已加载模型 {model_path} 进行评估")
            evaluate_policy(policy, env, episodes=10, render=False)
        else:
            print(f"无法加载模型，评估失败")
    else:
        # 训练模式
        train_dqn(env, args)