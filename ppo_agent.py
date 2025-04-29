import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import logging
import time
import os
import matplotlib.pyplot as plt
import datetime
import matplotlib.font_manager as fm


logger = logging.getLogger('PPOAgent')

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_std_init=0.6):
        super(Actor, self).__init__()
        
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()
        )
        
        self.action_var = torch.full((action_dim,), action_std_init * action_std_init)
        
    def forward(self, state):
        action_mean = self.actor(state)
        return action_mean
    
    def get_action(self, state):
        action_mean = self.actor(state)
        cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
        dist = Normal(action_mean, torch.sqrt(self.action_var))
        action = dist.sample()
        action_logprob = dist.log_prob(action).sum(dim=-1)
        return action, action_logprob
    
    def evaluate(self, state, action):
        action_mean = self.actor(state)
        dist = Normal(action_mean, torch.sqrt(self.action_var))
        action_logprobs = dist.log_prob(action).sum(dim=-1)
        dist_entropy = dist.entropy().sum(dim=-1)
        return action_logprobs, dist_entropy

class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, state):
        return self.critic(state)

class PPOMemory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.next_states = []
        self.is_terminals = []
    
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.is_terminals.clear()
    
    def add(self, state, action, logprob, reward, next_state, done):
        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.is_terminals.append(done)
    
    def size(self):
        return len(self.states)

class PPOAgent:
    def __init__(self, state_dim=12, action_dim=3, lr_actor=0.0003, lr_critic=0.001, gamma=0.99, 
                 eps_clip=0.2, K_epochs=10):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        # 初始化演员和评论家网络
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)
        
        # 初始化优化器
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # 初始化内存
        self.memory = PPOMemory()
        
        # 旧策略
        self.policy_old = Actor(state_dim, action_dim)
        self.policy_old.load_state_dict(self.actor.state_dict())
        
        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
        # 训练摘要
        self.train_summaries = []
        
        logger.info(f"PPO代理在 {self.device} 上初始化")
    
    def to(self, device):
        self.actor.to(device)
        self.critic.to(device)
        self.policy_old.to(device)
    
    def get_action(self, state, deterministic=False):
        # 将状态转换为张量
        if isinstance(state, list):
            state = np.array(state)
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        
        with torch.no_grad():
            if deterministic:
                action = self.policy_old(state)
            else:
                action, _ = self.policy_old.get_action(state)
        
        # 将动作作为numpy数组返回
        return action.cpu().numpy().flatten()
    
    def select_action(self, state, deterministic=False):
        return self.get_action(state, deterministic)

    def update(self, state, action, reward, next_state, done):
        # 如果需要，转换数据类型
        if isinstance(state, list):
            state = np.array(state)
        if isinstance(action, list):
            action = np.array(action)
        if isinstance(next_state, list):
            next_state = np.array(next_state)
        
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = torch.FloatTensor(action.reshape(1, -1)).to(self.device)
        
        # 获取动作对数概率
        with torch.no_grad():
            _, old_logprob = self.policy_old.evaluate(state, action)
        
        # 存储到内存
        self.memory.add(
            state.cpu().numpy().reshape(-1),
            action.cpu().numpy().reshape(-1),
            old_logprob.cpu().numpy().item(),
            reward,
            next_state.reshape(-1),
            done
        )
        
        # 如果有足够的经验，则学习
        if self.memory.size() >= 64:
            train_info = self.learn()
            return train_info
        return None
    
    def learn(self):
        # 从内存获取数据
        old_states = torch.FloatTensor(np.array(self.memory.states)).to(self.device)
        old_actions = torch.FloatTensor(np.array(self.memory.actions)).to(self.device)
        old_logprobs = torch.FloatTensor(np.array(self.memory.logprobs)).to(self.device)
        rewards = self.memory.rewards
        is_terminals = self.memory.is_terminals
        
        # 计算折扣奖励
        discounted_rewards = []
        discounted_reward = 0
        for reward, done in zip(reversed(rewards), reversed(is_terminals)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            discounted_rewards.insert(0, discounted_reward)
        
        # 转换为张量并标准化
        discounted_rewards = torch.FloatTensor(discounted_rewards).to(self.device)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8)
        
        # 优化策略
        total_loss = 0
        actor_loss = 0
        critic_loss = 0
        
        for _ in range(self.K_epochs):
            # 评估旧动作和值
            logprobs, dist_entropy = self.actor.evaluate(old_states, old_actions)
            state_values = self.critic(old_states).squeeze()
            
            # 计算比率
            ratios = torch.exp(logprobs - old_logprobs.detach())
            
            # 计算优势
            advantages = discounted_rewards - state_values.detach()
            
            # PPO损失
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = 0.5 * ((discounted_rewards - state_values)**2).mean()
            entropy_loss = 0.01 * dist_entropy.mean()
            
            loss = actor_loss + critic_loss - entropy_loss
            total_loss += loss.item()
            
            # 优化策略
            self.optimizer_actor.zero_grad()
            self.optimizer_critic.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            
            self.optimizer_actor.step()
            self.optimizer_critic.step()
        
        # 更新策略旧网络
        self.policy_old.load_state_dict(self.actor.state_dict())
        
        # 清空内存
        self.memory.clear()
        
        # 记录训练摘要
        train_info = {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'entropy': dist_entropy.mean().item(),
            'total_loss': total_loss / self.K_epochs,
            'value_mean': state_values.mean().item(),
            'advantage_mean': advantages.mean().item(),
            'ratio_mean': ratios.mean().item()
        }
        
        self.train_summaries.append(train_info)
        logger.info("完成PPO学习更新")
        
        return train_info
        
    def save(self, path):
        """保存模型到指定路径"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'optimizer_actor': self.optimizer_actor.state_dict(),
            'optimizer_critic': self.optimizer_critic.state_dict(),
        }, path)
        logger.info(f"模型已保存到 {path}")
    
    def load(self, path):
        """从指定路径加载模型"""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.actor.load_state_dict(checkpoint['actor'])
            self.critic.load_state_dict(checkpoint['critic'])
            self.optimizer_actor.load_state_dict(checkpoint['optimizer_actor'])
            self.optimizer_critic.load_state_dict(checkpoint['optimizer_critic'])
            self.policy_old.load_state_dict(self.actor.state_dict())
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

def train_ppo(env, args):
    """
    训练PPO模型
    
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
    
    # 初始化PPO代理
    policy = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        gamma=args.discount,
        eps_clip=args.eps_clip,
        K_epochs=args.K_epochs
    )
    
    # 加载已有模型（如果指定）
    if args.load_model:
        model_path = os.path.join(args.model_dir, f"ppo_{args.scenario}.pt")
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
            best_model_path = os.path.join(args.model_dir, f"ppo_{args.scenario}_best.pt")
            policy.save(best_model_path)
            logger.info(f"发现更好的模型 (奖励: {best_reward:.2f})，已保存到 {best_model_path}")
        else:
            episodes_without_improvement += 1
        
        # 学习率衰减
        if args.lr_decay and episodes_without_improvement > args.patience:
            for param_group in policy.optimizer_actor.param_groups:
                param_group['lr'] *= 0.9
            for param_group in policy.optimizer_critic.param_groups:
                param_group['lr'] *= 0.9
            episodes_without_improvement = 0
            logger.info(f"学习率衰减: Actor={policy.optimizer_actor.param_groups[0]['lr']:.6f}, " +
                      f"Critic={policy.optimizer_critic.param_groups[0]['lr']:.6f}")
        
        # 定期保存检查点
        if (episode + 1) % args.save_interval == 0:
            checkpoint_path = os.path.join(args.model_dir, f"ppo_{args.scenario}_ep{episode+1}.pt")
            policy.save(checkpoint_path)
            logger.info(f"检查点已保存到 {checkpoint_path}")
    
    # 保存最终模型
    final_model_path = os.path.join(args.model_dir, f"ppo_{args.scenario}_final.pt")
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
    result_path = os.path.join(args.results_dir, f"ppo_training_data_{args.scenario}.json")
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
        os.path.join(args.results_dir, f"ppo_training_results_{args.scenario}.png"),
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
    
    # 如果有训练信息，绘制PPO特有指标
    if train_info:
        # 提取信息
        actor_losses = [info.get('actor_loss', 0) for info in train_info]
        critic_losses = [info.get('critic_loss', 0) for info in train_info]
        
        # Actor损失图
        plt.subplot(3, 2, 5)
        plt.plot(actor_losses)
        # 平滑损失
        if len(actor_losses) > window_size:
            smoothed_loss = np.convolve(actor_losses, np.ones(window_size)/window_size, mode='valid')
            plt.plot(range(window_size-1, window_size-1+len(smoothed_loss)), smoothed_loss, 'r-', alpha=0.7)
        plt.title('Actor 损失')
        plt.xlabel('回合')
        plt.ylabel('损失值')
        plt.legend(['原始', f'平滑 (窗口={window_size})'])
        
        # Critic损失图
        plt.subplot(3, 2, 6)
        plt.plot(critic_losses)
        # 平滑损失
        if len(critic_losses) > window_size:
            smoothed_loss = np.convolve(critic_losses, np.ones(window_size)/window_size, mode='valid')
            plt.plot(range(window_size-1, window_size-1+len(smoothed_loss)), smoothed_loss, 'r-', alpha=0.7)
        plt.title('Critic 损失')
        plt.xlabel('回合')
        plt.ylabel('损失值')
        plt.legend(['原始', f'平滑 (窗口={window_size})'])
    
    plt.tight_layout()
    plt.suptitle(f'PPO训练结果 - {scenario}场景', fontsize=16)
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
            logging.FileHandler("ppo_agent.log"),
            logging.StreamHandler()
        ]
    )
    
    parser = argparse.ArgumentParser(description="训练PPO用于UAV-MEC任务卸载")
    # 环境参数
    parser.add_argument("--num_users", type=int, default=50, help="用户数量")
    parser.add_argument("--num_uavs", type=int, default=3, help="UAV数量")
    parser.add_argument("--scenario", type=str, default="default", help="场景名称 (default, urban, rural)")
    
    # 训练参数
    parser.add_argument("--episodes", type=int, default=1000, help="训练回合数")
    parser.add_argument("--steps", type=int, default=200, help="每回合最大步数")
    parser.add_argument("--lr_actor", type=float, default=0.0003, help="Actor学习率")
    parser.add_argument("--lr_critic", type=float, default=0.001, help="Critic学习率")
    parser.add_argument("--discount", type=float, default=0.99, help="折扣因子")
    parser.add_argument("--eps_clip", type=float, default=0.2, help="PPO裁剪参数")
    parser.add_argument("--K_epochs", type=int, default=10, help="PPO更新轮数")
    
    # 其他选项
    parser.add_argument("--load_model", action="store_true", help="加载现有模型")
    parser.add_argument("--model_dir", type=str, default="saved_models", help="模型保存目录")
    parser.add_argument("--results_dir", type=str, default="results/plots", help="结果保存目录")
    parser.add_argument("--log_interval", type=int, default=1, help="日志记录间隔")
    parser.add_argument("--save_interval", type=int, default=100, help="模型保存间隔")
    parser.add_argument("--evaluate", action="store_true", help="评估模式")
    parser.add_argument("--patience", type=int, default=50, help="学习率衰减的耐心参数") 
    parser.add_argument("--lr_decay", action="store_true", help="启用学习率衰减")
    
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
        policy = PPOAgent(state_dim, action_dim)
        model_path = os.path.join(args.model_dir, f"ppo_{args.scenario}_best.pt")
        
        if policy.load(model_path):
            print(f"已加载模型 {model_path} 进行评估")
            evaluate_policy(policy, env, episodes=10, render=False)
        else:
            print(f"无法加载模型，评估失败")
    else:
        # 训练模式
        train_ppo(env, args)