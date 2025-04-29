import numpy as np
import tensorflow as tf
import os
import time
import matplotlib.pyplot as plt
import datetime
import logging
# 设置中文字体
import matplotlib.font_manager as fm


class ReplayBuffer:
    def __init__(self, capacity, state_dim, action_dim):
        self.capacity = capacity
        self.counter = 0
        self.state_buffer = np.zeros((capacity, state_dim), dtype=np.float32)
        self.action_buffer = np.zeros((capacity, action_dim), dtype=np.float32)
        self.reward_buffer = np.zeros(capacity, dtype=np.float32)
        self.next_state_buffer = np.zeros((capacity, state_dim), dtype=np.float32)
        self.done_buffer = np.zeros(capacity, dtype=np.bool_)
        
        # 最近的奖励统计用于归一化
        self.recent_rewards = []
        self.max_recent_rewards = 1000

    def add(self, state, action, reward, next_state, done):
        index = self.counter % self.capacity
        self.state_buffer[index] = state
        self.action_buffer[index] = action
        self.reward_buffer[index] = reward
        self.next_state_buffer[index] = next_state
        self.done_buffer[index] = done
        
        # 记录最近的奖励
        if len(self.recent_rewards) >= self.max_recent_rewards:
            self.recent_rewards.pop(0)
        self.recent_rewards.append(reward)
        
        self.counter += 1
    
    def sample(self, batch_size):
        record_range = min(self.counter, self.capacity)
        batch_indices = np.random.choice(record_range, batch_size)
        
        return (
            self.state_buffer[batch_indices],
            self.action_buffer[batch_indices],
            self.reward_buffer[batch_indices],
            self.next_state_buffer[batch_indices],
            self.done_buffer[batch_indices]
        )
    
    def get_reward_stats(self):
        """获取奖励的均值和标准差，用于归一化"""
        if not self.recent_rewards:
            return 0.0, 1.0
        return np.mean(self.recent_rewards), max(np.std(self.recent_rewards), 1e-8)
    
    @property
    def size(self):
        return min(self.counter, self.capacity)

# DDPG代理
class DDPG:
    def __init__(self, state_dim, action_dim, action_bound=1.0, actor_lr=0.001, critic_lr=0.002, gamma=0.99, tau=0.005):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.gamma = gamma
        self.tau = tau
        
        # 使用函数式API创建Actor网络
        state_input = tf.keras.layers.Input(shape=(state_dim,))
        layer1 = tf.keras.layers.Dense(256, activation='relu')(state_input)
        layer2 = tf.keras.layers.Dense(128, activation='relu')(layer1)
        action_output = tf.keras.layers.Dense(action_dim, activation='tanh')(layer2)
        scaled_output = tf.keras.layers.Lambda(lambda x: x * action_bound)(action_output)
        
        self.actor = tf.keras.Model(inputs=state_input, outputs=scaled_output)
        self.target_actor = tf.keras.Model(inputs=state_input, outputs=scaled_output)
        
        # 使用函数式API创建Critic网络
        state_input_critic = tf.keras.layers.Input(shape=(state_dim,))
        action_input_critic = tf.keras.layers.Input(shape=(action_dim,))
        
        state_x = tf.keras.layers.Dense(256, activation='relu')(state_input_critic)
        concat = tf.keras.layers.Concatenate()([state_x, action_input_critic])
        x = tf.keras.layers.Dense(128, activation='relu')(concat)
        q_value = tf.keras.layers.Dense(1)(x)
        
        self.critic = tf.keras.Model(inputs=[state_input_critic, action_input_critic], outputs=q_value)
        self.target_critic = tf.keras.Model(inputs=[state_input_critic, action_input_critic], outputs=q_value)
        
        # 复制权重到目标网络
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())
        
        # 设置优化器
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)
        
        # 探索噪声
        self.noise_std = 0.2
        self.noise_clip = 0.5
        self.epsilon = 1.0  # 初始探索率
        self.epsilon_decay = 0.995  # 探索率衰减
        self.min_epsilon = 0.1  # 最小探索率
        
        # 训练摘要
        self.train_summaries = []
        
        # 训练迭代计数
        self.train_step = 0
        
        logging.info(f"DDPG代理初始化完成，状态维度: {state_dim}, 动作维度: {action_dim}")
    
    def select_action(self, state, add_noise=True):
        """选择动作，可以选择是否添加噪声"""
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        action = self.actor(state)[0].numpy()
        
        if add_noise:
            # 使用OU噪声或普通高斯噪声
            if np.random.random() < self.epsilon:
                noise = np.random.normal(0, self.noise_std, size=self.action_dim)
                noise = np.clip(noise, -self.noise_clip, self.noise_clip)
                action += noise
            
            # 衰减探索率
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        
        # 裁剪动作
        action = np.clip(action, -self.action_bound, self.action_bound)
        return action
    
    @tf.function
    def _update_target(self, target_weights, weights, tau):
        """软更新目标网络"""
        for a, b in zip(target_weights, weights):
            a.assign(b * tau + a * (1 - tau))
    
    def update_target_networks(self):
        """更新目标网络"""
        self._update_target(self.target_actor.variables, self.actor.variables, self.tau)
        self._update_target(self.target_critic.variables, self.critic.variables, self.tau)
    
    def train(self, replay_buffer, batch_size=64):
        """训练网络"""
        if replay_buffer.size < batch_size:
            return None
        
        self.train_step += 1
        
        # 获取随机样本
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = replay_buffer.sample(batch_size)
        
        # 获取奖励归一化统计信息
        reward_mean, reward_std = replay_buffer.get_reward_stats()
        
        # 训练Critic网络
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(next_state_batch)
            target_q = self.target_critic([next_state_batch, target_actions])
            
            # 使用done屏蔽目标Q值
            target_q = reward_batch + (1 - done_batch) * self.gamma * target_q
            
            current_q = self.critic([state_batch, action_batch])
            critic_loss = tf.reduce_mean(tf.square(target_q - current_q))
        
        # 应用Critic梯度
        critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic.trainable_variables))
        
        # 训练Actor网络
        with tf.GradientTape() as tape:
            actions = self.actor(state_batch)
            q_values = self.critic([state_batch, actions])
            actor_loss = -tf.reduce_mean(q_values)
        
        # 应用Actor梯度
        actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))
        
        # 更新目标网络
        self.update_target_networks()
        
        # 记录训练摘要
        train_info = {
            'critic_loss': float(critic_loss),
            'actor_loss': float(actor_loss),
            'q_mean': float(tf.reduce_mean(current_q)),
            'target_q_mean': float(tf.reduce_mean(target_q)),
            'reward_mean': float(reward_mean),
            'reward_std': float(reward_std),
            'epsilon': float(self.epsilon)
        }
        
        self.train_summaries.append(train_info)
        
        return train_info
    
    def save(self, actor_path, critic_path):
        """保存模型"""
        try:
            self.actor.save_weights(actor_path)
            self.critic.save_weights(critic_path)
            logging.info(f"已成功保存模型到 {actor_path} 和 {critic_path}")
            return True
        except Exception as e:
            logging.error(f"保存模型时出错: {str(e)}")
            return False
    
    def load(self, actor_path, critic_path):
        """加载模型"""
        try:
            self.actor.load_weights(actor_path)
            self.critic.load_weights(critic_path)
            
            # 复制到目标网络
            self.target_actor.set_weights(self.actor.get_weights())
            self.target_critic.set_weights(self.critic.get_weights())
            
            logging.info(f"已成功从 {actor_path} 和 {critic_path} 加载模型")
            return True
        except Exception as e:
            logging.error(f"加载模型时出错: {str(e)}")
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

def train_ddpg(env, args):
    """
    训练DDPG模型
    
    参数:
        env: 环境
        args: 训练参数
    """
    # 获取状态和动作维度
    state = env.reset()
    state_dim = len(state)
    action_dim = env.num_users + env.num_uavs * 3  # 任务卸载决策 + 无人机移动控制
    max_action = 1.0
    
    logging.info(f"环境初始化，状态维度={state_dim}，动作维度={action_dim}")
    logging.info(f"训练参数：{args.num_users}个用户，{args.num_uavs}个UAV，场景：{args.scenario}")
    
    # 初始化DDPG代理
    policy = DDPG(
        state_dim=state_dim,
        action_dim=action_dim,
        action_bound=max_action,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        gamma=args.discount,
        tau=args.tau
    )
    
    # 加载已有模型（如果指定）
    if args.load_model:
        actor_path = os.path.join(args.model_dir, f"ddpg_{args.scenario}_actor.h5")
        critic_path = os.path.join(args.model_dir, f"ddpg_{args.scenario}_critic.h5")
        
        if os.path.exists(actor_path) and os.path.exists(critic_path):
            logging.info(f"尝试从 {actor_path} 和 {critic_path} 加载模型")
            if policy.load(actor_path, critic_path):
                logging.info("模型加载成功")
            else:
                logging.warning("模型加载失败，使用随机初始化模型")
        else:
            logging.warning("未找到预训练模型，使用随机初始化模型")
    
    # 初始化经验回放缓冲区
    replay_buffer = ReplayBuffer(capacity=args.buffer_size, state_dim=state_dim, action_dim=action_dim)
    
    # 预填充缓冲区
    if args.prefill > 0:
        prefill_buffer(env, replay_buffer, args.prefill)
    
    # 记录训练数据
    episode_rewards = []
    avg_delays = []
    avg_energies = []
    success_rates = []
    train_infos = []
    
    # 为保存检查点创建目录
    os.makedirs(args.model_dir, exist_ok=True)
    
    # 开始训练
    logging.info(f"开始训练 {args.episodes} 个回合...")
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
            action = policy.select_action(state, add_noise=True)
            
            # 执行动作
            next_state, reward, done, info = env.step(action)
            
            # 存储经验样本
            replay_buffer.add(state, action, reward, next_state, done)
            
            # 更新状态
            state = next_state
            
            # 训练代理
            train_info = policy.train(replay_buffer, args.batch_size)
            if train_info:
                episode_train_info.append(train_info)
            
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
            logging.info(f"回合 {episode+1}/{args.episodes} - " +
                      f"奖励: {episode_reward:.2f}, 延迟: {avg_delay:.4f}s, " +
                      f"能耗: {avg_energy:.4f}J, 成功率: {episode_success*100:.1f}%, " +
                      f"步数: {episode_steps}, 时间: {elapsed_time:.1f}s")
        
        # 检查是否有改进
        if episode_reward > best_reward:
            best_reward = episode_reward
            episodes_without_improvement = 0
            
            # 保存最佳模型
            best_actor_path = os.path.join(args.model_dir, f"ddpg_{args.scenario}_best_actor.h5")
            best_critic_path = os.path.join(args.model_dir, f"ddpg_{args.scenario}_best_critic.h5")
            policy.save(best_actor_path, best_critic_path)
            logging.info(f"发现更好的模型 (奖励: {best_reward:.2f})，已保存")
        else:
            episodes_without_improvement += 1
        
        # 学习率衰减
        if args.lr_decay and episodes_without_improvement > args.patience:
            policy.actor_optimizer.learning_rate.assign(policy.actor_optimizer.learning_rate * 0.9)
            policy.critic_optimizer.learning_rate.assign(policy.critic_optimizer.learning_rate * 0.9)
            episodes_without_improvement = 0
            logging.info(f"学习率衰减: Actor={policy.actor_optimizer.learning_rate.numpy():.6f}, " +
                      f"Critic={policy.critic_optimizer.learning_rate.numpy():.6f}")
        
        # 定期保存检查点
        if (episode + 1) % args.save_interval == 0:
            checkpoint_actor_path = os.path.join(args.model_dir, f"ddpg_{args.scenario}_ep{episode+1}_actor.h5")
            checkpoint_critic_path = os.path.join(args.model_dir, f"ddpg_{args.scenario}_ep{episode+1}_critic.h5")
            policy.save(checkpoint_actor_path, checkpoint_critic_path)
            logging.info(f"检查点已保存到 ep{episode+1}")
    
    # 保存最终模型
    final_actor_path = os.path.join(args.model_dir, f"ddpg_{args.scenario}_final_actor.h5")
    final_critic_path = os.path.join(args.model_dir, f"ddpg_{args.scenario}_final_critic.h5")
    policy.save(final_actor_path, final_critic_path)
    logging.info(f"最终模型已保存")
    
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
    result_path = os.path.join(args.results_dir, f"ddpg_training_data_{args.scenario}.json")
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
        os.path.join(args.results_dir, f"ddpg_training_results_{args.scenario}.png"),
        args.scenario
    )
    
    logging.info(f"训练结果已保存到 {args.results_dir}")
    
    return policy, result_data

def prefill_buffer(env, replay_buffer, prefill_steps):
    """
    预填充经验回放缓冲区
    
    参数:
        env: 环境
        replay_buffer: 经验回放缓冲区
        prefill_steps: 预填充步数
    """
    logging.info(f"开始预填充经验回放缓冲区，目标步数: {prefill_steps}...")
    state = env.reset()
    
    for step in range(prefill_steps):
        # 随机选择动作
        action = np.random.uniform(-1, 1, size=env.action_space_dims)
        
        # 执行动作
        next_state, reward, done, info = env.step(action)
        
        # 存储经验
        replay_buffer.add(state, action, reward, next_state, done)
    
        # 更新状态
        if done:
            state = env.reset()
        else:
            state = next_state
        
        # 打印进度
        if (step + 1) % (prefill_steps // 10) == 0:
            logging.info(f"预填充进度: {step+1}/{prefill_steps}")
    
    logging.info(f"预填充完成，经验回放缓冲区大小: {replay_buffer.size}")
    
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
    if len(rewards) > window_size:
        smoothed_rewards = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, window_size-1+len(smoothed_rewards)), smoothed_rewards, 'r-', alpha=0.7)
        plt.legend(['原始', f'平滑 (窗口={window_size})'])
    
    # 延迟图
    plt.subplot(3, 2, 2)
    plt.plot(delays)
    if len(delays) > window_size:
        smoothed_delays = np.convolve(delays, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, window_size-1+len(smoothed_delays)), smoothed_delays, 'r-', alpha=0.7)
        plt.legend(['原始', f'平滑 (窗口={window_size})'])
    plt.title('平均延迟')
    plt.xlabel('回合')
    plt.ylabel('延迟 (秒)')
    
    # 能耗图
    plt.subplot(3, 2, 3)
    plt.plot(energies)
    if len(energies) > window_size:
        smoothed_energies = np.convolve(energies, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, window_size-1+len(smoothed_energies)), smoothed_energies, 'r-', alpha=0.7)
        plt.legend(['原始', f'平滑 (窗口={window_size})'])
    plt.title('平均能耗')
    plt.xlabel('回合')
    plt.ylabel('能耗 (焦耳)')
    
    # 成功率图
    plt.subplot(3, 2, 4)
    plt.plot(success_rates)
    if len(success_rates) > window_size:
        smoothed_success = np.convolve(success_rates, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, window_size-1+len(smoothed_success)), smoothed_success, 'r-', alpha=0.7)
        plt.legend(['原始', f'平滑 (窗口={window_size})'])
    plt.title('任务成功率')
    plt.xlabel('回合')
    plt.ylabel('成功率')
    
    # 如果有训练信息，绘制DDPG特有指标
    if train_info:
        # 提取信息
        critic_losses = [info.get('critic_loss', 0) for info in train_info]
        q_values = [info.get('q_mean', 0) for info in train_info]
        
        # Critic损失图
        plt.subplot(3, 2, 5)
        plt.plot(critic_losses)
        # 平滑损失
        if len(critic_losses) > window_size:
            smoothed_loss = np.convolve(critic_losses, np.ones(window_size)/window_size, mode='valid')
            plt.plot(range(window_size-1, window_size-1+len(smoothed_loss)), smoothed_loss, 'r-', alpha=0.7)
            plt.legend(['原始', f'平滑 (窗口={window_size})'])
        plt.title('Critic 损失')
        plt.xlabel('回合')
        plt.ylabel('损失值')
        
        # Q值图
        plt.subplot(3, 2, 6)
        plt.plot(q_values)
        # 平滑Q值
        if len(q_values) > window_size:
            smoothed_q = np.convolve(q_values, np.ones(window_size)/window_size, mode='valid')
            plt.plot(range(window_size-1, window_size-1+len(smoothed_q)), smoothed_q, 'r-', alpha=0.7)
            plt.legend(['原始', f'平滑 (窗口={window_size})'])
        plt.title('平均Q值')
        plt.xlabel('回合')
        plt.ylabel('Q值')
    
    plt.tight_layout()
    plt.suptitle(f'DDPG训练结果 - {scenario}场景', fontsize=16)
    plt.subplots_adjust(top=0.95)
    
    # 保存带时间戳的图表
    plt.savefig(timestamped_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path)
    
    logging.info(f"训练结果图表已保存到 {timestamped_path}")
    logging.info(f"训练结果图表已保存到 {save_path}")
    
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
            action = policy.select_action(state, add_noise=False)
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
            logging.FileHandler("ddpg_agent.log"),
            logging.StreamHandler()
        ]
    )
    
    parser = argparse.ArgumentParser(description="训练DDPG用于UAV-MEC任务卸载")
    # 环境参数
    parser.add_argument("--num_users", type=int, default=50, help="用户数量")
    parser.add_argument("--num_uavs", type=int, default=3, help="UAV数量")
    parser.add_argument("--scenario", type=str, default="default", help="场景名称 (default, urban, rural)")
    
    # 训练参数
    parser.add_argument("--episodes", type=int, default=1000, help="训练回合数")
    parser.add_argument("--steps", type=int, default=200, help="每回合最大步数")
    parser.add_argument("--batch_size", type=int, default=64, help="训练批次大小")
    parser.add_argument("--actor_lr", type=float, default=0.001, help="Actor学习率")
    parser.add_argument("--critic_lr", type=float, default=0.002, help="Critic学习率")
    parser.add_argument("--discount", type=float, default=0.99, help="折扣因子")
    parser.add_argument("--tau", type=float, default=0.005, help="软更新系数")
    parser.add_argument("--buffer_size", type=int, default=int(1e5), help="经验回放缓冲区容量")
    parser.add_argument("--prefill", type=int, default=1000, help="预填充步数")
    
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
    
    # 添加动作空间维度属性，用于预填充
    env.action_space_dims = env.num_users + env.num_uavs * 3
    
    if args.evaluate:
        # 评估模式
        state_dim = len(env.reset())
        action_dim = env.num_users + env.num_uavs * 3
        
        # 加载模型
        policy = DDPG(state_dim, action_dim)
        actor_path = os.path.join(args.model_dir, f"ddpg_{args.scenario}_best_actor.h5")
        critic_path = os.path.join(args.model_dir, f"ddpg_{args.scenario}_best_critic.h5")
        
        if policy.load(actor_path, critic_path):
            print(f"已加载模型 {actor_path} 和 {critic_path} 进行评估")
            evaluate_policy(policy, env, episodes=10, render=False)
        else:
            print(f"无法加载模型，评估失败")
    else:
        # 训练模式
        train_ddpg(env, args)