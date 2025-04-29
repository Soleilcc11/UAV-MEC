#!/usr/bin/env python3
"""
训练DDPG模型用于无人机辅助移动边缘计算任务卸载
"""

import numpy as np
import tensorflow as tf
import time
import os
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
from ddpg_agent import DDPG, ReplayBuffer
from uav_mec_env import UAVMECEnvironment
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("python/logs/train_ddpg.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("TrainDDPG")

# 确保目录存在
os.makedirs("python/logs", exist_ok=True)
os.makedirs("python/saved_models", exist_ok=True)
os.makedirs("python/results", exist_ok=True)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Train DDPG for UAV-MEC task offloading")
    parser.add_argument("--num_users", type=int, default=50, help="Number of users")
    parser.add_argument("--num_uavs", type=int, default=3, help="Number of UAVs")
    parser.add_argument("--episodes", type=int, default=5000, help="Number of training episodes")
    parser.add_argument("--steps", type=int, default=100, help="Maximum steps per episode")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training")
    parser.add_argument("--scenario", type=str, default="default", help="Scenario name (default, urban, rural)")
    parser.add_argument("--load_model", action="store_true", help="Load existing model if available")
    parser.add_argument("--prefill", type=int, default=1000, help="Number of steps to prefill buffer with random actions")
    return parser.parse_args()

#  确保模型构建完成
def ensure_model_is_built(policy, state_dim, action_dim):
    """确保模型已被构建，可以安全保存"""
    logger.info("确保模型已构建以便保存...")
    
    # 创建虚拟输入数据
    dummy_state = np.zeros((1, state_dim), dtype=np.float32)
    dummy_action = np.zeros((1, action_dim), dtype=np.float32)
    
    # 通过前向传播构建模型
    _ = policy.actor(dummy_state)
    _ = policy.critic(dummy_state, dummy_action)
    
    logger.info("模型已成功构建，现在可以安全保存")

# 预填充经验回放缓冲区
def prefill_buffer(env, replay_buffer, prefill_steps=1000):
    """使用随机动作预填充经验回放缓冲区"""
    logger.info(f"开始预填充经验回放缓冲区，目标步数: {prefill_steps}...")
    state = env.reset()
    
    for step in tqdm(range(prefill_steps), desc="预填充缓冲区"):
        # 随机选择动作
        action = np.random.uniform(-1, 1, size=env.num_users + env.num_uavs * 3)
        
        # 执行动作
        next_state, reward, done, info = env.step(action)
        
        # 存储经验
        replay_buffer.add(state, action, reward, next_state, done)
        
        # 更新状态
        if done:
            state = env.reset()
        else:
            state = next_state
    
    logger.info(f"预填充完成，经验回放缓冲区大小: {replay_buffer.size}")

# 动态批次大小
def get_dynamic_batch_size(buffer_size, min_batch=32, max_batch=256):
    """根据缓冲区大小动态调整批次大小"""
    return min(max(min_batch, buffer_size // 10), max_batch)

def train_ddpg(args):
    """训练DDPG模型"""
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
    
    # 获取状态和动作维度
    state = env.reset()
    state_dim = len(state)
    action_dim = env.num_users + env.num_uavs * 3  # 任务卸载决策 + 无人机移动控制
    max_action = 1.0
    
    logger.info(f"Environment initialized with state_dim={state_dim}, action_dim={action_dim}")
    logger.info(f"Training with {args.num_users} users, {args.num_uavs} UAVs in {args.scenario} scenario")
    
    # 初始化DDPG代理
    policy = DDPG(state_dim, action_dim, max_action)
    
    # 加载已有模型（如果指定）
    if args.load_model:
        model_path = "python/saved_models/ddpg_actor.h5"
        if os.path.exists(model_path):
            logger.info(f"Loading model from {model_path}")
            try:
                policy.actor.load_weights(model_path)
                policy.critic.load_weights("python/saved_models/ddpg_critic.h5")
                logger.info("Model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading model: {str(e)}")
                logger.info("Starting with a new model")
        else:
            logger.warning("No pre-trained model found, starting from scratch")
    
    # 初始化经验回放缓冲区
    replay_buffer_size = int(1e6)
    replay_buffer = ReplayBuffer(replay_buffer_size, state_dim, action_dim)
    
    # 预填充缓冲区
    if args.prefill > 0:
        prefill_buffer(env, replay_buffer, args.prefill)
    
    # 记录训练数据
    episode_rewards = []
    avg_delays = []
    avg_energies = []
    success_rates = []
    
    # 开始训练
    logger.info(f"Starting training for {args.episodes} episodes")
    for episode in range(args.episodes):
        start_time = time.time()
        
        # 重置环境
        state = env.reset()
        
        episode_reward = 0
        episode_delay = 0
        episode_energy = 0
        episode_success = 0
        episode_steps = 0
        
        # 一个回合的交互
        for step in range(args.steps):
            # 选择动作
            action = policy.select_action(state, noise=0.2)
            
            # 执行动作
            next_state, reward, done, info = env.step(action)
            
            # 存储经验样本
            replay_buffer.add(state, action, reward, next_state, done)
            
            # 更新状态
            state = next_state
            
            # 累积奖励和指标
            episode_reward += reward
            episode_delay += info["delay"]
            episode_energy += info["energy"]
            episode_success = info["success_rate"]  # 最后一个时间步的成功率
            episode_steps += 1
            
            # [修改] 动态批次大小
            current_batch_size = get_dynamic_batch_size(replay_buffer.size, min_batch=32, max_batch=args.batch_size)
            
            # 如果有足够样本，进行训练
            if replay_buffer.size >= current_batch_size:
                policy.train(replay_buffer, current_batch_size)
                if current_batch_size < args.batch_size:
                    print(f"Training with reduced batch size: {current_batch_size} (buffer size: {replay_buffer.size})")
                else:
                    print(f"Training at step {episode_steps}, buffer size: {replay_buffer.size}")
            else:
                print(f"Buffer size {replay_buffer.size} not yet sufficient for training (need {current_batch_size})")
                        
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
        
        # 打印训练进度
        elapsed_time = time.time() - start_time
        if (episode + 1) % 10 == 0 or episode == 0:
            logger.info(f"Episode {episode+1}/{args.episodes} - " +
                      f"Reward: {episode_reward:.2f}, Delay: {avg_delay:.4f}s, " +
                      f"Energy: {avg_energy:.4f}J, Success: {episode_success*100:.1f}%, " +
                      f"Time: {elapsed_time:.1f}s")
        
        # 每50个回合保存模型，确保模型已构建
        if (episode + 1) % 50 == 0:
            # 确保模型已构建
            ensure_model_is_built(policy, state_dim, action_dim)
            
            # 确保目录存在
            os.makedirs("python/saved_models", exist_ok=True)
            
            try:
                # 保存模型
                policy.actor.save_weights("python/saved_models/ddpg_actor.weights.h5")
                policy.critic.save_weights("python/saved_models/ddpg_critic.weights.h5")
                logger.info(f"Model saved at episode {episode+1}")
            except Exception as e:
                logger.error(f"Error saving model weights: {str(e)}")
                
                # 尝试使用SavedModel格式保存
                try:
                    tf.saved_model.save(policy.actor, "python/saved_models/ddpg_actor_model")
                    tf.saved_model.save(policy.critic, "python/saved_models/ddpg_critic_model")
                    logger.info("Successfully saved models using SavedModel format")
                except Exception as e2:
                    logger.error(f"Alternative saving method also failed: {str(e2)}")
    
    # 保存训练结果数据
    result_data = {
        "rewards": episode_rewards,
        "delays": avg_delays,
        "energies": avg_energies,
        "success_rates": success_rates
    }
    
    # 绘制训练结果图表
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(episode_rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    plt.subplot(2, 2, 2)
    plt.plot(avg_delays)
    plt.title('Average Delay')
    plt.xlabel('Episode')
    plt.ylabel('Delay (s)')
    
    plt.subplot(2, 2, 3)
    plt.plot(avg_energies)
    plt.title('Average Energy Consumption')
    plt.xlabel('Episode')
    plt.ylabel('Energy (J)')
    
    plt.subplot(2, 2, 4)
    plt.plot(success_rates)
    plt.title('Task Success Rate')
    plt.xlabel('Episode')
    plt.ylabel('Success Rate')
    
    plt.tight_layout()
    plt.savefig(f"python/results/training_results_{args.scenario}.png")
    
    logger.info(f"Training results saved to python/results/training_results_{args.scenario}.png")
    
    return result_data

if __name__ == "__main__":
    args = parse_args()
    train_ddpg(args)