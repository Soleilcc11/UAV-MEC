"""
优化的DDPG实现，用于无人机辅助移动边缘计算任务卸载
"""

import numpy as np
import tensorflow as tf
import time
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import datetime
import logging
from collections import deque
from tensorflow.keras import layers, initializers, regularizers
from matplotlib import font_manager


class PrioritizedReplayBuffer:
    """优先级经验回放缓冲区"""
    def __init__(self, capacity, state_dim, action_dim, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.alpha = alpha      # 优先级指数
        self.beta = beta        # 重要性采样指数
        self.beta_increment = beta_increment  # beta增量
        self.epsilon = 1e-6     # 优先级的小偏移量
        
        # 存储缓冲区
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)
        
        # 优先级缓冲区
        self.priorities = np.zeros(capacity, dtype=np.float32)
        
        self.index = 0
        self.size = 0
        self.max_priority = 1.0
        
        # 追踪最近的奖励，用于奖励归一化
        self.recent_rewards = deque(maxlen=1000)
    
    def add(self, state, action, reward, next_state, done, error=None):
        """添加经验样本"""
        # 如果没有提供TD误差，使用最大优先级
        priority = self.max_priority if error is None else (np.abs(error) + self.epsilon) ** self.alpha
        
        self.states[self.index] = state
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.next_states[self.index] = next_state
        self.dones[self.index] = done
        self.priorities[self.index] = priority
        
        # 记录奖励用于归一化
        self.recent_rewards.append(reward)
        
        # 更新索引和大小
        self.index = (self.index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size):
        """优先级采样经验批次"""
        if self.size < batch_size:
            # 如果缓冲区中的样本不足，返回None
            return None, None, None, None, None, None, None
        
        # 计算采样概率
        if self.size == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.size]
        
        probs = priorities / np.sum(priorities)
        
        # 采样索引
        indices = np.random.choice(len(probs), batch_size, p=probs)
        
        # 计算重要性权重
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # 归一化权重
        
        # 增加beta值
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
            weights,
            indices
        )
    
    def update_priorities(self, indices, errors):
        """更新TD误差对应的优先级"""
        for idx, error in zip(indices, errors):
            priority = (np.abs(error) + self.epsilon) ** self.alpha
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
    
    def get_reward_stats(self):
        """获取奖励的均值和标准差，用于归一化"""
        if len(self.recent_rewards) == 0:
            return 0.0, 1.0
        return np.mean(self.recent_rewards), max(np.std(self.recent_rewards), 1e-6)


class NoisyDense(layers.Layer):
    """带参数噪声的全连接层，用于参数空间的探索"""
    def __init__(self, units, activation=None, sigma_init=0.017):
        super(NoisyDense, self).__init__()
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.sigma_init = sigma_init
        
    def build(self, input_shape):
        self.w_mu = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer=initializers.GlorotUniform(),
            trainable=True,
            name='w_mu'
        )
        self.w_sigma = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer=initializers.Constant(self.sigma_init),
            trainable=True,
            name='w_sigma'
        )
        self.b_mu = self.add_weight(
            shape=(self.units,),
            initializer=initializers.Zeros(),
            trainable=True,
            name='b_mu'
        )
        self.b_sigma = self.add_weight(
            shape=(self.units,),
            initializer=initializers.Constant(self.sigma_init),
            trainable=True,
            name='b_sigma'
        )
        
    def call(self, inputs, training=None):
        if training:
            # 生成噪声
            eps_in = tf.random.normal(shape=(inputs.shape[-1], 1))
            eps_out = tf.random.normal(shape=(1, self.units))
            
            # 计算噪声因子
            eps_w = tf.matmul(eps_in, eps_out)
            eps_b = eps_out
            
            # 应用噪声
            w = self.w_mu + self.w_sigma * eps_w
            b = self.b_mu + self.b_sigma * eps_b[:, 0]
        else:
            w = self.w_mu
            b = self.b_mu
            
        output = tf.matmul(inputs, w) + b
        if self.activation is not None:
            output = self.activation(output)
            
        return output


class EnhancedActor(tf.keras.Model):
    """增强型Actor网络：输出动作"""
    def __init__(self, state_dim, action_dim, max_action, hidden_sizes=(512, 256), use_noisy=True):
        super(EnhancedActor, self).__init__()
        
        self.max_action = max_action
        self.use_noisy = use_noisy
        
        # 层归一化
        self.layer_norm = layers.LayerNormalization()
        
        # 使用更好的初始化方法
        kernel_init = initializers.RandomUniform(minval=-3e-3, maxval=3e-3)
        last_kernel_init = initializers.RandomUniform(minval=-3e-3, maxval=3e-3)
        
        # L2正则化
        regularizer = regularizers.l2(1e-4)
        
        # 构建网络层
        self.layers_list = []
        
        # 输入层和隐藏层
        input_dim = state_dim
        for i, hidden_size in enumerate(hidden_sizes):
            if self.use_noisy:
                self.layers_list.append(NoisyDense(hidden_size, activation='relu'))
            else:
                self.layers_list.append(
                    layers.Dense(hidden_size, 
                                activation='relu', 
                                kernel_initializer=kernel_init,
                                kernel_regularizer=regularizer,
                                input_shape=(input_dim,))
                )
            if i < len(hidden_sizes) - 1:  # 不在最后一个隐藏层后添加dropout
                self.layers_list.append(layers.Dropout(0.1))
            input_dim = hidden_size
        
        # 输出层 (使用tanh激活以约束动作范围)
        if self.use_noisy:
            self.output_layer = NoisyDense(action_dim, activation='tanh')
        else:
            self.output_layer = layers.Dense(action_dim, 
                                         activation='tanh',
                                         kernel_initializer=last_kernel_init,
                                         kernel_regularizer=regularizer)
        
        # 构建模型
        self.build_model(state_dim)
    
    def build_model(self, state_dim):
        """构建模型以确保权重初始化"""
        # 输入一个批次的虚拟数据以构建模型
        dummy_state = tf.random.normal((4, state_dim))
        self(dummy_state)
    
    def call(self, state, training=None):
        """前向传播"""
        # 应用层归一化
        x = self.layer_norm(state)
        
        # 应用隐藏层
        for layer in self.layers_list:
            if isinstance(layer, layers.Dropout):
                x = layer(x, training=training)
            elif isinstance(layer, NoisyDense):
                x = layer(x, training=training)
            else:
                x = layer(x)
        
        # 应用输出层
        if isinstance(self.output_layer, NoisyDense):
            x = self.output_layer(x, training=training)
        else:
            x = self.output_layer(x)
        
        return x * self.max_action
    
    def get_action(self, state, noise_std=0.1, training=False):
        """获取带噪声的动作，用于探索"""
        # 在无噪声层的情况下添加动作噪声
        if not self.use_noisy:
            action = self(state, training=training)
            noise = tf.random.normal(shape=action.shape, stddev=noise_std)
            noisy_action = action + noise
            return tf.clip_by_value(noisy_action, -self.max_action, self.max_action)
        
        # 在有噪声层的情况下直接使用前向传播
        return tf.clip_by_value(self(state, training=training), -self.max_action, self.max_action)


class EnhancedCritic(tf.keras.Model):
    """增强型Critic网络：评估状态和动作"""
    def __init__(self, state_dim, action_dim, hidden_sizes=(512, 256), use_noisy=True):
        super(EnhancedCritic, self).__init__()
        
        self.use_noisy = use_noisy
        
        # 层归一化
        self.state_norm = layers.LayerNormalization()
        
        # 使用更好的初始化方法
        kernel_init = initializers.GlorotUniform()
        last_kernel_init = initializers.RandomUniform(minval=-3e-3, maxval=3e-3)
        
        # L2正则化
        regularizer = regularizers.l2(1e-4)
        
        # 第一个Q网络
        self.q1_layers = []
        input_dim = state_dim + action_dim
        for hidden_size in hidden_sizes:
            if self.use_noisy:
                self.q1_layers.append(NoisyDense(hidden_size, activation='relu'))
            else:
                self.q1_layers.append(
                    layers.Dense(hidden_size, 
                                activation='relu', 
                                kernel_initializer=kernel_init,
                                kernel_regularizer=regularizer,
                                input_shape=(input_dim,))
                )
            input_dim = hidden_size
        
        if self.use_noisy:
            self.q1_output = NoisyDense(1)
        else:
            self.q1_output = layers.Dense(1, 
                                        kernel_initializer=last_kernel_init,
                                        kernel_regularizer=regularizer)
        
        # 第二个Q网络 (双Q学习)
        self.q2_layers = []
        input_dim = state_dim + action_dim
        for hidden_size in hidden_sizes:
            if self.use_noisy:
                self.q2_layers.append(NoisyDense(hidden_size, activation='relu'))
            else:
                self.q2_layers.append(
                    layers.Dense(hidden_size, 
                                activation='relu', 
                                kernel_initializer=kernel_init,
                                kernel_regularizer=regularizer,
                                input_shape=(input_dim,))
                )
            input_dim = hidden_size
        
        if self.use_noisy:
            self.q2_output = NoisyDense(1)
        else:
            self.q2_output = layers.Dense(1, 
                                        kernel_initializer=last_kernel_init,
                                        kernel_regularizer=regularizer)
        
        # 构建模型
        self.build_model(state_dim, action_dim)
    
    def build_model(self, state_dim, action_dim):
        """构建模型以确保权重初始化"""
        dummy_state = tf.random.normal((4, state_dim))
        dummy_action = tf.random.normal((4, action_dim))
        
        self([dummy_state, dummy_action])
    
    def call(self, inputs, training=None):
        """前向传播 - 输出两个Q值"""
        if isinstance(inputs, list) or isinstance(inputs, tuple):
            state, action = inputs
            # 对状态应用层归一化
            state = self.state_norm(state)
            sa = tf.concat([state, action], axis=-1)
        else:
            sa = inputs
        
        # 第一个Q网络
        q1 = sa
        for layer in self.q1_layers:
            if isinstance(layer, NoisyDense):
                q1 = layer(q1, training=training)
            else:
                q1 = layer(q1)
        
        if isinstance(self.q1_output, NoisyDense):
            q1 = self.q1_output(q1, training=training)
        else:
            q1 = self.q1_output(q1)
        
        # 第二个Q网络
        q2 = sa
        for layer in self.q2_layers:
            if isinstance(layer, NoisyDense):
                q2 = layer(q2, training=training)
            else:
                q2 = layer(q2)
        
        if isinstance(self.q2_output, NoisyDense):
            q2 = self.q2_output(q2, training=training)
        else:
            q2 = self.q2_output(q2)
        
        return q1, q2
    
    def Q1(self, inputs, training=None):
        """仅返回第一个Q值 (用于策略优化)"""
        if isinstance(inputs, list) or isinstance(inputs, tuple):
            state, action = inputs
            # 对状态应用层归一化
            state = self.state_norm(state)
            sa = tf.concat([state, action], axis=-1)
        else:
            sa = inputs
        
        q1 = sa
        for layer in self.q1_layers:
            if isinstance(layer, NoisyDense):
                q1 = layer(q1, training=training)
            else:
                q1 = layer(q1)
        
        if isinstance(self.q1_output, NoisyDense):
            q1 = self.q1_output(q1, training=training)
        else:
            q1 = self.q1_output(q1)
        
        return q1


class OptimizedDDPG:
    """优化的深度确定性策略梯度算法"""
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        actor_lr=3e-4,
        critic_lr=3e-4,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        use_prioritized_replay=True,
        use_noisy_layers=True,
        reward_scale=1.0,
        hidden_sizes=(512, 256),
        gamma_schedule=False
    ):
        # 初始化参数
        self.discount = discount
        self.initial_discount = discount  # 用于学习率调度
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.max_action = max_action
        self.use_prioritized_replay = use_prioritized_replay
        self.reward_scale = reward_scale
        self.gamma_schedule = gamma_schedule
        self.epsilon = 1.0  # 初始探索率
        self.min_epsilon = 0.1  # 最小探索率
        self.epsilon_decay = 0.99  # 探索率衰减
        
        # 初始化Actor网络
        self.actor = EnhancedActor(state_dim, action_dim, max_action, hidden_sizes, use_noisy_layers)
        self.actor_target = EnhancedActor(state_dim, action_dim, max_action, hidden_sizes, use_noisy_layers)
        
        # 确保目标网络初始权重相同
        self.actor_target.set_weights(self.actor.get_weights())
        
        # 初始化Actor优化器
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)
        
        # 初始化Critic网络
        self.critic = EnhancedCritic(state_dim, action_dim, hidden_sizes, use_noisy_layers)
        self.critic_target = EnhancedCritic(state_dim, action_dim, hidden_sizes, use_noisy_layers)
        
        # 确保目标网络初始权重相同
        self.critic_target.set_weights(self.critic.get_weights())
        
        # 初始化Critic优化器
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)
        
        # 初始化训练步数
        self.total_it = 0
        
        # 训练日志
        self.train_summaries = []
    
    def select_action(self, state, noise=0.1):
        """选择动作"""
        state = tf.convert_to_tensor(state.reshape(1, -1), dtype=tf.float32)
        
        # 使用当前探索率决定是随机探索还是使用策略
        if np.random.rand() < self.epsilon:
            # 随机动作 - 确保大小正确
            action_dim = self.actor.output_layer.units  # 获取动作维度
            return np.random.uniform(-self.max_action, self.max_action, size=action_dim)
        
        # 使用Actor网络预测动作
        action = self.actor(state, training=False).numpy().flatten()
        
        # 添加噪声进行探索
        if noise > 0:
            action += np.random.normal(0, noise * self.max_action, size=action.shape)
            
        # 裁剪动作范围
        return np.clip(action, -self.max_action, self.max_action)
    
    def train(self, replay_buffer, batch_size=256):
        """训练网络"""
        self.total_it += 1
        
        # 更新探索率
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        
        # 更新折扣因子（如果启用调度）
        if self.gamma_schedule:
            progress = min(self.total_it / 100000, 1)
            self.discount = self.initial_discount + (0.999 - self.initial_discount) * progress
        
        # 从回放缓冲区采样
        if self.use_prioritized_replay:
            # 使用优先级回放
            experience = replay_buffer.sample(batch_size)
            if experience[0] is None:  # 如果缓冲区中的样本不足
                return
                
            state, action, reward, next_state, done, weights, indices = experience
            weights = tf.convert_to_tensor(weights, dtype=tf.float32)
        else:
            # 使用普通回放
            state, action, reward, next_state, done = replay_buffer.sample(batch_size)
            weights = tf.ones(batch_size)  # 不使用重要性采样权重
            indices = None
        
        # 获取奖励归一化统计信息
        if hasattr(replay_buffer, 'get_reward_stats'):
            reward_mean, reward_std = replay_buffer.get_reward_stats()
            normalized_reward = (reward - reward_mean) / reward_std
            reward = normalized_reward * self.reward_scale
        
        # 转换为TensorFlow张量
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        action = tf.convert_to_tensor(action, dtype=tf.float32)
        reward = tf.convert_to_tensor(reward, dtype=tf.float32)
        next_state = tf.convert_to_tensor(next_state, dtype=tf.float32)
        done = tf.convert_to_tensor(done, dtype=tf.float32)
        
        # 更新Critic网络
        with tf.GradientTape() as tape:
            # 选择下一个动作并添加目标策略噪声
            noise = tf.clip_by_value(
                tf.random.normal(action.shape, stddev=self.policy_noise),
                -self.noise_clip,
                self.noise_clip
            )
            
            next_action = tf.clip_by_value(
                self.actor_target(next_state, training=True) + noise,
                -self.max_action,
                self.max_action
            )
            
            # 计算目标Q值
            target_Q1, target_Q2 = self.critic_target([next_state, next_action], training=True)
            target_Q = tf.minimum(target_Q1, target_Q2)
            target_Q = reward[:, None] + (1 - done[:, None]) * self.discount * target_Q
            
            # 获取当前Q估计
            current_Q1, current_Q2 = self.critic([state, action], training=True)
            
            # 计算TD误差
            td_error1 = tf.abs(current_Q1 - target_Q)
            td_error2 = tf.abs(current_Q2 - target_Q)
            td_error = tf.reduce_mean(tf.concat([td_error1, td_error2], axis=1), axis=1)
            
            # 应用重要性采样权重
            critic_loss = tf.reduce_mean(weights * (tf.square(current_Q1 - target_Q) + tf.square(current_Q2 - target_Q)))
        
        # 应用Critic梯度
        critic_gradients = tape.gradient(critic_loss, self.critic.trainable_variables)
        # 梯度裁剪
        critic_gradients = [tf.clip_by_norm(g, 1.0) for g in critic_gradients]
        self.critic_optimizer.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))
        
        # 更新优先级
        if self.use_prioritized_replay and indices is not None:
            # 将TD误差转换为numpy数组
            td_errors_np = td_error.numpy()
            # 更新回放缓冲区中样本的优先级
            replay_buffer.update_priorities(indices, td_errors_np)
        
        # 延迟策略更新
        actor_loss = 0
        if self.total_it % self.policy_freq == 0:
            # 更新Actor网络
            with tf.GradientTape() as tape:
                # 使用当前策略选择动作
                actor_actions = self.actor(state, training=True)
                # Actor的损失是负的Q值（我们想要最大化Q值）
                actor_loss = -tf.reduce_mean(self.critic.Q1([state, actor_actions], training=True))
            
            # 应用Actor梯度
            actor_gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
            # 梯度裁剪
            actor_gradients = [tf.clip_by_norm(g, 1.0) for g in actor_gradients]
            self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))
            
            # 软更新目标网络
            self._update_target_networks()
        
        # 记录训练摘要
        self.train_summaries.append({
            'critic_loss': float(critic_loss.numpy()),
            'actor_loss': float(actor_loss) if isinstance(actor_loss, tf.Tensor) else actor_loss,
            'td_error_mean': float(tf.reduce_mean(td_error).numpy()),
            'q_value_mean': float(tf.reduce_mean(current_Q1).numpy()),
            'target_q_mean': float(tf.reduce_mean(target_Q).numpy()),
            'reward_mean': float(tf.reduce_mean(reward).numpy()),
            'epsilon': self.epsilon,
            'discount': self.discount
        })
        
        return {
            'critic_loss': float(critic_loss.numpy()),
            'actor_loss': float(actor_loss) if isinstance(actor_loss, tf.Tensor) else actor_loss,
            'q_mean': float(tf.reduce_mean(current_Q1).numpy())
        }
    
    def _update_target_networks(self):
        """软更新目标网络"""
        # 更新目标Actor网络
        for target_param, param in zip(self.actor_target.variables, self.actor.variables):
            target_param.assign((1 - self.tau) * target_param + self.tau * param)
        
        # 更新目标Critic网络
        for target_param, param in zip(self.critic_target.variables, self.critic.variables):
            target_param.assign((1 - self.tau) * target_param + self.tau * param)
    
    def save(self, filepath):
        """保存模型"""
        # 确保路径存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        try:
            self.actor.save_weights(f"{filepath}_actor.h5")
            self.critic.save_weights(f"{filepath}_critic.h5")
            
            # 保存模型配置信息
            model_info = {
                "actor_lr": self.actor_optimizer.learning_rate.numpy(),
                "critic_lr": self.critic_optimizer.learning_rate.numpy(),
                "discount": self.discount,
                "tau": self.tau,
                "policy_noise": self.policy_noise,
                "noise_clip": self.noise_clip,
                "policy_freq": self.policy_freq,
                "total_it": self.total_it,
                "epsilon": self.epsilon
            }
            
            # 保存配置
            import json
            with open(f"{filepath}_config.json", 'w') as f:
                json.dump(model_info, f, indent=2)
                
            print(f"模型成功保存到 {filepath}")
            return True
        except Exception as e:
            print(f"保存模型时出错: {str(e)}")
            
            # 尝试使用SavedModel格式保存
            try:
                tf.saved_model.save(self.actor, f"{filepath}_actor_saved_model")
                tf.saved_model.save(self.critic, f"{filepath}_critic_saved_model")
                print(f"使用SavedModel格式成功保存模型到 {filepath}")
                return True
            except Exception as e2:
                print(f"备用保存方法也失败: {str(e2)}")
                return False
    
    def load(self, filepath):
        """加载模型"""
        try:
            self.actor.load_weights(f"{filepath}_actor.h5")
            self.critic.load_weights(f"{filepath}_critic.h5")
            self.actor_target.load_weights(f"{filepath}_actor.h5")
            self.critic_target.load_weights(f"{filepath}_critic.h5")
            
            # 加载配置信息
            import json
            with open(f"{filepath}_config.json", 'r') as f:
                model_info = json.load(f)
                
            # 更新部分配置参数
            self.discount = model_info.get("discount", self.discount)
            self.tau = model_info.get("tau", self.tau)
            self.policy_noise = model_info.get("policy_noise", self.policy_noise)
            self.noise_clip = model_info.get("noise_clip", self.noise_clip)
            self.policy_freq = model_info.get("policy_freq", self.policy_freq)
            self.total_it = model_info.get("total_it", self.total_it)
            self.epsilon = model_info.get("epsilon", self.epsilon)
            
            print(f"模型成功从 {filepath} 加载")
            return True
        except Exception as e:
            print(f"加载模型时出错: {str(e)}")
            # 尝试加载SavedModel格式
            try:
                # 尝试加载SavedModel格式
                self.actor = tf.saved_model.load(f"{filepath}_actor_saved_model")
                self.critic = tf.saved_model.load(f"{filepath}_critic_saved_model")
                # 复制到目标网络
                self.actor_target = tf.saved_model.load(f"{filepath}_actor_saved_model")
                self.critic_target = tf.saved_model.load(f"{filepath}_critic_saved_model")
                print(f"使用SavedModel格式成功加载模型")
                return True
            except Exception as e2:
                print(f"备用加载方法也失败: {str(e2)}")
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


# 修改后的训练函数
def train_optimized_ddpg(env, args):
    import logging
    logger = logging.getLogger("OptimizedDDPG")
    
    # 获取状态和动作维度
    state = env.reset()
    state_dim = len(state)
    action_dim = env.num_users + env.num_uavs * 3  # 任务卸载决策 + 无人机移动控制
    max_action = 1.0
    
    logger.info(f"环境初始化，状态维度={state_dim}，动作维度={action_dim}")
    logger.info(f"训练参数：{args.num_users}个用户，{args.num_uavs}个UAV，场景：{args.scenario}")
    
    # 初始化优化版DDPG代理
    policy = OptimizedDDPG(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        discount=args.discount,
        tau=args.tau,
        policy_noise=args.noise,
        noise_clip=args.noise_clip,
        policy_freq=args.policy_freq,
        use_prioritized_replay=args.use_per,
        use_noisy_layers=args.use_noisy,
        reward_scale=args.reward_scale,
        hidden_sizes=args.hidden_sizes,
        gamma_schedule=args.gamma_schedule
    )
    
    # 加载已有模型（如果指定）
    if args.load_model:
        model_path = os.path.join(args.model_dir, f"ddpg_{args.scenario}")
        if os.path.exists(f"{model_path}_actor.h5") or os.path.exists(f"{model_path}_actor_saved_model"):
            logger.info(f"尝试从 {model_path} 加载模型")
            if policy.load(model_path):
                logger.info("模型加载成功")
            else:
                logger.warning("模型加载失败，使用随机初始化模型")
        else:
            logger.warning("未找到预训练模型，使用随机初始化模型")
    
    # 初始化经验回放缓冲区
    if args.use_per:
        replay_buffer = PrioritizedReplayBuffer(
            capacity=args.buffer_size,
            state_dim=state_dim,
            action_dim=action_dim,
            alpha=args.per_alpha,
            beta=args.per_beta
        )
    else:
        # 使用普通经验回放
        from ddpg_agent import ReplayBuffer
        replay_buffer = ReplayBuffer(args.buffer_size, state_dim, action_dim)
    
    # 预填充缓冲区
    if args.prefill > 0:
        logger.info(f"开始预填充经验回放缓冲区 ({args.prefill} 步)...")
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
    logger.info(f"开始训练 {args.episodes} 个回合...")
    best_reward = -float('inf')
    episodes_without_improvement = 0

    exploration_episodes = int(args.episodes * 0.7)  # 将70%的回合用于探索

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

        # 修改：强制在探索期保持较高的探索率
        if episode < exploration_episodes:
            current_epsilon = max(policy.epsilon, 0.4)  # 探索期内最小保持0.4的探索率
        else:
            current_epsilon = policy.epsilon
            
        # 一个回合的交互
        for step in range(args.steps):
            # 使用当前探索率决定是随机探索还是使用策略
            if np.random.rand() < current_epsilon:
                # 随机动作 - 确保大小正确
                action = np.random.uniform(-1, 1, size=action_dim)
            else:
                # 使用Actor网络预测动作
                action = policy.select_action(state, noise=args.action_noise)

            # 选择动作
            # action = policy.select_action(state, noise=args.action_noise)
            
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
            
            # 确定当前批次大小
            current_batch_size = min(args.batch_size, replay_buffer.size)
            
            # 如果有足够样本，进行训练
            if replay_buffer.size >= args.min_train_size:
                train_info = policy.train(replay_buffer, current_batch_size)
                episode_train_info.append(train_info)
                if args.verbose and step % 10 == 0:
                    print(f"训练步骤 {episode_steps}，Q值均值: {train_info['q_mean']:.4f}, 损失: {train_info['critic_loss']:.4f}")
            else:
                if args.verbose:
                    print(f"缓冲区大小 {replay_buffer.size} 尚不足以开始训练 (需要 {args.min_train_size})")
                        
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
            best_model_path = os.path.join(args.model_dir, f"ddpg_{args.scenario}_best")
            policy.save(best_model_path)
            logger.info(f"发现更好的模型 (奖励: {best_reward:.2f})，已保存到 {best_model_path}")
        else:
            episodes_without_improvement += 1
        
        # 学习率衰减
        if args.lr_decay and episodes_without_improvement > args.patience:
            policy.actor_optimizer.learning_rate = policy.actor_optimizer.learning_rate * 0.9
            policy.critic_optimizer.learning_rate = policy.critic_optimizer.learning_rate * 0.9
            episodes_without_improvement = 0
            logger.info(f"学习率衰减: Actor={policy.actor_optimizer.learning_rate.numpy():.6f}, " +
                      f"Critic={policy.critic_optimizer.learning_rate.numpy():.6f}")
        
        # 定期保存检查点
        if (episode + 1) % args.save_interval == 0:
            checkpoint_path = os.path.join(args.model_dir, f"ddpg_{args.scenario}_ep{episode+1}")
            policy.save(checkpoint_path)
            logger.info(f"检查点已保存到 {checkpoint_path}")
    
    # 保存最终模型
    final_model_path = os.path.join(args.model_dir, f"ddpg_{args.scenario}_final")
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
    result_path = os.path.join(args.results_dir, f"priddpg_training_data_{args.scenario}.json")
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
        os.path.join(args.results_dir, f"training_results_{args.scenario}.png"),
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
    
    # 如果有训练信息，绘制损失和Q值
    if train_info:
        # 提取损失值
        critic_losses = [info.get('critic_loss', 0) for info in train_info]
        q_values = [info.get('q_mean', 0) for info in train_info]
        
        # 损失图
        plt.subplot(3, 2, 5)
        plt.plot(critic_losses)
        # 平滑损失
        if len(critic_losses) > window_size:
            smoothed_loss = np.convolve(critic_losses, np.ones(window_size)/window_size, mode='valid')
            plt.plot(range(window_size-1, window_size-1+len(smoothed_loss)), smoothed_loss, 'r-', alpha=0.7)
        plt.title('Critic 损失')
        plt.xlabel('回合')
        plt.ylabel('损失值')
        plt.yscale('log')
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
    plt.suptitle(f'UAV-MEC训练结果 - {scenario}场景', fontsize=16)
    plt.subplots_adjust(top=0.95)
    # 保存带时间戳的图表
    plt.savefig(timestamped_path, dpi=300, bbox_inches='tight')

    # plt.savefig(save_path)

    print(f"训练结果图表已保存到 {timestamped_path}")
    print(f"训练结果图表已保存到 {save_path}")

    plt.close()
    return save_path


def prefill_buffer(env, replay_buffer, prefill_steps=1000):
    """使用随机动作预填充经验回放缓冲区"""
    from tqdm import tqdm
    print(f"开始预填充经验回放缓冲区，目标步数: {prefill_steps}...")
    state = env.reset()
    
    action_dim = env.num_users + env.num_uavs * 3

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
    
    print(f"预填充完成，经验回放缓冲区大小: {replay_buffer.size}")


# 测试函数：用于验证训练后的模型
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
            action = policy.select_action(state, noise=0)
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
    
    parser = argparse.ArgumentParser(description="训练优化版DDPG用于UAV-MEC任务卸载")
    # 环境参数
    parser.add_argument("--num_users", type=int, default=50, help="用户数量")
    parser.add_argument("--num_uavs", type=int, default=3, help="UAV数量")
    parser.add_argument("--scenario", type=str, default="default", help="场景名称 (default, urban, rural)")
    
    # 训练参数
    parser.add_argument("--episodes", type=int, default=1000, help="训练回合数")
    parser.add_argument("--steps", type=int, default=200, help="每回合最大步数")
    parser.add_argument("--batch_size", type=int, default=256, help="训练批次大小")
    parser.add_argument("--min_train_size", type=int, default=1000, help="开始训练所需的最小样本数")
    parser.add_argument("--buffer_size", type=int, default=int(1e6), help="经验回放缓冲区容量")
    parser.add_argument("--prefill", type=int, default=1000, help="预填充步数")
    
    # 算法参数
    parser.add_argument("--actor_lr", type=float, default=3e-4, help="Actor学习率")
    parser.add_argument("--critic_lr", type=float, default=3e-4, help="Critic学习率")
    parser.add_argument("--discount", type=float, default=0.99, help="折扣因子")
    parser.add_argument("--tau", type=float, default=0.005, help="软更新系数")
    parser.add_argument("--noise", type=float, default=0.2, help="策略噪声标准差")
    parser.add_argument("--noise_clip", type=float, default=0.5, help="噪声裁剪范围")
    parser.add_argument("--policy_freq", type=int, default=2, help="策略更新频率")
    parser.add_argument("--action_noise", type=float, default=0.1, help="动作探索噪声")
    parser.add_argument("--reward_scale", type=float, default=0.1, help="奖励缩放因子")
    parser.add_argument("--hidden_sizes", nargs='+', type=int, default=[512, 256], help="隐藏层大小")
    
    # 优化选项
    parser.add_argument("--use_per", action="store_true", help="使用优先级经验回放")
    parser.add_argument("--per_alpha", type=float, default=0.6, help="PER alpha参数")
    parser.add_argument("--per_beta", type=float, default=0.4, help="PER beta参数")
    parser.add_argument("--use_noisy", action="store_true", help="使用噪声网络层")
    parser.add_argument("--gamma_schedule", action="store_true", help="使用折扣因子调度")
    parser.add_argument("--lr_decay", action="store_true", help="启用学习率衰减")
    parser.add_argument("--patience", type=int, default=50, help="学习率衰减的耐心参数")
    
    # 其他选项
    parser.add_argument("--load_model", action="store_true", help="加载现有模型")
    parser.add_argument("--model_dir", type=str, default="results/saved_models", help="模型保存目录")
    parser.add_argument("--results_dir", type=str, default="results/plots", help="结果保存目录")
    parser.add_argument("--log_interval", type=int, default=10, help="日志记录间隔")
    parser.add_argument("--save_interval", type=int, default=100, help="模型保存间隔")
    parser.add_argument("--evaluate", action="store_true", help="评估模式")
    parser.add_argument("--eval_episodes", type=int, default=10, help="评估回合数")
    parser.add_argument("--verbose", action="store_true", help="详细输出模式")
    
    args = parser.parse_args()
    
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)
 
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "optimized_ddpg.log")),
            logging.StreamHandler()
        ]
    )
    
    # 创建环境
    from uav_mec_env import UAVMECEnvironment
    
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
        max_action = 1.0
        
        # 加载模型
        policy = OptimizedDDPG(state_dim, action_dim, max_action)
        model_path = os.path.join(args.model_dir, f"ddpg_{args.scenario}_best")
        
        if policy.load(model_path):
            print(f"已加载模型 {model_path} 进行评估")
            evaluate_policy(policy, env, episodes=args.eval_episodes, render=True)
        else:
            print(f"无法加载模型，评估失败")
    else:
        # 训练模式
        policy, results = train_optimized_ddpg(env, args)