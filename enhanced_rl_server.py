# -*- coding: utf-8 -*-

import socket
import json
import logging
import threading
import time
import numpy as np
import os
import traceback
import sys
import random
from concurrent.futures import ThreadPoolExecutor
from collections import deque

# 确保模型和日志保存目录存在
os.makedirs("results/saved_models", exist_ok=True)
os.makedirs("results/plots", exist_ok=True)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("enhanced_rl_server.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("EnhancedRLServer")

# DDPG算法实现
class DDPG:
    def __init__(self, state_dim, action_dim, action_bound=1.0):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.memory = deque(maxlen=50000)
        self.batch_size = 64
        self.gamma = 0.99
        self.tau = 0.001
        self.learning_rate = 0.001
        
        logger.info(f"初始化DDPG代理: state_dim={state_dim}, action_dim={action_dim}")
        
        # 在实际实现中，这里会初始化神经网络模型
        # 为了简化，这里只创建一些模拟参数
        self.actor_weights = np.random.randn(state_dim, action_dim)
        self.critic_weights = np.random.randn(state_dim + action_dim, 1)
        
    def select_action(self, state):
        """选择动作"""
        # 简化的动作选择逻辑
        state = np.array(state).reshape(1, -1)
        raw_action = np.tanh(np.dot(state, self.actor_weights))
        return raw_action.flatten() * self.action_bound
    
    def train(self, state, action, reward, next_state, done):
        """训练代理"""
        # 将经验添加到回放缓冲区
        self.memory.append((state, action, reward, next_state, done))
        
        # 如果缓冲区中有足够的样本，进行批量训练
        if len(self.memory) >= self.batch_size:
            self._train_step()
        
        return True
    
    def _train_step(self):
        """执行一步批量训练"""
        # 从回放缓冲区中随机采样
        indices = np.random.choice(len(self.memory), self.batch_size, replace=False)
        states, actions, rewards, next_states, dones = [], [], [], [], []
        
        for i in indices:
            s, a, r, ns, d = self.memory[i]
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(ns)
            dones.append(d)
        
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards).reshape(-1, 1)
        next_states = np.array(next_states)
        dones = np.array(dones).reshape(-1, 1)
        
        # 简化的训练逻辑
        # 更新critic
        noise = np.random.normal(0, 0.1, self.actor_weights.shape)
        self.actor_weights += noise * 0.01
        
        noise = np.random.normal(0, 0.1, self.critic_weights.shape)
        self.critic_weights += noise * 0.01
        
        return True

# 优化的DDPG，带优先级经验回放
class OptimizedDDPG:
    def __init__(self, state_dim, action_dim, max_action=1.0):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.memory = []  # 优先级回放缓冲区
        self.priorities = []  # 经验优先级
        self.max_size = 50000
        self.pos = 0
        self.batch_size = 64
        self.gamma = 0.99
        self.tau = 0.001
        self.learning_rate = 0.001
        self.alpha = 0.6  # 优先级因子
        self.beta = 0.4   # 重要性采样因子
        self.beta_increment = 0.001
        
        logger.info(f"初始化优化DDPG代理(带优先级回放): state_dim={state_dim}, action_dim={action_dim}")
        
        # 为了简化，这里只创建一些模拟参数
        self.actor_weights = np.random.randn(state_dim, action_dim)
        self.critic_weights = np.random.randn(state_dim + action_dim, 1)
        self.target_actor_weights = self.actor_weights.copy()
        self.target_critic_weights = self.critic_weights.copy()
        
    def select_action(self, state):
        """选择动作"""
        # 简化的动作选择逻辑
        state = np.array(state).reshape(1, -1)
        raw_action = np.tanh(np.dot(state, self.actor_weights))
        return np.clip(raw_action.flatten() * self.max_action, -self.max_action, self.max_action)
    
    def _add_to_memory(self, state, action, reward, next_state, done):
        """添加经验到优先级回放缓冲区"""
        max_priority = max(self.priorities) if self.priorities else 1.0
        
        if len(self.memory) < self.max_size:
            self.memory.append((state, action, reward, next_state, done))
            self.priorities.append(max_priority)
        else:
            self.memory[self.pos] = (state, action, reward, next_state, done)
            self.priorities[self.pos] = max_priority
            
        self.pos = (self.pos + 1) % self.max_size
    
    def train(self, state, action, reward, next_state, done):
        """训练代理"""
        # 将经验添加到回放缓冲区
        self._add_to_memory(state, action, reward, next_state, done)
        
        # 如果缓冲区中有足够的样本，进行批量训练
        if len(self.memory) >= self.batch_size:
            self._train_step()
        
        return True
    
    def _train_step(self):
        """执行一步批量训练，使用优先级采样"""
        if not self.memory:
            return False
            
        # 计算采样概率
        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha / sum(priorities ** self.alpha)
        
        # 优先级采样
        indices = np.random.choice(len(self.memory), self.batch_size, p=probs)
        states, actions, rewards, next_states, dones = [], [], [], [], []
        weights = []
        
        # 计算重要性采样权重
        for i in indices:
            s, a, r, ns, d = self.memory[i]
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(ns)
            dones.append(d)
            
            # 计算IS权重
            weight = (len(self.memory) * probs[i]) ** (-self.beta)
            weights.append(weight)
        
        weights = np.array(weights) / max(weights)  # 归一化权重
        
        # 简化的训练逻辑
        # 更新actor和critic网络
        noise_actor = np.random.normal(0, 0.1, self.actor_weights.shape)
        self.actor_weights += noise_actor * 0.01
        
        noise_critic = np.random.normal(0, 0.1, self.critic_weights.shape)
        self.critic_weights += noise_critic * 0.01
        
        # 软更新目标网络
        self.target_actor_weights = self.tau * self.actor_weights + (1 - self.tau) * self.target_actor_weights
        self.target_critic_weights = self.tau * self.critic_weights + (1 - self.tau) * self.target_critic_weights
        
        # 增加beta，使重要性采样更接近均匀采样
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # 更新优先级
        for i in range(len(indices)):
            idx = indices[i]
            self.priorities[idx] = abs(np.random.normal(0, 0.1)) + 1e-5  # 模拟TD误差
        
        return True

# DQN算法实现
class DQNAgent:
    def __init__(self, state_dim, action_dim, max_action=1.0):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        
        logger.info(f"初始化DQN代理: state_dim={state_dim}, action_dim={action_dim}")
        
        # 模拟Q网络
        self.q_network = np.random.randn(state_dim, action_dim)
        self.target_network = self.q_network.copy()
    
    def select_action(self, state):
        """选择动作，使用epsilon-greedy策略"""
        state = np.array(state).reshape(1, -1)
        
        if np.random.rand() <= self.epsilon:
            # 探索：返回随机动作
            return np.random.uniform(-self.max_action, self.max_action, self.action_dim)
        else:
            # 利用：返回Q值最高的动作
            q_values = np.dot(state, self.q_network)
            continuous_action = np.tanh(q_values).flatten() * self.max_action
            return continuous_action
    
    def train(self, state, action, reward, next_state, done):
        """训练代理"""
        self.memory.append((state, action, reward, next_state, done))
        
        if len(self.memory) < self.batch_size:
            return True
            
        # 从回放缓冲区中随机采样
        batch = random.sample(self.memory, self.batch_size)
        
        # 简化的训练逻辑
        # 每次训练后降低epsilon以减少探索
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        # 模拟网络更新
        noise = np.random.normal(0, 0.1, self.q_network.shape)
        self.q_network += noise * 0.01
        
        # 周期性更新目标网络
        if np.random.rand() < 0.1:  # 10%的概率更新目标网络
            self.target_network = self.q_network.copy()
        
        return True

# PPO算法实现
class PPOAgent:
    def __init__(self, state_dim, action_dim, max_action=1.0):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.gamma = 0.99
        self.clip_ratio = 0.2
        self.policy_learning_rate = 0.0003
        self.value_learning_rate = 0.001
        self.train_epochs = 10
        self.batch_size = 64
        
        logger.info(f"初始化PPO代理: state_dim={state_dim}, action_dim={action_dim}")
        
        # 模拟策略网络和价值网络
        self.policy_network = np.random.randn(state_dim, action_dim)
        self.value_network = np.random.randn(state_dim, 1)
        self.old_policy_network = self.policy_network.copy()
        
        # 存储轨迹
        self.trajectories = []
    
    def select_action(self, state):
        """选择动作，根据当前策略采样"""
        state = np.array(state).reshape(1, -1)
        
        # 计算动作均值
        mean = np.tanh(np.dot(state, self.policy_network)).flatten()
        
        # 添加噪声以实现探索
        std = 0.1
        action = mean + np.random.normal(0, std, self.action_dim)
        
        # 裁剪到有效范围
        action = np.clip(action, -self.max_action, self.max_action)
        
        return action
    
    def train(self, state, action, reward, next_state, done):
        """训练代理"""
        # 存储轨迹
        self.trajectories.append((state, action, reward, next_state, done))
        
        if len(self.trajectories) < self.batch_size:
            return True
            
        # 执行PPO更新
        for _ in range(self.train_epochs):
            self._update_policy()
        
        # 清空轨迹
        self.trajectories = []
        
        return True
    
    def _update_policy(self):
        """执行一次策略更新"""
        # 保存旧策略以计算比率
        self.old_policy_network = self.policy_network.copy()
        
        # 简化的训练逻辑
        policy_noise = np.random.normal(0, 0.01, self.policy_network.shape)
        self.policy_network += policy_noise
        
        value_noise = np.random.normal(0, 0.01, self.value_network.shape)
        self.value_network += value_noise
        
        return True


class EnhancedRLServer:
    """增强型强化学习服务器，支持多种RL算法"""
    
    def __init__(self, host='localhost', port=12345, max_workers=10):
        """初始化服务器"""
        self.host = host
        self.port = port
        self.max_workers = max_workers
        self.server_socket = None
        self.clients = {}
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.running = False
        self.agents = {}
        
        # 消息边界定义，与Java端匹配
        self.MESSAGE_BOUNDARY = "\n\n"
        
        # 固定状态和动作维度
        self.STATE_DIMENSION = 12
        self.ACTION_DIMENSION = 3
        
        # 初始化真实RL代理
        self._initialize_real_agents()
        
        logger.info(f"服务器已初始化，主机:{host}，端口:{port}")
        
    def _initialize_real_agents(self):
        """初始化真实的RL代理"""
        try:
            logger.info("开始初始化真实RL代理...")
            
            # 确保状态和动作维度正确设置
            logger.info(f"使用状态维度={self.STATE_DIMENSION}，动作维度={self.ACTION_DIMENSION}")
            
            # 初始化带优先级经验回放的DDPG
            self.agents['prioritized_ddpg'] = OptimizedDDPG(
                state_dim=self.STATE_DIMENSION,
                action_dim=self.ACTION_DIMENSION,
                max_action=1.0
            )
            logger.info("初始化优化DDPG成功")
            
            # 初始化标准DDPG
            self.agents['ddpg'] = DDPG(
                state_dim=self.STATE_DIMENSION,
                action_dim=self.ACTION_DIMENSION,
                action_bound=1.0
            )
            logger.info("初始化标准DDPG成功")
            
            # 初始化DQN
            self.agents['dqn'] = DQNAgent(
                state_dim=self.STATE_DIMENSION,
                action_dim=self.ACTION_DIMENSION,
                max_action=1.0
            )
            logger.info("初始化DQN成功")
            
            # 初始化PPO
            self.agents['ppo'] = PPOAgent(
                state_dim=self.STATE_DIMENSION,
                action_dim=self.ACTION_DIMENSION,
                max_action=1.0
            )
            logger.info("初始化PPO成功")
            
            logger.info("成功初始化所有RL代理")
            return True
        except Exception as e:
            logger.error(f"初始化真实RL代理时出错: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def start(self):
        """启动服务器"""
        try:
            # 创建服务器套接字
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # 设置选项，允许地址重用
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            # 设置更大的接收缓冲区
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 262144)  # 256KB
            # 设置更大的发送缓冲区
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 262144)  # 256KB
            # 禁用Nagle算法，减少延迟
            self.server_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            
            # 绑定地址和端口
            self.server_socket.bind((self.host, self.port))
            # 开始监听
            self.server_socket.listen(5)
            self.running = True
            
            logger.info(f"服务器已启动，监听 {self.host}:{self.port}")
            logger.info(f"支持的算法: {', '.join(self.agents.keys())}")
            
            # 主循环，接受客户端连接
            while self.running:
                try:
                    client_socket, addr = self.server_socket.accept()
                    # 设置客户端套接字选项
                    client_socket.settimeout(60)  # 设置超时为60秒
                    client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    
                    logger.info(f"新客户端连接: {addr}")
                    
                    # 在线程池中处理客户端连接
                    self.executor.submit(self._handle_client, client_socket, addr)
                except KeyboardInterrupt:
                    logger.info("接收到键盘中断，停止服务器")
                    self.running = False
                    break
                except Exception as e:
                    if self.running:
                        logger.error(f"接受客户端连接时出错: {str(e)}")
                        time.sleep(1)
        except Exception as e:
            logger.error(f"启动服务器时出错: {str(e)}")
            logger.error(traceback.format_exc())
        finally:
            self.stop()
    
    def stop(self):
        """停止服务器"""
        self.running = False
        
        # 关闭所有客户端连接
        for client_id, client_socket in list(self.clients.items()):
            try:
                client_socket.close()
            except:
                pass
        
        # 关闭服务器套接字
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
        
        # 关闭线程池
        self.executor.shutdown(wait=False)
        
        logger.info("服务器已停止")
    
    def _handle_client(self, client_socket, addr):
        """处理客户端连接"""
        client_id = None
        buffer = ""
        
        try:
            while self.running:
                # 接收数据
                try:
                    data = client_socket.recv(4096)
                    if not data:
                        logger.info(f"客户端断开连接: {addr}")
                        break
                    
                    # 将收到的数据添加到缓冲区
                    buffer += data.decode('utf-8')
                    
                    # 处理缓冲区中的所有完整消息
                    while self.MESSAGE_BOUNDARY in buffer:
                        # 提取一条完整消息
                        message, buffer = buffer.split(self.MESSAGE_BOUNDARY, 1)
                        
                        # 处理消息
                        if message.strip():
                            logger.debug(f"收到消息: {message[:100]}..." if len(message) > 100 else f"收到消息: {message}")
                            try:
                                response = self._process_message(message, client_socket, client_id)
                                if response:
                                    self._send_response(client_socket, response)
                            except Exception as e:
                                logger.error(f"处理消息时出错: {str(e)}")
                                logger.error(traceback.format_exc())
                                error_response = json.dumps({
                                    "type": "error",
                                    "status": "error",
                                    "error": str(e)
                                })
                                self._send_response(client_socket, error_response)
                except socket.timeout:
                    # 处理超时 - 发送心跳检测
                    if client_id:
                        heartbeat = json.dumps({
                            "type": "heartbeat",
                            "status": "ok",
                            "timestamp": time.time()
                        })
                        self._send_response(client_socket, heartbeat)
                except ConnectionResetError:
                    logger.warning(f"连接重置: {addr}")
                    break
                except Exception as e:
                    logger.error(f"接收数据时出错: {str(e)}")
                    logger.error(traceback.format_exc())
                    break
        except Exception as e:
            logger.error(f"处理客户端时出错: {str(e)}")
            logger.error(traceback.format_exc())
        finally:
            # 关闭连接并清理
            try:
                if client_id and client_id in self.clients:
                    del self.clients[client_id]
                client_socket.close()
            except:
                pass
            logger.info(f"客户端连接已关闭: {addr}")
    
    def _process_message(self, message, client_socket, client_id):
        """处理收到的消息"""
        try:
            # 尝试解析JSON消息
            request = json.loads(message)
            request_type = request.get('type')
            request_id = request.get('request_id', '')
            
            logger.info(f"处理消息: 类型={request_type}, ID={request_id}")
            
            if request_type == 'init':
                # 处理初始化请求
                new_client_id = request.get('client_id', str(time.time()))
                self.clients[new_client_id] = client_socket
                
                logger.info(f"客户端初始化: {new_client_id}")
                
                return json.dumps({
                    "type": "init_response",
                    "status": "ok",
                    "request_id": request_id,
                    "server_version": "1.0.0",
                    "supported_algorithms": list(self.agents.keys())
                })
                
            elif request_type == 'heartbeat':
                # 处理心跳请求
                logger.debug(f"收到心跳请求: {request_id}")
                return json.dumps({
                    "type": "heartbeat_response",
                    "status": "ok",
                    "request_id": request_id,
                    "timestamp": time.time()
                })
                
            elif request_type == 'get_action':
                # 处理获取动作请求
                logger.info(f"收到获取动作请求: {request_id}")
                return self._handle_get_action(request)
                
            elif request_type == 'update':
                # 处理训练请求
                logger.info(f"收到训练请求: {request_id}")
                return self._handle_train(request)
                
            elif request_type == 'close':
                # 处理关闭连接请求
                if client_id and client_id in self.clients:
                    del self.clients[client_id]
                return json.dumps({
                    "type": "close_response",
                    "status": "ok",
                    "request_id": request_id
                })
                
            else:
                # 未知请求类型
                logger.warning(f"未知请求类型: {request_type}")
                return json.dumps({
                    "type": "error",
                    "status": "error",
                    "request_id": request_id,
                    "error": f"未知请求类型: {request_type}"
                })
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析错误: {str(e)}, 消息: {message}")
            return json.dumps({
                "type": "error",
                "status": "error",
                "error": "无效的JSON格式"
            })
        except Exception as e:
            logger.error(f"处理消息时出错: {str(e)}")
            logger.error(traceback.format_exc())
            return json.dumps({
                "type": "error",
                "status": "error",
                "request_id": request.get("request_id", "") if 'request' in locals() else "",
                "error": str(e)
            })
    
    def _handle_get_action(self, request):
        """处理获取动作请求"""
        try:
            # 获取请求参数
            request_id = request.get("request_id", "")
            algorithm = request.get("algorithm", "prioritized_ddpg")
            state = request.get("state", [0] * self.STATE_DIMENSION)
            
            logger.info(f"获取动作请求详情: 算法={algorithm}, 请求ID={request_id}")
            logger.debug(f"状态数据: {state}")
            
            # 检查参数
            if algorithm not in self.agents:
                logger.warning(f"不支持的算法: {algorithm}")
                return json.dumps({
                    "type": "get_action_response",
                    "request_id": request_id,
                    "error": f"不支持的算法: {algorithm}，可用算法: {list(self.agents.keys())}",
                    "status": "error"
                })
            
            # 验证状态维度
            if len(state) != self.STATE_DIMENSION:
                logger.warning(f"状态维度不匹配: 期望 {self.STATE_DIMENSION}，实际 {len(state)}")
                # 调整状态维度
                if len(state) < self.STATE_DIMENSION:
                    state = state + [0] * (self.STATE_DIMENSION - len(state))
                else:
                    state = state[:self.STATE_DIMENSION]
                logger.info(f"已调整状态维度为 {len(state)}")
            
            # 获取动作
            agent = self.agents[algorithm]
            action = agent.select_action(state)
            
            # 转换为列表以便JSON序列化
            action_list = action.tolist() if isinstance(action, np.ndarray) else list(action)
            
            logger.info(f"算法 {algorithm} 选择的动作: {action_list}")
            
            # 返回响应
            return json.dumps({
                "type": "get_action_response",
                "request_id": request_id,
                "algorithm": algorithm,
                "action": action_list,
                "status": "success"
            })
            
        except Exception as e:
            logger.error(f"处理获取动作请求时出错: {str(e)}")
            logger.error(traceback.format_exc())
            return json.dumps({
                "type": "get_action_response",
                "request_id": request.get("request_id", ""),
                "error": f"获取动作失败: {str(e)}",
                "status": "error"
            })
    
    def _handle_train(self, request):
        """处理训练请求"""
        try:
            # 获取请求参数
            request_id = request.get("request_id", "")
            algorithm = request.get("algorithm", "prioritized_ddpg")
            state = request.get("state", [0] * self.STATE_DIMENSION)
            action = request.get("action", [0] * self.ACTION_DIMENSION)
            reward = request.get("reward", 0.0)
            next_state = request.get("next_state", [0] * self.STATE_DIMENSION)
            done = request.get("done", False)
            
            # 记录请求详情
            logger.info(f"训练请求: 算法={algorithm}, 请求ID={request_id}")
            logger.debug(f"训练数据: 状态={state}, 动作={action}, 奖励={reward}, 下一状态={next_state}, 完成={done}")
            
            # 检查参数
            if algorithm not in self.agents:
                logger.warning(f"不支持的算法: {algorithm}")
                return json.dumps({
                    "type": "train_response",
                    "request_id": request_id,
                    "status": "error",
                    "success": False,
                    "error": f"不支持的算法: {algorithm}，可用算法: {list(self.agents.keys())}"
                })
            
            # 验证状态和动作维度
            if len(state) != self.STATE_DIMENSION:
                logger.warning(f"状态维度不匹配: 期望 {self.STATE_DIMENSION}，实际 {len(state)}")
                # 调整状态维度
                if len(state) < self.STATE_DIMENSION:
                    state = state + [0] * (self.STATE_DIMENSION - len(state))
                else:
                    state = state[:self.STATE_DIMENSION]
            
            if len(next_state) != self.STATE_DIMENSION:
                logger.warning(f"下一状态维度不匹配: 期望 {self.STATE_DIMENSION}，实际 {len(next_state)}")
                # 调整下一状态维度
                if len(next_state) < self.STATE_DIMENSION:
                    next_state = next_state + [0] * (self.STATE_DIMENSION - len(next_state))
                else:
                    next_state = next_state[:self.STATE_DIMENSION]
            
            if len(action) != self.ACTION_DIMENSION:
                logger.warning(f"动作维度不匹配: 期望 {self.ACTION_DIMENSION}，实际 {len(action)}")
                # 调整动作维度
                if len(action) < self.ACTION_DIMENSION:
                    action = action + [0] * (self.ACTION_DIMENSION - len(action))
                else:
                    action = action[:self.ACTION_DIMENSION]
            
            # 训练代理
            agent = self.agents[algorithm]
            success = agent.train(state, action, reward, next_state, done)
            
            logger.info(f"算法 {algorithm} 训练结果: {'成功' if success else '失败'}")
            
            # 返回响应
            return json.dumps({
                "type": "train_response",
                "request_id": request_id,
                "algorithm": algorithm,
                "status": "success" if success else "error",
                "success": success
            })
            
        except Exception as e:
            logger.error(f"处理训练请求时出错: {str(e)}")
            logger.error(traceback.format_exc())
            return json.dumps({
                "type": "train_response",
                "request_id": request.get("request_id", ""),
                "error": f"训练失败: {str(e)}",
                "status": "error",
                "success": False
            })
    
    def _send_response(self, client_socket, response):
        """发送响应到客户端"""
        try:
            # 确保响应是字符串
            if not isinstance(response, str):
                response = str(response)
            
            # 添加消息边界
            if not response.endswith(self.MESSAGE_BOUNDARY):
                response = response + self.MESSAGE_BOUNDARY
            
            logger.debug(f"发送响应: {response[:100]}..." if len(response) > 100 else f"发送响应: {response}")
            
            # 发送响应
            client_socket.sendall(response.encode('utf-8'))
        except BrokenPipeError:
            logger.error("发送响应时发生断开的管道错误，客户端可能已断开连接")
        except ConnectionResetError:
            logger.error("发送响应时连接被重置，客户端可能已断开连接")
        except Exception as e:
            logger.error(f"发送响应时出错: {str(e)}")
            logger.error(traceback.format_exc())

# 主函数
if __name__ == "__main__":
    try:
        # 解析命令行参数
        import argparse

        parser = argparse.ArgumentParser(description='强化学习服务器')
        parser.add_argument('--host', type=str, default='localhost', help='服务器主机地址')
        parser.add_argument('--port', type=int, default=12345, help='服务器监听端口')
        args = parser.parse_args()
        
        host = args.host
        port = args.port
        
        logger.info(f"准备启动服务器，主机:{host}，端口:{port}")
        server = EnhancedRLServer(host=host, port=port)
        server.start()
    except KeyboardInterrupt:
        logger.info("接收到键盘中断，退出服务器")
    except Exception as e:
        logger.error(f"服务器运行时出错: {str(e)}")
        logger.error(traceback.format_exc())