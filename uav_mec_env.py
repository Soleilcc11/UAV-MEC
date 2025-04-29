#!/usr/bin/env python3
"""
无人机辅助移动边缘计算环境模拟
"""
import numpy as np
import tensorflow as tf

class UAVMECEnvironment:
    """无人机辅助移动边缘计算环境模拟"""
    def __init__(self, num_users=50, num_uavs=3, area_size=1000, 
                 max_task_size=5, max_task_cycles=2, max_energy=1000):
        """
        初始化环境参数
        
        参数:
            num_users: 用户数量
            num_uavs: 无人机数量
            area_size: 区域大小 (m x m)
            max_task_size: 最大任务数据大小 (MB)
            max_task_cycles: 最大任务计算复杂度 (GHz)
            max_energy: 无人机最大能量
        """
        self.num_users = num_users
        self.num_uavs = num_uavs
        self.area_size = area_size
        self.max_task_size = max_task_size
        self.max_task_cycles = max_task_cycles
        self.max_energy = max_energy
        
        # 用户位置 - 随机分布在区域内
        self.user_positions = np.random.uniform(0, area_size, size=(num_users, 2))
        
        # 无人机初始位置和能量
        self.uav_positions = np.random.uniform(0, area_size, size=(num_uavs, 3))
        self.uav_positions[:, 2] = 100  # 设置无人机高度为100m
        self.uav_energy = np.ones(num_uavs) * max_energy
        
        # 任务特性 - 随机生成
        self.task_data_size = np.random.uniform(0.5, max_task_size, size=num_users)  # MB
        self.task_cycles = np.random.uniform(0.5, max_task_cycles, size=num_users)   # GHz
        self.task_deadlines = np.random.uniform(0.5, 2, size=num_users)  # 秒
        
        # 计算资源配置
        self.local_computing_capacity = np.random.uniform(0.5, 1.5, size=num_users)  # GHz
        self.uav_computing_capacity = np.random.uniform(2, 4, size=num_uavs)  # GHz
        self.cloud_computing_capacity = 10.0  # GHz
        
        # 通信参数
        self.bandwidth = 20  # MHz
        self.noise_power = 1e-13  # W
        self.transmit_power_user = 0.1  # W
        self.transmit_power_uav = 1.0  # W
        
        # 能量消耗系数
        self.local_energy_coefficient = 1e-27  # J/cycle
        self.uav_hover_power = 5.0  # W
        self.uav_flight_power = 10.0  # W
        
        # 初始化任务状态
        self.reset()
    
    def reset(self):
        """重置环境状态"""
        # 重置任务队列
        self.active_tasks = np.ones(self.num_users)  # 当前活跃任务指示器
        
        # 重置UAV状态
        self.uav_positions = np.random.uniform(0, self.area_size, size=(self.num_uavs, 3))
        self.uav_positions[:, 2] = 100  # 设置无人机高度为100m
        self.uav_energy = np.ones(self.num_uavs) * self.max_energy
        
        # 获取当前状态
        state = self._get_state()
        return state
    
    def _get_state(self):
        """Get the current environment state representation with improved robustness"""
        # Calculate the base dimension of the state vector
        uav_pos_dim = self.num_uavs * 3  # UAV positions (x,y,z for each UAV)
        uav_energy_dim = self.num_uavs   # Energy levels for each UAV
        task_feature_dim = 3             # Mean task size, cycles, and active task ratio
        user_pos_dim = 2                 # Mean user position (x,y)
        
        total_dim = user_pos_dim + uav_pos_dim + task_feature_dim + uav_energy_dim
        
        # If no active tasks, return a state with appropriate indicators
        if np.sum(self.active_tasks) == 0:
            state = np.zeros(total_dim)
            # Fill in UAV positions and energy levels (which are still valid)
            state[user_pos_dim:user_pos_dim+uav_pos_dim] = self.uav_positions.flatten()
            state[-uav_energy_dim:] = self.uav_energy / self.max_energy
            return state
        
        # Otherwise, compute the normal state
        active_user_positions = self.user_positions[self.active_tasks == 1]
        active_task_sizes = self.task_data_size[self.active_tasks == 1]
        active_task_cycles = self.task_cycles[self.active_tasks == 1]
        
        # Calculate statistics for active tasks
        mean_user_pos = np.mean(active_user_positions, axis=0)
        mean_task_size = np.mean(active_task_sizes)
        mean_task_cycles = np.mean(active_task_cycles)
        active_task_ratio = np.sum(self.active_tasks) / self.num_users
        
        # Assemble the state vector
        state = np.concatenate([
            mean_user_pos,                      # Average user position (2)
            self.uav_positions.flatten(),       # UAV positions (num_uavs*3)
            np.array([mean_task_size, mean_task_cycles, active_task_ratio]),  # Task features (3)
            self.uav_energy / self.max_energy,  # UAV energy ratios (num_uavs)
        ])
        
        return state
    
    def step(self, action):
        """
        执行动作并获取新状态、奖励和结束标志
        
        参数:
            action: 包含任务卸载决策和无人机移动的动作
                    前 num_users 个元素是任务卸载决策 (0: 本地, 1: UAV x, 2: 云)
                    后 num_uavs*3 个元素是无人机位置调整
        
        返回:
            state: 新的环境状态
            reward: 获得的奖励
            done: 环境是否结束
            info: 附加信息
        """
        # 解析动作
        task_offloading = action[:self.num_users]
        uav_movement = action[self.num_users:].reshape(self.num_uavs, 3)
        
        # 移动无人机
        self._move_uavs(uav_movement)
        
        # 执行任务卸载
        delay, energy, success_rate, offload_logs = self._offload_tasks(task_offloading)
        
        # 更新任务状态 (简化处理: 所有任务一次完成)
        self.active_tasks = np.zeros(self.num_users)
        
        # 获取新状态
        next_state = self._get_state()
        
        # 计算奖励: 平衡延迟、能耗和成功率
        reward = self._calculate_reward(delay, energy, success_rate)
        
        # 检查是否终止
        done = np.sum(self.active_tasks) == 0 or np.any(self.uav_energy <= 0)
        
        # 附加信息
        info = {
            "delay": delay,
            "energy": energy,
            "success_rate": success_rate,
            "uav_energy": self.uav_energy,
            "offload_logs": offload_logs
        }
        
        return next_state, reward, done, info
    
    def _move_uavs(self, movement):
        """移动无人机位置并消耗能量"""
        # 计算移动距离
        movement = np.clip(movement, -10, 10)  # 限制单步移动范围
        
        # 更新位置
        new_positions = self.uav_positions + movement
        
        # 确保无人机在区域范围内
        new_positions[:, 0] = np.clip(new_positions[:, 0], 0, self.area_size)
        new_positions[:, 1] = np.clip(new_positions[:, 1], 0, self.area_size)
        new_positions[:, 2] = np.clip(new_positions[:, 2], 50, 200)  # 高度限制
        
        # 计算移动距离
        distances = np.linalg.norm(new_positions - self.uav_positions, axis=1)
        
        # 计算能量消耗: 移动消耗 + 悬停消耗
        energy_consumption = distances * self.uav_flight_power + self.uav_hover_power
        
        # 更新无人机能量
        self.uav_energy -= energy_consumption
        self.uav_energy = np.maximum(self.uav_energy, 0)  # 防止能量变为负数
        
        # 更新无人机位置
        self.uav_positions = new_positions
    
    def _offload_tasks(self, offloading_decisions):
        """
        执行任务卸载并记录详细流程
        
        参数:
            offloading_decisions: 每个任务的卸载决策 (0: 本地, 1-num_uavs: UAV, num_uavs+1: 云)
        
        返回:
            total_delay: 总延迟
            total_energy: 总能耗
            success_rate: 任务成功率
            offload_logs: 详细任务卸载日志
        """
        total_delay = 0
        total_energy = 0
        successful_tasks = 0
        offload_logs = []
        
        # 对每个活跃任务进行卸载决策
        for i in range(self.num_users):
            if self.active_tasks[i] == 0:
                continue  # 跳过非活跃任务
            
            # 获取任务特性
            data_size = self.task_data_size[i]  # MB
            cycles = self.task_cycles[i] * 1e9  # 转换为cycles
            deadline = self.task_deadlines[i]  # 秒
            
            # 记录请求发送阶段
            log_entry = {
                "user_id": i,
                "task_size": data_size,
                "computation_cycles": cycles,
                "deadline": deadline,
                "decision": int(offloading_decisions[i]),
                "stages": []
            }
            
            # 执行卸载决策
            decision = int(offloading_decisions[i])
            
            # 计算时延和能耗
            if decision == 0:  # 本地执行
                # 计算时延
                computing_time = cycles / (self.local_computing_capacity[i] * 1e9)
                transmission_time = 0
                total_time = computing_time + transmission_time
                
                # 记录本地执行阶段
                log_entry["stages"].append({
                    "stage": "本地执行",
                    "start_time": 0,
                    "end_time": computing_time,
                    "duration": computing_time
                })
                
                # 计算能耗
                computing_energy = self.local_energy_coefficient * cycles
                transmission_energy = 0
                task_energy = computing_energy + transmission_energy
                
            elif 1 <= decision <= self.num_uavs:  # UAV执行
                uav_idx = decision - 1
                
                # 检查UAV能量
                if self.uav_energy[uav_idx] <= 0:
                    # UAV能量不足，视为执行失败
                    total_time = deadline * 2  # 超过截止时间
                    task_energy = 0
                    
                    # 记录失败阶段
                    log_entry["stages"].append({
                        "stage": "UAV能量不足",
                        "start_time": 0,
                        "end_time": 0,
                        "duration": 0,
                        "uav_id": uav_idx
                    })
                else:
                    # 计算传输速率
                    distance = np.linalg.norm(self.user_positions[i] - self.uav_positions[uav_idx, :2])
                    path_loss = 20 * np.log10(distance) + 20 * np.log10(2.4) + 32.44  # dB
                    channel_gain = 10 ** (-path_loss / 10)
                    snr = self.transmit_power_user * channel_gain / self.noise_power
                    data_rate = self.bandwidth * np.log2(1 + snr)  # Mbps
                    
                    # 计算时延
                    transmission_time = data_size / data_rate if data_rate > 0 else float('inf')
                    computing_time = cycles / (self.uav_computing_capacity[uav_idx] * 1e9)
                    total_time = transmission_time + computing_time
                    
                    # 记录传输阶段
                    log_entry["stages"].append({
                        "stage": "数据传输到UAV",
                        "start_time": 0,
                        "end_time": transmission_time,
                        "duration": transmission_time,
                        "uav_id": uav_idx,
                        "distance": distance,
                        "data_rate": data_rate
                    })
                    
                    # 记录UAV计算阶段
                    log_entry["stages"].append({
                        "stage": "UAV计算处理",
                        "start_time": transmission_time,
                        "end_time": transmission_time + computing_time,
                        "duration": computing_time,
                        "uav_id": uav_idx
                    })
                    
                    # 计算能耗
                    transmission_energy = self.transmit_power_user * transmission_time
                    # UAV的计算能耗
                    uav_computing_energy = 0.5 * computing_time
                    task_energy = transmission_energy
                    
                    # 更新UAV能量
                    self.uav_energy[uav_idx] -= uav_computing_energy
                    self.uav_energy[uav_idx] = max(0, self.uav_energy[uav_idx])
            
            else:  # 云执行
                # 假设通过UAV中继到云
                uav_idx = np.argmin(np.linalg.norm(self.uav_positions[:, :2] - self.user_positions[i], axis=1))
                
                # 计算用户到UAV的传输速率
                distance_to_uav = np.linalg.norm(self.user_positions[i] - self.uav_positions[uav_idx, :2])
                path_loss_1 = 20 * np.log10(distance_to_uav) + 20 * np.log10(2.4) + 32.44
                channel_gain_1 = 10 ** (-path_loss_1 / 10)
                snr_1 = self.transmit_power_user * channel_gain_1 / self.noise_power
                data_rate_1 = self.bandwidth * np.log2(1 + snr_1)  # Mbps
                
                # 假设UAV到云的传输速率固定为50Mbps
                data_rate_2 = 50  # Mbps
                
                # 计算时延
                transmission_time_1 = data_size / data_rate_1 if data_rate_1 > 0 else float('inf')
                transmission_time_2 = data_size / data_rate_2
                computing_time = cycles / (self.cloud_computing_capacity * 1e9)
                total_time = transmission_time_1 + transmission_time_2 + computing_time
                
                # 记录用户到UAV传输阶段
                log_entry["stages"].append({
                    "stage": "数据传输到UAV",
                    "start_time": 0,
                    "end_time": transmission_time_1,
                    "duration": transmission_time_1,
                    "uav_id": uav_idx,
                    "distance": distance_to_uav,
                    "data_rate": data_rate_1
                })
                
                # 记录UAV到云传输阶段
                log_entry["stages"].append({
                    "stage": "数据从UAV传输到云",
                    "start_time": transmission_time_1,
                    "end_time": transmission_time_1 + transmission_time_2,
                    "duration": transmission_time_2,
                    "uav_id": uav_idx,
                    "data_rate": data_rate_2
                })
                
                # 记录云计算阶段
                log_entry["stages"].append({
                    "stage": "云端计算处理",
                    "start_time": transmission_time_1 + transmission_time_2,
                    "end_time": total_time,
                    "duration": computing_time
                })
                
                # 计算能耗
                transmission_energy = self.transmit_power_user * transmission_time_1
                uav_relay_energy = self.transmit_power_uav * transmission_time_2
                task_energy = transmission_energy  # 只考虑移动设备的能耗
                
                # 更新UAV能量
                self.uav_energy[uav_idx] -= uav_relay_energy
                self.uav_energy[uav_idx] = max(0, self.uav_energy[uav_idx])
            
            # 记录总体执行情况
            log_entry["total_time"] = total_time
            log_entry["energy_consumption"] = task_energy
            log_entry["deadline_met"] = total_time <= deadline
            
            offload_logs.append(log_entry)
            
            # 检查任务是否在截止时间内完成
            if total_time <= deadline:
                successful_tasks += 1
            
            # 累加总延迟和能耗
            total_delay += total_time
            total_energy += task_energy
        
        # 计算成功率
        active_task_count = np.sum(self.active_tasks)
        success_rate = successful_tasks / active_task_count if active_task_count > 0 else 1.0
        
        return total_delay, total_energy, success_rate, offload_logs
    
    def _calculate_reward(self, delay, energy, success_rate):
        """
        计算奖励函数，平衡延迟、能耗和成功率
        
        参数:
            delay: 总延迟
            energy: 总能耗
            success_rate: 任务成功率
        
        返回:
            reward: 计算得到的奖励值
        """
        # 归一化延迟和能耗 (假设最大延迟为5s，最大能耗为10J)
        normalized_delay = min(delay / 5.0, 1.0)
        normalized_energy = min(energy / 10.0, 1.0)
        
        # 权重因子
        w_delay = 0.35
        w_energy = 0.3
        w_success = 0.35
        
        # 计算奖励: 高成功率奖励，低延迟和低能耗奖励
        reward = w_success * success_rate - w_delay * normalized_delay - w_energy * normalized_energy
        
        return reward
    