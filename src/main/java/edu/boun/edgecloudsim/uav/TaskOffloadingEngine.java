package edu.boun.edgecloudsim.uav;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

import org.cloudbus.cloudsim.core.CloudSim;

import edu.boun.edgecloudsim.utils.ArrayUtils;

import edu.boun.edgecloudsim.core.ExecutionTarget;
import edu.boun.edgecloudsim.core.SimManager;
import edu.boun.edgecloudsim.core.SimSettings;
import edu.boun.edgecloudsim.utils.SimLogger;
import edu.boun.edgecloudsim.utils.TaskProperty;

/**
 * 任务卸载引擎，负责使用RL进行任务卸载决策
 */
public class TaskOffloadingEngine {
    private SimManager simManager;
    private EnhancedPythonInterface pythonInterface;
    private StateActionManager stateActionManager;
    private RewardCalculator rewardCalculator;
    private Map<String, UAV.Task> activeTasks;
    private AtomicInteger totalTaskCount;
    private AtomicInteger completedTasks;
    private AtomicInteger rejectedTaskCount;
    private double totalLatency;
    private boolean useDQN; // 是否使用DQN算法
    
    private static final int MAX_TASK_QUEUE_LENGTH = 50; // 最大队列长度
    private static final double DEFAULT_TASK_LENGTH = 1000.0; // 默认任务长度(MI)
    
    /**
     * 构造函数
     * @param simManager 模拟管理器
     */
    public TaskOffloadingEngine(SimManager simManager) {
        this.simManager = simManager;
        // 初始化集合和计数器
        this.activeTasks = new HashMap<>();
        this.totalTaskCount = new AtomicInteger(0);
        this.completedTasks = new AtomicInteger(0);
        this.rejectedTaskCount = new AtomicInteger(0);
        this.totalLatency = 0.0;
        this.useDQN = false;
        
        // 初始化状态动作管理器和奖励计算器
        this.stateActionManager = new StateActionManager(simManager);
        this.rewardCalculator = new RewardCalculator(simManager);
        
        // 连接到Python RL服务器
        try {
            this.pythonInterface = new EnhancedPythonInterface("localhost", 12345);
            if (pythonInterface.connect()) {
                SimLogger.printLine("成功连接到RL服务器");
            } else {
                SimLogger.printLine("无法连接到RL服务器，将使用默认策略");
                pythonInterface.shutdown();
                this.pythonInterface = null;
            }
        } catch (Exception e) {
            SimLogger.printLine("初始化RL接口时出错: " + e.getMessage());
            SimLogger.printLine("将使用默认策略");
            this.pythonInterface = null;
        }
    }
    
    /**
     * 初始化引擎
     */
    public void initialize() {
        // 已在构造函数中完成初始化
    }
    
    /**
     * 获取卸载决策
     * @param task 任务
     * @return typed execution target
     */
    public ExecutionTarget getOffloadingTarget(UAV.Task task) {
        // 确保状态动作管理器已初始化
        if (stateActionManager == null) {
            SimLogger.printLine("警告: StateActionManager未初始化，使用默认决策");
            return getDefaultTarget();
        }
        
        // 生成当前状态
        double[] state = stateActionManager.generateState();
        
        if (pythonInterface != null && pythonInterface.isConnected()) {
            // 使用RL模型获取动作
            try {
                final ExecutionTarget[] decision = {null};
                final boolean[] completed = {false};
                
                pythonInterface.getActionAsync(state, new EnhancedPythonInterface.Callback<double[]>() {
                    @Override
                    public void onSuccess(double[] action) {
                        // 解析动作为卸载决策
                        List<Map<String, Object>> actions = stateActionManager.parseOffloadingDecision(action);
                        
                        // 在这个简化的实现中，我们只取第一个动作作为卸载目标
                        if (actions.size() > 0) {
                            decision[0] = ExecutionTarget.uav((int) actions.get(0).get("uavId"));
                        }
                        
                        completed[0] = true;
                    }
                    
                    @Override
                    public void onFailure(Exception e) {
                        SimLogger.printLine("获取RL动作失败: " + e.getMessage());
                        decision[0] = getDefaultTarget();
                        completed[0] = true;
                    }
                });
                
                // 等待回调完成（在实际系统中可能需要异步处理）
                int waitCount = 0;
                while (!completed[0] && waitCount < 100) {
                    try {
                        Thread.sleep(10);
                    } catch (InterruptedException e) {
                        SimLogger.printLine("等待RL决策时被中断");
                        break;
                    }
                    waitCount++;
                }
                
                if (!completed[0]) {
                    SimLogger.printLine("获取RL动作超时，使用默认决策");
                    return getDefaultTarget();
                }

                return decision[0] != null ? decision[0] : getDefaultTarget();
            } catch (Exception e) {
                SimLogger.printLine("使用RL进行卸载决策时出错: " + e.getMessage());
                return getDefaultTarget();
            }
        } else {
            // 使用默认策略
            return getDefaultTarget();
        }
    }

    /**
     * Compatibility adapter for callers that still consume EdgeCloudSim's
     * legacy integer destination IDs.
     */
    @Deprecated
    public int getOffloadingDecision(UAV.Task task) {
        return getOffloadingTarget(task).toLegacyDeviceId();
    }
    
    /**
     * 获取默认卸载决策（最简单的负载均衡）
     * @return typed target; local execution is explicit when no UAV is usable
     */
    private ExecutionTarget getDefaultTarget() {
        UAVManager uavManager = simManager.getUAVManager();
        if (uavManager == null) {
            SimLogger.printLine("警告: UAVManager为空，返回本地执行");
            return ExecutionTarget.local();
        }
        
        List<UAV> uavList = uavManager.getUAVs();
        if (uavList == null || uavList.isEmpty()) {
            SimLogger.printLine("警告: UAV列表为空，返回本地执行");
            return ExecutionTarget.local();
        }
        
        // 修改选择策略：考虑处理能力、任务队列长度和能量
        int bestUavId = -1;
        double bestScore = -1;
        
        for (UAV uav : uavList) {
            if (uav.getEnergy() <= 0) continue;
            if (uav.getTaskQueueLength() >= MAX_TASK_QUEUE_LENGTH) continue;
            
            // 计算综合分数: 处理能力/(队列长度+1) * 能量比例
            double energyRatio = uav.getEnergy() / simManager.getSimulationSettings().getUAVMaxEnergy();
            double processingCapacity = uav.getProcessingCapacity();
            int queueLength = uav.getTaskQueueLength();
            
            double score = (processingCapacity / (queueLength + 1)) * energyRatio;
            
            if (score > bestScore) {
                bestScore = score;
                bestUavId = uav.getId();
            }
        }
        
        if (bestUavId >= 0) {
            return ExecutionTarget.uav(bestUavId);
        }
        
        // 如果没有合适的UAV，返回本地执行
        return ExecutionTarget.local();
    }
    
    /**
     * 提交任务处理
     * @param taskProperty 任务属性
     */
    public void submitTask(TaskProperty taskProperty) {
        // 确保计数器已初始化
        if (totalTaskCount == null) {
            totalTaskCount = new AtomicInteger(0);
        }
        
        int taskId = totalTaskCount.incrementAndGet();
        SimLogger.printLine("处理任务提交: #" + taskId);
        
        // 创建UAV任务 - 从TaskProperty获取长度
        double taskLength = (taskProperty != null && taskProperty.getLength() > 0) ? 
                          taskProperty.getLength() : DEFAULT_TASK_LENGTH;
        
        UAV.Task task = new UAV.Task(String.valueOf(taskId), taskLength);
        
        // 获取卸载决策
        ExecutionTarget target = getOffloadingTarget(task);

        if (target.getType() != ExecutionTarget.Type.UAV) {
            SimLogger.printLine("遗留UAV引擎无法执行目标 " + target + "，任务被拒绝");
            rejectedTaskCount.incrementAndGet();
            return;
        }
        int targetUavId = target.getResourceId();
        
        // 提交任务到UAV
        UAVManager uavManager = simManager.getUAVManager();
        if (uavManager == null) {
            SimLogger.printLine("错误: UAV管理器为空");
            if (rejectedTaskCount == null) {
                rejectedTaskCount = new AtomicInteger(0);
            }
            rejectedTaskCount.incrementAndGet();
            return;
        }
        
        boolean assigned = uavManager.assignTask(targetUavId, task);
        
        if (assigned) {
            SimLogger.printLine("任务 " + taskId + " 已分配给UAV " + targetUavId);
            
            // 记录到活动任务
            if (activeTasks == null) {
                activeTasks = new HashMap<>();
            }
            activeTasks.put(task.getId(), task);
            
        } else {
            SimLogger.printLine("任务 " + taskId + " 分配失败");
            if (rejectedTaskCount == null) {
                rejectedTaskCount = new AtomicInteger(0);
            }
            rejectedTaskCount.incrementAndGet();
        }
    }

    /**
     * 处理任务完成
     * @param task 完成的任务
     */
    public void taskCompleted(UAV.Task task) {
        // 确保已初始化
        if (activeTasks == null) {
            activeTasks = new HashMap<>();
        }
        if (completedTasks == null) {
            completedTasks = new AtomicInteger(0);
        }
        
        String taskId = task.getId();
        if (!activeTasks.containsKey(taskId)) {
            SimLogger.printLine("[INFO] 完成未由卸载引擎提交的任务 #" + taskId);
        }
        
        // 从活动任务中移除
        activeTasks.remove(taskId);
        completedTasks.incrementAndGet();
        
        // 修复延迟计算 - 确保单位一致性
        long arrivalTimeMs = task.getArrivalTime();
        long completionTimeMs = task.getCompletionTime();
        
        if (completionTimeMs < arrivalTimeMs) {
            completionTimeMs = (long)(SimManager.getInstance().getSimulationTime() * 1000);
            task.setCompletionTime(completionTimeMs);
        }
        
        // 计算正确的延迟
        long latency = completionTimeMs - arrivalTimeMs;
        
        // 更新总延迟
        totalLatency += latency;
        
        double arrivalTimeSeconds = arrivalTimeMs / 1000.0;
        double completionTimeSeconds = completionTimeMs / 1000.0;
        SimLogger.getInstance().taskCompleted(
            taskId, arrivalTimeSeconds, completionTimeSeconds, task.getTotalMI());
        
        // RL模型训练部分添加错误处理
        if (pythonInterface != null && pythonInterface.isConnected()) {
            try {
                trainRLModel(task, latency);
            } catch (Exception e) {
                SimLogger.printLine("[ERROR] RL模型训练失败: " + e.getMessage());
            }
        }
    }
    
    /**
     * 训练RL模型
     * @param task 完成的任务
     * @param latency 任务延迟
     */
    private void trainRLModel(UAV.Task task, long latency) {
        // 获取状态、动作和奖励
        double[] state = stateActionManager.getLastState();
        double[] action = stateActionManager.getLastAction();
        double[] nextState = stateActionManager.generateState();
        double reward = rewardCalculator.calculateReward(latency, (long) task.getTotalMI());

        if (state == null || action == null) {
            SimLogger.printLine("跳过训练：当前任务没有完整的状态/动作轨迹");
            return;
        }
        
        // 异步训练RL模型
        pythonInterface.trainAsync(state, action, reward, nextState, false, 
            new EnhancedPythonInterface.Callback<Boolean>() {
                @Override
                public void onSuccess(Boolean result) {
                    if (result) {
                        SimLogger.printLine("RL模型训练成功");
                    } else {
                        SimLogger.printLine("RL模型训练失败");
                    }
                }
                
                @Override
                public void onFailure(Exception e) {
                    SimLogger.printLine("RL模型训练出错: " + e.getMessage());
                }
            });
    }
    
    /**
     * 获取Python接口
     * @return Python接口
     */
    public EnhancedPythonInterface getPythonInterface() {
        return pythonInterface;
    }
    
    /**
     * 获取状态动作管理器
     * @return 状态动作管理器
     */
    public StateActionManager getStateActionManager() {
        return stateActionManager;
    }
    
    /**
     * 获取总任务数
     * @return 总任务数
     */
    public int getTotalTaskCount() {
        return totalTaskCount != null ? totalTaskCount.get() : 0;
    }
    
    /**
     * 获取完成的任务数
     * @return 完成的任务数
     */
    public int getCompletedTaskCount() {
        return completedTasks != null ? completedTasks.get() : 0;
    }
    
    /**
     * 获取拒绝的任务数
     * @return 拒绝的任务数
     */
    public int getRejectedTaskCount() {
        return rejectedTaskCount != null ? rejectedTaskCount.get() : 0;
    }
    
    /**
     * 获取总延迟
     * @return 总延迟
     */
    public double getTotalLatency() {
        return totalLatency;
    }
    
    /**
     * 关闭引擎
     */
    public void close() {
        // 关闭Python接口
        if (pythonInterface != null) {
            try {
                pythonInterface.close();
            } catch (Exception e) {
                SimLogger.printLine("关闭Python接口时出错: " + e.getMessage());
            }
        }
    }
    
    /**
     * 检查卡死状态
     * @param currentTime 当前时间
     * @param lastTaskCompletionTime 上次任务完成时间
     * @return 是否卡死
     */
    public boolean isStalled(double currentTime, double lastTaskCompletionTime) {
        // 如果超过一定时间没有完成任务，认为系统卡死
        return (currentTime - lastTaskCompletionTime) > 1000.0;
    }
    
    /**
     * 紧急恢复处理
     */
    public void emergencyRecovery() {
        UAVManager uavManager = simManager.getUAVManager();
        if (uavManager != null) {
            uavManager.resetUAVs();
        }
        
        // 清理活动任务
        int activeTasks = this.activeTasks.size();
        this.activeTasks.clear();
        
        SimLogger.printLine("紧急恢复: 清除了 " + activeTasks + " 个活动任务");
    }
    
}
