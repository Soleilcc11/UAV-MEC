package edu.boun.edgecloudsim.uav;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

import org.cloudbus.cloudsim.core.CloudSim;

import edu.boun.edgecloudsim.utils.ArrayUtils;

import edu.boun.edgecloudsim.core.SimManager;
import edu.boun.edgecloudsim.core.SimSettings;
import edu.boun.edgecloudsim.utils.SimLogger;
import edu.boun.edgecloudsim.utils.TaskProperty;

/**
 * 任务卸载引擎，负责使用RL进行任务卸载决策
 */
public class TaskOffloadingEngine {
    public static final int LOCAL_EXECUTION = 0;
    public static final int CLOUD_EXECUTION = 1;
    public static final int UAV_EXECUTION = 2;

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
        this.rewardCalculator = new RewardCalculator();
        
        // 连接到Python RL服务器
        try {
            this.pythonInterface = new EnhancedPythonInterface("localhost", 12345);
            if (pythonInterface.connect()) {
                SimLogger.printLine("成功连接到RL服务器");
            } else {
                SimLogger.printLine("无法连接到RL服务器，将使用默认策略");
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
     * @return 卸载决策（目标UAV的ID）
     */
    public int getOffloadingDecision(UAV.Task task) {
        // 确保已初始化
        if (totalTaskCount == null) {
            totalTaskCount = new AtomicInteger(0);
        }
        totalTaskCount.incrementAndGet();
        
        // 确保状态动作管理器已初始化
        if (stateActionManager == null) {
            SimLogger.printLine("警告: StateActionManager未初始化，使用默认决策");
            return getDefaultDecision();
        }
        
        // 生成当前状态
        double[] state = stateActionManager.generateState();
        
        if (pythonInterface != null && pythonInterface.isConnected()) {
            // 使用RL模型获取动作
            try {
                final int[] decision = {-1};
                final boolean[] completed = {false};
                
                pythonInterface.getActionAsync(state, new EnhancedPythonInterface.Callback<double[]>() {
                    @Override
                    public void onSuccess(double[] action) {
                        // 解析动作为卸载决策
                        List<Map<String, Object>> actions = stateActionManager.parseOffloadingDecision(action);
                        
                        // 在这个简化的实现中，我们只取第一个动作作为卸载目标
                        if (actions.size() > 0) {
                            decision[0] = (int) actions.get(0).get("uavId");
                        }
                        
                        completed[0] = true;
                    }
                    
                    @Override
                    public void onFailure(Exception e) {
                        SimLogger.printLine("获取RL动作失败: " + e.getMessage());
                        decision[0] = getDefaultDecision();
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
                    return getDefaultDecision();
                }
                
                return decision[0];
            } catch (Exception e) {
                SimLogger.printLine("使用RL进行卸载决策时出错: " + e.getMessage());
                return getDefaultDecision();
            }
        } else {
            // 使用默认策略
            return getDefaultDecision();
        }
    }
    
    /**
     * 获取默认卸载决策（最简单的负载均衡）
     * @return 目标UAV的ID
     */
    private int getDefaultDecision() {
        UAVManager uavManager = simManager.getUAVManager();
        if (uavManager == null) {
            SimLogger.printLine("警告: UAVManager为空，返回本地执行");
            return LOCAL_EXECUTION;
        }
        
        List<UAV> uavList = uavManager.getUAVs();
        if (uavList == null || uavList.isEmpty()) {
            SimLogger.printLine("警告: UAV列表为空，返回本地执行");
            return LOCAL_EXECUTION;
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
            return bestUavId;
        }
        
        // 如果没有合适的UAV，返回本地执行
        return LOCAL_EXECUTION;
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
        int targetUavId = getOffloadingDecision(task);
        
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
            
            // 记录任务提交信息到日志
            try {
                SimLogger.getInstance().taskStarted(taskId, CloudSim.clock());
                // 设置其他任务属性
            } catch (Exception e) {
                SimLogger.printLine("记录任务信息时出错: " + e.getMessage());
            }
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
        
        // 确保任务被跟踪
        String taskId = task.getId();
        if (!activeTasks.containsKey(taskId)) {
            SimLogger.printLine("[WARNING] 任务 #" + taskId + " 不在活动任务列表中，添加它");
            activeTasks.put(taskId, task);
        }
        
        // 从活动任务中移除
        activeTasks.remove(taskId);
        completedTasks.incrementAndGet();
        
        // 修复延迟计算 - 确保单位一致性
        long arrivalTimeMs = task.getArrivalTime();
        long completionTimeMs = task.getCompletionTime();
        
        // 防止时间戳错误
        if (completionTimeMs <= 0) {
            // 使用当前仿真时间作为完成时间
            completionTimeMs = (long)(SimManager.getInstance().getSimulationTime() * 1000);
            task.setCompletionTime(completionTimeMs); 
        }
        
        // 计算正确的延迟
        long latency = (completionTimeMs > arrivalTimeMs) ? 
                    (completionTimeMs - arrivalTimeMs) : 0;
        
        // 更新总延迟
        totalLatency += latency;
        
        // 解析任务信息
        String taskIdStr = "SimTask_" + taskId;
        double startTime = SimManager.getInstance().getSimulationTime();
        double completionTime = startTime + 1.0; // 假设任务完成时间为当前时间+1秒
        double taskLength = 1000.0; // 假设默认任务长度为1000MI
        
        // 记录调试信息
        SimLogger.printLine("[完成] 任务 " + taskIdStr + " 完成，到达时间: " + startTime + "ms, 完成时间: " + completionTime + "ms, 延迟: " + 0 + "ms");
        
        // 记录任务完成
        SimLogger.getInstance().taskCompleted(taskIdStr, startTime, completionTime, taskLength);
        
        // 更新统计信息
        completedTasks.incrementAndGet();
        
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
    
    /**
     * 状态动作管理器内部类
     */
    public class StateActionManager {
        private SimManager simManager;
        private double[] lastState;
        private double[] lastAction;
        
        public StateActionManager(SimManager simManager) {
            this.simManager = simManager;
            this.lastState = null;
            this.lastAction = null;
        }
        
        public double[] generateState() {
            // 简化版：生成一个基本状态向量
            double[] state = new double[]{
                simManager.getSimulationTime(),
                getCompletedTaskCount(),
                getTotalTaskCount(),
                getRejectedTaskCount()
            };
            
            // 保存最后生成的状态
            this.lastState = state.clone();
            
            return state;
        }
        
        public List<Map<String, Object>> parseOffloadingDecision(double[] action) {
            // 简化版：将动作向量解析为决策列表
            List<Map<String, Object>> decisions = new ArrayList<>();
            Map<String, Object> decision = new HashMap<>();
            
            // 简单地将第一个值作为UAV ID
            int uavId = action.length > 0 ? (int)Math.abs(action[0] % simManager.getUAVManager().getUAVs().size()) : 0;
            
            decision.put("uavId", uavId);
            decisions.add(decision);
            
            // 保存最后的动作
            this.lastAction = action.clone();
            
            return decisions;
        }
        
        public double[] getLastState() {
            return lastState;
        }
        
        public double[] getLastAction() {
            return lastAction;
        }
    }
    
    /**
     * 奖励计算器内部类
     */
    public class RewardCalculator {
        public double calculateReward(long latency, long totalMI) {
            // 简化版：基于延迟和计算量的反比例奖励
            // 延迟越低，奖励越高
            return 1000.0 / (1.0 + latency);
        }
    }
}