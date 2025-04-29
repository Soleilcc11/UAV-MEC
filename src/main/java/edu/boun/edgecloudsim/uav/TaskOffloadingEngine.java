package edu.boun.edgecloudsim.uav;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

import edu.boun.edgecloudsim.core.SimManager;
import edu.boun.edgecloudsim.core.SimSettings;
import edu.boun.edgecloudsim.utils.SimLogger;

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
    private AtomicInteger completedTaskCount;
    private AtomicInteger rejectedTaskCount;
    private double totalLatency;
    private boolean useDQN; // 是否使用DQN算法
    
    /**
     * 构造函数
     * @param simManager 模拟管理器
     */
    public TaskOffloadingEngine(SimManager simManager) {
        this.simManager = simManager;
        // 初始化集合和计数器
        this.activeTasks = new HashMap<>();
        this.totalTaskCount = new AtomicInteger(0);
        this.completedTaskCount = new AtomicInteger(0);
        this.rejectedTaskCount = new AtomicInteger(0);
        this.totalLatency = 0.0;
        this.useDQN = false;
    }
    
    /**
     * 初始化引擎
     */
    public void initialize() {
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
        
        // 简单的轮询策略
        int minQueueLength = Integer.MAX_VALUE;
        int targetUavId = 0;
        
        for (UAV uav : uavList) {
            if (uav.getEnergy() > 0 && uav.getTaskQueueLength() < minQueueLength) {
                minQueueLength = uav.getTaskQueueLength();
                targetUavId = uav.getId();
            }
        }
        
        return targetUavId;
    }
    
    /**
     * 提交任务到指定UAV
     * @param task 任务
     * @param uavId 目标UAV的ID
     * @return 是否成功提交
     */
    public boolean submitTask(UAV.Task task, int uavId) {
        // 确保已初始化
        if (activeTasks == null) {
            activeTasks = new HashMap<>();
        }
        if (rejectedTaskCount == null) {
            rejectedTaskCount = new AtomicInteger(0);
        }
        
        UAVManager uavManager = simManager.getUAVManager();
        if (uavManager == null) {
            SimLogger.printLine("警告: UAVManager为空，任务提交失败");
            rejectedTaskCount.incrementAndGet();
            return false;
        }
        
        boolean success = uavManager.assignTask(uavId, task);
        
        if (success) {
            activeTasks.put(task.getId(), task);
        } else {
            rejectedTaskCount.incrementAndGet();
        }
        
        return success;
    }
    
    /**
     * 任务完成回调
     * @param task 完成的任务
     */
    public void taskCompleted(UAV.Task task) {
        // 确保已初始化
        if (activeTasks == null) {
            activeTasks = new HashMap<>();
        }
        if (completedTaskCount == null) {
            completedTaskCount = new AtomicInteger(0);
        }
        
        // 从活动任务中移除
        activeTasks.remove(task.getId());
        completedTaskCount.incrementAndGet();
        
        // 计算延迟
        long latency = task.getCompletionTime() - task.getArrivalTime();
        totalLatency += latency;
        
        // 记录完成情况
        SimLogger.printLine("任务 " + task.getId() + " 完成，延迟: " + latency + "ms");
        
        // 如果有RL模型，进行训练
        if (pythonInterface != null && pythonInterface.isConnected() && 
            stateActionManager != null && rewardCalculator != null) {
            trainRLModel(task, latency);
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
        double reward = rewardCalculator.calculateReward(latency, task.getTotalMI());
        
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
        return completedTaskCount != null ? completedTaskCount.get() : 0;
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
            // 实际实现中应该包含更多系统状态信息
            return new double[]{simManager.getSimulationTime(), getCompletedTaskCount()};
        }
        
        public List<Map<String, Object>> parseOffloadingDecision(double[] action) {
            // 简化版：将动作向量解析为决策列表
            List<Map<String, Object>> decisions = new ArrayList<>();
            Map<String, Object> decision = new HashMap<>();
            
            // 简单地将第一个值作为UAV ID
            decision.put("uavId", (int)action[0]);
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