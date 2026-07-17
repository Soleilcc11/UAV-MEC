package edu.boun.edgecloudsim.uav;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

import edu.boun.edgecloudsim.core.ExecutionTarget;
import edu.boun.edgecloudsim.core.SimManager;
import edu.boun.edgecloudsim.utils.SimLogger;
import edu.boun.edgecloudsim.utils.TaskProperty;

/**
 * Legacy UAV task adapter.
 *
 * <p>Gymnasium owns all learned decisions through {@link GymBridgeServer}.
 * This adapter intentionally contains no Python client or training path and
 * only provides a deterministic heuristic for non-Gym legacy callers.</p>
 */
public class TaskOffloadingEngine {
    private SimManager simManager;
    private Map<String, UAV.Task> activeTasks;
    private AtomicInteger totalTaskCount;
    private AtomicInteger completedTasks;
    private AtomicInteger rejectedTaskCount;
    private double totalLatency;
    
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
        return getDefaultTarget();
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
        // No external resources: GymBridge owns the Python connection.
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
