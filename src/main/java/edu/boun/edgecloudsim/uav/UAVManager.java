package edu.boun.edgecloudsim.uav;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Map;
import java.util.HashMap;
import java.util.concurrent.atomic.AtomicInteger;
import org.cloudbus.cloudsim.core.SimEvent;

import edu.boun.edgecloudsim.core.SimManager;
import edu.boun.edgecloudsim.core.SimSettings;
import edu.boun.edgecloudsim.utils.SimLogger;
import edu.boun.edgecloudsim.edge_orchestrator.EdgeOrchestrator;
import edu.boun.edgecloudsim.edge_client.DefaultMobileDeviceManager;
import edu.boun.edgecloudsim.edge_client.Task;

/**
 * UAV管理器类，负责管理无人机
 */
public class UAVManager {
    private SimManager simManager;
    private List<UAV> uavList;
    private Random random;
    private AtomicInteger completedTasks;
    private Map<String, Task> activeEdgeTasks;
    private int movementCommandCount;
    private int boundaryConstraintCount;
    
    // 定义任务完成事件类型
    public static final int UAV_TASK_COMPLETED = 9999;
    
    /**
     * 构造函数
     * @param simManager 模拟管理器
     */
    public UAVManager(SimManager simManager) {
        this.simManager = simManager;
        this.uavList = new ArrayList<>();
        this.random = new Random(42); // 固定随机种子以实现可重复的模拟
        this.completedTasks = new AtomicInteger(0);
        this.activeEdgeTasks = new HashMap<>();
        
        // 在构造函数中初始化，因为simManager还没有完全设置好
    }
    
    /**
     * 初始化UAV管理器
     */
    public void initialize() {
        if (!uavList.isEmpty()) {
            return;
        }
        SimSettings simSettings = simManager.getSimulationSettings();
        random.setSeed(simSettings.getSimulationSeed());
        int numUavs = simSettings.getNumOfUAVs();
        
        SimLogger.printLine("初始化 " + numUavs + " 个UAV...");
        
        // 创建UAVs
        for (int i = 0; i < numUavs; i++) {
            // 随机位置
            double[] position = generateRandomPosition();
            
            // 初始能量和处理能力
            double initialEnergy = simSettings.getUAVMaxEnergy();
            double processingCapacity = 1500 + random.nextDouble() * 1000; // 提高至1500-2500 MIPS

            UAV uav = new UAV(i, position, initialEnergy, processingCapacity);
            uav.configureEnergyModel(
                    simSettings.getUAVFlightPower(),
                    simSettings.getUAVHoverPower(),
                    simSettings.getUAVComputeEnergyPerMi());
            uavList.add(uav);
            
            SimLogger.printLine("UAV " + i + " 创建在位置 (" + 
                position[0] + ", " + position[1] + ", " + position[2] + 
                ") 能量为 " + initialEnergy +
                " 处理能力为 " + processingCapacity + " MIPS");
        }
    }
    
    /**
     * 更新所有UAV的位置和状态
     * @param actions UAV动作列表
     */
    public void updateUAVs(List<Map<String, Object>> actions) {
        for (Map<String, Object> action : actions) {
            int uavId = (int) action.get("uavId");
            double[] moveDirection = (double[]) action.get("moveDirection");
            
            if (uavId >= 0 && uavId < uavList.size()) {
                moveUav(uavId, moveDirection);
            }
        }
    }

    public boolean moveUav(int uavId, double[] displacement) {
        UAV uav = getUAV(uavId);
        if (uav == null || displacement == null || displacement.length != 3) {
            return false;
        }
        double[] newPosition = uav.updatePosition(displacement.clone());
        movementCommandCount++;
        if (enforceBoundaryConstraints(uav, newPosition)) {
            boundaryConstraintCount++;
        }
        return true;
    }
    
    /**
     * 查找距离给定位置最近的UAV
     * @param x X坐标
     * @param y Y坐标
     * @return 最近UAV的ID
     */
    public int findNearestUAV(double x, double y) {
        if (uavList.isEmpty()) {
            return -1;
        }
        
        int nearestUavId = -1;
        double minDistance = Double.MAX_VALUE;
        
        for (UAV uav : uavList) {
            double[] position = uav.getPosition();
            double distance = Math.sqrt(
                Math.pow(position[0] - x, 2) +
                Math.pow(position[1] - y, 2)
            );
            
            if (distance < minDistance) {
                minDistance = distance;
                nearestUavId = uav.getId();
            }
        }
        
        return nearestUavId;
    }
        
    /**
     * 处理所有UAV的任务
     * @param timeSlot 时间片
     * @return 处理完成的任务总数
     */
    public int processTasks(double timeSlot) {
        int newCompletedTaskCount = 0;
        
        for (UAV uav : uavList) {
            List<UAV.Task> newCompletedTasks = uav.processTasks(timeSlot);
            newCompletedTaskCount += newCompletedTasks.size();
            
            // 处理完成的任务
            for (UAV.Task task : newCompletedTasks) {
                handleUAVTaskCompletedEvent(task);
            }
        }
        
        // 记录并返回新完成的任务数量
        if (newCompletedTaskCount > 0) {
            SimLogger.printLine("处理UAV任务，完成数量: " + getCompletedTaskCount() + 
                               ", 时间: " + simManager.getSimulationTime());
        }
        
        return newCompletedTaskCount;
    }
    
    /**
     * 处理UAV任务完成事件
     */
    public void handleUAVTaskCompletedEvent(UAV.Task task) {
        if (task == null) {
            SimLogger.printLine("[WARNING] 收到空任务完成事件");
            return;
        }
        
        // 确保计数器初始化
        if (completedTasks == null) {
            completedTasks = new AtomicInteger(0);
            SimLogger.printLine("[INFO] 初始化completedTasks计数器");
        }
        
        // 更新计数
        int newCount = completedTasks.incrementAndGet();
        
        // 记录详细日志
        SimLogger.printLine("[TASK_COMPLETED] 任务 #" + task.getId() + 
                        " 完成时间: " + task.getCompletionTime() + 
                        ", 当前仿真时间: " + (long)(simManager.getSimulationTime() * 1000) + 
                        ", 总完成数: " + newCount);
        
        Task edgeTask = activeEdgeTasks.remove(task.getId());
        if (edgeTask != null) {
            if (simManager.getMobileDeviceManager() instanceof DefaultMobileDeviceManager) {
                ((DefaultMobileDeviceManager) simManager.getMobileDeviceManager())
                        .uavTaskCompleted(edgeTask);
            }
            return;
        }

        // 通知相关组件
        try {
            TaskOffloadingEngine engine = (TaskOffloadingEngine)simManager.getTaskOffloadingEngine();
            if (engine != null) {
                engine.taskCompleted(task);
            } else {
                SimLogger.printLine("[WARNING] TaskOffloadingEngine为空，无法通知任务完成");
            }
        } catch (Exception e) {
            SimLogger.printLine("[ERROR] 处理任务完成事件时出错: " + e.getMessage());
            e.printStackTrace();
        }
        
        // 定期记录 - 防止过多日志
        if (newCount % 10 == 0 || newCount < 20) {
            SimLogger.printLine("处理UAV任务，完成数量: " + newCount + 
                            ", 时间: " + simManager.getSimulationTime());
        }
    }
    
    /**
     * 生成随机位置
     * @return 随机位置坐标 [x, y, z]
     */
    private double[] generateRandomPosition() {
        SimSettings simSettings = simManager.getSimulationSettings();
        double[] position = new double[3];
        
        // X坐标
        position[0] = random.nextDouble() * simSettings.getSimulationSpace()[0];
        
        // Y坐标
        position[1] = random.nextDouble() * simSettings.getSimulationSpace()[1];
        
        // Z坐标（高度）
        position[2] = simSettings.getUAVInitialHeight();
        
        return position;
    }
    
    /**
     * 确保UAV位置在模拟空间边界内
     * @param uav UAV对象
     * @param position 位置坐标
     */
    private boolean enforceBoundaryConstraints(UAV uav, double[] position) {
        SimSettings simSettings = simManager.getSimulationSettings();
        double[] simSpace = simSettings.getSimulationSpace();
        double minHeight = simSettings.getUAVMinHeight();
        double maxHeight = simSettings.getUAVMaxHeight();
        
        double originalX = position[0];
        double originalY = position[1];
        double originalZ = position[2];

        // X边界
        if (position[0] < 0) position[0] = 0;
        if (position[0] > simSpace[0]) position[0] = simSpace[0];
        
        // Y边界
        if (position[1] < 0) position[1] = 0;
        if (position[1] > simSpace[1]) position[1] = simSpace[1];
        
        // Z边界（高度）
        if (position[2] < minHeight) position[2] = minHeight;
        if (position[2] > maxHeight) position[2] = maxHeight;
        return originalX != position[0] || originalY != position[1] || originalZ != position[2];
    }
    
    /**
     * 计算两个UAV之间的通信质量
     * @param uav1 第一个UAV
     * @param uav2 第二个UAV
     * @return 通信质量（0-1之间，1表示最佳）
     */
    public double calculateCommunicationQuality(UAV uav1, UAV uav2) {
        double[] pos1 = uav1.getPosition();
        double[] pos2 = uav2.getPosition();
        
        // 计算欧氏距离
        double distance = Math.sqrt(
            Math.pow(pos1[0] - pos2[0], 2) +
            Math.pow(pos1[1] - pos2[1], 2) +
            Math.pow(pos1[2] - pos2[2], 2)
        );
        
        // 应用简单的路径损耗模型
        SimSettings simSettings = simManager.getSimulationSettings();
        double pathLossParam = simSettings.getUAVPathLossParameter();
        double pathLossExponent = simSettings.getUAVPathLossExponent();
        double additionalLoss = simSettings.getUAVAdditionalPathLoss();
        
        // 计算路径损耗（dB）
        double pathLoss = pathLossParam + 10 * pathLossExponent * Math.log10(distance) + additionalLoss;
        
        // 将路径损耗转换为通信质量（0-1）
        double maxPathLoss = 100.0; // 假设100dB是最大可接受的路径损耗
        double quality = Math.max(0, 1 - (pathLoss / maxPathLoss));
        
        return quality;
    }
    
    /**
     * 计算UAV能量消耗率
     * @param uav UAV对象
     * @param isMoving 是否移动中
     * @return 能量消耗率（每秒）
     */
    public double calculateEnergyConsumptionRate(UAV uav, boolean isMoving) {
        SimSettings simSettings = simManager.getSimulationSettings();
        double flightPower = simSettings.getUAVFlightPower();
        double hoverPower = simSettings.getUAVHoverPower();
        
        // 根据UAV状态选择能耗率
        UAV.UAVState state = uav.getState();
        
        switch (state) {
            case MOVING:
                return flightPower;
            case HOVERING:
                return hoverPower;
            case PROCESSING:
                // 处理任务时额外能耗
                return hoverPower * 1.5;
            case IDLE:
                // 空闲状态最小能耗
                return hoverPower * 0.8;
            default:
                return hoverPower;
        }
    }
    
    /**
     * 获取所有UAV列表
     * @return UAV列表
     */
    public List<UAV> getUAVs() {
        return uavList;
    }
    
    /**
     * 获取指定ID的UAV
     * @param id UAV ID
     * @return UAV对象，如果不存在则返回null
     */
    public UAV getUAV(int id) {
        for (UAV uav : uavList) {
            if (uav.getId() == id) {
                return uav;
            }
        }
        return null;
    }
    
    /**
     * 向UAV分配任务
     * @param uavId UAV ID
     * @param task 任务
     * @return 是否成功分配
     */
    public boolean assignTask(int uavId, UAV.Task task) {
        UAV uav = getUAV(uavId);
        if (uav == null) {
            return false;
        }
        return uav.addTask(task);
    }

    /** Submit an EdgeCloudSim task to a concrete UAV resource. */
    public boolean submitEdgeTask(int uavId, Task edgeTask) {
        String internalId = "edgecloudsim-" + edgeTask.getCloudletId();
        UAV.Task uavTask = new UAV.Task(
                internalId,
                edgeTask.getCloudletLength(),
                (long) (org.cloudbus.cloudsim.core.CloudSim.clock() * 1000));
        if (!assignTask(uavId, uavTask)) {
            return false;
        }
        activeEdgeTasks.put(internalId, edgeTask);
        return true;
    }

    public int getActiveEdgeTaskCount() {
        return activeEdgeTasks.size();
    }

    public int getMovementCommandCount() {
        return movementCommandCount;
    }

    public int getBoundaryConstraintCount() {
        return boundaryConstraintCount;
    }
    
    /**
     * 获取完成的任务数量
     */
    public int getCompletedTaskCount() {
        return completedTasks.get();
    }
    
    /**
     * 重置UAV状态 - 用于恢复卡死
     */
    public void resetUAVs() {
        SimLogger.printLine("执行UAV紧急重置");
        SimSettings simSettings = simManager.getSimulationSettings();
        
        for (UAV uav : uavList) {
            // 清空任务队列
            uav.clearTaskQueue();
            
            // 恢复一定的能量
            double maxEnergy = simSettings.getUAVMaxEnergy();
            uav.setEnergy(maxEnergy * 0.5); // 恢复到50%能量
            
            SimLogger.printLine("重置UAV #" + uav.getId() + " 成功");
        }
    }
}
