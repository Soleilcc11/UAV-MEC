package edu.boun.edgecloudsim.uav;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Map;
import java.util.HashMap;
import edu.boun.edgecloudsim.core.SimManager;
import edu.boun.edgecloudsim.core.SimSettings;
import edu.boun.edgecloudsim.utils.SimLogger;
import edu.boun.edgecloudsim.edge_orchestrator.EdgeOrchestrator;

/**
 * UAV管理器类，负责管理无人机
 */
public class UAVManager {
    private SimManager simManager;
    private List<UAV> uavList;
    private Random random;
    
    /**
     * 构造函数
     * @param simManager 模拟管理器
     */
    public UAVManager(SimManager simManager) {
        this.simManager = simManager;
        this.uavList = new ArrayList<>();
        this.random = new Random(42); // 固定随机种子以实现可重复的模拟
        
        // 在构造函数中初始化，因为simManager还没有完全设置好
    }
    
    /**
     * 初始化UAV管理器
     */
    public void initialize() {
        SimSettings simSettings = simManager.getSimulationSettings();
        int numUavs = simSettings.getNumOfUAVs();
        
        SimLogger.printLine("初始化 " + numUavs + " 个UAV...");
        
        // 创建UAVs
        for (int i = 0; i < numUavs; i++) {
            // 随机位置
            double[] position = generateRandomPosition();
            
            // 初始能量和处理能力
            double initialEnergy = simSettings.getUAVMaxEnergy();
            double processingCapacity = 1000 + random.nextDouble() * 500; // 1000-1500 MIPS
            
            UAV uav = new UAV(i, position, initialEnergy, processingCapacity);
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
                UAV uav = uavList.get(uavId);
                double[] newPosition = uav.updatePosition(moveDirection);
                
                // 检查边界条件
                enforceBoundaryConstraints(uav, newPosition);
            }
        }
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
        int completedTaskCount = 0;
        
        for (UAV uav : uavList) {
            List<UAV.Task> completedTasks = uav.processTasks(timeSlot);
            completedTaskCount += completedTasks.size();
        }
        
        return completedTaskCount;
    }
    
    /**
     * 生成随机位置
     * @return 随机位置坐标 [x, y, z]
     */
    private double[] generateRandomPosition() {
        SimSettings simSettings = simManager.getSimulationSettings();
        double[] position = new double[3];
        
        // X坐标
        position[0] = simSettings.getRandomPositionX();
        
        // Y坐标
        position[1] = simSettings.getRandomPositionY();
        
        // Z坐标（高度）
        position[2] = simSettings.getUAVInitialHeight();
        
        return position;
    }
    
    /**
     * 确保UAV位置在模拟空间边界内
     * @param uav UAV对象
     * @param position 位置坐标
     */
    private void enforceBoundaryConstraints(UAV uav, double[] position) {
        SimSettings simSettings = simManager.getSimulationSettings();
        double[] simSpace = simSettings.getSimulationSpace();
        double minHeight = simSettings.getUAVMinHeight();
        double maxHeight = simSettings.getUAVMaxHeight();
        
        // X边界
        if (position[0] < 0) position[0] = 0;
        if (position[0] > simSpace[0]) position[0] = simSpace[0];
        
        // Y边界
        if (position[1] < 0) position[1] = 0;
        if (position[1] > simSpace[1]) position[1] = simSpace[1];
        
        // Z边界（高度）
        if (position[2] < minHeight) position[2] = minHeight;
        if (position[2] > maxHeight) position[2] = maxHeight;
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
}