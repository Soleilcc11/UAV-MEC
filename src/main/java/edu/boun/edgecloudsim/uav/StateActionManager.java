package edu.boun.edgecloudsim.uav;

import java.util.*;
import org.json.JSONArray;
import org.json.JSONObject;

import edu.boun.edgecloudsim.core.SimManager;
import edu.boun.edgecloudsim.core.SimSettings;
import edu.boun.edgecloudsim.utils.SimLogger;

/**
 * 状态动作管理器，负责生成状态和解析动作
 */
public class StateActionManager {
    private SimManager simManager;
    private int stateDimension;
    private int actionDimension;
    private double[] lastState;
    private double[] lastAction;
    private boolean useDynamicState;
    
    /**
     * 构造函数
     * @param simManager 模拟管理器
     */
    public StateActionManager(SimManager simManager) {
        this.simManager = simManager;
        this.useDynamicState = true; // 默认使用动态状态
        
        // 设置状态维度
        this.stateDimension = calculateStateDimension();
        
        // 设置动作维度（取决于UAV数量和每个UAV的行为）
        SimSettings simSettings = simManager.getSimulationSettings();
        int numUavs = simSettings.getNumOfUAVs();
        // 每个UAV有3个动作：移动方向(x,y,z)
        this.actionDimension = numUavs * 3;
        
        SimLogger.printLine("状态维度: " + stateDimension);
        SimLogger.printLine("动作维度: " + actionDimension);
    }
    
    /**
     * 计算状态维度
     * @return 状态维度
     */
    private int calculateStateDimension() {
        int dimension = 0;
        
        // 获取UAV数量
        SimSettings simSettings = simManager.getSimulationSettings();
        int numUavs = simSettings.getNumOfUAVs();
        
        if (useDynamicState) {
            // 动态状态包括：
            // 1. 每个UAV的位置(x,y,z): 3维
            // 2. 每个UAV的能量水平: 1维
            // 3. 每个UAV的任务队列长度: 1维
            // 4. 每个UAV的处理能力: 1维
            // 5. 系统当前时间: 1维
            dimension = numUavs * (3 + 1 + 1 + 1) + 1;
        } else {
            // 简化的状态表示
            dimension = numUavs * 4 + 1;
        }
        
        return dimension;
    }
    
    /**
     * 生成当前状态表示
     * @return 状态向量
     */
    public double[] generateState() {
        UAVManager uavManager = simManager.getUAVManager();
        List<UAV> uavList = uavManager.getUAVs();
        double[] state = new double[stateDimension];
        int index = 0;
        
        // 当前归一化的时间 (0-1)
        double currentTime = simManager.getSimulationTime();
        double simulationLength = 3600; // 假设模拟时长为3600秒
        state[index++] = currentTime / simulationLength;
        
        // 获取每个UAV的状态信息
        for (UAV uav : uavList) {
            // 归一化的位置坐标
            double[] position = uav.getPosition();
            double[] simSpace = simManager.getSimulationSettings().getSimulationSpace();
            
            state[index++] = position[0] / simSpace[0];
            state[index++] = position[1] / simSpace[1];
            state[index++] = position[2] / simSpace[2];
            
            // 归一化的能量水平 (0-1)
            double maxEnergy = simManager.getSimulationSettings().getUAVMaxEnergy();
            state[index++] = uav.getEnergy() / maxEnergy;
            
            if (useDynamicState) {
                // 归一化的任务队列长度
                int maxQueueLength = 50; // 假设最大队列长度
                state[index++] = uav.getTaskQueueLength() / (double) maxQueueLength;
                
                // 归一化的处理能力
                double maxProcessingCap = 1000; // 假设最大处理能力 (MIPS)
                state[index++] = uav.getProcessingCapacity() / maxProcessingCap;
            }
        }
        
        // 保存最后一个状态
        this.lastState = state;
        
        return state;
    }
    
    /**
     * 解析强化学习模型返回的动作
     * @param actionVector 动作向量
     * @return 结构化的动作对象
     */
    public List<Map<String, Object>> parseOffloadingDecision(double[] actionVector) {
        List<Map<String, Object>> actions = new ArrayList<>();
        this.lastAction = actionVector;
        
        UAVManager uavManager = simManager.getUAVManager();
        List<UAV> uavList = uavManager.getUAVs();
        int numUavs = uavList.size();
        
        // 确保动作向量尺寸正确
        if (actionVector.length != actionDimension) {
            SimLogger.printLine("警告: 动作向量尺寸不匹配! 预期: " + actionDimension + ", 实际: " + actionVector.length);
            return actions;
        }
        
        // 计算每个UAV的动作
        for (int i = 0; i < numUavs; i++) {
            Map<String, Object> uavAction = new HashMap<>();
            uavAction.put("uavId", i);
            
            // 提取移动方向
            double moveX = actionVector[i * 3];
            double moveY = actionVector[i * 3 + 1];
            double moveZ = actionVector[i * 3 + 2];
            
            // 缩放动作值(-1到1)为实际移动距离
            double[] simSpace = simManager.getSimulationSettings().getSimulationSpace();
            double maxMoveDistance = 10.0; // 假设UAV每步最大移动距离
            
            double scaledMoveX = moveX * maxMoveDistance;
            double scaledMoveY = moveY * maxMoveDistance;
            double scaledMoveZ = moveZ * maxMoveDistance;
            
            // 设置移动动作
            double[] moveDirection = new double[] {scaledMoveX, scaledMoveY, scaledMoveZ};
            uavAction.put("moveDirection", moveDirection);
            
            // 添加到动作列表
            actions.add(uavAction);
        }
        
        return actions;
    }
    
    /**
     * 获取最后一个生成的状态
     * @return 最后的状态向量
     */
    public double[] getLastState() {
        return lastState;
    }
    
    /**
     * 获取最后一个解析的动作
     * @return 最后的动作向量
     */
    public double[] getLastAction() {
        return lastAction;
    }
    
    /**
     * 获取状态维度
     * @return 状态维度
     */
    public int getStateDimension() {
        return stateDimension;
    }
    
    /**
     * 获取动作维度
     * @return 动作维度
     */
    public int getActionDimension() {
        return actionDimension;
    }
    
    /**
     * 设置是否使用动态状态表示
     * @param useDynamicState 是否使用动态状态
     */
    public void setUseDynamicState(boolean useDynamicState) {
        this.useDynamicState = useDynamicState;
        this.stateDimension = calculateStateDimension();
    }
}