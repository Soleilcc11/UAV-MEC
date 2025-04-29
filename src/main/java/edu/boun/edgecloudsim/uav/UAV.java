package edu.boun.edgecloudsim.uav;

import java.util.ArrayList;
import java.util.List;
import java.util.Queue;
import java.util.LinkedList;

/**
 * UAV类，代表一个无人机实体
 */
public class UAV {
    private int id;
    private double[] position;
    private double energy;
    private double processingCapacity;
    private Queue<Task> taskQueue;
    private double speed;
    private UAVState state;
    
    /**
     * UAV状态枚举
     */
    public enum UAVState {
        IDLE,
        MOVING,
        HOVERING,
        PROCESSING
    }
    
    /**
     * 构造函数
     * @param id UAV ID
     * @param initialPosition 初始位置
     * @param initialEnergy 初始能量
     * @param processingCapacity 处理能力（MIPS）
     */
    public UAV(int id, double[] initialPosition, double initialEnergy, double processingCapacity) {
        this.id = id;
        this.position = initialPosition;
        this.energy = initialEnergy;
        this.processingCapacity = processingCapacity;
        this.taskQueue = new LinkedList<>();
        this.speed = 5.0; // 默认速度 m/s
        this.state = UAVState.IDLE;
    }
    
    /**
     * 更新UAV位置
     * @param direction 移动方向 [deltaX, deltaY, deltaZ]
     * @return 移动后的新位置
     */
    public double[] updatePosition(double[] direction) {
        if (energy <= 0) {
            // 能量耗尽，不能移动
            return position;
        }
        
        // 计算实际移动距离
        double distance = Math.sqrt(
            direction[0] * direction[0] + 
            direction[1] * direction[1] + 
            direction[2] * direction[2]
        );
        
        // 应用速度限制
        if (distance > speed) {
            double scale = speed / distance;
            direction[0] *= scale;
            direction[1] *= scale;
            direction[2] *= scale;
            distance = speed;
        }
        
        // 更新位置
        position[0] += direction[0];
        position[1] += direction[1];
        position[2] += direction[2];
        
        // 更新状态
        if (distance > 0) {
            state = UAVState.MOVING;
        } else if (taskQueue.size() > 0) {
            state = UAVState.PROCESSING;
        } else {
            state = UAVState.HOVERING;
        }
        
        // 消耗能量
        consumeEnergy(distance);
        
        return position;
    }
    
    /**
     * 消耗能量
     * @param distance 移动距离
     */
    private void consumeEnergy(double distance) {
        // 简单能量消耗模型
        double movementEnergy = distance * 0.1; // 每米消耗0.1单位能量
        double processingEnergy = (state == UAVState.PROCESSING) ? 0.2 : 0; // 处理任务额外消耗
        double hoveringEnergy = (state == UAVState.HOVERING) ? 0.05 : 0; // 悬停消耗
        
        double totalConsumption = movementEnergy + processingEnergy + hoveringEnergy;
        energy = Math.max(0, energy - totalConsumption);
    }
    
    /**
     * 添加任务到队列
     * @param task 要添加的任务
     * @return 是否成功添加
     */
    public boolean addTask(Task task) {
        if (taskQueue.size() >= 50) { // 假设最大队列长度为50
            return false;
        }
        return taskQueue.offer(task);
    }
    
    /**
     * 处理队列中的任务
     * @param timeSlot 时间片
     * @return 处理完成的任务列表
     */
    public List<Task> processTasks(double timeSlot) {
        List<Task> completedTasks = new ArrayList<>();
        
        if (energy <= 0 || taskQueue.isEmpty()) {
            return completedTasks;
        }
        
        // 处理任务逻辑
        double availableMIPS = processingCapacity * timeSlot;
        double usedMIPS = 0;
        
        while (!taskQueue.isEmpty() && usedMIPS < availableMIPS) {
            Task currentTask = taskQueue.peek();
            double remainingMI = currentTask.getRemainingMI();
            
            if (remainingMI <= (availableMIPS - usedMIPS)) {
                // 任务可以在当前时间片完成
                taskQueue.poll(); // 从队列移除
                usedMIPS += remainingMI;
                currentTask.setRemainingMI(0);
                currentTask.setCompletionTime(System.currentTimeMillis());
                completedTasks.add(currentTask);
            } else {
                // 任务部分完成
                double processedMI = availableMIPS - usedMIPS;
                currentTask.setRemainingMI(remainingMI - processedMI);
                usedMIPS = availableMIPS;
            }
        }
        
        // 更新状态
        if (taskQueue.isEmpty()) {
            state = UAVState.HOVERING;
        } else {
            state = UAVState.PROCESSING;
        }
        
        // 消耗处理能量
        double processingEnergyConsumption = usedMIPS * 0.001; // 假设每MIPS消耗0.001单位能量
        energy = Math.max(0, energy - processingEnergyConsumption);
        
        return completedTasks;
    }
    /**
     * 获取可用计算能力
     * @return 可用计算能力 (MIPS)
     */
    public double getAvailableCapacity() {
        // 假设UAV有一个总计算能力和当前使用的计算能力
        return getProcessingCapacity() - getCurrentUsage();
    }

    /**
     * 获取当前使用的计算能力
     * @return 当前使用的计算能力 (MIPS)
     */
    private double getCurrentUsage() {
        // 实现计算当前使用的计算能力
        return 0.0; // 示例返回值
    }
    
    /**
     * 获取UAV ID
     */
    public int getId() {
        return id;
    }
    
    /**
     * 获取当前位置
     */
    public double[] getPosition() {
        return position;
    }
    
    /**
     * 获取当前能量
     */
    public double getEnergy() {
        return energy;
    }
    
    /**
     * 获取处理能力
     */
    public double getProcessingCapacity() {
        return processingCapacity;
    }
    
    /**
     * 获取任务队列长度
     */
    public int getTaskQueueLength() {
        return taskQueue.size();
    }
    
    /**
     * 获取当前状态
     */
    public UAVState getState() {
        return state;
    }
    
    /**
     * 设置速度
     */
    public void setSpeed(double speed) {
        this.speed = speed;
    }
    
    /**
     * 充电
     * @param amount 充电量
     * @param maxEnergy 最大能量
     */
    public void recharge(double amount, double maxEnergy) {
        energy = Math.min(energy + amount, maxEnergy);
    }
    
    /**
     * Task类，代表一个计算任务
     */
    public static class Task {
        private String id;
        private double totalMI;
        private double remainingMI;
        private long arrivalTime;
        private long completionTime;
        
        public Task(String id, double totalMI) {
            this.id = id;
            this.totalMI = totalMI;
            this.remainingMI = totalMI;
            this.arrivalTime = System.currentTimeMillis();
            this.completionTime = -1;
        }
        
        public String getId() {
            return id;
        }
        
        public double getTotalMI() {
            return totalMI;
        }
        
        public double getRemainingMI() {
            return remainingMI;
        }
        
        public void setRemainingMI(double remainingMI) {
            this.remainingMI = remainingMI;
        }
        
        public long getArrivalTime() {
            return arrivalTime;
        }
        
        public long getCompletionTime() {
            return completionTime;
        }
        
        public void setCompletionTime(long completionTime) {
            this.completionTime = completionTime;
        }
        
        public boolean isCompleted() {
            return remainingMI <= 0;
        }
    }
}