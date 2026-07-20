package edu.boun.edgecloudsim.uav;

import java.util.ArrayList;
import java.util.List;
import java.util.Queue;

import org.cloudbus.cloudsim.core.SimEvent;
import org.cloudbus.cloudsim.core.CloudSim;

import java.util.LinkedList;
import edu.boun.edgecloudsim.core.SimSettings;
import edu.boun.edgecloudsim.utils.SimLogger;


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
    private double flightPowerPerSecond = 0.1;
    private double hoverPowerPerSecond = 0.05;
    private double processingEnergyPerMi = 0.001;
    private double flightEnergyConsumed;
    private double hoverEnergyConsumed;
    private double processingEnergyConsumed;
    private int maxObservedQueueLength;
    
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

    public void configureEnergyModel(double flightPower, double hoverPower,
            double computeEnergyPerMi) {
        if (flightPower < 0 || hoverPower < 0 || computeEnergyPerMi < 0) {
            throw new IllegalArgumentException("Energy coefficients must be non-negative");
        }
        this.flightPowerPerSecond = flightPower;
        this.hoverPowerPerSecond = hoverPower;
        this.processingEnergyPerMi = computeEnergyPerMi;
    }
    
    /**
     * 更新UAV位置
     * @param direction 移动方向 [deltaX, deltaY, deltaZ]
     * @return 移动后的新位置
     */
    public double[] updatePosition(double[] direction) {
        return updatePosition(direction, 1.0);
    }

    /** Move for an explicit simulation-time interval and charge actual flight time. */
    public double[] updatePosition(double[] direction, double elapsedSeconds) {
        if (direction == null || direction.length != 3) {
            throw new IllegalArgumentException("UAV displacement must contain x, y, and z");
        }
        if (!Double.isFinite(elapsedSeconds) || elapsedSeconds < 0.0) {
            throw new IllegalArgumentException("Movement elapsed time must be non-negative");
        }
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
        double maxDistance = speed * elapsedSeconds;
        if (distance > maxDistance && distance > 0.0) {
            double scale = maxDistance / distance;
            direction[0] *= scale;
            direction[1] *= scale;
            direction[2] *= scale;
            distance = maxDistance;
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
        
        double flightSeconds = speed <= 0.0 ? 0.0 : distance / speed;
        // PROCESS_UAV_TASKS already charges the baseline hover power for every
        // simulated second. A movement command therefore adds only the flight
        // power above that baseline, avoiding double charging the same interval.
        double incrementalFlightPower = Math.max(
                0.0, flightPowerPerSecond - hoverPowerPerSecond);
        consumeFlightEnergy(incrementalFlightPower * flightSeconds);
        
        return position;
    }
    
    /**
     * 消耗能量
     * @param distance 移动距离
     */
    private double consume(double requested) {
        double consumed = Math.min(energy, Math.max(0.0, requested));
        energy -= consumed;
        return consumed;
    }

    private void consumeFlightEnergy(double requested) {
        flightEnergyConsumed += consume(requested);
    }

    private void consumeHoverEnergy(double requested) {
        hoverEnergyConsumed += consume(requested);
    }

    private void consumeProcessingEnergy(double requested) {
        processingEnergyConsumed += consume(requested);
    }
    
    /**
     * 添加任务到队列
     * @param task 要添加的任务
     * @return 是否成功添加
     */
    public boolean addTask(Task task) {
        if (task == null || energy <= 0 || taskQueue.size() >= 50) { // 假设最大队列长度为50
            return false;
        }
        boolean accepted = taskQueue.offer(task);
        if (accepted) {
            maxObservedQueueLength = Math.max(maxObservedQueueLength, taskQueue.size());
        }
        return accepted;
    }

    /** Remove one queued task after its backing EdgeCloudSim task is failed. */
    public boolean cancelTask(String taskId) {
        if (taskId == null) {
            return false;
        }
        return taskQueue.removeIf(task -> taskId.equals(task.getId()));
    }
    
    /**
     * 处理队列中的任务
     * @param timeSlot 时间片
     * @return 处理完成的任务列表
     */
    /**
     * 处理队列中的任务 - 简化版，不依赖事件系统
     * @param time 时间片
     * @return 处理完成的任务列表
     */
    public List<Task> processTasks(double time) {
        List<Task> completedTasks = new ArrayList<>();
        
        // 添加防护检查
        if (time <= 0) {
            SimLogger.printLine("[WARNING] UAV " + id + " 收到无效时间值: " + time);
            return completedTasks;
        }
        
        if (taskQueue.isEmpty()) {
            state = UAVState.HOVERING;
            consumeHoverEnergy(hoverPowerPerSecond * time);
            return completedTasks;
        }

        state = UAVState.PROCESSING;
        consumeHoverEnergy(hoverPowerPerSecond * time);
        if (energy <= 0) {
            return completedTasks;
        }
        
        // 详细记录处理状态
        SimLogger.printLine("UAV " + id + " 处理前状态: 能量=" + energy + 
                        ", 处理能力=" + processingCapacity + " MIPS" + 
                        ", 队列大小=" + taskQueue.size());
        
        double usedMIPS = 0;
        double availableMIPS = processingCapacity * time;
        if (processingEnergyPerMi > 0) {
            availableMIPS = Math.min(availableMIPS, energy / processingEnergyPerMi);
        }
        double processingEnergyConsumption = 0;
        
        // 处理队列中的任务
        while (!taskQueue.isEmpty() && usedMIPS < availableMIPS) {
            Task currentTask = taskQueue.peek();
            
            if (currentTask == null) {
                SimLogger.printLine("[WARNING] UAV " + id + " 队列中存在空任务，跳过处理");
                taskQueue.poll();
                continue;
            }
            
            double remainingMI = currentTask.getRemainingMI();
            currentTask.markStarted((long) (CloudSim.clock() * 1000));
            SimLogger.printLine("UAV " + id + " 正在处理任务 #" + currentTask.getId() + 
                            ", 剩余MI: " + remainingMI + 
                            ", 可用MIPS: " + (availableMIPS - usedMIPS));
            
            // 检查是否可以完成任务
            if (remainingMI <= (availableMIPS - usedMIPS)) {
                // 可以完成任务
                usedMIPS += remainingMI;
                
                // 计算能量消耗
                double taskEnergyConsumption = remainingMI * processingEnergyPerMi;
                processingEnergyConsumption += taskEnergyConsumption;
                
                // 从队列中移除任务
                Task task = taskQueue.poll();
                
                // 设置完成时间
                double currentSimTime = CloudSim.clock();
                long completionTimeMs = (long)(currentSimTime * 1000);
                task.setCompletionTime(completionTimeMs);
                
                // 确保任务已完成
                task.setRemainingMI(0);
                
                // 添加到完成列表
                completedTasks.add(task);
                
                SimLogger.printLine("UAV " + id + " 完成任务 #" + task.getId() + 
                                ", 仿真时间: " + currentSimTime);
            } else {
                // 无法完成任务，部分处理
                double processedMI = availableMIPS - usedMIPS;
                double newRemainingMI = remainingMI - processedMI;
                
                // 计算能量消耗
                double taskEnergyConsumption = processedMI * processingEnergyPerMi;
                processingEnergyConsumption += taskEnergyConsumption;
                
                // 更新剩余MI
                currentTask.setRemainingMI(newRemainingMI);
                usedMIPS = availableMIPS;
                
                SimLogger.printLine("UAV " + id + " 部分处理任务 #" + currentTask.getId() + 
                                ", 原剩余MI: " + remainingMI + 
                                ", 处理了: " + processedMI + 
                                ", 新剩余MI: " + newRemainingMI);
            }
        }
        
        // 更新能量消耗
        consumeProcessingEnergy(processingEnergyConsumption);
        state = taskQueue.isEmpty() ? UAVState.HOVERING : UAVState.PROCESSING;
        
        // 记录处理后状态
        SimLogger.printLine("UAV " + id + " 处理后状态: 能量=" + energy + 
                        ", 处理消耗=" + processingEnergyConsumption + 
                        ", 剩余队列=" + taskQueue.size() + 
                        ", 完成任务数: " + completedTasks.size());
        
        return completedTasks;
    }

    /**
     * 清空任务队列 - 用于处理卡死情况
     */
    public void clearTaskQueue() {
        int originalSize = taskQueue.size();
        taskQueue.clear();
        SimLogger.printLine("UAV " + id + " 已清空任务队列，原队列大小: " + originalSize);
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
        return taskQueue.size() > 0 ? processingCapacity * 0.5 : 0.0; // 简化计算，一旦有任务，使用50%能力
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
     * 设置能量水平 - 用于恢复
     */
    public void setEnergy(double energy) {
        this.energy = energy;
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

    public int getMaxObservedQueueLength() {
        return maxObservedQueueLength;
    }

    public double getFlightEnergyConsumed() {
        return flightEnergyConsumed;
    }

    public double getHoverEnergyConsumed() {
        return hoverEnergyConsumed;
    }

    public double getProcessingEnergyConsumed() {
        return processingEnergyConsumed;
    }

    public double getTotalEnergyConsumed() {
        return flightEnergyConsumed + hoverEnergyConsumed + processingEnergyConsumed;
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
        if (!Double.isFinite(speed) || speed < 0.0) {
            throw new IllegalArgumentException("UAV speed must be finite and non-negative");
        }
        this.speed = speed;
    }

    public double getSpeed() {
        return speed;
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
        private long startTime;
        private int status;
        
        // 任务状态常量
        public static final int CREATED_STATUS = 0;
        public static final int PROCESSING_STATUS = 1;
        public static final int COMPLETED_STATUS = 2;
        public static final int CANCELED_STATUS = 3;
        
        public Task(String id, double totalMI) {
            this(id, totalMI, (long) (CloudSim.clock() * 1000));
        }

        Task(String id, double totalMI, long arrivalTime) {
            this.id = id;
            this.totalMI = totalMI;
            this.remainingMI = totalMI;
            this.arrivalTime = arrivalTime;
            this.completionTime = -1;
            this.startTime = -1;
            this.status = CREATED_STATUS;
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
            if (remainingMI <= 0) {
                this.status = COMPLETED_STATUS;
            } else {
                this.status = PROCESSING_STATUS;
            }
        }
        
        /**
         * 获取任务到达时间
         * @return 任务到达时间（毫秒）
         */
        public long getArrivalTime() {
            return arrivalTime;
        }
        
        /**
         * 获取任务完成时间
         * @return 任务完成时间（毫秒）
         */
        public long getCompletionTime() {
            return completionTime;
        }

        void markStarted(long currentTime) {
            if (startTime < 0) {
                startTime = currentTime;
            }
        }

        public long getStartTime() {
            return startTime;
        }

        public long getQueueWaitTime() {
            return startTime < 0 ? -1 : Math.max(0, startTime - arrivalTime);
        }
        
        /**
         * 设置任务完成时间
         * @param completionTime 任务完成时间（毫秒）
         */
        public void setCompletionTime(long completionTime) {
            this.completionTime = completionTime;
            this.status = COMPLETED_STATUS;
        }
        
        public boolean isCompleted() {
            return remainingMI <= 0 || status == COMPLETED_STATUS;
        }
        
        public int getStatus() {
            return status;
        }
        
        public void setStatus(int status) {
            this.status = status;
        }
    }
}
