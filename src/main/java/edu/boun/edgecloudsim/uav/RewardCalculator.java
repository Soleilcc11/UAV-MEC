package edu.boun.edgecloudsim.uav;

import edu.boun.edgecloudsim.core.SimManager;

/**
 * 奖励计算器，计算RL模型的奖励值
 */
public class RewardCalculator {
    private SimManager simManager;
    private double latencyWeight;
    private double energyWeight;
    private double loadBalanceWeight;
    private double completionWeight;
    private double maxAcceptableLatency;
    
    /**
     * 构造函数
     * @param simManager 模拟管理器
     */
    public RewardCalculator(SimManager simManager) {
        this.simManager = simManager;
        
        // 默认奖励权重
        this.latencyWeight = 0.4;
        this.energyWeight = 0.3;
        this.loadBalanceWeight = 0.2;
        this.completionWeight = 0.1;
        
        // 最大可接受延迟（毫秒）
        this.maxAcceptableLatency = 1000.0;
    }
    
    /**
     * 计算奖励值
     * @param latency 任务延迟（毫秒）
     * @param taskSize 任务大小（MI）
     * @return 奖励值
     */
    public double calculateReward(double latency, double taskSize) {
        // 延迟组件（延迟越低，奖励越高）
        double latencyReward = calculateLatencyReward(latency);
        
        // 能量组件（剩余能量越高，奖励越高）
        double energyReward = calculateEnergyReward();
        
        // 负载均衡组件（任务分配均匀，奖励越高）
        double loadBalanceReward = calculateLoadBalanceReward();
        
        // 任务完成组件（完成率越高，奖励越高）
        double completionReward = calculateCompletionReward();
        
        // 总奖励
        double totalReward = 
            latencyWeight * latencyReward +
            energyWeight * energyReward +
            loadBalanceWeight * loadBalanceReward +
            completionWeight * completionReward;
        
        return totalReward;
    }
    
    /**
     * 计算延迟奖励
     * @param latency 任务延迟（毫秒）
     * @return 延迟奖励
     */
    private double calculateLatencyReward(double latency) {
        // 延迟奖励是一个负指数函数，随着延迟增加而减少
        // 当延迟为0时，奖励为1
        // 当延迟达到最大可接受延迟时，奖励接近0
        return Math.exp(-latency / maxAcceptableLatency);
    }
    
    /**
     * 计算能量奖励
     * @return 能量奖励
     */
    private double calculateEnergyReward() {
        // 获取所有UAV的平均能量水平
        UAVManager uavManager = simManager.getUAVManager();
        double totalEnergy = 0.0;
        double maxEnergy = simManager.getSimulationSettings().getUAVMaxEnergy();
        int uavCount = uavManager.getUAVs().size();
        
        for (UAV uav : uavManager.getUAVs()) {
            totalEnergy += uav.getEnergy();
        }
        
        // 平均能量水平（0-1）
        double averageEnergyLevel = totalEnergy / (uavCount * maxEnergy);
        
        // 返回平均能量水平作为奖励
        return averageEnergyLevel;
    }
    
    /**
     * 计算负载均衡奖励
     * @return 负载均衡奖励
     */
    private double calculateLoadBalanceReward() {
        // 计算任务队列长度的标准差，标准差越小，负载越均衡，奖励越高
        UAVManager uavManager = simManager.getUAVManager();
        int uavCount = uavManager.getUAVs().size();
        
        if (uavCount <= 1) {
            return 1.0; // 只有一个UAV时，负载必然均衡
        }
        
        // 计算平均队列长度
        double totalQueueLength = 0.0;
        for (UAV uav : uavManager.getUAVs()) {
            totalQueueLength += uav.getTaskQueueLength();
        }
        double avgQueueLength = totalQueueLength / uavCount;
        
        // 计算队列长度的方差
        double variance = 0.0;
        for (UAV uav : uavManager.getUAVs()) {
            double diff = uav.getTaskQueueLength() - avgQueueLength;
            variance += diff * diff;
        }
        variance /= uavCount;
        
        // 标准差
        double stdDev = Math.sqrt(variance);
        
        // 负载均衡奖励（标准差越小，奖励越高）
        // 使用指数函数，当标准差为0时，奖励为1
        // 当标准差增大时，奖励指数减小
        double maxStdDev = 10.0; // 假设最大可接受的标准差为10
        return Math.exp(-stdDev / maxStdDev);
    }
    
    /**
     * 计算任务完成奖励
     * @return 任务完成奖励
     */
    private double calculateCompletionReward() {
        // 获取任务完成率
        TaskOffloadingEngine offloadingEngine = (TaskOffloadingEngine) simManager.getTaskOffloadingEngine();
        
        if (offloadingEngine == null) {
            return 0.0;
        }
        
        int completed = offloadingEngine.getCompletedTaskCount();
        int total = offloadingEngine.getTotalTaskCount();
        
        if (total == 0) {
            return 1.0; // 没有任务时，完成率为100%
        }
        
        // 返回完成率作为奖励
        return (double) completed / total;
    }
    
    /**
     * 设置奖励权重
     * @param latencyWeight 延迟权重
     * @param energyWeight 能量权重
     * @param loadBalanceWeight 负载均衡权重
     * @param completionWeight 完成率权重
     */
    public void setWeights(double latencyWeight, double energyWeight, 
                           double loadBalanceWeight, double completionWeight) {
        // 归一化权重
        double sum = latencyWeight + energyWeight + loadBalanceWeight + completionWeight;
        
        this.latencyWeight = latencyWeight / sum;
        this.energyWeight = energyWeight / sum;
        this.loadBalanceWeight = loadBalanceWeight / sum;
        this.completionWeight = completionWeight / sum;
    }
    
    /**
     * 设置最大可接受延迟
     * @param maxAcceptableLatency 最大可接受延迟（毫秒）
     */
    public void setMaxAcceptableLatency(double maxAcceptableLatency) {
        this.maxAcceptableLatency = maxAcceptableLatency;
    }
}