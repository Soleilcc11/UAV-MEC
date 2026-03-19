package edu.boun.edgecloudsim.uav;

import java.io.BufferedWriter;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.Map;
import java.util.Queue;

import edu.boun.edgecloudsim.core.SimManager;
import edu.boun.edgecloudsim.utils.SimLogger;

/**
 * 性能监控器，用于监控和记录系统性能指标
 */
public class PerformanceMonitor {
    private SimManager simManager;
    private Map<String, Double> metrics;
    private Map<String, Queue<Double>> metricHistory;
    private int historySize;
    private BufferedWriter performanceLogBW;
    private boolean performanceLogInitialized = false;
    
    /**
     * 构造函数
     * @param simManager 模拟管理器
     */
    public PerformanceMonitor(SimManager simManager) {
        this.simManager = simManager;
        this.metrics = new HashMap<>();
        this.metricHistory = new HashMap<>();
        this.historySize = 100; // 保存最近100个指标数值
        
        // 初始化指标
        initializeMetrics();
    }
    
    /**
     * 初始化性能指标
     */
    private void initializeMetrics() {
        // 系统级指标
        metrics.put("system.uptime", 0.0);
        metrics.put("system.taskCompletionRate", 0.0);
        metrics.put("system.averageLatency", 0.0);
        metrics.put("system.energyEfficiency", 0.0);
        
        // UAV级指标
        metrics.put("uav.averageEnergy", 0.0);
        metrics.put("uav.averageUtilization", 0.0);
        metrics.put("uav.averageQueueLength", 0.0);
        metrics.put("uav.communicationQuality", 0.0);
        
        // 任务级指标
        metrics.put("task.averageWaitTime", 0.0);
        metrics.put("task.averageProcessingTime", 0.0);
        metrics.put("task.rejectionRate", 0.0);
        
        // 初始化历史记录队列
        for (String key : metrics.keySet()) {
            metricHistory.put(key, new LinkedList<>());
        }
    }

    /**
     * 初始化性能日志文件
     */
    public void initializePerformanceLog() {
        try {
            // 创建CSV文件头
            String header = "Time,TaskCompletionRate,AverageLatency,EnergyEfficiency,UAVAverageEnergy,UAVAverageUtilization,UAVAverageQueueLength,UAVCommunicationQuality";
            SimLogger.getInstance().writeToPerformanceLogFile(header);
            performanceLogInitialized = true;
            SimLogger.printLine("性能日志初始化成功");
        } catch (Exception e) {
            SimLogger.printLine("初始化性能日志时出错: " + e.getMessage());
        }
    }
    
    /**
     * 更新性能指标
     * @param timeSlot 当前时间片
     */
    public void updateMetrics(double timeSlot) {
        
        // 获取当前系统状态
        UAVManager uavManager = simManager.getUAVManager();
        double simulationTime = simManager.getSimulationTime();
        
        // 更新系统运行时间
        updateMetric("system.uptime", simulationTime);
        
        // 计算UAV相关指标
        double totalEnergy = 0.0;
        double totalUtilization = 0.0;
        double totalQueueLength = 0.0;
        double totalCommunicationQuality = 0.0;
        int uavCount = uavManager.getUAVs().size();
        int communicationPairs = 0;
        
        for (UAV uav : uavManager.getUAVs()) {
            totalEnergy += uav.getEnergy();
            totalQueueLength += uav.getTaskQueueLength();
            
            // 计算利用率（处理中的任务占处理能力的比例）
            double utilization = 0.0;
            if (uav.getState() == UAV.UAVState.PROCESSING) {
                utilization = uav.getTaskQueueLength() > 0 ? 1.0 : 0.0;
            }
            totalUtilization += utilization;
            
            // 计算与其他UAV的通信质量
            for (UAV otherUav : uavManager.getUAVs()) {
                if (uav.getId() != otherUav.getId()) {
                    totalCommunicationQuality += uavManager.calculateCommunicationQuality(uav, otherUav);
                    communicationPairs++;
                }
            }
        }
        
        // 更新UAV平均指标
        if (uavCount > 0) {
            updateMetric("uav.averageEnergy", totalEnergy / uavCount);
            updateMetric("uav.averageUtilization", totalUtilization / uavCount);
            updateMetric("uav.averageQueueLength", totalQueueLength / uavCount);
        }
        
        if (communicationPairs > 0) {
            updateMetric("uav.communicationQuality", totalCommunicationQuality / communicationPairs);
        }
        
        // 更新任务完成率和能源效率（这些指标可能需要从其他组件获取）
        // 这里是示例代码，根据实际情况调整
        TaskOffloadingEngine offloadingEngine = (TaskOffloadingEngine) simManager.getTaskOffloadingEngine();
        if (offloadingEngine != null) {
            double completionRate = offloadingEngine.getCompletedTaskCount() / Math.max(1.0, offloadingEngine.getTotalTaskCount());
            updateMetric("system.taskCompletionRate", completionRate);
            
            double averageLatency = offloadingEngine.getTotalLatency() / Math.max(1.0, offloadingEngine.getCompletedTaskCount());
            updateMetric("system.averageLatency", averageLatency);
            
            // 能源效率 = 完成的任务数 / 消耗的总能量
            double initialTotalEnergy = uavCount * simManager.getSimulationSettings().getUAVMaxEnergy();
            double consumedEnergy = initialTotalEnergy - totalEnergy;
            double energyEfficiency = offloadingEngine.getCompletedTaskCount() / Math.max(1.0, consumedEnergy);
            updateMetric("system.energyEfficiency", energyEfficiency);
        }
        
        // 记录当前指标
        logMetrics();
    }
    
    /**
     * 更新单个指标
     * @param key 指标键名
     * @param value 指标值
     */
    private void updateMetric(String key, double value) {
        // 更新当前值
        metrics.put(key, value);
        
        // 添加到历史记录
        Queue<Double> history = metricHistory.get(key);
        history.add(value);
        
        // 保持历史记录大小
        while (history.size() > historySize) {
            history.poll();
        }
    }
    
    /**
     * 记录当前指标
     */
    private void logMetrics() {
        double currentTime = simManager.getSimulationTime();
        // 每隔一定时间记录一次（例如每10秒）
        if (Math.round(currentTime) % 10 == 0) {
            SimLogger.printLine(String.format("===== 性能指标 (时间: %.2f) =====", currentTime));
            
            // 系统指标
            SimLogger.printLine(String.format("系统运行时间: %.2f", metrics.get("system.uptime")));
            SimLogger.printLine(String.format("任务完成率: %.2f%%", metrics.get("system.taskCompletionRate") * 100));
            SimLogger.printLine(String.format("平均延迟: %.2f ms", metrics.get("system.averageLatency")));
            SimLogger.printLine(String.format("能源效率: %.2f 任务/能量单位", metrics.get("system.energyEfficiency")));
            
            // UAV指标
            SimLogger.printLine(String.format("UAV平均能量: %.2f", metrics.get("uav.averageEnergy")));
            SimLogger.printLine(String.format("UAV平均利用率: %.2f%%", metrics.get("uav.averageUtilization") * 100));
            SimLogger.printLine(String.format("UAV平均队列长度: %.2f", metrics.get("uav.averageQueueLength")));
            SimLogger.printLine(String.format("UAV通信质量: %.2f", metrics.get("uav.communicationQuality")));
        }
        
        // 添加CSV记录
        try {
            if (!performanceLogInitialized) {
                // 如果日志还未初始化，先初始化
                initializePerformanceLog();
            }
            
            String csvLine = String.format("%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f",
                currentTime,
                metrics.get("system.taskCompletionRate") * 100,
                metrics.get("system.averageLatency"),
                metrics.get("system.energyEfficiency"),
                metrics.get("uav.averageEnergy"),
                metrics.get("uav.averageUtilization") * 100,
                metrics.get("uav.averageQueueLength"),
                metrics.get("uav.communicationQuality"));
            
            SimLogger.getInstance().writeToPerformanceLogFile(csvLine);
        } catch (Exception e) {
            SimLogger.printLine("写入性能日志时出错: " + e.getMessage());
        }
    }

    /**
     * 获取当前指标值
     * @param key 指标键名
     * @return 指标值
     */
    public double getMetric(String key) {
        return metrics.getOrDefault(key, 0.0);
    }
    
    /**
     * 获取指标历史记录
     * @param key 指标键名
     * @return 历史记录队列
     */
    public Queue<Double> getMetricHistory(String key) {
        return metricHistory.get(key);
    }
    
    /**
     * 获取所有当前指标
     * @return 指标映射
     */
    public Map<String, Double> getAllMetrics() {
        return new HashMap<>(metrics);
    }
}