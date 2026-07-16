package edu.boun.edgecloudsim.task_generator;
import edu.boun.edgecloudsim.utils.ArrayUtils;
import java.util.ArrayList;

import org.apache.commons.math3.distribution.ExponentialDistribution;

import edu.boun.edgecloudsim.core.SimSettings;
import edu.boun.edgecloudsim.utils.TaskProperty;
import edu.boun.edgecloudsim.utils.SimLogger;
import edu.boun.edgecloudsim.utils.SimUtils;

public class IdleActiveLoadGenerator extends LoadGeneratorModel {
    int taskTypeOfDevices[];
    
    public IdleActiveLoadGenerator(int _numberOfMobileDevices, double _simulationTime, String _simScenario) {
        super(_numberOfMobileDevices, _simulationTime, _simScenario);
    }

    @Override
    public void initializeModel() {
        taskList = new ArrayList<TaskProperty>();
        
        // 创建随机数生成器，用于任务输入大小、输出大小和任务长度
        int taskLookUpTableLength = ArrayUtils.length(SimSettings.getInstance().getTaskLookUpTable());
        ExponentialDistribution[][] expRngList = new ExponentialDistribution[taskLookUpTableLength][3];
        
        // 为每种任务类型创建随机数生成器
        for(int i=0; i<taskLookUpTableLength; i++) {
            if(ArrayUtils.getDoubleValueFromTable(SimSettings.getInstance().getTaskLookUpTable(), i, 0) == 0)
                continue;
            
            // 为任务输入大小、输出大小和长度创建指数分布
            expRngList[i][0] = new ExponentialDistribution(ArrayUtils.getDoubleValueFromTable(SimSettings.getInstance().getTaskLookUpTable(), i, 5));
            expRngList[i][1] = new ExponentialDistribution(ArrayUtils.getDoubleValueFromTable(SimSettings.getInstance().getTaskLookUpTable(), i, 6));
            expRngList[i][2] = new ExponentialDistribution(ArrayUtils.getDoubleValueFromTable(SimSettings.getInstance().getTaskLookUpTable(), i, 7));
        }
        
        // 为每个移动设备分配任务类型
        taskTypeOfDevices = new int[numberOfMobileDevices];
        for(int i=0; i<numberOfMobileDevices; i++) {
            int randomTaskType = -1;
            double taskTypeSelector = SimUtils.getRandomDoubleNumber(0,100);
            double taskTypePercentage = 0;
            
            // 根据任务类型使用百分比选择随机任务类型
            for (int j=0; j<taskLookUpTableLength; j++) {
                taskTypePercentage += ArrayUtils.getDoubleValueFromTable(SimSettings.getInstance().getTaskLookUpTable(), j, 0);
                if(taskTypeSelector <= taskTypePercentage){
                    randomTaskType = j;
                    break;
                }
            }
            
            if(randomTaskType == -1){
                SimLogger.printLine("任务类型选择错误！无法获取随机任务类型！");
                continue;
            }
            
            taskTypeOfDevices[i] = randomTaskType;
            
            // 获取任务生成参数
            double poissonMean = ArrayUtils.getDoubleValueFromTable(SimSettings.getInstance().getTaskLookUpTable(), randomTaskType, 2);
            double activePeriod = ArrayUtils.getDoubleValueFromTable(SimSettings.getInstance().getTaskLookUpTable(), randomTaskType, 3);
            double idlePeriod = ArrayUtils.getDoubleValueFromTable(SimSettings.getInstance().getTaskLookUpTable(), randomTaskType, 4);
            double activePeriodStartTime = SimUtils.getRandomDoubleNumber(
                    SimSettings.CLIENT_ACTIVITY_START_TIME, 
                    SimSettings.CLIENT_ACTIVITY_START_TIME + activePeriod);
                    
            double virtualTime = activePeriodStartTime;
            int deviceTasks = 0; // 该设备生成的任务数量计数器
            
            // 使用泊松分布生成任务
            ExponentialDistribution rng = new ExponentialDistribution(poissonMean);
            
            // 在模拟时间内生成任务
            while(virtualTime < simulationTime) {
                double interval = rng.sample();
                
                if(interval <= 0){
                    SimLogger.printLine("生成任务间隔时间错误！间隔时间为 " + interval + "，设备 " + i + "，时间 " + virtualTime);
                    continue;
                }

                virtualTime += interval;

                // The sample is a delay from the previous arrival. Do not enqueue
                // an arrival that falls outside the configured simulation window.
                if (virtualTime >= simulationTime) {
                    break;
                }

                // 检查是否超出活动期
                if(virtualTime > activePeriodStartTime + activePeriod){
                    activePeriodStartTime = activePeriodStartTime + activePeriod + idlePeriod;
                    virtualTime = activePeriodStartTime;
                    continue;
                }
                
                // 创建任务并添加到列表
                taskList.add(new TaskProperty(i, randomTaskType, virtualTime, expRngList));
                deviceTasks++;
            }
            
            // 为调试添加日志
            System.out.println("为设备" + i + "生成了" + deviceTasks + "个任务");
        }
        
        // 确认任务数量
        System.out.println("总共生成了" + taskList.size() + "个任务");
        
        // 如果没有生成任务，添加警告日志
        if (taskList.isEmpty()) {
            SimLogger.printLine("警告：未生成任何任务！请检查配置参数！");
            System.out.println("警告：未生成任何任务！请检查以下参数：");
            System.out.println("- 模拟时间: " + simulationTime);
            System.out.println("- 移动设备数量: " + numberOfMobileDevices);
            System.out.println("- 任务参数表大小: " + taskLookUpTableLength);
            System.out.println("- 任务生成间隔: 请检查applications.xml中的poisson_interarrival值");
            System.out.println("- 活动期/空闲期: 请检查applications.xml中的active_period和idle_period值");
        }
    }

    @Override
    public int getTaskTypeOfDevice(int deviceId) {
        return taskTypeOfDevices[deviceId];
    }
}
