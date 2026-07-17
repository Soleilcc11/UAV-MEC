package edu.boun.edgecloudsim.utils;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import edu.boun.edgecloudsim.core.SimManager;
import edu.boun.edgecloudsim.core.SimSettings;

/**
 * 模拟日志记录器类
 */
public class SimLogger {
    private static SimLogger instance = null;
    private String outputFolder;
    private Map<Integer, LogItem> taskMap;
    private BufferedWriter taskLogBW;
    private BufferedWriter locationLogBW;
    private BufferedWriter apDelayLogBW;
    private BufferedWriter vmLoadLogBW;
    private int numOfAppTypes;
    private boolean fileLogEnabled;
    private boolean pendingLogs;
    private int iterationNumber;
    private int taskIdCounter;
    private BufferedWriter performanceLogBW = null;
    private String performanceLogFileName = null;
    
    // 各种指标统计
    private int numOfFailedTask;
    private int numOfProcessedTask;
    private int numOfLowLatencyGainedTask;
    private int numOfHighProcessingGainedTask;
    private int numOfFailedTaskDueToBw;
    private int numOfFailedTaskDueToVm;
    private int numOfFailedTaskDueToMobility;
    
    /**
     * 内部类：日志条目
     */
    private class LogItem {
        public double taskStartTime;
        public double taskEndTime;
        public double offloadingTime;
        public double vmAssignTime;
        public double executionStartTime;
        public double executionFinishTime;
        public double returnTime;
        public int taskType;
        public int offloadingDestination;
        public int status;
        public double lanDelay;
        public double manDelay;
        public double wanDelay;
        public double gsmDelay;
        public double bwCost;
        public int hostId;
        public int vmId;
        public int vmType;
        public int taskLength;
        public int requestedCores;
        public double requestedMips;
        public int requestedRam;
        public int requestedStorage;
        
        public LogItem(int _taskType, int _offloadingDestination) {
            taskStartTime = 0;
            taskEndTime = 0;
            offloadingTime = 0;
            vmAssignTime = 0;
            executionStartTime = 0;
            executionFinishTime = 0;
            returnTime = 0;
            taskType = _taskType;
            offloadingDestination = _offloadingDestination;
            status = 0; // Created
            lanDelay = 0;
            manDelay = 0;
            wanDelay = 0;
            gsmDelay = 0;
            bwCost = 0;
            hostId = 0;
            vmId = 0;
            vmType = 0;
            taskLength = 0;
            requestedCores = 0;
            requestedMips = 0;
            requestedRam = 0;
            requestedStorage = 0;
        }
    }


    
    /**
     * 私有构造函数
     */
    private SimLogger() {
        taskMap = new HashMap<Integer, LogItem>();
        pendingLogs = false;
        numOfFailedTask = 0;
        numOfProcessedTask = 0;
        numOfLowLatencyGainedTask = 0;
        numOfHighProcessingGainedTask = 0;
        numOfFailedTaskDueToBw = 0;
        numOfFailedTaskDueToVm = 0;
        numOfFailedTaskDueToMobility = 0;
        taskIdCounter = 0;
    }
    
    /**
     * 获取实例
     */
    public static SimLogger getInstance() {
        if(instance == null) {
            instance = new SimLogger();
        }
        return instance;
    }
    
    /**
     * 设置输出文件夹
     */
    public void setOutputFolder(String _outputFolder) {
        outputFolder = _outputFolder;
        
        File folder = new File(outputFolder);
        if(!folder.exists())
            folder.mkdir();
    }
    
    /**
     * 模拟开始
     */
    public void simStarted(String outFolder, int _iterationNumber) {
        outputFolder = outFolder;
        iterationNumber = _iterationNumber;
        
        SimSettings SS = SimSettings.getInstance();
        numOfAppTypes = SS.getSimulationScenarios().length;
        fileLogEnabled = SS.getFileLoggingEnabled();
        
        taskIdCounter = 0;
        
        if(fileLogEnabled) {
            File folder = new File(outputFolder);
            if(!folder.exists())
                folder.mkdir();
            
            try {
                taskLogBW = new BufferedWriter(new FileWriter(outputFolder + "/task_log.csv", false));
                locationLogBW = new BufferedWriter(new FileWriter(outputFolder + "/location_log.csv", false));
                apDelayLogBW = new BufferedWriter(new FileWriter(outputFolder + "/ap_delay_log.csv", false));
                vmLoadLogBW = new BufferedWriter(new FileWriter(outputFolder + "/vm_load_log.csv", false));
                
                writeHeaders();
            } catch (IOException e) {
                e.printStackTrace();
                System.exit(1);
            }
        }
    }
    
    /**
     * 写入表头
     */
    private void writeHeaders() throws IOException {
        // Task log header
        String header = "TaskID" + SimSettings.DELIMITER;
        header += "TaskType" + SimSettings.DELIMITER;
        header += "Status" + SimSettings.DELIMITER;
        header += "DatacenterId" + SimSettings.DELIMITER;
        header += "HostId" + SimSettings.DELIMITER;
        header += "VmType" + SimSettings.DELIMITER;
        header += "VmId" + SimSettings.DELIMITER;
        header += "TaskLengthType" + SimSettings.DELIMITER;
        header += "TaskStartTime" + SimSettings.DELIMITER;
        header += "TaskEndTime" + SimSettings.DELIMITER;
        header += "NetworkDelay" + SimSettings.DELIMITER;
        header += "ExecutionTime" + SimSettings.DELIMITER;
        header += "ServiceTime";
        
        taskLogBW.write(header);
        taskLogBW.newLine();
        
        // Location log header
        header = "Time" + SimSettings.DELIMITER;
        header += "DeviceID" + SimSettings.DELIMITER;
        header += "X" + SimSettings.DELIMITER;
        header += "Y" + SimSettings.DELIMITER;
        header += "Z";
        
        locationLogBW.write(header);
        locationLogBW.newLine();
        
        // AP delay log header
        header = "Time" + SimSettings.DELIMITER;
        header += "DeviceID" + SimSettings.DELIMITER;
        header += "WLANID" + SimSettings.DELIMITER;
        header += "WLANDELAY";
        
        apDelayLogBW.write(header);
        apDelayLogBW.newLine();
        
        // VM load log header
        header = "Time" + SimSettings.DELIMITER;
        header += "DatacenterId" + SimSettings.DELIMITER;
        header += "HostId" + SimSettings.DELIMITER;
        header += "VmId" + SimSettings.DELIMITER;
        header += "Load";
        
        vmLoadLogBW.write(header);
        vmLoadLogBW.newLine();
    }
    
    /**
     * 模拟停止
     */
    public void simStopped() throws IOException {
        if(fileLogEnabled) {
            taskLogBW.close();
            locationLogBW.close();
            apDelayLogBW.close();
            vmLoadLogBW.close();
        }
    }
    
    /**
     * 添加任务记录
     */
    public int addLog(int taskType, int offloadingDestination) {
        int taskId = taskIdCounter++;
        taskMap.put(taskId, new LogItem(taskType, offloadingDestination));
        
        pendingLogs = true;
        
        return taskId;
    }
    
    /**
     * 将任务完成事件记录到文件
     */
    public synchronized void taskCompleted(String taskId, double taskStartTime, double taskEndTime, double taskLength) {
        // 检查任务ID有效性
        if (taskId == null || taskId.isEmpty()) {
            printLine("[ERROR] 尝试记录无效任务ID: " + taskId);
            return;
        }
        
        printLine("[任务完成] 任务ID:" + taskId + ", 开始时间:" + taskStartTime + 
                  ", 结束时间:" + taskEndTime + ", 处理时间:" + (taskEndTime - taskStartTime) + 
                  ", 任务长度:" + taskLength);
        
        try {
            if(fileLogEnabled) {
                BufferedWriter bw = new BufferedWriter(new FileWriter(outputFolder + "/task_log.csv", true));
                bw.write(taskId + SimSettings.DELIMITER + 
                        taskStartTime + SimSettings.DELIMITER + 
                        taskEndTime + SimSettings.DELIMITER + 
                        (taskEndTime - taskStartTime) + SimSettings.DELIMITER + 
                        taskLength + System.lineSeparator());
                bw.close();
                
                // 调试信息
                printLine("[DEBUG] 已写入任务记录到文件: " + taskId);
            }
        } catch (IOException e) {
            e.printStackTrace();
            printLine("[ERROR] 写入任务日志时发生错误: " + e.getMessage());
        }
    }
    
    /**
     * 日志任务失败
     */
    public void taskFailed(int taskId, int reason) {
        LogItem task = taskMap.get(taskId);
        if(task == null) {
            System.out.println("Cannot find task with ID: " + taskId);
            return;
        }
        
        // Record failure status
        task.status = -1; // Failed
        numOfFailedTask++;
        
        // Record failure reason
        if(reason == 1)
            numOfFailedTaskDueToBw++;
        else if(reason == 2)
            numOfFailedTaskDueToVm++;
        else if(reason == 3)
            numOfFailedTaskDueToMobility++;
    }
    
    /**
     * 日志任务开始时间
     */
    public void setTaskStartTime(int taskId, double time) {
        taskMap.get(taskId).taskStartTime = time;
    }
    
    /**
     * 日志任务结束时间
     */
    public void setTaskEndTime(int taskId, double time) {
        taskMap.get(taskId).taskEndTime = time;
    }
    
    /**
     * 日志卸载时间
     */
    public void setOffloadingStartTime(int taskId, double time) {
        taskMap.get(taskId).offloadingTime = time;
    }
    
    /**
     * 日志VM分配时间
     */
    public void setVmAssignTime(int taskId, double time) {
        taskMap.get(taskId).vmAssignTime = time;
    }
    
    /**
     * 日志执行开始时间
     */
    public void setExecutionStartTime(int taskId, double time) {
        taskMap.get(taskId).executionStartTime = time;
    }
    
    /**
     * 日志执行完成时间
     */
    public void setExecutionFinishTime(int taskId, double time) {
        taskMap.get(taskId).executionFinishTime = time;
    }
    
    /**
     * 日志返回时间
     */
    public void setReturnTime(int taskId, double time) {
        taskMap.get(taskId).returnTime = time;
    }
    
    /**
     * 打开性能日志文件
     */
    public void openPerformanceLogFile() {
        if (performanceLogBW != null)
            return;
            
        try {
            performanceLogFileName = outputFolder + "/performance_log.csv";
            performanceLogBW = new BufferedWriter(new FileWriter(performanceLogFileName, false));
        } catch (IOException e) {
            printLine("打开性能日志文件时出错: " + e.getMessage());
            e.printStackTrace();
        }
    }

    /**
     * 写入性能日志文件
     * @param line 要写入的行
     */
    public void writeToPerformanceLogFile(String line) {
        if (performanceLogBW == null) {
            openPerformanceLogFile();
        }
        
        try {
            performanceLogBW.write(line + "\n");
            performanceLogBW.flush();
        } catch (IOException e) {
            printLine("写入性能日志文件时出错: " + e.getMessage());
            e.printStackTrace();
        }
    }

    /**
     * 关闭性能日志文件
     */
    public void closePerformanceLogFile() {
        try {
            if (performanceLogBW != null) {
                performanceLogBW.close();
                performanceLogBW = null;
            }
        } catch (IOException e) {
            printLine("关闭性能日志文件时出错: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * 日志网络延迟
     */
    public void setNetworkDelay(int taskId, double delay, int delayType) {
        if(delayType == SimSettings.NETWORK_DELAY_TYPES.WLAN_DELAY.ordinal())
            taskMap.get(taskId).lanDelay = delay;
        else if(delayType == SimSettings.NETWORK_DELAY_TYPES.MAN_DELAY.ordinal())
            taskMap.get(taskId).manDelay = delay;
        else if(delayType == SimSettings.NETWORK_DELAY_TYPES.WAN_DELAY.ordinal())
            taskMap.get(taskId).wanDelay = delay;
        else if(delayType == SimSettings.NETWORK_DELAY_TYPES.GSM_DELAY.ordinal())
            taskMap.get(taskId).gsmDelay = delay;
    }
    
    /**
     * 日志带宽成本
     */
    public void setBandwidthCost(int taskId, double cost) {
        taskMap.get(taskId).bwCost = cost;
    }
    
    /**
     * 日志主机ID
     */
    public void setHostId(int taskId, int hostId) {
        taskMap.get(taskId).hostId = hostId;
    }
    
    /**
     * 日志VM信息
     */
    public void setVmId(int taskId, int vmId, int vmType) {
        taskMap.get(taskId).vmId = vmId;
        taskMap.get(taskId).vmType = vmType;
    }
    
    /**
     * 日志任务属性
     */
    public void setTaskProperty(int taskId, int taskLength, int requestedCores, double requestedMips, int requestedRam, int requestedStorage) {
        taskMap.get(taskId).taskLength = taskLength;
        taskMap.get(taskId).requestedCores = requestedCores;
        taskMap.get(taskId).requestedMips = requestedMips;
        taskMap.get(taskId).requestedRam = requestedRam;
        taskMap.get(taskId).requestedStorage = requestedStorage;
    }
    
    /**
     * 日志位置信息
     */
    public void saveLocation(double time, int deviceId, Location loc) {
        if(fileLogEnabled && SimManager.getInstance().getSimulationTime() > SimSettings.getInstance().getLocationLogInterval()) {
            try {
                locationLogBW.write(time + SimSettings.DELIMITER + deviceId + SimSettings.DELIMITER + loc.getXPos() 
                        + SimSettings.DELIMITER + loc.getYPos() + SimSettings.DELIMITER + loc.getZPos());
                locationLogBW.newLine();
            } catch (IOException e) {
                e.printStackTrace();
                System.exit(1);
            }
        }
    }
    
    /**
     * 日志AP延迟
     */
    public void saveApDelay(double time, int deviceId, int wlanId, double apDelay) {
        if(fileLogEnabled && SimManager.getInstance().getSimulationTime() > SimSettings.getInstance().getApDelayLogInterval()) {
            try {
                apDelayLogBW.write(time + SimSettings.DELIMITER + deviceId + SimSettings.DELIMITER + wlanId + SimSettings.DELIMITER + apDelay);
                apDelayLogBW.newLine();
            } catch (IOException e) {
                e.printStackTrace();
                System.exit(1);
            }
        }
    }
    
    /**
     * 日志VM负载
     */
    public void saveVmLoad(double time, int datacenterId, int hostId, int vmId, double load) {
        if(fileLogEnabled && SimManager.getInstance().getSimulationTime() > SimSettings.getInstance().getVmLoadLogInterval()) {
            try {
                vmLoadLogBW.write(time + SimSettings.DELIMITER + datacenterId + SimSettings.DELIMITER + hostId + SimSettings.DELIMITER + vmId + SimSettings.DELIMITER + load);
                vmLoadLogBW.newLine();
            } catch (IOException e) {
                e.printStackTrace();
                System.exit(1);
            }
        }
    }
    
    /**
     * 获取失败任务数
     */
    public int getFailedTaskCount() {
        return numOfFailedTask;
    }
    
    /**
     * 获取处理任务数
     */
    public int getProcessedTaskCount() {
        return numOfProcessedTask;
    }
    
    /**
     * 获取低延迟任务数
     */
    public int getLowLatencyGainedTask() {
        return numOfLowLatencyGainedTask;
    }
    
    /**
     * 获取高处理能力任务数
     */
    public int getHighProcessingGainedTask() {
        return numOfHighProcessingGainedTask;
    }
    
    /**
     * 获取由于带宽失败的任务数
     */
    public int getFailedTaskCountDueToBw() {
        return numOfFailedTaskDueToBw;
    }
    
    /**
     * 获取由于VM失败的任务数
     */
    public int getFailedTaskCountDueToVm() {
        return numOfFailedTaskDueToVm;
    }
    
    /**
     * 获取由于移动性失败的任务数
     */
    public int getFailedTaskCountDueToMobility() {
        return numOfFailedTaskDueToMobility;
    }
    
    /**
     * 打印行到控制台
     */
    public static void printLine(String msg) {
        System.out.println(msg);
    }
    
    /**
     * 打印到控制台（无换行）
     */
    public static void print(String msg) {
        System.out.print(msg);
    }
    
    /**
     * 启用控制台日志打印
     */
    public static void enablePrintLog() {
		getInstance().enablePrintLog0();
	}
	private void enablePrintLog0() {
	}

    public static void enableFileLog() {
		getInstance().enableFileLog0();
	}
	
	private void enableFileLog0() {
		// 原方法实现
	}
	
	public static void disableFileLog() {
		getInstance().disableFileLog0();
	}
	
	private void disableFileLog0() {
		// 原方法实现
	}

    /**
     * 记录任务执行
     */
    public void taskExecuted(int taskId) {
        // 任务执行记录的实现
        LogItem task = taskMap.get(taskId);
        if (task != null) {
            // 标记任务已执行
        }
    }

    /**
     * 记录任务开始
     */
    public void taskStarted(int taskId, double time) {
        setTaskStartTime(taskId, time);
    }

    /**
     * 记录任务结束
     */
    public void taskEnded(int taskId, double time) {
        setTaskEndTime(taskId, time);
    }

    /**
     * 设置上传延迟
     */
    public void setUploadDelay(int taskId, double delay, SimSettings.NETWORK_DELAY_TYPES delayType) {
        setNetworkDelay(taskId, delay, delayType.ordinal());
    }

    /**
     * 设置下载延迟
     */
    public void setDownloadDelay(int taskId, double delay, SimSettings.NETWORK_DELAY_TYPES delayType) {
        setNetworkDelay(taskId, delay, delayType.ordinal());
    }

    /**
     * 记录任务因移动性失败
     */
    public void failedDueToMobility(int taskId, double time) {
        taskFailed(taskId, 3); // 3表示移动性失败
        setTaskEndTime(taskId, time);
    }

    /**
     * 记录任务因带宽失败
     */
    public void failedDueToBandwidth(int taskId, double delay, SimSettings.NETWORK_DELAY_TYPES delayType) {
        taskFailed(taskId, 1); // 1表示带宽失败
        setNetworkDelay(taskId, delay, delayType.ordinal());
    }

    /**
     * 记录任务因带宽被拒绝
     */
    public void rejectedDueToBandwidth(int taskId, double time, int hostId, SimSettings.NETWORK_DELAY_TYPES delayType) {
        taskFailed(taskId, 1); // 1表示带宽失败
        setTaskEndTime(taskId, time);
        setHostId(taskId, hostId);
    }

    /**
     * 记录任务因VM容量被拒绝
     */
    public void rejectedDueToVMCapacity(int taskId, double time, int hostId) {
        taskFailed(taskId, 2); // 2表示VM容量失败
        setTaskEndTime(taskId, time);
        setHostId(taskId, hostId);
    }

    /**
     * 记录任务分配
     */
    public void taskAssigned(int taskId, int datacenterId, int hostId, int vmId, int vmType) {
        LogItem task = taskMap.get(taskId);
        if (task != null) {
            task.offloadingDestination = datacenterId;
            setHostId(taskId, hostId);
            setVmId(taskId, vmId, vmType);
        }
    }

    /**
     * 添加扩展日志（支持多参数）
     */
    public int addLog(int mobileDeviceId, int taskId, int taskType, int taskLength,
            int taskInputSize, int taskOutputSize) {
        taskMap.put(taskId, new LogItem(taskType, 0));
        setTaskProperty(taskId, taskLength, 0, 0.0, 0, 0);
        pendingLogs = true;
        return taskId;
    }
	public void addApDelayLog(double time, double[] source, double[] destination) {
		// 实现AP延迟日志记录
	}
	
	public void setQoE(int taskId, double qoe) {
		// 实现QoE设置
	}
	
	public void setOrchestratorOverhead(int taskId, long overhead) {
		// 实现编排器开销设置
	}
	
	public String getOutputFolder() {
		return outputFolder;
	}
	
}
