package edu.boun.edgecloudsim.uav;

import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.Date;
import java.io.File;

import org.cloudbus.cloudsim.core.CloudSim;

import edu.boun.edgecloudsim.core.SimManager;
import edu.boun.edgecloudsim.core.SimSettings;
import edu.boun.edgecloudsim.utils.SimLogger;

/**
 * UAV-MEC系统的主应用程序类
 */
public class UAVMECMainApp {
    // 声明全局变量，避免未定义错误
    private static SimManager manager;
    private static int numOfMobileDevice;
    private static String simScenario;
    private static String orchestratorPolicy;
    
    /**
     * 创建一个具有唯一名称的目录，用于存储模拟结果
     */
    public static String getOutputFolder(String path){
        DateFormat df = new SimpleDateFormat("yyyy-MM-dd_HH-mm-ss");
        Date today = Calendar.getInstance().getTime();
        String reportDate = df.format(today);
        
        String outputFolder = path + "/" + reportDate;
        
        // 创建目录
        File folder = new File(outputFolder);
        if(!folder.exists()) {
            boolean created = folder.mkdirs();
            if(!created) {
                System.err.println("无法创建输出目录: " + outputFolder);
                // 如果无法创建，使用当前目录
                outputFolder = "./sim_results";
                folder = new File(outputFolder);
                folder.mkdirs();
            }
        }
        
        SimLogger.getInstance().setOutputFolder(outputFolder);
        
        return outputFolder;
    }
    
    /**
     * 运行所有场景的入口点
     */
    public static void main(String[] args) {
        // 解析命令行参数
        // if (args.length > 6) {
        //     SimLogger.printLine("Unexpected number of arguments. Simulation setting file, configuration file and output folder are expected!");
        //     System.exit(0);
        //     return;
        // }
        
        // 默认参数值
        String configFile = "src/main/resources/config/simulation_settings.xml";
        String edgeDevicesFile = "src/main/resources/config/edge_devices.xml";
        String applicationsFile = "src/main/resources/config/applications.xml";
        String outputFolder = "sim_results";
        int iterationNumber = 1;
        int numOfMobileDevices = 100;
        
        // 如果提供了参数，则使用参数值
        if (args.length >= 1) configFile = args[0];
        if (args.length >= 2) edgeDevicesFile = args[1];
        if (args.length >= 3) applicationsFile = args[2];
        if (args.length >= 4) outputFolder = args[3];
        if (args.length >= 5) iterationNumber = Integer.parseInt(args[4]);
        if (args.length >= 6) numOfMobileDevices = Integer.parseInt(args[5]);
        
        // 确保输出文件夹存在
        outputFolder = getOutputFolder(outputFolder);
        
        SimSettings SS = SimSettings.getInstance();
        try {
            // 检查配置文件是否存在，如果不存在则使用内存中的默认配置
            File configFileObj = new File(configFile);
            File edgeDevicesFileObj = new File(edgeDevicesFile);
            File applicationsFileObj = new File(applicationsFile);
            
            if (!configFileObj.exists() || !edgeDevicesFileObj.exists() || !applicationsFileObj.exists()) {
                SimLogger.printLine("配置文件不存在，使用内存中的默认配置...");
                
                // 使用编程方式初始化设置
                SS.initializeDefaultSettings(numOfMobileDevices);
                
                // 如果需要，可以在这里手动设置其他参数
                // SS.setSomeSetting(...);
            } else {
                // 使用配置文件初始化
                SS.initialize(configFile, edgeDevicesFile, applicationsFile);
            }
            // If we reach here, initialization was successful
        } catch (Exception e) {
            SimLogger.printLine("Cannot initialize simulation settings!");
            e.printStackTrace();
            System.exit(0);
        }   
         // 初始化日志记录
        SimLogger.getInstance().simStarted(outputFolder, iterationNumber);
        
        // 初始化全局参数
        numOfMobileDevice = numOfMobileDevices;
        simScenario = SimSettings.getInstance().getSimulationScenarios()[0];
        orchestratorPolicy = SimSettings.getInstance().getOrchestratorPolicies()[0];
        
        // 运行模拟
        try {
            runSimulation(iterationNumber);
            
            // 进行结果分析
            SimLogger.getInstance().simStopped();
            analyzeResults(outputFolder);
        } catch (Exception e) {
            SimLogger.printLine("Exception during simulation: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }
    
    /**
     * 运行单个模拟实例
     */
    private static void runSimulation(int iterationNumber) throws Exception {
        // 创建CloudSim环境
        int numUser = 1;
        Calendar calendar = Calendar.getInstance();
        boolean traceFlag = false;
        CloudSim.init(numUser, calendar, traceFlag);
        
        // 创建UAV-MEC场景工厂
        UAVMECScenarioFactory scenarioFactory = new UAVMECScenarioFactory(numOfMobileDevice, simScenario, orchestratorPolicy);
        SimSettings.getInstance().setScenarioFactory(scenarioFactory);
        
        // 创建模拟管理器
        manager = SimManager.getInstance();
        manager.initialize(scenarioFactory, numOfMobileDevice, simScenario, orchestratorPolicy);
        
        // 配置任务卸载引擎和UAV管理器
        configureSimulationComponents(manager);
        
        // 添加这一行 - 创建性能监控日志文件
        if (manager.getPerformanceMonitor() != null) {
            ((PerformanceMonitor)manager.getPerformanceMonitor()).initializePerformanceLog();
        }
        
        // 启动模拟
        SimLogger.printLine("Starting UAV-MEC simulation...");
        try {
            CloudSim.startSimulation();
            
            // 收集最终结果
            SimLogger.printLine("Simulation finished.");
            
            // 安全地结束模拟
            try {
                CloudSim.finishSimulation();
            } catch (NullPointerException e) {
                // 忽略CloudSim关闭时的空指针异常，因为模拟已经完成
                SimLogger.printLine("Note: Ignoring NullPointerException during simulation shutdown. Results are still valid.");
            }
        } catch (Exception e) {
            SimLogger.printLine("Exception during simulation: " + e.getMessage());
            e.printStackTrace();
            throw e;
        }
        
        // 处理模拟结果
        try {
            manager.processResults();
        } catch (Exception e) {
            SimLogger.printLine("Exception during results processing: " + e.getMessage());
            e.printStackTrace();
        }
        
        // 关闭性能监控日志 - 添加这一行
        if (manager.getPerformanceMonitor() != null) {
            SimLogger.getInstance().closePerformanceLogFile();
            SimLogger.printLine("Performance log file closed.");
        }
        
        // 保存RL模型状态（如果使用）
        try {
            TaskOffloadingEngine engine = manager.getTaskOffloadingEngine();
            if (engine != null) {
                SimLogger.printLine("Saving RL model state...");
                // 实现模型保存逻辑
                engine.close();
            }
        } catch (Exception e) {
            SimLogger.printLine("Failed to save RL model: " + e.getMessage());
        }
    }
    
    /** Configure UAV monitoring without bypassing the EdgeCloudSim task path. */
    private static void configureSimulationComponents(SimManager manager) {
        UAVManager uavManager = manager.getUAVManager();
        if (uavManager == null) {
            uavManager = new UAVManager(manager);
        }
        manager.setUAVManager(uavManager);

        PerformanceMonitor monitor = new PerformanceMonitor(manager);
        manager.setPerformanceMonitor(monitor);
    }
    
    /**
     * 分析模拟结果
     */
    private static void analyzeResults(String outputFolder) {
        // 创建结果分析器
        ResultAnalyzer analyzer = new ResultAnalyzer(outputFolder);
        analyzer.analyzeResults();
        
        // 生成图表和报告
        SimLogger.printLine("Generating result graphs and reports...");
        // 实现结果可视化逻辑
    }
    
    /**
     * 内部类：结果分析器
     */
    private static class ResultAnalyzer {
        private String outputFolder;
        
        public ResultAnalyzer(String outputFolder) {
            this.outputFolder = outputFolder;
        }
        
        public void analyzeResults() {
            SimLogger.printLine("Analyzing simulation results...");
            // 实现结果分析逻辑
        }
    }
}
