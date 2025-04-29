package edu.boun.edgecloudsim.applications.sample_app1; // 路径根据实际情况修改

import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.Date;

import org.cloudbus.cloudsim.Log;
import org.cloudbus.cloudsim.core.CloudSim;

import edu.boun.edgecloudsim.core.ScenarioFactory;
import edu.boun.edgecloudsim.core.SimManager;
import edu.boun.edgecloudsim.core.SimSettings;
import edu.boun.edgecloudsim.utils.SimLogger;
import edu.boun.edgecloudsim.utils.SimUtils;

public class MainApp {
    
    /**
     * 创建场景工厂 - 修改为使用适当的UA VMECScenarioFactory
     */
    public static ScenarioFactory createScenarioFactory(int numOfMobileDevice, String simScenario, String orchestratorPolicy) {
        return new edu.boun.edgecloudsim.uav.UAVMECScenarioFactory(numOfMobileDevice, simScenario, orchestratorPolicy);
    }

    /**
     * 主方法入口点
     */
    public static void main(String[] args) {
        // 解析命令行参数
        SimSettings.getInstance().setSimulationParameters();
        
        // 禁止输出时间戳
        boolean experimentalTimeStamp = false;
        
        // 用于保存配置和结果的目录
        String configFile = "";
        String outputFolder = "";
        String edgeDevicesFile = "";
        String applicationsFile = "";
        
        // 默认参数
        int numOfMobileDevice = 100;
        String orchestratorPolicy = "RANDOM";
        String simScenario = "SINGLE_TIER";
        
        // 解析命令行参数，获取配置
        for (int i = 0; i < args.length; i++) {
            if (args[i].equals("-c")) {
                configFile = args[++i];
            } else if (args[i].equals("-o")) {
                outputFolder = args[++i];
            } else if (args[i].equals("-a")) {
                applicationsFile = args[++i];
            } else if (args[i].equals("-d")) {
                edgeDevicesFile = args[++i];
            } else if (args[i].equals("-n")) {
                numOfMobileDevice = Integer.parseInt(args[++i]);
            } else if (args[i].equals("-s")) {
                simScenario = args[++i];
            } else if (args[i].equals("-p")) {
                orchestratorPolicy = args[++i];
            }
        }
        
        // 加载配置文件
        if (configFile.isEmpty()) {
            System.out.println("缺少配置文件");
            System.exit(0);
        }
        if (edgeDevicesFile.isEmpty()) {
            System.out.println("缺少边缘设备文件");
            System.exit(0);
        }
        if (applicationsFile.isEmpty()) {
            System.out.println("缺少应用程序文件");
            System.exit(0);
        }
        
        // 加载边缘设备和应用程序配置
        SimSettings.getInstance().initialize(configFile, edgeDevicesFile, applicationsFile);
        
        // 记录配置参数
        System.out.println("模拟开始时间: " + SimSettings.getInstance().getSimulationTime());
        System.out.println("移动设备数量: " + numOfMobileDevice);
        System.out.println("模拟场景: " + simScenario);
        System.out.println("编排策略: " + orchestratorPolicy);
        
        // 准备输出文件夹
        if (outputFolder.isEmpty())
            outputFolder = SimUtils.getOutputFolder();
        SimLogger.getInstance().setOutputFolder(outputFolder);
        
        // 创建日志文件
        DateFormat df = new SimpleDateFormat("dd/MM/yyyy HH:mm:ss");
        Date simulationStartDate = Calendar.getInstance().getTime();
        String simulationStartTime = df.format(simulationStartDate);
        
        SimLogger.getInstance().simStarted(outputFolder, orchestratorPolicy, simScenario, numOfMobileDevice);
        SimLogger.getInstance().printLine("Simulation started at " + simulationStartTime);
        SimLogger.getInstance().printLine("----------------------------------------------------------------------");
        
        // 初始化 CloudSim 库
        int num_user = 2;
        Calendar calendar = Calendar.getInstance();
        boolean trace_flag = false;
        CloudSim.init(num_user, calendar, trace_flag, 0.01);
        
        // 创建并初始化场景工厂
        ScenarioFactory factory = createScenarioFactory(numOfMobileDevice, simScenario, orchestratorPolicy);
        
        // 创建SimManager实例并初始化
        SimManager manager = SimManager.getInstance();
        manager.initialize(factory, numOfMobileDevice, simScenario, orchestratorPolicy);
        
        // 开始模拟
        manager.startSimulation();
        
        // 处理结果
        Date simulationEndDate = Calendar.getInstance().getTime();
        String simulationEndTime = df.format(simulationEndDate);
        SimLogger.getInstance().simStopped(simulationEndTime);
    }
}