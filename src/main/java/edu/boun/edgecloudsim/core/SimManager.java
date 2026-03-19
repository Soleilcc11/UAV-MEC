package edu.boun.edgecloudsim.core;

import org.cloudbus.cloudsim.core.CloudSim;
import org.cloudbus.cloudsim.core.SimEntity;
import org.cloudbus.cloudsim.core.SimEvent;

import edu.boun.edgecloudsim.cloud_server.CloudServerManager;
import edu.boun.edgecloudsim.edge_client.MobileDeviceManager;
import edu.boun.edgecloudsim.edge_client.mobile_processing_unit.MobileServerManager;
import edu.boun.edgecloudsim.edge_orchestrator.EdgeOrchestrator;
import edu.boun.edgecloudsim.edge_server.EdgeServerManager;
import edu.boun.edgecloudsim.mobility.MobilityModel;
import edu.boun.edgecloudsim.network.NetworkModel;
import edu.boun.edgecloudsim.uav.UAVManager;
import edu.boun.edgecloudsim.uav.PerformanceMonitor;
import edu.boun.edgecloudsim.uav.TaskOffloadingEngine;
import edu.boun.edgecloudsim.uav.PeriodicEvent;
import edu.boun.edgecloudsim.utils.SimLogger;

/**
 * 模拟管理器扩展，增加UAV管理器支持
 */
public class SimManager extends SimEntity {
    private static SimManager instance = null;
    private SimSettings simSettings;
    private UAVManager uavManager;
    private TaskOffloadingEngine taskOffloadingEngine;
    private PerformanceMonitor performanceMonitor;
    
    // 添加缺失的核心组件
    private ScenarioFactory scenarioFactory;
    private MobilityModel mobilityModel;
    private NetworkModel networkModel;
    private EdgeOrchestrator orchestrator;
    private EdgeServerManager edgeServerManager;
    private CloudServerManager cloudServerManager;
    private MobileServerManager mobileServerManager;
    private MobileDeviceManager mobileDeviceManager;
    
    // 模拟配置参数
    private int numOfMobileDevice;
    private String simScenario;
    private String orchestratorPolicy;
    
    // 事件标识符常量
    public static final int BASE_EVENT_ID = 5000;
    public static final int PERIODIC_EVENT = BASE_EVENT_ID + 1;
    public static final int PROCESS_UAV_TASKS = BASE_EVENT_ID + 2;
    
    /**
     * 默认构造函数 - 私有化以实现单例模式
     */
    private SimManager() {
        super("SimManager");
        // 默认构造函数，不执行任何初始化
    }
    
    /**
     * 新增构造函数 - 支持带参数的初始化
     * 修复构造函数不匹配的问题
     */
    public SimManager(ScenarioFactory _scenarioFactory, int _numOfMobileDevice, 
                     String _simScenario, String _orchestratorPolicy) {
        super("SimManager");
        scenarioFactory = _scenarioFactory;
        numOfMobileDevice = _numOfMobileDevice;
        simScenario = _simScenario;
        orchestratorPolicy = _orchestratorPolicy;
        
        simSettings = SimSettings.getInstance();
        
        // 初始化核心组件
        try {
            mobilityModel = scenarioFactory.getMobilityModel();
            networkModel = scenarioFactory.getNetworkModel();
            orchestrator = scenarioFactory.getEdgeOrchestrator();
            edgeServerManager = scenarioFactory.getEdgeServerManager();
            cloudServerManager = scenarioFactory.getCloudServerManager();
            mobileServerManager = scenarioFactory.getMobileServerManager();
            mobileDeviceManager = scenarioFactory.getMobileDeviceManager();
            
            // 如果ScenarioFactory是UAVMECScenarioFactory，获取UAVManager
            if (scenarioFactory instanceof edu.boun.edgecloudsim.uav.UAVMECScenarioFactory) {
                uavManager = ((edu.boun.edgecloudsim.uav.UAVMECScenarioFactory) scenarioFactory).getUAVManager();
            }
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }
        
        // 更新单例实例
        instance = this;
    }
    
    // 获取实例（单例模式）
    public static SimManager getInstance(){
        if(instance == null){
            instance = new SimManager();
        }
        return instance;
    }
    
    /**
     * 初始化模拟管理器(使用单个配置对象)
     */
    public void initialize(Object config) {
        // 初始化代码...
        simSettings = SimSettings.getInstance();
    }
    
    /**
     * 初始化模拟管理器(使用多个参数 - 添加此方法满足API要求)
     */
    public void initialize(ScenarioFactory _scenarioFactory, int _numOfMobileDevice, 
                           String _simScenario, String _orchestratorPolicy) {
        scenarioFactory = _scenarioFactory;
        numOfMobileDevice = _numOfMobileDevice;
        simScenario = _simScenario;
        orchestratorPolicy = _orchestratorPolicy;
        
        simSettings = SimSettings.getInstance();
        
        // 初始化核心组件
        try {
            mobilityModel = scenarioFactory.getMobilityModel();
            networkModel = scenarioFactory.getNetworkModel();
            orchestrator = scenarioFactory.getEdgeOrchestrator();
            edgeServerManager = scenarioFactory.getEdgeServerManager();
            cloudServerManager = scenarioFactory.getCloudServerManager();
            mobileServerManager = scenarioFactory.getMobileServerManager();
            mobileDeviceManager = scenarioFactory.getMobileDeviceManager();
            
            // 如果ScenarioFactory是UAVMECScenarioFactory，获取UAVManager
            if (scenarioFactory instanceof edu.boun.edgecloudsim.uav.UAVMECScenarioFactory) {
                uavManager = ((edu.boun.edgecloudsim.uav.UAVMECScenarioFactory) scenarioFactory).getUAVManager();
            }
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }
    }
    
    /**
     * 处理事件方法 - 添加事件处理逻辑
     */
    @Override
    public void processEvent(SimEvent ev) {
        if (ev == null) {
            SimLogger.printLine("警告: SimManager收到空事件!");
            return;
        }
        
        switch (ev.getTag()) {
            case PERIODIC_EVENT:
                // 处理周期性事件
                if (ev.getData() instanceof PeriodicEvent) {
                    PeriodicEvent periodicEvent = (PeriodicEvent) ev.getData();
                    SimLogger.printLine("SimManager处理周期性事件，时间: " + CloudSim.clock());
                    periodicEvent.processEvent();
                    
                    // 安排下一次更新
                    schedule(getId(), periodicEvent.getInterval(), PERIODIC_EVENT, periodicEvent);
                }
                break;
                
            case PROCESS_UAV_TASKS:
                // 处理UAV任务
                if (uavManager != null) {
                    int processedTasks = uavManager.processTasks(1.0); // 处理1秒的任务
                    SimLogger.printLine("处理UAV任务，完成数量: " + processedTasks + ", 时间: " + CloudSim.clock());
                    
                    // 安排下一次任务处理
                    schedule(getId(), 1.0, PROCESS_UAV_TASKS);
                }
                break;
                
            default:
                SimLogger.printLine("SimManager收到未知事件，标签: " + ev.getTag());
                break;
        }
    }
    
    /**
     * 启动实体方法 - 初始化组件并安排首个事件
     */
    @Override
    public void startEntity() {
        SimLogger.printLine("SimManager启动...");
        
        // 初始化组件
        if (mobilityModel != null) mobilityModel.initialize();
        if (networkModel != null) networkModel.initialize();
        if (orchestrator != null) orchestrator.initialize();
        if (uavManager != null) uavManager.initialize();
        if (taskOffloadingEngine != null) taskOffloadingEngine.initialize();
        
        // 安排首个任务处理事件
        schedule(getId(), 1.0, PROCESS_UAV_TASKS);
        
        SimLogger.printLine("SimManager初始化完成，安排了任务处理事件");
    }
    
    /**
     * 关闭实体方法
     */
    @Override
    public void shutdownEntity() {
        SimLogger.printLine("SimManager关闭...");
    }
    
    /**
     * 启动模拟
     */
    public void startSimulation() {
        // 模拟启动逻辑
        CloudSim.startSimulation();
    }
    
    // ... [其余方法保持不变] ...
    
    /**
     * 获取移动模型
     */
    public MobilityModel getMobilityModel() {
        return mobilityModel;
    }
    
    /**
     * 获取网络模型
     */
    public NetworkModel getNetworkModel() {
        return networkModel;
    }
    
    /**
     * 获取边缘编排器
     */
    public EdgeOrchestrator getEdgeOrchestrator() {
        return orchestrator;
    }
    
    /**
     * 获取边缘服务器管理器
     */
    public EdgeServerManager getEdgeServerManager() {
        return edgeServerManager;
    }
    
    /**
     * 获取云服务器管理器
     */
    public CloudServerManager getCloudServerManager() {
        return cloudServerManager;
    }
    
    /**
     * 获取移动服务器管理器
     */
    public MobileServerManager getMobileServerManager() {
        return mobileServerManager;
    }
    
    /**
     * 获取移动设备管理器
     */
    public MobileDeviceManager getMobileDeviceManager() {
        return mobileDeviceManager;
    }
    
    /**
     * 获取UAV管理器
     */
    public UAVManager getUAVManager() {
        return uavManager;
    }
    
    /**
     * 设置UAV管理器
     */
    public void setUAVManager(UAVManager uavManager) {
        this.uavManager = uavManager;
    }
    
    /**
     * 获取任务卸载引擎
     */
    public TaskOffloadingEngine getTaskOffloadingEngine() {
        return taskOffloadingEngine;
    }
    
    /**
     * 设置任务卸载引擎
     */
    public void setTaskOffloadingEngine(TaskOffloadingEngine engine) {
        this.taskOffloadingEngine = engine;
    }
    
    /**
     * 设置性能监控器
     */
    public void setPerformanceMonitor(PerformanceMonitor monitor) {
        this.performanceMonitor = monitor;
    }
    
    /**
     * 获取性能监控器
     */
    public PerformanceMonitor getPerformanceMonitor() {
        return performanceMonitor;
    }
    
    /**
     * 获取模拟设置
     */
    public SimSettings getSimulationSettings() {
        return simSettings;
    }
    
    /**
     * 获取模拟场景
     */
    public String getSimulationScenario() {
        return simScenario;
    }

    /**
     * 获取移动设备数量
     */
    public int getNumOfMobileDevice() {
        return numOfMobileDevice;
    }
    
    /**
     * 获取当前模拟时间
     */
    public double getSimulationTime() {
        return CloudSim.clock();
    }
    
    /**
     * 处理模拟结果
     */
    public void processResults() {
        // 收集并处理结果
    }
}