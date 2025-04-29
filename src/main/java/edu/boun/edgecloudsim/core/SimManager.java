package edu.boun.edgecloudsim.core;

import org.cloudbus.cloudsim.core.CloudSim;


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

/**
 * 模拟管理器扩展，增加UAV管理器支持
 */
public class SimManager {
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
    
    /**
     * 默认构造函数 - 私有化以实现单例模式
     */
    private SimManager() {
        // 默认构造函数，不执行任何初始化
    }
    
    /**
     * 新增构造函数 - 支持带参数的初始化
     * 修复构造函数不匹配的问题
     */
    public SimManager(ScenarioFactory _scenarioFactory, int _numOfMobileDevice, 
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
     * 启动模拟
     */
    public void startSimulation() {
        // 模拟启动逻辑
        CloudSim.startSimulation();
    }
    
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