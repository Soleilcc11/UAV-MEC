package edu.boun.edgecloudsim.core;

import java.util.Properties;
import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import org.w3c.dom.Document;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.File;


import edu.boun.edgecloudsim.uav.UAVMECScenarioFactory;

/**
 * 模拟设置类，增加UAV配置支持
 */
public class SimSettings {

    private static SimSettings instance = null;
    private Properties configFile;
    private UAVMECScenarioFactory scenarioFactory;
    private Document edgeDevicesDoc = null;
    private Document applicationsDoc = null;
    
    // 添加分隔符常量
    public static final String DELIMITER = ",";
    
    public static final int CLOUD_DATACENTER_ID = 1000;
    public static final int MOBILE_DATACENTER_ID = 1001;
    public static final int EDGE_ORCHESTRATOR_ID = 1002;
    public static final int GENERIC_EDGE_DEVICE_ID = 1003;
    
    // 添加客户端活动时间常量
    public static final double CLIENT_ACTIVITY_START_TIME = 10;
    
    public int getNumOfEdgeVMs() { return 10; }
    public int getNumOfCloudVMs() { return 5; }
    public int getNumOfEdgeHosts() { return 5; }
    public int getNumOfCloudHost() { return 2; }
    public int getCoreForMobileVM() { return 4; }
    public int getMipsForMobileVM() { return 1000; }
    public int getRamForMobileVM() { return 1024; }
    public int getStorageForMobileVM() { return 10000; }
    public Object getTaskLookUpTable() { return new Object(); }
    
    // 网络相关方法
    public double getInternalLanDelay() { return 0.1; }
    public double getWlanBandwidth() { return 100; }
    public double getWanBandwidth() { return 50; }
    public double getWanPropagationDelay() { return 0.2; }
    
    // 添加边缘数据中心数量获取方法
    public int getNumOfEdgeDatacenters() {
        return Integer.parseInt(configFile.getProperty("num_of_edge_datacenters", "1"));
    }
    
    // 添加日志间隔获取方法
    public double getApDelayLogInterval() {
        return Double.parseDouble(configFile.getProperty("ap_delay_log_interval", "5"));
    }

    public double getVmLoadLogInterval() {
        return Double.parseDouble(configFile.getProperty("vm_load_log_interval", "5"));
    }

    public double getLocationLogInterval() {
        return Double.parseDouble(configFile.getProperty("location_log_interval", "5"));
    }

    // 添加任务相关方法
    public String getTaskName(int taskType) {
        return configFile.getProperty("task_" + taskType + "_name", "default_task");
    }

    public Object getTaskProperties(String taskName) {
        // 返回任务属性对象
        return new Object();
    }

    // 添加云VM相关方法
    public int getNumOfCloudVMsPerHost() {
        return Integer.parseInt(configFile.getProperty("cloud_vms_per_host", "4"));
    }

    // 添加日志深度配置
    public boolean getDeepFileLoggingEnabled() {
        return Boolean.parseBoolean(configFile.getProperty("deep_file_logging_enabled", "false"));
    }
    
    /**
     * 设备类型枚举
     */
    public enum DEVICE_TYPE {
        MOBILE_DEVICE(0),
        EDGE_SERVER(1),
        CLOUD_SERVER(2),
        UAV(3);
        
        private int value;
        
        private DEVICE_TYPE(int value) {
            this.value = value;
        }
        
        public int getValue() {
            return value;
        }
    }
		
	public enum VM_TYPES {
		MOBILE_VM,
		EDGE_VM,
		CLOUD_VM
	}

	public enum NETWORK_DELAY_TYPES {
		WLAN,
		MAN,
		WAN,
		GSM;
		
		// 静态引用
		public static final NETWORK_DELAY_TYPES WLAN_DELAY = WLAN;
		public static final NETWORK_DELAY_TYPES MAN_DELAY = MAN;
		public static final NETWORK_DELAY_TYPES WAN_DELAY = WAN;
		public static final NETWORK_DELAY_TYPES GSM_DELAY = GSM;
	}
		
    // 获取实例（单例模式）
    public static SimSettings getInstance(){
        if(instance == null){
            instance = new SimSettings();
        }
        return instance;
    }
    
    /**
     * 初始化模拟设置
     */
    public void initialize(String configFile, String edgeDevicesFile, String applicationsFile) {
        this.configFile = new Properties();
        try {
            this.configFile.load(new FileInputStream(configFile));
            
            // 加载边缘设备和应用程序XML文件
            DocumentBuilderFactory dbFactory = DocumentBuilderFactory.newInstance();
            DocumentBuilder dBuilder = dbFactory.newDocumentBuilder();
            
            edgeDevicesDoc = dBuilder.parse(new File(edgeDevicesFile));
            applicationsDoc = dBuilder.parse(new File(applicationsFile));
            
            edgeDevicesDoc.getDocumentElement().normalize();
            applicationsDoc.getDocumentElement().normalize();
            
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }
    }
    
    /**
     * 获取边缘设备文档
     */
    public Document getEdgeDevicesDocument() {
        return edgeDevicesDoc;
    }
    
    /**
     * 获取应用程序文档
     */
    public Document getApplicationsDocument() {
        return applicationsDoc;
    }
    
    /**
     * 获取是否启用文件日志
     */
    public boolean getFileLoggingEnabled() {
        return Boolean.parseBoolean(configFile.getProperty("file_logging_enabled", "true"));
    }
    
    /**
     * 获取最小移动设备数量
     */
    public int getMinNumOfMobileDev() {
        return Integer.parseInt(configFile.getProperty("min_number_of_mobile_devices", "100"));
    }
    
    /**
     * 获取最大移动设备数量
     */
    public int getMaxNumOfMobileDev() {
        return Integer.parseInt(configFile.getProperty("max_number_of_mobile_devices", "1000"));
    }
    
    /**
     * 获取移动设备计数器大小
     */
    public int getMobileDevCounterSize() {
        return Integer.parseInt(configFile.getProperty("mobile_device_counter_size", "1000"));
    }
    
    /**
     * 获取模拟场景列表
     */
    public String[] getSimulationScenarios() {
        String scenarioList = configFile.getProperty("simulation_scenarios", "DEFAULT_SCENARIO");
        return scenarioList.split(",");
    }
    
    /**
     * 获取编排策略列表
     */
    public String[] getOrchestratorPolicies() {
        String policyList = configFile.getProperty("orchestrator_policies", "DEFAULT_POLICY");
        return policyList.split(",");
    }
    
    /**
     * 获取模拟时间
     */
    public double getSimulationTime() {
        return Double.parseDouble(configFile.getProperty("simulation_time", "1000"));
    }
    
    /**
     * 获取预热期
     */
    public double getWarmUpPeriod() {
        return Double.parseDouble(configFile.getProperty("warm_up_period", "100"));
    }
    
    /**
     * 设置场景工厂
     */
    public void setScenarioFactory(UAVMECScenarioFactory factory) {
        this.scenarioFactory = factory;
    }
    
    /**
     * 获取场景工厂
     */
    public UAVMECScenarioFactory getScenarioFactory() {
        return scenarioFactory;
    }
    
    //======== UAV相关设置 ========//
    
    /**
     * 获取UAV数量
     */
    public int getNumOfUAVs() {
        return Integer.parseInt(configFile.getProperty("num_of_uavs", "5"));
    }
    
    /**
     * 获取模拟空间
     * @return [x, y, z]大小的空间
     */
    public double[] getSimulationSpace() {
        double[] space = new double[3];
        space[0] = Double.parseDouble(configFile.getProperty("simulation_space_x", "1000"));
        space[1] = Double.parseDouble(configFile.getProperty("simulation_space_y", "1000"));
        space[2] = Double.parseDouble(configFile.getProperty("simulation_space_z", "100"));
        return space;
    }
    
    /**
     * 获取随机X坐标
     */
    public double getRandomPositionX() {
        double[] space = getSimulationSpace();
        return Math.random() * space[0];
    }
    
    /**
     * 获取随机Y坐标
     */
    public double getRandomPositionY() {
        double[] space = getSimulationSpace();
        return Math.random() * space[1];
    }
    
    /**
     * 获取UAV初始高度
     */
    public double getUAVInitialHeight() {
        return Double.parseDouble(configFile.getProperty("uav_initial_height", "50"));
    }
    
    /**
     * 获取UAV最小高度
     */
    public double getUAVMinHeight() {
        return Double.parseDouble(configFile.getProperty("uav_min_height", "10"));
    }
    
    /**
     * 获取UAV最大高度
     */
    public double getUAVMaxHeight() {
        return Double.parseDouble(configFile.getProperty("uav_max_height", "100"));
    }
    
    /**
     * 获取UAV最大能量
     */
    public double getUAVMaxEnergy() {
        return Double.parseDouble(configFile.getProperty("uav_max_energy", "5000"));
    }
    
    /**
     * 获取UAV飞行功率
     */
    public double getUAVFlightPower() {
        return Double.parseDouble(configFile.getProperty("uav_flight_power", "50"));
    }
    
    /**
     * 获取UAV悬停功率
     */
    public double getUAVHoverPower() {
        return Double.parseDouble(configFile.getProperty("uav_hover_power", "30"));
    }
    
    /**
     * 获取UAV通信路径损耗参数
     */
    public double getUAVPathLossParameter() {
        return Double.parseDouble(configFile.getProperty("uav_path_loss_parameter", "32.4"));
    }
    
    /**
     * 获取UAV通信路径损耗指数
     */
    public double getUAVPathLossExponent() {
        return Double.parseDouble(configFile.getProperty("uav_path_loss_exponent", "2.0"));
    }
    
    /**
     * 获取UAV额外路径损耗
     */
    public double getUAVAdditionalPathLoss() {
        return Double.parseDouble(configFile.getProperty("uav_additional_path_loss", "0.0"));
    }

    public double getManBandwidth() {
        return Double.parseDouble(configFile.getProperty("man_bandwidth", "100"));
    }
    
    public double getGsmBandwidth() {
        return Double.parseDouble(configFile.getProperty("gsm_bandwidth", "50"));
    }
    
    public double getGsmPropagationDelay() {
        return Double.parseDouble(configFile.getProperty("gsm_propagation_delay", "0.5"));
    }
}