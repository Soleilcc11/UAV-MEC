package edu.boun.edgecloudsim.uav;

import edu.boun.edgecloudsim.cloud_server.CloudServerManager;
import edu.boun.edgecloudsim.core.ScenarioFactory;
import edu.boun.edgecloudsim.core.SimSettings;
import edu.boun.edgecloudsim.edge_client.MobileDeviceManager;
import edu.boun.edgecloudsim.edge_client.mobile_processing_unit.MobileServerManager;
import edu.boun.edgecloudsim.edge_orchestrator.EdgeOrchestrator;
import edu.boun.edgecloudsim.edge_server.EdgeServerManager;
import edu.boun.edgecloudsim.mobility.MobilityModel;
import edu.boun.edgecloudsim.network.NetworkModel;
import edu.boun.edgecloudsim.task_generator.LoadGeneratorModel;
import edu.boun.edgecloudsim.utils.ArrayUtils;
import edu.boun.edgecloudsim.utils.Location;
import edu.boun.edgecloudsim.utils.TaskProperty;

import org.cloudbus.cloudsim.core.SimEntity;
import org.cloudbus.cloudsim.core.SimEvent;
import org.cloudbus.cloudsim.Vm;
import org.apache.commons.math3.distribution.ExponentialDistribution;
import org.cloudbus.cloudsim.Host;
import org.cloudbus.cloudsim.VmAllocationPolicy;
import java.util.List;
import java.util.ArrayList;
import org.cloudbus.cloudsim.UtilizationModel;
import org.cloudbus.cloudsim.UtilizationModelFull;

public class UAVMECScenarioFactory implements ScenarioFactory {
    private int numOfMobileDevice;
    private String simScenario;
    private String orchestratorPolicy;
    
    public UAVMECScenarioFactory(int _numOfMobileDevice, String _simScenario, String _orchestratorPolicy) {
        numOfMobileDevice = _numOfMobileDevice;
        simScenario = _simScenario;
        orchestratorPolicy = _orchestratorPolicy;
    }
    
   @Override
    public LoadGeneratorModel getLoadGeneratorModel() {
        return new LoadGeneratorModel(numOfMobileDevice, SimSettings.getInstance().getSimulationTime(), simScenario) {
            
            @Override
            public void initializeModel() {
                taskList = new ArrayList<TaskProperty>();
                System.out.println("开始生成任务...");
                
                // 创建随机数生成器
                int taskLookUpTableLength = ArrayUtils.length(SimSettings.getInstance().getTaskLookUpTable());
                ExponentialDistribution[][] expRngList = new ExponentialDistribution[taskLookUpTableLength][3];
                
                // 初始化随机数生成器
                for(int i=0; i<taskLookUpTableLength; i++) {
                    if(ArrayUtils.getDoubleValueFromTable(SimSettings.getInstance().getTaskLookUpTable(), i, 0) == 0)
                        continue;
                    
                    expRngList[i][0] = new ExponentialDistribution(ArrayUtils.getDoubleValueFromTable(SimSettings.getInstance().getTaskLookUpTable(), i, 5));
                    expRngList[i][1] = new ExponentialDistribution(ArrayUtils.getDoubleValueFromTable(SimSettings.getInstance().getTaskLookUpTable(), i, 6));
                    expRngList[i][2] = new ExponentialDistribution(ArrayUtils.getDoubleValueFromTable(SimSettings.getInstance().getTaskLookUpTable(), i, 7));
                }
                
                // 为每个移动设备生成任务
                for(int deviceId=0; deviceId<numOfMobileDevice; deviceId++) {
                    int taskType = 0; // 所有设备使用相同任务类型，简化处理
                    
                    // 获取任务生成参数
                    double poissonMean = ArrayUtils.getDoubleValueFromTable(SimSettings.getInstance().getTaskLookUpTable(), taskType, 2);
                    double activePeriod = ArrayUtils.getDoubleValueFromTable(SimSettings.getInstance().getTaskLookUpTable(), taskType, 3);
                    double idlePeriod = ArrayUtils.getDoubleValueFromTable(SimSettings.getInstance().getTaskLookUpTable(), taskType, 4);
                    
                    // 设置初始活动期开始时间
                    double simulationTime = SimSettings.getInstance().getSimulationTime();
                    double activePeriodStartTime = 10.0; // 从10秒后开始
                    double virtualTime = activePeriodStartTime;
                    int deviceTasks = 0;
                    
                    // 使用泊松分布生成任务到达时间
                    ExponentialDistribution rng = new ExponentialDistribution(poissonMean);
                    
                    // 在整个模拟时间范围内生成任务
                    while(virtualTime < simulationTime) {
                        double interval = rng.sample();
                        if(interval <= 0) continue;
                        
                        virtualTime += interval;
                        
                        // 检查是否超出活动期
                        if(virtualTime > activePeriodStartTime + activePeriod) {
                            activePeriodStartTime = activePeriodStartTime + activePeriod + idlePeriod;
                            virtualTime = activePeriodStartTime;
                            continue;
                        }
                        
                        // 创建任务并添加到列表 - 使用正确的构造函数
                        TaskProperty task = new TaskProperty(deviceId, taskType, virtualTime, expRngList);
                        taskList.add(task);
                        deviceTasks++;
                    }
                    
                    System.out.println("为设备 " + deviceId + " 生成了 " + deviceTasks + " 个任务");
                }
                
                // 验证任务生成结果
                System.out.println("总共生成了 " + taskList.size() + " 个任务");
                
                // 如果没有任务，生成一些基本任务
                if(taskList.isEmpty()) {
                    System.out.println("警告：未生成任务，添加基本测试任务");
                    
                    // 为基本任务创建一个简单的ExpRngList
                    ExponentialDistribution[] basicExpRng = new ExponentialDistribution[3];
                    basicExpRng[0] = new ExponentialDistribution(1000); // 输入文件大小
                    basicExpRng[1] = new ExponentialDistribution(100);  // 输出文件大小
                    basicExpRng[2] = new ExponentialDistribution(10000); // 任务长度
                    
                    // 强制添加一些基本任务
                    for(int i=0; i<100; i++) {
                        double startTime = 10 + i*10;
                        if(startTime >= SimSettings.getInstance().getSimulationTime()) break;
                        
                        // 使用适当的构造函数 - 第三个构造函数
                        TaskProperty task = new TaskProperty(0, startTime, basicExpRng);
                        taskList.add(task);
                    }
                    
                    System.out.println("已添加 " + taskList.size() + " 个基本测试任务");
                }
            }
            
            @Override
            public int getTaskTypeOfDevice(int deviceId) {
                return 0; // 所有设备使用相同任务类型
            }
        };
    }
    
    @Override
    public EdgeOrchestrator getEdgeOrchestrator() {
        // 使用正确的参数创建TaskOffloadingOrchestrator
        return new TaskOffloadingOrchestrator(orchestratorPolicy, simScenario);
    }
    
    @Override
    public MobilityModel getMobilityModel() {
        // 修复构造函数参数 - 只使用两个参数
        return new MobilityModel(numOfMobileDevice, 0.0) {
            @Override
            public void initialize() {}
            
            @Override
            public Location getLocation(int deviceId, double time) {
                // 修复构造函数参数 - 4个参数
                return new Location(0, 0, 0, 0);
            }
        };
    }
    
    @Override
    public NetworkModel getNetworkModel() {
        // 实现 UAV 网络模型
        return new UAVMECNetworkModel(numOfMobileDevice, simScenario);
    }
    
    @Override
    public EdgeServerManager getEdgeServerManager() {
        return new EdgeServerManager() {
            @Override
            public void initialize() {}
            
            public void startEntity() {}
            
            public void shutdownEntity() {}
            
            public void processEvent(SimEvent ev) {}
            
            @Override
            public double getAvgUtilization() {
                return 0.0;
            }
            
            @Override
            public void createVmList(int brokerId) {
                // 实现创建VM列表的逻辑，但不返回任何内容
                // 符合EdgeServerManager接口定义
            }
            
            @Override
            public void startDatacenters() throws Exception {
                // 实现启动数据中心的逻辑
            }
            
            @Override
            public void terminateDatacenters() {
                // 实现终止数据中心的逻辑
            }
            
            @Override
            public VmAllocationPolicy getVmAllocationPolicy(List<? extends Host> list, int dataCenterIndex) {
                // 实现VM分配策略
                return null;
            }
            
            // 额外的辅助方法
            public Vm getVmInstance(int hostId, int vmId) {
                return null;
            }
        };
    }
    
    @Override
    public CloudServerManager getCloudServerManager() {
        return new CloudServerManager() {
            @Override
            public void initialize() {}
            
            public void startEntity() {}
            
            public void shutdownEntity() {}
            
            public void processEvent(SimEvent ev) {}
            
            @Override
            public double getAvgUtilization() {
                return 0.0;
            }
            
            @Override
            public void createVmList(int brokerId) {
                // 实现创建VM列表的逻辑，但不返回任何内容
                // 符合CloudServerManager接口定义
            }
            
            @Override
            public void startDatacenters() throws Exception {
                // 实现启动数据中心的逻辑
            }
            
            @Override
            public void terminateDatacenters() {
                // 实现终止数据中心的逻辑
            }
            
            @Override
            public VmAllocationPolicy getVmAllocationPolicy(List<? extends Host> list, int dataCenterIndex) {
                // 实现VM分配策略
                return null;
            }
            
            // 额外的辅助方法
            public Vm getVmInstance(int hostId, int vmId) {
                return null;
            }
        };
    }
    
    @Override
    public MobileServerManager getMobileServerManager() {
        return new MobileServerManager() {
            @Override
            public void initialize() {}
            
            @Override
            public double getAvgUtilization() {
                return 0.0;
            }
            
            @Override
            public void createVmList(int brokerId) {
                // 实现创建VM列表的逻辑，但不返回任何内容
                // 符合MobileServerManager接口定义
            }
            
            @Override
            public void startDatacenters() throws Exception {
                // 实现启动数据中心的逻辑
            }
            
            @Override
            public void terminateDatacenters() {
                // 实现终止数据中心的逻辑
            }
            
            @Override
            public VmAllocationPolicy getVmAllocationPolicy(List<? extends Host> list, int dataCenterIndex) {
                // 实现VM分配策略
                return null;
            }
            
            // 额外的辅助方法
            public Vm getVmInstance(int hostId, int vmId) {
                return null;
            }
        };
    }
    
    @Override
    public MobileDeviceManager getMobileDeviceManager() throws Exception {
        // 添加getCpuUtilizationModel方法
        return new MobileDeviceManager() {
            @Override
            public void initialize() {}
            
            @Override
            public void startEntity() {}
            
            @Override
            public void shutdownEntity() {}
            
            @Override
            public void processEvent(SimEvent ev) {}
            
            @Override
            public void submitTask(edu.boun.edgecloudsim.utils.TaskProperty task) {
                // 任务提交逻辑
            }
            
            @Override
            public UtilizationModel getCpuUtilizationModel() {
                return new UtilizationModelFull();
            }
        };
    }
    
    // 获取 UAV 管理器的附加方法
    public UAVManager getUAVManager() {
        return new UAVManager(edu.boun.edgecloudsim.core.SimManager.getInstance());
    }
}