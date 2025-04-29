package edu.boun.edgecloudsim.uav;

import org.cloudbus.cloudsim.Vm;
import org.cloudbus.cloudsim.core.SimEvent;
import edu.boun.edgecloudsim.edge_orchestrator.EdgeOrchestrator;
import edu.boun.edgecloudsim.core.SimManager;
import edu.boun.edgecloudsim.core.SimSettings;
import edu.boun.edgecloudsim.edge_client.Task;

/**
 * 任务卸载编排器，适配EdgeOrchestrator接口
 */
public class TaskOffloadingOrchestrator extends EdgeOrchestrator {
    
    private TaskOffloadingEngine engine;
    private UAVManager uavManager;
    private SimManager simManager;
    
    /**
     * 构造函数
     */
    public TaskOffloadingOrchestrator(String _policy, String _scenario) {
        super(_policy, _scenario);
        this.simManager = SimManager.getInstance();
        this.uavManager = simManager.getUAVManager();
        this.engine = new TaskOffloadingEngine(simManager);
    }

    @Override
    public void initialize() {
        // 初始化已在TaskOffloadingEngine构造函数中完成
        if (engine != null) {
            engine.initialize();
        }
    }
    
    @Override
    public int getDeviceToOffload(int mobileDeviceId) {
        // 根据移动设备ID获取卸载目标设备
        double[] location = getMobileDeviceLocation(mobileDeviceId);
        
        // 找到最近的UAV
        int nearestUavId = uavManager.findNearestUAV(location[0], location[1]);
        UAV nearestUAV = null;
        if (nearestUavId >= 0) {
            nearestUAV = uavManager.getUAV(nearestUavId);
        }
        
        // 简化的决策逻辑
        if (nearestUAV == null || nearestUAV.getEnergy() <= 0) {
            // 如果没有可用UAV，卸载到云
            return TaskOffloadingEngine.CLOUD_EXECUTION;
        } else if (nearestUAV.getAvailableCapacity() > 0) {
            // 如果最近的UAV有足够容量，卸载到UAV
            return TaskOffloadingEngine.UAV_EXECUTION;
        } else {
            // 否则本地执行
            return TaskOffloadingEngine.LOCAL_EXECUTION;
        }
    }
    
    /**
     * 获取移动设备位置
     * 这是一个帮助方法，根据移动设备ID返回其位置
     */
    private double[] getMobileDeviceLocation(int mobileDeviceId) {
        // 简化版：返回随机位置，实际应从SimManager获取
        return new double[] {
            SimSettings.getInstance().getRandomPositionX(),
            SimSettings.getInstance().getRandomPositionY()
        };
    }

    @Override
    public Vm getVmToOffload(Task task, int deviceId) {
        // 根据设备ID选择合适的VM
        int vmType = SimSettings.VM_TYPES.EDGE_VM.ordinal();
        return getVmToOffload(deviceId, deviceId, vmType, task.getTaskType());
    }

    /**
     * 获取移动设备VM
     */
    private Vm getMobileDeviceVm(int mobileDeviceId) {
        if (simManager.getMobileDeviceManager() != null) {
            // 尝试获取移动设备的VM（简化版实现）
            try {
                return simManager.getMobileServerManager().getVmList(mobileDeviceId).get(0);
            } catch (Exception e) {
                return null;
            }
        }
        return null;
    }
    
    /**
     * 获取边缘服务器VM
     */
    private Vm getEdgeServerVm(int hostId) {
        if (simManager.getEdgeServerManager() != null) {
            try {
                return simManager.getEdgeServerManager().getVmList(hostId).get(0);
            } catch (Exception e) {
                return null;
            }
        }
        return null;
    }
    
    /**
     * 获取云服务器VM
     */
    private Vm getCloudServerVm() {
        if (simManager.getCloudServerManager() != null) {
            try {
                return simManager.getCloudServerManager().getVmList(0).get(0);
            } catch (Exception e) {
                return null;
            }
        }
        return null;
    }
    
    /**
     * 获取内部的TaskOffloadingEngine实例
     */
    public TaskOffloadingEngine getEngine() {
        return engine;
    }
    
    @Override
    public void processEvent(SimEvent ev) {
        // EdgeOrchestrator的事件处理方法
    }
    
    @Override
    public void shutdownEntity() {
        // 关闭实体
        if (engine != null) {
            engine.close();
        }
    }
    
    @Override
    public void startEntity() {
        // 启动实体
    }
    
    /**
     * 获取任务执行决策类型
     */
    private int getExecutionDecisionType(int deviceId) {
        if (deviceId == SimSettings.CLOUD_DATACENTER_ID) {
            return TaskOffloadingEngine.CLOUD_EXECUTION;
        } else if (deviceId >= 0 && deviceId < simManager.getUAVManager().getUAVs().size()) {
            return TaskOffloadingEngine.UAV_EXECUTION;
        } else {
            return TaskOffloadingEngine.LOCAL_EXECUTION;
        }
    }
}