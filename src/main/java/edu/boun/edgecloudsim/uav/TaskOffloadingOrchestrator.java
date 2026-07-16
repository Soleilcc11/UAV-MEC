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
    
    private SimManager simManager;
    
    /**
     * 构造函数
     */
    public TaskOffloadingOrchestrator(String _policy, String _scenario) {
        super(_policy, _scenario);
        this.simManager = SimManager.getInstance();
    }

    @Override
    public void initialize() {
        // TaskOffloadingEngine is owned and initialized by SimManager.
    }
    
    /**
     * 实现EdgeOrchestrator接口的getDeviceToOffload方法
     */
    @Override
    public int getDeviceToOffload(Task task) {
        // 从任务中获取移动设备ID
        int mobileDeviceId = task.getMobileDeviceId();
        
        // 调用内部方法处理设备ID
        return getDeviceToOffload(mobileDeviceId);
    }
    
    /**
     * 内部方法：根据移动设备ID获取卸载目标设备
     */
    private int getDeviceToOffload(int mobileDeviceId) {
        // 根据移动设备ID获取卸载目标设备
        double[] location = getMobileDeviceLocation(mobileDeviceId);
        
        // 找到最近的UAV
        UAVManager uavManager = simManager.getUAVManager();
        if (uavManager == null) {
            return TaskOffloadingEngine.CLOUD_EXECUTION;
        }
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
        // 根据任务类型和设备ID获取合适的VM
        int vmType;
        
        if (deviceId == SimSettings.CLOUD_DATACENTER_ID) {
            vmType = SimSettings.VM_TYPES.CLOUD_VM.ordinal();
        } else if (deviceId == SimSettings.MOBILE_DATACENTER_ID) {
            vmType = SimSettings.VM_TYPES.MOBILE_VM.ordinal();
        } else {
            vmType = SimSettings.VM_TYPES.EDGE_VM.ordinal();
        }
        
        // 根据设备类型选择VM
        if (deviceId == SimSettings.CLOUD_DATACENTER_ID) {
            return getCloudServerVm();
        } else if (deviceId == SimSettings.MOBILE_DATACENTER_ID) {
            return getMobileDeviceVm(task.getMobileDeviceId());
        } else {
            return getEdgeServerVm(deviceId);
        }
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
        return simManager.getTaskOffloadingEngine();
    }
    
    @Override
    public void processEvent(SimEvent ev) {
        // EdgeOrchestrator的事件处理方法
    }
    
    @Override
    public void shutdownEntity() {
        // SimManager owns and closes the engine.
    }
    
    @Override
    public void startEntity() {
        // 启动实体
    }
}
