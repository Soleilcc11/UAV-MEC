package edu.boun.edgecloudsim.uav;

import edu.boun.edgecloudsim.core.SimManager;
import edu.boun.edgecloudsim.edge_client.Task;
import edu.boun.edgecloudsim.network.NetworkModel;
import edu.boun.edgecloudsim.utils.Location;

public class UAVMECNetworkModel extends NetworkModel {
    
    public UAVMECNetworkModel(int _numberOfMobileDevices, String _simScenario) {
        super(_numberOfMobileDevices, _simScenario);
    }

    @Override
    public void initialize() {
        // 初始化网络模型
    }

    @Override
    public double getUploadDelay(int sourceDeviceId, int destDeviceId, Task task) {
        // 计算并返回上传延迟
        return 10.0; // 默认延迟为10毫秒
    }

    @Override
    public double getDownloadDelay(int sourceDeviceId, int destDeviceId, Task task) {
        // 计算并返回下载延迟
        return 10.0; // 默认延迟为10毫秒
    }

    @Override
    public void uploadStarted(Location accessPointLocation, int destDeviceId) {
        // 处理上传开始事件
    }

    @Override
    public void uploadFinished(Location accessPointLocation, int destDeviceId) {
        // 处理上传完成事件
    }

    @Override
    public void downloadStarted(Location accessPointLocation, int sourceDeviceId) {
        // 处理下载开始事件
    }

    @Override
    public void downloadFinished(Location accessPointLocation, int sourceDeviceId) {
        // 处理下载完成事件
    }
}