package edu.boun.edgecloudsim.uav;

import edu.boun.edgecloudsim.core.SimSettings;
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
        if (task == null) {
            return 0.0;
        }
        boolean cloud = destDeviceId == SimSettings.CLOUD_DATACENTER_ID;
        double bandwidthMbps = cloud
                ? SimSettings.getInstance().getWanBandwidth()
                : SimSettings.getInstance().getWlanBandwidth();
        double propagationSeconds = cloud
                ? SimSettings.getInstance().getWanPropagationDelay()
                : SimSettings.getInstance().getInternalLanDelay();
        return transferDelaySeconds(task.getCloudletFileSize(), bandwidthMbps, propagationSeconds);
    }

    @Override
    public double getDownloadDelay(int sourceDeviceId, int destDeviceId, Task task) {
        if (task == null) {
            return 0.0;
        }
        boolean cloud = sourceDeviceId == SimSettings.CLOUD_DATACENTER_ID;
        double bandwidthMbps = cloud
                ? SimSettings.getInstance().getWanBandwidth()
                : SimSettings.getInstance().getWlanBandwidth();
        double propagationSeconds = cloud
                ? SimSettings.getInstance().getWanPropagationDelay()
                : SimSettings.getInstance().getInternalLanDelay();
        return transferDelaySeconds(task.getCloudletOutputSize(), bandwidthMbps, propagationSeconds);
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

    private double transferDelaySeconds(long sizeKb, double bandwidthMbps, double propagationSeconds) {
        if (bandwidthMbps <= 0) {
            return 0.0;
        }
        return propagationSeconds + (sizeKb * 8.0) / (bandwidthMbps * 1000.0);
    }
}
