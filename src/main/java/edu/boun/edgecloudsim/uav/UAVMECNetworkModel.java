package edu.boun.edgecloudsim.uav;

import edu.boun.edgecloudsim.core.SimSettings;
import edu.boun.edgecloudsim.core.ExecutionTarget;
import edu.boun.edgecloudsim.core.SimManager;
import edu.boun.edgecloudsim.edge_client.Task;
import edu.boun.edgecloudsim.network.NetworkModel;
import edu.boun.edgecloudsim.utils.Location;

public class UAVMECNetworkModel extends NetworkModel {
    private static final double SPEED_OF_LIGHT_METERS_PER_SECOND = 299_792_458.0;
    private static final double MAX_USABLE_PATH_LOSS_DB = 140.0;
    private static final double MIN_BANDWIDTH_FACTOR = 0.05;
    private int activeUavUploads;
    private int activeUavDownloads;
    private int completedUavUploads;
    private int completedUavDownloads;
    
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
        if (isUavDeviceId(destDeviceId)) {
            return getUavTransferDelay(task.getCloudletFileSize(), destDeviceId, task);
        }
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
        if (isUavDeviceId(sourceDeviceId)) {
            return getUavTransferDelay(task.getCloudletOutputSize(), sourceDeviceId, task);
        }
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
        if (isUavDeviceId(destDeviceId)) {
            activeUavUploads++;
        }
    }

    @Override
    public void uploadFinished(Location accessPointLocation, int destDeviceId) {
        if (isUavDeviceId(destDeviceId)) {
            activeUavUploads = Math.max(0, activeUavUploads - 1);
            completedUavUploads++;
        }
    }

    @Override
    public void downloadStarted(Location accessPointLocation, int sourceDeviceId) {
        if (isUavDeviceId(sourceDeviceId)) {
            activeUavDownloads++;
        }
    }

    @Override
    public void downloadFinished(Location accessPointLocation, int sourceDeviceId) {
        if (isUavDeviceId(sourceDeviceId)) {
            activeUavDownloads = Math.max(0, activeUavDownloads - 1);
            completedUavDownloads++;
        }
    }

    private double getUavTransferDelay(long sizeKb, int legacyUavId, Task task) {
        int uavId = legacyUavId - ExecutionTarget.UAV_DEVICE_ID_BASE;
        UAVManager manager = SimManager.getInstance().getUAVManager();
        UAV uav = manager == null ? null : manager.getUAV(uavId);
        if (uav == null || task.getSubmittedLocation() == null) {
            return 0.0;
        }
        SimSettings settings = SimSettings.getInstance();
        return calculateAirGroundTransferDelay(
                sizeKb,
                settings.getWlanBandwidth(),
                settings.getInternalLanDelay(),
                settings.getUAVPathLossParameter(),
                settings.getUAVPathLossExponent(),
                settings.getUAVAdditionalPathLoss(),
                task.getSubmittedLocation(),
                uav.getPosition());
    }

    static double calculateAirGroundTransferDelay(long sizeKb, double nominalBandwidthMbps,
            double basePropagationSeconds, double pathLossAtOneMeterDb,
            double pathLossExponent, double additionalLossDb,
            Location userLocation, double[] uavPosition) {
        if (nominalBandwidthMbps <= 0 || userLocation == null
                || uavPosition == null || uavPosition.length < 3) {
            return 0.0;
        }
        double dx = uavPosition[0] - userLocation.getXPos();
        double dy = uavPosition[1] - userLocation.getYPos();
        double dz = uavPosition[2] - userLocation.getZPos();
        double distanceMeters = Math.max(1.0, Math.sqrt(dx * dx + dy * dy + dz * dz));
        double pathLossDb = pathLossAtOneMeterDb
                + 10.0 * pathLossExponent * Math.log10(distanceMeters)
                + additionalLossDb;
        double bandwidthFactor = Math.max(MIN_BANDWIDTH_FACTOR,
                Math.min(1.0, 1.0 - pathLossDb / MAX_USABLE_PATH_LOSS_DB));
        double effectiveBandwidthMbps = nominalBandwidthMbps * bandwidthFactor;
        double propagationSeconds = basePropagationSeconds
                + distanceMeters / SPEED_OF_LIGHT_METERS_PER_SECOND;
        return transferDelay(sizeKb, effectiveBandwidthMbps, propagationSeconds);
    }

    private static double transferDelay(long sizeKb, double bandwidthMbps,
            double propagationSeconds) {
        return propagationSeconds + (sizeKb * 8.0) / (bandwidthMbps * 1000.0);
    }

    private boolean isUavDeviceId(int deviceId) {
        return deviceId >= ExecutionTarget.UAV_DEVICE_ID_BASE;
    }

    public int getActiveUavTransferCount() {
        return activeUavUploads + activeUavDownloads;
    }

    public int getCompletedUavUploadCount() {
        return completedUavUploads;
    }

    public int getCompletedUavDownloadCount() {
        return completedUavDownloads;
    }

    private double transferDelaySeconds(long sizeKb, double bandwidthMbps, double propagationSeconds) {
        if (bandwidthMbps <= 0) {
            return 0.0;
        }
        return propagationSeconds + (sizeKb * 8.0) / (bandwidthMbps * 1000.0);
    }
}
