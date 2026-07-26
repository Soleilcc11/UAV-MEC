package edu.boun.edgecloudsim.uav;

import edu.boun.edgecloudsim.core.ExecutionTarget;
import edu.boun.edgecloudsim.core.SimManager;
import edu.boun.edgecloudsim.core.SimSettings;
import edu.boun.edgecloudsim.edge_client.Task;
import edu.boun.edgecloudsim.network.NetworkModel;
import edu.boun.edgecloudsim.utils.Location;

/**
 * UAV access and UAV-to-cloud backhaul model used by the GymBridge scenario.
 *
 * <p>The access link uses the standard probabilistic LoS/NLoS mean path-loss
 * model and Shannon capacity. The configured access bandwidth is divided into
 * a fixed number of orthogonal channels; transfers beyond that channel count
 * time-share the least-loaded channel deterministically.</p>
 */
public class UAVMECNetworkModel extends NetworkModel {
    private static final double SPEED_OF_LIGHT_METERS_PER_SECOND = 299_792_458.0;

    private int activeUavUploads;
    private int activeUavDownloads;
    private int activeCloudUploads;
    private int activeCloudDownloads;
    private int completedUavUploads;
    private int completedUavDownloads;

    public UAVMECNetworkModel(int numberOfMobileDevices, String simScenario) {
        super(numberOfMobileDevices, simScenario);
    }

    @Override
    public void initialize() {
        activeUavUploads = 0;
        activeUavDownloads = 0;
        activeCloudUploads = 0;
        activeCloudDownloads = 0;
        completedUavUploads = 0;
        completedUavDownloads = 0;
    }

    @Override
    public double getUploadDelay(int sourceDeviceId, int destDeviceId, Task task) {
        if (task == null) {
            return 0.0;
        }
        if (isUavDeviceId(destDeviceId)) {
            return getUavTransferDelay(
                    task.getCloudletFileSize(), destDeviceId, task,
                    activeUavUploads + activeCloudUploads + 1);
        }
        if (destDeviceId == SimSettings.CLOUD_DATACENTER_ID) {
            return getCloudTwoHopTransferDelay(
                    task.getCloudletFileSize(), task,
                    activeUavUploads + activeCloudUploads + 1,
                    activeCloudUploads + 1);
        }
        return transferDelay(
                task.getCloudletFileSize(),
                SimSettings.getInstance().getWlanBandwidth(),
                SimSettings.getInstance().getInternalLanDelay());
    }

    @Override
    public double getDownloadDelay(int sourceDeviceId, int destDeviceId, Task task) {
        if (task == null) {
            return 0.0;
        }
        if (isUavDeviceId(sourceDeviceId)) {
            return getUavTransferDelay(
                    task.getCloudletOutputSize(), sourceDeviceId, task,
                    activeUavDownloads + activeCloudDownloads + 1);
        }
        if (sourceDeviceId == SimSettings.CLOUD_DATACENTER_ID) {
            return getCloudTwoHopTransferDelay(
                    task.getCloudletOutputSize(), task,
                    activeUavDownloads + activeCloudDownloads + 1,
                    activeCloudDownloads + 1);
        }
        return transferDelay(
                task.getCloudletOutputSize(),
                SimSettings.getInstance().getWlanBandwidth(),
                SimSettings.getInstance().getInternalLanDelay());
    }

    @Override
    public void uploadStarted(Location accessPointLocation, int destDeviceId) {
        if (isUavDeviceId(destDeviceId)) {
            activeUavUploads++;
        } else if (destDeviceId == SimSettings.CLOUD_DATACENTER_ID) {
            activeCloudUploads++;
        }
    }

    @Override
    public void uploadFinished(Location accessPointLocation, int destDeviceId) {
        if (isUavDeviceId(destDeviceId)) {
            activeUavUploads = Math.max(0, activeUavUploads - 1);
            completedUavUploads++;
        } else if (destDeviceId == SimSettings.CLOUD_DATACENTER_ID) {
            activeCloudUploads = Math.max(0, activeCloudUploads - 1);
        }
    }

    @Override
    public void downloadStarted(Location accessPointLocation, int sourceDeviceId) {
        if (isUavDeviceId(sourceDeviceId)) {
            activeUavDownloads++;
        } else if (sourceDeviceId == SimSettings.CLOUD_DATACENTER_ID) {
            activeCloudDownloads++;
        }
    }

    @Override
    public void downloadFinished(Location accessPointLocation, int sourceDeviceId) {
        if (isUavDeviceId(sourceDeviceId)) {
            activeUavDownloads = Math.max(0, activeUavDownloads - 1);
            completedUavDownloads++;
        } else if (sourceDeviceId == SimSettings.CLOUD_DATACENTER_ID) {
            activeCloudDownloads = Math.max(0, activeCloudDownloads - 1);
        }
    }

    private double getUavTransferDelay(long sizeKb, int legacyUavId, Task task,
            int concurrentAccessTransfers) {
        int uavId = legacyUavId - ExecutionTarget.UAV_DEVICE_ID_BASE;
        UAVManager manager = SimManager.getInstance().getUAVManager();
        UAV uav = manager == null ? null : manager.getUAV(uavId);
        if (uav == null || task.getSubmittedLocation() == null) {
            return 0.0;
        }
        return calculateConfiguredAirGroundTransferDelay(
                sizeKb, task.getSubmittedLocation(), uav.getPosition(),
                concurrentAccessTransfers);
    }

    private double getCloudTwoHopTransferDelay(long sizeKb, Task task,
            int concurrentAccessTransfers, int concurrentBackhaulTransfers) {
        if (task.getSubmittedLocation() == null) {
            return 0.0;
        }
        int relayUavId = task.getCloudRelayUavId();
        if (relayUavId < 0) {
            relayUavId = selectBestCloudRelayUavId(
                    task.getSubmittedLocation(),
                    task.getCloudletFileSize(),
                    task.getCloudletOutputSize());
            task.setCloudRelayUavId(relayUavId);
        }
        UAVManager manager = SimManager.getInstance().getUAVManager();
        UAV relay = manager == null ? null : manager.getUAV(relayUavId);
        if (relay == null || relay.getEnergy() <= 0.0) {
            return 0.0;
        }
        double accessDelay = calculateConfiguredAirGroundTransferDelay(
                sizeKb, task.getSubmittedLocation(), relay.getPosition(),
                concurrentAccessTransfers);
        SimSettings settings = SimSettings.getInstance();
        double backhaulPropagation = settings.getUAVCloudPropagationDelay()
                + settings.getUAVCloudDistanceMeters()
                        / SPEED_OF_LIGHT_METERS_PER_SECOND;
        return accessDelay + transferDelay(
                sizeKb, settings.getUAVCloudBandwidthMbps()
                        / Math.max(1, concurrentBackhaulTransfers),
                backhaulPropagation);
    }

    /**
     * Selects one relay for the full cloud round trip. Ties are resolved by
     * UAV ID because the manager list is stable and scanned in ascending order.
     */
    public int selectBestCloudRelayUavId(Location userLocation,
            long uploadSizeKb, long downloadSizeKb) {
        UAVManager manager = SimManager.getInstance().getUAVManager();
        if (manager == null || userLocation == null) {
            return -1;
        }
        double bestDelay = Double.POSITIVE_INFINITY;
        int bestUavId = -1;
        for (UAV uav : manager.getUAVs()) {
            if (uav.getEnergy() <= 0.0) {
                continue;
            }
            double delay = calculateConfiguredAirGroundTransferDelay(
                    uploadSizeKb, userLocation, uav.getPosition(), 1)
                    + calculateConfiguredAirGroundTransferDelay(
                            downloadSizeKb, userLocation, uav.getPosition(), 1);
            if (delay < bestDelay) {
                bestDelay = delay;
                bestUavId = uav.getId();
            }
        }
        return bestUavId;
    }

    private double calculateConfiguredAirGroundTransferDelay(long sizeKb,
            Location userLocation, double[] uavPosition,
            int concurrentAccessTransfers) {
        SimSettings settings = SimSettings.getInstance();
        return calculateProbabilisticAirGroundTransferDelay(
                sizeKb,
                settings.getUAVAccessBandwidthMbps(),
                settings.getInternalLanDelay(),
                settings.getUAVCarrierFrequencyHz(),
                settings.getUAVLoSA(),
                settings.getUAVLoSB(),
                settings.getUAVEtaLoSDb(),
                settings.getUAVEtaNLoSDb(),
                settings.getUAVTransmitPowerWatts(),
                settings.getUAVNoisePsdDbmPerHz(),
                settings.getUAVAccessChannelCount(),
                concurrentAccessTransfers,
                userLocation,
                uavPosition);
    }

    static double calculateProbabilisticAirGroundTransferDelay(long sizeKb,
            double totalBandwidthMbps, double basePropagationSeconds,
            double carrierFrequencyHz, double environmentA, double environmentB,
            double etaLosDb, double etaNlosDb, double transmitPowerWatts,
            double noisePsdDbmPerHz, int channelCount, int concurrentTransfers,
            Location userLocation, double[] uavPosition) {
        double rateMbps = calculateProbabilisticAirGroundRateMbps(
                totalBandwidthMbps, carrierFrequencyHz, environmentA,
                environmentB, etaLosDb, etaNlosDb, transmitPowerWatts,
                noisePsdDbmPerHz, channelCount, concurrentTransfers,
                userLocation, uavPosition);
        if (rateMbps <= 0.0) {
            return 0.0;
        }
        double distanceMeters = distanceMeters(userLocation, uavPosition);
        return transferDelay(sizeKb, rateMbps,
                Math.max(0.0, basePropagationSeconds)
                        + distanceMeters / SPEED_OF_LIGHT_METERS_PER_SECOND);
    }

    static double calculateProbabilisticAirGroundTransferDelay(long sizeKb,
            double totalBandwidthMbps, double basePropagationSeconds,
            double carrierFrequencyHz, double environmentA, double environmentB,
            double etaLosDb, double etaNlosDb, Location userLocation,
            double[] uavPosition) {
        return calculateProbabilisticAirGroundTransferDelay(
                sizeKb, totalBandwidthMbps, basePropagationSeconds,
                carrierFrequencyHz, environmentA, environmentB,
                etaLosDb, etaNlosDb, 0.1, -174.0, 4, 1,
                userLocation, uavPosition);
    }

    static double calculateProbabilisticAirGroundRateMbps(
            double totalBandwidthMbps, double carrierFrequencyHz,
            double environmentA, double environmentB, double etaLosDb,
            double etaNlosDb, double transmitPowerWatts,
            double noisePsdDbmPerHz, int channelCount, int concurrentTransfers,
            Location userLocation, double[] uavPosition) {
        if (totalBandwidthMbps <= 0.0 || carrierFrequencyHz <= 0.0
                || transmitPowerWatts <= 0.0 || channelCount <= 0
                || concurrentTransfers <= 0 || userLocation == null
                || uavPosition == null || uavPosition.length < 3) {
            return 0.0;
        }
        double dx = uavPosition[0] - userLocation.getXPos();
        double dy = uavPosition[1] - userLocation.getYPos();
        double dz = uavPosition[2] - userLocation.getZPos();
        double horizontalMeters = Math.sqrt(dx * dx + dy * dy);
        double distanceMeters = Math.max(1.0,
                Math.sqrt(horizontalMeters * horizontalMeters + dz * dz));
        double elevationDegrees = Math.toDegrees(Math.atan2(
                Math.max(0.0, dz), Math.max(1e-9, horizontalMeters)));
        double losProbability = calculateLoSProbability(
                elevationDegrees, environmentA, environmentB);
        double freeSpaceLossDb = 20.0 * Math.log10(
                4.0 * Math.PI * carrierFrequencyHz * distanceMeters
                        / SPEED_OF_LIGHT_METERS_PER_SECOND);
        double pathLossDb = freeSpaceLossDb
                + losProbability * etaLosDb
                + (1.0 - losProbability) * etaNlosDb;
        double channelGain = Math.pow(10.0, -pathLossDb / 10.0);
        double channelBandwidthHz =
                totalBandwidthMbps * 1_000_000.0 / channelCount;
        double noisePsdWattsPerHz =
                Math.pow(10.0, (noisePsdDbmPerHz - 30.0) / 10.0);
        double noiseWatts = noisePsdWattsPerHz * channelBandwidthHz;
        double snr = transmitPowerWatts * channelGain
                / Math.max(Double.MIN_NORMAL, noiseWatts);
        double channelCapacityMbps = channelBandwidthHz
                * (Math.log1p(snr) / Math.log(2.0)) / 1_000_000.0;
        int usersPerChannel = (concurrentTransfers + channelCount - 1)
                / channelCount;
        return channelCapacityMbps / Math.max(1, usersPerChannel);
    }

    static double calculateLoSProbability(double elevationDegrees,
            double environmentA, double environmentB) {
        if (environmentA <= 0.0 || environmentB <= 0.0) {
            throw new IllegalArgumentException(
                    "LoS environment parameters must be positive");
        }
        double probability = 1.0 / (1.0 + environmentA * Math.exp(
                -environmentB * (elevationDegrees - environmentA)));
        return Math.max(0.0, Math.min(1.0, probability));
    }

    /**
     * Legacy helper retained for historical tests and non-Gym code. New
     * protocol-1.2 paths use the Shannon-based method above.
     */
    static double calculateAirGroundTransferDelay(long sizeKb,
            double nominalBandwidthMbps, double basePropagationSeconds,
            double pathLossAtOneMeterDb, double pathLossExponent,
            double additionalLossDb, Location userLocation,
            double[] uavPosition) {
        if (nominalBandwidthMbps <= 0.0 || userLocation == null
                || uavPosition == null || uavPosition.length < 3) {
            return 0.0;
        }
        double distanceMeters = distanceMeters(userLocation, uavPosition);
        double pathLossDb = pathLossAtOneMeterDb
                + 10.0 * pathLossExponent * Math.log10(distanceMeters)
                + additionalLossDb;
        double bandwidthFactor = Math.max(0.05,
                Math.min(1.0, 1.0 - pathLossDb / 160.0));
        return transferDelay(sizeKb,
                nominalBandwidthMbps * bandwidthFactor,
                Math.max(0.0, basePropagationSeconds)
                        + distanceMeters / SPEED_OF_LIGHT_METERS_PER_SECOND);
    }

    private static double distanceMeters(Location userLocation,
            double[] uavPosition) {
        if (userLocation == null || uavPosition == null
                || uavPosition.length < 3) {
            return 0.0;
        }
        double dx = uavPosition[0] - userLocation.getXPos();
        double dy = uavPosition[1] - userLocation.getYPos();
        double dz = uavPosition[2] - userLocation.getZPos();
        return Math.max(1.0, Math.sqrt(dx * dx + dy * dy + dz * dz));
    }

    private static double transferDelay(long sizeKb, double bandwidthMbps,
            double propagationSeconds) {
        if (bandwidthMbps <= 0.0) {
            return 0.0;
        }
        return Math.max(0.0, propagationSeconds)
                + Math.max(0L, sizeKb) * 8.0
                        / (bandwidthMbps * 1000.0);
    }

    private boolean isUavDeviceId(int deviceId) {
        return deviceId >= ExecutionTarget.UAV_DEVICE_ID_BASE;
    }

    public int getActiveUavTransferCount() {
        return activeUavUploads + activeUavDownloads
                + activeCloudUploads + activeCloudDownloads;
    }

    public int getActiveAccessUploadCount() {
        return activeUavUploads + activeCloudUploads;
    }

    public int getActiveAccessDownloadCount() {
        return activeUavDownloads + activeCloudDownloads;
    }

    public int getActiveBackhaulUploadCount() {
        return activeCloudUploads;
    }

    public int getActiveBackhaulDownloadCount() {
        return activeCloudDownloads;
    }

    public int getCompletedUavUploadCount() {
        return completedUavUploads;
    }

    public int getCompletedUavDownloadCount() {
        return completedUavDownloads;
    }
}
