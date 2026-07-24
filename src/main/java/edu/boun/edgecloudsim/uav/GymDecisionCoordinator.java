package edu.boun.edgecloudsim.uav;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.cloudbus.cloudsim.Cloudlet;
import org.cloudbus.cloudsim.Vm;
import org.cloudbus.cloudsim.core.CloudSim;
import org.json.JSONArray;
import org.json.JSONObject;

import edu.boun.edgecloudsim.core.ExecutionTarget;
import edu.boun.edgecloudsim.core.SimManager;
import edu.boun.edgecloudsim.core.SimSettings;
import edu.boun.edgecloudsim.edge_client.Task;
import edu.boun.edgecloudsim.utils.Location;
import edu.boun.edgecloudsim.utils.TaskProperty;

/**
 * Coordinates task-arrival Gymnasium decision epochs without serializing task execution.
 * The CloudSim thread blocks only while the current arrival waits for its one action;
 * submitted tasks are then tracked independently until their terminal callbacks arrive.
 */
public class GymDecisionCoordinator {
    private final SimManager manager;
    private final Map<Integer, Decision> inFlightDecisions = new HashMap<>();
    private final TransitionAccumulator transition = new TransitionAccumulator();
    private TaskProperty currentTask;
    private Decision decisionForSimulation;
    private boolean initialDecisionReady;
    private boolean stepResultReady;
    private boolean firstDecisionPresented;
    private boolean terminated;
    private boolean truncated;
    private boolean closed;
    private StepResult frozenTerminalResult;
    private int totalTaskCount;
    private int settledTaskCount;
    private int successfulTaskCount;
    private double lastSimulationTime;
    private double lastMovementDecisionTime;
    private double lastStepObservationTime;
    private double lastReportedUavEnergy;
    private int lastReportedBoundaryConstraints;
    private int lastSelectedCloudRelayUavId = -1;

    public GymDecisionCoordinator(SimManager manager) {
        this.manager = manager;
    }

    public synchronized void setTotalTaskCount(int totalTaskCount) {
        this.totalTaskCount = totalTaskCount;
    }

    public synchronized Decision onTaskArrival(TaskProperty task) {
        if (closed) {
            return null;
        }
        if (task == null) {
            throw new IllegalArgumentException("Gym task arrival cannot be null");
        }
        if (currentTask != null || decisionForSimulation != null) {
            throw new IllegalStateException("A previous Gym decision is still pending");
        }

        currentTask = task;
        if (!firstDecisionPresented) {
            firstDecisionPresented = true;
            initialDecisionReady = true;
        } else {
            // The simulation thread stops at the next arrival, after all events
            // before this timestamp have contributed to the transition window.
            stepResultReady = true;
        }
        notifyAll();
        return awaitDecision();
    }

    public synchronized JSONObject awaitInitialObservation(long timeoutMillis)
            throws InterruptedException {
        waitFor(() -> initialDecisionReady || terminated || truncated || closed,
                timeoutMillis);
        if (closed) {
            throw new IllegalStateException("Gym session is closed");
        }
        if (!initialDecisionReady) {
            throw new IllegalStateException(
                    "Gym episode ended before the first task-arrival decision");
        }
        lastStepObservationTime = CloudSim.clock();
        return buildObservation();
    }

    public synchronized Decision beginDecision(int targetIndex, double[][] movements) {
        if (currentTask == null || decisionForSimulation != null) {
            throw new IllegalStateException("No Gymnasium decision is pending");
        }
        int[] mask = buildActionMask(currentTask);
        if (targetIndex < 0 || targetIndex >= mask.length) {
            throw new IllegalArgumentException("Unknown Gym execution target: " + targetIndex);
        }
        boolean constraintViolation = mask[targetIndex] == 0;
        ExecutionTarget target = mapTarget(targetIndex);
        lastSelectedCloudRelayUavId = -1;
        double decisionTime = CloudSim.clock();
        double movementElapsedSeconds = Math.max(
                0.0, decisionTime - lastMovementDecisionTime);
        decisionForSimulation = new Decision(
                currentTask,
                target,
                copyMovements(movements, movementElapsedSeconds),
                movementElapsedSeconds,
                decisionTime,
                deadlineSeconds(currentTask),
                constraintViolation);
        lastMovementDecisionTime = decisionTime;
        currentTask = null;
        initialDecisionReady = false;
        stepResultReady = false;
        notifyAll();
        return decisionForSimulation;
    }

    public synchronized void onTaskSubmitted(Task task, Decision decision) {
        if (decision == null) {
            throw new IllegalArgumentException("Submitted Gym task is missing its decision");
        }
        if (task == null) {
            settle(decision, false, true);
            return;
        }
        if (task.getCloudletStatus() != Cloudlet.CREATED) {
            settle(decision, false, true);
            return;
        }
        int legacyTarget = decision.target.toLegacyDeviceId();
        if (decision.target.getType() != ExecutionTarget.Type.LOCAL) {
            double uploadDelay = manager.getNetworkModel().getUploadDelay(
                    task.getMobileDeviceId(), legacyTarget, task);
            decision.ueEnergyJoules = Math.max(0.0, uploadDelay) * 0.1;
        }
        if (decision.target.getType() == ExecutionTarget.Type.CLOUD) {
            lastSelectedCloudRelayUavId = task.getCloudRelayUavId();
        }
        Decision previous = inFlightDecisions.put(task.getCloudletId(), decision);
        if (previous != null) {
            throw new IllegalStateException(
                    "Duplicate in-flight cloudlet id: " + task.getCloudletId());
        }
    }

    public synchronized boolean onTaskSettled(Task task) {
        if (task == null) {
            return false;
        }
        Decision decision = inFlightDecisions.remove(task.getCloudletId());
        if (decision == null) {
            return false;
        }
        settle(decision, task.getCloudletStatus() == Cloudlet.SUCCESS, true);
        return true;
    }

    private void settle(Decision decision, boolean success, boolean allowNaturalTermination) {
        lastSimulationTime = Math.max(lastSimulationTime, CloudSim.clock());
        double latencySeconds = Math.max(0.0, CloudSim.clock() - decision.decisionTime);
        boolean deadlineSuccess = success
                && latencySeconds <= decision.deadlineSeconds;
        double successComponent = deadlineSuccess ? 1.0 : 0.0;
        double latencyRatio = latencySeconds / decision.deadlineSeconds;
        double ueEnergyRatio = decision.ueEnergyJoules / 10.0;
        transition.add(
                successComponent,
                latencyRatio,
                ueEnergyRatio,
                latencySeconds,
                decision.deadlineSeconds,
                decision.ueEnergyJoules,
                decision.constraintViolation ? 1 : 0);
        settledTaskCount++;
        if (deadlineSuccess) {
            successfulTaskCount++;
        }

        if (allowNaturalTermination
                && totalTaskCount > 0 && settledTaskCount >= totalTaskCount) {
            terminated = true;
            stepResultReady = true;
            // Freeze every terminal value while the CloudSim thread still owns
            // the exact event boundary.  The socket thread must never race the
            // subsequent one-second UAV ticks when it constructs the response.
            frozenTerminalResult = createStepResult();
            if (CloudSim.running()) {
                // CloudSim 4.0 only observes this flag after completing the
                // current timestamp batch, so same-time events retain their
                // deterministic ordering while no later tick can add energy.
                CloudSim.abruptallyTerminate();
            }
        }
        notifyAll();
    }

    private Decision awaitDecision() {
        while (decisionForSimulation == null && !closed) {
            try {
                wait();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                closed = true;
            }
        }
        if (closed) {
            return null;
        }
        Decision decision = decisionForSimulation;
        decisionForSimulation = null;
        return decision;
    }

    public synchronized StepResult awaitStepResult(long timeoutMillis) throws InterruptedException {
        waitFor(() -> stepResultReady || closed, timeoutMillis);
        if (closed) {
            throw new IllegalStateException("Gym session is closed");
        }
        if (frozenTerminalResult != null) {
            StepResult result = frozenTerminalResult;
            frozenTerminalResult = null;
            stepResultReady = false;
            return result;
        }
        StepResult result = createStepResult();
        stepResultReady = false;
        return result;
    }

    private StepResult createStepResult() {
        double totalUavEnergy = totalUavEnergyConsumed();
        double intervalUavEnergy = Math.max(0.0,
                totalUavEnergy - lastReportedUavEnergy);
        lastReportedUavEnergy = totalUavEnergy;
        int totalBoundaryConstraints = manager.getUAVManager().getBoundaryConstraintCount();
        int intervalBoundaryConstraints = Math.max(0,
                totalBoundaryConstraints - lastReportedBoundaryConstraints);
        lastReportedBoundaryConstraints = totalBoundaryConstraints;
        double uavBudget = Math.max(1.0,
                manager.getUAVManager().getUAVs().size()
                        * manager.getSimulationSettings().getUAVMaxEnergy());
        TransitionSnapshot snapshot = transition.drain(
                intervalUavEnergy, uavBudget, intervalBoundaryConstraints);
        JSONObject metrics = snapshot.physicalMetrics;
        double simulationTime = Math.max(lastSimulationTime, CloudSim.clock());
        double elapsedSimulationTime = Math.max(
                0.0, simulationTime - lastStepObservationTime);
        lastStepObservationTime = simulationTime;
        SimSettings settings = manager.getSimulationSettings();
        double discountTimeUnit = Math.max(
                Double.MIN_NORMAL, settings.getSmdpDiscountTimeUnitSeconds());
        double effectiveDiscount = Math.pow(
                settings.getSmdpDiscountBase(),
                elapsedSimulationTime / discountTimeUnit);
        metrics = withEpisodeProgress(metrics, settledTaskCount, totalTaskCount,
                simulationTime);
        int awaitingDecision = currentTask == null ? 0 : 1;
        int inFlight = inFlightDecisions.size();
        int unsettled = Math.max(0, totalTaskCount - settledTaskCount);
        int notArrived = Math.max(0,
                totalTaskCount - settledTaskCount - inFlight - awaitingDecision);
        metrics.put("successful_tasks", successfulTaskCount)
                .put("throughput_tasks_per_second", simulationTime <= 0.0
                        ? 0.0 : successfulTaskCount / simulationTime)
                .put("failed_tasks", settledTaskCount - successfulTaskCount)
                .put("settled_in_transition", snapshot.settledCount)
                .put("in_flight_tasks", inFlight)
                .put("awaiting_decision_tasks", awaitingDecision)
                .put("not_arrived_tasks", notArrived)
                .put("unsettled_tasks", unsettled)
                .put("elapsed_simulation_time", elapsedSimulationTime)
                .put("effective_discount", effectiveDiscount)
                .put("selected_cloud_relay_uav", lastSelectedCloudRelayUavId)
                .put("uav_queue_length_total",
                        manager.getUAVManager().getUAVs().stream()
                                .mapToInt(UAV::getTaskQueueLength).sum())
                .put("uav_queue_length_max",
                        manager.getUAVManager().getUAVs().stream()
                                .mapToInt(UAV::getTaskQueueLength).max().orElse(0))
                .put("local_resource_utilization", averageLocalUtilization())
                .put("cloud_resource_utilization",
                        averageVmUtilization(resourceVms(
                                ExecutionTarget.cloud(), 0)))
                .put("uav_resource_utilization", averageUavUtilization());
        if (manager.getNetworkModel() instanceof UAVMECNetworkModel) {
            UAVMECNetworkModel network =
                    (UAVMECNetworkModel) manager.getNetworkModel();
            metrics.put("active_access_uploads",
                    network.getActiveAccessUploadCount())
                    .put("active_access_downloads",
                            network.getActiveAccessDownloadCount())
                    .put("active_backhaul_uploads",
                            network.getActiveBackhaulUploadCount())
                    .put("active_backhaul_downloads",
                            network.getActiveBackhaulDownloadCount())
                    .put("access_channel_count",
                            settings.getUAVAccessChannelCount())
                    .put("access_bandwidth_mbps",
                            settings.getUAVAccessBandwidthMbps())
                    .put("backhaul_bandwidth_mbps",
                            settings.getUAVCloudBandwidthMbps());
        }
        StepResult result = new StepResult(
                buildObservation(), snapshot.rewardComponents, snapshot.reward,
                snapshot.settledCount, terminated, truncated, metrics);
        return result;
    }

    public synchronized void onSimulationEnded() {
        lastSimulationTime = CloudSim.clock();
        if (terminated) {
            stepResultReady = true;
            notifyAll();
            return;
        }
        if (!inFlightDecisions.isEmpty()) {
            List<Decision> unfinished = new ArrayList<>(inFlightDecisions.values());
            inFlightDecisions.clear();
            for (Decision decision : unfinished) {
                settle(decision, false, false);
            }
        }
        // Reaching the configured simulation horizon is a truncation even when
        // failing the remaining in-flight work brings settled == total.
        terminated = false;
        truncated = true;
        stepResultReady = true;
        frozenTerminalResult = createStepResult();
        notifyAll();
    }

    public synchronized void close() {
        closed = true;
        currentTask = null;
        decisionForSimulation = null;
        initialDecisionReady = false;
        stepResultReady = false;
        frozenTerminalResult = null;
        notifyAll();
    }

    synchronized int getInFlightTaskCount() {
        return inFlightDecisions.size();
    }

    public synchronized JSONObject buildObservation() {
        int uavCount = manager.getUAVManager().getUAVs().size();
        if (currentTask == null) {
            return zeroObservation(uavCount);
        }
        SimSettings settings = manager.getSimulationSettings();
        Location user = manager.getMobilityModel().getLocation(
                currentTask.getMobileDeviceId(), CloudSim.clock());
        double[][] table = settings.getTaskLookUpTable();
        double deadline = deadlineSeconds(currentTask);
        double deltaTime = Math.max(
                0.0, CloudSim.clock() - lastMovementDecisionTime);

        JSONArray task = new JSONArray()
                .put(normalize(currentTask.getInputFileSize(), exponentialQ99Scale(table, 5)))
                .put(normalize(currentTask.getOutputFileSize(), exponentialQ99Scale(table, 6)))
                .put(normalize(currentTask.getLength(), exponentialQ99Scale(table, 7)))
                .put(normalize(currentTask.getPesNumber(), maxColumn(table, 8)))
                .put(normalize(deadline, maxColumn(table, 13)))
                .put(normalize(user.getXPos(), settings.getSimulationSpace()[0]))
                .put(normalize(user.getYPos(), settings.getSimulationSpace()[1]));

        ResourceSnapshot local = resourceSnapshot(ExecutionTarget.local(), currentTask);
        ResourceSnapshot cloud = resourceSnapshot(ExecutionTarget.cloud(), currentTask);
        JSONArray resources = new JSONArray()
                .put(local.toJson())
                .put(cloud.toJson());

        JSONArray uavs = new JSONArray();
        int activeUploads = 1;
        int activeDownloads = 1;
        if (manager.getNetworkModel() instanceof UAVMECNetworkModel) {
            UAVMECNetworkModel network =
                    (UAVMECNetworkModel) manager.getNetworkModel();
            activeUploads += network.getActiveAccessUploadCount();
            activeDownloads += network.getActiveAccessDownloadCount();
        }
        for (UAV uav : manager.getUAVManager().getUAVs()) {
            double[] position = uav.getPosition();
            double upload = UAVMECNetworkModel.calculateProbabilisticAirGroundTransferDelay(
                    currentTask.getInputFileSize(), settings.getUAVAccessBandwidthMbps(),
                    settings.getInternalLanDelay(), settings.getUAVCarrierFrequencyHz(),
                    settings.getUAVLoSA(), settings.getUAVLoSB(),
                    settings.getUAVEtaLoSDb(), settings.getUAVEtaNLoSDb(),
                    settings.getUAVTransmitPowerWatts(),
                    settings.getUAVNoisePsdDbmPerHz(),
                    settings.getUAVAccessChannelCount(), activeUploads,
                    user, position);
            double download = UAVMECNetworkModel.calculateProbabilisticAirGroundTransferDelay(
                    currentTask.getOutputFileSize(), settings.getUAVAccessBandwidthMbps(),
                    settings.getInternalLanDelay(), settings.getUAVCarrierFrequencyHz(),
                    settings.getUAVLoSA(), settings.getUAVLoSB(),
                    settings.getUAVEtaLoSDb(), settings.getUAVEtaNLoSDb(),
                    settings.getUAVTransmitPowerWatts(),
                    settings.getUAVNoisePsdDbmPerHz(),
                    settings.getUAVAccessChannelCount(), activeDownloads,
                    user, position);
            uavs.put(new JSONArray()
                    .put(normalize(position[0], settings.getSimulationSpace()[0]))
                    .put(normalize(position[1], settings.getSimulationSpace()[1]))
                    .put(normalize(position[2], settings.getSimulationSpace()[2]))
                    .put(normalize(uav.getEnergy(), settings.getUAVMaxEnergy()))
                    .put(normalize(uav.getTaskQueueLength(), 50.0))
                    .put(normalize(uav.getProcessingCapacity(), 2500.0))
                    .put(normalize(upload, deadline))
                    .put(normalize(download, deadline)));
        }

        JSONArray mask = new JSONArray();
        for (int value : buildActionMask(currentTask)) {
            mask.put(value);
        }
        return new JSONObject()
                .put("time", new JSONArray().put(normalize(CloudSim.clock(), settings.getSimulationTime())))
                .put("delta_time", new JSONArray().put(
                        normalize(deltaTime, settings.getSimulationTime())))
                .put("task", task)
                .put("resources", resources)
                .put("uavs", uavs)
                .put("action_mask", mask);
    }

    private int[] buildActionMask(TaskProperty task) {
        int[] mask = new int[2 + manager.getUAVManager().getUAVs().size()];
        mask[0] = resourceSnapshot(ExecutionTarget.local(), task).available ? 1 : 0;
        mask[1] = resourceSnapshot(ExecutionTarget.cloud(), task).available ? 1 : 0;
        for (UAV uav : manager.getUAVManager().getUAVs()) {
            mask[2 + uav.getId()] = uav.getEnergy() > 0 && uav.getTaskQueueLength() < 50 ? 1 : 0;
        }
        return mask;
    }

    private ExecutionTarget mapTarget(int targetIndex) {
        if (targetIndex == 0) return ExecutionTarget.local();
        if (targetIndex == 1) return ExecutionTarget.cloud();
        int uavId = targetIndex - 2;
        if (uavId < 0 || uavId >= manager.getUAVManager().getUAVs().size()) {
            throw new IllegalArgumentException("Unknown Gym execution target: " + targetIndex);
        }
        return ExecutionTarget.uav(uavId);
    }

    private double deadlineSeconds(TaskProperty task) {
        double[][] table = manager.getSimulationSettings().getTaskLookUpTable();
        int type = task.getTaskType();
        if (type >= 0 && type < table.length && table[type].length > 13 && table[type][13] > 0) {
            return table[type][13];
        }
        return Math.max(1.0, manager.getSimulationSettings().getSimulationTime());
    }

    private double totalUavEnergyConsumed() {
        return manager.getUAVManager().getUAVs().stream()
                .mapToDouble(UAV::getTotalEnergyConsumed).sum();
    }

    private ResourceSnapshot resourceSnapshot(ExecutionTarget target, TaskProperty task) {
        List<? extends Vm> candidates = resourceVms(target, task.getMobileDeviceId());
        double totalCapacity = 0.0;
        double availableCapacity = 0.0;
        boolean eligible = false;
        for (Vm vm : candidates) {
            if (vm == null || vm.getHost() == null) {
                continue;
            }
            double capacity = Math.max(0.0, vm.getMips())
                    * Math.max(1, vm.getNumberOfPes());
            double load = normalizeUtilization(
                    vm.getCloudletScheduler().getTotalUtilizationOfCpu(CloudSim.clock()));
            totalCapacity += capacity;
            availableCapacity += capacity * (1.0 - load);
            if (vm.getNumberOfPes() >= task.getPesNumber() && load < 1.0) {
                eligible = true;
            }
        }
        double capacityRatio = totalCapacity <= 0.0
                ? 0.0 : availableCapacity / totalCapacity;
        double loadRatio = totalCapacity <= 0.0
                ? 0.0 : 1.0 - capacityRatio;
        double linkDelay = normalizedFixedResourceDelay(target, task);
        boolean relayAvailable = target.getType() != ExecutionTarget.Type.CLOUD
                || manager.getUAVManager().getUAVs().stream()
                        .anyMatch(uav -> uav.getEnergy() > 0.0);
        return new ResourceSnapshot(
                eligible && capacityRatio > 0.0 && relayAvailable,
                normalize(capacityRatio, 1.0),
                normalize(loadRatio, 1.0),
                linkDelay);
    }

    private List<? extends Vm> resourceVms(ExecutionTarget target, int mobileDeviceId) {
        List<Vm> result = new ArrayList<>();
        switch (target.getType()) {
            case LOCAL:
                List<? extends Vm> mobile = manager.getMobileServerManager()
                        .getVmList(mobileDeviceId);
                if (mobile != null) result.addAll(mobile);
                break;
            case CLOUD:
                for (int host = 0;
                        host < manager.getSimulationSettings().getNumOfCloudHost(); host++) {
                    List<? extends Vm> vms = manager.getCloudServerManager().getVmList(host);
                    if (vms != null) result.addAll(vms);
                }
                break;
            case EDGE:
                for (int host = 0;
                        host < manager.getSimulationSettings().getNumOfEdgeHosts(); host++) {
                    List<? extends Vm> vms = manager.getEdgeServerManager().getVmList(host);
                    if (vms != null) result.addAll(vms);
                }
                break;
            default:
                break;
        }
        return result;
    }

    private double normalizedFixedResourceDelay(
            ExecutionTarget target, TaskProperty task) {
        if (target.getType() == ExecutionTarget.Type.LOCAL) {
            return 0.0;
        }
        SimSettings settings = manager.getSimulationSettings();
        boolean cloud = target.getType() == ExecutionTarget.Type.CLOUD;
        if (cloud) {
            return normalize(cloudTwoHopTransferDelay(task), deadlineSeconds(task));
        }
        double bandwidth = settings.getWlanBandwidth();
        double propagation = settings.getInternalLanDelay();
        double transfer = fixedTransferDelay(task.getInputFileSize(), bandwidth, propagation)
                + fixedTransferDelay(task.getOutputFileSize(), bandwidth, propagation);
        return normalize(transfer, deadlineSeconds(task));
    }

    private double cloudTwoHopTransferDelay(TaskProperty task) {
        SimSettings settings = manager.getSimulationSettings();
        Location user = manager.getMobilityModel().getLocation(
                task.getMobileDeviceId(), CloudSim.clock());
        double best = Double.POSITIVE_INFINITY;
        int concurrentUploads = 1;
        int concurrentDownloads = 1;
        int concurrentBackhaulUploads = 1;
        int concurrentBackhaulDownloads = 1;
        if (manager.getNetworkModel() instanceof UAVMECNetworkModel) {
            UAVMECNetworkModel network =
                    (UAVMECNetworkModel) manager.getNetworkModel();
            concurrentUploads += network.getActiveAccessUploadCount();
            concurrentDownloads += network.getActiveAccessDownloadCount();
            concurrentBackhaulUploads += network.getActiveBackhaulUploadCount();
            concurrentBackhaulDownloads += network.getActiveBackhaulDownloadCount();
        }
        for (UAV uav : manager.getUAVManager().getUAVs()) {
            if (uav.getEnergy() <= 0.0) {
                continue;
            }
            double access = UAVMECNetworkModel.calculateProbabilisticAirGroundTransferDelay(
                    task.getInputFileSize(), settings.getUAVAccessBandwidthMbps(),
                    settings.getInternalLanDelay(), settings.getUAVCarrierFrequencyHz(),
                    settings.getUAVLoSA(), settings.getUAVLoSB(),
                    settings.getUAVEtaLoSDb(), settings.getUAVEtaNLoSDb(),
                    settings.getUAVTransmitPowerWatts(),
                    settings.getUAVNoisePsdDbmPerHz(),
                    settings.getUAVAccessChannelCount(), concurrentUploads,
                    user, uav.getPosition());
            access += UAVMECNetworkModel.calculateProbabilisticAirGroundTransferDelay(
                    task.getOutputFileSize(), settings.getUAVAccessBandwidthMbps(),
                    settings.getInternalLanDelay(), settings.getUAVCarrierFrequencyHz(),
                    settings.getUAVLoSA(), settings.getUAVLoSB(),
                    settings.getUAVEtaLoSDb(), settings.getUAVEtaNLoSDb(),
                    settings.getUAVTransmitPowerWatts(),
                    settings.getUAVNoisePsdDbmPerHz(),
                    settings.getUAVAccessChannelCount(), concurrentDownloads,
                    user, uav.getPosition());
            best = Math.min(best, access);
        }
        if (!Double.isFinite(best)) {
            return Double.POSITIVE_INFINITY;
        }
        double backhaulPropagation = settings.getUAVCloudPropagationDelay()
                + settings.getUAVCloudDistanceMeters() / 299_792_458.0;
        return best
                + fixedTransferDelay(task.getInputFileSize(),
                        settings.getUAVCloudBandwidthMbps()
                                / concurrentBackhaulUploads,
                        backhaulPropagation)
                + fixedTransferDelay(task.getOutputFileSize(),
                        settings.getUAVCloudBandwidthMbps()
                                / concurrentBackhaulDownloads,
                        backhaulPropagation);
    }

    private double averageLocalUtilization() {
        List<Vm> vms = new ArrayList<>();
        for (int device = 0; device < manager.getNumOfMobileDevice(); device++) {
            List<? extends Vm> deviceVms =
                    manager.getMobileServerManager().getVmList(device);
            if (deviceVms != null) {
                vms.addAll(deviceVms);
            }
        }
        return averageVmUtilization(vms);
    }

    private double averageVmUtilization(List<? extends Vm> vms) {
        double total = 0.0;
        int count = 0;
        for (Vm vm : vms) {
            if (vm == null || vm.getCloudletScheduler() == null) {
                continue;
            }
            total += normalizeUtilization(
                    vm.getCloudletScheduler().getTotalUtilizationOfCpu(
                            CloudSim.clock()));
            count++;
        }
        return count == 0 ? 0.0 : total / count;
    }

    private double averageUavUtilization() {
        if (manager.getUAVManager().getUAVs().isEmpty()) {
            return 0.0;
        }
        return manager.getUAVManager().getUAVs().stream()
                .mapToDouble(uav -> {
                    double capacity = Math.max(
                            Double.MIN_NORMAL, uav.getProcessingCapacity());
                    return Math.max(0.0, Math.min(1.0,
                            1.0 - uav.getAvailableCapacity() / capacity));
                })
                .average()
                .orElse(0.0);
    }

    private static double fixedTransferDelay(
            long sizeKb, double bandwidthMbps, double propagationSeconds) {
        if (bandwidthMbps <= 0.0) {
            return Double.POSITIVE_INFINITY;
        }
        return Math.max(0.0, propagationSeconds)
                + (Math.max(0L, sizeKb) * 8.0) / (bandwidthMbps * 1000.0);
    }

    private static double normalizeUtilization(double utilizationPercent) {
        if (!Double.isFinite(utilizationPercent)) {
            return 1.0;
        }
        return Math.max(0.0, Math.min(1.0, utilizationPercent / 100.0));
    }

    private JSONObject zeroObservation(int uavCount) {
        JSONArray uavs = new JSONArray();
        for (int i = 0; i < uavCount; i++) {
            uavs.put(zeroArray(8));
        }
        return new JSONObject()
                .put("time", new JSONArray().put(1.0))
                .put("delta_time", new JSONArray().put(0.0))
                .put("task", zeroArray(7))
                .put("resources", new JSONArray().put(zeroArray(3)).put(zeroArray(3)))
                .put("uavs", uavs)
                .put("action_mask", zeroArray(2 + uavCount));
    }

    private JSONArray zeroArray(int size) {
        JSONArray values = new JSONArray();
        for (int i = 0; i < size; i++) values.put(0);
        return values;
    }

    private static JSONObject zeroRewardComponents() {
        return new JSONObject().put("success", 0.0).put("latency_ratio", 0.0)
                .put("ue_energy_ratio", 0.0).put("uav_energy_ratio", 0.0)
                .put("constraint_violations", 0.0);
    }

    static JSONObject rawPhysicalMetrics(double latencySeconds, double deadlineSeconds,
            double ueEnergyJoules, double uavEnergyJoules, int constraintViolations) {
        return new JSONObject()
                .put("latency_seconds", Math.max(0.0, latencySeconds))
                .put("deadline_seconds", Math.max(0.0, deadlineSeconds))
                .put("ue_energy_joules", Math.max(0.0, ueEnergyJoules))
                .put("uav_energy_joules", Math.max(0.0, uavEnergyJoules))
                .put("constraint_violations", Math.max(0, constraintViolations));
    }

    static JSONObject withEpisodeProgress(JSONObject physicalMetrics, int settledTasks,
            int totalTasks, double simulationTime) {
        return physicalMetrics
                .put("settled_tasks", Math.max(0, settledTasks))
                .put("total_tasks", Math.max(0, totalTasks))
                .put("simulation_time", Math.max(0.0, simulationTime));
    }

    private double maxColumn(double[][] table, int column) {
        double max = 1.0;
        for (double[] row : table) {
            if (row.length > column) max = Math.max(max, row[column]);
        }
        return max;
    }

    static double exponentialQ99Scale(double[][] table, int column) {
        return -Math.log(0.01) * maxColumnValue(table, column);
    }

    private static double maxColumnValue(double[][] table, int column) {
        double max = 1.0;
        for (double[] row : table) {
            if (row.length > column) {
                max = Math.max(max, row[column]);
            }
        }
        return max;
    }

    private double normalize(double value, double maximum) {
        return maximum <= 0 ? 0.0 : Math.max(0.0, Math.min(1.0, value / maximum));
    }

    private double[][] copyMovements(double[][] movements, double elapsedSeconds) {
        int expected = manager.getUAVManager().getUAVs().size();
        if (movements == null || movements.length != expected) {
            throw new IllegalArgumentException("Movement row count must equal UAV count");
        }
        double[][] copy = new double[expected][3];
        for (int i = 0; i < expected; i++) {
            if (movements[i] == null || movements[i].length != 3) {
                throw new IllegalArgumentException("Each movement must contain x, y, and z");
            }
            double magnitudeSquared = 0.0;
            for (int j = 0; j < 3; j++) {
                double value = movements[i][j];
                if (!Double.isFinite(value) || value < -1.0 || value > 1.0) {
                    throw new IllegalArgumentException(
                            "Movement values must be finite and lie in [-1,1]");
                }
                copy[i][j] = value;
                magnitudeSquared += value * value;
            }
            double magnitude = Math.sqrt(magnitudeSquared);
            double unitScale = magnitude > 1.0 ? 1.0 / magnitude : 1.0;
            double maxDistance = manager.getUAVManager().getUAVs().get(i).getSpeed()
                    * Math.max(0.0, elapsedSeconds);
            for (int j = 0; j < 3; j++) {
                copy[i][j] *= unitScale * maxDistance;
            }
        }
        return copy;
    }

    private void waitFor(BooleanCondition condition, long timeoutMillis) throws InterruptedException {
        long deadline = System.currentTimeMillis() + timeoutMillis;
        while (!condition.evaluate()) {
            long remaining = deadline - System.currentTimeMillis();
            if (remaining <= 0) throw new IllegalStateException("Gym decision timed out");
            wait(remaining);
        }
    }

    @FunctionalInterface
    private interface BooleanCondition { boolean evaluate(); }

    private static double boundedTaskReward(double success, double latencyRatio,
            double ueEnergyRatio) {
        double reward = success
                - 0.35 * Math.min(Math.max(latencyRatio, 0.0), 2.0)
                - 0.15 * Math.min(Math.max(ueEnergyRatio, 0.0), 2.0);
        return Math.max(-1.0, Math.min(1.0, reward));
    }

    /** Accumulates completed tasks between two consecutive decision epochs. */
    private static final class TransitionAccumulator {
        private double success;
        private double latencyRatio;
        private double ueEnergyRatio;
        private double latencySeconds;
        private double deadlineSeconds;
        private double ueEnergyJoules;
        private double reward;
        private int settledCount;
        private int constraintViolations;
        private final List<Double> latencySamplesSeconds = new ArrayList<>();

        private void add(double taskSuccess, double taskLatencyRatio,
                double taskUeEnergyRatio, double taskLatencySeconds,
                double taskDeadlineSeconds, double taskUeEnergyJoules,
                int taskConstraintViolations) {
            success += taskSuccess;
            latencyRatio += taskLatencyRatio;
            ueEnergyRatio += taskUeEnergyRatio;
            latencySeconds += taskLatencySeconds;
            deadlineSeconds += taskDeadlineSeconds;
            ueEnergyJoules += taskUeEnergyJoules;
            latencySamplesSeconds.add(Math.max(0.0, taskLatencySeconds));
            constraintViolations += Math.max(0, taskConstraintViolations);
            reward += boundedTaskReward(taskSuccess, taskLatencyRatio, taskUeEnergyRatio);
            settledCount++;
        }

        private TransitionSnapshot drain(double intervalUavEnergyJoules,
                double uavBudgetJoules, int intervalConstraintViolations) {
            double uavEnergyRatio = intervalUavEnergyJoules
                    / Math.max(1.0, uavBudgetJoules);
            int totalConstraintViolations = constraintViolations
                    + Math.max(0, intervalConstraintViolations);
            double boundedConstraintViolations = totalConstraintViolations > 0 ? 1.0 : 0.0;
            JSONObject components = settledCount == 0
                    ? zeroRewardComponents()
                    : new JSONObject()
                            .put("success", success)
                            .put("latency_ratio", latencyRatio)
                            .put("ue_energy_ratio", ueEnergyRatio)
                            .put("uav_energy_ratio", uavEnergyRatio)
                            .put("constraint_violations", boundedConstraintViolations);
            if (settledCount == 0) {
                components.put("uav_energy_ratio", uavEnergyRatio)
                        .put("constraint_violations", boundedConstraintViolations);
            }
            JSONObject physical = rawPhysicalMetrics(
                    latencySeconds, deadlineSeconds, ueEnergyJoules,
                    intervalUavEnergyJoules, totalConstraintViolations);
            JSONArray latencySamples = new JSONArray();
            for (double latency : latencySamplesSeconds) {
                latencySamples.put(latency);
            }
            physical.put("settled_task_latencies_seconds", latencySamples);
            double intervalReward = reward
                    - 0.20 * Math.min(Math.max(uavEnergyRatio, 0.0), 2.0)
                    - 0.30 * Math.min(Math.max(boundedConstraintViolations, 0.0), 1.0);
            TransitionSnapshot snapshot = new TransitionSnapshot(
                    components, physical, intervalReward, settledCount);
            success = 0.0;
            latencyRatio = 0.0;
            ueEnergyRatio = 0.0;
            latencySeconds = 0.0;
            deadlineSeconds = 0.0;
            ueEnergyJoules = 0.0;
            reward = 0.0;
            settledCount = 0;
            constraintViolations = 0;
            latencySamplesSeconds.clear();
            return snapshot;
        }
    }

    private static final class ResourceSnapshot {
        private final boolean available;
        private final double capacity;
        private final double load;
        private final double linkDelay;

        private ResourceSnapshot(boolean available, double capacity,
                double load, double linkDelay) {
            this.available = available;
            this.capacity = capacity;
            this.load = load;
            this.linkDelay = linkDelay;
        }

        private JSONArray toJson() {
            return new JSONArray().put(capacity).put(load).put(linkDelay);
        }
    }

    private static final class TransitionSnapshot {
        private final JSONObject rewardComponents;
        private final JSONObject physicalMetrics;
        private final double reward;
        private final int settledCount;

        private TransitionSnapshot(JSONObject rewardComponents, JSONObject physicalMetrics,
                double reward, int settledCount) {
            this.rewardComponents = rewardComponents;
            this.physicalMetrics = physicalMetrics;
            this.reward = reward;
            this.settledCount = settledCount;
        }
    }

    public static final class Decision {
        private final TaskProperty task;
        private final ExecutionTarget target;
        private final double[][] movements;
        private final double movementElapsedSeconds;
        private final double decisionTime;
        private final double deadlineSeconds;
        private final boolean constraintViolation;
        private double ueEnergyJoules;

        private Decision(TaskProperty task, ExecutionTarget target, double[][] movements,
                double movementElapsedSeconds, double decisionTime, double deadlineSeconds,
                boolean constraintViolation) {
            this.task = task;
            this.target = target;
            this.movements = movements;
            this.movementElapsedSeconds = movementElapsedSeconds;
            this.decisionTime = decisionTime;
            this.deadlineSeconds = deadlineSeconds;
            this.constraintViolation = constraintViolation;
        }

        public TaskProperty getTask() { return task; }
        public ExecutionTarget getTarget() { return target; }
        public double[][] getMovements() { return movements; }
        public double getMovementElapsedSeconds() { return movementElapsedSeconds; }
        public boolean isConstraintViolation() { return constraintViolation; }
    }

    public static final class StepResult {
        private final JSONObject observation;
        private final JSONObject rewardComponents;
        private final double reward;
        private final int settledInTransition;
        private final boolean terminated;
        private final boolean truncated;
        private final JSONObject metrics;

        private StepResult(JSONObject observation, JSONObject rewardComponents,
                double reward, int settledInTransition, boolean terminated,
                boolean truncated, JSONObject metrics) {
            this.observation = observation;
            this.rewardComponents = rewardComponents;
            this.reward = reward;
            this.settledInTransition = settledInTransition;
            this.terminated = terminated;
            this.truncated = truncated;
            this.metrics = metrics;
        }

        public JSONObject getObservation() { return observation; }
        public JSONObject getRewardComponents() { return rewardComponents; }
        public double getReward() { return reward; }
        public int getSettledInTransition() { return settledInTransition; }
        public boolean isTerminated() { return terminated; }
        public boolean isTruncated() { return truncated; }
        public JSONObject getMetrics() { return metrics; }
    }
}
