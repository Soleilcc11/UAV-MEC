package edu.boun.edgecloudsim.uav;

import java.util.ArrayDeque;
import java.util.Deque;

import org.cloudbus.cloudsim.Cloudlet;
import org.cloudbus.cloudsim.core.CloudSim;
import org.json.JSONArray;
import org.json.JSONObject;

import edu.boun.edgecloudsim.core.ExecutionTarget;
import edu.boun.edgecloudsim.core.SimManager;
import edu.boun.edgecloudsim.core.SimSettings;
import edu.boun.edgecloudsim.edge_client.Task;
import edu.boun.edgecloudsim.utils.Location;
import edu.boun.edgecloudsim.utils.TaskProperty;

/** Coordinates blocking task-arrival Gymnasium decision epochs on the simulation thread. */
public class GymDecisionCoordinator {
    private final SimManager manager;
    private final Deque<TaskProperty> waitingTasks = new ArrayDeque<>();
    private TaskProperty currentTask;
    private Task activeTask;
    private Decision activeDecision;
    private JSONObject pendingRewardComponents;
    private boolean initialDecisionReady;
    private boolean stepResultReady;
    private boolean terminated;
    private boolean truncated;
    private boolean closed;
    private int totalTaskCount;
    private int settledTaskCount;
    private double lastSimulationTime;

    public GymDecisionCoordinator(SimManager manager) {
        this.manager = manager;
    }

    public synchronized void setTotalTaskCount(int totalTaskCount) {
        this.totalTaskCount = totalTaskCount;
    }

    public synchronized Decision onTaskArrival(TaskProperty task) {
        waitingTasks.addLast(task);
        if (activeTask == null && currentTask == null) {
            currentTask = waitingTasks.removeFirst();
            if (pendingRewardComponents == null) {
                initialDecisionReady = true;
            } else {
                stepResultReady = true;
            }
            notifyAll();
            return awaitDecision();
        }
        return null;
    }

    public synchronized JSONObject awaitInitialObservation(long timeoutMillis)
            throws InterruptedException {
        waitFor(() -> initialDecisionReady || terminated || closed, timeoutMillis);
        if (closed) {
            throw new IllegalStateException("Gym session is closed");
        }
        return buildObservation();
    }

    public synchronized Decision beginDecision(int targetIndex, double[][] movements) {
        if (currentTask == null || activeTask != null || activeDecision != null) {
            throw new IllegalStateException("No Gymnasium decision is pending");
        }
        int[] mask = buildActionMask(currentTask);
        if (targetIndex < 0 || targetIndex >= mask.length || mask[targetIndex] == 0) {
            throw new IllegalArgumentException("Target is disabled by the action mask");
        }
        ExecutionTarget target = mapTarget(targetIndex);
        activeDecision = new Decision(
                currentTask,
                target,
                copyMovements(movements),
                CloudSim.clock(),
                totalUavEnergyConsumed(),
                manager.getUAVManager().getBoundaryConstraintCount(),
                false,
                deadlineSeconds(currentTask));
        currentTask = null;
        initialDecisionReady = false;
        stepResultReady = false;
        notifyAll();
        return activeDecision;
    }

    public synchronized boolean onTaskSubmitted(Task task, Decision decision) {
        activeTask = task;
        if (task == null) {
            return settle(null, false);
        }
        if (task.getCloudletStatus() != Cloudlet.CREATED) {
            return settle(task, false);
        }
        int legacyTarget = decision.target.toLegacyDeviceId();
        if (decision.target.getType() != ExecutionTarget.Type.LOCAL) {
            double uploadDelay = manager.getNetworkModel().getUploadDelay(
                    task.getMobileDeviceId(), legacyTarget, task);
            decision.ueEnergyJoules = Math.max(0.0, uploadDelay) * 0.1;
        }
        return false;
    }

    public synchronized boolean onTaskSettled(Task task) {
        if (activeTask == null || task == null
                || task.getCloudletId() != activeTask.getCloudletId()) {
            return false;
        }
        return settle(task, task.getCloudletStatus() == Cloudlet.SUCCESS);
    }

    private boolean settle(Task task, boolean success) {
        lastSimulationTime = Math.max(lastSimulationTime, CloudSim.clock());
        double latencySeconds = Math.max(0.0, CloudSim.clock() - activeDecision.decisionTime);
        double uavEnergy = Math.max(0.0,
                totalUavEnergyConsumed() - activeDecision.startUavEnergy);
        int boundaryDelta = Math.max(0,
                manager.getUAVManager().getBoundaryConstraintCount()
                        - activeDecision.startBoundaryConstraints);
        double uavBudget = Math.max(1.0,
                manager.getUAVManager().getUAVs().size()
                        * manager.getSimulationSettings().getUAVMaxEnergy());

        pendingRewardComponents = new JSONObject()
                .put("success", success ? 1.0 : 0.0)
                .put("latency_ratio", latencySeconds / activeDecision.deadlineSeconds)
                .put("ue_energy_ratio", activeDecision.ueEnergyJoules / 10.0)
                .put("uav_energy_ratio", uavEnergy / uavBudget)
                .put("constraint_violations",
                        activeDecision.constraintViolation || boundaryDelta > 0 ? 1.0 : 0.0);
        activeTask = null;
        activeDecision = null;
        settledTaskCount++;

        if (settledTaskCount >= totalTaskCount) {
            terminated = true;
            stepResultReady = true;
        } else if (!waitingTasks.isEmpty()) {
            currentTask = waitingTasks.removeFirst();
            stepResultReady = true;
        }
        notifyAll();
        return currentTask != null;
    }

    private Decision awaitDecision() {
        while (activeDecision == null && !closed) {
            try {
                wait();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                closed = true;
            }
        }
        return closed ? null : activeDecision;
    }

    public synchronized StepResult awaitStepResult(long timeoutMillis) throws InterruptedException {
        waitFor(() -> stepResultReady || closed, timeoutMillis);
        if (closed) {
            throw new IllegalStateException("Gym session is closed");
        }
        JSONObject reward = pendingRewardComponents == null
                ? zeroRewardComponents()
                : pendingRewardComponents;
        StepResult result = new StepResult(
                buildObservation(), reward, terminated, truncated,
                new JSONObject()
                        .put("settled_tasks", settledTaskCount)
                        .put("total_tasks", totalTaskCount)
                        .put("simulation_time", Math.max(lastSimulationTime, CloudSim.clock())));
        pendingRewardComponents = null;
        stepResultReady = false;
        return result;
    }

    public synchronized void onSimulationEnded() {
        lastSimulationTime = CloudSim.clock();
        if (activeDecision != null) {
            settle(activeTask, false);
        }
        if (!terminated) {
            truncated = true;
            stepResultReady = true;
        }
        notifyAll();
    }

    public synchronized void close() {
        closed = true;
        notifyAll();
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

        JSONArray task = new JSONArray()
                .put(normalize(currentTask.getInputFileSize(), maxColumn(table, 5)))
                .put(normalize(currentTask.getOutputFileSize(), maxColumn(table, 6)))
                .put(normalize(currentTask.getLength(), maxColumn(table, 7)))
                .put(normalize(currentTask.getPesNumber(), maxColumn(table, 8)))
                .put(normalize(deadline, maxColumn(table, 13)))
                .put(normalize(user.getXPos(), settings.getSimulationSpace()[0]))
                .put(normalize(user.getYPos(), settings.getSimulationSpace()[1]));

        JSONArray resources = new JSONArray()
                .put(new JSONArray().put(0.0).put(0.0).put(0.0))
                .put(new JSONArray().put(1.0).put(0.0)
                        .put(normalize(settings.getWanPropagationDelay(), 1.0)))
                .put(new JSONArray().put(1.0).put(0.0)
                        .put(normalize(settings.getInternalLanDelay(), 1.0)));

        JSONArray uavs = new JSONArray();
        for (UAV uav : manager.getUAVManager().getUAVs()) {
            double[] position = uav.getPosition();
            double upload = UAVMECNetworkModel.calculateAirGroundTransferDelay(
                    currentTask.getInputFileSize(), settings.getWlanBandwidth(),
                    settings.getInternalLanDelay(), settings.getUAVPathLossParameter(),
                    settings.getUAVPathLossExponent(), settings.getUAVAdditionalPathLoss(),
                    user, position);
            double download = UAVMECNetworkModel.calculateAirGroundTransferDelay(
                    currentTask.getOutputFileSize(), settings.getWlanBandwidth(),
                    settings.getInternalLanDelay(), settings.getUAVPathLossParameter(),
                    settings.getUAVPathLossExponent(), settings.getUAVAdditionalPathLoss(),
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
                .put("task", task)
                .put("resources", resources)
                .put("uavs", uavs)
                .put("action_mask", mask);
    }

    private int[] buildActionMask(TaskProperty task) {
        int[] mask = new int[3 + manager.getUAVManager().getUAVs().size()];
        mask[0] = 0; // Local execution is not connected to a mobile VM in this scenario yet.
        mask[1] = 1;
        mask[2] = 1;
        for (UAV uav : manager.getUAVManager().getUAVs()) {
            mask[3 + uav.getId()] = uav.getEnergy() > 0 && uav.getTaskQueueLength() < 50 ? 1 : 0;
        }
        return mask;
    }

    private ExecutionTarget mapTarget(int targetIndex) {
        if (targetIndex == 0) return ExecutionTarget.local();
        if (targetIndex == 1) return ExecutionTarget.cloud();
        if (targetIndex == 2) return ExecutionTarget.edge();
        int uavId = targetIndex - 3;
        if (uavId < 0 || uavId >= manager.getUAVManager().getUAVs().size()) {
            return ExecutionTarget.local();
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

    private JSONObject zeroObservation(int uavCount) {
        JSONArray uavs = new JSONArray();
        for (int i = 0; i < uavCount; i++) {
            uavs.put(zeroArray(8));
        }
        return new JSONObject()
                .put("time", new JSONArray().put(1.0))
                .put("task", zeroArray(7))
                .put("resources", new JSONArray().put(zeroArray(3)).put(zeroArray(3)).put(zeroArray(3)))
                .put("uavs", uavs)
                .put("action_mask", zeroArray(3 + uavCount));
    }

    private JSONArray zeroArray(int size) {
        JSONArray values = new JSONArray();
        for (int i = 0; i < size; i++) values.put(0);
        return values;
    }

    private JSONObject zeroRewardComponents() {
        return new JSONObject().put("success", 0.0).put("latency_ratio", 0.0)
                .put("ue_energy_ratio", 0.0).put("uav_energy_ratio", 0.0)
                .put("constraint_violations", 0.0);
    }

    private double maxColumn(double[][] table, int column) {
        double max = 1.0;
        for (double[] row : table) {
            if (row.length > column) max = Math.max(max, row[column]);
        }
        return max;
    }

    private double normalize(double value, double maximum) {
        return maximum <= 0 ? 0.0 : Math.max(0.0, Math.min(1.0, value / maximum));
    }

    private double[][] copyMovements(double[][] movements) {
        int expected = manager.getUAVManager().getUAVs().size();
        if (movements == null || movements.length != expected) {
            throw new IllegalArgumentException("Movement row count must equal UAV count");
        }
        double[][] copy = new double[expected][3];
        for (int i = 0; i < expected; i++) {
            if (movements[i] == null || movements[i].length != 3) {
                throw new IllegalArgumentException("Each movement must contain x, y, and z");
            }
            for (int j = 0; j < 3; j++) {
                copy[i][j] = Math.max(-1.0, Math.min(1.0, movements[i][j])) * 5.0;
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

    public static final class Decision {
        private final TaskProperty task;
        private final ExecutionTarget target;
        private final double[][] movements;
        private final double decisionTime;
        private final double startUavEnergy;
        private final int startBoundaryConstraints;
        private final boolean constraintViolation;
        private final double deadlineSeconds;
        private double ueEnergyJoules;

        private Decision(TaskProperty task, ExecutionTarget target, double[][] movements,
                double decisionTime, double startUavEnergy, int startBoundaryConstraints,
                boolean constraintViolation, double deadlineSeconds) {
            this.task = task;
            this.target = target;
            this.movements = movements;
            this.decisionTime = decisionTime;
            this.startUavEnergy = startUavEnergy;
            this.startBoundaryConstraints = startBoundaryConstraints;
            this.constraintViolation = constraintViolation;
            this.deadlineSeconds = deadlineSeconds;
        }

        public TaskProperty getTask() { return task; }
        public ExecutionTarget getTarget() { return target; }
        public double[][] getMovements() { return movements; }
    }

    public static final class StepResult {
        private final JSONObject observation;
        private final JSONObject rewardComponents;
        private final boolean terminated;
        private final boolean truncated;
        private final JSONObject metrics;

        private StepResult(JSONObject observation, JSONObject rewardComponents,
                boolean terminated, boolean truncated, JSONObject metrics) {
            this.observation = observation;
            this.rewardComponents = rewardComponents;
            this.terminated = terminated;
            this.truncated = truncated;
            this.metrics = metrics;
        }

        public JSONObject getObservation() { return observation; }
        public JSONObject getRewardComponents() { return rewardComponents; }
        public boolean isTerminated() { return terminated; }
        public boolean isTruncated() { return truncated; }
        public JSONObject getMetrics() { return metrics; }
    }
}
