package edu.boun.edgecloudsim.uav;

import java.util.Calendar;

import org.cloudbus.cloudsim.core.CloudSim;
import org.json.JSONArray;
import org.json.JSONObject;

import edu.boun.edgecloudsim.core.SimManager;
import edu.boun.edgecloudsim.core.SimSettings;

/** Owns one resettable EdgeCloudSim episode for the GymBridge protocol. */
public class GymBridgeSession implements AutoCloseable {
    public static final String PROTOCOL_VERSION = "1.0";
    private final String settingsPath;
    private final String edgeDevicesPath;
    private final String applicationsPath;
    private final int mobileDeviceCount;
    private SimManager manager;
    private GymDecisionCoordinator coordinator;
    private Thread simulationThread;
    private boolean settingsInitialized;

    public GymBridgeSession(String settingsPath, String edgeDevicesPath,
            String applicationsPath, int mobileDeviceCount) {
        this.settingsPath = settingsPath;
        this.edgeDevicesPath = edgeDevicesPath;
        this.applicationsPath = applicationsPath;
        this.mobileDeviceCount = mobileDeviceCount;
    }

    public synchronized JSONObject reset(long seed, long timeoutMillis) throws Exception {
        closeEpisode();
        SimManager.resetInstance();
        SimSettings settings = SimSettings.getInstance();
        settings.initialize(settingsPath, edgeDevicesPath, applicationsPath);
        settingsInitialized = true;
        settings.setSimulationSeed(seed);
        CloudSim.init(1, Calendar.getInstance(), false);

        String scenario = settings.getSimulationScenarios()[0];
        String policy = settings.getOrchestratorPolicies()[0];
        UAVMECScenarioFactory factory = new UAVMECScenarioFactory(
                mobileDeviceCount, scenario, policy);
        manager = SimManager.getInstance();
        manager.initialize(factory, mobileDeviceCount, scenario, policy);
        coordinator = new GymDecisionCoordinator(manager);
        manager.setGymCoordinator(coordinator);

        simulationThread = new Thread(CloudSim::startSimulation, "edgecloudsim-gym-episode");
        simulationThread.setDaemon(true);
        simulationThread.start();

        JSONObject observation = coordinator.awaitInitialObservation(timeoutMillis);
        return new JSONObject()
                .put("observation", observation)
                .put("info", new JSONObject().put("seed", seed));
    }

    public synchronized JSONObject step(int target, double[][] movement, long timeoutMillis)
            throws Exception {
        ensureActive();
        GymDecisionCoordinator.Decision decision = coordinator.beginDecision(target, movement);
        manager.applyGymDecision(decision);
        if (CloudSim.isPaused()) {
            CloudSim.resumeSimulation();
        }
        GymDecisionCoordinator.StepResult result = coordinator.awaitStepResult(timeoutMillis);
        return new JSONObject()
                .put("observation", result.getObservation())
                .put("reward_components", result.getRewardComponents())
                .put("terminated", result.isTerminated())
                .put("truncated", result.isTruncated())
                .put("metrics", result.getMetrics());
    }

    public synchronized JSONObject specification() {
        ensureSettingsInitialized();
        int uavCount = SimSettings.getInstance().getNumOfUAVs();
        return new JSONObject()
                .put("protocol_version", PROTOCOL_VERSION)
                .put("decision_cadence", "task_arrival")
                .put("number_of_uavs", uavCount)
                .put("action", new JSONObject()
                        .put("target_count", 3 + uavCount)
                        .put("movement_shape", new JSONArray().put(uavCount).put(3))
                        .put("movement_low", -1.0)
                        .put("movement_high", 1.0))
                .put("observation", new JSONObject()
                        .put("time_shape", new JSONArray().put(1))
                        .put("task_shape", new JSONArray().put(7))
                        .put("resources_shape", new JSONArray().put(3).put(3))
                        .put("uavs_shape", new JSONArray().put(uavCount).put(8))
                        .put("action_mask_shape", new JSONArray().put(3 + uavCount)));
    }

    @Override
    public synchronized void close() {
        closeEpisode();
    }

    private void ensureActive() {
        if (manager == null || coordinator == null || simulationThread == null) {
            throw new IllegalStateException("reset must be called before step");
        }
    }

    private void ensureSettingsInitialized() {
        if (!settingsInitialized) {
            SimSettings.getInstance().initialize(
                    settingsPath, edgeDevicesPath, applicationsPath);
            settingsInitialized = true;
        }
    }

    private void closeEpisode() {
        if (coordinator != null) coordinator.close();
        if (CloudSim.isPaused()) CloudSim.resumeSimulation();
        if (CloudSim.running()) CloudSim.stopSimulation();
        if (simulationThread != null) {
            try {
                simulationThread.join(2000);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }
        manager = null;
        coordinator = null;
        simulationThread = null;
    }
}
