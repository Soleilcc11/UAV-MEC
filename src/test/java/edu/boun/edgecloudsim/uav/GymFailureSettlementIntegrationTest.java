package edu.boun.edgecloudsim.uav;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Calendar;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;

import org.cloudbus.cloudsim.Cloudlet;
import org.cloudbus.cloudsim.Vm;
import org.cloudbus.cloudsim.core.CloudSim;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import edu.boun.edgecloudsim.core.ExecutionTarget;
import edu.boun.edgecloudsim.core.SimManager;
import edu.boun.edgecloudsim.core.SimSettings;
import edu.boun.edgecloudsim.edge_client.DefaultMobileDeviceManager;
import edu.boun.edgecloudsim.edge_client.MobileDeviceManager;
import edu.boun.edgecloudsim.edge_client.Task;
import edu.boun.edgecloudsim.edge_orchestrator.EdgeOrchestrator;
import edu.boun.edgecloudsim.mobility.MobilityModel;
import edu.boun.edgecloudsim.network.NetworkModel;
import edu.boun.edgecloudsim.task_generator.LoadGeneratorModel;
import edu.boun.edgecloudsim.utils.Location;
import edu.boun.edgecloudsim.utils.TaskProperty;

class GymFailureSettlementIntegrationTest {
    @Test
    @Timeout(20)
    void uploadRejectionsNotifyEveryCreatedGymTaskOnce() throws Exception {
        FailureFactory factory = new FailureFactory(FailureMode.UPLOAD_REJECTION);
        Harness harness = start(factory);
        try {
            harness.coordinator.awaitInitialObservation(5_000);
            int[] targets = {1, 2, 3}; // cloud, edge, UAV 0
            GymDecisionCoordinator.StepResult result = null;
            for (int target : targets) {
                harness.coordinator.beginDecision(target, zeroMovements(harness.manager));
                result = harness.coordinator.awaitStepResult(5_000);
            }

            assertTrue(result.isTerminated());
            assertEquals(3, result.getMetrics().getInt("settled_tasks"));
            assertEquals(3, result.getMetrics().getInt("failed_tasks"));
            assertTerminalFailureNotifications(factory.broker.getTasks(), 3);
        } finally {
            harness.close();
        }
    }

    @Test
    @Timeout(20)
    void vmMobilityAndDownloadFailuresEachNotifyOnce() throws Exception {
        FailureFactory factory = new FailureFactory(FailureMode.TERMINAL_PATHS);
        Harness harness = start(factory);
        try {
            harness.coordinator.awaitInitialObservation(5_000);
            int[] targets = {2, 2, 2}; // no edge VM, edge mobility, edge download
            GymDecisionCoordinator.StepResult result = null;
            for (int index = 0; index < targets.length; index++) {
                int target = targets[index];
                harness.coordinator.beginDecision(target, zeroMovements(harness.manager));
                result = harness.coordinator.awaitStepResult(5_000);
                assertEquals(1, result.getSettledInTransition(),
                        "failure path index " + index + ", metrics=" + result.getMetrics());
            }

            assertTrue(result.isTerminated());
            assertEquals(3, result.getMetrics().getInt("failed_tasks"));
            assertTerminalFailureNotifications(factory.broker.getTasks(), 3);
        } finally {
            harness.close();
        }
    }

    @Test
    @Timeout(20)
    void uavEnergyExhaustionFailsAndRemovesQueuedGymTask() throws Exception {
        FailureFactory factory = new FailureFactory(FailureMode.UAV_EXHAUSTION);
        Harness harness = start(factory);
        try {
            harness.coordinator.awaitInitialObservation(5_000);
            // One processing tick consumes 20 J of hover energy and the final
            // 0.1 J permits only a partial execution before exhaustion.
            harness.manager.getUAVManager().getUAV(0).setEnergy(20.1);
            harness.coordinator.beginDecision(3, zeroMovements(harness.manager));
            GymDecisionCoordinator.StepResult result =
                    harness.coordinator.awaitStepResult(5_000);

            assertTrue(result.isTerminated());
            assertEquals(1, result.getMetrics().getInt("failed_tasks"));
            assertEquals(0, harness.manager.getUAVManager().getActiveEdgeTaskCount());
            assertEquals(0, harness.manager.getUAVManager().getUAV(0).getTaskQueueLength());
            assertTerminalFailureNotifications(factory.broker.getTasks(), 1);
        } finally {
            harness.close();
        }
    }

    private static Harness start(FailureFactory factory) throws Exception {
        SimSettings settings = SimSettings.getInstance();
        settings.initialize(
                "src/test/resources/config/simulation_settings.xml",
                "src/test/resources/config/edge_devices.xml",
                "src/test/resources/config/applications.xml");
        SimManager.resetInstance();
        CloudSim.init(1, Calendar.getInstance(), false);
        SimManager manager = SimManager.getInstance();
        manager.initialize(factory, 1, "SINGLE_TIER", "RANDOM_FIT");
        GymDecisionCoordinator coordinator = new GymDecisionCoordinator(manager);
        manager.setGymCoordinator(coordinator);
        Thread simulation = new Thread(
                CloudSim::startSimulation, "gym-failure-settlement-simulation");
        simulation.start();
        return new Harness(manager, coordinator, simulation);
    }

    private static void assertTerminalFailureNotifications(List<Task> tasks, int expected) {
        assertEquals(expected, tasks.size());
        for (Task task : tasks) {
            assertEquals(Cloudlet.FAILED_RESOURCE_UNAVAILABLE, task.getCloudletStatus());
            assertTrue(task.isTerminalSettlementNotified());
            assertFalse(task.claimTerminalSettlementNotification(),
                    "A terminal task must not permit a second Gym callback");
        }
    }

    private static double[][] zeroMovements(SimManager manager) {
        return new double[manager.getUAVManager().getUAVs().size()][3];
    }

    private enum FailureMode {
        UPLOAD_REJECTION,
        TERMINAL_PATHS,
        UAV_EXHAUSTION
    }

    private static final class Harness implements AutoCloseable {
        private final SimManager manager;
        private final GymDecisionCoordinator coordinator;
        private final Thread simulation;

        private Harness(SimManager manager, GymDecisionCoordinator coordinator,
                Thread simulation) {
            this.manager = manager;
            this.coordinator = coordinator;
            this.simulation = simulation;
        }

        @Override
        public void close() throws Exception {
            if (CloudSim.running()) {
                CloudSim.terminateSimulation();
            }
            coordinator.close();
            simulation.join(5_000);
            assertFalse(simulation.isAlive());
            SimManager.resetInstance();
        }
    }

    private static final class FailureFactory extends UAVMECScenarioFactory {
        private final FailureMode mode;
        private RecordingMobileDeviceManager broker;

        private FailureFactory(FailureMode mode) {
            super(1, "SINGLE_TIER", "RANDOM_FIT");
            this.mode = mode;
        }

        @Override
        public LoadGeneratorModel getLoadGeneratorModel() {
            return new LoadGeneratorModel(1, 60, "SINGLE_TIER") {
                @Override
                public void initializeModel() {
                    taskList = new ArrayList<>();
                    if (mode == FailureMode.UPLOAD_REJECTION) {
                        addTasks(1.0, 2.0, 3.0);
                    } else if (mode == FailureMode.TERMINAL_PATHS) {
                        addTasks(1.0, 10.0, 45.0);
                    } else {
                        taskList.add(new TaskProperty(
                                1.0, 0, 0, 1, 1_000_000, 100, 20));
                    }
                }

                private void addTasks(Double... startTimes) {
                    Arrays.stream(startTimes).forEach(time -> taskList.add(
                            new TaskProperty(time, 0, 0, 1, 1_000, 100, 20)));
                }

                @Override
                public int getTaskTypeOfDevice(int deviceId) {
                    return 0;
                }
            };
        }

        @Override
        public MobileDeviceManager getMobileDeviceManager() throws Exception {
            broker = new RecordingMobileDeviceManager();
            return broker;
        }

        @Override
        public NetworkModel getNetworkModel() {
            if (mode == FailureMode.UPLOAD_REJECTION) {
                return new UAVMECNetworkModel(1, "SINGLE_TIER") {
                    @Override
                    public double getUploadDelay(int sourceDeviceId, int destDeviceId,
                            Task task) {
                        return 0.0;
                    }
                };
            }
            if (mode == FailureMode.TERMINAL_PATHS) {
                return new UAVMECNetworkModel(1, "SINGLE_TIER") {
                    @Override
                    public double getUploadDelay(int sourceDeviceId, int destDeviceId,
                            Task task) {
                        return 0.01;
                    }

                    @Override
                    public double getDownloadDelay(int sourceDeviceId, int destDeviceId,
                            Task task) {
                        return task.getCloudletId() == 3 ? 0.0 : 1.0;
                    }
                };
            }
            return super.getNetworkModel();
        }

        @Override
        public MobilityModel getMobilityModel() {
            if (mode != FailureMode.TERMINAL_PATHS) {
                return super.getMobilityModel();
            }
            return new MobilityModel(1, 60) {
                @Override
                public void initialize() {
                }

                @Override
                public Location getLocation(int deviceId, double time) {
                    int wlanId = time > 10.5 && time < 20.0 ? 1 : 0;
                    return new Location(0, wlanId, 500, 500);
                }
            };
        }

        @Override
        public EdgeOrchestrator getEdgeOrchestrator() {
            if (mode != FailureMode.TERMINAL_PATHS) {
                return super.getEdgeOrchestrator();
            }
            return new TaskOffloadingOrchestrator("RANDOM_FIT", "SINGLE_TIER") {
                @Override
                public Vm getVmToOffload(Task task, int deviceId) {
                    return task.getCloudletId() == 1
                            ? null : super.getVmToOffload(task, deviceId);
                }
            };
        }
    }

    private static final class RecordingMobileDeviceManager
            extends DefaultMobileDeviceManager {
        private final List<Task> tasks = new CopyOnWriteArrayList<>();

        private RecordingMobileDeviceManager() throws Exception {
        }

        @Override
        public Task submitTask(TaskProperty edgeTask, ExecutionTarget forcedTarget) {
            Task task = super.submitTask(edgeTask, forcedTarget);
            tasks.add(task);
            return task;
        }

        private List<Task> getTasks() {
            return tasks;
        }
    }
}
