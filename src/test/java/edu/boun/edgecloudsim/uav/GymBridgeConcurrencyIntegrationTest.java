package edu.boun.edgecloudsim.uav;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.Calendar;

import org.cloudbus.cloudsim.core.CloudSim;
import org.json.JSONObject;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import edu.boun.edgecloudsim.core.SimManager;
import edu.boun.edgecloudsim.core.SimSettings;
import edu.boun.edgecloudsim.task_generator.LoadGeneratorModel;
import edu.boun.edgecloudsim.utils.TaskProperty;

class GymBridgeConcurrencyIntegrationTest {
    @Test
    @Timeout(15)
    void decisionsFollowGlobalArrivalsWhileEarlierTasksRemainInFlight() throws Exception {
        SimSettings settings = SimSettings.getInstance();
        settings.initialize(
                "src/test/resources/config/simulation_settings.xml",
                "src/test/resources/config/edge_devices.xml",
                "src/test/resources/config/applications.xml");

        SimManager.resetInstance();
        CloudSim.init(1, Calendar.getInstance(), false);
        UAVMECScenarioFactory factory = new UAVMECScenarioFactory(
                1, "SINGLE_TIER", "RANDOM_FIT") {
            @Override
            public LoadGeneratorModel getLoadGeneratorModel() {
                return new LoadGeneratorModel(1, 60, "SINGLE_TIER") {
                    @Override
                    public void initializeModel() {
                        taskList = new ArrayList<>();
                        // Deliberately reverse the source list. SimManager must
                        // schedule the global workload by absolute arrival time.
                        taskList.add(new TaskProperty(
                                2.0, 0, 0, 1, 20_000, 100, 20));
                        taskList.add(new TaskProperty(
                                1.0, 0, 0, 1, 20_000, 100, 20));
                    }

                    @Override
                    public int getTaskTypeOfDevice(int deviceId) {
                        return 0;
                    }
                };
            }
        };
        SimManager manager = SimManager.getInstance();
        manager.initialize(factory, 1, "SINGLE_TIER", "RANDOM_FIT");
        GymDecisionCoordinator coordinator = new GymDecisionCoordinator(manager);
        manager.setGymCoordinator(coordinator);

        Thread simulationThread = new Thread(
                CloudSim::startSimulation, "gym-concurrency-test-simulation");
        simulationThread.start();
        try {
            JSONObject initial = coordinator.awaitInitialObservation(5_000);
            assertEquals(1.0 / 60.0,
                    initial.getJSONArray("time").getDouble(0), 1e-12);
            assertEquals(0, initial.getJSONArray("action_mask").getInt(0));
            assertEquals(1, initial.getJSONArray("action_mask").getInt(1));
            assertEquals(1, initial.getJSONArray("action_mask").getInt(2));
            assertEquals(0.0,
                    initial.getJSONArray("resources").getJSONArray(0).getDouble(0),
                    1e-12);
            assertTrue(initial.getJSONArray("resources").getJSONArray(1).getDouble(0) > 0.0);
            assertTrue(initial.getJSONArray("resources").getJSONArray(2).getDouble(0) > 0.0);
            assertTrue(initial.getJSONArray("resources").getJSONArray(1).getDouble(2)
                    > initial.getJSONArray("resources").getJSONArray(2).getDouble(2));

            coordinator.beginDecision(2, zeroMovements(settings.getNumOfUAVs()));
            GymDecisionCoordinator.StepResult first =
                    coordinator.awaitStepResult(5_000);

            assertFalse(first.isTerminated());
            assertFalse(first.isTruncated());
            assertEquals(2.0 / 60.0,
                    first.getObservation().getJSONArray("time").getDouble(0), 1e-12);
            assertEquals(0, first.getSettledInTransition());
            assertTrue(first.getReward() < 0.0,
                    "The interval must account for UAV hover energy even before a task settles");
            assertTrue(first.getRewardComponents().getDouble("uav_energy_ratio") > 0.0);
            assertEquals(1, coordinator.getInFlightTaskCount(),
                    "The first task must still be executing at the second arrival");
            assertEquals(1, first.getMetrics().getInt("in_flight_tasks"));
            assertEquals(1, first.getMetrics().getInt("awaiting_decision_tasks"));

            coordinator.beginDecision(2, zeroMovements(settings.getNumOfUAVs()));
            GymDecisionCoordinator.StepResult terminal =
                    coordinator.awaitStepResult(10_000);

            assertTrue(terminal.isTerminated());
            assertFalse(terminal.isTruncated());
            assertEquals(2, terminal.getSettledInTransition());
            assertEquals(2, terminal.getMetrics().getInt("settled_tasks"));
            assertEquals(0, terminal.getMetrics().getInt("unsettled_tasks"));
            assertEquals(0, coordinator.getInFlightTaskCount());
            assertEquals(2, manager.getSubmittedEdgeTaskCount());
            assertEquals(2.0,
                    terminal.getRewardComponents().getDouble("success"), 1e-12);
            assertTrue(terminal.getReward() > 1.0,
                    "Per-task bounded rewards must be summed, not clipped as one batch");

            simulationThread.join(5_000);
            assertFalse(simulationThread.isAlive());
        } finally {
            coordinator.close();
            if (CloudSim.running()) {
                CloudSim.stopSimulation();
            }
            simulationThread.join(5_000);
            SimManager.resetInstance();
        }
    }

    private double[][] zeroMovements(int numberOfUavs) {
        return new double[numberOfUavs][3];
    }
}
