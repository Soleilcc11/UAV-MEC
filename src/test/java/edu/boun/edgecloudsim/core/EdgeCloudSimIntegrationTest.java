package edu.boun.edgecloudsim.core;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.Calendar;

import org.cloudbus.cloudsim.Cloudlet;
import org.cloudbus.cloudsim.core.CloudSim;
import org.junit.jupiter.api.Test;

import edu.boun.edgecloudsim.uav.UAVMECScenarioFactory;
import edu.boun.edgecloudsim.uav.UAVMECNetworkModel;
import edu.boun.edgecloudsim.uav.StateActionManager;
import edu.boun.edgecloudsim.uav.RewardCalculator;
import edu.boun.edgecloudsim.uav.TaskOffloadingOrchestrator;
import edu.boun.edgecloudsim.edge_client.Task;
import edu.boun.edgecloudsim.edge_orchestrator.EdgeOrchestrator;
import edu.boun.edgecloudsim.task_generator.LoadGeneratorModel;
import edu.boun.edgecloudsim.utils.TaskProperty;

class EdgeCloudSimIntegrationTest {
    @Test
    void taskTraversesBrokerNetworkAndEdgeVm() {
        SimSettings settings = SimSettings.getInstance();
        settings.initialize(
                "src/test/resources/config/simulation_settings.xml",
                "src/test/resources/config/edge_devices.xml",
                "src/test/resources/config/applications.xml");

        CloudSim.init(1, Calendar.getInstance(), false);
        UAVMECScenarioFactory factory = new UAVMECScenarioFactory(1, "SINGLE_TIER", "RANDOM_FIT") {
            @Override
            public LoadGeneratorModel getLoadGeneratorModel() {
                return new LoadGeneratorModel(1, 60, "SINGLE_TIER") {
                    @Override
                    public void initializeModel() {
                        taskList = new ArrayList<>();
                        taskList.add(new TaskProperty(1.0, 0, 0, 1, 1000, 100, 20));
                        taskList.add(new TaskProperty(2.0, 0, 0, 1, 1000, 100, 20));
                    }

                    @Override
                    public int getTaskTypeOfDevice(int deviceId) {
                        return 0;
                    }
                };
            }

            @Override
            public EdgeOrchestrator getEdgeOrchestrator() {
                return new TaskOffloadingOrchestrator("RANDOM_FIT", "SINGLE_TIER") {
                    @Override
                    public ExecutionTarget getExecutionTarget(Task task) {
                        return task.getCloudletId() == 1
                                ? ExecutionTarget.edge()
                                : super.getExecutionTarget(task);
                    }
                };
            }
        };
        SimManager manager = SimManager.getInstance();
        manager.initialize(factory, 1, "SINGLE_TIER", "RANDOM_FIT");
        for (int i = 0; i < 9; i++) {
            manager.scheduleUavMovement(0.5, 0, new double[] {0.0, 0.0, -1000.0});
        }

        CloudSim.startSimulation();

        assertEquals(1, manager.getEdgeServerManager().getDatacenterList().size());
        assertEquals(1, manager.getEdgeServerManager().getVmList(0).size());
        assertEquals(manager.getScheduledEdgeTaskCount(), manager.getSubmittedEdgeTaskCount(),
                "Every scheduled workload task must reach SimManager");
        assertFalse(manager.getMobileDeviceManager().getCloudletReceivedList().isEmpty(),
                "At least one workload task must complete through an EdgeCloudSim VM");
        boolean completedOnEdge = manager.getMobileDeviceManager().getCloudletReceivedList().stream()
                .map(Task.class::cast)
                .anyMatch(task -> task.getExecutionTarget().getType() == ExecutionTarget.Type.EDGE);
        boolean completedOnUav = manager.getMobileDeviceManager().getCloudletReceivedList().stream()
                .map(Task.class::cast)
                .anyMatch(task -> task.getExecutionTarget().getType() == ExecutionTarget.Type.UAV
                        && task.getCloudletStatus() == Cloudlet.SUCCESS);
        assertTrue(completedOnEdge, "The Edge VM path must complete");
        assertTrue(completedOnUav, "The UAV resource path must complete");
        assertEquals(0, manager.getUAVManager().getActiveEdgeTaskCount());
        UAVMECNetworkModel networkModel = (UAVMECNetworkModel) manager.getNetworkModel();
        assertEquals(1, networkModel.getCompletedUavUploadCount());
        assertEquals(1, networkModel.getCompletedUavDownloadCount());
        assertEquals(0, networkModel.getActiveUavTransferCount());
        assertTrue(manager.getUAVManager().getUAVs().stream()
                .allMatch(uav -> uav.getTotalEnergyConsumed() > 0.0));
        assertEquals(9, manager.getUAVManager().getMovementCommandCount());
        assertTrue(manager.getUAVManager().getBoundaryConstraintCount() > 0);
        assertTrue(manager.getUAVManager().getUAV(0).getFlightEnergyConsumed() > 0.0);

        StateActionManager stateActionManager = new StateActionManager(manager);
        double[] state = stateActionManager.generateState();
        assertEquals(16, state.length);
        for (double value : state) {
            assertTrue(value >= 0.0 && value <= 1.0);
        }

        RewardCalculator rewardCalculator = new RewardCalculator(manager);
        assertTrue(rewardCalculator.calculateReward(100.0, 1000.0)
                > rewardCalculator.calculateReward(5000.0, 1000.0));
    }
}
