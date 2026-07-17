package edu.boun.edgecloudsim.uav;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Calendar;
import java.util.List;

import org.cloudbus.cloudsim.core.CloudSim;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class UAVTaskTest {
    @BeforeEach
    void initializeCloudSimClock() {
        CloudSim.init(1, Calendar.getInstance(), false);
    }

    @Test
    void taskUsesSimulationClockInsteadOfWallClock() {
        UAV.Task task = new UAV.Task("task-1", 1000.0);

        assertEquals(0L, task.getArrivalTime());
    }

    @Test
    void taskCompletionIsRecordedOnceWithNonNegativeSimulationTime() {
        UAV uav = new UAV(0, new double[] {0.0, 0.0, 100.0}, 100.0, 2000.0);
        uav.configureEnergyModel(10.0, 2.0, 0.001);
        UAV.Task task = new UAV.Task("task-1", 1000.0, 0L);
        assertTrue(uav.addTask(task));

        List<UAV.Task> completed = uav.processTasks(1.0);

        assertEquals(1, completed.size());
        assertEquals(0L, completed.get(0).getCompletionTime());
        assertTrue(completed.get(0).getCompletionTime() >= completed.get(0).getArrivalTime());
        assertEquals(UAV.Task.COMPLETED_STATUS, completed.get(0).getStatus());
        assertEquals(1, uav.getMaxObservedQueueLength());
        assertEquals(0L, completed.get(0).getQueueWaitTime());
        assertEquals(2.0, uav.getHoverEnergyConsumed(), 1e-9);
        assertEquals(1.0, uav.getProcessingEnergyConsumed(), 1e-9);
        assertEquals(3.0, uav.getTotalEnergyConsumed(), 1e-9);
    }

    @Test
    void movementAndHoverEnergyAreTrackedSeparately() {
        UAV uav = new UAV(0, new double[] {0.0, 0.0, 50.0}, 100.0, 2000.0);
        uav.configureEnergyModel(10.0, 2.0, 0.001);

        uav.updatePosition(new double[] {3.0, 4.0, 0.0});
        uav.processTasks(1.0);

        assertEquals(10.0, uav.getFlightEnergyConsumed(), 1e-9);
        assertEquals(2.0, uav.getHoverEnergyConsumed(), 1e-9);
        assertEquals(12.0, uav.getTotalEnergyConsumed(), 1e-9);
    }
}
