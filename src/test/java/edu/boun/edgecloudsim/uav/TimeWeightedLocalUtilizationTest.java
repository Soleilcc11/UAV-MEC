package edu.boun.edgecloudsim.uav;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

class TimeWeightedLocalUtilizationTest {
    @Test
    void integratesBusyTimeAcrossUnequalDecisionIntervals() {
        TimeWeightedLocalUtilization utilization = new TimeWeightedLocalUtilization(2);
        utilization.start(1, 10, 0.8, 1.0);
        assertEquals(0.4, utilization.drainBusySeconds(2.0), 1e-12);
        utilization.start(2, 10, 0.8, 2.0);
        assertEquals(1.0, utilization.drainBusySeconds(4.0), 1e-12);
        utilization.finish(1, 4.0);
        utilization.finish(2, 5.0);
        assertEquals(0.4, utilization.drainBusySeconds(5.0), 1e-12);
        assertEquals(0.0, utilization.drainBusySeconds(6.0), 1e-12);
    }

    @Test
    void rejectsTimeReversal() {
        TimeWeightedLocalUtilization utilization = new TimeWeightedLocalUtilization(1);
        utilization.start(1, 10, 0.8, 2.0);
        assertThrows(IllegalArgumentException.class,
                () -> utilization.finish(1, 1.0));
    }
}
