package edu.boun.edgecloudsim.core;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

class ExecutionTargetTest {
    @Test
    void legacyIdsRoundTripWithoutCollidingWithUavZero() {
        ExecutionTarget local = ExecutionTarget.local();
        ExecutionTarget uavZero = ExecutionTarget.uav(0);

        assertNotEquals(local.toLegacyDeviceId(), uavZero.toLegacyDeviceId());
        assertEquals(local, ExecutionTarget.fromLegacyDeviceId(local.toLegacyDeviceId()));
        assertEquals(uavZero, ExecutionTarget.fromLegacyDeviceId(uavZero.toLegacyDeviceId()));
        assertEquals(ExecutionTarget.cloud(),
                ExecutionTarget.fromLegacyDeviceId(SimSettings.CLOUD_DATACENTER_ID));
        assertEquals(ExecutionTarget.edge(),
                ExecutionTarget.fromLegacyDeviceId(SimSettings.GENERIC_EDGE_DEVICE_ID));
    }

    @Test
    void rejectsNegativeUavIds() {
        assertThrows(IllegalArgumentException.class, () -> ExecutionTarget.uav(-1));
    }
}
