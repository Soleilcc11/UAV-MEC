package edu.boun.edgecloudsim.uav;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import edu.boun.edgecloudsim.utils.Location;

class UAVMECNetworkModelTest {
    @Test
    void airGroundDelayUsesThreeDimensionalDistanceAndPathLoss() {
        Location user = new Location(0, 0, 0, 0);

        double nearDelay = UAVMECNetworkModel.calculateAirGroundTransferDelay(
                1000, 100, 0.01, 32.4, 2.0, 0.0,
                user, new double[] {0.0, 0.0, 50.0});
        double farDelay = UAVMECNetworkModel.calculateAirGroundTransferDelay(
                1000, 100, 0.01, 32.4, 2.0, 0.0,
                user, new double[] {800.0, 800.0, 200.0});

        assertTrue(nearDelay > 0.0);
        assertTrue(farDelay > nearDelay);
    }
}
