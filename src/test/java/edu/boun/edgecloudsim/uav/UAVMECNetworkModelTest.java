package edu.boun.edgecloudsim.uav;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.assertEquals;

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

    @Test
    void probabilisticLoSImprovesWithElevationAngle() {
        double lowElevation = UAVMECNetworkModel.calculateLoSProbability(
                5.0, 9.61, 0.16);
        double highElevation = UAVMECNetworkModel.calculateLoSProbability(
                80.0, 9.61, 0.16);

        assertTrue(lowElevation >= 0.0 && lowElevation <= 1.0);
        assertTrue(highElevation >= 0.0 && highElevation <= 1.0);
        assertTrue(highElevation > lowElevation);
    }

    @Test
    void shannonRateFallsWithDistance() {
        Location user = new Location(0, 0, 0, 0);

        double nearRate = UAVMECNetworkModel.calculateProbabilisticAirGroundRateMbps(
                10.0, 2_000_000_000.0, 9.61, 0.16, 1.0, 20.0,
                0.1, -174.0, 4, 1, user,
                new double[] {0.0, 0.0, 50.0});
        double farRate = UAVMECNetworkModel.calculateProbabilisticAirGroundRateMbps(
                10.0, 2_000_000_000.0, 9.61, 0.16, 1.0, 20.0,
                0.1, -174.0, 4, 1, user,
                new double[] {900.0, 900.0, 50.0});

        assertTrue(nearRate > farRate);
        assertTrue(farRate > 0.0);
    }

    @Test
    void fourChannelsShareOnlyAfterTheFourthConcurrentTransfer() {
        Location user = new Location(0, 0, 0, 0);
        double[] uav = {100.0, 100.0, 50.0};

        double oneTransfer = UAVMECNetworkModel.calculateProbabilisticAirGroundRateMbps(
                10.0, 2_000_000_000.0, 9.61, 0.16, 1.0, 20.0,
                0.1, -174.0, 4, 1, user, uav);
        double fourTransfers = UAVMECNetworkModel.calculateProbabilisticAirGroundRateMbps(
                10.0, 2_000_000_000.0, 9.61, 0.16, 1.0, 20.0,
                0.1, -174.0, 4, 4, user, uav);
        double fiveTransfers = UAVMECNetworkModel.calculateProbabilisticAirGroundRateMbps(
                10.0, 2_000_000_000.0, 9.61, 0.16, 1.0, 20.0,
                0.1, -174.0, 4, 5, user, uav);

        assertEquals(oneTransfer, fourTransfers, 1e-12);
        assertEquals(oneTransfer / 2.0, fiveTransfers, 1e-12);
    }

    @Test
    void exponentialTaskFeaturesUseTheNinetyNinthPercentileScale() {
        double[][] taskTable = {{0, 0, 0, 0, 0, 100, 20, 1000}};

        assertEquals(4605.170186, GymDecisionCoordinator.exponentialQ99Scale(
                taskTable, 7), 1e-6);
    }
}
