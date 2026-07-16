package edu.boun.edgecloudsim.core;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

import java.util.Calendar;

import org.cloudbus.cloudsim.core.CloudSim;
import org.junit.jupiter.api.Test;

import edu.boun.edgecloudsim.uav.UAVMECScenarioFactory;

class EdgeCloudSimIntegrationTest {
    @Test
    void taskTraversesBrokerNetworkAndEdgeVm() {
        SimSettings settings = SimSettings.getInstance();
        settings.initialize(
                "src/test/resources/config/simulation_settings.xml",
                "src/test/resources/config/edge_devices.xml",
                "src/test/resources/config/applications.xml");

        CloudSim.init(1, Calendar.getInstance(), false);
        UAVMECScenarioFactory factory = new UAVMECScenarioFactory(1, "SINGLE_TIER", "RANDOM_FIT");
        SimManager manager = SimManager.getInstance();
        manager.initialize(factory, 1, "SINGLE_TIER", "RANDOM_FIT");

        CloudSim.startSimulation();

        assertEquals(1, manager.getEdgeServerManager().getDatacenterList().size());
        assertEquals(1, manager.getEdgeServerManager().getVmList(0).size());
        assertFalse(manager.getMobileDeviceManager().getCloudletReceivedList().isEmpty());
    }
}
