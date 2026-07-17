package edu.boun.edgecloudsim.core;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

class SimSettingsTest {
    @Test
    void parsesEdgeCloudSimXmlConfiguration() {
        SimSettings settings = SimSettings.getInstance();
        settings.initialize(
                "src/test/resources/config/simulation_settings.xml",
                "src/test/resources/config/edge_devices.xml",
                "src/test/resources/config/applications.xml");

        assertEquals(60.0, settings.getSimulationTime());
        assertEquals(2, settings.getNumOfUAVs());
        assertEquals(1, settings.getNumOfEdgeHosts());
        assertEquals(1, settings.getNumOfEdgeVMs());
        assertEquals(1, settings.getTaskLookUpTable().length);
        assertEquals(1000.0, settings.getTaskLookUpTable()[0][7]);
        assertEquals("SMOKE_TASK", settings.getTaskName(0));
        assertEquals("SINGLE_TIER", settings.getSimulationScenarios()[0]);
        assertEquals("RANDOM_FIT", settings.getOrchestratorPolicies()[0]);
        assertEquals(42L, settings.getSimulationSeed());
        assertEquals(50.0, settings.getUAVFlightPower());
        assertEquals(20.0, settings.getUAVHoverPower());
        assertEquals(0.001, settings.getUAVComputeEnergyPerMi());
    }
}
