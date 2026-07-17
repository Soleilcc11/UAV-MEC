package edu.boun.edgecloudsim.uav;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.json.JSONObject;
import org.junit.jupiter.api.Test;

import edu.boun.edgecloudsim.core.SimSettings;

class GymBridgeServerTest {
    @Test
    void stepMetricsExposeRawPhysicalValuesAndEpisodeProgress() {
        JSONObject metrics = GymDecisionCoordinator.withEpisodeProgress(
                GymDecisionCoordinator.rawPhysicalMetrics(1.25, 4.0, 0.75, 12.5, 2),
                3, 8, 9.5);

        assertEquals(8, metrics.length());
        assertEquals(1.25, metrics.getDouble("latency_seconds"), 1e-12);
        assertEquals(4.0, metrics.getDouble("deadline_seconds"), 1e-12);
        assertEquals(0.75, metrics.getDouble("ue_energy_joules"), 1e-12);
        assertEquals(12.5, metrics.getDouble("uav_energy_joules"), 1e-12);
        assertEquals(2, metrics.getInt("constraint_violations"));
        assertEquals(3, metrics.getInt("settled_tasks"));
        assertEquals(8, metrics.getInt("total_tasks"));
        assertEquals(9.5, metrics.getDouble("simulation_time"), 1e-12);
    }

    @Test
    void heldoutConfigurationLoadsAChangedWorkloadDeadlineAndResourceProfile() {
        SimSettings settings = SimSettings.getInstance();
        settings.initialize(
                "src/test/resources/config/heldout/simulation_settings.xml",
                "src/test/resources/config/heldout/edge_devices.xml",
                "src/test/resources/config/heldout/applications.xml");

        assertEquals(8000.0, settings.getUAVMaxEnergy(), 1e-12);
        assertEquals(1.25, settings.getTaskLookUpTable()[0][2], 1e-12);
        assertEquals(2200.0, settings.getTaskLookUpTable()[0][7], 1e-12);
        assertEquals(45.0, settings.getTaskLookUpTable()[0][13], 1e-12);
        assertEquals("7000", settings.getEdgeDevicesDocument()
                .getElementsByTagName("mips").item(0).getTextContent().trim());
    }

    @Test
    void rejectsIncompatibleProtocolBeforeDispatch() {
        GymBridgeSession session = new GymBridgeSession("unused", "unused", "unused", 1);
        GymBridgeServer server = new GymBridgeServer(0, session);

        JSONObject response = server.handle(new JSONObject()
                .put("id", "request-1")
                .put("type", "hello")
                .put("protocol_version", "2.0"));

        assertFalse(response.getBoolean("ok"));
        assertTrue(response.getString("error").contains("Protocol version mismatch"));
    }
}
