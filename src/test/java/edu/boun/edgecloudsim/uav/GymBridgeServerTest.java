package edu.boun.edgecloudsim.uav;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.json.JSONObject;
import org.junit.jupiter.api.Test;

class GymBridgeServerTest {
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
