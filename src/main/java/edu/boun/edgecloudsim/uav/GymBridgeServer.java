package edu.boun.edgecloudsim.uav;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.net.ServerSocket;
import java.net.Socket;
import java.nio.charset.StandardCharsets;

import org.json.JSONArray;
import org.json.JSONObject;

/** Versioned newline-delimited JSON server for Gymnasium control. */
public class GymBridgeServer implements AutoCloseable {
    private final int port;
    private final GymBridgeSession session;
    private volatile boolean running;
    private ServerSocket serverSocket;

    public GymBridgeServer(int port, GymBridgeSession session) {
        this.port = port;
        this.session = session;
    }

    public void serve() throws Exception {
        running = true;
        serverSocket = new ServerSocket(port);
        while (running) {
            try (Socket socket = serverSocket.accept();
                    BufferedReader reader = new BufferedReader(new InputStreamReader(
                            socket.getInputStream(), StandardCharsets.UTF_8));
                    BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(
                            socket.getOutputStream(), StandardCharsets.UTF_8))) {
                String line;
                while (running && (line = reader.readLine()) != null) {
                    JSONObject response = handle(new JSONObject(line));
                    writer.write(response.toString());
                    writer.newLine();
                    writer.flush();
                }
            }
        }
    }

    JSONObject handle(JSONObject request) {
        String requestId = request.optString("id", "");
        String type = request.optString("type", "");
        JSONObject response = new JSONObject().put("id", requestId).put("type", type + "_response");
        try {
            requireVersion(request);
            switch (type) {
                case "hello":
                    response.put("spec", session.specification());
                    break;
                case "reset":
                    JSONObject reset = session.reset(request.getLong("seed"), 30000);
                    response.put("observation", reset.getJSONObject("observation"));
                    response.put("info", reset.getJSONObject("info"));
                    break;
                case "step":
                    JSONObject action = request.getJSONObject("action");
                    double[][] movement = parseMovement(action.getJSONArray("movement"));
                    JSONObject step = session.step(action.getInt("target"), movement, 30000);
                    for (String key : step.keySet()) response.put(key, step.get(key));
                    break;
                case "close":
                    session.close();
                    break;
                default:
                    throw new IllegalArgumentException("Unsupported request type: " + type);
            }
            response.put("ok", true);
        } catch (Exception e) {
            response.put("ok", false).put("error", e.getMessage());
        }
        return response;
    }

    private void requireVersion(JSONObject request) {
        String version = request.optString("protocol_version", "");
        if (!GymBridgeSession.PROTOCOL_VERSION.equals(version)) {
            throw new IllegalArgumentException("Protocol version mismatch: " + version);
        }
    }

    private double[][] parseMovement(JSONArray rows) {
        int uavCount = session.specification().getInt("number_of_uavs");
        if (rows.length() != uavCount) {
            throw new IllegalArgumentException("Movement row count must equal UAV count");
        }
        double[][] result = new double[uavCount][3];
        for (int i = 0; i < uavCount; i++) {
            JSONArray row = rows.getJSONArray(i);
            if (row.length() != 3) throw new IllegalArgumentException("Movement rows require xyz");
            for (int j = 0; j < 3; j++) {
                double value = row.getDouble(j);
                if (value < -1.0 || value > 1.0) {
                    throw new IllegalArgumentException("Movement values must be in [-1,1]");
                }
                result[i][j] = value;
            }
        }
        return result;
    }

    @Override
    public void close() {
        running = false;
        session.close();
        if (serverSocket != null) {
            try { serverSocket.close(); } catch (Exception ignored) { }
        }
    }
}
