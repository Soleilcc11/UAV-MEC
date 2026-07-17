package edu.boun.edgecloudsim.uav;

/** CLI entry point for the Java side of GymBridge. */
public final class GymBridgeMain {
    private GymBridgeMain() { }

    public static void main(String[] args) throws Exception {
        int port = args.length > 0 ? Integer.parseInt(args[0]) : 12346;
        String settings = args.length > 1 ? args[1]
                : "src/main/resources/config/simulation_settings.xml";
        String edgeDevices = args.length > 2 ? args[2]
                : "src/main/resources/config/edge_devices.xml";
        String applications = args.length > 3 ? args[3]
                : "src/main/resources/config/applications.xml";
        int mobileDevices = args.length > 4 ? Integer.parseInt(args[4]) : 100;

        GymBridgeSession session = new GymBridgeSession(
                settings, edgeDevices, applications, mobileDevices);
        try (GymBridgeServer server = new GymBridgeServer(port, session)) {
            server.serve();
        }
    }
}
