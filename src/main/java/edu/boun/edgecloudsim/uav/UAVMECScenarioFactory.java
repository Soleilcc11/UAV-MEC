package edu.boun.edgecloudsim.uav;

import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.NodeList;

import edu.boun.edgecloudsim.cloud_server.CloudServerManager;
import edu.boun.edgecloudsim.cloud_server.DefaultCloudServerManager;
import edu.boun.edgecloudsim.core.ScenarioFactory;
import edu.boun.edgecloudsim.core.SimManager;
import edu.boun.edgecloudsim.core.SimSettings;
import edu.boun.edgecloudsim.edge_client.DefaultMobileDeviceManager;
import edu.boun.edgecloudsim.edge_client.MobileDeviceManager;
import edu.boun.edgecloudsim.edge_client.mobile_processing_unit.MobileServerManager;
import edu.boun.edgecloudsim.applications.sample_app3.SampleMobileServerManager;
import edu.boun.edgecloudsim.edge_orchestrator.BasicEdgeOrchestrator;
import edu.boun.edgecloudsim.edge_orchestrator.EdgeOrchestrator;
import edu.boun.edgecloudsim.edge_server.DefaultEdgeServerManager;
import edu.boun.edgecloudsim.edge_server.EdgeServerManager;
import edu.boun.edgecloudsim.mobility.MobilityModel;
import edu.boun.edgecloudsim.network.NetworkModel;
import edu.boun.edgecloudsim.task_generator.IdleActiveLoadGenerator;
import edu.boun.edgecloudsim.task_generator.LoadGeneratorModel;
import edu.boun.edgecloudsim.utils.Location;

/**
 * EdgeCloudSim component factory for the UAV-MEC scenario.
 * UAV-specific resources are added without replacing the framework's
 * load generator, broker, orchestrator, datacenters, and VM managers.
 */
public class UAVMECScenarioFactory implements ScenarioFactory {
    private final int numOfMobileDevices;
    private final String simScenario;
    private final String orchestratorPolicy;
    private UAVManager uavManager;

    public UAVMECScenarioFactory(int numOfMobileDevices, String simScenario, String orchestratorPolicy) {
        this.numOfMobileDevices = numOfMobileDevices;
        this.simScenario = simScenario;
        this.orchestratorPolicy = normalizePolicy(orchestratorPolicy);
    }

    @Override
    public LoadGeneratorModel getLoadGeneratorModel() {
        return new IdleActiveLoadGenerator(
                numOfMobileDevices,
                SimSettings.getInstance().getSimulationTime(),
                simScenario);
    }

    @Override
    public EdgeOrchestrator getEdgeOrchestrator() {
        return new BasicEdgeOrchestrator(orchestratorPolicy, simScenario);
    }

    @Override
    public MobilityModel getMobilityModel() {
        return new MobilityModel(numOfMobileDevices, SimSettings.getInstance().getSimulationTime()) {
            @Override
            public void initialize() {
                // Static access-point association for the first integration baseline.
            }

            @Override
            public Location getLocation(int deviceId, double time) {
                Document document = SimSettings.getInstance().getEdgeDevicesDocument();
                NodeList datacenters = document.getElementsByTagName("datacenter");
                if (datacenters.getLength() == 0) {
                    return new Location(0, 0, 0, 0);
                }

                Element datacenter = (Element) datacenters.item(Math.floorMod(deviceId, datacenters.getLength()));
                Element location = (Element) datacenter.getElementsByTagName("location").item(0);
                int attractiveness = childInt(location, "attractiveness");
                int wlanId = childInt(location, "wlan_id");
                int x = childInt(location, "x_pos");
                int y = childInt(location, "y_pos");
                return new Location(attractiveness, wlanId, x, y);
            }
        };
    }

    @Override
    public NetworkModel getNetworkModel() {
        return new UAVMECNetworkModel(numOfMobileDevices, simScenario);
    }

    @Override
    public EdgeServerManager getEdgeServerManager() {
        return new DefaultEdgeServerManager();
    }

    @Override
    public CloudServerManager getCloudServerManager() {
        return new DefaultCloudServerManager();
    }

    @Override
    public MobileServerManager getMobileServerManager() {
        return new SampleMobileServerManager(numOfMobileDevices);
    }

    @Override
    public MobileDeviceManager getMobileDeviceManager() throws Exception {
        return new DefaultMobileDeviceManager();
    }

    public UAVManager getUAVManager() {
        if (uavManager == null) {
            uavManager = new UAVManager(SimManager.getInstance());
        }
        return uavManager;
    }

    private String normalizePolicy(String policy) {
        if (policy == null) {
            return "RANDOM_FIT";
        }
        String normalized = policy.toUpperCase();
        if (normalized.equals("RANDOM_FIT") || normalized.equals("WORST_FIT")
                || normalized.equals("BEST_FIT") || normalized.equals("FIRST_FIT")
                || normalized.equals("NEXT_FIT")) {
            return normalized;
        }
        return "RANDOM_FIT";
    }

    private static int childInt(Element element, String tag) {
        return Integer.parseInt(element.getElementsByTagName(tag).item(0).getTextContent().trim());
    }
}
