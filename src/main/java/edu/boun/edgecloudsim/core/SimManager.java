package edu.boun.edgecloudsim.core;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

import org.cloudbus.cloudsim.Vm;
import org.cloudbus.cloudsim.core.CloudSim;
import org.cloudbus.cloudsim.core.SimEntity;
import org.cloudbus.cloudsim.core.SimEvent;

import edu.boun.edgecloudsim.cloud_server.CloudServerManager;
import edu.boun.edgecloudsim.edge_client.MobileDeviceManager;
import edu.boun.edgecloudsim.edge_client.mobile_processing_unit.MobileServerManager;
import edu.boun.edgecloudsim.edge_orchestrator.EdgeOrchestrator;
import edu.boun.edgecloudsim.edge_server.EdgeServerManager;
import edu.boun.edgecloudsim.mobility.MobilityModel;
import edu.boun.edgecloudsim.network.NetworkModel;
import edu.boun.edgecloudsim.task_generator.LoadGeneratorModel;
import edu.boun.edgecloudsim.uav.PerformanceMonitor;
import edu.boun.edgecloudsim.uav.TaskOffloadingEngine;
import edu.boun.edgecloudsim.uav.UAVMECScenarioFactory;
import edu.boun.edgecloudsim.uav.UAVManager;
import edu.boun.edgecloudsim.uav.GymDecisionCoordinator;
import edu.boun.edgecloudsim.utils.SimLogger;
import edu.boun.edgecloudsim.utils.TaskProperty;
import edu.boun.edgecloudsim.utils.SimUtils;

/** Coordinates the EdgeCloudSim components and UAV extensions. */
public class SimManager extends SimEntity {
    private static SimManager instance;

    public static final int BASE_EVENT_ID = 5000;
    public static final int SUBMIT_EDGE_TASK = BASE_EVENT_ID + 1;
    public static final int PROCESS_UAV_TASKS = BASE_EVENT_ID + 2;
    public static final int SUBMIT_UAV_TASK = BASE_EVENT_ID + 3;
    public static final int MOVE_UAV = BASE_EVENT_ID + 4;
    private static final double TERMINATION_EPSILON = 1e-9;

    private SimSettings simSettings;
    private ScenarioFactory scenarioFactory;
    private LoadGeneratorModel loadGeneratorModel;
    private MobilityModel mobilityModel;
    private NetworkModel networkModel;
    private EdgeOrchestrator orchestrator;
    private EdgeServerManager edgeServerManager;
    private CloudServerManager cloudServerManager;
    private MobileServerManager mobileServerManager;
    private MobileDeviceManager mobileDeviceManager;

    private UAVManager uavManager;
    private TaskOffloadingEngine taskOffloadingEngine;
    private PerformanceMonitor performanceMonitor;
    private final List<TaskProperty> pendingUavTasks = new ArrayList<>();
    private final List<UavMovementCommand> pendingUavMovements = new ArrayList<>();

    private int numOfMobileDevice;
    private String simScenario;
    private String orchestratorPolicy;
    private boolean componentsInitialized;
    private int scheduledEdgeTaskCount;
    private int submittedEdgeTaskCount;
    private GymDecisionCoordinator gymCoordinator;

    private SimManager() {
        super("SimManager");
    }

    public SimManager(ScenarioFactory scenarioFactory, int numOfMobileDevice,
            String simScenario, String orchestratorPolicy) {
        super("SimManager");
        instance = this;
        initialize(scenarioFactory, numOfMobileDevice, simScenario, orchestratorPolicy);
    }

    public static SimManager getInstance() {
        if (instance == null) {
            instance = new SimManager();
        }
        return instance;
    }

    public static synchronized void resetInstance() {
        if (CloudSim.running()) {
            throw new IllegalStateException("Cannot reset SimManager while CloudSim is running");
        }
        instance = null;
    }

    public void initialize(Object config) {
        simSettings = SimSettings.getInstance();
    }

    public synchronized void initialize(ScenarioFactory scenarioFactory, int numOfMobileDevice,
            String simScenario, String orchestratorPolicy) {
        if (componentsInitialized) {
            return;
        }

        this.scenarioFactory = scenarioFactory;
        this.numOfMobileDevice = numOfMobileDevice;
        this.simScenario = simScenario;
        this.orchestratorPolicy = orchestratorPolicy;
        this.simSettings = SimSettings.getInstance();
        this.scheduledEdgeTaskCount = 0;
        this.submittedEdgeTaskCount = 0;

        try {
            SimUtils.setSeed(simSettings.getSimulationSeed());
            loadGeneratorModel = scenarioFactory.getLoadGeneratorModel();
            mobilityModel = scenarioFactory.getMobilityModel();
            networkModel = scenarioFactory.getNetworkModel();
            orchestrator = scenarioFactory.getEdgeOrchestrator();
            edgeServerManager = scenarioFactory.getEdgeServerManager();
            cloudServerManager = scenarioFactory.getCloudServerManager();
            mobileServerManager = scenarioFactory.getMobileServerManager();
            mobileDeviceManager = scenarioFactory.getMobileDeviceManager();

            if (scenarioFactory instanceof UAVMECScenarioFactory) {
                uavManager = ((UAVMECScenarioFactory) scenarioFactory).getUAVManager();
            }

            initializeEdgeCloudSimResources();
            componentsInitialized = true;
        } catch (Exception e) {
            throw new IllegalStateException("Cannot initialize EdgeCloudSim components", e);
        }
    }

    private void initializeEdgeCloudSimResources() throws Exception {
        loadGeneratorModel.initializeModel();
        edgeServerManager.initialize();
        cloudServerManager.initialize();
        mobileServerManager.initialize();
        mobileDeviceManager.initialize();

        edgeServerManager.startDatacenters();
        cloudServerManager.startDatacenters();
        mobileServerManager.startDatacenters();

        int brokerId = mobileDeviceManager.getId();
        edgeServerManager.createVmList(brokerId);
        cloudServerManager.createVmList(brokerId);
        mobileServerManager.createVmList(brokerId);

        List<Vm> vmList = new ArrayList<>();
        for (int hostId = 0; hostId < simSettings.getNumOfEdgeHosts(); hostId++) {
            if (edgeServerManager.getVmList(hostId) != null) {
                vmList.addAll(edgeServerManager.getVmList(hostId));
            }
        }
        for (int hostId = 0; hostId < simSettings.getNumOfCloudHost(); hostId++) {
            if (cloudServerManager.getVmList(hostId) != null) {
                vmList.addAll(cloudServerManager.getVmList(hostId));
            }
        }
        for (int mobileDeviceId = 0; mobileDeviceId < numOfMobileDevice;
                mobileDeviceId++) {
            if (mobileServerManager.getVmList(mobileDeviceId) != null) {
                vmList.addAll(mobileServerManager.getVmList(mobileDeviceId));
            }
        }
        mobileDeviceManager.submitVmList(vmList);
        SimLogger.printLine("EdgeCloudSim资源已初始化: " + vmList.size() + " 个VM");
    }

    @Override
    public void processEvent(SimEvent event) {
        if (event == null) {
            return;
        }

        switch (event.getTag()) {
            case SUBMIT_EDGE_TASK:
                if (mobileDeviceManager != null && event.getData() instanceof TaskProperty) {
                    submittedEdgeTaskCount++;
                    if (gymCoordinator != null) {
                        applyGymDecision(gymCoordinator.onTaskArrival((TaskProperty) event.getData()));
                    } else {
                        mobileDeviceManager.submitTask((TaskProperty) event.getData());
                    }
                }
                break;

            case PROCESS_UAV_TASKS:
                if (uavManager != null) {
                    uavManager.processTasks(1.0);
                    if (performanceMonitor != null) {
                        performanceMonitor.updateMetrics(1.0);
                    }
                    if (CloudSim.clock() + 1.0 <= simSettings.getSimulationTime()) {
                        schedule(getId(), 1.0, PROCESS_UAV_TASKS);
                    }
                }
                break;

            case SUBMIT_UAV_TASK:
                if (taskOffloadingEngine != null && event.getData() instanceof TaskProperty) {
                    taskOffloadingEngine.submitTask((TaskProperty) event.getData());
                }
                break;

            case MOVE_UAV:
                if (uavManager != null && event.getData() instanceof UavMovementCommand) {
                    UavMovementCommand command = (UavMovementCommand) event.getData();
                    uavManager.moveUav(command.uavId, command.displacement);
                }
                break;

            default:
                SimLogger.printLine("SimManager收到未知事件，标签: " + event.getTag());
                break;
        }
    }

    @Override
    public void startEntity() {
        mobilityModel.initialize();
        networkModel.initialize();
        orchestrator.initialize();

        if (uavManager != null) {
            uavManager.initialize();
        }
        if (taskOffloadingEngine != null) {
            taskOffloadingEngine.initialize();
        }

        scheduledEdgeTaskCount = loadGeneratorModel.getTaskList().size();
        if (gymCoordinator != null) {
            gymCoordinator.setTotalTaskCount(scheduledEdgeTaskCount);
        }
        List<TaskProperty> edgeTasks = new ArrayList<>(loadGeneratorModel.getTaskList());
        edgeTasks.sort(Comparator.comparingDouble(TaskProperty::getStartTime));
        for (TaskProperty task : edgeTasks) {
            // startEntity runs on CloudSim's simulation thread. Scheduling the
            // complete workload here keeps FutureQueue ownership off the socket
            // thread and lets independent task executions overlap naturally.
            schedule(getId(), Math.max(0.0, task.getStartTime()), SUBMIT_EDGE_TASK, task);
        }
        for (TaskProperty task : pendingUavTasks) {
            schedule(getId(), Math.max(0.0, task.getStartTime()), SUBMIT_UAV_TASK, task);
        }
        pendingUavTasks.clear();
        for (UavMovementCommand command : pendingUavMovements) {
            schedule(getId(), Math.max(0.0, command.time), MOVE_UAV, command);
        }
        pendingUavMovements.clear();

        if (uavManager != null) {
            schedule(getId(), 1.0, PROCESS_UAV_TASKS);
        }
        CloudSim.terminateSimulation(simSettings.getSimulationTime() + TERMINATION_EPSILON);
        SimLogger.printLine("EdgeCloudSim启动，已调度 " + loadGeneratorModel.getTaskList().size() + " 个任务");
    }

    @Override
    public void shutdownEntity() {
        if (taskOffloadingEngine != null) {
            taskOffloadingEngine.close();
        }
        if (gymCoordinator != null) {
            gymCoordinator.onSimulationEnded();
        }
        SimLogger.printLine("SimManager关闭");
    }

    public void startSimulation() {
        CloudSim.startSimulation();
    }

    public MobilityModel getMobilityModel() { return mobilityModel; }
    public NetworkModel getNetworkModel() { return networkModel; }
    public EdgeOrchestrator getEdgeOrchestrator() { return orchestrator; }
    public EdgeServerManager getEdgeServerManager() { return edgeServerManager; }
    public CloudServerManager getCloudServerManager() { return cloudServerManager; }
    public MobileServerManager getMobileServerManager() { return mobileServerManager; }
    public MobileDeviceManager getMobileDeviceManager() { return mobileDeviceManager; }
    public UAVManager getUAVManager() { return uavManager; }
    public TaskOffloadingEngine getTaskOffloadingEngine() { return taskOffloadingEngine; }
    public PerformanceMonitor getPerformanceMonitor() { return performanceMonitor; }
    public SimSettings getSimulationSettings() { return simSettings; }
    public String getSimulationScenario() { return simScenario; }
    public int getNumOfMobileDevice() { return numOfMobileDevice; }
    public double getSimulationTime() { return CloudSim.clock(); }
    public int getScheduledEdgeTaskCount() { return scheduledEdgeTaskCount; }
    public int getSubmittedEdgeTaskCount() { return submittedEdgeTaskCount; }

    public void setUAVManager(UAVManager uavManager) { this.uavManager = uavManager; }
    public void setTaskOffloadingEngine(TaskOffloadingEngine engine) { this.taskOffloadingEngine = engine; }
    public void setPerformanceMonitor(PerformanceMonitor monitor) { this.performanceMonitor = monitor; }
    public void setGymCoordinator(GymDecisionCoordinator coordinator) { this.gymCoordinator = coordinator; }
    public GymDecisionCoordinator getGymCoordinator() { return gymCoordinator; }

    public void scheduleUavTask(TaskProperty task) {
        if (task != null) {
            pendingUavTasks.add(task);
        }
    }

    public void scheduleUavMovement(double time, int uavId, double[] displacement) {
        if (displacement == null || displacement.length != 3) {
            throw new IllegalArgumentException("UAV displacement must contain x, y, and z");
        }
        pendingUavMovements.add(new UavMovementCommand(time, uavId, displacement.clone()));
    }

    public void applyGymDecision(GymDecisionCoordinator.Decision decision) {
        if (decision == null) {
            return;
        }
        double[][] movements = decision.getMovements();
        for (int i = 0; i < movements.length; i++) {
            uavManager.moveUav(
                    i, movements[i], decision.getMovementElapsedSeconds());
        }
        if (decision.isConstraintViolation()) {
            gymCoordinator.onTaskSubmitted(null, decision);
            return;
        }
        if (mobileDeviceManager instanceof edu.boun.edgecloudsim.edge_client.DefaultMobileDeviceManager) {
            edu.boun.edgecloudsim.edge_client.Task task =
                    ((edu.boun.edgecloudsim.edge_client.DefaultMobileDeviceManager) mobileDeviceManager)
                            .submitTask(decision.getTask(), decision.getTarget());
            gymCoordinator.onTaskSubmitted(task, decision);
        } else {
            // A Gym task must always reach a terminal accounting state even if a
            // custom scenario supplies a broker that cannot accept forced targets.
            gymCoordinator.onTaskSubmitted(null, decision);
        }
    }

    public void notifyGymTaskSettled(edu.boun.edgecloudsim.edge_client.Task task) {
        if (gymCoordinator != null && task != null
                && task.claimTerminalSettlementNotification()) {
            gymCoordinator.onTaskSettled(task);
        }
    }

    private static final class UavMovementCommand {
        private final double time;
        private final int uavId;
        private final double[] displacement;

        private UavMovementCommand(double time, int uavId, double[] displacement) {
            this.time = time;
            this.uavId = uavId;
            this.displacement = displacement;
        }
    }

    public void processResults() {
        // Result aggregation remains in SimLogger for now.
    }
}
