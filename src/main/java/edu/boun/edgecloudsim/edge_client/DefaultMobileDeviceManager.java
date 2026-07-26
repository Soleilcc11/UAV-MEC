/*
 * Title:        EdgeCloudSim - Mobile Device Manager
 * 
 * Description: 
 * DefaultMobileDeviceManager is responsible for submitting the tasks to the related
 * device by using the Edge Orchestrator. It also takes proper actions 
 * when the execution of the tasks are finished.
 * By default, DefaultMobileDeviceManager sends tasks to the edge servers or
 * cloud servers. If you want to use different topology, for example
 * MAN edge server, you should modify the flow defined in this class.
 * 
 * Licence:      GPL - http://www.gnu.org/copyleft/gpl.html
 * Copyright (c) 2017, Bogazici University, Istanbul, Turkey
 */

package edu.boun.edgecloudsim.edge_client;

import org.cloudbus.cloudsim.UtilizationModel;
import org.cloudbus.cloudsim.UtilizationModelFull;
import org.cloudbus.cloudsim.Vm;
import org.cloudbus.cloudsim.core.CloudSim;
import org.cloudbus.cloudsim.core.CloudSimTags;
import org.cloudbus.cloudsim.core.SimEvent;

import edu.boun.edgecloudsim.core.SimManager;
import edu.boun.edgecloudsim.core.SimSettings;
import edu.boun.edgecloudsim.core.ExecutionTarget;
import edu.boun.edgecloudsim.core.SimSettings.NETWORK_DELAY_TYPES;
import edu.boun.edgecloudsim.network.NetworkModel;
import edu.boun.edgecloudsim.utils.TaskProperty;
import edu.boun.edgecloudsim.utils.Location;
import edu.boun.edgecloudsim.utils.SimLogger;

public class DefaultMobileDeviceManager extends MobileDeviceManager {
	private static final int BASE = 100000; //start from base in order not to conflict cloudsim tag!
	private static final int REQUEST_RECEIVED_BY_CLOUD = BASE + 1;
	private static final int REQUEST_RECEIVED_BY_EDGE_DEVICE = BASE + 2;
	private static final int RESPONSE_RECEIVED_BY_MOBILE_DEVICE = BASE + 3;
	private static final int REQUEST_RECEIVED_BY_UAV = BASE + 4;
	private int taskIdCounter=0;
	
	public DefaultMobileDeviceManager() throws Exception{
	}

	@Override
	public void initialize() {
	}
	
	@Override
	public UtilizationModel getCpuUtilizationModel() {
		return new CpuUtilizationModel_Custom();
	}
	
	/**
	 * Submit cloudlets to the created VMs.
	 * 
	 * @pre $none
	 * @post $none
	 */
	protected void submitCloudlets() {
		//do nothing!
	}
	
	/**
	 * Process a cloudlet return event.
	 * 
	 * @param ev a SimEvent object
	 * @pre ev != $null
	 * @post $none
	 */
	protected void processCloudletReturn(SimEvent ev) {
		NetworkModel networkModel = SimManager.getInstance().getNetworkModel();
		Task task = (Task) ev.getData();

		// This class submits cloudlets after the modeled upload delay instead of
		// using DatacenterBroker.submitCloudlets(). Keep the broker bookkeeping
		// consistent when the datacenter returns a completed task.
		getCloudletReceivedList().add(task);
		getCloudletSubmittedList().remove(task);
		if (cloudletsSubmitted > 0) {
			cloudletsSubmitted--;
		}

		SimLogger.getInstance().taskExecuted(task.getCloudletId());

		if(task.getAssociatedDatacenterId() == SimSettings.MOBILE_DATACENTER_ID){
			SimLogger.getInstance().taskEnded(task.getCloudletId(), CloudSim.clock());
			SimManager.getInstance().notifyGymTaskSettled(task);
		}
		else if(task.getAssociatedDatacenterId() == SimSettings.CLOUD_DATACENTER_ID){
			//SimLogger.printLine(CloudSim.clock() + ": " + getName() + ": task #" + task.getCloudletId() + " received from cloud");
			double WanDelay = networkModel.getDownloadDelay(SimSettings.CLOUD_DATACENTER_ID, task.getMobileDeviceId(), task);
			if(WanDelay > 0)
			{
				Location currentLocation = SimManager.getInstance().getMobilityModel().getLocation(task.getMobileDeviceId(),CloudSim.clock()+WanDelay);
				if(task.getSubmittedLocation().getServingWlanId() == currentLocation.getServingWlanId())
				{
					networkModel.downloadStarted(task.getSubmittedLocation(), SimSettings.CLOUD_DATACENTER_ID);
					SimLogger.getInstance().setDownloadDelay(task.getCloudletId(), WanDelay, NETWORK_DELAY_TYPES.WAN_DELAY);
					schedule(getId(), WanDelay, RESPONSE_RECEIVED_BY_MOBILE_DEVICE, task);
				}
				else
				{
					SimLogger.getInstance().failedDueToMobility(task.getCloudletId(), CloudSim.clock());
					failAndSettle(task);
				}
			}
			else
			{
				SimLogger.getInstance().failedDueToBandwidth(task.getCloudletId(), CloudSim.clock(), NETWORK_DELAY_TYPES.WAN_DELAY);
				failAndSettle(task);
			}
		}
		else{
			//SimLogger.printLine(CloudSim.clock() + ": " + getName() + ": task #" + task.getCloudletId() + " received from edge");
			double WlanDelay = networkModel.getDownloadDelay(task.getAssociatedHostId(), task.getMobileDeviceId(), task);
			if(WlanDelay > 0)
			{
				Location currentLocation = SimManager.getInstance().getMobilityModel().getLocation(task.getMobileDeviceId(),CloudSim.clock()+WlanDelay);
				if(task.getSubmittedLocation().getServingWlanId() == currentLocation.getServingWlanId())
				{
					networkModel.downloadStarted(currentLocation, SimSettings.GENERIC_EDGE_DEVICE_ID);
					SimLogger.getInstance().setDownloadDelay(task.getCloudletId(), WlanDelay, NETWORK_DELAY_TYPES.WLAN_DELAY);
					schedule(getId(), WlanDelay, RESPONSE_RECEIVED_BY_MOBILE_DEVICE, task);
				}
				else
				{
					SimLogger.getInstance().failedDueToMobility(task.getCloudletId(), CloudSim.clock());
					failAndSettle(task);
				}
			}
			else
			{
				SimLogger.getInstance().failedDueToBandwidth(task.getCloudletId(), CloudSim.clock(), NETWORK_DELAY_TYPES.WLAN_DELAY);
				failAndSettle(task);
			}
		}
	}
	
	protected void processOtherEvent(SimEvent ev) {
		if (ev == null) {
			SimLogger.printLine(getName() + ".processOtherEvent(): " + "Error - an event is null! Terminating simulation...");
			System.exit(1);
			return;
		}
		
		NetworkModel networkModel = SimManager.getInstance().getNetworkModel();
		
		switch (ev.getTag()) {
			case REQUEST_RECEIVED_BY_CLOUD:
			{
				Task task = (Task) ev.getData();

				networkModel.uploadFinished(task.getSubmittedLocation(), SimSettings.CLOUD_DATACENTER_ID);

				submitTaskToVm(task,0,SimSettings.CLOUD_DATACENTER_ID);
				
				break;
			}
			case REQUEST_RECEIVED_BY_EDGE_DEVICE:
			{
				Task task = (Task) ev.getData();
				
				networkModel.uploadFinished(task.getSubmittedLocation(), SimSettings.GENERIC_EDGE_DEVICE_ID);
				
				submitTaskToVm(task, 0, SimSettings.GENERIC_EDGE_DEVICE_ID);
				
				break;
			}
			case REQUEST_RECEIVED_BY_UAV:
			{
				Task task = (Task) ev.getData();
				ExecutionTarget target = task.getExecutionTarget();
				networkModel.uploadFinished(task.getSubmittedLocation(), target.toLegacyDeviceId());

				boolean assigned = SimManager.getInstance().getUAVManager()
						.submitEdgeTask(target.getResourceId(), task);
				if (assigned) {
					task.setAssociatedDatacenterId(target.toLegacyDeviceId());
					task.setAssociatedHostId(target.getResourceId());
					task.setAssociatedVmId(-1);
					getCloudletSubmittedList().add(task);
					cloudletsSubmitted++;
					SimLogger.getInstance().taskAssigned(task.getCloudletId(),
							target.toLegacyDeviceId(), target.getResourceId(), -1,
							SimSettings.VM_TYPES.EDGE_VM.ordinal());
				} else {
					SimLogger.getInstance().rejectedDueToVMCapacity(
							task.getCloudletId(), CloudSim.clock(), SimSettings.VM_TYPES.EDGE_VM.ordinal());
					failAndSettle(task);
				}
				break;
			}
			case RESPONSE_RECEIVED_BY_MOBILE_DEVICE:
			{
				Task task = (Task) ev.getData();
				ExecutionTarget target = task.getExecutionTarget();

				if(target != null && target.getType() == ExecutionTarget.Type.CLOUD)
					networkModel.downloadFinished(task.getSubmittedLocation(), SimSettings.CLOUD_DATACENTER_ID);
				else if(target != null && target.getType() == ExecutionTarget.Type.UAV)
					networkModel.downloadFinished(task.getSubmittedLocation(), target.toLegacyDeviceId());
				else if(task.getAssociatedDatacenterId() != SimSettings.MOBILE_DATACENTER_ID)
					networkModel.downloadFinished(task.getSubmittedLocation(), SimSettings.GENERIC_EDGE_DEVICE_ID);
				
				SimLogger.getInstance().taskEnded(task.getCloudletId(), CloudSim.clock());
				SimManager.getInstance().notifyGymTaskSettled(task);
				break;
			}
			default:
				SimLogger.printLine(getName() + ".processOtherEvent(): " + "Error - event unknown by this DatacenterBroker. Terminating simulation...");
				System.exit(1);
				break;
		}
	}

	public void submitTask(TaskProperty edgeTask) {
		submitTask(edgeTask, null);
	}

	public Task submitTask(TaskProperty edgeTask, ExecutionTarget forcedTarget) {
		
		NetworkModel networkModel = SimManager.getInstance().getNetworkModel();
		
		//create a task
		Task task = createTask(edgeTask);
		
		Location currentLocation = SimManager.getInstance().getMobilityModel().
				getLocation(task.getMobileDeviceId(),CloudSim.clock());
		
		//set location of the mobile device which generates this task
		task.setSubmittedLocation(currentLocation);

		//add related task to log list
		SimLogger.getInstance().addLog(task.getMobileDeviceId(),
				task.getCloudletId(),
				task.getTaskType(),
				(int)task.getCloudletLength(),
				(int)task.getCloudletFileSize(),
				(int)task.getCloudletOutputSize());

		ExecutionTarget target = forcedTarget != null
				? forcedTarget
				: SimManager.getInstance().getEdgeOrchestrator().getExecutionTarget(task);
		task.setExecutionTarget(target);
		int nextHopId = target.toLegacyDeviceId();

		if(target.getType() == ExecutionTarget.Type.CLOUD){
			double WanDelay = networkModel.getUploadDelay(task.getMobileDeviceId(), nextHopId, task);
			
			if(WanDelay>0){
				networkModel.uploadStarted(currentLocation, nextHopId);
				SimLogger.getInstance().taskStarted(task.getCloudletId(), CloudSim.clock());
				SimLogger.getInstance().setUploadDelay(task.getCloudletId(), WanDelay, NETWORK_DELAY_TYPES.WAN_DELAY);
				schedule(getId(), WanDelay, REQUEST_RECEIVED_BY_CLOUD, task);
			}
			else
			{
				//SimLogger.printLine("Task #" + task.getCloudletId() + " cannot assign to any VM");
				SimLogger.getInstance().rejectedDueToBandwidth(
						task.getCloudletId(),
						CloudSim.clock(),
						SimSettings.VM_TYPES.CLOUD_VM.ordinal(),
						NETWORK_DELAY_TYPES.WAN_DELAY);
				failAndSettle(task);
			}
		}
		else if(target.getType() == ExecutionTarget.Type.EDGE) {
			double WlanDelay = networkModel.getUploadDelay(task.getMobileDeviceId(), nextHopId, task);
			
			if(WlanDelay > 0){
				networkModel.uploadStarted(currentLocation, nextHopId);
				schedule(getId(), WlanDelay, REQUEST_RECEIVED_BY_EDGE_DEVICE, task);
				SimLogger.getInstance().taskStarted(task.getCloudletId(), CloudSim.clock());
				SimLogger.getInstance().setUploadDelay(task.getCloudletId(), WlanDelay, NETWORK_DELAY_TYPES.WLAN_DELAY);
			}
			else {
				SimLogger.getInstance().rejectedDueToBandwidth(
						task.getCloudletId(),
						CloudSim.clock(),
						SimSettings.VM_TYPES.EDGE_VM.ordinal(),
						NETWORK_DELAY_TYPES.WLAN_DELAY);
				failAndSettle(task);
			}
		}
		else if(target.getType() == ExecutionTarget.Type.UAV) {
			double UavDelay = networkModel.getUploadDelay(task.getMobileDeviceId(), nextHopId, task);

			if (UavDelay > 0) {
				networkModel.uploadStarted(currentLocation, nextHopId);
				SimLogger.getInstance().taskStarted(task.getCloudletId(), CloudSim.clock());
				SimLogger.getInstance().setUploadDelay(
						task.getCloudletId(), UavDelay, NETWORK_DELAY_TYPES.WLAN_DELAY);
				schedule(getId(), UavDelay, REQUEST_RECEIVED_BY_UAV, task);
			} else {
				SimLogger.getInstance().rejectedDueToBandwidth(
						task.getCloudletId(), CloudSim.clock(),
						SimSettings.VM_TYPES.EDGE_VM.ordinal(), NETWORK_DELAY_TYPES.WLAN_DELAY);
				failAndSettle(task);
			}
		}
		else if(target.getType() == ExecutionTarget.Type.LOCAL) {
			SimLogger.getInstance().taskStarted(task.getCloudletId(), CloudSim.clock());
			submitTaskToVm(task, 0.0, SimSettings.MOBILE_DATACENTER_ID);
		}
		else {
			throw new IllegalArgumentException("Unsupported execution target: " + target);
		}
		return task;
	}

	/** Called by UAVManager when an EdgeCloudSim task finishes on a UAV. */
	public void uavTaskCompleted(Task task) {
		ExecutionTarget target = task.getExecutionTarget();
		NetworkModel networkModel = SimManager.getInstance().getNetworkModel();
		try {
			task.setCloudletStatus(org.cloudbus.cloudsim.Cloudlet.SUCCESS);
		} catch (Exception e) {
			throw new IllegalStateException("Cannot mark UAV task as completed", e);
		}

		getCloudletReceivedList().add(task);
		getCloudletSubmittedList().remove(task);
		if (cloudletsSubmitted > 0) {
			cloudletsSubmitted--;
		}
		SimLogger.getInstance().taskExecuted(task.getCloudletId());

		double delay = networkModel.getDownloadDelay(
				target.toLegacyDeviceId(), task.getMobileDeviceId(), task);
		if (delay > 0) {
			Location currentLocation = SimManager.getInstance().getMobilityModel()
					.getLocation(task.getMobileDeviceId(), CloudSim.clock() + delay);
			if (task.getSubmittedLocation().getServingWlanId() == currentLocation.getServingWlanId()) {
				networkModel.downloadStarted(currentLocation, target.toLegacyDeviceId());
				SimLogger.getInstance().setDownloadDelay(
						task.getCloudletId(), delay, NETWORK_DELAY_TYPES.WLAN_DELAY);
				schedule(getId(), delay, RESPONSE_RECEIVED_BY_MOBILE_DEVICE, task);
			} else {
				SimLogger.getInstance().failedDueToMobility(task.getCloudletId(), CloudSim.clock());
				failAndSettle(task);
			}
		} else {
			SimLogger.getInstance().failedDueToBandwidth(
					task.getCloudletId(), CloudSim.clock(), NETWORK_DELAY_TYPES.WLAN_DELAY);
			failAndSettle(task);
		}
	}

	/** Called by UAVManager when a queued EdgeCloudSim task cannot continue. */
	public void uavTaskFailed(Task task) {
		boolean wasSubmitted = getCloudletSubmittedList().remove(task);
		if (wasSubmitted && cloudletsSubmitted > 0) {
			cloudletsSubmitted--;
		}
		failAndSettle(task);
	}

	private void submitTaskToVm(Task task, double delay, int datacenterId) {
		//select a VM
		Vm selectedVM = SimManager.getInstance().getEdgeOrchestrator().getVmToOffload(task, datacenterId);
		
		int vmType = 0;
		if(datacenterId == SimSettings.CLOUD_DATACENTER_ID)
			vmType = SimSettings.VM_TYPES.CLOUD_VM.ordinal();
		else if(datacenterId == SimSettings.MOBILE_DATACENTER_ID)
			vmType = SimSettings.VM_TYPES.MOBILE_VM.ordinal();
		else
			vmType = SimSettings.VM_TYPES.EDGE_VM.ordinal();
		
		if(selectedVM != null){
			if(datacenterId == SimSettings.CLOUD_DATACENTER_ID)
				task.setAssociatedDatacenterId(SimSettings.CLOUD_DATACENTER_ID);
			else
				task.setAssociatedDatacenterId(selectedVM.getHost().getDatacenter().getId());

			//save related host id
			task.setAssociatedHostId(selectedVM.getHost().getId());
			
			//set related vm id
			task.setAssociatedVmId(selectedVM.getId());
			
			// Track the manually delayed submission in the same collections used by
			// DatacenterBroker's normal submitCloudlets() path.
			task.setVmId(selectedVM.getId());
			getCloudletSubmittedList().add(task);
			cloudletsSubmitted++;
			
			//SimLogger.printLine(CloudSim.clock() + ": Cloudlet#" + task.getCloudletId() + " is submitted to VM#" + task.getVmId());
			schedule(getVmsToDatacentersMap().get(task.getVmId()), delay, CloudSimTags.CLOUDLET_SUBMIT, task);

			SimLogger.getInstance().taskAssigned(task.getCloudletId(),
					selectedVM.getHost().getDatacenter().getId(),
					selectedVM.getHost().getId(),
					selectedVM.getId(),
					vmType);
		}
		else{
			//SimLogger.printLine("Task #" + task.getCloudletId() + " cannot assign to any VM");
			SimLogger.getInstance().rejectedDueToVMCapacity(task.getCloudletId(), CloudSim.clock(), vmType);
			failAndSettle(task);
		}
	}

	private void markFailed(Task task) {
		try {
			task.setCloudletStatus(org.cloudbus.cloudsim.Cloudlet.FAILED_RESOURCE_UNAVAILABLE);
		} catch (Exception ignored) {
			// GymBridge still records the terminal transition as a failed task.
		}
	}

	private void failAndSettle(Task task) {
		markFailed(task);
		SimManager.getInstance().notifyGymTaskSettled(task);
	}
	
	private Task createTask(TaskProperty edgeTask){
		UtilizationModel utilizationModel = new UtilizationModelFull(); /*UtilizationModelStochastic*/
		UtilizationModel utilizationModelCPU = getCpuUtilizationModel();

		Task task = new Task(edgeTask.getMobileDeviceId(), ++taskIdCounter,
				edgeTask.getLength(), edgeTask.getPesNumber(),
				edgeTask.getInputFileSize(), edgeTask.getOutputFileSize(),
				utilizationModelCPU, utilizationModel, utilizationModel);
		
		//set the owner of this task
		task.setUserId(this.getId());
		task.setTaskType(edgeTask.getTaskType());
		
		if (utilizationModelCPU instanceof CpuUtilizationModel_Custom) {
			((CpuUtilizationModel_Custom)utilizationModelCPU).setTask(task);
		}
		
		return task;
	}
}
