/*
 * Title:        EdgeCloudSim - M/M/1 Queue model implementation
 * 
 * Description: 
 * MM1Queue implements M/M/1 Queue model for WLAN and WAN communication
 * 
 * Licence:      GPL - http://www.gnu.org/copyleft/gpl.html
 * Copyright (c) 2017, Bogazici University, Istanbul, Turkey
 */

 package edu.boun.edgecloudsim.network;
 import edu.boun.edgecloudsim.utils.ArrayUtils;
 import org.cloudbus.cloudsim.core.CloudSim;
 
 import edu.boun.edgecloudsim.core.SimManager;
 import edu.boun.edgecloudsim.core.SimSettings;
 import edu.boun.edgecloudsim.edge_client.Task;
 import edu.boun.edgecloudsim.edge_server.EdgeHost;
 import edu.boun.edgecloudsim.utils.Location;
 
 public class MM1Queue extends NetworkModel {
	 private double WlanPoissonMean; //seconds
	 private double WanPoissonMean; //seconds
	 private double avgTaskInputSize; //bytes
	 private double avgTaskOutputSize; //bytes
	 private int maxNumOfClientsInPlace;
 
	 public MM1Queue(int _numberOfMobileDevices, String _simScenario) {
		 super(_numberOfMobileDevices, _simScenario);
	 }
 
 
	 @Override
	 public void initialize() {
		 WlanPoissonMean=0;
		 WanPoissonMean=0;
		 avgTaskInputSize=0;
		 avgTaskOutputSize=0;
		 maxNumOfClientsInPlace=0;
 
		 //Calculate interarrival time and task sizes
		 double numOfTaskType = 0;
		 SimSettings SS = SimSettings.getInstance();
		 // 修复: 检查对象是否为数组并安全转换
		 Object taskLookUpTable = SS.getTaskLookUpTable();
		 int tableLength = 0;
		 if (taskLookUpTable instanceof double[][]) {
			 tableLength = ((double[][])taskLookUpTable).length;
		 } else {
			 System.err.println("Warning: taskLookUpTable is not a double[][]");
			 return; // 无法处理，提前返回
		 }
		 
		 for (int i=0; i<tableLength; i++) {
			 // 修复: 安全转换数组访问
			 double[][] lookUpTable = (double[][])taskLookUpTable;
			 double weight = lookUpTable[i][0]/(double)100;
			 
			 if(weight != 0) {
				 WlanPoissonMean += (lookUpTable[i][2])*weight;
 
				 double percentageOfCloudCommunication = lookUpTable[i][1];
				 WanPoissonMean += (WlanPoissonMean)*((double)100/percentageOfCloudCommunication)*weight;
 
				 avgTaskInputSize += lookUpTable[i][5]*weight;
 
				 avgTaskOutputSize += lookUpTable[i][6]*weight;
 
				 numOfTaskType++;
			 }
		 }
 
		 WlanPoissonMean = WlanPoissonMean/numOfTaskType;
		 avgTaskInputSize = avgTaskInputSize/numOfTaskType;
		 avgTaskOutputSize = avgTaskOutputSize/numOfTaskType;
	 }
 
	 /**
	  * source device is always mobile device in our simulation scenarios!
	  */
	 @Override
	 public double getUploadDelay(int sourceDeviceId, int destDeviceId, Task task) {
		 double delay = 0;
		 Location accessPointLocation = SimManager.getInstance().getMobilityModel().getLocation(sourceDeviceId,CloudSim.clock());
 
		 //mobile device to cloud server
		 if(destDeviceId == SimSettings.CLOUD_DATACENTER_ID){
			 double wlanDelay = getWlanUploadDelay(accessPointLocation, CloudSim.clock());
			 double wanDelay = getWanUploadDelay(accessPointLocation, CloudSim.clock() + wlanDelay);
			 if(wlanDelay > 0 && wanDelay >0)
				 delay = wlanDelay + wanDelay;
		 }
		 //mobile device to edge orchestrator
		 else if(destDeviceId == SimSettings.EDGE_ORCHESTRATOR_ID){
			 delay = getWlanUploadDelay(accessPointLocation, CloudSim.clock()) +
					 SimSettings.getInstance().getInternalLanDelay();
		 }
		 //mobile device to edge device (wifi access point)
		 else if (destDeviceId == SimSettings.GENERIC_EDGE_DEVICE_ID) {
			 delay = getWlanUploadDelay(accessPointLocation, CloudSim.clock());
		 }
 
		 return delay;
	 }
 
	 /**
	  * destination device is always mobile device in our simulation scenarios!
	  */
	 @Override
	 public double getDownloadDelay(int sourceDeviceId, int destDeviceId, Task task) {
		 //Special Case -> edge orchestrator to edge device
		 if(sourceDeviceId == SimSettings.EDGE_ORCHESTRATOR_ID &&
				 destDeviceId == SimSettings.GENERIC_EDGE_DEVICE_ID){
			 return SimSettings.getInstance().getInternalLanDelay();
		 }
 
		 double delay = 0;
		 Location accessPointLocation = SimManager.getInstance().getMobilityModel().getLocation(destDeviceId,CloudSim.clock());
 
		 //cloud server to mobile device
		 if(sourceDeviceId == SimSettings.CLOUD_DATACENTER_ID){
			 double wlanDelay = getWlanDownloadDelay(accessPointLocation, CloudSim.clock());
			 double wanDelay = getWanDownloadDelay(accessPointLocation, CloudSim.clock() + wlanDelay);
			 if(wlanDelay > 0 && wanDelay >0)
				 delay = wlanDelay + wanDelay;
		 }
		 //edge device (wifi access point) to mobile device
		 else{
			 delay = getWlanDownloadDelay(accessPointLocation, CloudSim.clock());
 
			 EdgeHost host = (EdgeHost)(SimManager.
					 getInstance().
					 getEdgeServerManager().
					 getDatacenterList().get(sourceDeviceId).
					 getHostList().get(0));
 
			 //if source device id is the edge server which is located in another location, add internal lan delay
			 //in our scenario, serving wlan ID is equal to the host id, because there is only one host in one place
			 if(host.getLocation().getServingWlanId() != accessPointLocation.getServingWlanId())
				 delay += (SimSettings.getInstance().getInternalLanDelay() * 2);
		 }
 
		 return delay;
	 }
 
	 public int getMaxNumOfClientsInPlace(){
		 return maxNumOfClientsInPlace;
	 }
 
	 private int getDeviceCount(Location deviceLocation, double time){
		 int deviceCount = 0;
 
		 for(int i=0; i<numberOfMobileDevices; i++) {
			 Location location = SimManager.getInstance().getMobilityModel().getLocation(i,time);
			 if(location.equals(deviceLocation))
				 deviceCount++;
		 }
 
		 //record max number of client just for debugging
		 if(maxNumOfClientsInPlace<deviceCount)
			 maxNumOfClientsInPlace = deviceCount;
 
		 return deviceCount;
	 }
 
	 private double calculateMM1(double propagationDelay, int bandwidth /*Kbps*/, double PoissonMean, double avgTaskSize /*KB*/, int deviceCount){
		 double Bps=0, mu=0, lamda=0;
 
		 avgTaskSize = avgTaskSize * (double)1000; //convert from KB to Byte
 
		 Bps = bandwidth * (double)1000 / (double)8; //convert from Kbps to Byte per seconds
		 lamda = ((double)1/(double)PoissonMean); //task per seconds
		 mu = Bps / avgTaskSize ; //task per seconds
		 double result = (double)1 / (mu-lamda*(double)deviceCount);
 
		 result += propagationDelay;
 
		 return (result > 5) ? -1 : result;
	 }
 
	 private double getWlanDownloadDelay(Location accessPointLocation, double time) {
		 return calculateMM1(0.0,
				 (int) SimSettings.getInstance().getWlanBandwidth(),
				 WlanPoissonMean,
				 avgTaskOutputSize,
				 getDeviceCount(accessPointLocation, time));
	 }
 
	 private double getWlanUploadDelay(Location accessPointLocation, double time) {
		 return calculateMM1(0.0,
				 (int) SimSettings.getInstance().getWlanBandwidth(),
				 WlanPoissonMean,
				 avgTaskInputSize,
				 getDeviceCount(accessPointLocation, time));
	 }
 
	 private double getWanDownloadDelay(Location accessPointLocation, double time) {
		 int bandwidth = (int)SimSettings.getInstance().getWanBandwidth();
		 return calculateMM1(SimSettings.getInstance().getWanPropagationDelay(),
				 bandwidth,
				 WanPoissonMean,
				 avgTaskOutputSize,
				 getDeviceCount(accessPointLocation, time));
	 }
 
	 private double getWanUploadDelay(Location accessPointLocation, double time) {
		 // 修复: 将double转换为int时使用Math.round或显式转换
		 int bandwidth = (int) SimSettings.getInstance().getWanBandwidth();
		 return calculateMM1(SimSettings.getInstance().getWanPropagationDelay(),
				 bandwidth,
				 WanPoissonMean,
				 avgTaskInputSize,
				 getDeviceCount(accessPointLocation, time));
	 }
 
	 @Override
	 public void uploadStarted(Location accessPointLocation, int destDeviceId) {
 
	 }
 
	 @Override
	 public void uploadFinished(Location accessPointLocation, int destDeviceId) {
 
	 }
 
	 @Override
	 public void downloadStarted(Location accessPointLocation, int sourceDeviceId) {
 
	 }
 
	 @Override
	 public void downloadFinished(Location accessPointLocation, int sourceDeviceId) {
 
	 }
 }