package edu.boun.edgecloudsim.uav;

import java.util.HashMap;
import java.util.Map;

/** Integrates requested local-VM CPU load between CloudSim event times. */
final class TimeWeightedLocalUtilization {
    private final int vmCount;
    private final Map<Integer, ActiveTask> activeTasks = new HashMap<>();
    private final Map<Integer, Double> vmDemands = new HashMap<>();
    private double lastTime;
    private double intervalBusySeconds;

    TimeWeightedLocalUtilization(int vmCount) {
        this.vmCount = Math.max(0, vmCount);
    }

    void start(int cloudletId, int vmId, double requestedCpuFraction, double time) {
        advance(time);
        if (activeTasks.containsKey(cloudletId)) {
            throw new IllegalStateException("Duplicate local cloudlet id: " + cloudletId);
        }
        double demand = Math.max(0.0, Math.min(1.0, requestedCpuFraction));
        activeTasks.put(cloudletId, new ActiveTask(vmId, demand));
        vmDemands.merge(vmId, demand, Double::sum);
    }

    void finish(int cloudletId, double time) {
        advance(time);
        ActiveTask task = activeTasks.remove(cloudletId);
        if (task == null) {
            return;
        }
        double remaining = vmDemands.getOrDefault(task.vmId, 0.0) - task.demand;
        if (remaining <= 1e-12) {
            vmDemands.remove(task.vmId);
        } else {
            vmDemands.put(task.vmId, remaining);
        }
    }

    double drainBusySeconds(double time) {
        advance(time);
        double value = intervalBusySeconds;
        intervalBusySeconds = 0.0;
        return value;
    }

    private void advance(double time) {
        if (!Double.isFinite(time) || time < lastTime) {
            throw new IllegalArgumentException("Local utilization clock must be monotonic");
        }
        if (vmCount > 0) {
            double usedVmEquivalents = 0.0;
            for (double demand : vmDemands.values()) {
                usedVmEquivalents += Math.min(1.0, demand);
            }
            intervalBusySeconds += (time - lastTime) * usedVmEquivalents / vmCount;
        }
        lastTime = time;
    }

    private static final class ActiveTask {
        private final int vmId;
        private final double demand;

        private ActiveTask(int vmId, double demand) {
            this.vmId = vmId;
            this.demand = demand;
        }
    }
}
