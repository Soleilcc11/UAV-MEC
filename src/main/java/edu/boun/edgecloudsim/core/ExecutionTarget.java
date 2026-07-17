package edu.boun.edgecloudsim.core;

import java.util.Objects;

/**
 * Typed destination for an offloading decision.
 *
 * <p>The original UAV extension reused small integers for both execution
 * kinds and UAV identifiers. In particular, {@code 0} meant both local
 * execution and UAV 0. This value object keeps the target kind separate from
 * its resource identifier while still providing a collision-free adapter for
 * the legacy EdgeCloudSim integer API.</p>
 */
public final class ExecutionTarget {
    public static final int UAV_DEVICE_ID_BASE = 2000;

    public enum Type {
        LOCAL,
        CLOUD,
        EDGE,
        UAV
    }

    private final Type type;
    private final int resourceId;

    private ExecutionTarget(Type type, int resourceId) {
        this.type = Objects.requireNonNull(type, "type");
        this.resourceId = resourceId;
    }

    public static ExecutionTarget local() {
        return new ExecutionTarget(Type.LOCAL, SimSettings.MOBILE_DATACENTER_ID);
    }

    public static ExecutionTarget cloud() {
        return new ExecutionTarget(Type.CLOUD, SimSettings.CLOUD_DATACENTER_ID);
    }

    public static ExecutionTarget edge() {
        return edge(SimSettings.GENERIC_EDGE_DEVICE_ID);
    }

    public static ExecutionTarget edge(int resourceId) {
        if (resourceId < 0) {
            throw new IllegalArgumentException("Edge resource ID must be non-negative");
        }
        return new ExecutionTarget(Type.EDGE, resourceId);
    }

    public static ExecutionTarget uav(int uavId) {
        if (uavId < 0) {
            throw new IllegalArgumentException("UAV ID must be non-negative");
        }
        return new ExecutionTarget(Type.UAV, uavId);
    }

    public static ExecutionTarget fromLegacyDeviceId(int deviceId) {
        if (deviceId == SimSettings.MOBILE_DATACENTER_ID) {
            return local();
        }
        if (deviceId == SimSettings.CLOUD_DATACENTER_ID) {
            return cloud();
        }
        if (deviceId >= UAV_DEVICE_ID_BASE) {
            return uav(deviceId - UAV_DEVICE_ID_BASE);
        }
        return edge(deviceId);
    }

    public Type getType() {
        return type;
    }

    public int getResourceId() {
        return resourceId;
    }

    public int toLegacyDeviceId() {
        return type == Type.UAV ? UAV_DEVICE_ID_BASE + resourceId : resourceId;
    }

    @Override
    public boolean equals(Object other) {
        if (this == other) {
            return true;
        }
        if (!(other instanceof ExecutionTarget)) {
            return false;
        }
        ExecutionTarget that = (ExecutionTarget) other;
        return resourceId == that.resourceId && type == that.type;
    }

    @Override
    public int hashCode() {
        return Objects.hash(type, resourceId);
    }

    @Override
    public String toString() {
        return type + "(" + resourceId + ")";
    }
}
