package edu.boun.edgecloudsim.task_generator;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.List;

import org.junit.jupiter.api.Test;

import edu.boun.edgecloudsim.core.SimSettings;
import edu.boun.edgecloudsim.utils.SimUtils;
import edu.boun.edgecloudsim.utils.TaskProperty;

class IdleActiveLoadGeneratorTest {
    @Test
    void configuredSeedReproducesTheWorkload() {
        SimSettings settings = SimSettings.getInstance();
        settings.initialize(
                "src/test/resources/config/simulation_settings.xml",
                "src/test/resources/config/edge_devices.xml",
                "src/test/resources/config/applications.xml");

        SimUtils.setSeed(settings.getSimulationSeed());
        IdleActiveLoadGenerator first = new IdleActiveLoadGenerator(2, 60, "SINGLE_TIER");
        first.initializeModel();

        SimUtils.setSeed(settings.getSimulationSeed());
        IdleActiveLoadGenerator second = new IdleActiveLoadGenerator(2, 60, "SINGLE_TIER");
        second.initializeModel();

        assertTaskListsEqual(first.getTaskList(), second.getTaskList());
    }

    private void assertTaskListsEqual(List<TaskProperty> first, List<TaskProperty> second) {
        assertEquals(first.size(), second.size());
        for (int i = 0; i < first.size(); i++) {
            TaskProperty left = first.get(i);
            TaskProperty right = second.get(i);
            assertEquals(left.getStartTime(), right.getStartTime());
            assertEquals(left.getMobileDeviceId(), right.getMobileDeviceId());
            assertEquals(left.getTaskType(), right.getTaskType());
            assertEquals(left.getLength(), right.getLength());
            assertEquals(left.getInputFileSize(), right.getInputFileSize());
            assertEquals(left.getOutputFileSize(), right.getOutputFileSize());
        }
    }
}
