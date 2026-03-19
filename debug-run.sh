#!/bin/bash
# debug-run.sh

# Ensure directories exist
mkdir -p config
mkdir -p results/output

# Clear previous results
rm -f results/output/*.csv

# Set parameters
NUM_DEVICES=200
SIM_TIME=10800  # 3 hours in seconds

# Create enhanced config files with debugging parameters
cat > config/simulation_settings.xml << EOF
<?xml version="1.0" encoding="UTF-8"?>
<simulation_settings>
    <general>
        <simulation_time>${SIM_TIME}</simulation_time>
        <warm_up_period>60</warm_up_period>
        <vm_load_check_interval>1</vm_load_check_interval>
        <location_check_interval>1</location_check_interval>
        <file_log_enabled>true</file_log_enabled>
        <deep_file_log_enabled>true</deep_file_log_enabled>
    </general>
    <uav>
        <num_of_uavs>20</num_of_uavs>
        <uav_max_energy>10000</uav_max_energy>
        <uav_initial_height>50</uav_initial_height>
    </uav>
    <simulation_space>
        <x>1000</x>
        <y>1000</y>
        <z>200</z>
    </simulation_space>
    <scenario>
        <single_tier>true</single_tier>
        <two_tier>false</two_tier>
        <two_tier_with_EO>false</two_tier_with_EO>
    </scenario>
    <orchestrator>
        <random>true</random>
        <ddpg>false</ddpg>
    </orchestrator>
</simulation_settings>
EOF

cat > config/applications.xml << EOF
<?xml version="1.0" encoding="UTF-8"?>
<applications>
    <application name="REAL_TIME_VIDEO">
        <usage_percentage>100</usage_percentage>
        <prob_cloud_selection>20</prob_cloud_selection>
        <poisson_interarrival>1</poisson_interarrival>
        <active_period>100</active_period>
        <idle_period>1</idle_period>
        <data_upload>1500</data_upload>
        <data_download>25</data_download>
        <task_length>3000</task_length>
        <required_core>1</required_core>
        <vm_utilization_on_edge>20</vm_utilization_on_edge>
        <vm_utilization_on_cloud>5</vm_utilization_on_cloud>
        <vm_utilization_on_mobile>80</vm_utilization_on_mobile>
        <delay_sensitivity>0.8</delay_sensitivity>
        <max_delay_requirement>100</max_delay_requirement>
    </application>
</applications>
EOF

echo "===== Starting Debug UAV-MEC Simulation ====="
echo "Mobile Devices: $NUM_DEVICES"
echo "Simulation Time: $SIM_TIME seconds"
echo "Task Interarrival: 1 second"
echo "Idle Period: 1 second"

# Run with verbose output
java -cp "target/classes:src/main/resources/lib/*" edu.boun.edgecloudsim.uav.UAVMECMainApp \
  config/simulation_settings.xml \
  config/edge_devices.xml \
  config/applications.xml \
  results/output \
  1 \
  $NUM_DEVICES > simulation-debug.log 2>&1

# Check results
echo -e "\n===== Checking Simulation Results ====="
for file in results/output/*.csv; do
  COUNT=$(wc -l < "$file")
  echo "File: $file - Lines: $COUNT"
  if [ $COUNT -le 1 ]; then
    echo "WARNING: File only contains headers!"
  else
    echo "SUCCESS: File contains data rows!"
    head -n 3 "$file"
  fi
done

echo -e "\n===== Examining Log for Issues ====="
grep -i "error\|exception\|fail" simulation-debug.log

echo -e "\n===== Simulation Complete ====="
