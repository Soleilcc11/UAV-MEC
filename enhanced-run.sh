#!/bin/bash
# 确保目录存在
mkdir -p config
mkdir -p results/output

# 清除旧结果
rm -f results/output/*.csv

# 设置更大的设备数量
NUM_DEVICES=100

echo "===== 开始UAV-MEC模拟 ====="
echo "移动设备数量: $NUM_DEVICES"
echo "配置文件: config/simulation_settings.xml"
echo "输出目录: results/output"

# 使用直接的Java命令运行
java -cp "target/classes:src/main/resources/lib/*" edu.boun.edgecloudsim.uav.UAVMECMainApp \
  config/simulation_settings.xml \
  config/edge_devices.xml \
  config/applications.xml \
  results/output \
  1 \
  $NUM_DEVICES

# 检查结果
echo -e "\n===== 检查模拟结果 ====="
for file in results/output/*.csv; do
  COUNT=$(wc -l < "$file")
  echo "文件: $file - 行数: $COUNT"
  if [ $COUNT -le 1 ]; then
    echo "警告: 文件只包含标题行!"
  fi
done

echo -e "\n===== 模拟完成 ====="
