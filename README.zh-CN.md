# UAV-MEC

[English](README.md) | [简体中文](README.zh-CN.md)

这是一个基于 EdgeCloudSim 的事件驱动 UAV 辅助 MEC 研究系统，并通过严格的
Gymnasium 边界训练和评估策略。Java 是物理状态与状态转移的唯一来源，Python
只通过 GymBridge 1.2 与仿真器交互。

项目定位是系统与实证研究：不把算法名称本身当成创新，也不把 protocol 1.1
的历史结果当作 protocol 1.2 的证据。

## 固定的 protocol 1.2

每个 Gym step 对应一次任务到达决策。此前提交的任务继续在 EdgeCloudSim 中
并发执行，因此问题建模为事件驱动 SMDP/POMDP。每次转移的有效折扣为：

```text
gamma_k = gamma_0 ** (delta_t_k / tau)
```

并通过 `info["effective_discount"]` 返回。

动作空间：

```text
target   = Discrete(2 + U)       # 本地、云、UAV 0 ... UAV U-1
movement = Box(-1, 1, (U, 3))    # 归一化 XYZ 位移
```

动作空间中没有固定 Edge。云任务由后端选择预计往返接入时延最小的活动 UAV
作为中继，并在上传、下载全过程保持同一中继。

观测空间：

```text
time(1)
delta_time(1)
task(7)
resources(2, 3)
uavs(U, 8)
action_mask(2 + U)
```

2 UAV 时共有 31 个连续特征和 4 位 mask。选择被 mask 的合法索引会生成一条
可审计的失败转移和约束违规，不会静默回退。

空地链路使用概率 LoS/NLoS 路损、接收信噪比和 Shannon 容量。接入总带宽
划分为 4 个正交信道，超出信道数的传输按确定性方式共享；并发云任务也共享
UAV—云回传带宽。

完整契约、奖励、终态和溯源规则见
[ADR 0001](docs/adr/0001-gymnasium-contract.md)。

## 配置规模

| 配置 | UAV | 移动设备 | 区域 | 初始高度 | 时长 | 用途 |
|---|---:|---:|---:|---:|---:|---|
| Pilot | 1 | 20 | 400 × 400 m | 50 m | 300 s | 端到端审计 |
| Urban | 2 | 25 | 1000 × 1000 m | 80 m | 3600 s | 训练/验证 |
| Rural | 2 | 15 | 2000 × 2000 m | 80 m | 3600 s | held-out 评估 |

配置文件位于 `src/main/resources/config/{pilot,urban,rural}`。正式 runner 会把
移动设备数量作为显式参数传给 GymBridge。

## 算法与基线

三个主算法共享完全相同的观测、混合动作、奖励、交互预算和配对训练种子：

- masked parameterized-action DDPG：单 critic、straight-through masked
  离散目标头、连续移动头；
- mixed-action PPO：masked categorical 目标头和 tanh-squashed 移动头；
- mixed-action TD3：双 critic、延迟 actor 更新，仅对移动分支做目标平滑。

Masked DQN 使用相同观测和奖励，但移动恒为零。它是“动作能力消融”，不能当作
同能力的公平主基线。

非学习基线包括 masked random、minimum estimated delay、local only 和
cloud only。

## 安装与回归测试

需要 Python 3.11+、Maven 和 JDK。

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt

mvn test -q
.venv/bin/pytest -q
mvn package -q -DskipTests
```

真实 Java↔Python 集成测试需要在 `127.0.0.1` 绑定临时端口。

## 手动启动 GymBridge

以下命令启动城市配置：

```bash
mvn package -q -DskipTests

java -cp 'target/classes:target/lib/*' \
  edu.boun.edgecloudsim.uav.GymBridgeMain \
  12470 \
  src/main/resources/config/urban/simulation_settings.xml \
  src/main/resources/config/edge_devices.xml \
  src/main/resources/config/urban/applications.xml \
  25
```

运行结束后应关闭服务。正式自动化会为每份训练或评估报告启动全新的 JVM。

## 试运行验收

Pilot 使用同一个训练种子依次运行 4 个算法，每个算法进行 512 次真实 GymBridge
交互，再做 5 个配对种子的评估：

```bash
.venv/bin/python scripts/run_pilot_audit.py --dry-run
.venv/bin/python scripts/run_pilot_audit.py
```

以下任一情况会使审计失败：出现 NaN/Inf；任务数、吞吐量与仿真时钟不一致；
时延样本不完整；Git、源码、环境、配置或 checkpoint 哈希不一致；指数输入变量
上限饱和率超过 1.5%。结果写入
`results/pilot/protocol_1_2_1seed_v1/pilot_audit.json`。

## 正式实验

机器可读协议为
[formal_protocol_1_2_10seed.json](experiments/formal_protocol_1_2_10seed.json)：

- 4 个学习算法；
- 每个算法使用相同的一组 10 个独立训练种子；
- 每个 checkpoint 固定 200,000 次真实 GymBridge 交互；
- 城市验证种子 `201–210`；
- 农村 held-out 种子 `301–310`；
- 对独立 checkpoint 均值执行 20,000 次非参数 bootstrap。

只能在已经提交且干净的工作树上运行：

```bash
.venv/bin/python scripts/run_formal_experiments.py --dry-run --max-runs 1
.venv/bin/python scripts/run_formal_experiments.py
.venv/bin/python scripts/summarize_formal_experiments.py
.venv/bin/python scripts/plot_formal_results.py
```

完整矩阵耗时很长，不会在实现阶段或 CI 中自动启动。runner 支持恢复，但只有当
配置、Git、源码、环境、预算、种子和 checkpoint 哈希全部一致时才会跳过结果。

主指标为截止时间内成功率。次要指标包括平均/P95 时延、UE/UAV 能耗、吞吐量、
约束次数、队列长度、资源利用率，以及本地/云/UAV/总卸载比例。学习策略的置信
区间以 10 个独立训练 checkpoint 的均值为重采样单位；配对差值先按同一环境
种子减去基线，再进行重采样。

详细说明见[正式协议](docs/experiments/formal-protocol.md)和
[实验产物发布计划](docs/experiments/artifact-publication.md)。

## 可复现边界

GymBridge `hello` 会返回实际 XML、Java class、Java 源码树、Git commit 的哈希
以及 class 是否过期。Python 在 reset 前核对这些信息。报告还记录算法配置、
种子分区、原始转移、checkpoint SHA-256 和环境清单。

1.2 loader 会拒绝 protocol 1.1 checkpoint。旧版
`experiments/formal_protocol_1_1_10seed.json`、合成 Python 环境和历史结果只作
审计归档，不参与新结论。
