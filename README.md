# UAV-MEC

[English](README.md) | [简体中文](README.zh-CN.md)

An event-driven UAV-assisted MEC research system built on EdgeCloudSim with a
strict Gymnasium boundary. Java is the only source of physical state and
transitions; Python trains and evaluates policies through GymBridge 1.2.

The repository supports system and empirical research. It does not treat an
algorithm name by itself as a novelty claim, and it does not reuse protocol 1.1
results as evidence for protocol 1.2.

## Frozen protocol 1.2

Each Gym step is one task-arrival decision epoch. Previously submitted work
continues concurrently in EdgeCloudSim, so the process is modeled as an
event-driven SMDP/POMDP. The effective transition discount is

```text
gamma_k = gamma_0 ** (delta_t_k / tau)
```

and is returned as `info["effective_discount"]`.

The action is:

```text
target   = Discrete(2 + U)       # local, cloud, UAV 0 ... UAV U-1
movement = Box(-1, 1, (U, 3))    # normalized XYZ movement
```

There is no fixed-edge action. A cloud task is relayed through the active UAV
with the smallest predicted round-trip access delay, and the same relay is
retained for upload and download.

The observation is:

```text
time(1)
delta_time(1)
task(7)
resources(2, 3)
uavs(U, 8)
action_mask(2 + U)
```

For two UAVs this is 31 continuous values and a four-bit mask. Masked in-range
targets produce an auditable failed transition and constraint violation; they
never silently fall back.

The air-ground link uses probabilistic LoS/NLoS path loss, received SNR, and
Shannon capacity. Access bandwidth is split across four orthogonal channels;
excess transfers time-share channels deterministically. UAV-cloud backhaul
bandwidth is also shared by concurrent cloud transfers.

Full semantics, reward, terminal behavior, and provenance are frozen in
[ADR 0001](docs/adr/0001-gymnasium-contract.md).

## Configurations

| Profile | UAVs | Mobile devices | Area | Height | Horizon | Role |
|---|---:|---:|---:|---:|---:|---|
| Pilot | 1 | 20 | 400 × 400 m | 50 m initial | 300 s | end-to-end audit |
| Urban | 2 | 25 | 1000 × 1000 m | 80 m initial | 3600 s | training/validation |
| Rural | 2 | 15 | 2000 × 2000 m | 80 m initial | 3600 s | held-out evaluation |

The profile XML files are under `src/main/resources/config/{pilot,urban,rural}`.
The formal runner passes the mobile-device count explicitly to GymBridge.

## Algorithms and baselines

The main learned comparisons share the same observation, mixed action, reward,
interaction budget, and paired training seeds:

- masked parameterized-action DDPG: one critic, straight-through masked target
  head, continuous movement head;
- mixed-action PPO: masked categorical target head and tanh-squashed movement;
- mixed-action TD3: twin critics, delayed actor update, and movement-only target
  smoothing.

Masked DQN observes the same state and reward but always emits zero movement.
It is an action-capability ablation, not a same-capability main baseline.

Deterministic or seeded non-learning baselines are masked random, minimum
estimated delay, local only, and cloud only.

## Setup and regression tests

Python 3.11+, Maven, and a JDK are required.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt

mvn test -q
.venv/bin/pytest -q
mvn package -q -DskipTests
```

The integration suite opens a loopback TCP port for the real Java-Python bridge.
Run it in an environment that permits binding to `127.0.0.1`.

## Start GymBridge manually

This example starts the urban profile:

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

Stop the service after a run. Formal automation starts a fresh JVM for every
training or evaluation report.

## Pilot acceptance gate

The pilot runs one paired training seed for all four algorithms with 512 real
GymBridge interactions each, followed by five-seed paired evaluation:

```bash
.venv/bin/python scripts/run_pilot_audit.py --dry-run
.venv/bin/python scripts/run_pilot_audit.py
```

It rejects the run if any value is NaN/Inf, task counts disagree with throughput
or simulation time, latency samples are incomplete, any provenance hash differs,
or the exponential-input saturation rate exceeds 1.5%. The result is written to
`results/pilot/protocol_1_2_1seed_v2/pilot_audit.json`.

## Formal experiment

The machine-readable protocol is
[formal_protocol_1_2_10seed.json](experiments/formal_protocol_1_2_10seed.json):

- four learned algorithms;
- 10 shared, independent training seeds per algorithm;
- 200,000 real GymBridge interactions per checkpoint;
- urban validation seeds `201–210`;
- rural held-out seeds `301–310`;
- 20,000-resample nonparametric bootstrap over independent checkpoint means.

Run only from a committed, clean tree:

```bash
.venv/bin/python scripts/run_formal_experiments.py --dry-run --max-runs 1
.venv/bin/python scripts/run_formal_experiments.py
.venv/bin/python scripts/summarize_formal_experiments.py
.venv/bin/python scripts/plot_formal_results.py
```

The formal matrix is intentionally not launched as part of implementation or
CI. It is a long-running research experiment. The runner is resumable and skips
only reports whose config, Git, source, environment, budget, seed, and checkpoint
hashes all match.

The primary metric is deadline success rate. Secondary outputs include mean and
P95 latency, UE/UAV energy, throughput, constraint count, queue length, resource
utilization, and local/cloud/UAV/offload ratios. Learned-policy confidence
intervals resample ten independently trained checkpoint means; paired effects
subtract the same-seed baseline before resampling.

See [the formal protocol](docs/experiments/formal-protocol.md) and
[artifact publication plan](docs/experiments/artifact-publication.md).

## Reproducibility boundary

GymBridge `hello` reports the loaded XML hashes, Java class hash, Java source
hash, Git commit, and stale-class status. Python verifies these before reset.
Reports additionally record algorithm config, seed partitions, raw transitions,
checkpoint SHA-256, and the exact environment manifest.

Protocol 1.1 checkpoints are rejected by 1.2 loaders. The old
`experiments/formal_protocol_1_1_10seed.json`, legacy synthetic Python
environment, and historical outputs remain audit-only and are excluded from new
claims.
