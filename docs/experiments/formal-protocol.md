# Formal protocol 1.2 experiment

## Scope

The formal matrix evaluates the real Java/EdgeCloudSim GymBridge 1.2 system.
The legacy synthetic Python environment and protocol 1.1 checkpoints/results are
excluded.

The authoritative specification is
[`experiments/formal_protocol_1_2_10seed.json`](../../experiments/formal_protocol_1_2_10seed.json).

## Environment and actions

- Urban training/validation: 2 UAVs, 25 users, 3600 seconds.
- Rural held-out: 2 UAVs, 15 users, 3600 seconds.
- Targets: local, cloud, UAV 0, UAV 1.
- Movement: bounded XYZ control for both UAVs.
- Access link: probabilistic LoS/NLoS, SNR, Shannon rate, four channels.
- Cloud: one deterministic UAV relay retained for both directions, with shared
  backhaul.
- Transition discount: `gamma_0 ** (delta_t / tau)`.

## Learned policies and baselines

Main comparisons:

1. `masked_parameterized_action_ddpg`;
2. `mixed_action_ppo`;
3. `mixed_action_td3`.

They share the same observations, actions, reward, budget, environment, and
training-seed starts. DDPG and TD3 are parameterized-action extensions, not
claims about the original continuous-only algorithms.

`masked_dqn_zero_movement` is an action-capability ablation. It observes the same
state and chooses a masked target, but every movement value is zero.

Non-learning baselines are masked random, minimum estimated delay, local only,
and cloud only.

## Replication and seed partitions

- Each learned policy has 10 independently trained checkpoints.
- All algorithms use the same 10 training seed starts.
- Each checkpoint receives exactly 200,000 real GymBridge interactions.
- Urban validation uses paired seeds `201–210`.
- Rural held-out evaluation uses paired seeds `301–310`.
- Held-out seeds cannot tune hyperparameters, select checkpoints, or stop
  training.
- The selected checkpoint is the final fixed-budget checkpoint regardless of
  observed performance.

The learned-policy replication unit is the independently trained checkpoint.
Metrics are averaged over the ten environment seeds within a checkpoint, then
uncertainty is calculated across ten checkpoint means. The 100
checkpoint-by-environment episodes are not treated as 100 independent training
runs.

## Metrics

Primary metric:

- successful tasks completed by their deadline divided by settled tasks.

Secondary metrics:

- mean and P95 task latency;
- UE and UAV energy;
- throughput;
- constraint violations;
- mean and maximum UAV queue length;
- local, cloud, and UAV resource utilization;
- local/cloud/UAV target ratios and total offload ratio.

All physical values originate in Java. Raw episode rows are retained even when a
learned policy loses to a baseline.

## Uncertainty

The summary uses 20,000 deterministic nonparametric bootstrap resamples.

- Candidate intervals resample the ten independent checkpoint means.
- Baseline intervals resample the ten paired environment seeds.
- Candidate-minus-baseline effects first subtract the same environment seed
  within each checkpoint, average within checkpoint, and then resample the ten
  checkpoint differences.

These intervals quantify uncertainty under the declared seed population; they
do not prove universal superiority.

## Pilot gate

Before the formal matrix, run:

```bash
.venv/bin/python scripts/run_pilot_audit.py
```

The pilot covers all four algorithms with one shared training seed and 512
interactions each. It must report `passed`, contain no non-finite values, keep
exponential saturation at or below 1.5%, reconcile tasks/latencies/throughput
with the simulation clock, and match all provenance hashes.

## Formal execution

```bash
.venv/bin/python scripts/run_formal_experiments.py --dry-run --max-runs 1
.venv/bin/python scripts/run_formal_experiments.py
.venv/bin/python scripts/summarize_formal_experiments.py
.venv/bin/python scripts/plot_formal_results.py
```

The runner requires a clean committed tree, performs a clean Java build, and
uses a fresh JVM for each training or evaluation report. A completed item is
skipped only when its protocol/config/Git/source/environment/seed/budget and
checkpoint hashes all match.

The aggregate audit refuses to complete unless every algorithm has ten
checkpoints, every checkpoint contains exactly 200,000 contiguous transitions,
all paired evaluations pass, repeated baselines are deterministic, and the
entire matrix shares one Git/source/config provenance.
