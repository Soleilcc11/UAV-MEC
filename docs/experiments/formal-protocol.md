# Formal protocol 1.1 experiment

## Scope

The formal matrix compares masked random, minimum estimated delay,
mixed-action PPO, and mixed-action TD3 on the real Java/EdgeCloudSim
GymBridge. The legacy synthetic Python environment and historical DDPG/DQN
artifacts are excluded.

The machine-readable specification is
[`experiments/formal_protocol_1_1_10seed.json`](../../experiments/formal_protocol_1_1_10seed.json).

## Independent replication and budgets

- PPO and TD3 each use 10 independent training seed starts.
- Every learned-policy replicate receives exactly 4,096 real GymBridge
  interactions, 64 times the protocol 1.1 smoke budget.
- Validation uses 10 paired environment seeds (`201`–`210`).
- Held-out evaluation uses 10 disjoint paired seeds (`301`–`310`).
- Training seed ranges are spaced by 10,000, so even the upper bound of one new
  environment seed per interaction cannot overlap another replicate.
- Checkpoint selection is the final fixed-budget checkpoint. There is no
  held-out-driven checkpoint selection or hyperparameter tuning.

The formal unit of replication for learned policies is the independently
trained checkpoint. Each checkpoint is evaluated on all 10 paired environment
seeds; metrics are first averaged within checkpoint and uncertainty is then
computed across the 10 training-seed means. This avoids treating the 100
checkpoint-by-environment episodes as 100 independent training runs.

## Mixed-action TD3 definition

TD3's twin critics, delayed actor updates, and target-policy smoothing follow
the original algorithm described by Fujimoto, van Hoof, and Meger,
“[Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/abs/1802.09477).”
Because original TD3 assumes continuous actions, this repository explicitly
uses a parameterized-action extension:

- the actor emits masked target logits and a `tanh` movement vector;
- execution uses the legal target argmax and bounded movement;
- critics consume target one-hot plus movement;
- target smoothing applies only to movement;
- the discrete actor head uses a hard straight-through masked softmax during
  gradient calculation.
- time-limit transitions bootstrap only when GymBridge exposes a legal next
  decision; the terminal zero-observation has no valid target action and is
  therefore treated as non-bootstrappable.

This is named `mixed_action_td3`, not “vanilla TD3.” Parameterized action-space
background is described by Hausknecht and Stone,
“[Deep Reinforcement Learning in Parameterized Action Space](https://arxiv.org/abs/1511.04143),”
and Xiong et al.,
“[Parametrized Deep Q-Networks Learning](https://arxiv.org/abs/1810.06394).”

## Metrics and uncertainty

The primary held-out views are:

- total episode reward (higher is better);
- successful tasks divided by settled tasks (higher is better);
- latency seconds per settled task (lower is better);
- UE plus UAV energy joules per settled task (lower is better);
- constraint violations per episode (lower is better).

All raw metrics originate in Java and retain their physical units. The summary
uses a deterministic 20,000-resample nonparametric bootstrap. Candidate
intervals resample 10 independent training-seed means. Baseline intervals
resample 10 paired environment seeds. Paired-effect intervals subtract the
same environment seed's baseline before averaging within training checkpoint,
then resample the 10 checkpoint means.

These intervals quantify uncertainty for the declared seed populations; they
do not establish universal algorithm superiority or substitute for additional
workloads.

## Run and resume

The runner refuses a dirty tracked tree, performs a clean Maven package, starts
the exact protocol 1.1 services, and resumes only when the report matches the
formal config hash, Git commit, environment hash, source-tree hash, training
seed, interaction budget, and checkpoint hash. Every training report,
validation report, and held-out report runs in a fresh JVM. This isolates
EdgeCloudSim process-level static state and makes repeated deterministic
baselines comparable across learned checkpoints. Dry runs only print commands
and never create or replace the orchestration manifest.

At natural termination, GymBridge freezes the terminal observation, reward,
physical metrics, and episode counters atomically on the CloudSim event thread,
then stops the simulator after the current timestamp batch. Terminal UAV energy
therefore cannot depend on when the socket thread reads the response.

```bash
.venv/bin/python scripts/run_formal_experiments.py
.venv/bin/python scripts/summarize_formal_experiments.py
.venv/bin/python scripts/plot_formal_results.py
```

Useful bounded checks:

```bash
# Print the first matrix command without executing it.
.venv/bin/python scripts/run_formal_experiments.py --dry-run --max-runs 1

# Resume one algorithm; completed hash-valid runs are skipped.
.venv/bin/python scripts/run_formal_experiments.py --only mixed_action_td3
```

## Machine audit

The summary refuses to complete unless:

- all 20 training reports contain exactly 4,096 contiguous raw transitions;
- every checkpoint hash matches training and both evaluation reports;
- all evaluation audits pass on the declared paired seeds;
- Git and environment hashes are constant within the matrix;
- repeated deterministic baseline metrics are identical across reports;
- orchestration records the fresh-JVM-per-report lifecycle;
- each algorithm has exactly 10 independent training replicates.

Smoke results remain under `results/phase4/` and are not merged into the formal
matrix.
