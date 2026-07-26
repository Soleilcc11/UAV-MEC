# ADR 0001: GymBridge 1.2 SMDP/POMDP contract

- Status: Accepted
- Date: 2026-07-24
- Supersedes: protocol 1.1 contract

## Decision boundary

Java/EdgeCloudSim owns the complete simulator state, event queue, workload,
mobility, resources, network, and terminal settlement. Python may only call
`hello`, `reset`, `step`, and `close`.

A decision epoch occurs at each task arrival. Applying one action does not wait
for that task alone to complete: previously submitted tasks remain in flight and
all EdgeCloudSim events run until the next task arrival or episode end. The
underlying process is an SMDP; because the agent receives a compressed view of
the full simulator state, the learning interface is also partially observable.

For elapsed simulation time `delta_t_k`, discount base `gamma_0`, and time unit
`tau`, the backend supplies:

```text
gamma_k = gamma_0 ** (delta_t_k / tau)
```

Algorithms must store this transition-specific value. A fixed per-step gamma is
not protocol compliant.

## Action

```text
Dict(
  target   = Discrete(2 + U),
  movement = Box(-1, 1, shape=(U, 3), dtype=float32)
)
```

Target mapping:

- `0`: local mobile execution;
- `1`: cloud through a UAV relay;
- `2 + i`: UAV `i`.

There is no fixed-edge target. For a cloud action, Java deterministically chooses
the active UAV with minimum predicted upload-plus-download access delay. Ties
resolve by ascending UAV ID. The chosen ID is stored on the task and reused on
download.

Each movement row is an XYZ direction. Java limits its norm, scales it by
`speed × elapsed simulation time`, enforces physical bounds, and charges actual
flight/hover energy. Simultaneous task arrivals provide zero extra movement time.

Selecting a masked but in-range target is a constraint violation and failed task
transition; it is never silently redirected. An out-of-range action is a protocol
error.

## Observation

All continuous fields are finite `float32` values in `[0, 1]`:

- `time: (1,)`;
- `delta_time: (1,)`;
- `task: (7,)`: input, output, MI, cores, deadline, user X, user Y;
- `resources: (2, 3)`: local/cloud capacity, load, and link-delay proxy;
- `uavs: (U, 8)`: XYZ, energy, queue, capacity, upload delay, download delay;
- `action_mask: MultiBinary(2 + U)`.

For `U=2`, the continuous part contains 31 values and the mask contains four
bits. Exponential task variables use a configured 99th-percentile scale; pilot
acceptance requires their upper-bound saturation rate to remain at or below
1.5%.

## Network

The air-ground path loss is free-space loss plus the expected LoS/NLoS excess
loss. LoS probability depends on elevation angle and environment parameters.
Received SNR uses configured transmit power and noise spectral density, and link
rate is `B log2(1 + SNR)`.

Total access bandwidth is divided across four orthogonal channels. The least
loaded deterministic allocation is equivalent to dividing each channel by
`ceil(active transfers / channels)`. Cloud transfers include both the access hop
and UAV-cloud backhaul; concurrent cloud transfers share backhaul bandwidth.

## Reward and info

Tasks settling between two decision epochs contribute bounded success, latency,
and UE-energy terms. Interval UAV energy and constraint penalties are charged
once. Several tasks may settle in one transition, so the interval reward is not
restricted to `[-1, 1]`.

`info` includes normalized reward components and raw physical/audit data:

- elapsed simulation time and effective discount;
- settled/total/success/failed/in-flight task counts;
- task latency samples, UE/UAV energy, and constraints;
- throughput;
- selected cloud relay;
- active access/backhaul transfers;
- UAV queue total/maximum;
- local/cloud/UAV resource utilization.

## Episode and seed semantics

`terminated=True` means every configured task has naturally reached a terminal
state. A GymBridge `truncated=True` means the simulation horizon ended first;
Java explicitly fails work that was already in flight. The training runner's
fixed interaction-budget cutoff is recorded separately as
`interaction_budget_truncated=True` and is treated as a truncation by the
learning buffer without inventing an extra simulator transition. The remaining
episode state is discarded when that report's JVM closes. `terminated` and
`truncated` cannot both be true.

`reset(seed)` rebuilds the Java simulation and propagates the seed to workload,
mobility, UAV initialization, and other simulator randomness. The same seed and
action sequence must be bitwise deterministic across fresh JVMs.

## Checkpoints and provenance

GymBridge `hello` returns protocol version, XML manifest, Git commit, Java source
hash, loaded class hash, and stale-class status. Python verifies them before
reset and records them in every report.

Protocol 1.2 checkpoints include algorithm state, optimizers, pending
rollout/replay, normalizer, RNG streams, counters, configuration, UAV count, and
training-seed cursor. Protocol 1.1 checkpoints are rejected rather than migrated
implicitly.

## Required validation

- Gymnasium environment checker and shape/mask tests;
- LoS monotonicity, Shannon-rate monotonicity, and four-channel competition;
- fixed cloud relay across both directions;
- local/cloud/UAV execution paths and urban/rural/pilot config tests;
- train/save/load tests for PPO, DDPG, TD3, and DQN zero movement;
- real Java-Python round trip and fresh-JVM bitwise determinism;
- pilot checks for finite values, saturation, task/throughput/clock consistency,
  and all provenance hashes.
