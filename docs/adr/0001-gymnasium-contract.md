# ADR 0001: Gymnasium contract for UAV-MEC

- Status: Accepted
- Date: 2026-07-17

## Decision

The environment uses **task-arrival decision epochs**. One `step(action)` applies to exactly one current task. The backend applies the offloading target and all UAV movement commands, then advances EdgeCloudSim until the next task-arrival decision point or episode termination.

This avoids a variable-length list of offloading decisions and preserves a fixed Gymnasium action space.

## Action

`action_space` is a `Dict`:

- `target: Discrete(3 + number_of_uavs)`
  - `0`: local mobile execution
  - `1`: cloud
  - `2`: edge
  - `3 + i`: UAV `i`
- `movement: Box(-1, 1, shape=(number_of_uavs, 3))`
  - normalized XYZ movement commands
  - the Java backend scales each row by the movement budget for the elapsed simulation time
  - physical and energy bounds are enforced by the backend

The observation includes `action_mask`; selecting a masked target is a constraint violation, not an implicit fallback.

## Observation

All continuous values are `float32` and normalized to `[0, 1]` using configuration limits.

- `time: (1,)`: current simulation time / configured time limit
- `task: (7,)`: upload size, output size, MI, required cores, deadline, user X, user Y
- `resources: (3, 3)`: local/cloud/edge available capacity, normalized queue/load, and link delay
- `uavs: (U, 8)`: X, Y, Z, remaining energy, queue, capacity, upload delay, download delay
- `action_mask: (3 + U,)`: legal local/cloud/edge/UAV targets for the current task

No silent padding, truncation, integer coercion, or fallback target is allowed.

## Reward

The environment computes one bounded scalar after the selected task settles:

```text
r = clip(
    success
    - 0.35 * min(latency / deadline, 2)
    - 0.15 * min(UE energy / UE budget, 2)
    - 0.20 * min(UAV energy / UAV budget, 2)
    - 0.30 * min(constraint violations, 1),
    -1,
    1
)
```

The unweighted physical metrics and normalized reward components are returned in `info` for auditability.

## Episode semantics

- `terminated=True`: all configured workload tasks have reached a terminal state and no upload, execution, or download remains in flight; or the system reaches an unrecoverable physical terminal state.
- `truncated=True`: configured simulation-time or external step limit is reached before natural termination.
- They are never collapsed into a single `done` flag.

## Seeding

`reset(seed)` must recreate the complete Java simulation and propagate the seed to workload generation, mobility, UAV initialization, algorithm sampling, and all other random sources. Repeating `reset(seed)` and the same action sequence must reproduce observations, rewards, terminal flags, and physical metrics.

## Validation

- `gymnasium.utils.env_checker.check_env` must pass without `skip_render_check` exceptions other than no render mode being configured.
- Every observation must satisfy `observation_space.contains`.
- Every accepted action must satisfy `action_space.contains`.
- Contract tests use a deterministic protocol backend only; it is not a training simulator. The production backend is supplied by GymBridge in phase 3.
