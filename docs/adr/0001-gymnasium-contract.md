# ADR 0001: Gymnasium contract for UAV-MEC

- Status: Accepted
- Date: 2026-07-17

## Decision

The environment uses **task-arrival decision epochs**. One `step(action)` applies to exactly one current task. The backend applies the offloading target and all UAV movement commands, then advances EdgeCloudSim until the next task-arrival decision point or episode termination. Previously submitted tasks remain in flight, so execution and resource contention are not serialized by the Gym interface.

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
  - the Java backend scales each row by `UAV speed × elapsed simulation time` since the previous decision; simultaneous arrivals have zero additional movement budget
  - movement energy is based on actual bounded flight distance/time, while the simulation tick accounts for baseline hover energy
  - physical and energy bounds are enforced by the backend

The observation includes `action_mask`; selecting a masked target is a constraint violation, not an implicit fallback.

## Observation

All continuous values are `float32` and normalized to `[0, 1]` using configuration limits.

- `time: (1,)`: current simulation time / configured time limit
- `task: (7,)`: upload size, output size, MI, required cores, deadline, user X, user Y
- `resources: (3, 3)`: local/cloud/edge available capacity, live VM load, and task-size-aware link delay; an unavailable resource has a zero action mask entry
- `uavs: (U, 8)`: X, Y, Z, remaining energy, queue, capacity, upload delay, download delay
- `action_mask: (3 + U,)`: legal local/cloud/edge/UAV targets for the current task

No silent padding, truncation, integer coercion, or fallback target is allowed.

## Reward

For the set `S_t` of tasks settled since the preceding decision epoch, the
backend computes:

```text
r_t = sum over i in S_t of clip(
          success_i
          - 0.35 * min(latency_i / deadline_i, 2)
          - 0.15 * min(UE energy_i / UE budget, 2),
          -1, 1
      )
      - 0.20 * min(interval UAV energy / UAV budget, 2)
      - 0.30 * min(interval constraint violations, 1)
```

The reward returned at a decision epoch is the sum of task contributions that settled since the previous epoch, with interval UAV-energy and boundary penalties counted once. It can therefore be zero/negative when no task settles, or exceed 1 when several tasks settle concurrently. `settled_in_transition`, cumulative task counts, unweighted physical metrics, and normalized reward components are returned in `info` for auditability.

## Episode semantics

- `terminated=True`: all configured workload tasks have reached a terminal state and no upload, execution, or download remains in flight; or the system reaches an unrecoverable physical terminal state.
- `truncated=True`: configured simulation-time or external step limit is reached before natural termination. Any in-flight task is explicitly settled as failed at the configured horizon.
- They are never collapsed into a single `done` flag.

## Seeding

`reset(seed)` must recreate the complete Java simulation and propagate the seed to workload generation, mobility, UAV initialization, algorithm sampling, and all other random sources. Repeating `reset(seed)` and the same action sequence must reproduce observations, rewards, terminal flags, and physical metrics.

## Runtime provenance

GymBridge protocol 1.1 `hello` returns SHA-256 hashes for the actual XML inputs, loaded Java class artifact, and Java source tree, plus Git HEAD and a stale-class check. The Python experiment client compares environment hashes, Git commit, and source-tree hash before reset and records the service provenance in every report.

## Validation

- `gymnasium.utils.env_checker.check_env` must pass without `skip_render_check` exceptions other than no render mode being configured.
- Every observation must satisfy `observation_space.contains`.
- Every accepted action must satisfy `action_space.contains`.
- Contract tests use a deterministic protocol backend only; it is not a training simulator. The production backend is supplied by GymBridge in phase 3.
