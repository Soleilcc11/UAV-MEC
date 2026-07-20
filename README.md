# UAV-MEC

UAV-assisted MEC simulation with an event-driven EdgeCloudSim backend and a
strict Gymnasium training boundary.

## System boundary

- Java/EdgeCloudSim is the only source of simulation state and transitions.
- Python/Gymnasium is the only training and evaluation boundary.
- GymBridge exposes only `hello`, `reset`, `step`, and `close`.
- The frozen action is `Dict(target, movement)`: a masked categorical execution
  target plus a bounded `(number_of_uavs, 3)` movement command.
- `uav_mec_env.py`, the old algorithm files, and historical checkpoints/results
  are retained only for audit. They are not valid inputs to the new experiments.

The contract and reward are documented in
[`docs/adr/0001-gymnasium-contract.md`](docs/adr/0001-gymnasium-contract.md).

GymBridge protocol 1.1 binds each run to the listening Java service. `hello`
exposes the actual XML hashes, loaded class-artifact hash, Java source-tree hash,
Git HEAD, and a stale-class check. The experiment CLI rejects a server whose
configuration, commit, or source tree differs from the client checkout and
records the verified service provenance in its JSON output.

## Setup and regression tests

Python 3.11+ and Maven are required.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt

mvn test -q
.venv/bin/pytest -q
mvn package -q -DskipTests
```

PyTorch PPO is deliberately CPU-only at present so checkpoint/restart and
seeded tests do not depend on accelerator-specific kernels.

## Start the real GymBridge

Use explicit classpath, XML paths, port, and mobile-device count. The following
starts the development/training configuration with two UAVs and two mobile
devices:

```bash
java -cp 'target/classes:target/lib/*' \
  edu.boun.edgecloudsim.uav.GymBridgeMain \
  12347 \
  src/test/resources/config/simulation_settings.xml \
  src/test/resources/config/edge_devices.xml \
  src/test/resources/config/applications.xml \
  2
```

The held-out configuration changes workload intensity and deadline, UAV energy
parameters, and edge capacity:

```bash
java -cp 'target/classes:target/lib/*' \
  edu.boun.edgecloudsim.uav.GymBridgeMain \
  12348 \
  src/test/resources/config/heldout/simulation_settings.xml \
  src/test/resources/config/heldout/edge_devices.xml \
  src/test/resources/config/heldout/applications.xml \
  2
```

Stop each local server when its run finishes so an old Java process cannot
pollute a later reset.

## Reproducible baselines

The baseline evaluator uses paired seeds and records reward, termination,
settled/success counts, simulation time, and raw latency/energy/constraint
metrics returned by EdgeCloudSim.

```bash
.venv/bin/python -m uav_mec_gym.evaluation \
  --port 12347 --uavs 2 \
  --seeds 41 42 43 44 45 \
  --output results/phase4/baselines_test_config.json
```

`RandomMaskedPolicy` samples only enabled targets. The deterministic
`MinimumEstimatedDelayPolicy` uses the observable transfer-delay proxy and zero
movement.

## PPO training

The migrated PPO has a masked categorical target head and a tanh-squashed
Normal movement head. PPO ratios use the actual executed joint action log-prob;
`terminated` never bootstraps, while truncated bootstrapping is explicit in the
saved configuration.

Use a fixed environment interaction budget, not an episode count. The budget
is a **cumulative target**: on a fresh agent `--interaction-budget 64` performs
64 interactions and leaves `training_steps == 64`.

```bash
.venv/bin/python -m uav_mec_gym.experiment train-ppo \
  --port 12347 --uavs 2 \
  --environment-config \
    src/test/resources/config/simulation_settings.xml \
    src/test/resources/config/edge_devices.xml \
    src/test/resources/config/applications.xml \
  --interaction-budget 64 \
  --training-seed-start 101 \
  --validation-seeds 201 202 203 204 205 \
  --heldout-seeds 301 302 303 304 305 \
  --rollout-steps 32 --minibatch-size 16 --update-epochs 4 \
  --checkpoint results/phase4/checkpoints/ppo_smoke.pt \
  --output results/phase4/ppo_training_smoke.json
```

The checkpoint includes actor, critic, optimizer, pending rollout, step/update
counters, configuration, observation-normalization statistics, Python, NumPy,
and PyTorch RNG states, and the next environment-training-seed cursor. The
training JSON keeps raw per-transition and per-episode evidence, the cumulative
interaction target, the interactions performed by this invocation, config
hashes, seed partitions, checkpoint hash, and Git commit SHA.

To continue training, load the prior checkpoint and set a larger cumulative
target. Keep `--training-seed-start` equal to the original run; the command
rejects a different value instead of reusing training seeds. The checkpoint's
PPO configuration is authoritative on resume, and its action-sampling RNG is
continued without reseeding:

```bash
.venv/bin/python -m uav_mec_gym.experiment train-ppo \
  --port 12347 --uavs 2 \
  --environment-config \
    src/test/resources/config/simulation_settings.xml \
    src/test/resources/config/edge_devices.xml \
    src/test/resources/config/applications.xml \
  --resume-from results/phase4/checkpoints/ppo_smoke.pt \
  --interaction-budget 128 \
  --training-seed-start 101 \
  --validation-seeds 201 202 203 204 205 \
  --heldout-seeds 301 302 303 304 305 \
  --checkpoint results/phase4/checkpoints/ppo_smoke_128.pt \
  --output results/phase4/ppo_training_resume_128.json
```

For an exact interrupted-versus-uninterrupted replay, checkpoint at a completed
episode and rollout-update boundary. A forced budget cutoff is recorded as a
truncation and the short rollout is flushed, so choosing a cutoff inside either
boundary intentionally defines a different (but fully recorded) training run.

## Mixed-action TD3

The TD3 candidate is an explicitly parameterized-action extension: a masked
target-logit head is executed by legal argmax, a `tanh` head controls movement,
twin critics consume target one-hot plus movement, and target-policy smoothing
is limited to movement. Replay, target networks, optimizers, normalizer state,
all RNG streams, counters, and the training-seed cursor are checkpointed.

```bash
.venv/bin/python -m uav_mec_gym.experiment train-td3 \
  --port 12347 --uavs 2 \
  --environment-config \
    src/test/resources/config/simulation_settings.xml \
    src/test/resources/config/edge_devices.xml \
    src/test/resources/config/applications.xml \
  --interaction-budget 4096 \
  --training-seed-start 10001 \
  --validation-seeds 201 202 203 204 205 206 207 208 209 210 \
  --heldout-seeds 301 302 303 304 305 306 307 308 309 310 \
  --learning-starts 512 --batch-size 128 --replay-capacity 10000 \
  --checkpoint results/formal/td3.pt \
  --output results/formal/td3_training.json
```

Evaluation uses the same `evaluate` command with `--algorithm td3`. See
[`docs/experiments/formal-protocol.md`](docs/experiments/formal-protocol.md) for
the exact mixed-action definition and statistical protocol.

## Paired validation and held-out evaluation

Validation seeds (`201`–`205`) and held-out test seeds (`301`–`305`) are
separate from training seeds. Evaluation is deterministic: PPO has no sampling
noise, epsilon, or dropout. The command evaluates the two baselines and the
same PPO checkpoint on an identical seed set, then reports mean, sample
standard deviation, normal 95% CI, and per-seed PPO-minus-baseline differences.

With the development GymBridge still listening on port 12347:

```bash
.venv/bin/python -m uav_mec_gym.experiment evaluate \
  --port 12347 --uavs 2 --split validation \
  --seeds 201 202 203 204 205 \
  --environment-config \
    src/test/resources/config/simulation_settings.xml \
    src/test/resources/config/edge_devices.xml \
    src/test/resources/config/applications.xml \
  --checkpoint results/phase4/checkpoints/ppo_smoke.pt \
  --output results/phase4/ppo_validation.json
```

After stopping that server and starting the held-out server on port 12348:

```bash
.venv/bin/python -m uav_mec_gym.experiment evaluate \
  --port 12348 --uavs 2 --split heldout \
  --seeds 301 302 303 304 305 \
  --environment-config \
    src/test/resources/config/heldout/simulation_settings.xml \
    src/test/resources/config/heldout/edge_devices.xml \
    src/test/resources/config/heldout/applications.xml \
  --checkpoint results/phase4/checkpoints/ppo_smoke.pt \
  --output results/phase4/ppo_heldout.json
```

The XML files passed to `--environment-config` must be the same files used to
launch that server. Their content hash is recorded in every result row.

## Result interpretation

All generated phase-4 artifacts are ignored under `results/phase4/`. Before any
plot or algorithm claim, audit:

- paired seed sets and fixed interaction budgets;
- `successful_tasks <= settled_tasks <= total_tasks`;
- termination/truncation rates and non-zero terminal simulation time;
- latency in seconds, UE/UAV energy in joules, and constraint counts;
- reward-component totals against the raw physical metrics;
- Git commit, environment/config hashes, and checkpoint hash.

Smoke runs prove the pipeline and reproducibility contract only. They are not
paper evidence of algorithm superiority.

## Formal 10-seed evidence and release artifacts

The tracked formal config declares 10 independent training seeds per learned
algorithm, a fixed 4,096-interaction budget per seed, and 10 paired seeds for
each validation split. The runner is resumable and refuses a dirty tracked tree.

```bash
.venv/bin/python scripts/run_formal_experiments.py
.venv/bin/python scripts/summarize_formal_experiments.py
.venv/bin/python scripts/plot_formal_results.py
```

The summary script audits all raw transitions, checkpoint/config/Git hashes,
paired seeds, and deterministic baseline repetition before producing tables.
The plotting script emits PDF, SVG, and 300-DPI PNG figures with bootstrap
intervals across independent training-seed means.

Generated evidence stays ignored under `results/`; Maven output stays ignored
under `target/`. Use `scripts/package_experiment_artifacts.py` for deterministic
checksummed release bundles, following
[`docs/experiments/artifact-publication.md`](docs/experiments/artifact-publication.md).
