from __future__ import annotations

import socket
import subprocess
import time
from pathlib import Path

import numpy as np

from uav_mec_gym import JavaGymBridgeBackend, UAVMECGymEnv
from uav_mec_gym.experiment import (
    current_commit_sha,
    environment_manifest,
    repository_source_sha256,
)


def _free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_until_listening(port: int, process: subprocess.Popen[bytes]) -> None:
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("GymBridge Java process exited before accepting connections")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return
        except OSError:
            time.sleep(0.05)
    raise TimeoutError("GymBridge did not start within 15 seconds")


def test_real_java_python_bridge_performs_verified_transition() -> None:
    repository = Path(__file__).resolve().parents[1]
    subprocess.run(
        ["mvn", "package", "-q", "-DskipTests"],
        cwd=repository,
        check=True,
        timeout=60,
    )
    config_paths = [
        repository / "src/test/resources/config/simulation_settings.xml",
        repository / "src/test/resources/config/edge_devices.xml",
        repository / "src/test/resources/config/applications.xml",
    ]
    environment = environment_manifest(config_paths)
    port = _free_loopback_port()
    process = subprocess.Popen(
        [
            "java",
            "-cp",
            "target/classes:target/lib/*:src/main/resources/lib/*",
            "edu.boun.edgecloudsim.uav.GymBridgeMain",
            str(port),
            *(str(path) for path in config_paths),
            "2",
        ],
        cwd=repository,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    env: UAVMECGymEnv | None = None
    try:
        _wait_until_listening(port, process)
        backend = JavaGymBridgeBackend(
            port=port,
            expected_number_of_uavs=2,
            expected_environment_manifest=environment,
            expected_git_commit_sha=current_commit_sha(repository),
            expected_source_tree_sha256=repository_source_sha256(repository),
        )
        env = UAVMECGymEnv(2, backend)
        observation, reset_info = env.reset(seed=2026)
        assert reset_info["seed"] == 2026
        assert env.observation_space.contains(observation)
        assert observation["action_mask"][2] == 1

        next_observation, reward, terminated, truncated, info = env.step(
            {
                "target": 2,
                "movement": np.zeros((2, 3), dtype=np.float32),
            }
        )

        assert env.observation_space.contains(next_observation)
        assert np.isfinite(reward)
        assert not (terminated and truncated)
        assert info["in_flight_tasks"] >= 0
        assert info["settled_in_transition"] >= 0
        assert backend.provenance is not None
        assert backend.provenance["runtime"]["classes_current"] is True
    finally:
        if env is not None:
            env.close()
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
