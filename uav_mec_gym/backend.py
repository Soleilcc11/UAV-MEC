from __future__ import annotations

import json
import socket
from itertools import count
from typing import Any

import numpy as np

from .contract import BackendStep, Observation, RewardComponents


class GymBridgeError(RuntimeError):
    """Raised when the Java GymBridge rejects a request or violates its protocol."""


class JavaGymBridgeBackend:
    """Newline-delimited JSON client for the EdgeCloudSim GymBridge server."""

    PROTOCOL_VERSION = "1.1"

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 12346,
        *,
        timeout: float = 35.0,
        expected_number_of_uavs: int | None = None,
        expected_environment_manifest: dict[str, Any] | None = None,
        expected_git_commit_sha: str | None = None,
        expected_source_tree_sha256: str | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self._expected_number_of_uavs = expected_number_of_uavs
        self._expected_environment_manifest = expected_environment_manifest
        self._expected_git_commit_sha = expected_git_commit_sha
        self._expected_source_tree_sha256 = expected_source_tree_sha256
        self._socket: socket.socket | None = None
        self._reader: Any = None
        self._writer: Any = None
        self._ids = count(1)
        self.specification: dict[str, Any] | None = None
        self.provenance: dict[str, Any] | None = None

    @property
    def number_of_uavs(self) -> int:
        self._connect()
        assert self.specification is not None
        return int(self.specification["number_of_uavs"])

    def reset(self, seed: int) -> tuple[Observation, dict[str, Any]]:
        if seed is None:
            raise ValueError(
                "JavaGymBridgeBackend requires an explicit episode seed"
            )
        response = self._request("reset", seed=int(seed))
        return response["observation"], dict(response["info"])

    def step(self, target: int, movement: np.ndarray) -> BackendStep:
        response = self._request(
            "step",
            action={
                "target": int(target),
                "movement": np.asarray(movement, dtype=np.float32).tolist(),
            },
        )
        components = response["reward_components"]
        reward = float(response["reward"])
        if not np.isfinite(reward):
            raise GymBridgeError("GymBridge returned a non-finite reward")
        return BackendStep(
            observation=response["observation"],
            reward_components=RewardComponents(
                success=float(components["success"]),
                latency_ratio=float(components["latency_ratio"]),
                ue_energy_ratio=float(components["ue_energy_ratio"]),
                uav_energy_ratio=float(components["uav_energy_ratio"]),
                constraint_violations=float(components["constraint_violations"]),
            ),
            reward=reward,
            terminated=bool(response["terminated"]),
            truncated=bool(response["truncated"]),
            metrics=dict(response["metrics"]),
        )

    def close(self) -> None:
        if self._socket is None:
            return
        try:
            self._request("close")
        except (OSError, GymBridgeError):
            pass
        finally:
            self._disconnect()

    def _connect(self) -> None:
        if self._socket is not None:
            return
        sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        sock.settimeout(self.timeout)
        self._socket = sock
        self._reader = sock.makefile("r", encoding="utf-8", newline="\n")
        self._writer = sock.makefile("w", encoding="utf-8", newline="\n")
        try:
            response = self._request("hello", connect=False)
            specification = dict(response["spec"])
            if specification.get("protocol_version") != self.PROTOCOL_VERSION:
                raise GymBridgeError("GymBridge hello returned an incompatible protocol")
            uav_count = int(specification["number_of_uavs"])
            if self._expected_number_of_uavs is not None and (
                uav_count != self._expected_number_of_uavs
            ):
                raise GymBridgeError(
                    "GymBridge UAV count does not match the Gymnasium environment"
                )
            self._validate_provenance(specification)
            self.specification = specification
        except Exception:
            self._disconnect()
            raise

    def _validate_provenance(self, specification: dict[str, Any]) -> None:
        raw = specification.get("provenance")
        provenance_required = any(
            value is not None
            for value in (
                self._expected_environment_manifest,
                self._expected_git_commit_sha,
                self._expected_source_tree_sha256,
            )
        )
        if not isinstance(raw, dict):
            if provenance_required:
                raise GymBridgeError("GymBridge hello omitted service provenance")
            return
        environment = raw.get("environment")
        runtime = raw.get("runtime")
        if not isinstance(environment, dict) or not isinstance(runtime, dict):
            raise GymBridgeError("GymBridge returned malformed service provenance")
        artifact_hash = str(runtime.get("artifact_sha256", ""))
        if len(artifact_hash) != 64:
            raise GymBridgeError("GymBridge runtime artifact hash is invalid")
        if runtime.get("classes_current") is not True:
            raise GymBridgeError("GymBridge runtime classes are stale relative to source")

        if self._expected_environment_manifest is not None:
            expected_files = {
                str(entry["name"]): str(entry["sha256"])
                for entry in self._expected_environment_manifest["files"]
            }
            actual_files = {
                str(entry["name"]): str(entry["sha256"])
                for entry in environment.get("files", [])
            }
            if actual_files != expected_files:
                raise GymBridgeError(
                    "GymBridge server environment config hashes do not match the client"
                )
        if self._expected_git_commit_sha is not None and (
            runtime.get("git_commit_sha") != self._expected_git_commit_sha
        ):
            raise GymBridgeError("GymBridge server Git commit does not match the client")
        if self._expected_source_tree_sha256 is not None and (
            runtime.get("source_tree_sha256")
            != self._expected_source_tree_sha256
        ):
            raise GymBridgeError("GymBridge server source tree does not match the client")
        self.provenance = dict(raw)

    def _request(self, request_type: str, *, connect: bool = True, **payload: Any) -> dict[str, Any]:
        if connect:
            self._connect()
        if self._writer is None or self._reader is None:
            raise GymBridgeError("GymBridge connection is not open")
        request_id = str(next(self._ids))
        request = {
            "id": request_id,
            "type": request_type,
            "protocol_version": self.PROTOCOL_VERSION,
            **payload,
        }
        self._writer.write(json.dumps(request, separators=(",", ":")) + "\n")
        self._writer.flush()
        line = self._reader.readline()
        if not line:
            raise GymBridgeError("GymBridge closed the connection without a response")
        response = json.loads(line)
        if response.get("id") != request_id:
            raise GymBridgeError("GymBridge response id does not match the request")
        if not response.get("ok"):
            raise GymBridgeError(str(response.get("error", "unknown GymBridge error")))
        return response

    def _disconnect(self) -> None:
        for stream in (self._reader, self._writer):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass
        if self._socket is not None:
            try:
                self._socket.close()
            except OSError:
                pass
        self._reader = None
        self._writer = None
        self._socket = None
