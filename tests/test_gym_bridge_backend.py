from __future__ import annotations

import json
import socketserver
import threading

import numpy as np

from uav_mec_gym import JavaGymBridgeBackend


def _observation():
    return {
        "time": [0.25],
        "task": [0.25] * 7,
        "resources": [[0.5] * 3] * 3,
        "uavs": [[0.5] * 8] * 2,
        "action_mask": [0, 1, 1, 1, 1],
    }


class _Handler(socketserver.StreamRequestHandler):
    def handle(self):
        while line := self.rfile.readline():
            request = json.loads(line)
            response = {
                "id": request["id"],
                "type": request["type"] + "_response",
                "ok": True,
            }
            if request["type"] == "hello":
                response["spec"] = {
                    "protocol_version": "1.1",
                    "number_of_uavs": 2,
                    "provenance": {
                        "environment": {
                            "files": [{"name": "config.xml", "sha256": "c" * 64}]
                        },
                        "runtime": {
                            "artifact_sha256": "a" * 64,
                            "classes_current": True,
                            "git_commit_sha": "abc123",
                            "source_tree_sha256": "s" * 64,
                        },
                    },
                }
            elif request["type"] == "reset":
                response.update(observation=_observation(), info={"seed": request["seed"]})
            elif request["type"] == "step":
                response.update(
                    observation=_observation(),
                    reward_components={
                        "success": 1.0,
                        "latency_ratio": 0.2,
                        "ue_energy_ratio": 0.1,
                        "uav_energy_ratio": 0.05,
                        "constraint_violations": 0.0,
                    },
                    reward=0.9,
                    terminated=True,
                    truncated=False,
                    metrics={"settled_tasks": 1},
                )
            self.wfile.write(json.dumps(response).encode() + b"\n")


def test_java_backend_performs_versioned_round_trip():
    server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    backend = JavaGymBridgeBackend(
        port=server.server_address[1],
        expected_number_of_uavs=2,
        expected_environment_manifest={
            "files": [{"name": "config.xml", "sha256": "c" * 64}]
        },
        expected_git_commit_sha="abc123",
        expected_source_tree_sha256="s" * 64,
    )
    try:
        observation, info = backend.reset(42)
        result = backend.step(3, np.zeros((2, 3), dtype=np.float32))
        assert info == {"seed": 42}
        assert observation["action_mask"] == [0, 1, 1, 1, 1]
        assert result.reward_components.success == 1.0
        assert result.reward == 0.9
        assert backend.provenance["runtime"]["artifact_sha256"] == "a" * 64
        assert result.terminated is True
    finally:
        backend.close()
        server.shutdown()
        server.server_close()
