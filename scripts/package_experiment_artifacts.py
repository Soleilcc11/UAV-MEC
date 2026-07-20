#!/usr/bin/env python3
"""Build deterministic, checksummed release bundles for formal experiments."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path
from typing import Any, Iterable, Mapping


REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPOSITORY / "results/formal/protocol_1_1_10seed_v2"
SOURCE_PATTERNS = (
    "README.md",
    "pom.xml",
    "pyproject.toml",
    "requirements-dev.txt",
    "experiments/*.json",
    "scripts/*.py",
    "uav_mec_gym/*.py",
    "docs/adr/*.md",
    "docs/experiments/*.md",
    "src/main/java/**/*.java",
    "src/main/resources/config/*",
    "src/main/resources/maven-repository/**/*",
    "src/test/resources/config/**/*",
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _source_files() -> set[Path]:
    files: set[Path] = set()
    for pattern in SOURCE_PATTERNS:
        files.update(path for path in REPOSITORY.glob(pattern) if path.is_file())
    return files


def _artifact_files(experiment_root: Path, profile: str) -> set[Path]:
    files = _source_files()
    files.add(experiment_root / "orchestration.json")
    files.update(path for path in (experiment_root / "summary").rglob("*") if path.is_file())
    if profile in {"evidence", "full"}:
        files.update(
            path
            for path in (experiment_root / "runs").rglob("*.json")
            if path.is_file()
        )
    if profile in {"checkpoints", "full"}:
        files.update(
            path
            for path in (experiment_root / "runs").rglob("*.pt")
            if path.is_file()
        )
    return files


def _archive_name(path: Path, experiment_root: Path) -> str:
    if path.is_relative_to(experiment_root):
        return f"experiment/{path.relative_to(experiment_root).as_posix()}"
    return f"source/{path.relative_to(REPOSITORY).as_posix()}"


def _validate_experiment(experiment_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    orchestration = _read_json(experiment_root / "orchestration.json")
    if orchestration.get("status") != "complete":
        raise ValueError("Formal orchestration is not complete")
    if orchestration.get("service_lifecycle") != (
        "fresh_jvm_per_training_or_evaluation_report"
    ):
        raise ValueError("Formal orchestration did not isolate reports in fresh JVMs")
    summary = _read_json(experiment_root / "summary/formal_summary.json")
    if summary.get("audit", {}).get("status") != "passed":
        raise ValueError("Formal summary audit did not pass")
    figure_manifest = _read_json(experiment_root / "summary/figures/figure_manifest.json")
    source_hash = figure_manifest.get("source_summary_sha256")
    if source_hash != _sha256_file(experiment_root / "summary/formal_summary.json"):
        raise ValueError("Figure manifest does not match formal summary")
    if orchestration.get("git_commit_sha") != summary.get("git_commit_sha"):
        raise ValueError("Orchestration and summary Git commits differ")
    if _git_head() != summary.get("git_commit_sha"):
        raise ValueError("Current checkout does not match the formal result commit")
    return orchestration, summary


def _tar_info(name: str, size: int, mode: int = 0o644) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = size
    info.mode = mode
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    return info


def package(
    experiment_root: Path,
    destination: Path,
    *,
    profile: str,
) -> dict[str, Any]:
    orchestration, summary = _validate_experiment(experiment_root)
    files = _artifact_files(experiment_root, profile)
    missing = [path for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing[0])
    entries = []
    for path in sorted(files, key=lambda item: _archive_name(item, experiment_root)):
        entries.append(
            {
                "path": _archive_name(path, experiment_root),
                "sha256": _sha256_file(path),
                "bytes": path.stat().st_size,
                "source_path": str(path),
            }
        )
    inventory = {
        "format_version": 1,
        "created_at_utc": summary["generated_at_utc"],
        "profile": profile,
        "experiment_id": summary["experiment_id"],
        "git_commit_sha": summary["git_commit_sha"],
        "config_sha256": orchestration["config_sha256"],
        "formal_summary_sha256": _sha256_file(
            experiment_root / "summary/formal_summary.json"
        ),
        "files": [
            {key: entry[key] for key in ("path", "sha256", "bytes")}
            for entry in entries
        ],
    }
    inventory_bytes = (
        json.dumps(inventory, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    manifest_lines = [
        f"{entry['sha256']}  {entry['path']}" for entry in inventory["files"]
    ]
    manifest_lines.append(
        f"{_sha256_bytes(inventory_bytes)}  ARTIFACT_INVENTORY.json"
    )
    manifest_bytes = ("\n".join(manifest_lines) + "\n").encode("utf-8")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                for entry in entries:
                    path = Path(entry["source_path"])
                    data = path.read_bytes()
                    archive.addfile(_tar_info(entry["path"], len(data)), io.BytesIO(data))
                archive.addfile(
                    _tar_info("ARTIFACT_INVENTORY.json", len(inventory_bytes)),
                    io.BytesIO(inventory_bytes),
                )
                archive.addfile(
                    _tar_info("MANIFEST.sha256", len(manifest_bytes)),
                    io.BytesIO(manifest_bytes),
                )
    bundle_hash = _sha256_file(destination)
    destination.with_suffix(destination.suffix + ".sha256").write_text(
        f"{bundle_hash}  {destination.name}\n", encoding="utf-8"
    )
    _verify_archive(destination)
    return {
        "path": str(destination),
        "sha256": bundle_hash,
        "bytes": destination.stat().st_size,
        "profile": profile,
        "file_count": len(entries),
    }


def _verify_archive(path: Path) -> None:
    with tarfile.open(path, mode="r:gz") as archive:
        members = {member.name: member for member in archive.getmembers()}
        inventory_handle = archive.extractfile(members["ARTIFACT_INVENTORY.json"])
        if inventory_handle is None:
            raise ValueError("Archive inventory is unreadable")
        inventory = json.loads(inventory_handle.read())
        for entry in inventory["files"]:
            member = members.get(entry["path"])
            if member is None or not member.isfile():
                raise ValueError(f"Archive is missing {entry['path']}")
            handle = archive.extractfile(member)
            if handle is None or _sha256_bytes(handle.read()) != entry["sha256"]:
                raise ValueError(f"Archive checksum mismatch: {entry['path']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--profile", choices=["evidence", "checkpoints", "full"], default="evidence"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    experiment_root = args.experiment_root.resolve()
    summary = _read_json(experiment_root / "summary/formal_summary.json")
    destination = (
        args.output.resolve()
        if args.output
        else REPOSITORY
        / "artifacts"
        / (
            f"uav-mec-{summary['experiment_id']}-"
            f"{summary['git_commit_sha'][:12]}-{args.profile}.tar.gz"
        )
    )
    result = package(experiment_root, destination, profile=args.profile)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
