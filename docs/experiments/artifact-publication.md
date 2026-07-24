# Experiment artifact publication plan

## Release boundary

Release source and evidence as one immutable protocol 1.2 version identified by:

1. a protected or signed Git tag, suggested
   `uav-mec-formal-protocol-1.2-v1`;
2. the full Git commit recorded by all reports;
3. the formal config and environment SHA-256 values;
4. every final checkpoint SHA-256;
5. bundle SHA-256 sidecars;
6. a version DOI, with the concept DOI cited by the paper.

Generated checkpoints, raw transitions, figures, and Maven output remain outside
Git. The source repository contains code, tests, protocol/config files, and
packaging automation.

## Bundle profiles

```bash
# Raw JSON reports, audited summary, tables, figures, and source snapshot.
.venv/bin/python scripts/package_experiment_artifacts.py --profile evidence

# Checkpoints plus summaries/config/source for exact replay.
.venv/bin/python scripts/package_experiment_artifacts.py --profile checkpoints

# Combined archive when repository limits permit.
.venv/bin/python scripts/package_experiment_artifacts.py --profile full
```

Archives have normalized tar metadata, `ARTIFACT_INVENTORY.json`,
`MANIFEST.sha256`, and an external archive checksum. The source snapshot includes
both READMEs, recursive environment configs, Java/Python source, tests referenced
by the protocol, and experiment documentation.

## Publication sequence

1. Confirm the pilot audit, formal orchestration, and aggregate summary all say
   `passed`.
2. Confirm the formal result commit is checked out and the tree is clean.
3. Render and visually inspect PNG/PDF/SVG figures for units, clipping, labels,
   and confidence intervals.
4. Build each archive twice and confirm identical SHA-256 values.
5. Tag the exact formal commit and create a source release.
6. Upload evidence, checkpoints, sidecars, and protocol metadata to the release
   and archival repository.
7. In a clean checkout, verify `MANIFEST.sha256`, rebuild Java/Python, and replay
   one held-out checkpoint.

## Required assets

| Asset | Purpose |
|---|---|
| `*-evidence.tar.gz` | raw reports, aggregate, tables, figures, source snapshot |
| `*-checkpoints.tar.gz` | all final learned-policy checkpoints |
| `*.sha256` | transport integrity |
| tagged source archive | immutable human-readable source |
| protocol/config metadata | scale, seeds, hashes, metric definitions |

Console logs may contain machine-local paths and are operational diagnostics;
they are excluded unless separately scrubbed and documented.

## Publication blockers

Public release must pause until the owner supplies an authoritative license,
author list, affiliations, and citation metadata. Automation must not invent
these. Vendored third-party material must be checked for redistribution rights;
when redistribution is not permitted, publish its checksum and retrieval
instructions instead.

Any change to simulator semantics, reward, contract, algorithm implementation,
seed partition, interaction budget, or environment requires a new experiment ID
and Git tag. A rerun with identical code/config but more repetitions is a new
artifact version under the same concept DOI.
