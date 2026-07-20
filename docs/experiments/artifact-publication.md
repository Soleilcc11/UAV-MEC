# Experiment artifact publication plan

## Release boundary

Publish code and experiment data as one immutable release identified by all of:

1. a signed or protected Git tag, suggested name
   `uav-mec-formal-protocol-1.1-v1`;
2. the full 40-character Git commit recorded by every report;
3. the experiment config SHA-256;
4. SHA-256 sidecars for every release bundle;
5. a Zenodo version DOI, with its concept DOI used by the paper.

Do not put checkpoints, raw training transitions, generated figures, or Maven
`target/` output back into Git history. Git contains source, tests, config, and
the packaging scripts. GitHub Releases and the archival repository contain the
generated evidence.

## Bundle tiers

The packaging command creates deterministic `tar.gz` archives with normalized
tar metadata, an `ARTIFACT_INVENTORY.json`, and `MANIFEST.sha256`.

```bash
# Raw JSON reports, audited summaries, figures, and the minimal source snapshot.
.venv/bin/python scripts/package_experiment_artifacts.py --profile evidence

# Checkpoints plus summaries/config/source for exact policy replay.
.venv/bin/python scripts/package_experiment_artifacts.py --profile checkpoints

# One self-contained archive when repository limits permit it.
.venv/bin/python scripts/package_experiment_artifacts.py --profile full
```

Recommended publication assets:

| Asset | Purpose | Required |
|---|---|---|
| `*-evidence.tar.gz` | raw reports, audited aggregate, CSV/Markdown tables, PDF/SVG/PNG figures, source snapshot | yes |
| `*-checkpoints.tar.gz` | all 20 final checkpoints | yes for exact evaluation replay |
| `*.sha256` | transport integrity | yes |
| tagged source archive | human-readable source release | yes |

The raw simulator console logs are operational diagnostics and may contain
machine paths; they are intentionally excluded from publication bundles.

## Publication sequence

1. Confirm the formal orchestration and aggregate audit both say `passed`.
2. Render the PNG figures and visually inspect labels, units, confidence
   intervals, and clipping; keep PDF/SVG as the paper source.
3. Build the evidence and checkpoint bundles twice and confirm identical
   SHA-256 digests.
4. Add the immutable Git tag at the recorded formal commit.
5. Create a GitHub Release for that tag and upload bundles plus sidecars.
6. Archive the same assets in Zenodo (or an institutional repository), record
   the version DOI in the GitHub Release, and use the concept DOI in the paper.
7. From a clean directory, verify `MANIFEST.sha256`, rebuild Java, recreate the
   Python environment, and rerun one held-out checkpoint before announcing the
   release.

## Metadata and policy blockers

The repository currently has no top-level `LICENSE` or authoritative author
list. Public archival release must pause until the owner chooses a source/data
license and supplies citation metadata; the automation must not invent either.
Recommended metadata fields are title, authors/ORCIDs, affiliations, license,
keywords, related GitHub URL, related paper DOI (when available), funding, and
the protocol/config/Git hashes.

If third-party redistribution terms prevent bundling a vendored dependency,
publish its checksum and retrieval instructions instead and update the clean
room verification before release.

## Versioning

- A rerun with identical code/config but additional repetitions is a new
  artifact version under the same concept DOI.
- Any change to simulator semantics, reward, action/observation contract,
  algorithm implementation, seed partition, or interaction budget requires a
  new Git tag and experiment ID.
- Regenerating a plot from the same audited summary does not change the result
  dataset, but the figure bundle hash and release version must still be updated.
