# Vendored Maven artifacts

This repository contains only the two upstream binaries required by the legacy
EdgeCloudSim samples that are not reliably resolvable from Maven Central. They
are exposed as normal Maven dependencies so the project does not use
`systemPath`.

| Coordinate | SHA-256 |
|---|---|
| `org.cloudbus.cloudsim:cloudsim-vendored:4.0` | `c1803a3567ff7b6607b0a077f835e90ad256fb06f91a379d7ffb0d6a13231d80` |
| `net.sourceforge.jfuzzylogic:jfuzzylogic:1.0` | `6c7007464a8609d49188cb4322b7cd59612488d3a28330e245eeafca46e8904e` |

The JAR bytes are unchanged from the previously checked-in
`src/main/resources/lib` copies. Maven `.sha1` sidecars protect repository
resolution; release bundles additionally publish SHA-256 checksums.
