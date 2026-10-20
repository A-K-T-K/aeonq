# Evidence archives

Each `source-<SHA256>.zip` contains the exact experimental source, tests,
dependency specifications and manifest captured at the start of a process.
`SOURCE_MANIFEST.json` maps paths to file hashes; the SHA-256 of its canonical
JSON bytes is the identifier recorded in experiment metadata. Verification
checks the manifest digest and every archived file. The code snapshot hash
does not include raw data.

Timing and statistical studies can reference different source snapshots.
Each statistical population is required to have one consistent snapshot.
Historical archives are retained when results are regenerated. In the v0.0.5
campaign, some statistical runs began before an import-path change to the
validator. Those records were validated with the updated validator without
changing their sampling code, seeds, or archived source identifiers; that
release's regeneration log retains the failure and recovery details. Each later
campaign records its own source identifiers and regeneration log.
