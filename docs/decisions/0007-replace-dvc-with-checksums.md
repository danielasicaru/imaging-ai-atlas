# ADR 0007: Replace DVC with committed SHA-256 checksums for raw snapshots

**Status:** accepted
**Date:** 2026-11-17

## Context

DVC was used for one thing: two pointer files recording the hash of each raw snapshot
(`arxiv-metadata-oai-snapshot.json`, `pubmed_cervical_spine_v2.xml`). No `dvc.yaml`
pipeline existed and no code imported DVC.

The scheduled dependency scan flagged `diskcache` 5.6.3 (GHSA-w8v5-vhqr-4h9v / PYSEC-2026-2447,
moderate): it deserializes cache entries with pickle, so anyone able to write to its cache
directory can execute code as the user running DVC. No fixed release exists. It came in only
through DVC (`dvc-data`), and on Windows DVC places part of that cache in a site-wide
directory under `C:\ProgramData` that every local account can write to.

The project's standard is that nothing it installs should let anything outside the
owner's account act on the machine. A transitive dependency with an unpatched
code-execution path in a shared directory doesn't meet it.

## Decision

Remove DVC. Each raw snapshot is pinned instead by its SHA-256 in `data/raw/SHA256SUMS`,
in standard `sha256sum` format so any `sha256sum -c` can verify it (command in the README).
The checksums were computed from the same files whose MD5 matched the former DVC pointers.

## Alternatives considered

| Option | Pros | Cons |
|---|---|---|
| Keep DVC, ignore the advisory in the scanner | No change to the workflow | Leaves an unpatched code-execution path in a shared directory; depends on a fix that may never ship |
| Keep DVC, move its site cache to a user-only folder | Closes the shared-directory path | A per-machine setting a fresh clone wouldn't have; the vulnerable dependency stays installed |
| Committed SHA-256 checksums (chosen) | No dependency at all; verification uses a standard tool | No built-in fetch/restore of the snapshots |

## Consequences

- Drops DVC and its whole dependency tree from the environment.
- Verifying that a snapshot is the exact one the results came from is one standard command.
- Restoring a snapshot is manual: re-download it per ADR 0001 / ADR 0006 and check it against
  `SHA256SUMS`. A re-fetched PubMed result can differ from the original if records changed
  upstream, so keeping a private backup copy of the snapshots is the owner's responsibility.
- Revisit if the project ever needs a real multi-stage pipeline with cached intermediate
  outputs; at that point compare tools on their dependency footprint as well as features.
