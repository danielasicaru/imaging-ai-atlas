# ADR 0001: Data source and licensing

**Status:** proposed
**Date:** 2026-11-03

## Context

The project needs a corpus of medical-imaging AI papers to build the topic-shift story on. arXiv is the
candidate source: it has a long history, a documented API/metadata license, and existing bulk snapshots
(e.g. the Kaggle arXiv metadata dataset) that avoid re-scraping arXiv directly.

Two things need to be pinned down before `ingest` is built in M1:
1. What exactly is CC0 — the metadata, or the abstracts/full text too?
2. What can this repo ship: code only, code + derived aggregates, or code + raw abstracts?

## Decision

- Use the **arXiv metadata** (via the Kaggle snapshot, re-exported from the arXiv API) as the M0–M6 data
  source. PubMed is deferred to M7 as a separate adapter behind the same `Source` interface.
- Ship **code + derived aggregates + arXiv IDs** in the repo. Raw abstracts and the full snapshot are
  **not** committed — they are pulled via the documented reproduction path and tracked with DVC locally.
- No claims of arXiv endorsement anywhere in the repo or report.

## Alternatives considered

| Option | Pros | Cons |
|---|---|---|
| Ship raw abstracts in the repo | Self-contained, no external download step for a reader | CC0 metadata license doesn't automatically clear full abstract text; bloats repo size for no story value |
| Scrape arXiv directly instead of the Kaggle snapshot | No third-party intermediary | Slower, rate-limited, duplicates work the snapshot already did; snapshot is the dataset ArXiv Atlas-style projects already use |
| PubMed first, arXiv later | Larger structured-metadata ecosystem (E-utilities) | PubMed abstracts carry copyright risk requiring a check before shipping any text; arXiv preprints are the more natural "AI research trend" corpus |

## Consequences

- `ingest` (M1) needs a `Source` interface from day one so the PubMed adapter (M7) can slot in without
  reworking the pipeline.
- The exact CC0 scope (metadata vs. abstract text) still needs verification against the current arXiv API
  terms before `ingest` ships — tracked as a TODO for M1, not resolved here.
- Status: **proposed**, to be moved to **accepted** once the CC0 scope is verified in M1.
