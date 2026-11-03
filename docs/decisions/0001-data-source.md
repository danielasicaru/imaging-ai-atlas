# ADR 0001: Data source and licensing

**Status:** accepted
**Date:** 2026-11-03

## Context

The project needs a corpus of medical-imaging AI papers to build the topic-shift story on. arXiv is the
candidate source: it has a long history, a documented API/metadata license, and existing bulk snapshots
(e.g. the Kaggle arXiv metadata dataset) that avoid re-scraping arXiv directly.

Two things needed to be pinned down before `ingest` was built:
1. What exactly is CC0 — the metadata, or the abstracts/full text too?
2. What can this repo ship: code only, code + derived aggregates, or code + raw abstracts?

## Decision

- Use the **arXiv metadata** (via the Kaggle snapshot, re-exported from the arXiv API) as the data source.
  A `Source` protocol keeps a future second source (e.g. PubMed) pluggable without reworking the pipeline.
- Ship **code + derived aggregates + arXiv IDs** in the repo. Raw abstracts and the full snapshot are
  **not** committed — they are pulled via the documented reproduction path and tracked with DVC locally.
- No claims of arXiv endorsement anywhere in the repo or report.

## Verification

arXiv's API Terms of Use (https://info.arxiv.org/help/api/tou.html) state verbatim:

> "You are free to use descriptive metadata about arXiv e-prints under the terms of the Creative Commons
> Universal (CC0 1.0) Public Domain Declaration."

with a footnote explicitly defining the scope:

> "Descriptive metadata includes information for discovery and identification purposes, and includes
> fields such as title, abstract, authors, identifiers, and classification terms."

So the abstract text is explicitly CC0, confirmed by arXiv's own wording, not inferred. Full-text articles
(PDFs/source) are licensed separately per paper and are out of scope for this project regardless of CC0
status — this project never fetches full text.

## Alternatives considered

| Option | Pros | Cons |
|---|---|---|
| Ship raw abstracts in the repo | Self-contained, no external download step for a reader | Bloats repo size for no story value, even though CC0 covers it |
| Scrape arXiv directly instead of the Kaggle snapshot | No third-party intermediary | Slower, rate-limited, duplicates work the snapshot already did |
| PubMed first, arXiv later | Larger structured-metadata ecosystem (E-utilities) | PubMed abstracts carry copyright risk requiring a check before shipping any text; arXiv preprints are the more natural "AI research trend" corpus |

## Consequences

- `ingest` needed a `Source` interface from day one so a PubMed adapter can slot in later without
  reworking the pipeline — built that way from the start.
- CC0 scope is now verified; no further licensing check is needed for metadata-only fields.
