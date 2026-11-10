# ADR 0006: Second data source (PubMed) and a second filter domain (cervical spine fracture)

**Status:** accepted
**Date:** 2026-11-10

## Context

ADR 0005 made the filter rule config-driven specifically so a second, narrower research
topic could be added as a YAML file with no code change, and ADR 0001 built `Source` as a
protocol specifically so PubMed could be added as a second adapter without reworking
`ingest`. Both deferrals were cashed in together: cervical spine fracture
detection/classification/segmentation is sparsely represented on arXiv but well
represented in the clinical/radiology literature PubMed indexes, so this topic needed
PubMed before it needed anything else.

## Decision

- **`PubmedSource`** (`data/pubmed.py`) implements the existing `Source` protocol,
  parsing NCBI's `PubmedArticleSet` XML with stdlib `xml.etree.ElementTree` (no new
  parsing dependency) into the same `PaperRecord` schema `ArxivSource` already produces -
  MeSH headings fill the `categories` field, the same field arXiv's category tags
  populate, so one `FilterConfig.category_scope` list works across both sources with no
  source-specific branch anywhere in `filter.py`.
- **`fetch_pubmed_xml`** (`data/pubmed_fetch.py`) wraps NCBI E-utilities (`esearch` then
  batched `efetch`), respecting NCBI's ≤3 req/s rate limit and `email` identification
  requirement. New `requests` dependency (the only new one this ADR introduces).
- **`combine_corpora`** (`data/combine.py`) merges multiple sources' parquet files,
  deduping on `(source, external_id)` - the same key `ingest()` already uses - so a
  multi-source corpus is one extra CLI step, not a schema change.
- **`configs/filter_cervical_spine.yaml`** is the second filter domain ADR 0005
  deferred, measured (informally - see Limitations) rather than guessed.
- Only titles, MeSH headings, dates, and PMIDs are ever shipped in this repo's derived
  report content (`reports/report_cervical_spine.md`) - no PubMed abstract text is
  quoted or committed. ADR 0001 flagged PubMed abstracts as carrying copyright risk
  distinct from arXiv's verified CC0 metadata scope; this project avoids resolving that
  risk by simply not needing to ship abstract text to tell the story.

## Verification

Full suite green (95 passed, 87.54% coverage), `ruff check`/`pyright` clean. New tests:
`test_pubmed_source.py`, `test_pubmed_fetch.py` (mocked NCBI calls, no real network
access in CI), `test_combine.py`.

A real run against live data (21 arXiv + 161 PubMed papers after filtering) found and
fixed three distinct filter-design problems that unit tests alone would not have caught,
since each one is a property of the *filter rule's content*, not its code:

1. **Unanchored keyword patterns.** An early draft included generic task vocabulary
   (`"spine segmentation"`, `"fracture detection"`) on the theory that it would narrow
   results. Since `keyword_patterns` are OR'd together, any single generic pattern
   matching let non-cervical spine papers (lumbar, thoracic) through. Fixed by requiring
   every pattern to carry a cervical-specific anchor.
2. **A broad vertebral-level regex matched animal anatomy.** `\bC[1-7]\b` (numbering
   human cervical vertebrae) also matches comparative-anatomy/paleontology papers using
   the identical notation for animal necks (sauropod, giraffe). Dropped the pattern
   entirely - every clinical case it was meant to catch is already covered by named
   patterns (`"cervical vertebra\w*"`, `"odontoid fracture\w*"`, etc.).
3. **`category_scope` let anatomy-only MeSH headings substitute for an AI-relatedness
   signal that nothing else enforced.** `category_scope` entries are OR'd internally and
   then AND'd with the keyword match (`matches_filter`). An earlier version of the
   PubMed category list mixed anatomy/injury MeSH headings (`Cervical Vertebrae`,
   `Spinal Fractures`) with AI-task headings (`Deep Learning`, `Machine Learning`) - so a
   purely clinical paper with zero AI/ML content could clear the category gate on an
   anatomy tag alone. This is the more serious of the three: it is a logic gap in what
   the filter actually enforces, not just an imprecise keyword. Fixed by restricting
   `category_scope` to AI/imaging-task MeSH headings only, making it play the same role
   on the PubMed side it already plays on the arXiv side (`category_scope` answers
   "could this involve AI/imaging computation"; `keyword_patterns` alone answers "is this
   about the cervical spine").

## Alternatives considered

| Option | Pros | Cons |
|---|---|---|
| PubMed via `Source` protocol (chosen) | No changes to `filter.py`, `ingest()`, or the topic-modeling stages; mirrors ADR 0001's stated design intent exactly | New dependency (`requests`); a second XML dialect to parse and keep correct |
| arXiv-only, accept a smaller/noisier cervical-spine corpus | No new source, no new dependency | The topic is specifically one arXiv under-covers (clinical literature); would have meant shipping a weaker result to avoid building the planned extension point |
| A general-purpose literature-search API/aggregator instead of PubMed directly | One integration for many future sources | No such aggregator was vetted for license terms; PubMed's E-utilities terms are a known, already-partially-evaluated quantity (ADR 0001's alternatives table) |

## Consequences

Easier: a third data source or a third filter domain is now a known, exercised path
(`Source` protocol + `FilterConfig` YAML + `combine_corpora`), not a hypothetical one.

Harder/unchanged: corpus size for a narrow domain (182 papers vs. the main corpus's
30,872) is small enough that several pipeline defaults tuned for the main corpus
(`min_cluster_size`, `min_df`, UMAP `n_neighbors`) needed CLI overrides rather than
working out of the box - `register-topic` gained `--min-cluster-size`/`--min-df`/
`--umap-n-neighbors` for this reason.

Deferred, not resolved: filter precision for this domain was spot-checked informally
(~93% on a 30-paper manual sample), not measured against a hand-labeled, stratified
sample the way ADR 0003 measures the main corpus's rule (0.895 precision, Wilson 95% CI).
Recall was not estimated at all. Doing so properly is the natural next step before
treating `reports/report_cervical_spine.md`'s findings as more than a provisional
reading - this ADR records the gap rather than papering over it with an unverified
number.
