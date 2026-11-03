# Data quality — arXiv metadata snapshot

Computed by `imaging_ai_atlas.data.quality` against the full ingested corpus (Kaggle snapshot,
`Cornell-University/arxiv`, downloaded 2026-11-03).

## Headline numbers

| Metric | Value |
|---|---|
| Rows after ingest (deduplicated by `(source, external_id)`) | 3,233,527 |
| Duplicate ids collapsed during ingest | 93 |
| Date range | 1986-04-25 to 2026-10-29 |
| Records dated before arXiv's actual launch (1991-08-01) | 67 |
| Degenerate short abstracts (≤20 chars) | 52 |
| Abstracts containing LaTeX markup | 1,062,499 (33% of the corpus) |

## Findings

**Pre-launch dates are a backdating artifact, not a parsing bug.** arXiv went live in August 1991, but
67 records carry an earlier `created` date. Example: `physics/9403001`'s arXiv ID encodes a March 1994
submission, but its `versions[0].created` is 1986-04-25. These are old preprints where the metadata was
set to the paper's *original* publication date rather than its arXiv upload date. Conclusion: for this
small subset, the version timestamp is not a reliable submission-time signal — the ID is. This affects
~0.002% of the corpus and isn't worth a special case in `ingest`, but matters if a later stage ever needs
exact submission timing for all records.

**Degenerate abstracts exist and are findable by length alone.** The shortest abstracts are literally
`"No."` and `"Yes."` — almost certainly withdrawn or placeholder submissions. 52 records, negligible by
count, but each one would otherwise produce a meaningless embedding once the embedding stage runs. Worth
excluding before embedding, not before ingest (ingest's job is to parse, not to curate).

**A third of all abstracts contain LaTeX markup.** This is far larger than a few parsing edge cases — it
reflects how physics/math abstracts are actually written (inline `$x^2$`, `\alpha`-style macros). This is
a real decision point for the embedding stage: strip LaTeX before embedding (risk: losing meaning encoded
in notation) or leave it (risk: a sentence-transformer tokenizer treating `\alpha` as noise rather than
signal). Deferred to an ADR when that stage starts; noted here so it isn't rediscovered from scratch.

**eess.IV's year-by-year counts refine ADR 0001's claim.** The category wasn't a clean on/off switch at
2017: a handful of papers (1-5/year) were retroactively tagged back to 2010-2016, then real uptake starts
in 2017 (96) and explodes through 2018-2019 (600 → 3,139). The growth-trap risk for the eventual report:
reading this curve as "medical imaging AI research exploded in 2018" would conflate a category-availability
effect with actual field growth. The filter stage needs to account for this — filtering on the union
of overlapping categories (eess.IV + cs.CV etc.), not eess.IV alone, and the final report should show the
eess.IV-specific share separately rather than folding it into an undifferentiated total.

| Year | eess.IV papers |
|---|---|
| 2010-2016 | 1-5/year (negligible, retroactive tagging) |
| 2017 | 96 |
| 2018 | 600 |
| 2019 | 3,139 |
| 2020 | 4,889 |
| 2021-2025 | 3,800-4,700/year (plateau) |
| 2026 (partial year) | 2,196 |
