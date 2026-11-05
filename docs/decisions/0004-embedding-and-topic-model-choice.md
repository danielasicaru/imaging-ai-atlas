# ADR 0004: Embedding and topic-model choice

**Status:** accepted
**Date:** 2026-11-04

## Context

The embedding sweep produced 7 variants (MiniLM, SPECTER, and nomic-embed-text-v1.5
Matryoshka-truncated to 64/128/256/512/768 dims). One of them needed to be picked to carry
forward as the primary embedding for the topic model that the rest of the project
(dynamics, story) builds on, backed by real comparison numbers rather than a model-card
preference.

## Decision

**Primary embedding: nomic-embed-text-v1.5 @ 64 dims**, with UMAP (5D, cosine) -> HDBSCAN ->
CountVectorizer -> c-TF-IDF as the topic model, as specified in `configs/topics.yaml`.

Chosen as a deliberate tradeoff, not the single best number on every metric: nomic@64 has the
highest topic diversity (0.696) and the smallest embedding (fastest/cheapest for the downstream
dynamics and story stages), with stability (ARI 0.632) a close second to SPECTER's best. This
directly answers the original question, "how many dimensions does this corpus need?" — the answer
is relatively few; truncating nomic from 768 to 64 dims did not meaningfully cost diversity or
stability.

## Measurement

Full comparison (`atlas topics`, one BERTopic fit + 5-seed stability refit per variant, logged to
MLflow):

| Model | Dims | Topics | Diversity | NPMI | Outliers (before→after) | ARI stability | Runtime |
|---|---|---|---|---|---|---|---|
| nomic | 64 | 189 | 0.695 | 0.071 | 38.4%→0% | 0.632 | 38s |
| nomic | 128 | 216 | 0.696 | 0.071 | 33.3%→0% | 0.569 | 39s |
| nomic | 256 | 253 | 0.674 | 0.068 | 31.9%→0% | 0.581 | 40s |
| nomic | 512 | 276 | 0.661 | 0.065 | 32.7%→0% | 0.585 | 41s |
| nomic | 768 | 283 | 0.659 | 0.062 | 33.5%→0% | 0.582 | 41s |
| SPECTER | 768 | 213 | 0.652 | 0.068 | 41.7%→0% | **0.653** | 40s |
| MiniLM | 384 | 223 | 0.676 | 0.068 | 36.7%→0% | 0.572 | 84s |

SPECTER has the single best stability number, which is a real, defensible alternative choice (its
citation-based training plausibly produces more consistent clusters). It was not picked because it
has the worst diversity of the seven variants and offers no Matryoshka story — nomic@64's diversity
+ efficiency tradeoff was judged to matter more for this project's "how many dims does this corpus
need" narrative.

NPMI barely discriminates between variants (0.062–0.071 across the board) — consistent with the
literature's warning that automated coherence correlates only weakly with topic quality; it was not
treated as a deciding metric here.

**Outlier reduction (`c-tf-idf` strategy) drove every variant's outlier rate to exactly 0%** —
flagged as suspicious rather than accepted uncritically; this means every document, however
marginal, got force-assigned to its nearest topic. The *before* rate (31.9%-41.7% across variants)
is the more honest signal of how much HDBSCAN actually clustered confidently; the *after* number
is a methodological artifact of the reduction strategy, not evidence of a cleaner corpus. The
final report must present outlier rates as the before number with the reduction step's effect
described, not quote 0% as a quality result.

**PCA-5D control** (same nomic@64 embedding, PCA instead of UMAP, otherwise identical pipeline):

| Reducer | Topics | Diversity | NPMI | Outlier rate | Runtime |
|---|---|---|---|---|---|
| UMAP (chosen) | 189 | 0.695 | 0.071 | 38.4% | 38s |
| PCA (control) | 3 | 0.767 | 0.061 | **81.2%** | 22s |

PCA collapsed the corpus into effectively one dominant cluster (5,727 docs), two tiny ones (72 and
20 docs), and 25,053 of 30,872 documents (81.2%) as unclustered noise. This is a strong, clean
result: UMAP's non-linear structure is doing real work here, not just adding complexity for a
marginal gain. PCA's higher diversity score (0.767) is not meaningful — with only 3 topics, a
handful of keyword lists are nearly guaranteed to have little overlap; diversity as a metric assumes
a reasonable number of topics to be a sane comparison at all.

## Alternatives considered

| Option | Pros | Cons |
|---|---|---|
| SPECTER @ 768 | Best measured stability (0.653 vs 0.632) | Worst diversity of all 7 variants; no Matryoshka story, fixed dim only |
| nomic @ 768 (full dim) | Slightly higher NPMI, more topics found | Diversity actually *lower* than 64 dims; no efficiency benefit for the extra 12x storage/compute |
| Escalate to a 4th embedding model | Could in principle beat all 7 measured variants | Not justified: the plan called for comparing 3 models, not an open-ended search: diminishing-returns risk for a portfolio project |

## Consequences

- `configs/topics.yaml`'s `embedding_variants` list stays as the full 7-way comparison set
  (useful ablation evidence for the report); the primary pipeline (`atlas register-topic`,
  `atlas story`) points at nomic@64 specifically.
- The outlier-rate caveat above must carry into the final report's Limitations section, not
  be silently dropped in favor of the cleaner-looking 0% post-reduction number.
- If recall/topic quality concerns arise later that specifically implicate low dimensionality,
  revisit by comparing nomic@64 against nomic@256 on the human-labeled 30-topic check (not done
  yet) before assuming 64 is adequate everywhere.
