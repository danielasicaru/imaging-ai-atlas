# Medical-imaging AI research, 1986-2026: a topic-modeling view of arXiv

## 1. Corpus

3,233,527 papers ingested from the full arXiv metadata snapshot (Cornell University's
Kaggle mirror), spanning 1986-04-25 to 2026-10-29. 93 duplicate `(source, external_id)`
pairs were collapsed during ingest.

## 2. Data quality

A third of all abstracts (1,062,499, 33%) contain LaTeX markup - far more than a few
parsing edge cases, reflecting how physics/math abstracts are actually written. Whether
to strip this before embedding is a real tradeoff (losing notation-encoded meaning vs. a
tokenizer treating `\alpha` as noise) and was deferred to the embedding stage rather than
decided at ingest.

52 abstracts are degenerate placeholders (as short as `"No."` or `"Yes."`), excluded
before embedding rather than at ingest, since ingest's job is to parse, not curate.

67 records carry a `created` date before arXiv's actual 1991-08-01 launch - a backdating
artifact in old preprints where the metadata was set to the paper's original publication
date, not its arXiv upload date. Negligible by count (~0.002% of the corpus).

The `eess.IV` category's year-by-year counts matter for how the rest of this report reads
growth: 1-5 papers/year from 2010-2016 are a retroactive-tagging artifact, not real
activity; real uptake starts in 2017 (96 papers) and explodes through 2018-2019
(600 -> 3,139). Reading this curve alone as "the field exploded in 2018" would conflate a
category-availability effect with actual research growth - the filter rule (below) uses
the union of several overlapping categories specifically to avoid depending on this one
category's availability history.

## 3. Representation

Three embedding models (MiniLM-L6-v2, SPECTER, nomic-embed-text-v1.5) were compared
across 7 variants (nomic's Matryoshka truncations at 64/128/256/512/768 dims) on topic
diversity, NPMI coherence, outlier rate, and 5-seed ARI stability.

**Primary embedding: nomic-embed-text-v1.5 @ 64 dims** - highest diversity (0.695) and
the cheapest to run downstream, with stability (ARI 0.632) a close second to SPECTER's
best (0.653). Truncating nomic from 768 to 64 dimensions cost neither diversity nor
stability - this corpus does not need a large embedding. NPMI barely discriminated between
variants (0.062-0.071 across the board), consistent with the literature's warning that
automated coherence correlates only weakly with topic quality; it was not treated as a
deciding metric.

A PCA-5D control run on the same nomic@64 embedding is a clean validation that UMAP's
non-linear structure is doing real work, not adding complexity for a marginal gain: UMAP
finds 189 topics with 38.4% outliers, while PCA on the identical embedding collapses to 3
topics with 81.2% outliers.

## 4. Themes

![Topic map](figures/topic_map.png)

30 of the largest topics were human-reviewed for coherence; all 30 were marked coherent,
covering themes including brain tumor MRI, retinal OCT/fundus imaging, cardiac MRI, breast
cancer/mammography, COVID-19 chest X-ray, SAM-based segmentation adaptation, diffusion
generative models, dental CBCT, knee osteoarthritis imaging, and whole-slide-image/
multiple-instance-learning histopathology.

## 5. Evidence

The filter rule that defines "in scope" for this whole project (category scope
`{eess.IV, cs.CV, q-bio.QM, q-bio.TO, physics.med-ph}` AND an imaging/clinical keyword
match) was measured on a hand-labeled, stratified sample (200 accepted, 100 rejected,
seed 42):

- **Precision: 0.895** (Wilson 95% CI: 0.845-0.930) - comfortably above the 0.85
  threshold that would have triggered escalating to an embedding-similarity classifier.
- **Recall is explicitly unresolved, not measured-and-good.** The point estimate from the
  excluded-pool sample is 1.0 (zero misses in 100 labeled rejections), but the excluded
  pool is ~104x the accepted pool, so this estimate is extremely sensitive to sample size:
  at the upper end of the excluded pool's 95% CI on its true relevant-rate (3.7%), recall
  would be as low as **0.189**. Zero observed misses in 100 samples cannot distinguish "the
  filter misses almost nothing" from "the filter misses a few percent of 3.2M papers,"
  and those two cases differ by roughly 100,000 papers in absolute terms.

The baseline filter rule passes 30,872 of 3,233,527 papers (~1.0% of all of arXiv).

## 6. Shifts

![Topic dynamics](figures/topic_dynamics.png)

Per-year topic share was bootstrapped (1,000 resamples within each year) to put a
confidence interval on each of the 30 coherent topics' trend, ranked by the slope of
share-vs-year (`reports/topic_dynamics_summary.csv`).

The fastest-rising topic by slope is **Radiology report generation** (slope 0.00327/year,
95% CI [0.00210, 0.00474]) - a clear, unambiguous rise since its whole CI sits above zero.
Close behind: **whole-slide image classification (multiple instance learning)** (slope
0.00211/year, 95% CI [0.00108, 0.00272]) and **diffusion generative models for medical
imaging** (slope 0.00211/year, 95% CI [0.00131, 0.00356]).

The fastest-fading topic is **Radiation detectors and therapy physics** (slope
-0.00754/year, 95% CI [-0.01111, -0.00470]), followed by **COVID-19 detection from chest
X-ray and CT** (slope -0.00628/year, 95% CI [-0.02262, -0.00570]) - both CIs sit entirely
below zero, so unlike several mid-table topics whose CIs straddle zero (e.g. cardiac MRI,
federated learning), these two declines are not just bootstrap noise.

## 7. Takeaways and limitations

**Takeaways:**
- A relatively small, efficient embedding (64 dimensions) captures this corpus's thematic
  structure as well as a 768-dimension model - dimensionality was not the bottleneck here.
- UMAP's non-linear projection is doing substantive work; a linear control (PCA) on the
  identical embedding collapses almost the entire corpus into one dominant cluster plus
  noise.
- Natural-language radiology report generation is the most unambiguously rising theme in
  the corpus (slope's full 95% CI stays positive) - a shift from purely perceptual tasks
  (segmentation, classification) toward generating clinical text, consistent with the
  broader trend of applying large language models to medical imaging.

**Limitations:**
- Filter recall is not actually known to be good - only that zero misses were observed in
  a 100-row sample far too small to rule out a few-percent true miss rate against a
  3.2-million-paper excluded pool (section 5).
- The 0% post-reduction outlier rate reported by BERTopic's `reduce_outliers` is a
  methodological artifact: every document, however marginal, gets force-assigned to its
  nearest topic. The honest signal is the *before* rate (31.9%-41.7% across the 7 compared
  variants) - this report does not treat the post-reduction 0% as evidence of a cleanly
  clustered corpus.
- Whether to strip LaTeX markup before embedding was deferred, not resolved - abstracts
  with heavy notation may be embedded somewhat differently than prose-only abstracts as a
  result.
- `eess.IV`'s 2017-2019 growth is partly a category-availability effect (the category only
  started seeing real uptake in 2017), not purely organic research growth - the filter
  rule's reliance on a union of categories mitigates this but does not eliminate it as a
  confound in any category-specific trend reading.
- The rising/fading ranking in section 6 is restricted to the 30 largest, human-reviewed
  topics - it says nothing about trends among the ~160 smaller topics this project did not
  review for coherence.
