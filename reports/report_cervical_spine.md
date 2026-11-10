# Cervical spine fracture AI research: a topic-modeling view of arXiv + PubMed

## 1. Corpus

182 papers matched the cervical-spine-fracture filter (`configs/filter_cervical_spine.yaml`)
out of a combined arXiv + PubMed pool: 21 from arXiv (filtered the same way as the main
medical-imaging corpus) and 161 from PubMed (fetched via NCBI E-utilities,
`data/pubmed_fetch.py`, from a broadened esearch query covering fracture detection,
segmentation, classification, and cervical vertebral maturation). Dates span
2001-01-01 to 2026-10-27, with the large majority concentrated in 2024-2026 - this is a
young, still-forming sub-literature, not a mature one with decades of steady output.

This reuses the existing `ingest -> filter -> embed -> topic-model -> story` pipeline
unchanged (`Source` protocol, `FilterConfig`) - `PubmedSource` is a second implementation
of the same protocol `ArxivSource` already satisfies, and the same YAML-driven filter
narrows both sources with no code branch per source.

## 2. Data quality and filter design

Getting precision right here took three corrections, not one, each caught by inspecting
actual matched titles rather than trusting the regex:

- **Unanchored keyword patterns let in generic spine papers.** An early draft included
  patterns like `"spine segmentation"` and `"fracture detection"` on the theory that task
  vocabulary alone would narrow things down. It didn't - these patterns are OR'd together
  (`matches_filter`), so any one of them matching pulled in lumbar/thoracic spine papers
  with no cervical content at all. Fixed by requiring every keyword pattern to carry a
  cervical-specific anchor (`"cervical spine"`, `"cervical vertebra\w*"`, or a named
  cervical fracture: Jefferson, hangman's, odontoid, atlantoaxial).
- **A broad `\bC[1-7]\b` pattern matched animal anatomy, not just human vertebral
  levels.** Comparative-anatomy and paleontology papers (sauropod and giraffe neck
  biomechanics) use "C1"-"C7" to number animal cervical vertebrae. Dropped the pattern
  entirely rather than trying to regex around it - every real clinical case it caught is
  already covered by the named-fracture patterns above.
- **`category_scope` let anatomy-only MeSH headings satisfy the filter with no
  AI-relatedness signal at all.** An earlier version of the PubMed category list mixed
  anatomy/injury MeSH headings (`Cervical Vertebrae`, `Spinal Fractures`, `Spine`, ...)
  with AI-task headings. Since `category_scope` entries are OR'd internally and then
  AND'd with the keyword match, a purely clinical paper (e.g. a trauma-management case
  report) could clear the category gate on an anatomy MeSH tag alone, with nothing
  requiring any AI/ML signal. Fixed by restricting `category_scope` to AI/imaging-task
  MeSH headings only (`Deep Learning`, `Machine Learning`, `Artificial Intelligence`,
  `Neural Networks, Computer`, `Tomography, X-Ray Computed`, `Radiography`) - mirroring
  the arXiv side, where `category_scope` answers "could this involve AI/imaging
  computation" and `keyword_patterns` alone answers "is this about the cervical spine."

An informal 30-paper manual spot-check after these fixes found roughly 93% on-topic
(28/30) - comparable to, though not as rigorously measured as, the main corpus's
hand-labeled, stratified ADR-0003 baseline (89.5% precision, Wilson 95% CI
0.845-0.930). This spot-check was not saved as a reproducible labeled sample and should
be read as a sanity check, not a citable precision estimate.

## 3. Representation

Reused the main corpus's chosen embedding (nomic-embed-text-v1.5 @ 64 dims) and topic
pipeline unchanged in kind, but not in hyperparameters: BERTopic's defaults
(`min_cluster_size=15`, vectorizer `min_df=5`, UMAP `n_neighbors=15`) were tuned for the
~30,000-document main corpus and do not transfer to a corpus two orders of magnitude
smaller.

- At the main corpus's defaults, `register-topic` crashed outright (`max_df corresponds
  to < documents than min_df`) - fixed by adding `--min-cluster-size`/`--min-df`
  overrides to the CLI.
- At a first attempt on a 55-document corpus, a human-coherence review of the labeling
  sample found both resulting topics not coherent. Diagnosis: UMAP's `n_neighbors=15`
  treats a large fraction of a 55-document corpus as "nearby," flattening the local
  structure HDBSCAN needs to find separable clusters - compounded by the corpus itself
  being too small for BERTopic to discover reliable subtopics at all. This is the same
  methodological point the main project's ADR-0003 makes about automated topic metrics:
  a model will produce *some* clustering even on data too sparse to support it, which is
  why a human-coherence review gate exists rather than trusting NPMI/diversity alone.
- Responded by both expanding the corpus (55 -> 182 documents, mainly via the PubMed
  query broadening and the category-scope fix in section 2) and adding a tunable
  `--umap-n-neighbors` CLI override, run at `n_neighbors=10` for this corpus size.

## 4. Themes

![2D UMAP scatter plot of all 182 filtered papers, with the 3 topics colored and
direct-labeled (fracture detection/segmentation on CT, cervical vertebral maturation
(CVM) assessment, and clinical cervical-spine imaging), and 17 outlier papers shown in
grey](figures_cervical_spine/topic_map.png)

Topic 1 (cervical vertebral maturation) is clearly separated in the UMAP projection;
Topic 0 (fracture detection/segmentation) and Topic 2 (clinical cervical-spine imaging)
occupy adjacent but distinguishable arcs - a visible improvement over the 55-document
run's indistinct single blob, though with less separation than the main corpus's 8
largest topics on 30,872 documents.

All 3 topics were human-reviewed for coherence on a 3-row labeling sample
(`data/interim/topic_labeling_sample_cervical_spine.csv`):

- **Topic 1 - cervical vertebral maturation (CVM) assessment** (52 papers, e.g. CVM
  staging via deep learning, cephalometric landmark detection): marked coherent.
- **Topic 2 - clinical cervical-spine imaging** (44 papers, e.g. surgical outcome
  prediction, canal stenosis measurement, cervical lordotic curve trends): marked
  coherent.
- **Topic 0 - fracture detection/segmentation on CT** (69 papers): marked **not**
  coherent. Investigation confirmed two specific examples the reviewer flagged (a rugby
  scrummaging-biomechanics paper, a student-motion-tracking paper) are genuine
  keyword+category matches, not filter bugs - both contain literal "cervical spine"
  mentions and legitimate `Neural Networks, Computer`/`Machine Learning` MeSH tags, just
  outside the clinical-fracture scope this topic was meant to capture. This is accepted
  as the same precision ceiling the main corpus's own filter documents (section 2) rather
  than chased further with narrower regexes.

17 papers (9% of the corpus) were left as outliers (topic -1), unlike the main corpus's
reported post-reduction 0% - `reduce_outliers` was not forced to zero here, so this
number is the honest signal of how much of this smaller corpus didn't clearly fit any of
the 3 discovered topics.

## 5. Shifts

![Line chart of each topic's share of papers by year; flat beyond 2026 is an
extrapolation artifact of the bootstrap, not real data](figures_cervical_spine/topic_dynamics.png)

**Not reported as a trend finding.** The per-topic, per-year bootstrap
(`reports/topic_dynamics_summary_cervical_spine.csv`) returned zero rows - no topic's
year-over-year share met the minimum-count threshold the main report applies before
computing a confidence interval. With only 182 papers spread across a corpus where the
real coverage is effectively 3 years (2024-2026), per-year, per-topic counts are too
small (single digits to low tens) for any slope estimate to be more than noise. The
figure above is included only to show what the raw, unbootstrapped shares look like -
the apparent 2024-2025 swings and the flat line after 2026 are an artifact of sparse
data and the plotting code's extrapolation, not a measured trend, and should not be read
as one.

## 6. Takeaways, limitations, and ethics

**Takeaways:**
- The same `Source`/`FilterConfig` abstractions built for the arXiv-only main corpus
  extended to a second data source (PubMed) and a much narrower topic scope with no
  changes to `filter.py`, `ingest()`, or the topic-modeling stages themselves - only new
  plumbing (`PubmedSource`, `fetch_pubmed_xml`, `combine_corpora`) and CLI parameter
  overrides.
- Cervical vertebral maturation (orthodontic growth assessment) and general clinical
  cervical-spine imaging emerged as clearly separable, coherent themes even at this
  corpus size - cervical fracture detection/segmentation did not cleanly separate from
  adjacent sports-biomechanics and motion-tracking use of "cervical spine" terminology.

**Limitations:**
- **This corpus is two orders of magnitude smaller than the main one** (182 vs. 30,872
  filtered papers) - every stage from topic discovery to trend analysis is working with
  far less statistical power, and several hyperparameters tuned for the main corpus
  (`min_cluster_size`, `min_df`, UMAP `n_neighbors`) had to be manually overridden rather
  than left at their validated defaults.
- **Trend analysis (section 5) could not be performed at all** - not a null result, an
  absence of a result, because the per-year sample is too sparse for the bootstrap to
  produce a meaningful confidence interval.
- **The 93% spot-check precision figure (section 2) is informal**, not a hand-labeled
  stratified sample like the main corpus's ADR-0003 measurement - it should not be
  quoted as a rigorous precision estimate, and recall was not estimated at all for this
  filter.
- **Topic 0's incoherence is a known, accepted limitation, not fully resolved** - two
  specific false positives were traced and explained (section 4), but the underlying
  cause (fracture/cervical-spine vocabulary overlapping with sports-biomechanics and
  motion-tracking papers) is not something further regex narrowing is expected to fully
  close, mirroring the main corpus's own documented precision ceiling (0.895).
- **17 of 182 papers (9%) are unassigned outliers** - a corpus this small has less
  structure for every paper to fit a topic, and this was left visible rather than forced
  to zero via `reduce_outliers`.

**Ethics:**
- Same as the main report: only public metadata (titles, abstracts, categories/MeSH
  headings, dates) was used from both arXiv and PubMed - no patient data, clinical
  records, or protected health information. PubMed access used NCBI E-utilities under
  their stated usage policy (rate-limited, identified by an email parameter, not
  authenticated access to any restricted data).
- The same risk the main report flags applies here with more force given the smaller
  sample: a 3-topic split on 182 papers is a much weaker signal than the main corpus's
  30-topic review on 30,872 papers, and should not be read as a definitive map of the
  cervical-spine-fracture-AI literature - it is a narrow, methodologically-limited slice,
  presented with its limitations explicit rather than smoothed over.
