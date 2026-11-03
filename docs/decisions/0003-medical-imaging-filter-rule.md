# ADR 0003: Medical-imaging filter rule

**Status:** accepted
**Date:** 2026-11-03

## Context

The corpus (3,233,527 papers, all of arXiv) is far larger than the "medical-imaging AI" scope this
project tells a story about. A filter rule is needed to cut it down, and its precision and recall need
measuring on a hand-labeled sample before trusting it for the rest of the pipeline (§M2 of the plan).

Candidate rules, cheapest to most expensive: category-only, category + keyword, embedding-similarity
classifier. The plan was to start with category + keyword and escalate only if precision/recall fell
below ~0.85.

## Decision

Use the baseline rule in `data/filter.py`: a paper passes if it is in the category scope
`{eess.IV, cs.CV, q-bio.QM, q-bio.TO, physics.med-ph}` **and** its title+abstract matches an
imaging/clinical-vocabulary keyword regex (modality terms like MRI/CT/ultrasound/histopathology, plus
general clinical-imaging terms). The category net is deliberately broad (cs.CV alone is mostly
non-medical computer vision); the keyword match does the real narrowing.

This rule passes **30,872 of 3,233,527 papers (~1.0%)**.

## Measurement

Stratified sample (seed 42, `data/sample.py`): 200 papers the filter accepted, 100 it rejected
(`data/interim/filter_labeling_sample_validation.csv`), labeled by hand.

**Precision** (accepted stratum, n=200): 179 relevant, 21 not relevant.
- Point estimate: **0.895**
- Wilson 95% CI: **(0.845, 0.930)**
- Above the 0.85 escalation threshold — no need to move to an embedding-similarity classifier on
  precision grounds.

**Recall** (combining both strata, `evaluate/precision_recall.estimate_precision_recall`): 0 relevant
papers found in the 100-row excluded-pool sample.
- Point estimate: **1.0** (no misses observed)
- **This number is not trustworthy as stated.** The excluded pool (3,202,655 papers) is ~104x the
  accepted pool (30,872), so recall is extremely sensitive to the *true* miss rate, not just the
  point estimate from 100 samples. The Wilson 95% CI on the excluded-pool's true relevant-rate is
  (0.0%, 3.7%) — and at the upper end of that CI alone, recall would be **0.189**, not 1.0:

  | excluded-pool relevant rate | estimated false negatives | recall |
  |---|---|---|
  | 0.0% (point estimate) | 0 | 1.000 |
  | 3.7% (Wilson CI upper bound) | ~118,600 | 0.189 |

  Zero misses in 100 samples is the best outcome this sample size *could* show, but it can't
  distinguish "the filter misses almost nothing" from "the filter misses a few percent of a
  3.2M-paper pool," and those two cases are wildly different in absolute terms (0 vs. ~100k papers).

## Alternatives considered

| Option | Pros | Cons |
|---|---|---|
| Category-only (no keyword) | Simpler, higher recall by construction | Precision would collapse — cs.CV alone is mostly non-medical vision; not measured, but expected well below 0.85 given the keyword rule is already doing the real work |
| Embedding-similarity classifier | Could improve recall, not threshold-brittle like regex | Not needed: precision already clears 0.85; recall's uncertainty here comes from sample size, not rule quality, so a fancier rule wouldn't fix the measurement problem |
| Larger excluded-pool sample (e.g. 500-1000) | Would tighten the recall CI enough to actually distinguish "misses nothing" from "misses a few percent" | More hand-labeling; deferred, not blocking M2 since precision is the metric that gated escalation |

## Consequences

- The baseline category + keyword rule stands; no escalation to an embedding classifier.
- Recall is reported as **unresolved, not measured-and-good** — the report (§5, Evidence) must state
  this plainly rather than quoting the 1.0 point estimate unqualified. This is a limitation to carry
  into the final story's "Limitations & ethics" section (M6), not a solved question.
- If recall becomes load-bearing later (e.g. a reviewer questions whether a known theme is
  under-represented), revisit by labeling a larger excluded-pool sample rather than re-deriving a point
  estimate from n=100.
