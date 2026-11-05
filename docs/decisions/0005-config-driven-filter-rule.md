# ADR 0005: Config-driven filter rule instead of hardcoded category/keyword constants

**Status:** accepted
**Date:** 2026-11-05

## Context

The filter rule (ADR 0003) was implemented as module-level constants in `data/filter.py`:
a hardcoded `CATEGORY_SCOPE` frozenset and two hardcoded keyword-pattern lists. This was
fine while the project had exactly one topic domain (medical imaging AI), but it hardcodes
a domain-specific decision into code. Retargeting this pipeline at a different, narrower
research topic (e.g. cervical spine fracture classification, AO Spine classification
systems) would have meant editing `filter.py` directly - the same shape of change as
editing `embed.yaml` to add a model, except not exposed as data.

## Decision

`data/filter.py` now defines a pydantic `FilterConfig` (`name`, `category_scope`,
`keyword_patterns`) loaded from a YAML file via `load_filter_config(path)`, the same
pattern already used for `embed.yaml`/`topics.yaml`. `matches_filter` and `apply_filter`
take a `FilterConfig` argument instead of reading module constants. The current rule moved
verbatim into `configs/filter_medical_imaging.yaml`. Every CLI command that filters the
corpus (`embed`, `topics`, `register-topic`, `story`) gained a `--filter-config-path`
option defaulting to that file, loaded through one shared `_load_filtered_corpus` helper
so the loading/filtering logic isn't duplicated four times.

A new topic domain is now a new YAML file with no code change: a different
`category_scope`/`keyword_patterns` list under a different `name`, pointed at via
`--filter-config-path`. `keyword_patterns` are validated at load time (each must compile
as a regex) so a malformed pattern fails fast with a clear error, not a confusing
downstream `re.error` deep in a vectorized `str.contains` call.

## Alternatives considered

| Option | Pros | Cons |
|---|---|---|
| Config-driven `FilterConfig` (chosen) | Matches existing `embed.yaml`/`topics.yaml` pattern; new domain = new file, zero code change; regex validated at load time | One more config file to keep straight; doesn't generalize anything beyond category+keyword (e.g. a domain needing date-range or venue filtering would still need code changes) |
| Subclass `FilterConfig` per domain in Python | Type-safe per-domain fields if a domain ever needs more than category+keyword | Overkill for the one dimension that actually varies today; reintroduces "new domain needs a code change" |
| Leave hardcoded, just note the domain in a comment | No work | Directly blocks the stated future use case; the whole point of this ADR |

## Consequences

Easier: adding a new research-topic domain (e.g. a narrower clinical-imaging subtopic) is
now a YAML file plus a labeled precision/recall sample (ADR 0003's method still applies
per-domain), not a `filter.py` edit. Every stage's CLI command threads the same
`filter_config_path` consistently.

Harder/unchanged: this only generalizes the *filter* stage. `configs/embed.yaml` and
`configs/topics.yaml` still assume the same three embedding models are reasonable for any
domain - a genuinely different domain might want a different embedding sweep, which is a
separate decision to make if/when a second domain is actually run, not something this ADR
tries to solve speculatively.

Deferred: no second domain's keyword list was written as part of this decision - inventing
one (e.g. for cervical spine fracture classification) without a labeled sample to validate
precision/recall against would be guessing, not engineering. The capability is proven by
tests (`test_a_different_topic_domain_config_changes_what_matches`), not by a second real
config file.
