import re

import pandas as pd

# Categories where a medical-imaging AI paper could plausibly live. Deliberately broad —
# this is the "could this paper be relevant at all" net, not the final answer. cs.CV alone
# is almost entirely non-medical computer vision; the keyword rule below does the real
# narrowing. eess.IV is kept separate in reporting (see reports/data_quality.md) because its
# adoption curve is itself a category-introduction artifact, not organic growth.
CATEGORY_SCOPE = frozenset({"eess.IV", "cs.CV", "q-bio.QM", "q-bio.TO", "physics.med-ph"})

# Imaging modalities and clinical-imaging vocabulary. Intentionally conservative (misses are
# expected and will be measured against the excluded pool) rather than overly broad (which
# would need recall measured against a much larger labeled sample to catch false positives).
_MODALITY_TERMS = [
    r"MRI",
    r"magnetic resonance imag\w*",
    r"\bCT\b",
    r"computed tomography",
    r"\bX-ray\b",
    r"radiograph\w*",
    r"mammograph\w*",
    r"ultrasound",
    r"sonograph\w*",
    r"PET scan",
    r"PET/CT",
    r"histopatholog\w*",
    r"whole[- ]slide imag\w*",
    r"endoscop\w*",
    r"\bOCT\b",
    r"optical coherence tomography",
]
_CLINICAL_TERMS = [
    r"medical imag\w*",
    r"clinical imag\w*",
    r"radiolog\w*",
    r"tumor segmentation",
    r"lesion segmentation",
    r"diagnos\w* imag\w*",
]

_KEYWORD_PATTERN_STR = "|".join(_MODALITY_TERMS + _CLINICAL_TERMS)
_KEYWORD_PATTERN = re.compile(_KEYWORD_PATTERN_STR, re.IGNORECASE)


def matches_filter(categories: list[str], title: str, abstract: str) -> bool:
    """The baseline filter rule: in the category scope AND mentions imaging/clinical vocabulary."""
    in_scope_category = any(cat in CATEGORY_SCOPE for cat in categories)
    has_keyword = bool(_KEYWORD_PATTERN.search(f"{title} {abstract}"))
    return in_scope_category and has_keyword


def apply_filter(df: pd.DataFrame) -> pd.Series:
    """Boolean mask over `df` for the baseline filter rule.

    Keyword matching is vectorized (str.contains over the whole column); category-list
    membership isn't, since pandas has no vectorized "any element of this list column is in
    this set" operation without exploding the column first — not worth it for one pass.
    """
    title_and_abstract = df["title"] + " " + df["abstract"]
    has_keyword = title_and_abstract.str.contains(_KEYWORD_PATTERN_STR, case=False, regex=True)
    in_scope_category = df["categories"].apply(lambda cats: any(c in CATEGORY_SCOPE for c in cats))
    return in_scope_category & has_keyword
