import re
from pathlib import Path

import pandas as pd
import yaml
from pydantic import BaseModel, field_validator


class FilterConfig(BaseModel):
    """A topic-domain filter rule: (category_scope AND any keyword_pattern) over title +
    abstract. Parameterizing this (rather than hardcoding one domain's categories and
    keywords as module constants) is what lets this project point at a different research
    topic - e.g. cervical spine fracture classification instead of medical imaging AI -
    via a new YAML file, with no code change.
    """

    name: str
    category_scope: list[str]
    keyword_patterns: list[str]

    @field_validator("category_scope", "keyword_patterns")
    @classmethod
    def _lists_must_be_non_empty(_cls, values: list[str]) -> list[str]:
        if not values:
            raise ValueError(
                "must be non-empty - an empty category_scope rejects every row (silently "
                "fails closed), but an empty keyword_patterns joins to the empty regex, "
                "which re.search/str.contains match against *any* string (silently fails "
                "open, accepting everything in-category). Neither is ever intended."
            )
        return values

    @field_validator("keyword_patterns")
    @classmethod
    def _patterns_must_compile(_cls, patterns: list[str]) -> list[str]:
        for pattern in patterns:
            re.compile(pattern)  # raises re.error with a clear message if malformed
        return patterns

    @property
    def keyword_pattern_str(self) -> str:
        return "|".join(self.keyword_patterns)


def load_filter_config(path: Path) -> FilterConfig:
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return FilterConfig.model_validate(raw)


def matches_filter(categories: list[str], title: str, abstract: str, config: FilterConfig) -> bool:
    """The baseline filter rule: in the category scope AND mentions the configured vocabulary."""
    in_scope_category = any(cat in config.category_scope for cat in categories)
    has_keyword = bool(re.search(config.keyword_pattern_str, f"{title} {abstract}", re.IGNORECASE))
    return in_scope_category and has_keyword


def apply_filter(df: pd.DataFrame, config: FilterConfig) -> pd.Series:
    """Boolean mask over `df` for the baseline filter rule.

    Keyword matching is vectorized (str.contains over the whole column); category-list
    membership isn't, since pandas has no vectorized "any element of this list column is in
    this set" operation without exploding the column first — not worth it for one pass.
    """
    title_and_abstract = df["title"] + " " + df["abstract"]
    has_keyword = title_and_abstract.str.contains(
        config.keyword_pattern_str, case=False, regex=True
    )
    category_scope = frozenset(config.category_scope)
    in_scope_category = df["categories"].apply(lambda cats: any(c in category_scope for c in cats))
    return in_scope_category & has_keyword
