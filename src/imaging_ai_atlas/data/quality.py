import re

import pandas as pd
from pydantic import BaseModel

# arXiv went live in August 1991. A created date before this means the version metadata
# reflects the paper's original (pre-arXiv) publication date, not its arXiv submission —
# a known backdating artifact in old physics/math preprints, not a parsing bug.
ARXIV_LAUNCH_DATE = pd.Timestamp("1991-08-01")

# Abstracts at or under this length can't carry enough signal for topic embedding either way;
# they're withdrawal placeholders ("No.", "Yes.") more often than real short abstracts.
SHORT_ABSTRACT_THRESHOLD = 20

# Cheap heuristic for literal LaTeX leaking into abstract text: a backslash command or
# unescaped math delimiters. Not a full LaTeX parser — false positives (e.g. a stray "$")
# are expected and fine for a report-level count, not a filtering decision.
_LATEX_PATTERN = re.compile(r"\\[a-zA-Z]+|\$[^$]+\$")


class QualityReport(BaseModel):
    row_count: int
    date_min: str
    date_max: str
    pre_launch_count: int
    short_abstract_count: int
    latex_leakage_count: int
    papers_per_year: dict[int, int]


def category_year_counts(df: pd.DataFrame, category: str) -> dict[int, int]:
    """Per-year paper counts for one category — surfaces category-introduction artifacts
    (e.g. a category barely used for years, then jumping once arXiv promotes/splits it).
    """
    has_category = df["categories"].apply(lambda cats: category in cats)
    years = pd.to_datetime(df.loc[has_category, "created"]).dt.year
    return years.value_counts().sort_index().to_dict()


def compute_quality_report(df: pd.DataFrame) -> QualityReport:
    created = pd.to_datetime(df["created"])

    return QualityReport(
        row_count=len(df),
        date_min=str(created.min().date()),
        date_max=str(created.max().date()),
        pre_launch_count=int((created < ARXIV_LAUNCH_DATE).sum()),
        short_abstract_count=int((df["abstract"].str.len() <= SHORT_ABSTRACT_THRESHOLD).sum()),
        latex_leakage_count=int(df["abstract"].str.contains(_LATEX_PATTERN, regex=True).sum()),
        papers_per_year=created.dt.year.value_counts().sort_index().to_dict(),
    )
