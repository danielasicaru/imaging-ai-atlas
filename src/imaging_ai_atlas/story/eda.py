import pandas as pd


def papers_per_year(years: pd.Series) -> pd.Series:
    """Paper count per year, with every year between the first and last present (0
    where nothing was published) so a chart can't silently skip a gap.
    """
    counts = years.value_counts()
    full_range = range(int(years.min()), int(years.max()) + 1)
    return counts.reindex(full_range, fill_value=0)


def category_share_by_year(
    years: list[int], categories: list[list[str]], top_n: int, min_year_total: int
) -> pd.DataFrame:
    """Per year, the fraction of papers tagged with each of the `top_n` most-used
    categories, as a year x category table in descending order of use. A paper counts
    once for every category it carries, so shares can sum past 1. Years with fewer than
    `min_year_total` papers are dropped: a handful of papers turns every share into 0%
    or 100%.
    """
    papers = pd.DataFrame({"year": years, "category": categories})
    year_total = papers["year"].value_counts()
    tags = papers.explode("category")
    top = tags["category"].value_counts().index[:top_n]

    counts = pd.crosstab(tags["year"], tags["category"]).reindex(columns=top, fill_value=0)
    totals = year_total.reindex(counts.index)
    return counts.div(totals, axis=0).loc[totals >= min_year_total]


def largest_topics(topics: list[int], topic_labels: dict[int, str], n: int) -> pd.DataFrame:
    """The `n` largest topics by paper count (`label`, `size`), largest first. The -1
    outlier bucket is skipped: it holds papers that fit no theme, so it isn't one.
    """
    sizes = pd.Series([t for t in topics if t != -1]).value_counts().head(n)
    labels = [topic_labels.get(int(t), f"Topic {t}") for t in sizes.index]
    return pd.DataFrame({"label": labels, "size": sizes.to_numpy()})
