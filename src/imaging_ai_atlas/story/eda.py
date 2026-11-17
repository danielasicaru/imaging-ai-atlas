import pandas as pd

from imaging_ai_atlas.story.labels import topic_label


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
    labels = [topic_label(int(t), topic_labels) for t in sizes.index]
    return pd.DataFrame({"label": labels, "size": sizes.to_numpy()})


def topic_mix_by_source(
    topics: list[int], sources: list[str], topic_labels: dict[int, str]
) -> pd.DataFrame:
    """Papers per theme for each source, as raw counts and as a share of that source's
    papers (columns `("count", source)` and `("share", source)`, one row per theme).
    Shares are taken within each source because sources can differ a lot in size; the
    counts stay alongside so a share built on a handful of papers is visible as such.
    Topic -1 is reported as "No clear theme" rather than dropped.
    """
    labels = ["No clear theme" if t == -1 else topic_label(t, topic_labels) for t in topics]
    counts = pd.crosstab(pd.Series(labels, name="theme"), pd.Series(sources, name="source"))
    shares = counts / counts.sum()
    return pd.concat({"count": counts, "share": shares}, axis=1)
