import pandas as pd


def topic_volume_by_year(topics: list[int], years: list[int]) -> pd.DataFrame:
    """Per-topic, per-year document counts - the raw input to both an absolute-volume
    and a share-of-corpus dynamics chart. The outlier bucket (-1) is excluded, it isn't a
    real research theme and would dwarf every other topic's volume.
    """
    df = pd.DataFrame({"topic_id": topics, "year": years})
    df = df[df["topic_id"] != -1]
    counts = df.groupby(["year", "topic_id"]).size()
    counts.name = "count"
    return counts.reset_index()


def topic_share_by_year(volume: pd.DataFrame) -> pd.DataFrame:
    """Each topic's count as a fraction of that year's total (non-outlier) paper count -
    volume alone conflates "this topic grew" with "the whole corpus grew that year" (arXiv
    submissions trend upward over time regardless of topic), share disentangles the two.
    `year_total` is kept (not dropped) so callers can filter out years with too few papers
    to be a meaningful share - a handful of documents gives wild, meaningless swings.
    """
    merged = volume.copy()
    merged["year_total"] = merged.groupby("year")["count"].transform("sum")
    merged["share"] = merged["count"] / merged["year_total"]
    return merged
