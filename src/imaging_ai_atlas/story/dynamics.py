import math

import numpy as np
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


def bootstrap_topic_share(
    topics: list[int], years: list[int], n_iterations: int = 1000, seed: int = 42
) -> pd.DataFrame:
    """Resample each year's non-outlier topic labels with replacement `n_iterations`
    times, recomputing every topic's share of that year per resample. Feeds
    `rank_topics_by_slope`'s confidence interval - resampling within each year (not across
    years) keeps each resample's year totals realistic, since the real question is "how
    much would this year's share estimate wobble with different papers," not "what if
    years had different relative sizes."
    """
    df = pd.DataFrame({"topic_id": topics, "year": years})
    df = df[df["topic_id"] != -1]
    rng = np.random.default_rng(seed)

    rows: list[dict[str, int | float]] = []
    for year in np.unique(np.asarray(df["year"])):
        year_topic_ids = np.asarray(df.loc[df["year"] == year, "topic_id"])
        n = len(year_topic_ids)
        for iteration in range(n_iterations):
            resample = rng.choice(year_topic_ids, size=n, replace=True)
            resampled_topic_ids, counts = np.unique(resample, return_counts=True)
            for topic_id, count in zip(resampled_topic_ids, counts, strict=True):
                rows.append(
                    {
                        "iteration": iteration,
                        "year": int(year),
                        "topic_id": int(topic_id),
                        "share": float(count) / n,
                    }
                )
    return pd.DataFrame(rows, columns=["iteration", "year", "topic_id", "share"])


def rank_topics_by_slope(
    share: pd.DataFrame,
    bootstrap_shares: pd.DataFrame,
    topic_labels: dict[int, str],
    min_year_total: int = 20,
) -> pd.DataFrame:
    """Rank topics by the slope of share-vs-year (rising first), with a percentile
    confidence interval on that slope from `bootstrap_shares`. Restricted to
    `topic_labels`'s keys - the human-reviewed, coherent topics - since an unlabeled
    topic has no trustworthy name to put in a ranking table, matching the same scoping
    `story/labels.py` already applies. Uses the same `min_year_total` threshold as the
    dynamics chart, for the same small-sample-noise reason.
    """
    valid_years = set(share.loc[share["year_total"] >= min_year_total, "year"].to_numpy().tolist())
    topic_ids = sorted(topic_labels)

    def _slope(years_values: np.ndarray, share_values: np.ndarray) -> float:
        if len(years_values) < 2:
            return float("nan")
        slope, _ = np.polyfit(years_values, share_values, 1)
        return float(slope)

    real_slopes = {}
    for topic_id in topic_ids:
        is_topic = share["topic_id"].to_numpy() == topic_id
        is_valid_year = np.isin(share["year"].to_numpy(), list(valid_years))
        subset = share.loc[is_topic & is_valid_year]
        real_slopes[topic_id] = _slope(subset["year"].to_numpy(), subset["share"].to_numpy())

    is_valid_year = np.isin(bootstrap_shares["year"].to_numpy(), list(valid_years))
    boot = bootstrap_shares.loc[is_valid_year]
    slope_samples: dict[int, list[float]] = {topic_id: [] for topic_id in topic_ids}
    for iteration in np.unique(boot["iteration"].to_numpy()):
        iteration_df = boot.loc[boot["iteration"].to_numpy() == iteration]
        for topic_id in topic_ids:
            points = iteration_df.loc[iteration_df["topic_id"].to_numpy() == topic_id]
            slope = _slope(points["year"].to_numpy(), points["share"].to_numpy())
            if not math.isnan(slope):
                slope_samples[topic_id].append(slope)

    rows = []
    for topic_id in topic_ids:
        samples = slope_samples[topic_id]
        ci_low, ci_high = (
            (float(x) for x in np.percentile(samples, [2.5, 97.5]))
            if samples
            else (float("nan"), float("nan"))
        )
        rows.append(
            {
                "topic_id": topic_id,
                "label": topic_labels[topic_id],
                "slope": real_slopes[topic_id],
                "slope_ci_low": ci_low,
                "slope_ci_high": ci_high,
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("slope", ascending=False, na_position="last")
        .reset_index(drop=True)
    )
