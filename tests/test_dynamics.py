import math

import pandas as pd

from imaging_ai_atlas.story.dynamics import (
    bootstrap_topic_share,
    rank_topics_by_slope,
    topic_share_by_year,
    topic_volume_by_year,
)


def test_topic_volume_by_year_counts_per_topic_per_year_and_excludes_outliers() -> None:
    topics = [0, 0, 1, -1, 0]
    years = [2020, 2020, 2021, 2020, 2021]

    volume = topic_volume_by_year(topics, years)

    keys = list(zip(volume["year"], volume["topic_id"], strict=True))
    rows = dict(zip(keys, volume["count"], strict=True))
    assert rows[(2020, 0)] == 2
    assert rows[(2021, 0)] == 1
    assert rows[(2021, 1)] == 1
    assert (2020, -1) not in rows


def test_topic_share_by_year_normalizes_within_each_year() -> None:
    topics = [0, 0, 1, 0, 1]
    years = [2020, 2020, 2020, 2021, 2021]
    volume = topic_volume_by_year(topics, years)

    share = topic_share_by_year(volume)

    year_2020: dict[int, float] = (
        share[share["year"] == 2020].set_index("topic_id")["share"].to_dict()
    )
    assert math.isclose(year_2020[0], 2 / 3)
    assert math.isclose(year_2020[1], 1 / 3)
    year_2021: dict[int, float] = (
        share[share["year"] == 2021].set_index("topic_id")["share"].to_dict()
    )
    assert math.isclose(year_2021[0], 0.5)
    assert math.isclose(year_2021[1], 0.5)


def test_bootstrap_topic_share_is_deterministic_given_a_fixed_seed() -> None:
    topics = [0, 0, 1, 1, 0, 1, 0, 1]
    years = [2020, 2020, 2020, 2020, 2021, 2021, 2021, 2021]

    result1 = bootstrap_topic_share(topics, years, n_iterations=5, seed=7)
    result2 = bootstrap_topic_share(topics, years, n_iterations=5, seed=7)

    pd.testing.assert_frame_equal(result1, result2)


def test_bootstrap_topic_share_resamples_are_valid_and_average_near_true_share() -> None:
    # 100 docs in one year, true share of topic 0 is exactly 0.3.
    topics = [0] * 30 + [1] * 70
    years = [2020] * 100

    result = bootstrap_topic_share(topics, years, n_iterations=500, seed=1)

    assert bool((result["share"] >= 0).all())
    assert bool((result["share"] <= 1).all())
    mean_share_topic_0 = result.loc[result["topic_id"] == 0, "share"].mean()
    assert math.isclose(mean_share_topic_0, 0.3, abs_tol=0.03)


def _expand_counts(counts_by_year_topic: dict[int, dict[int, int]]) -> tuple[list[int], list[int]]:
    topics: list[int] = []
    years: list[int] = []
    for year, counts in counts_by_year_topic.items():
        for topic_id, count in counts.items():
            topics.extend([topic_id] * count)
            years.extend([year] * count)
    return topics, years


def test_rank_topics_by_slope_orders_rising_above_flat_above_falling() -> None:
    # topic 0 share rises .1 -> .5, topic 1 share is exactly .25 every year (flat),
    # topic 2 absorbs the rest and falls. 200 docs/year clears min_year_total=20 easily
    # and keeps bootstrap variance small enough for a stable CI-straddles-zero check.
    counts_by_year_topic = {
        2018: {0: 20, 1: 50, 2: 130},
        2019: {0: 40, 1: 50, 2: 110},
        2020: {0: 60, 1: 50, 2: 90},
        2021: {0: 80, 1: 50, 2: 70},
        2022: {0: 100, 1: 50, 2: 50},
    }
    topics, years = _expand_counts(counts_by_year_topic)
    volume = topic_volume_by_year(topics, years)
    share = topic_share_by_year(volume)
    bootstrap_shares = bootstrap_topic_share(topics, years, n_iterations=200, seed=42)
    topic_labels = {0: "rising", 1: "flat", 2: "falling"}

    ranking = rank_topics_by_slope(share, bootstrap_shares, topic_labels, min_year_total=20)

    assert list(ranking["topic_id"]) == [0, 1, 2]
    rising_row = ranking[ranking["topic_id"] == 0].iloc[0]
    assert rising_row["slope"] > 0
    falling_row = ranking[ranking["topic_id"] == 2].iloc[0]
    assert falling_row["slope"] < 0
    flat_row = ranking[ranking["topic_id"] == 1].iloc[0]
    assert math.isclose(flat_row["slope"], 0.0, abs_tol=1e-9)
    flat_ci_low = float(flat_row["slope_ci_low"])
    flat_ci_high = float(flat_row["slope_ci_high"])
    assert flat_ci_low < 0
    assert flat_ci_high > 0
