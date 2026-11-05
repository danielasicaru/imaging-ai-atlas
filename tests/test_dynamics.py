import math

from imaging_ai_atlas.story.dynamics import topic_share_by_year, topic_volume_by_year


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
