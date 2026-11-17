import pandas as pd

from imaging_ai_atlas.story.eda import category_share_by_year, largest_topics, papers_per_year


def test_papers_per_year_counts_and_fills_empty_years_with_zero() -> None:
    # A year with no papers must still appear (as 0) - otherwise a bar chart silently
    # closes the gap and the growth curve looks smoother than the data is.
    counts = papers_per_year(pd.Series([2018, 2018, 2020]))

    assert counts.to_dict() == {2018: 2, 2019: 0, 2020: 1}


def test_category_share_by_year_counts_multi_tagged_papers_once_per_category() -> None:
    # Papers carry several categories, so each share is "fraction of that year's papers
    # tagged X" - shares across categories can sum past 1, and that's correct.
    years = [2020, 2020, 2021, 2021]
    categories = [["cs.CV", "eess.IV"], ["cs.CV"], ["eess.IV"], ["cs.LG"]]

    share = category_share_by_year(years, categories, top_n=2, min_year_total=1)

    assert share.to_dict() == {
        "cs.CV": {2020: 1.0, 2021: 0.0},
        "eess.IV": {2020: 0.5, 2021: 0.5},
    }


def test_category_share_by_year_drops_years_below_min_total() -> None:
    # One paper in a year makes every share 0% or 100% - noise, not a trend.
    years = [2010, 2020, 2020]
    categories = [["cs.CV"], ["cs.CV"], ["eess.IV"]]

    share = category_share_by_year(years, categories, top_n=2, min_year_total=2)

    assert share.index.tolist() == [2020]


def test_largest_topics_ranks_by_size_and_skips_outliers() -> None:
    # -1 is HDBSCAN's "no topic" bucket - often the biggest group, but not a theme.
    topics = [-1] * 5 + [0] * 3 + [1] * 1 + [2] * 2

    ranked = largest_topics(topics, {0: "Brain MRI", 2: "Retinal imaging"}, n=2)

    assert ranked["label"].tolist() == ["Brain MRI", "Retinal imaging"]
    assert ranked["size"].tolist() == [3, 2]


def test_largest_topics_falls_back_to_topic_id_without_a_label() -> None:
    ranked = largest_topics([7, 7], {}, n=5)

    assert ranked["label"].tolist() == ["Topic 7"]
