from pathlib import Path

import pandas as pd

from imaging_ai_atlas.story.eda import category_share_by_year, largest_topics, papers_per_year
from imaging_ai_atlas.story.eda_plots import (
    plot_category_share,
    plot_corpus_funnel,
    plot_largest_topics,
    plot_papers_per_year,
)


def test_plot_papers_per_year_writes_a_file(tmp_path: Path) -> None:
    counts = papers_per_year(pd.Series([2018, 2019, 2019]))
    save_path = tmp_path / "growth.png"

    plot_papers_per_year(counts, save_path, last_year_partial=True)

    assert save_path.stat().st_size > 0


def test_plot_corpus_funnel_writes_a_file(tmp_path: Path) -> None:
    save_path = tmp_path / "funnel.png"

    plot_corpus_funnel(
        [("All papers", 1_000_000), ("In scope", 5_000), ("In a theme", 3_000)], save_path
    )

    assert save_path.stat().st_size > 0


def test_plot_category_share_writes_a_file(tmp_path: Path) -> None:
    share = category_share_by_year(
        [2020, 2020, 2021, 2021],
        [["cs.CV"], ["eess.IV"], ["cs.CV"], ["cs.LG"]],
        top_n=3,
        min_year_total=1,
    )
    save_path = tmp_path / "categories.png"

    plot_category_share(share, save_path)

    assert save_path.stat().st_size > 0


def test_plot_largest_topics_writes_a_file(tmp_path: Path) -> None:
    ranked = largest_topics([0, 0, 0, 1, 1, 2], {0: "Brain MRI", 1: "CT scan"}, n=3)
    save_path = tmp_path / "themes.png"

    plot_largest_topics(ranked, save_path)

    assert save_path.stat().st_size > 0
