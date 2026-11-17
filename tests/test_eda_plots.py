from pathlib import Path

import pandas as pd

from imaging_ai_atlas.story.eda import category_share_by_year, largest_topics, papers_per_year
from imaging_ai_atlas.story.eda_plots import (
    build_eda_figures,
    plot_category_share,
    plot_corpus_funnel,
    plot_largest_topics,
    plot_papers_per_year,
)


def test_plot_papers_per_year_writes_a_file(tmp_path: Path) -> None:
    counts = papers_per_year(pd.Series([2018, 2019, 2019]))
    save_path = tmp_path / "growth.png"

    plot_papers_per_year(counts, save_path, last_year_partial=True, title="Papers per year")

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


def test_build_eda_figures_writes_all_four_charts(tmp_path: Path) -> None:
    # Shared by `atlas story` and the EDA notebook, so the charts have one producer.
    filtered = pd.DataFrame(
        {"categories": [["cs.CV"], ["cs.CV", "eess.IV"], ["eess.IV"], ["cs.LG"]]}
    )

    build_eda_figures(
        filtered,
        years=[2019, 2020, 2020, 2021],
        corpus_label="Spine AI",
        raw_count=100,
        topics=[0, 0, 1, -1],
        topic_labels={0: "Brain MRI"},
        figures_dir=tmp_path,
        min_year_total=1,
    )

    written = sorted(p.name for p in tmp_path.glob("*.png"))
    assert written == [
        "category_share.png",
        "corpus_funnel.png",
        "largest_themes.png",
        "papers_per_year.png",
    ]


def test_build_eda_figures_skips_the_category_chart_when_no_year_is_big_enough(
    tmp_path: Path,
) -> None:
    # A small corpus can have no year with min_year_total papers; the other charts must
    # still be written rather than the whole run failing on an empty share table.
    build_eda_figures(
        pd.DataFrame({"categories": [["cs.CV"], ["eess.IV"]]}),
        years=[2020, 2021],
        corpus_label="Spine AI",
        raw_count=10,
        topics=[0, 0],
        topic_labels={},
        figures_dir=tmp_path,
        min_year_total=20,
    )

    written = sorted(p.name for p in tmp_path.glob("*.png"))
    assert written == ["corpus_funnel.png", "largest_themes.png", "papers_per_year.png"]
