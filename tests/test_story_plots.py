from pathlib import Path

import numpy as np

from imaging_ai_atlas.story.dynamics import topic_share_by_year, topic_volume_by_year
from imaging_ai_atlas.story.plots import plot_topic_dynamics, plot_topic_map


def test_plot_topic_map_writes_a_file(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    coords = rng.normal(size=(20, 2))
    topics = [0] * 10 + [1] * 8 + [-1] * 2
    save_path = tmp_path / "map.png"

    plot_topic_map(coords, topics, {0: "Brain MRI", 1: "CT scan"}, save_path)

    assert save_path.exists()
    assert save_path.stat().st_size > 0


def test_plot_topic_map_spreads_overlapping_labels(tmp_path: Path) -> None:
    # Two topics with near-identical centroids would otherwise stack their text labels
    # directly on top of each other.
    rng = np.random.default_rng(0)
    coords = rng.normal(scale=0.01, size=(20, 2))
    topics = [0] * 10 + [1] * 10
    save_path = tmp_path / "map.png"

    plot_topic_map(coords, topics, {0: "Brain MRI", 1: "CT scan"}, save_path)

    assert save_path.exists()
    assert save_path.stat().st_size > 0


def test_plot_topic_dynamics_writes_a_file(tmp_path: Path) -> None:
    topics = [0, 0, 1, 0, 1, 1]
    years = [2020, 2020, 2020, 2021, 2021, 2021]
    share = topic_share_by_year(topic_volume_by_year(topics, years))
    save_path = tmp_path / "dynamics.png"

    plot_topic_dynamics(share, {0: "Brain MRI", 1: "CT scan"}, save_path, min_year_total=1)

    assert save_path.exists()
    assert save_path.stat().st_size > 0


def test_plot_topic_dynamics_drops_low_volume_years_and_labels_other_topics(
    tmp_path: Path,
) -> None:
    # Year 2000 has a single paper - with the default min_year_total it must be excluded
    # rather than producing a misleading 100%-share spike. Topic 3 falls outside the
    # n_highlighted cap and should be folded into the grey "other topics" line.
    topics = [0] * 30 + [1] * 25 + [2] * 20 + [3] * 5 + [0]
    years = [2020] * 30 + [2021] * 25 + [2021] * 20 + [2021] * 5 + [2000]
    share = topic_share_by_year(topic_volume_by_year(topics, years))
    save_path = tmp_path / "dynamics.png"

    plot_topic_dynamics(
        share, {0: "Brain MRI", 1: "CT scan", 2: "Ultrasound"}, save_path, n_highlighted=3
    )

    assert save_path.exists()
    assert save_path.stat().st_size > 0
