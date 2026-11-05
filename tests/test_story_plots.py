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


def test_plot_topic_dynamics_writes_a_file(tmp_path: Path) -> None:
    topics = [0, 0, 1, 0, 1, 1]
    years = [2020, 2020, 2020, 2021, 2021, 2021]
    share = topic_share_by_year(topic_volume_by_year(topics, years))
    save_path = tmp_path / "dynamics.png"

    plot_topic_dynamics(share, {0: "Brain MRI", 1: "CT scan"}, save_path)

    assert save_path.exists()
    assert save_path.stat().st_size > 0
