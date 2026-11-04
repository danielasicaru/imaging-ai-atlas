import math

import pytest

from imaging_ai_atlas.topics.stability import mean_pairwise_ari


def test_mean_pairwise_ari_is_one_for_identical_assignments() -> None:
    assignments = [[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]]

    assert math.isclose(mean_pairwise_ari(assignments), 1.0)


def test_mean_pairwise_ari_is_low_for_unrelated_assignments() -> None:
    # One run splits into 2 clean clusters, the other assigns every point its own cluster
    # (ARI near 0 for a random/degenerate partition vs. a structured one).
    assignments = [[0, 0, 1, 1, 0, 0, 1, 1], [0, 1, 2, 3, 4, 5, 6, 7]]

    assert mean_pairwise_ari(assignments) < 0.3


def test_mean_pairwise_ari_averages_over_all_seed_pairs() -> None:
    # 3 runs -> 3 pairs (0,1),(0,2),(1,2). Two identical + one totally different should
    # average to somewhere strictly between the two extremes, not collapse to either.
    identical_a = [0, 0, 0, 1, 1, 1]
    identical_b = [0, 0, 0, 1, 1, 1]
    different = [0, 0, 1, 1, 2, 2]

    result = mean_pairwise_ari([identical_a, identical_b, different])

    assert 0.0 < result < 1.0


def test_mean_pairwise_ari_requires_at_least_two_runs() -> None:
    with pytest.raises(ValueError, match="at least two"):
        mean_pairwise_ari([[0, 0, 1, 1]])
