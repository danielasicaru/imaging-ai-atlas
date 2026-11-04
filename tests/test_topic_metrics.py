import math

from imaging_ai_atlas.topics.metrics import (
    hungarian_keyword_overlap,
    npmi_coherence,
    outlier_rate,
    topic_diversity,
)


def test_topic_diversity_counts_unique_over_total_words() -> None:
    topics = [["a", "b", "c"], ["a", "d", "e"]]

    assert math.isclose(topic_diversity(topics), 5 / 6)


def test_topic_diversity_is_one_when_all_words_unique() -> None:
    topics = [["a", "b"], ["c", "d"]]

    assert topic_diversity(topics) == 1.0


def test_topic_diversity_handles_no_topics() -> None:
    assert topic_diversity([]) == 0.0


def test_outlier_rate_counts_minus_one_fraction() -> None:
    assert math.isclose(outlier_rate([-1, -1, 0, 1, 0]), 0.4)


def test_outlier_rate_is_zero_with_no_outliers() -> None:
    assert outlier_rate([0, 1, 2]) == 0.0


def test_hungarian_keyword_overlap_matches_best_pairs() -> None:
    topics_a = [["mri", "scan", "brain"], ["x", "y", "z"]]
    topics_b = [["x", "y", "q"], ["mri", "scan", "liver"]]

    # a[0] best matches b[1] (jaccard 2/4), a[1] best matches b[0] (jaccard 2/4) -> mean 0.5
    assert math.isclose(hungarian_keyword_overlap(topics_a, topics_b), 0.5)


def test_hungarian_keyword_overlap_is_one_for_identical_topic_sets() -> None:
    topics = [["a", "b"], ["c", "d"]]

    assert hungarian_keyword_overlap(topics, topics) == 1.0


def test_npmi_coherence_returns_a_finite_score_in_range() -> None:
    docs = [
        ["mri", "scan", "brain", "tumor"],
        ["mri", "brain", "segmentation", "deep"],
        ["ct", "scan", "lung", "nodule"],
        ["ct", "lung", "segmentation", "deep"],
    ]
    topics = [["mri", "brain"], ["ct", "lung"]]

    score = npmi_coherence(topics, docs)

    assert math.isfinite(score)
    assert -1.01 <= score <= 1.01
