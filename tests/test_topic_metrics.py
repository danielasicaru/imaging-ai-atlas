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


def test_hungarian_keyword_overlap_drops_unmatched_topics_in_the_longer_run() -> None:
    # topics_a has an extra topic ("x","y") with no counterpart in topics_b. It should be
    # silently excluded from the average, not penalize the score as a zero-overlap pair -
    # this is the documented caveat, pinned down so a future change can't quietly "fix" it
    # into penalizing unmatched topics without a deliberate decision.
    topics_a = [["mri", "scan"], ["ct", "lung"], ["x", "y"]]
    topics_b = [["mri", "scan"], ["ct", "lung"]]

    assert hungarian_keyword_overlap(topics_a, topics_b) == 1.0


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


def test_npmi_coherence_returns_nan_rather_than_crashing_on_degenerate_input() -> None:
    # A topic word that never co-occurs with any other word in the (tiny, single-word)
    # corpus - gensim itself returns NaN here via a mean-of-empty-slice, not an exception,
    # but the function must not propagate that as a crash either way.
    score = npmi_coherence(topic_words=[["a", "b"]], tokenized_docs=[["a"]])

    assert math.isnan(score)
