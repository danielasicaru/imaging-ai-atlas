import math

from imaging_ai_atlas.evaluate.precision_recall import estimate_precision_recall, wilson_interval


def test_wilson_interval_matches_known_value() -> None:
    # Hand-derived check: phat=0.2, z=1.959964, n=100 -> Wilson interval ~= (0.133, 0.289).
    low, high = wilson_interval(20, 100)

    assert math.isclose(low, 0.133, abs_tol=0.001)
    assert math.isclose(high, 0.289, abs_tol=0.001)


def test_wilson_interval_handles_zero_trials() -> None:
    assert wilson_interval(0, 0) == (0.0, 0.0)


def test_wilson_interval_handles_all_successes() -> None:
    low, high = wilson_interval(10, 10)

    assert low > 0.6
    assert math.isclose(high, 1.0)


def test_estimate_precision_recall_combines_both_strata() -> None:
    # Filter accepted 1000 papers, rejected 4000. Labeled 200 of the accepted (180 truly
    # relevant) and 100 of the rejected (5 truly relevant) -> estimate total relevant and recall.
    result = estimate_precision_recall(
        n_accepted=1000,
        n_rejected=4000,
        accepted_sample_size=200,
        accepted_relevant=180,
        rejected_sample_size=100,
        rejected_relevant=5,
    )

    assert math.isclose(result.precision, 0.9)
    assert result.precision_ci[0] < 0.9 < result.precision_ci[1]

    # estimated true positives = 1000 * 0.9 = 900; estimated false negatives = 4000 * 0.05 = 200
    # recall = 900 / (900 + 200)
    assert math.isclose(result.estimated_recall, 900 / 1100, abs_tol=1e-9)
