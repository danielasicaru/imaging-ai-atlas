import math

from pydantic import BaseModel

# 95% two-sided normal quantile (z such that P(-z < Z < z) = 0.95 for Z ~ N(0,1)).
_Z_95 = 1.959963984540054


def wilson_interval(successes: int, n: int) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion. Preferred over the normal
    (Wald) approximation here because the labeled sample is small (~100-200) and the
    true rate can sit near 0 or 1, where Wald coverage breaks down.
    """
    if n == 0:
        return (0.0, 0.0)
    phat = successes / n
    z = _Z_95
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    half_width = (z * math.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, center - half_width), min(1.0, center + half_width))


class PrecisionRecallEstimate(BaseModel):
    precision: float
    precision_ci: tuple[float, float]
    rejected_relevant_rate: float
    estimated_recall: float


def estimate_precision_recall(
    n_accepted: int,
    n_rejected: int,
    accepted_sample_size: int,
    accepted_relevant: int,
    rejected_sample_size: int,
    rejected_relevant: int,
) -> PrecisionRecallEstimate:
    """Precision from the accepted-stratum sample directly; recall from combining both
    strata, since the filter's true positives and false negatives live in different pools.

    Scaling each stratum's labeled relevant-rate up by its full size gives an estimated
    count of relevant papers on each side of the filter boundary:
      estimated true positives  = n_accepted * precision
      estimated false negatives = n_rejected * rejected_relevant_rate
      recall = true positives / (true positives + false negatives)

    No CI on recall: it's a ratio of two scaled, independently-sampled proportions, and a
    correct interval needs the delta method or a bootstrap over both strata — deferred
    until recall is actually load-bearing for a go/no-go decision, not estimated here.
    """
    precision = accepted_relevant / accepted_sample_size
    precision_ci = wilson_interval(accepted_relevant, accepted_sample_size)
    rejected_relevant_rate = (
        rejected_relevant / rejected_sample_size if rejected_sample_size else 0.0
    )

    estimated_true_positives = n_accepted * precision
    estimated_false_negatives = n_rejected * rejected_relevant_rate
    denominator = estimated_true_positives + estimated_false_negatives
    estimated_recall = estimated_true_positives / denominator if denominator else 0.0

    return PrecisionRecallEstimate(
        precision=precision,
        precision_ci=precision_ci,
        rejected_relevant_rate=rejected_relevant_rate,
        estimated_recall=estimated_recall,
    )
