from itertools import combinations

from sklearn.metrics import adjusted_rand_score


def mean_pairwise_ari(assignments: list[list[int]]) -> float:
    """Mean Adjusted Rand Index across every pair of document-topic assignments from
    different seeds. Compares assignments directly (not topic word lists), since ARI
    measures whether two partitions of the *same documents* agree, independent of which
    arbitrary integer label each run assigned to a given cluster.
    """
    if len(assignments) < 2:
        raise ValueError("mean_pairwise_ari needs at least two runs to compare")
    scores = [adjusted_rand_score(a, b) for a, b in combinations(assignments, 2)]
    return sum(scores) / len(scores)
