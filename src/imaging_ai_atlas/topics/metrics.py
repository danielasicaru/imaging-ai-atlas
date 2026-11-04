import numpy as np
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
from scipy.optimize import linear_sum_assignment


def topic_diversity(topic_words: list[list[str]]) -> float:
    """Fraction of unique words among all topics' keyword lists - low diversity means
    topics are repeating the same words, a sign HDBSCAN found redundant clusters rather
    than genuinely distinct themes.
    """
    all_words = [word for topic in topic_words for word in topic]
    if not all_words:
        return 0.0
    return len(set(all_words)) / len(all_words)


def outlier_rate(topics: list[int]) -> float:
    """Fraction of documents HDBSCAN assigned to no cluster (topic -1)."""
    if not topics:
        return 0.0
    return sum(1 for t in topics if t == -1) / len(topics)


def _jaccard(a: list[str], b: list[str]) -> float:
    set_a, set_b = set(a), set(b)
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def hungarian_keyword_overlap(topics_a: list[list[str]], topics_b: list[list[str]]) -> float:
    """Match topics between two runs (e.g. two random seeds) via the Hungarian algorithm on
    Jaccard keyword similarity, then average the overlap of matched pairs. Topic *labels*
    aren't stable across HDBSCAN runs - topic 3 in one run isn't "the same" cluster as
    topic 3 in another - so topics must be matched by content before comparing.
    """
    cost = np.array([[-_jaccard(a, b) for b in topics_b] for a in topics_a])
    row_idx, col_idx = linear_sum_assignment(cost)
    overlaps = [-cost[r, c] for r, c in zip(row_idx, col_idx, strict=True)]
    return float(np.mean(overlaps)) if overlaps else 0.0


def npmi_coherence(topic_words: list[list[str]], tokenized_docs: list[list[str]]) -> float:
    """Normalized Pointwise Mutual Information coherence (gensim's c_npmi) for a set of
    topics against the corpus they were fit on. Correlates only weakly with human judgment
    of topic quality (Lau et al. findings consistently cited in the topic-modeling
    literature) - a sanity signal, not a substitute for the human label check in M4.
    """
    dictionary = Dictionary(tokenized_docs)
    coherence_model = CoherenceModel(
        topics=topic_words, texts=tokenized_docs, dictionary=dictionary, coherence="c_npmi"
    )
    return float(coherence_model.get_coherence())
