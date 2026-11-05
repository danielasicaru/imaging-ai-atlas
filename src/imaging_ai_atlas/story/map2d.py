import numpy as np
from umap import UMAP


def project_2d(embeddings: np.ndarray, seed: int) -> np.ndarray:
    """A separate UMAP run at `n_components=2`, purely for the display map - distinct from
    the 5D UMAP used for clustering in M4 (that one is tuned for HDBSCAN's density
    assumptions, not for human-readable layout; conflating the two would mean every change
    to the clustering config also reshuffles the map, or vice versa). Cosine metric
    matches the clustering UMAP's choice, for a comparable notion of "nearby" in both.
    """
    reducer = UMAP(n_components=2, n_neighbors=15, min_dist=0.0, metric="cosine", random_state=seed)
    return reducer.fit_transform(embeddings)
