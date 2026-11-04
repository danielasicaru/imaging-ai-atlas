from typing import Protocol

import numpy as np


class EmbeddingModel(Protocol):
    """Anything that can turn texts into embedding vectors - satisfied by
    sentence_transformers.SentenceTransformer, and by a fake in tests so embed_texts'
    prefixing/truncation logic doesn't need a real model download to verify.
    """

    def encode(self, texts: list[str], batch_size: int) -> np.ndarray: ...


def truncate_and_normalize(embeddings: np.ndarray, dims: int | None) -> np.ndarray:
    """Matryoshka truncation: slice to the first `dims` dimensions, then re-normalize.
    A Matryoshka-trained model's dimensions are ordered by importance, so a prefix slice
    is meaningful - but slicing a unit vector changes its norm, and cosine-similarity-based
    downstream steps (UMAP, HDBSCAN) need unit vectors, not whatever norm survives the cut.
    """
    if dims is None:
        return embeddings
    truncated = embeddings[:, :dims]
    norms = np.linalg.norm(truncated, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError(
            f"{int(np.sum(norms == 0))} row(s) have a zero-norm vector after truncating to "
            f"{dims} dims - re-normalizing would produce NaN/inf, which UMAP/HDBSCAN would "
            "silently accept and corrupt downstream, so this fails loudly instead."
        )
    return truncated / norms


def embed_texts(
    model: EmbeddingModel,
    texts: list[str],
    task_prefix: str,
    batch_size: int,
    dims: int | None,
) -> np.ndarray:
    """Embed `texts`, prefixed per the model's required task instruction (e.g. nomic-embed's
    "clustering: "; empty string for models that don't need one), truncated/re-normalized
    to `dims` if given.
    """
    prefixed = [task_prefix + text for text in texts]
    embeddings = model.encode(prefixed, batch_size=batch_size)
    return truncate_and_normalize(embeddings, dims)
