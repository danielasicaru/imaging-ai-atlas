import numpy as np
from sentence_transformers import SentenceTransformer


class SentenceTransformerModel:
    """Thin adapter around sentence_transformers.SentenceTransformer that exposes exactly
    the embedder.EmbeddingModel protocol - SentenceTransformer.encode's own signature is a
    large overload set, which structurally can't match a narrow Protocol.
    """

    def __init__(self, name: str, revision: str | None, trust_remote_code: bool) -> None:
        self._model = SentenceTransformer(
            name, revision=revision, trust_remote_code=trust_remote_code
        )

    def encode(self, texts: list[str], batch_size: int) -> np.ndarray:
        return self._model.encode(texts, batch_size=batch_size, convert_to_numpy=True)
