import numpy as np
import pytest

from imaging_ai_atlas.embed.embedder import embed_texts, truncate_and_normalize


class _FakeModel:
    """Returns a fixed, deterministic (but not pre-normalized) embedding per call so tests
    can check what embed_texts actually passed in (prefix, batch_size) and did with the
    output (truncation, re-normalization), without downloading a real model.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], int]] = []

    def encode(self, texts: list[str], batch_size: int) -> np.ndarray:
        self.calls.append((texts, batch_size))
        return np.tile(np.array([3.0, 4.0, 0.0, 0.0]), (len(texts), 1))


def test_truncate_and_normalize_keeps_native_dim_when_none() -> None:
    embeddings = np.array([[3.0, 4.0]])

    result = truncate_and_normalize(embeddings, dims=None)

    assert result.shape == (1, 2)
    np.testing.assert_array_equal(result, embeddings)


def test_truncate_and_normalize_renormalizes_after_truncation() -> None:
    # [3, 4, 0, 0] has norm 5; truncating to the first 1 dim leaves [3], which must be
    # re-normalized to [1.0] (its own norm), not left as 3.0.
    embeddings = np.array([[3.0, 4.0, 0.0, 0.0]])

    result = truncate_and_normalize(embeddings, dims=1)

    np.testing.assert_allclose(result, [[1.0]])


def test_truncate_and_normalize_raises_on_zero_norm_row() -> None:
    # Truncating to the first 2 dims of this vector leaves [0, 0] - a zero-norm row that
    # would silently become NaN/inf on re-normalization if not caught explicitly.
    embeddings = np.array([[0.0, 0.0, 3.0, 4.0]])

    with pytest.raises(ValueError, match="zero-norm"):
        truncate_and_normalize(embeddings, dims=2)


def test_embed_texts_prepends_task_prefix_and_passes_batch_size() -> None:
    model = _FakeModel()

    embed_texts(model, ["scan of lungs"], task_prefix="clustering: ", batch_size=8, dims=None)

    assert model.calls == [(["clustering: scan of lungs"], 8)]


def test_embed_texts_truncates_output_to_requested_dims() -> None:
    model = _FakeModel()

    result = embed_texts(model, ["a"], task_prefix="", batch_size=8, dims=2)

    assert result.shape == (1, 2)
    np.testing.assert_allclose(np.linalg.norm(result, axis=1), [1.0])
