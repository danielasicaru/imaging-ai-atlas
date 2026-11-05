import numpy as np

from imaging_ai_atlas.story.map2d import project_2d


def test_project_2d_is_deterministic_for_a_fixed_seed() -> None:
    rng = np.random.default_rng(0)
    embeddings = rng.normal(size=(30, 16)).astype(np.float32)

    first = project_2d(embeddings, seed=0)
    second = project_2d(embeddings, seed=0)

    assert first.shape == (30, 2)
    np.testing.assert_array_equal(first, second)
