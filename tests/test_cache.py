import numpy as np

from imaging_ai_atlas.embed.cache import cache_key, load_cached, save_cached


def test_cache_key_is_stable_for_the_same_inputs() -> None:
    key1 = cache_key("minilm", "abc123", "", 256, "hash1", None)
    key2 = cache_key("minilm", "abc123", "", 256, "hash1", None)

    assert key1 == key2


def test_cache_key_differs_when_any_input_differs() -> None:
    base = cache_key("minilm", "abc123", "", 256, "hash1", None)

    assert base != cache_key("specter", "abc123", "", 256, "hash1", None)
    assert base != cache_key("minilm", "def456", "", 256, "hash1", None)
    assert base != cache_key("minilm", "abc123", "clustering: ", 256, "hash1", None)
    assert base != cache_key("minilm", "abc123", "", 512, "hash1", None)
    assert base != cache_key("minilm", "abc123", "", 256, "hash2", None)
    assert base != cache_key("minilm", "abc123", "", 256, "hash1", 128)


def test_load_cached_returns_none_on_miss(tmp_path) -> None:
    assert load_cached("missing-key", tmp_path) is None


def test_save_then_load_cached_round_trips(tmp_path) -> None:
    embeddings = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    save_cached("my-key", embeddings, tmp_path)
    loaded = load_cached("my-key", tmp_path)

    np.testing.assert_array_equal(loaded, embeddings)
