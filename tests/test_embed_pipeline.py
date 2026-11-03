import pandas as pd

from imaging_ai_atlas.embed.pipeline import corpus_hash


def test_corpus_hash_is_order_independent() -> None:
    df_a = pd.DataFrame({"external_id": ["1", "2", "3"]})
    df_b = pd.DataFrame({"external_id": ["3", "1", "2"]})

    assert corpus_hash(df_a) == corpus_hash(df_b)


def test_corpus_hash_changes_when_the_id_set_changes() -> None:
    df_a = pd.DataFrame({"external_id": ["1", "2", "3"]})
    df_b = pd.DataFrame({"external_id": ["1", "2", "4"]})

    assert corpus_hash(df_a) != corpus_hash(df_b)
