import pandas as pd

from imaging_ai_atlas.data.sample import build_labeling_sample


def _row(external_id: str) -> dict:
    return {
        "source": "arxiv",
        "external_id": external_id,
        "title": "t",
        "abstract": "a",
        "categories": ["cs.CV"],
        "created": "2020-01-01",
        "update_date": "2020-01-01",
    }


def test_build_labeling_sample_draws_from_both_strata() -> None:
    df = pd.DataFrame.from_records([_row(str(i)) for i in range(20)])
    mask = pd.Series([i % 2 == 0 for i in range(20)])  # 10 in, 10 out

    sample = build_labeling_sample(df, mask, n_in=4, n_out=3, seed=0)

    assert len(sample) == 7
    assert len(sample[sample["in_filter"]]) == 4
    assert len(sample[~sample["in_filter"]]) == 3
    assert list(sample["label"]) == [""] * 7
    assert sample["external_id"].is_unique
    # cross-check against the mask itself, not just the function's own bookkeeping: every
    # row flagged in_filter=True must actually be one of the even ids the mask selected.
    assert all(int(x) % 2 == 0 for x in sample.loc[sample["in_filter"], "external_id"])
    assert all(int(x) % 2 == 1 for x in sample.loc[~sample["in_filter"], "external_id"])


def test_build_labeling_sample_is_reproducible_for_a_fixed_seed() -> None:
    df = pd.DataFrame.from_records([_row(str(i)) for i in range(20)])
    mask = pd.Series([i % 2 == 0 for i in range(20)])

    first = build_labeling_sample(df, mask, n_in=4, n_out=3, seed=0)
    second = build_labeling_sample(df, mask, n_in=4, n_out=3, seed=0)

    assert list(first["external_id"]) == list(second["external_id"])


def test_build_labeling_sample_caps_at_available_rows() -> None:
    df = pd.DataFrame.from_records([_row(str(i)) for i in range(3)])
    mask = pd.Series([True, True, False])

    sample = build_labeling_sample(df, mask, n_in=10, n_out=10, seed=0)

    assert len(sample) == 3


def test_build_labeling_sample_aligns_by_index_not_position() -> None:
    # df's index is NOT 0..n-1 in row order; mask is keyed by the same labels but given in a
    # different order. A positional (.to_numpy()) mask would silently pick the wrong rows.
    labels = [3, 1, 2, 0]
    df = pd.DataFrame.from_records([_row(str(label)) for label in labels], index=labels)
    mask = pd.Series({3: False, 1: True, 2: False, 0: True})

    sample = build_labeling_sample(df, mask, n_in=10, n_out=10, seed=0)

    assert set(sample.loc[sample["in_filter"], "external_id"]) == {"1", "0"}
    assert set(sample.loc[~sample["in_filter"], "external_id"]) == {"3", "2"}
