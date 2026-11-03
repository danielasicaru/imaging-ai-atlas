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
    assert (sample["in_filter"]).sum() == 4
    assert (~sample["in_filter"]).sum() == 3
    assert list(sample["label"]) == [""] * 7
    assert sample["external_id"].is_unique


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
