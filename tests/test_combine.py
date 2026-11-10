from pathlib import Path

import pandas as pd

from imaging_ai_atlas.data.combine import combine_corpora


def _record(source: str, external_id: str) -> dict:
    return {
        "source": source,
        "external_id": external_id,
        "title": "t",
        "abstract": "a",
        "categories": ["eess.IV"],
        "created": "2020-01-01",
        "update_date": None,
    }


def test_combine_corpora_concatenates_and_dedupes(tmp_path: Path) -> None:
    arxiv_path = tmp_path / "arxiv.parquet"
    pubmed_path = tmp_path / "pubmed.parquet"
    pd.DataFrame.from_records([_record("arxiv", "1"), _record("arxiv", "2")]).to_parquet(arxiv_path)
    pd.DataFrame.from_records(
        [_record("pubmed", "100"), _record("arxiv", "2")]  # duplicate of arxiv/2
    ).to_parquet(pubmed_path)

    out_path = tmp_path / "combined.parquet"
    count = combine_corpora([arxiv_path, pubmed_path], out_path)

    assert count == 3
    combined = pd.read_parquet(out_path)
    keys = set(zip(combined["source"], combined["external_id"], strict=True))
    assert keys == {("arxiv", "1"), ("arxiv", "2"), ("pubmed", "100")}
