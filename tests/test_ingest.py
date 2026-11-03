from datetime import date
from pathlib import Path

import pandas as pd

from imaging_ai_atlas.data.arxiv import ArxivSource
from imaging_ai_atlas.data.ingest import ingest

FIXTURE = Path(__file__).parent / "fixtures" / "mini_arxiv.jsonl"


def test_arxiv_source_parses_date_from_first_version() -> None:
    records = list(ArxivSource().iter_records(FIXTURE))

    first = next(r for r in records if r.external_id == "0704.0001")
    assert first.created == date(2007, 4, 2)
    assert first.update_date == date(2007, 5, 23)

    second = next(r for r in records if r.external_id == "0704.0002")
    # created must come from v1, not the later v2 timestamp or update_date.
    assert second.created == date(2007, 4, 3)


def test_ingest_dedupes_by_source_and_external_id(tmp_path: Path) -> None:
    out_path = tmp_path / "papers.parquet"

    row_count = ingest(ArxivSource(), FIXTURE, out_path)

    assert row_count == 2  # the duplicate 0704.0001 line collapses to one row
    df = pd.read_parquet(out_path)
    assert sorted(df["external_id"]) == ["0704.0001", "0704.0002"]
