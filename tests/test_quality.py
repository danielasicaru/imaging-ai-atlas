import pandas as pd

from imaging_ai_atlas.data.quality import category_year_counts, compute_quality_report


def _row(
    external_id: str, created: str, abstract: str, categories: list[str] | None = None
) -> dict:
    return {
        "source": "arxiv",
        "external_id": external_id,
        "title": "t",
        "abstract": abstract,
        "categories": categories or ["cs.CV"],
        "created": created,
        "update_date": created,
    }


def test_quality_report_flags_each_issue_independently() -> None:
    df = pd.DataFrame.from_records(
        [
            _row("pre1991", "1988-01-01", "A perfectly normal abstract of reasonable length."),
            _row("short", "2010-01-01", "No."),
            _row("latex", "2015-06-01", r"Uses $x^2$ and \emph{bold} markup in the text."),
            _row("normal", "2020-01-01", "A perfectly normal abstract of reasonable length."),
        ]
    )

    report = compute_quality_report(df)

    assert report.row_count == 4
    assert report.date_min == "1988-01-01"
    assert report.date_max == "2020-01-01"
    assert report.pre_launch_count == 1
    assert report.short_abstract_count == 1
    assert report.latex_leakage_count == 1
    assert report.papers_per_year == {1988: 1, 2010: 1, 2015: 1, 2020: 1}


def test_category_year_counts_isolates_one_category() -> None:
    df = pd.DataFrame.from_records(
        [
            _row("a", "2016-01-01", "x", categories=["eess.IV"]),
            _row("b", "2016-01-01", "x", categories=["cs.CV"]),
            _row("c", "2018-01-01", "x", categories=["eess.IV", "cs.CV"]),
        ]
    )

    counts = category_year_counts(df, "eess.IV")

    assert counts == {2016: 1, 2018: 1}
