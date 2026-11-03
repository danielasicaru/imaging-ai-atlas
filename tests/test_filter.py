import pandas as pd

from imaging_ai_atlas.data.filter import apply_filter, matches_filter


def test_matches_filter_requires_both_category_and_keyword() -> None:
    # in scope + keyword -> match
    assert matches_filter(["eess.IV"], "MRI segmentation", "We study MRI scans.")
    # in scope, no keyword -> no match (generic computer vision, not medical)
    assert not matches_filter(["cs.CV"], "Object detection", "We study car detection.")
    # keyword, wrong category -> no match (recall risk, measured later against the excluded pool)
    assert not matches_filter(["stat.ML"], "A statistical model", "We analyze MRI datasets.")
    # neither -> no match
    assert not matches_filter(["math.CO"], "Graph theory", "We study combinatorics.")


def test_apply_filter_matches_row_by_row_logic() -> None:
    df = pd.DataFrame.from_records(
        [
            {"categories": ["eess.IV"], "title": "MRI reconstruction", "abstract": "scans"},
            {"categories": ["cs.CV"], "title": "Car detection", "abstract": "vision"},
            {"categories": ["stat.ML"], "title": "x", "abstract": "MRI"},
        ]
    )

    mask = apply_filter(df)

    assert list(mask) == [True, False, False]
