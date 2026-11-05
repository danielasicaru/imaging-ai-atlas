import re

import pandas as pd
import pytest

from imaging_ai_atlas.data.filter import (
    FilterConfig,
    apply_filter,
    load_filter_config,
    matches_filter,
)

_MEDICAL_IMAGING_CONFIG = FilterConfig(
    name="medical-imaging",
    category_scope=["eess.IV", "cs.CV", "stat.ML"],
    keyword_patterns=[r"MRI"],
)


def test_matches_filter_requires_both_category_and_keyword() -> None:
    # in scope + keyword -> match
    assert matches_filter(
        ["eess.IV"], "MRI segmentation", "We study MRI scans.", _MEDICAL_IMAGING_CONFIG
    )
    # in scope, no keyword -> no match
    assert not matches_filter(
        ["cs.CV"], "Object detection", "We study car detection.", _MEDICAL_IMAGING_CONFIG
    )
    # keyword, wrong category -> no match
    assert not matches_filter(
        ["math.CO"], "A statistical model", "We analyze MRI datasets.", _MEDICAL_IMAGING_CONFIG
    )
    # neither -> no match
    assert not matches_filter(
        ["math.CO"], "Graph theory", "We study combinatorics.", _MEDICAL_IMAGING_CONFIG
    )


def test_apply_filter_matches_row_by_row_logic() -> None:
    df = pd.DataFrame.from_records(
        [
            {"categories": ["eess.IV"], "title": "MRI reconstruction", "abstract": "scans"},
            {"categories": ["cs.CV"], "title": "Car detection", "abstract": "vision"},
            {"categories": ["math.CO"], "title": "x", "abstract": "MRI"},
        ]
    )

    mask = apply_filter(df, _MEDICAL_IMAGING_CONFIG)

    assert list(mask) == [True, False, False]


def test_a_different_topic_domain_config_changes_what_matches() -> None:
    # The whole point of making this a config: a cervical-spine topic domain accepts a
    # paper the medical-imaging config would reject (no MRI/CT/etc. keyword) and vice
    # versa, with zero code changes - just a different FilterConfig.
    spine_config = FilterConfig(
        name="cervical-spine-fracture",
        category_scope=["eess.IV", "cs.CV"],
        keyword_patterns=[r"cervical spine fracture", r"AO Spine classification"],
    )
    df = pd.DataFrame.from_records(
        [
            {
                "categories": ["eess.IV"],
                "title": "Cervical spine fracture detection",
                "abstract": "using the AO Spine classification system",
            },
            {"categories": ["eess.IV"], "title": "MRI reconstruction", "abstract": "scans"},
        ]
    )

    assert list(apply_filter(df, spine_config)) == [True, False]
    assert list(apply_filter(df, _MEDICAL_IMAGING_CONFIG)) == [False, True]


def test_load_filter_config_round_trips_through_yaml(tmp_path) -> None:
    path = tmp_path / "filter.yaml"
    path.write_text(
        "name: medical-imaging\n"
        "category_scope: [eess.IV, cs.CV]\n"
        "keyword_patterns: ['MRI', 'CT scan']\n",
        encoding="utf-8",
    )

    config = load_filter_config(path)

    assert config.name == "medical-imaging"
    assert config.category_scope == ["eess.IV", "cs.CV"]
    assert config.keyword_patterns == ["MRI", "CT scan"]


def test_filter_config_rejects_a_malformed_regex_pattern() -> None:
    with pytest.raises(re.error):
        FilterConfig(name="broken", category_scope=["eess.IV"], keyword_patterns=["(unclosed"])
