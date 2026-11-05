import pandas as pd
from typer.testing import CliRunner

from imaging_ai_atlas.cli import app

runner = CliRunner()


def test_version_command() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "imaging-ai-atlas" in result.stdout


def test_embed_command_exits_cleanly_when_papers_file_is_missing(tmp_path) -> None:
    result = runner.invoke(app, ["embed", "--papers-path", str(tmp_path / "nonexistent.parquet")])
    assert result.exit_code == 1


def test_embed_command_exits_cleanly_when_config_file_is_missing(tmp_path) -> None:
    papers_path = tmp_path / "papers.parquet"
    papers_path.touch()

    result = runner.invoke(
        app,
        [
            "embed",
            "--papers-path",
            str(papers_path),
            "--config-path",
            str(tmp_path / "nonexistent.yaml"),
        ],
    )
    assert result.exit_code == 1


def test_register_topic_refuses_to_overwrite_an_already_labeled_sample_without_force(
    tmp_path,
) -> None:
    # papers_path must exist so the guard under test - not the earlier missing-file
    # check - is what actually triggers the exit.
    papers_path = tmp_path / "papers.parquet"
    pd.DataFrame(
        {
            "external_id": ["1"],
            "title": ["MRI brain tumor segmentation"],
            "abstract": ["deep learning for medical imaging"],
            "categories": [["eess.IV"]],
            "created": ["2020-01-01"],
        }
    ).to_parquet(papers_path)

    labeling_sample_path = tmp_path / "topic_labeling_sample.csv"
    pd.DataFrame({"topic_id": [0], "coherent": ["yes"], "suggested_label": ["Brain MRI"]}).to_csv(
        labeling_sample_path, index=False
    )

    result = runner.invoke(
        app,
        [
            "register-topic",
            "--papers-path",
            str(papers_path),
            "--labeling-sample-path",
            str(labeling_sample_path),
        ],
    )

    assert result.exit_code == 1


def test_story_command_exits_cleanly_when_assignments_dont_match_filtered_corpus(
    tmp_path,
) -> None:
    papers_path = tmp_path / "papers.parquet"
    pd.DataFrame(
        {
            "external_id": ["1", "2"],
            "title": ["MRI brain tumor segmentation"] * 2,
            "abstract": ["deep learning for medical imaging"] * 2,
            "categories": [["eess.IV"], ["eess.IV"]],
            "created": ["2020-01-01", "2020-01-02"],
        }
    ).to_parquet(papers_path)

    # Deliberately mismatched: a different external_id than what's in papers_path, so the
    # alignment check must catch it rather than silently zipping topics to the wrong docs.
    assignments_path = tmp_path / "assignments.parquet"
    pd.DataFrame({"external_id": ["not-the-same-id"], "topic_id": [0]}).to_parquet(assignments_path)

    result = runner.invoke(
        app,
        [
            "story",
            "--papers-path",
            str(papers_path),
            "--assignments-path",
            str(assignments_path),
        ],
    )

    assert result.exit_code == 1
