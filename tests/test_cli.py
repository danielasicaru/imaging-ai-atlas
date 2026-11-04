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
