from typer.testing import CliRunner

from imaging_ai_atlas.cli import app

runner = CliRunner()


def test_version_command() -> None:
    result = runner.invoke(app, [])
    assert result.exit_code == 0
    assert "imaging-ai-atlas" in result.stdout
