import typer

app = typer.Typer(help="imaging-ai-atlas: a data story on medical-imaging AI research trends.")


@app.command()
def version() -> None:
    """Print the installed package version."""
    typer.echo("imaging-ai-atlas 0.1.0")


if __name__ == "__main__":
    app()
