from pathlib import Path

import pandas as pd
import typer

from imaging_ai_atlas.data.filter import apply_filter
from imaging_ai_atlas.embed.config import load_embed_config
from imaging_ai_atlas.embed.pipeline import run_embedding_sweep

app = typer.Typer(help="imaging-ai-atlas: a data story on medical-imaging AI research trends.")


@app.command()
def version() -> None:
    """Print the installed package version."""
    typer.echo("imaging-ai-atlas 0.1.0")


@app.command()
def embed(
    papers_path: Path = Path("data/interim/papers.parquet"),
    config_path: Path = Path("configs/embed.yaml"),
    cache_dir: Path = Path("data/interim/embeddings"),
) -> None:
    """Embed the M2-filtered corpus with every model in `config_path`, caching to
    `cache_dir` and logging one MLflow run per (model, dims) combination.
    """
    df = pd.read_parquet(papers_path)
    filtered = df.loc[apply_filter(df)].reset_index(drop=True)
    typer.echo(f"Embedding {len(filtered)} filtered papers.")

    config = load_embed_config(config_path)
    for model_config in config.models:
        typer.echo(f"Running {model_config.name} (dims={model_config.dims})...")
        run_embedding_sweep(filtered, model_config, cache_dir)


if __name__ == "__main__":
    app()
