from pathlib import Path

import pandas as pd
import typer

from imaging_ai_atlas.data.filter import apply_filter
from imaging_ai_atlas.embed.config import load_embed_config
from imaging_ai_atlas.embed.pipeline import corpus_hash, run_embedding_sweep
from imaging_ai_atlas.topics.config import load_topic_sweep_config
from imaging_ai_atlas.topics.pipeline import evaluate_variant, load_variant_embeddings

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
    if not papers_path.exists():
        typer.echo(f"No papers file at {papers_path} - run the ingest stage first.", err=True)
        raise typer.Exit(1)
    if not config_path.exists():
        typer.echo(f"No embed config at {config_path}.", err=True)
        raise typer.Exit(1)

    df = pd.read_parquet(papers_path)
    filtered = df.loc[apply_filter(df)].reset_index(drop=True)
    typer.echo(f"Embedding {len(filtered)} filtered papers.")

    config = load_embed_config(config_path)
    for model_config in config.models:
        typer.echo(f"Running {model_config.name} (dims={model_config.dims})...")
        run_embedding_sweep(filtered, model_config, cache_dir)


@app.command()
def topics(
    papers_path: Path = Path("data/interim/papers.parquet"),
    config_path: Path = Path("configs/topics.yaml"),
    embeddings_dir: Path = Path("data/interim/embeddings"),
) -> None:
    """Fit + evaluate BERTopic on every cached embedding variant in `config_path`
    (diversity, NPMI, outlier rate before/after, 5-seed ARI stability, runtime), logging
    one MLflow run per variant. Requires `atlas embed` to have been run first.
    """
    if not papers_path.exists():
        typer.echo(f"No papers file at {papers_path} - run the ingest stage first.", err=True)
        raise typer.Exit(1)
    if not config_path.exists():
        typer.echo(f"No topics config at {config_path}.", err=True)
        raise typer.Exit(1)

    df = pd.read_parquet(papers_path)
    filtered = df.loc[apply_filter(df)].reset_index(drop=True)
    data_hash = corpus_hash(filtered)
    typer.echo(f"Evaluating topic models on {len(filtered)} filtered papers.")

    config = load_topic_sweep_config(config_path)
    for variant in config.embedding_variants:
        typer.echo(f"Running {variant.model_name} (dims={variant.dims})...")
        embeddings = load_variant_embeddings(variant, data_hash, embeddings_dir)
        evaluate_variant(filtered, embeddings, variant, config.topic_model)


if __name__ == "__main__":
    app()
