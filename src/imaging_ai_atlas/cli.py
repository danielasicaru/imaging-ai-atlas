import shutil
import subprocess
from pathlib import Path

import pandas as pd
import typer

from imaging_ai_atlas.data.clean_text import strip_latex
from imaging_ai_atlas.data.filter import apply_filter
from imaging_ai_atlas.embed.config import load_embed_config
from imaging_ai_atlas.embed.pipeline import corpus_hash, run_embedding_sweep
from imaging_ai_atlas.story.dynamics import topic_share_by_year, topic_volume_by_year
from imaging_ai_atlas.story.labels import load_topic_labels
from imaging_ai_atlas.story.map2d import project_2d
from imaging_ai_atlas.story.plots import plot_topic_dynamics, plot_topic_map
from imaging_ai_atlas.topics.config import (
    EmbeddingVariant,
    TopicModelConfig,
    load_topic_sweep_config,
)
from imaging_ai_atlas.topics.labeling import build_topic_label_sample
from imaging_ai_atlas.topics.model import fit_topic_model
from imaging_ai_atlas.topics.pipeline import evaluate_variant, load_variant_embeddings
from imaging_ai_atlas.topics.registry import register_topic_model, save_topic_assignments

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


@app.command()
def register_topic(
    papers_path: Path = Path("data/interim/papers.parquet"),
    embeddings_dir: Path = Path("data/interim/embeddings"),
    model_dir: Path = Path("data/interim/topic_model"),
    labeling_sample_path: Path = Path("data/interim/topic_labeling_sample.csv"),
    assignments_path: Path = Path("data/interim/topic_assignments.parquet"),
    model_name: str = "nomic-ai/nomic-embed-text-v1.5",
    revision: str = "e9b6763023c676ca8431644204f50c2b100d9aab",
    task_prefix: str = "clustering: ",
    max_len: int = 256,
    dims: int = 64,
    n_label_topics: int = 30,
) -> None:
    """Fit the chosen primary topic model (nomic@64 per ADR 0004) once at seed 0, register
    it as a standard MLflow artifact, and write a human coherence-check sample for the
    `n_label_topics` largest topics. Requires `atlas embed` to have cached this variant.
    """
    if not papers_path.exists():
        typer.echo(f"No papers file at {papers_path} - run the ingest stage first.", err=True)
        raise typer.Exit(1)

    df = pd.read_parquet(papers_path)
    filtered = df.loc[apply_filter(df)].reset_index(drop=True)
    data_hash = corpus_hash(filtered)

    variant = EmbeddingVariant(
        model_name=model_name,
        revision=revision,
        task_prefix=task_prefix,
        max_len=max_len,
        dims=dims,
    )
    embeddings = load_variant_embeddings(variant, data_hash, embeddings_dir)
    texts = [
        strip_latex(f"{title} {abstract}")
        for title, abstract in zip(filtered["title"], filtered["abstract"], strict=True)
    ]

    config = TopicModelConfig()
    typer.echo(f"Fitting final topic model on {len(texts)} docs...")
    topic_model, topics = fit_topic_model(embeddings, texts, config.seeds[0], config)

    sample = build_topic_label_sample(topic_model, topics, texts, n_label_topics, config.seeds[0])
    labeling_sample_path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(labeling_sample_path, index=False)
    typer.echo(f"Wrote labeling sample ({len(sample)} topics) to {labeling_sample_path}")

    git_executable = shutil.which("git")
    if git_executable is None:
        typer.echo("git executable not found on PATH.", err=True)
        raise typer.Exit(1)
    git_sha = subprocess.run(  # noqa: S603 - fixed argv, no user input
        [git_executable, "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    register_topic_model(topic_model, model_dir, data_hash, model_name, git_sha)
    typer.echo(f"Registered topic model to MLflow (artifacts under {model_dir}).")

    assignments_path.parent.mkdir(parents=True, exist_ok=True)
    save_topic_assignments(filtered["external_id"].tolist(), topics, assignments_path)
    typer.echo(f"Wrote per-document topic assignments to {assignments_path}")


@app.command()
def story(
    papers_path: Path = Path("data/interim/papers.parquet"),
    embeddings_dir: Path = Path("data/interim/embeddings"),
    assignments_path: Path = Path("data/interim/topic_assignments.parquet"),
    labeling_sample_path: Path = Path("data/interim/topic_labeling_sample.csv"),
    figures_dir: Path = Path("reports/figures"),
    model_name: str = "nomic-ai/nomic-embed-text-v1.5",
    revision: str = "e9b6763023c676ca8431644204f50c2b100d9aab",
    task_prefix: str = "clustering: ",
    max_len: int = 256,
    dims: int = 64,
    seed: int = 0,
) -> None:
    """Build the M5 2D topic map and topic-dynamics-over-time charts from the registered
    nomic@64 topic model's saved assignments. Requires `atlas register-topic` to have run
    first.
    """
    if not papers_path.exists():
        typer.echo(f"No papers file at {papers_path} - run the ingest stage first.", err=True)
        raise typer.Exit(1)
    if not assignments_path.exists():
        typer.echo(
            f"No topic assignments at {assignments_path} - run `atlas register-topic` first.",
            err=True,
        )
        raise typer.Exit(1)

    df = pd.read_parquet(papers_path)
    filtered = df.loc[apply_filter(df)].reset_index(drop=True)
    data_hash = corpus_hash(filtered)

    assignments = pd.read_parquet(assignments_path)
    if assignments["external_id"].tolist() != filtered["external_id"].tolist():
        typer.echo(
            "Topic assignments don't align with the current filtered corpus - "
            "rerun `atlas register-topic`.",
            err=True,
        )
        raise typer.Exit(1)
    topics = assignments["topic_id"].tolist()

    variant = EmbeddingVariant(
        model_name=model_name,
        revision=revision,
        task_prefix=task_prefix,
        max_len=max_len,
        dims=dims,
    )
    embeddings = load_variant_embeddings(variant, data_hash, embeddings_dir)

    typer.echo("Projecting to 2D (UMAP)...")
    coords = project_2d(embeddings, seed)

    years = pd.to_datetime(filtered["created"]).dt.year.tolist()
    volume = topic_volume_by_year(topics, years)
    share = topic_share_by_year(volume)

    topic_labels = load_topic_labels(labeling_sample_path) if labeling_sample_path.exists() else {}

    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_topic_map(coords, topics, topic_labels, figures_dir / "topic_map.png")
    plot_topic_dynamics(share, topic_labels, figures_dir / "topic_dynamics.png")
    typer.echo(f"Wrote figures to {figures_dir}")


if __name__ == "__main__":
    app()
