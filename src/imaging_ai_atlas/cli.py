import shutil
import subprocess
from pathlib import Path

import pandas as pd
import typer

from imaging_ai_atlas.data.arxiv import ArxivSource
from imaging_ai_atlas.data.clean_text import strip_latex
from imaging_ai_atlas.data.combine import combine_corpora
from imaging_ai_atlas.data.filter import apply_filter, load_filter_config
from imaging_ai_atlas.data.ingest import ingest as ingest_source
from imaging_ai_atlas.data.pubmed import PubmedSource
from imaging_ai_atlas.data.pubmed_fetch import fetch_pubmed_xml
from imaging_ai_atlas.embed.config import load_embed_config
from imaging_ai_atlas.embed.pipeline import corpus_hash, run_embedding_sweep
from imaging_ai_atlas.story.dynamics import (
    bootstrap_topic_share,
    rank_topics_by_slope,
    topic_share_by_year,
    topic_volume_by_year,
)
from imaging_ai_atlas.story.labels import load_topic_labels
from imaging_ai_atlas.story.map2d import project_2d
from imaging_ai_atlas.story.plots import plot_topic_dynamics, plot_topic_map
from imaging_ai_atlas.topics.config import (
    EmbeddingVariant,
    HdbscanConfig,
    TopicModelConfig,
    UmapConfig,
    VectorizerConfig,
    load_topic_sweep_config,
)
from imaging_ai_atlas.topics.labeling import build_topic_label_sample
from imaging_ai_atlas.topics.model import fit_topic_model
from imaging_ai_atlas.topics.pipeline import evaluate_variant, load_variant_embeddings
from imaging_ai_atlas.topics.registry import register_topic_model, save_topic_assignments

app = typer.Typer(help="imaging-ai-atlas: a data story on a configurable research topic.")

_DEFAULT_FILTER_CONFIG = Path("configs/filter_medical_imaging.yaml")


def _require_exists(path: Path, message: str) -> None:
    """Fail fast with an actionable message when an input file is missing - every command
    opens with a few of these, so the echo-then-Exit(1) pair lives in one place.
    """
    if not path.exists():
        typer.echo(message, err=True)
        raise typer.Exit(1)


def _require_corpus_inputs(papers_path: Path, filter_config_path: Path) -> None:
    """The two inputs every corpus-reading command needs before it can do anything."""
    _require_exists(papers_path, f"No papers file at {papers_path} - run the ingest stage first.")
    _require_exists(filter_config_path, f"No filter config at {filter_config_path}.")


def _load_filtered_corpus(papers_path: Path, filter_config_path: Path) -> pd.DataFrame:
    """Every stage starts from the same filtered corpus - factored out so the filter
    config (which topic this run is about) is loaded and applied identically everywhere,
    rather than four near-identical copies of the same two lines.
    """
    df = pd.read_parquet(papers_path)
    config = load_filter_config(filter_config_path)
    return df.loc[apply_filter(df, config)].reset_index(drop=True)


@app.command()
def version() -> None:
    """Print the installed package version."""
    typer.echo("imaging-ai-atlas 0.1.0")


_SOURCES = {"arxiv": ArxivSource, "pubmed": PubmedSource}


@app.command()
def ingest(
    source: str = typer.Argument(..., help="Which source to ingest: 'arxiv' or 'pubmed'."),
    raw_path: Path = Path("data/raw/snapshot"),
    out_path: Path = Path("data/interim/papers.parquet"),
    force: bool = typer.Option(False, "--force", help="Overwrite out_path if it already exists."),
) -> None:
    """Parse a raw snapshot from `source` into the normalized `PaperRecord` parquet
    schema every later pipeline stage reads.
    """
    if source not in _SOURCES:
        typer.echo(f"Unknown source '{source}' - must be one of {sorted(_SOURCES)}.", err=True)
        raise typer.Exit(1)
    _require_exists(raw_path, f"No raw snapshot at {raw_path}.")
    if out_path.exists() and not force:
        typer.echo(
            f"{out_path} already exists - rerun with --out-path <new path> or --force to "
            "overwrite it (the default path is shared with other corpora's ingest runs).",
            err=True,
        )
        raise typer.Exit(1)

    count = ingest_source(_SOURCES[source](), raw_path, out_path)
    typer.echo(f"Ingested {count} {source} papers to {out_path}")


@app.command()
def fetch_pubmed(
    query: str,
    email: str,
    raw_path: Path = Path("data/raw/pubmed.xml"),
    max_results: int = 2000,
    api_key: str | None = None,
    force: bool = typer.Option(False, "--force", help="Overwrite raw_path if it already exists."),
) -> None:
    """Search PubMed for `query` via NCBI E-utilities and save the raw XML to `raw_path`,
    ready for `atlas ingest pubmed --raw-path <raw_path>`.
    """
    if raw_path.exists() and not force:
        typer.echo(
            f"{raw_path} already exists - rerun with --raw-path <new path> or --force to "
            "overwrite it (refetching re-runs the full NCBI query under its rate limit).",
            err=True,
        )
        raise typer.Exit(1)

    count = fetch_pubmed_xml(query, raw_path, email, max_results, api_key)
    typer.echo(f"Fetched {count} PubMed records to {raw_path}")


@app.command("combine-corpora")
def combine_corpora_command(
    paths: list[Path],
    out_path: Path = Path("data/interim/papers.parquet"),
    force: bool = typer.Option(False, "--force", help="Overwrite out_path if it already exists."),
) -> None:
    """Concatenate multiple sources' `PaperRecord` parquet files into one corpus,
    deduping on (source, external_id).
    """
    for path in paths:
        _require_exists(path, f"No papers file at {path}.")
    if out_path.exists() and not force:
        typer.echo(
            f"{out_path} already exists - rerun with --out-path <new path> or --force to "
            "overwrite it (the default path is shared with other corpora's combine runs).",
            err=True,
        )
        raise typer.Exit(1)
    count = combine_corpora(paths, out_path)
    typer.echo(f"Combined {count} papers into {out_path}")


@app.command()
def embed(
    papers_path: Path = Path("data/interim/papers.parquet"),
    config_path: Path = Path("configs/embed.yaml"),
    filter_config_path: Path = _DEFAULT_FILTER_CONFIG,
    cache_dir: Path = Path("data/interim/embeddings"),
) -> None:
    """Embed the filtered corpus with every model in `config_path`, caching to
    `cache_dir` and logging one MLflow run per (model, dims) combination.
    """
    _require_corpus_inputs(papers_path, filter_config_path)
    _require_exists(config_path, f"No embed config at {config_path}.")

    filtered = _load_filtered_corpus(papers_path, filter_config_path)
    typer.echo(f"Embedding {len(filtered)} filtered papers.")

    config = load_embed_config(config_path)
    for model_config in config.models:
        typer.echo(f"Running {model_config.name} (dims={model_config.dims})...")
        run_embedding_sweep(filtered, model_config, cache_dir)


@app.command()
def topics(
    papers_path: Path = Path("data/interim/papers.parquet"),
    config_path: Path = Path("configs/topics.yaml"),
    filter_config_path: Path = _DEFAULT_FILTER_CONFIG,
    embeddings_dir: Path = Path("data/interim/embeddings"),
) -> None:
    """Fit + evaluate BERTopic on every cached embedding variant in `config_path`
    (diversity, NPMI, outlier rate before/after, 5-seed ARI stability, runtime), logging
    one MLflow run per variant. Requires `atlas embed` to have been run first.
    """
    _require_corpus_inputs(papers_path, filter_config_path)
    _require_exists(config_path, f"No topics config at {config_path}.")

    filtered = _load_filtered_corpus(papers_path, filter_config_path)
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
    filter_config_path: Path = _DEFAULT_FILTER_CONFIG,
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
    min_cluster_size: int = HdbscanConfig().min_cluster_size,
    min_df: int = VectorizerConfig().min_df,
    umap_n_neighbors: int = UmapConfig().n_neighbors,
    force: bool = typer.Option(
        False, "--force", help="Overwrite an existing, human-labeled labeling sample CSV."
    ),
) -> None:
    """Fit the chosen primary topic model (nomic@64 per ADR 0004) once at seed 0, register
    it as a standard MLflow artifact, and write a human coherence-check sample for the
    `n_label_topics` largest topics. Requires `atlas embed` to have cached this variant.

    `min_cluster_size`/`min_df`/`umap_n_neighbors` default to the values tuned for the full
    medical-imaging corpus (ADR 0004) - a much smaller filtered corpus (e.g. a narrow
    topic-domain config) needs smaller values: HDBSCAN can't form clusters bigger than the
    corpus, CountVectorizer's `min_df` can't exceed the smallest cluster's document count,
    and UMAP's `n_neighbors` relative to corpus size controls how local vs. global its
    neighborhood graph is - at `n_neighbors=15` on a few hundred documents or fewer, a
    meaningful fraction of the whole corpus counts as "nearby," which flattens real local
    structure HDBSCAN needs to find separable clusters.
    """
    _require_corpus_inputs(papers_path, filter_config_path)
    if labeling_sample_path.exists() and not force:
        existing = pd.read_csv(labeling_sample_path)
        has_labels = (
            "coherent" in existing.columns
            and existing["coherent"].astype(str).str.strip().ne("").any()
        )
        if has_labels:
            typer.echo(
                f"{labeling_sample_path} already has human-filled labels - rerunning would "
                "overwrite them (this command always regenerates the file from scratch). "
                "Back it up, or rerun with --force to overwrite anyway.",
                err=True,
            )
            raise typer.Exit(1)

    filtered = _load_filtered_corpus(papers_path, filter_config_path)
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

    config = TopicModelConfig(
        umap=UmapConfig(n_neighbors=umap_n_neighbors),
        hdbscan=HdbscanConfig(min_cluster_size=min_cluster_size),
        vectorizer=VectorizerConfig(min_df=min_df),
    )
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
    filter_config_path: Path = _DEFAULT_FILTER_CONFIG,
    embeddings_dir: Path = Path("data/interim/embeddings"),
    assignments_path: Path = Path("data/interim/topic_assignments.parquet"),
    labeling_sample_path: Path = Path("data/interim/topic_labeling_sample.csv"),
    figures_dir: Path = Path("reports/figures"),
    dynamics_summary_path: Path = Path("reports/topic_dynamics_summary.csv"),
    n_bootstrap_iterations: int = 1000,
    model_name: str = "nomic-ai/nomic-embed-text-v1.5",
    revision: str = "e9b6763023c676ca8431644204f50c2b100d9aab",
    task_prefix: str = "clustering: ",
    max_len: int = 256,
    dims: int = 64,
    seed: int = 0,
) -> None:
    """Build the 2D topic map and topic-dynamics-over-time charts from the registered
    nomic@64 topic model's saved assignments, plus a bootstrapped rising/fading topic
    ranking written to `dynamics_summary_path`. Requires `atlas register-topic` to have
    run first.
    """
    _require_corpus_inputs(papers_path, filter_config_path)
    _require_exists(
        assignments_path,
        f"No topic assignments at {assignments_path} - run `atlas register-topic` first.",
    )

    filtered = _load_filtered_corpus(papers_path, filter_config_path)
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

    typer.echo(
        f"Bootstrapping topic-share confidence intervals ({n_bootstrap_iterations} iterations)..."
    )
    bootstrap_shares = bootstrap_topic_share(topics, years, n_iterations=n_bootstrap_iterations)
    ranking = rank_topics_by_slope(share, bootstrap_shares, topic_labels)
    dynamics_summary_path.parent.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(dynamics_summary_path, index=False)
    typer.echo(f"Wrote rising/fading topic ranking to {dynamics_summary_path}")

    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_topic_map(coords, topics, topic_labels, figures_dir / "topic_map.png")
    plot_topic_dynamics(share, topic_labels, figures_dir / "topic_dynamics.png")
    typer.echo(f"Wrote figures to {figures_dir}")


if __name__ == "__main__":
    app()
