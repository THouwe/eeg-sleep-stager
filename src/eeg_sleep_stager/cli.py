"""Typer CLI wiring the pipeline subcommands.

    eeg-sleep-stager ingest   --limit 6
    eeg-sleep-stager etl      --master "local[*]"
    eeg-sleep-stager split    --seed 42
    eeg-sleep-stager train    --model cnn
    eeg-sleep-stager evaluate --model cnn

In S1 every subcommand is wired to its module function; the pipeline stages
themselves raise NotImplementedError until their milestone lands.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from . import __version__
from .config import load_config

app = typer.Typer(
    name="eeg-sleep-stager",
    help="Distributed EEG sleep staging: PySpark ETL + Keras 1-D CNN.",
    no_args_is_help=True,
    add_completion=False,
)

# Shared --config option.
ConfigOpt = typer.Option(
    "config.yaml",
    "--config",
    "-c",
    help="Path to config.yaml.",
    exists=False,
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"eeg-sleep-stager {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", callback=_version_callback, is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    """Distributed EEG sleep staging pipeline."""


@app.command()
def ingest(
    config: Path = ConfigOpt,
    limit: Optional[int] = typer.Option(
        None, "--limit", "-n", help="Only pull the first N subjects."
    ),
    dataset: str = typer.Option(
        "sleep-cassette", "--dataset", help="PhysioNet subset name."
    ),
) -> None:
    """Download recordings and build manifest.csv (S2)."""
    from . import ingest as ingest_mod

    cfg = load_config(config)
    ingest_mod.ingest(cfg, limit=limit)


@app.command()
def etl(
    config: Path = ConfigOpt,
    master: Optional[str] = typer.Option(
        None, "--master", help="Spark master URL (defaults to config)."
    ),
    channel: Optional[str] = typer.Option(
        None, "--channel", help="EEG channel to extract (defaults to config)."
    ),
    epoch_sec: Optional[int] = typer.Option(
        None, "--epoch-sec", help="Epoch length in seconds (defaults to config)."
    ),
) -> None:
    """Run the PySpark ETL: manifest -> epochs.parquet (S4)."""
    from . import etl as etl_mod

    cfg = load_config(config)
    if channel is not None:
        cfg.dataset.channel = channel
    if epoch_sec is not None:
        cfg.dataset.epoch_sec = epoch_sec
    etl_mod.run_etl(cfg, master=master)


@app.command()
def split(
    config: Path = ConfigOpt,
    seed: Optional[int] = typer.Option(None, "--seed", help="Split RNG seed."),
    val_frac: Optional[float] = typer.Option(None, "--val-frac", help="Val fraction."),
    test_frac: Optional[float] = typer.Option(
        None, "--test-frac", help="Test fraction."
    ),
) -> None:
    """Subject-wise seeded split -> splits.json (S5)."""
    from . import split as split_mod

    cfg = load_config(config)
    split_mod.make_split(cfg, seed=seed, val_frac=val_frac, test_frac=test_frac)


@app.command()
def train(
    config: Path = ConfigOpt,
    model: str = typer.Option("cnn", "--model", help="Model: 'cnn' or 'baseline'."),
    epochs: Optional[int] = typer.Option(None, "--epochs", help="Training epochs."),
    batch_size: Optional[int] = typer.Option(None, "--batch-size", help="Batch size."),
    lr: Optional[float] = typer.Option(None, "--lr", help="Learning rate."),
    class_weights: Optional[str] = typer.Option(
        None, "--class-weights", help="'auto' or 'none'."
    ),
) -> None:
    """Train the CNN or the features baseline (S6)."""
    from . import train as train_mod

    cfg = load_config(config)
    train_mod.train(
        cfg,
        model=model,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        class_weights=class_weights,
    )


@app.command()
def evaluate(
    config: Path = ConfigOpt,
    model: str = typer.Option("cnn", "--model", help="Model to evaluate."),
) -> None:
    """Evaluate on held-out subjects -> reports/ (S7)."""
    from . import evaluate as evaluate_mod

    cfg = load_config(config)
    evaluate_mod.evaluate(cfg, model=model)


if __name__ == "__main__":
    app()
