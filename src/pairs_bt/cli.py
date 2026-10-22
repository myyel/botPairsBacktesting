"""Komut satırı girişi: `uv run pairs-bt --help`."""

from __future__ import annotations

from pathlib import Path

import typer

from pairs_bt.config import load_config

app = typer.Typer(help="ACKPT-50 pairs trading backtest CLI", no_args_is_help=True)


@app.command()
def validate_config(path: Path = typer.Argument(Path("config/strategy_v1.yaml"))) -> None:
    """Config dosyasını doğrular ve hash'ini yazdırır."""
    cfg = load_config(path)
    typer.echo(f"OK  {cfg.run.name}  hash={cfg.config_hash()[:12]}")


@app.command()
def version() -> None:
    """Paket sürümü."""
    from pairs_bt import __version__

    typer.echo(__version__)


if __name__ == "__main__":
    app()
