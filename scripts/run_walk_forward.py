"""Walk-forward backtest çalıştırıcı (Faz 3+ — henüz implement edilmedi). Bkz. RAPOR §12."""

from __future__ import annotations

from pathlib import Path

import typer

from pairs_bt.config import load_config

app = typer.Typer(no_args_is_help=True)


@app.command()
def main(config: Path = typer.Option(Path("config/strategy_v1.yaml"))) -> None:
    cfg = load_config(config)
    typer.echo(f"config OK: {cfg.run.name} hash={cfg.config_hash()[:12]}")
    raise NotImplementedError("Faz 3: BacktestEngine + WalkForwardRunner")


if __name__ == "__main__":
    app()
