"""raw/ -> normalize -> parquet -> doğrulama -> manifest (Faz 1, CLAUDE.md §7).

Çıktılar `data/manifests/` altında: dataset manifestleri, `dataset_index.json`
(`data_manifest_hash`), `symbol_lifecycle.json`, `derived_ticks.json`,
`validation_report.{json,md}` ve `build_summary.md` (Türkçe özet).

Örnek:
    uv run python scripts/build_dataset.py --config config/data_v1.yaml
"""

from __future__ import annotations

from pathlib import Path

import typer

from pairs_bt.config import load_data_config
from pairs_bt.data.pipeline import apply_overrides, run_build

app = typer.Typer(no_args_is_help=False)


@app.command()
def main(
    config: Path = typer.Option(Path("config/data_v1.yaml"), help="Data config YAML"),
    start: str | None = typer.Option(None, help="YYYY-MM-DD (UTC), config'i ezer"),
    end: str | None = typer.Option(None, help="YYYY-MM-DD (UTC, dahil), config'i ezer"),
    datasets: str | None = typer.Option(None, help="klines_1m,klines_5m,aggTrades,fundingRate"),
) -> None:
    cfg = apply_overrides(load_data_config(config), start=start, end=end, datasets=datasets)
    summary = run_build(cfg)
    typer.echo(
        f"semboller={len(summary.lifecycle.entries)} hatalı={summary.report.error_count} "
        f"data_manifest_hash={summary.data_manifest_hash[:16]}"
    )
    typer.echo(f"özet: {cfg.paths.manifests_dir / 'build_summary.md'}")


if __name__ == "__main__":
    app()
