"""Binance public data arşivinden USDⓈ-M futures verisi indirir (Faz 1, RAPOR §4, CLAUDE.md §7).

klines (1m/5m), aggTrades, fundingRate zip'leri `.CHECKSUM` doğrulamasıyla `data/raw/` altına
değiştirilmeden kaydedilir; güncel exchangeInfo snapshot'ı alınır;
`data/manifests/download_report.md`
yazılır. Tekrar çalıştırmak güvenlidir (doğrulanmış dosyalar atlanır).

Örnek:
    uv run python scripts/download_binance_data.py --config config/data_v1.yaml
    uv run python scripts/download_binance_data.py --symbols BTCUSDT,ETHUSDT \
        --start 2025-09-01 --end 2025-09-02 --datasets aggTrades
"""

from __future__ import annotations

from pathlib import Path

import httpx
import typer

from pairs_bt.config import load_data_config
from pairs_bt.data.pipeline import apply_overrides, run_download

app = typer.Typer(no_args_is_help=False)


@app.command()
def main(
    config: Path = typer.Option(Path("config/data_v1.yaml"), help="Data config YAML"),
    symbols: str | None = typer.Option(None, help="Virgülle ayrılmış semboller (universe yerine)"),
    start: str | None = typer.Option(None, help="YYYY-MM-DD (UTC), config'i ezer"),
    end: str | None = typer.Option(None, help="YYYY-MM-DD (UTC, dahil), config'i ezer"),
    datasets: str | None = typer.Option(None, help="klines_1m,klines_5m,aggTrades,fundingRate"),
) -> None:
    cfg = apply_overrides(
        load_data_config(config), symbols=symbols, start=start, end=end, datasets=datasets
    )
    with httpx.Client(http2=False, follow_redirects=True) as client:
        summary = run_download(cfg, client)
    typer.echo(
        f"universe={len(summary.universe)} indirilen={summary.downloaded} "
        f"atlanan={summary.skipped} başarısız={len(summary.failed)}"
    )
    typer.echo(f"rapor: {cfg.paths.manifests_dir / 'download_report.md'}")
    if summary.failed:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
