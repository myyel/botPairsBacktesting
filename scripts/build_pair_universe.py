"""4 saatlik pair universe üretimi (Faz 2 çıkış kriteri, RAPOR §8).

Walk-forward penceresindeki her refresh cutoff'u için UniverseFilter + PairSelector çalıştırır.
Çıktı: `artifacts/pair_universe/<run_id>/` (diagnostics/universe/symbol_eligibility/
refresh_summary parquet + summary.md + run_meta.json + config_resolved.yaml).

Örnek:
    uv run python scripts/build_pair_universe.py --config config/strategy_v1.yaml --workers 16
    uv run python scripts/build_pair_universe.py --start 2025-10-01 --end 2025-10-08
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import structlog
import typer
import yaml

from pairs_bt.config import DatasetName, load_config, load_data_config
from pairs_bt.data.repository import DataRepository
from pairs_bt.provenance import git_info, make_run_id, package_versions
from pairs_bt.research.panel import load_panel, panel_cached, save_panel
from pairs_bt.research.universe_report import write_summary
from pairs_bt.research.universe_run import UniverseInputs, run_pair_universe, write_outputs
from pairs_bt.research.window import DAY_MS, refresh_schedule

app = typer.Typer(no_args_is_help=False)
ROOT = Path(__file__).resolve().parent.parent
_INTERVAL_MS = {"1m": 60_000, "5m": 300_000}


def _parse_day(s: str) -> int:
    d = datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=UTC)
    return int(d.timestamp() * 1000)


@app.command()
def main(
    config: Path = typer.Option(Path("config/strategy_v1.yaml"), help="Strategy config YAML"),
    data_config: Path = typer.Option(Path("config/data_v1.yaml"), help="Data config YAML"),
    start: str | None = typer.Option(None, help="YYYY-MM-DD UTC (dahil); varsayılan WF start"),
    end: str | None = typer.Option(None, help="YYYY-MM-DD UTC (hariç); varsayılan WF end"),
    workers: int = typer.Option(8, help="paralel process sayısı"),
    out_root: Path = typer.Option(Path("artifacts/pair_universe")),
    cache_root: Path = typer.Option(Path("artifacts/cache")),
) -> None:
    cfg = load_config(config)
    dcfg = load_data_config(data_config)
    cfg_hash = cfg.config_hash()
    run_id = make_run_id(cfg_hash)
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    )
    structlog.contextvars.bind_contextvars(run_id=run_id)
    log = structlog.get_logger("build_pair_universe")

    wf = cfg.walk_forward
    lo = _parse_day(start) if start else int(wf.start.timestamp() * 1000)
    hi = _parse_day(end) if end else int(wf.end.timestamp() * 1000)
    cutoffs = refresh_schedule(lo, hi, cfg.pair_selection.refresh_hours)
    if cutoffs.size == 0:
        raise typer.BadParameter("aralıkta refresh cutoff'u yok")

    manifests = dcfg.paths.manifests_dir
    index = json.loads((manifests / "dataset_index.json").read_text(encoding="utf-8"))
    data_hash = index["data_manifest_hash"]
    interval = cfg.market.formation_interval
    dataset: DatasetName = "klines_5m" if interval == "5m" else "klines_1m"
    interval_ms = _INTERVAL_MS[interval]
    p_start = int(cutoffs[0]) - cfg.pair_selection.long_window_days * DAY_MS
    p_end = int(cutoffs[-1]) + 1
    repo = DataRepository(dcfg.paths.parquet_dir)
    symbols = repo.symbols(dataset)
    key_src = json.dumps(
        [data_hash, dataset, p_start, p_end, cfg.market.max_ffill_bars, symbols], sort_keys=True
    )
    panel_dir = cache_root / f"panel_{hashlib.sha256(key_src.encode()).hexdigest()[:16]}"
    if not panel_cached(panel_dir):
        log.info("panel.load", symbols=len(symbols), dataset=dataset, dir=str(panel_dir))
        panel = load_panel(
            repo, dataset, symbols, p_start, p_end, interval_ms, cfg.market.max_ffill_bars
        )
        save_panel(panel, panel_dir)
        del panel
    else:
        log.info("panel.cache_hit", dir=str(panel_dir))

    inputs = UniverseInputs(
        panel_dir=panel_dir,
        lifecycle_path=manifests / "symbol_lifecycle.json",
        exchange_info_dir=dcfg.paths.raw_dir / "exchange_info",
        derived_ticks_path=manifests / "derived_ticks.json",
    )
    log.info("pair_universe.start", refreshes=int(cutoffs.size), workers=workers)
    t0 = datetime.now(UTC)
    result = run_pair_universe(cfg, inputs, cutoffs.tolist(), workers=workers)
    elapsed = (datetime.now(UTC) - t0).total_seconds()

    out_dir = out_root / run_id
    write_outputs(result, out_dir)
    commit, dirty = git_info(ROOT)
    meta = {
        "run_id": run_id,
        "config_hash": cfg_hash,
        "data_manifest_hash": data_hash,
        "git_commit": commit,
        "git_dirty": dirty,
        "seed": cfg.run.seed,
        "formation_dataset": dataset,
        "refreshes": int(cutoffs.size),
        "first_cutoff_ms": int(cutoffs[0]),
        "last_cutoff_ms": int(cutoffs[-1]),
        "elapsed_seconds": round(elapsed, 1),
        "packages": package_versions(),
    }
    (out_dir / "run_meta.json").write_text(
        json.dumps(meta, indent=1, sort_keys=True), encoding="utf-8"
    )
    (out_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=True, allow_unicode=True),
        encoding="utf-8",
    )
    summary_meta = {k: v for k, v in meta.items() if k != "packages"}
    write_summary(result, out_dir, summary_meta)
    log.info("pair_universe.done", out=str(out_dir), seconds=round(elapsed, 1))
    typer.echo(f"çıktı: {out_dir}  (refresh={cutoffs.size}, süre={elapsed:.0f}s)")


if __name__ == "__main__":
    app()
