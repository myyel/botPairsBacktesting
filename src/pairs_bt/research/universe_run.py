"""4 saatlik pair universe üretimi — tüm refresh cutoff'ları için (Faz 2 çıkış kriteri).

Her cutoff bağımsızdır (yalnız `(cutoff − long_window, cutoff]` verisi; fold'lar/refresh'ler arası
mutable state yok — invariant #2). Bu yüzden cutoff'lar process pool'da paralel işlenir; sonuçlar
cutoff'a göre sıralanır, dolayısıyla worker sayısı çıktıyı değiştirmez (invariant #7).

Çıktılar (`out_dir`):
- `diagnostics.parquet`: korelasyon ön filtresinden geçen TÜM adaylar + red nedeni (invariant #10)
- `universe.parquet`: seçilen pair'ler (rank sıralı)
- `symbol_eligibility.parquet`: refresh başına sembol bazlı universe kararı
- `refresh_summary.parquet`: refresh başına sayaçlar
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl
import structlog

from pairs_bt.config import StrategyConfig
from pairs_bt.data.exchange_info import ExchangeInfoStore, load_derived_ticks
from pairs_bt.data.lifecycle import SymbolLifecycle
from pairs_bt.data.synchronizer import SyncedPanel
from pairs_bt.research.pair_selector import (
    DIAGNOSTICS_SCHEMA,
    PairSelector,
    PairUniverse,
    RejectReason,
)
from pairs_bt.research.panel import open_panel
from pairs_bt.research.universe import UniverseFilter, UniverseReason, UniverseResult

log = structlog.get_logger(__name__)

ELIGIBILITY_SCHEMA = pl.Schema(
    {
        "cutoff_ms": pl.Int64,
        "symbol": pl.String,
        "reason": pl.String,
        "coverage": pl.Float64,
        "zero_trade_frac": pl.Float64,
        "median_quote_volume_24h_usdt": pl.Float64,
        "listing_age_days": pl.Float64,
    }
)


@dataclass(frozen=True, slots=True)
class UniverseInputs:
    panel_dir: Path
    lifecycle_path: Path
    exchange_info_dir: Path
    derived_ticks_path: Path | None = None


@dataclass(frozen=True, slots=True)
class UniverseRunResult:
    diagnostics: pl.DataFrame
    eligibility: pl.DataFrame
    refresh_summary: pl.DataFrame

    @property
    def selected(self) -> pl.DataFrame:
        return self.diagnostics.filter(pl.col("selected")).sort(["cutoff_ms", "rank"])


# ------------------------------------------------------------------ worker
class _Worker:
    def __init__(self, inputs: UniverseInputs, cfg: StrategyConfig) -> None:
        self.panel: SyncedPanel = open_panel(inputs.panel_dir)
        self.lifecycle = SymbolLifecycle.read(inputs.lifecycle_path)
        ticks = (
            load_derived_ticks(inputs.derived_ticks_path)
            if inputs.derived_ticks_path is not None and inputs.derived_ticks_path.exists()
            else None
        )
        self.store = ExchangeInfoStore.load(inputs.exchange_info_dir, ticks)
        self.filter = UniverseFilter(cfg.market, cfg.pair_selection.long_window_days)
        self.selector = PairSelector(cfg)

    def run(self, cutoff_ms: int) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, Any]]:
        universe = self.filter.evaluate(self.panel, cutoff_ms, self.lifecycle, self.store)
        pu = self.selector.select(self.panel, cutoff_ms, universe)
        return pu.to_frame(), _eligibility_frame(universe), _summary_row(pu)


_WORKER: _Worker | None = None


def _init_worker(inputs: UniverseInputs, cfg_json: dict[str, Any]) -> None:
    global _WORKER
    _WORKER = _Worker(inputs, StrategyConfig.model_validate(cfg_json))


def _run_one(cutoff_ms: int) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, Any]]:
    if _WORKER is None:
        raise RuntimeError("worker başlatılmadı")
    return _WORKER.run(cutoff_ms)


# ------------------------------------------------------------------ public
def run_pair_universe(
    cfg: StrategyConfig,
    inputs: UniverseInputs,
    cutoffs: Sequence[int],
    *,
    workers: int = 1,
) -> UniverseRunResult:
    cut = sorted({int(c) for c in cutoffs})
    cfg_json = cfg.model_dump(mode="json")
    if workers <= 1:
        _init_worker(inputs, cfg_json)
        results = [_run_one(c) for c in cut]
    else:
        chunk = max(1, len(cut) // (workers * 8))
        with ProcessPoolExecutor(
            max_workers=workers, initializer=_init_worker, initargs=(inputs, cfg_json)
        ) as ex:
            results = []
            for i, r in enumerate(ex.map(_run_one, cut, chunksize=chunk), start=1):
                results.append(r)
                if i % 50 == 0 or i == len(cut):
                    log.info("pair_universe.progress", done=i, total=len(cut))
    diags = [r[0] for r in results]
    elig = [r[1] for r in results]
    summary = pl.DataFrame([r[2] for r in results], schema=_summary_schema(), orient="row")
    return UniverseRunResult(
        diagnostics=pl.concat(diags) if diags else pl.DataFrame(schema=DIAGNOSTICS_SCHEMA),
        eligibility=pl.concat(elig) if elig else pl.DataFrame(schema=ELIGIBILITY_SCHEMA),
        refresh_summary=summary,
    )


def write_outputs(result: UniverseRunResult, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "diagnostics.parquet": result.diagnostics,
        "universe.parquet": result.selected,
        "symbol_eligibility.parquet": result.eligibility,
        "refresh_summary.parquet": result.refresh_summary,
    }
    paths = []
    for name, df in files.items():
        p = out_dir / name
        df.write_parquet(p, compression="zstd", statistics=True)
        paths.append(p)
    return paths


# ------------------------------------------------------------------ yardımcılar
def _eligibility_frame(u: UniverseResult) -> pl.DataFrame:
    rows = [
        (
            u.cutoff_ms,
            r.symbol,
            str(r.reason),
            r.coverage,
            r.zero_trade_frac,
            r.median_quote_volume_24h_usdt,
            r.listing_age_days,
        )
        for r in u.rows
    ]
    return pl.DataFrame(rows, schema=ELIGIBILITY_SCHEMA, orient="row")


def _summary_schema() -> pl.Schema:
    cols: dict[str, pl.DataType] = {"cutoff_ms": pl.Int64()}
    for c in ("n_symbols", "n_pairs_total", "n_corr_pass", "n_candidates", "n_selected"):
        cols[c] = pl.Int64()
    for r in RejectReason:
        cols[f"rej_{r.value}"] = pl.Int64()
    for u in UniverseReason:
        cols[f"uni_{u.value}"] = pl.Int64()
    return pl.Schema(cols)


def _summary_row(pu: PairUniverse) -> dict[str, Any]:
    row: dict[str, Any] = {
        "cutoff_ms": pu.cutoff_ms,
        "n_symbols": pu.n_symbols,
        "n_pairs_total": pu.n_pairs_total,
        "n_corr_pass": pu.n_corr_pass,
        "n_candidates": len(pu.diagnostics),
        "n_selected": len(pu.selected),
    }
    for r in RejectReason:
        row[f"rej_{r.value}"] = sum(d.reject_reason is r for d in pu.diagnostics)
    for u in UniverseReason:
        row[f"uni_{u.value}"] = sum(e.reason is u for e in pu.universe.rows)
    return row
