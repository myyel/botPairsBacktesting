"""Faz 2 uçtan uca: repository -> panel önbelleği -> 4 saatlik universe -> parquet + summary.md."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal
from tests.fixtures.binance_archive import ei_symbol, exchange_info_json
from tests.fixtures.synthetic_panel import (
    BARS_PER_DAY,
    DAY,
    M5,
    T0,
    make_lifecycle,
    make_pair_market,
)

from pairs_bt.config import StrategyConfig
from pairs_bt.data.normalize import KLINES_SCHEMA
from pairs_bt.data.repository import DataRepository
from pairs_bt.research.panel import load_panel, save_panel
from pairs_bt.research.universe_report import turnover, write_summary
from pairs_bt.research.universe_run import UniverseInputs, run_pair_universe, write_outputs
from pairs_bt.research.window import refresh_schedule

N_DAYS = 31


def _bars(ts: np.ndarray, log_close: np.ndarray) -> pl.DataFrame:
    close = np.exp(log_close)
    n = ts.size
    return pl.DataFrame(
        {
            "ts_ms": ts,
            "open_time_ms": ts - M5,
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": np.ones(n),
            "quote_volume": np.full(n, 1e6),
            "trades": np.full(n, 10, dtype=np.int64),
            "taker_buy_volume": np.zeros(n),
            "taker_buy_quote_volume": np.zeros(n),
        },
        schema=KLINES_SCHEMA,
    )


@pytest.fixture
def env(
    tmp_path: Path, rng: np.random.Generator, strategy_config: StrategyConfig
) -> tuple[StrategyConfig, UniverseInputs, np.ndarray]:
    prices = make_pair_market(rng, N_DAYS * BARS_PER_DAY)
    repo = DataRepository(tmp_path / "parquet")
    ts = T0 + M5 * (np.arange(N_DAYS * BARS_PER_DAY, dtype=np.int64) + 1)
    for sym, lp in prices.items():
        repo.write_symbol("klines_5m", sym, _bars(ts, lp))
    syms = sorted(prices)
    lc_path = make_lifecycle(syms).write(tmp_path / "manifests" / "symbol_lifecycle.json")
    ei_dir = tmp_path / "raw" / "exchange_info"
    ei_dir.mkdir(parents=True)
    snap_ms = T0 + 500 * DAY
    (ei_dir / f"exchangeInfo_{snap_ms}.json").write_bytes(
        exchange_info_json([ei_symbol(s) for s in syms], snap_ms)
    )
    cutoffs = refresh_schedule(T0 + 29 * DAY, T0 + 30 * DAY, 4)
    panel = load_panel(
        repo, "klines_5m", syms, int(cutoffs[0]) - 28 * DAY, int(cutoffs[-1]) + 1, M5, 3
    )
    panel_dir = save_panel(panel, tmp_path / "cache" / "panel")
    slip = strategy_config.costs.slippage.model_copy(update={"k2": 0.1})
    cfg = strategy_config.model_copy(
        update={"costs": strategy_config.costs.model_copy(update={"slippage": slip})}
    )
    return cfg, UniverseInputs(panel_dir, lc_path, ei_dir, None), cutoffs


def test_end_to_end(env: tuple[StrategyConfig, UniverseInputs, np.ndarray], tmp_path: Path) -> None:
    cfg, inputs, cutoffs = env
    res = run_pair_universe(cfg, inputs, cutoffs.tolist(), workers=1)
    assert res.refresh_summary.height == cutoffs.size == 6
    assert res.refresh_summary["cutoff_ms"].to_list() == sorted(cutoffs.tolist())
    sel = res.selected
    pairs = {frozenset(p) for p in sel.select("a", "b").iter_rows()}
    true_pairs = {frozenset(("AAAUSDT", "BBBUSDT")), frozenset(("CCCUSDT", "DDDUSDT"))}
    assert true_pairs <= pairs
    # kabul edilen risk (karar B): diğer seçimler spurious_risk ile işaretli
    others = sel.filter(
        ~pl.struct("a", "b").map_elements(
            lambda r: frozenset((r["a"], r["b"])) in true_pairs, return_dtype=pl.Boolean
        )
    )
    assert others["spurious_risk"].all()
    assert set(sel["cutoff_ms"].unique().to_list()) == set(cutoffs.tolist())
    # tüm cutoff'larda 8 sembol uygun
    assert res.eligibility.filter(pl.col("reason") == "ok").height == 8 * 6

    out = tmp_path / "out"
    files = write_outputs(res, out)
    assert all(f.exists() for f in files)
    summary = write_summary(res, out, {"run_id": "test"})
    text = summary.read_text(encoding="utf-8")
    assert "Pair Universe Özeti" in text
    assert "AAAUSDT" in text
    to_all = turnover(sel)
    assert to_all.height == 5
    assert to_all["retained"].is_between(0.0, 1.0).all()
    # gerçek çiftler her refresh'te korunur
    true_sel = sel.filter(~pl.col("spurious_risk"))
    to = turnover(true_sel)
    assert to["retained"].to_list() == [1.0] * 5


@pytest.mark.slow
def test_parallel_equals_sequential(env: tuple[StrategyConfig, UniverseInputs, np.ndarray]) -> None:
    """Invariant #7: worker sayısı sonucu değiştirmez."""
    cfg, inputs, cutoffs = env
    seq = run_pair_universe(cfg, inputs, cutoffs.tolist(), workers=1)
    par = run_pair_universe(cfg, inputs, list(reversed(cutoffs.tolist())), workers=2)
    assert_frame_equal(seq.diagnostics, par.diagnostics)
    assert_frame_equal(seq.eligibility, par.eligibility)
    assert_frame_equal(seq.refresh_summary, par.refresh_summary)


def test_empty_summary(tmp_path: Path) -> None:
    from pairs_bt.research.pair_selector import DIAGNOSTICS_SCHEMA
    from pairs_bt.research.universe_run import (
        ELIGIBILITY_SCHEMA,
        UniverseRunResult,
        _summary_schema,
    )

    res = UniverseRunResult(
        pl.DataFrame(schema=DIAGNOSTICS_SCHEMA),
        pl.DataFrame(schema=ELIGIBILITY_SCHEMA),
        pl.DataFrame(schema=_summary_schema()),
    )
    assert res.selected.height == 0
    text = write_summary(res, tmp_path, {"run_id": "x"}).read_text(encoding="utf-8")
    assert "Hiç refresh işlenmedi" in text
