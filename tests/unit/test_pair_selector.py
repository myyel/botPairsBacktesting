"""PairSelector (RAPOR §8.2–§8.3) — sentetik pazarda uçtan uca seçim."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from pairs_bt.config import StrategyConfig
from pairs_bt.research.pair_selector import PairSelector, RejectReason, build_pair_universe
from pairs_bt.research.universe import UniverseFilter
from tests.fixtures.synthetic_panel import (
    BARS_PER_DAY,
    DAY,
    T0,
    make_lifecycle,
    make_pair_market,
    make_panel,
    make_store,
)

N_DAYS = 31
CUTOFF = T0 + 29 * DAY  # 28 günlük formation + 2 gün cutoff sonrası veri


def _cfg(cfg: StrategyConfig, **pair_selection: object) -> StrategyConfig:
    """Seçim mantığını maliyet kalibrasyonundan ayırmak için slippage k2 küçültülür."""
    slip = cfg.costs.slippage.model_copy(update={"k2": 0.1})
    costs = cfg.costs.model_copy(update={"slippage": slip})
    ps = cfg.pair_selection.model_copy(update=pair_selection)
    return cfg.model_copy(update={"costs": costs, "pair_selection": ps})


@pytest.fixture
def market(rng: np.random.Generator) -> dict[str, np.ndarray]:
    return make_pair_market(rng, N_DAYS * BARS_PER_DAY)


def _run(cfg: StrategyConfig, prices: dict[str, np.ndarray]):  # type: ignore[no-untyped-def]
    syms = sorted(prices)
    return build_pair_universe(
        make_panel(prices), CUTOFF, make_lifecycle(syms), make_store(syms), cfg
    )


def _pairset(universe) -> set[frozenset[str]]:  # type: ignore[no-untyped-def]
    return {frozenset((d.a, d.b)) for d in universe.selected}


def test_selects_cointegrated_pairs(
    strategy_config: StrategyConfig, market: dict[str, np.ndarray]
) -> None:
    u = _run(_cfg(strategy_config), market)
    true_pairs = {frozenset(("AAAUSDT", "BBBUSDT")), frozenset(("CCCUSDT", "DDDUSDT"))}
    assert true_pairs <= _pairset(u)
    by_pair = {frozenset((d.a, d.b)): d for d in u.selected}
    # gerçek cointegrated çiftler bayraksız
    assert not any(by_pair[p].spurious_risk for p in true_pairs)
    ab = by_pair[frozenset(("AAAUSDT", "BBBUSDT"))]
    if ab.a == "AAAUSDT":
        assert ab.beta == pytest.approx(1.2, abs=0.1)
    else:
        assert ab.beta == pytest.approx(1 / 1.2, abs=0.1)
    assert 5.0 < ab.half_life_min < 30.0
    assert ab.hurst < strategy_config.pair_selection.max_hurst
    assert ab.p_long < 0.01
    assert ab.p_short < strategy_config.pair_selection.adf_p_short
    assert ab.expected_gross_bps > ab.expected_cost_bps
    assert ab.rank in (1, 2)
    assert 0 <= ab.score <= 1
    # Faz 3 için Kalman başlangıç state'i formation sonundan
    assert ab.kalman_beta == pytest.approx(ab.beta, abs=0.2)
    assert ab.kalman_p_bb > 0


def test_noise_pairs_rejected_with_reasons(
    strategy_config: StrategyConfig, market: dict[str, np.ndarray]
) -> None:
    u = _run(_cfg(strategy_config), market)
    rejected = [d for d in u.diagnostics if not d.selected]
    assert rejected
    assert all(d.reject_reason is not RejectReason.SELECTED for d in rejected)
    noise = [d for d in rejected if "AAAUSDT" not in (d.a, d.b) and "CCCUSDT" not in (d.a, d.b)]
    assert any(
        d.reject_reason in (RejectReason.COINT_LONG, RejectReason.COINT_SHORT) for d in noise
    )
    assert u.n_symbols == 8
    assert u.n_pairs_total == 28
    assert u.n_corr_pass == len(u.diagnostics)


def test_max_pairs_per_asset(strategy_config: StrategyConfig, rng: np.random.Generator) -> None:
    prices = make_pair_market(rng, N_DAYS * BARS_PER_DAY, extra_b_pair=True)
    u = _run(_cfg(strategy_config, max_pairs_per_asset=1), prices)
    degree: dict[str, int] = {}
    for d in u.selected:
        for s in (d.a, d.b):
            degree[s] = degree.get(s, 0) + 1
    assert max(degree.values()) == 1
    assert any(d.reject_reason is RejectReason.DIVERSIFICATION for d in u.diagnostics)


def test_active_pairs_cap(strategy_config: StrategyConfig, market: dict[str, np.ndarray]) -> None:
    u = _run(_cfg(strategy_config, active_pairs=1), market)
    assert len(u.selected) == 1
    assert u.selected[0].rank == 1


def test_beta_bounds_reject(strategy_config: StrategyConfig, market: dict[str, np.ndarray]) -> None:
    u = _run(_cfg(strategy_config, beta_bounds=(3.0, 5.0)), market)
    assert u.selected == ()


def test_deterministic(strategy_config: StrategyConfig, market: dict[str, np.ndarray]) -> None:
    cfg = _cfg(strategy_config)
    assert_frame_equal(_run(cfg, market).to_frame(), _run(cfg, market).to_frame())


def test_data_after_cutoff_does_not_change_selection(
    strategy_config: StrategyConfig, market: dict[str, np.ndarray], rng: np.random.Generator
) -> None:
    """Invariant #1/#2: cutoff sonrası fiyatları bozmak seçimi ve diagnostics'i değiştirmez."""
    cfg = _cfg(strategy_config)
    base = _run(cfg, market).to_frame()
    cut_idx = (CUTOFF - T0) // 300_000  # cutoff bar'ının (dahil) sayısı
    perturbed = {
        s: np.concatenate([v[:cut_idx], v[cut_idx:] + rng.standard_normal(v.size - cut_idx)])
        for s, v in market.items()
    }
    assert_frame_equal(base, _run(cfg, perturbed).to_frame())


def test_data_before_window_does_not_change_selection(
    strategy_config: StrategyConfig, market: dict[str, np.ndarray], rng: np.random.Generator
) -> None:
    """Formation yalnız (cutoff − long_window, cutoff] kullanır."""
    cfg = _cfg(strategy_config)
    base = _run(cfg, market).to_frame()
    n_before = BARS_PER_DAY - 1  # ilk gün (pencere dışı) bozulur
    perturbed = {
        s: np.concatenate([v[:n_before] + rng.standard_normal(n_before), v[n_before:]])
        for s, v in market.items()
    }
    assert_frame_equal(base, _run(cfg, perturbed).to_frame())


def test_empty_universe(strategy_config: StrategyConfig, market: dict[str, np.ndarray]) -> None:
    syms = sorted(market)
    panel = make_panel(market)
    u = build_pair_universe(panel, CUTOFF, make_lifecycle([]), make_store(syms), strategy_config)
    assert u.selected == ()
    assert u.diagnostics == ()
    assert u.to_frame().height == 0


def test_selector_uses_given_universe(
    strategy_config: StrategyConfig, market: dict[str, np.ndarray]
) -> None:
    cfg = _cfg(strategy_config)
    syms = sorted(market)
    panel = make_panel(market)
    universe = UniverseFilter(cfg.market, cfg.pair_selection.long_window_days).evaluate(
        panel, CUTOFF, make_lifecycle(syms, overrides={}), make_store(["AAAUSDT", "BBBUSDT"])
    )
    u = PairSelector(cfg).select(panel, CUTOFF, universe)
    assert _pairset(u) == {frozenset(("AAAUSDT", "BBBUSDT"))}
    frame = u.to_frame()
    assert isinstance(frame, pl.DataFrame)
    assert frame["cutoff_ms"].unique().to_list() == [CUTOFF]


def test_jump_in_short_window_rejects_pair(
    strategy_config: StrategyConfig, market: dict[str, np.ndarray]
) -> None:
    """Çöküş fitilleri (kısa pencerede) → JUMP; diğer pair etkilenmez."""
    cfg = _cfg(strategy_config)
    p = dict(market)
    a = p["AAAUSDT"].copy()
    cut_idx = (CUTOFF - T0) // 300_000
    # kısa pencerede 12 fitil → 24 sıçrama artışı (> %0.5 · 2016 ≈ 10). Boy 0.06: sıçrama eşiğinin
    # (8·robust σ ≈ 0.036) üstünde, ama korelasyonu ön filtre altına düşürmeyecek kadar küçük
    for k in range(12):
        a[cut_idx - 100 - 50 * k] += 0.06
    p["AAAUSDT"] = a
    u = _run(cfg, p)
    assert frozenset(("CCCUSDT", "DDDUSDT")) in _pairset(u)
    assert frozenset(("AAAUSDT", "BBBUSDT")) not in _pairset(u)
    ab = [d for d in u.diagnostics if {d.a, d.b} == {"AAAUSDT", "BBBUSDT"}]
    assert ab
    assert ab[0].reject_reason is RejectReason.JUMP
    assert ab[0].jump_frac > cfg.pair_selection.max_jump_frac


def test_few_jumps_are_winsorized_not_rejected(
    strategy_config: StrategyConfig, market: dict[str, np.ndarray]
) -> None:
    cfg = _cfg(strategy_config)
    base = {frozenset((d.a, d.b)): d for d in _run(cfg, market).selected}
    p = dict(market)
    a = p["AAAUSDT"].copy()
    a[(CUTOFF - T0) // 300_000 - 200] += 0.3  # tek fitil
    p["AAAUSDT"] = a
    u = _run(cfg, p)
    assert frozenset(("AAAUSDT", "BBBUSDT")) in _pairset(u)
    ab = next(d for d in u.selected if {d.a, d.b} == {"AAAUSDT", "BBBUSDT"})
    ref = base[frozenset(("AAAUSDT", "BBBUSDT"))]
    # kırpma sayesinde fitil sigma/HL tahminini bozmaz
    assert ab.sigma_eq == pytest.approx(ref.sigma_eq, rel=0.05)
    assert ab.half_life_min == pytest.approx(ref.half_life_min, rel=0.1)


def test_level_method_still_supported(
    strategy_config: StrategyConfig, market: dict[str, np.ndarray]
) -> None:
    u = _run(_cfg(strategy_config, half_life_method="level"), market)
    assert len(u.selected) == 2
    for d in u.selected:
        assert d.half_life_min == d.half_life_level_min


def test_spurious_pairs_are_flagged(
    strategy_config: StrategyConfig, market: dict[str, np.ndarray]
) -> None:
    """Kabul edilen risk (karar B): RW seviye + hızlı gürültü spread'leri EWMA sapma HL'ini geçip
    seçilebilir; seçilirlerse `spurious_risk` ile işaretli olmalıdır (seviye HL > ufuk)."""
    cfg = _cfg(strategy_config)
    u = _run(cfg, market)
    true_pairs = {frozenset(("AAAUSDT", "BBBUSDT")), frozenset(("CCCUSDT", "DDDUSDT"))}
    spurious = [d for d in u.selected if frozenset((d.a, d.b)) not in true_pairs]
    assert spurious, "fixture en az bir sahte çifti seçtirmeli (risk görünür olsun)"
    hl_max = cfg.pair_selection.half_life_minutes[1]
    for d in spurious:
        assert d.spurious_risk
        assert d.half_life_level_min > hl_max
    assert u.to_frame()["spurious_risk"].dtype == pl.Boolean
