"""UniverseFilter — refresh anında işlem görebilir sembol evreni (RAPOR §8.1).

Kontroller (sırayla; ilk başarısız olan red nedenidir):
1. `exclude_symbols` (stablecoin / emtia tokenı vb.)
2. `SymbolLifecycle` girdisi var ve cutoff'ta aktif (invariant #3: gelecekteki delist bilinmez)
3. contract type izinli (varsayılan yalnız PERPETUAL)
4. cutoff'ta emir kuralları (exchange-info filtreleri) var (invariant #9)
5. listing yaşı >= `min_listing_age_days`
6. formation penceresinde stale olmayan bar oranı >= `min_coverage`
7. gözlenen bar'larda işlemsiz (quote_volume == 0) oran <= `max_zero_trade_frac`
8. 24 saatlik quote volume bloklarının medyanı >= `min_quote_volume_24h_usdt`

Yalnız `(cutoff − window_days, cutoff]` verisi kullanılır (invariant #1).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from pairs_bt.config import MarketConfig
from pairs_bt.data.exchange_info import ExchangeInfoStore
from pairs_bt.data.lifecycle import SymbolLifecycle
from pairs_bt.data.synchronizer import SyncedPanel
from pairs_bt.research.window import DAY_MS, formation_window


class UniverseReason(StrEnum):
    OK = "ok"
    EXCLUDED = "excluded"
    NOT_ACTIVE = "not_active"
    CONTRACT_TYPE = "contract_type"
    NO_FILTERS = "no_filters"
    LISTING_AGE = "listing_age"
    LOW_COVERAGE = "low_coverage"
    ZERO_TRADE = "zero_trade"
    LOW_VOLUME = "low_volume"


@dataclass(frozen=True, slots=True)
class SymbolEligibility:
    symbol: str
    reason: UniverseReason
    coverage: float  # stale olmayan bar / beklenen bar
    zero_trade_frac: float  # gözlenen bar'lar içinde işlemsiz oran (gözlem yoksa 1)
    median_quote_volume_24h_usdt: float
    listing_age_days: float  # lifecycle yoksa NaN


@dataclass(frozen=True, slots=True)
class UniverseResult:
    cutoff_ms: int
    rows: tuple[SymbolEligibility, ...]  # sembole göre sıralı

    @property
    def eligible(self) -> tuple[str, ...]:
        return tuple(r.symbol for r in self.rows if r.reason is UniverseReason.OK)

    def get(self, symbol: str) -> SymbolEligibility | None:
        for r in self.rows:
            if r.symbol == symbol:
                return r
        return None


class UniverseFilter:
    def __init__(self, market: MarketConfig, window_days: int) -> None:
        self.market = market
        self.window_days = window_days

    def evaluate(
        self,
        panel: SyncedPanel,
        cutoff_ms: int,
        lifecycle: SymbolLifecycle,
        exchange_info: ExchangeInfoStore,
    ) -> UniverseResult:
        interval = _interval_ms(panel)
        win = formation_window(panel.ts_ms, cutoff_ms, self.window_days, interval)
        stale = panel.stale[win.rows]
        observed = panel.observed[win.rows]
        qv = panel.quote_volume[win.rows]
        n_expected = max(win.expected_bars, 1)
        coverage = (~stale).sum(axis=0) / n_expected
        n_obs = observed.sum(axis=0)
        n_zero = (observed & (qv <= 0)).sum(axis=0)
        zero_frac = np.where(n_obs > 0, n_zero / np.maximum(n_obs, 1), 1.0)
        median_qv = _median_daily_volume(qv, DAY_MS // interval)

        rows = []
        for j, sym in enumerate(panel.symbols):
            entry = lifecycle.get(sym)
            age = entry.listing_age_days(cutoff_ms) if entry is not None else float("nan")
            stats = (float(coverage[j]), float(zero_frac[j]), float(median_qv[j]), age)
            reason = self._reason(sym, j, cutoff_ms, lifecycle, exchange_info, stats)
            rows.append(SymbolEligibility(sym, reason, *stats))
        return UniverseResult(cutoff_ms, tuple(sorted(rows, key=lambda r: r.symbol)))

    def _reason(
        self,
        sym: str,
        j: int,
        cutoff_ms: int,
        lifecycle: SymbolLifecycle,
        exchange_info: ExchangeInfoStore,
        stats: tuple[float, float, float, float],
    ) -> UniverseReason:
        m = self.market
        coverage, zero_frac, median_qv, age = stats
        entry = lifecycle.get(sym)
        if sym in m.exclude_symbols:
            return UniverseReason.EXCLUDED
        if entry is None or not entry.active_at(cutoff_ms):
            return UniverseReason.NOT_ACTIVE
        if entry.contract_type not in m.contract_types:
            return UniverseReason.CONTRACT_TYPE
        if exchange_info.filters_at(sym, cutoff_ms) is None:
            return UniverseReason.NO_FILTERS
        if age < m.min_listing_age_days:
            return UniverseReason.LISTING_AGE
        if coverage < m.min_coverage:
            return UniverseReason.LOW_COVERAGE
        if zero_frac > m.max_zero_trade_frac:
            return UniverseReason.ZERO_TRADE
        if median_qv < m.min_quote_volume_24h_usdt:
            return UniverseReason.LOW_VOLUME
        return UniverseReason.OK


def _interval_ms(panel: SyncedPanel) -> int:
    if panel.ts_ms.size < 2:
        raise ValueError("panel en az 2 bar içermeli (interval çıkarımı)")
    return int(panel.ts_ms[1] - panel.ts_ms[0])


def _median_daily_volume(qv: np.ndarray, bars_per_day: int) -> np.ndarray:
    """Pencerenin SONUNDAN geriye doğru tam 24 saatlik blokların quote volume toplamı, medyan."""
    n_days = qv.shape[0] // bars_per_day
    if n_days == 0:
        return np.zeros(qv.shape[1])
    tail = qv[qv.shape[0] - n_days * bars_per_day :]
    daily = tail.reshape(n_days, bars_per_day, qv.shape[1]).sum(axis=1)
    return np.asarray(np.median(daily, axis=0), dtype=np.float64)
