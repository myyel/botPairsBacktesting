"""Tarihli exchange-info: snapshot'lar + türetilmiş tickSize (CLAUDE.md invariant #9).

data.binance.vision geçmiş exchangeInfo sunmaz. Politika (kullanıcı kararı, Faz 1):
- `fapi/v1/exchangeInfo` periyodik olarak ham snapshot olarak saklanır (`data/raw/exchange_info/`).
- ts >= bir snapshot'ın `serverTime`'ı ise o tarihteki en son snapshot kullanılır (kesin).
- ts tüm snapshot'lardan önceyse: tickSize o dönemin fiyat grid'inden türetilir
  (`derive_tick_size`),
  stepSize/minQty/minNotional en erken snapshot'tan alınır ve `approximate=True` işaretlenir.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from pairs_bt.domain.instruments import SymbolFilters

_MAX_TICK_DECIMALS = 8
_MANTISSAS = ("1", "2", "2.5", "5")


@dataclass(frozen=True, slots=True)
class SymbolMeta:
    symbol: str
    contract_type: str
    quote_asset: str
    status: str
    onboard_ms: int | None
    delivery_ms: int | None


@dataclass(frozen=True, slots=True)
class ExchangeInfoSnapshot:
    server_time_ms: int
    filters: Mapping[str, SymbolFilters]
    meta: Mapping[str, SymbolMeta]

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> ExchangeInfoSnapshot:
        ts = int(data["serverTime"])
        filters: dict[str, SymbolFilters] = {}
        meta: dict[str, SymbolMeta] = {}
        for s in data.get("symbols", []):
            sym = str(s["symbol"])
            by_type = {f["filterType"]: f for f in s.get("filters", [])}
            price, lot = by_type.get("PRICE_FILTER"), by_type.get("LOT_SIZE")
            notional = by_type.get("MIN_NOTIONAL", {})
            meta[sym] = SymbolMeta(
                symbol=sym,
                contract_type=str(s.get("contractType", "")),
                quote_asset=str(s.get("quoteAsset", "")),
                status=str(s.get("status", "")),
                onboard_ms=_opt_int(s.get("onboardDate")),
                delivery_ms=_opt_int(s.get("deliveryDate")),
            )
            if price is None or lot is None:
                continue
            filters[sym] = SymbolFilters(
                symbol=sym,
                tick_size=Decimal(price["tickSize"]),
                step_size=Decimal(lot["stepSize"]),
                min_qty=Decimal(lot["minQty"]),
                min_notional=Decimal(notional.get("notional", "0")),
                status=meta[sym].status,
                valid_from_ms=ts,
            )
        return cls(ts, filters, meta)


def _opt_int(x: object) -> int | None:
    return None if x is None else int(x)  # type: ignore[call-overload]


class ExchangeInfoStore:
    def __init__(
        self,
        snapshots: Sequence[ExchangeInfoSnapshot],
        derived_ticks: Mapping[str, Sequence[tuple[int, Decimal]]] | None = None,
    ) -> None:
        self._snaps = sorted(snapshots, key=lambda s: s.server_time_ms)
        self._derived = {k: sorted(v) for k, v in (derived_ticks or {}).items()}

    @classmethod
    def load(
        cls,
        directory: Path,
        derived_ticks: Mapping[str, Sequence[tuple[int, Decimal]]] | None = None,
    ) -> ExchangeInfoStore:
        snaps = [
            ExchangeInfoSnapshot.from_json(json.loads(p.read_text(encoding="utf-8")))
            for p in sorted(directory.glob("exchangeInfo_*.json"))
        ]
        return cls(snaps, derived_ticks)

    @property
    def snapshots(self) -> tuple[ExchangeInfoSnapshot, ...]:
        return tuple(self._snaps)

    def filters_at(self, symbol: str, ts_ms: int) -> SymbolFilters | None:
        exact = [s for s in self._snaps if s.server_time_ms <= ts_ms and symbol in s.filters]
        if exact:
            return exact[-1].filters[symbol]
        later = [s for s in self._snaps if symbol in s.filters]
        if not later:
            return None
        base = later[0].filters[symbol]
        tick = base.tick_size
        for valid_from, derived in self._derived.get(symbol, ()):
            if valid_from <= ts_ms:
                tick = derived
        return replace(base, tick_size=tick, valid_from_ms=0, approximate=True)

    def meta(self, symbol: str) -> SymbolMeta | None:
        for s in reversed(self._snaps):
            if symbol in s.meta:
                return s.meta[symbol]
        return None


def derive_tick_size(prices: npt.NDArray[np.float64], min_fraction: float = 1.0) -> Decimal | None:
    """Fiyat grid'inden tickSize tahmini (RAPOR §4.2 exchange-info yaklaşığı).

    Aday tick'ler `{1, 2, 2.5, 5} × 10^k` (k: -8..6). Fiyatların en az `min_fraction` oranı
    grid'deyse aday geçerlidir; en büyük geçerli aday döner. `min_fraction < 1` birkaç aykırı
    fiyatın (ör. tek tük alt-tick fill) tahmini küçültmesini engeller. Sonuç gerçek tick'in katı
    olabilir (fiyat yalnız kaba adımlarla hareket ettiyse) — bu yüzden `approximate` işaretlenir.
    """
    p = prices[np.isfinite(prices)]
    if p.size == 0:
        return None
    if not 0 < min_fraction <= 1:
        raise ValueError("min_fraction (0, 1] aralığında olmalı")
    candidates = sorted(
        (Decimal(m) * Decimal(10) ** k for k in range(-_MAX_TICK_DECIMALS, 7) for m in _MANTISSAS),
        reverse=True,
    )
    for tick in candidates:
        ratio = p / float(tick)
        on_grid = np.abs(ratio - np.rint(ratio)) <= 1e-9 * np.maximum(1.0, np.abs(ratio))
        if on_grid.mean() >= min_fraction:
            return tick.normalize()
    return None
