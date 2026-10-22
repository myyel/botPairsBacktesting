"""Sembol işlem kuralları ve tick/step yuvarlama (CLAUDE.md invariant #9).

Yuvarlama `Decimal` ile yapılır; float girdiler önce `Decimal(str(x))` ile en kısa ondalık
gösterimine çevrilir, böylece `0.1 + 0.2` gibi binary artefaktlar bir step kaybettirmez.
Bu modülün `pairs_bt` içinde hiçbir bağımlılığı yoktur.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal

Number = float | int | Decimal


@dataclass(frozen=True, slots=True)
class SymbolFilters:
    """Bir sembolün `valid_from_ms` anından itibaren geçerli emir kuralları.

    `approximate=True`: kural o tarihe ait bir exchange-info snapshot'ından değil, türetilerek
    (ör. tickSize fiyat grid'inden) elde edilmiştir — bkz. `data.exchange_info`.
    """

    symbol: str
    tick_size: Decimal
    step_size: Decimal
    min_qty: Decimal
    min_notional: Decimal
    status: str
    valid_from_ms: int
    approximate: bool = False

    def __post_init__(self) -> None:
        for name in ("tick_size", "step_size"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{self.symbol}: {name} pozitif olmalı")
        for name in ("min_qty", "min_notional"):
            if getattr(self, name) < 0:
                raise ValueError(f"{self.symbol}: {name} negatif olamaz")


def _to_decimal(x: Number) -> Decimal:
    return x if isinstance(x, Decimal) else Decimal(str(x))


def _round_to_increment(x: Decimal, inc: Decimal, *, up: bool) -> Decimal:
    n = (x / inc).to_integral_value(rounding=ROUND_CEILING if up else ROUND_FLOOR)
    return (n * inc).quantize(inc)


def round_qty_down(qty: Number, filters: SymbolFilters) -> Decimal:
    """Miktarı stepSize'ın katına aşağı yuvarlar (asla fazla miktar emri verilmez)."""
    q = _to_decimal(qty)
    if q < 0:
        raise ValueError(f"qty negatif olamaz: {qty}")
    return _round_to_increment(q, filters.step_size, up=False)


def round_price_to_tick(price: Number, filters: SymbolFilters, *, up: bool) -> Decimal:
    """Fiyatı tickSize'ın katına yuvarlar. `up=True` yukarı, aksi halde aşağı.

    Yön seçimi çağıranın sorumluluğundadır (ör. konservatif buy limit için aşağı).
    """
    p = _to_decimal(price)
    if p < 0:
        raise ValueError(f"price negatif olamaz: {price}")
    return _round_to_increment(p, filters.tick_size, up=up)


def passes_min_rules(qty: Decimal, price: Decimal, filters: SymbolFilters) -> bool:
    """minQty ve minNotional kontrolü (qty > 0 zorunlu)."""
    return qty > 0 and qty >= filters.min_qty and qty * price >= filters.min_notional
