"""Immutable event tipleri (RAPOR §7.1).

Zaman: tüm `ts_ms` alanları UTC epoch milisaniyedir (CLAUDE.md invariant #8).
Bu modülün `pairs_bt` içinde hiçbir bağımlılığı yoktur.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, StrEnum


class Side(StrEnum):
    BUY = "BUY"
    SELL = "SELL"


class PairSide(StrEnum):
    """LONG_SPREAD: A long / B short. SHORT_SPREAD: A short / B long."""

    LONG_SPREAD = "LONG_SPREAD"
    SHORT_SPREAD = "SHORT_SPREAD"


class Liquidity(StrEnum):
    MAKER = "MAKER"
    TAKER = "TAKER"


class PositionState(Enum):
    """RAPOR §9.3 state machine."""

    FLAT = "FLAT"
    ENTRY_PENDING = "ENTRY_PENDING"
    OPEN = "OPEN"
    EXIT_PENDING = "EXIT_PENDING"
    LEG_RISK = "LEG_RISK"
    FORCED_EXIT = "FORCED_EXIT"


@dataclass(frozen=True, slots=True)
class PairId:
    a: str
    b: str

    def __str__(self) -> str:
        return f"{self.a}|{self.b}"


@dataclass(frozen=True, slots=True)
class BarEvent:
    ts_ms: int  # bar kapanış zamanı (close_time); karar en erken bu anda verilir
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: float
    trades: int
    taker_buy_volume: float


@dataclass(frozen=True, slots=True)
class TradeEvent:
    ts_ms: int
    symbol: str
    price: float
    qty: float
    buyer_is_maker: bool


@dataclass(frozen=True, slots=True)
class PairRefreshEvent:
    ts_ms: int
    formation_cutoff_ms: int


@dataclass(frozen=True, slots=True)
class SignalEvent:
    ts_ms: int
    pair: PairId
    side: PairSide
    z: float
    beta: float
    expected_gross_bps: float
    expected_cost_bps: float
    confidence: float
    is_exit: bool = False

    @property
    def expected_net_bps(self) -> float:
        return self.expected_gross_bps - self.expected_cost_bps


@dataclass(frozen=True, slots=True)
class LegOrder:
    symbol: str
    side: Side
    qty: float
    limit_price: float | None  # None => market


@dataclass(frozen=True, slots=True)
class PairOrderEvent:
    ts_ms: int
    group_id: str
    pair: PairId
    leg_a: LegOrder
    leg_b: LegOrder
    timeout_ms: int


@dataclass(frozen=True, slots=True)
class FillEvent:
    ts_ms: int
    group_id: str
    symbol: str
    side: Side
    qty: float
    price: float
    fee_usdt: float
    slippage_usdt: float
    liquidity: Liquidity
    latency_ms: int


@dataclass(frozen=True, slots=True)
class FundingEvent:
    ts_ms: int
    symbol: str
    rate: float  # pozitif: long öder, short alır


class RiskAction(StrEnum):
    HALT_NEW_ENTRIES = "HALT_NEW_ENTRIES"
    REDUCE = "REDUCE"
    CLOSE_PAIR = "CLOSE_PAIR"
    RESUME = "RESUME"


@dataclass(frozen=True, slots=True)
class RiskEvent:
    ts_ms: int
    action: RiskAction
    reason: str
    pair: PairId | None = None
