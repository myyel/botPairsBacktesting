"""Trade kaydı (RAPOR Ek B.3). Maliyet bileşenleri AYRI tutulur (CLAUDE.md invariant #5)."""

from __future__ import annotations

from dataclasses import dataclass

from pairs_bt.domain.events import PairId, PairSide


@dataclass(frozen=True, slots=True)
class PairTradeRecord:
    pair: PairId
    entry_ts_ms: int
    exit_ts_ms: int
    side: PairSide
    beta_entry: float
    beta_exit: float
    z_entry: float
    z_exit: float
    half_life_min: float
    notional_a_usdt: float
    notional_b_usdt: float
    gross_pnl_usdt: float
    fees_usdt: float
    slippage_usdt: float
    funding_pnl_usdt: float
    leg_risk_cost_usdt: float
    mae_usdt: float
    mfe_usdt: float
    exit_reason: str
    formation_fold_id: int
    execution_model: str

    @property
    def net_pnl_usdt(self) -> float:
        """Net = brüt − fee − slippage + funding − leg-risk (funding işaretli nakit akışıdır)."""
        return (
            self.gross_pnl_usdt
            - self.fees_usdt
            - self.slippage_usdt
            + self.funding_pnl_usdt
            - self.leg_risk_cost_usdt
        )

    @property
    def holding_minutes(self) -> float:
        return (self.exit_ts_ms - self.entry_ts_ms) / 60_000
