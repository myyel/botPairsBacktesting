from __future__ import annotations

import dataclasses

import pytest

from pairs_bt.domain import PairId, PairSide, PairTradeRecord, SignalEvent


def _record(**overrides: object) -> PairTradeRecord:
    base: dict[str, object] = {
        "pair": PairId("ETHUSDT", "BTCUSDT"),
        "entry_ts_ms": 0,
        "exit_ts_ms": 30 * 60_000,
        "side": PairSide.LONG_SPREAD,
        "beta_entry": 1.2,
        "beta_exit": 1.21,
        "z_entry": -2.0,
        "z_exit": -0.2,
        "half_life_min": 12.0,
        "notional_a_usdt": 500.0,
        "notional_b_usdt": 600.0,
        "gross_pnl_usdt": 10.0,
        "fees_usdt": 2.0,
        "slippage_usdt": 1.0,
        "funding_pnl_usdt": -0.5,
        "leg_risk_cost_usdt": 0.25,
        "mae_usdt": -3.0,
        "mfe_usdt": 11.0,
        "exit_reason": "convergence",
        "formation_fold_id": 0,
        "execution_model": "hybrid",
    }
    base.update(overrides)
    return PairTradeRecord(**base)  # type: ignore[arg-type]


def test_net_pnl_decomposition() -> None:
    assert _record().net_pnl_usdt == pytest.approx(10.0 - 2.0 - 1.0 - 0.5 - 0.25)


def test_holding_minutes() -> None:
    assert _record().holding_minutes == pytest.approx(30.0)


def test_records_are_immutable() -> None:
    rec = _record()
    with pytest.raises(dataclasses.FrozenInstanceError):
        rec.fees_usdt = 0.0  # type: ignore[misc]


def test_signal_net_edge() -> None:
    sig = SignalEvent(
        ts_ms=1,
        pair=PairId("A", "B"),
        side=PairSide.SHORT_SPREAD,
        z=2.1,
        beta=0.9,
        expected_gross_bps=30.0,
        expected_cost_bps=9.0,
        confidence=0.7,
    )
    assert sig.expected_net_bps == pytest.approx(21.0)
    assert str(sig.pair) == "A|B"
