"""costs.estimates — formation ve cost gate'in paylaştığı saf maliyet tahminleri (RAPOR §10)."""

from __future__ import annotations

import math

import pytest

from pairs_bt.costs.estimates import liquidity_scaled_slippage_bps, pair_round_trip_cost_bps


def test_slippage_formula() -> None:
    bps = liquidity_scaled_slippage_bps(
        base_bps=1.0,
        k1=50.0,
        k2=0.5,
        order_notional_usdt=1_000.0,
        bar_quote_volume_usdt=100_000.0,
        volatility_1m_bps=8.0,
        stress_addon_bps=0.5,
    )
    assert bps == pytest.approx(1.0 + 50.0 * 0.01 + 0.5 * 8.0 + 0.5)


def test_slippage_without_liquidity_is_infinite() -> None:
    bps = liquidity_scaled_slippage_bps(
        base_bps=1.0,
        k1=50.0,
        k2=0.0,
        order_notional_usdt=1_000.0,
        bar_quote_volume_usdt=0.0,
        volatility_1m_bps=0.0,
    )
    assert math.isinf(bps)


def test_slippage_rejects_negative_inputs() -> None:
    with pytest.raises(ValueError, match="negatif"):
        liquidity_scaled_slippage_bps(
            base_bps=1.0,
            k1=1.0,
            k2=1.0,
            order_notional_usdt=-1.0,
            bar_quote_volume_usdt=1.0,
            volatility_1m_bps=0.0,
        )


def test_round_trip_cost_weights_legs_by_notional() -> None:
    # G = 2000, beta = 1 -> 1000 / 1000; her bacak giriş+çıkış = 2 işlem
    cost = pair_round_trip_cost_bps(
        notional_a_usdt=1_000.0,
        notional_b_usdt=1_000.0,
        fee_bps=5.0,
        slippage_a_bps=1.0,
        slippage_b_bps=3.0,
        leg_risk_reserve_bps=1.5,
    )
    expected = (2 * 1_000 * (5 + 1) + 2 * 1_000 * (5 + 3)) / 2_000 + 1.5
    assert cost == pytest.approx(expected)


def test_round_trip_cost_zero_notional_rejected() -> None:
    with pytest.raises(ValueError, match="pozitif"):
        pair_round_trip_cost_bps(
            notional_a_usdt=0.0,
            notional_b_usdt=0.0,
            fee_bps=5.0,
            slippage_a_bps=1.0,
            slippage_b_bps=1.0,
            leg_risk_reserve_bps=0.0,
        )
