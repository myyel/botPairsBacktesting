"""Saf maliyet tahmin fonksiyonları (RAPOR §10.1, §10.3).

Formation'daki edge/maliyet ön filtresi (RAPOR §8.2 adım 6) ile Faz 3 cost gate'i (§3.4) AYNI
formülleri kullanır. Birimler: bps = 1e-4, notional USDT.
"""

from __future__ import annotations

import math


def liquidity_scaled_slippage_bps(
    *,
    base_bps: float,
    k1: float,
    k2: float,
    order_notional_usdt: float,
    bar_quote_volume_usdt: float,
    volatility_1m_bps: float,
    stress_addon_bps: float = 0.0,
) -> float:
    """RAPOR §10.3: base + k1·(order_notional / bar_quote_volume) + k2·volatility_1m + stress.

    `volatility_1m_bps`: 1 dakikalık log getirilerin std'si, bps cinsinden (VARSAYIM: k2 bu
    birimle çarpılır). `bar_quote_volume_usdt == 0` (likidite yok) -> sonsuz slippage.
    """
    values = (base_bps, k1, k2, order_notional_usdt, volatility_1m_bps, stress_addon_bps)
    if any(v < 0 for v in values) or bar_quote_volume_usdt < 0:
        raise ValueError("slippage girdileri negatif olamaz")
    if bar_quote_volume_usdt == 0:
        return math.inf
    participation = order_notional_usdt / bar_quote_volume_usdt
    return base_bps + k1 * participation + k2 * volatility_1m_bps + stress_addon_bps


def pair_round_trip_cost_bps(
    *,
    notional_a_usdt: float,
    notional_b_usdt: float,
    fee_bps: float,
    slippage_a_bps: float,
    slippage_b_bps: float,
    leg_risk_reserve_bps: float,
) -> float:
    """Bir pair round-trip'inin (giriş + çıkış, iki bacak) brüt notional'a oranla maliyeti, bps.

    Her bacak iki kez işlem görür; fee ve slippage o bacağın notional'ıyla ağırlıklanır.
    Funding dahil DEĞİLDİR (holding süresine bağlı; Faz 3 FundingModel ayrı kaydeder).
    """
    gross = notional_a_usdt + notional_b_usdt
    if notional_a_usdt < 0 or notional_b_usdt < 0 or gross <= 0:
        raise ValueError("bacak notional'ları negatif olamaz ve toplam pozitif olmalı")
    leg_cost = notional_a_usdt * (fee_bps + slippage_a_bps) + notional_b_usdt * (
        fee_bps + slippage_b_bps
    )
    return 2.0 * leg_cost / gross + leg_risk_reserve_bps
