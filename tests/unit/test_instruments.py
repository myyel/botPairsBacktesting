from __future__ import annotations

from decimal import Decimal

import pytest
from hypothesis import given
from hypothesis import strategies as st

from pairs_bt.domain.instruments import (
    SymbolFilters,
    passes_min_rules,
    round_price_to_tick,
    round_qty_down,
)


def _filters(**overrides: object) -> SymbolFilters:
    base: dict[str, object] = {
        "symbol": "BTCUSDT",
        "tick_size": Decimal("0.10"),
        "step_size": Decimal("0.001"),
        "min_qty": Decimal("0.001"),
        "min_notional": Decimal("50"),
        "status": "TRADING",
        "valid_from_ms": 0,
    }
    base.update(overrides)
    return SymbolFilters(**base)  # type: ignore[arg-type]


TICKS = st.sampled_from([Decimal(x) for x in ("0.1", "0.01", "0.0001", "0.000001", "1", "5")])
STEPS = st.sampled_from([Decimal(x) for x in ("1", "0.1", "0.001", "10")])
POS_FLOATS = st.floats(min_value=1e-6, max_value=1e7, allow_nan=False, allow_infinity=False)


@given(qty=POS_FLOATS, step=STEPS)
def test_round_qty_down_is_multiple_of_step(qty: float, step: Decimal) -> None:
    f = _filters(step_size=step, min_qty=step)
    q = round_qty_down(qty, f)
    assert q % step == 0
    assert q <= Decimal(str(qty))
    assert Decimal(str(qty)) - q < step


@given(price=POS_FLOATS, tick=TICKS, up=st.booleans())
def test_round_price_is_multiple_of_tick(price: float, tick: Decimal, up: bool) -> None:
    f = _filters(tick_size=tick)
    p = round_price_to_tick(price, f, up=up)
    assert p % tick == 0
    exact = Decimal(str(price))
    assert abs(p - exact) < tick
    assert (p >= exact) if up else (p <= exact)


def test_round_exact_multiple_unchanged() -> None:
    f = _filters()
    assert round_price_to_tick(108221.4, f, up=True) == Decimal("108221.4")
    assert round_price_to_tick(108221.4, f, up=False) == Decimal("108221.4")
    assert round_qty_down(0.3, f) == Decimal("0.300")


def test_float_artifacts_do_not_lose_a_step() -> None:
    # 0.1 + 0.2 = 0.30000000000000004 — binary artefakt bir step kaybettirmemeli
    assert round_qty_down(0.1 + 0.2, _filters(step_size=Decimal("0.1"))) == Decimal("0.3")


@pytest.mark.parametrize(
    ("qty", "price", "ok"),
    [
        (Decimal("0.001"), Decimal("50000"), True),  # 50 >= 50
        (Decimal("0.001"), Decimal("49999.9"), False),  # notional < 50
        (Decimal("0.0005"), Decimal("1000000"), False),  # qty < minQty
        (Decimal("0"), Decimal("100000"), False),
    ],
)
def test_passes_min_rules(qty: Decimal, price: Decimal, ok: bool) -> None:
    assert passes_min_rules(qty, price, _filters()) is ok


@pytest.mark.parametrize("field", ["tick_size", "step_size"])
def test_non_positive_increment_rejected(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        _filters(**{field: Decimal("0")})


def test_negative_min_rejected() -> None:
    with pytest.raises(ValueError, match="min_notional"):
        _filters(min_notional=Decimal("-1"))


def test_zero_qty_rounds_to_zero() -> None:
    assert round_qty_down(0.0, _filters()) == 0


def test_negative_inputs_rejected() -> None:
    with pytest.raises(ValueError, match="qty"):
        round_qty_down(-1.0, _filters())
    with pytest.raises(ValueError, match="price"):
        round_price_to_tick(-1.0, _filters(), up=True)
