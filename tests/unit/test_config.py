from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from pairs_bt.config import DataConfig, StrategyConfig


def test_default_config_loads(strategy_config: StrategyConfig) -> None:
    assert strategy_config.signal.entry_z == pytest.approx(1.8)
    assert strategy_config.pair_selection.half_life_minutes == (3, 45)
    assert strategy_config.walk_forward.formation_days == 28
    assert strategy_config.costs.fees.taker_bps > 0


def test_config_hash_is_deterministic(strategy_config: StrategyConfig) -> None:
    clone = StrategyConfig.model_validate(strategy_config.model_dump())
    assert clone.config_hash() == strategy_config.config_hash()


def test_config_hash_changes_with_param(strategy_config: StrategyConfig) -> None:
    data = strategy_config.model_dump()
    data["signal"]["entry_z"] = 2.0
    assert StrategyConfig.model_validate(data).config_hash() != strategy_config.config_hash()


def test_config_is_frozen(strategy_config: StrategyConfig) -> None:
    with pytest.raises(ValidationError):
        strategy_config.signal.entry_z = 5.0  # type: ignore[misc]


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("signal", "exit_z", 2.0),  # exit >= entry
        ("signal", "stop_z", 1.0),  # stop <= entry
        ("risk", "daily_hard_stop_pct", -0.001),  # hard stop soft'tan gevşek
        ("pair_selection", "half_life_minutes", (45, 3)),
        ("pair_selection", "short_window_days", 60),
    ],
)
def test_invalid_values_rejected(
    strategy_config: StrategyConfig, section: str, key: str, value: object
) -> None:
    data = strategy_config.model_dump()
    data[section][key] = value
    with pytest.raises(ValidationError):
        StrategyConfig.model_validate(data)


def test_unknown_key_rejected(strategy_config: StrategyConfig) -> None:
    data = strategy_config.model_dump()
    data["signal"]["entry_zz"] = 1.0
    with pytest.raises(ValidationError):
        StrategyConfig.model_validate(data)


def test_score_weights_must_sum_to_one(strategy_config: StrategyConfig) -> None:
    data = strategy_config.model_dump()
    data["pair_selection"]["score_weights"]["liquidity"] = 0.5
    with pytest.raises(ValidationError):
        StrategyConfig.model_validate(data)


def test_walk_forward_requires_utc(strategy_config: StrategyConfig) -> None:
    data = strategy_config.model_dump()
    data["walk_forward"]["start"] = "2025-10-01T00:00:00+03:00"
    with pytest.raises(ValidationError):
        StrategyConfig.model_validate(data)


def test_max_leverage_capped_at_three(strategy_config: StrategyConfig) -> None:
    data = strategy_config.model_dump()
    data["risk"]["max_leverage"] = 5.0
    with pytest.raises(ValidationError):
        StrategyConfig.model_validate(data)


# ---------------------------------------------------------------- DataConfig (Faz 1)


def test_data_config_loads(data_config: DataConfig) -> None:
    assert data_config.start_date == date(2025, 9, 1)
    assert data_config.end_date == date(2026, 9, 30)
    assert set(data_config.datasets) == {"klines_1m", "klines_5m", "fundingRate"}
    assert data_config.universe.contract_types == ("PERPETUAL",)
    assert data_config.universe.quote_asset == "USDT"


def test_data_config_rejects_reversed_dates(data_config: DataConfig) -> None:
    data = data_config.model_dump()
    data["start_date"], data["end_date"] = data["end_date"], data["start_date"]
    with pytest.raises(ValidationError):
        DataConfig.model_validate(data)


def test_data_config_rejects_unknown_dataset(data_config: DataConfig) -> None:
    data = data_config.model_dump()
    data["datasets"] = ["klines_1m", "bookTicker"]
    with pytest.raises(ValidationError):
        DataConfig.model_validate(data)


def test_data_config_rejects_duplicate_dataset(data_config: DataConfig) -> None:
    data = data_config.model_dump()
    data["datasets"] = ["klines_1m", "klines_1m"]
    with pytest.raises(ValidationError):
        DataConfig.model_validate(data)


def test_data_config_hash_is_deterministic(data_config: DataConfig) -> None:
    clone = DataConfig.model_validate(data_config.model_dump())
    assert clone.config_hash() == data_config.config_hash()


# ---------------------------------------------------------------- Faz 2 alanları


def test_research_fields_load(strategy_config: StrategyConfig) -> None:
    ps = strategy_config.pair_selection
    assert ps.eg_direction == "min_p"
    assert ps.beta_bounds == (0.2, 5.0)
    assert ps.adf_lags >= 0
    assert ps.hurst_lags[0] < ps.hurst_lags[1]
    assert ps.stability_subwindows >= 2
    m = strategy_config.market
    assert 0 < m.min_coverage <= 1
    assert 0 <= m.max_zero_trade_frac < 1
    assert "USDCUSDT" in m.exclude_symbols


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("beta_bounds", (5.0, 0.2)),
        ("beta_bounds", (-1.0, 2.0)),
        ("hurst_lags", (10, 2)),
        ("hurst_lags", (1, 10)),
        ("stability_subwindows", 1),
        ("max_coint_candidates", 0),
    ],
)
def test_invalid_research_values_rejected(
    strategy_config: StrategyConfig, key: str, value: object
) -> None:
    data = strategy_config.model_dump()
    data["pair_selection"][key] = value
    with pytest.raises(ValidationError):
        StrategyConfig.model_validate(data)


@pytest.mark.parametrize(("key", "value"), [("delta", 1.0), ("delta", 0.0), ("obs_var", 0.0)])
def test_invalid_kalman_values_rejected(
    strategy_config: StrategyConfig, key: str, value: float
) -> None:
    data = strategy_config.model_dump()
    data["models"]["kalman"][key] = value
    with pytest.raises(ValidationError):
        StrategyConfig.model_validate(data)


def test_ewma_deviation_requires_hl_max_below_ewma_halflife(
    strategy_config: StrategyConfig,
) -> None:
    """RW'nin EWMA sapma HL'i = EWMA HL; üst sınır onun altında değilse filtre anlamsızdır."""
    data = strategy_config.model_dump()
    assert data["pair_selection"]["half_life_method"] == "ewma_deviation"
    data["pair_selection"]["half_life_minutes"] = (3, 90)  # EWMA HL = 60 × 1m
    with pytest.raises(ValidationError, match="EWMA"):
        StrategyConfig.model_validate(data)
    data["pair_selection"]["half_life_method"] = "level"
    StrategyConfig.model_validate(data)


@pytest.mark.parametrize(("key", "value"), [("jump_mad_k", 0.0), ("max_jump_frac", 1.5)])
def test_invalid_jump_values_rejected(
    strategy_config: StrategyConfig, key: str, value: float
) -> None:
    data = strategy_config.model_dump()
    data["pair_selection"][key] = value
    with pytest.raises(ValidationError):
        StrategyConfig.model_validate(data)
