"""Config şeması: YAML -> frozen pydantic modelleri (RAPOR §3.5, §14.2).

Tüm eşikler buradan gelir; kodda sihirli sayı kullanılmaz.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class RunConfig(_Frozen):
    name: str
    seed: int = 42
    initial_equity_usdt: float = Field(gt=0)


class MarketConfig(_Frozen):
    venue: Literal["binance_um"] = "binance_um"
    quote_asset: str = "USDT"
    execution_interval: Literal["1m", "5m"] = "1m"
    formation_interval: Literal["1m", "5m"] = "5m"
    max_ffill_bars: int = Field(ge=0, default=3)
    min_listing_age_days: int = Field(ge=0, default=30)
    min_quote_volume_24h_usdt: float = Field(ge=0, default=0.0)
    exclude_symbols: tuple[str, ...] = ()


class ScoreWeights(_Frozen):
    coint_stability: float
    mean_reversion_quality: float
    expected_net_edge: float
    liquidity: float
    beta_stability: float
    diversification: float

    @model_validator(mode="after")
    def _sum_to_one(self) -> ScoreWeights:
        total = sum(self.model_dump().values())
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"score_weights toplamı 1 olmalı, bulunan: {total}")
        return self


class PairSelectionConfig(_Frozen):
    short_window_days: int = Field(gt=0)
    long_window_days: int = Field(gt=0)
    refresh_hours: int = Field(gt=0)
    adf_p_short: float = Field(gt=0, lt=1)
    adf_p_long: float = Field(gt=0, lt=1)
    min_corr: float = Field(ge=-1, le=1)
    max_hurst: float = Field(gt=0, lt=1)
    half_life_minutes: tuple[float, float]
    active_pairs: int = Field(gt=0)
    max_pairs_per_asset: int = Field(gt=0)
    score_weights: ScoreWeights

    @model_validator(mode="after")
    def _check(self) -> PairSelectionConfig:
        lo, hi = self.half_life_minutes
        if not 0 < lo < hi:
            raise ValueError("half_life_minutes [min, max] ve 0 < min < max olmalı")
        if self.short_window_days >= self.long_window_days:
            raise ValueError("short_window_days < long_window_days olmalı")
        return self


class KalmanConfig(_Frozen):
    delta: float = Field(gt=0)
    obs_var: float = Field(gt=0)


class ZScoreConfig(_Frozen):
    ewma_halflife_bars: int = Field(gt=0)


class ModelsConfig(_Frozen):
    kalman: KalmanConfig
    zscore: ZScoreConfig


class SignalConfig(_Frozen):
    entry_z: float = Field(gt=0)
    exit_z: float = Field(ge=0)
    stop_z: float = Field(gt=0)
    min_net_edge_bps: float = Field(ge=0)
    cost_multiplier: float = Field(ge=1)
    max_holding_hl_mult: float = Field(gt=0)
    max_holding_minutes: float = Field(gt=0)
    require_reversion_trigger: bool = True

    @model_validator(mode="after")
    def _ordered(self) -> SignalConfig:
        if not self.exit_z < self.entry_z < self.stop_z:
            raise ValueError("exit_z < entry_z < stop_z olmalı")
        return self


class RiskConfig(_Frozen):
    pair_risk_pct: float = Field(gt=0, lt=0.05)
    max_open_pairs: int = Field(gt=0)
    daily_soft_stop_pct: float = Field(lt=0)
    daily_hard_stop_pct: float = Field(lt=0)
    max_leverage: float = Field(gt=0, le=3)  # RAPOR §11.2: 1-3x

    @model_validator(mode="after")
    def _stops(self) -> RiskConfig:
        if not self.daily_hard_stop_pct < self.daily_soft_stop_pct:
            raise ValueError("daily_hard_stop_pct, soft stop'tan daha negatif olmalı")
        return self


class ExecutionConfig(_Frozen):
    model: Literal["conservative_taker", "maker_touch", "hybrid"] = "hybrid"
    maker_timeout_seconds: float = Field(ge=0)
    leg_timeout_seconds: float = Field(ge=0)
    leg_timeout_action: Literal["aggress", "unwind"] = "aggress"
    latency_ms: int = Field(ge=0)
    queue_haircut: float = Field(ge=0, le=1)


class FeesConfig(_Frozen):
    source: Literal["config", "account_tier"] = "config"
    maker_bps: float
    taker_bps: float = Field(ge=0)
    bnb_discount: bool = False


class SlippageConfig(_Frozen):
    model: Literal["fixed", "liquidity_scaled"] = "liquidity_scaled"
    base_bps: float = Field(ge=0)
    k1: float = Field(ge=0)
    k2: float = Field(ge=0)
    stress_addon_bps: float = Field(ge=0, default=0.0)


class FundingConfig(_Frozen):
    enabled: bool = True
    interval_hours: int = Field(gt=0, default=8)


class LegRiskConfig(_Frozen):
    reserve_bps: float = Field(ge=0)


class StressScenario(_Frozen):
    fee_mult: float = 1.0
    slippage_mult: float = 1.0
    force_taker: bool = False


class CostsConfig(_Frozen):
    fees: FeesConfig
    slippage: SlippageConfig
    funding: FundingConfig
    leg_risk: LegRiskConfig
    stress_scenarios: dict[str, StressScenario] = Field(default_factory=dict)


class WalkForwardConfig(_Frozen):
    start: datetime
    end: datetime
    formation_days: int = Field(gt=0)
    test_days: int = Field(gt=0)
    step_days: int = Field(gt=0)
    embargo_minutes: int = Field(ge=0, default=0)
    open_position_policy_on_refresh: Literal["grandfather", "force_exit"] = "grandfather"

    @model_validator(mode="after")
    def _utc(self) -> WalkForwardConfig:
        for ts in (self.start, self.end):
            if ts.tzinfo is None or ts.utcoffset() is None or ts.utcoffset().total_seconds() != 0:  # type: ignore[union-attr]
                raise ValueError("walk-forward start/end UTC (tz-aware, offset 0) olmalı")
        if self.start >= self.end:
            raise ValueError("start < end olmalı")
        return self


class StrategyConfig(_Frozen):
    run: RunConfig
    market: MarketConfig
    pair_selection: PairSelectionConfig
    models: ModelsConfig
    signal: SignalConfig
    risk: RiskConfig
    execution: ExecutionConfig
    costs: CostsConfig
    walk_forward: WalkForwardConfig

    def config_hash(self) -> str:
        """Çözümlenmiş config'in deterministik SHA-256 hash'i (reproducibility, CLAUDE.md §8)."""
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path} bir YAML mapping olmalı")
    return data


def load_config(path: str | Path, root: str | Path | None = None) -> StrategyConfig:
    """Strateji YAML'ını ve referans verdiği costs/walk-forward dosyalarını yükler.

    `costs_file` ve `walk_forward_file` yolları `root`'a (varsayılan: proje kökü = config
    klasörünün üstü) göre çözülür.
    """
    path = Path(path)
    base = Path(root) if root is not None else path.resolve().parent.parent
    raw = _read_yaml(path)
    costs_file = raw.pop("costs_file")
    wf_file = raw.pop("walk_forward_file")
    raw["costs"] = _read_yaml(base / costs_file)
    raw["walk_forward"] = _read_yaml(base / wf_file)
    return StrategyConfig.model_validate(raw)


# ===================================================================== Data pipeline (Faz 1)

DatasetName = Literal["klines_1m", "klines_5m", "aggTrades", "fundingRate"]


class UniverseConfig(_Frozen):
    """İndirilecek sembol evreni (RAPOR §8.1, CLAUDE.md invariant #3).

    `include_symbols` boşsa arşiv listing'indeki (delist edilenler dahil) tüm semboller,
    quote/contract filtresinden geçenler alınır.
    """

    quote_asset: str = "USDT"
    contract_types: tuple[str, ...] = ("PERPETUAL",)
    include_symbols: tuple[str, ...] = ()
    exclude_symbols: tuple[str, ...] = ()


class DataPathsConfig(_Frozen):
    raw_dir: Path = Path("data/raw")
    normalized_dir: Path = Path("data/normalized")
    parquet_dir: Path = Path("data/parquet")
    manifests_dir: Path = Path("data/manifests")


class DownloadConfig(_Frozen):
    archive_base_url: str = "https://data.binance.vision"
    s3_listing_url: str = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
    exchange_info_url: str = "https://fapi.binance.com/fapi/v1/exchangeInfo"
    max_workers: int = Field(gt=0, default=8)
    max_retries: int = Field(ge=0, default=5)
    backoff_base_seconds: float = Field(ge=0, default=1.0)
    timeout_seconds: float = Field(gt=0, default=60.0)


class ValidationConfig(_Frozen):
    crosscheck_price_rel_tol: float = Field(ge=0, default=1e-9)
    crosscheck_volume_rel_tol: float = Field(ge=0, default=1e-6)
    max_examples: int = Field(gt=0, default=20)  # rapor başına örnek ihlal sayısı
    # türetilen tickSize, fiyatların en az bu oranını kapsamalı (aykırı alt-tick fill'lere tolerans)
    tick_grid_min_fraction: float = Field(gt=0, le=1, default=0.999)


class DataConfig(_Frozen):
    """Veri indirme/derleme config'i. Tarihler UTC gün sınırlarıdır (dahil)."""

    start_date: date
    end_date: date
    datasets: tuple[DatasetName, ...]
    universe: UniverseConfig = UniverseConfig()
    paths: DataPathsConfig = DataPathsConfig()
    download: DownloadConfig = DownloadConfig()
    validation: ValidationConfig = ValidationConfig()

    @model_validator(mode="after")
    def _check(self) -> DataConfig:
        if self.start_date > self.end_date:
            raise ValueError("start_date <= end_date olmalı")
        if len(set(self.datasets)) != len(self.datasets):
            raise ValueError("datasets tekrar içeremez")
        return self

    def config_hash(self) -> str:
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


def load_data_config(path: str | Path) -> DataConfig:
    return DataConfig.model_validate(_read_yaml(Path(path)))
