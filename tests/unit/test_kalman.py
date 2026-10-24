"""KalmanHedgeModel (RAPOR §3.1): dinamik alpha/beta."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pairs_bt.models.kalman import KalmanHedgeModel, KalmanState
from tests.conftest import make_cointegrated_pair, make_ou_spread

FAR_INIT = KalmanState(alpha=0.0, beta=0.0, p_aa=1.0, p_ab=0.0, p_bb=1.0)


def test_converges_to_constant_beta(rng: np.random.Generator) -> None:
    log_a, log_b = make_cointegrated_pair(rng, 5_000, beta=1.3, alpha=0.1, sigma_b=0.02)
    model = KalmanHedgeModel(delta=1e-6, obs_var=4e-5)
    path = model.run(log_a, log_b, FAR_INIT)
    assert path.beta[-1] == pytest.approx(1.3, abs=0.05)
    assert path.final.beta == path.beta[-1]
    assert path.alpha.shape == path.beta.shape == path.innovation.shape == (5_000,)


def test_tracks_beta_jump(rng: np.random.Generator) -> None:
    n = 6_000
    log_b = np.cumsum(0.02 * rng.standard_normal(n)) + 4.0
    beta = np.where(np.arange(n) < n // 2, 1.0, 1.5)
    log_a = 0.2 + beta * log_b + make_ou_spread(rng, n, kappa=0.1, sigma=0.002)
    model = KalmanHedgeModel(delta=1e-4, obs_var=4e-5)
    path = model.run(log_a, log_b, FAR_INIT)
    assert path.beta[n // 2 - 1] == pytest.approx(1.0, abs=0.1)
    assert path.beta[-1] == pytest.approx(1.5, abs=0.1)


def test_tracks_beta_sign_change(rng: np.random.Generator) -> None:
    n = 6_000
    log_b = np.cumsum(0.02 * rng.standard_normal(n))
    beta = np.where(np.arange(n) < n // 2, 0.8, -0.8)
    log_a = beta * log_b + make_ou_spread(rng, n, kappa=0.1, sigma=0.002)
    path = KalmanHedgeModel(delta=1e-4, obs_var=4e-5).run(log_a, log_b, FAR_INIT)
    assert path.beta[n // 2 - 1] > 0.5
    assert path.beta[-1] < -0.5


def test_nan_observation_is_predict_only() -> None:
    model = KalmanHedgeModel(delta=1e-3, obs_var=1e-4)
    init = KalmanState(alpha=0.1, beta=1.0, p_aa=0.01, p_ab=0.0, p_bb=0.01)
    step = model.step(init, float("nan"), 2.0)
    assert np.isnan(step.innovation)
    assert (step.state.alpha, step.state.beta) == (0.1, 1.0)
    assert step.state.p_aa > init.p_aa
    assert step.state.p_bb > init.p_bb


def test_innovation_is_prior_residual() -> None:
    """Inovasyon, güncellemeden ÖNCEKİ state ile hesaplanır (aynı bar'ı tahmin için kullanmaz)."""
    model = KalmanHedgeModel(delta=1e-3, obs_var=1e-4)
    init = KalmanState(alpha=0.1, beta=1.0, p_aa=0.01, p_ab=0.0, p_bb=0.01)
    step = model.step(init, 2.5, 2.0)
    assert step.innovation == pytest.approx(2.5 - 0.1 - 1.0 * 2.0)
    assert step.innovation_var > 1e-4


def test_single_bar_and_empty() -> None:
    model = KalmanHedgeModel(delta=1e-4, obs_var=1e-4)
    one = model.run(np.array([1.0]), np.array([1.0]), FAR_INIT)
    assert one.beta.shape == (1,)
    empty = model.run(np.array([]), np.array([]), FAR_INIT)
    assert empty.beta.shape == (0,)
    assert empty.final == FAR_INIT


@pytest.mark.parametrize(("delta", "obs_var"), [(0.0, 1e-4), (1.0, 1e-4), (1e-4, 0.0)])
def test_invalid_params(delta: float, obs_var: float) -> None:
    with pytest.raises(ValueError, match=r"pozitif|aralığında"):
        KalmanHedgeModel(delta=delta, obs_var=obs_var)


def test_shape_mismatch_rejected() -> None:
    with pytest.raises(ValueError, match="aynı uzunlukta"):
        KalmanHedgeModel(1e-4, 1e-4).run(np.zeros(3), np.zeros(4), FAR_INIT)


def test_nan_in_b_leg_is_predict_only() -> None:
    model = KalmanHedgeModel(delta=1e-3, obs_var=1e-4)
    init = KalmanState(alpha=0.1, beta=1.0, p_aa=0.01, p_ab=0.0, p_bb=0.01)
    step = model.step(init, 2.0, float("nan"))
    assert np.isnan(step.innovation)
    assert np.isnan(step.innovation_var)
    assert (step.state.alpha, step.state.beta) == (0.1, 1.0)


def test_constant_b_leg_stays_finite(rng: np.random.Generator) -> None:
    """Sıfır volatilite: alpha/beta ayrıştırılamaz ama filtre patlamamalı (NaN/inf yok)."""
    n = 3_000
    log_b = np.full(n, 4.0)
    log_a = 0.5 + 1.2 * log_b + make_ou_spread(rng, n, kappa=0.2, sigma=0.002)
    path = KalmanHedgeModel(delta=1e-4, obs_var=4e-5).run(log_a, log_b, FAR_INIT)
    assert np.all(np.isfinite(path.alpha))
    assert np.all(np.isfinite(path.beta))
    # yalnız alpha + 4·beta tanımlıdır; tahmin o kombinasyonu bulmalı
    assert path.alpha[-1] + 4.0 * path.beta[-1] == pytest.approx(0.5 + 1.2 * 4.0, abs=0.01)


def test_step_equals_run(rng: np.random.Generator) -> None:
    log_a, log_b = make_cointegrated_pair(rng, 500, sigma_b=0.02)
    model = KalmanHedgeModel(delta=1e-4, obs_var=4e-5)
    path = model.run(log_a, log_b, FAR_INIT)
    state = FAR_INIT
    for t in range(500):
        st = model.step(state, float(log_a[t]), float(log_b[t]))
        state = st.state
        assert state.beta == path.beta[t]
        assert st.innovation == path.innovation[t]
    assert state == path.final


@settings(max_examples=100, deadline=None)
@given(
    seed=st.integers(0, 2**32 - 1),
    delta=st.floats(1e-8, 1e-2),
    obs_var=st.floats(1e-7, 1e-2),
    level=st.floats(-5.0, 12.0),
)
def test_covariance_stays_positive_definite(
    seed: int, delta: float, obs_var: float, level: float
) -> None:
    """Sayısal stabilite: posterior P simetrik pozitif tanımlı kalır (ln B ölçeğinden bağımsız)."""
    g = np.random.default_rng(seed)
    n = 400
    log_b = level + np.cumsum(0.01 * g.standard_normal(n))
    log_a = 0.1 + 0.9 * log_b + 0.003 * g.standard_normal(n)
    log_a[g.random(n) < 0.05] = np.nan  # ara sıra stale bar
    model = KalmanHedgeModel(delta=delta, obs_var=obs_var)
    state = FAR_INIT
    for t in range(n):
        state = model.step(state, float(log_a[t]), float(log_b[t])).state
        assert state.p_aa > 0
        assert state.p_bb > 0
        assert state.p_aa * state.p_bb - state.p_ab**2 > -1e-12 * max(1.0, state.p_aa * state.p_bb)
