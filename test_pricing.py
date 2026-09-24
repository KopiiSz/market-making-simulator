import math

import numpy as np
import pytest

from mmsim import MarketModel, bs_implied_vol, bs_price, simulate_paths, true_price
from mmsim.pricing import cf_call_price

K = np.array([70.0, 90.0, 100.0, 110.0, 140.0])


def test_cf_pricer_reduces_to_black_scholes():
    m = MarketModel("merton", lam=0.0)  # jump-free Merton goes through the CF pricer
    for T in (0.02, 0.25, 1.0):
        np.testing.assert_allclose(cf_call_price(100, K, T, m), bs_price(100, K, T, m.r, m.sigma),
                                   atol=1e-7)


def test_merton_matches_series():
    m = MarketModel("merton")
    T = 0.25
    k = m.jump_comp
    lp = m.lam * (1 + k)
    series = 0.0
    for n in range(60):
        sn = math.sqrt(m.sigma ** 2 + n * m.jump_std ** 2 / T)
        rn = m.r - m.lam * k + n * math.log(1 + k) / T
        series += math.exp(-lp * T) * (lp * T) ** n / math.factorial(n) * bs_price(100, K, T, rn, sn)
    np.testing.assert_allclose(true_price(100, K, T, m), series, atol=1e-6)


@pytest.mark.parametrize("kind", ["heston", "bates"])
def test_sv_models_match_monte_carlo(kind):
    m = MarketModel(kind, mu=0.02)  # mu = r -> simulate under Q
    T = 0.25
    rng = np.random.default_rng(7)
    ST = simulate_paths(m, 100, 200, T / 200, 100_000, rng).S[:, -1]
    ST *= 100 * np.exp(m.r * T) / ST.mean()  # moment-match the forward (variance reduction)
    mc = np.exp(-m.r * T) * np.maximum(ST[:, None] - K, 0).mean(0)
    np.testing.assert_allclose(true_price(100, K, T, m), mc, atol=0.04)


@pytest.mark.parametrize("kind", ["gbm", "merton", "heston", "bates"])
def test_put_call_parity(kind):
    m = MarketModel(kind)
    T = 0.5
    c = true_price(100, K, T, m, True)
    p = true_price(100, K, T, m, False)
    np.testing.assert_allclose(c - p, 100 - K * np.exp(-m.r * T), atol=1e-8)


def test_implied_vol_roundtrip():
    for sig in (0.05, 0.2, 0.8):
        px = bs_price(100, K[1:4], 0.3, 0.02, sig)
        np.testing.assert_allclose(bs_implied_vol(px, 100, K[1:4], 0.3, 0.02), sig, atol=1e-6)


def test_merton_has_negative_skew_smile():
    """Negative mean jumps -> OTM puts carry higher implied vol than OTM calls."""
    from mmsim import smile
    Kg, iv, _ = smile(MarketModel("merton", jump_mean=-0.1), 0.25, True, n=11)
    assert iv[0] > iv[-1]
    assert np.all(iv > 0.2)  # jumps add variance on top of the 20% diffusion
