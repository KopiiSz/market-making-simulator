"""Option pricing.

* Black-Scholes closed form — what the market maker uses.
* A characteristic-function (Lewis 2001) pricer for GBM / Merton / Heston /
  Bates — the "true" risk-neutral value of the option in the simulated world.

The true models are priced under Q with the same parameters as under P
(i.e. no jump or volatility risk premia): only the drift changes, mu -> r.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr

from .models import MarketModel

# ---------------------------------------------------------------------------
# Black-Scholes
# ---------------------------------------------------------------------------

def _d1d2(S, K, tau, r, sigma):
    tau = np.maximum(tau, 1e-12)
    vs = np.maximum(sigma, 1e-12) * np.sqrt(tau)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * tau) / vs
    return d1, d1 - vs


def bs_price(S, K, tau, r, sigma, is_call: bool = True):
    S, K, tau = np.broadcast_arrays(np.asarray(S, float), np.asarray(K, float),
                                    np.asarray(tau, float))
    d1, d2 = _d1d2(S, K, tau, r, sigma)
    disc = np.exp(-r * np.maximum(tau, 0.0))
    call = S * ndtr(d1) - K * disc * ndtr(d2)
    price = call if is_call else call - S + K * disc
    intrinsic = np.maximum(S - K, 0.0) if is_call else np.maximum(K - S, 0.0)
    return np.where(tau <= 1e-12, intrinsic, price)


def bs_delta(S, K, tau, r, sigma, is_call: bool = True):
    S, K, tau = np.broadcast_arrays(np.asarray(S, float), np.asarray(K, float),
                                    np.asarray(tau, float))
    d1, _ = _d1d2(S, K, tau, r, sigma)
    delta = ndtr(d1) if is_call else ndtr(d1) - 1.0
    expired = tau <= 1e-12
    itm = (S > K) if is_call else (S < K)
    exp_delta = np.where(itm, 1.0 if is_call else -1.0, 0.0)
    return np.where(expired, exp_delta, delta)


def bs_implied_vol(price, S, K, tau, r, is_call: bool = True,
                   lo: float = 1e-4, hi: float = 3.0, iters: int = 80):
    """Vectorised bisection. Returns NaN where the price is outside no-arb bounds."""
    price, S, K = np.broadcast_arrays(np.asarray(price, float), np.asarray(S, float),
                                      np.asarray(K, float))
    a = np.full(price.shape, lo)
    b = np.full(price.shape, hi)
    for _ in range(iters):
        m = 0.5 * (a + b)
        too_high = bs_price(S, K, tau, r, m, is_call) > price
        b = np.where(too_high, m, b)
        a = np.where(too_high, a, m)
    iv = 0.5 * (a + b)
    pmin = bs_price(S, K, tau, r, lo, is_call)
    pmax = bs_price(S, K, tau, r, hi, is_call)
    return np.where((price < pmin - 1e-10) | (price > pmax + 1e-10), np.nan, iv)


# ---------------------------------------------------------------------------
# Characteristic functions of X_T = ln(S_T / S_0) - r T under Q
# ---------------------------------------------------------------------------

def log_cf(u, T: float, m: MarketModel, v0=None):
    """log E_Q[exp(i u X_T)], u complex (any shape); v0 broadcasts against u."""
    u = np.asarray(u, complex)
    iu = 1j * u
    if m.has_sv:
        v0 = m.v0 if v0 is None else v0
        k, th, xi, rho = m.kappa, m.theta, m.xi, m.rho
        b = k - rho * xi * iu
        d = np.sqrt(b ** 2 + xi ** 2 * (iu + u ** 2))
        g = (b - d) / (b + d)
        edT = np.exp(-d * T)
        C = k * th / xi ** 2 * ((b - d) * T - 2.0 * np.log((1 - g * edT) / (1 - g)))
        D = (b - d) / xi ** 2 * (1 - edT) / (1 - g * edT)
        out = C + D * v0
    else:
        out = -0.5 * m.sigma ** 2 * T * (iu + u ** 2)
    if m.has_jumps:
        out = out + m.lam * T * (np.exp(iu * m.jump_mean - 0.5 * m.jump_std ** 2 * u ** 2)
                                 - 1.0 - iu * m.jump_comp)
    return out


_GL_X, _GL_W = np.polynomial.legendre.leggauss(400)


def cf_call_price(S, K, T: float, m: MarketModel, v0=None):
    """European call via the Lewis (2001) single-integral formula.

    C = S - e^{-rT} sqrt(F K) / pi * int_0^inf Re[e^{i u x} phi(u - i/2)] / (u^2 + 1/4) du,
    x = ln(F / K), F = S e^{rT}.
    S, K, v0 broadcast together; T is a scalar.
    """
    S = np.asarray(S, float)
    K = np.asarray(K, float)
    v0_arr = np.asarray(m.v0 if v0 is None else v0, float)
    S, K, v0_arr = np.broadcast_arrays(S, K, v0_arr)

    # integration range from the slowest-decaying variance in play
    v_lo = max(min(m.sigma ** 2, float(np.min(v0_arr)) if m.has_sv else m.sigma ** 2), 4e-3)
    U = min(max(40.0, 14.0 / np.sqrt(v_lo * T)), 4000.0)
    u = 0.5 * U * (_GL_X + 1.0)
    w = 0.5 * U * _GL_W

    F = S * np.exp(m.r * T)
    x = np.log(F / K)[..., None]
    phi = np.exp(log_cf(u - 0.5j, T, m, v0_arr[..., None] if m.has_sv else None))
    integrand = np.real(np.exp(1j * u * x) * phi) / (u ** 2 + 0.25)
    integral = integrand @ w
    call = S - np.exp(-m.r * T) * np.sqrt(F * K) / np.pi * integral
    # clamp to no-arbitrage bounds (tiny quadrature noise deep ITM/OTM)
    lower = np.maximum(S - K * np.exp(-m.r * T), 0.0)
    return np.clip(call, lower, S)


def true_price(S, K, T: float, m: MarketModel, is_call: bool = True, v0=None):
    """True (model-consistent) price of a European option under market model m."""
    if m.kind == "gbm":
        return bs_price(S, K, T, m.r, m.sigma, is_call)
    call = cf_call_price(S, K, T, m, v0)
    if is_call:
        return call
    return call - np.asarray(S, float) + np.asarray(K, float) * np.exp(-m.r * T)
