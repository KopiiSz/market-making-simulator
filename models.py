"""Market models: parameters and physical-measure path simulation.

Four "true" dynamics are supported, all nested inside Bates (1996):

    GBM     dS/S = mu dt + sigma dW
    Merton  GBM + compound-Poisson log-normal jumps
    Heston  stochastic variance  dv = kappa (theta - v) dt + xi sqrt(v) dZ,  corr(dW, dZ) = rho
    Bates   Heston + Merton jumps

For Heston/Bates the long-run variance theta and the starting variance v0 are
both set to sigma**2, so "Vol sigma" means the same thing in every model.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

MODEL_LABELS = {
    "gbm": "Geometric Brownian Motion",
    "merton": "Merton jump-diffusion",
    "heston": "Heston stochastic vol",
    "bates": "Bates (Heston + jumps)",
}


@dataclass(frozen=True)
class MarketModel:
    kind: str = "gbm"          # gbm | merton | heston | bates
    mu: float = 0.05           # physical drift of S
    sigma: float = 0.20        # diffusion vol (sqrt(theta) = sqrt(v0) for SV models)
    r: float = 0.02            # risk-free rate
    lam: float = 1.0           # jump intensity per year
    jump_mean: float = -0.05   # mean of log jump size
    jump_std: float = 0.10     # std of log jump size
    kappa: float = 3.0         # variance mean reversion speed
    xi: float = 0.4            # vol of vol
    rho: float = -0.7          # spot/vol correlation

    # ---- convenience flags -------------------------------------------------
    @property
    def has_jumps(self) -> bool:
        return self.kind in ("merton", "bates") and self.lam > 0

    @property
    def has_sv(self) -> bool:
        return self.kind in ("heston", "bates")

    @property
    def theta(self) -> float:
        return self.sigma ** 2

    @property
    def v0(self) -> float:
        return self.sigma ** 2

    @property
    def jump_comp(self) -> float:
        """kappa_J = E[e^J] - 1, the jump compensator."""
        if not self.has_jumps:
            return 0.0
        return float(np.exp(self.jump_mean + 0.5 * self.jump_std ** 2) - 1.0)

    @property
    def total_vol(self) -> float:
        """Annualised stdev of log-returns including jumps (long-run)."""
        var = self.sigma ** 2
        if self.has_jumps:
            var += self.lam * (self.jump_mean ** 2 + self.jump_std ** 2)
        return float(np.sqrt(var))

    def label(self) -> str:
        return MODEL_LABELS[self.kind]

    def as_gbm(self, sigma: float) -> "MarketModel":
        """The world the Black-Scholes market maker *thinks* it lives in."""
        return replace(self, kind="gbm", sigma=sigma)


@dataclass
class Paths:
    S: np.ndarray        # (R, N+1) spot
    v: np.ndarray        # (R, N+1) instantaneous variance
    jumps: np.ndarray    # (R, N+1) log-jump that happened during (t-1, t]; 0 if none
    dt: float


def simulate_paths(model: MarketModel, S0: float, n_steps: int, dt: float,
                   n_paths: int, rng: np.random.Generator, substeps: int | None = None) -> Paths:
    """Simulate spot (and variance) under the physical measure P.

    Log-Euler for S, full-truncation Euler for v. Jumps are compensated so
    that E[S_t] = S0 * exp(mu t) in every model. Stochastic-vol models take
    several Euler sub-steps per tick for accuracy.
    """
    R, N = n_paths, n_steps
    m_sub = substeps or (4 if model.has_sv else 1)
    h = dt / m_sub
    sqh = np.sqrt(h)
    logS = np.empty((R, N + 1))
    v = np.empty((R, N + 1))
    J = np.zeros((R, N + 1))
    logS[:, 0] = np.log(S0)
    v[:, 0] = model.v0
    lamk = model.lam * model.jump_comp if model.has_jumps else 0.0

    ls = logS[:, 0].copy()
    vc = v[:, 0].copy()
    for t in range(N):
        for _ in range(m_sub):
            vp = np.maximum(vc, 0.0)
            z1 = rng.standard_normal(R)
            ls = ls + (model.mu - lamk - 0.5 * vp) * h + np.sqrt(vp) * sqh * z1
            if model.has_sv:
                z2 = model.rho * z1 + np.sqrt(1 - model.rho ** 2) * rng.standard_normal(R)
                vc = vc + model.kappa * (model.theta - vp) * h + model.xi * np.sqrt(vp) * sqh * z2
        if model.has_jumps:
            n = rng.poisson(model.lam * dt, R)
            jump = np.where(n > 0,
                            n * model.jump_mean
                            + np.sqrt(n) * model.jump_std * rng.standard_normal(R),
                            0.0)
            J[:, t + 1] = jump
            ls = ls + jump
        logS[:, t + 1] = ls
        v[:, t + 1] = vc

    return Paths(S=np.exp(logS), v=np.maximum(v, 0.0), jumps=J, dt=dt)
