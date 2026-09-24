"""The market-making engine.

Each tick the market maker (MM):

1. prices a rolling option contract (strike = moneyness x spot, maturity T)
   with Black-Scholes at its own vol,
2. quotes  bid/ask = BS fair + inventory skew -/+ half-spread,
3. meets at most one customer, who is either
      * a noise trader (buys with prob p_buy, otherwise sells), or
      * an informed trader who knows the TRUE model price and only trades
        when it is outside the MM's quotes,
4. delta-hedges the whole option book with the underlying (BS deltas),
5. settles expiring contracts at intrinsic value.

Quoting stops at the horizon; the book is then run off until the last contract
expires so every dollar of P/L is fully realised.

Everything is vectorised across independent runs, so the same code produces a
single animated path (n_runs=1) and a Monte Carlo P/L distribution.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .models import MarketModel, Paths, simulate_paths
from .pricing import bs_delta, bs_implied_vol, bs_price, true_price


@dataclass(frozen=True)
class MMParams:
    S0: float = 100.0
    strike: float = 100.0          # strike at S0; contracts roll at the same moneyness
    T: float = 0.25                # maturity of every new contract (years)
    is_call: bool = True
    half_spread: float = 0.15      # $ per contract
    quote_size: int = 1
    fill_intensity: float = 0.35   # prob. a customer shows up each tick
    informed_share: float = 0.30   # share of customers who know the true price
    p_buy: float = 0.50            # noise traders' prob. of buying
    risk_aversion: float = 0.02    # $ quote skew per contract of inventory
    mm_vol_mode: str = "auto"      # auto | calibrated | manual
    mm_vol_manual: float = 0.20
    horizon: float = 1.0           # years of quoting
    ticks: int = 250               # ticks over the horizon
    hedge_every: int = 1           # ticks between delta rebalances (0 = never hedge)

    @property
    def moneyness(self) -> float:
        return self.strike / self.S0


def mm_vol(model: MarketModel, p: MMParams) -> float:
    """The vol the Black-Scholes market maker plugs in."""
    if p.mm_vol_mode == "manual":
        return float(p.mm_vol_manual)
    if p.mm_vol_mode == "calibrated":
        dt = p.horizon / p.ticks
        T_eff = max(1, round(p.T / dt)) * dt
        tp = true_price(p.S0, p.strike, T_eff, model, p.is_call)
        iv = bs_implied_vol(tp, p.S0, p.strike, T_eff, model.r, p.is_call)
        return float(iv) if np.isfinite(iv) else model.sigma
    return float(model.sigma)  # auto: the diffusion vol, i.e. "the textbook sigma"


def true_unit_price_fn(model: MarketModel, p: MMParams, T_eff: float, v_max: float):
    """Return f(v) -> true price of the rolling contract per $1 of spot.

    Rolling strike = moneyness x spot makes the price homogeneous of degree 1
    in S, so the true value depends only on the current variance v. We price
    on a v-grid once and interpolate — this keeps Monte Carlo fast.
    """
    k = p.moneyness
    if not model.has_sv:
        c = float(true_price(1.0, k, T_eff, model, p.is_call))
        return lambda v: np.full(np.shape(v), c)
    grid = np.linspace(0.0, max(v_max, 4 * model.theta) * 1.05, 160)
    grid[0] = 1e-6
    vals = true_price(np.ones_like(grid), k, T_eff, model, p.is_call, v0=grid)
    return lambda v: np.interp(v, grid, vals)


@dataclass
class SimResult:
    params: MMParams
    model: MarketModel
    sigma_mm: float
    dt: float
    n_quote: int
    n_T: int
    time: np.ndarray                     # (N+1,)
    paths: Paths
    fair: np.ndarray                     # (R, n_quote) MM model value of new contract
    true_val: np.ndarray                 # (R, n_quote) true value of new contract
    bid: np.ndarray
    ask: np.ndarray
    trade_side: np.ndarray               # (R, n_quote) +1 MM bought, -1 MM sold, 0 none
    trade_informed: np.ndarray           # (R, n_quote) bool
    pnl: np.ndarray                      # (R, N+1) realised P/L, book marked at MM model
    model_edge_cum: np.ndarray           # (R, N+1) cumulative edge vs MM model
    true_edge_cum: np.ndarray            # (R, N+1) cumulative edge vs true value
    inventory: np.ndarray                # (R, N+1) net open option contracts
    hedge: np.ndarray                    # (R, N+1) shares held
    extra: dict = field(default_factory=dict)

    # ---- per-run summaries ------------------------------------------------
    @property
    def final_pnl(self):
        return self.pnl[:, -1]

    @property
    def spread_capture(self):
        return self.model_edge_cum[:, -1]

    @property
    def model_error(self):
        return self.true_edge_cum[:, -1] - self.model_edge_cum[:, -1]

    @property
    def hedge_residual(self):
        return self.pnl[:, -1] - self.true_edge_cum[:, -1]

    @property
    def n_contracts(self):
        return np.abs(self.trade_side).sum(axis=1) * self.params.quote_size


def run_simulation(model: MarketModel, p: MMParams, n_runs: int = 1,
                   seed: int | None = None, sigma_mm: float | None = None) -> SimResult:
    rng = np.random.default_rng(seed)
    dt = p.horizon / p.ticks
    n_quote = int(p.ticks)
    n_T = max(1, int(round(p.T / dt)))
    T_eff = n_T * dt
    N = n_quote + n_T
    R = n_runs
    r = model.r
    sig = mm_vol(model, p) if sigma_mm is None else float(sigma_mm)
    hs, Q = float(p.half_spread), int(p.quote_size)
    k = p.moneyness

    paths = simulate_paths(model, p.S0, N, dt, R, rng)
    S = paths.S
    f_true = true_unit_price_fn(model, p, T_eff, float(paths.v.max()))
    c_mm = float(bs_price(1.0, k, T_eff, r, sig, p.is_call))  # MM price per $ spot

    # customer draws (independent of the MM's behaviour)
    arrive = rng.random((R, n_quote)) < p.fill_intensity
    informed = rng.random((R, n_quote)) < p.informed_share
    noise_buy = rng.random((R, n_quote)) < p.p_buy

    q = np.zeros((R, n_quote))            # MM quantity by birth tick (+ long)
    Kb = np.zeros((R, n_quote))           # strike by birth tick
    cash = np.zeros(R)
    H = np.zeros(R)

    fair = np.zeros((R, n_quote)); tv = np.zeros((R, n_quote))
    bid = np.zeros((R, n_quote)); ask = np.zeros((R, n_quote))
    side = np.zeros((R, n_quote), dtype=np.int8)
    inf_fill = np.zeros((R, n_quote), dtype=bool)
    pnl = np.zeros((R, N + 1)); me = np.zeros((R, N + 1)); te = np.zeros((R, N + 1))
    inv = np.zeros((R, N + 1)); hed = np.zeros((R, N + 1))
    me_c = np.zeros(R); te_c = np.zeros(R)
    growth = np.exp(r * dt)

    for t in range(N + 1):
        St = S[:, t]
        if t > 0:
            cash *= growth
        # 1) expiries: contracts born at t - n_T settle at intrinsic value
        j_exp = t - n_T
        if 0 <= j_exp < n_quote:
            payoff = (np.maximum(St - Kb[:, j_exp], 0.0) if p.is_call
                      else np.maximum(Kb[:, j_exp] - St, 0.0))
            cash += q[:, j_exp] * payoff

        lo = max(0, t - n_T + 1)            # oldest still-open birth tick
        hi = min(t, n_quote - 1)            # newest birth tick (after trading)
        net_open = q[:, lo:hi + 1].sum(axis=1) if hi >= lo else np.zeros(R)

        # 2) quote & trade
        if t < n_quote:
            f = c_mm * St
            true_v = f_true(paths.v[:, t]) * St
            skew = -p.risk_aversion * net_open
            b = np.maximum(f + skew - hs, 0.0)
            a = f + skew + hs
            inf_buy = informed[:, t] & (true_v > a)
            inf_sell = informed[:, t] & (true_v < b)
            cust_buy = arrive[:, t] & np.where(informed[:, t], inf_buy, noise_buy[:, t])
            cust_sell = arrive[:, t] & np.where(informed[:, t], inf_sell, ~noise_buy[:, t])
            s = np.where(cust_buy, -1, np.where(cust_sell, 1, 0)).astype(np.int8)  # MM side
            px = np.where(s < 0, a, b)
            cash -= s * Q * px
            q[:, t] = s * Q
            Kb[:, t] = k * St
            me_c += np.where(s != 0, s * Q * (f - px), 0.0)
            te_c += np.where(s != 0, s * Q * (true_v - px), 0.0)
            fair[:, t], tv[:, t], bid[:, t], ask[:, t], side[:, t] = f, true_v, b, a, s
            inf_fill[:, t] = (s != 0) & informed[:, t]
            hi = t

        # 3) value open book with the MM's model; delta hedge on schedule
        if hi >= lo:
            births = np.arange(lo, hi + 1)
            tau = (births + n_T - t) * dt
            qq, KK = q[:, lo:hi + 1], Kb[:, lo:hi + 1]
            mtm = (qq * bs_price(St[:, None], KK, tau[None, :], r, sig, p.is_call)).sum(1)
            book_delta = (qq * bs_delta(St[:, None], KK, tau[None, :], r, sig, p.is_call)).sum(1)
            net_open = qq.sum(axis=1)
        else:
            mtm = np.zeros(R); book_delta = np.zeros(R); net_open = np.zeros(R)

        rebalance = p.hedge_every > 0 and (t % p.hedge_every == 0 or t >= N - 1)
        if rebalance:
            target = -book_delta
            cash -= (target - H) * St
            H = target

        pnl[:, t] = cash + H * St + mtm
        me[:, t], te[:, t] = me_c, te_c
        inv[:, t], hed[:, t] = net_open, H

    return SimResult(
        params=p, model=model, sigma_mm=sig, dt=dt, n_quote=n_quote, n_T=n_T,
        time=np.arange(N + 1) * dt, paths=paths, fair=fair, true_val=tv,
        bid=bid, ask=ask, trade_side=side, trade_informed=inf_fill, pnl=pnl,
        model_edge_cum=me, true_edge_cum=te, inventory=inv, hedge=hed,
        extra={"T_eff": T_eff, "c_mm": c_mm},
    )


# ---------------------------------------------------------------------------
# Analytics for the "Return Distributions" tab
# ---------------------------------------------------------------------------

def horizon_returns(model: MarketModel, T: float, dt: float, n_paths: int, seed=None):
    """Log-returns over the option maturity under the true model (physical measure)."""
    rng = np.random.default_rng(seed)
    n = max(1, int(round(T / dt)))
    paths = simulate_paths(model, 1.0, n, dt, n_paths, rng)
    return np.log(paths.S[:, -1]), n * dt


def smile(model: MarketModel, T: float, is_call: bool, S0: float = 100.0,
          m_lo: float = 0.75, m_hi: float = 1.25, n: int = 41):
    K = S0 * np.linspace(m_lo, m_hi, n)
    # OTM options are the numerically stable ones for implied vol
    calls = true_price(S0, K, T, model, True)
    puts = true_price(S0, K, T, model, False)
    use_call = K >= S0
    px = np.where(use_call, calls, puts)
    iv = np.where(use_call,
                  bs_implied_vol(calls, S0, K, T, model.r, True),
                  bs_implied_vol(puts, S0, K, T, model.r, False))
    return K, iv, px


def pnl_stats(x: np.ndarray) -> dict:
    x = np.asarray(x, float)
    q05 = np.quantile(x, 0.05)
    tail = x[x <= q05]
    return {
        "mean": float(x.mean()),
        "std": float(x.std(ddof=1)) if x.size > 1 else 0.0,
        "p_loss": float((x < 0).mean()),
        "var5": float(-q05),
        "cvar5": float(-tail.mean()) if tail.size else float(-q05),
        "worst": float(x.min()),
        "skew": float(((x - x.mean()) ** 3).mean() / (x.std() ** 3 + 1e-12)),
    }
