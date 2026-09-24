"""Market Making Simulator — model risk in options market making.

Run locally:   streamlit run app.py
"""
from __future__ import annotations


import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from mmsim import (MODEL_LABELS, MarketModel, MMParams, bs_implied_vol, bs_price, horizon_returns,
                   mm_vol, pnl_stats, run_simulation, smile, true_price)
from mmsim import charts
from mmsim.animation import player_html

st.set_page_config(page_title="Market Making Simulator", page_icon=":material/candlestick_chart:",
                   layout="wide", initial_sidebar_state="auto")

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
:root { --ink:#0b0b0b; --ink2:#52514e; --muted:#898781; --line:#e6e5df; --card:#fcfcfb;
        --pos:#006300; --neg:#c12f2f; --accent:#0e9f6e; }
html, body, [class*="css"], .stMarkdown, .stText, button, input, select, label { font-family: Inter, system-ui, -apple-system, "Segoe UI", sans-serif; }
.block-container { padding-top: 2.6rem; padding-bottom: 3rem; max-width: 1500px; }
section[data-testid="stSidebar"] { min-width: 340px; border-right: 1px solid var(--line); }
section[data-testid="stSidebar"] .block-container, section[data-testid="stSidebar"] > div { padding-top: 0.6rem; }
section[data-testid="stSidebar"] input { font-family: "JetBrains Mono", ui-monospace, monospace; font-size: 0.86rem; }
.mm-section { font-size: 0.7rem; font-weight: 700; letter-spacing: 0.1em; color: var(--muted);
              text-transform: uppercase; margin: 1.1rem 0 0.2rem; padding-bottom: 0.3rem; border-bottom: 1px solid var(--line); }
.mm-header { display:flex; align-items:center; gap:14px; padding: 4px 0 14px; border-bottom:1px solid var(--line); margin-bottom: 10px; }
.mm-logo { width:30px; height:30px; border-radius:6px; background: linear-gradient(90deg, var(--accent) 0 45%, #b7ead7 45% 100%); flex:none; }
.mm-title { font-size: 1.45rem; font-weight: 700; color: var(--ink); line-height:1.1; }
.mm-sub { font-size: 0.86rem; color: var(--ink2); margin-top:2px; }
.mm-cards { display:grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px; margin: 6px 0 12px; }
.mm-card { background: var(--card); border:1px solid var(--line); border-radius:10px; padding: 10px 14px; }
.mm-card .k { font-size: 0.66rem; font-weight:700; letter-spacing:0.09em; color: var(--muted); text-transform: uppercase; }
.mm-card .v { font-family: "JetBrains Mono", ui-monospace, monospace; font-size: 1.15rem; font-weight:600; color: var(--ink); margin-top: 3px; white-space: nowrap; }
.mm-card .s { font-size: 0.72rem; color: var(--muted); margin-top: 1px; }
.pos { color: var(--pos) !important; } .neg { color: var(--neg) !important; }
.mm-banner { border-radius: 8px; padding: 9px 12px; font-size: 0.8rem; line-height: 1.4; margin: 8px 0 2px; }
.mm-ok { background:#ecf8f1; border:1px solid #9fd8b8; color:#0d5b2f; }
.mm-warn { background:#fff6e8; border:1px solid #f3c77a; color:#6b4300; }
.mm-summary { background: var(--card); border:1px solid var(--line); border-radius: 10px; padding: 14px 18px; font-size: 0.92rem; color: var(--ink); line-height:1.55; }
.mm-summary b { font-weight: 600; }
.mm-summary code, .mono { font-family: "JetBrains Mono", ui-monospace, monospace; font-size: 0.85em; }
div[data-testid="stTabs"] button p { font-family: "JetBrains Mono", ui-monospace, monospace; font-size: 0.9rem; }
button[data-testid="stNumberInputStepUp"], button[data-testid="stNumberInputStepDown"] { display: none; }
div[data-testid="stPlotlyChart"] { background:#ffffff; border:1px solid var(--line); border-radius: 10px; padding: 6px 6px 0; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# State & presets
# ---------------------------------------------------------------------------
VOL_MODES = {
    "Auto (= true diffusion σ)": "auto",
    "Calibrated to the true price": "calibrated",
    "Manual": "manual",
}
DEFAULTS = dict(
    true_model="gbm", opt_type="Call", S0=100.0, K=100.0, T=0.25,
    mu=0.05, sigma=0.20, r=0.02, lam=1.0, jump_mean=-0.05, jump_std=0.10,
    kappa=3.0, xi=0.4, rho=-0.7,
    half_spread=0.15, quote_size=1, fill=0.35, risk_aversion=0.02,
    informed=0.30, p_buy=0.50, vol_mode="Auto (= true diffusion σ)", mm_vol_manual=0.20,
    horizon=1.0, ticks=250, hedge_every=1, seed="42", speed=30, animate=True, mc_runs=400,
)
PRESETS = {
    "matched": dict(),
    "steamroller": dict(true_model="merton", opt_type="Put", K=95.0, lam=0.4, jump_mean=-0.15,
                        jump_std=0.05, informed=0.05, p_buy=0.8, seed="0"),
    "vol clustering": dict(true_model="heston", kappa=3.0, xi=0.5, rho=-0.7, seed="11"),
    "crash + vol": dict(true_model="bates", lam=1.0, jump_mean=-0.08, jump_std=0.08, xi=0.4, seed="9"),
}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)
st.session_state.setdefault("res", None)
st.session_state.setdefault("player", None)
st.session_state.setdefault("mc", None)


def apply_preset(name: str):
    for k, v in {**DEFAULTS, **PRESETS[name]}.items():
        st.session_state[k] = v
    st.session_state.run_request = True


def request_run():
    st.session_state.run_request = True


def section(title: str, where=st):
    where.markdown(f'<div class="mm-section">{title}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
sb = st.sidebar
sb.markdown("### Controls")

section("Models", sb)
sb.selectbox("MM pricing model", ["Black-Scholes (GBM)"], disabled=True,
             help="The market maker always prices and hedges with Black-Scholes. "
                  "That is the point: what happens when reality isn't GBM?")
sb.selectbox("True market-path model", list(MODEL_LABELS), key="true_model",
             format_func=lambda k: MODEL_LABELS[k])
sb.selectbox("MM volatility input", list(VOL_MODES), key="vol_mode",
             help="Auto: the MM uses the true diffusion σ (the 'textbook' vol).\n\n"
                  "Calibrated: the MM backs out the Black-Scholes implied vol from the true price "
                  "of its contract at t=0, so it starts with zero mispricing, but a single flat vol "
                  "still can't capture jumps or moving volatility.\n\nManual: pick any vol.")
if VOL_MODES[st.session_state.vol_mode] == "manual":
    sb.number_input("MM vol", min_value=0.01, max_value=2.0, step=0.01, key="mm_vol_manual", format="%.3f")
banner_slot = sb.empty()

section("Contract", sb)
c1, c2 = sb.columns(2)
c1.number_input("S₀ (spot)", min_value=1.0, step=1.0, key="S0")
c2.number_input("Strike K", min_value=0.01, step=1.0, key="K",
                help="Strike at S₀. New contracts are struck at the same moneyness K/S₀ of the current spot.")
c1, c2 = sb.columns(2)
c1.number_input("Rolling maturity T (yrs)", min_value=0.02, max_value=2.0, step=0.05, key="T", format="%.2f")
c2.selectbox("Type", ["Call", "Put"], key="opt_type")

section("True dynamics", sb)
kind = st.session_state.true_model
c1, c2 = sb.columns(2)
c1.number_input("Drift μ", step=0.01, key="mu", format="%.3f")
c2.number_input("Vol σ", min_value=0.01, max_value=2.0, step=0.01, key="sigma", format="%.3f",
                help="Diffusion vol. For Heston/Bates it is the long-run and starting vol (θ = v₀ = σ²).")
c1, c2 = sb.columns(2)
c1.number_input("Rate r", step=0.005, key="r", format="%.3f")
if kind in ("merton", "bates"):
    c2.number_input("Jump λ/yr", min_value=0.0, step=0.1, key="lam", format="%.2f")
    c1, c2 = sb.columns(2)
    c1.number_input("Jump mean", step=0.01, key="jump_mean", format="%.3f", help="Mean log jump size.")
    c2.number_input("Jump std", min_value=0.0, step=0.01, key="jump_std", format="%.3f")
if kind in ("heston", "bates"):
    c1, c2 = sb.columns(2)
    c1.number_input("Mean reversion κ", min_value=0.01, step=0.1, key="kappa", format="%.2f")
    c2.number_input("Vol of vol ξ", min_value=0.01, step=0.05, key="xi", format="%.2f")
    c1, c2 = sb.columns(2)
    c1.number_input("Corr ρ(S, v)", min_value=-0.99, max_value=0.99, step=0.05, key="rho", format="%.2f")

section("Market making", sb)
c1, c2 = sb.columns(2)
c1.number_input("Half-spread", min_value=0.0, step=0.01, key="half_spread", format="%.3f")
c2.number_input("Quote size", min_value=1, step=1, key="quote_size")
c1, c2 = sb.columns(2)
c1.number_input("Fill intensity", min_value=0.0, max_value=1.0, step=0.05, key="fill", format="%.2f",
                help="Probability that a customer arrives each tick.")
c2.number_input("Risk aversion", min_value=0.0, step=0.01, key="risk_aversion", format="%.3f",
                help="$ of quote skew per contract of inventory. Short inventory → quotes move up.")
c1, c2 = sb.columns(2)
c1.number_input("Informed share", min_value=0.0, max_value=1.0, step=0.05, key="informed", format="%.2f",
                help="Share of customers who know the TRUE option value and only trade when it is "
                     "outside your quotes. They are how model error turns into losses.")
c2.number_input("Noise buy prob", min_value=0.0, max_value=1.0, step=0.05, key="p_buy", format="%.2f",
                help="Probability a noise trader buys (0.5 = balanced flow).")

section("Simulation", sb)
c1, c2 = sb.columns(2)
c1.number_input("Horizon (yrs)", min_value=0.1, max_value=5.0, step=0.25, key="horizon", format="%.2f")
c2.number_input("Ticks", min_value=20, max_value=2000, step=10, key="ticks")
c1, c2 = sb.columns(2)
c1.number_input("Hedge every (ticks)", min_value=0, step=1, key="hedge_every",
                help="Delta-rebalance frequency. 0 = never hedge.")
c2.text_input("Seed (blank=random)", key="seed")
c1, c2 = sb.columns(2)
c1.number_input("Speed (ms/tick)", min_value=0, max_value=500, step=10, key="speed")
c2.checkbox("Animate", key="animate")

sb.button("▶  Run & Animate", type="primary", width="stretch", on_click=request_run)
sb.caption("Pause, restart or skip to the end with the buttons under the charts.")
sb.caption("Presets")
p1, p2 = sb.columns(2)
p1.button("matched", width="stretch", on_click=apply_preset, args=("matched",))
p2.button("steamroller", width="stretch", on_click=apply_preset, args=("steamroller",))
p1, p2 = sb.columns(2)
p1.button("vol clustering", width="stretch", on_click=apply_preset, args=("vol clustering",))
p2.button("crash + vol", width="stretch", on_click=apply_preset, args=("crash + vol",))

# ---------------------------------------------------------------------------
# Build model objects from state
# ---------------------------------------------------------------------------
ss = st.session_state
model = MarketModel(kind=ss.true_model, mu=ss.mu, sigma=ss.sigma, r=ss.r, lam=ss.lam,
                    jump_mean=ss.jump_mean, jump_std=ss.jump_std, kappa=ss.kappa, xi=ss.xi, rho=ss.rho)
params = MMParams(S0=ss.S0, strike=ss.K, T=ss.T, is_call=ss.opt_type == "Call",
                  half_spread=ss.half_spread, quote_size=int(ss.quote_size), fill_intensity=ss.fill,
                  informed_share=ss.informed, p_buy=ss.p_buy, risk_aversion=ss.risk_aversion,
                  mm_vol_mode=VOL_MODES[ss.vol_mode], mm_vol_manual=ss.mm_vol_manual,
                  horizon=ss.horizon, ticks=int(ss.ticks), hedge_every=int(ss.hedge_every))
signature = (model, params)


def parse_seed(s: str):
    s = (s or "").strip()
    try:
        return int(s) if s else None
    except ValueError:
        return abs(hash(s)) % (2 ** 31)


@st.cache_data(show_spinner=False)
def contract_snapshot(model: MarketModel, params: MMParams):
    """BS vs true price of the quoted contract at t=0."""
    sig = mm_vol(model, params)
    dt = params.horizon / params.ticks
    T_eff = max(1, round(params.T / dt)) * dt
    bs = float(bs_price(params.S0, params.strike, T_eff, model.r, sig, params.is_call))
    tp = float(true_price(params.S0, params.strike, T_eff, model, params.is_call))
    iv = float(bs_implied_vol(tp, params.S0, params.strike, T_eff, model.r, params.is_call))
    return dict(sig=sig, bs=bs, true=tp, iv=iv, T_eff=T_eff)


snap = contract_snapshot(model, params)
mis = snap["true"] - snap["bs"]
matched = model.kind == "gbm" and abs(snap["sig"] - model.sigma) < 1e-9
if matched:
    banner_slot.markdown(
        '<div class="mm-banner mm-ok">✓ <b>Model matches reality</b>: the MM should capture the '
        'spread with controlled variance.</div>', unsafe_allow_html=True)
else:
    why = {
        "gbm": "the market is GBM, but your vol is wrong",
        "merton": "the market jumps; BS assumes continuous paths and thin tails",
        "heston": "volatility moves around; BS assumes one constant vol",
        "bates": "the market jumps <i>and</i> volatility moves; BS assumes neither",
    }[model.kind]
    edge_note = ("more than your half-spread, so informed flow will pick you off"
                 if abs(mis) > ss.half_spread else "inside your half-spread at t=0")
    banner_slot.markdown(
        f'<div class="mm-banner mm-warn">⚠ <b>Model mismatch</b>: {why}.<br>'
        f'Your BS price <span class="mono">{snap["bs"]:.3f}</span> vs true '
        f'<span class="mono">{snap["true"]:.3f}</span> (implied vol '
        f'<span class="mono">{snap["iv"] * 100:.1f}%</span> vs your '
        f'<span class="mono">{snap["sig"] * 100:.1f}%</span>): a gap of '
        f'<span class="mono">{mis:+.3f}</span>, {edge_note}.</div>',
        unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="mm-header"><div class="mm-logo"></div><div>
<div class="mm-title">Market Making Simulator</div>
<div class="mm-sub">Model risk in options market making: quote with Black-Scholes, trade in a world that isn't.</div>
</div></div>""", unsafe_allow_html=True)

# Run request -> new simulation
if ss.pop("run_request", False):
    with st.spinner("Simulating…"):
        ss.res = run_simulation(model, params, n_runs=1, seed=parse_seed(ss.seed))
    ss.res_signature = signature
    ss.player = None  # built lazily below (needs post_mortem)
    ss.player_autoplay = bool(ss.animate)
    ss.player_speed = float(ss.speed)

tab_mm, tab_dist, tab_how = st.tabs(["Market Making", "Return Distributions", "How it works"])


# ---------------------------------------------------------------------------
# Helpers for the Market Making tab
# ---------------------------------------------------------------------------
def signed(x: float, fmt: str = "{:+.2f}") -> str:
    cls = "pos" if x > 1e-9 else "neg" if x < -1e-9 else ""
    return f'<span class="{cls}">{fmt.format(x)}</span>'


def cards_html(res, i: int | None) -> str:
    if res is None or i is None:
        items = [("Tick", "–", ""), ("Spot", "–", ""), ("Contracts", "–", ""),
                 ("BS edge / ct", "–", ""), ("True edge / ct", "–", ""), ("Realised P/L", "–", "")]
    else:
        nq = res.n_quote
        j = min(i, nq - 1)
        n_ct = int(np.abs(res.trade_side[0, :j + 1]).sum() * res.params.quote_size)
        me, te = res.model_edge_cum[0, i], res.true_edge_cum[0, i]
        phase = "quoting" if i < nq else "run-off"
        items = [
            ("Tick", f"{i}/{len(res.time) - 1}", f"t = {res.time[i]:.2f}y · {phase}"),
            ("Spot", f"{res.paths.S[0, i]:.2f}", f"vol now {np.sqrt(res.paths.v[0, i]) * 100:.1f}%"),
            ("Contracts", f"{n_ct}", f"net inventory {res.inventory[0, i]:+.0f}"),
            ("BS edge / ct", signed(me / n_ct if n_ct else 0.0, "{:+.3f}"), "what the model says"),
            ("True edge / ct", signed(te / n_ct if n_ct else 0.0, "{:+.3f}"), "vs true value"),
            ("Realised P/L", signed(res.pnl[0, i]), f"BS promised {me:+.2f}"),
        ]
    cells = "".join(f'<div class="mm-card"><div class="k">{k}</div><div class="v">{v}</div>'
                    f'<div class="s">{s}</div></div>' for k, v, s in items)
    return f'<div class="mm-cards">{cells}</div>'


def post_mortem(res) -> str:
    p = res.params
    side = res.trade_side[0]
    buys = int((side > 0).sum()) * p.quote_size
    sells = int((side < 0).sum()) * p.quote_size
    n_inf = int(res.trade_informed[0].sum()) * p.quote_size
    spread, merr, hres = res.spread_capture[0], res.model_error[0], res.hedge_residual[0]
    total = res.final_pnl[0]
    J = res.paths.jumps[0]
    jumps = np.nonzero(J)[0]
    avg_gap = float(np.mean(res.true_val[0] - res.fair[0]))
    lines = [
        f"You traded <b>{buys + sells}</b> contracts: bought <b>{buys}</b> at the bid, sold <b>{sells}</b> "
        f"at the ask; <b>{n_inf}</b> of them against informed traders.",
        f"Black-Scholes told you every fill was worth the half-spread: <b>{spread:+.2f}</b> in total. "
        f"Measured against the <i>true</i> option value, the same trades were worth "
        f"<b>{res.true_edge_cum[0, -1]:+.2f}</b> (model error {signed(merr)}).",
        f"Delta-hedging with BS deltas and running the book off to expiry added {signed(hres)}"
        + (f"; the path had <b>{len(jumps)}</b> jump(s), the largest <b>{J[jumps].min() * 100:+.1f}%</b>"
           if len(jumps) and J[jumps].min() < 0 else
           (f"; the path had <b>{len(jumps)}</b> jump(s)" if len(jumps) else "")) + ".",
        f"<b>Realised P/L: {signed(total)}</b>.",
    ]
    model = res.model
    if model.kind == "gbm" and abs(res.sigma_mm - model.sigma) < 1e-9:
        verdict = ("Model matched reality: the true edge equals the BS edge and what's left is "
                   "discrete-hedging noise around zero. This is what market making is supposed to look like.")
    else:
        verdict = (f"On average the true value sat <b>{avg_gap:+.3f}</b> away from your BS fair value "
                   f"(half-spread {p.half_spread:.3f}). ")
        if merr < 0:
            verdict += ("Informed flow traded exactly when your model was wrong, so the spread you thought "
                        "you were earning was handed back. ")
        if model.has_jumps and len(jumps):
            short = res.inventory[0, jumps[0] - 1] < 0
            verdict += ("Jumps can't be delta-hedged: " + (
                "you were short options going into the gap, so the hedge covered almost none of it. "
                if short else "a gap move breaks the hedge whichever side you're on. "))
        elif model.has_jumps:
            verdict += ("<b>No jump hit this time</b>, so you kept the crash premium you under-charged for. "
                        "That's the steamroller before it arrives: re-run with another seed, or run the Monte Carlo. ")
        if model.has_sv:
            verdict += ("Volatility kept moving, so a single BS vol mis-priced options and mis-sized hedges. ")
        verdict += "Open <b>Return Distributions</b> to see how different the true distribution is, and how often this happens."
    return "<br>".join(lines) + f'<div style="margin-top:8px;color:#52514e">{verdict}</div>'


def run_table(res) -> pd.DataFrame:
    n = len(res.time)
    nq = res.n_quote
    pad = lambda a: np.concatenate([a, np.full(n - nq, np.nan)])
    return pd.DataFrame({
        "t": res.time, "spot": res.paths.S[0], "vol": np.sqrt(res.paths.v[0]),
        "jump": res.paths.jumps[0], "bs_fair": pad(res.fair[0]), "true_value": pad(res.true_val[0]),
        "bid": pad(res.bid[0]), "ask": pad(res.ask[0]),
        "mm_side": pad(res.trade_side[0].astype(float)), "informed": pad(res.trade_informed[0].astype(float)),
        "inventory": res.inventory[0], "hedge_shares": res.hedge[0],
        "bs_edge_cum": res.model_edge_cum[0], "true_edge_cum": res.true_edge_cum[0], "pnl": res.pnl[0],
    })


# ---------------------------------------------------------------------------
# Return Distributions tab
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def dist_data(model: MarketModel, params: MMParams, sigma_mm: float, T_eff: float):
    dt = params.horizon / params.ticks
    lr, T_used = horizon_returns(model, T_eff, dt, 40_000, seed=123)
    K, iv, _ = smile(model, T_used, params.is_call, S0=100.0, m_lo=0.7, m_hi=1.3, n=49)
    return lr, T_used, K, iv


@st.cache_data(show_spinner=False)
def monte_carlo(model: MarketModel, params: MMParams, n: int, seed: int):
    sig = mm_vol(model, params)
    true_w = run_simulation(model, params, n_runs=n, seed=seed)
    bs_world = run_simulation(model.as_gbm(sig), params, n_runs=n, seed=seed, sigma_mm=sig)
    pick = lambda r: dict(pnl=r.final_pnl, spread=r.spread_capture, merr=r.model_error,
                          hres=r.hedge_residual, n=r.n_contracts)
    return pick(true_w), pick(bs_world)


with tab_dist:
    lr, T_used, Kg, ivg = dist_data(model, params, snap["sig"], snap["T_eff"])
    sd_bs = snap["sig"] * np.sqrt(T_used)
    mean_bs = (model.mu - 0.5 * snap["sig"] ** 2) * T_used
    m_true = lr.mean(); s_true = lr.std()
    skew_true = float(((lr - m_true) ** 3).mean() / s_true ** 3)
    kurt_true = float(((lr - m_true) ** 4).mean() / s_true ** 4 - 3)
    q01_true = float(np.quantile(lr, 0.01))
    q01_bs = mean_bs - 2.326 * sd_bs
    st.markdown(
        '<div class="mm-cards">'
        f'<div class="mm-card"><div class="k">Vol (true)</div><div class="v">{s_true / np.sqrt(T_used) * 100:.1f}%</div><div class="s">annualised, incl. jumps</div></div>'
        f'<div class="mm-card"><div class="k">Vol (your BS)</div><div class="v">{snap["sig"] * 100:.1f}%</div><div class="s">{ss.vol_mode.split(" (")[0].lower()}</div></div>'
        f'<div class="mm-card"><div class="k">Skewness</div><div class="v">{skew_true:+.2f}</div><div class="s">BS assumes 0.00</div></div>'
        f'<div class="mm-card"><div class="k">Excess kurtosis</div><div class="v">{kurt_true:+.2f}</div><div class="s">BS assumes 0.00</div></div>'
        f'<div class="mm-card"><div class="k">1% worst return</div><div class="v">{q01_true * 100:+.1f}%</div><div class="s">BS says {q01_bs * 100:+.1f}%</div></div>'
        '</div>', unsafe_allow_html=True)

    d1, d2 = st.columns([1.15, 1])
    with d1:
        log_y = st.toggle("Log scale (see the tails)", value=False, key="logy")
        st.plotly_chart(charts.returns_fig(lr, mean_bs, sd_bs, T_used, log_y), width="stretch",
                        theme=None, config={"displayModeBar": False})
        st.caption("Distribution of the underlying's log-return over one option life. "
                   "Where the orange bars stick out past the blue curve, the market does things your model says can't happen.")
    with d2:
        st.plotly_chart(charts.smile_fig(Kg, ivg, snap["sig"], params.strike / params.S0 * 100, 100.0),
                        width="stretch", theme=None, config={"displayModeBar": False})
        st.caption("The true model's option prices, expressed as Black-Scholes implied vols. "
                   "A flat line would mean BS is right. Any gap at your strike is edge handed to informed traders.")

    st.markdown("#### Monte Carlo: how often does the market maker lose?")
    m1, m2 = st.columns([3, 1])
    m1.slider("Number of simulated runs", 100, 2000, step=100, key="mc_runs")
    if m2.button("Run Monte Carlo", type="primary", width="stretch"):
        with st.spinner(f"Simulating {ss.mc_runs} runs in both worlds…"):
            ss.mc = dict(sig=signature, data=monte_carlo(model, params, int(ss.mc_runs),
                                                         parse_seed(ss.seed) or 0))
    if ss.mc is None:
        st.info("Run the same market maker many times, once in the **true world** and once in the world "
                "**Black-Scholes assumes** (GBM at your vol), and compare the P/L distributions.")
    else:
        if ss.mc["sig"] != signature:
            st.warning("Parameters changed since this Monte Carlo. Press **Run Monte Carlo** to refresh.")
        tw, bw = ss.mc["data"]
        st.plotly_chart(charts.mc_fig(tw["pnl"], bw["pnl"]), width="stretch",
                        theme=None, config={"displayModeBar": False})
        s_t, s_b = pnl_stats(tw["pnl"]), pnl_stats(bw["pnl"])
        def col(st_, w):
            return [f"{st_['mean']:+.2f}", f"{np.median(w['pnl']):+.2f}", f"{st_['std']:.2f}",
                    f"{st_['p_loss'] * 100:.1f}%", f"{st_['var5']:.2f}", f"{st_['cvar5']:.2f}",
                    f"{st_['worst']:+.2f}", f"{st_['skew']:+.2f}", f"{w['n'].mean():.0f}",
                    f"{w['spread'].mean():+.2f}", f"{w['merr'].mean():+.2f}", f"{w['hres'].mean():+.2f}"]
        table = pd.DataFrame(
            {"If Black-Scholes were right": col(s_b, bw), "True world": col(s_t, tw)},
            index=["Mean P/L", "Median P/L", "Std dev", "P(loss)", "VaR 5% (loss)",
                   "CVaR 5% (expected shortfall)", "Worst run", "Skewness", "Contracts traded (avg)",
                   "avg Spread capture", "avg Model error", "avg Hedging & path"])
        st.dataframe(table, width="stretch", height=458)


# ---------------------------------------------------------------------------
# How it works tab
# ---------------------------------------------------------------------------
with tab_how:
    st.markdown(r"""
### The game
You are an options market maker. Every tick you quote a bid and an ask on a rolling option
(strike at a fixed moneyness **K/S₀** of the current spot, maturity **T**). Your fair value comes
from **Black-Scholes** with a single volatility, and you delta-hedge your whole book with BS deltas.

Customers arrive with probability *fill intensity* per tick:

* **Noise traders** buy or sell for reasons unrelated to value (buy with probability *noise buy prob*).
* **Informed traders** know the **true** value of the option, the one implied by the real market dynamics.
  They buy only when true value > your ask, and sell only when true value < your bid.

After the quoting horizon you stop quoting and run the book off until every contract has expired, so the
final P/L is fully realised: no marks, no model.

### Where the P/L comes from
$$
\text{Realised P/L} = \underbrace{\sum \text{(price} - \text{BS value)}}_{\text{spread capture}}
+ \underbrace{\sum (\text{BS value} - \text{true value})}_{\text{model error}}
+ \underbrace{\text{hedging \& path}}_{\text{jumps, vol moves, discrete hedging}}
$$
(each term signed from the MM's side of the trade).

* If your model is right, *model error* is 0, *hedging & path* is zero-mean noise, and you earn the spread.
* If your model is wrong, informed traders only trade when it hurts you (**adverse selection**), and
  hedging can't neutralise jumps or volatility moves, so the P/L distribution picks up a **fat left tail**.

### The true worlds
| Model | Dynamics | What BS misses |
|---|---|---|
| GBM | $dS/S = \mu\,dt + \sigma\,dW$ | nothing (if σ is right) |
| Merton | GBM + Poisson jumps, log-jump $\sim N(m, s^2)$ | fat tails, skew, gap risk |
| Heston | $dv = \kappa(\theta - v)dt + \xi\sqrt{v}\,dZ$, $\rho = \text{corr}(dW, dZ)$ | vol clustering, skew, vol-of-vol |
| Bates | Heston + Merton jumps | all of the above |

True option values are computed with the characteristic function (Lewis 2001) under the risk-neutral
measure (same parameters, drift → r). Paths are simulated under the physical measure with drift μ.

### Presets
* **matched**: GBM world, correct vol. The benchmark: steady spread capture.
* **steamroller**: customers love buying OTM puts; the MM sells them at "fair" BS prices. Pennies,
  pennies, pennies… then a −15% jump.
* **vol clustering**: Heston world. Informed traders buy when vol is high and sell when it's low.
* **crash + vol**: Bates world: jumps *and* stochastic vol.

Things to try: switch the MM vol to *Calibrated*, widen the half-spread until you break even, set
*informed share* to 0, or hedge less often.
""")


# ---------------------------------------------------------------------------
# Market Making tab
# ---------------------------------------------------------------------------
with tab_mm:
    res = ss.res
    CFG = {"displayModeBar": False}
    if res is None:
        st.markdown(cards_html(None, None), unsafe_allow_html=True)
        e1, e2 = st.columns([1.08, 1])
        e1.plotly_chart(charts.empty_fig("Underlying & MM quotes", 600), width="stretch", theme=None, config=CFG)
        e2.plotly_chart(charts.empty_fig("Cumulative P/L", 600), width="stretch", theme=None, config=CFG)
        st.markdown('<div class="mm-summary"><b>Run summary</b><br>'
                    '<span class="mono" style="color:#898781">Run a simulation to see the post-mortem.</span></div>',
                    unsafe_allow_html=True)
    else:
        if ss.player is None:
            ss.player = player_html(res, post_mortem(res), ss.player_speed, ss.player_autoplay)
        components.html(ss.player, height=1320, scrolling=True)
        if ss.get("res_signature") != signature:
            st.caption("Controls changed since this run. Press **Run & Animate** to re-simulate.")
        with st.expander("Data for this run"):
            df = run_table(res)
            st.dataframe(df, width="stretch", height=260)
            st.download_button("Download CSV", df.to_csv(index=False).encode(), "mm_run.csv", "text/csv")
