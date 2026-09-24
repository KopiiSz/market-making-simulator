"""Plotly figures. Colour semantics are fixed across the whole app:

    orange  = the TRUE market (spot path, true option value, true distribution)
    blue    = the market maker's Black-Scholes MODEL
    ink     = what actually happened to the MM's account (realised P/L)
    aqua / violet = MM buys / MM sells
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from engine import SimResult

TRUE = "#eb6834"
MODEL = "#2a78d6"
MODEL_BAND = "rgba(42,120,214,0.13)"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#ecebe6"
AXIS = "#c3c2b7"
BUY = "#1baf7a"
SELL = "#4a3aa7"
POS = "#2a78d6"
NEG = "#e34948"

FONT = "Inter, system-ui, -apple-system, 'Segoe UI', sans-serif"
MONO = "'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace"


def _style(fig: go.Figure, height: int, legend: bool = True, legend_bottom: bool = False) -> go.Figure:
    if legend_bottom:
        leg = dict(orientation="h", yanchor="top", y=-0.1, xanchor="left", x=0)
        margin = dict(l=56, r=24, t=28, b=110)
    else:
        leg = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
        margin = dict(l=56, r=24, t=48 if legend else 24, b=52)
    fig.update_layout(
        height=height,
        margin=margin,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        font=dict(family=FONT, size=12, color=INK2),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#ffffff", bordercolor=AXIS, font=dict(family=MONO, size=11, color=INK)),
        showlegend=legend,
        legend=dict(**leg, font=dict(size=11, color=INK2), bgcolor="rgba(0,0,0,0)"),
        uirevision="keep",
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID, gridwidth=1, zeroline=False,
                     linecolor=AXIS, ticks="outside", tickcolor=AXIS, ticklen=4,
                     tickfont=dict(family=MONO, size=10, color=MUTED),
                     title_font=dict(size=11, color=MUTED))
    fig.update_yaxes(showgrid=True, gridcolor=GRID, gridwidth=1, zeroline=False,
                     linecolor=AXIS, tickfont=dict(family=MONO, size=10, color=MUTED),
                     title_font=dict(size=11, color=MUTED))
    for a in fig.layout.annotations:  # subplot titles
        if a.text and a.text.startswith("<b>"):
            a.update(font=dict(size=12, color=INK), x=0, xanchor="left")
    return fig


def _pad(lo, hi, frac=0.06):
    span = max(hi - lo, 1e-6)
    return [lo - frac * span, hi + frac * span]


def _vline(fig, x, rows, label=None):
    for rrow in rows:
        fig.add_vline(x=x, line=dict(color=AXIS, width=1, dash="dot"), row=rrow, col=1)
    if label:
        fig.add_annotation(x=x, y=1, yref="paper", xref="x", text=label, showarrow=False,
                           xanchor="right", yanchor="bottom", font=dict(size=10, color=MUTED),
                           xshift=-4)


# ---------------------------------------------------------------------------
# Market Making tab
# ---------------------------------------------------------------------------

def market_fig(res: SimResult, upto: int, run: int = 0, height: int = 600) -> go.Figure:
    t = res.time
    S = res.paths.S[run]
    nq = res.n_quote
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        row_heights=[0.52, 0.48],
                        subplot_titles=("<b>Underlying spot</b>",
                                        "<b>Option: MM quotes vs true value</b>"))
    u = upto + 1
    fig.add_trace(go.Scatter(x=t[:u], y=S[:u], name="Spot (true path)", mode="lines",
                             line=dict(color=TRUE, width=2),
                             hovertemplate="%{y:.2f}"), row=1, col=1)
    J = res.paths.jumps[run, :u]
    jm = np.nonzero(J)[0]
    if jm.size:
        fig.add_trace(go.Scatter(
            x=t[jm], y=S[jm], mode="markers", name="Jump",
            marker=dict(symbol=np.where(J[jm] < 0, "triangle-down", "triangle-up"),
                        size=11, color="#ffffff", line=dict(color=INK, width=2)),
            customdata=J[jm] * 100, hovertemplate="jump %{customdata:+.1f}%"), row=1, col=1)

    q = min(u, nq)
    if q > 0:
        tq = t[:q]
        fig.add_trace(go.Scatter(x=tq, y=res.ask[run, :q], mode="lines", name="Ask",
                                 line=dict(color="rgba(42,120,214,0.45)", width=1),
                                 hovertemplate="%{y:.3f}", showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=tq, y=res.bid[run, :q], mode="lines", name="Bid / ask",
                                 line=dict(color="rgba(42,120,214,0.45)", width=1),
                                 fill="tonexty", fillcolor=MODEL_BAND,
                                 hovertemplate="%{y:.3f}"), row=2, col=1)
        fig.add_trace(go.Scatter(x=tq, y=res.fair[run, :q], mode="lines", name="BS fair (MM model)",
                                 line=dict(color=MODEL, width=1.5, dash="dash"),
                                 hovertemplate="%{y:.3f}"), row=2, col=1)
        fig.add_trace(go.Scatter(x=tq, y=res.true_val[run, :q], mode="lines", name="True value",
                                 line=dict(color=TRUE, width=2), hovertemplate="%{y:.3f}"), row=2, col=1)
        side = res.trade_side[run, :q]
        inf = res.trade_informed[run, :q]
        for s, nm, col, sym, px in ((1, "MM buys (at bid)", BUY, "triangle-up", res.bid),
                                    (-1, "MM sells (at ask)", SELL, "triangle-down", res.ask)):
            idx = np.nonzero(side == s)[0]
            if idx.size:
                fig.add_trace(go.Scatter(
                    x=tq[idx], y=px[run, idx], mode="markers", name=nm,
                    marker=dict(symbol=sym, size=np.where(inf[idx], 11, 8), color=col,
                                line=dict(color=np.where(inf[idx], INK, "#ffffff"),
                                          width=np.where(inf[idx], 1.8, 1))),
                    customdata=np.where(inf[idx], "informed", "noise"),
                    hovertemplate="%{y:.3f} · %{customdata}"), row=2, col=1)

    # fixed axes so the animation doesn't jump around
    fig.update_xaxes(range=[0, t[-1] * 1.01])
    fig.update_xaxes(title_text="time (yrs)", row=2, col=1)
    fig.update_yaxes(range=_pad(S.min(), S.max()), title_text="spot", row=1, col=1)
    olo = min(res.bid[run].min(), res.true_val[run].min())
    ohi = max(res.ask[run].max(), res.true_val[run].max())
    fig.update_yaxes(range=_pad(olo, ohi), title_text="option $", row=2, col=1)
    if res.n_T > 0:
        _vline(fig, t[nq], rows=(1, 2), label="quoting stops | run-off →")
    return _style(fig, height, legend_bottom=True)


def pnl_fig(res: SimResult, upto: int, run: int = 0, height: int = 600,
            end_labels: bool = True) -> go.Figure:
    t = res.time
    u = upto + 1
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12,
                        row_heights=[0.66, 0.34],
                        subplot_titles=("<b>Cumulative P/L ($)</b>",
                                        "<b>Net option inventory (contracts)</b>"))
    series = [
        ("What BS promised (spread capture)", res.model_edge_cum[run], MODEL, "dash", 1.6),
        ("True edge (vs true value)", res.true_edge_cum[run], TRUE, "solid", 2),
        ("Realised P/L (hedged, settled)", res.pnl[run], INK, "solid", 2.4),
    ]
    for name, y, col, dash, w in series:
        fig.add_trace(go.Scatter(x=t[:u], y=y[:u], name=name, mode="lines",
                                 line=dict(color=col, width=w, dash=dash),
                                 hovertemplate="%{y:+.2f}"), row=1, col=1)
        if end_labels:
            fig.add_annotation(x=t[upto], y=y[upto], text=f"{y[upto]:+.2f}", showarrow=False,
                               xanchor="left", xshift=6, font=dict(family=MONO, size=10, color=INK2),
                               row=1, col=1)
    fig.add_hline(y=0, line=dict(color=AXIS, width=1), row=1, col=1)
    fig.add_trace(go.Scatter(x=t[:u], y=res.inventory[run, :u], name="Net contracts",
                             mode="lines", line=dict(color=INK2, width=1.6, shape="hv"),
                             fill="tozeroy", fillcolor="rgba(82,81,78,0.08)",
                             hovertemplate="%{y:+.0f}", showlegend=False), row=2, col=1)
    fig.add_hline(y=0, line=dict(color=AXIS, width=1), row=2, col=1)

    allv = np.concatenate([s[1] for s in series])
    fig.update_xaxes(range=[0, t[-1] * 1.08])
    fig.update_xaxes(title_text="time (yrs)", row=2, col=1)
    fig.update_yaxes(range=_pad(min(allv.min(), 0), max(allv.max(), 0)), row=1, col=1)
    inv = res.inventory[run]
    fig.update_yaxes(range=_pad(min(inv.min(), 0), max(inv.max(), 0)), row=2, col=1)
    if res.n_T > 0:
        _vline(fig, t[res.n_quote], rows=(1, 2))
    return _style(fig, height, legend_bottom=True)


def empty_fig(title: str, height: int = 300) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text="Press <b>Run &amp; Animate</b> to start", x=0.5, y=0.5,
                       xref="paper", yref="paper", showarrow=False,
                       font=dict(size=13, color=MUTED))
    fig.update_xaxes(range=[0, 1], title_text="time (yrs)")
    fig.update_yaxes(range=[0, 1], showticklabels=False)
    fig.update_layout(title=dict(text=f"<b>{title}</b>", x=0, font=dict(size=12, color=INK)))
    return _style(fig, height, legend=False)


def attribution_fig(spread: float, model_err: float, hedge: float, total: float,
                    height: int = 300) -> go.Figure:
    labels = ["Spread capture", "Model error", "Hedging & path", "Realised P/L"]
    fig = go.Figure(go.Waterfall(
        x=labels, y=[spread, model_err, hedge, total],
        measure=["relative", "relative", "relative", "total"],
        text=[f"{v:+.2f}" for v in (spread, model_err, hedge, total)],
        textposition="outside", textfont=dict(family=MONO, size=11, color=INK),
        increasing=dict(marker=dict(color=POS)), decreasing=dict(marker=dict(color=NEG)),
        totals=dict(marker=dict(color=INK)), connector=dict(line=dict(color=AXIS, width=1)),
        hovertemplate="%{x}: %{y:+.2f}<extra></extra>",
    ))
    fig.add_hline(y=0, line=dict(color=AXIS, width=1))
    vals = np.cumsum([0, spread, model_err, hedge])
    lo, hi = min(vals.min(), total, 0), max(vals.max(), total, 0)
    fig.update_yaxes(range=_pad(lo, hi, 0.18), title_text="$")
    _style(fig, height, legend=False)
    fig.update_layout(hovermode="closest", margin=dict(t=36, b=40),
                      title=dict(text="<b>P/L attribution ($)</b>", x=0.01, y=0.97,
                                 font=dict(size=12, color=INK)))
    fig.update_xaxes(showgrid=False, tickfont=dict(family=FONT, size=11, color=INK2))
    return fig


# ---------------------------------------------------------------------------
# Return Distributions tab
# ---------------------------------------------------------------------------

def returns_fig(logret: np.ndarray, mean_bs: float, sd_bs: float, T: float,
                log_y: bool = False, height: int = 380) -> go.Figure:
    lo, hi = np.quantile(logret, [0.0005, 0.9995])
    lo, hi = min(lo, mean_bs - 4.5 * sd_bs), max(hi, mean_bs + 4.5 * sd_bs)
    bins = np.linspace(lo, hi, 90)
    dens, edges = np.histogram(np.clip(logret, lo, hi), bins=bins, density=True)
    mids = 0.5 * (edges[1:] + edges[:-1])
    fig = go.Figure()
    fig.add_trace(go.Bar(x=mids * 100, y=dens / 100, name="True model (simulated)",
                         marker=dict(color=TRUE, line=dict(color="#ffffff", width=1)),
                         opacity=0.85, hovertemplate="%{x:.1f}%: %{y:.4f}"))
    xs = np.linspace(lo, hi, 400)
    pdf = np.exp(-0.5 * ((xs - mean_bs) / sd_bs) ** 2) / (sd_bs * np.sqrt(2 * np.pi))
    fig.add_trace(go.Scatter(x=xs * 100, y=pdf / 100, name="Black-Scholes assumption (normal)",
                             mode="lines", line=dict(color=MODEL, width=2.2),
                             hovertemplate="%{x:.1f}%: %{y:.4f}"))
    fig.update_layout(bargap=0.02, hovermode="closest")
    fig.update_xaxes(title_text=f"log-return over option life T = {T:.2f}y (%)")
    fig.update_yaxes(title_text="density (log scale)" if log_y else "density",
                     type="log" if log_y else "linear")
    if log_y:
        fig.update_yaxes(range=[np.log10(max(dens[dens > 0].min() / 100, 1e-6)),
                                np.log10(max(dens.max(), pdf.max()) / 100 * 1.5)])
    return _style(fig, height)


def smile_fig(K: np.ndarray, iv: np.ndarray, sigma_mm: float, k_traded: float, S0: float,
              height: int = 380) -> go.Figure:
    m = K / S0 * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=m, y=iv * 100, mode="lines", name="True model implied vol",
                             line=dict(color=TRUE, width=2.4), hovertemplate="%{y:.2f}%"))
    fig.add_trace(go.Scatter(x=m, y=np.full_like(m, sigma_mm * 100), mode="lines",
                             name="MM's Black-Scholes vol (flat)",
                             line=dict(color=MODEL, width=2, dash="dash"), hovertemplate="%{y:.2f}%"))
    fig.add_vline(x=k_traded / S0 * 100, line=dict(color=AXIS, width=1, dash="dot"))
    fig.add_annotation(x=k_traded / S0 * 100, y=1, yref="paper", text="quoted strike",
                       showarrow=False, xanchor="left", yanchor="top", xshift=4,
                       font=dict(size=10, color=MUTED))
    fig.update_xaxes(title_text="strike / spot (%)")
    fig.update_yaxes(title_text="implied vol (%)")
    finite = iv[np.isfinite(iv)] * 100
    lo = min(finite.min() if finite.size else sigma_mm * 100, sigma_mm * 100)
    hi = max(finite.max() if finite.size else sigma_mm * 100, sigma_mm * 100)
    mid, half = 0.5 * (lo + hi), max(0.5 * (hi - lo) * 1.25, 2.0)
    fig.update_yaxes(range=[mid - half, mid + half])
    return _style(fig, height)


def mc_fig(pnl_true: np.ndarray, pnl_model: np.ndarray | None, height: int = 400) -> go.Figure:
    allx = pnl_true if pnl_model is None else np.concatenate([pnl_true, pnl_model])
    lo, hi = np.quantile(allx, [0.002, 0.998])
    bins = np.linspace(lo, hi, 60)
    fig = go.Figure()
    if pnl_model is not None:
        fig.add_trace(go.Histogram(x=np.clip(pnl_model, lo, hi), xbins=dict(start=lo, end=hi, size=bins[1] - bins[0]),
                                   name="If Black-Scholes were right", histnorm="probability",
                                   marker=dict(color=MODEL, line=dict(color="#ffffff", width=1)),
                                   opacity=0.55, hovertemplate="%{x}: %{y:.3f}"))
    fig.add_trace(go.Histogram(x=np.clip(pnl_true, lo, hi), xbins=dict(start=lo, end=hi, size=bins[1] - bins[0]),
                               name="True world", histnorm="probability",
                               marker=dict(color=TRUE, line=dict(color="#ffffff", width=1)),
                               opacity=0.75, hovertemplate="%{x}: %{y:.3f}"))
    fig.update_layout(barmode="overlay", hovermode="closest")
    fig.add_vline(x=0, line=dict(color=INK2, width=1))
    marks = [(np.mean(pnl_true), TRUE, "true-world mean")]
    if pnl_model is not None:
        marks.append((np.mean(pnl_model), MODEL, "BS-world mean"))
    marks.sort(key=lambda m: m[0])
    for i, (x, col, lbl) in enumerate(marks):
        left = i == 0 and len(marks) > 1
        fig.add_vline(x=x, line=dict(color=col, width=2, dash="dash"))
        fig.add_annotation(x=x, y=1, yref="paper", text=f"{lbl} {x:+.1f}", showarrow=False,
                           xanchor="right" if left else "left", yanchor="top",
                           xshift=-4 if left else 4, bgcolor="rgba(255,255,255,0.85)",
                           font=dict(family=MONO, size=10, color=INK2))
    fig.update_xaxes(title_text="final realised P/L per run ($)")
    fig.update_yaxes(title_text="share of runs")
    return _style(fig, height)
