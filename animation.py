"""Client-side animation of a single run.

The whole run is shipped to the browser once and animated with Plotly.js, so
playback is smooth and needs no server round-trips (important on Streamlit
Community Cloud). Play/pause, restart and skip-to-end live inside the player.
"""
from __future__ import annotations

import base64
import json

import numpy as np
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs_version

from . import charts
from .engine import SimResult


def _plain(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(type(obj))


def _decode(obj):
    """Plotly >= 6 serialises arrays as base64 typed arrays; the player slices plain lists."""
    if isinstance(obj, dict):
        if "bdata" in obj and "dtype" in obj:
            arr = np.frombuffer(base64.b64decode(obj["bdata"]), dtype=np.dtype(obj["dtype"]))
            if "shape" in obj:
                shape = obj["shape"]
                arr = arr.reshape([int(x) for x in str(shape).split(",")] if isinstance(shape, str) else shape)
            return arr.tolist()
        return {k: _decode(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_decode(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _fig_json(fig: go.Figure) -> str:
    return json.dumps(_decode(fig.to_plotly_json()), default=_plain, separators=(",", ":"))


def _r(a, d=5):
    return np.round(np.asarray(a, float), d).tolist()


def player_html(res: SimResult, summary_html: str, speed_ms: float, autoplay: bool,
                height_charts: int = 600) -> str:
    last = len(res.time) - 1
    nq = res.n_quote
    mkt = charts.market_fig(res, last, height=height_charts)
    pnl_anim = charts.pnl_fig(res, last, height=height_charts, end_labels=False)
    pnl_final = charts.pnl_fig(res, last, height=height_charts)
    attr = charts.attribution_fig(res.spread_capture[0], res.model_error[0],
                                  res.hedge_residual[0], res.final_pnl[0], height=330)
    side = np.abs(res.trade_side[0]).astype(float) * res.params.quote_size
    nct = np.concatenate([np.cumsum(side), np.full(last + 1 - nq, side.sum())])
    data = dict(
        t=_r(res.time, 6), S=_r(res.paths.S[0], 4), vol=_r(np.sqrt(res.paths.v[0]), 4),
        pnl=_r(res.pnl[0], 4), me=_r(res.model_edge_cum[0], 4), te=_r(res.true_edge_cum[0], 4),
        inv=_r(res.inventory[0], 0), nct=_r(nct, 0), nq=nq, N=last,
    )
    v = get_plotlyjs_version()
    sources = [f"https://cdn.plot.ly/plotly-{v}.min.js",
               f"https://cdn.jsdelivr.net/npm/plotly.js-dist-min@{v}/plotly.min.js",
               "https://cdn.plot.ly/plotly-2.35.2.min.js",
               "https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"]
    return (_TEMPLATE
            .replace("__SOURCES__", json.dumps(sources))
            .replace("__DATA__", json.dumps(data, separators=(",", ":")))
            .replace("__MKT__", _fig_json(mkt))
            .replace("__PNLA__", _fig_json(pnl_anim))
            .replace("__PNLF__", _fig_json(pnl_final))
            .replace("__ATTR__", _fig_json(attr))
            .replace("__SUMMARY__", json.dumps(summary_html))
            .replace("__SPEED__", str(float(speed_ms)))
            .replace("__AUTOPLAY__", "true" if autoplay else "false")
            .replace("__CH__", str(height_charts)))


_TEMPLATE = r"""<!doctype html><html><head><meta charset="utf-8">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root { --ink:#0b0b0b; --ink2:#52514e; --muted:#898781; --line:#e6e5df; --card:#fcfcfb;
        --pos:#006300; --neg:#c12f2f; --accent:#0e9f6e; }
* { box-sizing: border-box; }
html, body { margin:0; padding:0; background:#ffffff; color:var(--ink);
  font-family: Inter, system-ui, -apple-system, "Segoe UI", sans-serif; overflow-x:hidden; }
.mono { font-family: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, monospace; }
.cards { display:grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap:10px; margin: 2px 0 12px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:10px 14px; }
.card .k { font-size:10.5px; font-weight:700; letter-spacing:.09em; color:var(--muted); text-transform:uppercase; }
.card .v { font-family:"JetBrains Mono", ui-monospace, monospace; font-size:18px; font-weight:600; margin-top:3px; white-space:nowrap; }
.card .s { font-size:11.5px; color:var(--muted); margin-top:1px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.pos { color:var(--pos); } .neg { color:var(--neg); }
.grid { display:grid; grid-template-columns: 1.08fr 1fr; gap:16px; }
.panel { background:#fff; border:1px solid var(--line); border-radius:10px; padding:6px 6px 0; min-width:0; }
.bar { display:flex; align-items:center; gap:10px; margin:12px 0; flex-wrap:wrap; }
button { font-family:"JetBrains Mono", ui-monospace, monospace; font-size:13px; padding:7px 14px; border-radius:8px;
  border:1px solid var(--line); background:#fff; color:var(--ink); cursor:pointer; }
button:hover { border-color:var(--accent); color:var(--accent); }
button.primary { background:var(--accent); border-color:var(--accent); color:#fff; }
button.primary:hover { background:#0b8a5f; color:#fff; }
button:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
.prog { flex:1; min-width:120px; height:6px; background:#efeee9; border-radius:3px; overflow:hidden; }
.prog > div { height:100%; width:0; background:var(--accent); }
.plabel { font-size:12px; color:var(--muted); min-width:120px; text-align:right; }
.sumgrid { display:grid; grid-template-columns: 1.15fr 1fr; gap:16px; }
.summary { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px 18px;
  font-size:13.5px; line-height:1.5; }
.summary b { font-weight:600; }
.summary .ph { color:var(--muted); font-family:"JetBrains Mono", ui-monospace, monospace; font-size:13px; }
.hidden { visibility:hidden; }
.err { padding:24px; color:var(--neg); }
@media (max-width: 860px) { .grid, .sumgrid { grid-template-columns: 1fr; } }
</style></head><body>
<div id="cards" class="cards"></div>
<div class="grid"><div class="panel"><div id="mkt" style="height:__CH__px"></div></div>
<div class="panel"><div id="pnl" style="height:__CH__px"></div></div></div>
<div class="bar">
  <button id="play" class="primary">❚❚ Pause</button>
  <button id="restart">⟲ Restart</button>
  <button id="skip">⏭ Skip to end</button>
  <div class="prog"><div id="fill"></div></div><span id="plabel" class="plabel mono"></span>
</div>
<div class="sumgrid">
  <div id="summary" class="summary"><b>Run summary</b><br><span class="ph">Running… the post-mortem appears when the book has run off.</span></div>
  <div id="attrwrap" class="panel hidden"><div id="attr" style="height:330px"></div></div>
</div>
<script>
const SOURCES = __SOURCES__;
const D = __DATA__;
const MKT = __MKT__, PNLA = __PNLA__, PNLF = __PNLF__, ATTR = __ATTR__;
const SUMMARY = __SUMMARY__;
const SPEED = __SPEED__;
const AUTOPLAY = __AUTOPLAY__;
const CFG = {displayModeBar:false, responsive:true};
const N = D.N;
let i = 0, playing = false, finished = false, acc = 0, lastTs = null, attrDrawn = false;

function sliceArr(a, k) { return Array.isArray(a) ? a.slice(0, k) : a; }
function sliceTrace(tr, tmax) {
  const x = tr.x;
  if (!Array.isArray(x)) return tr;
  let k = 0; while (k < x.length && x[k] <= tmax + 1e-9) k++;
  const o = Object.assign({}, tr);
  for (const key of ["x","y","customdata","text","hovertext"]) if (key in o) o[key] = sliceArr(o[key], k);
  if (o.marker) {
    const m = Object.assign({}, o.marker);
    for (const key of ["size","symbol","color","opacity"]) if (key in m) m[key] = sliceArr(m[key], k);
    if (m.line) { const l = Object.assign({}, m.line); for (const key of ["color","width"]) if (key in l) l[key] = sliceArr(l[key], k); m.line = l; }
    o.marker = m;
  }
  return o;
}
function sliceFig(fig, tmax) { return fig.data.map(tr => sliceTrace(tr, tmax)); }

function fmt(x, d) { return (x >= 0 ? "+" : "") + x.toFixed(d); }
function signed(x, d) { const c = x > 1e-9 ? "pos" : (x < -1e-9 ? "neg" : ""); return `<span class="${c}">${fmt(x, d)}</span>`; }
function cards(j) {
  const n = D.nct[j], me = D.me[j], te = D.te[j];
  const phase = j < D.nq ? "quoting" : "run-off";
  const items = [
    ["Tick", `${j}/${N}`, `t = ${D.t[j].toFixed(2)}y · ${phase}`],
    ["Spot", D.S[j].toFixed(2), `vol now ${(D.vol[j]*100).toFixed(1)}%`],
    ["Contracts", `${n}`, `net inventory ${fmt(D.inv[j], 0)}`],
    ["BS edge / ct", signed(n ? me / n : 0, 3), "what the model says"],
    ["True edge / ct", signed(n ? te / n : 0, 3), "vs true value"],
    ["Realised P/L", signed(D.pnl[j], 2), `BS said ${fmt(me, 2)}`],
  ];
  document.getElementById("cards").innerHTML = items.map(([k, v, s]) =>
    `<div class="card"><div class="k">${k}</div><div class="v">${v}</div><div class="s">${s}</div></div>`).join("");
  document.getElementById("fill").style.width = (100 * j / N).toFixed(2) + "%";
  document.getElementById("plabel").textContent = `t = ${D.t[j].toFixed(2)}y`;
}
function draw(j) {
  const tm = D.t[j];
  Plotly.react("mkt", sliceFig(MKT, tm), MKT.layout, CFG);
  Plotly.react("pnl", sliceFig(PNLA, tm), PNLA.layout, CFG);
  cards(j);
}
function finish() {
  i = N; finished = true; playing = false;
  Plotly.react("mkt", MKT.data, MKT.layout, CFG);
  Plotly.react("pnl", PNLF.data, PNLF.layout, CFG);
  cards(N);
  document.getElementById("summary").innerHTML = "<b>Run summary</b><br>" + SUMMARY;
  document.getElementById("attrwrap").classList.remove("hidden");
  if (!attrDrawn) { Plotly.newPlot("attr", ATTR.data, ATTR.layout, CFG); attrDrawn = true; }
  document.getElementById("play").textContent = "▶ Replay";
}
function loop(ts) {
  if (!playing) return;
  if (lastTs === null) lastTs = ts;
  acc += ts - lastTs; lastTs = ts;
  const adv = SPEED <= 0 ? N : Math.floor(acc / SPEED);
  if (adv >= 1) { acc -= adv * SPEED; i = Math.min(N, i + adv); if (i >= N) { finish(); return; } draw(i); }
  requestAnimationFrame(loop);
}
function play() {
  if (finished) { restart(); return; }
  playing = true; lastTs = null; acc = 0;
  document.getElementById("play").textContent = "❚❚ Pause";
  requestAnimationFrame(loop);
}
function pause() { playing = false; document.getElementById("play").textContent = "▶ Play"; }
function restart() {
  finished = false; i = 0;
  document.getElementById("summary").innerHTML = '<b>Run summary</b><br><span class="ph">Running… the post-mortem appears when the book has run off.</span>';
  document.getElementById("attrwrap").classList.add("hidden");
  draw(0); play();
}
function start() {
  document.getElementById("play").onclick = () => playing ? pause() : play();
  document.getElementById("restart").onclick = restart;
  document.getElementById("skip").onclick = finish;
  Plotly.newPlot("mkt", sliceFig(MKT, D.t[0]), MKT.layout, CFG);
  Plotly.newPlot("pnl", sliceFig(PNLA, D.t[0]), PNLA.layout, CFG);
  cards(0);
  if (AUTOPLAY) play(); else finish();
}
function load(k) {
  if (k >= SOURCES.length) { document.body.innerHTML = '<div class="err">Could not load Plotly.js from any CDN.</div>'; return; }
  const s = document.createElement("script");
  s.src = SOURCES[k]; s.onload = start; s.onerror = () => load(k + 1);
  document.head.appendChild(s);
}
load(0);
</script></body></html>"""
