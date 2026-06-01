"""Tablero Streamlit de análisis de opciones GGAL — v2 "for dummies".

Ejecutar con:
    streamlit run optionsdesk/ui/dashboard.py

Tab por defecto: Inicio (3 tarjetas de veredicto para cada perfil de riesgo).
Modo avanzado: desbloquea Cadena completa, Simulador P&L e Historial.
"""
from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

from optionsdesk.config.settings import settings
from optionsdesk.core.benchmark import Benchmark, ZERO_BENCHMARK
from optionsdesk.core.rates import RateResult
from optionsdesk.data.providers.base import MarketDataProvider, OptionsChain
from optionsdesk.data.providers.demo import DemoProvider
from optionsdesk.execution.base import Order, OrderSide
from optionsdesk.execution.paper import PaperExecutor
from optionsdesk.signals.alerts import TelegramAlerter
from optionsdesk.signals.directional_ideas import (
    DirectionalIdea,
    build_directional_idea,
    build_directional_spread,
)
from optionsdesk.signals.recommender import (
    Recommendation,
    Recommender,
    RiskProfile,
)
from optionsdesk.signals.screener import Screener, ScreenerConfig
from optionsdesk.strategies.covered_call import CoveredCallConfig, CoveredCallScanner
from optionsdesk.strategies.short_put import ShortPutConfig, ShortPutScanner


# ── Design System: Minimalist Dark ───────────────────────────────────────────

_CSS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">

<style>
/* ── Variables ────────────────────────────────────────────────────── */
:root {
  --bg:         #0A0A0F;
  --bg-alt:     #12121A;
  --bg-muted:   #1A1A24;
  --fg:         #FAFAFA;
  --fg-muted:   #71717A;
  --accent:     #F59E0B;
  --accent-dim: rgba(245,158,11,0.15);
  --border:     rgba(255,255,255,0.08);
  --border-h:   rgba(255,255,255,0.15);
  --card:       rgba(26,26,36,0.60);
  --glow-sm:    0 0 20px rgba(245,158,11,0.15);
  --glow-md:    0 0 40px rgba(245,158,11,0.20);
  --glow-lg:    0 0 60px rgba(245,158,11,0.25);
}

/* ── Base ─────────────────────────────────────────────────────────── */
html, body, .stApp {
  background-color: var(--bg) !important;
  font-family: 'Inter', system-ui, sans-serif !important;
  color: var(--fg) !important;
}

/* Ambient orbs decorativos */
.stApp::before {
  content: '';
  position: fixed;
  top: -200px;
  left: 50%;
  transform: translateX(-50%);
  width: 600px;
  height: 600px;
  background: radial-gradient(ellipse, rgba(245,158,11,0.04) 0%, transparent 70%);
  filter: blur(60px);
  pointer-events: none;
  z-index: 0;
}
.stApp::after {
  content: '';
  position: fixed;
  bottom: -200px;
  right: -100px;
  width: 500px;
  height: 500px;
  background: radial-gradient(ellipse, rgba(245,158,11,0.03) 0%, transparent 70%);
  filter: blur(80px);
  pointer-events: none;
  z-index: 0;
}

/* ── Tipografía ───────────────────────────────────────────────────── */
h1, h2, h3, h4, h5, h6,
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
  font-family: 'Space Grotesk', system-ui, sans-serif !important;
  font-weight: 600 !important;
  letter-spacing: -0.025em !important;
  color: var(--fg) !important;
}
.stMarkdown p, .stMarkdown li { font-family: 'Inter', sans-serif !important; }
code, pre, .stCode { font-family: 'JetBrains Mono', monospace !important; }

/* ── Sidebar ──────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
  background: var(--bg-alt) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3,
[data-testid="stSidebar"] header {
  font-family: 'Space Grotesk', sans-serif !important;
}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
  padding-top: 0.5rem;
}

/* ── Header principal ─────────────────────────────────────────────── */
[data-testid="stHeader"] {
  background: transparent !important;
  border-bottom: 1px solid var(--border) !important;
}

/* ── Título ───────────────────────────────────────────────────────── */
.stApp [data-testid="stMarkdownContainer"] h1 {
  font-size: 1.75rem;
  font-weight: 700;
  background: linear-gradient(135deg, var(--fg) 60%, var(--accent));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

/* ── Métricas ─────────────────────────────────────────────────────── */
[data-testid="metric-container"] {
  background: var(--card) !important;
  backdrop-filter: blur(8px) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
  padding: 1rem 1.25rem !important;
  transition: all 300ms ease-out !important;
}
[data-testid="metric-container"]:hover {
  border-color: var(--border-h) !important;
  box-shadow: var(--glow-sm) !important;
}
[data-testid="metric-container"] [data-testid="stMetricLabel"] {
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 0.75rem !important;
  letter-spacing: 0.05em !important;
  color: var(--fg-muted) !important;
  text-transform: uppercase !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
  font-family: 'Space Grotesk', sans-serif !important;
  font-size: 1.5rem !important;
  font-weight: 600 !important;
  color: var(--fg) !important;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 0.8rem !important;
}

/* ── Tabs ─────────────────────────────────────────────────────────── */
[data-testid="stTabs"] [role="tablist"] {
  background: var(--bg-alt) !important;
  border-radius: 10px !important;
  padding: 4px !important;
  border: 1px solid var(--border) !important;
  gap: 2px !important;
}
[data-testid="stTabs"] [role="tab"] {
  font-family: 'Inter', sans-serif !important;
  font-size: 0.875rem !important;
  font-weight: 500 !important;
  color: var(--fg-muted) !important;
  border-radius: 7px !important;
  padding: 6px 16px !important;
  transition: all 200ms ease-out !important;
  border: none !important;
  background: transparent !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
  background: var(--bg-muted) !important;
  color: var(--fg) !important;
  border-color: var(--border) !important;
  box-shadow: var(--glow-sm) !important;
}
[data-testid="stTabs"] [role="tab"]:hover:not([aria-selected="true"]) {
  color: var(--fg) !important;
  background: rgba(255,255,255,0.04) !important;
}

/* ── Botones ──────────────────────────────────────────────────────── */
.stButton button {
  font-family: 'Inter', sans-serif !important;
  font-weight: 500 !important;
  border-radius: 10px !important;
  transition: all 200ms ease-out !important;
  border: 1px solid var(--border) !important;
  background: rgba(26,26,36,0.6) !important;
  color: var(--fg) !important;
}
.stButton button:hover {
  border-color: var(--accent) !important;
  color: var(--accent) !important;
  box-shadow: var(--glow-sm) !important;
  transform: scale(1.01) !important;
}
.stButton button:active {
  transform: scale(0.98) !important;
}
/* Botón primario (el primero en cada grupo suele ser el CTA) */
.stButton [kind="primary"] button,
.stButton button[kind="primary"] {
  background: var(--accent) !important;
  color: #0A0A0F !important;
  border: none !important;
  box-shadow: var(--glow-sm) !important;
}
.stButton [kind="primary"] button:hover {
  filter: brightness(1.1) !important;
  box-shadow: 0 0 20px rgba(245,158,11,0.4) !important;
  color: #0A0A0F !important;
}

/* ── Containers con borde (st.container(border=True)) ─────────────── */
[data-testid="stVerticalBlockBorderWrapper"] {
  background: var(--card) !important;
  backdrop-filter: blur(8px) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
  transition: all 300ms ease-out !important;
}
[data-testid="stVerticalBlockBorderWrapper"]:hover {
  border-color: var(--border-h) !important;
  box-shadow: var(--glow-sm) !important;
}

/* ── Inputs / Selectbox / Slider ──────────────────────────────────── */
.stTextInput input, .stNumberInput input, .stSelectbox select,
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input {
  background: var(--card) !important;
  backdrop-filter: blur(8px) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  color: var(--fg) !important;
  font-family: 'Inter', sans-serif !important;
  transition: all 200ms !important;
}
.stTextInput input:focus, .stNumberInput input:focus {
  border-color: rgba(245,158,11,0.5) !important;
  box-shadow: 0 0 0 2px rgba(245,158,11,0.2), 0 0 20px rgba(245,158,11,0.1) !important;
  outline: none !important;
}

[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
  background: var(--accent) !important;
  box-shadow: var(--glow-sm) !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] [data-testid="stThumbValue"] {
  color: var(--accent) !important;
}

/* ── Toggle / Checkbox ────────────────────────────────────────────── */
[data-testid="stToggle"] [data-testid="stMarkdownContainer"] p {
  font-size: 0.875rem !important;
  color: var(--fg) !important;
}

/* ── Alerts (info, warning, error, success) ───────────────────────── */
[data-testid="stAlert"] {
  border-radius: 10px !important;
  border-left-width: 3px !important;
  backdrop-filter: blur(8px) !important;
}
[data-testid="stAlert"][data-baseweb="notification"][kind="info"] {
  background: rgba(245,158,11,0.08) !important;
  border-left-color: var(--accent) !important;
}
[data-testid="stAlert"][data-baseweb="notification"][kind="warning"] {
  background: rgba(245,158,11,0.10) !important;
  border-left-color: var(--accent) !important;
}
[data-testid="stAlert"][data-baseweb="notification"][kind="success"] {
  background: rgba(0,184,148,0.08) !important;
  border-left-color: #00b894 !important;
}
[data-testid="stAlert"][data-baseweb="notification"][kind="error"] {
  background: rgba(220,38,38,0.08) !important;
  border-left-color: #dc2626 !important;
}

/* ── Expanders ────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  backdrop-filter: blur(8px) !important;
  transition: all 200ms ease-out !important;
}
[data-testid="stExpander"]:hover {
  border-color: var(--border-h) !important;
}
[data-testid="stExpander"] summary {
  font-family: 'Inter', sans-serif !important;
  font-weight: 500 !important;
  color: var(--fg) !important;
}

/* ── Dataframes / Tablas ──────────────────────────────────────────── */
[data-testid="stDataFrame"] {
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  overflow: hidden !important;
}
[data-testid="stDataFrame"] iframe {
  border-radius: 10px !important;
}

/* ── Captions / Labels ────────────────────────────────────────────── */
[data-testid="stCaptionContainer"] p,
.stCaption {
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 0.75rem !important;
  color: var(--fg-muted) !important;
  letter-spacing: 0.02em !important;
}

/* ── Divider ──────────────────────────────────────────────────────── */
hr {
  border-color: var(--border) !important;
  margin: 1.5rem 0 !important;
}

/* ── Scrollbar ────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-alt); }
::-webkit-scrollbar-thumb { background: var(--bg-muted); border-radius: 99px; }
::-webkit-scrollbar-thumb:hover { background: var(--fg-muted); }

/* ── Code blocks ──────────────────────────────────────────────────── */
.stCode, [data-testid="stCode"] {
  background: var(--bg-muted) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  font-family: 'JetBrains Mono', monospace !important;
}

/* ── Charts ───────────────────────────────────────────────────────── */
[data-testid="stVegaLiteChart"],
[data-testid="stArrowVegaLiteChart"] {
  background: transparent !important;
}

/* ── Footer / main block ──────────────────────────────────────────── */
footer { display: none !important; }
#MainMenu { visibility: hidden !important; }
[data-testid="stToolbar"] { display: none !important; }
</style>
"""


def _inject_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


# ── Recursos cacheados ────────────────────────────────────────────────────────

@st.cache_resource
def _build_provider(demo: bool) -> MarketDataProvider:
    if demo:
        return DemoProvider()
    try:
        from optionsdesk.data.providers.homebroker import HomeBrokerProvider
        p = HomeBrokerProvider()
        p.connect()
        return p
    except Exception as exc:
        st.warning(f"HomeBroker no disponible ({exc}). Usando modo demo.")
        return DemoProvider()


@st.cache_resource
def _load_expiry_calendar() -> dict:
    try:
        from optionsdesk.core.instruments import load_expiry_calendar
        return load_expiry_calendar()
    except Exception:
        return {}


@st.cache_resource
def _get_paper_executor() -> PaperExecutor:
    return PaperExecutor()


# ── Helpers de datos ──────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def _load_spot_history(days: int = 180) -> Optional[pd.DataFrame]:
    """Historial OHLCV de GGAL vía PyOBD (con cache local y fallback sintético)."""
    try:
        from optionsdesk.data.history import UnderlyingHistory
        df = UnderlyingHistory().daily("GGAL", days=days)
        return df if not df.empty else None
    except Exception:
        return None


@st.cache_data(ttl=3600)
def _load_htf_history(weeks: int = 52) -> Optional[pd.DataFrame]:
    """Historial semanal (HTF) de GGAL — resampleado desde daily."""
    try:
        from optionsdesk.data.history import UnderlyingHistory
        df = UnderlyingHistory().weekly("GGAL", weeks=weeks)
        return df if not df.empty else None
    except Exception:
        return None


@st.cache_data(ttl=300)
def _load_ltf_history() -> Optional[pd.DataFrame]:
    """Historial intradiario de GGAL (LTF). TTL corto para reflejar precio en tiempo real."""
    try:
        from optionsdesk.data.history import UnderlyingHistory
        df = UnderlyingHistory().intraday("GGAL")
        return df if not df.empty else None
    except Exception:
        return None


# ── Helpers de formateo ───────────────────────────────────────────────────────

def _fmt_signals(results: list[RateResult]) -> pd.DataFrame:
    if not results:
        return pd.DataFrame()
    return pd.DataFrame([
        {
            "Simbolo": r.symbol,
            "Strike": r.strike,
            "Venc.": r.expiration.strftime("%d/%m"),
            "Dias": r.days,
            "Prima": round(r.premium, 1),
            "TNA %": round(r.tna_pct, 1),
            "TEA %": round(r.tea_pct, 1),
            "Spread": round(r.spread_vs_caucion_pct, 1),
            "Colchon %": round(r.cushion_pct, 1),
            "Delta": round(r.delta, 2) if r.delta is not None else None,
            "IV %": round(r.iv * 100, 0) if r.iv else None,
            "M'ness": r.moneyness,
        }
        for r in results
    ])


_CHAIN_RE = re.compile(r"^GFG([CV])(\d+(?:[.,]\d+)?)([A-Z]+)$")


def _build_chain_pivot(chain: OptionsChain) -> dict[str, pd.DataFrame]:
    rows = []
    for sym, q in chain.options.items():
        m = _CHAIN_RE.match(sym)
        if not m:
            continue
        rows.append({
            "code": m.group(3),
            "strike": float(m.group(2).replace(",", ".")),
            "type": m.group(1),
            "bid": q.bid,
            "ask": q.ask,
            "mid": round(q.mid, 1),
            "vol": q.volume,
        })
    if not rows:
        return {}

    df = pd.DataFrame(rows)
    result: dict[str, pd.DataFrame] = {}
    for code in sorted(df["code"].unique()):
        sub = df[df["code"] == code]
        calls = (
            sub[sub["type"] == "C"]
            .set_index("strike")[["vol", "bid", "ask", "mid"]]
            .rename(columns={"vol": "C-Vol", "bid": "C-Bid", "ask": "C-Ask", "mid": "C-Mid"})
        )
        puts = (
            sub[sub["type"] == "P"]
            .set_index("strike")[["mid", "bid", "ask", "vol"]]
            .rename(columns={"mid": "P-Mid", "bid": "P-Bid", "ask": "P-Ask", "vol": "P-Vol"})
        )
        merged = calls.join(puts, how="outer").reset_index().sort_values("strike")
        result[code] = merged
    return result


def _style_chain(df: pd.DataFrame, spot: float) -> "pd.io.formats.style.Styler":
    def _row(row: pd.Series) -> list[str]:
        styles = [""] * len(row)
        s = float(row["strike"])
        idx = list(row.index)
        if s < spot:
            for i, c in enumerate(idx):
                if c.startswith("C-"):
                    styles[i] = "background-color: rgba(0,160,0,0.18)"
        if s > spot:
            for i, c in enumerate(idx):
                if c.startswith("P-"):
                    styles[i] = "background-color: rgba(220,0,0,0.15)"
        return styles

    return df.style.apply(_row, axis=1).format(
        {c: "{:.1f}" for c in df.columns if c not in ("strike", "C-Vol", "P-Vol")}
    ).format({"strike": "${:,.0f}"})


def _payoff_df(r: RateResult) -> pd.DataFrame:
    s_min = min(r.strike, r.spot) * 0.55
    s_max = max(r.strike, r.spot) * 1.45
    spots = np.linspace(s_min, s_max, 400)

    if r.strategy == "COVERED_CALL":
        div_adj = r.net_proceeds - r.strike
        pnl = np.where(
            spots >= r.strike,
            r.net_proceeds - r.net_outlay,
            spots + div_adj - r.net_outlay,
        )
    else:
        pnl = np.where(
            spots >= r.strike,
            r.net_proceeds,
            spots - r.strike + r.net_proceeds,
        )

    pct = pnl / r.net_outlay * 100.0
    return pd.DataFrame(
        {"P&L ($)": pnl, "P&L (%)": pct, "Cero": np.zeros(len(spots))},
        index=spots,
    )


def _spread_payoff_df(sr) -> pd.DataFrame:
    """Payoff al vencimiento para un SpreadResult de 2 patas."""
    K_lo = min(sr.long_leg.strike, sr.short_leg.strike)
    K_hi = max(sr.long_leg.strike, sr.short_leg.strike)
    mp   =  sr.max_profit     # positivo
    ml   = -sr.max_loss       # negativo

    spots = np.linspace(sr.spot * 0.70, sr.spot * 1.30, 300)

    bullish = sr.strategy in ("BULL_CALL", "BULL_PUT")
    pnl_lo  = ml if bullish else mp
    pnl_hi  = mp if bullish else ml

    pnl = np.where(
        spots <= K_lo, pnl_lo,
        np.where(
            spots >= K_hi, pnl_hi,
            pnl_lo + (spots - K_lo) / (K_hi - K_lo) * (pnl_hi - pnl_lo),
        ),
    )
    return pd.DataFrame({"P&L ($)": pnl, "Cero": np.zeros(len(spots))}, index=spots)


# ── Tab: Inicio ───────────────────────────────────────────────────────────────

_PROFILE_META: dict[RiskProfile, dict[str, str]] = {
    RiskProfile.CONSERVADOR: {
        "label": "Conservadora",
        "desc":  "Prioriza colchon y probabilidad. Para quien no quiere sorpresas.",
    },
    RiskProfile.EQUILIBRADO: {
        "label": "Equilibrada",
        "desc":  "Balance entre tasa y seguridad. La opcion predeterminada.",
    },
    RiskProfile.AGRESIVO: {
        "label": "Agresiva",
        "desc":  "Maxima tasa posible. Mayor riesgo asumido.",
    },
}


def _submit_paper_orders(rec: Recommendation) -> None:
    executor = _get_paper_executor()
    n = max(rec.contracts, 1)
    ref = f"{rec.result.strategy}:{rec.result.symbol}"
    if rec.result.strategy == "COVERED_CALL":
        executor.submit(Order(
            symbol="GGAL",
            side=OrderSide.BUY,
            quantity=n * 100,
            limit_price=rec.result.spot,
            strategy_ref=ref,
        ))
        executor.submit(Order(
            symbol=rec.result.symbol,
            side=OrderSide.SELL,
            quantity=n,
            limit_price=rec.result.premium,
            strategy_ref=ref,
        ))
    else:
        executor.submit(Order(
            symbol=rec.result.symbol,
            side=OrderSide.SELL,
            quantity=n,
            limit_price=rec.result.premium,
            strategy_ref=ref,
        ))


def _render_single_rec(
    rec: Recommendation,
    profile: RiskProfile,
    idx: int,
    alerter: TelegramAlerter,
) -> None:
    """Tarjeta interna para una Recommendation individual."""
    # Semáforo
    if rec.light == "verde":
        st.success(f"APROBADA  —  Score {rec.score:.0f}/100")
    elif rec.light == "amarillo":
        st.warning(f"CON ADVERTENCIAS  —  Score {rec.score:.0f}/100")

    # Badge HOLD / SWING
    hp = rec.horizon_plan
    if hp is not None:
        if hp.mode == "SWING":
            st.info(
                f"**SWING** — Cerrá en ~{hp.target_exit_days} días  "
                f"({hp.target_capture_pct:.0f}% captura · TNA opt. {hp.optimized_tna_pct:.1f}%)"
            )
        else:
            st.info(f"**HOLD** hasta vencimiento — TNA proyectada {hp.optimized_tna_pct:.1f}%")

    # VolEdge liviano
    ve = rec.vol_edge
    if ve is not None and ve.label != "sin datos":
        color_map = {"positivo": "verde", "neutro": "gris", "negativo": "rojo"}
        ve_icon = {"positivo": "IV cara — edge del vendedor",
                   "neutro": "IV neutra",
                   "negativo": "IV barata — precaucion"}
        st.caption(f"Vol edge: {ve_icon.get(ve.label, ve.label)}"
                   + (f"  |  VRP {ve.vrp*100:+.1f}%" if ve.vrp is not None else ""))

    # Métricas clave
    c1, c2 = st.columns(2)
    tna_label = f"{hp.optimized_tna_pct:.1f}%" if hp else f"{rec.result.tna_pct:.1f}%"
    c1.metric("TNA anual", tna_label)
    c2.metric("Colchon", f"{rec.result.cushion_pct:.1f}%")
    c3, c4 = st.columns(2)
    c3.metric("Probabilidad", f"{rec.success_probability * 100:.0f}%")
    if rec.expected_profit_ars is not None:
        c4.metric("Ganancia estimada", f"${rec.expected_profit_ars:,.0f}")
    else:
        c4.metric("Spread vs caucion", f"+{rec.result.spread_vs_caucion_pct:.1f}%")

    # Detalle expandible
    with st.expander("Ver detalle y pasos"):
        st.write(rec.plain_explanation)
        st.markdown("**Pasos:**")
        for step in rec.action_steps:
            st.markdown(f"- {step}")

        if hp is not None:
            st.divider()
            st.markdown(f"**Plan de salida ({hp.mode})**")
            st.write(hp.plain_explanation)
            gc1, gc2, gc3 = st.columns(3)
            gc1.metric("Theta/día (contrato)", f"${hp.theta_daily_ars:,.0f}")
            gc2.metric("Gamma", f"{hp.gamma:.5f}")
            gc3.metric("Vega (1% IV)", f"{hp.vega_1pct:.2f}")
            if hp.warnings:
                for w in hp.warnings:
                    st.caption(f"Aviso: {w}")

        if rec.warnings:
            st.markdown("**Advertencias:**")
            for w in rec.warnings:
                st.warning(w)

        df_pay = _payoff_df(rec.result)
        chart_df = df_pay[["P&L ($)", "Cero"]].copy()
        chart_df.index.name = "Spot"
        st.caption(f"P&L al vencimiento — {rec.result.symbol}")
        st.line_chart(chart_df, color=["#00b894", "#636e72"])

    # Botones de accion
    col_a, col_b = st.columns(2)
    ticket_key = f"ticket_shown_{profile.value}_{idx}"
    with col_a:
        if st.button("Generar ticket", key=f"btn_ticket_{profile.value}_{idx}"):
            _submit_paper_orders(rec)
            st.session_state[ticket_key] = True
    with col_b:
        if st.button("Telegram", key=f"btn_tg_{profile.value}_{idx}"):
            ok = alerter.send_recommendation(rec)
            if ok:
                st.toast("Enviado a Telegram")
            else:
                st.toast("Telegram no configurado")

    if st.session_state.get(ticket_key):
        st.code(rec.ticket_text, language="text")
        st.caption("Copia y pega en Bull Market. Orden registrada en data/paper_orders.jsonl")


def _render_rec_card(
    recs: list[Recommendation],
    profile: RiskProfile,
    alerter: TelegramAlerter,
) -> None:
    meta = _PROFILE_META[profile]

    with st.container(border=True):
        st.subheader(meta["label"])
        st.caption(meta["desc"])

        if not recs:
            st.info("Sin oportunidades para este perfil con los datos actuales.")
            return

        # Tarjeta principal (mejor candidato)
        primary = recs[0]
        st.markdown(f"**{primary.result.symbol}** — {primary.result.strategy.replace('_', ' ').title()}")
        _render_single_rec(primary, profile, 0, alerter)

        # Alternativas en expander
        if len(recs) > 1:
            with st.expander(f"Alternativas ({len(recs) - 1})"):
                for i, alt in enumerate(recs[1:], start=1):
                    st.markdown(
                        f"**#{i+1}** — {alt.result.symbol} | "
                        f"TNA {alt.result.tna_pct:.1f}% | "
                        f"Colchon {alt.result.cushion_pct:.1f}% | "
                        f"Score {alt.score:.0f}"
                    )
                    hp_alt = alt.horizon_plan
                    if hp_alt:
                        mode_badge = f"**{hp_alt.mode}**" + (
                            f" ~{hp_alt.target_exit_days}d" if hp_alt.mode == "SWING" else ""
                        )
                        st.caption(mode_badge)
                    alt_ticket_key = f"ticket_shown_{profile.value}_{i}"
                    if st.button("Ticket", key=f"btn_alt_ticket_{profile.value}_{i}"):
                        _submit_paper_orders(alt)
                        st.session_state[alt_ticket_key] = True
                    if st.session_state.get(alt_ticket_key):
                        st.code(alt.ticket_text, language="text")
                    if i < len(recs) - 1:
                        st.divider()


def _tab_home(
    recs: dict[RiskProfile, list[Recommendation]],
    alerter: TelegramAlerter,
) -> None:
    cols = st.columns(3)
    for col, profile in zip(cols, RiskProfile):
        with col:
            _render_rec_card(recs.get(profile, []), profile, alerter)


# ── Tab: Oportunidades ────────────────────────────────────────────────────────

def _tab_opportunities(
    cc_all: list[RateResult],
    sp_all: list[RateResult],
    cc_filtered: list[RateResult],
    sp_filtered: list[RateResult],
    show_all: bool,
    caucion_tna: float,
    send_alerts: bool,
    alerter: TelegramAlerter,
) -> None:
    col1, col2 = st.columns(2)
    col_config = {
        "Strike": st.column_config.NumberColumn("Strike", format="$%,.0f"),
        "TNA %": st.column_config.NumberColumn("TNA %", format="%.1f %%"),
        "TEA %": st.column_config.NumberColumn("TEA %", format="%.1f %%"),
        "Spread": st.column_config.NumberColumn("Spread", format="+%.1f %%"),
        "Colchon %": st.column_config.NumberColumn("Colchon %", format="%.1f %%"),
    }

    with col1:
        st.subheader("Lanzamientos cubiertos")
        display = cc_all if show_all else cc_filtered
        if display:
            st.caption(f"{len(display)} oportunidad{'es' if len(display) != 1 else ''}")
            st.dataframe(
                _fmt_signals(display), width="stretch",
                hide_index=True, column_config=col_config,
            )
        else:
            st.info("Ninguna supera el filtro de spread. Baja el umbral en el sidebar.")

    with col2:
        st.subheader("Venta de puts (cash-secured)")
        display = sp_all if show_all else sp_filtered
        if display:
            st.caption(f"{len(display)} oportunidad{'es' if len(display) != 1 else ''}")
            st.dataframe(
                _fmt_signals(display), width="stretch",
                hide_index=True, column_config=col_config,
            )
        else:
            st.info("Ninguna supera el filtro de spread.")

    if caucion_tna > 0:
        st.caption(
            f"Spread = TNA opcion - caucion ({caucion_tna:.1f}% TNA). "
            "Ordenado por spread, de mayor a menor."
        )

    # Advertencias top oportunidad
    top = (cc_filtered or cc_all)[:1]
    if top:
        from optionsdesk.risk.limits import RiskChecker
        _, warnings = RiskChecker().check_opportunity(top[0])
        if warnings:
            with st.expander("Advertencias de riesgo — oportunidad top"):
                for w in warnings:
                    st.warning(w)

    if send_alerts and cc_filtered:
        alerter.send_top_opportunities(cc_filtered[:3])


# ── Tab: Aprende ──────────────────────────────────────────────────────────────

def _tab_learn() -> None:
    guide_path = Path("GUIA.md")
    if not guide_path.exists():
        # Fallback: buscar relativo al archivo del dashboard
        guide_path = Path(__file__).parent.parent.parent / "GUIA.md"

    if guide_path.exists():
        st.markdown(guide_path.read_text(encoding="utf-8"))
    else:
        st.info("No se encontro GUIA.md en la raiz del proyecto.")


# ── Tab: Cadena completa ──────────────────────────────────────────────────────

def _tab_chain(chain: OptionsChain, spot: float) -> None:
    pivots = _build_chain_pivot(chain)
    if not pivots:
        st.info("Sin datos de cadena.")
        return

    st.caption(
        f"GGAL spot: ${spot:,.0f}  |  "
        "Calls (C-*) izquierda, puts (P-*) derecha. "
        "Verde = call ITM. Rojo = put ITM."
    )
    venc_tabs = st.tabs([f"Venc. {code}" for code in pivots])
    for tab, (code, df) in zip(venc_tabs, pivots.items()):
        with tab:
            st.dataframe(_style_chain(df, spot), width="stretch", hide_index=True)


# ── Tab: Simulador P&L ────────────────────────────────────────────────────────

def _tab_simulator(cc_all: list[RateResult], sp_all: list[RateResult], spot: float) -> None:
    options = (
        [(f"CC | {r.symbol} | TNA {r.tna_pct:.0f}%", r) for r in cc_all[:10]]
        + [(f"SP | {r.symbol} | TNA {r.tna_pct:.0f}%", r) for r in sp_all[:10]]
    )
    if not options:
        st.info("Sin oportunidades. Relaja los filtros en el sidebar.")
        return

    label_map = dict(options)
    sel = st.selectbox("Selecciona una oportunidad", list(label_map))
    r = label_map[sel]

    if r.strategy == "COVERED_CALL":
        breakeven = r.net_outlay - (r.net_proceeds - r.strike)
        max_profit = r.net_proceeds - r.net_outlay
        title = "P&L al vencimiento — Lanzamiento cubierto"
    else:
        breakeven = r.strike - r.net_proceeds
        max_profit = r.net_proceeds
        title = "P&L al vencimiento — Venta de put"

    max_profit_pct = max_profit / r.net_outlay * 100.0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Strike", f"${r.strike:,.0f}")
    c2.metric("Prima", f"${r.premium:,.1f}")
    c3.metric("Break-even", f"${breakeven:,.0f}")
    c4.metric("Ganancia max.", f"${max_profit:,.0f}  ({max_profit_pct:.1f}%)")
    c5.metric("TNA / TEA", f"{r.tna_pct:.1f}% / {r.tea_pct:.1f}%")

    st.subheader(title)
    pct_view = st.toggle("Ver en % del capital comprometido", value=False)
    col = "P&L (%)" if pct_view else "P&L ($)"
    df_pay = _payoff_df(r)
    chart_df = df_pay[[col, "Cero"]].copy()
    chart_df.index.name = "Spot al vencimiento"
    st.line_chart(chart_df, color=["#00b894", "#636e72"])

    iv_note = f"  |  IV: {r.iv * 100:.0f}%" if r.iv else ""
    st.caption(
        f"Spot actual: ${spot:,.0f}  |  Strike: ${r.strike:,.0f}  |  "
        f"Capital comprometido: ${r.net_outlay:,.0f}  |  Dias: {r.days}{iv_note}"
    )


# ── Tab: Direccional ─────────────────────────────────────────────────────────

_MTF_LABEL = {
    "alineado_alcista": ("Marcos alineados alcistas", "success"),
    "alineado_bajista": ("Marcos alineados bajistas", "error"),
    "conflicto":        ("Conflicto HTF / Diario",    "warning"),
    "neutral":          ("Marcos neutros",            "info"),
}
_VOL_LABEL = {
    "expansion":  ("Vol en expansion",   "warning"),
    "contraccion":("Vol en contraccion", "info"),
    "normal":     ("Vol normal",         "info"),
}


def _tab_directional(
    spot_history: Optional[pd.DataFrame],
    chain: Optional["OptionsChain"],
    spot: float,
    context,
    htf_df: Optional[pd.DataFrame] = None,
    ltf_df: Optional[pd.DataFrame] = None,
    expiry_calendar: Optional[dict] = None,
    benchmark=None,
) -> None:
    st.subheader("Analisis tecnico — GGAL")
    st.caption(
        "Indicadores sobre precio diario. Fuente: BYMA Open Data (PyOBD, ~20 min delay). "
        "Esta tab es para swing de dias, no scalping."
    )

    if spot_history is None or spot_history.empty:
        st.info("Sin historial disponible. Esperando datos de PyOBD o recorder.")
        return

    from optionsdesk.signals.technical import analyze as _analyze, sma as _sma, rsi as _rsi

    snap = _analyze(spot_history)

    # ── MTF + Vol badges ──────────────────────────────────────────────────
    if context is not None:
        alignment  = getattr(context, "mtf_alignment", "neutral")
        vol_regime = getattr(context, "vol_regime",    "normal")
        htf_trend  = getattr(context, "htf_trend",    "lateral")
        ltf_trend  = getattr(context, "ltf_trend",    "lateral")

        b1, b2, b3, b4 = st.columns(4)
        mtf_txt, mtf_fn = _MTF_LABEL.get(alignment, ("Neutral", "info"))
        getattr(b1, mtf_fn)(mtf_txt)
        vol_txt, vol_fn = _VOL_LABEL.get(vol_regime, ("Vol normal", "info"))
        getattr(b2, vol_fn)(vol_txt)
        b3.info(f"Semanal: {htf_trend.capitalize()}")
        ltf_lbl = f"Intradiario: {ltf_trend.capitalize()}" if ltf_df is not None else "Intradiario: sin datos"
        b4.info(ltf_lbl)

    # ── Panel 3 marcos temporales ─────────────────────────────────────────
    with st.expander("Panel 3 marcos temporales", expanded=False):
        tf1, tf2, tf3 = st.columns(3)

        # Semanal (HTF)
        with tf1:
            st.markdown("**Semanal (HTF)**")
            htf_snap = getattr(context, "htf_snap", None) if context else None
            if htf_snap is not None:
                st.metric("Tendencia", htf_snap.trend.capitalize())
                st.metric("RSI(14)",   f"{htf_snap.rsi:.0f}")
                st.metric("ATR %",     f"{htf_snap.atr_pct:.1f}%")
                st.metric("Señal",     htf_snap.signal_strength.capitalize())
            elif htf_df is not None and not htf_df.empty:
                st.caption("Datos semanales disponibles pero sin análisis suficiente.")
            else:
                st.caption("Sin historial semanal.")
            if htf_df is not None and not htf_df.empty and len(htf_df) >= 4:
                n = min(len(htf_df), 26)
                htf_sub = htf_df.tail(n)
                st.line_chart(
                    pd.DataFrame({"Semanal": htf_sub["close"].values},
                                 index=htf_sub["date"].values),
                    color=["#7c3aed"], height=130,
                )

        # Diario (base)
        with tf2:
            st.markdown("**Diario (base)**")
            st.metric("Tendencia", snap.trend.capitalize())
            st.metric("RSI(14)",   f"{snap.rsi:.0f}")
            st.metric("ATR %",     f"{snap.atr_pct:.1f}%")
            st.metric("Señal",     snap.signal_strength.capitalize())
            n = min(len(spot_history), 20)
            d_sub = spot_history.tail(n)
            st.line_chart(
                pd.DataFrame({"Diario": d_sub["close"].values}, index=d_sub["date"].values),
                color=["#F59E0B"], height=130,
            )

        # Intradiario (LTF)
        with tf3:
            st.markdown("**Intradiario (LTF)**")
            ltf_snap = getattr(context, "ltf_snap", None) if context else None
            if ltf_snap is not None:
                st.metric("Tendencia", ltf_snap.trend.capitalize())
                st.metric("RSI(14)",   f"{ltf_snap.rsi:.0f}")
                st.metric("ATR %",     f"{ltf_snap.atr_pct:.1f}%")
                st.metric("Señal",     ltf_snap.signal_strength.capitalize())
            else:
                st.caption("Sin analisis intradía (fuera de horario o sin datos).")
            if ltf_df is not None and not ltf_df.empty and "close" in ltf_df.columns:
                ltf_sub = ltf_df.tail(60)
                st.line_chart(
                    pd.DataFrame({"Intradiario": ltf_sub["close"].values}),
                    color=["#00cec9"], height=130,
                )

    # ── Gráfico de precio con medias móviles ──────────────────────────────
    close = spot_history["close"]
    dates = spot_history["date"]

    sma5_s  = _sma(close, 5).values
    sma20_s = _sma(close, 20).values
    chart_df = pd.DataFrame({
        "GGAL":    close.values,
        "SMA(5)":  sma5_s,
        "SMA(20)": sma20_s,
    }, index=dates)
    st.line_chart(chart_df, color=["#F59E0B", "#00b894", "#636e72"])

    # ── Métricas clásicas ─────────────────────────────────────────────────
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Tendencia", snap.trend.capitalize())
    c2.metric("RSI(14)", f"{snap.rsi:.0f}")
    c3.metric("ATR %", f"{snap.atr_pct:.1f}%")
    c4.metric("Momentum 10d", f"{snap.momentum:+.1f}%")
    c5.metric("MACD hist.", f"{snap.macd_hist:+.0f}")
    c6.metric("Señal", snap.signal_strength.capitalize())

    st.caption(snap.read)

    # ── SMC badges ────────────────────────────────────────────────────────
    smc_cols = st.columns(4)
    with smc_cols[0]:
        if snap.bos == "BOS_UP":
            st.success("BOS alcista")
        elif snap.bos == "BOS_DOWN":
            st.error("BOS bajista")
        else:
            st.info("Sin BOS reciente")

    with smc_cols[1]:
        if snap.choch == "CHOCH_UP":
            st.success("CHoCH: giro alcista")
        elif snap.choch == "CHOCH_DOWN":
            st.warning("CHoCH: giro bajista")
        else:
            st.info("Sin CHoCH")

    with smc_cols[2]:
        ob = snap.order_block
        if ob:
            ob_tipo = "Demanda" if ob.type == "bullish" else "Oferta"
            st.info(f"OB {ob_tipo}: ${ob.low:,.0f}–${ob.high:,.0f}")
        else:
            st.info("Sin Order Block")

    with smc_cols[3]:
        fvg = snap.nearest_fvg
        if fvg:
            tipo = "alcista" if fvg.type == "bullish" else "bajista"
            st.info(f"FVG {tipo}: ${fvg.low:,.0f}–${fvg.high:,.0f}")
        else:
            st.info("Sin FVG cercano")

    with st.expander("RSI(14) — detalle"):
        rsi14  = _rsi(close, 14)
        rsi_df = pd.DataFrame({"RSI(14)": rsi14.values}, index=dates)
        st.line_chart(rsi_df, color=["#00cec9"])
        st.caption("Sobrecomprado >70 | Sobrevendido <30")

    st.divider()

    # ── Idea direccional (opcion desnuda) ─────────────────────────────────
    st.subheader("Idea direccional")
    st.warning(
        "Las siguientes ideas son **especulativas**. El bot no tiene alpha direccional "
        "validado. La pérdida máxima es la prima pagada. Solo operar con capital de riesgo."
    )

    if context is None or chain is None:
        st.info("Sin contexto de mercado disponible.")
        return

    daily_snap = getattr(context, "snap", None)
    idea = build_directional_idea(context, chain, spot, snap=daily_snap)

    if idea is None:
        st.info(
            f"Sin idea para el mercado actual "
            f"(tendencia '{context.trend}', confianza '{context.confidence}', "
            f"fuerza '{context.signal_strength}'). "
            "Se requiere confianza media+ y momentum >1.5% para generar una sugerencia."
        )
    else:
        with st.container(border=True):
            tipo_txt = "COMPRA DE CALL (alcista)" if idea.idea_type == "BUY_CALL" else "COMPRA DE PUT (bajista)"
            st.markdown(f"**{tipo_txt}** — {idea.symbol}")

            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Strike",               f"${idea.strike:,.0f}")
            d2.metric("Prima (mid)",           f"${idea.mid_price:,.0f}")
            d3.metric("Costo total (1 cto.)", f"${idea.total_cost_ars:,.0f}")
            d4.metric("Dias al vencimiento",   str(idea.days_to_expiry))

            e1, e2, e3 = st.columns(3)
            e1.metric("Break-even", f"${idea.breakeven:,.0f}")
            e2.metric("Objetivo",   f"${idea.target_spot:,.0f}")
            e3.metric("Stop",       f"${idea.stop_spot:,.0f}")

            st.write(idea.rational)
            st.caption(idea.disclaimer)

    st.divider()

    # ── Spread vertical (riesgo definido) ─────────────────────────────────
    st.subheader("Spread vertical — riesgo definido")
    st.caption(
        "Spread de 2 patas con perdida maxima conocida. Mas disciplinado que la opcion desnuda. "
        "Requiere liquidez en ambas patas (spread bid-ask acotado)."
    )

    spread_result = build_directional_spread(
        context, chain, spot, snap=daily_snap,
        expiry_calendar=expiry_calendar or {},
        benchmark=benchmark,
    )

    if spread_result is None:
        st.info(
            "Sin spread viable con los datos actuales. "
            "Posibles causas: opciones poco líquidas, señal débil o conflicto multi-TF."
        )
    else:
        sr = spread_result
        credit_lbl = "CREDITO" if sr.is_credit else "DEBITO"
        st.markdown(f"**{sr.strategy.replace('_', ' ')}** — {credit_lbl}")

        f1, f2, f3, f4, f5 = st.columns(5)
        f1.metric("Long leg",   f"K={sr.long_leg.strike:,.0f}  ({sr.long_leg.action})")
        f2.metric("Short leg",  f"K={sr.short_leg.strike:,.0f} ({sr.short_leg.action})")
        if sr.is_credit:
            f3.metric("Credito neto / accion", f"${-sr.net_debit:,.0f}")
        else:
            f3.metric("Debito neto / accion",  f"${sr.net_debit:,.0f}")
        f4.metric("Max ganancia (1 cto.)", f"${sr.max_profit:,.0f}")
        f5.metric("Max perdida (1 cto.)",  f"${sr.max_loss:,.0f}")

        g1, g2, g3, g4 = st.columns(4)
        g1.metric("Break-even",    f"${sr.breakeven:,.0f}")
        g2.metric("R:R",           f"{sr.risk_reward:.1f}x")
        g3.metric("Prob. ganancia",f"{sr.prob_of_profit * 100:.0f}%")
        g4.metric("Dias",          str(sr.days))

        with st.expander("Payoff al vencimiento"):
            pay_df = _spread_payoff_df(sr)
            pay_df.index.name = "Spot al vencimiento"
            st.line_chart(pay_df[["P&L ($)", "Cero"]], color=["#00b894", "#636e72"])
            st.caption(
                f"Break-even ${sr.breakeven:,.0f}  |  "
                f"Max ganancia ${sr.max_profit:,.0f}  |  "
                f"Max perdida ${sr.max_loss:,.0f}"
            )

        st.write(sr.rational)
        st.caption(sr.disclaimer)


# ── Tab: Historial ────────────────────────────────────────────────────────────

def _tab_history() -> None:
    snapshots_dir = Path("data/snapshots")
    if not snapshots_dir.exists() or not any(snapshots_dir.iterdir()):
        st.info(
            "No hay historial grabado. "
            "Corre el recorder en horario de mercado para acumular datos."
        )
        st.code("python -m optionsdesk.data.recorder", language="bash")
        return

    dates = sorted(
        [d.name for d in snapshots_dir.iterdir() if d.is_dir()], reverse=True
    )
    sel_date = st.selectbox("Fecha", dates)
    try:
        from optionsdesk.data.recorder import ChainRecorder
        df = ChainRecorder.load_day(sel_date)
        if df is None or df.empty:
            st.info("Sin datos para esa fecha.")
        else:
            n_snaps = df["timestamp"].nunique() if "timestamp" in df.columns else "?"
            st.caption(f"{len(df)} registros — {n_snaps} snapshots")
            st.dataframe(df, width="stretch", hide_index=True)
    except Exception as exc:
        st.error(f"Error cargando historial: {exc}")


# ── Footer con auto-refresh via fragment ─────────────────────────────────────

@st.fragment(run_every=2)
def _auto_refresh_footer() -> None:
    """Actualiza el contador y dispara rerun de la app cuando los datos vencen.

    Al usar @st.fragment(run_every=2) el resto de la pagina queda idle —
    sin spinner — y el screenshot / preview funciona correctamente.
    """
    rs: int = st.session_state.get("_refresh_s", 60)
    elapsed = time.time() - st.session_state.get("last_fetch", time.time())
    remaining = max(0, rs - elapsed)

    col_foot1, col_foot2 = st.columns([4, 1])
    with col_foot1:
        st.caption(
            f"Datos: {pd.Timestamp.now().strftime('%H:%M:%S')} — "
            f"proxima actualizacion en {remaining:.0f}s"
        )
    with col_foot2:
        if st.button("Actualizar ahora", key="btn_refresh_main"):
            st.session_state.pop("last_fetch", None)
            st.rerun(scope="app")

    if remaining <= 0:
        st.rerun(scope="app")


# ── App principal ─────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="OptionsDesk GGAL",
        layout="wide",
        page_icon=":chart_with_upwards_trend:",
    )
    _inject_css()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("OptionsDesk")

        demo_mode = st.toggle(
            "Modo demo", value=not settings.is_configured(),
            help="Datos sinteticos. Activa cuando no hay credenciales.",
        )
        refresh_s = st.slider("Refresh (seg)", 15, 300, 60)

        st.divider()

        capital_input = st.number_input(
            "Capital disponible (ARS)",
            min_value=0,
            max_value=100_000_000,
            value=int(settings.default_capital or 0),
            step=50_000,
            help="Para calcular cuantos lotes entran y la ganancia estimada en pesos.",
        )
        capital = float(capital_input) if capital_input > 0 else None

        directional_on = st.toggle(
            "Contexto de mercado (experimental)",
            value=settings.directional_enabled,
            help="Ajuste leve basado en tendencia. Mejora con historial acumulado.",
        )

        advanced_mode = st.toggle(
            "Modo avanzado",
            value=False,
            help="Desbloquea Cadena completa, Simulador P&L e Historial.",
        )

        # Filtros: solo visibles en modo avanzado
        min_spread = float(settings.min_tna_spread_pct)
        price_mode = "mid"
        capital_mode = "net"
        show_all = False
        send_alerts = False

        if advanced_mode:
            st.divider()
            st.subheader("Filtros")
            min_spread = st.slider(
                "Spread min vs caucion (%)", 0.0, 30.0,
                float(settings.min_tna_spread_pct), 0.5,
            )
            price_mode = st.selectbox(
                "Precio opcion", ["mid", "bid", "ask", "last"],
                help="mid = analisis; bid = mas realista al vender",
            )
            capital_mode = st.selectbox(
                "Capital puts", ["net", "gross"],
                help="net = K-prima; gross = K (mas conservador)",
            )
            show_all = st.checkbox("Mostrar toda la cadena sin filtro", False)
            st.divider()
            send_alerts = st.checkbox("Enviar alertas Telegram", False)

        st.divider()
        st.caption("GGAL — BYMA — opciones americanas")
        if demo_mode:
            st.info("Datos sinteticos. Configura .env para datos reales.")

    # ── Providers y scanners ──────────────────────────────────────────────────
    provider = _build_provider(demo_mode)
    expiry_cal = _load_expiry_calendar()

    cc_scanner = CoveredCallScanner(CoveredCallConfig(price_mode=price_mode), expiry_cal)
    sp_scanner = ShortPutScanner(
        ShortPutConfig(price_mode=price_mode, capital_mode=capital_mode), expiry_cal
    )
    screener = Screener(ScreenerConfig(min_tna_spread_pct=min_spread))
    alerter = TelegramAlerter()

    # ── Datos (cacheados hasta que vence el intervalo) ────────────────────────
    now = time.time()
    if (now - st.session_state.get("last_fetch", 0.0)) >= refresh_s or "chain" not in st.session_state:
        st.session_state.chain = provider.get_options_chain()
        st.session_state.caucion_tna = provider.get_caucion_tna() or 0.0
        st.session_state.last_fetch = now

    chain: Optional[OptionsChain] = st.session_state.chain
    caucion_tna: float = st.session_state.caucion_tna

    if chain is None:
        st.error("Sin datos de mercado. Verifica la conexion o activa modo demo.")
        time.sleep(2)
        st.rerun()
        return

    benchmark = (
        Benchmark(caucion_tna_pct=caucion_tna, days=30)
        if caucion_tna > 0
        else ZERO_BENCHMARK
    )
    spot = chain.spot.mid

    # ── Scan ──────────────────────────────────────────────────────────────────
    cc_all = cc_scanner.scan(chain, benchmark)
    sp_all = sp_scanner.scan(chain, benchmark)
    cc_filtered, sp_filtered = screener.rank(cc_all, sp_all, benchmark)

    # ── Historial del subyacente (para AT + VolEdge) ──────────────────────────
    spot_history_df = _load_spot_history(days=180)
    htf_df          = _load_htf_history(weeks=52)
    ltf_df          = _load_ltf_history()

    spot_history_list: Optional[list[float]] = (
        spot_history_df["close"].dropna().tolist()
        if spot_history_df is not None and not spot_history_df.empty
        else None
    )

    # ── Contexto de mercado (opcional) ────────────────────────────────────────
    context = None
    if directional_on:
        from optionsdesk.signals.directional import compute_market_context
        ctx_data = spot_history_df if spot_history_df is not None else None
        context = compute_market_context(ctx_data, htf=htf_df, ltf=ltf_df)

    # ── Recomendaciones ───────────────────────────────────────────────────────
    recommender = Recommender()
    recs = recommender.recommend(
        cc_all, sp_all, benchmark, context=context, capital=capital,
        spot_history=spot_history_list, top_n=3,
    )

    # ── Header ───────────────────────────────────────────────────────────────
    st.title("OptionsDesk — Tasa implicita GGAL")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("GGAL", f"${spot:,.0f}")
    c2.metric("Caucion TNA", f"{caucion_tna:.1f}%" if caucion_tna > 0 else "—")
    c3.metric("Opciones en cadena", len(chain.options))
    c4.metric("Modo", "Demo" if demo_mode else "Live")
    c5.metric("Hora", pd.Timestamp.now().strftime("%H:%M:%S"), delta_color="off")

    if context is not None:
        conf_badge = {
            "alta": "alta confianza",
            "media": "confianza media",
            "baja": "baja confianza",
            "sin datos": "sin datos",
        }.get(context.confidence, context.confidence)
        color = "normal" if context.confidence == "sin datos" else "info"
        st.info(f"Mercado: {context.note}")

    st.divider()

    # ── Tabs ──────────────────────────────────────────────────────────────────
    if advanced_mode:
        tab_labels = [
            "Inicio", "Oportunidades", "Direccional", "Aprende",
            "Cadena completa", "Simulador P&L", "Historial",
        ]
        t_home, t_ops, t_dir, t_learn, t_chain, t_sim, t_hist = st.tabs(tab_labels)
    else:
        t_home, t_ops, t_dir, t_learn = st.tabs(
            ["Inicio", "Oportunidades", "Direccional", "Aprende"]
        )

    with t_home:
        _tab_home(recs, alerter)

    with t_ops:
        _tab_opportunities(
            cc_all, sp_all, cc_filtered, sp_filtered,
            show_all, caucion_tna, send_alerts, alerter,
        )

    with t_dir:
        _tab_directional(
            spot_history_df, chain, spot, context,
            htf_df=htf_df, ltf_df=ltf_df,
            expiry_calendar=expiry_cal,
            benchmark=benchmark,
        )

    with t_learn:
        _tab_learn()

    if advanced_mode:
        with t_chain:
            _tab_chain(chain, spot)
        with t_sim:
            _tab_simulator(cc_all, sp_all, spot)
        with t_hist:
            _tab_history()

    # ── Footer y auto-refresh ─────────────────────────────────────────────────
    # Guardamos refresh_s en session_state para que el fragment lo lea
    st.session_state["_refresh_s"] = refresh_s
    _auto_refresh_footer()


if __name__ == "__main__":
    main()
