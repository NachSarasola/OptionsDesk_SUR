"""Tablero Streamlit de análisis de opciones GGAL — v2 "for dummies".

Ejecutar con:
    streamlit run optionsdesk/ui/dashboard.py

Tab por defecto: Inicio (3 tarjetas de veredicto para cada perfil de riesgo).
Modo avanzado: desbloquea Cadena completa, Simulador P&L e Historial.
"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

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

    if settings.is_iol_configured():
        try:
            from optionsdesk.data.providers.iol import IOLProvider
            p = IOLProvider()
            p.connect()
            return p
        except Exception as exc:
            st.warning(f"IOL no disponible ({exc}). Intentando fuente alternativa.")

    if settings.is_configured():
        try:
            from optionsdesk.data.providers.homebroker import HomeBrokerProvider
            p = HomeBrokerProvider()
            p.connect()
            return p
        except Exception as exc:
            st.warning(f"HomeBroker no disponible ({exc}). Intentando BYMA Open Data.")

    try:
        from optionsdesk.data.providers.byma_open import BymaOpenProvider
        return BymaOpenProvider()
    except Exception as exc:
        st.warning(f"BYMA Open Data no disponible ({exc}). Usando modo demo.")
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


@st.cache_data(ttl=20)
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


# ── Gráfico de velas (estilo TradingView) ─────────────────────────────────────

def _candlestick_chart(
    df: pd.DataFrame,
    *,
    title: str = "",
    height: int = 420,
    smas: Optional[dict[str, int]] = None,
    levels: Optional[list[dict]] = None,
    show_volume: bool = True,
    max_bars: int = 120,
) -> None:
    """Render de velas japonesas tema oscuro con volumen, medias y niveles.

    df: DataFrame OHLCV con columnas open/high/low/close (+ volume opcional) y
        un índice temporal en 'date' o 'time'. Cae a st.line_chart si plotly
        no está disponible o faltan columnas OHLC.
    levels: lista de {"y": float, "label": str, "color": str, "dash": str}.
    """
    needed = {"open", "high", "low", "close"}
    if df is None or df.empty or not needed.issubset(df.columns):
        if df is not None and not df.empty and "close" in df.columns:
            st.line_chart(df.set_index(df.columns[0])["close"] if "close" in df.columns else df)
        else:
            st.info("Sin datos OHLC para graficar.")
        return

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        st.line_chart(df[["close"]])
        return

    data = df.tail(max_bars).copy()
    x_col = "time" if "time" in data.columns else ("date" if "date" in data.columns else None)
    x = data[x_col] if x_col else data.index

    up, down = "#26a69a", "#ef5350"
    has_vol = show_volume and "volume" in data.columns and data["volume"].fillna(0).abs().sum() > 0

    if has_vol:
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            row_heights=[0.78, 0.22], vertical_spacing=0.02,
        )
    else:
        fig = make_subplots(rows=1, cols=1)

    fig.add_trace(
        go.Candlestick(
            x=x, open=data["open"], high=data["high"], low=data["low"], close=data["close"],
            increasing_line_color=up, decreasing_line_color=down,
            increasing_fillcolor=up, decreasing_fillcolor=down,
            line_width=1, name="GGAL", showlegend=False,
        ),
        row=1, col=1,
    )

    sma_palette = ["#f5b301", "#4f9bff", "#b06bff"]
    if smas:
        for i, (label, period) in enumerate(smas.items()):
            if len(data) >= period:
                ma = data["close"].rolling(period).mean()
                fig.add_trace(
                    go.Scatter(
                        x=x, y=ma, mode="lines", name=label,
                        line=dict(color=sma_palette[i % len(sma_palette)], width=1.2),
                    ),
                    row=1, col=1,
                )

    if levels:
        for lv in levels:
            y = lv.get("y")
            if y is None or y <= 0:
                continue
            fig.add_hline(
                y=y, line_color=lv.get("color", "#9ca3af"),
                line_dash=lv.get("dash", "dash"), line_width=1.1,
                annotation_text=lv.get("label", ""),
                annotation_position="right",
                annotation_font_color=lv.get("color", "#9ca3af"),
                annotation_font_size=11,
                row=1, col=1,
            )

    if has_vol:
        vol_colors = [up if c >= o else down for o, c in zip(data["open"], data["close"])]
        fig.add_trace(
            go.Bar(x=x, y=data["volume"], marker_color=vol_colors, opacity=0.5,
                   name="Vol", showlegend=False),
            row=2, col=1,
        )

    fig.update_layout(
        template="plotly_dark",
        height=height,
        margin=dict(l=8, r=56, t=28 if title else 8, b=8),
        title=dict(text=title, font=dict(size=14, color="#d4d4d8")) if title else None,
        paper_bgcolor="#0e0e12", plot_bgcolor="#0e0e12",
        xaxis_rangeslider_visible=False,
        showlegend=bool(smas),
        legend=dict(orientation="h", y=1.04, x=0, font=dict(size=11),
                    bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified",
        dragmode="pan",
    )
    fig.update_xaxes(showgrid=False, color="#71717a", row=1, col=1)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.05)",
                     color="#71717a", side="right", row=1, col=1)
    if has_vol:
        fig.update_xaxes(showgrid=False, color="#71717a", row=2, col=1)
        fig.update_yaxes(showgrid=False, color="#71717a", side="right",
                         showticklabels=False, row=2, col=1)

    st.plotly_chart(fig, use_container_width=True, config={
        "displayModeBar": False, "scrollZoom": True,
    })


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

    # ── Gráfico de precio (velas diarias con medias) ──────────────────────
    close = spot_history["close"]
    dates = spot_history["date"]
    _candlestick_chart(
        spot_history,
        height=420,
        smas={"SMA5": 5, "SMA20": 20},
        max_bars=120,
    )

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
    st.caption("Especulativo, sin alpha validado. Pérdida máxima = prima pagada.")

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
def _close_position_live(symbol: str, strategy: str, lotes: int, provider) -> None:
    is_buy = "SHORT" in strategy
    success = provider.send_order(symbol, is_buy, lotes, 0.0) # Using 0.0 for now, should pull quote for limit order
    if success:
        st.success(f"Orden de cierre enviada para {lotes} lotes de {symbol}")
    else:
        st.error(f"Error al enviar orden de cierre para {symbol}")

def _tab_portfolio(provider, chain, spot):
    st.subheader("Portfolio Activo (Scalping)")
    try:
        from optionsdesk.signals.monitor import PositionMonitor
        monitor = PositionMonitor(settings.open_positions_file)
        positions = monitor.load_positions()
    except Exception:
        positions = []

    if not positions:
        st.info("No hay posiciones abiertas.")
        return

    rows = []
    for p in positions:
        quote = chain.options.get(p.symbol) if chain else None
        mid = quote.mid if quote else p.spot_entry
        pnl_gross = (mid - p.net_outlay) * p.contracts * 100 if "LONG" in p.strategy else (p.net_outlay - mid) * p.contracts * 100
        # Comisiones IOL aprox 0.5% + IVA
        commissions = (p.net_outlay * p.contracts * 100 * 0.005 * 1.21) + (mid * p.contracts * 100 * 0.005 * 1.21)
        pnl_net = pnl_gross - commissions
        pnl_pct = (pnl_net / (p.net_outlay * p.contracts * 100)) * 100 if p.net_outlay > 0 else 0.0
        
        rows.append({
            "Símbolo": p.symbol,
            "Estrategia": p.strategy,
            "Lotes": p.contracts,
            "Entrada": _scalp_money(p.net_outlay, 2),
            "Mark (Mid)": _scalp_money(mid, 2),
            "Comisiones": _scalp_money(commissions, 2),
            "PnL Neto": _scalp_money(pnl_net, 2),
            "PnL %": f"{pnl_pct:+.2f}%",
        })

    import pandas as pd
    df = pd.DataFrame(rows)
    st.dataframe(df, hide_index=True, use_container_width=True)

    st.markdown("### Acciones Rápidas")
    cols = st.columns(min(4, len(positions)))
    for i, p in enumerate(positions[:4]):
        with cols[i]:
            st.markdown(f"**{p.symbol}**")
            if st.button("Cerrar Posición", key=f"close_{p.symbol}_{i}", type="secondary"):
                _close_position_live(p.symbol, p.strategy, p.contracts, provider)
                st.rerun()

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


# ── Tab: Scalping ─────────────────────────────────────────────────────────────

def _scalp_money(value: float | int | None, decimals: int = 0) -> str:
    if value is None:
        return "-"
    try:
        return f"${float(value):,.{decimals}f}"
    except (TypeError, ValueError):
        return "-"


def _scalp_pct(value: float | int | None, decimals: int = 1, signed: bool = False) -> str:
    if value is None:
        return "-"
    try:
        sign = "+" if signed else ""
        return f"{float(value):{sign}.{decimals}f}%"
    except (TypeError, ValueError):
        return "-"


def _scalp_card(label: str, value: str, tag: str, color: str) -> str:
    return (
        f"<div style='background:rgba(26,26,36,0.85);border:1px solid {color}33;"
        f"border-radius:8px;padding:10px 12px;min-height:88px;'>"
        f"<div style='font-size:0.62rem;color:#71717A;text-transform:uppercase;"
        f"letter-spacing:.07em;margin-bottom:4px;font-family:JetBrains Mono,monospace;'>{label}</div>"
        f"<div style='font-size:1.2rem;font-weight:700;color:#FAFAFA;line-height:1.15;'>{value}</div>"
        f"<div style='font-size:0.72rem;color:{color};font-weight:600;margin-top:5px;'>{tag}</div>"
        f"</div>"
    )


def _signal_color(sig) -> str:
    if getattr(sig, "confidence", "") == "actionable":
        return "#22c55e"
    if any("stale" in f or "capital" in f or "spread" in f for f in getattr(sig, "risk_flags", [])):
        return "#ef4444"
    return "#f59e0b"


def _execute_scalp_trade_live(sig, spot: float, lotes: int, tna: float, provider) -> None:
    is_buy = sig.action.startswith("BUY")
    price = sig.ask if is_buy else sig.bid
    success = provider.send_order(sig.symbol, is_buy, lotes, price)
    if not success:
        st.error(f"Fallo al enviar la orden real a IOL para {sig.symbol}. Verifica logs.")
        return
    # Si la orden se envío bien, la registramos localmente en el monitor
    _simulate_scalp_trade(sig, spot, lotes, tna)

def _simulate_scalp_trade(sig, spot: float, lotes: int, tna: float) -> None:
    import json
    from datetime import date

    file = settings.open_positions_file
    file.parent.mkdir(parents=True, exist_ok=True)

    strat = (
        "SCALP_LONG_CALL" if sig.action == "BUY_CALL"
        else "SCALP_LONG_PUT" if sig.action == "BUY_PUT"
        else "SCALP_SHORT_CALL" if sig.action == "SELL_CALL"
        else "SCALP_SHORT_PUT"
    )
    is_long = "LONG" in strat
    entry = float(sig.plan_entry or sig.mid or 0.0)
    stop_frac = abs(entry - float(sig.plan_sl or 0.0)) / entry if entry > 0 else 0.35
    target_pct = ((float(sig.plan_tp or entry) - entry) / entry * 100.0) if is_long and entry > 0 else 50.0

    pos = {
        "symbol": sig.symbol,
        "strategy": strat,
        "strike": sig.strike,
        "spot_entry": spot,
        "premium_received": -entry if is_long else entry,
        "net_outlay": entry if is_long else max(sig.max_loss_ars / 100.0, entry),
        "iv_entry": sig.iv or 0.0,
        "days_entry": max(int(sig.days_to_expiry), 1),
        "entry_date": date.today().isoformat(),
        "target_exit_days": max(min(int(sig.days_to_expiry), 2), 1),
        "target_capture_pct": max(min(target_pct, 80.0), 15.0),
        "caucion_tna": tna,
        "contracts": int(lotes),
        "max_loss_mult": max(min(stop_frac, 0.80), 0.10),
        "roll_dte": 2,
        "defend_delta": 0.65,
        "scalp_plan_entry": sig.plan_entry,
        "scalp_plan_sl": sig.plan_sl,
        "scalp_plan_tp": sig.plan_tp,
        "scalp_plan_rr": sig.plan_rr,
        "scalp_pop": sig.probability_of_profit,
        "scalp_expected_value_ars": sig.expected_value_ars,
        "scalp_edge_r": getattr(sig, "edge_r", 0.0),
        "scalp_fill_probability": getattr(sig, "fill_probability", 0.0),
        "scalp_cost_to_target_pct": getattr(sig, "cost_to_target_pct", 0.0),
        "scalp_time_stop_min": getattr(sig, "time_stop_min", 20),
    }
    with file.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(pos, ensure_ascii=False) + "\n")


def _tab_scalping(
    chain: OptionsChain,
    spot: float,
    provider: MarketDataProvider,
    context,
    expiry_calendar: dict,
    spot_history_df: Optional[pd.DataFrame],
    capital: Optional[float],
    risk_profile: str,
    caucion_tna: float,
    ltf_df: Optional[pd.DataFrame] = None,
) -> None:
    from optionsdesk.signals.scalping import scan_all

    try:
        from optionsdesk.signals.monitor import PositionMonitor
        positions = PositionMonitor(settings.open_positions_file).load_positions()
    except Exception:
        positions = []

    st.subheader("Scalping & Momentum")
    st.caption("Mesa de scalping intradía. El botón EJECUTAR EN MERCADO envía orden real a IOL.")

    ctl_cap, ctl_refresh = st.columns([3, 1])
    with ctl_cap:
        capital_default = int(st.session_state.get("scalp_capital", capital or settings.default_capital or 0) or 0)
        capital_input = st.number_input(
            "Capital máximo para scalping (ARS)",
            min_value=0,
            max_value=100_000_000,
            value=capital_default,
            step=50_000,
            help="Este capital define sizing, riesgo por trade y bloqueo de señales reales.",
            key="capital_scalping_main",
        )
        capital = float(capital_input) if capital_input > 0 else None
        st.session_state["scalp_capital"] = float(capital_input or 0)
    with ctl_refresh:
        st.write("")
        if st.button("Actualizar datos", key="btn_refresh_scalping", use_container_width=True):
            st.session_state["force_refresh"] = True
            st.rerun()

    realized_vol = None
    if spot_history_df is not None and not spot_history_df.empty:
        try:
            from optionsdesk.signals.volatility import yang_zhang_volatility, realized_volatility
            realized_vol = yang_zhang_volatility(spot_history_df, window=20)
            if realized_vol is None and "close" in spot_history_df.columns:
                realized_vol = realized_volatility(spot_history_df["close"].dropna().tolist(), window=20)
        except Exception:
            realized_vol = None

    snap = None
    if context is not None:
        snap = getattr(context, "ltf_snap", None) or getattr(context, "snap", None)

    try:
        greeks, signals = scan_all(
            chain,
            expiry_calendar,
            snap=snap,
            realized_vol=realized_vol,
            r=0.0,
            capital=capital,
            positions=positions,
            risk_profile=risk_profile,
        )
    except Exception as exc:
        st.error(f"Scalping no pudo escanear la cadena: {exc}")
        return

    actionables = [s for s in signals if s.confidence == "actionable"]
    watchlist = [s for s in signals if s.confidence != "actionable"]
    tradeable = [g for g in greeks.values() if g.is_tradeable]
    health = provider.get_health()

    if not capital or capital <= 0:
        actionables = []

    h1, h2, h3, h4, h5, h6 = st.columns(6)
    latency = f"{health.last_latency_ms:.0f} ms" if health.last_latency_ms is not None else "-"

    now_ba = datetime.now(ZoneInfo("America/Argentina/Buenos_Aires"))
    session_end = now_ba.replace(hour=17, minute=0, second=0, microsecond=0)
    mins_left = max((session_end - now_ba).total_seconds() / 60, 0)
    session_color = "#22c55e" if mins_left > 30 else ("#f59e0b" if mins_left > 10 else "#ef4444")
    session_phase = getattr(snap, "session_phase", None) if snap else None
    if session_phase is None:
        h = now_ba.hour + now_ba.minute / 60
        session_phase = "cerrada" if (h < 11 or h >= 17) else ("apertura" if h < 11.33 else ("cierre" if h >= 16.75 else "regular"))

    h1.markdown(_scalp_card("FEED", health.source, "conectado" if health.connected else "revisar", "#22c55e" if health.connected else "#ef4444"), unsafe_allow_html=True)
    h2.markdown(_scalp_card("LATENCIA", latency, f"timeouts {health.timeouts}", "#22c55e" if not health.timeouts else "#f59e0b"), unsafe_allow_html=True)
    h3.markdown(_scalp_card("CADENA", f"{len(tradeable)}/{len(greeks)}", "operable/parseada", "#22c55e" if tradeable else "#ef4444"), unsafe_allow_html=True)
    h4.markdown(_scalp_card("SESIÓN", f"{int(mins_left)}m", session_phase, session_color), unsafe_allow_html=True)
    h5.markdown(_scalp_card("RV 20D", f"{realized_vol * 100:.1f}%" if realized_vol else "-", "Yang-Zhang/close", "#71717a"), unsafe_allow_html=True)
    h6.markdown(_scalp_card("ACCIONABLES", str(len(actionables)), f"watchlist {len(watchlist)}", "#22c55e" if actionables else "#71717a"), unsafe_allow_html=True)

    preflight = []
    if not settings.is_iol_configured() or health.source != "IOL":
        preflight.append("No estás usando IOL como fuente primaria; tratá cualquier señal como radar.")
    if capital is None or capital <= 0:
        preflight.append("Cargá capital disponible para sizing real por riesgo.")
    if snap is None:
        preflight.append("Falta contexto intradiario: sin ola/rebote confirmado, las señales quedan degradadas.")
    if realized_vol is None:
        preflight.append("Falta volatilidad realizada: gamma scalp queda menos confiable.")
    if health.timeouts:
        preflight.append(f"El feed tuvo {health.timeouts} timeout(s). No operes si el dato se congela.")

    if preflight:
        st.warning("Pre-flight: " + " | ".join(preflight))
    else:
        st.success("Pre-flight OK: feed, contexto, volatilidad y sizing disponibles.")

    # ── Gráfico de velas intradía con niveles de la señal ─────────────────────
    intraday = ltf_df is not None and not ltf_df.empty
    chart_df = ltf_df if intraday else spot_history_df
    levels = [{"y": spot, "label": f"GGAL ${spot:,.0f}", "color": "#9ca3af", "dash": "dot"}]
    chart_sig = None
    if actionables:
        labels = [
            f"{s.action} {s.symbol} · PoP {s.probability_of_profit * 100:.0f}% · R:R {s.plan_rr:.1f}"
            for s in actionables
        ]
        pick = st.selectbox(
            "Señal sobre el gráfico",
            range(len(actionables)),
            format_func=lambda i: labels[i],
            key="scalp_chart_pick",
        )
        chart_sig = actionables[pick]
        if chart_sig.underlying_target > 0:
            levels.append({"y": chart_sig.underlying_target, "label": f"TP ${chart_sig.underlying_target:,.0f}", "color": "#26a69a", "dash": "dash"})
        if chart_sig.underlying_stop > 0:
            levels.append({"y": chart_sig.underlying_stop, "label": f"Stop ${chart_sig.underlying_stop:,.0f}", "color": "#ef5350", "dash": "dash"})

    _candlestick_chart(
        chart_df,
        height=400,
        smas={"SMA9": 9, "SMA20": 20},
        levels=levels,
        max_bars=120 if intraday else 60,
    )
    st.caption(
        "Velas intradía GGAL con tu entrada/stop/TP proyectados sobre el subyacente."
        if intraday else
        "Sin intradía ahora (fuera de horario o sin feed): se muestra el diario."
    )

    if actionables:
        best = actionables[0]
        st.markdown("### Mejor trade por PoP viable")
        st.success(
            f"{best.action} {best.symbol} | PoP {best.probability_of_profit * 100:.0f}% | "
            f"EV/lote {_scalp_money(best.expected_value_ars)} | "
            f"Edge/R {getattr(best, 'edge_r', 0.0):+.2f} | "
            f"Friccion/target {getattr(best, 'cost_to_target_pct', 0.0):.1f}%"
        )
        st.caption(
            "Seleccionado por: actionable > PoP > EV/riesgo > menor friccion > liquidez > score."
        )

    st.markdown("### Señales accionables")
    if not actionables:
        st.info("No hay trades liberados ahora. Esto es deseable si el dato, el spread o el contexto no dan ventaja neta.")
    else:
        cols = st.columns(min(3, len(actionables)))
        for i, sig in enumerate(actionables[:3]):
            color = _signal_color(sig)
            with cols[i]:
                st.markdown(
                    f"<div style='background:rgba(26,26,36,0.86);border-top:3px solid {color};"
                    f"border-radius:8px;padding:12px;'>"
                    f"<div style='font-size:0.72rem;color:#71717A;text-transform:uppercase;'>{sig.playbook} - {sig.confidence}</div>"
                    f"<div style='font-size:1.2rem;font-weight:700;margin-top:2px;'>{sig.action} {sig.symbol}</div>"
                    f"<div style='font-size:0.78rem;color:{color};font-weight:600;margin-top:4px;'>{sig.expected_move}</div>"
                    f"<div style='font-size:0.78rem;color:#d4d4d8;margin-top:8px;line-height:1.35;'>{sig.rationale}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                with st.expander("Ticket manual", expanded=True):
                    valid_txt = sig.valid_until.split("T")[-1][:8] if sig.valid_until else "-"
                    st.write(f"Entrada límite: **{_scalp_money(sig.plan_entry, 2)}** ({sig.entry_style})")
                    st.write(f"Stop opción: **{_scalp_money(sig.plan_sl, 2)}** | TP opción: **{_scalp_money(sig.plan_tp, 2)}**")
                    st.write(f"Stop GGAL: **{_scalp_money(sig.underlying_stop, 2)}** | TP GGAL: **{_scalp_money(sig.underlying_target, 2)}**")
                    st.caption(
                        f"PoP {sig.probability_of_profit * 100:.0f}% | EV/lote {_scalp_money(sig.expected_value_ars)} | "
                        f"R:R {sig.plan_rr:.2f} | Riesgo stop/lote {_scalp_money(sig.planned_risk_ars)} | "
                        f"Fricción {_scalp_money(sig.friction_ars)} ({sig.friction_pct:.2f}%) | "
                        f"Fill {getattr(sig, 'fill_probability', 0.0) * 100:.0f}% | "
                        f"Time stop {getattr(sig, 'time_stop_min', 20)} min | válido hasta {valid_txt}"
                    )
                    if getattr(sig, "warning_flags", []):
                        st.caption("Advertencias: " + ", ".join(sig.warning_flags))
                    default_lotes = max(int(sig.suggested_lots or 1), 1)
                    if sig.suggested_lots <= 0:
                        st.error("Sizing bloqueado: cargá capital o bajá riesgo antes de registrar.")
                        continue
                    lotes = st.number_input(
                        "Lotes",
                        min_value=1,
                        max_value=max(int(sig.suggested_lots), 1),
                        value=default_lotes,
                        step=1,
                        key=f"scalp_lotes_{sig.symbol}_{i}",
                    )
                    if st.button("EJECUTAR EN MERCADO", key=f"scalp_reg_{sig.symbol}_{i}", type="primary"):
                        _execute_scalp_trade_live(sig, spot, lotes, caucion_tna, provider)
                        st.success("¡Orden ENVIADA A IOL y registrada en Portfolio!")

    def _signals_df(items: list) -> pd.DataFrame:
        rows = []
        for s in items:
            g = greeks.get(s.symbol)
            rows.append({
                "Estado": s.confidence,
                "Acción": s.action,
                "Símbolo": s.symbol,
                "Setup": s.playbook or s.signal_type,
                "Score": round(s.score, 1),
                "PoP": f"{s.probability_of_profit * 100:.0f}%",
                "EV/lote": _scalp_money(s.expected_value_ars),
                "Entrada": _scalp_money(s.plan_entry, 2),
                "Stop": _scalp_money(s.plan_sl, 2),
                "TP": _scalp_money(s.plan_tp, 2),
                "R:R": f"{s.plan_rr:.2f}",
                "Riesgo/lote": _scalp_money(s.planned_risk_ars),
                "Fricción": _scalp_money(s.friction_ars),
                "Edge/R": f"{getattr(s, 'edge_r', 0.0):+.2f}",
                "Fill": f"{getattr(s, 'fill_probability', 0.0) * 100:.0f}%",
                "Costo/target": f"{getattr(s, 'cost_to_target_pct', 0.0):.1f}%",
                "BE mov": f"{s.breakeven_move_pct:.2f}%",
                "Spread": f"{s.spread_pct:.1f}%",
                "Edad": f"{getattr(g, 'quote_age_s', 0.0):.0f}s" if g else "-",
                "Bloqueos": ", ".join(getattr(s, "blocking_flags", [])) if getattr(s, "blocking_flags", []) else "-",
                "Advertencias": ", ".join(getattr(s, "warning_flags", [])) if getattr(s, "warning_flags", []) else "-",
            })
        return pd.DataFrame(rows)

    if actionables:
        st.dataframe(_signals_df(actionables), hide_index=True, use_container_width=True)

    st.markdown("### Watchlist")
    if not watchlist:
        st.info("Watchlist vacía.")
    else:
        reason_counts: dict[str, int] = {}
        for s in watchlist:
            for flag in getattr(s, "blocking_flags", []) or s.risk_flags:
                reason_counts[flag] = reason_counts.get(flag, 0) + 1
        if reason_counts:
            top_reasons = sorted(reason_counts.items(), key=lambda kv: kv[1], reverse=True)[:6]
            st.caption("Bloqueos principales: " + " | ".join(f"{k} ({v})" for k, v in top_reasons))
        st.dataframe(_signals_df(watchlist[:40]), hide_index=True, use_container_width=True)

    with st.expander("Auditoria / backtest scalping", expanded=False):
        snap_dir = settings.snapshots_dir
        files = list(snap_dir.glob("**/*.parquet")) if snap_dir.exists() else []
        if files:
            st.write(f"Snapshots disponibles para replay: **{len(files)}** archivos.")
            st.caption(
                "El motor optionsdesk.backtest.scalping replayea planes sin lookahead: "
                "entrada posterior a la senal, compra al ask, salida al bid, costos, TP, SL y time stop."
            )
        else:
            st.info("Sin snapshots intradiarios todavia. El recorder debe acumular data antes de auditar parametros reales.")

    with st.expander("Griegas de la cadena y calidad de quotes", expanded=False):
        rows = []
        for g in sorted(greeks.values(), key=lambda x: (x.days, abs(x.moneyness_pct), x.symbol)):
            rows.append({
                "Símbolo": g.symbol,
                "Tipo": "CALL" if g.option_type == "C" else "PUT",
                "Strike": g.strike,
                "DTE": g.days,
                "Bid": g.bid,
                "Ask": g.ask,
                "Mid": g.mid,
                "Spread %": g.spread_pct,
                "IV %": f"{g.iv * 100:.1f}%" if g.iv else "-",
                "Delta": g.delta,
                "Gamma": g.gamma,
                "Theta": g.theta,
                "Vol": g.volume,
                "Edad quote": f"{g.quote_age_s:.0f}s",
                "Operable": g.is_tradeable,
                "Motivo": g.liquidity_reason or "ok",
            })
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        else:
            st.info("No se pudieron calcular griegas parseables para la cadena actual.")


# ── Footer manual ─────────────────────────────────────────────────────────────

def _data_footer() -> None:
    """Pie estable: no fuerza reruns ni parpadeos."""
    last_fetch = st.session_state.get("last_fetch_label", "sin cargar")
    col_foot1, col_foot2 = st.columns([4, 1])
    with col_foot1:
        st.caption(f"Datos cargados: {last_fetch}")
    with col_foot2:
        if st.button("Actualizar datos", key="btn_refresh_main"):
            st.session_state["force_refresh"] = True
            st.rerun()


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
            "Modo demo", value=not (settings.is_iol_configured() or settings.is_configured()),
            help="Datos sinteticos. Activa cuando no hay credenciales.",
        )
        if st.button("Actualizar datos", key="btn_refresh_sidebar", use_container_width=True):
            st.session_state["force_refresh"] = True
            st.rerun()
        scalp_profile = st.selectbox(
            "Perfil scalping",
            ["balanced", "conservative", "aggressive"],
            index=0,
            help="balanced es el default para operar manualmente con dinero real.",
        )

        st.divider()

        capital_default = int(st.session_state.get("scalp_capital", settings.default_capital or 0) or 0)
        capital_input = st.number_input(
            "Capital disponible (ARS)",
            min_value=0,
            max_value=100_000_000,
            value=capital_default,
            step=50_000,
            help="Para calcular cuantos lotes entran y la ganancia estimada en pesos.",
            key="capital_sidebar",
        )
        capital = float(capital_input) if capital_input > 0 else None
        st.session_state["scalp_capital"] = float(capital_input or 0)

        directional_on = st.toggle(
            "Contexto de mercado (experimental)",
            value=True,
            help="Ajuste leve basado en tendencia. Mejora con historial acumulado.",
        )

        advanced_mode = st.toggle(
            "Modo avanzado",
            value=False,
            help="Suma tabs de análisis de tasas: Inicio, Oportunidades, Aprende, "
                 "Cadena completa, Simulador P&L e Historial.",
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

    # ── Datos: carga inicial + refresh manual, sin parpadeo por auto-rerun ─────
    force_refresh = bool(st.session_state.pop("force_refresh", False))
    if force_refresh or "chain" not in st.session_state:
        with st.spinner("Cargando datos de mercado..."):
            st.session_state.chain = provider.get_options_chain()
            st.session_state.caucion_tna = provider.get_caucion_tna() or settings.default_caucion_tna
            st.session_state.last_fetch_label = pd.Timestamp.now().strftime("%H:%M:%S")

    chain: Optional[OptionsChain] = st.session_state.chain
    caucion_tna: float = st.session_state.caucion_tna

    if chain is None:
        st.error("Sin datos de mercado. Verifica la conexion o activa modo demo.")
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
    from optionsdesk.signals.directional import compute_market_context
    ctx_data = spot_history_df if spot_history_df is not None else None
    context = compute_market_context(ctx_data, htf=htf_df, ltf=ltf_df)
    recommender_context = context if directional_on else None

    # ── Recomendaciones ───────────────────────────────────────────────────────
    recommender = Recommender()
    recs = recommender.recommend(
        cc_all, sp_all, benchmark, context=recommender_context, capital=capital,
        spot_history=spot_history_list, top_n=3,
    )

    # ── Header ───────────────────────────────────────────────────────────────
    st.title("OptionsDesk — Scalping GGAL")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("GGAL", f"${spot:,.0f}")
    c2.metric("Caucion TNA", f"{caucion_tna:.1f}%" if caucion_tna > 0 else "—")
    c3.metric("Opciones en cadena", len(chain.options))
    c4.metric("Modo", "Demo" if demo_mode else "Live")
    c5.metric("Hora", pd.Timestamp.now().strftime("%H:%M:%S"), delta_color="off")

    if context is not None:
        st.info(f"Mercado: {context.note}")

    st.divider()

    # ── Tabs ──────────────────────────────────────────────────────────────────
    # Operatoria diaria: Scalping, Portfolio y Direccional. El resto (análisis
    # de tasas, tutoriales, simuladores) queda detrás de "Modo avanzado".
    if advanced_mode:
        tab_labels = [
            "Scalping", "Portfolio", "Direccional",
            "Inicio", "Oportunidades", "Aprende",
            "Cadena completa", "Simulador P&L", "Historial",
        ]
        (t_scalping, t_portfolio, t_dir, t_home, t_ops, t_learn,
         t_chain, t_sim, t_hist) = st.tabs(tab_labels)
    else:
        t_scalping, t_portfolio, t_dir = st.tabs(["Scalping", "Portfolio", "Direccional"])

    with t_scalping:
        _tab_scalping(
            chain=chain,
            spot=spot,
            provider=provider,
            context=context,
            expiry_calendar=expiry_cal,
            spot_history_df=spot_history_df,
            capital=capital,
            risk_profile=scalp_profile,
            caucion_tna=caucion_tna,
            ltf_df=ltf_df,
        )

    with t_portfolio:
        _tab_portfolio(provider, chain, spot)

    with t_dir:
        _tab_directional(
            spot_history_df, chain, spot, context,
            htf_df=htf_df, ltf_df=ltf_df,
            expiry_calendar=expiry_cal,
            benchmark=benchmark,
        )

    if advanced_mode:
        with t_home:
            _tab_home(recs, alerter)
        with t_ops:
            _tab_opportunities(
                cc_all, sp_all, cc_filtered, sp_filtered,
                show_all, caucion_tna, send_alerts, alerter,
            )
        with t_learn:
            _tab_learn()
        with t_chain:
            _tab_chain(chain, spot)
        with t_sim:
            _tab_simulator(cc_all, sp_all, spot)
        with t_hist:
            _tab_history()

    # ── Footer manual ─────────────────────────────────────────────────────────
    _data_footer()


if __name__ == "__main__":
    main()
