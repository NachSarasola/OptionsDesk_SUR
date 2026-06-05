"""Tablero Streamlit profesional para análisis de opciones GGAL.

Ejecutar con:
    streamlit run optionsdesk/ui/dashboard.py

Tab por defecto: Inicio (3 tarjetas de veredicto para cada perfil de riesgo).
Modo avanzado: desbloquea Cadena completa, Simulador P&L e Historial.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st

# `streamlit run optionsdesk/ui/dashboard.py` ejecuta el script con sys.path[0]
# apuntando a optionsdesk/ui/, no a la raiz del proyecto. Si el paquete no esta
# instalado (sin venv activado / sin `pip install -e .`), el import de abajo
# falla con ModuleNotFoundError. Agregamos la raiz al path para que corra igual.
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from optionsdesk.config.settings import settings
from optionsdesk.config.costs import DEFAULT_COSTS
from optionsdesk.core.benchmark import Benchmark, ZERO_BENCHMARK
from optionsdesk.core.rates import RateResult
from optionsdesk.data.history import (
    daily_with_live_spot as _daily_with_live_spot,
    load_spot_tape,
    tape_ohlc as _tape_ohlc,
    weekly_from_daily as _weekly_from_daily,
)
from optionsdesk.data.providers.base import MarketDataProvider, MarketDataHealth, OptionsChain, Quote
from optionsdesk.execution.base import Order, OrderSide, OrderStatus
from optionsdesk.execution.paper import PaperExecutor
from optionsdesk.signals.alerts import TelegramAlerter
from optionsdesk.signals.directional_ideas import (
    build_directional_idea,
    build_directional_spread,
    _confluence,
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
  --glow-sm:    0 0 12px rgba(245,158,11,0.10);
}

/* ── Base ─────────────────────────────────────────────────────────── */
html, body, .stApp {
  background-color: var(--bg) !important;
  font-family: 'Inter', system-ui, sans-serif !important;
  color: var(--fg) !important;
}
[data-testid="stMainBlockContainer"] {
  max-width: 1500px !important;
  padding-top: 1.25rem !important;
  padding-bottom: 2rem !important;
}
[data-testid="stMainBlockContainer"] [data-testid="stVerticalBlock"] {
  gap: 0.75rem !important;
}

/* ── Tipografía ───────────────────────────────────────────────────── */
h1, h2, h3, h4, h5, h6,
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
  font-family: 'Space Grotesk', system-ui, sans-serif !important;
  font-weight: 600 !important;
  letter-spacing: 0 !important;
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
  color: var(--fg) !important;
}

/* ── Métricas ─────────────────────────────────────────────────────── */
[data-testid="metric-container"] {
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  padding: 0.78rem 0.95rem !important;
}
[data-testid="metric-container"]:hover {
  border-color: var(--border-h) !important;
}
[data-testid="metric-container"] [data-testid="stMetricLabel"] {
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 0.75rem !important;
  letter-spacing: 0 !important;
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
  border-radius: 8px !important;
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
  border: none !important;
  background: transparent !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
  background: var(--bg-muted) !important;
  color: var(--fg) !important;
  border-color: var(--border) !important;
}
[data-testid="stTabs"] [role="tab"]:hover:not([aria-selected="true"]) {
  color: var(--fg) !important;
  background: rgba(255,255,255,0.04) !important;
}

/* ── Botones ──────────────────────────────────────────────────────── */
.stButton button {
  font-family: 'Inter', sans-serif !important;
  font-weight: 500 !important;
  border-radius: 8px !important;
  border: 1px solid var(--border) !important;
  background: rgba(26,26,36,0.6) !important;
  color: var(--fg) !important;
}
.stButton button:hover {
  border-color: var(--accent) !important;
  color: var(--accent) !important;
}
.stButton button:active {
  opacity: 0.92 !important;
}
/* Botón primario (el primero en cada grupo suele ser el CTA) */
.stButton [kind="primary"] button,
.stButton button[kind="primary"] {
  background: var(--accent) !important;
  color: #0A0A0F !important;
  border: none !important;
}
.stButton [kind="primary"] button:hover {
  filter: brightness(1.1) !important;
  color: #0A0A0F !important;
}

/* ── Containers con borde (st.container(border=True)) ─────────────── */
[data-testid="stVerticalBlockBorderWrapper"] {
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
}
[data-testid="stVerticalBlockBorderWrapper"]:hover {
  border-color: var(--border-h) !important;
}

/* ── Inputs / Selectbox / Slider ──────────────────────────────────── */
.stTextInput input, .stNumberInput input, .stSelectbox select,
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input {
  background: var(--card) !important;
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
  border-radius: 8px !important;
  border-left-width: 3px !important;
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
  border-radius: 8px !important;
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
  border-radius: 8px !important;
  overflow: hidden !important;
}
[data-testid="stDataFrame"] iframe {
  border-radius: 8px !important;
}

/* ── Captions / Labels ────────────────────────────────────────────── */
[data-testid="stCaptionContainer"] p,
.stCaption {
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 0.75rem !important;
  color: var(--fg-muted) !important;
  letter-spacing: 0 !important;
}

/* ── Divider ──────────────────────────────────────────────────────── */
hr {
  border-color: var(--border) !important;
  margin: 1rem 0 !important;
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
def _build_provider() -> MarketDataProvider:
    preference = getattr(settings, "market_data_provider", "AUTO")
    if preference in {"AUTO", "PRIMARY"} and settings.is_primary_configured():
        try:
            from optionsdesk.data.providers.primary import PrimaryProvider
            p = PrimaryProvider()
            p.connect()
            return p
        except Exception as exc:
            st.warning(f"Primary no disponible ({exc}). Intentando fuente alternativa.")
            if preference == "PRIMARY":
                raise

    if preference in {"AUTO", "IOL"} and settings.is_iol_configured():
        try:
            from optionsdesk.data.providers.iol import IOLProvider
            p = IOLProvider()
            p.connect()
            return p
        except Exception as exc:
            st.warning(f"IOL no disponible ({exc}). Intentando fuente alternativa.")
            if preference == "IOL":
                raise

    try:
        from optionsdesk.data.providers.byma_open import BymaOpenProvider
        return BymaOpenProvider()
    except Exception as exc:
        raise RuntimeError(f"BYMA Open Data no disponible: {exc}") from exc


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


def _effective_expiry_calendar(chain: Optional[OptionsChain], fallback: Optional[dict] = None) -> dict:
    from optionsdesk.core.instruments import merge_expiry_calendars

    return merge_expiry_calendars(fallback, getattr(chain, "expiry_calendar", None))


def _is_realtime_feed(provider_health) -> bool:
    return provider_health.source in {"PRIMARY", "IOL"} and provider_health.connected


# ── Helpers de datos ──────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def _load_spot_history(days: int = 180, allow_synthetic: bool = True) -> Optional[pd.DataFrame]:
    """Historial OHLCV de GGAL vía PyOBD (con cache local y fallback sintético)."""
    try:
        from optionsdesk.data.history import UnderlyingHistory
        df = UnderlyingHistory().daily("GGAL", days=days, allow_synthetic=allow_synthetic)
        return df if not df.empty else None
    except Exception:
        return None


@st.cache_data(ttl=3600)
def _load_htf_history(weeks: int = 52, allow_synthetic: bool = True) -> Optional[pd.DataFrame]:
    """Historial semanal (HTF) de GGAL — resampleado desde daily."""
    try:
        from optionsdesk.data.history import UnderlyingHistory
        df = UnderlyingHistory().weekly("GGAL", weeks=weeks, allow_synthetic=allow_synthetic)
        return df if not df.empty else None
    except Exception:
        return None


@st.cache_data(ttl=3600)
def _load_symbol_daily(symbol: str, days: int = 180, allow_synthetic: bool = False) -> Optional[pd.DataFrame]:
    """Historial diario real para acciones argentinas del panel Acciones."""
    try:
        from optionsdesk.data.history import UnderlyingHistory
        df = UnderlyingHistory().daily(symbol.upper(), days=days, allow_synthetic=allow_synthetic)
        return df if not df.empty else None
    except Exception:
        return None


@st.cache_data(ttl=3600)
def _load_symbol_weekly(symbol: str, weeks: int = 52, allow_synthetic: bool = False) -> Optional[pd.DataFrame]:
    """Historial semanal real para acciones argentinas del panel Acciones."""
    try:
        from optionsdesk.data.history import UnderlyingHistory
        df = UnderlyingHistory().weekly(symbol.upper(), weeks=weeks, allow_synthetic=allow_synthetic)
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
    boxes: Optional[list[dict]] = None,
    vzones: Optional[list[dict]] = None,
    show_volume: bool = True,
    max_bars: int = 120,
) -> None:
    """Render de velas japonesas tema oscuro con volumen, medias y niveles.

    df: DataFrame OHLCV con columnas open/high/low/close (+ volume opcional) y
        un índice temporal en 'date' o 'time'. Cae a st.line_chart si plotly
        no está disponible o faltan columnas OHLC.
    levels: lista de {"y": float, "label": str, "color": str, "dash": str}.
    boxes: lista de {"x0", "x1", "y0", "y1", "color", "opacity", "label"} para
        zonas rectangulares (OB, OTE, premium/discount bands, etc.).
        x0/x1 son valores de la misma unidad que el eje x (datetime o str).
        x0=None → borde izquierdo del gráfico. x1=None → borde derecho (último bar).
    vzones: lista de {"x0", "x1", "color", "opacity", "label"} para franjas
        verticales (killzones intradía).
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
    if len(data) < 2:
        st.info("Formando grafico live: se necesitan al menos 2 observaciones.")
        return
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

    sma_palette = ["#f5b301", "#4f9bff", "#b06bff", "#a8e6cf"]
    if smas:
        for i, (label, period) in enumerate(smas.items()):
            if len(data) >= period:
                # Detecta si la etiqueta pide EMA (ej. "EMA13", "EMA50") para
                # calcular media exponencial en lugar de simple.
                close_s = data["close"]
                if label.upper().startswith("EMA"):
                    ma = close_s.ewm(span=period, adjust=False, min_periods=period).mean()
                else:
                    ma = close_s.rolling(period).mean()
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

    # Cajas rectangulares (OB, OTE, premium/discount, FVG)
    if boxes:
        x_last = x.iloc[-1] if hasattr(x, "iloc") else x[-1]
        x_first = x.iloc[0] if hasattr(x, "iloc") else x[0]
        for bx in boxes:
            x0 = bx.get("x0", x_first)
            x1 = bx.get("x1", x_last)
            if x0 is None:
                x0 = x_first
            if x1 is None:
                x1 = x_last
            y0 = bx.get("y0")
            y1 = bx.get("y1")
            if y0 is None or y1 is None:
                continue
            color   = bx.get("color", "#4f9bff")
            opacity = bx.get("opacity", 0.12)
            label   = bx.get("label", "")
            fig.add_shape(
                type="rect",
                x0=x0, x1=x1, y0=y0, y1=y1,
                fillcolor=color,
                opacity=opacity,
                line=dict(color=color, width=0.5),
                row=1, col=1,
            )
            if label:
                mid_y = (float(y0) + float(y1)) / 2
                fig.add_annotation(
                    x=x1, y=mid_y,
                    text=label, showarrow=False,
                    font=dict(size=9, color=color),
                    xanchor="left", yanchor="middle",
                    row=1, col=1,
                )

    # Franjas verticales (killzones intradía)
    if vzones:
        for vz in vzones:
            x0 = vz.get("x0")
            x1 = vz.get("x1")
            if x0 is None or x1 is None:
                continue
            fig.add_vrect(
                x0=x0, x1=x1,
                fillcolor=vz.get("color", "#f5b301"),
                opacity=vz.get("opacity", 0.07),
                line_width=0,
                annotation_text=vz.get("label", ""),
                annotation_position="top left",
                annotation_font_size=9,
                annotation_font_color=vz.get("color", "#f5b301"),
                row=1, col=1,
            )

    if has_vol:
        vol_colors = [up if c >= o else down for o, c in zip(data["open"], data["close"])]
        fig.add_trace(
            go.Bar(x=x, y=data["volume"], marker_color=vol_colors, opacity=0.5,
                   name="Vol", showlegend=False),
            row=2, col=1,
        )

    layout = dict(
        template="plotly_dark",
        height=height,
        margin=dict(l=8, r=56, t=28 if title else 8, b=8),
        paper_bgcolor="#0e0e12", plot_bgcolor="#0e0e12",
        xaxis_rangeslider_visible=False,
        showlegend=bool(smas),
        legend=dict(orientation="h", y=1.04, x=0, font=dict(size=11),
                    bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified",
        dragmode="pan",
    )
    if title:
        layout["title"] = dict(text=title, font=dict(size=14, color="#d4d4d8"))
    fig.update_layout(**layout)
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
        st.markdown(f"**💡 Intención:** {rec.intention}")
        st.markdown(f"**✅ Escenario Ganador:** {rec.win_scenario}")
        st.markdown(f"**❌ Escenario Perdedor:** {rec.lose_scenario}")
        st.divider()
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
        if st.button(
            "Generar ticket",
            key=f"btn_ticket_{profile.value}_{idx}",
            disabled=not settings.costs_verified,
            help="Requiere un perfil de costos documentado del ALyC.",
        ):
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
        st.caption("Carga y confirma manualmente en Matriz. Ticket registrado en data/paper_orders.jsonl")


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
                    if st.button(
                        "Ticket",
                        key=f"btn_alt_ticket_{profile.value}_{i}",
                        disabled=not settings.costs_verified,
                    ):
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
                _fmt_signals(display), use_container_width=True,
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
                _fmt_signals(display), use_container_width=True,
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


# ── Tab: Referencia ───────────────────────────────────────────────────────────

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
            st.dataframe(_style_chain(df, spot), use_container_width=True, hide_index=True)


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


def _build_smc_overlays(
    smc_ctx: object,
    df: "pd.DataFrame",
) -> tuple[list[dict], list[dict]]:
    """Construye las listas `levels` y `boxes` para _candlestick_chart desde el SmcContext.

    levels: líneas horizontales BSL/SSL (add_hline).
    boxes: cajas rectangulares OB, OTE, premium/discount bands (add_shape rect).
    """
    if smc_ctx is None or df is None or df.empty:
        return [], []

    levels: list[dict] = []
    boxes: list[dict]  = []

    x_last  = df["date"].iloc[-1]  if "date"  in df.columns else df.index[-1]
    x_first = df["date"].iloc[-60] if "date"  in df.columns and len(df) >= 60 else (
              df["date"].iloc[0]   if "date"  in df.columns else df.index[0])

    # BSL/SSL como líneas horizontales
    for lv in getattr(smc_ctx, "liquidity", []):
        if getattr(lv, "swept", True):
            continue  # nivel ya barrido, no dibujar
        kind  = getattr(lv, "kind", "")
        price = getattr(lv, "price", 0.0)
        label = getattr(lv, "label", "")
        if kind == "BSL":
            color = "#ef5350"   # rojo: liquidity arriba (zona de resistencia/stops)
            dash  = "dot"
        else:
            color = "#26a69a"   # verde: liquidity abajo (zona de soporte/stops)
            dash  = "dot"
        levels.append({"y": price, "label": label, "color": color, "dash": dash})

    # PRZ de Peak Formations M/W (armónicos Fase 3)
    peak_formations = getattr(smc_ctx, "peak_formations", [])
    if peak_formations:
        try:
            from optionsdesk.signals.harmonics import prz_boxes as _prz_boxes
            # Resolver fechas relativas a las fechas del df
            pf_boxes = _prz_boxes(peak_formations)
            for bx in pf_boxes:
                if bx.get("x0") is None:
                    bx["x0"] = x_first
            boxes.extend(pf_boxes)
        except Exception:
            pass

    # PDA premium/discount bands
    pda = getattr(smc_ctx, "pda", None)
    if pda is not None:
        sh = getattr(pda, "swing_high", 0.0)
        sl = getattr(pda, "swing_low",  0.0)
        pb = getattr(pda, "premium_band",  0.0)
        db = getattr(pda, "discount_band", 0.0)
        if sh > 0 and sl > 0 and sh > sl:
            # Banda premium (sobre 0.618)
            boxes.append({
                "x0": x_first, "x1": x_last,
                "y0": pb, "y1": sh,
                "color": "#ef5350", "opacity": 0.06,
                "label": "Premium",
            })
            # Banda discount (bajo 0.382)
            boxes.append({
                "x0": x_first, "x1": x_last,
                "y0": sl, "y1": db,
                "color": "#26a69a", "opacity": 0.06,
                "label": "Discount",
            })

    # OTE zone del swing activo
    ote = getattr(smc_ctx, "ote", None)
    if ote is not None:
        ote_lo   = getattr(ote, "lo", 0.0)
        ote_hi   = getattr(ote, "hi", 0.0)
        ote_dir  = getattr(ote, "direction", "")
        if ote_lo > 0 and ote_hi > ote_lo:
            ote_color = "#26a69a" if ote_dir == "bullish" else "#ef5350"
            boxes.append({
                "x0": x_first, "x1": x_last,
                "y0": ote_lo, "y1": ote_hi,
                "color": ote_color, "opacity": 0.14,
                "label": f"OTE {ote_dir[:4]}",
            })

    return levels, boxes


def _render_smc_panel(smc_ctx: object) -> None:
    """Renderiza el expander 'Smart Money Concepts' en el tab direccional."""
    with st.expander("Smart Money Concepts — Liquidez & Ciclo MM", expanded=False):
        pda       = getattr(smc_ctx, "pda", None)
        liquidity = getattr(smc_ctx, "liquidity", [])
        sweeps    = getattr(smc_ctx, "sweeps", [])
        mm        = getattr(smc_ctx, "mm", None)
        killzone  = getattr(smc_ctx, "killzone", None)
        ote       = getattr(smc_ctx, "ote", None)
        adr_conf  = getattr(smc_ctx, "adr_confluence", None)
        sig       = getattr(smc_ctx, "signal", None)

        # ── SmcSignal — señal de trading directa ─────────────────────────────
        if sig is not None:
            _Q_FN = {"S": st.success, "A": st.success, "B": st.info,
                     "C": st.warning, "none": st.info}
            _Q_LABELS = {"S": "SNIPER (S)", "A": "Fuerte (A)", "B": "Moderada (B)",
                         "C": "Marginal (C)", "none": "Sin señal"}
            quality = getattr(sig, "quality", "none")
            direction = getattr(sig, "direction", "neutral")
            reason = getattr(sig, "reason", "")

            fn_use = _Q_FN.get(quality, st.info)
            fn_use(
                f"Calidad **{_Q_LABELS.get(quality, quality)}** — "
                f"Dirección **{direction.upper()}**  ·  {reason}"
            )

            _sig_cols = st.columns(3)
            entry_lbl = getattr(sig, "entry_label", "")
            target_lbl = getattr(sig, "target_label", "")
            stop_lbl  = getattr(sig, "stop_label", "")
            if entry_lbl:  _sig_cols[0].caption(f"Entry: {entry_lbl}")
            if target_lbl: _sig_cols[1].caption(f"Target: {target_lbl}")
            if stop_lbl:   _sig_cols[2].caption(f"Stop: {stop_lbl}")

            _opts = st.columns(3)
            _opts[0].metric("Short Put", "SI" if getattr(sig, "favors_short_put", False) else "NO")
            _opts[1].metric("Call Cubierto", "SI" if getattr(sig, "favors_covered_call", False) else "NO")
            _opts[2].metric("Esperar", "SI" if getattr(sig, "favors_wait", True) else "NO")
            st.divider()

        # Fila 1: zona PDA + killzone + ADR
        pcols = st.columns([2, 1, 1])
        with pcols[0]:
            if pda is not None:
                zone = getattr(pda, "zone", "?")
                eq   = getattr(pda, "equilibrium", 0.0)
                sh   = getattr(pda, "swing_high", 0.0)
                sl   = getattr(pda, "swing_low", 0.0)
                zone_fn = {"premium": st.error, "discount": st.success, "equilibrium": st.info}
                zone_lbl = {"premium": "PREMIUM — zona de calls", "discount": "DISCOUNT — zona de puts", "equilibrium": "EQUILIBRIO (0.5 fib)"}
                zone_fn.get(zone, st.info)(zone_lbl.get(zone, zone))
                st.caption(f"Swing: ${sl:,.0f} – ${sh:,.0f} | Equilibrio: ${eq:,.0f}")
            else:
                st.info("PDA array sin datos suficientes.")

        with pcols[1]:
            if killzone:
                kz_map = {"manipulacion": ("Killzone Apertura", "warning"),
                           "distribucion": ("Killzone Cierre", "warning")}
                lbl, fn = kz_map.get(killzone, (killzone, "info"))
                getattr(st, fn)(lbl)
            else:
                st.info("Fuera de killzone")

        with pcols[2]:
            if adr_conf == "confirmed":
                st.success("ADR confirma")
            elif adr_conf == "fx_driven":
                st.warning("Movido por dólar")
            else:
                st.info("ADR: sin datos")

        # OTE del swing activo
        if ote is not None:
            ote_lo  = getattr(ote, "lo", 0.0)
            ote_hi  = getattr(ote, "hi", 0.0)
            ote_dir = getattr(ote, "direction", "")
            f618    = getattr(ote, "fib_618", 0.0)
            f786    = getattr(ote, "fib_786", 0.0)
            f886    = getattr(ote, "fib_886", 0.0)
            st.caption(
                f"OTE {ote_dir}: ${ote_lo:,.0f}–${ote_hi:,.0f} "
                f"| Fib 0.618=${f618:,.0f} | 0.786=${f786:,.0f} | 0.886=${f886:,.0f}"
            )

        # BSL/SSL table
        active_lv = [lv for lv in liquidity if not getattr(lv, "swept", True)]
        if active_lv:
            lv_rows = []
            for lv in sorted(active_lv, key=lambda l: getattr(l, "price", 0), reverse=True):
                lv_rows.append({
                    "Tipo": getattr(lv, "kind", ""),
                    "Precio": f"${getattr(lv, 'price', 0):,.0f}",
                    "Label": getattr(lv, "label", ""),
                    "TF": getattr(lv, "timeframe", ""),
                })
            st.dataframe(lv_rows, use_container_width=True, hide_index=True)
        else:
            st.caption("No hay niveles de liquidez activos (todos barridos).")

        # Sweeps recientes
        rev_sweeps = [sw for sw in sweeps if getattr(sw, "reversed", False)]
        if rev_sweeps:
            sw_strs = []
            for sw in rev_sweeps[-3:]:
                lv    = getattr(sw, "level", None)
                p     = getattr(lv, "price", 0.0) if lv else 0.0
                direc = getattr(sw, "direction", "")
                label = getattr(lv, "label", "") if lv else ""
                kind  = getattr(lv, "kind", "") if lv else ""
                sw_strs.append(f"{kind} {label} ${p:,.0f} (barrido {'↑' if direc=='up' else '↓'} + reversión)")
            st.warning("Stop hunts con reversión: " + " | ".join(sw_strs))

        # Peak Formations M/W (armónicos — Fase 3)
        peak_formations = getattr(smc_ctx, "peak_formations", [])
        if peak_formations:
            st.markdown("**Peak Formations M/W**")
            for pf in peak_formations[:3]:
                pf_kind  = getattr(pf, "kind", "")
                pf_dir   = getattr(pf, "direction", "")
                pf_d     = getattr(pf, "d", 0.0)
                pf_prz_lo = getattr(pf, "prz_lo", 0.0)
                pf_prz_hi = getattr(pf, "prz_hi", 0.0)
                pf_tp1   = getattr(pf, "tp1", 0.0)
                pf_tp2   = getattr(pf, "tp2", 0.0)
                pf_fn    = st.success if pf_dir == "bullish" else st.error
                pf_fn(
                    f"Formación {pf_kind} ({pf_dir}) | D=${pf_d:,.0f} | "
                    f"PRZ ${pf_prz_lo:,.0f}–${pf_prz_hi:,.0f} | "
                    f"TP1=${pf_tp1:,.0f} | TP2=${pf_tp2:,.0f}"
                )

        # MM Read
        if mm is not None:
            st.markdown("**Market Maker Method**")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Stack EMAs", mm.stack.capitalize())
            m2.metric("EMA13 vs 50", mm.ema13_cross_50)
            m3.metric("Niveles MM", str(mm.level_count))
            m4.metric("Peak Formation", "Sí" if mm.peak_formation else "No")
            st.caption(
                f"EMA5={mm.ema5:,.0f} | EMA13={mm.ema13:,.0f} | "
                f"EMA50={mm.ema50:,.0f} | EMA200={mm.ema200:,.0f}"
            )


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

    from optionsdesk.signals.technical import analyze as _analyze, rsi as _rsi

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

    # ── Gráfico de precio con overlays SMC ───────────────────────────────
    close = spot_history["close"]
    dates = spot_history["date"]

    smc_ctx = getattr(context, "smc", None) if context else None
    _smc_levels, _smc_boxes = _build_smc_overlays(smc_ctx, spot_history)

    _candlestick_chart(
        spot_history,
        height=480,
        smas={"EMA13": 13, "EMA50": 50, "EMA200": 200},
        levels=_smc_levels,
        boxes=_smc_boxes,
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

    # ── SMC badges (structure) ────────────────────────────────────────────
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

    # ── Panel Smart Money ─────────────────────────────────────────────────
    if smc_ctx is not None:
        _render_smc_panel(smc_ctx)

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

            e1, e2, e3, e4 = st.columns(4)
            e1.metric("Break-even", f"${idea.breakeven:,.0f}")
            e2.metric("Objetivo",   f"${idea.target_spot:,.0f}")
            e3.metric("Stop",       f"${idea.stop_spot:,.0f}")
            e4.metric("R/R",        f"{idea.rr_ratio:.1f}:1")

            st.write(idea.rational)
            if idea.confluence:
                st.caption("Confluencia SMC: " + " · ".join(idea.confluence))
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
        spread_conf = _confluence(context, daily_snap, getattr(context, "smc", None), context.trend in ("alcista", "ALCISTA"), "", "", spot=spot)
        if spread_conf:
            st.caption("Confluencia SMC: " + " · ".join(spread_conf))
        st.caption(sr.disclaimer)


# ── Tab: Historial ────────────────────────────────────────────────────────────
def _close_position_manual(index: int, monitor, exit_price: Optional[float] = None, reason: str = "manual_local_close") -> None:
    if monitor.close_position_at(index, reason=reason, exit_price=exit_price):
        st.success("Posicion archivada como cerrada localmente. No se envio ninguna orden a Matriz.")
    else:
        st.error("No se pudo actualizar el registro local.")

def _scalp_money(value: float | int | None, decimals: int = 0) -> str:
    if value is None:
        return "-"
    try:
        val = float(value)
        if decimals == 0:
            return f"${val:,.0f}"
        else:
            return f"${val:,.{decimals}f}"
    except (ValueError, TypeError):
        return str(value)

def _scalp_pct(value: float | int | None, decimals: int = 2) -> str:
    if value is None:
        return "-"
    try:
        val = float(value)
        return f"{val:+.{decimals}f}%"
    except (ValueError, TypeError):
        return str(value)

def _stock_quote_age_s(quote, now: datetime) -> float:
    ts = getattr(quote, "book_ts", None) or getattr(quote, "received_at", None) or getattr(quote, "timestamp", None)
    if ts is None:
        return float("inf")
    if now.tzinfo is not None and ts.tzinfo is None:
        ts = ts.replace(tzinfo=now.tzinfo)
    elif now.tzinfo is None and ts.tzinfo is not None:
        ts = ts.astimezone().replace(tzinfo=None)
    return max((now - ts).total_seconds(), 0.0)

def _stock_age_label(age_s: Optional[float]) -> str:
    if age_s is None or age_s == float("inf"):
        return "-"
    if age_s < 60:
        return f"{age_s:.0f}s"
    return f"{age_s / 60:.1f}m"

def _stock_closed_pnl_delta(trades) -> str:
    if not trades:
        return ""
    last = trades[-1].pnl_ars
    return f"ultimo {_scalp_money(last)}"

def _tab_stocks(provider: MarketDataProvider, provider_health: MarketDataHealth, adaptive_context: Optional[AdaptiveContext] = None) -> None:
    """Acciones argentinas: scanner long-only, demo automatico y operacion manual real/simulada."""
    import streamlit as st
    import pandas as pd
    from pathlib import Path
    from datetime import datetime
    
    from optionsdesk.backtest.stock_demo import (
        positions_frame,
        run_stock_demo_tick,
        trades_frame,
    )
    from optionsdesk.data.stock_tape import append_stock_quotes, load_stock_tape, stock_tape_bars
    from optionsdesk.signals.stock_signals import scan_stock_symbol
    from optionsdesk.execution.base import Order, OrderSide, OrderStatus

    st.subheader("Acciones Argentinas")
    st.caption(
        "Acciones líderes del panel general. El bot demo automático opera swing long-only basándose en confluencias SMC."
    )

    universe = settings.stock_universe_symbols()
    if not universe:
        st.warning("STOCK_UNIVERSE está vacío. Carga símbolos en .env para activar el panel.")
        return

    symbols = sorted([s.upper() for s in universe])
    now = datetime.now()

    # 1. Fetch live quotes
    try:
        quotes = provider.get_quotes(symbols)
    except Exception as exc:
        st.error(f"No se pudieron obtener cotizaciones de acciones: {exc}")
        return

    # Persist live quotes to stock tape
    append_stock_quotes(quotes.values(), now=now, min_interval_s=10.0)
    tape_df = load_stock_tape(symbols=symbols, now=now)

    # Edge realizado: qué setups/grades ya probaron perder (no mostrar como operables).
    blocked_setups: set = set()
    blocked_grades: set = set()
    try:
        from optionsdesk.performance.attribution import block_lists
        blocked_setups, blocked_grades = block_lists(settings.stock_demo_trades_file.parent)
    except Exception:
        pass
    from optionsdesk.signals.stock_signals import signal_grade, signal_edge_status

    # Scan and generate signals
    all_signals = []
    table_rows = []
    fresh_quotes = 0
    executable_quotes = 0

    for symbol in symbols:
        quote = quotes.get(symbol)
        if quote is None:
            table_rows.append({
                "Símbolo": symbol,
                "Precio": "-",
                "Spread": "-",
                "Señal": "sin quote",
                "Setup": "-",
                "Score": "-",
                "Stop": "-",
                "TP": "-",
                "Frescura": "-",
                "Motivo": "BYMA/IOL no devolvió cotización",
            })
            continue

        # Check freshness
        age_s = _stock_quote_age_s(quote, now)
        if age_s is not None and age_s <= settings.stock_demo_quote_max_age_s:
            fresh_quotes += 1
        if quote.bid > 0 and quote.ask > quote.bid:
            executable_quotes += 1

        # Retrieve intraday bars from tape
        bars_1m = stock_tape_bars(tape_df, symbol, "1min")
        
        # Load daily and weekly histories for scanning
        daily = _load_symbol_daily(symbol, days=180, allow_synthetic=False)
        weekly = _load_symbol_weekly(symbol, weeks=52, allow_synthetic=False)
        
        # Scan symbol for SMC confluence setups
        signals = scan_stock_symbol(
            symbol, quote, bars_1m=bars_1m, daily=daily, weekly=weekly, now=now,
            adaptive_context=adaptive_context
        )
        all_signals.extend(signals)
        
        best = signals[0] if signals else None
        blocker = "-"
        if best is None:
            if quote.bid <= 0 or quote.ask <= quote.bid:
                blocker = "sin bid/ask ejecutable"
            elif age_s is not None and age_s > settings.stock_demo_quote_max_age_s:
                blocker = "quote viejo"
            elif len(bars_1m) < 8:
                blocker = "formando tape intradía"
            else:
                blocker = "sin setup long"
                
        # SMC enrichment fields (v2.4) — mostrar zona PDA, ADR y liquidez cercana
        smc_col_parts: list[str] = []
        if best is not None:
            pda_z   = getattr(best, "pda_zone", None)
            adr_c   = getattr(best, "adr_confluence", None)
            near_lv = getattr(best, "nearest_liquidity", None)
            _PDA_LABELS = {"premium": "▼PREM", "discount": "▲DISC", "equilibrium": "=EQ"}
            if pda_z:
                smc_col_parts.append(_PDA_LABELS.get(pda_z, pda_z))
            if adr_c and adr_c != "unknown":
                smc_col_parts.append("ADR✓" if adr_c == "confirmed" else "CCL⚠")
            if near_lv:
                smc_col_parts.append(near_lv[:16])
        smc_col = " | ".join(smc_col_parts) if smc_col_parts else "-"

        # Volumen: RVOL + OBV del día — detecta la ola temprano en BYMA.
        # `daily` ya está disponible del scan anterior — no re-descargamos.
        vol_col = "-"
        try:
            from optionsdesk.signals.volume_momentum import volume_snapshot as _vs
            _snp = _vs(daily) if daily is not None else None
            if _snp is not None and _snp.rvol is not None:
                _icons = {"explosivo": "🔥", "alto": "⬆", "normal": "—", "bajo": "⬇"}
                _icon = _icons.get(_snp.rvol_label, "")
                vol_col = f"{_icon}{_snp.rvol:.1f}x"
                if _snp.obv_divergence == "acumulacion_silenciosa":
                    vol_col += " ⚡"
                elif _snp.obv_divergence == "distribucion":
                    vol_col += " ⚠"
        except Exception:
            pass

        # Grade + veredicto de edge realizado (no mostrar basura como operable).
        edge_flag = ""
        grade_col = "-"
        intel_col = "-"
        if best is not None:
            grade_col = signal_grade(best.score)
            status = signal_edge_status(best, blocked_setups, blocked_grades)
            if status:
                edge_flag = f"  ⚠ {status}"
            # Columna Intel: analistas + social
            analyst = getattr(best, "analyst_note", None)
            social  = getattr(best, "social_note", None)
            parts   = [p for p in [analyst, social] if p]
            intel_col = " · ".join(parts) if parts else "-"

        table_rows.append({
            "Símbolo": symbol,
            "Precio": _scalp_money(quote.mid or quote.last, 2),
            "Spread": _scalp_pct(quote.spread_pct, 2),
            "Señal": best.strategy if best else "RADAR",
            "Setup": best.signal_type if best else "-",
            "Grade": grade_col,
            "Score": best.score if best else "-",
            "Vol": vol_col,
            "Stop": _scalp_money(best.stop_price, 2) if best else "-",
            "TP": _scalp_money(best.target_price, 2) if best else "-",
            "Intel": intel_col,
            "SMC": smc_col,
            "Frescura": _stock_age_label(age_s),
            "Motivo": (best.rationale + edge_flag) if best else blocker,
        })

    # Regime diario: ¿risk-on, cauteloso o risk-off?
    regime = None
    try:
        from optionsdesk.signals.regime import compute_daily_regime
        regime = compute_daily_regime(symbols)
    except Exception:
        pass

    # Run the demo engine tick (focus strictly on swings, disable scalping)
    swing_signals = [s for s in all_signals if s.strategy == "SWING"]
    # El regime filtra señales de baja calidad en días de espejismo macro.
    if regime is not None and regime.min_grade_override:
        from optionsdesk.signals.stock_signals import signal_grade as _sg
        _grade_order = {"S": 0, "A": 1, "B": 2, "C": 3, "D": 4}
        _min = _grade_order.get(regime.min_grade_override, 99)
        swing_signals = [
            s for s in swing_signals
            if _grade_order.get(_sg(s.score), 99) <= _min
        ]
    demo = run_stock_demo_tick(quotes, swing_signals, max_scalps=0, now=now, adaptive_context=adaptive_context)
    
    try:
        from optionsdesk.backtest.genetic import EvolutionEngine
        if "genetic_engine" not in st.session_state:
            st.session_state.genetic_engine = EvolutionEngine(population_size=20)
        st.session_state.genetic_engine.step(all_signals, quotes)
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Error in genetic step: {e}")
    
    # Calculate open positions P&L
    open_positions_df = positions_frame(demo.positions, quotes)
    open_pnl = 0.0
    if not open_positions_df.empty and "PnL abierto" in open_positions_df.columns:
        open_pnl = float(pd.to_numeric(open_positions_df["PnL abierto"], errors="coerce").fillna(0.0).sum())

    # --- Top summary metrics bar ---
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Feed", "OK" if provider_health.connected else "DEG", provider_health.source)
    m2.metric("Universo", f"{fresh_quotes}/{len(symbols)}", "quotes frescos")
    m3.metric("Book", f"{executable_quotes}/{len(symbols)}", "bid/ask válidos")
    m4.metric("Señales", len(all_signals))
    m5.metric("PnL demo", _scalp_money(demo.total_pnl_ars), _stock_closed_pnl_delta(demo.closed_trades))
    m6.metric("Abiertas", len(demo.positions), f"abierto {_scalp_money(open_pnl)}")

    if adaptive_context:
        mode_color = {"NORMAL": "🟢", "DEFENSIVE": "🟡", "AGGRESSIVE": "🔥", "HALTED": "🛑"}.get(adaptive_context.mode, "")
        kelly_label = f" | Modo: {mode_color} {adaptive_context.mode} | Sizing: {adaptive_context.half_kelly_pct * 100:.1f}% | Min Score: {adaptive_context.min_score_for_entry}"
        if getattr(adaptive_context, "halted", False):
            kelly_label += f" | CIRCUIT BREAKER ACTIVO: {adaptive_context.reason}"
    else:
        kelly_label = " | Sizing: 5.0% (static)"

    interval_label = f"{settings.stock_demo_refresh_interval_s}s"
    # Regime del día en el header
    if regime is not None:
        _r_icons = {"risk-on": "🟢", "cauteloso": "🟡", "risk-off": "🔴"}
        _r_icon = _r_icons.get(regime.label, "⚪")
        st.caption(
            f"{_r_icon} Regime {regime.label.upper()}: {regime.summary}"
        )
    st.caption(
        f"Refresh {interval_label} | guardado tape en {settings.stock_tape_dir}{kelly_label}"
    )

    # Indicador de suficiencia del adaptativo: hace honesto "el motor está ciego"
    # hasta acumular suficiente historial real de swings.
    try:
        from optionsdesk.performance.attribution import load_all_closed_trades
        _n_swings = sum(1 for t in load_all_closed_trades() if t.realized_pnl_ars is not None)
        _min_s = 15
        if _n_swings < _min_s:
            st.warning(
                f"Adaptive: {_n_swings}/{_min_s} trades cerrados. Motor aprendiendo — "
                "sizing y filtros son conservadores hasta tener historial suficiente. "
                "No aumentar capital real hasta llegar al umbral."
            )
        elif _n_swings < 40:
            st.info(f"Adaptive: {_n_swings} trades. Motor entrenando — resultados aún preliminares.")
    except Exception:
        pass

    if demo.skipped_reasons:
        top = sorted(demo.skipped_reasons.items(), key=lambda kv: kv[1], reverse=True)[:5]
        st.caption("Bloqueos demo: " + " | ".join(f"{k} ({v})" for k, v in top))

    # --- Display details layout ---
    left, right = st.columns([1.1, 1], gap="large")

    with left:
        st.markdown("#### Scanner & Señales en Tiempo Real")

        # Enriquecer con Grade y ordenar por score descendente
        from optionsdesk.signals.stock_signals import signal_grade as _sg
        _GRADE_COLORS = {"S": "🟣", "A": "🟢", "B": "🟡", "C": "🟠", "D": "🔴"}
        for row in table_rows:
            sc = row.get("Score")
            if isinstance(sc, (int, float)):
                g = _sg(float(sc))
                row["Grade"] = _GRADE_COLORS.get(g, "") + g
            else:
                row["Grade"] = "—"
        table_rows_sorted = sorted(
            table_rows,
            key=lambda r: (r.get("Score") if isinstance(r.get("Score"), (int, float)) else -1),
            reverse=True,
        )

        # Top pick highlight (solo si hay señal real con Grade S o A)
        top = next((r for r in table_rows_sorted if r.get("Grade", "—")[:2] in ("🟣S", "🟢A")), None)
        if top:
            sig_type = top.get("Setup", "-")
            is_smcrev = sig_type == "SMC_REVERSAL"
            badge_txt = "SNIPER ENTRY" if is_smcrev else "TOP SETUP"
            st.success(
                f"**{badge_txt}** — {top['Símbolo']} | "
                f"{top.get('Grade','')} Score {top.get('Score','-')} | "
                f"Entrada {top.get('Precio','-')} | "
                f"Stop {top.get('Stop','-')} | TP {top.get('TP','-')} | "
                f"{top.get('Motivo','')[:80]}"
            )

        # Reordenar columnas: Grade al frente
        cols_order = ["Grade", "Símbolo", "Precio", "Spread", "Señal", "Setup",
                      "Score", "Stop", "TP", "SMC", "Frescura", "Motivo"]
        df_display = pd.DataFrame(table_rows_sorted)
        available_cols = [c for c in cols_order if c in df_display.columns]
        st.dataframe(df_display[available_cols], hide_index=True, use_container_width=True)
        
        st.divider()
        # Interactive Real Trading order cockpit for Stocks
        st.markdown("#### Operar Acciones (BYMA)")
        if not settings.is_iol_configured():
            st.info("Configura IOL_USER / IOL_PASSWORD en el .env para habilitar ruteo de órdenes.")
        else:
            if "iol_executor" not in st.session_state:
                try:
                    from optionsdesk.execution.iol_executor import IOLExecutor
                    st.session_state["iol_executor"] = IOLExecutor.from_settings()
                except Exception as exc:
                    st.error(f"Error al iniciar ejecución IOL: {exc}")
            
            if "iol_executor" in st.session_state:
                executor = st.session_state["iol_executor"]
                
                # Interactive Quick Order Entry Form
                sc1, sc2 = st.columns(2)
                stock_sym = sc1.selectbox("Símbolo Acción", symbols, key="stock_tk_symbol")
                stock_side = sc2.radio("Lado", ["Comprar", "Vender"], key="stock_tk_side", horizontal=True)
                
                sc3, sc4, sc5 = st.columns(3)
                stock_qty = sc3.number_input("Cantidad Acciones", min_value=1, step=10, key="stock_tk_qty", value=100)
                
                # Get current stock quote for prefilling/displaying info
                stk_q = quotes.get(stock_sym)
                ask_px = float(stk_q.ask) if stk_q and stk_q.ask > 0 else (stk_q.last if stk_q else 0.0)
                bid_px = float(stk_q.bid) if stk_q and stk_q.bid > 0 else (stk_q.last if stk_q else 0.0)
                
                st.session_state.setdefault("stock_tk_price", ask_px)
                stock_px = sc4.number_input("Precio Límite (ARS)", min_value=0.0, step=0.1, key="stock_tk_price", value=ask_px)
                stock_term = sc5.selectbox("Plazo de liquidación", ["t1", "t0", "t2"], key="stock_tk_term")
                
                gross = stock_px * stock_qty
                fee_side = "stock_buy" if stock_side == "Comprar" else "stock_sell"
                fees = DEFAULT_COSTS.gross_cost(gross, fee_side)
                total_cash = gross + fees if stock_side == "Comprar" else gross - fees
                
                verb = "Pagas" if stock_side == "Comprar" else "Cobras"
                st.caption(
                    f"{verb} aproximado: **${total_cash:,.2f}** (bruto ${gross:,.2f} + comisiones ${fees:,.2f})"
                )
                
                confirmed = st.checkbox("Confirmo orden REAL de Acciones en IOL", key="stock_operar_confirm")
                if st.button(f"Enviar Orden: {stock_side} {stock_qty} {stock_sym}", key="stock_send_btn", type="primary", disabled=not confirmed or stock_px <= 0, use_container_width=True):
                    order_side = OrderSide.BUY if stock_side == "Comprar" else OrderSide.SELL
                    order = Order(
                        symbol=stock_sym,
                        side=order_side,
                        quantity=stock_qty,
                        limit_price=stock_px,
                        strategy_ref="MANUAL_STOCKS",
                        term=stock_term
                    )
                    with st.spinner("Enviando orden a IOL..."):
                        res = executor.submit(order)
                    if res.status == OrderStatus.SUBMITTED:
                        st.success(f"¡Orden enviada con éxito! ID: {res.broker_id}. {res.notes}")
                    else:
                        st.error(f"Orden rechazada: {res.notes}")
                        
                    # Trigger account refresh in operar tab
                    if "operar_acct" in st.session_state:
                        del st.session_state["operar_acct"]

    with right:
        st.markdown("#### Posiciones Abiertas (Demo Swing)")
        if open_positions_df.empty:
            st.info("Sin posiciones demo abiertas.")
        else:
            view = open_positions_df.copy()
            if "Hora" in view.columns:
                view["Hora"] = pd.to_datetime(view["Hora"]).dt.strftime("%H:%M:%S")
            for col in ("Entrada", "Bid actual", "Stop", "TP", "PnL abierto"):
                if col in view.columns:
                    view[col] = pd.to_numeric(view[col], errors="coerce").round(2)
            st.dataframe(view, hide_index=True, use_container_width=True)

        st.divider()
        st.markdown("#### Historial de Trades Cerrados (Demo)")
        all_trades = sorted(demo.closed_trades, key=lambda t: t.exit_ts, reverse=True)
        if not all_trades:
            st.info("Sin trades cerrados todavía.")
        else:
            t_rows = []
            for t in all_trades[:15]:
                t_rows.append({
                    "Símbolo": t.symbol,
                    "Estrategia": t.strategy,
                    "Entrada": round(t.entry_price, 2),
                    "Salida": round(t.exit_price, 2),
                    "Cantidad": t.quantity,
                    "PnL neto": round(t.pnl_ars, 2),
                    "Motivo": t.exit_reason,
                    "Cierre": t.exit_ts.strftime("%Y-%m-%d %H:%M"),
                })
            st.dataframe(pd.DataFrame(t_rows), hide_index=True, use_container_width=True)

def _tab_portfolio(provider, chain, spot, cc_filtered=None, sp_filtered=None, now=None, caucion_tna=60.0, adaptive_context=None):
    import streamlit as st
    import pandas as pd
    st.subheader("Opciones: Rentas en Vivo (Demo)")
    try:
        from optionsdesk.backtest.options_demo import run_options_demo_tick
    except ImportError:
        st.error("No se pudo importar options_demo")
        return
        
    candidates = []
    if cc_filtered: candidates.extend(cc_filtered[:3])
    if sp_filtered: candidates.extend(sp_filtered[:3])
    candidates.sort(key=lambda x: getattr(x, "score", 0), reverse=True)
    
    try:
        demo = run_options_demo_tick(
            chain=chain,
            candidates=candidates,
            now=now,
            adaptive_context=adaptive_context,
            caucion_tna_pct=caucion_tna,
        )
        positions = demo.get("positions", [])
        closed = demo.get("closed", [])
    except Exception as e:
        st.error(f"Error corriendo options demo: {e}")
        positions = []
        closed = []

    if not positions:
        st.info("Buscando oportunidades seguras de yield...")
    else:
        st.caption("Posiciones vivas automatizadas (Short Puts / Covered Calls)")
        rows = []
        import datetime
        from dataclasses import asdict
        for p in positions:
            quote = chain.options.get(p.symbol) if chain else None
            mark = float(quote.ask) if quote and quote.ask > 0 else None
            pnl_net = None
            if mark is not None:
                pnl_gross = (p.premium_received - mark) * p.contracts * 100
                pnl_net = pnl_gross
                
            entry_d = p.entry_date
            if isinstance(entry_d, str): entry_d = datetime.date.fromisoformat(entry_d)
            now_d = now.date() if now else datetime.date.today()
            days_remaining = (entry_d + datetime.timedelta(days=p.days_entry) - now_d).days
            
            rows.append({
                "Símbolo": p.symbol,
                "Estrategia": p.strategy,
                "DTE": days_remaining,
                "Strike": p.strike,
                "Lotes": p.contracts,
                "Prima (Entrada)": _scalp_money(p.premium_received, 2),
                "Recompra (Ask)": _scalp_money(mark, 2) if mark is not None else "-",
                "PnL Abierto": _scalp_money(pnl_net, 2) if pnl_net is not None else "-",
            })

        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    if closed:
        st.caption("Últimas posiciones cerradas")
        c_rows = []
        from dataclasses import asdict
        for c in closed[-5:]:
            d = c.__dict__ if hasattr(c, '__dict__') else asdict(c)
            c_rows.append(d)
        st.dataframe(pd.DataFrame(c_rows), hide_index=True)


# ── Operatoria real (IOL) ─────────────────────────────────────────────────────

def _operar_option_meta(symbol: str, spot: float) -> tuple[str, float]:
    """Devuelve (tipo, strike aproximado) a partir del simbolo GFG."""
    m = re.match(r"^GFG([CV])(\d+(?:[.,]\d+)?)", symbol.upper())
    if not m:
        return ("?", 0.0)
    tipo = "CALL" if m.group(1) == "C" else "PUT"
    try:
        strike = float(m.group(2).replace(",", "."))
    except ValueError:
        return (tipo, 0.0)
    if spot > 0:
        while strike > spot * 3:
            strike /= 10.0
        while strike < spot / 3:
            strike *= 10.0
    return (tipo, strike)


def _operar_set_ticket(symbol: str, side: str, price: float, qty: Optional[int] = None) -> None:
    """Callback de los botones rapidos: prellena el ticket."""
    st.session_state["tk_symbol"] = symbol
    st.session_state["tk_side"] = side
    st.session_state["tk_price"] = round(float(price), 2)
    if qty is not None:
        st.session_state["tk_qty"] = int(qty)
    st.session_state["operar_confirm"] = False


def _operar_load_account(executor) -> None:
    """Trae ordenes vivas, posiciones e historial en una sola pasada."""
    try:
        st.session_state["operar_acct"] = {
            "orders": executor.get_open_orders(),
            "positions": executor.get_positions(),
            "history": executor.get_history(days=7),
            "ts": pd.Timestamp.now().strftime("%H:%M:%S"),
            "error": "",
        }
    except Exception as exc:  # noqa: BLE001 - mostrar causa al usuario
        st.session_state["operar_acct"] = {
            "orders": [], "positions": [], "history": [], "ts": "", "error": str(exc),
        }


def _tab_operar(provider: MarketDataProvider, chain: Optional[OptionsChain]) -> None:
    """Cockpit de trading de opciones GGAL via API de IOL.

    Precios en vivo, ticket rapido, posiciones con cierre en 1 click, ordenes
    activas e historial. Una confirmacion por orden (es plata real).
    """
    from optionsdesk.execution.iol_executor import IOLExecutor

    if not settings.is_iol_configured():
        st.info("Configura IOL_USER / IOL_PASSWORD en el .env para operar.")
        return
    if chain is None or not chain.options:
        st.warning("Sin cadena de opciones fresca. Actualiza los datos arriba.")
        return

    if "iol_executor" not in st.session_state:
        try:
            st.session_state["iol_executor"] = IOLExecutor.from_settings()
        except Exception as exc:  # noqa: BLE001
            st.error(f"No se pudo iniciar la operatoria IOL: {exc}")
            return
    executor = st.session_state["iol_executor"]

    spot = chain.spot.mid
    symbols = sorted(chain.options.keys(), key=lambda s: abs(_operar_option_meta(s, spot)[1] - spot))
    if st.session_state.get("tk_symbol") not in symbols:
        st.session_state["tk_symbol"] = symbols[0]
    st.session_state.setdefault("tk_side", "Comprar")
    st.session_state.setdefault("tk_qty", 1)
    st.session_state.setdefault("tk_term", "t1")

    if "operar_acct" not in st.session_state:
        _operar_load_account(executor)
    acct = st.session_state.get("operar_acct", {})

    # ── Barra superior ────────────────────────────────────────────────────────
    b1, b2, b3 = st.columns([1, 1, 1])
    b1.metric("GGAL", f"${spot:,.0f}")
    b2.metric("Cuenta IOL", f"actualizada {acct.get('ts', '—')}")
    b3.button("Actualizar cuenta", key="operar_refresh_acct", use_container_width=True,
              on_click=_operar_load_account, args=(executor,))
    if acct.get("error"):
        st.error(f"IOL: {acct['error']}")

    left, right = st.columns([1.05, 1], gap="large")

    with left:
        _operar_prices(chain, spot)
        st.divider()
        _operar_ticket(executor, chain, symbols)

    with right:
        _operar_positions(acct.get("positions", []), chain)
        st.divider()
        _operar_open_orders(executor, acct.get("orders", []))
        st.divider()
        _operar_history(acct.get("history", []))


def _operar_prices(chain: OptionsChain, spot: float, top: int = 14) -> None:
    """Tabla de puntas de las opciones mas cercanas al ATM con compra/venta rapida."""
    st.markdown("#### Precios")
    rows = sorted(chain.options.items(), key=lambda kv: abs(_operar_option_meta(kv[0], spot)[1] - spot))[:top]
    h = st.columns([1.6, 0.8, 0.9, 0.9, 0.9, 0.8, 0.8])
    for c, label in zip(h, ["Opcion", "Tipo", "Bid", "Ask", "Ult.", "", ""]):
        c.caption(label)
    for sym, q in rows:
        tipo, strike = _operar_option_meta(sym, spot)
        c = st.columns([1.6, 0.8, 0.9, 0.9, 0.9, 0.8, 0.8])
        c[0].write(f"**{sym}**")
        c[1].caption(f"{tipo} {strike:,.0f}")
        c[2].write(f"{q.bid:,.1f}")
        c[3].write(f"{q.ask:,.1f}")
        c[4].write(f"{q.last:,.1f}")
        buy_px = q.ask if q.ask > 0 else (q.mid if q.mid > 0 else q.last)
        sell_px = q.bid if q.bid > 0 else (q.mid if q.mid > 0 else q.last)
        c[5].button("Comprar", key=f"qbuy_{sym}", on_click=_operar_set_ticket,
                    args=(sym, "Comprar", buy_px), use_container_width=True)
        c[6].button("Vender", key=f"qsell_{sym}", on_click=_operar_set_ticket,
                    args=(sym, "Vender", sell_px), use_container_width=True)


def _operar_ticket(executor, chain: OptionsChain, symbols: list[str]) -> None:
    """Ticket de envio con una confirmacion."""
    st.markdown("#### Ticket")
    symbol = st.session_state["tk_symbol"]
    quote = chain.options.get(symbol)
    if quote is not None:
        st.caption(f"{symbol} — bid {quote.bid:,.1f} / ask {quote.ask:,.1f}")
    st.session_state.setdefault("tk_price", round(float(quote.ask if quote else 0.0), 2))

    c1, c2 = st.columns(2)
    c1.selectbox("Opcion", symbols, key="tk_symbol")
    c2.radio("Lado", ["Comprar", "Vender"], key="tk_side", horizontal=True)
    c3, c4, c5 = st.columns(3)
    c3.number_input("Cantidad", min_value=1, step=1, key="tk_qty")
    c4.number_input("Precio (ARS)", min_value=0.0, step=0.5, key="tk_price")
    c5.selectbox("Plazo", ["t1", "t0", "t2"], key="tk_term")

    side = OrderSide.BUY if st.session_state["tk_side"] == "Comprar" else OrderSide.SELL
    qty = int(st.session_state["tk_qty"])
    price = float(st.session_state["tk_price"])
    gross = price * qty * 100
    fees = DEFAULT_COSTS.gross_cost(gross, "option_buy" if side == OrderSide.BUY else "option_sell")
    cash = gross + fees if side == OrderSide.BUY else gross - fees
    verb = "Pagas" if side == OrderSide.BUY else "Cobras"
    st.caption(
        f"{verb} aprox **\\${cash:,.0f}** (bruto \\${gross:,.0f} + costos \\${fees:,.0f}). "
        "Validez: fin de rueda."
    )

    confirmed = st.checkbox("Confirmo orden REAL", key="operar_confirm")
    if st.button(f"Enviar: {st.session_state['tk_side']} {qty} {symbol}",
                 key="operar_send_btn", type="primary",
                 disabled=not confirmed or price <= 0, use_container_width=True):
        order = Order(
            symbol=symbol, side=side, quantity=qty, limit_price=price,
            strategy_ref="MANUAL_IOL", term=st.session_state["tk_term"],
        )
        with st.spinner("Enviando orden a IOL..."):
            result = executor.submit(order)
        if result.status == OrderStatus.SUBMITTED:
            st.success(f"Orden enviada. N° {result.broker_id}. {result.notes}")
        else:
            st.error(f"Rechazada: {result.notes}")
        _operar_load_account(executor)


def _operar_positions(positions: list[dict], chain: OptionsChain) -> None:
    """Tenencia de opciones GGAL con cierre rapido (carga el ticket en venta)."""
    st.markdown("#### Posiciones")
    if not positions:
        st.caption("Sin opciones GGAL en cartera.")
        return
    for p in positions:
        sym = p["simbolo"]
        pnl = p.get("ganancia_pct", 0.0)
        c = st.columns([1.7, 1.1, 1.1, 0.9])
        c[0].write(f"**{sym}**")
        c[1].caption(f"x{p['cantidad']:,.0f} @ {p['ppc']:,.1f}")
        c[2].caption(f"ult {p['ultimo']:,.1f} ({pnl:+.1f}%)")
        q = chain.options.get(sym)
        sell_px = (q.bid if q and q.bid > 0 else p.get("ultimo", 0.0)) or 0.0
        c[3].button("Cerrar", key=f"close_{sym}", on_click=_operar_set_ticket,
                    args=(sym, "Vender", sell_px, int(p["cantidad"])), use_container_width=True)
    st.caption("«Cerrar» carga el ticket en venta; revisa cantidad y precio antes de confirmar.")


def _operar_open_orders(executor, orders: list) -> None:
    st.markdown("#### Ordenes activas")
    if not orders:
        st.caption("Sin ordenes pendientes.")
        return
    for o in orders:
        c = st.columns([2.6, 0.9])
        c[0].write(f"N° {o.broker_id} — {o.side.value} {o.quantity} {o.symbol} @ \\${o.limit_price:,.1f}")
        if c[1].button("Cancelar", key=f"operar_cancel_{o.broker_id}", use_container_width=True):
            with st.spinner("Cancelando..."):
                cancelled = executor.cancel(o)
            if cancelled.status == OrderStatus.CANCELLED:
                st.success(f"Orden {o.broker_id} cancelada.")
            else:
                st.error(cancelled.notes)
            _operar_load_account(executor)


def _operar_history(history: list[dict]) -> None:
    st.markdown("#### Historial (7 dias)")
    if not history:
        st.caption("Sin operaciones recientes.")
        return
    recs = []
    for item in history:
        if not isinstance(item, dict):
            continue
        titulo = item.get("titulo") or {}
        recs.append({
            "Fecha": str(item.get("fechaOperada") or item.get("fechaOrden") or item.get("fechaAlta") or "")[:16],
            "Tipo": item.get("tipo") or "",
            "Simbolo": item.get("simbolo") or titulo.get("simbolo") or "",
            "Cant.": item.get("cantidad") or "",
            "Precio": item.get("precio") or "",
            "Estado": item.get("estado") or item.get("estadoActual") or "",
        })
    if recs:
        st.dataframe(pd.DataFrame(recs), hide_index=True, use_container_width=True)


# ── Tab nueva: Ahora (qué comprar / qué vender) ──────────────────────────────

_MGMT_BADGE: dict[str, tuple[str, str]] = {
    "TAKE_PROFIT":   ("Tomar ganancia", "success"),
    "STOP":          ("Stop",            "error"),
    "TIME_STOP":     ("Time stop",       "warning"),
    "SESSION_CLOSE": ("Cerrar la sesion","warning"),
    "ROLL":          ("Rolar",           "warning"),
    "DEFEND":        ("Defender",        "warning"),
    "HOLD":          ("Mantener",        "info"),
}


def _now_load_iv_map(chain: Optional[OptionsChain], expiry_cal: dict) -> dict[str, float]:
    """Mapa simbolo→IV actual desde la cadena, para reevaluar posiciones abiertas."""
    if chain is None or not chain.options:
        return {}
    try:
        from optionsdesk.core.greeks_engine import compute_chain_greeks
        greeks = compute_chain_greeks(chain, expiry_cal, r=0.0)
        return {sym: g.iv for sym, g in greeks.items() if g.iv}
    except Exception as exc:
        logger.debug("greeks load fallo: %s", exc)
        return {}


def _now_open_positions() -> list:
    """Carga posiciones abiertas desde el JSONL (demos, paper, scalp tickets).

    Usa settings.open_positions_file si existe para evitar depender del CWD.
    """
    try:
        from optionsdesk.signals.monitor import PositionMonitor
        from pathlib import Path as _P
        pos_path = None
        _attr = getattr(settings, "open_positions_file", None)
        if _attr:
            pos_path = _P(_attr)
        monitor = PositionMonitor(positions_file=pos_path) if pos_path else PositionMonitor()
        return monitor.load_positions()
    except Exception as exc:
        logger.debug("load_positions fallo: %s", exc)
        return []


def _now_buy_row(rec, profile: RiskProfile, idx: int, alerter: TelegramAlerter) -> None:
    """Una fila accionable de COMPRAR/ABRIR en lenguaje llano.

    Estructura visual:
    [semáforo + acción + símbolo] [TNA] [Colchón] [Prob] [Score/Grade]
    [Contexto SMC en una línea si está disponible]
    [Explicación breve]
    [Expander: detalle / pasos]
    [Botones: Cargar | Telegram | Paper]
    """
    r = rec.result
    is_call      = r.strategy == "COVERED_CALL"
    accion_verbo = "Lanzar call cubierto" if is_call else "Vender put"
    light_icon   = {"verde": "🟢", "amarillo": "🟡", "rojo": "🔴"}.get(rec.light, "⚪")

    # Score grade
    from optionsdesk.signals.stock_signals import signal_grade as _sg
    _grade_icon = {"S": "🟣", "A": "🟢", "B": "🟡", "C": "🟠", "D": "🔴"}
    grade = _sg(rec.score)

    with st.container(border=True):
        head = st.columns([2.2, 1.2, 1.2, 1.2, 1.2])
        head[0].markdown(f"**{light_icon} {accion_verbo} {r.symbol}**")
        head[1].metric("TNA", f"{r.tna_pct:.1f}%")
        head[2].metric("Colchon", f"{r.cushion_pct:.1f}%")
        head[3].metric("Prob.", f"{rec.success_probability * 100:.0f}%")
        head[4].metric("Score", f"{_grade_icon.get(grade,'')}{grade} {rec.score:.0f}")

        # ── Acción concreta — visible sin clicks ─────────────────────────────────
        if rec.action_steps:
            st.markdown(f"**→ {rec.action_steps[0]}**")

        # ── SMC tags + contexto en una línea ─────────────────────────────────
        _plain = rec.plain_explanation or ""
        _base_plain, *_smc_parts = _plain.split(" | ")
        _base_plain = _base_plain.strip()
        if len(_base_plain) > 160:
            _base_plain = _base_plain[:157].rstrip() + "..."

        smc_tags = [p.strip() for p in _smc_parts if p.strip()]
        _tag_icons = {
            "DISCOUNT": "▲", "PREMIUM": "▼", "ADR confirma": "✓",
            "MANIPUL": "⚠", "CIERRE": "⏰", "CCL": "⚡",
            "dolar": "💵", "equilibrio": "◉",
        }
        if smc_tags:
            rendered_tags = []
            for tag in smc_tags[:3]:
                icon = next((v for k, v in _tag_icons.items() if k.upper() in tag.upper()), "·")
                rendered_tags.append(f"{icon} {tag}")
            st.caption("  |  ".join(rendered_tags))

        st.caption(_base_plain)

        with st.expander("Por qué + escenarios"):
            st.markdown(f"**Si sale bien:** {rec.win_scenario}")
            st.markdown(f"**Si sale mal:** {rec.lose_scenario}")
            if len(rec.action_steps) > 1:
                st.markdown("**Pasos restantes:**")
                for step in rec.action_steps[1:]:
                    st.markdown(f"- {step}")
            st.markdown(f"*{rec.intention}*")
            if rec.warnings:
                for w in rec.warnings:
                    st.warning(w)

        btns = st.columns([1.2, 1.0, 1.0])
        if btns[0].button(
            "Cargar en Operar",
            key=f"now_load_{profile.value}_{idx}",
            use_container_width=True,
            help="Prellena el ticket de Operar con esta opcion.",
            type="primary",
        ):
            side = "Vender"
            _operar_set_ticket(r.symbol, side, float(r.premium), qty=max(rec.contracts, 1))
            st.toast(f"Ticket cargado: {side} {r.symbol}")
        if btns[1].button(
            "Telegram",
            key=f"now_tg_{profile.value}_{idx}",
            use_container_width=True,
        ):
            ok = alerter.send_recommendation(rec)
            st.toast("Enviado" if ok else "Telegram no configurado")
        if btns[2].button(
            "Paper",
            key=f"now_paper_{profile.value}_{idx}",
            disabled=not settings.costs_verified,
            use_container_width=True,
            help="Registra la operacion en el ledger simulado.",
        ):
            _submit_paper_orders(rec)
            st.toast("Registrada en paper")


def _now_directional_row(idea, chain: Optional[OptionsChain]) -> None:
    """Idea direccional (comprar call/put) — especulativa, riesgo definido."""
    side_label = "Comprar call" if idea.idea_type == "BUY_CALL" else "Comprar put"
    is_call    = idea.idea_type == "BUY_CALL"
    dir_icon   = "📈" if is_call else "📉"

    with st.container(border=True):
        head = st.columns([3.0, 0.9, 0.9, 0.9, 0.9])
        head[0].markdown(f"**{dir_icon} {side_label} {idea.symbol}**  ·  especulativo")
        head[1].metric("Strike", f"${idea.strike:,.0f}")
        head[2].metric("R/R", f"{idea.rr_ratio:.1f}:1")
        head[3].metric("Costo", f"${idea.total_cost_ars:,.0f}")
        head[4].metric("Dias", str(idea.days_to_expiry))

        # Línea de trade — lo esencial en 1 line
        st.caption(
            f"Objetivo ${idea.target_spot:,.0f}  ·  "
            f"Stop ${idea.stop_spot:,.0f}  ·  "
            f"BE ${idea.breakeven:,.0f}  ·  "
            f"Perdida max = prima pagada"
        )

        # Confluencias SMC directamente visibles (no en expander)
        if idea.confluence:
            st.caption("Confluencia: " + "  ·  ".join(idea.confluence[:4]))

        with st.expander("Ver racional completo"):
            st.write(idea.rational)
            if len(idea.confluence) > 4:
                st.caption("Más factores: " + "  ·  ".join(idea.confluence[4:]))
            if idea.confluence:
                st.caption("Confluencia: " + " · ".join(idea.confluence))
            st.caption(idea.disclaimer)

        if st.button(
            "Cargar en Operar",
            key=f"now_dir_load_{idea.symbol}",
            help="Comprar la opcion al ask.",
        ):
            q = chain.options.get(idea.symbol) if chain else None
            px = float(q.ask) if q and q.ask > 0 else float(idea.mid_price)
            _operar_set_ticket(idea.symbol, "Comprar", px, qty=1)
            st.toast(f"Ticket cargado: Comprar {idea.symbol}")


def _now_close_row(pos, signal, spot: float, chain: Optional[OptionsChain], idx: int) -> None:
    """Una fila de CERRAR/GESTIONAR para una posicion abierta con señal accionable."""
    badge, fn = _MGMT_BADGE.get(signal.signal_type.value, ("Revisar", "info"))
    color_fn = {"success": st.success, "error": st.error,
                "warning": st.warning, "info": st.info}[fn]

    with st.container(border=True):
        color_fn(f"**{badge}** — {pos.symbol}  ·  captura {signal.capture_pct:+.0f}%")
        st.caption(signal.reason)
        st.caption(signal.suggested_action)
        # Boton: prellena ticket de cierre (comprar de vuelta lo que vendimos).
        if chain is not None and pos.symbol in chain.options:
            q = chain.options[pos.symbol]
            is_long = "LONG" in pos.strategy.upper()
            if is_long:
                side = "Vender"
                # Prioridad: bid > mid > last. `or` es incorrecto cuando mid==0.0.
                px = float(q.bid if q.bid > 0 else (q.mid if q.mid is not None else (q.last or 0.0)))
            else:
                side = "Comprar"
                px = float(q.ask if q.ask > 0 else (q.mid if q.mid is not None else (q.last or 0.0)))
            if st.button(
                f"Cerrar en Operar ({side} {pos.symbol})",
                key=f"now_close_{idx}_{pos.symbol}",
                use_container_width=True,
            ):
                _operar_set_ticket(pos.symbol, side, px, qty=max(pos.contracts, 1))
                st.toast(f"Ticket cargado: {side} {pos.symbol}")


def _render_market_state_bar(context, spot: float) -> None:
    """Barra de estado del mercado — responde en un vistazo a: ¿entro ahora?

    Muestra en una fila: tendencia · zona PDA · timing (killzone) · sweeps · señal sugerida.
    El objetivo es que el operador sepa en <5 segundos si el momento es bueno o no.
    """
    if context is None:
        st.info("Sin datos de contexto — actualizá la cadena.")
        return

    trend    = getattr(context, "trend", "lateral")
    conf     = getattr(context, "confidence", "sin datos")
    smc      = getattr(context, "smc", None)
    mom      = getattr(context, "momentum_pct", 0.0)

    # Zona PDA
    pda_zone = None
    eq_price = None
    if smc is not None:
        pda = getattr(smc, "pda", None)
        if pda is not None:
            pda_zone = getattr(pda, "zone", None)
            eq_price = getattr(pda, "equilibrium", None)

    # Killzone
    killzone = getattr(smc, "killzone", None) if smc is not None else None

    # Sweeps recientes con reversión
    sweep_signal = ""
    if smc is not None:
        sweeps = getattr(smc, "sweeps", [])
        ssl_rev = any(
            getattr(sw, "direction", "") == "down" and getattr(sw, "reversed", False)
            for sw in sweeps
        )
        bsl_rev = any(
            getattr(sw, "direction", "") == "up" and getattr(sw, "reversed", False)
            for sw in sweeps
        )
        if ssl_rev:
            sweep_signal = "Stop-hunt alcista detectado"
        elif bsl_rev:
            sweep_signal = "Stop-hunt bajista detectado"

    # ADR
    adr_conf = getattr(smc, "adr_confluence", None) if smc is not None else None

    # ── SmcSignal — respuesta directa del cascade multi-TF ────────────────────
    smc_sig = getattr(smc, "signal", None) if smc is not None else None

    if smc_sig is not None:
        direction    = getattr(smc_sig, "direction", "neutral")
        quality      = getattr(smc_sig, "quality", "none")
        favors_sp    = getattr(smc_sig, "favors_short_put", False)
        favors_cc    = getattr(smc_sig, "favors_covered_call", False)
        favors_wait  = getattr(smc_sig, "favors_wait", True)
        sig_reason   = getattr(smc_sig, "reason", "")
        sig_target   = getattr(smc_sig, "target", 0.0)
        sig_stop     = getattr(smc_sig, "stop", 0.0)
        target_lbl   = getattr(smc_sig, "target_label", "")
        stop_lbl     = getattr(smc_sig, "stop_label", "")

        _Q_ICON = {"S": "🟣S", "A": "🟢A", "B": "🟡B", "C": "🟠C", "none": "⚪—"}
        q_badge = _Q_ICON.get(quality, "⚪—")

        if favors_sp and not killzone:
            action_txt = f"VENDER PUT GGAL — calidad {q_badge}"
            if sig_target > 0:
                action_txt += f"  ·  target ${sig_target:,.0f} ({target_lbl.split('$')[0].strip()})"
            if sig_stop > 0:
                action_txt += f"  ·  stop ${sig_stop:,.0f}"
            rec_fn = st.success if quality in ("S", "A") else st.info
        elif favors_cc and not killzone:
            action_txt = f"LANZAR CALL CUBIERTO GGAL — calidad {q_badge}"
            if sig_target > 0:
                action_txt += f"  ·  target ${sig_target:,.0f}"
            rec_fn = st.error if quality in ("S", "A") else st.warning
        elif killzone == "manipulacion":
            action_txt = f"KILLZONE APERTURA (11-12h) — esperar desarrollo  ·  {q_badge}"
            rec_fn = st.warning
        elif killzone == "distribucion":
            if favors_sp or direction == "long":
                action_txt = "CIERRE (15:30-17h) — si hay posición abierta, es buen momento para TP"
            else:
                action_txt = "Killzone distribución — movimientos de cierre de ALyCs"
            rec_fn = st.warning
        else:
            action_txt = f"ESPERAR — calidad {q_badge}  ·  sin confluencia suficiente para actuar"
            rec_fn = st.info

        if sig_reason and "killzone" not in sig_reason.lower():
            action_txt += f"  ·  {sig_reason[:80]}"
        if adr_conf == "fx_driven":
            action_txt = "⚠ MOVIDO POR DOLAR (CCL)  ·  " + action_txt
            rec_fn = st.warning

        rec_fn(action_txt)
    else:
        # Fallback sin SmcSignal: señal simple por tendencia y zona PDA
        if trend in ("alcista", "ALCISTA") and pda_zone == "discount":
            rec_fn, rec_signal = st.success, "VENDER PUT — zona discount + tendencia alcista"
        elif trend in ("bajista", "BAJISTA") and pda_zone == "premium":
            rec_fn, rec_signal = st.error, "LANZAR CALL CUBIERTO — zona premium + bajista"
        elif killzone == "manipulacion":
            rec_fn, rec_signal = st.warning, "KILLZONE APERTURA — esperar 12h"
        else:
            rec_fn, rec_signal = st.info, "Sin señal clara — esperar confluencia"
        if sweep_signal:
            rec_signal += f"  ·  {sweep_signal}"
        if adr_conf == "fx_driven":
            rec_signal = "ALERTA CCL  ·  " + rec_signal
            rec_fn = st.warning
        rec_fn(rec_signal)

    # ── Métricas en una fila ────────────────────────────────────────────────────
    cols = st.columns(6)
    _TREND_ICON = {"alcista": "📈", "bajista": "📉", "lateral": "↔"}.get(trend.lower(), "")
    _ZONE_ICON  = {"discount": "▲ DISCOUNT", "premium": "▼ PREMIUM",
                   "equilibrium": "= EQUILIBRIO"}.get(pda_zone or "", "— sin datos")
    _KZ_MAP     = {"manipulacion": "⚠ APERTURA 11-12h", "distribucion": "⏰ CIERRE 15:30-17h"}
    _ADR_MAP    = {"confirmed": "✓ ADR OK", "fx_driven": "⚠ CCL", "unknown": "— ADR"}

    smc_sig_q = "—"
    smc_sig_dir = "—"
    if smc_sig is not None:
        _Q_LABELS = {"S": "🟣S (sniper)", "A": "🟢A (fuerte)", "B": "🟡B (ok)",
                     "C": "🟠C (débil)", "none": "⚪ esperar"}
        smc_sig_q   = _Q_LABELS.get(getattr(smc_sig, "quality", "none"), "—")
        smc_sig_dir = getattr(smc_sig, "direction", "neutral").capitalize()

    cols[0].metric("Tendencia",  f"{_TREND_ICON} {trend.capitalize()} ({conf})")
    cols[1].metric("Zona PDA",   _ZONE_ICON)
    cols[2].metric("Calidad SMC", smc_sig_q)
    cols[3].metric("Momentum",   f"{mom:+.1f}%")
    cols[4].metric("Killzone",   _KZ_MAP.get(killzone or "", "Fuera de KZ"))
    cols[5].metric("ADR/CCL",    _ADR_MAP.get(adr_conf or "", "— sin datos"))

    if eq_price and spot > 0:
        dist_pct = (spot - eq_price) / eq_price * 100.0
        rsi_txt  = ""
        obv_txt  = ""
        if smc_sig is not None:
            rsi_div_v = getattr(smc_sig, "rsi_div", "none")
            if rsi_div_v != "none":
                rsi_txt = f"  ·  RSI div {rsi_div_v}"
            if getattr(smc_sig, "obv_alcista", False):
                obv_txt = "  ·  OBV acumulando"
        st.caption(
            f"Spot ${spot:,.0f}  ·  Equilibrio PDA ${eq_price:,.0f}  "
            f"({'arriba' if dist_pct > 0 else 'abajo'} {abs(dist_pct):.1f}%){rsi_txt}{obv_txt}"
        )
    st.divider()


def _tab_now(
    *,
    recs: dict[RiskProfile, list[Recommendation]],
    chain: Optional[OptionsChain],
    spot: float,
    context,
    caucion_tna: float,
    alerter: TelegramAlerter,
    adaptive_context=None,
) -> None:
    """Tablero de accion: que comprar ahora y que cerrar ahora.

    No es analisis, es decision. Cada fila lleva un boton para precargar la
    operacion en la tab Operar.
    """
    # ── Barra de market state: qué dice el mercado AHORA ────────────────────────
    _render_market_state_bar(context, spot)

    sel_profile = st.radio(
        "Perfil",
        list(RiskProfile),
        format_func=lambda p: _PROFILE_META[p]["label"],
        index=1,
        horizontal=True,
        key="now_profile",
    )
    st.caption(_PROFILE_META[sel_profile]["desc"])

    col_buy, col_close = st.columns(2, gap="large")

    # ── COMPRAR / ABRIR ────────────────────────────────────────────────
    with col_buy:
        st.markdown("### Comprar / abrir")
        profile_recs = recs.get(sel_profile, [])

        if not profile_recs:
            st.info("Sin oportunidades para este perfil con los datos actuales.")
        else:
            for i, rec in enumerate(profile_recs[:3]):
                _now_buy_row(rec, sel_profile, i, alerter)

        # Idea direccional (solo si el contexto es claro).
        if context is not None and chain is not None:
            try:
                daily_snap = getattr(context, "snap", None)
                idea = build_directional_idea(context, chain, spot, snap=daily_snap)
                if idea is not None:
                    st.markdown("#### Idea direccional")
                    _now_directional_row(idea, chain)
            except Exception as exc:
                logger.debug("directional idea fallo: %s", exc)

    # ── CERRAR / GESTIONAR ─────────────────────────────────────────────
    # Si la version de Streamlit soporta st.fragment, lo envolvemos para que
    # se refresque solo cada 30s (releer JSONL + recalcular senales) sin
    # parpadear el resto del tablero. Caemos a llamada directa si no esta.
    def _render_close_panel() -> None:
        # Prevenir NameError en contexto aislado si variables no persisten
        current_chain = st.session_state.get("chain") or chain
        current_spot = current_chain.spot.mid if current_chain and getattr(current_chain, "spot", None) else spot

        st.markdown("### Cerrar / gestionar")
        positions = _now_open_positions()

        if not positions:
            st.info("Sin posiciones abiertas. Cuando el bot o el ticket guarden "
                    "una posicion, aparece aca con su senal de gestion.")
            return

        from optionsdesk.signals.management import evaluate_position, SignalType
        # _load_expiry_calendar() es @st.cache_resource en este mismo archivo.
        iv_map = _now_load_iv_map(
            current_chain,
            _effective_expiry_calendar(current_chain, _load_expiry_calendar()),
        )
        actionable = []
        holding = []
        for pos in positions:
            iv_now = iv_map.get(pos.symbol)
            try:
                sig = evaluate_position(pos, current_spot, iv_now)
            except Exception as exc:
                logger.debug("evaluate fallo %s: %s", pos.symbol, exc)
                continue
            if sig.signal_type != SignalType.HOLD:
                actionable.append((pos, sig))
            else:
                holding.append((pos, sig))

        if not actionable:
            st.success("Mantener. Ninguna posicion pide accion ahora.")
        else:
            for i, (pos, sig) in enumerate(actionable):
                _now_close_row(pos, sig, current_spot, current_chain, i)

        if holding:
            with st.expander(f"Posiciones en HOLD ({len(holding)})"):
                for pos, sig in holding:
                    st.caption(
                        f"{pos.symbol}  ·  captura {sig.capture_pct:+.0f}%  ·  "
                        f"{pos.days_remaining}d restantes"
                    )

    with col_close:
        _frag_dec = getattr(st, "fragment", None)
        if _frag_dec is not None:
            try:
                # FIX: Llamamos al wrapper que devuelve el decorador (evita bug de Streamlit)
                wrapper = _frag_dec(run_every=30)(_render_close_panel)
                wrapper()
            except TypeError:
                # Streamlit antiguo (sin run_every) — fragment sin auto-refresh.
                wrapper = _frag_dec(_render_close_panel)
                wrapper()
        else:
            _render_close_panel()


# ── Tab nueva: Posiciones (vista unificada) ──────────────────────────────────

_IOL_POSITIONS_TTL_S = 300  # 5 minutos entre fetches automaticos


def _positions_iol_rows() -> list[dict]:
    """Posiciones de opciones reales en la cuenta IOL (sin gestion CRR).

    Solo llama a _operar_load_account si el cache esta vacio O tiene mas de
    _IOL_POSITIONS_TTL_S segundos. Evita los 3 API calls en cada rerun de Streamlit.
    El TTL se mide con time.time() (epoch float) para evitar el bug de medianoche
    que producía un timedelta negativo al parsear solo la hora sin fecha.
    """
    import time as _time
    if not settings.is_iol_configured():
        return []
    try:
        if "iol_executor" not in st.session_state:
            from optionsdesk.execution.iol_executor import IOLExecutor
            st.session_state["iol_executor"] = IOLExecutor.from_settings()
        executor = st.session_state["iol_executor"]

        fetched_epoch = st.session_state.get("operar_acct_epoch", 0.0)
        cache_stale = (_time.time() - fetched_epoch) > _IOL_POSITIONS_TTL_S

        if cache_stale:
            _operar_load_account(executor)
            st.session_state["operar_acct_epoch"] = _time.time()

        acct = st.session_state.get("operar_acct", {}) or {}
        return [p for p in acct.get("positions", []) if isinstance(p, dict)]
    except Exception as exc:
        logger.debug("IOL positions fetch fallo: %s", exc)
        return []


# ── Analizador intertemporal de posiciones ────────────────────────────────────

_ACTION_COLOR = {
    "CLOSE_NOW": "error",
    "ROLL":      "warning",
    "DEFEND":    "warning",
    "ROTATE":    "warning",
    "PARTIAL":   "info",
    "HOLD":      "success",
}
_ACTION_ICON = {
    "CLOSE_NOW": "🔴",
    "ROLL":      "🔄",
    "DEFEND":    "🛡",
    "ROTATE":    "🔃",
    "PARTIAL":   "🟡",
    "HOLD":      "🟢",
}


def _render_advice_card(adv, chain: Optional[OptionsChain]) -> None:
    """Tarjeta de consejo intertemporal para una posicion abierta."""
    icon = _ACTION_ICON.get(adv.action, "⚪")
    color_fn = {"error": st.error, "warning": st.warning,
                "success": st.success, "info": st.info}[
        _ACTION_COLOR.get(adv.action, "info")
    ]
    with st.container(border=True):
        # Fila de métricas
        h = st.columns([3.5, 1, 1, 1, 1])
        h[0].markdown(
            f"**{icon} {adv.position.symbol}** — "
            f"{adv.position.strategy.replace('_', ' ').title()}"
        )
        h[1].metric("Captura", f"{adv.capture_pct:.0f}%")
        h[2].metric("DTE", str(adv.dte))
        h[3].metric("Delta", f"{adv.current_delta:.2f}" if adv.current_delta is not None else "—")
        urgency_badge = {"alta": "🔴 Alta", "media": "🟡 Media", "baja": "🟢 Baja"}
        h[4].metric("Urgencia", urgency_badge.get(adv.urgency, adv.urgency))

        # Headline accionable
        color_fn(adv.headline)
        st.caption(adv.reason)

        # Flags de riesgo
        for flag in adv.risk_flags:
            st.caption(f"Riesgo: {flag}")

        # Trayectoria temporal
        if adv.scenarios:
            with st.expander("Trayectoria (theta decay — spot congelado)"):
                sc_rows = []
                for sc in adv.scenarios:
                    sc_rows.append({
                        "En": f"+{sc.days_from_now}d",
                        "DTE rest.": sc.dte_remaining,
                        "Captura": f"{sc.capture_pct:.0f}%",
                        "P&L ($)": f"${sc.pnl_ars:,.0f}",
                        "Theta/dia": f"${sc.theta_daily_ars:,.0f}",
                    })
                st.dataframe(
                    pd.DataFrame(sc_rows), hide_index=True, use_container_width=True
                )
                if adv.days_to_target is not None and adv.days_to_target > 0:
                    st.caption(
                        f"Alcanza objetivo ({adv.position.target_capture_pct:.0f}%) "
                        f"en ~{adv.days_to_target} dias si el spot no se mueve."
                    )
                elif adv.days_to_target == 0:
                    st.caption("Objetivo ya alcanzado.")
                else:
                    st.caption(
                        "La theta sola no alcanza el objetivo en los proximos 21 dias."
                    )
                lag_label = (
                    f"Captura {abs(adv.capture_lag_pct):.0f}pp "
                    f"{'adelantada' if adv.capture_lag_pct > 0 else 'rezagada'} "
                    f"respecto al ritmo esperado del plan."
                )
                (st.success if adv.plan_on_track else st.warning)(lag_label)

        # Rotacion
        if adv.rotate_to is not None:
            with st.expander(f"Oportunidad de rotacion → {adv.rotate_to}"):
                st.info(adv.rotation_rationale or "")
                if st.button(
                    f"Cargar cierre {adv.position.symbol} en Operar",
                    key=f"adv_rotate_{adv.position.symbol}",
                ):
                    q = chain.options.get(adv.position.symbol) if chain else None
                    is_long = "LONG" in adv.position.strategy.upper()
                    side = "Vender" if is_long else "Comprar"
                    px = float(
                        (q.bid if is_long else q.ask) if q and (q.bid if is_long else q.ask) > 0
                        else (q.mid if q and q.mid else 0.0)
                    )
                    _operar_set_ticket(adv.position.symbol, side, px, qty=max(adv.position.contracts, 1))
                    st.toast(f"Ticket de cierre: {side} {adv.position.symbol}")

        # Botón de acción rápida
        if adv.action in ("CLOSE_NOW", "ROLL", "DEFEND"):
            q = chain.options.get(adv.position.symbol) if chain else None
            is_long = "LONG" in adv.position.strategy.upper()
            side = "Vender" if is_long else "Comprar"
            px = float(
                (q.bid if is_long else q.ask) if q and (q.bid if is_long else q.ask) > 0
                else (q.mid if q and q.mid else 0.0)
            )
            lbl = f"Cargar {'cierre' if adv.action == 'CLOSE_NOW' else adv.action.lower()} en Operar"
            if st.button(lbl, key=f"adv_btn_{adv.position.symbol}",
                         use_container_width=True, type="primary"):
                _operar_set_ticket(adv.position.symbol, side, px, qty=max(adv.position.contracts, 1))
                st.toast(f"Ticket: {side} {adv.position.symbol}")


def _render_advisor_panel(
    *,
    positions: list,
    spot: float,
    chain: Optional[OptionsChain],
    context,
    caucion_tna: float,
    recs: dict,
) -> None:
    """Panel principal del analizador intertemporal."""
    from optionsdesk.signals.position_advisor import advise_portfolio

    if not positions:
        st.info(
            "Sin posiciones en el ledger (data/open_positions.jsonl). "
            "Cuando el paper executor o el bot registren una posicion, "
            "el analizador muestra aqui que hacer con ella."
        )
        return

    all_recs = [r for rs in recs.values() for r in rs] if recs else []
    iv_map = _now_load_iv_map(chain, _effective_expiry_calendar(chain, _load_expiry_calendar()))

    try:
        portfolio = advise_portfolio(
            positions,
            current_spot=spot,
            current_iv_map=iv_map,
            context=context,
            available_recs=all_recs or None,
            caucion_tna=caucion_tna,
        )
    except Exception as exc:
        st.warning(f"Error en el analizador: {exc}")
        logger.exception("advise_portfolio fallo")
        return

    # Resumen del libro
    urgency_fn = {
        "alta":    st.error,
        "media":   st.warning,
        "baja":    st.success,
        "ninguna": st.info,
    }.get(portfolio.highest_urgency, st.info)
    urgency_fn(
        f"Urgencia maxima: **{portfolio.highest_urgency.upper()}**  ·  "
        f"P&L capturado: **${portfolio.total_capture_ars:,.0f}**  ·  "
        f"Theta diario del libro: **${portfolio.theta_daily_ars:,.0f}**"
    )
    for flag in portfolio.portfolio_flags:
        st.caption(flag)

    # Tarjeta por posicion, de más urgente a menos
    for adv in portfolio.advices:
        _render_advice_card(adv, chain=chain)


def _tab_positions(
    *,
    provider,
    chain: Optional[OptionsChain],
    spot: float,
    caucion_tna: float,
    cc_filtered=None,
    sp_filtered=None,
    now=None,
    adaptive_context=None,
    recs: Optional[dict] = None,
    context=None,
) -> None:
    """Posiciones: Analizador intertemporal · Cartera IOL · Demo."""

    # Demo tick (no bloquea el render)
    try:
        from optionsdesk.backtest.options_demo import run_options_demo_tick
        if chain is not None:
            cands = []
            if cc_filtered: cands.extend(cc_filtered)
            if sp_filtered: cands.extend(sp_filtered)
            run_options_demo_tick(
                chain=chain, candidates=cands, now=now,
                adaptive_context=adaptive_context, caucion_tna_pct=caucion_tna,
            )
    except Exception as exc:
        logger.debug("options demo tick fallo: %s", exc)

    bot_positions = _now_open_positions()
    iol_rows = _positions_iol_rows()
    
    if iol_rows:
        from optionsdesk.backtest.stock_demo import StockDemoPosition
        from datetime import datetime
        for p in iol_rows:
            qty = p.get("cantidad", 0)
            if qty <= 0: continue
            sym = p.get("simbolo", "")
            ppc = p.get("ppc", 0.0)
            if not any(bp.symbol == sym for bp in bot_positions):
                bot_positions.append(StockDemoPosition(
                    id=f"IOL-{sym}",
                    symbol=sym,
                    strategy="REAL",
                    signal_type="Sincronizado",
                    entry_ts=datetime.now(),
                    entry_price=ppc,
                    quantity=int(qty),
                    stop_price=ppc * 0.95,
                    target_price=ppc * 1.10,
                    entry_fee_ars=0.0,
                    time_stop_min=0,
                    max_holding_days=30,
                    rationale="Posición importada de IOL",
                    score=0.0
                ))

    sub_adv, sub_iol, sub_demo = st.tabs([
        f"Analizador ({len(bot_positions)})", "Cartera IOL", "Demo",
    ])

    with sub_adv:
        _render_advisor_panel(
            positions=bot_positions, spot=spot, chain=chain,
            context=context, caucion_tna=caucion_tna, recs=recs or {},
        )

    with sub_iol:
        if iol_rows:
            st.caption("Cuenta IOL — Sincronizada con el broker")
            
            total_invertido = 0.0
            total_valorizado = 0.0
            
            view = []
            for p in iol_rows:
                qty = p.get("cantidad", 0)
                ppc = p.get("ppc", 0.0)
                ult = p.get("ultimo", 0.0)
                val = p.get("valorizado", 0.0)
                if val == 0 and qty > 0:
                    val = qty * ult
                    
                invertido = qty * ppc
                pnl_pesos = val - invertido
                
                total_invertido += invertido
                total_valorizado += val
                
                view.append({
                    "Símbolo": p.get("simbolo", ""),
                    "Cantidad": int(qty),
                    "PPC ($)": f"{ppc:,.2f}",
                    "Último ($)": f"{ult:,.2f}",
                    "Invertido ($)": f"{invertido:,.2f}",
                    "Valorizado ($)": f"{val:,.2f}",
                    "PnL ($)": f"{pnl_pesos:+,.2f}",
                    "Ganancia %": f"{p.get('ganancia_pct', 0.0):+.2f}%",
                })
            
            total_pnl = total_valorizado - total_invertido
            total_pnl_pct = (total_pnl / total_invertido * 100.0) if total_invertido > 0 else 0.0
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Capital Invertido", f"${total_invertido:,.0f}")
            c2.metric("Total Valorizado", f"${total_valorizado:,.0f}", f"{total_pnl_pct:+.2f}%")
            c3.metric("PnL Abierto", f"${total_pnl:+,.0f}")
            c4.metric("Posiciones Activas", len(view))
            
            st.dataframe(pd.DataFrame(view), hide_index=True, use_container_width=True)
            st.caption("Las posiciones de IOL se inyectan automáticamente en el Analizador.")
        else:
            st.caption("Sin posiciones en IOL o broker no configurado.")

    with sub_demo:
        try:
            from optionsdesk.backtest.stock_demo import load_stock_demo_trades
            from pathlib import Path as _P
            if _P(settings.stock_demo_trades_file).exists():
                closed = load_stock_demo_trades(_P(settings.stock_demo_trades_file), limit=15)
                if closed:
                    rows = [
                        {
                            "Simbolo": getattr(t, "symbol", "-"),
                            "Estrategia": getattr(t, "strategy", "-"),
                            "PnL $": getattr(t, "pnl_ars", 0.0),
                            "Motivo": getattr(t, "exit_reason", "-"),
                        }
                        for t in closed
                    ]
                    st.caption("Ultimos trades cerrados del demo de acciones")
                    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
                else:
                    st.caption("Sin trades cerrados en el demo todavia.")
            else:
                st.caption("Demo sin archivo de trades.")
        except Exception as exc:
            logger.debug("stock demo historial fallo: %s", exc)


# ── Tab nueva: Mercado (scanner + analisis tecnico) ──────────────────────────

def _tab_market(
    *,
    provider,
    provider_health,
    spot_history,
    chain: Optional[OptionsChain],
    spot: float,
    context,
    htf_df=None,
    ltf_df=None,
    expiry_cal=None,
    benchmark=None,
    adaptive_context=None,
) -> None:
    """Mercado en vivo: scanner de acciones + analisis tecnico GGAL.

    Reutiliza _tab_stocks (scanner) y _tab_directional (analisis tecnico)
    organizados en sub-tabs para que el usuario elija enfoque.
    """
    sub_acc, sub_at = st.tabs(["Acciones (scanner)", "GGAL (analisis tecnico)"])

    with sub_acc:
        _tab_stocks(provider, provider_health, adaptive_context=adaptive_context)

    with sub_at:
        _tab_directional(
            spot_history=spot_history,
            chain=chain,
            spot=spot,
            context=context,
            htf_df=htf_df,
            ltf_df=ltf_df,
            expiry_calendar=expiry_cal,
            benchmark=benchmark,
        )


# ── Footer manual ─────────────────────────────────────────────────────────────

def _data_footer() -> None:
    """Pie estable: no fuerza reruns ni parpadeos."""
    last_fetch = st.session_state.get("last_fetch_label", "sin cargar")
    st.caption(f"Datos cargados: {last_fetch}")


# ── App principal ─────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="OptionsDesk",
        layout="wide",
        page_icon=":chart_with_upwards_trend:",
    )
    _inject_css()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("OptionsDesk")

        if st.button("Actualizar datos", key="btn_refresh_sidebar", use_container_width=True):
            st.session_state["force_refresh"] = True
            st.rerun()
        scalp_profile = st.selectbox(
            "Perfil scalping",
            ["balanced", "conservative", "aggressive"],
            index=0,
            help="balanced es el default para operar manualmente con dinero real.",
        )

        capital_value = float(st.session_state.get("scalp_capital", settings.default_capital or 0) or 0)
        capital = capital_value if capital_value > 0 else None

        directional_on = st.toggle(
            "Contexto tecnico multi-TF",
            value=True,
            help="Ajuste leve basado en tendencia. Mejora con historial acumulado.",
        )

        advanced_mode = st.toggle(
            "Modo avanzado",
            value=False,
            help="Suma tabs analiticas: Oportunidades (tabla CC/SP completa), "
                 "Simulador P&L y Edge realizado por estrategia.",
        )

        # Filtros: solo visibles en modo avanzado
        min_spread = float(settings.min_tna_spread_pct)
        price_mode = "bid"
        capital_mode = "gross"
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
                "Precio opcion", ["bid", "mid", "ask", "last"],
                help="mid = analisis; bid = mas realista al vender",
            )
            capital_mode = st.selectbox(
                "Capital puts", ["gross", "net"],
                help="net = K-prima; gross = K (mas conservador)",
            )
            show_all = st.checkbox("Mostrar toda la cadena sin filtro", False)
            st.divider()
            send_alerts = st.checkbox("Enviar alertas Telegram", False)

        st.divider()
        with st.expander("Que significan estos terminos"):
            st.markdown(
                "- **TNA**: tasa anualizada de la operacion. Sirve para comparar "
                "contra la caucion (renta sin riesgo).\n"
                "- **Colchon**: cuanto puede caer GGAL antes de empezar a perder.\n"
                "- **Captura**: que porcentaje de la prima ya cobraste vs el total "
                "que pagaria al vencimiento si todo sale bien.\n"
                "- **Delta**: probabilidad aproximada de que la opcion termine ITM. "
                "Tambien indica cuanto se mueve la prima por cada $1 que se mueve GGAL.\n"
                "- **Prima**: lo que cobras (vendiendo) o pagas (comprando) por el contrato.\n"
                "- **Caucion**: tasa testigo overnight del mercado argentino. Es tu "
                "alternativa libre de riesgo.\n"
                "- **VRP / Vol edge**: diferencia entre la IV (lo que estima el mercado) "
                "y la vol realizada (lo que efectivamente paso). Positivo = vender prima paga.\n"
                "- **DTE**: dias al vencimiento."
            )
        with st.expander("Guia (referencia completa)"):
            _gp = Path("GUIA.md")
            if not _gp.exists():
                _gp = Path(__file__).parent.parent.parent / "GUIA.md"
            if _gp.exists():
                st.markdown(_gp.read_text(encoding="utf-8"))
            else:
                st.caption("No se encontro GUIA.md en la raiz.")
        st.caption("BYMA - opciones y acciones argentinas")

    # ── Providers y scanners ──────────────────────────────────────────────────
    try:
        provider = _build_provider()
    except Exception as exc:
        st.error(f"No se pudo iniciar una fuente de mercado real: {exc}")
        return
    expiry_cal = _load_expiry_calendar()

    screener = Screener(ScreenerConfig(min_tna_spread_pct=min_spread))
    alerter = TelegramAlerter()

    # ── Datos: carga inicial + refresh manual, sin parpadeo por auto-rerun ─────
    force_refresh = bool(st.session_state.pop("force_refresh", False))
    provider_key = f"live:{provider.__class__.__name__}"
    if st.session_state.get("provider_key") != provider_key:
        st.session_state["provider_key"] = provider_key
        force_refresh = True
    if force_refresh or "chain" not in st.session_state or st.session_state.chain is None:
        with st.spinner("Cargando datos de mercado..."):
            st.session_state.chain = provider.get_options_chain()
            st.session_state.caucion_tna = provider.get_caucion_tna() or settings.default_caucion_tna
            st.session_state.last_fetch_label = pd.Timestamp.now().strftime("%H:%M:%S")

    chain: Optional[OptionsChain] = st.session_state.chain
    caucion_tna: float = st.session_state.caucion_tna
    provider_health = provider.get_health()

    if chain is None:
        spot_quote = provider.get_quote("GGAL")
        if spot_quote is None:
            st.error("Sin datos de mercado reales. Verifica la conexion y actualiza nuevamente.")
            return
        chain = OptionsChain(underlying="GGAL", spot=spot_quote, options={})
        st.session_state.chain = chain
        st.warning(
            "Sin cadena fresca de opciones. El panel de Acciones sigue disponible; "
            "Scalping GGAL esperara una cadena valida."
        )
    if not _is_realtime_feed(provider_health):
        st.warning(
            "El feed actual no es realtime. Se muestra solo como referencia; "
            "las señales directas de scalping quedan bloqueadas."
        )
    if not settings.costs_verified:
        st.warning(
            "Perfil de costos del ALyC sin verificar. Se usan costos conservadores y "
            "los tickets directos de scalping quedan bloqueados."
        )
    else:
        st.caption(f"Perfil de costos activo: {settings.costs_profile}")
    expiry_cal = _effective_expiry_calendar(chain, expiry_cal)
    cc_scanner = CoveredCallScanner(CoveredCallConfig(price_mode=price_mode), expiry_cal)
    sp_scanner = ShortPutScanner(
        ShortPutConfig(price_mode=price_mode, capital_mode=capital_mode), expiry_cal
    )

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
    spot_history_df = _load_spot_history(days=180, allow_synthetic=False)
    htf_df          = _load_htf_history(weeks=52, allow_synthetic=False)
    ltf_df          = _load_ltf_history()

    spot_history_list: Optional[list[float]] = (
        spot_history_df["close"].dropna().tolist()
        if spot_history_df is not None and not spot_history_df.empty
        else None
    )

    # ── Contexto de mercado (opcional) ────────────────────────────────────────
    from optionsdesk.signals.directional import compute_market_context
    ctx_data = spot_history_df if spot_history_df is not None else None
    context = compute_market_context(ctx_data, htf=htf_df, ltf=ltf_df, symbol="GGAL")
    recommender_context = context if directional_on else None

    # ── Recomendaciones ───────────────────────────────────────────────────────
    recommender = Recommender()
    recs = recommender.recommend(
        cc_all, sp_all, benchmark, context=recommender_context, capital=capital,
        spot_history=spot_history_list, top_n=3,
    )

    # Persistir la IV ATM de hoy (una por día) para alimentar iv_rank a futuro.
    try:
        from optionsdesk.data.iv_history import record_daily_iv, representative_atm_iv
        record_daily_iv(representative_atm_iv(list(cc_all) + list(sp_all)), symbol="GGAL")
    except Exception:
        pass

    # `now` se usa abajo en los demos de opciones/acciones para calcular DTE y
    # frescura de cotizaciones. Se define una sola vez por render.
    now = datetime.now()
    st.session_state["now"] = now

    # ── Header ───────────────────────────────────────────────────────────────
    st.title("OptionsDesk")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("GGAL", f"${spot:,.0f}")
    c2.metric("Caucion TNA", f"{caucion_tna:.1f}%" if caucion_tna > 0 else "—")
    c3.metric("Opciones en cadena", len(chain.options))
    c4.metric("Feed", provider_health.source)
    c5.metric("Hora", pd.Timestamp.now().strftime("%H:%M:%S"), delta_color="off")

    if context is not None:
        _note_parts = [context.note]
        _smc_hdr = getattr(context, "smc", None)
        if _smc_hdr is not None:
            _pda_hdr = getattr(_smc_hdr, "pda", None)
            if _pda_hdr is not None:
                _zone_hdr = getattr(_pda_hdr, "zone", None)
                _zone_map = {"premium": "PREMIUM", "discount": "DISCOUNT", "equilibrium": "EQUILIBRIO"}
                if _zone_hdr:
                    _note_parts.append(f"Zona PDA: {_zone_map.get(_zone_hdr, _zone_hdr)}")
            _kz_hdr = getattr(_smc_hdr, "killzone", None)
            if _kz_hdr:
                _note_parts.append(f"Killzone: {_kz_hdr.upper()}")
            _adr_hdr = getattr(_smc_hdr, "adr_confluence", None)
            if _adr_hdr and _adr_hdr != "unknown":
                _note_parts.append("ADR confirma" if _adr_hdr == "confirmed" else "Movido por dólar (CCL)")
        st.caption("  |  ".join(_note_parts))

    st.divider()

    # Unified Adaptive Context — cacheado en session_state 60s para no leer
    # dos archivos de disco + analyze_performance() en cada rerun por widget.
    import time as _time_mod
    import json
    _ADAPTIVE_TTL = 60
    _adaptive_epoch = st.session_state.get("adaptive_context_epoch", 0.0)
    if (_time_mod.time() - _adaptive_epoch) > _ADAPTIVE_TTL or "adaptive_context" not in st.session_state:
        from optionsdesk.backtest.stock_demo import load_stock_demo_trades
        from optionsdesk.backtest.adaptive import analyze_performance
        from types import SimpleNamespace as _SN

        unified_trades = (
            load_stock_demo_trades(Path(settings.stock_demo_trades_file), limit=50)
            if Path(settings.stock_demo_trades_file).exists()
            else []
        )
        options_file = Path("data/options_demo_trades.jsonl")
        if options_file.exists():
            try:
                with options_file.open("r", encoding="utf-8") as fh:
                    for line in fh:
                        if line.strip():
                            data = json.loads(line)
                            if "pnl_ars" in data:
                                # SimpleNamespace evita definir una clase de un solo uso
                                unified_trades.append(_SN(pnl_ars=data["pnl_ars"]))
            except Exception:
                pass
        st.session_state["adaptive_context"] = analyze_performance(unified_trades) if unified_trades else None
        st.session_state["adaptive_context_epoch"] = _time_mod.time()

    adaptive_context = st.session_state["adaptive_context"]

    # ── Tabs ──────────────────────────────────────────────────────────────────
    # Operativas (siempre visibles): Ahora (qué comprar / qué vender), Operar
    # (cockpit IOL), Posiciones (todo lo abierto + gestión), Mercado (scanner
    # + análisis técnico). Avanzadas detrás del toggle.
    if advanced_mode:
        tab_labels = [
            "Ahora", "Operar", "Posiciones", "Mercado",
            "Oportunidades", "Simulador P&L", "Edge", "Genética",
        ]
        (t_now, t_operar, t_pos, t_market,
         t_ops, t_sim, t_edge, t_genetic) = st.tabs(tab_labels)
    else:
        (t_now, t_operar, t_pos, t_market, t_genetic) = st.tabs(
            ["Ahora", "Operar", "Posiciones", "Mercado", "Genética"]
        )

    with t_now:
        _tab_now(
            recs=recs,
            chain=chain,
            spot=spot,
            context=context,
            caucion_tna=caucion_tna,
            alerter=alerter,
            adaptive_context=adaptive_context,
        )

    with t_operar:
        _tab_operar(provider, chain)

    with t_pos:
        _tab_positions(
            provider=provider,
            chain=chain,
            spot=spot,
            caucion_tna=caucion_tna,
            cc_filtered=cc_filtered,
            sp_filtered=sp_filtered,
            now=now,
            adaptive_context=adaptive_context,
            recs=recs,
            context=context,
        )

    with t_market:
        _tab_market(
            provider=provider,
            provider_health=provider_health,
            spot_history=spot_history_df,
            chain=chain,
            spot=spot,
            context=context,
            htf_df=htf_df,
            ltf_df=ltf_df,
            expiry_cal=expiry_cal,
            benchmark=benchmark,
            adaptive_context=adaptive_context,
        )

    with t_genetic:
        import streamlit as st
        st.subheader("🧬 Laboratorio Genético")
        st.caption("Bots autónomos operando en Forward-Testing continuo con mutaciones.")
        if "genetic_engine" in st.session_state:
            engine = st.session_state.genetic_engine
            
            view_mode = st.radio("Vista", ["🏆 Clasificación Activa", "🌳 Árbol Genealógico"], horizontal=True)
            
            if view_mode == "🏆 Clasificación Activa":
                c1, c2 = st.columns([3, 1])
                with c1:
                    bots = sorted(engine.bots, key=lambda b: b.pnl_pct(), reverse=True)
                    view = []
                    for b in bots:
                        view.append({
                            "Bot ID": b.id,
                            "Gen": b.generation,
                            "PnL %": f"{b.pnl_pct():.2f}%",
                            "Capital": f"${b.capital:,.0f}",
                            "Abiertas": len(b.open_positions),
                            "Cerradas": len(b.closed_trades),
                            "Stop (ATR)": b.genome.atr_stop_mult,
                            "Min RR": b.genome.min_rr,
                            "Req Bull": "Sí" if b.genome.require_merval_bull else "No",
                            "Req CCL": "Sí" if b.genome.require_ccl_stable else "No"
                        })
                    import pandas as pd
                    st.dataframe(pd.DataFrame(view), hide_index=True, use_container_width=True)
                    
                with c2:
                    st.markdown("### Panel de Control")
                    st.metric("Población Activa", len(engine.bots))
                    
                    if st.button("✂️ Forzar Poda y Evolución"):
                        engine.prune_and_breed()
                        st.success("¡Nueva generación engendrada!")
                        st.rerun()

            else:
                st.markdown("### Ramificaciones y Descartes")
                st.caption("Verde = Sobrevivientes | Rojo = Descartados por mal rendimiento")
                
                # Build Graphviz DOT string
                graph = "digraph Lineage {\\n"
                graph += '  rankdir=TB;\\n'
                graph += '  node [shape=box, style=filled, fontname="sans-serif", fontsize=10];\\n'
                graph += '  edge [color="#666666", arrowhead=open];\\n'
                
                for b in engine.historical_bots:
                    color = '"#A8E6CF"' if b.status == "alive" else '"#FF8B94"'
                    label = f"Gen {b.generation}\\\\nBot: {b.id}\\\\nATR: {b.genome.atr_stop_mult}x | RR: {b.genome.min_rr}"
                    if b.status == "dead":
                        label += f"\\\\n(Muerto en G{b.death_generation})"
                    else:
                        label += f"\\\\nPnL: {b.pnl_pct():.1f}%"
                        
                    graph += f'  "{b.id}" [label="{label}", fillcolor={color}];\\n'
                    if getattr(b, "parent_id", None):
                        graph += f'  "{b.parent_id}" -> "{b.id}";\\n'
                        
                graph += "}\\n"
                st.graphviz_chart(graph)
                
        else:
            st.info("El motor genético aún no ha arrancado. Espera el próximo tick del scanner.")

    if advanced_mode:
        with t_ops:
            _tab_opportunities(
                cc_all, sp_all, cc_filtered, sp_filtered,
                show_all, caucion_tna, send_alerts, alerter,
            )
        with t_sim:
            _tab_simulator(cc_all, sp_all, spot)
        with t_edge:
            _tab_edge(spot)

    # ── Footer manual ─────────────────────────────────────────────────────────
    _data_footer()


def _tab_edge(spot: Optional[float] = None) -> None:
    """Edge realizado por estrategia + riesgo de gap del libro real.

    Mide lo que cada estrategia REALMENTE dejó (renta, momentum de opciones, scalp,
    SMC de acciones) sin tocar la lógica de entrada de ninguna, y stresa el libro
    abierto ante un gap de GGAL. Capital donde hay edge, cortar la basura, y ver la
    pérdida de cola que el delta/theta del día esconden.
    """
    from optionsdesk.performance.attribution import (
        Verdict, attribute, load_all_closed_trades,
    )

    # Estado del runner continuo (si está corriendo en background).
    try:
        from optionsdesk.backtest.demo_runner import load_runner_status
        rs = load_runner_status()
    except Exception:
        rs = None
    if rs:
        msg = (
            f"Runner: último tick {rs.get('last_tick', '?')} · modo {rs.get('mode', '?')} · "
            f"PnL ${rs.get('realized_pnl_ars', 0):,.0f} · drawdown {rs.get('drawdown_pct', 0):.1f}% · "
            f"½Kelly {rs.get('half_kelly_pct', 0) * 100:.1f}%"
        )
        (st.warning if rs.get("halted") else st.info)(
            msg + (" · FRENO por drawdown" if rs.get("halted") else "")
        )
        if rs.get("blocked"):
            st.caption("Setups bloqueados (sin edge realizado): " + ", ".join(rs["blocked"]))
        if rs.get("regime"):
            st.caption(f"Estrategia desplegada por contexto: {rs['regime']}")
        extra = []
        if rs.get("market_data_source"):
            extra.append(f"feed {rs['market_data_source']}")
        if rs.get("policy_status"):
            extra.append(f"politica {rs['policy_status']}")
        if rs.get("universe_source"):
            extra.append(f"universo {rs['universe_source']}")
        if extra:
            st.caption(" | ".join(extra))
        if rs.get("stock_universe"):
            st.caption("Top universo: " + ", ".join(rs["stock_universe"]))
        if rs.get("fetch_error"):
            st.warning(f"Fetch runner: {rs['fetch_error']}")

    _render_strategy_tree()

    _render_param_learner()

    _render_gap_stress(spot)

    st.subheader("Edge realizado por estrategia")
    st.caption(
        "El número que importa: lo que cada estrategia dejó DESPUÉS de costos, no lo "
        "que prometía. Cortá la basura, escalá lo que calibra."
    )
    all_trades = load_all_closed_trades()   # una sola lectura; se reutiliza abajo
    edges = attribute(all_trades)
    if not edges:
        st.info(
            "Sin trades cerrados todavía. A medida que se cierran posiciones "
            "(renta, momentum, scalp, acciones) aparece acá el edge realizado."
        )
        return

    badge = {
        Verdict.EDGE: "🟢 Edge confirmado",
        Verdict.MARGINAL: "🟡 Marginal",
        Verdict.NO_EDGE: "🔴 Sin edge (basura)",
        Verdict.INSUFFICIENT: "⚪ Muestra insuficiente",
        Verdict.NO_RESULT: "⚪ Sin resultados",
    }
    rows = []
    for e in edges:
        rows.append({
            "Estrategia": e.strategy,
            "N": e.n,
            "Win%": f"{e.win_rate * 100:.0f}%" if e.n else "-",
            "Expectancy": f"${e.expectancy_ars:,.0f}" if e.n else "-",
            "PF": f"{e.profit_factor:.2f}" if e.profit_factor is not None else "-",
            "PnL total": f"${e.total_pnl_ars:,.0f}",
            "PoP esp.": f"{e.expected_pop * 100:.0f}%" if e.expected_pop is not None else "-",
            "Calib.": f"{e.calibration_gap:+.2f}" if e.calibration_gap is not None else "-",
            "EV realiz.": f"{e.ev_realization:.2f}x" if e.ev_realization is not None else "-",
            "Veredicto": badge.get(e.verdict, e.verdict),
            "Alertas": " · ".join(e.flags),
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    confirmed = [e.strategy for e in edges if e.verdict == Verdict.EDGE]
    junk = [e.strategy for e in edges if e.verdict == Verdict.NO_EDGE]
    if confirmed:
        st.success("Edge confirmado (poné capital acá): " + ", ".join(confirmed))
    if junk:
        st.error("Sin edge con muestra suficiente (cortar o revisar): " + ", ".join(junk))
    st.caption(
        "Calib. = win rate real − PoP esperada (negativo = el modelo sobreestima el "
        "acierto). EV realiz. = expectancy real / EV esperado (1.0 = perfecto). Las "
        "estrategias nuevas con pocos trades quedan en 'muestra insuficiente' y nunca "
        "se bloquean: se protege capital sin matar ideas en desarrollo."
    )

    # ── Validación del grading SMC: ¿el grade predice el resultado? ───────────
    try:
        from optionsdesk.performance.attribution import attribute_by_grade
        grade_edges = attribute_by_grade(all_trades)   # reutiliza la lectura de arriba
    except Exception:
        grade_edges = []
    if grade_edges:
        st.divider()
        st.subheader("¿El grade SMC predice el resultado?")
        st.caption(
            "Si los grades altos (S/A) no le ganan a los bajos (C/D), el grading es "
            "ruido y hay que recalibrarlo. Validación del sistema de calidad de señales."
        )
        grows = [{
            "Grade": e.strategy.split()[-1],
            "N": e.n,
            "Win%": f"{e.win_rate * 100:.0f}%" if e.n else "-",
            "Expectancy": f"${e.expectancy_ars:,.0f}" if e.n else "-",
            "PF": f"{e.profit_factor:.2f}" if e.profit_factor is not None else "-",
            "PnL total": f"${e.total_pnl_ars:,.0f}",
            "Veredicto": badge.get(e.verdict, e.verdict),
        } for e in grade_edges]
        st.dataframe(pd.DataFrame(grows), hide_index=True, use_container_width=True)


def _render_strategy_tree() -> None:
    """Panel del optimizador por contexto: la mejor estrategia para cada regimen.

    El edge es condicional al contexto — una estrategia que pierde en lateral puede
    ganar en tendencia. El arbol busca y guarda un ESPECIALISTA por regimen; en vivo
    el runner despliega el que corresponde al regimen actual.
    """
    try:
        from optionsdesk.backtest.strategy_tree import load_policy_doc
        from optionsdesk.backtest.strategy_context import regime_label_es
    except Exception:
        return

    doc = load_policy_doc()
    st.subheader("Estrategia optima por contexto (arbol de regimenes)")
    if not doc:
        st.caption(
            "Todavia no se corrio la optimizacion. Generala con "
            "`python -m optionsdesk.backtest.strategy_tree --evals 40` "
            "(o desde el runner). Aprende que estrategia rinde en cada regimen de mercado."
        )
        return

    c1, c2 = st.columns(2)
    c1.metric("Genomas evaluados", doc.get("evaluated", 0))
    c2.metric("Mejor fitness global", f"{doc.get('overall_fitness', 0):,.0f}")
    st.caption(f"Ultima optimizacion: {doc.get('updated_at', '?')}")

    # Tabla regimen → estrategia (especialista o fallback global).
    regimes = doc.get("regimes", {})
    rows = []
    for reg, slot in regimes.items():
        if slot.get("n", 0) <= 0 and not slot.get("specialist"):
            continue
        flags = (slot.get("genome", {}) or {}).get("flags", {})
        active = [k.replace("enable_", "") for k, v in flags.items() if v]
        rows.append({
            "Regimen": slot.get("label") or regime_label_es(reg),
            "Tipo": "especialista" if slot.get("specialist") else "global (fallback)",
            "Setups activos": ", ".join(active) if active else "—",
            "N": slot.get("n", 0),
            "Win%": f"{slot['win_rate'] * 100:.0f}%" if slot.get("win_rate") is not None else "-",
            "PnL": f"${slot['total_pnl_ars']:,.0f}" if slot.get("total_pnl_ars") is not None else "-",
            "Fitness": f"{slot['fitness']:,.0f}" if slot.get("fitness") is not None else "-",
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption(
            "Especialista = estrategia con edge propio y muestra suficiente en ESE contexto. "
            "Donde no hay especialista confiable se cae al mejor global (nunca a un perdedor)."
        )
    else:
        st.caption("Aun sin especialistas confiables — opera el mejor genoma global en todos los regimenes.")

    # Mejor genoma global (params + setups).
    best = doc.get("best_overall", {})
    with st.expander("Mejor estrategia global (params + setups)"):
        bp = best.get("params", {})
        bf = best.get("flags", {})
        if bp:
            st.dataframe(
                pd.DataFrame([{"param": k, "valor": v} for k, v in bp.items()]),
                hide_index=True, use_container_width=True,
            )
        st.caption("Setups: " + ", ".join(f"{k}={'on' if v else 'off'}" for k, v in bf.items()))


def _render_param_learner() -> None:
    """Panel del optimizador de parametros: que esta probando y como le fue.

    Muestra el campeon (mejor set probado), el retador en curso, la temperatura de
    exploracion (sube cuando falla) y los parametros que se desviaron del default.
    """
    try:
        from optionsdesk.backtest.param_learner import load_learner_status
        from optionsdesk.backtest.param_store import describe_active, TUNABLES
    except Exception:
        return

    state = load_learner_status()
    st.subheader("Auto-tuning de parametros (SMC + señales)")
    if not state:
        st.caption(
            "El optimizador todavia no corrio. Arranca con el runner continuo "
            "(python -m optionsdesk.backtest.demo_runner) y aprende de cada trade cerrado."
        )
        return

    action_label = {
        "init_baseline":    "Midiendo el baseline del campeon",
        "gathering":        "Juntando muestra del set en prueba",
        "deploy_challenger":"Retador desplegado en vivo",
        "promote":          "Retador PROMOVIDO a campeon",
        "revert":           "Retador descartado",
    }.get(state.get("last_action", ""), state.get("last_action", "—"))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Iteracion", state.get("iterations", 0))
    c2.metric("Temperatura", f"{state.get('temperature', 0):.2f}",
              help="Exploracion: sube cuando un retador falla, baja al promover.")
    c3.metric("Reverts seguidos", state.get("consecutive_reverts", 0))
    cf = state.get("champion_fitness")
    c4.metric("Fitness campeon", f"{cf:,.0f}" if cf is not None else "—")

    st.caption(f"Estado: {action_label} · {state.get('note', '')}")

    # Tabla de parametros activos vs default (lo que el learner movio).
    rows = describe_active()
    moved = [r for r in rows if abs(r["delta_vs_default"]) > 1e-9]
    if moved:
        st.caption("Parametros que el learner desvio del valor de fabrica:")
        disp = [
            {
                "Parametro": r["param"],
                "Actual": r["actual"],
                "Default": r["default"],
                "Δ": r["delta_vs_default"],
                "Que controla": TUNABLES[r["param"]].help if r["param"] in TUNABLES else "",
            }
            for r in moved
        ]
        st.dataframe(pd.DataFrame(disp), hide_index=True, use_container_width=True)
    else:
        st.caption("Aun operando con los parametros de fabrica (sin desvios probados todavia).")

    # Curva de fitness reciente (si hay historial).
    hist = state.get("history", [])
    if len(hist) >= 2:
        with st.expander("Historial de fitness por iteracion"):
            fit_df = pd.DataFrame([
                {"iter": h.get("iter"), "fitness": h.get("fitness"), "accion": h.get("action")}
                for h in hist
            ]).set_index("iter")
            st.line_chart(fit_df[["fitness"]])


def _render_gap_stress(spot: Optional[float]) -> None:
    """Stress de gap GGAL sobre el libro abierto real (data/open_positions.jsonl).

    Todo el libro está sobre el mismo subyacente; un gap atraviesa calls, puts y
    caución a la vez. Muestra la pérdida de cola que el delta/theta del día esconden.
    """
    st.subheader("Riesgo de gap (concentración GGAL)")
    positions = []
    _agg_ok = False
    try:
        from optionsdesk.portfolio.aggregator import (
            load_open_positions, gap_stress, worst_case_loss, concentration_report,
        )
        _agg_ok = True
        positions = load_open_positions()
    except ImportError:
        st.warning("Modulo optionsdesk.portfolio.aggregator no disponible — stress de gap desactivado.")
        st.divider()
        return
    except Exception as exc:
        st.warning(f"No se pudo cargar el libro de posiciones: {exc}")
        st.divider()
        return

    if not positions:
        st.caption("Sin posiciones en el libro real (data/open_positions.jsonl).")
        st.divider()
        return

    ref_spot = float(spot) if spot and spot > 0 else float(getattr(positions[0], "spot_entry", 0) or 0)
    if ref_spot <= 0:
        st.caption("Sin spot de referencia para el stress.")
        st.divider()
        return

    sc = gap_stress(positions, ref_spot)
    shock, worst = worst_case_loss(sc)
    rows = [
        {"Gap GGAL": f"{s:+.0f}%", "P&L libro": f"${p:,.0f}"}
        for s, p in zip(sc.shock_pcts, sc.pnl_ars)
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    if worst < 0:
        st.error(
            f"Peor caso: gap {shock:+.0f}% → ${worst:,.0f} sobre {len(positions)} "
            "posiciones (todas GGAL)."
        )
    for flag in concentration_report(positions).flags:
        st.warning(flag)
    st.caption(
        "Un gap de GGAL atraviesa todo el libro a la vez. Las puts vendidas explotan "
        "en el gap a la baja — verificá que el peor caso sea tolerable para tu capital."
    )
    st.divider()


if __name__ == "__main__":
    main()
