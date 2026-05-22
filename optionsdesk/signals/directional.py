"""Modulo de contexto de mercado (opcional, experimental).

Calcula tendencia y momentum a partir del historial OHLCV del subyacente.
Sin historial suficiente devuelve confidence='sin datos' y no cambia ningun
comportamiento — el bot opera como renta pura si el toggle esta desactivado
o si no hay datos.

v2.2: si el historial tiene columnas OHLCV completas, consume `technical.analyze`
para enriquecer el contexto (rsi, atr_pct). Si no (ej. solo columna 'spot' del
recorder), usa el cálculo SMA simple original.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


@dataclass
class MarketContext:
    # Campos originales — usados por _apply_directional_boost en recommender.py
    trend: str           # "alcista", "bajista", "lateral"
    momentum_pct: float  # % de cambio sobre ventana corta
    confidence: str      # "alta", "media", "baja", "sin datos"
    note: str            # texto listo para mostrar en el dashboard
    # v2.2 — opcionales para no romper tests existentes
    rsi: float = 50.0
    atr_pct: float = 0.0
    signal_strength: str = "neutral"   # "fuerte", "moderada", "neutral", "debil"
    snap: object = None   # TechnicalSnapshot | None (evita import circular)
    # v2.3 — contexto multi-timeframe (todos con defaults, backward-compatible)
    htf_trend: str = "lateral"          # tendencia semanal (HTF)
    ltf_trend: str = "lateral"          # tendencia intradiaria (LTF)
    mtf_alignment: str = "neutral"      # "alineado_alcista" | "alineado_bajista" | "conflicto" | "neutral"
    vol_regime: str = "normal"           # "expansion" | "contraccion" | "normal"
    htf_snap: object = None             # TechnicalSnapshot semanal | None
    ltf_snap: object = None             # TechnicalSnapshot intradiario | None


_MIN_ROWS  = 5
_LONG_WINDOW = 20
_OHLCV_COLS  = {"open", "high", "low", "close"}


def compute_market_context(
    spot_history: Optional[pd.DataFrame],
    htf: Optional[pd.DataFrame] = None,   # OHLCV semanal (HTF)
    ltf: Optional[pd.DataFrame] = None,   # OHLCV intradiario (LTF)
) -> MarketContext:
    """Calcula contexto de mercado a partir del historial de precios.

    spot_history: DataFrame del recorder (columna 'spot') **o** OHLCV completo
    (columnas open/high/low/close) de UnderlyingHistory.daily().
    htf: OHLCV semanal opcional para contexto de marco alto.
    ltf: OHLCV intradiario opcional para contexto de marco bajo.
    Si no hay suficientes datos, devuelve confidence='sin datos'.
    """
    if spot_history is None or len(spot_history) < _MIN_ROWS:
        return _no_data()

    # Si el DataFrame tiene columnas OHLCV usamos el analisis tecnico completo
    if _OHLCV_COLS.issubset(set(spot_history.columns)):
        ctx = _from_ohlcv(spot_history)
    else:
        # Fallback: serie de spots del recorder (solo cierre)
        spots = _extract_spots(spot_history)
        if spots is None or len(spots) < _MIN_ROWS:
            return _no_data()
        ctx = _from_spots(spots)

    # Enriquecer con HTF y LTF si estan disponibles
    if htf is not None and not htf.empty and _OHLCV_COLS.issubset(set(htf.columns)):
        ctx = _enrich_htf(ctx, htf)
    if ltf is not None and not ltf.empty and _OHLCV_COLS.issubset(set(ltf.columns)):
        ctx = _enrich_ltf(ctx, ltf)

    # Calcular alineacion multi-TF y regimen de volatilidad
    ctx.mtf_alignment = _mtf_alignment(ctx.trend, ctx.htf_trend, ctx.ltf_trend)
    if spot_history is not None and _OHLCV_COLS.issubset(set(spot_history.columns)):
        ctx.vol_regime = _vol_regime(spot_history)

    return ctx


# ── Implementaciones ──────────────────────────────────────────────────────────

def _from_ohlcv(df: pd.DataFrame) -> MarketContext:
    from optionsdesk.signals.technical import analyze   # import local para no circularizar

    snap = analyze(df)
    n    = len(df)

    # Mapeo de tendencia: TechnicalSnapshot usa mayúsculas
    trend_map = {"ALCISTA": "alcista", "BAJISTA": "bajista", "LATERAL": "lateral"}
    trend     = trend_map.get(snap.trend, "lateral")

    confidence = "alta" if n >= _LONG_WINDOW else "media" if n >= _MIN_ROWS * 2 else "baja"

    return MarketContext(
        trend=trend,
        momentum_pct=round(snap.momentum, 2),
        confidence=confidence,
        note=_build_note(trend, snap.momentum, confidence, n),
        rsi=snap.rsi,
        atr_pct=snap.atr_pct,
        signal_strength=snap.signal_strength,   # ya calculado por technical.analyze()
        snap=snap,
    )


def _from_spots(spots: pd.Series) -> MarketContext:
    """Cálculo SMA simple (historial sin OHLCV, solo spot del recorder)."""
    n       = len(spots)
    short_w = min(5, n)
    long_w  = min(_LONG_WINDOW, n)

    sma_short  = spots.iloc[-short_w:].mean()
    sma_long   = spots.iloc[-long_w:].mean()
    momentum   = (spots.iloc[-1] / spots.iloc[-short_w] - 1.0) * 100.0

    if sma_short > sma_long * 1.01:
        trend = "alcista"
    elif sma_short < sma_long * 0.99:
        trend = "bajista"
    else:
        trend = "lateral"

    confidence = "alta" if n >= _LONG_WINDOW else "media" if n >= _MIN_ROWS * 2 else "baja"

    return MarketContext(
        trend=trend,
        momentum_pct=round(momentum, 2),
        confidence=confidence,
        note=_build_note(trend, momentum, confidence, n),
    )


def _no_data() -> MarketContext:
    return MarketContext(
        trend="lateral",
        momentum_pct=0.0,
        confidence="sin datos",
        note="Juntando datos — el bot necesita al menos 5 dias de historial.",
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_spots(df: pd.DataFrame) -> Optional[pd.Series]:
    for col in ("spot", "underlying", "close", "last"):
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(s) >= _MIN_ROWS:
                return s
    if df.shape[1] == 1:
        s = pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna()
        return s if len(s) >= _MIN_ROWS else None
    return None


def _build_note(trend: str, momentum_pct: float, confidence: str, n_days: int) -> str:
    direction  = {"alcista": "subiendo", "bajista": "bajando", "lateral": "lateral"}[trend]
    conf_label = {"alta": "alta confianza", "media": "confianza media", "baja": "confianza baja"}
    return (
        f"GGAL {direction}  |  Momentum: {momentum_pct:+.1f}%  "
        f"({conf_label.get(confidence, confidence)}, {n_days} dias de historial)"
    )


# ── Multi-timeframe helpers (v2.3) ────────────────────────────────────────────

def _enrich_htf(ctx: MarketContext, htf: pd.DataFrame) -> MarketContext:
    """Corre analyze() sobre el OHLCV semanal y rellena htf_trend/htf_snap."""
    from optionsdesk.signals.technical import analyze   # local para no circularizar
    try:
        snap = analyze(htf)
        trend_map = {"ALCISTA": "alcista", "BAJISTA": "bajista", "LATERAL": "lateral"}
        ctx.htf_trend = trend_map.get(snap.trend, "lateral")
        ctx.htf_snap  = snap
    except Exception:
        pass
    return ctx


def _enrich_ltf(ctx: MarketContext, ltf: pd.DataFrame) -> MarketContext:
    """Corre analyze() sobre el OHLCV intradiario y rellena ltf_trend/ltf_snap.

    Para LTF intradiario: usa las ultimas 60 barras (1h de datos a 1-min)
    y omite SMC (muy ruidoso en 1-min). Solo indicadores clasicos.
    """
    from optionsdesk.signals.technical import analyze
    try:
        df = ltf.tail(60).reset_index(drop=True)
        if len(df) < 5:
            return ctx
        snap = analyze(df)
        trend_map = {"ALCISTA": "alcista", "BAJISTA": "bajista", "LATERAL": "lateral"}
        ctx.ltf_trend = trend_map.get(snap.trend, "lateral")
        ctx.ltf_snap  = snap
    except Exception:
        pass
    return ctx


def _mtf_alignment(daily: str, htf: str, ltf: str) -> str:
    """Determina la alineacion entre los tres timeframes.

    Reglas (de mas fuerte a mas debil):
    - HTF + diario alineados alcistas → 'alineado_alcista' (LTF puede diferir)
    - HTF + diario alineados bajistas → 'alineado_bajista'
    - HTF y diario en direcciones opuestas (excl. lateral) → 'conflicto'
    - Resto (lateral o datos insuficientes) → 'neutral'
    """
    bullish = ("alcista",)
    bearish = ("bajista",)

    htf_bull = htf in bullish
    htf_bear = htf in bearish
    day_bull = daily in bullish
    day_bear = daily in bearish

    if htf_bull and day_bull:
        return "alineado_alcista"
    if htf_bear and day_bear:
        return "alineado_bajista"
    if (htf_bull and day_bear) or (htf_bear and day_bull):
        return "conflicto"
    return "neutral"


def _vol_regime(df_daily: pd.DataFrame, short_n: int = 5, long_n: int = 20) -> str:
    """Compara ATR reciente vs historico para detectar expansion/contraccion."""
    from optionsdesk.signals.technical import atr as _atr
    try:
        if len(df_daily) < long_n + 2:
            return "normal"
        atr_series = _atr(df_daily, long_n).dropna()
        if len(atr_series) < short_n + 1:
            return "normal"
        recent  = float(atr_series.iloc[-short_n:].mean())
        hist    = float(atr_series.mean())
        if hist <= 0:
            return "normal"
        ratio = recent / hist
        if ratio > 1.25:
            return "expansion"
        if ratio < 0.75:
            return "contraccion"
        return "normal"
    except Exception:
        return "normal"
