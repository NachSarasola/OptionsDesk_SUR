"""Clasificador de regimen de mercado — el CONTEXTO en que opera una estrategia.

Premisa (la observacion clave del operador): una estrategia que pierde en un
contexto puede ganar en otro. El edge es CONDICIONAL al regimen. Por eso el
optimizador no busca "la mejor estrategia global" sino la mejor estrategia POR
regimen, y en vivo despliega la adecuada al contexto vigente.

Diseño modular por "ejes": cada eje reduce el estado del mercado a una etiqueta
de baja cardinalidad. El regimen es la combinacion de ejes. Default: tendencia x
volatilidad (6 cubetas), facil de extender con momentum, RSI, zona PDA, etc.

Es PURO (pandas) y se computa identico offline (backtest, sobre barras truncadas)
y en vivo (sobre las ultimas barras) → la politica entrenada se aplica sin sesgo.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable, Optional

import pandas as pd

# Minimo de barras para clasificar con confianza. Por debajo → regimen desconocido.
_MIN_BARS = 30


def _ema(series: pd.Series, span: int) -> float:
    return float(series.ewm(span=span, adjust=False).mean().iloc[-1])


def _trend_axis(daily: pd.DataFrame) -> str:
    """Tendencia: up | side | down, desde el apilamiento de medias y el precio."""
    close = daily["close"].astype(float)
    if len(close) < _MIN_BARS:
        return "side"
    sma20 = float(close.tail(20).mean())
    sma50 = float(close.tail(50).mean()) if len(close) >= 50 else float(close.mean())
    last = float(close.iloc[-1])
    # Pendiente reciente de la SMA20 (20 ruedas) como confirmacion de direccion.
    sma20_prev = float(close.iloc[-21:-1].mean()) if len(close) >= 21 else sma20
    slope_up = sma20 > sma20_prev
    if last > sma20 >= sma50 and slope_up:
        return "up"
    if last < sma20 <= sma50 and not slope_up:
        return "down"
    return "side"


def _vol_axis(daily: pd.DataFrame) -> str:
    """Volatilidad: hi | lo, comparando la vol realizada reciente contra su mediana."""
    close = daily["close"].astype(float)
    if len(close) < _MIN_BARS:
        return "lo"
    ret = close.pct_change().dropna()
    if len(ret) < 20:
        return "lo"
    recent = float(ret.tail(10).std())
    # Mediana movil de la vol de 10 ruedas como umbral adaptativo (no hardcode).
    rolling = ret.rolling(10).std().dropna()
    if rolling.empty:
        return "lo"
    median_vol = float(rolling.median())
    return "hi" if recent >= median_vol else "lo"


# ── Registro de ejes (extensible: agregar mas dimensiones aca) ────────────────

@dataclass(frozen=True)
class RegimeAxis:
    name: str
    labels: tuple[str, ...]
    classify: Callable[[pd.DataFrame], str]


DEFAULT_AXES: tuple[RegimeAxis, ...] = (
    RegimeAxis("trend", ("up", "side", "down"), _trend_axis),
    RegimeAxis("vol", ("hi", "lo"), _vol_axis),
)

UNKNOWN_REGIME = "unknown"


def classify_regime(
    daily: pd.DataFrame,
    axes: tuple[RegimeAxis, ...] = DEFAULT_AXES,
) -> str:
    """Etiqueta de regimen ('up/hi', 'down/lo', ...) desde las barras dadas.

    `daily` debe estar truncado hasta la barra a clasificar (sin look-ahead).
    Devuelve UNKNOWN_REGIME si no hay barras suficientes.
    """
    if daily is None or len(daily) < _MIN_BARS:
        return UNKNOWN_REGIME
    return "/".join(axis.classify(daily) for axis in axes)


def all_regimes(axes: tuple[RegimeAxis, ...] = DEFAULT_AXES) -> list[str]:
    """Todas las cubetas posibles (producto cartesiano de los ejes) + unknown."""
    combos = product(*(axis.labels for axis in axes))
    return ["/".join(c) for c in combos] + [UNKNOWN_REGIME]


def regime_label_es(regime: str) -> str:
    """Etiqueta legible en castellano para el dashboard."""
    if regime == UNKNOWN_REGIME:
        return "sin datos"
    parts = regime.split("/")
    trend_map = {"up": "alcista", "side": "lateral", "down": "bajista"}
    vol_map = {"hi": "vol alta", "lo": "vol baja"}
    out = []
    if len(parts) >= 1:
        out.append(trend_map.get(parts[0], parts[0]))
    if len(parts) >= 2:
        out.append(vol_map.get(parts[1], parts[1]))
    out.extend(parts[2:])
    return " · ".join(out)
