"""Momentum de volumen para detección temprana de movimientos en el MERVAL.

En un mercado chico como BYMA, si un fondo local o un extranjero arranca a comprar,
el volumen lo grita ANTES que el precio lo confirme — no hay profundidad para absorber
el flujo en silencio. Detectar la expansión de volumen temprano = entrar en la ola,
no perseguirla.

Conceptos implementados:
  RVOL       Relative Volume: volumen hoy / media(N ruedas). >1.5 = alguien está
             comprando. >2.5 = flujo institucional. La señal más confiable.
  OBV        On-Balance Volume: acumula volumen en la dirección del precio. Una
             tendencia OBV alcista mientras el precio está lateral = acumulación
             silenciosa → precursor de movimiento.
  Vol Trend  ¿El volumen está expandiendo o contrayendo? Expansión en rally =
             confirmación. Contracción en rally = distribución.
  VWAP vol   Concentración de volumen cerca del VWAP = precio justo. Lejos = se
             alimenta de stops, no de compradores nuevos.

Solo trabaja con daily bars (volumen real). El tape intradía de BYMA = 0.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class VolumeSnapshot:
    """Estado de volumen de una acción en el día actual."""
    rvol: Optional[float]          # volumen hoy / media(20 ruedas)
    rvol_label: str                # "explosivo" | "alto" | "normal" | "bajo" | "sin datos"
    vol_trend: str                 # "expandiendo" | "contrayendo" | "neutral"
    obv_trend: str                 # "alcista" | "bajista" | "lateral"
    obv_divergence: Optional[str]  # "acumulacion_silenciosa" | "distribucion" | None
    price_vol_ok: bool             # precio sube con volumen > media (confirmado)
    note: str                      # resumen legible para el panel


# Umbrales de RVOL — auto-calibrables via param_store
_RVOL_EXPLOSIVE = 2.5    # >2.5x = flujo institucional real
_RVOL_HIGH      = 1.5    # >1.5x = participación alta, ola arrancando
_RVOL_LOW       = 0.70   # <0.7x = falta convicción (no perseguir)
_RVOL_LOOKBACK  = 20     # ruedas para el volumen promedio de referencia


def _learned(name: str, default: float) -> float:
    """Lee un parametro aprendido por el learner; default si no hay (==hoy)."""
    try:
        from optionsdesk.backtest.param_store import param
        return param(name, default)
    except Exception:
        return default


def volume_snapshot(
    daily: pd.DataFrame,
    *,
    lookback: Optional[int] = None,
    obv_lookback: int = 10,
) -> Optional[VolumeSnapshot]:
    """Analiza el estado de volumen del día actual.

    daily: DataFrame con columnas date, open, high, low, close, volume.
           Ordenado cronológicamente. Mínimo 5 filas.
    lookback: None → usa el valor aprendido (rvol_lookback) o el default 20.
    """
    if daily is None or len(daily) < 5:
        return None
    if "volume" not in daily.columns:
        return None

    if lookback is None:
        lookback = int(_learned("rvol_lookback", _RVOL_LOOKBACK))
    rvol_explosive = _learned("rvol_explosive_threshold", _RVOL_EXPLOSIVE)
    rvol_high = _learned("rvol_high_threshold", _RVOL_HIGH)

    close  = daily["close"].astype(float)
    volume = daily["volume"].astype(float)

    # Ignorar si el volumen está todo en cero (tape intradiario)
    if float(volume.tail(lookback).sum()) == 0.0:
        return None

    # ── RVOL ─────────────────────────────────────────────────────────────────
    window = min(lookback, len(volume) - 1)
    vol_mean = float(volume.iloc[-window - 1:-1].mean() or 0.0)
    today_vol = float(volume.iloc[-1])
    rvol = round(today_vol / vol_mean, 2) if vol_mean > 0 else None

    if rvol is None:
        rvol_label = "sin datos"
    elif rvol >= rvol_explosive:
        rvol_label = "explosivo"    # flujo institucional — montar la ola
    elif rvol >= rvol_high:
        rvol_label = "alto"         # ola arrancando — entrada temprana
    elif rvol < _RVOL_LOW:
        rvol_label = "bajo"         # sin convicción — no perseguir
    else:
        rvol_label = "normal"

    # ── Vol Trend (expandiendo vs contrayendo) ────────────────────────────────
    if len(volume) >= 6:
        short_avg = float(volume.tail(3).mean() or 0)
        long_avg  = float(volume.tail(min(lookback, len(volume))).mean() or 0)
        if long_avg > 0:
            ratio = short_avg / long_avg
            if ratio > 1.15:
                vol_trend = "expandiendo"
            elif ratio < 0.85:
                vol_trend = "contrayendo"
            else:
                vol_trend = "neutral"
        else:
            vol_trend = "neutral"
    else:
        vol_trend = "neutral"

    # ── OBV ──────────────────────────────────────────────────────────────────
    obv_series = _compute_obv(close, volume)
    obv_trend, obv_div = _obv_analysis(close, obv_series, obv_lookback)

    # ── Price × Volume confirmation ───────────────────────────────────────────
    price_up  = len(close) >= 2 and float(close.iloc[-1]) > float(close.iloc[-2])
    price_vol_ok = price_up and (rvol is not None and rvol >= rvol_high)

    # ── Nota legible ─────────────────────────────────────────────────────────
    note = _build_note(rvol, rvol_label, vol_trend, obv_trend, obv_div)

    return VolumeSnapshot(
        rvol=rvol,
        rvol_label=rvol_label,
        vol_trend=vol_trend,
        obv_trend=obv_trend,
        obv_divergence=obv_div,
        price_vol_ok=price_vol_ok,
        note=note,
    )


def _compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume clásico. Positivo = acumulación, negativo = distribución."""
    direction = close.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    return (direction * volume).cumsum()


def _obv_analysis(
    close: pd.Series,
    obv: pd.Series,
    lookback: int,
) -> tuple[str, Optional[str]]:
    """Tendencia OBV + divergencia precio/volumen."""
    if len(obv) < lookback + 1:
        return "lateral", None

    window_obv   = obv.tail(lookback)
    window_close = close.tail(lookback)

    obv_slope   = float(window_obv.iloc[-1] - window_obv.iloc[0])
    price_slope = float(window_close.iloc[-1] - window_close.iloc[0])

    if obv_slope > 0:
        obv_trend = "alcista"
    elif obv_slope < 0:
        obv_trend = "bajista"
    else:
        obv_trend = "lateral"

    # Divergencias precio/volumen — las señales más tempranas del MERVAL
    divergence = None
    if price_slope < 0 and obv_slope > 0:
        # Precio baja pero OBV sube → alguien acumula en silencio
        divergence = "acumulacion_silenciosa"
    elif price_slope > 0 and obv_slope < 0:
        # Precio sube pero OBV baja → distribución disfrazada de rally
        divergence = "distribucion"

    return obv_trend, divergence


def _build_note(
    rvol: Optional[float],
    rvol_label: str,
    vol_trend: str,
    obv_trend: str,
    obv_div: Optional[str],
) -> str:
    parts: list[str] = []
    if rvol is not None:
        parts.append(f"RVOL {rvol:.1f}x ({rvol_label})")
    parts.append(f"vol {vol_trend}")
    parts.append(f"OBV {obv_trend}")
    if obv_div == "acumulacion_silenciosa":
        parts.append("⚡ acumulacion silenciosa detectada")
    elif obv_div == "distribucion":
        parts.append("⚠ distribucion (precio sube, OBV baja)")
    return " | ".join(parts)


def volume_score_delta(snap: VolumeSnapshot) -> tuple[float, list[str]]:
    """Boost/penalización al score SMC basado en el análisis de volumen.

    Diseño conservador para MERVAL: boost grande cuando hay evidencia fuerte
    (RVOL explosivo + OBV alcista), penalización cuando la ola no tiene flujo.
    Solo complements la SMC — no genera ni bloquea señales por sí solo.
    """
    delta = 0.0
    notes: list[str] = []

    if snap.rvol is None:
        return 0.0, []

    # RVOL: la señal más directa de flujo institucional en BYMA
    if snap.rvol >= _learned("rvol_explosive_threshold", _RVOL_EXPLOSIVE):
        delta += 14.0; notes.append(f"RVOL {snap.rvol:.1f}x — flujo institucional (entrada temprana)")
    elif snap.rvol >= _learned("rvol_high_threshold", _RVOL_HIGH):
        delta += 7.0; notes.append(f"RVOL {snap.rvol:.1f}x — ola arrancando")
    elif snap.rvol < _RVOL_LOW:
        delta -= 8.0; notes.append(f"RVOL {snap.rvol:.1f}x — sin convicción (no perseguir)")

    # Acumulación silenciosa: precio lateral/bajista pero OBV sube
    if snap.obv_divergence == "acumulacion_silenciosa":
        delta += 10.0; notes.append("OBV divergencia alcista — acumulacion silenciosa")
    elif snap.obv_divergence == "distribucion":
        delta -= 10.0; notes.append("OBV divergencia bajista — distribucion disfrazada")

    # Volumen expandiendo en la dirección del trade
    if snap.vol_trend == "expandiendo":
        delta += 5.0; notes.append("volumen expandiendo — momentum confirmado")
    elif snap.vol_trend == "contrayendo":
        delta -= 4.0; notes.append("volumen contrayendo — momentum débil")

    # Precio sube CON volumen alto = confirmado
    if snap.price_vol_ok:
        delta += 3.0   # bonus tranquilo, ya está cubierto por RVOL arriba

    return round(delta, 1), notes
