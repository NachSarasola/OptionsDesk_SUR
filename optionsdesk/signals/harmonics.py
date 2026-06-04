"""Patrones armónicos adaptados al MERVAL — Vol 2 Smart Money by BEIG.

Implementa las formaciones M (bullish Peak Formation) y W (bearish Peak Formation)
del Market Maker Method (Steve Mauro) combinadas con zonas OTE de Fibonacci.

Adaptación al MERVAL:
- Solo Diario / Semanal (intradía inviable por spread/slippage/gaps).
- Tolerancias ampliadas (vs. Forex/cripto) por gaps nocturnos y liquidez baja.
- El punto D (PRZ) debe coincidir con liquidez SMC (BSL/SSL o OB) para alta prob.
- TPs sobre swing highs/lows previos (preferible) o 0.382/0.618 de la pierna AD.

Arquitectura:
    find_peak_formations(swings, pda, daily) → list[PeakFormation]
    fib_levels(lo, hi)                        → dict[str, float]  (reutilizable Fase 4)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Tolerancias para MERVAL (amplias vs. Forex por gaps y baja liquidez)
_D_TOLERANCE   = 0.07   # D debe estar dentro del 7% de X (punto de origen)
_OTE_LO        = 0.50   # límite inferior OTE permisivo para MERVAL
_OTE_HI        = 0.90   # límite superior OTE (incluye FYRS 0.886)
_MIN_SWING_PCT = 0.03   # la pierna XA debe ser ≥ 3% para que tenga relevancia


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class PeakFormation:
    """Formación M (bullish) o W (bearish) del Market Maker Method.

    Estructura de 4 puntos: X → A → B/C → D (punto de entrada / PRZ).
    El 'punto D' es donde el precio debería revertir — coincide con OTE y liquidez.

    M bullish (en discount zone):  X=low, A=high(manipulación), D≈X (2º mínimo en OTE)
    W bearish (en premium zone):   X=high, A=low(manipulación), D≈X (2º máximo en OTE)
    """
    kind: str           # "M" (bullish) | "W" (bearish)
    direction: str      # "bullish" | "bearish"
    x: float           # punto de partida (low para M, high para W)
    a: float           # swing de manipulación (high para M, low para W)
    d: float           # punto de entrada / PRZ
    prz_lo: float      # límite inferior de la Potential Reversal Zone
    prz_hi: float      # límite superior de la PRZ
    tp1: float         # take profit 1 (0.382 de XA desde D)
    tp2: float         # take profit 2 (swing A — objetivo de vuelta al extremo)
    fib_618: float     # nivel 0.618 de la pierna XA (OTE)
    fib_786: float     # nivel 0.786 de la pierna XA (OTE precision)
    fib_886: float     # nivel 0.886 de la pierna XA (FYRS / sniper)
    valid: bool        # True si D está dentro del OTE y tolerancias de ratio
    bar_index_d: int   # índice (en el df de origen) del punto D
    date_d: Optional[datetime] = None   # fecha del punto D (si disponible)


# ── Fibonacci helper ───────────────────────────────────────────────────────────

def fib_levels(lo: float, hi: float) -> dict[str, float]:
    """Niveles de Fibonacci de `lo` a `hi`.

    Reutilizable para cualquier pierna (no solo para armónicos).
    Genera retrocesos estándar + extensiones negativas.
    """
    rng = hi - lo
    if rng <= 0:
        return {}
    return {
        "0.000": lo,
        "0.236": lo + 0.236 * rng,
        "0.382": lo + 0.382 * rng,
        "0.500": lo + 0.500 * rng,
        "0.618": lo + 0.618 * rng,
        "0.705": lo + 0.705 * rng,
        "0.786": lo + 0.786 * rng,
        "0.886": lo + 0.886 * rng,
        "1.000": hi,
        "1.272": hi + 0.272 * rng,
        "1.618": hi + 0.618 * rng,
        # Extensiones negativas (desde el punto de origen)
        "-0.272": lo - 0.272 * rng,
        "-0.618": lo - 0.618 * rng,
    }


# ── Detector de Peak Formations ───────────────────────────────────────────────

def find_peak_formations(
    swings: pd.DataFrame,
    pda: Optional[object] = None,
    daily: Optional[pd.DataFrame] = None,
    lookback: int = 20,
    tolerance_d: float = _D_TOLERANCE,
) -> list[PeakFormation]:
    """Detecta formaciones M (bullish) y W (bearish) en los swings confirmados.

    Algoritmo:
    1. Para cada par de swing lows (X, D) donde |D - X| / X ≤ tolerance_d:
       - Buscar un swing high (A) entre ellos → forma la M
       - Si D está en OTE de XA y la zona es discount → PeakFormation M válida
    2. Análogo para W (par de swing highs X, D con swing low A entre ellos).
    3. Devuelve las últimas `max_patterns` formaciones encontradas.

    Returns: lista ordenada de más reciente a más antigua.
    """
    if swings is None or swings.empty or len(swings) < 3:
        return []

    recent = swings.tail(lookback).reset_index(drop=True)
    patterns: list[PeakFormation] = []

    pda_zone = getattr(pda, "zone", None) if pda is not None else None

    # ── M patterns (bullish): L-H-L ──────────────────────────────────────────
    lows  = recent[recent["HighLow"] == -1].reset_index(drop=True)
    highs = recent[recent["HighLow"] ==  1].reset_index(drop=True)

    for i in range(len(lows) - 1):
        x_row = lows.iloc[i]
        for j in range(i + 1, len(lows)):
            d_row = lows.iloc[j]
            x_val = float(x_row["Level"])
            d_val = float(d_row["Level"])
            x_idx = int(x_row["index"])
            d_idx = int(d_row["index"])

            if d_idx <= x_idx:
                continue

            # D debe estar cerca de X (tolerancia amplia para MERVAL)
            if abs(d_val - x_val) / max(x_val, 1e-9) > tolerance_d:
                continue

            # Buscar swing high A entre X y D
            between_highs = highs[
                (highs["index"] > x_idx) & (highs["index"] < d_idx)
            ]
            if between_highs.empty:
                continue
            a_row = between_highs.loc[between_highs["Level"].idxmax()]
            a_val = float(a_row["Level"])
            a_idx = int(a_row["index"])

            # Pierna XA debe ser significativa
            xa_rng = a_val - x_val
            if xa_rng / max(x_val, 1e-9) < _MIN_SWING_PCT:
                continue

            pf = _build_m_formation(
                x_val=x_val, a_val=a_val, d_val=d_val,
                x_idx=x_idx, a_idx=a_idx, d_idx=d_idx,
                d_row=d_row, daily=daily,
                pda_zone=pda_zone,
            )
            if pf is not None:
                patterns.append(pf)

    # ── W patterns (bearish): H-L-H ──────────────────────────────────────────
    for i in range(len(highs) - 1):
        x_row = highs.iloc[i]
        for j in range(i + 1, len(highs)):
            d_row = highs.iloc[j]
            x_val = float(x_row["Level"])
            d_val = float(d_row["Level"])
            x_idx = int(x_row["index"])
            d_idx = int(d_row["index"])

            if d_idx <= x_idx:
                continue

            if abs(d_val - x_val) / max(x_val, 1e-9) > tolerance_d:
                continue

            between_lows = lows[
                (lows["index"] > x_idx) & (lows["index"] < d_idx)
            ]
            if between_lows.empty:
                continue
            a_row = between_lows.loc[between_lows["Level"].idxmin()]
            a_val = float(a_row["Level"])
            a_idx = int(a_row["index"])

            xa_rng = x_val - a_val
            if xa_rng / max(a_val, 1e-9) < _MIN_SWING_PCT:
                continue

            pf = _build_w_formation(
                x_val=x_val, a_val=a_val, d_val=d_val,
                x_idx=x_idx, a_idx=a_idx, d_idx=d_idx,
                d_row=d_row, daily=daily,
                pda_zone=pda_zone,
            )
            if pf is not None:
                patterns.append(pf)

    # Ordenar por bar_index_d descendente (más reciente primero), eliminar dup
    seen: set[int] = set()
    unique: list[PeakFormation] = []
    for pf in sorted(patterns, key=lambda p: p.bar_index_d, reverse=True):
        key = (pf.kind, round(pf.d))
        if key not in seen:
            seen.add(key)
            unique.append(pf)

    return unique[:5]   # máx 5 patrones más recientes


# ── Constructores de formaciones ──────────────────────────────────────────────

def _build_m_formation(
    x_val: float, a_val: float, d_val: float,
    x_idx: int, a_idx: int, d_idx: int,
    d_row: pd.Series,
    daily: Optional[pd.DataFrame],
    pda_zone: Optional[str],
    tolerance_d: float = _D_TOLERANCE,
) -> Optional[PeakFormation]:
    """Construye una PeakFormation M (bullish) si cumple las condiciones."""
    xa_rng = a_val - x_val

    # D debe estar en la zona OTE de XA (retroceso de la pierna alcista XA)
    d_fib  = (a_val - d_val) / max(xa_rng, 1e-9)   # retroceso desde A
    if not (_OTE_LO <= d_fib <= _OTE_HI):
        return None

    # Preferencia por zona discount (PDA), pero permitir unknown/None también
    if pda_zone == "premium":
        return None

    fibs   = fib_levels(x_val, a_val)
    f618   = fibs.get("0.618", x_val + 0.618 * xa_rng)
    f786   = fibs.get("0.786", x_val + 0.786 * xa_rng)
    f886   = fibs.get("0.886", x_val + 0.886 * xa_rng)

    # PRZ centrado en D con banda proporcional al ATR (o 2% por defecto)
    band   = _estimate_band(daily, d_idx, d_price=d_val)
    prz_lo = max(d_val - band, x_val * 0.95)
    prz_hi = d_val + band

    # Take profits: TP1 = 0.382 de AD (precio objetivo parcial)
    ad_rng = a_val - d_val
    tp1    = d_val + 0.382 * ad_rng
    tp2    = a_val   # objetivo pleno = swing A

    date_d = _get_date(d_row, daily, d_idx)

    return PeakFormation(
        kind="M",
        direction="bullish",
        x=x_val, a=a_val, d=d_val,
        prz_lo=prz_lo, prz_hi=prz_hi,
        tp1=tp1, tp2=tp2,
        fib_618=f618, fib_786=f786, fib_886=f886,
        valid=True,
        bar_index_d=d_idx,
        date_d=date_d,
    )


def _build_w_formation(
    x_val: float, a_val: float, d_val: float,
    x_idx: int, a_idx: int, d_idx: int,
    d_row: pd.Series,
    daily: Optional[pd.DataFrame],
    pda_zone: Optional[str],
    tolerance_d: float = _D_TOLERANCE,
) -> Optional[PeakFormation]:
    """Construye una PeakFormation W (bearish) si cumple las condiciones."""
    xa_rng = x_val - a_val   # rng bajista

    d_fib  = (d_val - a_val) / max(xa_rng, 1e-9)   # retroceso desde A (desde abajo)
    if not (_OTE_LO <= d_fib <= _OTE_HI):
        return None

    if pda_zone == "discount":
        return None

    fibs   = fib_levels(a_val, x_val)
    f618   = fibs.get("0.618", x_val - 0.618 * xa_rng)
    f786   = fibs.get("0.786", x_val - 0.786 * xa_rng)
    f886   = fibs.get("0.886", x_val - 0.886 * xa_rng)

    band   = _estimate_band(daily, d_idx, d_price=d_val)
    prz_lo = d_val - band
    prz_hi = min(d_val + band, x_val * 1.05)

    ad_rng = d_val - a_val
    tp1    = d_val - 0.382 * ad_rng
    tp2    = a_val

    date_d = _get_date(d_row, daily, d_idx)

    return PeakFormation(
        kind="W",
        direction="bearish",
        x=x_val, a=a_val, d=d_val,
        prz_lo=prz_lo, prz_hi=prz_hi,
        tp1=tp1, tp2=tp2,
        fib_618=f618, fib_786=f786, fib_886=f886,
        valid=True,
        bar_index_d=d_idx,
        date_d=date_d,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _estimate_band(
    daily: Optional[pd.DataFrame], bar_index: int, n: int = 14, d_price: float = 0.0
) -> float:
    """Estima la banda alrededor del PRZ usando ATR de las últimas n barras.

    Si no hay datos de OHLCV devuelve 0.5% del precio D (PRZ mínimo viable).
    """
    default = max(d_price * 0.005, 1.0)   # 0.5% del precio, mínimo $1
    if daily is None or daily.empty:
        return default
    window = daily.iloc[max(0, bar_index - n): bar_index]
    if len(window) < 3:
        return default
    hi   = window["high"].astype(float)
    lo   = window["low"].astype(float)
    cl   = window["close"].astype(float)
    prev = cl.shift(1)
    tr   = pd.concat([hi - lo, (hi - prev).abs(), (lo - prev).abs()], axis=1).max(axis=1)
    atr  = float(tr.mean())
    return max(atr * 0.5, default)   # al menos el fallback mínimo


def _get_date(
    d_row: pd.Series,
    daily: Optional[pd.DataFrame],
    bar_index: int,
) -> Optional[datetime]:
    """Extrae la fecha del punto D del swing o del df original."""
    if "date" in d_row.index:
        v = d_row["date"]
        if pd.notna(v):
            return pd.Timestamp(v).to_pydatetime()
    if daily is not None and bar_index < len(daily) and "date" in daily.columns:
        v = daily["date"].iloc[bar_index]
        if pd.notna(v):
            return pd.Timestamp(v).to_pydatetime()
    return None


# ── Zona PRZ para overlays de dashboard ───────────────────────────────────────

def prz_boxes(formations: list[PeakFormation]) -> list[dict]:
    """Convierte las PeakFormations en boxes para _candlestick_chart.

    Devuelve una lista de dicts con x0/x1/y0/y1/color/opacity/label.
    x0=None hace que el chart use el primer bar como borde izquierdo.
    """
    boxes: list[dict] = []
    for pf in formations:
        if not pf.valid:
            continue
        color   = "#26a69a" if pf.direction == "bullish" else "#ef5350"
        x0_date = pf.date_d   # primer punto de la PRZ
        boxes.append({
            "x0":     x0_date,
            "x1":     None,       # extiende hasta el borde derecho del chart
            "y0":     pf.prz_lo,
            "y1":     pf.prz_hi,
            "color":  color,
            "opacity": 0.20,
            "label":  f"PRZ {pf.kind}",
        })
        # TP1 como línea horizontal (solo info visual)
        boxes.append({
            "x0": x0_date, "x1": None,
            "y0": pf.tp1 * 0.9995, "y1": pf.tp1 * 1.0005,   # franja fina
            "color": color,
            "opacity": 0.15,
            "label": f"TP1 {pf.kind}",
        })
    return boxes
