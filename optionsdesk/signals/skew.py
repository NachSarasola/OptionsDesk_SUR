"""Skew de volatilidad put/call para GGAL.

El put/call skew mide la diferencia de IV entre opciones OTM simétricas.
Positivo = puts caros vs calls (skew bajista, lo más común en acciones).
La señal de skew le dice al vendedor de prima qué lado del mercado paga más.

Toma `list[RateResult]` (ya con IV computada por los scanners), no la
cadena cruda `OptionsChain`, que solo tiene precios bid/ask sin IV.
"""
from __future__ import annotations

import re
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from optionsdesk.core.rates import RateResult

_SYM_RE = re.compile(r"^GFG([CV])(\d+(?:[.,]\d+)?)([A-Z]+)$")


def _parse_symbol(symbol: str):
    """Devuelve (opt_type, strike, expiry_code) o None si no matchea."""
    m = _SYM_RE.match(symbol)
    if not m:
        return None
    try:
        strike = float(m.group(2).replace(",", "."))
    except ValueError:
        return None
    return m.group(1), strike, m.group(3)


def put_call_skew(
    results: "list[RateResult]",
    spot: float,
    otm_pct: float = 0.05,
) -> Optional[float]:
    """IV(put OTM) − IV(call OTM) para opciones simétricamente OTM.

    Busca el put cuyo strike sea ≈ spot*(1-otm_pct) y el call ≈ spot*(1+otm_pct).
    Si no hay opciones dentro del ±2% de esos targets, devuelve None.

    Args:
        results: lista de RateResult del recommender (CC + SP juntos).
        spot: precio actual del subyacente.
        otm_pct: distancia OTM como fracción (default 5%).

    Returns:
        Skew en pp (positivo = puts caros). None si no hay datos suficientes.
    """
    if spot <= 0:
        return None

    put_target = spot * (1.0 - otm_pct)
    call_target = spot * (1.0 + otm_pct)
    tolerance = spot * 0.02   # ±2% de tolerancia

    best_put_iv: Optional[float] = None
    best_call_iv: Optional[float] = None
    best_put_dist = float("inf")
    best_call_dist = float("inf")

    for r in results:
        if r.iv is None or r.iv <= 0:
            continue
        parsed = _parse_symbol(r.symbol)
        if not parsed:
            continue
        opt_type, strike, _ = parsed

        if opt_type == "V":   # put
            dist = abs(strike - put_target)
            if dist <= tolerance and dist < best_put_dist:
                best_put_iv = r.iv
                best_put_dist = dist
        elif opt_type == "C":  # call
            dist = abs(strike - call_target)
            if dist <= tolerance and dist < best_call_dist:
                best_call_iv = r.iv
                best_call_dist = dist

    if best_put_iv is None or best_call_iv is None:
        return None

    return (best_put_iv - best_call_iv) * 100.0


def skew_signal(
    skew_pp: Optional[float],
    bearish_threshold: float = 5.0,
    bullish_threshold: float = -2.0,
) -> str:
    """Clasifica el skew como señal de sentimiento de mercado.

    Args:
        skew_pp: put/call skew en puntos porcentuales.
        bearish_threshold: skew ≥ este valor → "bearish" (puts caros, mercado defensivo).
        bullish_threshold: skew ≤ este valor → "bullish" (calls caros, mercado codicioso).

    Returns:
        "bearish" | "neutral" | "bullish" | "sin datos"
    """
    if skew_pp is None:
        return "sin datos"
    if skew_pp >= bearish_threshold:
        return "bearish"
    if skew_pp <= bullish_threshold:
        return "bullish"
    return "neutral"


def skew_for_expiry(
    results: "list[RateResult]",
    spot: float,
    expiry_code: str,
    otm_pct: float = 0.05,
) -> Optional[float]:
    """Como `put_call_skew` pero filtra solo un vencimiento específico.

    Útil para comparar skew entre vencimientos (estructura temporal del skew).
    """
    filtered = [
        r for r in results
        if _parse_symbol(r.symbol) and _parse_symbol(r.symbol)[2] == expiry_code
    ]
    return put_call_skew(filtered, spot, otm_pct)
