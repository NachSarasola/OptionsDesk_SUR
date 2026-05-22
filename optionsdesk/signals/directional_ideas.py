"""Ideas direccionales especulativas (BUY_CALL / BUY_PUT).

ADVERTENCIA: estas ideas son especulativas. El bot no tiene un modelo de alpha
direccional validado. Son sugerencias basadas en AT + SMC, no recomendaciones
de inversión. El usuario asume el riesgo total de la prima pagada.

build_directional_idea(context, chain, spot, snap) → DirectionalIdea | None
  - Devuelve None si la señal es débil o si no hay opciones líquidas adecuadas.
  - Gate adicional SMC: si hay CHoCH contra la tendencia → None (posible reversal).
  - Target: FVG sin llenar si existe cerca del spot, sino 2×ATR.
  - Stop: zona del Order Block si existe, sino 1×ATR.
  - Horizonte: swing de días, no scalping.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from optionsdesk.core.instruments import days_to_expiry
from optionsdesk.core.spreads import SpreadResult
from optionsdesk.data.providers.base import OptionsChain
from optionsdesk.signals.directional import MarketContext
from optionsdesk.signals.technical import FairValueGap, OrderBlock, TechnicalSnapshot

_DISCLAIMER = (
    "IDEA ESPECULATIVA — pérdida máxima = prima pagada. "
    "El bot no tiene alpha direccional validado. Operar con capital de riesgo."
)

_MIN_VOLUME       = 1           # volumen mínimo para considerar líquida la opción
_ATM_BAND_PCT     = 0.08        # rango ±8% del spot para buscar strike
_MIN_MOMENTUM_PCT = 1.5         # momentum mínimo para generar idea
_VALID_CONFIDENCE = {"alta", "media"}
_VALID_STRENGTH   = {"fuerte", "moderada"}

# Horizonte del swing: entre estos días buscamos el front-month
_MIN_DAYS = 7
_MAX_DAYS = 45

# Filtros SMC adicionales
_CHOCH_BLACKLIST = {"CHOCH_UP", "CHOCH_DOWN"}   # CHoCH contra-tendencia bloquea la idea


@dataclass
class DirectionalIdea:
    idea_type: str          # "BUY_CALL" | "BUY_PUT"
    symbol: str             # símbolo de la opción (ej. "GFGC8500OC")
    strike: float
    days_to_expiry: int
    mid_price: float        # prima mid al momento del análisis
    total_cost_ars: float   # prima * 100 (un contrato)
    breakeven: float        # spot en el que el comprador empieza a ganar
    max_loss_ars: float     # = total_cost_ars (pérdida máxima de comprar)
    target_spot: float      # objetivo de precio basado en ATR
    stop_spot: float        # stop sugerido basado en ATR
    rational: str           # por qué se sugiere esta idea
    disclaimer: str = _DISCLAIMER


def build_directional_idea(
    context: MarketContext,
    chain: OptionsChain,
    spot: float,
    snap: Optional[TechnicalSnapshot] = None,
) -> Optional[DirectionalIdea]:
    """Construye una idea direccional si la señal AT + SMC es suficientemente fuerte.

    Gates (devuelve None si cualquiera falla):
    - confidence en {'alta', 'media'}
    - signal_strength en {'fuerte', 'moderada'}
    - |momentum_pct| >= _MIN_MOMENTUM_PCT
    - No CHoCH contra la tendencia (señal de reversión pendiente)
    - Hay opción líquida ATM/OTM en el rango de días buscado

    Targets y stops:
    - Target: FVG sin llenar más cercano en la dirección del trade (si existe y es razonable),
              sino 2×ATR.
    - Stop:   Zona del Order Block (si existe), sino 1×ATR.
    """
    if context.confidence not in _VALID_CONFIDENCE:
        return None
    if context.signal_strength not in _VALID_STRENGTH:
        return None
    if abs(context.momentum_pct) < _MIN_MOMENTUM_PCT:
        return None

    is_call = context.trend in ("alcista", "ALCISTA")
    is_put  = context.trend in ("bajista", "BAJISTA")
    if not (is_call or is_put):
        return None

    # Si no se pasó snap explícito, intentamos obtenerlo del contexto
    if snap is None and context.snap is not None:
        snap = context.snap   # type: ignore[assignment]

    # Gate SMC: CHoCH contra la tendencia → estructura debilitada, no entramos
    if snap is not None and snap.choch is not None:
        if is_call and snap.choch == "CHOCH_DOWN":
            return None
        if is_put and snap.choch == "CHOCH_UP":
            return None

    opt_type_prefix = "C" if is_call else "V"
    candidates = _find_candidates(chain, spot, opt_type_prefix, is_call)
    if not candidates and not is_call:
        candidates = _find_candidates(chain, spot, "P", is_call)
    if not candidates:
        return None

    best = min(candidates, key=lambda x: abs(x["strike"] - spot))
    atr  = context.atr_pct / 100.0 * spot if context.atr_pct > 0 else spot * 0.025

    # ── Target: FVG > ATR si existe en la dirección del trade ────────────────
    target_spot, target_note = _resolve_target(snap, spot, is_call, atr)
    # ── Stop: Order Block si existe, sino 1×ATR ───────────────────────────────
    stop_spot, stop_note = _resolve_stop(snap, spot, is_call, atr)

    if is_call:
        breakeven = best["strike"] + best["mid"]
        idea_type = "BUY_CALL"
    else:
        breakeven = best["strike"] - best["mid"]
        idea_type = "BUY_PUT"

    rational = _build_rational(context, snap, best["strike"], best["days"], atr, spot,
                                target_note, stop_note)

    return DirectionalIdea(
        idea_type=idea_type,
        symbol=best["symbol"],
        strike=best["strike"],
        days_to_expiry=best["days"],
        mid_price=round(best["mid"], 2),
        total_cost_ars=round(best["mid"] * 100, 2),
        breakeven=round(breakeven, 2),
        max_loss_ars=round(best["mid"] * 100, 2),
        target_spot=round(target_spot, 2),
        stop_spot=round(stop_spot, 2),
        rational=rational,
    )


def build_directional_spread(
    context: MarketContext,
    chain: OptionsChain,
    spot: float,
    snap: Optional[TechnicalSnapshot] = None,
    expiry_calendar: Optional[dict] = None,
    benchmark=None,
) -> Optional[SpreadResult]:
    """Construye el mejor spread vertical en la direccion del contexto multi-TF.

    Gates propios (mas estrictos que el carry — aqui es especulativo):
    - confidence en {'alta', 'media'}
    - |momentum_pct| >= _MIN_MOMENTUM_PCT
    - mtf_alignment != 'conflicto'  (para especulativo si bloquea el conflicto)
    - Hay al menos un spread liquido en la direccion del contexto

    Devuelve None si la senial es debil o no hay spreads viables.
    """
    if context.confidence not in _VALID_CONFIDENCE:
        return None
    if abs(context.momentum_pct) < _MIN_MOMENTUM_PCT:
        return None

    # Para ideas especulativas el conflicto multi-TF es un gate duro
    alignment = getattr(context, "mtf_alignment", "neutral")
    if alignment == "conflicto":
        return None

    is_bullish = context.trend in ("alcista", "ALCISTA")
    is_bearish = context.trend in ("bajista", "BAJISTA")
    if not (is_bullish or is_bearish):
        return None

    # Gate SMC: CHoCH contra la tendencia
    if snap is None and context.snap is not None:
        snap = context.snap  # type: ignore[assignment]
    if snap is not None and snap.choch is not None:
        if is_bullish and snap.choch == "CHOCH_DOWN":
            return None
        if is_bearish and snap.choch == "CHOCH_UP":
            return None

    direction = "bullish" if is_bullish else "bearish"

    # Necesitamos un Benchmark para la tasa de descuento. Si no se pasa, usamos 0.
    from optionsdesk.core.benchmark import Benchmark as _Benchmark
    bm = benchmark if benchmark is not None else _Benchmark(caucion_tna_pct=0.0, days=1)

    from optionsdesk.strategies.vertical_spread import VerticalSpreadScanner
    scanner = VerticalSpreadScanner(expiry_calendar=expiry_calendar or {})
    spreads = scanner.scan(chain, bm, direction=direction)

    if not spreads:
        return None

    # El scanner ya ordena por risk_reward x PoP; tomamos el mejor
    return spreads[0]


# ── Internals ─────────────────────────────────────────────────────────────────

def _find_candidates(
    chain: OptionsChain,
    spot: float,
    opt_prefix: str,
    is_call: bool,
) -> list[dict]:
    """Busca opciones líquidas ATM/OTM con vencimiento en el rango de swing."""
    lo = spot * (1 - _ATM_BAND_PCT)
    hi = spot * (1 + _ATM_BAND_PCT)

    results = []
    for sym, quote in chain.options.items():
        if quote.bid <= 0 and quote.ask <= 0:
            continue
        if quote.volume < _MIN_VOLUME and quote.last <= 0:
            continue

        strike = _parse_strike(sym)
        if strike is None:
            continue
        if not (lo <= strike <= hi):
            continue

        # Para calls buscamos strikes >= spot (OTM o ATM); para puts, <= spot
        if is_call and strike < spot * 0.97:
            continue
        if not is_call and strike > spot * 1.03:
            continue

        try:
            days = days_to_expiry(sym)  # noqa: F841 — ya importado a nivel módulo
        except Exception:
            continue

        if not (_MIN_DAYS <= days <= _MAX_DAYS):
            continue

        mid = (quote.bid + quote.ask) / 2.0 if quote.bid > 0 and quote.ask > 0 else quote.last
        if mid <= 0:
            continue

        results.append({"symbol": sym, "strike": strike, "days": days, "mid": mid})

    return results


def _parse_strike(symbol: str) -> Optional[float]:
    """Extrae el strike del símbolo BYMA (ej. GFGC8500OC → 8500.0)."""
    import re
    # Formato típico: prefijo + strike + sufijo (OC/OV/etc.)
    m = re.search(r"(\d{3,6}(?:\.\d+)?)", symbol)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def _resolve_target(
    snap: Optional[TechnicalSnapshot],
    spot: float,
    is_call: bool,
    atr: float,
) -> tuple[float, str]:
    """Devuelve (target_price, nota). Prioriza FVG sin llenar en la dirección correcta."""
    if snap is not None and snap.nearest_fvg is not None:
        fvg = snap.nearest_fvg
        mid = (fvg.high + fvg.low) / 2
        fvg_is_above = mid > spot
        if is_call and fvg_is_above and mid <= spot * 1.15:
            return mid, f"FVG {fvg.type} sin llenar (${fvg.low:,.0f}–${fvg.high:,.0f})"
        if not is_call and not fvg_is_above and mid >= spot * 0.85:
            return mid, f"FVG {fvg.type} sin llenar (${fvg.low:,.0f}–${fvg.high:,.0f})"
    default = spot + 2.0 * atr if is_call else spot - 2.0 * atr
    return default, "2×ATR"


def _resolve_stop(
    snap: Optional[TechnicalSnapshot],
    spot: float,
    is_call: bool,
    atr: float,
) -> tuple[float, str]:
    """Devuelve (stop_price, nota). Prioriza Order Block como zona de invalidación."""
    if snap is not None and snap.order_block is not None:
        ob = snap.order_block
        if is_call and ob.type == "bullish":
            # Stop por debajo del OB bullish (si cae dentro del OB, la idea se invalida)
            stop = ob.low * 0.995
            return stop, f"bajo OB alcista (${ob.low:,.0f}–${ob.high:,.0f})"
        if not is_call and ob.type == "bearish":
            stop = ob.high * 1.005
            return stop, f"sobre OB bajista (${ob.low:,.0f}–${ob.high:,.0f})"
    default = spot - 1.0 * atr if is_call else spot + 1.0 * atr
    return default, "1×ATR"


def _build_rational(
    ctx: MarketContext,
    snap: Optional[TechnicalSnapshot],
    strike: float,
    days: int,
    atr: float,
    spot: float,
    target_note: str,
    stop_note: str,
) -> str:
    trend_txt = "alcista" if ctx.trend in ("alcista", "ALCISTA") else "bajista"
    bos_txt = ""
    if snap is not None and snap.bos is not None:
        bos_txt = f" BOS {snap.bos.replace('_', ' ').lower()} confirmado."
    return (
        f"GGAL tendencia {trend_txt} | momentum {ctx.momentum_pct:+.1f}% "
        f"({ctx.signal_strength}, confianza {ctx.confidence}).{bos_txt} "
        f"ATR {ctx.atr_pct:.1f}% = ${atr:,.0f}/día. "
        f"Strike K={strike:,.0f} a {days}d. "
        f"Objetivo: {target_note}. Stop: {stop_note}."
    )
