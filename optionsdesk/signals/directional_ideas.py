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

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

from optionsdesk.core.instruments import days_to_expiry
from optionsdesk.core.spreads import SpreadResult
from optionsdesk.data.providers.base import OptionsChain
from optionsdesk.signals.directional import MarketContext
from optionsdesk.signals.technical import TechnicalSnapshot

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

# Gestión de riesgo (medida sobre el subyacente, no sobre la prima)
_MIN_RR            = 1.5    # recompensa/riesgo mínima para emitir una idea
_STOP_ATR_CAP      = 2.0    # distancia máxima del stop, en múltiplos de ATR
_STOP_ATR_FLOOR    = 0.8    # distancia mínima del stop, para que no sea ruido
_TARGET_ATR_FALLBACK = 1.5  # objetivo = break-even ± k·ATR cuando no hay FVG válido
_FVG_TARGET_BAND   = 0.20   # un FVG vale como objetivo si está dentro del ±20% del BE


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
    prob_of_profit: float = 0.0   # v2.4: PoP física (drift=0, vol realizada)
    rr_ratio: float = 0.0         # recompensa/riesgo sobre el subyacente (objetivo vs stop)
    confluence: list = field(default_factory=list)  # factores SMC alineados (display)


def build_directional_idea(
    context: MarketContext,
    chain: OptionsChain,
    spot: float,
    snap: Optional[TechnicalSnapshot] = None,
    realized_vol: Optional[float] = None,   # v2.4: para PoP física
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

    # Gate multi-TF: no especular contra el marco temporal alto (paridad con el spread)
    if getattr(context, "mtf_alignment", "neutral") == "conflicto":
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

    # Gate ADR/CCL: si el quiebre local es solo movimiento de dólar → no especulamos
    smc_ctx = getattr(context, "smc", None)
    if smc_ctx is not None:
        adr_conf = getattr(smc_ctx, "adr_confluence", None)
        if adr_conf == "fx_driven":
            return None

    opt_type_prefix = "C" if is_call else "V"
    candidates = _find_candidates(chain, spot, opt_type_prefix, is_call)
    if not candidates and not is_call:
        candidates = _find_candidates(chain, spot, "P", is_call)
    if not candidates:
        return None

    best = min(candidates, key=lambda x: abs(x["strike"] - spot))
    atr  = context.atr_pct / 100.0 * spot if context.atr_pct > 0 else spot * 0.025

    if is_call:
        breakeven = best["strike"] + best["mid"]
        idea_type = "BUY_CALL"
    else:
        breakeven = best["strike"] - best["mid"]
        idea_type = "BUY_PUT"

    # ── Target: BSL/SSL de SMC (próxima zona de liquidez), fallback FVG/ATR ───
    target_spot, target_note = _resolve_target_smc(smc_ctx, snap, spot, is_call, atr, breakeven)
    # ── Stop: OTE zone o Order Block acotado a [floor, cap]×ATR ──────────────
    stop_spot, stop_note = _resolve_stop_smc(smc_ctx, snap, spot, is_call, atr)

    # ── Gate de calidad: R/R asimétrico sobre el subyacente ──────────────────
    potential = abs(target_spot - spot)
    risk      = abs(spot - stop_spot)
    rr        = potential / risk if risk > 0 else 0.0
    if rr < _MIN_RR:
        return None

    confluence = _confluence(context, snap, smc_ctx, is_call, target_note, stop_note, spot=spot)

    rational = _build_rational(context, snap, best["strike"], best["days"], atr, spot,
                                target_note, stop_note, rr)

    # PoP física: drift=0, vol realizada (o fallback al ATR como proxy de vol)
    pop = _naked_pop(spot, breakeven, best["days"], is_call, context, realized_vol)

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
        prob_of_profit=round(pop, 3),
        rr_ratio=round(rr, 2),
        confluence=confluence,
    )


def build_directional_spread(
    context: MarketContext,
    chain: OptionsChain,
    spot: float,
    snap: Optional[TechnicalSnapshot] = None,
    expiry_calendar: Optional[dict] = None,
    benchmark=None,
    realized_vol: Optional[float] = None,   # v2.4: vol realizada para EV/PoP
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
    spreads = scanner.scan(chain, bm, direction=direction, realized_vol=realized_vol)

    if not spreads:
        return None

    best = spreads[0]
    # Gate de EV: no recomendar spread con expectativa negativa
    if best.expected_value <= 0:
        return None

    return best


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

        strike = _parse_strike(sym, spot_hint=spot)
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
            days = days_to_expiry(sym)
        except Exception as e:
            logger.debug("days_to_expiry fallo para %s: %s", sym, e)
            continue

        if not (_MIN_DAYS <= days <= _MAX_DAYS):
            continue

        mid = (quote.bid + quote.ask) / 2.0 if quote.bid > 0 and quote.ask > 0 else quote.last
        if mid <= 0:
            continue

        results.append({"symbol": sym, "strike": strike, "days": days, "mid": mid})

    return results


def _parse_strike(symbol: str, spot_hint: Optional[float] = None) -> Optional[float]:
    """Extrae el strike usando el mismo regex que instruments.py (_TICKER_RE).

    Reemplaza el regex propio anterior que podia capturar digitos del prefijo.
    Para simbolos malformados devuelve None (el candidato se omite correctamente).
    """
    from optionsdesk.core.instruments import _TICKER_RE
    m = _TICKER_RE.match(symbol.strip().upper())
    if m:
        try:
            strike = float(m.group("strike").replace(",", "."))
            if spot_hint is not None and spot_hint > 0:
                while strike > spot_hint * 3:
                    strike /= 10.0
                while strike < spot_hint / 3:
                    strike *= 10.0
            return strike
        except ValueError:
            return None
    return None


def _resolve_target_smc(
    smc_ctx: object,
    snap: Optional[TechnicalSnapshot],
    spot: float,
    is_call: bool,
    atr: float,
    breakeven: float,
) -> tuple[float, str]:
    """Target usando BSL/SSL del SMC como primera opción, FVG como segunda, ATR fallback.

    Lógica SMC:
    - BUY_CALL: el precio sube a barrer el BSL más cercano (buy stops arriba).
      Target = primer BSL no barrido por encima del breakeven.
    - BUY_PUT:  el precio cae a barrer el SSL más cercano (sell stops abajo).
      Target = primer SSL no barrido por debajo del breakeven.
    """
    # 1er intento: BSL/SSL de SMC
    liquidity = list(getattr(smc_ctx, "liquidity", []) or []) if smc_ctx is not None else []
    if liquidity:
        want_kind = "BSL" if is_call else "SSL"
        candidates = [
            lv for lv in liquidity
            if getattr(lv, "kind", "") == want_kind
            and not getattr(lv, "swept", True)
            and (
                (is_call  and getattr(lv, "price", 0) > breakeven) or
                (not is_call and getattr(lv, "price", 0) < breakeven)
            )
            and abs(getattr(lv, "price", 0) - breakeven) / max(breakeven, 1e-9) <= _FVG_TARGET_BAND
        ]
        if candidates:
            best_lv = min(candidates, key=lambda lv: abs(getattr(lv, "price", 0) - breakeven))
            p     = getattr(best_lv, "price", 0.0)
            label = getattr(best_lv, "label", "")
            kind  = "BSL" if is_call else "SSL"
            return p, f"{kind} {label} ${p:,.0f} (zona de liquidez)"

    # 2do intento: FVG de technical.py (comportamiento original)
    fvgs = list(getattr(snap, "fvgs", []) or []) if snap is not None else []
    if fvgs:
        valid = []
        for fvg in fvgs:
            mid = (fvg.high + fvg.low) / 2
            if is_call and mid > breakeven and mid <= breakeven * (1 + _FVG_TARGET_BAND):
                valid.append((mid, fvg))
            elif not is_call and mid < breakeven and mid >= breakeven * (1 - _FVG_TARGET_BAND):
                valid.append((mid, fvg))
        if valid:
            mid, fvg = min(valid, key=lambda x: abs(x[0] - breakeven))
            return mid, f"FVG {fvg.type} sin llenar (${fvg.low:,.0f}–${fvg.high:,.0f})"

    # Fallback: breakeven ± k·ATR
    default = breakeven + _TARGET_ATR_FALLBACK * atr if is_call else breakeven - _TARGET_ATR_FALLBACK * atr
    return default, f"break-even + {_TARGET_ATR_FALLBACK:g}×ATR"


def _resolve_stop_smc(
    smc_ctx: object,
    snap: Optional[TechnicalSnapshot],
    spot: float,
    is_call: bool,
    atr: float,
) -> tuple[float, str]:
    """Stop usando OTE zone del SMC, luego Order Block, luego ATR.

    Lógica SMC:
    - BUY_CALL: si el precio cae bajo el extremo de la OTE bullish, la onda correctiva
      es más profunda de lo esperado → invalidación estructural.
    - BUY_PUT: si el precio sube sobre el extremo de la OTE bearish → invalidación.
    Acotado a [floor, cap]×ATR (un stop demasiado lejos no es operable con una opción).
    """
    structural = None
    note       = f"{_STOP_ATR_FLOOR:g}×ATR"

    # 1er intento: OTE zone del SMC
    ote = getattr(smc_ctx, "ote", None) if smc_ctx is not None else None
    if ote is not None:
        ote_dir = getattr(ote, "direction", "")
        ote_f886 = getattr(ote, "fib_886", 0.0)
        if is_call and ote_dir == "bullish" and ote_f886 > 0:
            structural = ote_f886 * 0.995   # bajo el 0.886 fib de la pierna bullish
            note = f"bajo OTE bullish 0.886 (${ote_f886:,.0f})"
        elif not is_call and ote_dir == "bearish" and ote_f886 > 0:
            structural = ote_f886 * 1.005
            note = f"sobre OTE bearish 0.886 (${ote_f886:,.0f})"

    # 2do intento: Order Block de technical.py
    if structural is None and snap is not None and snap.order_block is not None:
        ob = snap.order_block
        if is_call and ob.type == "bullish":
            structural = ob.low * 0.995
            note = f"bajo OB alcista (${ob.low:,.0f}–${ob.high:,.0f})"
        elif not is_call and ob.type == "bearish":
            structural = ob.high * 1.005
            note = f"sobre OB bajista (${ob.low:,.0f}–${ob.high:,.0f})"

    raw_dist = abs(spot - structural) if structural is not None else atr
    dist     = min(max(raw_dist, _STOP_ATR_FLOOR * atr), _STOP_ATR_CAP * atr)
    if structural is not None and dist < raw_dist:
        note += f", ajustado a {_STOP_ATR_CAP:g}×ATR"
    stop = spot - dist if is_call else spot + dist
    return stop, note


# Keep old names as aliases for backward compat in tests
def _resolve_target(snap, spot, is_call, atr, breakeven):
    return _resolve_target_smc(None, snap, spot, is_call, atr, breakeven)


def _resolve_stop(snap, spot, is_call, atr):
    return _resolve_stop_smc(None, snap, spot, is_call, atr)


def _confluence(
    context: "MarketContext",
    snap: Optional[TechnicalSnapshot],
    smc_ctx: object,
    is_call: bool,
    target_note: str,
    stop_note: str,
    spot: float = 0.0,
) -> list[str]:
    """Lista de factores SMC/AT alineados con la idea (solo informativo para el panel)."""
    factors: list[str] = []

    # Factores clásicos AT
    want_bos = "BOS_UP" if is_call else "BOS_DOWN"
    if snap is not None and snap.bos == want_bos:
        factors.append("BOS en dirección")
    if snap is not None and getattr(snap, "vol_confirms", False):
        factors.append("volumen confirma")
    alignment = getattr(context, "mtf_alignment", "neutral")
    if (is_call and alignment == "alineado_alcista") or (not is_call and alignment == "alineado_bajista"):
        factors.append("marcos temporales alineados")
    if context.signal_strength == "fuerte":
        factors.append("señal fuerte")

    # Target/stop origin
    if "BSL" in target_note or "SSL" in target_note:
        factors.append("objetivo en zona de liquidez institucional")
    elif "FVG" in target_note:
        factors.append("objetivo en FVG de liquidez")
    if "OTE" in stop_note:
        factors.append("stop en OTE bullish" if is_call else "stop en OTE bearish")
    elif "OB" in stop_note:
        factors.append("stop en Order Block")

    # Factores SMC extendidos
    if smc_ctx is not None:
        # PDA zone
        pda = getattr(smc_ctx, "pda", None)
        if pda is not None:
            zone = getattr(pda, "zone", None)
            if is_call and zone == "discount":
                factors.append("zona discount (comprar en el suelo)")
            elif not is_call and zone == "premium":
                factors.append("zona premium (vender en el techo)")

        # OTE — ¿el precio actual está dentro de la zona de entrada óptima?
        ote = getattr(smc_ctx, "ote", None)
        if ote is not None:
            ote_dir = getattr(ote, "direction", "")
            ote_lo  = getattr(ote, "lo", 0.0)
            ote_hi  = getattr(ote, "hi", 0.0)
            # spot se recibe como parámetro; fallback a sma_fast si no disponible
            _spot_ref = spot if spot > 0 else getattr(snap, "sma_fast", 0.0)
            if ote_lo > 0 and ote_hi > ote_lo and _spot_ref > 0:
                in_ote = ote_lo <= _spot_ref <= ote_hi
                if is_call and ote_dir == "bullish":
                    if in_ote:
                        factors.append(f"GGAL en OTE alcista ${ote_lo:,.0f}–${ote_hi:,.0f} (entrada optima)")
                    else:
                        factors.append(f"OTE bullish ${ote_lo:,.0f}–${ote_hi:,.0f} (zona objetivo)")
                elif not is_call and ote_dir == "bearish":
                    if in_ote:
                        factors.append(f"GGAL en OTE bajista ${ote_lo:,.0f}–${ote_hi:,.0f} (entrada optima)")
                    else:
                        factors.append(f"OTE bearish ${ote_lo:,.0f}–${ote_hi:,.0f} (zona objetivo)")

        # Sweeps con reversión
        sweeps = getattr(smc_ctx, "sweeps", [])
        for sw in sweeps:
            if not getattr(sw, "reversed", False):
                continue
            sw_dir = getattr(sw, "direction", "")
            if is_call and sw_dir == "down":
                factors.append("sweep de SSL + reversión alcista")
                break
            elif not is_call and sw_dir == "up":
                factors.append("sweep de BSL + reversión bajista")
                break

        # Peak formations M/W
        pfs = getattr(smc_ctx, "peak_formations", [])
        for pf in pfs:
            pf_dir = getattr(pf, "direction", "")
            if is_call and pf_dir == "bullish":
                factors.append("formación M (bullish peak)")
                break
            elif not is_call and pf_dir == "bearish":
                factors.append("formación W (bearish peak)")
                break

        # ADR confluence
        adr_conf = getattr(smc_ctx, "adr_confluence", None)
        if adr_conf == "confirmed":
            factors.append("ADR confirma quiebre")
        # fx_driven ya bloquea la idea antes de llegar aquí

    return factors


def _naked_pop(
    spot: float,
    breakeven: float,
    days: int,
    is_call: bool,
    context: "MarketContext",
    realized_vol: Optional[float],
) -> float:
    """PoP física de una opción desnuda (drift=0, vol realizada o proxy ATR)."""
    from optionsdesk.core.pricing import risk_neutral_prob_above
    T = max(days / 365.0, 1e-6)
    if realized_vol and realized_vol > 0:
        sigma = realized_vol
    else:
        # Fallback: proxy ATR→sigma. ATR es rango suavizado, NO desvío log
        # diario — multiplicar por sqrt(252) sobreestima la dispersión ~13%.
        # Acotado a [20%, 150%] para evitar artefactos: ATR=10% (crash mode)
        # daría 160% sin cap, produciendo PoP ≈ 0.5 para cualquier breakeven.
        sigma_proxy = context.atr_pct / 100.0 * (252 ** 0.5)
        sigma = min(max(sigma_proxy, 0.20), 1.50)
    p_above = risk_neutral_prob_above(spot, breakeven, T, 0.0, sigma)
    return p_above if is_call else (1.0 - p_above)


def _build_rational(
    ctx: MarketContext,
    snap: Optional[TechnicalSnapshot],
    strike: float,
    days: int,
    atr: float,
    spot: float,
    target_note: str,
    stop_note: str,
    rr: float,
) -> str:
    trend_txt = "alcista" if ctx.trend in ("alcista", "ALCISTA") else "bajista"
    bos_txt = ""
    if snap is not None and snap.bos is not None:
        bos_txt = f" BOS {snap.bos.replace('_', ' ').lower()} confirmado."

    # SMC context en el rational (zona PDA + ciclo MM)
    smc_txt = ""
    smc_ctx = getattr(ctx, "smc", None)
    if smc_ctx is not None:
        pda = getattr(smc_ctx, "pda", None)
        mm  = getattr(smc_ctx, "mm", None)
        parts = []
        if pda is not None:
            _zone_map = {"premium": "zona PREMIUM", "discount": "zona DISCOUNT", "equilibrium": "equilibrio PDA"}
            zone = getattr(pda, "zone", None)
            if zone:
                parts.append(_zone_map.get(zone, zone))
        if mm is not None:
            stack = getattr(mm, "stack", None)
            cross = getattr(mm, "ema13_cross_50", None)
            if stack:
                parts.append(f"EMAs {stack}")
            if cross == "crossing":
                parts.append("cruce EMA13/50")
        adr_conf = getattr(smc_ctx, "adr_confluence", None)
        if adr_conf == "confirmed":
            parts.append("ADR confirma")
        if parts:
            smc_txt = " SMC: " + ", ".join(parts) + "."

    return (
        f"GGAL tendencia {trend_txt} | momentum {ctx.momentum_pct:+.1f}% "
        f"({ctx.signal_strength}, confianza {ctx.confidence}).{bos_txt}{smc_txt} "
        f"ATR {ctx.atr_pct:.1f}% = ${atr:,.0f}/día. "
        f"Strike K={strike:,.0f} a {days}d. "
        f"Objetivo: {target_note}. Stop: {stop_note}. R/R {rr:.1f}:1."
    )
