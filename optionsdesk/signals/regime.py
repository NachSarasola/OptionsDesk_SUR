"""Regime de mercado diario — contexto de coyuntura para el MERVAL.

Responde: ¿el día de hoy es risk-on o risk-off para acciones argentinas?

Dos dimensiones complementarias:
1. Tendencia CCL: si el dólar implícito (precio local / ADR × ratio) está subiendo
   sostenidamente, los "breakouts" en pesos son espejismos. CCL alcista → más exigente.
2. Breadth de ADRs: cuántos de los 9 ADR tienen confluencia "confirmed" (el activo se
   mueve, no el dólar) vs "fx_driven". Si la mayoría está fx_driven, el mercado entero
   se está moviendo por el tipo de cambio — señal de cautela macro.

El regime modula la agresividad: risk-off sube el piso de grade mínimo en el demo y
baja el Kelly. No bloquea señales individuales (eso lo hace SMC + CCL por activo).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class DailyRegime:
    """Contexto de coyuntura del día para el MERVAL."""
    label: str                       # "risk-on" | "cauteloso" | "risk-off"
    ccl_trend: str                   # "estable" | "subiendo" | "bajando"
    adr_breadth: str                 # "confirmado" | "mixto" | "fx_driven"
    confirmed_count: int             # ADRs con confluencia confirmed
    fx_driven_count: int             # ADRs con confluencia fx_driven
    total_adrs: int
    kelly_mult: float                # multiplicador para half-Kelly (1.0=normal, 0.7=cauteloso)
    min_grade_override: Optional[str]  # "B" | "A" | None (el demo no acepta nada peor)
    summary: str

    def is_risk_off(self) -> bool:
        return self.label == "risk-off"

    def is_cautious(self) -> bool:
        return self.label in ("cauteloso", "risk-off")


def compute_daily_regime(
    symbols: Optional[list[str]] = None,
    *,
    ccl_days: int = 5,
) -> DailyRegime:
    """Computa el regime del día consultando el ADR de todos los símbolos del universo.

    Nunca lanza: devuelve un regime NEUTRO si falla cualquier parte.
    """
    try:
        from optionsdesk.config.settings import settings as _s
        adr_map = _s.adr_map()
        universe = symbols or list(adr_map.keys())
    except Exception:
        return _neutral_regime("no se pudo leer la configuración")

    confirmed = fx_driven = unknown = 0
    ccl_vals: list[float] = []

    for sym in universe:
        conf, delta = _assess_symbol_once(sym, adr_map, ccl_days)
        if delta is not None:
            ccl_vals.append(delta)
        if conf == "confirmed":
            confirmed += 1
        elif conf == "fx_driven":
            fx_driven += 1
        else:
            unknown += 1

    total = confirmed + fx_driven + unknown
    return _build_regime(confirmed, fx_driven, total, ccl_vals)


def _assess_symbol_once(
    sym: str,
    adr_map: dict,
    ccl_days: int,
) -> tuple[str, Optional[float]]:
    """Una sola llamada a adr_daily por símbolo — retorna (confluencia, ccl_delta%).

    Antes se llamaba dos veces (double-fetch); ahora se reutiliza el mismo DataFrame
    para calcular tanto el CCL trend como la confluencia ADR.
    """
    if sym not in adr_map:
        return "unknown", None
    try:
        adr_ticker, _ = adr_map[sym]
        from optionsdesk.data.providers.adr import adr_daily, adr_confluence
        lookback_days = max(ccl_days + 5, 10)
        adr_df = adr_daily(adr_ticker, days=lookback_days)
        if adr_df is None or len(adr_df) < 2:
            return "unknown", None

        # CCL proxy: cambio % del ADR en el período
        prev = float(adr_df["close"].iloc[-2])
        last = float(adr_df["close"].iloc[-1])
        delta = (last / prev - 1.0) * 100.0 if prev > 0 else None

        # Confluencia ADR (usa "up" como dirección de referencia)
        conf = adr_confluence("up", adr_df, lookback=3)
        return conf, delta
    except Exception as exc:
        logger.debug("_assess_symbol_once(%s) fallo: %s", sym, exc)
        return "unknown", None


def _build_regime(
    confirmed: int,
    fx_driven: int,
    total: int,
    ccl_deltas: list[float],
) -> DailyRegime:
    # CCL trend desde los cambios de ADR (proxy)
    if ccl_deltas:
        avg_delta = sum(ccl_deltas) / len(ccl_deltas)
        if avg_delta < -0.5:
            ccl_trend = "bajando"   # dólar bajando → pesos valen más → bueno para acciones
        elif avg_delta > 0.5:
            ccl_trend = "subiendo"  # dólar subiendo → espejismo probable
        else:
            ccl_trend = "estable"
    else:
        ccl_trend = "estable"

    # Breadth de ADRs
    if total == 0:
        adr_breadth = "mixto"
    elif fx_driven / max(total, 1) > 0.5:
        adr_breadth = "fx_driven"
    elif confirmed / max(total, 1) > 0.5:
        adr_breadth = "confirmado"
    else:
        adr_breadth = "mixto"

    # Label y parámetros
    if adr_breadth == "fx_driven" or ccl_trend == "subiendo":
        label       = "risk-off"
        kelly_mult  = 0.5
        min_grade   = "A"    # solo setups de alta calidad
        summary     = f"CCL sube + {fx_driven}/{total} ADRs fx_driven. Espejismo probable — solo señales A+."
    elif adr_breadth == "confirmado" and ccl_trend in ("estable", "bajando"):
        label       = "risk-on"
        kelly_mult  = 1.0
        min_grade   = None
        summary     = f"{confirmed}/{total} ADRs confirman movimiento real. CCL {ccl_trend}."
    else:
        label       = "cauteloso"
        kelly_mult  = 0.7
        min_grade   = "B"    # no operar señales C/D
        summary     = f"Señal mixta: {confirmed} conf, {fx_driven} fx, {total-confirmed-fx_driven} unknown. CCL {ccl_trend}."

    return DailyRegime(
        label=label,
        ccl_trend=ccl_trend,
        adr_breadth=adr_breadth,
        confirmed_count=confirmed,
        fx_driven_count=fx_driven,
        total_adrs=total,
        kelly_mult=kelly_mult,
        min_grade_override=min_grade,
        summary=summary,
    )


def _neutral_regime(reason: str) -> DailyRegime:
    return DailyRegime(
        label="cauteloso", ccl_trend="estable", adr_breadth="mixto",
        confirmed_count=0, fx_driven_count=0, total_adrs=0,
        kelly_mult=0.7, min_grade_override="B",
        summary=f"Regime desconocido: {reason}",
    )
