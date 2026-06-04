"""Gestión activa de posiciones de prima vendida.

Implementa las reglas de salida de una mesa de vendedores de prima:
- TAKE_PROFIT a ~50% de la prima (libera capital, evita gamma de fin de ciclo)
- STOP a 2× el crédito cobrado (limita la pérdida máxima)
- ROLL cuando quedan ≤21 DTE con extrínseco relevante (antes del gamma blow-up)
- DEFEND cuando el delta de la posición short supera 0.50 (strike testeado)
- HOLD si ninguna condición anterior se cumple

La prioridad es fija: TAKE_PROFIT > STOP > ROLL > DEFEND > HOLD.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import date, datetime, time as dt_time
from enum import Enum
from typing import Optional, TYPE_CHECKING

logger = logging.getLogger(__name__)

from optionsdesk.core.pricing import crr_delta, crr_price

if TYPE_CHECKING:
    from optionsdesk.signals.monitor import OpenPosition

_CRR_STEPS = 50


class SignalType(str, Enum):
    TAKE_PROFIT = "TAKE_PROFIT"
    STOP        = "STOP"
    TIME_STOP   = "TIME_STOP"
    SESSION_CLOSE = "SESSION_CLOSE"
    ROLL        = "ROLL"
    DEFEND      = "DEFEND"
    HOLD        = "HOLD"


@dataclass
class ManagementSignal:
    signal_type: SignalType
    reason: str
    suggested_action: str
    capture_pct: float          # % de la prima ya capturado
    current_delta: Optional[float] = None


def _current_option_value(
    pos: "OpenPosition",
    current_spot: float,
    current_iv: Optional[float],
) -> float:
    """Valor actual de la opción (CRR) para calcular el P&L."""
    T_rem = pos.days_remaining / 365.0
    iv = current_iv or pos.iv_entry
    r  = math.log(1.0 + pos.caucion_tna / 100.0)

    if T_rem < 1.0 / 365.0 or iv <= 0:
        return (
            max(current_spot - pos.strike, 0.0)
            if pos.opt_type == "C"
            else max(pos.strike - current_spot, 0.0)
        )
    return crr_price(current_spot, pos.strike, T_rem, r, iv, pos.opt_type, steps=_CRR_STEPS)


def _current_delta_abs(
    pos: "OpenPosition",
    current_spot: float,
    current_iv: Optional[float],
) -> Optional[float]:
    """Delta absoluto de la opción escrita. None si no se puede calcular."""
    T_rem = pos.days_remaining / 365.0
    iv = current_iv or pos.iv_entry
    r  = math.log(1.0 + pos.caucion_tna / 100.0)

    if T_rem < 1.0 / 365.0 or iv <= 0 or current_spot <= 0:
        return None
    try:
        d = crr_delta(current_spot, pos.strike, T_rem, r, iv, pos.opt_type, steps=_CRR_STEPS)
        return abs(d)
    except (ValueError, ArithmeticError, ZeroDivisionError) as e:
        logger.debug("_current_delta_abs fallo para %s: %s", pos.symbol, e)
        return None


def _capture_pct(pos: "OpenPosition", current_ov: float) -> float:
    """% de la prima ya capturada (0–100+)."""
    if pos.premium_received <= 0:
        return 0.0
    pnl = pos.premium_received - current_ov
    return max(pnl / pos.premium_received * 100.0, 0.0)


def evaluate_scalp_quote_position(
    pos: "OpenPosition",
    current_option_price: float,
    now: Optional[datetime] = None,
) -> ManagementSignal:
    """Evalua un scalp contra su quote ejecutable y su reloj intradia."""
    strategy = str(pos.strategy).upper()
    if "SCALP" not in strategy:
        raise ValueError("evaluate_scalp_quote_position requiere una posicion SCALP")

    mark = max(float(current_option_price), 0.0)
    is_long = "LONG" in strategy
    entry = float(pos.scalp_plan_entry or (pos.net_outlay if is_long else pos.premium_received) or 0.0)
    sl = pos.scalp_plan_sl
    tp = pos.scalp_plan_tp
    if entry <= 0:
        return ManagementSignal(
            signal_type=SignalType.HOLD,
            reason="Falta el precio de entrada del scalp.",
            suggested_action="Revisa la carga local antes de decidir.",
            capture_pct=0.0,
        )

    pnl_pct = ((mark - entry) if is_long else (entry - mark)) / entry * 100.0
    if tp is not None and ((is_long and mark >= tp) or (not is_long and mark <= tp)):
        return ManagementSignal(
            signal_type=SignalType.TAKE_PROFIT,
            reason=f"Quote ejecutable ${mark:,.2f}: objetivo ${tp:,.2f} alcanzado.",
            suggested_action=f"Cerra {pos.symbol} con orden limite y marca la posicion como cerrada.",
            capture_pct=pnl_pct,
        )
    if sl is not None and ((is_long and mark <= sl) or (not is_long and mark >= sl)):
        return ManagementSignal(
            signal_type=SignalType.STOP,
            reason=f"Quote ejecutable ${mark:,.2f}: stop ${sl:,.2f} alcanzado.",
            suggested_action=f"Cerra {pos.symbol}; el setup intradia dejo de ser valido.",
            capture_pct=pnl_pct,
        )

    current = now or datetime.now(tz=pos.opened_at.tzinfo if pos.opened_at else None)
    if (
        not getattr(pos, "scalp_allow_overnight", False)
        and current.weekday() < 5
        and current.time() >= dt_time(16, 50)
    ):
        return ManagementSignal(
            signal_type=SignalType.SESSION_CLOSE,
            reason="Quedan menos de 10 min de rueda y el scalp fue abierto como intradia.",
            suggested_action=f"Cerra {pos.symbol}; no conviertas un scalp en overnight por inercia.",
            capture_pct=pnl_pct,
        )

    opened_at = pos.opened_at
    time_stop_min = pos.scalp_time_stop_min
    if opened_at is not None and time_stop_min is not None and time_stop_min > 0:
        current = now or datetime.now(tz=opened_at.tzinfo)
        if opened_at.tzinfo is None and current.tzinfo is not None:
            current = current.replace(tzinfo=None)
        held_min = max((current - opened_at).total_seconds() / 60.0, 0.0)
        if held_min >= time_stop_min:
            return ManagementSignal(
                signal_type=SignalType.TIME_STOP,
                reason=f"Pasaron {held_min:.0f} min sin salida; time stop {time_stop_min} min.",
                suggested_action=f"Revalida el momentum y prioriza cerrar {pos.symbol}.",
                capture_pct=pnl_pct,
            )

    return ManagementSignal(
        signal_type=SignalType.HOLD,
        reason=f"Scalp activo: quote ejecutable ${mark:,.2f}, P&L bruto {pnl_pct:+.1f}%.",
        suggested_action="Mantenelo solo mientras el setup minuto a minuto siga vigente.",
        capture_pct=pnl_pct,
    )


def evaluate_position(
    pos: "OpenPosition",
    current_spot: float,
    current_iv: Optional[float] = None,
    today: Optional[date] = None,
) -> ManagementSignal:
    """Evalúa una posición abierta y devuelve la señal de gestión.

    Prioridad fija: TAKE_PROFIT → STOP → ROLL → DEFEND → HOLD.

    Args:
        pos: posición abierta (OpenPosition).
        current_spot: precio actual del subyacente.
        current_iv: IV actual de la opción; si es None usa iv_entry.
        today: fecha actual para calcular días restantes; None = date.today().
    """
    # Calcular días restantes respecto a today (permite testing con fecha fija)
    if today is not None:
        days_remaining = max(pos.days_entry - (today - pos.entry_date).days, 0)
    else:
        days_remaining = pos.days_remaining

    current_ov = _current_option_value(pos, current_spot, current_iv)
    capture = _capture_pct(pos, current_ov)
    delta_abs = _current_delta_abs(pos, current_spot, current_iv)
    is_long = "LONG" in str(pos.strategy).upper()

    target_capture = pos.target_capture_pct
    max_loss_mult  = pos.max_loss_mult
    roll_dte       = pos.roll_dte
    defend_delta   = pos.defend_delta

    if is_long:
        entry_debit = max(float(pos.net_outlay), 0.0)
        if entry_debit <= 0:
            return ManagementSignal(
                signal_type=SignalType.HOLD,
                reason="No se puede evaluar el scalp comprado porque falta el débito de entrada.",
                suggested_action="Revisá la carga de la posición antes de decidir.",
                capture_pct=0.0,
                current_delta=delta_abs,
            )

        pnl_pct = (current_ov - entry_debit) / entry_debit * 100.0
        target_gain_pct = max(float(target_capture), 15.0)
        stop_loss_pct = 35.0 if max_loss_mult >= 1.0 else max_loss_mult * 100.0

        if pnl_pct >= target_gain_pct:
            return ManagementSignal(
                signal_type=SignalType.TAKE_PROFIT,
                reason=(
                    f"El scalp comprado gana {pnl_pct:.1f}% sobre la prima pagada "
                    f"(objetivo: {target_gain_pct:.0f}%)."
                ),
                suggested_action=(
                    f"Cerrá {pos.symbol} con orden límite. En scalping conviene asegurar "
                    "la expansión antes de que se evapore el movimiento."
                ),
                capture_pct=pnl_pct,
                current_delta=delta_abs,
            )

        if pnl_pct <= -stop_loss_pct:
            return ManagementSignal(
                signal_type=SignalType.STOP,
                reason=(
                    f"El scalp comprado pierde {abs(pnl_pct):.1f}% de la prima pagada "
                    f"(stop operativo: {stop_loss_pct:.0f}%)."
                ),
                suggested_action=(
                    f"Cerrá o achicá {pos.symbol}. Si el rebote/ola no apareció, "
                    "no dejes que el theta convierta el scalp en esperanza."
                ),
                capture_pct=pnl_pct,
                current_delta=delta_abs,
            )

        if days_remaining <= 2 and current_ov > 0.01:
            return ManagementSignal(
                signal_type=SignalType.ROLL,
                reason=(
                    f"Quedan {days_remaining} días al vencimiento. En una opción comprada "
                    "el gamma y el spread pueden dominar el trade."
                ),
                suggested_action=(
                    f"Definí salida para {pos.symbol}: cerrar, tomar parcial o pasar a "
                    "un vencimiento con más aire si el contexto sigue válido."
                ),
                capture_pct=pnl_pct,
                current_delta=delta_abs,
            )

        return ManagementSignal(
            signal_type=SignalType.HOLD,
            reason=(
                f"Scalp comprado sin gatillo: P&L {pnl_pct:+.1f}%, "
                f"días restantes: {days_remaining}."
            ),
            suggested_action="Mantené solo si el setup minuto a minuto sigue vigente.",
            capture_pct=pnl_pct,
            current_delta=delta_abs,
        )

    # 1. TAKE_PROFIT
    if capture >= target_capture:
        return ManagementSignal(
            signal_type=SignalType.TAKE_PROFIT,
            reason=f"Capturaste el {capture:.1f}% de la prima (objetivo: {target_capture:.0f}%).",
            suggested_action=(
                f"Cerrá la posición {pos.symbol} recomprando la opción. "
                "Liberás el capital para el próximo ciclo."
            ),
            capture_pct=capture,
            current_delta=delta_abs,
        )

    # 2. STOP — pérdida = prima cobrada × max_loss_mult
    # La opción vale más de (1 + max_loss_mult) × prima → perdemos > max_loss_mult × prima
    stop_threshold = pos.premium_received * (1.0 + max_loss_mult)
    if current_ov >= stop_threshold:
        loss = current_ov - pos.premium_received
        return ManagementSignal(
            signal_type=SignalType.STOP,
            reason=(
                f"La opción vale ${current_ov:,.2f} — pérdida de ${loss:,.2f} "
                f"({loss / pos.premium_received:.1f}× la prima cobrada)."
            ),
            suggested_action=(
                f"Cerrá la posición {pos.symbol} ahora. "
                f"El stop está a {max_loss_mult:.1f}× la prima recibida."
            ),
            capture_pct=capture,
            current_delta=delta_abs,
        )

    # 3. ROLL — quedan ≤ roll_dte días y la opción aún tiene valor extrínseco
    if days_remaining <= roll_dte and current_ov > 0.01:
        return ManagementSignal(
            signal_type=SignalType.ROLL,
            reason=(
                f"Quedan {days_remaining} días al vencimiento (umbral: {roll_dte} DTE). "
                "El gamma se acelera — el riesgo/recompensa deteriora."
            ),
            suggested_action=(
                f"Rolá {pos.symbol} al próximo vencimiento con delta similar. "
                "Recomprá la opción actual y vendé el mismo strike (o ajustado) más lejos."
            ),
            capture_pct=capture,
            current_delta=delta_abs,
        )

    # 4. DEFEND — delta supera el umbral (strike siendo testeado)
    if delta_abs is not None and delta_abs >= defend_delta:
        return ManagementSignal(
            signal_type=SignalType.DEFEND,
            reason=(
                f"Delta de la opción escrita = {delta_abs:.2f} "
                f"(umbral: {defend_delta:.2f}) — el strike está siendo testeado."
            ),
            suggested_action=(
                f"Revisá la posición {pos.symbol}. Considerá rolarte a un strike más OTM "
                "o agregar una cobertura si el contexto direccional es adverso."
            ),
            capture_pct=capture,
            current_delta=delta_abs,
        )

    return ManagementSignal(
        signal_type=SignalType.HOLD,
        reason=f"Sin señal de acción. Captura: {capture:.1f}%. Días restantes: {days_remaining}.",
        suggested_action="Mantené la posición.",
        capture_pct=capture,
        current_delta=delta_abs,
    )
