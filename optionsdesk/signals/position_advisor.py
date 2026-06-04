"""Analizador intertemporal de posiciones abiertas.

Para cada posición abierta calcula NO solo el estado actual (ya lo hace
management.evaluate_position) sino la TRAYECTORIA FUTURA:

  - ¿Cuánto theta queda por capturar y en cuántos días?
  - ¿La captura va por buen ritmo o está rezagada respecto al plan?
  - ¿Conviene cerrar ya (liberar capital) y rotar a una oportunidad mejor?
  - ¿Qué pasa con el P&L en t+3, t+7, t+14 días si el spot no se mueve?
  - ¿Hay riesgo de delta drift que transforme la operación antes del trigger?

El módulo es PURO (sin dependencias de Streamlit ni de config global) para
poder testearlo independientemente del dashboard.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date
from typing import Optional, TYPE_CHECKING

from optionsdesk.config.costs import DEFAULT_COSTS
from optionsdesk.core.pricing import crr_price, crr_delta, crr_theta

if TYPE_CHECKING:
    from optionsdesk.signals.monitor import OpenPosition
    from optionsdesk.signals.recommender import Recommendation

logger = logging.getLogger(__name__)

_CRR_STEPS = 50
_THETA_ACCEL_DTE = 21   # DTE a partir del cual la theta se acelera (gamma blow-up)
_ROTATE_MIN_EV_GAP = 5.0   # % de TNA extra mínima para sugerir rotación


# ── Dataclasses de salida ────────────────────────────────────────────────────

@dataclass
class TimeScenario:
    """P&L proyectado en un punto futuro asumiendo spot congelado (theta puro)."""
    days_from_now: int
    dte_remaining: int           # días al vencimiento en ese punto
    option_value: float          # valor CRR de la opción
    pnl_ars: float               # P&L del leg de opción por contrato (100 acc.)
    capture_pct: float           # % capturado sobre prima total
    theta_daily_ars: float       # theta proyectado en ese punto


@dataclass
class PositionAdvice:
    """Consejo intertemporal completo para una posición abierta."""

    position: "OpenPosition"
    # ── Decisión principal ────────────────────────────────────────────
    action: str          # "HOLD" | "CLOSE_NOW" | "ROLL" | "DEFEND" | "ROTATE" | "PARTIAL"
    urgency: str         # "alta" | "media" | "baja"
    headline: str        # una línea accionable en lenguaje llano
    reason: str          # por qué (plain Spanish)

    # ── Estado actual ─────────────────────────────────────────────────
    capture_pct: float              # % de prima capturado ahora
    current_delta: Optional[float]  # delta absoluto actual
    dte: int                        # días restantes al vencimiento

    # ── Trayectoria temporal ──────────────────────────────────────────
    scenarios: list[TimeScenario]       # P&L proyectado a t+3, t+7, t+14, t+21
    days_to_target: Optional[int]       # días estimados hasta alcanzar target capture
    plan_on_track: bool                 # ¿la captura va al ritmo esperado?
    capture_lag_pct: float              # cuánto atrás está vs el plan (negativo = adelantado)

    # ── Oportunidad de rotación ───────────────────────────────────────
    rotate_to: Optional[str]               # símbolo alternativo sugerido
    rotation_tna_gain_pct: Optional[float] # TNA extra al rotar
    rotation_rationale: Optional[str]      # por qué rotar

    # ── Score para priorizar en el dashboard ─────────────────────────
    score: float           # 0–100; mayor = más urgente de revisar
    risk_flags: list[str] = field(default_factory=list)


@dataclass
class PortfolioAdvice:
    """Resumen del libro completo ordenado por urgencia."""
    advices: list[PositionAdvice]
    total_capture_ars: float      # P&L acumulado (leg opción) en el libro
    theta_daily_ars: float        # theta total del libro hoy
    highest_urgency: str          # "alta" | "media" | "baja" | "ninguna"
    portfolio_flags: list[str]    # alertas agregadas del libro


# ── Computación de escenarios temporales ─────────────────────────────────────

def _r_from_tna(caucion_tna: float) -> float:
    return math.log(1.0 + max(caucion_tna, 0.0) / 100.0)


def project_scenarios(
    pos: "OpenPosition",
    current_spot: float,
    current_iv: Optional[float] = None,
    days_list: Optional[list[int]] = None,
) -> list[TimeScenario]:
    """P&L proyectado asumiendo spot congelado (θ-decay puro).

    Congela el spot porque el módulo no hace predicción direccional — el
    objetivo es mostrar cuánto capturás SI el subyacente no se mueve, que
    es el escenario central de una venta de prima neutral.
    """
    if days_list is None:
        days_list = [3, 7, 14, 21]

    iv = current_iv if (current_iv and current_iv > 0) else pos.iv_entry
    r = _r_from_tna(pos.caucion_tna)
    S = current_spot if current_spot > 0 else pos.spot_entry
    K = pos.strike
    opt = pos.opt_type
    premium = pos.premium_received
    contracts = max(pos.contracts, 1)

    results: list[TimeScenario] = []
    for d in sorted(days_list):
        dte = max(pos.days_remaining - d, 0)
        T = dte / 365.0

        if T < 1.0 / 365.0 or iv <= 0:
            ov = max(S - K, 0.0) if opt == "C" else max(K - S, 0.0)
            theta_d = 0.0
        else:
            ov = crr_price(S, K, T, r, iv, opt, steps=_CRR_STEPS)
            # Theta: diferencia de precio entre T y T-1d
            T_prev = max(T - 1.0 / 365.0, 0.0)
            if T_prev > 0:
                ov_prev = crr_price(S, K, T_prev, r, iv, opt, steps=_CRR_STEPS)
                theta_d = (ov - ov_prev) * contracts * 100.0  # negativo para vendedor
            else:
                theta_d = 0.0

        pnl = (premium - ov) * contracts * 100.0
        cap = max(pnl / (premium * contracts * 100.0) * 100.0, 0.0) if premium > 0 else 0.0

        results.append(TimeScenario(
            days_from_now=d,
            dte_remaining=dte,
            option_value=round(ov, 2),
            pnl_ars=round(pnl, 0),
            capture_pct=round(cap, 1),
            theta_daily_ars=round(abs(theta_d), 0),
        ))
    return results


# ── Diagnóstico del ritmo de captura ─────────────────────────────────────────

def _capture_plan_lag(
    pos: "OpenPosition",
    current_capture: float,
) -> tuple[bool, float]:
    """¿La captura va al ritmo esperado del plan?

    Retorna (on_track, lag_pct).
    lag_pct > 0 significa que la captura va ADELANTADA al plan.
    lag_pct < 0 significa que va REZAGADA.
    """
    if pos.days_entry <= 0:
        return True, 0.0

    days_elapsed = pos.days_held
    if days_elapsed <= 0:
        return True, 0.0

    # Ritmo lineal esperado hasta el target (si era SWING) o hasta full
    target = pos.target_capture_pct
    expected_now = target * (days_elapsed / pos.days_entry)
    lag = current_capture - expected_now
    on_track = lag >= -10.0   # tolerar hasta 10pp de retraso
    return on_track, round(lag, 1)


# ── Días estimados hasta alcanzar el target ───────────────────────────────────

def _days_to_target(
    pos: "OpenPosition",
    current_capture: float,
    scenarios: list[TimeScenario],
) -> Optional[int]:
    """Estima en cuántos días (desde hoy) se alcanzará el target capture.

    Usa la curva de escenarios proyectados (theta decay puro).
    Retorna None si ningún escenario lo supera (theta no alcanza).
    """
    target = pos.target_capture_pct
    if current_capture >= target:
        return 0
    for sc in sorted(scenarios, key=lambda s: s.days_from_now):
        if sc.capture_pct >= target:
            return sc.days_from_now
    return None


# ── Oportunidad de rotación ───────────────────────────────────────────────────

def _rotation_opportunity(
    pos: "OpenPosition",
    current_capture: float,
    caucion_tna: float,
    available_recs: Optional[list["Recommendation"]],
) -> tuple[Optional[str], Optional[float], Optional[str]]:
    """¿Hay una oportunidad mejor a la que rotar?

    Compara la TNA *residual* de la posición (TNA que quedaría por ganar si
    la cerraras hoy y abrieras algo nuevo) vs la TNA de las mejores
    recomendaciones disponibles.

    Retorna (symbol, tna_gain_pct, rationale) o (None, None, None).
    """
    if not available_recs:
        return None, None, None

    # TNA residual: si ya capturaste X%, la TNA de seguir holding es sobre
    # la prima RESTANTE (1 - capture%) en los días que quedan.
    remaining_pct = max(100.0 - current_capture, 0.0) / 100.0
    remaining_premium = pos.premium_received * remaining_pct * pos.contracts * 100.0
    capital = pos.net_outlay * pos.contracts * 100.0
    dte = pos.days_remaining

    if capital <= 0 or dte <= 0 or remaining_premium <= 0:
        return None, None, None

    residual_tna = (remaining_premium / capital) * (365.0 / dte) * 100.0

    # La mejor recomendación disponible (ya scorer por el recommender)
    best = max(available_recs, key=lambda r: r.result.tna_pct, default=None)
    if best is None:
        return None, None, None

    tna_gain = best.result.tna_pct - residual_tna
    if tna_gain < _ROTATE_MIN_EV_GAP:
        return None, None, None

    rationale = (
        f"TNA residual de la posicion actual: {residual_tna:.1f}%. "
        f"{best.result.symbol} ofrece {best.result.tna_pct:.1f}% "
        f"(+{tna_gain:.1f}pp). Cerrar y rotar mejora la tasa anualizada del capital."
    )
    return best.result.symbol, round(tna_gain, 1), rationale


# ── Scoring de urgencia ───────────────────────────────────────────────────────

_ACTION_SCORE: dict[str, float] = {
    "CLOSE_NOW":  90.0,
    "ROLL":       75.0,
    "DEFEND":     70.0,
    "ROTATE":     55.0,
    "PARTIAL":    45.0,
    "HOLD":       10.0,
}

_URGENCY_THRESHOLDS = {"alta": 65.0, "media": 40.0}


def _urgency(score: float) -> str:
    if score >= _URGENCY_THRESHOLDS["alta"]:
        return "alta"
    if score >= _URGENCY_THRESHOLDS["media"]:
        return "media"
    return "baja"


# ── Motor principal ───────────────────────────────────────────────────────────

def advise_position(
    pos: "OpenPosition",
    current_spot: float,
    current_iv: Optional[float] = None,
    context=None,
    available_recs: Optional[list["Recommendation"]] = None,
    caucion_tna: float = 60.0,
) -> PositionAdvice:
    """Genera un consejo intertemporal completo para una posición abierta.

    Args:
        pos: la posición abierta (OpenPosition).
        current_spot: precio actual del subyacente.
        current_iv: IV actual de la opción; None → usa iv_entry.
        context: MarketContext de compute_market_context (opcional).
        available_recs: lista de Recommendation del Recommender para oportunidad cost.
        caucion_tna: TNA de caución actual (%).
    """
    from optionsdesk.signals.management import evaluate_position, SignalType
    from optionsdesk.signals.monitor import PositionMonitor

    monitor = PositionMonitor()
    risk_flags: list[str] = []

    # ── 1. Señal de gestión actual (reglas base de management) ───────────────
    mgmt = evaluate_position(pos, current_spot, current_iv)
    capture = mgmt.capture_pct
    delta_abs = mgmt.current_delta
    dte = pos.days_remaining

    # ── 2. Escenarios temporales (theta decay) ───────────────────────────────
    scenarios = project_scenarios(pos, current_spot, current_iv)

    # ── 3. Ritmo de captura vs plan ──────────────────────────────────────────
    on_track, capture_lag = _capture_plan_lag(pos, capture)
    days_to_t = _days_to_target(pos, capture, scenarios)

    # ── 4. Oportunidad de rotación ───────────────────────────────────────────
    rotate_sym, rotate_gain, rotate_rationale = _rotation_opportunity(
        pos, capture, caucion_tna, available_recs
    )

    # ── 5. Contexto de mercado ────────────────────────────────────────────────
    context_flag: Optional[str] = None
    if context is not None:
        trend = getattr(context, "trend", None)
        conf = getattr(context, "confidence", "sin datos")
        if conf not in ("sin datos", "baja") and trend is not None:
            is_short_put = "PUT" in pos.strategy.upper() and "LONG" not in pos.strategy.upper()
            is_cc = "COVERED_CALL" in pos.strategy.upper() or "CALL" in pos.strategy.upper()
            if is_short_put and trend == "bajista" and conf in ("alta", "media"):
                context_flag = "tendencia bajista activa — put vendida en riesgo"
                risk_flags.append(context_flag)
            elif is_cc and trend == "alcista" and conf == "alta":
                context_flag = "tendencia alcista — posible ejercicio anticipado de la call"
                risk_flags.append(context_flag)

    # ── 6. Alertas por delta drift ────────────────────────────────────────────
    if delta_abs is not None:
        delta_entry_approx = 1.0 - (pos.target_capture_pct / 100.0)  # proxy
        drift = delta_abs - delta_entry_approx
        if drift > 0.15 and delta_abs < pos.defend_delta:
            risk_flags.append(
                f"delta en {delta_abs:.2f} y subiendo — ya al {drift*100:.0f}pp del umbral de defensa"
            )

    # ── 7. Alerta por theta cliff ─────────────────────────────────────────────
    if 0 < dte <= _THETA_ACCEL_DTE and capture < 70.0:
        risk_flags.append(
            f"quedan {dte} DTE con solo {capture:.0f}% capturado — gamma acelerando"
        )

    # ── 8. Plan rezagado ──────────────────────────────────────────────────────
    if not on_track and capture_lag < -20.0:
        risk_flags.append(
            f"captura {abs(capture_lag):.0f}pp rezagada respecto al ritmo esperado del plan"
        )

    # ── 9. Acción final y headline ────────────────────────────────────────────
    # La señal de management tiene prioridad para TAKE_PROFIT/STOP/ROLL/DEFEND.
    # ROTATE y PARTIAL son capas adicionales de este módulo.
    if mgmt.signal_type == SignalType.TAKE_PROFIT:
        action = "CLOSE_NOW"
        headline = f"Tomar ganancia: cerrá {pos.symbol} ({capture:.0f}% capturado)"
        reason = mgmt.reason

    elif mgmt.signal_type == SignalType.STOP:
        action = "CLOSE_NOW"
        headline = f"Stop: cerrá {pos.symbol} ahora para limitar la pérdida"
        reason = mgmt.reason

    elif mgmt.signal_type in (SignalType.TIME_STOP, SignalType.SESSION_CLOSE):
        action = "CLOSE_NOW"
        headline = f"Time stop: cerrá {pos.symbol} — venció el tiempo del trade"
        reason = mgmt.reason

    elif mgmt.signal_type == SignalType.ROLL:
        action = "ROLL"
        headline = f"Rolar {pos.symbol}: quedan {dte} DTE — pasá al próximo vencimiento"
        reason = mgmt.reason

    elif mgmt.signal_type == SignalType.DEFEND:
        action = "DEFEND"
        headline = f"Defender {pos.symbol}: delta {delta_abs:.2f} — el strike está siendo testeado"
        reason = mgmt.reason

    elif rotate_sym is not None and capture >= 40.0:
        # Si ya capturaste bastante y hay algo mejor, proponer rotación.
        action = "ROTATE"
        headline = (
            f"Considerar rotación: cerrá {pos.symbol} ({capture:.0f}% cap.) "
            f"→ abrí {rotate_sym} (+{rotate_gain:.1f}pp TNA)"
        )
        reason = rotate_rationale or ""

    elif not on_track and dte > _THETA_ACCEL_DTE and capture_lag < -25.0:
        # Plan muy rezagado con tiempo suficiente para salir sin perder más.
        action = "PARTIAL"
        headline = (
            f"Plan rezagado en {pos.symbol}: captura {capture:.0f}% vs esperado "
            f"{capture - capture_lag:.0f}% — evaluar salida parcial"
        )
        reason = (
            f"Llevas {pos.days_held}d y la captura va {abs(capture_lag):.0f}pp atrás del ritmo "
            f"esperado. Con {dte} DTE todavía hay tiempo para salir parcialmente y preservar capital."
        )

    else:
        action = "HOLD"
        if days_to_t is not None and days_to_t > 0:
            headline = (
                f"Mantener {pos.symbol}: en ~{days_to_t}d alcanzás el objetivo "
                f"({pos.target_capture_pct:.0f}% captura, hoy {capture:.0f}%)"
            )
        elif days_to_t == 0:
            headline = f"Objetivo alcanzado en {pos.symbol} — podés cerrar cuando quieras"
        else:
            headline = (
                f"Mantener {pos.symbol}: {capture:.0f}% capturado, "
                f"{dte}d restantes"
            )
        reason = (
            mgmt.reason if mgmt.reason else
            f"Captura {capture:.0f}% de {pos.target_capture_pct:.0f}% objetivo. "
            f"Sin señal de salida por ahora."
        )
        if context_flag:
            reason += f" Ojo: {context_flag}."

    base_score = _ACTION_SCORE.get(action, 10.0)
    # Ajustes por flags adicionales
    flag_bonus = min(len(risk_flags) * 8.0, 25.0)
    score = min(base_score + flag_bonus, 100.0)

    return PositionAdvice(
        position=pos,
        action=action,
        urgency=_urgency(score),
        headline=headline,
        reason=reason,
        capture_pct=round(capture, 1),
        current_delta=delta_abs,
        dte=dte,
        scenarios=scenarios,
        days_to_target=days_to_t,
        plan_on_track=on_track,
        capture_lag_pct=capture_lag,
        rotate_to=rotate_sym,
        rotation_tna_gain_pct=rotate_gain,
        rotation_rationale=rotate_rationale,
        score=round(score, 1),
        risk_flags=risk_flags,
    )


def advise_portfolio(
    positions: list["OpenPosition"],
    current_spot: float,
    current_iv_map: Optional[dict[str, float]] = None,
    context=None,
    available_recs: Optional[list["Recommendation"]] = None,
    caucion_tna: float = 60.0,
) -> PortfolioAdvice:
    """Analiza el libro completo y lo devuelve ordenado por urgencia.

    Args:
        positions: lista de OpenPosition (del JSONL).
        current_spot: precio actual del subyacente (GGAL).
        current_iv_map: dict símbolo → IV actual (de greeks_engine).
        context: MarketContext.
        available_recs: recomendaciones del Recommender (para opportunity cost).
        caucion_tna: TNA de caución actual (%).
    """
    iv_map = current_iv_map or {}
    advices: list[PositionAdvice] = []

    for pos in positions:
        iv = iv_map.get(pos.symbol)
        try:
            adv = advise_position(
                pos, current_spot, iv, context, available_recs, caucion_tna
            )
            advices.append(adv)
        except Exception as exc:
            logger.warning("advise_position fallo para %s: %s", pos.symbol, exc)

    # Ordenar por score descendente (más urgente primero)
    advices.sort(key=lambda a: a.score, reverse=True)

    # ── Métricas agregadas del libro ─────────────────────────────────────────
    from optionsdesk.signals.monitor import PositionMonitor
    monitor = PositionMonitor()
    theta_total = 0.0
    pnl_total = 0.0
    for adv in advices:
        pos = adv.position
        iv = iv_map.get(pos.symbol)
        try:
            capture_ars = (
                adv.capture_pct / 100.0
                * pos.premium_received * pos.contracts * 100.0
            )
            pnl_total += capture_ars
            # Theta diario: primer escenario (t+3) / 3
            if adv.scenarios:
                theta_total += adv.scenarios[0].theta_daily_ars
        except Exception:
            pass

    portfolio_flags: list[str] = []

    # Flag de concentración: todo es GGAL
    if len(positions) > 1:
        portfolio_flags.append(
            f"Libro 100% concentrado en GGAL ({len(positions)} posiciones). "
            "Un gap atraviesa todo el libro a la vez."
        )

    # Flag de delta neta agregada
    total_delta = sum(
        (adv.current_delta or 0.0) * adv.position.contracts
        for adv in advices
        if adv.current_delta is not None
    )
    if total_delta > 1.5:
        portfolio_flags.append(
            f"Delta neta del libro = {total_delta:.2f} contratos — exposición direccional alta."
        )

    # Flag de vencimientos próximos
    near_expiry = [a for a in advices if a.dte <= _THETA_ACCEL_DTE]
    if near_expiry:
        syms = ", ".join(a.position.symbol for a in near_expiry)
        portfolio_flags.append(
            f"{len(near_expiry)} posicion(es) con ≤{_THETA_ACCEL_DTE} DTE: {syms}."
        )

    highest = advices[0].urgency if advices else "ninguna"

    return PortfolioAdvice(
        advices=advices,
        total_capture_ars=round(pnl_total, 0),
        theta_daily_ars=round(theta_total, 0),
        highest_urgency=highest,
        portfolio_flags=portfolio_flags,
    )
