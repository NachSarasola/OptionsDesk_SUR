"""Optimizador hold-vs-swing intertemporal.

Decide si conviene holdear la posición corta hasta vencimiento o salir antes
(swing), maximizando el retorno anualizado ajustado por riesgo.

Núcleo del modelo
-----------------
Para cada día t en [1, D]:
  1. Reprecia la opción corta con crr_price en t*, con spot congelado en S0 e IV de entrada.
  2. P&L(t) = prima recibida − valor de recompra(t) − costos de cierre.
     r_ann(t) = P&L(t) / capital * 365 / t
  3. Riesgo(t): reprecia en el escenario adverso S0 * exp(-z*σ*√(t/365)).
     risk(t) = max(P&L_base(t) − P&L_adverso(t), 0) / capital
  4. Score: u(t) = r_ann(t) − λ * risk(t), con λ por perfil.
  5. t* = argmax u(t). Si t* ≥ 0.85*D → HOLD; si no → SWING.

Por qué no es HOLD casi siempre: la theta de una opción ATM *acelera* cerca del
vencimiento, pero el gamma también — cada día que aguantás añade riesgo
desproporcionado. El perfil Conservador (λ=2.0) sale temprano; el Agresivo
(λ=0.3) aguanta hasta capturar la theta del final.

El spot se congela (sin vista direccional): esto es honesto y quant-correcto.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from optionsdesk.config.costs import CostModel, DEFAULT_COSTS
from optionsdesk.core.pricing import crr_gamma, crr_price, crr_theta, crr_vega
from optionsdesk.core.rates import RateResult


# λ de aversión al riesgo por perfil (nombre == RiskProfile.value).
_LAMBDA: dict[str, float] = {
    "CONSERVADOR": 2.0,
    "EQUILIBRADO": 1.0,
    "AGRESIVO":    0.3,
}

_HOLD_THRESHOLD  = 0.85   # t* >= 85% del plazo restante → HOLD
_Z_STRESS        = 1.5    # z-score del escenario adverso (≈ 93° percentil)
_CRR_STEPS       = 50     # pasos del árbol CRR para la curva (velocidad > precisión de ejecución)


@dataclass
class HorizonPlan:
    """Plan óptimo de salida para una posición de opción corta."""

    mode: str                       # "HOLD" | "SWING"
    target_exit_days: int           # días óptimos desde entrada; = D si HOLD
    target_capture_pct: float       # P&L(t*) / P&L(D) * 100
    optimized_tna_pct: float        # TNA anualizada en t*
    theta_daily_ars: float          # decaimiento diario (ARS) por contrato de 100 acciones
    gamma: float                    # gamma al momento de entrada
    vega_1pct: float                # vega por 1 p.p. de movimiento en IV
    plain_explanation: str          # texto en castellano para el dashboard
    warnings: list[str] = field(default_factory=list)


def optimize_horizon(
    result: RateResult,
    profile: str,
    caucion_tna: float = 60.0,
    cost_model: CostModel = DEFAULT_COSTS,
    spread_buyback_pct: float = 0.10,
) -> HorizonPlan:
    """Calcula el plan óptimo HOLD vs SWING para una posición de opción corta.

    profile: "CONSERVADOR" | "EQUILIBRADO" | "AGRESIVO"
    caucion_tna: TNA de la caución (%) — se convierte a tasa continua para el árbol CRR.
    spread_buyback_pct: costo extra estimado de cruzar el spread al recomprar (fracción).
    """
    warnings: list[str] = []
    D       = result.days
    S0      = result.spot
    K       = result.strike
    iv      = result.iv
    capital = result.net_outlay
    premium = result.premium
    opt_type = "C" if result.strategy == "COVERED_CALL" else "P"
    r_cont   = math.log(1.0 + caucion_tna / 100.0)
    lam      = _LAMBDA.get(profile, 1.0)

    # ── HOLD forzado ──────────────────────────────────────────────────────────
    if not result.is_liquid:
        warnings.append("Opción ilíquida: recompra en mercado sin cruzar spread imposible.")
        return _forced_hold(result, D, capital, r_cont, iv, opt_type, warnings)

    if iv is None or iv <= 0.0:
        warnings.append("IV no disponible: no se puede calcular la curva de decaimiento.")
        return _forced_hold(result, D, capital, r_cont, iv, opt_type, warnings)

    if D < 5:
        warnings.append(f"Plazo muy corto ({D} días): análisis hold/swing no aplica.")
        return _forced_hold(result, D, capital, r_cont, iv, opt_type, warnings)

    # Guard: prima menor al intrínseco (opción deep ITM con datos sintéticos / inconsistentes).
    # En mercado real esto es imposible (arbitraje). Forzamos HOLD y devolvemos la TNA
    # del rate calculator, que ya modela correctamente el ejercicio físico.
    intrinsic_entry = max(S0 - K, 0.0) if opt_type == "C" else max(K - S0, 0.0)
    if premium < intrinsic_entry - 0.01:
        warnings.append(
            "Prima recibida menor al valor intrínseco: el modelo de decaimiento "
            "no aplica. Se usa la TNA del rate calculator."
        )
        return _forced_hold(result, D, capital, r_cont, iv, opt_type, warnings)

    # ── Curva de P&L en [1, D] ────────────────────────────────────────────────

    def _ov(S: float, t: int) -> float:
        """Valor de la opción con t días transcurridos (D-t restantes)."""
        T_rem = (D - t) / 365.0
        if T_rem < 1.0 / 365.0:
            return max(S - K, 0.0) if opt_type == "C" else max(K - S, 0.0)
        return crr_price(S, K, T_rem, r_cont, iv, opt_type, steps=_CRR_STEPS)

    def _close_cost(ov: float) -> float:
        """Costo total de cerrar: comisión de compra + medio spread estimado."""
        val = max(ov, 0.001)
        return cost_model.gross_cost(val, "option_buy") + val * spread_buyback_pct / 2.0

    pnl_base:   list[float] = []
    pnl_stress: list[float] = []

    for t in range(1, D + 1):
        ov_b  = _ov(S0, t)
        pnl_base.append(premium - ov_b - _close_cost(ov_b))

        # Escenario adverso simétrico: tomamos el peor de los dos movimientos.
        # Para covered call, subir el spot encarece el call (adverso para el vendedor).
        # Para short put, bajar el spot encarece el put (adverso para el vendedor).
        move  = _Z_STRESS * iv * math.sqrt(t / 365.0)
        S_dn  = S0 * math.exp(-move)
        S_up  = S0 * math.exp(+move)
        ov_dn = _ov(S_dn, t)
        ov_up = _ov(S_up, t)
        pnl_dn = premium - ov_dn - _close_cost(ov_dn)
        pnl_up = premium - ov_up - _close_cost(ov_up)
        # El peor P&L de los dos extremos
        pnl_stress.append(min(pnl_dn, pnl_up))

    # ── Score intertemporal u(t) = r_ann(t) − λ · riesgo(t) ──────────────────
    scores: list[float] = []
    for idx, t in enumerate(range(1, D + 1)):
        if capital <= 0:
            scores.append(-1e9)
            continue
        pnl_t = pnl_base[idx]
        r_ann = pnl_t / capital * 365.0 / t
        # Riesgo: cuánto peor es el escenario adverso respecto al base
        risk_t = max(pnl_t - pnl_stress[idx], 0.0) / capital
        scores.append(r_ann - lam * risk_t)

    t_star = scores.index(max(scores)) + 1   # rango [1, D]

    # ── HOLD vs SWING ─────────────────────────────────────────────────────────
    pnl_hold    = pnl_base[-1]
    pnl_t_star  = pnl_base[t_star - 1]
    mode        = "HOLD" if t_star >= _HOLD_THRESHOLD * D else "SWING"

    if mode == "HOLD":
        # Ground truth: usamos la TNA del rate calculator (ya incluye el leg del stock).
        tna_t_star = result.tna_pct
    else:
        tna_t_star = (pnl_t_star / capital * 365.0 / t_star * 100.0) if capital > 0 else 0.0

    # Captura %: qué fracción del P&L terminal del modelo se obtiene saliendo en t*.
    # pnl_hold es el P&L del modelo al vencimiento (= prima cobrada menos costos mínimos).
    capture_pct = (pnl_t_star / pnl_hold * 100.0) if pnl_hold > 0.001 else 100.0

    # ── Greeks al momento de entrada ─────────────────────────────────────────
    T_entry = D / 365.0
    kw = {"steps": _CRR_STEPS}
    theta_share = crr_theta(S0, K, T_entry, r_cont, iv, opt_type, **kw)
    gamma_val   = crr_gamma(S0, K, T_entry, r_cont, iv, opt_type, **kw)
    vega_val    = crr_vega( S0, K, T_entry, r_cont, iv, opt_type, **kw)
    theta_contr = abs(theta_share) * 100.0   # por contrato = 100 acciones

    explanation = _explain(
        mode, t_star, D, capture_pct, tna_t_star,
        theta_contr, gamma_val, vega_val, result, profile,
    )

    return HorizonPlan(
        mode=mode,
        target_exit_days=t_star,
        target_capture_pct=round(capture_pct, 1),
        optimized_tna_pct=round(tna_t_star, 1),
        theta_daily_ars=round(theta_contr, 2),
        gamma=round(gamma_val, 6),
        vega_1pct=round(vega_val, 2),
        plain_explanation=explanation,
        warnings=warnings,
    )


# ── Helpers internos ──────────────────────────────────────────────────────────

def _forced_hold(
    result: RateResult,
    D: int,
    capital: float,
    r_cont: float,
    iv: Optional[float],
    opt_type: str,
    warnings: list[str],
) -> HorizonPlan:
    """HOLD forzado. Calcula Greeks si IV disponible, de lo contrario los pone en 0."""
    theta_c = gamma_v = vega_v = 0.0
    if iv and iv > 0:
        T = D / 365.0
        kw = {"steps": _CRR_STEPS}
        theta_c = abs(crr_theta(result.spot, result.strike, T, r_cont, iv, opt_type, **kw)) * 100
        gamma_v = crr_gamma(result.spot, result.strike, T, r_cont, iv, opt_type, **kw)
        vega_v  = crr_vega( result.spot, result.strike, T, r_cont, iv, opt_type, **kw)

    note = f" {warnings[0]}" if warnings else ""
    explanation = (
        f"Holdear hasta el vencimiento ({D} días). "
        f"TNA proyectada: {result.tna_pct:.1f}%.{note}"
    )
    return HorizonPlan(
        mode="HOLD",
        target_exit_days=D,
        target_capture_pct=100.0,
        optimized_tna_pct=round(result.tna_pct, 1),
        theta_daily_ars=round(theta_c, 2),
        gamma=round(gamma_v, 6),
        vega_1pct=round(vega_v, 2),
        plain_explanation=explanation,
        warnings=warnings,
    )


def _explain(
    mode: str,
    t_star: int,
    D: int,
    capture_pct: float,
    tna_tstar: float,
    theta_contract: float,
    gamma: float,
    vega: float,
    result: RateResult,
    profile: str,
) -> str:
    strat   = "call" if result.strategy == "COVERED_CALL" else "put"
    greeks  = (
        f"Theta: ~${theta_contract:,.0f}/día por contrato. "
        f"Gamma: {gamma:.5f}. "
        f"Vega: {vega:.2f} (por 1% de IV)."
    )
    if mode == "HOLD":
        return (
            f"Aguantar hasta el vencimiento en {D} días. "
            f"TNA proyectada {result.tna_pct:.1f}%. "
            f"El ajuste por riesgo confirma que no hay ventaja en salir antes "
            f"para el perfil {profile}. {greeks}"
        )
    return (
        f"Cerrar el {strat} en ~{t_star} días, capturando el "
        f"{capture_pct:.0f}% de la ganancia (TNA optimizada: {tna_tstar:.1f}%). "
        f"Después de ese punto, el gamma creciente añade más riesgo del que "
        f"compensa la theta restante para el perfil {profile}. "
        f"Reutilizá el capital liberado. {greeks}"
    )
