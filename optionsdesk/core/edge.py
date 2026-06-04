"""Edge cuantitativo para estrategias de prima vendida.

Calcula probabilidad de ganancia (PoP) y valor esperado (EV) bajo la medida
física con distribución lognormal.

La clave: la distribución de resultados se modela con la **vol realizada**
(lo que la acción efectivamente mueve) mientras que la prima cobrada es la del
mercado (priceada con IV). EV > 0 ⟺ VRP > 0 — el edge estructural del vendedor
de prima expresado en pesos.

Drift por default = 0 (medida física neutral, ligeramente conservadora).
"""
from __future__ import annotations

import math
from typing import Optional, TYPE_CHECKING

from scipy.stats import norm

if TYPE_CHECKING:
    from optionsdesk.core.rates import RateResult


# ── Helpers internos (lognormal física) ──────────────────────────────────────

def _d1_d2(
    spot: float,
    strike: float,
    sigma: float,
    T: float,
    drift: float = 0.0,
) -> tuple[float, float]:
    """d1 y d2 de la fórmula lognormal con drift físico."""
    sqrtT = math.sqrt(T)
    d1 = (math.log(spot / strike) + (drift + 0.5 * sigma ** 2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return d1, d2


def _prob_above(
    spot: float,
    strike: float,
    sigma: float,
    T: float,
    drift: float = 0.0,
) -> float:
    """P(S_T > strike) bajo lognormal física con el drift dado."""
    if T <= 0 or sigma <= 0 or spot <= 0 or strike <= 0:
        return 1.0 if spot > strike else 0.0
    _, d2 = _d1_d2(spot, strike, sigma, T, drift)
    return float(norm.cdf(d2))


def _expected_put_payoff(
    spot: float,
    strike: float,
    sigma: float,
    T: float,
    drift: float = 0.0,
) -> float:
    """E[(K − S_T)⁺] bajo lognormal física — valor esperado del put al vencimiento.

    Con drift=0: E[S_T]=spot, y el resultado es la prima 'justa' bajo
    la distribución real (no risk-neutral). Si prima recibida > este valor
    → expectativa positiva.
    """
    if T <= 0 or sigma <= 0 or spot <= 0 or strike <= 0:
        return max(strike - spot, 0.0)
    d1, d2 = _d1_d2(spot, strike, sigma, T, drift)
    fwd = spot * math.exp(drift * T)
    return strike * norm.cdf(-d2) - fwd * norm.cdf(-d1)


def _expected_min_with_strike(
    spot: float,
    strike: float,
    sigma: float,
    T: float,
    drift: float = 0.0,
) -> float:
    """E[min(S_T, K)] bajo lognormal física.

    Usado para el EV del covered call: E[flujo al vencimiento] = E[min(S_T, K)].
    """
    if T <= 0 or sigma <= 0 or spot <= 0 or strike <= 0:
        return min(spot, strike)
    d1, d2 = _d1_d2(spot, strike, sigma, T, drift)
    fwd = spot * math.exp(drift * T)
    # E[min(S_T, K)] = E[S_T] − E[(S_T − K)⁺]
    expected_call_payoff = fwd * norm.cdf(d1) - strike * norm.cdf(d2)
    return fwd - expected_call_payoff


# ── API pública ───────────────────────────────────────────────────────────────

def prob_and_ev(
    strategy: str,
    spot: float,
    strike: float,
    days: int,
    net_proceeds: float,
    net_outlay: float,
    cushion_pct: float,
    realized_vol: float,
    drift: float = 0.0,
) -> tuple[float, float, float]:
    """Probabilidad de ganancia y valor esperado bajo la distribución física.

    Args:
        strategy: "COVERED_CALL" o "SHORT_PUT".
        spot: precio del subyacente.
        strike: strike del contrato.
        days: días al vencimiento.
        net_proceeds: para SP = prima neta recibida por acción (ARS).
                      Para CC = net_proceeds de RateResult (no usado directamente).
        net_outlay: capital comprometido por acción (ARS).
        cushion_pct: colchón % del spot — define el breakeven = spot*(1−cushion/100).
        realized_vol: vol realizada anualizada (p.ej. 0.55 = 55%).
        drift: drift anual del subyacente (default 0 = neutral).

    Returns:
        (prob_profit, ev_pct_anual, ev_ars_por_lote):
          - prob_profit: P(S_T >= breakeven) — probabilidad de no perder plata.
          - ev_pct_anual: expectativa anualizada como % del capital comprometido.
          - ev_ars_por_lote: expectativa en ARS para 1 lote (100 acciones).

    Propiedad clave: EV > 0 ⟺ VRP > 0 (prima cobrada > costo esperado del riesgo).
    """
    T = days / 365.0
    if T <= 0 or realized_vol <= 0 or spot <= 0 or net_outlay <= 0:
        return 0.5, 0.0, 0.0

    breakeven = spot * (1.0 - cushion_pct / 100.0)

    if strategy == "SHORT_PUT":
        # PoP: P(S_T >= strike) — put expira OTM, mantenemos toda la prima
        prob_profit = _prob_above(spot, strike, realized_vol, T, drift)
        # EV = prima recibida − E[(K − S_T)⁺]
        ev_per_unit = net_proceeds - _expected_put_payoff(
            spot, strike, realized_vol, T, drift
        )
    else:
        # COVERED_CALL
        # PoP: P(S_T >= breakeven) — no perder plata (stock no cae bajo el costo neto)
        if breakeven > 0:
            prob_profit = _prob_above(spot, breakeven, realized_vol, T, drift)
        else:
            prob_profit = 1.0
        # EV = E[min(S_T, strike)] − net_outlay
        ev_per_unit = (
            _expected_min_with_strike(spot, strike, realized_vol, T, drift)
            - net_outlay
        )

    ev_pct_period = ev_per_unit / net_outlay * 100.0
    ev_pct_ann = ev_pct_period * 365.0 / days
    ev_ars_lote = ev_per_unit * 100.0  # lote = 100 acciones

    return float(prob_profit), float(ev_pct_ann), float(ev_ars_lote)


def enrich_rate_result(
    result: "RateResult",
    spot_history: list[float],
    iv_history: Optional[list[float]] = None,
    drift: float = 0.0,
) -> None:
    """Enriquece un RateResult in-place con vol_edge, prob_profit y EV.

    Si no hay suficiente historia o la IV no está disponible, los campos
    quedan en None y el resultado se comporta como en v2.2.
    """
    from optionsdesk.signals.volatility import realized_volatility, VolEdge

    vol_edge = VolEdge.compute(result.iv, spot_history, iv_history)
    result.vol_edge = vol_edge

    rv = realized_volatility(spot_history)
    if rv is None or rv <= 0 or result.net_outlay <= 0:
        result.prob_profit = None
        result.expected_value_pct = None
        result.expected_value_ars = None
        return

    prob, ev_pct, ev_ars = prob_and_ev(
        strategy=result.strategy,
        spot=result.spot,
        strike=result.strike,
        days=result.days,
        net_proceeds=result.net_proceeds,
        net_outlay=result.net_outlay,
        cushion_pct=result.cushion_pct,
        realized_vol=rv,
        drift=drift,
    )
    result.prob_profit = prob
    result.expected_value_pct = ev_pct
    result.expected_value_ars = ev_ars
