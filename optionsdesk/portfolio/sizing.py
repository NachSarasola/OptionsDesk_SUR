"""Kelly fraccional para sizing de posiciones de prima vendida.

El Kelly criterion clásico: f* = (p × b − q) / b
donde p = prob de ganar, b = razón ganancia/pérdida, q = 1 - p.

Para una posición de prima: b = EV / max_loss_per_lote.

Usamos Kelly cuartil (0.25 × f*) para controlar la volatilidad del capital.
En opciones el Kelly pleno es excesivo: la distribución de retornos no es
binaria y el Kelly standard sobreestima el leverage óptimo.
"""
from __future__ import annotations

import math
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from optionsdesk.signals.recommender import Recommendation

# Fracción de Kelly a usar (0.25 = Kelly cuartil, estándar de la industria)
_DEFAULT_KELLY_FRACTION = 0.25

# Cap máximo por trade como % del capital (evita Kelly > 100% del capital)
_MAX_CAPITAL_PER_TRADE_PCT = 0.20

# Mínimo: siempre al menos 1 contrato
_MIN_CONTRACTS = 1


def kelly_fraction(
    ev_ars: float,
    max_loss_ars: float,
    prob_profit: float,
    kelly_factor: float = _DEFAULT_KELLY_FRACTION,
) -> float:
    """Calcula la fracción de Kelly ajustada para un trade de prima.

    Args:
        ev_ars: valor esperado del trade en ARS (positivo = edge a favor).
        max_loss_ars: pérdida máxima posible en ARS (positivo).
        prob_profit: probabilidad de ganar (0–1).
        kelly_factor: multiplicador (0.25 = Kelly cuartil).

    Returns:
        Fracción del capital a asignar (0–1). 0 si no hay edge.
    """
    if ev_ars <= 0 or prob_profit <= 0 or max_loss_ars <= 0:
        return 0.0

    prob_loss = 1.0 - prob_profit
    if prob_loss <= 0:
        return 1.0  # 100% probabilidad de ganar → no hay Kelly relevante

    # Kelly clásico = (p × gain − loss × q) / gain
    # Aquí: gain ≈ EV/prob_profit, loss ≈ max_loss/prob_loss (expectativas)
    # Simplificado: f* = EV / max_loss (válido cuando la distribución es binaria)
    f_star = ev_ars / max_loss_ars

    return min(max(f_star * kelly_factor, 0.0), 1.0)


def size_position(
    capital: float,
    rec: "Recommendation",
    kelly_factor: float = _DEFAULT_KELLY_FRACTION,
    max_per_trade_pct: float = _MAX_CAPITAL_PER_TRADE_PCT,
) -> int:
    """Calcula cuántos contratos abrir para una Recommendation usando Kelly.

    Si la Recommendation no tiene datos de EV/PoP, cae al fixed fractional
    (el comportamiento original del recommender).

    Args:
        capital: capital disponible en ARS.
        rec: Recommendation del recommender.
        kelly_factor: fracción de Kelly (default 0.25).
        max_per_trade_pct: tope máximo por trade (default 20% del capital).

    Returns:
        Número de contratos (≥ 1, o 0 si el capital no alcanza ni para 1).
    """
    r = rec.result
    net_outlay_per_lot = r.net_outlay * 100.0  # 1 contrato = 100 acciones

    if net_outlay_per_lot <= 0 or capital <= 0:
        return 0

    # Cap por trade
    max_capital = capital * max_per_trade_pct
    max_contracts_by_cap = max(math.floor(max_capital / net_outlay_per_lot), 0)

    if max_contracts_by_cap == 0:
        return 0

    # Kelly si tenemos EV y PoP
    ev_ars  = r.expected_value_ars
    pop     = r.prob_profit
    max_loss_ars = r.net_outlay * 100.0 * 2.0  # stop a 2× la prima como max_loss

    if ev_ars is not None and pop is not None and ev_ars > 0 and pop > 0:
        f = kelly_fraction(ev_ars, max_loss_ars, pop, kelly_factor)
        kelly_capital = capital * f
        kelly_contracts = max(math.floor(kelly_capital / net_outlay_per_lot), _MIN_CONTRACTS)
        n = min(kelly_contracts, max_contracts_by_cap)
    else:
        # Fallback fixed fractional (comportamiento original)
        n = max_contracts_by_cap

    return max(n, _MIN_CONTRACTS) if capital >= net_outlay_per_lot else 0


def size_with_risk_budget(
    capital: float,
    rec: "Recommendation",
    current_delta_net: float = 0.0,
    max_net_delta: float = 100.0,
    kelly_factor: float = _DEFAULT_KELLY_FRACTION,
    max_per_trade_pct: float = _MAX_CAPITAL_PER_TRADE_PCT,
) -> int:
    """Como size_position, pero reduce contratos si el delta neto del portafolio
    ya está cerca del límite configurado.

    Args:
        current_delta_net: delta neto actual del portafolio (acciones equiv.).
        max_net_delta: límite máximo de delta neto.
        otros: igual a size_position.
    """
    base_n = size_position(capital, rec, kelly_factor, max_per_trade_pct)
    if base_n == 0:
        return 0

    # Delta del nuevo trade (estimado): delta del strike × contratos × 100
    r = rec.result
    if r.delta is None:
        return base_n

    trade_delta = abs(r.delta) * base_n * 100.0
    if abs(current_delta_net) + trade_delta > max_net_delta:
        # Reducir contratos para no superar el límite
        remaining_budget = max(max_net_delta - abs(current_delta_net), 0.0)
        if remaining_budget < abs(r.delta) * 100.0:
            return 0   # ni un contrato cabe en el budget
        n_by_delta = max(math.floor(remaining_budget / (abs(r.delta) * 100.0)), 0)
        return min(base_n, n_by_delta)

    return base_n
