"""Vista agregada del portafolio de posiciones abiertas.

Lee open_positions.jsonl, recomputa griegas con el spot y IV actuales,
y expone métricas de portafolio: delta neto, theta diaria, concentración
por vencimiento y scenario P&L.
"""
from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

from optionsdesk.core.pricing import crr_delta, crr_theta, crr_vega, crr_price
from optionsdesk.signals.monitor import OpenPosition, PositionMonitor

_CRR_STEPS = 50
_SYMBOL_RE = re.compile(r"^GFG([CV])(\d+(?:[.,]\d+)?)([A-Z]+)$")


def _strategy_code(strategy: str) -> str:
    strategy_u = str(strategy).upper()
    if strategy_u == "COVERED_CALL":
        return "CC"
    if strategy_u == "SHORT_PUT":
        return "SP"
    if "LONG_CALL" in strategy_u:
        return "LC"
    if "LONG_PUT" in strategy_u:
        return "LP"
    if "SHORT_CALL" in strategy_u:
        return "SC"
    if "SHORT_PUT" in strategy_u:
        return "SP"
    return strategy_u or "OTRA"


# ── Dataclasses de output ─────────────────────────────────────────────────────

@dataclass
class PositionGreeks:
    symbol: str
    strategy: str
    strike: float
    contracts: int               # contratos (lotes de 100 acciones)
    delta: Optional[float]       # delta de la opción escrita (absoluto)
    delta_net: float             # delta neto en acciones equivalentes
    theta_daily_ars: float       # theta diaria en ARS
    vega_1pct: float             # vega para 1% de cambio en IV
    current_value: float         # valor actual de la opción (CRR)
    capture_pct: float           # % de prima ya capturada


@dataclass
class AggregatedGreeks:
    n_positions: int
    delta_net: float             # acciones equivalentes (puede ser neg. si vendemos calls)
    theta_daily_ars: float       # ganancia diaria por theta (positivo = ganamos)
    vega_total: float            # exposición a movimientos de IV
    total_capital_at_risk: float  # suma de net_outlay × contratos
    positions: list[PositionGreeks] = field(default_factory=list)


@dataclass
class ConcentrationReport:
    by_expiry: dict[str, dict]   # expiry_code → {n_pos, capital, strategies}
    by_strategy: dict[str, int]  # "CC" | "SP" → n posiciones
    flags: list[str]             # alertas de concentración


@dataclass
class ScenarioPnL:
    shock_pcts: list[float]      # e.g. [-10, -5, 0, 5, 10]
    pnl_ars: list[float]         # PnL del portafolio para cada shock


# ── Funciones principales ─────────────────────────────────────────────────────

def load_open_positions(
    path: Path = Path("data/open_positions.jsonl"),
) -> list[OpenPosition]:
    """Carga las posiciones abiertas del archivo JSONL."""
    return PositionMonitor(path).load_positions()


def aggregate_greeks(
    positions: list[OpenPosition],
    current_spot: float,
    current_iv_map: Optional[dict[str, float]] = None,
    contracts_per_position: int = 1,
) -> AggregatedGreeks:
    """Computa griegas agregadas del portafolio.

    Args:
        positions: lista de posiciones abiertas.
        current_spot: precio actual de GGAL.
        current_iv_map: dict {symbol → IV} para usar IV viva; fallback a iv_entry.
        contracts_per_position: contratos por posición si no están en el objeto.
    """
    if not positions:
        return AggregatedGreeks(
            n_positions=0, delta_net=0.0, theta_daily_ars=0.0,
            vega_total=0.0, total_capital_at_risk=0.0,
        )

    iv_map = current_iv_map or {}
    pos_greeks: list[PositionGreeks] = []

    sum_delta_net = 0.0
    sum_theta     = 0.0
    sum_vega      = 0.0
    sum_capital   = 0.0

    for pos in positions:
        days_rem = pos.days_remaining
        T_rem    = max(days_rem, 1) / 365.0
        iv       = iv_map.get(pos.symbol, pos.iv_entry) or pos.iv_entry
        r        = math.log(1.0 + pos.caucion_tna / 100.0)
        # Usar contratos reales del objeto; contracts_per_position como fallback
        n = getattr(pos, "contracts", contracts_per_position) or contracts_per_position

        if iv <= 0 or current_spot <= 0:
            continue

        # Inicializar con defaults antes del try para evitar scope leak entre iteraciones
        delta_val  = None
        delta_net  = 0.0
        theta_ars  = 0.0
        vega_ars   = 0.0
        current_ov = 0.0
        capture    = 0.0

        try:
            delta_val = crr_delta(
                current_spot, pos.strike, T_rem, r, iv, pos.opt_type,
                steps=_CRR_STEPS,
            )
            is_long = "LONG" in pos.strategy
            sign = 1.0 if is_long else -1.0
            
            # Para posición short: multiplicamos por -1 (vendemos)
            delta_net = sign * delta_val * n * 100.0   # acciones equivalentes

            theta_val = crr_theta(
                current_spot, pos.strike, T_rem, r, iv, pos.opt_type,
                steps=_CRR_STEPS,
            )
            # Theta positiva para el vendedor (ganamos por paso del tiempo)
            theta_ars = sign * theta_val * n * 100.0

            vega_val = crr_vega(
                current_spot, pos.strike, T_rem, r, iv, pos.opt_type,
                steps=_CRR_STEPS,
            )
            # Vega en ARS por 1% de cambio de IV; negativa para el vendedor
            vega_ars = sign * vega_val * 0.01 * n * 100.0

            current_ov = crr_price(
                current_spot, pos.strike, T_rem, r, iv, pos.opt_type,
                steps=_CRR_STEPS,
            )
            
            if is_long:
                pnl_now = current_ov - pos.net_outlay
                capture = (pnl_now / pos.net_outlay * 100.0) if pos.net_outlay > 0 else 0.0
            else:
                pnl_now = pos.premium_received - current_ov
                capture = max(pnl_now / pos.premium_received * 100.0, 0.0) if pos.premium_received > 0 else 0.0

        except Exception as e:
            logger.debug("griegas para %s fallaron: %s", pos.symbol, e)

        capital = pos.net_outlay * n * 100.0
        sum_delta_net += delta_net
        sum_theta     += theta_ars
        sum_vega      += vega_ars
        sum_capital   += capital

        pos_greeks.append(PositionGreeks(
            symbol=pos.symbol,
            strategy=pos.strategy,
            strike=pos.strike,
            contracts=n,
            delta=delta_val,
            delta_net=delta_net,
            theta_daily_ars=theta_ars,
            vega_1pct=vega_ars,
            current_value=current_ov,
            capture_pct=capture,
        ))

    return AggregatedGreeks(
        n_positions=len(positions),
        delta_net=round(sum_delta_net, 2),
        theta_daily_ars=round(sum_theta, 0),
        vega_total=round(sum_vega, 0),
        total_capital_at_risk=round(sum_capital, 0),
        positions=pos_greeks,
    )


def concentration_report(
    positions: list[OpenPosition],
    max_per_expiry: int = 3,
) -> ConcentrationReport:
    """Analiza la concentración por vencimiento y estrategia."""
    by_expiry: dict[str, dict] = {}
    by_strategy: dict[str, int] = {"CC": 0, "SP": 0}

    for pos in positions:
        # Extraer código de vencimiento del símbolo
        m = _SYMBOL_RE.match(pos.symbol)
        expiry_code = m.group(3) if m else "???"
        code = _strategy_code(pos.strategy)
        n = getattr(pos, "contracts", 1) or 1

        if expiry_code not in by_expiry:
            by_expiry[expiry_code] = {
                "n_pos": 0, "strategies": [], "capital": 0.0, "max_loss": 0.0
            }
        by_expiry[expiry_code]["n_pos"] += 1
        by_expiry[expiry_code]["strategies"].append(code)
        by_expiry[expiry_code]["capital"] += pos.net_outlay * n * 100.0
        by_expiry[expiry_code]["max_loss"] += abs(pos.net_outlay) * n * 100.0
        by_strategy[code] = by_strategy.get(code, 0) + 1

    flags: list[str] = []
    for code, data in by_expiry.items():
        if data["n_pos"] > max_per_expiry:
            flags.append(
                f"{data['n_pos']} posiciones en {code} — concentración alta (máx {max_per_expiry})"
            )

    if by_strategy["SP"] > 0 and by_strategy["CC"] > 0:
        flags.append("Portafolio mixto CC+SP — verificar delta neto total.")

    return ConcentrationReport(
        by_expiry=by_expiry,
        by_strategy=by_strategy,
        flags=flags,
    )


def scenario_pnl(
    positions: list[OpenPosition],
    current_spot: float,
    shock_pcts: Optional[list[float]] = None,
    current_iv_map: Optional[dict[str, float]] = None,
    contracts_per_position: int = 1,
) -> ScenarioPnL:
    """Calcula el P&L del portafolio bajo distintos shocks de spot.

    Para cada shock: recomputa el valor de cada opción con el nuevo spot
    (IV y tiempo sin cambio) y calcula el P&L neto vs. la posición actual.
    """
    if shock_pcts is None:
        shock_pcts = [-10.0, -5.0, 0.0, +5.0, +10.0]
    iv_map = current_iv_map or {}

    pnl_by_shock: list[float] = []
    for shock_pct in shock_pcts:
        new_spot = current_spot * (1.0 + shock_pct / 100.0)
        total_pnl = 0.0

        for pos in positions:
            T_rem = max(pos.days_remaining, 1) / 365.0
            iv    = iv_map.get(pos.symbol, pos.iv_entry) or pos.iv_entry
            r     = math.log(1.0 + pos.caucion_tna / 100.0)
            n     = getattr(pos, "contracts", contracts_per_position) or contracts_per_position

            if iv <= 0 or new_spot <= 0:
                continue
            try:
                new_ov = crr_price(
                    new_spot, pos.strike, T_rem, r, iv, pos.opt_type,
                    steps=_CRR_STEPS,
                )
                if "LONG" in pos.strategy:
                    pnl = (new_ov - pos.net_outlay) * n * 100.0
                else:
                    pnl = (pos.premium_received - new_ov) * n * 100.0
                total_pnl += pnl
            except Exception as e:
                logger.debug("scenario pnl para %s fallo: %s", pos.symbol, e)

        pnl_by_shock.append(round(total_pnl, 0))

    return ScenarioPnL(shock_pcts=shock_pcts, pnl_ars=pnl_by_shock)
