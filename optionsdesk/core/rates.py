"""Motor de cálculo de tasa implícita para lanzamiento cubierto y venta de puts.

La "tasa" es la métrica central del mercado de opciones argentino: el rendimiento
anualizado equivalente a una renta fija que genera el lanzamiento cubierto o la
venta de put, neto de costos operativos.

Fórmulas (sin costos para claridad):
  Lanzamiento cubierto: TNA = (strike / (spot - prima) - 1) × 365/días
  Venta de put:         TNA = (prima / (strike - prima)) × 365/días

Con costos, se ajustan tanto el desembolso neto como los ingresos netos.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from optionsdesk.config.costs import CostModel, DEFAULT_COSTS
from optionsdesk.core.instruments import OptionContract, OptionType


@dataclass
class RateResult:
    """Tasa computada para una oportunidad de renta en opciones GGAL."""

    symbol: str
    strategy: str            # "COVERED_CALL" o "SHORT_PUT"
    strike: float
    expiration: date
    days: int
    spot: float
    premium: float           # prima efectiva usada en el cálculo
    net_outlay: float        # capital comprometido neto de costos
    net_proceeds: float      # ingresos netos si se ejerce / expira OTM
    period_rate_pct: float   # retorno del período (%)
    tna_pct: float           # tasa nominal anual (%)
    tea_pct: float           # tasa efectiva anual (%)
    cushion_pct: float       # caída máxima del spot sin perder la tasa (%)
    delta: Optional[float]
    iv: Optional[float]
    moneyness: str
    is_liquid: bool
    _spread_vs_caucion_pct: float = field(default=0.0, repr=False)

    @property
    def spread_vs_caucion_pct(self) -> float:
        return self._spread_vs_caucion_pct

    def set_spread(self, caucion_tna_pct: float) -> None:
        self._spread_vs_caucion_pct = self.tna_pct - caucion_tna_pct


# ── Helpers de precio ─────────────────────────────────────────────────────────

def _mid(bid: float, ask: float) -> float:
    if bid > 0 and ask > 0:
        return (bid + ask) / 2.0
    return bid or ask


def _effective_price(
    bid: float, ask: float, last: Optional[float], mode: str, side: str
) -> float:
    """Precio efectivo según el mode y el lado de la operación.

    Para análisis conservador usar side='sell' → toma el bid (lo que recibís
    al vender). Para análisis neutral usar mode='mid'.
    """
    if mode == "bid":
        return bid
    if mode == "ask":
        return ask
    if mode == "last":
        return last or _mid(bid, ask)
    # mid (default)
    return _mid(bid, ask)


# ── Lanzamiento cubierto ───────────────────────────────────────────────────────

def compute_covered_call(
    contract: OptionContract,
    spot_bid: float,
    spot_ask: float,
    call_bid: float,
    call_ask: float,
    call_last: Optional[float] = None,
    spot_last: Optional[float] = None,
    dividends: float = 0.0,
    delta: Optional[float] = None,
    iv: Optional[float] = None,
    cost_model: CostModel = DEFAULT_COSTS,
    price_mode: str = "mid",
    is_liquid: bool = True,
    atm_threshold_pct: float = 2.0,
) -> Optional[RateResult]:
    """Calcula la tasa anualizada del lanzamiento cubierto (neta de costos).

    Se compra la acción a spot_ask y se vende el call al precio según price_mode.
    dividends: dividendos en ARS esperados antes del vencimiento.

    Retorna None si el contrato no es un call, ya expiró, o los precios son inválidos.
    """
    if contract.option_type != OptionType.CALL:
        return None
    days = contract.days_to_expiry
    if days <= 0:
        return None

    # Compramos la acción al ask
    spot = spot_ask if spot_ask > 0 else _mid(spot_bid, spot_ask)
    if spot <= 0:
        return None

    # Vendemos el call; conservadoramente tomamos el bid, o el mid/last según config
    call_prem = _effective_price(call_bid, call_ask, call_last, price_mode, "sell")
    if call_prem <= 0:
        return None

    # Costos
    cost_buy_stock = cost_model.gross_cost(spot, "stock_buy")
    cost_sell_call = cost_model.gross_cost(call_prem, "option_sell")
    cost_exercise = cost_model.exercise_cost(contract.strike)

    net_outlay = spot + cost_buy_stock + cost_sell_call - call_prem
    net_proceeds = contract.strike + dividends - cost_exercise

    if net_outlay <= 0:
        return None

    period_rate = net_proceeds / net_outlay - 1.0
    T = days / 365.0
    tna = period_rate * 365.0 / days
    tea = (1.0 + period_rate) ** (1.0 / T) - 1.0

    # Colchón: cuánto puede caer el spot sin que perdamos la tasa
    # (breakeven = net_outlay - dividends es el precio mínimo de la acción al ejercicio)
    breakeven_spot = net_outlay - dividends
    cushion_pct = (spot - breakeven_spot) / spot * 100.0

    moneyness = contract.moneyness(spot, atm_threshold_pct)

    return RateResult(
        symbol=contract.symbol,
        strategy="COVERED_CALL",
        strike=contract.strike,
        expiration=contract.expiration,
        days=days,
        spot=spot,
        premium=call_prem,
        net_outlay=net_outlay,
        net_proceeds=net_proceeds,
        period_rate_pct=period_rate * 100.0,
        tna_pct=tna * 100.0,
        tea_pct=tea * 100.0,
        cushion_pct=cushion_pct,
        delta=delta,
        iv=iv,
        moneyness=moneyness,
        is_liquid=is_liquid,
    )


# ── Venta de put cubierta con efectivo ────────────────────────────────────────

def compute_short_put(
    contract: OptionContract,
    put_bid: float,
    put_ask: float,
    spot: float,
    put_last: Optional[float] = None,
    delta: Optional[float] = None,
    iv: Optional[float] = None,
    cost_model: CostModel = DEFAULT_COSTS,
    price_mode: str = "mid",
    capital_mode: str = "net",
    is_liquid: bool = True,
    atm_threshold_pct: float = 2.0,
) -> Optional[RateResult]:
    """Calcula la tasa anualizada de la venta de put cash-secured (neta de costos).

    capital_mode:
      'net'   → capital = strike - prima_neta (más eficiente)
      'gross' → capital = strike (más conservador)
    """
    if contract.option_type != OptionType.PUT:
        return None
    days = contract.days_to_expiry
    if days <= 0:
        return None

    put_prem = _effective_price(put_bid, put_ask, put_last, price_mode, "sell")
    if put_prem <= 0:
        return None

    cost_sell_put = cost_model.gross_cost(put_prem, "option_sell")
    cost_assignment = cost_model.exercise_cost(contract.strike)

    net_premium = put_prem - cost_sell_put  # lo que efectivamente recibimos

    if capital_mode == "net":
        capital = contract.strike - net_premium
    else:
        capital = contract.strike

    if capital <= 0:
        return None

    period_rate = net_premium / capital
    T = days / 365.0
    tna = period_rate * 365.0 / days
    tea = (1.0 + period_rate) ** (1.0 / T) - 1.0

    # Breakeven: spot al cual se empieza a perder si se ejerce
    breakeven_spot = contract.strike - net_premium - cost_assignment
    cushion_pct = (spot - breakeven_spot) / spot * 100.0 if spot > 0 else 0.0

    moneyness = contract.moneyness(spot, atm_threshold_pct)

    return RateResult(
        symbol=contract.symbol,
        strategy="SHORT_PUT",
        strike=contract.strike,
        expiration=contract.expiration,
        days=days,
        spot=spot,
        premium=put_prem,
        net_outlay=capital,
        net_proceeds=net_premium,
        period_rate_pct=period_rate * 100.0,
        tna_pct=tna * 100.0,
        tea_pct=tea * 100.0,
        cushion_pct=cushion_pct,
        delta=delta,
        iv=iv,
        moneyness=moneyness,
        is_liquid=is_liquid,
    )
