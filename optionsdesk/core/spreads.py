"""Spreads verticales (2 patas, riesgo definido) para GGAL en BYMA.

Estrategias:
    BULL_CALL: compra call K_low  + vende call K_high  — debito, alcista
    BEAR_CALL: vende call K_low   + compra call K_high  — credito, bajista
    BULL_PUT:  vende put K_high   + compra put K_low    — credito, alcista
    BEAR_PUT:  compra put K_high  + vende put K_low     — debito, bajista

Riesgo maximo siempre definido y acotado al ancho del spread.

ADVERTENCIA: ejecucion pata a pata (legging risk). Liquidez de ambas patas
requerida. Con spreads bid-ask amplios el edge se reduce rapido.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional

from optionsdesk.config.costs import CostModel, DEFAULT_COSTS
from optionsdesk.core.pricing import risk_neutral_prob_above

_DISCLAIMER = (
    "SPREAD ESPECULATIVO — perdida maxima = debito neto (debit) o ancho - credito (credit). "
    "Ejecucion pata a pata: existe legging risk. Verificar liquidez de ambas patas."
)
_DEFAULT_SIGMA = 0.65   # vol anual de referencia GGAL si no se dispone de IV


# ── Tipos de datos ────────────────────────────────────────────────────────────

@dataclass
class SpreadLeg:
    """Una pata de un spread vertical."""
    symbol: str
    option_type: str   # "C" | "P"
    strike: float
    action: str        # "BUY" | "SELL"
    price: float       # prima efectiva usada en el calculo


@dataclass
class SpreadResult:
    """Resultado de un spread vertical de 2 patas."""
    strategy: str           # "BULL_CALL" | "BEAR_CALL" | "BULL_PUT" | "BEAR_PUT"
    long_leg: SpreadLeg     # pata comprada
    short_leg: SpreadLeg    # pata vendida
    expiration: date
    days: int
    spot: float
    net_debit: float        # >0 = debito (pagas); <0 = credito (cobras)
    max_profit: float       # ARS por contrato (1 contrato = 100 acciones)
    max_loss: float         # ARS por contrato
    breakeven: float        # precio spot de equilibrio
    risk_reward: float      # max_profit / max_loss
    prob_of_profit: float   # 0–1
    is_credit: bool
    is_liquid: bool
    rational: str
    disclaimer: str = _DISCLAIMER


# ── Helpers internos ──────────────────────────────────────────────────────────

def _mid(bid: float, ask: float) -> float:
    if bid > 0 and ask > 0:
        return (bid + ask) / 2.0
    return bid or ask


def _eff_price(bid: float, ask: float, mode: str) -> float:
    if mode == "bid":
        return bid
    if mode == "ask":
        return ask
    return _mid(bid, ask)


def _net_debit_per_share(
    buy_price: float,
    sell_price: float,
    cost_model: CostModel,
) -> float:
    """Debito neto por accion incluyendo comisiones de ambas patas.

    Positivo = se paga (spread de debito).
    Negativo = se cobra (spread de credito).
    """
    buy_cost  = cost_model.gross_cost(buy_price,  "option_buy")
    sell_cost = cost_model.gross_cost(sell_price, "option_sell")
    return (buy_price + buy_cost) - (sell_price - sell_cost)


# ── Builders (funciones puras, testeables) ────────────────────────────────────

def build_bull_call_spread(
    long_symbol: str,  long_strike: float,  long_bid: float,  long_ask: float,
    short_symbol: str, short_strike: float, short_bid: float, short_ask: float,
    expiration: date,
    days: int,
    spot: float,
    sigma: float = _DEFAULT_SIGMA,
    r: float = 0.0,
    cost_model: CostModel = DEFAULT_COSTS,
    price_mode: str = "mid",
) -> Optional[SpreadResult]:
    """BUY call K_low + SELL call K_high.  Alcista, debito, max-loss = debito neto."""
    if long_strike >= short_strike:
        return None
    long_p  = _eff_price(long_bid,  long_ask,  price_mode)
    short_p = _eff_price(short_bid, short_ask, price_mode)
    if long_p <= 0 or short_p <= 0:
        return None

    width      = short_strike - long_strike
    net_dbt    = _net_debit_per_share(long_p, short_p, cost_model)
    max_profit = (width - net_dbt) * 100.0
    max_loss   = net_dbt * 100.0
    if max_profit <= 0 or max_loss <= 0:
        return None

    T         = days / 365.0
    breakeven = long_strike + net_dbt
    rr        = max_profit / max_loss
    sigma_use = sigma if sigma and sigma > 0 else _DEFAULT_SIGMA
    pop       = risk_neutral_prob_above(spot, breakeven, T, r, sigma_use)

    return SpreadResult(
        strategy="BULL_CALL",
        long_leg=SpreadLeg(long_symbol,  "C", long_strike,  "BUY",  round(long_p, 2)),
        short_leg=SpreadLeg(short_symbol, "C", short_strike, "SELL", round(short_p, 2)),
        expiration=expiration, days=days, spot=spot,
        net_debit=round(net_dbt, 2),
        max_profit=round(max_profit, 2), max_loss=round(max_loss, 2),
        breakeven=round(breakeven, 2),   risk_reward=round(rr, 2),
        prob_of_profit=round(pop, 3),    is_credit=False,
        is_liquid=True,
        rational=(
            f"Bull Call Spread K={long_strike:,.0f}/{short_strike:,.0f} | "
            f"Debito ${net_dbt * 100:,.0f} | Max ganancia ${max_profit:,.0f} | R:R {rr:.1f}x"
        ),
    )


def build_bear_call_spread(
    short_symbol: str, short_strike: float, short_bid: float, short_ask: float,
    long_symbol: str,  long_strike: float,  long_bid: float,  long_ask: float,
    expiration: date,
    days: int,
    spot: float,
    sigma: float = _DEFAULT_SIGMA,
    r: float = 0.0,
    cost_model: CostModel = DEFAULT_COSTS,
    price_mode: str = "mid",
) -> Optional[SpreadResult]:
    """SELL call K_low + BUY call K_high.  Bajista, credito, max-loss = ancho - credito."""
    if short_strike >= long_strike:
        return None
    short_p = _eff_price(short_bid, short_ask, price_mode)
    long_p  = _eff_price(long_bid,  long_ask,  price_mode)
    if long_p <= 0 or short_p <= 0:
        return None

    width      = long_strike - short_strike
    net_dbt    = _net_debit_per_share(long_p, short_p, cost_model)   # <0 = credito
    net_credit = -net_dbt
    max_profit = net_credit * 100.0
    max_loss   = (width - net_credit) * 100.0
    if max_profit <= 0 or max_loss <= 0:
        return None

    T         = days / 365.0
    breakeven = short_strike + net_credit
    rr        = max_profit / max_loss
    sigma_use = sigma if sigma and sigma > 0 else _DEFAULT_SIGMA
    pop       = 1.0 - risk_neutral_prob_above(spot, breakeven, T, r, sigma_use)

    return SpreadResult(
        strategy="BEAR_CALL",
        long_leg=SpreadLeg(long_symbol,  "C", long_strike,  "BUY",  round(long_p, 2)),
        short_leg=SpreadLeg(short_symbol, "C", short_strike, "SELL", round(short_p, 2)),
        expiration=expiration, days=days, spot=spot,
        net_debit=round(net_dbt, 2),
        max_profit=round(max_profit, 2), max_loss=round(max_loss, 2),
        breakeven=round(breakeven, 2),   risk_reward=round(rr, 2),
        prob_of_profit=round(pop, 3),    is_credit=True,
        is_liquid=True,
        rational=(
            f"Bear Call Spread K={short_strike:,.0f}/{long_strike:,.0f} | "
            f"Credito ${net_credit * 100:,.0f} | Max ganancia ${max_profit:,.0f} | R:R {rr:.1f}x"
        ),
    )


def build_bull_put_spread(
    short_symbol: str, short_strike: float, short_bid: float, short_ask: float,
    long_symbol: str,  long_strike: float,  long_bid: float,  long_ask: float,
    expiration: date,
    days: int,
    spot: float,
    sigma: float = _DEFAULT_SIGMA,
    r: float = 0.0,
    cost_model: CostModel = DEFAULT_COSTS,
    price_mode: str = "mid",
) -> Optional[SpreadResult]:
    """SELL put K_high + BUY put K_low.  Alcista, credito, max-loss = ancho - credito."""
    if short_strike <= long_strike:
        return None
    short_p = _eff_price(short_bid, short_ask, price_mode)
    long_p  = _eff_price(long_bid,  long_ask,  price_mode)
    if long_p <= 0 or short_p <= 0:
        return None

    width      = short_strike - long_strike
    net_dbt    = _net_debit_per_share(long_p, short_p, cost_model)   # <0 = credito
    net_credit = -net_dbt
    max_profit = net_credit * 100.0
    max_loss   = (width - net_credit) * 100.0
    if max_profit <= 0 or max_loss <= 0:
        return None

    T         = days / 365.0
    breakeven = short_strike - net_credit
    rr        = max_profit / max_loss
    sigma_use = sigma if sigma and sigma > 0 else _DEFAULT_SIGMA
    pop       = risk_neutral_prob_above(spot, breakeven, T, r, sigma_use)

    return SpreadResult(
        strategy="BULL_PUT",
        long_leg=SpreadLeg(long_symbol,  "P", long_strike,  "BUY",  round(long_p, 2)),
        short_leg=SpreadLeg(short_symbol, "P", short_strike, "SELL", round(short_p, 2)),
        expiration=expiration, days=days, spot=spot,
        net_debit=round(net_dbt, 2),
        max_profit=round(max_profit, 2), max_loss=round(max_loss, 2),
        breakeven=round(breakeven, 2),   risk_reward=round(rr, 2),
        prob_of_profit=round(pop, 3),    is_credit=True,
        is_liquid=True,
        rational=(
            f"Bull Put Spread K={long_strike:,.0f}/{short_strike:,.0f} | "
            f"Credito ${net_credit * 100:,.0f} | Max ganancia ${max_profit:,.0f} | R:R {rr:.1f}x"
        ),
    )


def build_bear_put_spread(
    long_symbol: str,  long_strike: float,  long_bid: float,  long_ask: float,
    short_symbol: str, short_strike: float, short_bid: float, short_ask: float,
    expiration: date,
    days: int,
    spot: float,
    sigma: float = _DEFAULT_SIGMA,
    r: float = 0.0,
    cost_model: CostModel = DEFAULT_COSTS,
    price_mode: str = "mid",
) -> Optional[SpreadResult]:
    """BUY put K_high + SELL put K_low.  Bajista, debito, max-loss = debito neto."""
    if long_strike <= short_strike:
        return None
    long_p  = _eff_price(long_bid,  long_ask,  price_mode)
    short_p = _eff_price(short_bid, short_ask, price_mode)
    if long_p <= 0 or short_p <= 0:
        return None

    width      = long_strike - short_strike
    net_dbt    = _net_debit_per_share(long_p, short_p, cost_model)
    max_profit = (width - net_dbt) * 100.0
    max_loss   = net_dbt * 100.0
    if max_profit <= 0 or max_loss <= 0:
        return None

    T         = days / 365.0
    breakeven = long_strike - net_dbt
    rr        = max_profit / max_loss
    sigma_use = sigma if sigma and sigma > 0 else _DEFAULT_SIGMA
    pop       = 1.0 - risk_neutral_prob_above(spot, breakeven, T, r, sigma_use)

    return SpreadResult(
        strategy="BEAR_PUT",
        long_leg=SpreadLeg(long_symbol,  "P", long_strike,  "BUY",  round(long_p, 2)),
        short_leg=SpreadLeg(short_symbol, "P", short_strike, "SELL", round(short_p, 2)),
        expiration=expiration, days=days, spot=spot,
        net_debit=round(net_dbt, 2),
        max_profit=round(max_profit, 2), max_loss=round(max_loss, 2),
        breakeven=round(breakeven, 2),   risk_reward=round(rr, 2),
        prob_of_profit=round(pop, 3),    is_credit=False,
        is_liquid=True,
        rational=(
            f"Bear Put Spread K={short_strike:,.0f}/{long_strike:,.0f} | "
            f"Debito ${net_dbt * 100:,.0f} | Max ganancia ${max_profit:,.0f} | R:R {rr:.1f}x"
        ),
    )
