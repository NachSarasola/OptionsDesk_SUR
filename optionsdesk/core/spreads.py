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

from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import numpy as np

from optionsdesk.config.costs import CostModel, DEFAULT_COSTS
from optionsdesk.core.pricing import risk_neutral_prob_above

_DISCLAIMER = (
    "SPREAD ESPECULATIVO — perdida maxima = debito neto (debit) o ancho - credito (credit). "
    "Ejecucion pata a pata: existe legging risk. Verificar liquidez de ambas patas."
)
from optionsdesk.config.constants import DEFAULT_SIGMA_GGAL as _DEFAULT_SIGMA


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
    prob_of_profit: float   # 0–1 con drift=0, vol realizada
    is_credit: bool
    is_liquid: bool
    rational: str
    disclaimer: str = _DISCLAIMER
    # v2.4 — valor esperado honesto (drift=0, precios ejecutables)
    expected_value: float = 0.0     # ARS por contrato, integración numérica
    ev_return_pct: float = 0.0      # expected_value / max_loss * 100
    # v2.6 — Greeks netos (long − short), ARS por contrato (×100)
    net_delta: float = 0.0    # exposición direccional implícita
    net_theta: float = 0.0    # ARS/día: credit >0 (ganás tiempo), debit <0
    net_vega:  float = 0.0    # ARS por 1pp de Δvol: credit <0, debit >0


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


def _eff_price_leg(bid: float, ask: float, mode: str, is_buy: bool) -> float:
    """Precio efectivo de una pata.

    "executable" usa ask para compras y bid para ventas (fill realista, peor caso).
    """
    if mode == "executable":
        return ask if is_buy else bid
    return _eff_price(bid, ask, mode)


# ── Helpers de valor esperado ─────────────────────────────────────────────────

def spread_payoff(
    strategy: str,
    k_lo: float,
    k_hi: float,
    max_profit: float,
    max_loss: float,
    spots: "np.ndarray | list[float]",
) -> np.ndarray:
    """Payoff al vencimiento (ARS por contrato) para un arreglo de spots.

    Tramos: flat en [-max_loss] por debajo de k_lo, lineal entre k_lo y k_hi,
    flat en [+max_profit] por encima de k_hi.  Para bajistas, dirección inversa.
    """
    s = np.asarray(spots, dtype=float)
    total = max_profit + max_loss          # = (k_hi - k_lo) * 100 (por construcción)
    width = k_hi - k_lo

    if strategy in ("BULL_CALL", "BULL_PUT"):
        raw = -max_loss + total * (s - k_lo) / width
    else:                                  # BEAR_CALL, BEAR_PUT
        raw = max_profit - total * (s - k_lo) / width

    return np.clip(raw, -max_loss, max_profit)


def spread_expected_value(
    strategy: str,
    k_lo: float,
    k_hi: float,
    max_profit: float,
    max_loss: float,
    spot: float,
    days: int,
    sigma: float,
    drift: float = 0.0,
    n_points: int = 400,
) -> float:
    """EV en ARS por contrato via integración numérica (medida física, drift=0).

    E[payoff] = integral payoff(S_T) * phi(z) dz
    donde S_T = spot * exp((drift - sigma²/2)*T + sigma*sqrt(T)*z)

    drift=0 y sigma=vol_realizada dan la distribución física conservadora.
    """
    T = max(days / 365.0, 1e-6)
    z   = np.linspace(-5.0, 5.0, n_points)
    dz  = z[1] - z[0]
    phi = np.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi)
    s_T = spot * np.exp((drift - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * z)
    payoffs = spread_payoff(strategy, k_lo, k_hi, max_profit, max_loss, s_T)
    return round(float(np.sum(payoffs * phi * dz)), 2)


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


def _net_greeks(
    spot: float,
    sigma: float,
    T: float,
    long_K: float,
    long_type: str,
    short_K: float,
    short_type: str,
) -> tuple[float, float, float]:
    """Greeks netos (long − short), escalados a 1 contrato (×100).

    Usa Black-Scholes: 10× más rápido que CRR y <1pp de error para spreads
    no-deep-ITM con T ≥ 7d. Drift/tasa = 0, consistente con PoP/EV del bot.
    Retorna (net_delta, net_theta, net_vega).
    """
    from optionsdesk.core.pricing import bs_delta, bs_theta, bs_vega
    if T <= 0 or sigma <= 0:
        return 0.0, 0.0, 0.0
    d = (bs_delta(spot, long_K,  T, 0.0, sigma, long_type)
         - bs_delta(spot, short_K, T, 0.0, sigma, short_type)) * 100.0
    t = (bs_theta(spot, long_K,  T, 0.0, sigma, long_type)
         - bs_theta(spot, short_K, T, 0.0, sigma, short_type)) * 100.0
    v = (bs_vega(spot, long_K,  T, 0.0, sigma)
         - bs_vega(spot, short_K, T, 0.0, sigma)) * 100.0
    return d, t, v


# ── Builders (funciones puras, testeables) ────────────────────────────────────

def build_bull_call_spread(
    long_symbol: str,  long_strike: float,  long_bid: float,  long_ask: float,
    short_symbol: str, short_strike: float, short_bid: float, short_ask: float,
    expiration: date,
    days: int,
    spot: float,
    sigma: float = _DEFAULT_SIGMA,
    r: float = 0.0,           # interpretado como drift (default 0 = fisico)
    cost_model: CostModel = DEFAULT_COSTS,
    price_mode: str = "mid",  # "executable" → buy@ask / sell@bid
) -> Optional[SpreadResult]:
    """BUY call K_low + SELL call K_high.  Alcista, debito, max-loss = debito neto."""
    if long_strike >= short_strike:
        return None
    long_p  = _eff_price_leg(long_bid,  long_ask,  price_mode, is_buy=True)
    short_p = _eff_price_leg(short_bid, short_ask, price_mode, is_buy=False)
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
    ev        = spread_expected_value(
        "BULL_CALL", long_strike, short_strike,
        max_profit, max_loss, spot, days, sigma_use, drift=r,
    )
    ev_ret    = round(ev / max_loss * 100.0, 2) if max_loss > 0 else 0.0
    nd, nt, nv = _net_greeks(spot, sigma_use, T, long_strike, "C", short_strike, "C")

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
        expected_value=ev,   ev_return_pct=ev_ret,
        net_delta=round(nd, 2), net_theta=round(nt, 2), net_vega=round(nv, 2),
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
    short_p = _eff_price_leg(short_bid, short_ask, price_mode, is_buy=False)
    long_p  = _eff_price_leg(long_bid,  long_ask,  price_mode, is_buy=True)
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
    ev        = spread_expected_value(
        "BEAR_CALL", short_strike, long_strike,
        max_profit, max_loss, spot, days, sigma_use, drift=r,
    )
    ev_ret    = round(ev / max_loss * 100.0, 2) if max_loss > 0 else 0.0
    # BEAR_CALL: long leg es la más alta (K_hi), short es K_lo
    nd, nt, nv = _net_greeks(spot, sigma_use, T, long_strike, "C", short_strike, "C")

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
        expected_value=ev,   ev_return_pct=ev_ret,
        net_delta=round(nd, 2), net_theta=round(nt, 2), net_vega=round(nv, 2),
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
    short_p = _eff_price_leg(short_bid, short_ask, price_mode, is_buy=False)
    long_p  = _eff_price_leg(long_bid,  long_ask,  price_mode, is_buy=True)
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
    ev        = spread_expected_value(
        "BULL_PUT", long_strike, short_strike,
        max_profit, max_loss, spot, days, sigma_use, drift=r,
    )
    ev_ret    = round(ev / max_loss * 100.0, 2) if max_loss > 0 else 0.0
    nd, nt, nv = _net_greeks(spot, sigma_use, T, long_strike, "P", short_strike, "P")

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
        expected_value=ev,   ev_return_pct=ev_ret,
        net_delta=round(nd, 2), net_theta=round(nt, 2), net_vega=round(nv, 2),
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
    long_p  = _eff_price_leg(long_bid,  long_ask,  price_mode, is_buy=True)
    short_p = _eff_price_leg(short_bid, short_ask, price_mode, is_buy=False)
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
    ev        = spread_expected_value(
        "BEAR_PUT", short_strike, long_strike,
        max_profit, max_loss, spot, days, sigma_use, drift=r,
    )
    ev_ret    = round(ev / max_loss * 100.0, 2) if max_loss > 0 else 0.0
    nd, nt, nv = _net_greeks(spot, sigma_use, T, long_strike, "P", short_strike, "P")

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
        expected_value=ev,   ev_return_pct=ev_ret,
        net_delta=round(nd, 2), net_theta=round(nt, 2), net_vega=round(nv, 2),
    )
