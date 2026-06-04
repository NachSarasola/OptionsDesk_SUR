"""Motor batch de griegas para la cadena de opciones."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional

from optionsdesk.config.constants import DEFAULT_SIGMA_GGAL
from optionsdesk.config.settings import settings
from optionsdesk.core.instruments import OptionType, merge_expiry_calendars, parse_option_symbol
from optionsdesk.core.pricing import bs_delta, bs_gamma, bs_theta, bs_vega, implied_vol
from optionsdesk.data.providers.base import OptionsChain

logger = logging.getLogger(__name__)

# Fallback IV cuando implied_vol() no converge — usa la vol estructural de GGAL
_DEFAULT_IV = DEFAULT_SIGMA_GGAL


@dataclass
class OptionGreeks:
    """Griegas y datos de mercado para una opcion individual."""

    symbol: str
    strike: float
    option_type: str
    days: int
    spot: float
    bid: float
    ask: float
    mid: float
    iv: Optional[float]
    delta: float
    gamma: float
    theta: float
    vega: float
    moneyness: str
    moneyness_pct: float
    volume: float
    spread_pct: float
    bid_size: float = 0.0
    ask_size: float = 0.0
    quote_age_s: float = 0.0
    is_tradeable: bool = True
    liquidity_reason: str = ""


def _quote_age_seconds(ts: datetime) -> float:
    now = datetime.now(ts.tzinfo) if ts.tzinfo else datetime.now()
    return max((now - ts).total_seconds(), 0.0)


def _liquidity_check(
    bid: float,
    ask: float,
    mid: float,
    spread_pct: float,
    volume: float,
    quote_age_s: float,
) -> tuple[bool, str]:
    if bid <= 0 or ask <= 0:
        return False, "sin puntas bid/ask"
    if mid <= 0:
        return False, "mid invalido"
    if volume < settings.scalping_min_volume:
        return False, "volumen bajo"
    if quote_age_s > max(settings.scalping_max_quote_age_s * 10, 900):
        return False, f"quote stale ({quote_age_s:.0f}s)"
    return True, ""


def compute_chain_greeks(
    chain: OptionsChain,
    expiry_calendar: dict,
    r: float = 0.0,
) -> dict[str, OptionGreeks]:
    """Calcula griegas BS para toda la cadena de opciones parseable."""
    S = chain.spot.mid
    if S <= 0:
        return {}

    result: dict[str, OptionGreeks] = {}
    effective_calendar = merge_expiry_calendars(expiry_calendar, chain.expiry_calendar)
    today = date.today()

    for sym, quote in chain.options.items():
        contract = parse_option_symbol(sym, effective_calendar, spot_hint=S)
        if contract is None:
            continue

        days = max((contract.expiration - today).days, 0)
        if days == 0:
            continue

        mid = quote.mid if quote.mid > 0 else quote.last
        if mid <= 0:
            continue

        T = days / 365.0
        K = contract.strike
        ot = "C" if contract.option_type == OptionType.CALL else "P"

        iv: Optional[float] = None
        if T > 0 and mid > 0:
            try:
                iv = implied_vol(
                    market_price=mid,
                    S=S,
                    K=K,
                    T=T,
                    r=r,
                    option_type=ot,
                    american=False,
                    sigma_lo=0.01,
                    sigma_hi=5.0,
                )
            except Exception:
                iv = None

        sigma = iv if iv is not None else _DEFAULT_IV
        if T > 0 and sigma > 0:
            delta = bs_delta(S, K, T, r, sigma, ot)
            gamma = bs_gamma(S, K, T, r, sigma)
            theta = bs_theta(S, K, T, r, sigma, ot)
            vega = bs_vega(S, K, T, r, sigma)
        else:
            delta = 1.0 if (ot == "C" and S > K) else (-1.0 if (ot == "P" and S < K) else 0.0)
            gamma = 0.0
            theta = 0.0
            vega = 0.0

        moneyness_pct = (K - S) / S * 100.0 if ot == "C" else (S - K) / S * 100.0
        if abs(K - S) / S <= 0.02:
            moneyness = "ATM"
        elif (ot == "C" and K < S) or (ot == "P" and K > S):
            moneyness = "ITM"
        else:
            moneyness = "OTM"

        spread_pct = quote.spread_pct
        age_s = _quote_age_seconds(quote.timestamp)
        is_tradeable, reason = _liquidity_check(
            bid=quote.bid,
            ask=quote.ask,
            mid=mid,
            spread_pct=spread_pct,
            volume=quote.volume,
            quote_age_s=age_s,
        )

        result[sym] = OptionGreeks(
            symbol=sym,
            strike=K,
            option_type=ot,
            days=days,
            spot=S,
            bid=quote.bid,
            ask=quote.ask,
            mid=round(mid, 2),
            iv=round(iv, 4) if iv is not None else None,
            delta=round(delta, 4),
            gamma=round(gamma, 6),
            theta=round(theta, 2),
            vega=round(vega, 2),
            moneyness=moneyness,
            moneyness_pct=round(moneyness_pct, 2),
            volume=quote.volume,
            spread_pct=round(spread_pct, 2),
            bid_size=quote.bid_size,
            ask_size=quote.ask_size,
            quote_age_s=round(age_s, 1),
            is_tradeable=is_tradeable,
            liquidity_reason=reason,
        )

    return result
