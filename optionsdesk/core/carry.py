"""Tagged carry estimates for option pricing.

For live decisions we prefer put-call parity observed in the option chain.
Configured caucion is an explicit fallback, never an invisible r=0 default.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from statistics import median

from optionsdesk.core.instruments import OptionType, merge_expiry_calendars, parse_option_symbol
from optionsdesk.data.providers.base import OptionsChain


@dataclass(frozen=True)
class CarryEstimate:
    tna_pct: float
    continuous_rate: float
    source: str
    pairs_used: int = 0


def estimate_carry(
    chain: OptionsChain,
    expiry_calendar: dict,
    *,
    fallback_tna_pct: float,
    max_quote_age_s: float = 120.0,
) -> CarryEstimate:
    """Infer annual carry from fresh ATM call-put pairs, with tagged fallback."""
    spot = float(chain.spot.mid or 0.0)
    if spot <= 0:
        return _fallback(fallback_tna_pct)

    effective_calendar = merge_expiry_calendars(expiry_calendar, chain.expiry_calendar)
    pairs: dict[tuple[str, float, int], dict[OptionType, float]] = {}
    today = datetime.now().date()

    for symbol, quote in chain.options.items():
        if quote.bid <= 0 or quote.ask <= 0 or quote.is_stale(max_quote_age_s):
            continue
        contract = parse_option_symbol(symbol, effective_calendar, spot_hint=spot)
        if contract is None:
            continue
        days = (contract.expiration - today).days
        if days < 5 or days > 60:
            continue
        if abs(contract.strike - spot) / spot > 0.12:
            continue
        pairs.setdefault((contract.expiry_code, contract.strike, days), {})[
            contract.option_type
        ] = quote.mid

    estimates: list[tuple[float, float]] = []
    for (_expiry, strike, days), sides in pairs.items():
        call = sides.get(OptionType.CALL)
        put = sides.get(OptionType.PUT)
        if call is None or put is None or strike <= 0:
            continue
        present_value_strike = spot - (call - put)
        if present_value_strike <= 0:
            continue
        years = days / 365.0
        r = -math.log(present_value_strike / strike) / years
        effective = math.exp(r) - 1.0
        if -0.20 <= effective <= 3.0:
            estimates.append((abs(strike - spot), effective))

    if not estimates:
        return _fallback(fallback_tna_pct)

    estimates.sort(key=lambda item: item[0])
    nearest = [rate for _, rate in estimates[: min(len(estimates), 8)]]
    tna = median(nearest) * 100.0
    return CarryEstimate(
        tna_pct=round(tna, 4),
        continuous_rate=math.log1p(tna / 100.0),
        source="put_call_parity",
        pairs_used=len(nearest),
    )


def _fallback(tna_pct: float) -> CarryEstimate:
    tna = max(float(tna_pct or 0.0), 0.0)
    return CarryEstimate(
        tna_pct=round(tna, 4),
        continuous_rate=math.log1p(tna / 100.0),
        source="configured_caucion_fallback",
    )
