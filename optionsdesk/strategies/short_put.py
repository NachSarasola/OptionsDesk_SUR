"""Scanner de venta de puts cash-secured en la cadena de GGAL."""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

from optionsdesk.config.costs import CostModel, DEFAULT_COSTS
from optionsdesk.core.benchmark import Benchmark
from optionsdesk.core.instruments import OptionType, merge_expiry_calendars, parse_option_symbol
from optionsdesk.core.pricing import crr_delta, implied_vol
from optionsdesk.core.rates import RateResult, compute_short_put
from optionsdesk.data.providers.base import OptionsChain, Quote
from optionsdesk.strategies.base import Strategy

logger = logging.getLogger(__name__)


@dataclass
class ShortPutConfig:
    min_bid: float = 1.0
    max_spread_pct: float = 30.0
    max_price_age_s: float = 300.0
    price_mode: str = "mid"
    min_days: int = 5
    max_days: int = 90
    pricing_steps: int = 100
    capital_mode: str = "net"       # 'net' (K-P) o 'gross' (K)
    atm_threshold_pct: float = 2.0
    cost_model: Optional[CostModel] = field(default=None)

    def __post_init__(self) -> None:
        if self.cost_model is None:
            self.cost_model = DEFAULT_COSTS


class ShortPutScanner(Strategy):
    """Scanner de venta de puts. Rankea por spread TNA vs caución."""

    def __init__(
        self,
        config: Optional[ShortPutConfig] = None,
        expiry_calendar: Optional[dict] = None,
    ) -> None:
        self._cfg = config or ShortPutConfig()
        self._expiry_cal = expiry_calendar or {}

    def scan(self, chain: OptionsChain, benchmark: Benchmark) -> list[RateResult]:
        S = chain.spot.mid
        if S <= 0:
            return []

        r = math.log(1.0 + benchmark.caucion_tna_pct / 100.0) if benchmark.caucion_tna_pct > 0 else 0.0
        results: list[RateResult] = []
        effective_calendar = merge_expiry_calendars(self._expiry_cal, chain.expiry_calendar)

        for sym, quote in chain.options.items():
            contract = parse_option_symbol(sym, effective_calendar, spot_hint=S)
            if contract is None or contract.option_type != OptionType.PUT:
                continue

            days = contract.days_to_expiry
            if not (self._cfg.min_days <= days <= self._cfg.max_days):
                continue

            if not self._is_liquid(quote):
                continue

            T = contract.years_to_expiry
            put_mid = quote.mid if quote.mid > 0 else quote.last

            iv = implied_vol(
                put_mid, S, contract.strike, T, r, "P",
                american=True, steps=self._cfg.pricing_steps,
            )
            delta = (
                crr_delta(S, contract.strike, T, r, iv, "P",
                          american=True, steps=self._cfg.pricing_steps)
                if iv else None
            )

            result = compute_short_put(
                contract=contract,
                put_bid=quote.bid,
                put_ask=quote.ask,
                spot=S,
                put_last=quote.last,
                delta=delta,
                iv=iv,
                cost_model=self._cfg.cost_model,
                price_mode=self._cfg.price_mode,
                capital_mode=self._cfg.capital_mode,
                is_liquid=True,
                atm_threshold_pct=self._cfg.atm_threshold_pct,
            )
            if result is None or result.tna_pct <= 0:
                continue

            result.set_spread(benchmark.caucion_tna_pct)
            results.append(result)

        results.sort(key=lambda x: x.spread_vs_caucion_pct, reverse=True)
        return results

    def _is_liquid(self, quote: Quote) -> bool:
        if quote.bid < self._cfg.min_bid:
            return False
        if quote.ask <= 0:
            return False
        mid = (quote.bid + quote.ask) / 2.0
        if mid > 0 and (quote.ask - quote.bid) / mid * 100.0 > self._cfg.max_spread_pct:
            return False
        if quote.is_stale(self._cfg.max_price_age_s):
            return False
        return True
