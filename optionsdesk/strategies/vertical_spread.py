"""Scanner de spreads verticales (2 patas, riesgo definido) para GGAL en BYMA.

Arma bull/bear call spreads y bull/bear put spreads segun el contexto de mercado.
Rankea por risk_reward x prob_of_profit.

Restriccion de liquidez: ambas patas deben pasar los filtros individuales.
Con dos spreads bid-ask amplios, el edge se acumula — el filtro es estricto.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

from optionsdesk.config.costs import CostModel, DEFAULT_COSTS
from optionsdesk.core.benchmark import Benchmark
from optionsdesk.core.instruments import OptionType, parse_option_symbol
from optionsdesk.core.pricing import implied_vol
from optionsdesk.core.spreads import (
    SpreadResult,
    build_bull_call_spread, build_bear_call_spread,
    build_bull_put_spread,  build_bear_put_spread,
)
from optionsdesk.data.providers.base import OptionsChain, Quote

logger = logging.getLogger(__name__)

_DEFAULT_SIGMA = 0.65   # vol anual de referencia si no se puede calcular IV


@dataclass
class VerticalSpreadConfig:
    min_bid: float          = 1.0
    max_spread_pct: float   = 30.0   # realista para BYMA; single-leg usa 30% tambien
    max_price_age_s: float  = 300.0
    min_days: int           = 7
    max_days: int           = 45
    spot_band_pct: float    = 0.15   # busca strikes dentro del ±15% del spot
    max_width_pct: float    = 0.08   # ancho maximo del spread como % del spot
    price_mode: str         = "mid"
    pricing_steps: int      = 50     # pasos CRR para IV (menos que single-leg para velocidad)
    cost_model: Optional[CostModel] = field(default=None)

    def __post_init__(self) -> None:
        if self.cost_model is None:
            self.cost_model = DEFAULT_COSTS


class VerticalSpreadScanner:
    """Scana spreads verticales de 2 patas y los rankea por risk_reward x PoP."""

    def __init__(
        self,
        config: Optional[VerticalSpreadConfig] = None,
        expiry_calendar: Optional[dict] = None,
    ) -> None:
        self._cfg = config or VerticalSpreadConfig()
        self._expiry_cal = expiry_calendar or {}

    def scan(
        self,
        chain: OptionsChain,
        benchmark: Benchmark,
        direction: str = "both",   # "bullish" | "bearish" | "both"
    ) -> list[SpreadResult]:
        """Devuelve spreads ordenados por risk_reward x PoP.

        direction: 'bullish' → solo bull spreads (call/put)
                   'bearish' → solo bear spreads (call/put)
                   'both'    → todos los lados
        """
        S = chain.spot.mid
        if S <= 0:
            return []

        r = math.log(1.0 + benchmark.caucion_tna_pct / 100.0) if benchmark.caucion_tna_pct > 0 else 0.0

        # ── Agrupar opciones liquidas por (tipo, vencimiento) ─────────────────
        # key  : (OptionType, expiry_code, expiration_date, days)
        # value: {strike → (symbol, Quote, implied_vol_or_default)}
        groups: dict[tuple, dict[float, tuple[str, Quote, float]]] = {}

        lo_band = S * (1 - self._cfg.spot_band_pct)
        hi_band = S * (1 + self._cfg.spot_band_pct)

        for sym, quote in chain.options.items():
            contract = parse_option_symbol(sym, self._expiry_cal)
            if contract is None:
                continue
            days = contract.days_to_expiry
            if not (self._cfg.min_days <= days <= self._cfg.max_days):
                continue
            if not self._is_liquid(quote):
                continue
            if not (lo_band <= contract.strike <= hi_band):
                continue

            T   = contract.years_to_expiry
            mid = quote.mid if quote.mid > 0 else quote.last
            ot  = "C" if contract.option_type == OptionType.CALL else "P"

            iv_val = _DEFAULT_SIGMA
            if mid > 0 and T > 0:
                try:
                    computed = implied_vol(
                        mid, S, contract.strike, T, r, ot,
                        american=True, steps=self._cfg.pricing_steps,
                    )
                    if computed:
                        iv_val = computed
                except Exception:
                    pass

            key = (contract.option_type, contract.expiry_code, contract.expiration, days)
            groups.setdefault(key, {})[contract.strike] = (sym, quote, iv_val)

        # ── Componer pares de strikes adyacentes ──────────────────────────────
        results: list[SpreadResult] = []
        max_width = S * self._cfg.max_width_pct

        for (opt_type, _code, expiration, days), strike_map in groups.items():
            strikes = sorted(strike_map.keys())
            if len(strikes) < 2:
                continue

            for i in range(len(strikes) - 1):
                K_lo, K_hi = strikes[i], strikes[i + 1]
                if K_hi - K_lo > max_width:
                    continue

                sym_lo, q_lo, iv_lo = strike_map[K_lo]
                sym_hi, q_hi, iv_hi = strike_map[K_hi]
                sigma = (iv_lo + iv_hi) / 2.0

                kwargs = dict(
                    expiration=expiration, days=days, spot=S,
                    sigma=sigma, r=r,
                    cost_model=self._cfg.cost_model,
                    price_mode=self._cfg.price_mode,
                )

                if opt_type == OptionType.CALL:
                    if direction in ("bullish", "both"):
                        sr = build_bull_call_spread(
                            sym_lo, K_lo, q_lo.bid, q_lo.ask,
                            sym_hi, K_hi, q_hi.bid, q_hi.ask,
                            **kwargs,
                        )
                        if sr:
                            results.append(sr)
                    if direction in ("bearish", "both"):
                        sr = build_bear_call_spread(
                            sym_lo, K_lo, q_lo.bid, q_lo.ask,
                            sym_hi, K_hi, q_hi.bid, q_hi.ask,
                            **kwargs,
                        )
                        if sr:
                            results.append(sr)

                else:  # PUT
                    if direction in ("bullish", "both"):
                        # bull put: sell K_hi, buy K_lo
                        sr = build_bull_put_spread(
                            sym_hi, K_hi, q_hi.bid, q_hi.ask,
                            sym_lo, K_lo, q_lo.bid, q_lo.ask,
                            **kwargs,
                        )
                        if sr:
                            results.append(sr)
                    if direction in ("bearish", "both"):
                        # bear put: buy K_hi, sell K_lo
                        sr = build_bear_put_spread(
                            sym_hi, K_hi, q_hi.bid, q_hi.ask,
                            sym_lo, K_lo, q_lo.bid, q_lo.ask,
                            **kwargs,
                        )
                        if sr:
                            results.append(sr)

        # Score: risk_reward x PoP (premia R:R bueno con alta probabilidad)
        results.sort(key=lambda x: x.risk_reward * x.prob_of_profit, reverse=True)
        return results

    def _is_liquid(self, quote: Quote) -> bool:
        if quote.bid < self._cfg.min_bid:
            return False
        if quote.ask <= 0:
            return False
        mid = (quote.bid + quote.ask) / 2.0 if quote.bid > 0 and quote.ask > 0 else 0.0
        if mid > 0 and (quote.ask - quote.bid) / mid * 100.0 > self._cfg.max_spread_pct:
            return False
        if quote.is_stale(self._cfg.max_price_age_s):
            return False
        return True
