"""Proveedor de datos sintéticos para desarrollo, tests y demos offline.

Genera una cadena de opciones de GGAL realista usando Black-Scholes con un
leve ruido aleatorio para simular un feed en vivo. Funciona sin credenciales.
"""
from __future__ import annotations

import math
import random
from datetime import date, datetime, timedelta
from typing import Optional

from optionsdesk.core.pricing import bs_price
from optionsdesk.data.providers.base import MarketDataProvider, OptionsChain, Quote

# Parámetros de mercado sintético
_SPOT = 8_500.0
_SIGMA = 0.65       # ~65% vol anual (típico para GGAL)
_CAUCION_TNA = 115.0
_STRIKES = [6_000, 6_500, 7_000, 7_500, 8_000, 8_500, 9_000, 9_500, 10_000, 11_000, 12_000]


def _future_expiry_codes(n: int = 3) -> list[tuple[str, date]]:
    """Devuelve los primeros N códigos futuros del calendario de instruments.yaml."""
    try:
        from optionsdesk.core.instruments import load_expiry_calendar
        cal = load_expiry_calendar()
        today = date.today()
        future = sorted(
            [(code, dt) for code, dt in cal.items() if dt > today],
            key=lambda x: x[1],
        )
        return future[:n]
    except Exception:
        pass
    # Fallback: generar vencimientos sintéticos si no hay calendario
    today = date.today()
    return [
        (f"X{i}", today + timedelta(days=21 * (i + 1)))
        for i in range(n)
    ]


def _gen_quote(symbol: str, mid: float, spread_frac: float = 0.05) -> Quote:
    spread = max(mid * spread_frac, 1.0)
    bid = max(mid - spread / 2.0, 0.5)
    ask = bid + spread
    last = mid * (1.0 + random.gauss(0, 0.008))
    return Quote(
        symbol=symbol,
        bid=round(bid, 2),
        ask=round(ask, 2),
        last=round(max(last, 0.5), 2),
        volume=random.randint(10, 500),
        timestamp=datetime.now(),
    )


class DemoProvider(MarketDataProvider):
    """Cadena sintética de GGAL para desarrollo y demo sin credenciales."""

    def __init__(
        self,
        spot: float = _SPOT,
        sigma: float = _SIGMA,
        caucion_tna_pct: float = _CAUCION_TNA,
    ) -> None:
        self._spot = spot
        self._sigma = sigma
        self._caucion_tna = caucion_tna_pct
        self._drift = 0.0

    def get_spot(self) -> Quote:
        self._drift += random.gauss(0, self._spot * 0.002)
        s = self._spot + self._drift
        return _gen_quote("GGAL", s, spread_frac=0.002)

    def get_caucion_tna(self, days: int = 30) -> float:
        return self._caucion_tna + random.gauss(0, 0.3)

    def get_options_chain(self) -> OptionsChain:
        spot_quote = self.get_spot()
        S = spot_quote.mid
        # Tasa continua equivalente a la caución TNA
        r_cont = math.log(1.0 + self._caucion_tna / 100.0) * 30.0 / 365.0

        future_expiries = _future_expiry_codes(3)

        options: dict[str, Quote] = {}
        for code, expiry in future_expiries:
            T = (expiry - date.today()).days / 365.0
            if T <= 0:
                continue
            for K in _STRIKES:
                for otype, prefix in [("C", "GFGC"), ("P", "GFGV")]:
                    mid = bs_price(S, K, T, r_cont, self._sigma, otype)
                    if mid < 0.5:
                        continue
                    moneyness_frac = abs(S - K) / S
                    spread_f = 0.05 + moneyness_frac * 0.12
                    symbol = f"{prefix}{int(K)}{code}"
                    options[symbol] = _gen_quote(symbol, mid, spread_frac=spread_f)

        return OptionsChain(
            underlying="GGAL",
            spot=spot_quote,
            options=options,
        )

    def is_connected(self) -> bool:
        return True
