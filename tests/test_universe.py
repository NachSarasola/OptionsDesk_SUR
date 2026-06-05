from __future__ import annotations

from datetime import datetime

from optionsdesk.data.providers.base import Quote
from optionsdesk.data.universe import rank_stock_universe, select_top_symbols


def _quote(symbol: str, price: float, volume: float) -> Quote:
    return Quote(
        symbol=symbol,
        bid=price * 0.999,
        ask=price * 1.001,
        last=price,
        volume=volume,
        timestamp=datetime(2026, 6, 3, 12, 0),
    )


def test_rank_stock_universe_prefers_value_traded():
    quotes = {
        "AAA": _quote("AAA", 10.0, 100.0),
        "BBB": _quote("BBB", 100.0, 50.0),
        "CCC": _quote("CCC", 20.0, 10.0),
    }
    ranked = rank_stock_universe(["AAA", "BBB", "CCC"], quotes=quotes, top_n=2)

    assert [m.symbol for m in ranked] == ["BBB", "AAA"]
    assert ranked[0].avg_value_ars > ranked[1].avg_value_ars


def test_select_top_symbols_falls_back_to_input_order_without_liquidity():
    assert select_top_symbols(["ZZZZ", "YYYY"], quotes={}, top_n=1) == ["ZZZZ"]
