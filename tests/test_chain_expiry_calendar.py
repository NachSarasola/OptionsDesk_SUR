from datetime import date, datetime, timedelta

from optionsdesk.core.benchmark import ZERO_BENCHMARK
from optionsdesk.data.providers.base import OptionsChain, Quote
from optionsdesk.data.providers.demo import DemoProvider
from optionsdesk.strategies.covered_call import CoveredCallScanner
from optionsdesk.strategies.short_put import ShortPutScanner
from optionsdesk.strategies.vertical_spread import VerticalSpreadScanner


def _quote(symbol: str, bid: float, ask: float, last: float | None = None) -> Quote:
    return Quote(
        symbol=symbol,
        bid=bid,
        ask=ask,
        last=last if last is not None else (bid + ask) / 2.0,
        volume=100.0,
        timestamp=datetime.now(),
    )


def test_rate_scanners_use_observed_calendar_when_local_calendar_is_empty():
    expiry = date.today() + timedelta(days=30)
    chain = OptionsChain(
        underlying="GGAL",
        spot=_quote("GGAL", 99.0, 100.0, 100.0),
        options={
            "GFGC90JU": _quote("GFGC90JU", 11.0, 12.0),
            "GFGV90JU": _quote("GFGV90JU", 2.0, 2.2),
        },
        expiry_calendar={"JU": expiry},
    )

    assert CoveredCallScanner(expiry_calendar={}).scan(chain, ZERO_BENCHMARK)
    assert ShortPutScanner(expiry_calendar={}).scan(chain, ZERO_BENCHMARK)


def test_vertical_spread_uses_observed_calendar_when_local_calendar_is_empty():
    expiry = date.today() + timedelta(days=30)
    chain = OptionsChain(
        underlying="GGAL",
        spot=_quote("GGAL", 8290.0, 8310.0, 8300.0),
        options={
            "GFGC8000JU": _quote("GFGC8000JU", 280.0, 320.0),
            "GFGC8500JU": _quote("GFGC8500JU", 130.0, 170.0),
        },
        expiry_calendar={"JU": expiry},
    )

    results = VerticalSpreadScanner(expiry_calendar={}).scan(
        chain,
        ZERO_BENCHMARK,
        direction="bullish",
    )

    assert any(result.strategy == "BULL_CALL" for result in results)


def test_demo_chain_is_self_describing():
    chain = DemoProvider().get_options_chain()

    assert chain.expiry_calendar
