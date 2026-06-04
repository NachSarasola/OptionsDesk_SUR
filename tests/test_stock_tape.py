from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from optionsdesk.data.providers.base import Quote
from optionsdesk.data.stock_tape import append_stock_quotes, load_stock_tape, stock_tape_bars


BA = ZoneInfo("America/Argentina/Buenos_Aires")


def _quote(symbol: str, bid: float, ask: float, last: float, ts: datetime) -> Quote:
    return Quote(
        symbol=symbol,
        bid=bid,
        ask=ask,
        last=last,
        volume=1000.0,
        timestamp=ts,
        source="TEST",
        received_at=ts,
        book_ts=ts,
    )


def test_stock_tape_appends_loads_and_deduplicates(tmp_path):
    now = datetime(2026, 6, 3, 11, 30, tzinfo=BA)
    q1 = _quote("GGAL", 99.0, 101.0, 100.0, now)
    q2 = _quote("YPFD", 199.0, 201.0, 200.0, now)

    assert append_stock_quotes([q1, q2], tape_dir=tmp_path, now=now, min_interval_s=10) == 2
    assert append_stock_quotes([q1], tape_dir=tmp_path, now=now + timedelta(seconds=5), min_interval_s=10) == 0

    df = load_stock_tape(tape_dir=tmp_path, now=now + timedelta(seconds=20), symbols=["GGAL"])

    assert len(df) == 1
    assert df.iloc[0]["symbol"] == "GGAL"
    assert df.iloc[0]["mid"] == pytest.approx(100.0)


def test_stock_tape_bars_uses_real_mid_prices(tmp_path):
    now = datetime(2026, 6, 3, 11, 30, tzinfo=BA)
    quotes = [
        _quote("GGAL", 99.0, 101.0, 100.0, now),
        _quote("GGAL", 100.0, 102.0, 101.0, now + timedelta(minutes=1)),
        _quote("GGAL", 101.0, 103.0, 102.0, now + timedelta(minutes=2)),
    ]
    for quote in quotes:
        append_stock_quotes([quote], tape_dir=tmp_path, now=quote.book_ts, min_interval_s=0)

    df = load_stock_tape(tape_dir=tmp_path, now=now + timedelta(minutes=3), symbols=["GGAL"])
    bars = stock_tape_bars(df, "GGAL", "1min")

    assert len(bars) == 3
    assert bars.iloc[0]["open"] == pytest.approx(100.0)
    assert bars.iloc[-1]["close"] == pytest.approx(102.0)
    assert bars["volume"].sum() == pytest.approx(0.0)
