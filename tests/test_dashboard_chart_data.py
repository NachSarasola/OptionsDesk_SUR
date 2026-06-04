from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from optionsdesk.data.history import daily_with_live_spot, tape_ohlc, weekly_from_daily


_BA = ZoneInfo("America/Argentina/Buenos_Aires")


def test_tape_ohlc_builds_real_one_minute_bars():
    start = datetime(2026, 6, 1, 12, 0, tzinfo=_BA)
    samples = [
        (start, 100.0),
        (start + timedelta(seconds=20), 102.0),
        (start + timedelta(seconds=50), 99.0),
        (start + timedelta(minutes=1, seconds=5), 101.0),
    ]

    bars = tape_ohlc(samples, "1min")

    assert len(bars) == 2
    assert bars.iloc[0]["open"] == pytest.approx(100.0)
    assert bars.iloc[0]["high"] == pytest.approx(102.0)
    assert bars.iloc[0]["low"] == pytest.approx(99.0)
    assert bars.iloc[0]["close"] == pytest.approx(99.0)
    assert bars["volume"].eq(0.0).all()


def test_daily_with_live_spot_appends_today_candle():
    now = datetime(2026, 6, 1, 14, 0, tzinfo=_BA)
    history = pd.DataFrame([{
        "date": pd.Timestamp("2026-05-29"),
        "open": 95.0,
        "high": 101.0,
        "low": 94.0,
        "close": 100.0,
        "volume": 1000.0,
    }])
    samples = [
        (now - timedelta(minutes=2), 103.0),
        (now - timedelta(minutes=1), 105.0),
    ]

    result = daily_with_live_spot(history, samples, spot=104.0, now=now)

    assert len(result) == 2
    today = result.iloc[-1]
    assert today["date"] == pd.Timestamp("2026-06-01")
    assert today["open"] == pytest.approx(103.0)
    assert today["high"] == pytest.approx(105.0)
    assert today["low"] == pytest.approx(103.0)
    assert today["close"] == pytest.approx(104.0)


def test_daily_with_live_spot_updates_existing_today_candle():
    now = datetime(2026, 6, 1, 14, 0, tzinfo=_BA)
    history = pd.DataFrame([{
        "date": pd.Timestamp("2026-06-01"),
        "open": 100.0,
        "high": 104.0,
        "low": 98.0,
        "close": 102.0,
        "volume": 1234.0,
    }])

    result = daily_with_live_spot(history, [(now, 105.0)], spot=103.0, now=now)

    today = result.iloc[-1]
    assert today["open"] == pytest.approx(100.0)
    assert today["high"] == pytest.approx(105.0)
    assert today["low"] == pytest.approx(98.0)
    assert today["close"] == pytest.approx(103.0)
    assert today["volume"] == pytest.approx(1234.0)


def test_weekly_from_daily_includes_current_live_week():
    daily = pd.DataFrame([
        {"date": pd.Timestamp("2026-05-29"), "open": 95.0, "high": 101.0, "low": 94.0, "close": 100.0, "volume": 10.0},
        {"date": pd.Timestamp("2026-06-01"), "open": 103.0, "high": 105.0, "low": 102.0, "close": 104.0, "volume": 0.0},
    ])

    result = weekly_from_daily(daily)

    assert len(result) == 2
    assert result.iloc[-1]["date"] == pd.Timestamp("2026-06-05")
    assert result.iloc[-1]["close"] == pytest.approx(104.0)
