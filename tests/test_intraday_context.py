from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from optionsdesk.signals.intraday_context import build_intraday_snapshot, resample_ohlc_bars
from optionsdesk.signals.scalping import LivePulse

BA = ZoneInfo("America/Argentina/Buenos_Aires")


def _bars(start: datetime, closes: list[float], step: timedelta = timedelta(minutes=1)) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "time": start + step * i,
                "open": close - 0.5,
                "high": close + 0.8,
                "low": close - 0.8,
                "close": close,
                "volume": 0.0,
            }
            for i, close in enumerate(closes)
        ]
    )


def test_intraday_snapshot_detects_local_bullish_structure_from_completed_bars():
    now = datetime(2026, 6, 2, 12, 10, tzinfo=BA)
    bars_1m = _bars(
        datetime(2026, 6, 2, 11, 58, tzinfo=BA),
        [1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1009, 1011, 1013],
    )
    bars_5m = resample_ohlc_bars(bars_1m, "5min")

    snap = build_intraday_snapshot(
        bars_1m,
        bars_5m,
        LivePulse(True, direction="BULL", observations=5, span_s=90.0),
        now=now,
        momentum_threshold_pct=0.18,
    )

    assert snap is not None
    assert snap.trend == "ALCISTA"
    assert snap.bos == "BOS_UP"
    assert snap.vol_confirms is True
    assert snap.momentum_threshold_pct == 0.18


def test_intraday_snapshot_returns_none_until_context_exists():
    now = datetime(2026, 6, 2, 12, 10, tzinfo=BA)

    snap = build_intraday_snapshot(
        _bars(datetime(2026, 6, 2, 12, 5, tzinfo=BA), [1000, 1001]),
        None,
        LivePulse(False),
        now=now,
    )

    assert snap is None


def test_intraday_snapshot_excludes_current_open_bar_from_structure():
    now = datetime(2026, 6, 2, 12, 10, 30, tzinfo=BA)
    bars_1m = _bars(
        datetime(2026, 6, 2, 12, 3, tzinfo=BA),
        [100.0, 100.1, 100.2, 100.3, 100.4, 100.5, 100.6, 110.0],
    )

    snap = build_intraday_snapshot(
        bars_1m,
        None,
        LivePulse(True, direction="BULL", observations=5, span_s=90.0),
        now=now,
        momentum_threshold_pct=0.18,
    )

    assert snap is not None
    assert snap.bos is None
