from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

from optionsdesk.backtest.scalping import (
    ScalpReplayPlan,
    simulate_plan_on_snapshots,
    simulate_plans_on_snapshots,
)


def _row(ts: datetime, bid: float, ask: float, symbol: str = "GFGC4000A") -> dict:
    return {
        "timestamp": ts,
        "symbol": symbol,
        "spot": 4000.0,
        "bid": bid,
        "ask": ask,
        "last": (bid + ask) / 2.0,
        "volume": 100.0,
    }


def test_replay_takes_profit_without_lookahead():
    t0 = datetime(2026, 5, 22, 11, 0)
    df = pd.DataFrame(
        [
            _row(t0, 90.0, 95.0),   # before signal, should be ignored
            _row(t0 + timedelta(minutes=1), 99.0, 100.0),
            _row(t0 + timedelta(minutes=2), 121.0, 123.0),
        ]
    )
    plan = ScalpReplayPlan(
        symbol="GFGC4000A",
        action="BUY_CALL",
        entry_limit=101.0,
        stop_price=90.0,
        target_price=120.0,
        created_ts=t0,
    )

    trade = simulate_plan_on_snapshots(df, plan)

    assert trade is not None
    assert trade.entry_ts == t0 + timedelta(minutes=1)
    assert trade.exit_reason == "TAKE_PROFIT"
    assert trade.pnl_ars > 0


def test_replay_stops_long_scalp():
    t0 = datetime(2026, 5, 22, 11, 0)
    df = pd.DataFrame(
        [
            _row(t0, 99.0, 100.0),
            _row(t0 + timedelta(minutes=1), 88.0, 90.0),
        ]
    )
    plan = ScalpReplayPlan(
        symbol="GFGC4000A",
        action="BUY_CALL",
        entry_limit=101.0,
        stop_price=90.0,
        target_price=120.0,
    )

    trade = simulate_plan_on_snapshots(df, plan)

    assert trade is not None
    assert trade.exit_reason == "STOP"
    assert trade.pnl_ars < 0


def test_replay_time_stop_and_session_close():
    t0 = datetime(2026, 5, 22, 11, 0)
    df_time = pd.DataFrame(
        [
            _row(t0, 99.0, 100.0),
            _row(t0 + timedelta(minutes=21), 100.0, 102.0),
        ]
    )
    time_plan = ScalpReplayPlan(
        symbol="GFGC4000A",
        action="BUY_CALL",
        entry_limit=101.0,
        stop_price=90.0,
        target_price=120.0,
        time_stop_min=20,
    )
    time_trade = simulate_plan_on_snapshots(df_time, time_plan)
    assert time_trade is not None
    assert time_trade.exit_reason == "TIME_STOP"

    close_t = datetime(2026, 5, 22, 16, 45)
    df_close = pd.DataFrame(
        [
            _row(close_t, 99.0, 100.0),
            _row(close_t + timedelta(minutes=6), 100.0, 102.0),
        ]
    )
    close_plan = ScalpReplayPlan(
        symbol="GFGC4000A",
        action="BUY_CALL",
        entry_limit=101.0,
        stop_price=90.0,
        target_price=120.0,
        time_stop_min=90,
        flat_before_close_min=10,
    )
    close_trade = simulate_plan_on_snapshots(df_close, close_plan)
    assert close_trade is not None
    assert close_trade.exit_reason == "SESSION_CLOSE"


def test_replay_result_builds_equity_curve():
    t0 = datetime(2026, 5, 22, 11, 0)
    df = pd.DataFrame(
        [
            _row(t0, 99.0, 100.0),
            _row(t0 + timedelta(minutes=1), 121.0, 123.0),
        ]
    )
    plan = ScalpReplayPlan(
        symbol="GFGC4000A",
        action="BUY_CALL",
        entry_limit=101.0,
        stop_price=90.0,
        target_price=120.0,
    )

    result = simulate_plans_on_snapshots(df, [plan], capital_initial=100_000.0)

    assert len(result.trades) == 1
    assert result.pnl_series == [result.trades[0].pnl_ars]
    assert not result.equity_curve.empty
