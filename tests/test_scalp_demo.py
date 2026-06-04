from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from optionsdesk.backtest.scalp_demo import append_demo_trades, run_signal_demo_on_snapshots
from optionsdesk.config.settings import settings

BA = ZoneInfo("America/Argentina/Buenos_Aires")


def _snapshot_rows() -> pd.DataFrame:
    rows = []
    start = datetime(2026, 6, 2, 11, 30, tzinfo=BA)
    spots = [4000, 4002, 4004, 4006, 4008, 4010, 4012, 4018, 4020, 4024, 4030, 4035, 4031, 4045]
    bids = [100, 101, 102, 103, 104, 105, 106, 108, 135, 140, 139, 170, 168, 200]
    asks = [104, 105, 106, 107, 108, 109, 110, 112, 139, 144, 142, 174, 172, 204]
    for i, (spot, bid, ask) in enumerate(zip(spots, bids, asks)):
        ts = start + timedelta(minutes=5 * i)
        rows.append(
            {
                "timestamp": ts,
                "symbol": "GFGC4050T",
                "spot": float(spot),
                "spot_bid": float(spot) - 1,
                "spot_ask": float(spot) + 1,
                "spot_last": float(spot),
                "bid": float(bid),
                "ask": float(ask),
                "last": (bid + ask) / 2.0,
                "volume": 500.0,
                "bid_size": 10.0,
                "ask_size": 10.0,
                "source": "IOL",
                "quote_source": "IOL_REST",
                "received_at": ts,
                "book_ts": ts,
                "carry_tna_pct": 45.0,
            }
        )
        rows.append(
            {
                "timestamp": ts,
                "symbol": "GFGV4050T",
                "spot": float(spot),
                "spot_bid": float(spot) - 1,
                "spot_ask": float(spot) + 1,
                "spot_last": float(spot),
                "bid": 95.0,
                "ask": 100.0,
                "last": 97.5,
                "volume": 500.0,
                "bid_size": 10.0,
                "ask_size": 10.0,
                "source": "IOL",
                "quote_source": "IOL_REST",
                "received_at": ts,
                "book_ts": ts,
                "carry_tna_pct": 45.0,
            }
        )
    return pd.DataFrame(rows)


def _settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "costs_verified", True)
    monkeypatch.setattr(settings, "iol_provisional_tickets_enabled", True)
    monkeypatch.setattr(settings, "iol_ticket_max_quote_age_s", 300)
    monkeypatch.setattr(settings, "iol_ticket_max_spread_pct", 6.0)
    monkeypatch.setattr(settings, "scalping_risk_per_trade_pct", 1.0)
    monkeypatch.setattr(settings, "scalping_start_after_open_min", 20)
    monkeypatch.setattr(settings, "scalping_flat_before_close_min", 15)


def test_signal_demo_replays_generated_ticket_on_realistic_snapshots(monkeypatch: pytest.MonkeyPatch):
    _settings(monkeypatch)
    result = run_signal_demo_on_snapshots(
        _snapshot_rows(),
        {"T": date.today() + timedelta(days=20)},
        capital=2_000_000.0,
        risk_profile="balanced",
        max_trades=5,
        momentum_threshold_pct=0.10,
    )

    assert result.snapshots == 14
    assert len(result.trades) >= 1
    assert result.trades[0].action == "BUY_CALL"
    assert result.trades[0].pnl_ars > 0
    assert result.total_pnl_ars > 0


def test_append_demo_trades_persists_jsonl(tmp_path, monkeypatch: pytest.MonkeyPatch):
    _settings(monkeypatch)
    result = run_signal_demo_on_snapshots(
        _snapshot_rows(),
        {"T": date.today() + timedelta(days=20)},
        capital=2_000_000.0,
        risk_profile="balanced",
        max_trades=1,
        momentum_threshold_pct=0.10,
    )
    path = tmp_path / "scalp_demo_trades.jsonl"

    saved = append_demo_trades(result, path)

    assert saved == len(result.trades)
    assert path.exists()
    assert "BUY_CALL" in path.read_text(encoding="utf-8")


def test_append_demo_trades_deduplicates_ticket_ids(tmp_path, monkeypatch: pytest.MonkeyPatch):
    _settings(monkeypatch)
    result = run_signal_demo_on_snapshots(
        _snapshot_rows(),
        {"T": date.today() + timedelta(days=20)},
        capital=2_000_000.0,
        risk_profile="balanced",
        max_trades=1,
        momentum_threshold_pct=0.10,
    )
    path = tmp_path / "scalp_demo_trades.jsonl"

    first = append_demo_trades(result, path)
    second = append_demo_trades(result, path)

    assert first == len(result.trades)
    assert second == 0
