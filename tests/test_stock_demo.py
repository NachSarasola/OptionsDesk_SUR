from __future__ import annotations

import json
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from optionsdesk.backtest.stock_demo import (
    load_stock_demo_positions,
    load_stock_demo_trades,
    run_stock_demo_tick,
)
from optionsdesk.config.costs import CostModel
from optionsdesk.data.providers.base import Quote
from optionsdesk.signals.stock_signals import StockSignal


BA = ZoneInfo("America/Argentina/Buenos_Aires")
ZERO_COSTS = CostModel(
    stock_commission_pct=0.0,
    option_commission_pct=0.0,
    stock_market_fee_pct=0.0,
    option_market_fee_pct=0.0,
    exercise_fee_pct=0.0,
    exercise_market_fee_pct=0.0,
    iva_rate=0.0,
)


def _quote(symbol: str, bid: float, ask: float, last: float, ts: datetime) -> Quote:
    return Quote(
        symbol=symbol,
        bid=bid,
        ask=ask,
        last=last,
        volume=10_000.0,
        timestamp=ts,
        source="TEST",
        received_at=ts,
        book_ts=ts,
    )


def _signal(symbol: str, now: datetime, *, strategy: str = "SWING") -> StockSignal:
    # Target lejano a proposito: las salidas las maneja el trailing stop universal.
    return StockSignal(
        symbol=symbol,
        strategy=strategy,
        signal_type="SWING_BREAKOUT" if strategy == "SWING" else "SCALP_CONTINUATION",
        created_at=now,
        entry_price=100.0,
        stop_price=90.0,
        target_price=130.0,
        score=80.0,
        rationale="test signal",
        time_stop_min=30 if strategy == "SCALP" else None,
        max_holding_days=None if strategy == "SCALP" else 10,
    )


def test_stock_demo_opens_at_ask_and_persists_position(tmp_path):
    now = datetime(2026, 6, 3, 11, 30, tzinfo=BA)
    positions = tmp_path / "positions.json"
    trades = tmp_path / "trades.jsonl"
    quote = _quote("GGAL", 99.5, 100.0, 99.75, now)   # spread tight: sin slippage

    result = run_stock_demo_tick(
        {"GGAL": quote},
        [_signal("GGAL", now)],
        now=now,
        positions_path=positions,
        trades_path=trades,
        trade_ars=1_000.0,
        costs=ZERO_COSTS,
    )

    assert result.opened_count == 1
    assert result.positions[0].entry_price == pytest.approx(100.0)
    # Risk = 10.0 (100.0 - 90.0). Max risk = 1.8M * 0.05 = 90000. Qty = 9000
    assert result.positions[0].quantity == 9000
    assert load_stock_demo_positions(positions)[0].symbol == "GGAL"
    assert load_stock_demo_trades(trades) == []


def test_stock_demo_does_not_duplicate_position_on_repeated_refresh(tmp_path):
    now = datetime(2026, 6, 3, 11, 30, tzinfo=BA)
    positions = tmp_path / "positions.json"
    trades = tmp_path / "trades.jsonl"
    quote = _quote("GGAL", 99.0, 100.0, 99.5, now)
    signal = _signal("GGAL", now)

    run_stock_demo_tick({"GGAL": quote}, [signal], now=now, positions_path=positions, trades_path=trades, trade_ars=1_000.0, costs=ZERO_COSTS)
    result = run_stock_demo_tick({"GGAL": quote}, [signal], now=now + timedelta(seconds=5), positions_path=positions, trades_path=trades, trade_ars=1_000.0, costs=ZERO_COSTS)

    assert len(result.positions) == 1
    assert result.opened_count == 0
    assert result.skipped_reasons["posicion_existente"] == 1


def test_stock_demo_closes_at_bid_and_appends_trade(tmp_path):
    now = datetime(2026, 6, 3, 11, 30, tzinfo=BA)
    positions = tmp_path / "positions.json"
    trades = tmp_path / "trades.jsonl"
    entry_quote = _quote("GGAL", 99.5, 100.0, 99.75, now)   # spread tight: sin slippage
    exit_quote = _quote("GGAL", 111.0, 112.0, 111.5, now + timedelta(minutes=1))

    run_stock_demo_tick({"GGAL": entry_quote}, [_signal("GGAL", now)], now=now, positions_path=positions, trades_path=trades, trade_ars=1_000.0, costs=ZERO_COSTS)
    
    # First update moves the trailing stop up (bid 111 - 10 risk = 101 stop)
    run_stock_demo_tick({"GGAL": exit_quote}, [], now=now + timedelta(minutes=1), positions_path=positions, trades_path=trades, trade_ars=1_000.0, costs=ZERO_COSTS)
    
    # Second update drops price to hit trailing stop
    stop_quote = _quote("GGAL", 100.5, 101.5, 101.0, now + timedelta(minutes=2))
    result = run_stock_demo_tick({"GGAL": stop_quote}, [], now=now + timedelta(minutes=2), positions_path=positions, trades_path=trades, trade_ars=1_000.0, costs=ZERO_COSTS)

    assert result.closed_count == 1
    assert result.positions == []
    assert result.closed_trades[-1].exit_price == pytest.approx(100.5)
    # Entry 100.0, Exit 100.5, Qty 9000 -> PnL = 4500.0
    assert result.closed_trades[-1].pnl_ars == pytest.approx(4500.0)
    assert result.closed_trades[-1].exit_reason == "STOP"


def test_stock_demo_requires_valid_and_fresh_book(tmp_path):
    now = datetime(2026, 6, 3, 11, 30, tzinfo=BA)
    old = now - timedelta(minutes=10)
    positions = tmp_path / "positions.json"
    trades = tmp_path / "trades.jsonl"
    stale_quote = _quote("GGAL", 99.0, 100.0, 99.5, old)
    bad_book = _quote("YPFD", 0.0, 100.0, 99.5, now)

    result = run_stock_demo_tick(
        {"GGAL": stale_quote, "YPFD": bad_book},
        [_signal("GGAL", now), _signal("YPFD", now)],
        now=now,
        positions_path=positions,
        trades_path=trades,
        quote_max_age_s=60,
        costs=ZERO_COSTS,
    )

    assert result.positions == []
    assert result.skipped_reasons["quote_stale_entrada"] == 1
    assert result.skipped_reasons["sin_ask_para_entrar"] == 1


def test_stock_demo_respects_max_positions_and_one_position_per_symbol(tmp_path):
    now = datetime(2026, 6, 3, 11, 30, tzinfo=BA)
    positions = tmp_path / "positions.json"
    trades = tmp_path / "trades.jsonl"
    quotes = {
        "GGAL": _quote("GGAL", 99.0, 100.0, 99.5, now),
        "YPFD": _quote("YPFD", 199.0, 200.0, 199.5, now),
    }
    signals = [_signal("GGAL", now), _signal("GGAL", now), _signal("YPFD", now)]

    result = run_stock_demo_tick(
        quotes,
        signals,
        now=now,
        positions_path=positions,
        trades_path=trades,
        trade_ars=1_000.0,
        max_swings=1,
        costs=ZERO_COSTS,
    )

    assert len(result.positions) == 1
    assert result.positions[0].symbol == "GGAL"
    assert result.skipped_reasons["posicion_existente"] == 1
    assert result.skipped_reasons["max_swings"] == 1


def _seed_losing_breakouts(tmp_path, n: int) -> None:
    """Siembra la fuente de atribución con n breakouts perdedores."""
    lines = [
        json.dumps({"strategy": "SWING", "signal_type": "SWING_BREAKOUT",
                    "pnl_ars": -500.0, "exit_reason": "STOP"})
        for _ in range(n)
    ]
    (tmp_path / "stock_demo_trades.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_demo_blocks_setup_with_proven_negative_edge(tmp_path):
    _seed_losing_breakouts(tmp_path, 16)   # muestra suficiente + perdedor
    now = datetime(2026, 6, 3, 11, 30, tzinfo=BA)
    quote = _quote("GGAL", 99.0, 100.0, 99.5, now)
    result = run_stock_demo_tick(
        {"GGAL": quote}, [_signal("GGAL", now)], now=now,
        positions_path=tmp_path / "positions.json", trades_path=tmp_path / "trades.jsonl",
        trade_ars=1_000.0, costs=ZERO_COSTS,
    )
    assert result.opened_count == 0
    assert result.skipped_reasons.get("sin_edge_realizado") == 1


def test_demo_does_not_block_new_setup_with_small_sample(tmp_path):
    _seed_losing_breakouts(tmp_path, 5)    # pocos trades: idea en desarrollo, no se corta
    now = datetime(2026, 6, 3, 11, 30, tzinfo=BA)
    quote = _quote("GGAL", 99.0, 100.0, 99.5, now)
    result = run_stock_demo_tick(
        {"GGAL": quote}, [_signal("GGAL", now)], now=now,
        positions_path=tmp_path / "positions.json", trades_path=tmp_path / "trades.jsonl",
        trade_ars=1_000.0, costs=ZERO_COSTS,
    )
    assert result.opened_count == 1


def test_demo_block_junk_can_be_disabled(tmp_path):
    _seed_losing_breakouts(tmp_path, 16)
    now = datetime(2026, 6, 3, 11, 30, tzinfo=BA)
    quote = _quote("GGAL", 99.0, 100.0, 99.5, now)
    result = run_stock_demo_tick(
        {"GGAL": quote}, [_signal("GGAL", now)], now=now,
        positions_path=tmp_path / "positions.json", trades_path=tmp_path / "trades.jsonl",
        trade_ars=1_000.0, costs=ZERO_COSTS, block_junk=False,
    )
    assert result.opened_count == 1
