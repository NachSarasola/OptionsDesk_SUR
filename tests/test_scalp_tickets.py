from __future__ import annotations

import math
from datetime import date, datetime, timedelta
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from optionsdesk.config.settings import settings
from optionsdesk.core.carry import CarryEstimate
from optionsdesk.core.greeks_engine import OptionGreeks
from optionsdesk.data.providers.base import MarketDataHealth, OptionsChain, Quote
from optionsdesk.signals.scalp_tickets import (
    DailyScalpPerformance,
    OperationalMode,
    UnderlyingTrigger,
    build_manual_scalp_ticket,
    confirm_manual_reconciliation,
    detect_underlying_trigger,
    manual_reconciliation_is_current,
    resolve_operational_mode,
)
from optionsdesk.signals.scalping import LivePulse, ScalpSignal

BA = ZoneInfo("America/Argentina/Buenos_Aires")


def _set_ticket_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "costs_verified", True)
    monkeypatch.setattr(settings, "costs_profile", "iol-gold-test")
    monkeypatch.setattr(settings, "iol_provisional_tickets_enabled", True)
    monkeypatch.setattr(settings, "iol_ticket_ttl_s", 60)
    monkeypatch.setattr(settings, "iol_ticket_max_quote_age_s", 75)
    monkeypatch.setattr(settings, "iol_ticket_max_spread_pct", 6.0)
    monkeypatch.setattr(settings, "scalping_risk_per_trade_pct", 0.35)
    monkeypatch.setattr(settings, "scalping_daily_loss_pct", 1.5)
    monkeypatch.setattr(settings, "scalping_max_stops_per_day", 2)
    monkeypatch.setattr(settings, "scalping_initial_lot_cap", 2)
    monkeypatch.setattr(settings, "scalping_manual_sizing", True)
    monkeypatch.setattr(settings, "scalping_manual_max_lots", 10)


def _signal(symbol: str = "GFGC4000JU") -> ScalpSignal:
    return ScalpSignal(
        signal_type="WAVE_RIDE",
        action="BUY_CALL",
        symbol=symbol,
        strike=4000.0,
        option_type="C",
        days_to_expiry=20,
        bid=100.0,
        ask=104.0,
        mid=102.0,
        spread_pct=3.92,
        delta=0.50,
        gamma=0.001,
        theta=-2.0,
        vega=10.0,
        iv=0.50,
        score=90.0,
        edge_source="test",
        rationale="test",
        urgency="alta",
        plan_entry=102.03,
        plan_sl=82.0,
        plan_tp=134.0,
        confidence="actionable",
        planned_risk_ars=2500.0,
        fees_ars=130.0,
        breakeven_move_pct=0.4,
        time_stop_min=18,
    )


def _greek(source: str = "IOL_REST") -> OptionGreeks:
    return OptionGreeks(
        symbol="GFGC4000JU",
        strike=4000.0,
        option_type="C",
        days=20,
        spot=4005.0,
        bid=100.0,
        ask=104.0,
        mid=102.0,
        iv=0.50,
        delta=0.50,
        gamma=0.001,
        theta=-2.0,
        vega=10.0,
        moneyness="ATM",
        moneyness_pct=-0.12,
        volume=250.0,
        spread_pct=3.92,
        bid_size=8.0,
        ask_size=5.0,
        quote_age_s=10.0,
        source=source,
        min_price_increment=0.05,
    )


def _chain(now: datetime) -> OptionsChain:
    spot = Quote("GGAL", 4000.0, 4010.0, 4005.0, 100_000.0, now, source="IOL_REST", book_ts=now)
    opt = Quote(
        "GFGC4000JU",
        100.0,
        104.0,
        102.0,
        250.0,
        now,
        bid_size=8.0,
        ask_size=5.0,
        source="IOL_REST",
        book_ts=now,
    )
    return OptionsChain(
        "GGAL",
        spot,
        {"GFGC4000JU": opt},
        expiry_calendar={"JU": date.today() + timedelta(days=20)},
        min_price_increments={"GFGC4000JU": 0.05},
    )


def _trigger(now: datetime) -> UnderlyingTrigger:
    return UnderlyingTrigger(
        setup="CONTINUATION",
        side="BULL",
        created_at=now,
        spot=4005.0,
        invalidation_spot=3985.0,
        target_spot=4040.0,
        rationale="test",
    )


def test_iol_provisional_builds_one_manual_ticket(monkeypatch: pytest.MonkeyPatch):
    _set_ticket_settings(monkeypatch)
    now = datetime(2026, 6, 2, 12, 15, tzinfo=BA)
    health = MarketDataHealth(source="IOL", connected=True, ticket_capable=True)
    carry = CarryEstimate(45.0, math.log1p(0.45), "configured_caucion_fallback")

    ticket, blockers = build_manual_scalp_ticket(
        [_signal()],
        {"GFGC4000JU": _greek()},
        _chain(now),
        health,
        _trigger(now),
        carry,
        capital=1_000_000.0,
        local_positions=[],
        account_reconciled=True,
        daily_performance=DailyScalpPerformance(),
        now=now,
    )

    assert blockers == []
    assert ticket is not None
    assert ticket.mode == OperationalMode.IOL_PROVISIONAL
    assert ticket.selection.symbol == "GFGC4000JU"
    assert ticket.selection.limit_price == pytest.approx(102.05)
    assert ticket.max_lots == 10
    assert ticket.valid_until == now + timedelta(seconds=60)
    assert ticket.warnings


def test_iol_ticket_blocks_listing_quotes(monkeypatch: pytest.MonkeyPatch):
    _set_ticket_settings(monkeypatch)
    now = datetime(2026, 6, 2, 12, 15, tzinfo=BA)

    ticket, blockers = build_manual_scalp_ticket(
        [_signal()],
        {"GFGC4000JU": _greek(source="IOL_LISTING")},
        _chain(now),
        MarketDataHealth(source="IOL", connected=True, ticket_capable=True),
        _trigger(now),
        CarryEstimate(45.0, math.log1p(0.45), "configured_caucion_fallback"),
        capital=1_000_000.0,
        local_positions=[],
        account_reconciled=True,
        now=now,
    )

    assert ticket is None
    assert blockers == ["sin_contrato_ejecutable"]


def test_ticket_blocks_without_manual_matriz_reconciliation(monkeypatch: pytest.MonkeyPatch):
    _set_ticket_settings(monkeypatch)
    now = datetime(2026, 6, 2, 12, 15, tzinfo=BA)

    ticket, blockers = build_manual_scalp_ticket(
        [_signal()],
        {"GFGC4000JU": _greek()},
        _chain(now),
        MarketDataHealth(source="IOL", connected=True, ticket_capable=True),
        _trigger(now),
        CarryEstimate(45.0, math.log1p(0.45), "configured_caucion_fallback"),
        capital=1_000_000.0,
        local_positions=[],
        account_reconciled=False,
        now=now,
    )

    assert ticket is None
    assert "matriz_no_conciliada" in blockers


def test_manual_sizing_allows_ticket_without_loaded_capital(monkeypatch: pytest.MonkeyPatch):
    _set_ticket_settings(monkeypatch)
    now = datetime(2026, 6, 2, 12, 15, tzinfo=BA)

    ticket, blockers = build_manual_scalp_ticket(
        [_signal()],
        {"GFGC4000JU": _greek()},
        _chain(now),
        MarketDataHealth(source="IOL", connected=True, ticket_capable=True),
        _trigger(now),
        CarryEstimate(45.0, math.log1p(0.45), "configured_caucion_fallback"),
        capital=None,
        local_positions=[],
        account_reconciled=True,
        now=now,
    )

    assert blockers == []
    assert ticket is not None
    assert ticket.max_lots == 10


def test_ticket_blocks_when_any_local_position_is_open(monkeypatch: pytest.MonkeyPatch):
    _set_ticket_settings(monkeypatch)
    now = datetime(2026, 6, 2, 12, 15, tzinfo=BA)

    ticket, blockers = build_manual_scalp_ticket(
        [_signal()],
        {"GFGC4000JU": _greek()},
        _chain(now),
        MarketDataHealth(source="IOL", connected=True, ticket_capable=True),
        _trigger(now),
        CarryEstimate(45.0, math.log1p(0.45), "configured_caucion_fallback"),
        capital=1_000_000.0,
        local_positions=[SimpleNamespace(strategy="COVERED_CALL")],
        account_reconciled=True,
        now=now,
    )

    assert ticket is None
    assert "ya_hay_posicion_abierta" in blockers


def test_manual_reconciliation_is_day_scoped(tmp_path):
    path = tmp_path / "manual_reconciliation.json"
    today = datetime(2026, 6, 2, 12, 0, tzinfo=BA)
    yesterday = today - timedelta(days=1)

    assert not manual_reconciliation_is_current(path=path, now=today)
    confirm_manual_reconciliation(0, path=path, now=yesterday)
    assert not manual_reconciliation_is_current(path=path, now=today)
    confirm_manual_reconciliation(0, path=path, now=today)
    assert manual_reconciliation_is_current(path=path, now=today)


def test_resolve_mode_distinguishes_iol_radar_from_provisional(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings, "iol_provisional_tickets_enabled", True)

    assert (
        resolve_operational_mode(MarketDataHealth(source="IOL", connected=True, ticket_capable=False))
        == OperationalMode.RADAR_IOL
    )
    assert (
        resolve_operational_mode(MarketDataHealth(source="IOL", connected=True, ticket_capable=True))
        == OperationalMode.IOL_PROVISIONAL
    )


def _bars(start: datetime, closes: list[float], step: timedelta) -> pd.DataFrame:
    rows = []
    for i, close in enumerate(closes):
        rows.append(
            {
                "time": start + step * i,
                "open": close - 0.5,
                "high": close + 0.8,
                "low": close - 0.8,
                "close": close,
                "volume": 1000,
            }
        )
    return pd.DataFrame(rows)


def test_detect_underlying_trigger_uses_completed_1m_and_5m_bars():
    now = datetime(2026, 6, 2, 12, 10, tzinfo=BA)
    bars_1m = _bars(
        datetime(2026, 6, 2, 11, 59, tzinfo=BA),
        [1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1011, 1013],
        timedelta(minutes=1),
    )
    bars_5m = _bars(
        datetime(2026, 6, 2, 11, 40, tzinfo=BA),
        [1000, 1003, 1006, 1009, 1012],
        timedelta(minutes=5),
    )

    trigger = detect_underlying_trigger(
        bars_1m,
        bars_5m,
        LivePulse(True, direction="BULL", observations=5, span_s=90.0),
        now=now,
    )

    assert trigger is not None
    assert trigger.setup == "CONTINUATION"
    assert trigger.side == "BULL"
    assert trigger.spot == pytest.approx(1013.0)


def _flat_bars(start: datetime, closes: list[float], step: timedelta) -> pd.DataFrame:
    """Velas degeneradas O=H=L=C: el true range = |paso|, atr predecible para tests."""
    return pd.DataFrame(
        [
            {"time": start + step * i, "open": c, "high": c, "low": c, "close": c, "volume": 0.0}
            for i, c in enumerate(closes)
        ]
    )


def test_trigger_rejects_breakout_inside_noise_band():
    """Cinta lateral con poke de 1 tick sobre el rango → chop, no dispara (anti-chop)."""
    now = datetime(2026, 6, 2, 12, 12, tzinfo=BA)
    chop = [1000, 1001, 1000, 1001, 1000, 1001, 1000, 1001, 1000, 1001]
    bars_1m = _flat_bars(datetime(2026, 6, 2, 12, 0, tzinfo=BA), chop, timedelta(minutes=1))
    bars_5m = _flat_bars(datetime(2026, 6, 2, 11, 40, tzinfo=BA), [1000, 1001, 1000, 1001, 1000], timedelta(minutes=5))

    trigger = detect_underlying_trigger(
        bars_1m, bars_5m,
        LivePulse(True, direction="BULL", observations=6, span_s=120.0, efficiency=0.9),
        now=now,
    )
    assert trigger is None


def test_trigger_rejects_parabolic_extension_on_continuation():
    """Salto de varios ATR sobre el máximo de 3 velas → chase parabólico, no perseguir."""
    now = datetime(2026, 6, 2, 12, 12, tzinfo=BA)
    ramp = [1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1020]  # último: +13 sobre nivel 1007
    bars_1m = _flat_bars(datetime(2026, 6, 2, 12, 0, tzinfo=BA), ramp, timedelta(minutes=1))
    bars_5m = _flat_bars(datetime(2026, 6, 2, 11, 40, tzinfo=BA), [1000, 1003, 1006, 1009, 1012], timedelta(minutes=5))

    trigger = detect_underlying_trigger(
        bars_1m, bars_5m,
        LivePulse(True, direction="BULL", observations=6, span_s=120.0, efficiency=0.95),
        now=now,
    )
    assert trigger is None
