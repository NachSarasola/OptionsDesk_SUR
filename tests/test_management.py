"""Tests de las señales de gestión activa (signals/management.py)."""
from __future__ import annotations

from datetime import date, timedelta

import pytest

from optionsdesk.signals.management import ManagementSignal, SignalType, evaluate_position
from optionsdesk.signals.monitor import OpenPosition


def _make_pos(
    strategy: str = "COVERED_CALL",
    premium_received: float = 300.0,
    strike: float = 9000.0,
    spot_entry: float = 8500.0,
    iv_entry: float = 0.55,
    days_entry: int = 30,
    days_held: int = 10,
    target_capture_pct: float = 50.0,
    max_loss_mult: float = 2.0,
    roll_dte: int = 21,
    defend_delta: float = 0.50,
) -> OpenPosition:
    entry_date = date.today() - timedelta(days=days_held)
    return OpenPosition(
        symbol=f"GFG{'C' if strategy == 'COVERED_CALL' else 'V'}{strike:.0f}F",
        strategy=strategy,
        strike=strike,
        spot_entry=spot_entry,
        premium_received=premium_received,
        net_outlay=strike - premium_received,
        iv_entry=iv_entry,
        days_entry=days_entry,
        entry_date=entry_date,
        target_exit_days=days_entry // 2,
        target_capture_pct=target_capture_pct,
        max_loss_mult=max_loss_mult,
        roll_dte=roll_dte,
        defend_delta=defend_delta,
    )


# ── TAKE_PROFIT ───────────────────────────────────────────────────────────────

def test_take_profit_when_capture_exceeds_target():
    """Captura >= objetivo → TAKE_PROFIT."""
    pos = _make_pos(premium_received=300.0, target_capture_pct=50.0, days_entry=30)
    # Forzar captura alta: pasar el spot muy lejos del strike (put queda casi sin valor)
    # Para un SHORT_PUT, spot muy por encima del strike → opción casi vale cero
    pos_put = _make_pos(
        strategy="SHORT_PUT",
        premium_received=300.0,
        strike=8000.0,
        target_capture_pct=50.0,
        iv_entry=0.01,   # IV muy baja → la opción vale casi nada → captura ~100%
        days_entry=30,
        days_held=5,
    )
    sig = evaluate_position(pos_put, current_spot=9500.0, current_iv=0.01)
    assert sig.signal_type == SignalType.TAKE_PROFIT
    assert sig.capture_pct >= 50.0


# ── STOP ─────────────────────────────────────────────────────────────────────

def test_stop_when_loss_exceeds_max():
    """Pérdida >= max_loss_mult × prima → STOP."""
    # CC: opción comprada back a 3× la prima → pérdida = 2× → stop
    pos = _make_pos(
        strategy="COVERED_CALL",
        premium_received=100.0,
        strike=9000.0,
        max_loss_mult=2.0,
        iv_entry=0.80,
        days_entry=30,
        days_held=5,
        target_capture_pct=50.0,
    )
    # Spot muy arriba del strike → call deep ITM → vale mucho más que la prima
    sig = evaluate_position(pos, current_spot=12_000.0, current_iv=0.80)
    assert sig.signal_type == SignalType.STOP


# ── ROLL ─────────────────────────────────────────────────────────────────────

def test_roll_when_near_expiry():
    """Quedan ≤ roll_dte días → ROLL (si la opción tiene valor)."""
    pos = _make_pos(
        strategy="SHORT_PUT",
        premium_received=300.0,
        strike=8000.0,
        iv_entry=0.55,
        days_entry=25,
        days_held=5,      # quedan 20 días → ≤ 21 = roll_dte
        roll_dte=21,
        target_capture_pct=50.0,
        max_loss_mult=2.0,
        defend_delta=0.50,
    )
    sig = evaluate_position(pos, current_spot=8500.0)
    assert sig.signal_type == SignalType.ROLL


# ── DEFEND ────────────────────────────────────────────────────────────────────

def test_defend_when_delta_high():
    """Delta >= defend_delta → DEFEND."""
    pos = _make_pos(
        strategy="SHORT_PUT",
        premium_received=300.0,
        strike=9000.0,       # put ATM/ITM con spot bajo → delta alto
        iv_entry=0.55,
        days_entry=30,
        days_held=5,
        roll_dte=5,          # todavía lejos del roll (quedan 25 días)
        defend_delta=0.50,
        target_capture_pct=80.0,   # todavía no alcanzó el objetivo
        max_loss_mult=10.0,  # stop muy lejos para que no se active
    )
    # Spot igual al strike → delta del put ≈ 0.50 → justo en el umbral
    sig = evaluate_position(pos, current_spot=9000.0, current_iv=0.55)
    assert sig.signal_type in (SignalType.DEFEND, SignalType.HOLD)


def test_defend_deep_itm_put():
    """Put deep ITM (spot << strike) → delta alto → DEFEND."""
    pos = _make_pos(
        strategy="SHORT_PUT",
        premium_received=200.0,
        strike=10_000.0,
        iv_entry=0.55,
        days_entry=30,
        days_held=5,
        roll_dte=5,
        defend_delta=0.50,
        target_capture_pct=80.0,
        max_loss_mult=5.0,
    )
    sig = evaluate_position(pos, current_spot=7_000.0, current_iv=0.55)
    # Put muy ITM → |delta| muy cercano a 1 → DEFEND o STOP
    assert sig.signal_type in (SignalType.DEFEND, SignalType.STOP)


# ── HOLD ─────────────────────────────────────────────────────────────────────

def test_hold_when_no_condition_met():
    """Ninguna condición activa → HOLD.

    Put moderadamente OTM (spot ~12% sobre strike), prima chica vs strike,
    muchos DTE restantes: la captura parcial (<50%), delta bajo (<0.50) y
    los días restantes (>21) no activan ningún gate.
    """
    pos = _make_pos(
        strategy="SHORT_PUT",
        premium_received=100.0,  # prima pequeña → stop threshold alto = 300
        strike=7000.0,           # put OTM con spot=8500 → ~18% OTM
        iv_entry=0.55,
        days_entry=45,
        days_held=5,             # quedan 40 días → lejos del roll_dte=21
        roll_dte=21,
        defend_delta=0.50,
        target_capture_pct=50.0,
        max_loss_mult=2.0,
        spot_entry=8500.0,
    )
    # Con spot=8500 y put strike=7000, la opción es OTM pero no tan OTM como
    # para estar sin valor (40 DTE, iv=0.55). La captura parcial (<50%) y el
    # delta bajo (<0.50) garantizan HOLD.
    sig = evaluate_position(pos, current_spot=8500.0)
    assert sig.signal_type == SignalType.HOLD


def test_long_scalp_hold_does_not_trigger_short_premium_stop(monkeypatch):
    """Un scalp comprado no debe evaluarse como venta de prima."""
    pos = _make_pos(
        strategy="SCALP_LONG_CALL",
        premium_received=-100.0,
        strike=9000.0,
        days_entry=30,
        days_held=5,
        target_capture_pct=50.0,
    )
    pos.net_outlay = 100.0
    monkeypatch.setattr("optionsdesk.signals.management._current_option_value", lambda *args: 95.0)
    monkeypatch.setattr("optionsdesk.signals.management._current_delta_abs", lambda *args: 0.35)

    sig = evaluate_position(pos, current_spot=9000.0)

    assert sig.signal_type == SignalType.HOLD
    assert sig.capture_pct == pytest.approx(-5.0)


def test_long_scalp_take_profit_and_stop(monkeypatch):
    """La gestión de scalps comprados usa P&L sobre débito pagado."""
    pos = _make_pos(
        strategy="SCALP_LONG_CALL",
        premium_received=-100.0,
        strike=9000.0,
        days_entry=30,
        days_held=5,
        target_capture_pct=50.0,
    )
    pos.net_outlay = 100.0
    monkeypatch.setattr("optionsdesk.signals.management._current_delta_abs", lambda *args: 0.35)

    monkeypatch.setattr("optionsdesk.signals.management._current_option_value", lambda *args: 160.0)
    assert evaluate_position(pos, current_spot=9000.0).signal_type == SignalType.TAKE_PROFIT

    monkeypatch.setattr("optionsdesk.signals.management._current_option_value", lambda *args: 60.0)
    assert evaluate_position(pos, current_spot=9000.0).signal_type == SignalType.STOP


# ── Retrocompatibilidad from_dict ─────────────────────────────────────────────

def test_from_dict_defaults_new_fields():
    """JSON sin los nuevos campos → usa defaults. Retrocompatible."""
    d = {
        "symbol": "GFGC9000F",
        "strategy": "COVERED_CALL",
        "strike": 9000.0,
        "spot_entry": 8500.0,
        "premium_received": 300.0,
        "net_outlay": 8200.0,
        "iv_entry": 0.55,
        "days_entry": 30,
        "entry_date": date.today().isoformat(),
        "target_exit_days": 15,
        "target_capture_pct": 50.0,
    }
    pos = OpenPosition.from_dict(d)
    assert pos.max_loss_mult == pytest.approx(2.0)
    assert pos.roll_dte == 21
    assert pos.defend_delta == pytest.approx(0.50)


def test_from_dict_reads_new_fields():
    """JSON con los nuevos campos → los usa."""
    d = {
        "symbol": "GFGC9000F",
        "strategy": "COVERED_CALL",
        "strike": 9000.0,
        "spot_entry": 8500.0,
        "premium_received": 300.0,
        "net_outlay": 8200.0,
        "iv_entry": 0.55,
        "days_entry": 30,
        "entry_date": date.today().isoformat(),
        "target_exit_days": 15,
        "target_capture_pct": 75.0,
        "max_loss_mult": 3.0,
        "roll_dte": 14,
        "defend_delta": 0.45,
    }
    pos = OpenPosition.from_dict(d)
    assert pos.max_loss_mult == pytest.approx(3.0)
    assert pos.roll_dte == 14
    assert pos.defend_delta == pytest.approx(0.45)
