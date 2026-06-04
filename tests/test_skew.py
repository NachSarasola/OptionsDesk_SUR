"""Tests de signals/skew.py — put/call skew sobre list[RateResult]."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from optionsdesk.signals.skew import put_call_skew, skew_signal, skew_for_expiry


def _make_result(symbol: str, iv: float, is_liquid: bool = True):
    r = MagicMock()
    r.symbol = symbol
    r.iv = iv
    r.is_liquid = is_liquid
    return r


def _make_results(spot: float, put_iv: float, call_iv: float, expiry: str = "JUN"):
    """Resultados con 1 put OTM ~5% y 1 call OTM ~5%."""
    put_strike = round(spot * 0.95, 0)
    call_strike = round(spot * 1.05, 0)
    return [
        _make_result(f"GFGV{put_strike:.0f}{expiry}", put_iv),
        _make_result(f"GFGC{call_strike:.0f}{expiry}", call_iv),
    ]


# ── put_call_skew ─────────────────────────────────────────────────────────────

def test_put_call_skew_basic():
    results = _make_results(spot=1500.0, put_iv=0.65, call_iv=0.50)
    result = put_call_skew(results, spot=1500.0)
    assert result is not None
    assert result == pytest.approx(15.0, abs=1.0)


def test_put_call_skew_zero_difference():
    results = _make_results(spot=1500.0, put_iv=0.55, call_iv=0.55)
    result = put_call_skew(results, spot=1500.0)
    assert result is not None
    assert abs(result) < 1.0


def test_put_call_skew_no_puts():
    results = [_make_result("GFGC1575JUN", 0.50)]
    assert put_call_skew(results, spot=1500.0) is None


def test_put_call_skew_no_calls():
    results = [_make_result("GFGV1425JUN", 0.60)]
    assert put_call_skew(results, spot=1500.0) is None


def test_put_call_skew_empty_results():
    assert put_call_skew([], spot=1500.0) is None


def test_put_call_skew_invalid_spot():
    results = _make_results(spot=1500.0, put_iv=0.60, call_iv=0.50)
    assert put_call_skew(results, spot=0.0) is None


def test_put_call_skew_no_iv():
    results = [
        _make_result("GFGV1425JUN", 0.0),   # IV inválida
        _make_result("GFGC1575JUN", 0.50),
    ]
    assert put_call_skew(results, spot=1500.0) is None


def test_put_call_skew_far_strikes_outside_tolerance():
    """Strikes muy alejados del target OTM no son usados."""
    results = [
        _make_result("GFGV1200JUN", 0.80),   # target 1425, muy lejos
        _make_result("GFGC1575JUN", 0.50),
    ]
    assert put_call_skew(results, spot=1500.0) is None


# ── skew_signal ───────────────────────────────────────────────────────────────

def test_skew_signal_bearish():
    assert skew_signal(10.0) == "bearish"


def test_skew_signal_bullish():
    assert skew_signal(-5.0) == "bullish"


def test_skew_signal_neutral():
    assert skew_signal(2.0) == "neutral"


def test_skew_signal_none():
    assert skew_signal(None) == "sin datos"


def test_skew_signal_boundary_bearish():
    assert skew_signal(5.0) == "bearish"


def test_skew_signal_boundary_bullish():
    assert skew_signal(-2.0) == "bullish"


# ── skew_for_expiry ───────────────────────────────────────────────────────────

def test_skew_for_expiry_filters_correctly():
    results = [
        _make_result("GFGV1425JUN", 0.65),
        _make_result("GFGC1575JUN", 0.50),
        _make_result("GFGV1425JUL", 0.70),
        _make_result("GFGC1575JUL", 0.55),
    ]
    skew_jun = skew_for_expiry(results, spot=1500.0, expiry_code="JUN")
    skew_jul = skew_for_expiry(results, spot=1500.0, expiry_code="JUL")

    assert skew_jun is not None
    assert skew_jul is not None
    assert skew_jun == pytest.approx(15.0, abs=1.0)
    assert skew_jul == pytest.approx(15.0, abs=1.0)


def test_skew_for_expiry_wrong_expiry_returns_none():
    results = [
        _make_result("GFGV1425JUN", 0.65),
        _make_result("GFGC1575JUN", 0.50),
    ]
    assert skew_for_expiry(results, spot=1500.0, expiry_code="AGO") is None
