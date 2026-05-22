"""Tests del modulo de analisis tecnico."""
from __future__ import annotations

import math

import pandas as pd
import pytest

from optionsdesk.signals.technical import (
    TechnicalSnapshot,
    analyze,
    atr,
    ema,
    momentum_pct,
    rsi,
    sma,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_ohlcv(closes: list[float]) -> pd.DataFrame:
    """DataFrame OHLCV mínimo a partir de una serie de cierres."""
    return pd.DataFrame({
        "date":   pd.date_range("2025-01-01", periods=len(closes), freq="D"),
        "open":   [c * 0.99 for c in closes],
        "high":   [c * 1.01 for c in closes],
        "low":    [c * 0.98 for c in closes],
        "close":  closes,
        "volume": [1_000_000] * len(closes),
    })


# ── SMA ───────────────────────────────────────────────────────────────────────

def test_sma_simple_values():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    result = sma(s, 3)
    assert pd.isna(result.iloc[0])
    assert pd.isna(result.iloc[1])
    assert result.iloc[2] == pytest.approx(2.0)
    assert result.iloc[4] == pytest.approx(4.0)


def test_sma_period_1_equals_input():
    s = pd.Series([10.0, 20.0, 30.0])
    assert sma(s, 1).tolist() == pytest.approx([10.0, 20.0, 30.0])


# ── EMA ───────────────────────────────────────────────────────────────────────

def test_ema_converges():
    s = pd.Series([100.0] * 30)
    result = ema(s, 10)
    assert result.dropna().iloc[-1] == pytest.approx(100.0, abs=1e-6)


def test_ema_responds_to_change():
    s = pd.Series([100.0] * 20 + [200.0] * 10)
    result = ema(s, 5)
    # EMA debe subir después del salto
    assert result.iloc[-1] > result.iloc[19]


# ── RSI ───────────────────────────────────────────────────────────────────────

def test_rsi_constant_series_does_not_raise():
    """Serie sin cambios → RSI puede ser 0 o 100 según la implementación, pero no debe explotar."""
    s = pd.Series([100.0] * 20)
    result = rsi(s, 14)
    non_nan = result.dropna()
    # Si hay valores válidos, deben estar en [0, 100]; si todos son NaN, no debe explotar
    if not non_nan.empty:
        assert 0.0 <= float(non_nan.iloc[-1]) <= 100.0


def test_rsi_all_gains_near_100():
    # Serie siempre subiendo → RSI debe ser alto
    s = pd.Series([float(i) for i in range(1, 35)])
    result = rsi(s, 14)
    assert result.dropna().iloc[-1] > 80.0


def test_rsi_all_losses_near_zero():
    # Serie siempre bajando → RSI bajo
    s = pd.Series([float(35 - i) for i in range(35)])
    result = rsi(s, 14)
    assert result.dropna().iloc[-1] < 20.0


def test_rsi_bounded():
    import random
    rng = random.Random(123)
    s = pd.Series([100.0 + rng.gauss(0, 3) for _ in range(100)])
    result = rsi(s, 14).dropna()
    assert (result >= 0.0).all() and (result <= 100.0).all()


# ── ATR ───────────────────────────────────────────────────────────────────────

def test_atr_positive():
    closes = [float(100 + i) for i in range(20)]
    df = _make_ohlcv(closes)
    result = atr(df, 14).dropna()
    assert (result > 0).all()


def test_atr_zero_range_is_small():
    # Velas sin rango → ATR debe ser 0 o muy cercano
    df = pd.DataFrame({
        "high":  [100.0] * 20,
        "low":   [100.0] * 20,
        "close": [100.0] * 20,
    })
    result = atr(df, 14).dropna()
    assert (result.abs() < 1e-6).all()


# ── Momentum ──────────────────────────────────────────────────────────────────

def test_momentum_positive_uptrend():
    s = pd.Series([float(100 + i) for i in range(20)])
    result = momentum_pct(s, 10)
    assert result.dropna().iloc[-1] > 0.0


def test_momentum_zero_flat():
    s = pd.Series([100.0] * 20)
    result = momentum_pct(s, 10)
    assert result.dropna().iloc[-1] == pytest.approx(0.0)


def test_momentum_negative_downtrend():
    s = pd.Series([float(200 - i) for i in range(20)])
    result = momentum_pct(s, 10)
    assert result.dropna().iloc[-1] < 0.0


# ── analyze() ─────────────────────────────────────────────────────────────────

def test_analyze_returns_snapshot():
    df = _make_ohlcv([float(100 + i) for i in range(60)])
    snap = analyze(df)
    assert isinstance(snap, TechnicalSnapshot)
    assert snap.trend in ("ALCISTA", "BAJISTA", "LATERAL")
    assert 0.0 <= snap.rsi <= 100.0
    assert snap.atr_pct >= 0.0
    assert isinstance(snap.read, str) and len(snap.read) > 10


def test_analyze_uptrend_detected():
    # Serie claramente alcista durante 60 días
    closes = [float(5000 + i * 30) for i in range(60)]
    df = _make_ohlcv(closes)
    snap = analyze(df)
    assert snap.trend == "ALCISTA"
    assert snap.momentum > 0.0


def test_analyze_downtrend_detected():
    closes = [float(8000 - i * 30) for i in range(60)]
    df = _make_ohlcv(closes)
    snap = analyze(df)
    assert snap.trend == "BAJISTA"
    assert snap.momentum < 0.0


def test_analyze_insufficient_data_returns_lateral():
    df = _make_ohlcv([100.0, 101.0, 99.0])
    snap = analyze(df)
    assert snap.trend == "LATERAL"
    assert "insuficientes" in snap.read.lower()


def test_analyze_sma_fast_above_slow_in_uptrend():
    closes = [float(5000 + i * 20) for i in range(60)]
    df = _make_ohlcv(closes)
    snap = analyze(df)
    assert snap.sma_fast > snap.sma_slow
